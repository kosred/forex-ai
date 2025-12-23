import contextlib
import inspect
import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core.storage import StrategyLedger

from ..core.config import Settings
from ..domain.events import PreparedDataset, SignalResult
from ..features.pipeline import FeatureEngineer
from ..models.evolution import EvoExpertCMA
from ..models.onnx_exporter import ONNXInferenceEngine
from ..models.rl import RLExpertPPO, RLExpertSAC
from ..models.transformers import TransformerExpertTorch
from ..training.ensemble import MetaBlender

try:
    from joblib import load
except ImportError:
    load = None

try:
    from ..models.kan import KANExpert
except Exception:  # pragma: no cover
    KANExpert = None

try:
    from ..models.tabnet import TabNetExpert
except Exception:  # pragma: no cover
    TabNetExpert = None

logger = logging.getLogger(__name__)
EPSILON = 1e-9
LABEL_FWD = {1: 1, -1: 2, 0: 0}  # Map signal to index


class SignalEngine:
    """Signal generation engine with ensemble of experts and models (inference-focused)."""

    def __init__(self, settings: Settings, strategy_ledger: Optional["StrategyLedger"] = None):
        self.config = settings
        self.settings = settings  # Alias

        if strategy_ledger is None:
            from ..core.storage import StrategyLedger

            self.strategy_ledger = StrategyLedger(settings.system.strategy_ledger_path)
        else:
            self.strategy_ledger = strategy_ledger

        self.models: dict[str, Any] = {}
        self.meta_blender = None
        self.feature_engineer = FeatureEngineer(settings)

        self._tfm = None
        self._rl = None
        self._sac = None
        self._evo = None
        self._nbeats = None
        self._tide = None
        self._tabnet = None
        self._kan = None

        self._scaler = None
        self._feature_columns = None
        self._expert_weights = None
        self._trade_prob_calibrator = None
        self._run_summary: dict[str, Any] = {}
        self._onnx_engine: ONNXInferenceEngine | None = None
        self._use_onnx = False  # Automatically enabled if available
        # Prop firm safety gates
        self._daily_dd_cutoff = float(getattr(settings.risk, "daily_drawdown_limit", 0.04) or 0.04)
        self._total_dd_cutoff = float(getattr(settings.risk, "max_drawdown_limit", 0.07) or 0.07)
        self._profit_target = float(getattr(settings.risk, "monthly_profit_target", 0.04) or 0.04)
        self._max_trades_per_day = int(getattr(settings.risk, "max_trades_per_day", 20) or 20)

    def prepare_dataset(
        self,
        frames: dict[str, pd.DataFrame],
        news_features: pd.DataFrame | None = None,
        order_book_features: pd.DataFrame | None = None,
    ) -> PreparedDataset:
        return self.feature_engineer.prepare(
            frames, news_features, order_book_features, symbol=self.settings.system.symbol
        )

    def _rule_expert_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Rule-based expert using EMA and RSI."""
        # Validate required columns exist
        required_cols = ["ema_fast", "ema_slow", "rsi"]
        missing = [col for col in required_cols if col not in X.columns]
        if missing:
            logger.warning(f"Rule expert missing columns: {missing}. Returning neutral probabilities.")
            probs = np.zeros((len(X), 3), dtype=float)
            probs[:, LABEL_FWD[0]] = 1.0  # All neutral
            return probs

        ema_f = X["ema_fast"].values
        ema_s = X["ema_slow"].values
        rsi = X["rsi"].values

        diff = ema_f - ema_s
        diff_std = np.std(diff)

        z = np.tanh((diff / (diff_std + EPSILON)) + (rsi - 50.0) / 10.0)

        p_up = (z + 1.0) / 2.0
        p_dn = 1.0 - p_up
        p_neutral = np.clip(1.0 - np.abs(z), 0.0, 1.0) * 0.2

        probs = np.zeros((len(X), 3), dtype=float)
        probs[:, LABEL_FWD[1]] = p_up * (1.0 - p_neutral)
        probs[:, LABEL_FWD[-1]] = p_dn * (1.0 - p_neutral)
        probs[:, LABEL_FWD[0]] = p_neutral

        return probs / (probs.sum(axis=1, keepdims=True) + EPSILON)

    def _compute_expert_weights(self, summary: dict[str, Any]) -> list[float] | None:
        """Derive expert weights from model_metrics in run_summary."""
        try:
            model_metrics = summary.get("model_metrics", {})
            if not model_metrics or not self.models:
                return None
            weights = []
            keys = list(self.models.keys())
            for name in keys:
                raw = model_metrics.get(name, {})
                if isinstance(raw, dict) and "fast" in raw and isinstance(raw.get("fast"), dict):
                    m_fast = raw.get("fast") or {}
                    m_prop = raw.get("prop") if isinstance(raw.get("prop"), dict) else {}
                else:
                    m_fast = {}
                    m_prop = raw if isinstance(raw, dict) else {}

                # Prefer fast backtest metrics (price-based SL/TP) when available.
                score = m_fast.get("net_profit", None)
                if score is None:
                    score = m_prop.get("pnl_score", None)

                # Penalize drawdown if available
                dd = None
                if "max_dd" in m_fast:
                    dd = float(m_fast.get("max_dd"))
                elif "max_dd_pct" in m_prop:
                    dd = float(m_prop.get("max_dd_pct"))
                else:
                    for dd_key in ("max_dd", "max_drawdown", "mdd"):
                        if dd_key in m_prop:
                            dd = float(m_prop.get(dd_key))
                            break
                dd_penalty = 1.0
                if dd is not None:
                    dd_penalty = max(0.1, 1.0 - dd / 0.20)  # fade weight if DD > 20%

                if score is not None:
                    # PnL based weighting: strict exclusion for losers
                    w = max(0.0, float(score)) * dd_penalty
                else:
                    # Accuracy based: strict exclusion for random/worse
                    acc = float(m_prop.get("accuracy", 0.0))
                    w = max(0.0, (acc - 0.5) * 2.0) * dd_penalty  # Scale 0.5->0, 1.0->1.0

                weights.append(w)

            wsum = sum(weights)
            if wsum <= 0:
                return None
            normalized = [w / wsum for w in weights]
            # Cap concentration so no single model dominates
            cap = 0.4
            capped = [min(w, cap) for w in normalized]
            capped_sum = sum(capped)
            if capped_sum <= 0:
                logger.warning("All weights capped to zero, using equal weighting")
                return normalized

            # Renormalize capped weights to sum to 1.0
            renormalized = [w / capped_sum for w in capped]

            # Verify sum is approximately 1.0
            final_sum = sum(renormalized)
            if abs(final_sum - 1.0) > 0.01:
                logger.warning(f"Expert weights sum to {final_sum:.3f} instead of 1.0, normalizing")
                return [w / final_sum for w in renormalized]

            return renormalized
        except Exception:
            return None

    def _momentum_expert_proba(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        """Momentum expert based on EMA crossover and slope."""
        close = meta["close"].astype(np.float32).values
        ema_f = X["ema_fast"].values
        ema_s = X["ema_slow"].values

        slope = np.concatenate([[0.0], np.diff(close)])

        ema_diff_std = np.std(ema_f - ema_s)
        slope_std = np.std(slope)

        z = np.tanh((ema_f - ema_s) / (ema_diff_std + EPSILON) + slope / (slope_std + EPSILON))

        p_up = (z + 1.0) / 2.0
        p_dn = 1.0 - p_up
        p_neu = np.clip(1.0 - np.abs(z), 0.0, 1.0) * 0.1

        probs = np.zeros((len(X), 3), dtype=float)
        probs[:, LABEL_FWD[1]] = p_up * (1.0 - p_neu)
        probs[:, LABEL_FWD[-1]] = p_dn * (1.0 - p_neu)
        probs[:, LABEL_FWD[0]] = p_neu

        return probs / (probs.sum(axis=1, keepdims=True) + EPSILON)

    def _mean_reversion_expert_proba(self, meta: pd.DataFrame) -> np.ndarray:
        """Mean reversion expert based on Bollinger bands."""
        close = meta["close"].astype(np.float32)
        ma = close.rolling(20).mean().ffill()
        sd = close.rolling(20).std().ffill() + EPSILON

        z = (close - ma) / sd
        zc = z.clip(-3.0, 3.0).values

        p_dn = np.clip((zc - 1.0) / 2.0, 0.0, 1.0)
        p_up = np.clip((-zc - 1.0) / 2.0, 0.0, 1.0)
        p_neu = 1.0 - np.maximum(p_up, p_dn)

        probs = np.zeros((len(meta), 3), dtype=float)
        probs[:, LABEL_FWD[1]] = p_up
        probs[:, LABEL_FWD[-1]] = p_dn
        probs[:, LABEL_FWD[0]] = p_neu

        return probs / (probs.sum(axis=1, keepdims=True) + EPSILON)

    @staticmethod
    def _softmax(p: np.ndarray) -> np.ndarray:
        """Softmax normalization with numerical stability."""
        p = np.asarray(p, dtype=float)
        return p / (p.sum(axis=1, keepdims=True) + EPSILON)

    @staticmethod
    def _pad_probs(probs: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
        """
        Normalize probability outputs to shape (n,3) ordered as [neutral, buy, sell].

        If `classes` is provided, it is interpreted as the class label for each column and used to reorder/map.
        Supported label conventions:
          - {-1, 0, 1}: -1=sell, 0=neutral, 1=buy
          - {0, 1, 2}: 0=neutral, 1=buy, 2=sell
        """
        if probs is None:
            return np.zeros((0, 3), dtype=float)
        arr = np.asarray(probs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n = arr.shape[0]

        # Fast path: already 3 columns and no mapping provided.
        if arr.shape[1] >= 3 and not classes:
            return arr[:, :3]

        out = np.zeros((n, 3), dtype=float)

        if classes and len(classes) == arr.shape[1]:
            for col, cls in enumerate(classes):
                if cls == 0:
                    out[:, 0] = arr[:, col]
                elif cls == 1:
                    out[:, 1] = arr[:, col]
                elif cls == -1:
                    out[:, 2] = arr[:, col]
                elif cls == 2:
                    out[:, 2] = arr[:, col]
            return out

        # Fallback mapping when class labels are unknown.
        if arr.shape[1] >= 3:
            return arr[:, :3]
        if arr.shape[1] == 2:
            # Assume [neutral, buy]
            out[:, 0] = arr[:, 0]
            out[:, 1] = arr[:, 1]
            return out
        # Assume a single probability corresponds to "buy" (neutral vs buy)
        out[:, 0] = 1.0 - arr[:, 0]
        out[:, 1] = arr[:, 0]
        return out

    def load_models(self, models_dir: str) -> None:
        """Load saved models from disk."""
        try:
            from pathlib import Path

            from joblib import load

            try:
                onnx_dir = Path(models_dir) / "onnx"
                if onnx_dir.exists() and (onnx_dir / "export_manifest.joblib").exists():
                    self._onnx_engine = ONNXInferenceEngine(str(onnx_dir))
                    self._onnx_engine.load_models()
                    if len(self._onnx_engine.sessions) > 0:
                        self._use_onnx = True
                        logger.info(f"ONNX inference enabled ({len(self._onnx_engine.sessions)} models)")
                    else:
                        self._use_onnx = False
                else:
                    self._use_onnx = False
            except Exception as e:
                logger.info(f"ONNX models not available: {e}")
                self._use_onnx = False

            try:
                from ..training.trainer import ModelTrainer

                trainer = ModelTrainer(self.settings)
                trainer.models_dir = Path(models_dir)
                trainer.load_models()
                if trainer.models:
                    run_summary = getattr(trainer, "run_summary", {}) or {}
                    exclude = set()
                    evo_meta = run_summary.get("evo_metrics", {})
                    if evo_meta.get("exclude"):
                        exclude.add("evolution")
                    filtered = {k: v for k, v in trainer.models.items() if k not in exclude}
                    self.models = filtered
                    self.meta_blender = trainer.meta_blender
                    self._run_summary = run_summary
                    self._expert_weights = self._compute_expert_weights(run_summary)
                    if "feature_columns" in run_summary and isinstance(run_summary["feature_columns"], list):
                        self._feature_columns = [str(c) for c in run_summary["feature_columns"]]
                    elif "feature_stats" in run_summary and isinstance(run_summary["feature_stats"], dict):
                        self._feature_columns = list(run_summary["feature_stats"].keys())
                    else:
                        self._feature_columns = None
                    return
            except Exception as e:
                logger.warning(f"Transformer model loading failed: {e}", exc_info=True)

            p = Path(models_dir) / "models.joblib"
            if load is None:
                logger.error("joblib.load is not available - cannot load models")
                raise RuntimeError("joblib module required for model loading")

            if not p.exists():
                # Not fatal: inference can still run via rule/momentum/mean-reversion experts,
                # and auxiliary models (RL/Evo/etc.) may be available as separate files.
                logger.warning(f"Models file not found at {p}. Continuing with built-in experts only.")
                payload = {}
            else:
                payload = load(p)
            self.models = payload.get("models", {})

            self._scaler = payload.get("scaler")
            self._feature_columns = payload.get("feature_columns")
            self._expert_weights = payload.get("expert_weights")
            self._trade_prob_calibrator = payload.get("trade_prob_calibrator")

            if TransformerExpertTorch is not None:
                try:
                    self._tfm = TransformerExpertTorch()
                    self._tfm.load(models_dir)
                    if getattr(self._tfm, "model", None) is None:
                        self._tfm = None
                except Exception as e:
                    logger.warning(f"Failed to load transformer: {e}")
                    self._tfm = None

            if RLExpertPPO is not None:
                try:
                    self._rl = RLExpertPPO(
                        timesteps=int(getattr(self.config.models, "rl_timesteps", 1_000_000)),
                        max_time_sec=int(getattr(self.config.models, "rl_train_seconds", 1800)),
                        network_arch=list(getattr(self.config.models, "rl_network_arch", [512, 512])),
                        device=getattr(self.config.system, "device", "cpu"),
                        parallel_envs=int(max(1, getattr(self.config.models, "rl_parallel_envs", 1))),
                    )
                    self._rl.load(models_dir)
                    if getattr(self._rl, "model", None) is None:
                        self._rl = None
                except Exception as e:
                    logger.warning(f"Failed to load RL agent: {e}")
                    self._rl = None

            if RLExpertSAC is not None:
                try:
                    self._sac = RLExpertSAC(
                        timesteps=int(getattr(self.config.models, "rl_timesteps", 1_000_000)),
                        max_time_sec=int(getattr(self.config.models, "rl_train_seconds", 1800)),
                        network_arch=list(getattr(self.config.models, "rl_network_arch", [256, 256, 256])),
                        device=getattr(self.config.system, "device", "cpu"),
                        parallel_envs=int(max(1, getattr(self.config.models, "rl_parallel_envs", 1))),
                    )
                    self._sac.load(models_dir)
                    if getattr(self._sac, "model", None) is None:
                        self._sac = None
                except Exception as e:
                    logger.warning(f"Failed to load SAC agent: {e}")
                    self._sac = None

            if EvoExpertCMA is not None:
                try:
                    self._evo = EvoExpertCMA(
                        max_time_sec=int(getattr(self.config.models, "evo_train_seconds", 3600)),
                        hidden_size=int(getattr(self.config.models, "evo_hidden_size", 64)),
                        population=int(getattr(self.config.models, "evo_population", 32)),
                        num_islands=int(getattr(self.config.models, "evo_islands", 4)),
                    )
                    self._evo.load(models_dir)
                    if getattr(self._evo, "theta", None) is None:
                        self._evo = None
                except Exception as e:
                    logger.warning(f"Failed to load neuroevolution: {e}")
                    self._evo = None

            if TabNetExpert is not None:
                try:
                    self._tabnet = TabNetExpert(
                        hidden_dim=64,
                        max_time_sec=int(getattr(self.config.models, "tabnet_train_seconds", 1800)),
                        device=getattr(self.config.system, "device", "cpu"),
                    )
                    self._tabnet.load(models_dir)
                    if getattr(self._tabnet, "model", None) is None:
                        self._tabnet = None
                except Exception as e:
                    logger.warning(f"Failed to load TabNet: {e}")
                    self._tabnet = None

            if KANExpert is not None:
                try:
                    self._kan = KANExpert(
                        hidden_dim=64,
                        max_time_sec=int(getattr(self.config.models, "kan_train_seconds", 1800)),
                        device=getattr(self.config.system, "device", "cpu"),
                    )
                    self._kan.load(models_dir)
                    if getattr(self._kan, "model", None) is None:
                        self._kan = None
                except Exception as e:
                    logger.warning(f"Failed to load KAN: {e}")
                    self._kan = None

            try:
                root = Path(models_dir)  # models_dir is typically inside model root
                path = root / "meta_blender.joblib"
                if path.exists():
                    self.meta_blender = MetaBlender.load(path)
            except Exception as e:
                logger.warning(f"Failed to load meta blender: {e}")
                self.meta_blender = None

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.models = {}
            self._scaler = None
            self._feature_columns = None
            self._expert_weights = None
            self._trade_prob_calibrator = None
            self._tfm = self._rl = self._sac = self._evo = None
            self.meta_blender = None

    async def train_all_models(self, *_, **__):
        """Deprecated training path; use ModelTrainer.train_all instead."""
        raise RuntimeError("Training is handled by ModelTrainer. SignalEngine is inference-only.")

    async def train_enhanced_models(self, *_, **__):
        """Deprecated training path; use ModelTrainer.train_all instead."""
        raise RuntimeError("Training is handled by ModelTrainer. SignalEngine is inference-only.")

    def get_latest_oof_predictions(self) -> dict[str, Any] | None:
        """Legacy stub for OOF predictions."""
        return None

    def generate_ensemble_signals(self, dataset: PreparedDataset) -> SignalResult:
        """Generate ensemble signals with all optimizations."""
        # Shallow copies to avoid doubling memory on large frames (10y data)
        X = dataset.X.copy(deep=False)
        X_full = dataset.X.copy(deep=False)  # Keep full features for rule/mean-reversion
        meta = dataset.metadata if dataset.metadata is not None else pd.DataFrame(index=X.index)
        probs_list: list[np.ndarray] = []
        components: list[tuple[str, np.ndarray]] = []

        if self._feature_columns is not None:
            for col in self._feature_columns:
                if col not in X:
                    X[col] = 0.0
            X = X[self._feature_columns]

        if self._evo:
            try:
                evo_probs = self._pad_probs(self._evo.predict_proba(X))
                evo_signal = evo_probs[:, 1] - evo_probs[:, 2]
                X["evo_signal"] = evo_signal
                if "evo_signal" not in dataset.X.columns:
                    dataset.X["evo_signal"] = (
                        evo_signal  # Ensure dataset.X has it for downstream models using dataset.X
                    )
                X_full["evo_signal"] = evo_signal  # Also add to X_full just in case
                probs_list.append(evo_probs)  # Still add to voting
                components.append(("evolution", evo_probs))
            except Exception as e:
                logger.warning(f"Evo signal injection failed: {e}")
                X["evo_signal"] = 0.0
                if "evo_signal" not in dataset.X.columns:
                    dataset.X["evo_signal"] = 0.0
                X_full["evo_signal"] = 0.0

        try:
            rule_p = self._rule_expert_proba(X_full)
            mom_p = self._momentum_expert_proba(X_full, meta)
            mean_p = self._mean_reversion_expert_proba(meta)
            probs_list.extend([rule_p, mom_p, mean_p])
            components.extend([("rule", rule_p), ("momentum", mom_p), ("mean_reversion", mean_p)])
        except Exception as e:
            logger.warning(f"Expert predictions failed: {e}")

        model_pred_store: dict[str, np.ndarray] = {}

        if self._use_onnx and self._onnx_engine:
            onnx_preds: dict[str, np.ndarray] = {}
            onnx_missing: set[str] = set()
            available = set(self._onnx_engine.model_names)

            for name in self._onnx_engine.model_names:
                if name == "evolution" or name == "meta_blender":
                    continue  # Already handled or handled later
                try:
                    classes = None
                    with contextlib.suppress(Exception):
                        classes = getattr(self._onnx_engine, "model_classes", {}).get(name)
                    if classes is None:
                        classes = [-1, 0, 1]  # common sklearn class order for y ∈ {-1,0,1}
                    probs = self._pad_probs(self._onnx_engine.predict_proba(name, X), classes=classes)
                    probs = self._softmax(probs)
                    onnx_preds[name] = probs
                except Exception as e:
                    logger.warning(f"ONNX {name} prediction failed: {e}")
                    onnx_missing.add(name)

            # All-or-nothing: if any ONNX model missing/fails, return neutral (no trade)
            if onnx_missing:
                logger.warning(f"ONNX missing/failed models: {sorted(onnx_missing)}; disabling trading signals.")
                return np.zeros((len(X), 3)), {"components": components}
            if set(onnx_preds.keys()) != available:
                missing = available - set(onnx_preds.keys())
                if missing:
                    logger.warning(f"ONNX models missing predictions: {sorted(missing)}; disabling trading signals.")
                    return np.zeros((len(X), 3)), {"components": components}

            for name, probs in onnx_preds.items():
                probs_list.append(probs)
                components.append((name, probs))
                model_pred_store[name] = probs

        else:
            for name, wrapper in self.models.items():
                if name == "evolution":
                    continue  # Already handled
                try:
                    pred_kwargs = {}
                    try:
                        sig = inspect.signature(wrapper.predict_proba)
                        if "metadata" in sig.parameters:
                            pred_kwargs["metadata"] = meta
                    except Exception as e:
                        logger.warning(f"Model prediction failed: {e}", exc_info=True)

                    probs = self._pad_probs(wrapper.predict_proba(X, **pred_kwargs))
                    probs = self._softmax(probs)
                    probs_list.append(probs)
                    components.append((name, probs))
                    model_pred_store[name] = probs
                except Exception as e:
                    logger.warning(f"{name} prediction failed: {e}")

        if self._tfm:
            try:
                p = self._pad_probs(self._tfm.predict_proba(X))  # Use X to include evo_signal
                probs_list.append(p)
                components.append(("transformer", p))
            except Exception as e:
                logger.warning(f"Transformer prediction failed: {e}")
        if self._rl:
            try:
                p = self._pad_probs(self._rl.predict_proba(X))
                probs_list.append(p)
                components.append(("rl_ppo", p))
            except Exception as e:
                logger.warning(f"RL PPO prediction failed: {e}")
        if self._sac:
            try:
                p = self._pad_probs(self._sac.predict_proba(X))
                probs_list.append(p)
                components.append(("rl_sac", p))
            except Exception as e:
                logger.warning(f"RL SAC prediction failed: {e}")
        if self._nbeats:
            try:
                p = self._pad_probs(self._nbeats.predict_proba(X))
                probs_list.append(p)
                components.append(("nbeats", p))
            except Exception as e:
                logger.warning(f"NBeats prediction failed: {e}")
        if self._tide:
            try:
                p = self._pad_probs(self._tide.predict_proba(X))
                probs_list.append(p)
                components.append(("tide", p))
            except Exception as e:
                logger.warning(f"TiDE prediction failed: {e}")
        if self._tabnet:
            try:
                p = self._pad_probs(self._tabnet.predict_proba(X))
                probs_list.append(p)
                components.append(("tabnet", p))
            except Exception as e:
                logger.warning(f"TabNet prediction failed: {e}")
        if self._kan:
            try:
                p = self._pad_probs(self._kan.predict_proba(X))
                probs_list.append(p)
                components.append(("kan", p))
            except Exception as e:
                logger.warning(f"KAN prediction failed: {e}")

        n = len(X)
        probs = np.zeros((n, 3), dtype=float)

        uncertainty = np.zeros(n, dtype=float)
        if len(probs_list) > 1:
            stacked = np.stack(probs_list, axis=0)
            std_devs = np.std(stacked, axis=0)
            uncertainty = np.mean(std_devs, axis=1)
            uncertainty = np.clip(uncertainty / 0.5, 0.0, 1.0)

        if self.meta_blender and model_pred_store:
            try:
                meta_features = pd.DataFrame(index=X.index)
                for name, preds in model_pred_store.items():
                    preds = self._pad_probs(preds)
                    meta_features[f"{name}_neutral"] = preds[:, 0]
                    meta_features[f"{name}_buy"] = preds[:, 1]
                    meta_features[f"{name}_sell"] = preds[:, 2]
                meta_features["symbol"] = self.config.system.symbol
                probs = self.meta_blender.predict_proba(meta_features)
            except Exception as e:
                logger.warning(f"Meta blender prediction failed, falling back to average: {e}")
                probs = np.mean(probs_list, axis=0) if probs_list else probs
        elif probs_list:
            if self._expert_weights:
                weight_map = dict(zip(self.models.keys(), self._expert_weights, strict=False))
                weighted_parts = []
                total_weight = 0.0
                baseline_parts = []
                for name, pr in components:
                    if name in weight_map:
                        w = weight_map[name]
                        weighted_parts.append(w * pr)
                        total_weight += w
                    else:
                        baseline_parts.append(pr)
                if total_weight > 0:
                    weighted_avg = np.sum(weighted_parts, axis=0) / total_weight
                    mix = baseline_parts + [weighted_avg]
                    probs = np.mean(mix, axis=0)
                else:
                    probs = np.mean([p for _, p in components], axis=0)
            else:
                probs = np.mean([p for _, p in components], axis=0)
            probs = self._softmax(probs)

        p_up = probs[:, LABEL_FWD[1]]
        p_dn = probs[:, LABEL_FWD[-1]]
        trade_prob = np.maximum(p_up, p_dn)

        # Concentration/uncertainty gating: reduce trade_prob when disagreement is high
        if len(probs_list) > 1:
            # reuse uncertainty from std dev above
            trade_prob = trade_prob * (1.0 - 0.5 * uncertainty)
        if self._expert_weights and len(probs_list) == len(self._expert_weights):
            max_w = max(self._expert_weights)
            if max_w > 0.6:
                trade_prob = trade_prob * (1.0 - (max_w - 0.6))

        # Correlation penalty: if models are highly correlated, reduce aggression
        if len(probs_list) > 1:
            try:
                # Limit to first 15 components to keep correlation O(n^2) manageable
                sample_probs = probs_list[:15]
                vote_matrix = []
                for pb in sample_probs:
                    vote_matrix.append(pb[:, LABEL_FWD[1]] - pb[:, LABEL_FWD[-1]])
                vote_matrix = np.stack(vote_matrix, axis=0)
                corr = np.corrcoef(vote_matrix)
                if corr.shape[0] > 1:
                    off_diag = np.abs(corr - np.eye(corr.shape[0]))
                    mean_corr = off_diag.sum() / (corr.shape[0] * (corr.shape[0] - 1))
                    if mean_corr > 0.7:
                        penalty = min(0.5, (mean_corr - 0.7))  # up to 50% reduction
                        trade_prob = trade_prob * (1.0 - penalty)
            except Exception as e:
                logger.warning(f"Correlation penalty calc failed: {e}", exc_info=True)

        reg_val = np.abs(X["ema_fast"] - X["ema_slow"]).values
        reg_thresh = np.percentile(reg_val, 60) if len(reg_val) > 0 else 0
        regimes = np.where(reg_val > reg_thresh, "trend", "neutral")

        base_conf = float(getattr(self.config.risk, "min_confidence_threshold", 0.55))
        thresholds = np.where(regimes == "trend", max(0.50, base_conf - 0.05), min(0.65, base_conf + 0.05))

        signals = np.where(trade_prob < thresholds, 0, np.where(p_up >= p_dn, 1, -1))

        win_prob = np.where(signals == 1, p_up, np.where(signals == -1, p_dn, 0.0))

        rr_values = np.full(n, 2.0)  # Default
        sl_pips_values = np.full(n, 20.0)  # Default fallback

        if "dist_liquidity" in X.columns and "atr" in X.columns:
            dist = X["dist_liquidity"].abs().values
            atr = X["atr"].values

            safe_atr = np.maximum(atr, 1e-6)
            est_stop_dist = 1.5 * safe_atr

            dynamic_rr = dist / est_stop_dist

            rr_values = np.clip(dynamic_rr, 1.0, 6.0)

            is_forex = np.nanmedian(atr) < 0.5
            pip_size = 0.0001 if is_forex else 0.01  # Rough guess, refined in bot.py
            sl_pips_values = est_stop_dist / pip_size

        rr_series = pd.Series(rr_values, index=X.index)
        sl_pips = pd.Series(sl_pips_values, index=X.index)

        return SignalResult(
            signals=pd.Series(signals, index=X.index),
            stacking_confidence=pd.Series(probs.max(axis=1), index=X.index),
            trade_probability=pd.Series(trade_prob, index=X.index),
            regimes=pd.Series(regimes, index=X.index),
            uncertainty=pd.Series(uncertainty, index=X.index),
            recommended_rr=rr_series,
            recommended_sl=sl_pips,
            win_probability=pd.Series(win_prob, index=X.index),
            signal=int(signals[-1]),
            confidence=float(trade_prob[-1]),
            probs=probs[-1].tolist(),
            model_votes={},
            regime=str(regimes[-1]),
            meta_features={},
            timestamp=X.index[-1],
        )

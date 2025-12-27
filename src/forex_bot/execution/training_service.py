import asyncio
import concurrent.futures
import json
import logging
import multiprocessing
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch.distributed as dist  # DDP Sync

from ..core.config import Settings
from ..data.loader import DataLoader
from ..data.news.client import get_sentiment_analyzer
from ..domain.events import PreparedDataset
from ..features.pipeline import FeatureEngineer
from ..strategy.discovery import AutonomousDiscoveryEngine
from ..training.trainer import ModelTrainer

logger = logging.getLogger(__name__)



class TrainingService:
    """
    Manages model training, feature engineering for training,
    and strategy discovery cycles.
    """

    def __init__(
        self,
        settings: Settings,
        data_loader: DataLoader,
        trainer: ModelTrainer,
        feature_engineer: FeatureEngineer,
        discovery_engine: AutonomousDiscoveryEngine,
        autotune_hints: Any,
    ):
        self.settings = settings
        self.data_loader = data_loader
        self.trainer = trainer
        self.feature_engineer = feature_engineer
        self.discovery_engine = discovery_engine
        self.autotune_hints = autotune_hints
        self._ray_started = False
        self._progress_path = self.trainer.models_dir / "global_incremental_progress.json"

    def _load_progress(self) -> set[str]:
        """Load completed-symbol list to allow resuming long incremental runs."""
        try:
            data = json.loads(Path(self._progress_path).read_text())
            if isinstance(data, list):
                return {str(s) for s in data}
        except FileNotFoundError:
            return set()
        except Exception as exc:
            logger.warning(f"Failed to load incremental progress: {exc}")
        return set()

    def _save_progress(self, completed: set[str]) -> None:
        try:
            Path(self._progress_path).write_text(json.dumps(sorted(completed)))
        except Exception as exc:
            logger.warning(f"Failed to persist incremental progress: {exc}")

    def _build_discovery_frames_for_tensor(
        self,
        frames: dict[str, pd.DataFrame],
        news_feats: pd.DataFrame | None,
        symbol: str | None,
        base_dataset: PreparedDataset | None = None,
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Build discovery frames with either base-TF propagation or full per-TF features."""
        full_tf = str(os.environ.get("FOREX_BOT_DISCOVERY_FULL_TF_FEATURES", "1") or "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        base_tf = self.settings.system.base_timeframe

        cfg_tfs = list(getattr(self.settings.system, "higher_timeframes", []) or [])
        if not cfg_tfs:
            cfg_tfs = list(getattr(self.settings.system, "required_timeframes", []) or [])
        timeframes = [base_tf] + cfg_tfs
        timeframes = [tf for tf in dict.fromkeys(timeframes) if tf in frames]
        for tf in frames.keys():
            if tf not in timeframes:
                timeframes.append(tf)
        if not timeframes:
            timeframes = list(frames.keys())

        if not full_tf:
            discovery_frames = frames.copy()
            if base_tf in discovery_frames and base_dataset is not None:
                rich_df = base_dataset.X.copy()
                if "close" not in rich_df.columns:
                    orig = discovery_frames[base_tf].reindex(rich_df.index).ffill()
                    for col in ["open", "high", "low", "close"]:
                        if col in orig.columns:
                            rich_df[col] = orig[col]
                discovery_frames[base_tf] = rich_df

            reference_df = discovery_frames.get(base_tf)
            aligned_frames = {}
            for tf in timeframes:
                if tf in frames and reference_df is not None:
                    aligned = reference_df.reindex(frames[tf].index).ffill().fillna(0.0)
                    aligned["open"] = frames[tf]["open"]
                    aligned["high"] = frames[tf]["high"]
                    aligned["low"] = frames[tf]["low"]
                    aligned["close"] = frames[tf]["close"]
                    aligned_frames[tf] = aligned
            aligned_frames = self._inject_discovery_mixer_signals(
                aligned_frames if aligned_frames else discovery_frames,
                base_tf=base_tf,
                per_tf=False,
            )
            return aligned_frames, timeframes

        # Full per-TF feature engineering (slow but pure)
        per_tf = {}
        for tf in timeframes:
            if tf not in frames:
                continue
            try:
                if base_dataset is not None and tf == base_tf:
                    rich_df = base_dataset.X.copy()
                else:
                    tf_settings = self.settings.model_copy()
                    tf_settings.system.base_timeframe = tf
                    fe = FeatureEngineer(tf_settings)
                    ds_tf = fe.prepare(frames, news_features=news_feats, symbol=symbol)
                    rich_df = ds_tf.X.copy()
                if "close" not in rich_df.columns:
                    orig = frames[tf].reindex(rich_df.index).ffill()
                    for col in ["open", "high", "low", "close"]:
                        if col in orig.columns:
                            rich_df[col] = orig[col]
                per_tf[tf] = rich_df
            except Exception as exc:
                logger.warning(f"Discovery per-TF feature gen failed for {tf}: {exc}", exc_info=True)

        if not per_tf:
            return frames.copy(), timeframes

        all_cols: list[str] = []
        for df in per_tf.values():
            all_cols.extend(list(df.columns))
        # Preserve order, de-duplicate
        all_cols = list(dict.fromkeys(all_cols))

        aligned_frames = {}
        for tf, df in per_tf.items():
            aligned = df.reindex(columns=all_cols).ffill().fillna(0.0)
            orig = frames[tf].reindex(aligned.index).ffill()
            for col in ["open", "high", "low", "close"]:
                if col in orig.columns:
                    aligned[col] = orig[col]
            aligned_frames[tf] = aligned

        aligned_frames = self._inject_discovery_mixer_signals(
            aligned_frames,
            base_tf=base_tf,
            per_tf=True,
        )
        return aligned_frames, timeframes

    def _inject_discovery_mixer_signals(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        base_tf: str,
        per_tf: bool,
    ) -> dict[str, pd.DataFrame]:
        use_mixer = str(os.environ.get("FOREX_BOT_DISCOVERY_USE_TALIB_MIXER", "1") or "1").strip().lower()
        if use_mixer not in {"1", "true", "yes", "on"}:
            return frames

        try:
            from ..features.talib_mixer import TALibStrategyMixer, TALIB_AVAILABLE
        except Exception as exc:
            logger.warning(f"Discovery: TA-Lib mixer import failed: {exc}")
            return frames
        if not TALIB_AVAILABLE:
            logger.warning("Discovery: TA-Lib not available; skipping mixer signals.")
            return frames

        try:
            n_strategies = int(os.environ.get("FOREX_BOT_DISCOVERY_MIXER_STRATEGIES", "24") or 24)
        except Exception:
            n_strategies = 24
        if n_strategies <= 0:
            return frames
        try:
            max_indicators = int(os.environ.get("FOREX_BOT_DISCOVERY_MIXER_MAX_INDICATORS", "20") or 20)
        except Exception:
            max_indicators = 20
        max_indicators = max(1, max_indicators)

        try:
            import torch

            mixer_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            mixer_device = "cpu"

        mixer = TALibStrategyMixer(
            device=mixer_device,
            use_volume_features=bool(getattr(self.settings.system, "use_volume_features", False)),
        )
        if not getattr(mixer, "available_indicators", []):
            logger.warning("Discovery: TA-Lib mixer has no available indicators; skipping.")
            return frames

        max_indicators = min(max_indicators, max(1, len(mixer.available_indicators)))
        genes = [mixer.generate_random_strategy(max_indicators=max_indicators) for _ in range(n_strategies)]

        def _apply(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            local = df.copy()
            cache = mixer.bulk_calculate_indicators(local, genes)
            for idx, gene in enumerate(genes):
                col = f"tmx_sig_{idx}"
                if col in local.columns:
                    continue
                try:
                    sig = mixer.compute_signals(local, gene, cache=cache)
                    if isinstance(sig, pd.Series):
                        local[col] = sig.to_numpy(dtype=np.float32, copy=False)
                    else:
                        local[col] = np.asarray(sig, dtype=np.float32)
                except Exception as exc:
                    logger.warning(f"Discovery: mixer signal {idx} failed: {exc}")
                    local[col] = np.zeros(len(local), dtype=np.float32)
            return local

        if per_tf:
            out = {}
            for tf, df in frames.items():
                out[tf] = _apply(df)
            logger.info(f"Discovery: Injected {n_strategies} TA-Lib mixer signals per TF.")
            return out

        if base_tf not in frames:
            return frames
        base_df = _apply(frames[base_tf])
        sig_cols = [c for c in base_df.columns if c.startswith("tmx_sig_")]
        out = {base_tf: base_df}
        for tf, df in frames.items():
            if tf == base_tf:
                continue
            if not sig_cols:
                out[tf] = df
                continue
            aligned = base_df[sig_cols].reindex(df.index).ffill().fillna(0.0)
            local = df.copy()
            for col in sig_cols:
                if col not in local.columns:
                    local[col] = aligned[col].to_numpy(dtype=np.float32, copy=False)
            out[tf] = local
        logger.info(f"Discovery: Injected {len(sig_cols)} TA-Lib mixer signals (base TF aligned).")
        return out

    @staticmethod
    def _parse_int_env(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None:
            return None
        try:
            return int(str(raw).strip())
        except Exception:
            return None

    def _infer_global_pool_cap_per_symbol(self, *, n_features: int, n_symbols: int) -> int | None:
        """
        Determine a safe per-symbol row cap for multi-symbol pooled training.

        Order of precedence:
        1) `FOREX_BOT_GLOBAL_MAX_ROWS_PER_SYMBOL` (int)
        2) `FOREX_BOT_GLOBAL_MAX_ROWS` (int) divided by symbol count
        3) Auto-fit to available RAM (conservative estimate)

        Set either env var to a value <= 0 to disable capping.
        """
        n_symbols = max(1, int(n_symbols))
        n_features = max(1, int(n_features))

        explicit_per_symbol = self._parse_int_env("FOREX_BOT_GLOBAL_MAX_ROWS_PER_SYMBOL")
        if explicit_per_symbol is None:
            explicit_per_symbol = int(
                getattr(getattr(self.settings, "models", None), "global_max_rows_per_symbol", 0) or 0
            )
        if explicit_per_symbol is not None and explicit_per_symbol <= 0:
            return None
        if explicit_per_symbol is not None and explicit_per_symbol > 0:
            return int(explicit_per_symbol)

        explicit_total = self._parse_int_env("FOREX_BOT_GLOBAL_MAX_ROWS")
        if explicit_total is None:
            explicit_total = int(getattr(getattr(self.settings, "models", None), "global_max_rows", 0) or 0)
        if explicit_total is not None and explicit_total <= 0:
            return None
        if explicit_total is not None and explicit_total > 0:
            return max(1, int(explicit_total) // n_symbols)

        # Auto cap based on available RAM. Keep conservative to avoid OOM on Windows.
        try:
            import psutil

            available = float(psutil.virtual_memory().available)
        except Exception:
            return None

        try:
            mem_frac = float(os.environ.get("FOREX_BOT_GLOBAL_POOL_MEM_FRAC", "0.10") or 0.10)
        except Exception:
            mem_frac = 0.10
        mem_frac = float(min(0.40, max(0.05, mem_frac)))

        try:
            overhead = float(os.environ.get("FOREX_BOT_GLOBAL_POOL_OVERHEAD", "3.0") or 3.0)
        except Exception:
            overhead = 3.0
        overhead = float(min(10.0, max(1.5, overhead)))

        bytes_per_row = float(n_features) * 4.0 * overhead  # float32 payload + pandas overhead factor
        budget = available * mem_frac
        total_rows = int(budget // max(bytes_per_row, 1.0))
        per_symbol = max(1, total_rows // n_symbols)

        try:
            floor = int(os.environ.get("FOREX_BOT_GLOBAL_MIN_ROWS_PER_SYMBOL", "50000") or 50000)
        except Exception:
            floor = 50000
        per_symbol = max(1, max(floor, per_symbol))
        return int(per_symbol)

    @staticmethod
    def _tail_dataset(ds: PreparedDataset, rows: int) -> PreparedDataset:
        if rows <= 0:
            return ds
        try:
            n = len(ds.X)
        except Exception:
            return ds
        if n <= rows:
            return ds

        def _tail(obj):
            try:
                if hasattr(obj, "iloc"):
                    return obj.iloc[-rows:]
                return obj[-rows:]
            except Exception:
                return obj

        X = _tail(ds.X)
        y = _tail(ds.y)
        labels = _tail(ds.labels) if ds.labels is not None else None
        meta = _tail(ds.metadata) if isinstance(ds.metadata, pd.DataFrame) else ds.metadata
        return PreparedDataset(
            X=X,
            y=y,
            index=getattr(X, "index", None),
            feature_names=list(getattr(X, "columns", ds.feature_names)),
            metadata=meta,
            labels=labels,
        )

    async def train(self, optimize: bool = True, stop_event: asyncio.Event | None = None) -> None:
        """Run the full training pipeline for the single active symbol."""
        symbol = self.settings.system.symbol
        logger.info(f"Starting training for {symbol}...")

        logger.info("Ensuring historical data...")
        has_data = await self.data_loader.ensure_history(symbol)
        if not has_data:
            logger.error(f"No data found for {symbol}. Cannot proceed with training.")
            return

        try:
            frames = await self.data_loader.get_training_data(symbol)
        except Exception as e:
            logger.error(f"Failed to load training data frames: {e}")
            return

        analyzer = None
        if self.settings.news.enable_news:
            try:
                analyzer = await get_sentiment_analyzer(self.settings)
            except Exception as exc:
                logger.warning(f"News analyzer unavailable: {exc}")

        # News Backfill (Best Effort)
        if analyzer is not None:
            try:
                base_tf = self.settings.system.base_timeframe
                base_df = frames.get(base_tf)
                if base_df is None:
                    base_df = frames.get("M1")

                if base_df is not None and hasattr(base_df, "empty") and not base_df.empty:
                    if "timestamp" in base_df.columns:
                        start_ts = pd.to_datetime(base_df["timestamp"].min(), utc=True, errors="coerce").to_pydatetime()
                        end_ts = pd.to_datetime(base_df["timestamp"].max(), utc=True, errors="coerce").to_pydatetime()
                    else:
                        start_ts = pd.to_datetime(base_df.index.min(), utc=True, errors="coerce").to_pydatetime()
                        end_ts = pd.to_datetime(base_df.index.max(), utc=True, errors="coerce").to_pydatetime()
                    analyzer.ensure_news_history(symbol, start_ts, end_ts, use_live_llm=False)
            except Exception as exc:
                logger.debug(f"News backfill skipped: {exc}")

        logger.info("Preparing feature set...")

        # Prepare News Features
        news_feats = None
        if analyzer is not None:
            try:
                currencies = [symbol[:3], symbol[3:]] if len(symbol) == 6 else []
                base_tf = self.settings.system.base_timeframe
                base_df = frames.get(base_tf)
                if base_df is None:
                    base_df = frames.get("M1")

                if base_df is not None and hasattr(base_df, "empty") and not base_df.empty:
                    if "timestamp" in base_df.columns:
                        start_ts = pd.to_datetime(base_df["timestamp"].min(), utc=True).to_pydatetime()
                        end_ts = pd.to_datetime(base_df["timestamp"].max(), utc=True).to_pydatetime()
                        base_idx = pd.to_datetime(base_df["timestamp"], utc=True)
                    else:
                        start_ts = pd.to_datetime(base_df.index.min(), utc=True).to_pydatetime()
                        end_ts = pd.to_datetime(base_df.index.max(), utc=True).to_pydatetime()
                        base_idx = pd.to_datetime(base_df.index, utc=True)

                    events = analyzer.db.fetch_events(start_ts, end_ts, currencies=currencies)
                    if events is not None and len(events) > 0:
                        news_feats = analyzer.build_features(events, base_idx)
                        if news_feats is not None and len(news_feats) > 0:
                            logger.info(f"Generated news features for {symbol}: {news_feats.shape}")
                        else:
                            news_feats = None
            except Exception as e:
                logger.warning(f"Failed to build training news features: {e}")

        dataset = self.feature_engineer.prepare(frames, news_features=news_feats, symbol=symbol)

        # Quick runtime estimate for visibility
        try:
            est_seconds = self.trainer.estimate_time_for_dataset(
                dataset.X,
                len(dataset.X),
                context="full",
                simulate_gpu=self.settings.system.device if self.settings.system.enable_gpu else None,
            )
            logger.info(
                f"[ESTIMATE] ~{est_seconds / 3600:.2f} hours expected for training {symbol} ({len(dataset.X)} rows)."
            )
        except Exception:
            pass

        logger.info("Launching GPU-Native Expert Discovery...")
        from ..strategy.discovery_tensor import TensorDiscoveryEngine

        discovery_frames, timeframes = self._build_discovery_frames_for_tensor(
            frames,
            news_feats,
            symbol,
            base_dataset=dataset,
        )
        if "M1" in discovery_frames:
            logger.debug(f"M1 Columns: {len(discovery_frames['M1'].columns)}")

        # New Unsupervised Tensor Engine (Million-Search)
        # We increase experts to 100 to create a diverse "Council of 100" for the deep models
        discovery_tensor = TensorDiscoveryEngine(
            device="cuda",
            n_experts=100,
            timeframes=timeframes,
            max_rows=int(getattr(self.settings.system, "discovery_max_rows", 0) or 0),
            stream_mode=bool(getattr(self.settings.system, "discovery_stream", False)),
            auto_cap=bool(getattr(self.settings.system, "discovery_auto_cap", True)),
        )
        
        # We need to pass the enriched multi-timeframe frames to the engine
        discovery_tensor.run_unsupervised_search(
            discovery_frames, 
            news_features=news_feats,
            iterations=1000
        )
        discovery_tensor.save_experts(self.settings.system.cache_dir + "/tensor_knowledge.pt")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(asyncio.to_thread(self.trainer.train_all, dataset, optimize, stop_event))


        logger.info("Training cycle complete.")
        self._maybe_stop_ray()

    async def train_incremental_all(self, optimize: bool = False, stop_event: asyncio.Event | None = None) -> None:
        """
        Best-effort incremental retraining for the active symbol.
        Runs in a thread to avoid blocking the event loop when triggered from the trading loop.
        """
        symbol = self.settings.system.symbol
        logger.info(f"Starting incremental retraining for drift recovery on {symbol}...")

        try:
            frames = await self.data_loader.get_training_data(symbol)
            if not frames:
                raise RuntimeError("No training data available for incremental retrain")

            dataset = self.feature_engineer.prepare(frames, symbol=symbol)

            # Offload synchronous training work to a thread to keep async loop responsive
            await asyncio.to_thread(self.trainer.train_incremental, dataset, symbol, optimize, stop_event)
            logger.info("Incremental retraining finished.")
        except Exception as exc:
            logger.error(f"Incremental retraining failed: {exc}", exc_info=True)
            raise

    async def train_global(
        self, symbols: list[str], optimize: bool = True, stop_event: asyncio.Event | None = None
    ) -> None:
        """Train one global model across all provided symbols (Sequential or HPC)."""
        logger.info(f"Starting global training for symbols: {symbols}")
        orig_symbol = self.settings.system.symbol

        is_hpc = getattr(self.autotune_hints, "is_hpc", False)

        if is_hpc:
            await self._train_global_hpc(symbols, optimize, stop_event)
        else:
            await self._train_global_sequential(symbols, optimize, stop_event)

        self.settings.system.symbol = orig_symbol
        self._maybe_stop_ray()

    def _build_news_features(self, analyzer: Any, symbol: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """Best-effort build of time-aligned news features for a symbol using the local news DB."""
        if analyzer is None:
            return None
        try:
            currencies = [symbol[:3], symbol[3:]] if isinstance(symbol, str) and len(symbol) == 6 else []
            base_tf = self.settings.system.base_timeframe
            base_df = frames.get(base_tf)
            if base_df is None:
                base_df = frames.get("M1")
            if base_df is None or not hasattr(base_df, "empty") or base_df.empty:
                return None

            if "timestamp" in base_df.columns:
                base_idx = pd.to_datetime(base_df["timestamp"], utc=True, errors="coerce")
                start_ts = pd.to_datetime(base_df["timestamp"].min(), utc=True, errors="coerce").to_pydatetime()
                end_ts = pd.to_datetime(base_df["timestamp"].max(), utc=True, errors="coerce").to_pydatetime()
            else:
                base_idx = pd.to_datetime(base_df.index, utc=True, errors="coerce")
                start_ts = pd.to_datetime(base_df.index.min(), utc=True, errors="coerce").to_pydatetime()
                end_ts = pd.to_datetime(base_df.index.max(), utc=True, errors="coerce").to_pydatetime()

            # Optional: rescore existing archive/DB events so training uses better sentiment/confidence.
            if bool(getattr(self.settings.news, "auto_rescore_enabled", False)):
                try:
                    days = int(getattr(self.settings.news, "auto_rescore_days", 30) or 0)
                    start_r = start_ts
                    if days > 0:
                        start_r = max(start_ts, end_ts - timedelta(days=days))
                    max_events = int(getattr(self.settings.news, "auto_rescore_max_events", 200) or 200)
                    only_missing = bool(getattr(self.settings.news, "auto_rescore_only_missing", True))
                    rescored = analyzer.rescore_existing_events(
                        symbol,
                        start_r,
                        end_ts,
                        max_events=max_events,
                        only_missing=only_missing,
                    )
                    # Safe check to prevent "The truth value of a DataFrame is ambiguous" error
                    has_rescored = False
                    count_str = "0"

                    if rescored is not None:
                        if isinstance(rescored, int) and rescored > 0:
                            has_rescored = True
                            count_str = str(rescored)
                        elif isinstance(rescored, list) and len(rescored) > 0:
                            has_rescored = True
                            count_str = str(len(rescored))
                        elif hasattr(rescored, "empty"):
                            # Handle DataFrame/Series
                            if not rescored.empty:
                                has_rescored = True
                                count_str = str(len(rescored))
                        elif hasattr(rescored, "size") and getattr(rescored, "size", 0) > 0:
                            # Handle numpy array or similar
                            has_rescored = True
                            count_str = str(rescored.size)
                        else:
                            try:
                                if rescored:
                                    has_rescored = True
                                    count_str = "some"
                            except ValueError:
                                # "The truth value of ... is ambiguous"
                                has_rescored = True  # Assume if it's ambiguous, it's a non-empty container
                                count_str = "unknown"

                    if has_rescored:
                        logger.info(f"Rescored {count_str} news events for {symbol} (lookback={days}d).")
                except Exception as exc:
                    logger.debug(f"News rescore skipped for {symbol}: {exc}")

            events = analyzer.db.fetch_events(start_ts, end_ts, currencies=currencies)

            # Safe check for events (list or DataFrame)
            has_events = False
            if events is not None:
                if isinstance(events, list):
                    has_events = len(events) > 0
                elif hasattr(events, "empty"):
                    has_events = not events.empty
                else:
                    try:
                        has_events = len(events) > 0
                    except Exception:
                        has_events = bool(events)

            if not has_events:
                return None

            nf = analyzer.build_features(events, base_idx)

            # Safe check for features (DataFrame)
            has_features = False
            if nf is not None:
                if hasattr(nf, "empty"):
                    has_features = not nf.empty
                else:
                    try:
                        has_features = len(nf) > 0
                    except Exception:
                        has_features = bool(nf)

            return nf if has_features else None
        except Exception as exc:
            logger.debug(f"News feature build skipped for {symbol}: {exc}")
            return None

    @staticmethod
    def _aggregate_metrics(
        metrics_list: list[dict[str, Any]],
        *,
        numeric_keys: list[str],
        bool_any_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        if not metrics_list:
            return {}

        def _num(key: str) -> list[float]:
            vals: list[float] = []
            for m in metrics_list:
                v = m.get(key)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.append(float(v))
            return vals

        agg: dict[str, Any] = {}
        for key in numeric_keys:
            vals = _num(key)
            if not vals:
                continue
            agg[key] = float(np.mean(vals))

        for key in bool_any_keys or []:
            agg[key] = any(bool(m.get(key)) for m in metrics_list)
        return agg

    def _align_global_feature_space(
        self, datasets: list[tuple[str, PreparedDataset]]
    ) -> tuple[list[str], list[tuple[str, PreparedDataset]]]:
        all_cols: set[str] = set()
        for _, d in datasets:
            try:
                all_cols.update([str(c) for c in d.X.columns])
            except Exception:
                continue
        cols = sorted(all_cols)

        aligned: list[tuple[str, PreparedDataset]] = []
        for sym, d in datasets:
            try:
                X = d.X.reindex(columns=cols, fill_value=0.0).astype(np.float32, copy=False)
                y = d.y
                if not isinstance(y, pd.Series):
                    y = pd.Series(y, index=X.index)
                meta = d.metadata
                if isinstance(meta, pd.DataFrame) and len(meta) != len(X):
                    meta = meta.reindex(X.index)

                aligned.append(
                    (
                        sym,
                        PreparedDataset(
                            X=X,
                            y=y,
                            index=X.index,
                            feature_names=cols,
                            metadata=meta,
                            labels=y,
                        ),
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to align dataset for {sym}: {exc}")
        return cols, aligned

    def _split_global_train_eval(
        self,
        datasets: list[tuple[str, PreparedDataset]],
        *,
        train_ratio: float,
        embargo_bars: int,
        min_train_rows: int = 1000,
        min_eval_rows: int = 500,
    ) -> tuple[list[tuple[str, PreparedDataset]], dict[str, PreparedDataset], dict[str, Any]]:
        # Compute a global time cutoff so no symbol trains on "future" relative to any other.
        times: list[np.ndarray] = []
        for _, d in datasets:
            idx = d.index
            if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
                times.append(idx.view("int64"))
        if not times:
            raise RuntimeError("Global split failed: no timestamps found.")

        all_times = np.concatenate(times)
        all_times.sort()
        cut_i = int(max(0, min(len(all_times) - 1, int(len(all_times) * train_ratio) - 1)))
        cutoff_ts = pd.to_datetime(int(all_times[cut_i]), utc=True)

        train_parts: list[tuple[str, PreparedDataset]] = []
        eval_map: dict[str, PreparedDataset] = {}
        split_meta: dict[str, Any] = {
            "cutoff": cutoff_ts.isoformat(),
            "embargo_bars": int(embargo_bars),
            "per_symbol": {},
        }

        for sym, d in datasets:
            X = d.X
            y = d.y if isinstance(d.y, pd.Series) else pd.Series(d.y, index=X.index)
            meta = d.metadata
            n = len(X)
            if n == 0:
                continue

            idx = X.index
            try:
                cut_right = int(idx.searchsorted(cutoff_ts, side="right"))
            except Exception:
                cut_right = int(n * train_ratio)

            train_end = max(0, cut_right - max(0, int(embargo_bars)))
            eval_start = cut_right

            if train_end < min_train_rows or (n - eval_start) < min_eval_rows:
                # Fallback to per-symbol ratio split if the global cutoff is too skewed for this symbol.
                cut_right = int(n * train_ratio)
                train_end = max(0, cut_right - max(0, int(embargo_bars)))
                eval_start = cut_right

            if train_end < max(100, min_train_rows // 5) or (n - eval_start) < max(200, min_eval_rows // 3):
                logger.warning(
                    f"Skipping {sym}: insufficient rows after split (train={train_end}, eval={n - eval_start})."
                )
                continue

            train_ds = PreparedDataset(
                X=X.iloc[:train_end],
                y=y.iloc[:train_end],
                index=X.index[:train_end],
                feature_names=list(X.columns),
                metadata=meta.iloc[:train_end] if isinstance(meta, pd.DataFrame) else None,
                labels=y.iloc[:train_end],
            )
            eval_ds = PreparedDataset(
                X=X.iloc[eval_start:],
                y=y.iloc[eval_start:],
                index=X.index[eval_start:],
                feature_names=list(X.columns),
                metadata=meta.iloc[eval_start:] if isinstance(meta, pd.DataFrame) else None,
                labels=y.iloc[eval_start:],
            )

            train_parts.append((sym, train_ds))
            eval_map[sym] = eval_ds
            split_meta["per_symbol"][sym] = {"train_rows": int(len(train_ds.X)), "eval_rows": int(len(eval_ds.X))}

        return train_parts, eval_map, split_meta

    async def _train_global_from_datasets(
        self,
        datasets: list[tuple[str, PreparedDataset]],
        symbols: list[str],
        optimize: bool,
        stop_event: asyncio.Event | None,
    ) -> None:
        if not datasets:
            logger.error("Global training: no datasets provided.")
            return

        # Apply an additional cap here (covers HPC path and any callers that didn't cap during dataset creation).
        try:
            first = next((d for _sym, d in datasets if getattr(d, "X", None) is not None), None)
            n_features = int(getattr(first.X, "shape", (0, 0))[1]) if first is not None else 0
        except Exception:
            n_features = 0
        cap = self._infer_global_pool_cap_per_symbol(n_features=n_features, n_symbols=len(datasets))
        if cap is not None:
            capped: list[tuple[str, PreparedDataset]] = []
            for sym, d in datasets:
                try:
                    if len(d.X) > cap:
                        d = self._tail_dataset(d, cap)
                except Exception:
                    pass
                capped.append((sym, d))
            datasets = capped

        # Align feature spaces across symbols.
        cols, aligned = self._align_global_feature_space(datasets)
        if not aligned:
            logger.error("Global training: all datasets failed to align.")
            return

        train_ratio = float(getattr(self.settings.models, "global_train_ratio", 0.8) or 0.8)
        train_ratio = float(min(0.95, max(0.50, train_ratio)))
        embargo_bars = int(
            max(
                int(getattr(self.settings.risk, "meta_label_max_hold_bars", 0) or 0),
                int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0),
            )
        )

        train_parts, eval_map, split_meta = self._split_global_train_eval(
            aligned, train_ratio=train_ratio, embargo_bars=embargo_bars
        )
        if not train_parts:
            logger.error("Global training: no train splits produced.")
            return
        if not eval_map:
            logger.warning("Global training: no eval splits produced; metrics will be limited.")

        # Pool training data.
        X_parts: list[pd.DataFrame] = []
        y_parts: list[pd.Series] = []
        pooled_meta: list[pd.DataFrame] = []
        for sym, d in train_parts:
            X_part = d.X
            y_part = d.y if isinstance(d.y, pd.Series) else pd.Series(d.y, index=X_part.index)
            X_parts.append(X_part)
            y_parts.append(y_part)

            if d.metadata is not None and isinstance(d.metadata, pd.DataFrame) and len(d.metadata) == len(X_part):
                try:
                    m = d.metadata[["high", "low", "close"]].copy()
                    m["symbol"] = sym
                    pooled_meta.append(m)
                except Exception:
                    pass

        X_train = pd.concat(X_parts, axis=0, copy=False)
        y_train = pd.concat(y_parts, axis=0, copy=False).astype(int, copy=False)
        if not X_train.index.equals(y_train.index):
            raise ValueError(
                f"Training data index misalignment: X has {len(X_train)} rows, "
                f"y has {len(y_train)} rows, indices don't match"
            )
        X_train = X_train.astype(np.float32, copy=False)

        meta_train: pd.DataFrame | None = None
        if pooled_meta:
            try:
                meta_train = pd.concat(pooled_meta, axis=0, copy=False)
                # Ensure 1:1 alignment with pooled training rows.
                if len(meta_train) != len(X_train) or not meta_train.index.equals(X_train.index):
                    logger.warning(
                        "Global training: pooled metadata index misaligned; disabling metadata for optimizer."
                    )
                    meta_train = None
                else:
                    meta_train = meta_train.astype(
                        {"high": np.float32, "low": np.float32, "close": np.float32},
                        copy=False,
                    )
                    meta_train["symbol"] = meta_train["symbol"].astype("category")
            except Exception:
                meta_train = None

        full_ds = PreparedDataset(
            X=X_train,
            y=y_train,
            index=X_train.index,
            feature_names=list(X_train.columns),
            # Provide symbol-aware OHLC metadata so HPO can score profitability by symbol.
            # The trainer guards against leaking multi-symbol metadata into models that require single-series OHLC.
            metadata=meta_train,
            labels=y_train,
        )

        # Ensure the primary symbol is stable for any symbol-dependent utilities.
        if symbols:
            self.settings.system.symbol = symbols[0]

        logger.info(
            f"GLOBAL: Training pooled dataset (symbols={len(train_parts)}, rows={len(full_ds.X):,}, "
            f"features={len(full_ds.feature_names)})"
        )
        await asyncio.to_thread(self.trainer.train_all, full_ds, optimize, stop_event)

        # Post-train: evaluate each trained model out-of-sample per symbol.
        if stop_event and stop_event.is_set():
            return

        try:
            import inspect

            from ..strategy.fast_backtest import (
                fast_evaluate_strategy,
                infer_pip_metrics,
                infer_sl_tp_pips_auto,
            )
            from ..training.evaluation import probs_to_signals, prop_backtest

            model_metrics: dict[str, Any] = {}
            for name, model in (self.trainer.models or {}).items():
                per_symbol: dict[str, Any] = {}
                for sym, ds_eval in eval_map.items():
                    X_e = ds_eval.X.reindex(columns=cols, fill_value=0.0).astype(np.float32)
                    meta_e = ds_eval.metadata
                    if meta_e is None or not isinstance(meta_e, pd.DataFrame) or len(meta_e) != len(X_e):
                        continue

                    pred_kwargs: dict[str, Any] = {}
                    try:
                        sig = inspect.signature(model.predict_proba)
                        if "metadata" in sig.parameters:
                            pred_kwargs["metadata"] = meta_e
                    except Exception:
                        pass

                    try:
                        probs = model.predict_proba(X_e, **pred_kwargs)
                        sig_arr = probs_to_signals(np.asarray(probs))
                        sig_s = pd.Series(sig_arr, index=X_e.index)

                        prop = prop_backtest(
                            meta_e,
                            sig_s,
                            max_daily_dd_pct=float(getattr(self.settings.risk, "daily_drawdown_limit", 0.05)),
                            daily_dd_warn_pct=float(getattr(self.settings.risk, "daily_drawdown_limit", 0.05)) * 0.8,
                            max_trades_per_day=int(getattr(self.settings.risk, "max_trades_per_day", 10)),
                            use_gpu=False,
                        )

                        idx = meta_e.index
                        if not isinstance(idx, pd.DatetimeIndex):
                            idx = pd.to_datetime(idx, utc=True, errors="coerce")
                        month_idx = (idx.year.astype(np.int32) * 12 + idx.month.astype(np.int32)).to_numpy(
                            dtype=np.int64
                        )
                        day_idx = (
                            idx.year.astype(np.int32) * 10000
                            + idx.month.astype(np.int32) * 100
                            + idx.day.astype(np.int32)
                        ).to_numpy(dtype=np.int64)

                        pip_size, pip_value_per_lot = infer_pip_metrics(sym)
                        sl_cfg = getattr(self.settings.risk, "meta_label_sl_pips", None)
                        tp_cfg = getattr(self.settings.risk, "meta_label_tp_pips", None)
                        rr = float(getattr(self.settings.risk, "min_risk_reward", 2.0))
                        if sl_cfg is None or float(sl_cfg) <= 0:
                            atr_vals = meta_e["atr"].to_numpy(dtype=np.float64) if "atr" in meta_e.columns else None
                            auto = infer_sl_tp_pips_auto(
                                open_prices=meta_e["open"].to_numpy(dtype=np.float64)
                                if "open" in meta_e.columns
                                else meta_e["close"].to_numpy(dtype=np.float64),
                                high_prices=meta_e["high"].to_numpy(dtype=np.float64),
                                low_prices=meta_e["low"].to_numpy(dtype=np.float64),
                                close_prices=meta_e["close"].to_numpy(dtype=np.float64),
                                atr_values=atr_vals,
                                pip_size=pip_size,
                                atr_mult=float(getattr(self.settings.risk, "atr_stop_multiplier", 1.5)),
                                min_rr=rr,
                                min_dist=float(getattr(self.settings.risk, "meta_label_min_dist", 0.0)),
                                settings=self.settings,
                            )
                            if auto is None:
                                raise RuntimeError("Cannot infer SL/TP pips from metadata.")
                            sl_pips, tp_pips = auto
                        else:
                            sl_pips = float(sl_cfg)
                            if tp_cfg is None or float(tp_cfg) <= 0:
                                tp_pips = sl_pips * rr
                            else:
                                tp_pips = max(float(tp_cfg), sl_pips * rr)

                        spread = float(getattr(self.settings.risk, "backtest_spread_pips", 1.5))
                        commission = float(getattr(self.settings.risk, "commission_per_lot", 0.0))
                        max_hold = int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0)
                        trailing_enabled = bool(getattr(self.settings.risk, "trailing_enabled", False))
                        trailing_mult = float(getattr(self.settings.risk, "trailing_atr_multiplier", 1.0) or 1.0)
                        trailing_trigger_r = float(getattr(self.settings.risk, "trailing_be_trigger_r", 1.0) or 1.0)

                        arr = fast_evaluate_strategy(
                            close_prices=meta_e["close"].to_numpy(dtype=np.float64),
                            high_prices=meta_e["high"].to_numpy(dtype=np.float64),
                            low_prices=meta_e["low"].to_numpy(dtype=np.float64),
                            signals=sig_s.to_numpy(dtype=np.int8),
                            month_indices=month_idx,
                            day_indices=day_idx,
                            sl_pips=sl_pips,
                            tp_pips=tp_pips,
                            max_hold_bars=max_hold,
                            trailing_enabled=trailing_enabled,
                            trailing_atr_multiplier=trailing_mult,
                            trailing_be_trigger_r=trailing_trigger_r,
                            pip_value=pip_size,
                            spread_pips=spread,
                            commission_per_trade=commission,
                            pip_value_per_lot=pip_value_per_lot,
                        )
                        keys = [
                            "net_profit",
                            "sharpe",
                            "sortino",
                            "max_dd",
                            "win_rate",
                            "profit_factor",
                            "expectancy",
                            "sqn",
                            "trades",
                            "consistency_score",
                            "daily_dd",
                        ]
                        fast = {k: float(v) for k, v in zip(keys, arr.tolist(), strict=False)}

                        per_symbol[sym] = {"prop": prop, "fast": fast}
                    except Exception as exc:
                        logger.debug(f"Global eval failed for {name} on {sym}: {exc}")
                        continue

                prop_list = [v.get("prop", {}) for v in per_symbol.values() if isinstance(v, dict)]
                fast_list = [v.get("fast", {}) for v in per_symbol.values() if isinstance(v, dict)]

                agg_prop = self._aggregate_metrics(
                    prop_list,
                    numeric_keys=["pnl_score", "win_rate", "max_dd_pct", "trades"],
                    bool_any_keys=["daily_dd_violation", "trade_limit_violation"],
                )
                agg_fast = self._aggregate_metrics(
                    fast_list,
                    numeric_keys=[
                        "net_profit",
                        "sharpe",
                        "sortino",
                        "max_dd",
                        "win_rate",
                        "profit_factor",
                        "expectancy",
                        "sqn",
                        "trades",
                        "consistency_score",
                        "daily_dd",
                    ],
                )

                model_metrics[name] = {"prop": agg_prop, "fast": agg_fast, "per_symbol": per_symbol}

            self.trainer.run_summary["global_training"] = {
                "symbols": list(symbols),
                "train_ratio": float(train_ratio),
                "feature_columns": list(cols),
                **split_meta,
            }
            if model_metrics:
                self.trainer.run_summary["model_metrics"] = model_metrics
            self.trainer.persistence.save_run_summary(self.trainer.run_summary)
        except Exception as exc:
            logger.warning(f"Global post-train evaluation skipped: {exc}", exc_info=True)

    async def _train_global_hpc(self, symbols: list[str], optimize: bool, stop_event: asyncio.Event | None) -> None:
        logger.info("?? HPC Mode Active: Switching to Parallel Global Training.")

        # Single-process mode for stability (Avoids NCCL timeouts)
        rank = 0
        world_size = 1

        # Cache path for data persistence
        cache_path = Path(self.settings.system.cache_dir) / "hpc_datasets.pkl"
        datasets: list[tuple[str, PreparedDataset]] = []
        datasets_loaded = False

        # --- Data Loading & Feature Engineering ---
        reuse_cache = str(os.environ.get("FOREX_BOT_HPC_DATASET_CACHE", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
        if reuse_cache and cache_path.exists():
            try:
                cached = joblib.load(cache_path)
                if isinstance(cached, list) and cached:
                    datasets = cached
                    datasets_loaded = True
                    logger.info(f"HPC: Loaded cached datasets from {cache_path}.")
            except Exception as e:
                logger.warning(f"HPC: Failed to load cached datasets: {e}")

        if not datasets_loaded:
            raw_frames_map = {}
            news_map: dict[str, pd.DataFrame | None] = {}
            
            analyzer = await get_sentiment_analyzer(self.settings) if self.settings.news.enable_news else None

            # 1. Parallel loading of raw data (using asyncio.gather for speed)
            logger.info(f"HPC: Loading raw data for {len(symbols)} symbols in parallel...")
            
            async def _load_single(s):
                await self.data_loader.ensure_history(s)
                f = await self.data_loader.get_training_data(s)
                n = self._build_news_features(analyzer, s, f) if analyzer else None
                return s, f, n

            load_results = await asyncio.gather(*[_load_single(s) for s in symbols])
            for sym, f, n in load_results:
                raw_frames_map[sym] = f
                news_map[sym] = n
                logger.info(f"HPC: Ready data for {sym}")

            # 2. Multiprocessing Feature Engineering (Use 200+ cores)
            max_workers = int(os.environ.get("FOREX_BOT_FEATURE_WORKERS", 200))
            logger.info(f"HPC: Launching feature engineering on {max_workers} workers...")
            
            spawn_ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as executor:
                futures = {executor.submit(_hpc_feature_worker, self.settings.model_copy(), raw_frames_map[sym], sym, news_map.get(sym)): sym for sym in symbols}
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    if res: datasets.append((futures[fut], res))

            # Save cache for next time
            if reuse_cache:
                joblib.dump(datasets, cache_path)

        # --- Launch Discovery (Uses internal 8-GPU ThreadPool) ---
        logger.info("Launching GPU-Native Expert Discovery (Multi-GPU Pool)...")
        from ..strategy.discovery_tensor import TensorDiscoveryEngine
        
        target_sym = "EURUSD" if "EURUSD" in [s for s, _ in datasets] else datasets[0][0]
        target_ds = next(ds for sym, ds in datasets if sym == target_sym)
        
        # Load raw frame for OHLC data
        await self.data_loader.ensure_history(target_sym)
        raw_frames = await self.data_loader.get_training_data(target_sym)

        discovery_frames, timeframes = self._build_discovery_frames_for_tensor(
            raw_frames, None, target_sym, base_dataset=target_ds
        )

        discovery_tensor = TensorDiscoveryEngine(n_experts=100, timeframes=timeframes)
        discovery_tensor.run_unsupervised_search(discovery_frames, iterations=1000)
        discovery_tensor.save_experts(self.settings.system.cache_dir + "/tensor_knowledge.pt")

        # --- Final Global Training ---
        await self._train_global_from_datasets(datasets, symbols, optimize, stop_event)

    async def _train_global_sequential(
        self, symbols: list[str], optimize: bool, stop_event: asyncio.Event | None
    ) -> None:
        logger.info(
            "GLOBAL: Building pooled dataset sequentially (feature engineering per symbol, then one pooled train)."
        )
        datasets: list[tuple[str, PreparedDataset]] = []
        total = len(symbols)

        analyzer = None
        if self.settings.news.enable_news:
            try:
                analyzer = await get_sentiment_analyzer(self.settings)
            except Exception as exc:
                logger.warning(f"News analyzer unavailable (global): {exc}")

        for idx, sym in enumerate(symbols, start=1):
            if stop_event and stop_event.is_set():
                break

            logger.info(f"[GLOBAL {idx}/{total}] Preparing features for {sym}...")
            try:
                self.settings.system.symbol = sym
                has_data = await self.data_loader.ensure_history(sym)
                if not isinstance(has_data, bool) or not has_data:
                    continue

                frames = await self.data_loader.get_training_data(sym)
                if not isinstance(frames, dict) or not frames:
                    continue

                news_feats = self._build_news_features(analyzer, sym, frames) if analyzer is not None else None
                ds = self.feature_engineer.prepare(frames, news_features=news_feats, symbol=sym)

                cap = self._infer_global_pool_cap_per_symbol(n_features=int(ds.X.shape[1]), n_symbols=len(symbols))
                if cap is not None and len(ds.X) > cap:
                    logger.info(
                        f"[GLOBAL {idx}/{total}] Capping {sym} dataset from {len(ds.X):,} -> {cap:,} rows "
                        "(override via FOREX_BOT_GLOBAL_MAX_ROWS[_PER_SYMBOL])."
                    )
                    ds = self._tail_dataset(ds, cap)
                datasets.append((sym, ds))
            except Exception as e:
                logger.error(f"Failed to prepare {sym}: {e}", exc_info=True)
                continue

        await self._train_global_from_datasets(datasets, symbols, optimize, stop_event)

    def _maybe_start_ray(self) -> None:
        from ..models.rllib_agent import RAY_AVAILABLE, _maybe_init_ray

        try:
            # Always attempt to start Ray when available; ignore feature flags.
            if RAY_AVAILABLE and not self._ray_started:
                if _maybe_init_ray():
                    self._ray_started = True
                    logger.info("Ray initialized for RLlib agents.")
        except Exception as exc:
            logger.warning(f"Ray init skipped: {exc}")

    def _maybe_stop_ray(self) -> None:
        from ..models.rllib_agent import RAY_AVAILABLE

        try:
            if self._ray_started and RAY_AVAILABLE:
                import ray

                if ray.is_initialized():
                    ray.shutdown()
                    logger.info("Ray shutdown.")
        except Exception as e:
            logger.warning(f"Ray shutdown failed: {e}", exc_info=True)


# Standalone worker function for ProcessPoolExecutor (must be picklable)
def _hpc_feature_worker(settings, frames, sym, news_features=None, assigned_gpu=0):
    try:
        import sys
        import os
        from pathlib import Path

        # Limit internal math threads to 1 per worker to prevent thrashing
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_MAX_THREADS"] = "64"

        # Optionally force CPU-only feature engineering to avoid GPU OOM in workers.
        force_cpu = str(os.environ.get("FOREX_BOT_FEATURE_CPU_ONLY", "1")).strip().lower() in {
            "1", "true", "yes", "on"
        }
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                settings.system.enable_gpu_preference = "cpu"
            except Exception:
                pass
        else:
            # Set GPU affinity before any torch/cupy imports
            os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        
        # Add src/ to path explicitly to ensure latest code is loaded in the worker process
        project_root = Path(__file__).resolve().parents[3]
        src_path = str(project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Re-instantiate FE inside worker
        from forex_bot.features.pipeline import FeatureEngineer

        fe = FeatureEngineer(settings)
        return fe.prepare(frames, news_features=news_features, symbol=sym)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Worker failed for {sym}: {e}", exc_info=True)
        return None

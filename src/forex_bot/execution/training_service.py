import asyncio
import concurrent.futures
import json
import logging
import multiprocessing
import os
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
        
        # New Unsupervised Tensor Engine (Million-Search)
        discovery_tensor = TensorDiscoveryEngine(device="cuda", n_experts=20)
        
        # We need to pass the multi-timeframe frames to the engine
        discovery_tensor.run_unsupervised_search(frames, iterations=1000)
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
            # Provide symbol-aware OHLC metadata so Optuna can score profitability by symbol.
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

            from ..strategy.fast_backtest import fast_evaluate_strategy, infer_pip_metrics
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

                        pip_size, pip_value_per_lot = infer_pip_metrics(sym)
                        sl_pips = float(getattr(self.settings.risk, "meta_label_sl_pips", 20.0))
                        rr = float(getattr(self.settings.risk, "min_risk_reward", 2.0))
                        tp_pips_cfg = float(getattr(self.settings.risk, "meta_label_tp_pips", sl_pips * rr))
                        tp_pips = max(tp_pips_cfg, sl_pips * rr)

                        spread = float(getattr(self.settings.risk, "backtest_spread_pips", 1.5))
                        commission = float(getattr(self.settings.risk, "commission_per_lot", 0.0))

                        arr = fast_evaluate_strategy(
                            close_prices=meta_e["close"].to_numpy(dtype=np.float64),
                            high_prices=meta_e["high"].to_numpy(dtype=np.float64),
                            low_prices=meta_e["low"].to_numpy(dtype=np.float64),
                            signals=sig_s.to_numpy(dtype=np.int8),
                            month_indices=month_idx,
                            sl_pips=sl_pips,
                            tp_pips=tp_pips,
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
        logger.info("?? HPC Mode Active: Switching to PARALLEL Global Training (All data in RAM).")

        # DDP Synchronization
        is_ddp = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_ddp else 0
        world_size = dist.get_world_size() if is_ddp else 1

        # Cache path for sharing data between ranks
        cache_path = Path(self.settings.system.cache_dir) / "hpc_datasets.pkl"

        datasets: list[tuple[str, PreparedDataset]] = []

        if rank == 0:
            # --- RANK 0: Heavy Lifting ---
            # Allow overriding feature-engineering worker count via env to avoid pool crashes/OOM.
            env_workers = os.getenv("FEATURE_WORKERS")
            if env_workers:
                try:
                    max_workers = max(1, int(env_workers))
                except ValueError:
                    max_workers = max(1, min(self.settings.system.n_jobs, os.cpu_count() or 1))
            else:
                max_workers = max(1, min(self.settings.system.n_jobs, os.cpu_count() or 1))
            max_workers = max(1, min(max_workers, len(symbols)))

            analyzer = None
            if self.settings.news.enable_news:
                try:
                    analyzer = await get_sentiment_analyzer(self.settings)
                except Exception as exc:
                    logger.warning(f"News analyzer unavailable (global HPC): {exc}")

            raw_frames_map = {}
            news_map: dict[str, pd.DataFrame | None] = {}

            for sym in symbols:
                self.settings.system.symbol = sym
                await self.data_loader.ensure_history(sym)
                frames = await self.data_loader.get_training_data(sym)
                raw_frames_map[sym] = frames
                logger.info(f"HPC (Rank 0): Loaded raw data for {sym}")
                if analyzer is not None:
                    news_map[sym] = self._build_news_features(analyzer, sym, frames)

            # Parallel Feature Engineering (use spawn to avoid CUDA fork issues)
            logger.info(f"HPC (Rank 0): Launching feature engineering on {max_workers} workers...")
            spawn_ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as executor:
                futures: dict[concurrent.futures.Future, str] = {}
                for sym, frames in raw_frames_map.items():
                    fut = executor.submit(
                        _hpc_feature_worker,
                        self.settings.model_copy(),
                        frames,
                        sym,
                        news_map.get(sym),
                    )
                    futures[fut] = sym

                for fut in concurrent.futures.as_completed(list(futures.keys())):
                    try:
                        sym = futures.get(fut, "UNKNOWN")
                        ds = fut.result()
                        datasets.append((sym, ds))
                    except Exception as e:
                        logger.error(f"HPC Feature Gen failed: {e}")

            # If any symbols failed in the pool, fall back to sequential for the missing ones.
            got_syms = {s for s, _ in datasets}
            missing_syms = [s for s in symbols if s not in got_syms]
            if missing_syms:
                logger.warning(
                    f"HPC feature generation missing {len(missing_syms)}/{len(symbols)} symbols; "
                    "retrying sequentially for the missing symbols."
                )
                for sym in missing_syms:
                    try:
                        frames = raw_frames_map.get(sym)
                        if not frames:
                            continue
                        ds = self.feature_engineer.prepare(frames, news_features=news_map.get(sym), symbol=sym)
                        datasets.append((sym, ds))
                    except Exception as e:
                        logger.error(f"Sequential feature gen failed for {sym}: {e}", exc_info=True)

            # Save for other ranks
            if is_ddp:
                try:
                    logger.info("HPC (Rank 0): Saving datasets to disk for other ranks...")
                    joblib.dump(datasets, cache_path)
                except Exception as e:
                    logger.error(f"Failed to save HPC datasets: {e}")

        # --- BARRIER ---
        if is_ddp:
            logger.info(f"Rank {rank}: Waiting for data preparation...")
            dist.barrier()

            if rank > 0:
                logger.info(f"Rank {rank}: Loading datasets from {cache_path}...")
                try:
                    datasets = joblib.load(cache_path)
                except Exception as e:
                    logger.error(f"Rank {rank} failed to load datasets: {e}")
                    return

            # Allow distributed training for deep models only.
            if rank != 0:
                # For non-deep models, skip training on non-zero ranks to avoid duplication.
                # Deep models will handle DDP internally when enabled.
                logger.info(f"Rank {rank}: Skipping non-deep training to avoid duplicated work.")
                return

        if not datasets:
            logger.error("HPC: No datasets generated.")
            return
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
def _hpc_feature_worker(settings, frames, sym, news_features=None):
    try:
        # Re-instantiate FE inside worker
        from ..features.pipeline import FeatureEngineer

        fe = FeatureEngineer(settings)
        return fe.prepare(frames, news_features=news_features, symbol=sym)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Worker failed for {sym}: {e}")
        return None

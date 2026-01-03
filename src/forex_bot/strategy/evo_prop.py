import concurrent.futures
import csv
import json
import logging
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.distributed as dist

from ..core.config import Settings
from ..core.system import resolve_cpu_budget
from ..features.talib_mixer import TALIB_AVAILABLE, TALibStrategyMixer
from .fast_backtest import detailed_backtest, fast_evaluate_strategy, infer_pip_metrics
from .stop_target import infer_stop_target_pips
from .genetic import GeneticGene, GeneticStrategyEvolution

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FitnessResult:
    fitness: float
    net_profit: float
    max_dd: float
    trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    sharpe: float
    sortino: float
    sqn: float


# MODULE-LEVEL WORKER FUNCTION (must be picklable for ProcessPoolExecutor)
def _evaluate_gene_worker(
    gene: GeneticGene,
    df: pd.DataFrame,
    mixer: TALibStrategyMixer,
    settings: Settings,
    symbol: str,
    month_indices: np.ndarray,
    day_indices: np.ndarray,
    indicator_cache: dict[str, np.ndarray],
) -> tuple[str, FitnessResult]:
    """
    Worker function for parallel gene evaluation.
    Returns (strategy_id, FitnessResult) tuple.
    This must be a MODULE-LEVEL function (not instance method) for pickling.
    """
    if not TALIB_AVAILABLE:
        return (gene.strategy_id, FitnessResult(0.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    try:
        signals = mixer.compute_signals(df, gene, cache=indicator_cache).to_numpy()
        # Strict no-lookahead: act on next bar
        if len(signals) > 0:
            signals = np.roll(signals, 1)
            signals[0] = 0
    except Exception as exc:
        logger.warning(f"Signal generation failed for {gene.strategy_id}: {exc}")
        return (gene.strategy_id, FitnessResult(-1e9, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    tp_pips = float(getattr(gene, "tp_pips", 40.0))
    sl_pips = float(getattr(gene, "sl_pips", 20.0))
    spread = float(getattr(settings.risk, "backtest_spread_pips", 1.5))
    commission = float(getattr(settings.risk, "commission_per_lot", 7.0))
    max_hold = int(getattr(settings.risk, "triple_barrier_max_bars", 0) or 0)
    trailing_enabled = bool(getattr(settings.risk, "trailing_enabled", False))
    trailing_mult = float(getattr(settings.risk, "trailing_atr_multiplier", 1.0) or 1.0)
    trailing_trigger_r = float(getattr(settings.risk, "trailing_be_trigger_r", 1.0) or 1.0)
    pip_size, pip_value_per_lot = infer_pip_metrics(symbol)
    mode = str(getattr(settings.risk, "stop_target_mode", "blend") or "blend").lower()
    if mode not in {"fixed", "gene"}:
        res = infer_stop_target_pips(df, settings=settings, pip_size=pip_size)
        if res is not None:
            sl_pips, tp_pips, _rr = res
    metrics = fast_evaluate_strategy(
        close_prices=close,
        high_prices=high,
        low_prices=low,
        signals=signals,
        month_indices=month_indices,
        day_indices=day_indices,
        sl_pips=sl_pips,
        tp_pips=tp_pips,
        max_hold_bars=max_hold,
        trailing_enabled=trailing_enabled,
        trailing_atr_multiplier=trailing_mult,
        trailing_be_trigger_r=trailing_trigger_r,
        spread_pips=spread,
        commission_per_trade=commission,
        pip_value=pip_size,
        pip_value_per_lot=pip_value_per_lot,
    )

    (
        net_profit,
        sharpe,
        sortino,
        max_dd,
        win_rate,
        profit_factor,
        expectancy,
        sqn,
        trades,
        _consistency,
        daily_dd,
    ) = metrics

    # Calculate prop fitness (inline to avoid method dependency)
    fitness = 0.0
    dd_cap = float(getattr(settings.risk, "total_drawdown_limit", 0.07) or 0.07)
    daily_dd_cap = float(getattr(settings.risk, "daily_drawdown_limit", 0.04) or 0.04)
    profit_target_pct = float(getattr(settings.risk, "profit_target_pct", 0.04) or 0.04)
    actual_balance = 100000.0  # default

    if trades < 10:
        fitness -= 50.0
    if max_dd >= dd_cap:
        fitness -= 1000.0
    if daily_dd >= daily_dd_cap:
        fitness -= 500.0

    monthly_profit_pct = float(net_profit) / float(actual_balance) if actual_balance > 0 else 0.0
    if monthly_profit_pct >= profit_target_pct:
        fitness += 100.0 * (monthly_profit_pct / profit_target_pct)
    else:
        fitness -= 50.0

    fitness += 10.0 * float(sharpe)
    fitness += 0.1 * (win_rate * 100.0)

    res = FitnessResult(
        fitness=float(fitness),
        net_profit=float(net_profit),
        max_dd=float(max_dd),
        trades=int(trades),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        expectancy=float(expectancy),
        sharpe=float(sharpe),
        sortino=float(sortino),
        sqn=float(sqn),
    )
    return (gene.strategy_id, res)


class PropAwareStrategySearch:
    """
    Long-running evolutionary search for prop-compliant strategies.
    - Uses TALibMixer to generate signals.
    - Scores via fast_evaluate_strategy on PnL with prop-style constraints.
    - Supports checkpoint/resume for multi-day runs.
    """

    def __init__(
        self,
        settings: Settings,
        df: pd.DataFrame,
        population: int = 1000,
        mutation_rate: float = 0.20,
        checkpoint_path: Path = Path("models") / "strategy_evo_checkpoint.json",
        max_time_hours: float = 120.0,  # allow multi-day runs
        max_workers: int | None = None,
        actual_balance: float | None = None,
    ) -> None:
        self.settings = settings
        self.df_full = df
        self.df_val: pd.DataFrame | None = None
        self._val_summary: dict[str, dict[str, Any]] = {}
        self._val_monthly: dict[str, dict[str, Any]] = {}
        self._val_month_labels: list[str] = []
        self.df, self.df_val = self._split_train_val(df)
        self._month_indices, self._day_indices = self._compute_month_day_indices(self.df)
        self.checkpoint_path = checkpoint_path
        self.max_time_hours = max_time_hours
        try:
            model_cfg = getattr(settings, "models", None)
            prop_device = getattr(model_cfg, "prop_search_device", None)
        except Exception:
            prop_device = None
        mixer_device = prop_device or getattr(settings.system, "device", "cpu")
        self.mixer = TALibStrategyMixer(
            device=mixer_device,
            use_volume_features=bool(getattr(settings.system, "use_volume_features", False)),
        )
        self.evolver = GeneticStrategyEvolution(
            population_size=population, mutation_rate=mutation_rate, mixer=self.mixer
        )
        self.fitness_cache: dict[str, FitnessResult] = {}
        self.actual_balance = actual_balance if actual_balance is not None else 100000.0
        self.symbol = getattr(settings.system, "symbol", "")
        self._indicator_cache: dict[str, np.ndarray] = {}
        # Auto-detect distributed rank/world for sharding
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if max_workers is None:
            try:
                self.max_workers = resolve_cpu_budget()
            except Exception:
                self.max_workers = 1
        else:
            try:
                cpu_total = multiprocessing.cpu_count()
            except Exception:
                cpu_total = 1
            self.max_workers = max(1, min(cpu_total, int(max_workers)))
        try:
            self._prop_portfolio_size = int(
                getattr(self.settings.models, "prop_search_portfolio_size", 100) or 100
            )
        except Exception:
            self._prop_portfolio_size = 100
        try:
            self._prop_min_trades = int(
                getattr(self.settings.models, "prop_min_trades", 0) or 0
            )
        except Exception:
            self._prop_min_trades = 0

    def _symbol_tag(self) -> str:
        symbol = str(getattr(self, "symbol", "") or "")
        if not symbol:
            try:
                symbol = str(self.df.attrs.get("symbol", "") or "")
            except Exception:
                symbol = ""
        if not symbol:
            try:
                symbol = str(getattr(self.settings.system, "symbol", "") or "")
            except Exception:
                symbol = ""
        if not symbol:
            return ""
        return "".join(c for c in symbol if c.isalnum() or c in ("-", "_"))

    @staticmethod
    def _compute_month_day_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        month_indices = np.zeros(len(df), dtype=np.int64)
        day_indices = np.zeros(len(df), dtype=np.int64)
        try:
            idx = getattr(df, "index", None)
            if isinstance(idx, pd.DatetimeIndex) and len(idx) == len(df):
                ts = idx.tz_convert("UTC") if idx.tz is not None else idx
                raw = ts.year.to_numpy() * 12 + ts.month.to_numpy()
                month_indices = np.nan_to_num(raw, nan=0.0).astype(np.int64, copy=False)
                day_raw = ts.year.to_numpy() * 10000 + ts.month.to_numpy() * 100 + ts.day.to_numpy()
                day_indices = np.nan_to_num(day_raw, nan=0.0).astype(np.int64, copy=False)
        except Exception:
            month_indices = np.zeros(len(df), dtype=np.int64)
            day_indices = np.zeros(len(df), dtype=np.int64)
        return month_indices, day_indices

    def _split_train_val(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        try:
            train_years = int(getattr(self.settings.models, "prop_search_train_years", 0) or 0)
        except Exception:
            train_years = 0
        try:
            val_years = int(getattr(self.settings.models, "prop_search_val_years", 0) or 0)
        except Exception:
            val_years = 0

        if val_years <= 0:
            return df, None

        idx = getattr(df, "index", None)
        if not isinstance(idx, pd.DatetimeIndex):
            logger.warning("Prop search validation split requires DatetimeIndex; using full data.")
            return df, None

        if not idx.is_monotonic_increasing:
            df = df.sort_index(kind="mergesort")
            idx = df.index

        idx_cmp = idx
        if idx.tz is not None:
            try:
                idx_cmp = idx.tz_convert("UTC").tz_localize(None)
            except Exception:
                idx_cmp = idx.tz_localize(None)

        end_ts = idx_cmp.max()
        if end_ts is None or end_ts is pd.NaT:
            return df, None

        val_start = (end_ts - pd.DateOffset(years=val_years)).to_period("M").to_timestamp()
        if train_years > 0:
            train_start = (val_start - pd.DateOffset(years=train_years)).to_period("M").to_timestamp()
            train_df = df[(idx_cmp >= train_start) & (idx_cmp < val_start)]
        else:
            train_df = df[idx_cmp < val_start]
        val_df = df[idx_cmp >= val_start]
        try:
            train_df.attrs = dict(df.attrs)
            val_df.attrs = dict(df.attrs)
        except Exception:
            pass

        if train_df.empty or val_df.empty:
            logger.warning(
                "Prop search split produced empty train/val (train=%s, val=%s); using full data.",
                len(train_df),
                len(val_df),
            )
            return df, None

        logger.info(
            "Prop search split: train=%s→%s (%s rows), val=%s→%s (%s rows)",
            train_df.index.min(),
            train_df.index.max(),
            f"{len(train_df):,}",
            val_df.index.min(),
            val_df.index.max(),
            f"{len(val_df):,}",
        )
        return train_df, val_df

    @staticmethod
    def _build_month_slices(df: pd.DataFrame) -> tuple[list[str], list[tuple[int, int]]]:
        if df.empty:
            return [], []
        idx = getattr(df, "index", None)
        if not isinstance(idx, pd.DatetimeIndex):
            return [], []
        idx_cmp = idx
        if idx.tz is not None:
            try:
                idx_cmp = idx.tz_convert("UTC").tz_localize(None)
            except Exception:
                idx_cmp = idx.tz_localize(None)
        month_labels = idx_cmp.to_period("M").astype(str).to_numpy()
        if len(month_labels) == 0:
            return [], []
        changes = np.nonzero(month_labels[1:] != month_labels[:-1])[0] + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(month_labels)]))
        labels = [str(month_labels[s]) for s in starts]
        slices = [(int(s), int(e)) for s, e in zip(starts, ends, strict=False)]
        return labels, slices

    def _validate_candidates_monthly(self) -> None:
        if self.df_val is None or self.df_val.empty:
            return
        df_val = self.df_val
        idx = getattr(df_val, "index", None)
        if not isinstance(idx, pd.DatetimeIndex):
            logger.warning("Monthly validation requires DatetimeIndex; skipping.")
            return
        if not idx.is_monotonic_increasing:
            df_val = df_val.sort_index(kind="mergesort")

        month_labels, month_slices = self._build_month_slices(df_val)
        if not month_labels:
            return

        try:
            candidates = int(getattr(self.settings.models, "prop_search_val_candidates", 0) or 0)
        except Exception:
            candidates = 0
        if candidates <= 0:
            candidates = self._prop_portfolio_size

        sorted_pop = sorted(self.evolver.population, key=lambda g: g.fitness, reverse=True)
        genes = sorted_pop[: max(0, min(candidates, len(sorted_pop)))]
        if not genes:
            return

        try:
            min_trades_month = int(
                getattr(self.settings.models, "prop_search_val_min_trades_per_month", 0) or 0
            )
        except Exception:
            min_trades_month = 0
        try:
            opp_min_trades_month = int(
                getattr(self.settings.models, "prop_search_opportunistic_min_trades_per_month", 0) or 0
            )
        except Exception:
            opp_min_trades_month = 0
        try:
            min_trades_per_day = float(
                getattr(self.settings.models, "prop_search_val_min_trades_per_day", 0.0) or 0.0
            )
        except Exception:
            min_trades_per_day = 0.0

        try:
            log_trades = bool(getattr(self.settings.models, "prop_search_val_log_trades", False))
        except Exception:
            log_trades = False
        try:
            log_trades_max = int(getattr(self.settings.models, "prop_search_val_trade_log_max", 0) or 0)
        except Exception:
            log_trades_max = 0
        if log_trades_max <= 0:
            log_trades_max = len(genes)

        # Precompute indicator cache for validation
        try:
            val_cache = self.mixer.bulk_calculate_indicators(df_val, genes)
        except Exception as exc:
            logger.warning(f"Validation indicator cache failed; continuing without cache: {exc}")
            val_cache = {}

        close = df_val["close"].to_numpy()
        high = df_val["high"].to_numpy()
        low = df_val["low"].to_numpy()
        timestamps = df_val.index.to_numpy()
        month_indices, day_indices = self._compute_month_day_indices(df_val)
        month_label_map: dict[int, str] = {}
        for label, (start, _end) in zip(month_labels, month_slices, strict=False):
            try:
                month_label_map[int(month_indices[start])] = label
            except Exception:
                continue

        self._val_month_labels = month_labels
        self._val_summary = {}
        self._val_monthly = {}

        logger.info(
            "Monthly validation: %s candidates over %s months (min trades/month=%s, min trades/day=%s).",
            len(genes),
            len(month_labels),
            min_trades_month,
            min_trades_per_day,
        )

        # Precompute day counts per month for trades/day requirement
        monthly_days: list[int] = []
        monthly_min_trades: list[int] = []
        total_days = 0
        try:
            all_days = [int(d) for d in np.unique(day_indices) if int(d) > 0]
        except Exception:
            all_days = []
        total_days = len(all_days)
        for (label, (start, end)) in zip(month_labels, month_slices, strict=False):
            try:
                days_in_month = int(np.unique(day_indices[start:end]).size)
            except Exception:
                days_in_month = 0
            if days_in_month <= 0:
                days_in_month = 1
            req = min_trades_month
            if min_trades_per_day > 0.0:
                req = max(req, int(np.ceil(min_trades_per_day * days_in_month)))
            monthly_days.append(days_in_month)
            monthly_min_trades.append(req)

        def _trade_day_id(value: Any) -> int | None:
            try:
                if isinstance(value, (np.datetime64, pd.Timestamp)):
                    ts = pd.Timestamp(value)
                elif isinstance(value, (int, np.integer)):
                    idx_val = int(value)
                    if 0 <= idx_val < len(df_val):
                        ts = df_val.index[idx_val]
                    else:
                        return None
                else:
                    ts = pd.Timestamp(value)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert("UTC")
                return int(ts.year * 10000 + ts.month * 100 + ts.day)
            except Exception:
                return None

        for idx_gene, gene in enumerate(genes):
            try:
                signals = self.mixer.compute_signals(df_val, gene, cache=val_cache).to_numpy()
                # Strict no-lookahead: act on next bar
                if len(signals) > 0:
                    signals = np.roll(signals, 1)
                    signals[0] = 0
            except Exception as exc:
                logger.warning(f"Validation signal generation failed for {gene.strategy_id}: {exc}")
                continue

            tp_pips = float(getattr(gene, "tp_pips", 40.0))
            sl_pips = float(getattr(gene, "sl_pips", 20.0))
            spread = float(getattr(self.settings.risk, "backtest_spread_pips", 1.5))
            commission = float(getattr(self.settings.risk, "commission_per_lot", 7.0))
            max_hold = int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0)
            trailing_enabled = bool(getattr(self.settings.risk, "trailing_enabled", False))
            trailing_mult = float(getattr(self.settings.risk, "trailing_atr_multiplier", 1.0) or 1.0)
            trailing_trigger_r = float(getattr(self.settings.risk, "trailing_be_trigger_r", 1.0) or 1.0)
            pip_size, pip_value_per_lot = infer_pip_metrics(self.symbol)
            mode = str(getattr(self.settings.risk, "stop_target_mode", "blend") or "blend").lower()
            if mode not in {"fixed", "gene"}:
                res = infer_stop_target_pips(df_val, settings=self.settings, pip_size=pip_size)
                if res is not None:
                    sl_pips, tp_pips, _rr = res

            details = detailed_backtest(
                close_prices=close,
                high_prices=high,
                low_prices=low,
                signals=signals,
                timestamps=timestamps,
                month_indices=month_indices,
                day_indices=day_indices,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                max_hold_bars=max_hold,
                trailing_enabled=trailing_enabled,
                trailing_atr_multiplier=trailing_mult,
                trailing_be_trigger_r=trailing_trigger_r,
                spread_pips=spread,
                commission_per_trade=commission,
                pip_value=pip_size,
                pip_value_per_lot=pip_value_per_lot,
                month_label_map=month_label_map,
                base_balance=self.actual_balance,
            )

            # Build per-day trade counts (entry day)
            day_trade_counts: dict[int, int] = {}
            for trade in details.get("trades", []) or []:
                day_id = _trade_day_id(trade.get("entry_time"))
                if day_id is None or day_id <= 0:
                    continue
                day_trade_counts[day_id] = day_trade_counts.get(day_id, 0) + 1
            max_trade_return_pct = 0.0
            if self.actual_balance > 0:
                for trade in details.get("trades", []) or []:
                    try:
                        pnl = float(trade.get("pnl", 0.0) or 0.0)
                    except Exception:
                        pnl = 0.0
                    if pnl <= 0:
                        continue
                    trade_pct = (pnl / float(self.actual_balance)) * 100.0
                    if trade_pct > max_trade_return_pct:
                        max_trade_return_pct = trade_pct

            monthly_dict = details.get("monthly", {})
            summary = details.get("summary", {})
            monthly_profit_pct: list[float] = []
            monthly_trades: list[int] = []
            monthly_net_profit: list[float] = []
            monthly_win_rate: list[float] = []
            monthly_profit_factor: list[float] = []
            monthly_max_dd: list[float] = []
            monthly_avg_trade: list[float] = []
            monthly_avg_hold: list[float] = []
            monthly_exposure: list[float] = []
            monthly_max_daily_dd: list[float] = []
            monthly_gross_profit: list[float] = []
            monthly_gross_loss: list[float] = []
            monthly_return_pct: list[float] = []
            monthly_start_equity: list[float] = []
            monthly_end_equity: list[float] = []
            monthly_sharpe: list[float] = []
            monthly_day_trades: list[list[int]] = []
            monthly_day_failures: list[int] = []

            for label in month_labels:
                m = monthly_dict.get(label, {})
                monthly_profit_pct.append(float(m.get("base_return_pct", 0.0) or 0.0))
                monthly_return_pct.append(float(m.get("return_pct", 0.0) or 0.0))
                monthly_trades.append(int(m.get("trades", 0) or 0))
                monthly_net_profit.append(float(m.get("net_profit", 0.0) or 0.0))
                monthly_win_rate.append(float(m.get("win_rate", 0.0) or 0.0))
                monthly_profit_factor.append(float(m.get("profit_factor", 0.0) or 0.0))
                monthly_max_dd.append(float(m.get("max_dd", 0.0) or 0.0))
                monthly_avg_trade.append(float(m.get("avg_trade", 0.0) or 0.0))
                monthly_avg_hold.append(float(m.get("avg_hold_bars", 0.0) or 0.0))
                monthly_exposure.append(float(m.get("exposure_pct", 0.0) or 0.0))
                monthly_max_daily_dd.append(float(m.get("max_daily_dd", 0.0) or 0.0))
                monthly_gross_profit.append(float(m.get("gross_profit", 0.0) or 0.0))
                monthly_gross_loss.append(float(m.get("gross_loss", 0.0) or 0.0))
                monthly_start_equity.append(float(m.get("start_equity", 0.0) or 0.0))
                monthly_end_equity.append(float(m.get("end_equity", 0.0) or 0.0))
                monthly_sharpe.append(float(m.get("sharpe", 0.0) or 0.0))

            # Compute per-day enforcement for each month
            for (start, end) in month_slices:
                try:
                    days_in_month = [int(d) for d in np.unique(day_indices[start:end]) if int(d) > 0]
                except Exception:
                    days_in_month = []
                day_counts = [int(day_trade_counts.get(d, 0)) for d in days_in_month]
                monthly_day_trades.append(day_counts)
                if min_trades_per_day > 0.0:
                    monthly_day_failures.append(int(sum(1 for c in day_counts if c < min_trades_per_day)))
                else:
                    monthly_day_failures.append(0)

            opp_months_meeting_min_trades = 0
            opp_positive_months = 0
            if opp_min_trades_month > 0:
                for t, p in zip(monthly_trades, monthly_profit_pct, strict=False):
                    if t >= opp_min_trades_month:
                        opp_months_meeting_min_trades += 1
                        if p > 0.0:
                            opp_positive_months += 1
            else:
                for t, p in zip(monthly_trades, monthly_profit_pct, strict=False):
                    if t > 0:
                        opp_months_meeting_min_trades += 1
                        if p > 0.0:
                            opp_positive_months += 1

            active_mask = []
            for t, req, failures in zip(monthly_trades, monthly_min_trades, monthly_day_failures, strict=False):
                ok = True
                if req > 0 and t < req:
                    ok = False
                if min_trades_per_day > 0.0 and failures > 0:
                    ok = False
                active_mask.append(ok)
            active_months = int(sum(1 for a in active_mask if a))
            months_below_min_trades = int(sum(1 for a in active_mask if not a))
            positive_months = int(
                sum(1 for a, p in zip(active_mask, monthly_profit_pct, strict=False) if a and p > 0.0)
            )
            active_profits = [
                p for a, p in zip(active_mask, monthly_profit_pct, strict=False) if a
            ]
            avg_profit = float(np.mean(active_profits)) if active_profits else 0.0
            min_profit = float(np.min(active_profits)) if active_profits else 0.0
            total_trades = int(summary.get("trades", 0) or 0)
            avg_trades_per_day = (
                float(total_trades) / float(total_days) if total_days > 0 else 0.0
            )
            days_below_min = 0
            if min_trades_per_day > 0.0:
                for day_id in all_days:
                    if day_trade_counts.get(day_id, 0) < min_trades_per_day:
                        days_below_min += 1
            days_meeting_min = int(total_days - days_below_min) if total_days > 0 else 0

            self._val_summary[gene.strategy_id] = {
                "positive_months": positive_months,
                "active_months": active_months,
                "total_months": len(month_labels),
                "avg_monthly_profit_pct": avg_profit,
                "min_monthly_profit_pct": min_profit,
                "months_below_min_trades": months_below_min_trades,
                "net_profit": float(summary.get("net_profit", 0.0) or 0.0),
                "max_dd": float(summary.get("max_dd", 0.0) or 0.0),
                "trades": total_trades,
                "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
                "profit_factor": float(summary.get("profit_factor", 0.0) or 0.0),
                "expectancy": float(summary.get("expectancy", 0.0) or 0.0),
                "sharpe": float(summary.get("sharpe", 0.0) or 0.0),
                "sortino": float(summary.get("sortino", 0.0) or 0.0),
                "sqn": float(summary.get("sqn", 0.0) or 0.0),
                "max_daily_dd": float(summary.get("max_daily_dd", 0.0) or 0.0),
                "total_days": total_days,
                "avg_trades_per_day": avg_trades_per_day,
                "min_trades_per_day": min_trades_per_day,
                "days_meeting_min_trades_per_day": days_meeting_min,
                "days_below_min_trades_per_day": days_below_min,
                "max_trade_return_pct": max_trade_return_pct,
                "opp_active_months": opp_months_meeting_min_trades,
                "opp_positive_months": opp_positive_months,
            }
            payload: dict[str, Any] = {
                "monthly_profit_pct": monthly_profit_pct,
                "monthly_return_pct": monthly_return_pct,
                "monthly_trades": monthly_trades,
                "monthly_days": monthly_days,
                "monthly_min_trades": monthly_min_trades,
                "monthly_day_trades": monthly_day_trades,
                "monthly_day_failures": monthly_day_failures,
                "monthly_net_profit": monthly_net_profit,
                "monthly_win_rate": monthly_win_rate,
                "monthly_profit_factor": monthly_profit_factor,
                "monthly_max_dd": monthly_max_dd,
                "monthly_avg_trade": monthly_avg_trade,
                "monthly_avg_hold_bars": monthly_avg_hold,
                "monthly_exposure_pct": monthly_exposure,
                "monthly_max_daily_dd": monthly_max_daily_dd,
                "monthly_gross_profit": monthly_gross_profit,
                "monthly_gross_loss": monthly_gross_loss,
                "monthly_start_equity": monthly_start_equity,
                "monthly_end_equity": monthly_end_equity,
                "monthly_sharpe": monthly_sharpe,
                "months": month_labels,
                "summary": summary,
            }
            if log_trades and idx_gene < log_trades_max:
                trade_log_path = self._persist_trade_log(gene.strategy_id, details.get("trades", []))
                if trade_log_path:
                    payload["trade_log"] = trade_log_path
            self._val_monthly[gene.strategy_id] = payload

        self._persist_monthly_validation()

    def _persist_trade_log(self, strategy_id: str, trades: list[dict[str, Any]]) -> str:
        if not trades:
            return ""
        try:
            cache_dir = Path(getattr(self.settings.system, "cache_dir", "cache"))
            trade_dir = cache_dir / "prop_validation_trades"
            trade_dir.mkdir(parents=True, exist_ok=True)

            symbol = self._symbol_tag()
            safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in strategy_id)
            if symbol:
                safe_id = f"{symbol}_{safe_id}"
            path = trade_dir / f"{safe_id}.csv"

            def _fmt(value: Any) -> Any:
                if isinstance(value, pd.Timestamp):
                    return value.isoformat()
                if isinstance(value, np.datetime64):
                    try:
                        return pd.Timestamp(value).isoformat()
                    except Exception:
                        return str(value)
                return value

            rows = []
            for trade in trades:
                rows.append({k: _fmt(v) for k, v in trade.items()})

            if not rows:
                return ""
            fieldnames = list(rows[0].keys())
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return str(path)
        except Exception as exc:
            logger.warning(f"Failed to persist trade log for {strategy_id}: {exc}")
            return ""

    def _persist_monthly_validation(self) -> None:
        if not self._val_monthly:
            return
        try:
            cache_dir = Path(getattr(self.settings.system, "cache_dir", "cache"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            tf = ""
            try:
                tf = str(self.df_val.attrs.get("timeframe") or self.df_val.attrs.get("tf") or "")
            except Exception:
                tf = ""
            symbol = self._symbol_tag()
            payload = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "symbol": self.symbol,
                "timeframe": tf,
                "months": self._val_month_labels,
                "strategies": self._val_monthly,
            }
            base_name = cache_dir / "prop_validation_monthly.json"
            if symbol:
                symbol_path = cache_dir / f"prop_validation_monthly_{symbol}.json"
                symbol_path.write_text(json.dumps(payload, indent=2))
                logger.info("Monthly validation saved to %s.", symbol_path)
                # Also keep a generic copy for backward compatibility.
                base_name.write_text(json.dumps(payload, indent=2))
            else:
                base_name.write_text(json.dumps(payload, indent=2))
                logger.info("Monthly validation saved to %s.", base_name)
        except Exception as exc:
            logger.warning(f"Failed to persist monthly validation: {exc}")

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            payload = json.loads(self.checkpoint_path.read_text())
            self.evolver.generation = int(payload.get("generation", 0))
            pop = payload.get("population", [])
            genes = []
            for g in pop:
                gene = GeneticGene.from_dict(g)
                genes.append(gene)
                if "fitness" in g:
                    # Handle both old format (float) and new format (dict)
                    fit_data = g["fitness"]
                    if isinstance(fit_data, dict):
                        # New format: full FitnessResult dict
                        self.fitness_cache[gene.strategy_id] = FitnessResult(**fit_data)
                    elif isinstance(fit_data, (int, float)):
                        # Old format: just fitness score - create FitnessResult with defaults
                        self.fitness_cache[gene.strategy_id] = FitnessResult(
                            fitness=float(fit_data),
                            net_profit=0.0,
                            max_dd=0.0,
                            trades=0,
                            win_rate=0.0,
                            profit_factor=0.0,
                            expectancy=0.0,
                            sharpe=0.0,
                            sortino=0.0,
                            sqn=0.0,
                        )
            self.evolver.population = genes
            logger.info(f"Strategy evo checkpoint loaded (gen={self.evolver.generation}, pop={len(genes)})")
        except Exception as exc:
            logger.warning(f"Failed to load strategy evo checkpoint: {exc}")

    def _save_checkpoint(self) -> None:
        try:
            data = {
                "generation": self.evolver.generation,
                "population": [],
            }
            for gene in self.evolver.population:
                gd = gene.to_dict()
                fit = self.fitness_cache.get(gene.strategy_id)
                if fit:
                    gd["fitness"] = asdict(fit)
                data["population"].append(gd)
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_text(json.dumps(data))
        except Exception as exc:
            logger.warning(f"Failed to save strategy evo checkpoint: {exc}")

    def _prop_metric(self, metrics: np.ndarray) -> float:
        """
        HPC FIX: Hard-Stop Survival Enforcement.
        """
        (
            net_profit,
            sharpe,
            sortino,
            max_dd,
            win_rate,
            profit_factor,
            expectancy,
            sqn,
            trades,
            consistency,
            daily_dd,
        ) = metrics

        dd_limit = float(getattr(self.settings.risk, "total_drawdown_limit", 0.07) or 0.07)
        daily_limit = float(getattr(self.settings.risk, "daily_drawdown_limit", 0.04) or 0.04)

        # 1. Survival Check (Hard Limit)
        if max_dd >= dd_limit or daily_dd >= daily_limit:
            return -100.0 # Disqualified

        monthly_ret_pct = (net_profit / self.actual_balance) * 100.0
        target_monthly_pct = 4.0

        profit_score = monthly_ret_pct * 10.0
        target_bonus = max(0.0, (monthly_ret_pct - target_monthly_pct) * 50.0)

        fitness = profit_score + target_bonus
        fitness += 0.5 * sharpe
        fitness += 0.2 * profit_factor
        fitness += 0.1 * (win_rate * 100.0)
        
        # Stability: penalize zero trade strategies
        if trades < 10:
            fitness -= 50.0

        return float(fitness)

    def _evaluate_gene(self, gene: GeneticGene) -> FitnessResult:
        if gene.strategy_id in self.fitness_cache:
            return self.fitness_cache[gene.strategy_id]

        if not TALIB_AVAILABLE:
            logger.warning("TA-Lib not available; skipping evaluation.")
            res = FitnessResult(0.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self.fitness_cache[gene.strategy_id] = res
            return res

        try:
            signals = self.mixer.compute_signals(self.df, gene, cache=self._indicator_cache).to_numpy()
            # Strict no-lookahead: act on next bar
            if len(signals) > 0:
                signals = np.roll(signals, 1)
                signals[0] = 0
        except Exception as exc:
            logger.warning(f"Signal generation failed for {gene.strategy_id}: {exc}")
            res = FitnessResult(-1e9, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self.fitness_cache[gene.strategy_id] = res
            return res

        close = self.df["close"].to_numpy()
        high = self.df["high"].to_numpy()
        low = self.df["low"].to_numpy()
        tp_pips = float(getattr(gene, "tp_pips", 40.0))
        sl_pips = float(getattr(gene, "sl_pips", 20.0))
        spread = float(getattr(self.settings.risk, "backtest_spread_pips", 1.5))
        commission = float(getattr(self.settings.risk, "commission_per_lot", 7.0))
        max_hold = int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0)
        trailing_enabled = bool(getattr(self.settings.risk, "trailing_enabled", False))
        trailing_mult = float(getattr(self.settings.risk, "trailing_atr_multiplier", 1.0) or 1.0)
        trailing_trigger_r = float(getattr(self.settings.risk, "trailing_be_trigger_r", 1.0) or 1.0)
        pip_size, pip_value_per_lot = infer_pip_metrics(self.symbol)
        mode = str(getattr(self.settings.risk, "stop_target_mode", "blend") or "blend").lower()
        if mode not in {"fixed", "gene"}:
            res = infer_stop_target_pips(self.df, settings=self.settings, pip_size=pip_size)
            if res is not None:
                sl_pips, tp_pips, _rr = res
        metrics = fast_evaluate_strategy(
            close_prices=close,
            high_prices=high,
            low_prices=low,
            signals=signals,
            month_indices=self._month_indices,
            day_indices=self._day_indices,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            max_hold_bars=max_hold,
            trailing_enabled=trailing_enabled,
            trailing_atr_multiplier=trailing_mult,
            trailing_be_trigger_r=trailing_trigger_r,
            spread_pips=spread,
            commission_per_trade=commission,
            pip_value=pip_size,
            pip_value_per_lot=pip_value_per_lot,
        )

        (
            net_profit,
            sharpe,
            sortino,
            max_dd,
            win_rate,
            profit_factor,
            expectancy,
            sqn,
            trades,
            _consistency,
            daily_dd,
        ) = metrics
        fitness = self._prop_metric(metrics)
        res = FitnessResult(
            fitness=fitness,
            net_profit=float(net_profit),
            max_dd=float(max_dd),
            trades=int(trades),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            expectancy=float(expectancy),
            sharpe=float(sharpe),
            sortino=float(sortino),
            sqn=float(sqn),
        )
        self.fitness_cache[gene.strategy_id] = res
        return res

    def _evaluate_population_parallel(self, genes: list[GeneticGene]) -> None:
        """
        Evaluate a population in parallel.
        HPC FIX: Use ProcessPoolExecutor for TRUE GIL bypass.
        WINDOWS FIX: Falls back to sequential if RAM is high (pickling 5M rows is too expensive).
        """
        # Shard by rank if running under torchrun
        if self.world_size > 1:
            genes = [g for i, g in enumerate(genes) if i % self.world_size == self.rank]
        genes_to_eval = [g for g in genes if g.strategy_id not in self.fitness_cache]
        if not genes_to_eval:
            return

        def _evaluate_sequential(reason: str | None = None) -> None:
            remaining = [g for g in genes_to_eval if g.strategy_id not in self.fitness_cache]
            if not remaining:
                return
            if reason:
                logger.info(reason)
            logger.info(
                f"Evaluating {len(remaining)} genes sequentially (this will take a while)..."
            )
            for i, g in enumerate(remaining, 1):
                res = self._evaluate_gene(g)
                self.fitness_cache[g.strategy_id] = res
                # Progress logging every 10 genes
                if i % 10 == 0 or i == len(remaining):
                    logger.info(
                        f"Progress: {i}/{len(remaining)} genes evaluated ({100*i/len(remaining):.1f}%)"
                    )

        max_workers = max(1, int(self.max_workers or 1))

        # RAM check: On Windows with 'spawn', pickling huge DataFrames is too expensive
        # If RAM is high or the dataset is too large for parallel workers, use sequential evaluation instead.
        force_sequential = os.name == "nt"
        mem_percent: float | None = None
        mem_reason: str | None = None
        try:
            import psutil

            mem = psutil.virtual_memory()
            mem_percent = float(mem.percent)
            if mem_percent > 70:
                force_sequential = True
                mem_reason = f"RAM={mem_percent:.1f}%"

            # Optional guardrail: avoid parallelism when dataset copies would exceed RAM budget.
            try:
                df_bytes = int(self.df.memory_usage(deep=True).sum())
            except Exception:
                df_bytes = int(len(self.df) * max(1, self.df.shape[1]) * 4)

            try:
                max_rows_env = int(os.environ.get("FOREX_BOT_PROP_PARALLEL_MAX_ROWS", "0") or 0)
            except Exception:
                max_rows_env = 0
            if max_rows_env > 0 and len(self.df) > max_rows_env:
                force_sequential = True
                mem_reason = mem_reason or f"rows>{max_rows_env:,}"

            try:
                mem_frac = float(os.environ.get("FOREX_BOT_PROP_PARALLEL_MEM_FRAC", "0.25") or 0.25)
            except Exception:
                mem_frac = 0.25
            try:
                overhead = float(os.environ.get("FOREX_BOT_PROP_PARALLEL_OVERHEAD", "3.0") or 3.0)
            except Exception:
                overhead = 3.0

            mem_frac = float(min(0.80, max(0.05, mem_frac)))
            overhead = float(min(10.0, max(1.0, overhead)))
            projected = float(df_bytes) * float(max_workers) * float(overhead)
            if projected > (float(mem.available) * mem_frac):
                force_sequential = True
                if mem_reason is None:
                    mem_reason = "RAM budget"
        except Exception:
            # If psutil is missing, still force sequential on Windows for safety.
            mem_percent = None

        if force_sequential:
            reason_parts = []
            if os.name == "nt":
                reason_parts.append("Windows spawn overhead")
            if mem_reason:
                reason_parts.append(mem_reason)
            reason = "Using SEQUENTIAL evaluation"
            if reason_parts:
                reason += f" ({' or '.join(reason_parts)})"
            _evaluate_sequential(reason)
            return

        ctx_name = str(os.environ.get("FOREX_BOT_PROP_MP_CONTEXT", "auto") or "auto").strip().lower()
        if ctx_name == "auto":
            # Prefer fork on Unix for copy-on-write to reduce RAM, unless GPU is used.
            if os.name != "nt" and not getattr(self.mixer, "use_gpu", False):
                ctx_name = "fork"
            else:
                ctx_name = "spawn"
        spawn_ctx = multiprocessing.get_context(ctx_name)
        
        # HPC: True Multi-Process Evaluation
        # CRITICAL FIX: Use module-level worker function (not instance method) for pickling
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers, mp_context=spawn_ctx
            ) as ex:
                future_map = {
                    ex.submit(
                        _evaluate_gene_worker,
                        g,
                        self.df,
                        self.mixer,
                        self.settings,
                        self.symbol,
                        self._month_indices,
                        self._day_indices,
                        self._indicator_cache,
                    ): g
                    for g in genes_to_eval
                }
                for fut in concurrent.futures.as_completed(future_map):
                    g = future_map[fut]
                    try:
                        strategy_id, res = fut.result()  # Worker returns (strategy_id, FitnessResult)
                    except MemoryError:
                        logger.warning(
                            "Parallel eval ran out of memory; falling back to sequential.",
                            exc_info=True,
                        )
                        _evaluate_sequential("Falling back to sequential evaluation after MemoryError.")
                        return
                    except Exception as exc:
                        logger.warning(f"Parallel eval failed for {g.strategy_id}: {exc}", exc_info=True)
                        res = FitnessResult(-1e9, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    self.fitness_cache[g.strategy_id] = res
        except Exception:
            logger.warning(
                "Parallel evaluation failed; falling back to sequential evaluation.",
                exc_info=True,
            )
            _evaluate_sequential("Falling back to sequential evaluation after parallel failure.")

    def run(self, generations: int = 50) -> tuple[GeneticGene, FitnessResult]:
        if not TALIB_AVAILABLE:
            raise RuntimeError("TA-Lib not installed; cannot run strategy evolution.")

        start = time.time()
        self._load_checkpoint()
        if not self.evolver.population:
            self.evolver.initialize_population(regime="any")

        best_gene: GeneticGene | None = None
        best_fit: FitnessResult | None = None

        for _ in range(generations):
            if (time.time() - start) / 3600.0 >= self.max_time_hours:
                logger.info("Max evolution wall-clock reached; stopping.")
                break

            # Pre-compute indicator cache for current population to avoid recomputation
            try:
                self._indicator_cache = self.mixer.bulk_calculate_indicators(self.df, self.evolver.population)
            except Exception as exc:
                logger.warning(f"Indicator cache generation failed, continuing without cache: {exc}")
                self._indicator_cache = {}

            self._evaluate_population_parallel(self.evolver.population)
            for gene in self.evolver.population:
                fit = self.fitness_cache.get(gene.strategy_id)
                if not fit:
                    fit = self._evaluate_gene(gene)
                gene.fitness = fit.fitness
                gene.win_rate = fit.win_rate
                gene.max_drawdown = fit.max_dd
                gene.expectancy = fit.expectancy
                gene.profit_factor = fit.profit_factor
                gene.sharpe_ratio = fit.sharpe

            sorted_pop = sorted(self.evolver.population, key=lambda g: g.fitness, reverse=True)
            top = sorted_pop[0]
            top_fit = self.fitness_cache[top.strategy_id]
            if best_fit is None or top_fit.fitness > best_fit.fitness:
                best_gene = top
                best_fit = top_fit
                logger.info(
                    f"New best strategy {top.strategy_id} fit={top_fit.fitness:.2f} "
                    f"ret={top_fit.net_profit:.1f} dd={top_fit.max_dd:.2%} trades={top_fit.trades}"
                )

            self.evolver.evolve()
            self._save_checkpoint()

        if best_gene is None or best_fit is None:
            best_gene = sorted(self.evolver.population, key=lambda g: g.fitness, reverse=True)[0]
            best_fit = self.fitness_cache.get(best_gene.strategy_id, FitnessResult(0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
        return best_gene, best_fit

    def _gene_payload(
        self,
        gene: GeneticGene,
        fit: FitnessResult | None,
        val_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tf = ""
        try:
            tf = str(self.df.attrs.get("timeframe") or self.df.attrs.get("tf") or "")
        except Exception:
            tf = ""
        symbol = self._symbol_tag()
        val_summary = val_summary or {}
        return {
            "indicators": list(getattr(gene, "indicators", []) or []),
            "params": dict(getattr(gene, "params", {}) or {}),
            "combination_method": getattr(gene, "combination_method", "weighted_vote"),
            "long_threshold": float(getattr(gene, "long_threshold", 0.6)),
            "short_threshold": float(getattr(gene, "short_threshold", -0.6)),
            "weights": dict(getattr(gene, "weights", {}) or {}),
            "preferred_regime": getattr(gene, "preferred_regime", "any"),
            "fitness": float(getattr(fit, "fitness", 0.0)) if fit else float(getattr(gene, "fitness", 0.0)),
            "sharpe_ratio": float(getattr(fit, "sharpe", getattr(gene, "sharpe_ratio", 0.0))) if fit else float(getattr(gene, "sharpe_ratio", 0.0)),
            "win_rate": float(getattr(fit, "win_rate", getattr(gene, "win_rate", 0.0))) if fit else float(getattr(gene, "win_rate", 0.0)),
            "profit_factor": float(getattr(fit, "profit_factor", getattr(gene, "profit_factor", 0.0))) if fit else float(getattr(gene, "profit_factor", 0.0)),
            "expectancy": float(getattr(fit, "expectancy", getattr(gene, "expectancy", 0.0))) if fit else float(getattr(gene, "expectancy", 0.0)),
            "max_dd": float(getattr(fit, "max_dd", getattr(gene, "max_drawdown", 0.0))) if fit else float(getattr(gene, "max_drawdown", 0.0)),
            "trades": int(getattr(fit, "trades", getattr(gene, "trades_count", 0))) if fit else int(getattr(gene, "trades_count", 0)),
            "strategy_id": getattr(gene, "strategy_id", ""),
            "use_ob": bool(getattr(gene, "use_ob", False)),
            "use_fvg": bool(getattr(gene, "use_fvg", False)),
            "use_liq_sweep": bool(getattr(gene, "use_liq_sweep", False)),
            "mtf_confirmation": bool(getattr(gene, "mtf_confirmation", False)),
            "use_premium_discount": bool(getattr(gene, "use_premium_discount", False)),
            "use_inducement": bool(getattr(gene, "use_inducement", False)),
            "tp_pips": float(getattr(gene, "tp_pips", 40.0)),
            "sl_pips": float(getattr(gene, "sl_pips", 20.0)),
            "source": "propaware",
            "timeframe": tf,
            "symbol": symbol,
            "val_positive_months": int(val_summary.get("positive_months", 0) or 0),
            "val_active_months": int(val_summary.get("active_months", 0) or 0),
            "val_total_months": int(val_summary.get("total_months", 0) or 0),
            "val_avg_monthly_profit_pct": float(val_summary.get("avg_monthly_profit_pct", 0.0) or 0.0),
            "val_min_monthly_profit_pct": float(val_summary.get("min_monthly_profit_pct", 0.0) or 0.0),
            "val_months_below_min_trades": int(val_summary.get("months_below_min_trades", 0) or 0),
            "val_net_profit": float(val_summary.get("net_profit", 0.0) or 0.0),
            "val_max_dd": float(val_summary.get("max_dd", 0.0) or 0.0),
            "val_trades": int(val_summary.get("trades", 0) or 0),
            "val_win_rate": float(val_summary.get("win_rate", 0.0) or 0.0),
            "val_profit_factor": float(val_summary.get("profit_factor", 0.0) or 0.0),
            "val_expectancy": float(val_summary.get("expectancy", 0.0) or 0.0),
            "val_sharpe": float(val_summary.get("sharpe", 0.0) or 0.0),
            "val_sortino": float(val_summary.get("sortino", 0.0) or 0.0),
            "val_sqn": float(val_summary.get("sqn", 0.0) or 0.0),
            "val_max_daily_dd": float(val_summary.get("max_daily_dd", 0.0) or 0.0),
            "val_total_days": int(val_summary.get("total_days", 0) or 0),
            "val_avg_trades_per_day": float(val_summary.get("avg_trades_per_day", 0.0) or 0.0),
            "val_min_trades_per_day": float(val_summary.get("min_trades_per_day", 0.0) or 0.0),
            "val_days_meeting_min_trades_per_day": int(val_summary.get("days_meeting_min_trades_per_day", 0) or 0),
            "val_days_below_min_trades_per_day": int(val_summary.get("days_below_min_trades_per_day", 0) or 0),
            "val_max_trade_return_pct": float(val_summary.get("max_trade_return_pct", 0.0) or 0.0),
            "val_opp_active_months": int(val_summary.get("opp_active_months", 0) or 0),
            "val_opp_positive_months": int(val_summary.get("opp_positive_months", 0) or 0),
        }

    def export_portfolio(self, cache_dir: Path, *, merge: bool = True, max_genes: int | None = None) -> int:
        max_genes = int(max_genes or self._prop_portfolio_size or 100)
        if not self.evolver.population:
            return 0

        # Build payloads for top strategies
        sorted_pop = sorted(self.evolver.population, key=lambda g: g.fitness, reverse=True)
        core_payloads: list[dict[str, Any]] = []
        try:
            min_pos_months = int(
                getattr(self.settings.models, "prop_search_val_min_positive_months", 0) or 0
            )
        except Exception:
            min_pos_months = 0
        try:
            min_profit_pct = float(
                getattr(self.settings.models, "prop_search_val_min_monthly_profit_pct", 0.0) or 0.0
            )
        except Exception:
            min_profit_pct = 0.0
        try:
            min_trades_per_day = float(
                getattr(self.settings.models, "prop_search_val_min_trades_per_day", 0.0) or 0.0
            )
        except Exception:
            min_trades_per_day = 0.0
        try:
            min_trades_month = int(
                getattr(self.settings.models, "prop_search_val_min_trades_per_month", 0) or 0
            )
        except Exception:
            min_trades_month = 0

        for gene in sorted_pop:
            fit = self.fitness_cache.get(gene.strategy_id)
            if self._prop_min_trades > 0 and fit and fit.trades < self._prop_min_trades:
                continue
            val_summary = self._val_summary.get(gene.strategy_id)
            if min_pos_months > 0:
                if not val_summary or int(val_summary.get("positive_months", 0) or 0) < min_pos_months:
                    continue
            if min_profit_pct > 0.0:
                if not val_summary or float(val_summary.get("min_monthly_profit_pct", 0.0) or 0.0) < min_profit_pct:
                    continue
            if min_trades_month > 0 or min_trades_per_day > 0.0:
                months_below = int(val_summary.get("months_below_min_trades", 0) or 0) if val_summary else 0
                if months_below > 0:
                    continue
            core_payloads.append(self._gene_payload(gene, fit, val_summary))
            if len(core_payloads) >= max_genes:
                break

        opp_payloads: list[dict[str, Any]] = []
        try:
            opp_enabled = bool(
                getattr(self.settings.models, "prop_search_opportunistic_enabled", True)
            )
        except Exception:
            opp_enabled = True
        if opp_enabled:
            try:
                opp_min_pos_months = int(
                    getattr(self.settings.models, "prop_search_opportunistic_min_positive_months", 0) or 0
                )
            except Exception:
                opp_min_pos_months = 0
            try:
                opp_min_trades_month = int(
                    getattr(self.settings.models, "prop_search_opportunistic_min_trades_per_month", 0) or 0
                )
            except Exception:
                opp_min_trades_month = 0
            try:
                opp_min_trade_return_pct = float(
                    getattr(self.settings.models, "prop_search_opportunistic_min_trade_return_pct", 0.0) or 0.0
                )
            except Exception:
                opp_min_trade_return_pct = 0.0
            try:
                opp_max_dd = float(
                    getattr(self.settings.models, "prop_search_opportunistic_max_dd", 1.0) or 1.0
                )
            except Exception:
                opp_max_dd = 1.0

            for gene in sorted_pop:
                fit = self.fitness_cache.get(gene.strategy_id)
                val_summary = self._val_summary.get(gene.strategy_id)
                if not val_summary:
                    continue
                opp_pos_months = int(val_summary.get("opp_positive_months", 0) or 0)
                max_trade_return_pct = float(val_summary.get("max_trade_return_pct", 0.0) or 0.0)
                max_dd = float(val_summary.get("max_dd", 1.0) or 1.0)

                keep = False
                # 1) Always keep "big trade" strategies (>= threshold in a single trade).
                if opp_min_trade_return_pct > 0.0 and max_trade_return_pct >= opp_min_trade_return_pct:
                    keep = True
                # 2) Otherwise keep if they have enough positive months (non-consecutive allowed).
                if not keep and opp_min_pos_months > 0 and opp_pos_months >= opp_min_pos_months:
                    keep = True
                if not keep and opp_min_pos_months <= 0 and opp_pos_months > 0:
                    keep = True
                if not keep:
                    continue
                opp_payloads.append(self._gene_payload(gene, fit, val_summary))
                if len(opp_payloads) >= max_genes:
                    break

        if not core_payloads and not opp_payloads:
            return 0

        symbol = self._symbol_tag()
        knowledge_path = cache_dir / "talib_knowledge.json"
        total_written = 0

        def _write_portfolio(
            payloads: list[dict[str, Any]],
            target_path: Path,
            generic_path: Path | None,
            source_tag: str,
        ) -> int:
            if not payloads:
                return 0
            combined = payloads

            if merge and target_path.exists():
                try:
                    existing = json.loads(target_path.read_text())
                    existing_genes = []
                    if isinstance(existing, dict):
                        if "best_genes" in existing:
                            existing_genes = list(existing.get("best_genes") or [])
                        elif "best_gene" in existing:
                            existing_genes = [existing.get("best_gene")]
                    if existing_genes:
                        combined = existing_genes + payloads
                except Exception as exc:
                    logger.warning(f"Failed to read existing knowledge for merge: {exc}")

            seen: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for gene in combined:
                if not isinstance(gene, dict):
                    continue
                key = str(gene.get("strategy_id") or "")
                if not key:
                    key = json.dumps(
                        {
                            "indicators": gene.get("indicators", []),
                            "params": gene.get("params", {}),
                            "long_threshold": gene.get("long_threshold", 0.0),
                            "short_threshold": gene.get("short_threshold", 0.0),
                            "weights": gene.get("weights", {}),
                        },
                        sort_keys=True,
                    )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(gene)

            deduped.sort(key=lambda g: float(g.get("fitness", 0.0) or 0.0), reverse=True)
            deduped = deduped[:max_genes]

            payload = {
                "best_genes": deduped,
                "timestamp": pd.Timestamp.now().isoformat(),
                "sources": {
                    source_tag: len(payloads),
                    "merged_total": len(deduped),
                },
            }
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(json.dumps(payload, indent=2))
            logger.info("Prop-aware portfolio merged into %s (%s strategies).", target_path, len(deduped))
            if generic_path and target_path != generic_path:
                generic_path.write_text(json.dumps(payload, indent=2))
            return len(deduped)

        core_target = knowledge_path
        if symbol:
            core_target = cache_dir / f"talib_knowledge_{symbol}.json"
        total_written = max(
            total_written,
            _write_portfolio(core_payloads, core_target, knowledge_path, "propaware"),
        )

        if opp_payloads:
            opp_base = cache_dir / "talib_knowledge_opportunistic.json"
            opp_target = opp_base
            if symbol:
                opp_target = cache_dir / f"talib_knowledge_opportunistic_{symbol}.json"
            total_written = max(
                total_written,
                _write_portfolio(opp_payloads, opp_target, opp_base, "opportunistic"),
            )

        return total_written


def run_evo_search(
    ohlcv: pd.DataFrame,
    settings: Settings | None = None,
    population: int = 64,
    generations: int = 50,
    checkpoint: str = "models/strategy_evo_checkpoint.json",
    max_time_hours: float = 120.0,
    actual_balance: float | None = None,
    max_workers: int | None = None,
) -> tuple[GeneticGene, FitnessResult]:
    settings = settings or Settings()
    search = PropAwareStrategySearch(
        settings=settings,
        df=ohlcv,
        population=population,
        checkpoint_path=Path(checkpoint),
        max_time_hours=max_time_hours,
        actual_balance=actual_balance,
        max_workers=max_workers,
    )
    result = search.run(generations=generations)
    try:
        search._validate_candidates_monthly()
    except Exception as exc:
        logger.warning(f"Monthly validation failed: {exc}")
    try:
        cache_dir = Path(getattr(settings.system, "cache_dir", "cache"))
        search.export_portfolio(cache_dir, merge=True)
    except Exception as exc:
        logger.warning(f"Failed to export prop-aware portfolio: {exc}")
    return result

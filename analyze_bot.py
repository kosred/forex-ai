#!/usr/bin/env python3
"""
Offline multi-run analysis harness.

Goal:
  - Load local historical data (no MT5 required)
  - Run feature engineering + ensemble signal generation
  - Slice the timeline into multiple segments ("multi-run")
  - Backtest each segment and summarize stability + speed

This is NOT financial advice. Use for research and debugging only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forex_bot.core.config import Settings
from forex_bot.core.system import AutoTuner, HardwareProbe
from forex_bot.data.loader import DataLoader
from forex_bot.data.news.client import get_sentiment_analyzer
from forex_bot.features.engine import SignalEngine
from forex_bot.features.pipeline import FeatureEngineer
from forex_bot.strategy.fast_backtest import fast_evaluate_strategy, infer_pip_metrics
from forex_bot.training.evaluation import prop_backtest


@dataclass(slots=True)
class SegmentResult:
    start: str
    end: str
    bars: int
    prop: dict[str, Any]
    fast: dict[str, Any] | None = None


@dataclass(slots=True)
class SymbolReport:
    symbol: str
    base_timeframe: str
    rows_cap: int
    features: int
    bars: int
    durations_sec: dict[str, float]
    aggregate_prop: dict[str, Any]
    aggregate_fast: dict[str, Any]
    segments: list[SegmentResult]


def _disable_network_features(settings: Settings, *, keep_news: bool) -> None:
    # Keep analysis deterministic/offline.
    settings.news.enable_news = bool(keep_news)
    settings.news.enable_llm_helper = False
    settings.news.llm_helper_enabled = False
    settings.news.openai_news_enabled = False
    settings.news.perplexity_enabled = False
    settings.news.strategist_enabled = False

    # Prevent any accidental MT5 requirement during analysis.
    settings.system.mt5_required = False


def _discover_symbols(data_dir: str) -> list[str]:
    p = Path(data_dir)
    if not p.exists():
        return []
    out: list[str] = []
    for child in p.iterdir():
        if child.is_dir() and child.name.startswith("symbol="):
            out.append(child.name.split("=", 1)[1])
    return sorted(set(out))


def _segment_indices(n: int, segments: int) -> list[slice]:
    if segments <= 1 or n <= 0:
        return [slice(0, n)]
    parts = np.array_split(np.arange(n, dtype=int), segments)
    out: list[slice] = []
    for p in parts:
        if len(p) == 0:
            continue
        out.append(slice(int(p[0]), int(p[-1]) + 1))
    return out


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
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
        agg[f"{key}_min"] = float(np.min(vals))
        agg[f"{key}_max"] = float(np.max(vals))

    for key in bool_any_keys or []:
        agg[f"{key}_any"] = any(bool(m.get(key)) for m in metrics_list)
    return agg


def _fast_backtest_dict(symbol: str, df: pd.DataFrame, signals: pd.Series, settings: Settings) -> dict[str, Any]:
    # Numba backtester expects np arrays.
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    sig = signals.to_numpy(dtype=np.int8)

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True, errors="coerce")
    month_idx = (idx.year.astype(np.int32) * 12 + idx.month.astype(np.int32)).to_numpy(dtype=np.int64)

    pip_size, pip_value_per_lot = infer_pip_metrics(symbol)

    sl_pips = float(getattr(settings.risk, "meta_label_sl_pips", 20.0))
    rr = float(getattr(settings.risk, "min_risk_reward", 2.0))
    tp_pips_cfg = float(getattr(settings.risk, "meta_label_tp_pips", sl_pips * rr))
    tp_pips = max(tp_pips_cfg, sl_pips * rr)

    spread = float(getattr(settings.risk, "backtest_spread_pips", 1.5))
    commission = float(getattr(settings.risk, "commission_per_lot", 0.0))

    arr = fast_evaluate_strategy(
        close_prices=close,
        high_prices=high,
        low_prices=low,
        signals=sig,
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
    return {k: float(v) for k, v in zip(keys, arr.tolist(), strict=False)}


async def analyze_symbol(settings: Settings, symbol: str, segments: int) -> SymbolReport:
    settings.system.symbol = symbol
    base_tf = str(getattr(settings.system, "base_timeframe", "M1"))

    durations: dict[str, float] = {}

    t0 = time.perf_counter()
    loader = DataLoader(settings)
    ok = await loader.ensure_history(symbol)
    if not ok:
        raise RuntimeError(f"Missing data for {symbol} (ensure_history failed).")
    frames = await loader.get_training_data(symbol)
    durations["load_data"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    fe = FeatureEngineer(settings)
    news_feats = None
    if settings.news.enable_news:
        try:
            analyzer = await get_sentiment_analyzer(settings)
            currencies = [symbol[:3], symbol[3:]] if len(symbol) == 6 else []
            base_tf = settings.system.base_timeframe
            base_df = frames.get(base_tf)
            if base_df is None:
                base_df = frames.get("M1")
            if base_df is not None and hasattr(base_df, "empty") and not base_df.empty:
                if "timestamp" in base_df.columns:
                    start_ts = pd.to_datetime(base_df["timestamp"].min(), utc=True, errors="coerce").to_pydatetime()
                    end_ts = pd.to_datetime(base_df["timestamp"].max(), utc=True, errors="coerce").to_pydatetime()
                    base_idx = pd.to_datetime(base_df["timestamp"], utc=True, errors="coerce")
                else:
                    start_ts = pd.to_datetime(base_df.index.min(), utc=True, errors="coerce").to_pydatetime()
                    end_ts = pd.to_datetime(base_df.index.max(), utc=True, errors="coerce").to_pydatetime()
                    base_idx = pd.to_datetime(base_df.index, utc=True, errors="coerce")

                events = analyzer.db.fetch_events(start_ts, end_ts, currencies=currencies)
                if events is not None and len(events) > 0:
                    news_feats = analyzer.build_features(events, pd.DatetimeIndex(base_idx))
                    if news_feats is not None and len(news_feats) == 0:
                        news_feats = None
        except Exception:
            news_feats = None

    dataset = fe.prepare(frames, news_features=news_feats, symbol=symbol)
    durations["features"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    engine = SignalEngine(settings)
    engine.load_models("models")
    result = engine.generate_ensemble_signals(dataset)
    durations["signals"] = time.perf_counter() - t2

    if dataset.metadata is None or result.signals is None:
        raise RuntimeError("Missing metadata/signals for backtest.")

    df = dataset.metadata
    sig = result.signals
    n = len(df)
    durations["bars"] = float(n)

    seg_results: list[SegmentResult] = []
    per_segment_prop: list[dict[str, Any]] = []
    per_segment_fast: list[dict[str, Any]] = []

    t3 = time.perf_counter()
    for slc in _segment_indices(n, segments):
        df_s = df.iloc[slc]
        sig_s = sig.iloc[slc]
        if len(df_s) < 200:
            continue
        prop_metrics = prop_backtest(
            df_s,
            sig_s,
            max_daily_dd_pct=float(getattr(settings.risk, "daily_drawdown_limit", 0.05)),
            daily_dd_warn_pct=float(getattr(settings.risk, "daily_drawdown_limit", 0.03)) * 0.8,
            max_trades_per_day=int(getattr(settings.risk, "max_trades_per_day", 10)),
        )
        per_segment_prop.append(prop_metrics)

        fast_metrics = _fast_backtest_dict(symbol, df_s, sig_s, settings)
        per_segment_fast.append(fast_metrics)

        seg_results.append(
            SegmentResult(
                start=str(df_s.index[0]),
                end=str(df_s.index[-1]),
                bars=int(len(df_s)),
                prop=prop_metrics,
                fast=fast_metrics,
            )
        )
    durations["backtest_total"] = time.perf_counter() - t3

    report = SymbolReport(
        symbol=symbol,
        base_timeframe=base_tf,
        rows_cap=int(getattr(settings.system, "max_training_rows_per_tf", 0) or 0),
        features=int(getattr(dataset.X, "shape", (0, 0))[1]),
        bars=int(n),
        durations_sec={k: float(v) for k, v in durations.items() if k != "bars"},
        aggregate_prop=_aggregate_metrics(
            per_segment_prop,
            numeric_keys=["pnl_score", "win_rate", "max_dd_pct", "trades"],
            bool_any_keys=["daily_dd_violation", "trade_limit_violation"],
        ),
        aggregate_fast=_aggregate_metrics(
            per_segment_fast,
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
        ),
        segments=seg_results,
    )
    return report


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Offline multi-run analysis for forex-bot.")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols. Default: from config.yaml")
    parser.add_argument("--rows", type=int, default=50_000, help="Row cap per timeframe (tail). 0 = no cap.")
    parser.add_argument("--segments", type=int, default=6, help="Number of time segments per symbol.")
    parser.add_argument("--out", type=str, default="reports/bot_analysis.json", help="Output JSON path.")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature caching.")
    parser.add_argument(
        "--with-news",
        action="store_true",
        help="Include local news features from cache/news.sqlite and data/news/*.csv (no online fetch).",
    )
    args = parser.parse_args()

    settings = Settings()
    _disable_network_features(settings, keep_news=bool(args.with_news))

    # Apply hardware detection (n_jobs, GPU flags, batch sizes).
    profile = HardwareProbe().detect()
    AutoTuner(settings, profile).apply()

    settings.system.max_training_rows_per_tf = int(args.rows)
    settings.system.cache_enabled = not bool(args.no_cache)

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _discover_symbols(str(getattr(settings.system, "data_dir", "data")))
        if not symbols:
            symbols = list(getattr(settings.system, "symbols", [])) or [
                str(getattr(settings.system, "symbol", "EURUSD"))
            ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reports: list[SymbolReport] = []
    for sym in symbols:
        rep = await analyze_symbol(settings, sym, int(args.segments))
        reports.append(rep)
        print(
            f"{sym}: pnl_mean={rep.aggregate_prop.get('pnl_score_mean', 0.0):.4f} "
            f"win_rate_mean={rep.aggregate_prop.get('win_rate_mean', 0.0):.3f} "
            f"mdd_mean={rep.aggregate_prop.get('max_dd_pct_mean', 0.0):.3f} "
            f"fast_np_mean={rep.aggregate_fast.get('net_profit_mean', 0.0):.1f} "
            f"segments={len(rep.segments)} "
            f"time={sum(rep.durations_sec.values()):.1f}s"
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows_cap": int(args.rows),
        "segments": int(args.segments),
        "symbols": [r.symbol for r in reports],
        "reports": [asdict(r) for r in reports],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

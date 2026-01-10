from __future__ import annotations

import os
from typing import Iterable, Tuple

import numpy as np

_FOREX_CORE_ERROR: Exception | None = None
try:
    import forex_core as _forex_core
except Exception as exc:  # pragma: no cover - depends on native build
    _forex_core = None
    _FOREX_CORE_ERROR = exc


def _require_rust() -> None:
    if _forex_core is not None:
        return
    if str(os.environ.get("FOREX_BOT_ALLOW_PY_BACKTEST", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    raise ImportError(
        "forex_core Rust extension not available. Build the Rust core or set "
        "FOREX_BOT_ALLOW_PY_BACKTEST=1 to allow slow Python fallback."
    ) from _FOREX_CORE_ERROR


def _as_contig(arr: Iterable, dtype: np.dtype) -> np.ndarray:
    out = np.asarray(arr, dtype=dtype)
    if not out.flags.c_contiguous:
        out = np.ascontiguousarray(out, dtype=dtype)
    return out


def infer_pip_metrics(symbol: str) -> Tuple[float, float]:
    sym = (symbol or "").upper()
    if sym.endswith("JPY") or sym.startswith("JPY"):
        pip_size = 0.01
        pip_value = 10.0
    elif sym.startswith("XAU") or sym.startswith("XAG"):
        pip_size = 0.01
        pip_value = 10.0
    elif "BTC" in sym or "ETH" in sym or "LTC" in sym:
        pip_size = 1.0
        pip_value = 1.0
    else:
        pip_size = 0.0001
        pip_value = 10.0
    return float(pip_size), float(pip_value)


def infer_sl_tp_pips_auto(
    *,
    open_prices: Iterable,
    high_prices: Iterable,
    low_prices: Iterable,
    close_prices: Iterable,
    atr_values: Iterable | None,
    pip_size: float,
    atr_mult: float,
    min_rr: float,
    min_dist: float,
    settings: object | None = None,
) -> Tuple[float, float] | None:
    close_arr = _as_contig(close_prices, np.float64)
    high_arr = _as_contig(high_prices, np.float64)
    low_arr = _as_contig(low_prices, np.float64)
    if atr_values is not None:
        atr_arr = _as_contig(atr_values, np.float64)
    else:
        atr_arr = None

    if close_arr.size < 2:
        return None

    if atr_arr is None or atr_arr.size == 0:
        prev_close = np.roll(close_arr, 1)
        tr = np.maximum(high_arr - low_arr, np.maximum(np.abs(high_arr - prev_close), np.abs(low_arr - prev_close)))
        atr_period = 14
        try:
            if settings is not None and hasattr(settings, "risk"):
                atr_period = int(getattr(settings.risk, "atr_period", 14) or 14)
        except Exception:
            atr_period = 14
        window = max(2, atr_period)
        if tr.size < window:
            atr = float(np.nanmean(tr))
        else:
            atr = float(np.nanmean(tr[-window:]))
    else:
        atr = float(atr_arr[-1])

    if not np.isfinite(atr) or atr <= 0.0:
        return None

    atr_mult = float(max(0.1, atr_mult))
    min_rr = float(max(0.1, min_rr))
    min_dist = float(max(0.0, min_dist))
    sl_dist = max(atr * atr_mult, min_dist)
    if sl_dist <= 0.0:
        return None
    sl_pips = float(sl_dist / max(pip_size, 1e-9))
    tp_pips = float(sl_pips * min_rr)
    return sl_pips, tp_pips


def fast_evaluate_strategy(
    *,
    close_prices: Iterable,
    high_prices: Iterable,
    low_prices: Iterable,
    signals: Iterable,
    month_indices: Iterable,
    day_indices: Iterable,
    sl_pips: float,
    tp_pips: float,
    max_hold_bars: int = 0,
    trailing_enabled: bool = False,
    trailing_atr_multiplier: float = 1.0,
    trailing_be_trigger_r: float = 1.0,
    pip_value: float = 0.0001,
    spread_pips: float = 1.5,
    commission_per_trade: float = 0.0,
    pip_value_per_lot: float = 10.0,
) -> np.ndarray:
    _require_rust()

    close_arr = _as_contig(close_prices, np.float64)
    high_arr = _as_contig(high_prices, np.float64)
    low_arr = _as_contig(low_prices, np.float64)
    sig_arr = _as_contig(signals, np.int8)
    month_arr = _as_contig(month_indices, np.int64)
    day_arr = _as_contig(day_indices, np.int64)

    if _forex_core is None:
        # Optional fallback path (very slow); return zeros if explicitly allowed.
        return np.zeros(11, dtype=np.float64)

    out = _forex_core.fast_evaluate_strategy(
        close_arr,
        high_arr,
        low_arr,
        sig_arr,
        month_arr,
        day_arr,
        float(sl_pips),
        float(tp_pips),
        int(max_hold_bars),
        bool(trailing_enabled),
        float(trailing_atr_multiplier),
        float(trailing_be_trigger_r),
        float(pip_value),
        float(spread_pips),
        float(commission_per_trade),
        float(pip_value_per_lot),
    )
    return np.asarray(out, dtype=np.float64)


def batch_evaluate_strategies(
    close_prices: Iterable,
    high_prices: Iterable,
    low_prices: Iterable,
    signals: Iterable,
    month_indices: Iterable,
    day_indices: Iterable,
    sl_pips: Iterable,
    tp_pips: Iterable,
    max_hold_bars: int = 0,
    trailing_enabled: bool = False,
    trailing_atr_multiplier: float = 1.0,
    trailing_be_trigger_r: float = 1.0,
    pip_value: float = 0.0001,
    spread_pips: float = 1.5,
    commission_per_trade: float = 0.0,
    pip_value_per_lot: float = 10.0,
) -> np.ndarray:
    _require_rust()

    close_arr = _as_contig(close_prices, np.float64)
    high_arr = _as_contig(high_prices, np.float64)
    low_arr = _as_contig(low_prices, np.float64)
    sig_arr = _as_contig(signals, np.int8)
    month_arr = _as_contig(month_indices, np.int64)
    day_arr = _as_contig(day_indices, np.int64)
    sl_arr = _as_contig(sl_pips, np.float64)
    tp_arr = _as_contig(tp_pips, np.float64)

    if _forex_core is None:
        return np.zeros((0, 11), dtype=np.float64)

    out = _forex_core.batch_evaluate_strategies(
        close_arr,
        high_arr,
        low_arr,
        sig_arr,
        month_arr,
        day_arr,
        sl_arr,
        tp_arr,
        int(max_hold_bars),
        bool(trailing_enabled),
        float(trailing_atr_multiplier),
        float(trailing_be_trigger_r),
        float(pip_value),
        float(spread_pips),
        float(commission_per_trade),
        float(pip_value_per_lot),
    )
    return np.asarray(out, dtype=np.float64)

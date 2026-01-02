from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import Settings

logger = logging.getLogger(__name__)


def _safe_log(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return np.log(np.clip(arr, 1e-12, None))


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def _rolling_var(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.zeros_like(values)
    return pd.Series(values).rolling(window, min_periods=window).var(ddof=1).to_numpy()


def _vol_parkinson(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    hl = _safe_log(high) - _safe_log(low)
    return (hl * hl) / (4.0 * np.log(2.0))


def _vol_ewma(close: np.ndarray, *, window: int, lam: float) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if n < 2:
        return np.zeros_like(close)
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0.0 or lam >= 1.0:
        lam = 0.94
    r = np.diff(_safe_log(close))
    var = np.full(n, np.nan, dtype=np.float64)
    if window <= 1:
        init = float(np.nanmean(r * r)) if r.size > 0 else 0.0
    else:
        span = min(window, r.size)
        init = float(np.nanmean(r[:span] * r[:span])) if span > 0 else 0.0
    if not np.isfinite(init):
        init = 0.0
    if n > 1:
        var[1] = init
    for i in range(1, r.size):
        prev = var[i]
        if not np.isfinite(prev):
            prev = init
        var[i + 1] = (lam * prev) + ((1.0 - lam) * (r[i] * r[i]))
    sigma = np.sqrt(np.maximum(var, 0.0))
    last = 0.0
    for i in range(n):
        val = sigma[i]
        if np.isfinite(val) and val > 0:
            last = float(val)
        else:
            sigma[i] = last
    return sigma


try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    prange = range

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, parallel=True)
    def _vol_parkinson_numba(high, low, window):
        n = len(high)
        out = np.zeros(n, dtype=np.float64)
        hl = np.log(high) - np.log(low)
        sq_hl = hl * hl
        
        for i in prange(window, n):
            # Rolling mean of squared HL
            out[i] = np.sqrt(np.mean(sq_hl[i-window+1 : i+1]) / (4.0 * np.log(2.0)))
        return out

    @njit(cache=True, fastmath=True)
    def _estimate_hurst_fast(series):
        # Fast Hurst approximation without heavy polyfit
        n = len(series)
        if n < 20: return 0.5
        
        # Simple rescaled range approach
        m = np.mean(series)
        z = np.cumsum(series - m)
        r = np.max(z) - np.min(z)
        s = np.std(series)
        if s < 1e-12: return 0.5
        return math.log(r/s) / math.log(n)
else:
    def _vol_parkinson_numba(high, low, window):
        n = len(high)
        out = np.zeros(n, dtype=np.float64)
        hl = np.log(high) - np.log(low)
        sq_hl = hl * hl
        
        # Vectorized rolling mean
        series = pd.Series(sq_hl)
        out = np.sqrt(series.rolling(window).mean() / (4.0 * np.log(2.0))).fillna(0.0).values
        return out

    def _estimate_hurst_fast(series):
        n = len(series)
        if n < 20: return 0.5
        m = np.mean(series)
        z = np.cumsum(series - m)
        r = np.max(z) - np.min(z)
        s = np.std(series)
        if s < 1e-12: return 0.5
        return math.log(r/s) / math.log(n)


def _vol_rogers_satchell(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    ho = _safe_log(high) - _safe_log(open_)
    hc = _safe_log(high) - _safe_log(close)
    lo = _safe_log(low) - _safe_log(open_)
    lc = _safe_log(low) - _safe_log(close)
    return (ho * hc) + (lo * lc)


def _vol_garman_klass(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    h = _safe_log(high)
    l = _safe_log(low)
    o = _safe_log(open_)
    c = _safe_log(close)
    hl = h - l
    co = c - o
    return 0.5 * (hl * hl) - (2.0 * np.log(2.0) - 1.0) * (co * co)


def _vol_yang_zhang(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    if len(close) < 2:
        return np.zeros_like(close)
    o = _safe_log(open_)
    c = _safe_log(close)
    h = _safe_log(high)
    l = _safe_log(low)

    o_ret = o[1:] - c[:-1]  # log(O_t / C_{t-1})
    c_ret = c[1:] - o[1:]   # log(C_t / O_t)
    rs = _vol_rogers_satchell(open_[1:], high[1:], low[1:], close[1:])

    # Align to close index by padding a leading nan
    o_ret = np.concatenate(([np.nan], o_ret))
    c_ret = np.concatenate(([np.nan], c_ret))
    rs = np.concatenate(([np.nan], rs))

    k = 0.0
    if window > 1:
        k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))

    sigma_o2 = _rolling_var(o_ret, window)
    sigma_c2 = _rolling_var(c_ret, window)
    sigma_rs2 = _rolling_mean(rs, window)

    sigma2 = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
    sigma2 = np.where(np.isfinite(sigma2), sigma2, 0.0)
    return np.sqrt(np.maximum(sigma2, 0.0))


def _normalize_ensemble_weights(weights: dict[str, float] | None) -> dict[str, float] | None:
    if not weights:
        return None
    normalized: dict[str, float] = {}
    for key, value in weights.items():
        if value is None:
            continue
        name = str(key).strip().lower()
        if name in {"yz", "yang-zhang"}:
            name = "yang_zhang"
        elif name in {"gk", "garman-klass", "garman klass"}:
            name = "garman_klass"
        elif name in {"rs", "rogers-satchell", "rogers satchell"}:
            name = "rogers_satchell"
        elif name in {"pk", "park"}:
            name = "parkinson"
        try:
            val = float(value)
        except Exception:
            continue
        if val > 0:
            normalized[name] = val
    if not normalized:
        return None
    total = float(sum(normalized.values()))
    if total <= 0:
        return None
    for key in list(normalized.keys()):
        normalized[key] = normalized[key] / total
    return normalized


def _get_ensemble_weights(settings: Settings | None, regime: str | None = None) -> dict[str, float] | None:
    if settings is None:
        return None
    if regime:
        override = getattr(settings.risk, f"vol_ensemble_weights_{regime}", None)
        if isinstance(override, dict) and override:
            return _normalize_ensemble_weights(override)
    base = getattr(settings.risk, "vol_ensemble_weights", None)
    if isinstance(base, dict) and base:
        return _normalize_ensemble_weights(base)
    return None


def _resolve_ewma_lambda(settings: Settings | None, df: pd.DataFrame | None = None) -> float:
    lam = 0.94
    if settings is None:
        return lam
    try:
        lam = float(getattr(settings.risk, "ewma_lambda", 0.94) or 0.94)
    except Exception:
        lam = 0.94
    tf_map = getattr(settings.risk, "ewma_lambda_by_timeframe", None)
    tf = None
    if df is not None:
        try:
            tf = df.attrs.get("timeframe") or df.attrs.get("tf")
        except Exception:
            tf = None
    if not tf:
        try:
            tf = getattr(settings.system, "base_timeframe", None)
        except Exception:
            tf = None
    if tf_map and tf:
        try:
            tf_key = str(tf).strip().upper()
            if tf_key in tf_map:
                lam = float(tf_map[tf_key])
        except Exception:
            pass
    if not np.isfinite(lam) or lam <= 0.0 or lam >= 1.0:
        lam = 0.94
    return lam


def estimate_volatility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    window: int,
    method: str,
    weights: dict[str, float] | None = None,
    ewma_lambda: float | None = None,
) -> np.ndarray:
    method = (method or "yang_zhang").lower()
    if method in {"ensemble", "mix", "blend"}:
        v_pk = _vol_parkinson(high, low)
        sigma_pk = np.sqrt(np.maximum(_rolling_mean(v_pk, window), 0.0))

        v_gk = _vol_garman_klass(open_, high, low, close)
        sigma_gk = np.sqrt(np.maximum(_rolling_mean(v_gk, window), 0.0))

        v_rs = _vol_rogers_satchell(open_, high, low, close)
        sigma_rs = np.sqrt(np.maximum(_rolling_mean(v_rs, window), 0.0))

        sigma_yz = _vol_yang_zhang(open_, high, low, close, window)

        stacked = np.vstack([sigma_yz, sigma_gk, sigma_rs, sigma_pk])
        weights = _normalize_ensemble_weights(weights)
        if weights:
            w = np.array(
                [
                    weights.get("yang_zhang", 0.0),
                    weights.get("garman_klass", 0.0),
                    weights.get("rogers_satchell", 0.0),
                    weights.get("parkinson", 0.0),
                ],
                dtype=np.float64,
            )
            if np.sum(w) > 0:
                return np.nansum(stacked * w[:, None], axis=0)
        return np.nanmedian(stacked, axis=0)
    if method in {"parkinson", "park"}:
        v = _vol_parkinson(high, low)
        return np.sqrt(np.maximum(_rolling_mean(v, window), 0.0))
    if method in {"garman_klass", "gk"}:
        v = _vol_garman_klass(open_, high, low, close)
        return np.sqrt(np.maximum(_rolling_mean(v, window), 0.0))
    if method in {"rogers_satchell", "rs"}:
        v = _vol_rogers_satchell(open_, high, low, close)
        return np.sqrt(np.maximum(_rolling_mean(v, window), 0.0))
    if method in {"ewma", "riskmetrics"}:
        lam = float(ewma_lambda) if ewma_lambda is not None else 0.94
        return _vol_ewma(close, window=window, lam=lam)
    # Default: Yang-Zhang
    return _vol_yang_zhang(open_, high, low, close, window)


def estimate_expected_shortfall(
    close: np.ndarray,
    *,
    window: int,
    alpha: float,
) -> float | None:
    if window <= 2 or len(close) <= 2:
        return None
    r = np.diff(_safe_log(close))
    if len(r) < window:
        return None
    tail = r[-window:]
    q = np.quantile(tail, 1.0 - alpha)
    losses = tail[tail <= q]
    if losses.size == 0:
        return None
    return float(abs(np.mean(losses)))


def estimate_expected_shortfall_series(
    close: np.ndarray,
    *,
    window: int,
    alpha: float,
    step: int,
    max_bars: int,
) -> np.ndarray | None:
    if window <= 2 or len(close) <= 2:
        return None
    if len(close) > max_bars:
        return None
    r = np.diff(_safe_log(close))
    if len(r) < window:
        return None
    step = max(1, int(step))
    es = np.full(len(r), np.nan, dtype=np.float64)
    for i in range(window - 1, len(r), step):
        win = r[i - window + 1 : i + 1]
        q = np.quantile(win, 1.0 - alpha)
        tail = win[win <= q]
        if tail.size > 0:
            es[i] = float(np.mean(np.abs(tail)))
    # Forward-fill sparse ES values to align with time progression.
    es = pd.Series(es).ffill().to_numpy(dtype=np.float64)
    # Align to close length by padding a leading nan.
    return np.concatenate(([np.nan], es))


def estimate_hurst(close: np.ndarray, *, window: int, max_lag: int = 20) -> float | None:
    if window <= 10 or len(close) <= window:
        return None
    series = np.diff(_safe_log(close[-(window + 1) :]))
    if len(series) < 20:
        return None
    max_lag = min(max_lag, max(2, len(series) // 2))
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(series[lag:] - series[:-lag]) for lag in lags])
    tau = np.where(tau > 0, tau, 1e-12)
    slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)
    if not np.isfinite(slope):
        return None
    return float(slope)


def infer_regime(
    df: pd.DataFrame,
    *,
    adx_trend: float,
    adx_range: float,
    hurst_window: int,
    hurst_trend: float,
    hurst_range: float,
) -> str:
    adx_val = None
    if "adx" in df.columns:
        try:
            adx_val = float(df["adx"].iloc[-1])
        except Exception:
            adx_val = None

    hurst_val = None
    if "close" in df.columns:
        try:
            close = df["close"].to_numpy(dtype=np.float64)
            hurst_val = estimate_hurst(close, window=hurst_window)
        except Exception:
            hurst_val = None

    if adx_val is not None and hurst_val is not None:
        if adx_val >= adx_trend and hurst_val >= hurst_trend:
            return "trend"
        if adx_val <= adx_range and hurst_val <= hurst_range:
            return "range"

    if adx_val is not None:
        if adx_val >= adx_trend:
            return "trend"
        if adx_val <= adx_range:
            return "range"

    if hurst_val is not None:
        if hurst_val >= hurst_trend:
            return "trend"
        if hurst_val <= hurst_range:
            return "range"
    # Fallback: EMA spread / ATR
    if {"ema_fast", "ema_slow", "atr"}.issubset(df.columns):
        try:
            atr = float(df["atr"].iloc[-1])
            if atr > 0:
                spread = abs(float(df["ema_fast"].iloc[-1]) - float(df["ema_slow"].iloc[-1]))
                strength = spread / atr
                if strength >= 0.6:
                    return "trend"
                if strength <= 0.3:
                    return "range"
        except Exception:
            pass
    return "neutral"


def compute_stop_distance_series(
    df: pd.DataFrame,
    *,
    settings: Settings,
) -> np.ndarray | None:
    if df is None or df.empty:
        return None

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    method = str(getattr(settings.risk, "vol_estimator", "yang_zhang"))
    window = int(getattr(settings.risk, "vol_window", 50) or 50)
    ewma_lambda = _resolve_ewma_lambda(settings, df)
    horizon = int(getattr(settings.risk, "vol_horizon_bars", 5) or 5)
    tail_window = int(getattr(settings.risk, "tail_window", 100) or 100)
    tail_alpha = float(getattr(settings.risk, "tail_alpha", 0.975) or 0.975)
    tail_step = int(getattr(settings.risk, "tail_step", 5) or 5)
    tail_max_bars = int(getattr(settings.risk, "tail_max_bars", 300_000) or 300_000)
    k_vol = float(getattr(settings.risk, "stop_k_vol", 1.0) or 1.0)
    k_tail = float(getattr(settings.risk, "stop_k_tail", 1.25) or 1.25)
    min_dist = float(getattr(settings.risk, "meta_label_min_dist", 0.0) or 0.0)

    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    if len(close) < max(window, tail_window, 5):
        return None

    weights = _get_ensemble_weights(settings)
    adx_trend = float(getattr(settings.risk, "regime_adx_trend", 25.0) or 25.0)
    adx_range = float(getattr(settings.risk, "regime_adx_range", 20.0) or 20.0)
    hurst_window = int(getattr(settings.risk, "hurst_window", 100) or 100)
    hurst_trend = float(getattr(settings.risk, "hurst_trend", 0.55) or 0.55)
    hurst_range = float(getattr(settings.risk, "hurst_range", 0.45) or 0.45)
    regime = infer_regime(
        df,
        adx_trend=adx_trend,
        adx_range=adx_range,
        hurst_window=hurst_window,
        hurst_trend=hurst_trend,
        hurst_range=hurst_range,
    )
    weights = _get_ensemble_weights(settings, regime=regime)
    sigma = estimate_volatility(
        open_,
        high,
        low,
        close,
        window=window,
        method=method,
        weights=weights,
        ewma_lambda=ewma_lambda,
    )
    vol_dist = close * sigma * float(np.sqrt(max(1, horizon)))

    es_series = estimate_expected_shortfall_series(
        close,
        window=tail_window,
        alpha=tail_alpha,
        step=tail_step,
        max_bars=tail_max_bars,
    )
    if es_series is not None:
        tail_dist = close * es_series * float(np.sqrt(max(1, horizon)))
    else:
        tail_dist = np.zeros_like(vol_dist)

    dist = np.maximum(k_vol * vol_dist, k_tail * tail_dist)
    if min_dist > 0:
        dist = np.maximum(dist, min_dist)

    dist = dist.astype(np.float64, copy=False)
    dist[~np.isfinite(dist)] = np.nan

    med = np.nanmedian(dist)
    if not np.isfinite(med) or med <= 0:
        return None

    dist = np.where(np.isfinite(dist), dist, med)
    return dist


def infer_stop_target_pips(
    df: pd.DataFrame,
    *,
    settings: Settings,
    pip_size: float,
    symbol_info: dict[str, Any] | None = None,
) -> tuple[float, float, float] | None:
    if df is None or df.empty:
        return None

    mode = str(getattr(settings.risk, "stop_target_mode", "blend")).lower()
    if mode not in {"blend", "auto"}:
        return None

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    # Configurable parameters (defaults are conservative).
    method = str(getattr(settings.risk, "vol_estimator", "yang_zhang"))
    window = int(getattr(settings.risk, "vol_window", 50) or 50)
    ewma_lambda = _resolve_ewma_lambda(settings, df)
    horizon = int(getattr(settings.risk, "vol_horizon_bars", 5) or 5)
    tail_window = int(getattr(settings.risk, "tail_window", 100) or 100)
    tail_alpha = float(getattr(settings.risk, "tail_alpha", 0.975) or 0.975)
    k_vol = float(getattr(settings.risk, "stop_k_vol", 1.0) or 1.0)
    k_tail = float(getattr(settings.risk, "stop_k_tail", 1.25) or 1.25)
    min_dist = float(getattr(settings.risk, "meta_label_min_dist", 0.0) or 0.0)
    adx_trend = float(getattr(settings.risk, "regime_adx_trend", 25.0) or 25.0)
    adx_range = float(getattr(settings.risk, "regime_adx_range", 20.0) or 20.0)
    hurst_window = int(getattr(settings.risk, "hurst_window", 100) or 100)
    hurst_trend = float(getattr(settings.risk, "hurst_trend", 0.55) or 0.55)
    hurst_range = float(getattr(settings.risk, "hurst_range", 0.45) or 0.45)
    rr_trend = float(getattr(settings.risk, "rr_trend", 2.5) or 2.5)
    rr_range = float(getattr(settings.risk, "rr_range", 1.5) or 1.5)
    rr_neutral = float(getattr(settings.risk, "rr_neutral", (rr_trend + rr_range) / 2.0))

    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    if len(close) < max(window, tail_window, 5):
        return None

    regime = infer_regime(
        df,
        adx_trend=adx_trend,
        adx_range=adx_range,
        hurst_window=hurst_window,
        hurst_trend=hurst_trend,
        hurst_range=hurst_range,
    )
    weights = _get_ensemble_weights(settings, regime=regime)
    sigma = estimate_volatility(
        open_,
        high,
        low,
        close,
        window=window,
        method=method,
        weights=weights,
        ewma_lambda=ewma_lambda,
    )
    sigma_last = float(sigma[-1]) if np.isfinite(sigma[-1]) else 0.0

    es = estimate_expected_shortfall(close, window=tail_window, alpha=tail_alpha)
    es = float(es) if es is not None else 0.0

    price = float(close[-1])
    scale = float(np.sqrt(max(1, horizon)))
    vol_dist = price * sigma_last * scale
    tail_dist = price * es * scale
    dist = max(k_vol * vol_dist, k_tail * tail_dist, min_dist)
    if not np.isfinite(dist) or dist <= 0:
        return None

    sl_pips = dist / max(float(pip_size), 1e-9)

    if regime == "trend":
        rr = rr_trend
    elif regime == "range":
        rr = rr_range
    else:
        rr = rr_neutral

    tp_pips = sl_pips * rr
    if not np.isfinite(tp_pips) or tp_pips <= 0:
        return None

    return float(sl_pips), float(tp_pips), float(rr)

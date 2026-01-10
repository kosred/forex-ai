from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import Settings


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean().iloc[-1]
    try:
        return float(atr)
    except Exception:
        return 0.0


def infer_stop_target_pips(
    df: pd.DataFrame,
    *,
    settings: Settings,
    pip_size: float,
) -> tuple[float, float, float] | None:
    if df is None or df.empty:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    atr = _atr(high, low, close)
    if not np.isfinite(atr) or atr <= 0.0:
        return None
    atr_mult = float(getattr(settings.risk, "atr_stop_multiplier", 1.5) or 1.5)
    min_dist = float(getattr(settings.risk, "meta_label_min_dist", 0.0) or 0.0)
    sl_dist = max(atr * max(atr_mult, 0.1), min_dist)
    if sl_dist <= 0:
        return None
    rr = float(getattr(settings.risk, "min_risk_reward", 2.0) or 2.0)
    sl_pips = float(sl_dist / max(pip_size, 1e-9))
    tp_pips = float(sl_pips * rr)
    return sl_pips, tp_pips, rr


__all__ = ["infer_stop_target_pips"]

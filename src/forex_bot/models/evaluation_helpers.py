from typing import Any

import numpy as np
import pandas as pd


def probs_to_signals(probs: np.ndarray) -> np.ndarray:
    sig_idx = probs.argmax(axis=1)
    signals = np.zeros(len(probs), dtype=int)
    signals[sig_idx == 1] = 1
    signals[sig_idx == 2] = -1
    return signals


def simple_backtest(df: pd.DataFrame, signals: pd.Series) -> dict[str, Any]:
    if df is None or signals is None or len(df) == 0 or len(signals) != len(df):
        return {}
    close = df["close"].to_numpy()
    future = np.roll(close, -1)
    ret = future - close
    pnl = []
    for s, r in zip(signals.to_numpy(), ret, strict=False):
        if s == 0:
            pnl.append(0.0)
        elif s == 1:
            pnl.append(1.0 if r > 0 else -1.0)
        else:
            pnl.append(1.0 if r < 0 else -1.0)
    pnl = np.array(pnl[:-1])
    return {
        "pnl_score": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()) if len(pnl) > 0 else 0.0,
        "trades": int(np.count_nonzero(signals)),
    }

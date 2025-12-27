from typing import Any

import numpy as np
import pandas as pd

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False


def pad_probs(probs: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
    """
    HPC UNIFIED PROTOCOL: Force output to [Neutral, Buy, Sell].
    Standard indices: 0=Neutral, 1=Buy, 2=Sell.
    """
    if probs is None or len(probs) == 0:
        return np.zeros((0, 3), dtype=float)
    
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    out = np.zeros((n, 3), dtype=float)

    # 1. Handle Model-Specific Mapping (If classes provided)
    if classes is not None and len(classes) == arr.shape[1]:
        for col, cls_val in enumerate(classes):
            # Map from Project Labels {-1, 0, 1} to Unified Indices {0, 1, 2}
            if cls_val == 0: out[:, 0] = arr[:, col] # Neutral -> 0
            elif cls_val == 1: out[:, 1] = arr[:, col] # Buy -> 1
            elif cls_val == -1 or cls_val == 2: out[:, 2] = arr[:, col] # Sell -> 2
        return out

    # 2. Automated Heuristics (If no class map)
    if arr.shape[1] == 3:
        return arr # Assume [Neutral, Buy, Sell]
    elif arr.shape[1] == 2:
        out[:, 0] = arr[:, 0] # Neutral
        out[:, 1] = arr[:, 1] # Buy
    else:
        out[:, 0] = 1.0 - arr[:, 0] # Neutral
        out[:, 1] = arr[:, 0] # Buy
        
    return out


def probs_to_signals(probs: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
    """
    Map probability matrix [neutral, buy, sell] to discrete signals {-1,0,1}.
    Chooses highest prob; ties fall back to neutral.
    """
    p = pad_probs(probs, classes=classes)
    if len(p) == 0:
        return np.zeros(0, dtype=int)
    sig_idx = p.argmax(axis=1)
    signals = np.zeros(len(p), dtype=int)
    signals[sig_idx == 1] = 1
    signals[sig_idx == 2] = -1
    return signals


def quick_backtest(df: pd.DataFrame, signals: pd.Series) -> dict[str, Any]:
    """
    Lightweight backtest: assumes df has 'close', uses next-bar direction as outcome.
    Returns accuracy and simple PnL proxy.
    """
    if df is None or signals is None or len(df) == 0 or len(signals) != len(df):
        return {}
    close = df["close"].to_numpy()
    future = np.roll(close, -1)
    ret = future - close
    sig_arr = signals.to_numpy()

    pnl = np.where(
        sig_arr == 0, 0.0, np.where(sig_arr == 1, np.where(ret > 0, 1.0, -1.0), np.where(ret < 0, 1.0, -1.0))
    )
    pnl = pnl[:-1]
    acc = float((sig_arr[:-1] == np.sign(ret[:-1])).mean()) if len(ret) > 1 else 0.0
    return {
        "accuracy": acc,
        "pnl_score": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()) if len(pnl) > 0 else 0.0,
    }


from ..strategy.fast_backtest import fast_evaluate_strategy, infer_pip_metrics

def prop_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    max_daily_dd_pct: float = 0.05,
    daily_dd_warn_pct: float = 0.03,
    max_trades_per_day: int = 10,
    use_gpu: bool | None = None,
) -> dict[str, Any]:
    """
    HPC Unified Backtest: Uses the master Numba engine for identical results.
    """
    if df is None or signals is None or len(df) == 0 or len(signals) != len(df):
        return {}

    # 1. Prepare Inputs
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    sig_arr = signals.to_numpy(dtype=np.int8)
    
    # 2. Extract Time Indices
    idx = df.index
    month_idx = (idx.year.astype(np.int32) * 12 + idx.month.astype(np.int32)).to_numpy(dtype=np.int64)
    day_idx = (idx.year.astype(np.int32) * 10000 + idx.month.astype(np.int32) * 100 + idx.day.astype(np.int32)).to_numpy(dtype=np.int64)
    
    # 3. Dynamic Pip Metrics
    symbol = df.attrs.get("symbol", "EURUSD")
    pip_size, pip_val_lot = infer_pip_metrics(symbol)
    
    # 4. Run Master Engine
    # Note: We use 30 pip SL and 60 pip TP as defaults for scoring consistency
    arr = fast_evaluate_strategy(
        close_prices=close,
        high_prices=high,
        low_prices=low,
        signals=sig_arr,
        month_indices=month_idx,
        day_indices=day_idx,
        sl_pips=30.0,
        tp_pips=60.0,
        pip_value=pip_size,
        pip_value_per_lot=pip_val_lot,
        spread_pips=1.5,
        commission_per_trade=7.0
    )
    
    keys = ["net_profit", "sharpe", "sortino", "max_dd_pct", "win_rate", "profit_factor", "expectancy", "sqn", "trades", "consistency", "daily_dd"]
    return {k: float(v) for k, v in zip(keys, arr.tolist())}

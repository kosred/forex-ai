import numpy as np
import pandas as pd

from forex_bot.core.config import Settings
from forex_bot.features.pipeline import FeatureEngineer


def _make_ohlcv(index: pd.DatetimeIndex, *, start: float) -> pd.DataFrame:
    close = np.linspace(start, start + 1.0, len(index), dtype=np.float64)
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    volume = np.full(len(index), 100.0, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_mtf_merge_uses_last_closed_htf_bar():
    settings = Settings()
    settings.system.base_timeframe = "M1"
    settings.system.cache_enabled = False

    fe = FeatureEngineer(settings)

    # Cover >1 H1 candle so HTF features vary (otherwise constant columns get dropped as near-zero variance).
    idx_m1 = pd.date_range("2025-01-01 10:00", periods=120, freq="1min", tz="UTC")
    base = _make_ohlcv(idx_m1, start=100.0)

    idx_h1 = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01 09:00", tz="UTC"),
            pd.Timestamp("2025-01-01 10:00", tz="UTC"),
            pd.Timestamp("2025-01-01 11:00", tz="UTC"),
        ]
    )
    h1 = _make_ohlcv(idx_h1, start=200.0)

    ds = fe.prepare({"M1": base, "H1": h1}, symbol="EURUSD")

    # Reproduce the HTF feature pipeline path and expected merge behavior.
    htf = h1.copy()
    htf = fe._compute_basic_features(htf, use_gpu=False)
    htf = fe._compute_volatility_features(htf)
    htf = fe._compute_volume_profile_features(htf)
    htf = fe._compute_obi_features(htf, use_gpu=False)

    key_feats = [
        "rsi",
        "macd",
        "macd_hist",
        "atr14",
        "bb_width",
        "dist_to_poc",
        "in_value_area",
        "vol_imbalance",
    ]
    subset = htf[[c for c in key_feats if c in htf.columns]].copy().shift(1)
    subset.columns = [f"H1_{c}" for c in subset.columns]
    expected = subset.reindex(ds.X.index, method="ffill").fillna(0.0)

    assert "H1_macd_hist" in ds.X.columns
    np.testing.assert_allclose(
        ds.X["H1_macd_hist"].to_numpy(dtype=float),
        expected["H1_macd_hist"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-7,
    )

    # Spot-check: within the 10:00 hour, values must come from the 09:00 H1 candle (after shift).
    t1 = pd.Timestamp("2025-01-01 10:30", tz="UTC")
    assert t1 in ds.X.index
    prev_10 = float(htf.loc[pd.Timestamp("2025-01-01 09:00", tz="UTC"), "macd_hist"])
    got_10 = float(ds.X.loc[t1, "H1_macd_hist"])
    assert abs(got_10 - prev_10) < 1e-7

    # And within the 11:00 hour, values must come from the 10:00 H1 candle.
    t2 = pd.Timestamp("2025-01-01 11:30", tz="UTC")
    assert t2 in ds.X.index
    prev_11 = float(htf.loc[pd.Timestamp("2025-01-01 10:00", tz="UTC"), "macd_hist"])
    got_11 = float(ds.X.loc[t2, "H1_macd_hist"])
    assert abs(got_11 - prev_11) < 1e-7

import numpy as np

from forex_bot.features.pipeline import _compute_adx_numba


def test_adx_calculation_continuity():
    """
    Verify that ADX calculation does not skip the initialization index (period).
    """
    n = 50
    period = 14
    rng = np.random.default_rng(seed=42)

    # Simple synthetic data to ensure directional movement
    high = np.linspace(100, 150, n) + rng.normal(scale=0.1, size=n)
    low = high - 1.0
    close = (high + low) / 2

    adx = _compute_adx_numba(high, low, close, period=period)

    # Check that we don't have a zero at index 'period' (index 14) which would indicate a skip
    # IF there is a trend (which linspace ensures).

    # Specifically, the smoothing relies on previous values.
    # If index 14 is skipped (0.0), then index 15 would be calculated from 0.0, dropping the value significantly.

    # We just assert it's not zero, assuming there's trend.
    assert adx[period] != 0.0, "ADX at index 'period' should not be zero (init skip bug)"
    assert not np.isnan(adx).any(), "ADX should not contain NaNs"

import numpy as np
import pandas as pd

from forex_bot.features import talib_mixer as tm
from forex_bot.features.talib_mixer import TALibStrategyGene, TALibStrategyMixer


class _DummyFunc:
    def __init__(self, name: str):
        self.name = name
        self.info = {"parameters": {}}

    def __call__(self, df, **_params):
        return pd.Series(np.linspace(0.0, 1.0, len(df)), index=df.index)


class _DummyAbstract:
    Function = _DummyFunc


def test_compute_signals_without_cache(monkeypatch):
    monkeypatch.setattr(tm, "TALIB_AVAILABLE", True)
    monkeypatch.setattr(tm, "abstract", _DummyAbstract)
    monkeypatch.setattr(tm, "ALL_INDICATORS", ["RSI"])
    monkeypatch.setattr(tm, "TALIB_INDICATORS", {"momentum": ["RSI"]})

    mixer = TALibStrategyMixer()
    idx = pd.date_range("2020-01-01", periods=50, freq="min")
    df = pd.DataFrame({"close": np.linspace(1.0, 2.0, len(idx))}, index=idx)
    gene = TALibStrategyGene(
        indicators=["RSI"],
        params={"RSI": {"timeperiod": 14}},
        weights={"RSI": 1.0},
        long_threshold=0.0,
        short_threshold=0.0,
        strategy_id="unit",
    )

    sig = mixer.compute_signals(df, gene, cache=None)
    assert len(sig) == len(df)
    assert (sig != 0).any()

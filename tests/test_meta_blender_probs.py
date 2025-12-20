import numpy as np
import pandas as pd

from forex_bot.training.ensemble import MetaBlender


class _StubProbaModel:
    def __init__(self, probs: np.ndarray, classes: list[int]) -> None:
        self._probs = np.asarray(probs, dtype=float)
        self.classes_ = np.asarray(classes, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        n = len(X)
        base = self._probs.reshape(1, -1) if self._probs.ndim == 1 else self._probs
        if len(base) == n:
            return base
        return np.repeat(base[:1], n, axis=0)


def test_meta_blender_reorders_to_neutral_buy_sell():
    blender = MetaBlender()
    blender.model = _StubProbaModel(probs=np.array([0.10, 0.20, 0.70]), classes=[-1, 0, 1])  # [-1,0,1] order
    blender.feature_columns = pd.Index(["f1"])

    out = blender.predict_proba(pd.DataFrame({"f1": [1.0, 2.0]}))
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], [0.20, 0.70, 0.10], rtol=0, atol=1e-12)


def test_meta_blender_pads_missing_class():
    blender = MetaBlender()
    blender.model = _StubProbaModel(probs=np.array([0.25, 0.75]), classes=[-1, 1])  # sell,buy only
    blender.feature_columns = pd.Index(["f1"])

    out = blender.predict_proba(pd.DataFrame({"f1": [1.0]}))
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0], [0.0, 0.75, 0.25], rtol=0, atol=1e-12)

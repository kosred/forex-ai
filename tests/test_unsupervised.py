import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from forex_bot.models.unsupervised import ClusterExpert


def test_cluster_expert_flow():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.001, size=1200)
    close = 100.0 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": close})

    model = ClusterExpert(n_regimes=5, lookback=1000)

    model.fit(df)
    assert model.is_fitted is True
    assert len(model.regime_map) > 0

    regime = model.predict(df)
    assert regime not in {"Unknown", "Error"}

    preds = model.predict_proba(df)
    assert preds.shape == (len(df), 3)
    assert np.allclose(preds.sum(axis=1), 1.0)


if __name__ == "__main__":
    test_cluster_expert_flow()

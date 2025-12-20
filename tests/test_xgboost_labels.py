import numpy as np
import pandas as pd
import pytest

from forex_bot.models.trees import XGB_AVAILABLE, XGBoostExpert


def test_xgboost_expert_accepts_negative_label_class():
    if not XGB_AVAILABLE:
        pytest.skip("xgboost not available")

    X = pd.DataFrame({"f1": np.linspace(-1.0, 1.0, 60), "f2": np.sin(np.linspace(0.0, 3.0, 60))})
    y = pd.Series(([0] * 20) + ([1] * 20) + ([-1] * 20), dtype=int)

    model = XGBoostExpert(
        params={
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.2,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 7,
            "n_jobs": 1,
            "verbosity": 0,
        }
    )
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-6)

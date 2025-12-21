from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CPCVResult:
    """Results from CPCV evaluation"""

    mean_score: float
    std_score: float
    scores: list[float]
    n_combinations: int
    phi: float  # Uniqueness score (lower = more overfitting risk)
    avg_max_dd: float = 0.0
    avg_trades: float = 0.0
    avg_win_rate: float = 0.0
    any_daily_loss_breach: bool = False
    any_trade_limit_violation: bool = False
    all_min_trading_days_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_score": float(self.mean_score),
            "std_score": float(self.std_score),
            "scores": [float(s) for s in self.scores],
            "n_combinations": int(self.n_combinations),
            "phi": float(self.phi),
            "avg_max_dd": float(self.avg_max_dd),
            "avg_trades": float(self.avg_trades),
            "avg_win_rate": float(self.avg_win_rate),
            "any_daily_loss_breach": bool(self.any_daily_loss_breach),
            "any_trade_limit_violation": bool(self.any_trade_limit_violation),
            "all_min_trading_days_ok": bool(self.all_min_trading_days_ok),
        }


class CombinatorialPurgedCV:
    """
    CPCV: The truth serum for backtesting.

    Unlike standard K-Fold, CPCV:
    - Purges samples with overlapping outcome periods from test set
    - Adds embargo periods between train/test to prevent information leakage
    - Tests all combinations of K groups, not just sequential splits

    Parameters
    ----------
    n_splits : int
        Number of groups to split data into (default 5)
    n_test_groups : int
        Number of groups to use for testing in each combination (default 2)
    embargo_pct : float
        Percentage of samples to embargo after train set (default 0.01 = 1%)
    purge_pct : float
        Percentage of samples to purge before test set (default 0.02 = 2%)
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.02,
    ) -> None:
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        x: pd.DataFrame,
        y: pd.Series | None = None,
        sample_weights: pd.Series | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorial purged train/test splits.

        A small warm-up window is always kept for training so that even the earliest
        test combinations have non-empty train sets. An embargo buffer is inserted
        between the warm-up window and the first test fold to guarantee a visible
        gap between train and test indices.

        Returns list of (train_indices, test_indices) tuples.
        """
        # FIX: Check for time-ordering
        if hasattr(x, "index") and not x.index.is_monotonic_increasing:
            logger.warning(
                "⚠️ CPCV Warning: Data index is NOT monotonic increasing. "
                "CPCV assumes time-ordered data. Purging/Embargo logic will be invalid on shuffled data!"
            )

        n_samples = len(x)
        if n_samples == 0:
            raise ValueError("CPCV requires non-empty data")

        purge_size = int(np.ceil(n_samples * self.purge_pct)) if self.purge_pct > 0 else 0
        embargo_size = int(np.ceil(n_samples * self.embargo_pct)) if self.embargo_pct > 0 else 0

        warmup_size = max(n_samples // self.n_splits, purge_size + embargo_size, self.n_splits)
        if warmup_size + self.n_splits >= n_samples:
            warmup_size = max(1, n_samples // (self.n_splits + 1))

        cv_start = min(n_samples, warmup_size + embargo_size)
        if n_samples - cv_start < self.n_splits:
            raise ValueError("Not enough samples after warm-up/embargo for requested splits")

        indices = np.arange(n_samples)
        base_train = indices[:warmup_size]
        cv_indices = indices[cv_start:]

        kfold = KFold(n_splits=self.n_splits, shuffle=False)
        groups: list[np.ndarray] = []
        for _, test_idx in kfold.split(cv_indices):
            groups.append(cv_indices[test_idx])

        test_combinations = list(combinations(range(self.n_splits), self.n_test_groups))
        logger.info(
            f"CPCV: {len(test_combinations)} combinations of {self.n_test_groups} test"
            f" groups from {self.n_splits} splits"
        )

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for test_group_indices in test_combinations:
            test_idx = np.concatenate([groups[i] for i in test_group_indices])
            test_idx.sort()

            if len(test_idx) == 0:
                splits.append((np.array([], dtype=int), np.array([], dtype=int)))
                continue

            earliest_group = min(test_group_indices)
            train_parts = [base_train]
            if earliest_group > 0:
                train_parts.extend(groups[i] for i in range(earliest_group) if i not in test_group_indices)

            train_idx = np.concatenate(train_parts) if len(train_parts) > 0 else np.array([], dtype=int)
            train_idx.sort()

            train_idx, test_idx = self._purge_and_embargo(train_idx, test_idx, n_samples, purge_size, embargo_size)

            if len(train_idx) == 0 and len(base_train) > 0:
                train_idx = base_train.copy()
                train_idx, test_idx = self._purge_and_embargo(train_idx, test_idx, n_samples, purge_size, embargo_size)

            splits.append((train_idx, test_idx))

        return splits

    def _purge_and_embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_samples: int,
        purge_size: int,
        embargo_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply purging and embargo to prevent data leakage.

        Purging: Remove train samples that overlap with test outcomes
        Embargo: Add time gap between train and test
        """
        if len(test_idx) == 0 or len(train_idx) == 0:
            return train_idx, test_idx

        test_start = int(test_idx[0])

        purge_threshold = max(0, test_start - purge_size)
        train_idx = train_idx[train_idx < purge_threshold]

        if len(train_idx) > 0:
            train_end = int(train_idx[-1])
            embargo_threshold = min(n_samples, train_end + embargo_size)
            test_idx = test_idx[test_idx >= embargo_threshold]

        return train_idx, test_idx

    def calculate_phi(
        self,
        x: pd.DataFrame,
    ) -> float:
        """
        Calculate uniqueness score φ (phi).

        φ = average number of times each sample appears in test set / n_combinations

        φ close to 1.0 = good (each sample tested ~once)
        φ much lower = potential overfitting (some samples rarely tested)
        """
        n_samples = len(x)
        test_counts = np.zeros(n_samples)

        splits = self.split(x)

        for _, test_idx in splits:
            test_counts[test_idx] += 1

        if n_samples == 0:
            return 0.0

        tested_mask = test_counts > 0
        coverage = tested_mask.mean()

        if tested_mask.any():
            avg_tests = float(test_counts[tested_mask].mean())
            frequency_score = avg_tests / max(1, len(splits))
        else:
            frequency_score = 0.0

        phi = float(np.clip(0.5 * coverage + 0.5 * frequency_score, 0.0, 1.0))
        return phi

    def score(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable,
        scoring_func: Callable | None = None,
        sample_weights: pd.Series | None = None,
        n_jobs: int = 1,
        max_daily_loss_pct: float = 0.05,
        max_trades_per_day: int = 15,
        min_trading_days: int = 3,
    ) -> CPCVResult:
        """
        Evaluate model using CPCV.

        Parameters
        ----------
        x : pd.DataFrame
            Features
        y : pd.Series
            Labels
        model_factory : Callable
            Function that returns a new untrained model instance
        scoring_func : Callable, optional
            Function(y_true, y_pred) -> score. If None, uses accuracy.
        sample_weights : pd.Series, optional
            Sample weights for training
        n_jobs : int
            Number of parallel jobs (default 1)

        Returns
        -------
        CPCVResult
            Evaluation results with mean score, std, and phi
        """
        splits = self.split(x, y, sample_weights)

        def default_scoring_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return (y_true == y_pred).mean()

        if scoring_func is None:
            scoring_func = default_scoring_func

        scores = []
        dd_list: list[float] = []
        trades_list: list[float] = []
        winrate_list: list[float] = []
        daily_loss_breaches = []
        trade_limit_breaches = []
        min_days_ok_list = []

        if n_jobs > 1:
            import os

            if os.name == "nt":
                logger.warning(
                    "CPCV: Forcing n_jobs=1 on Windows to avoid spawn recursion. "
                    "Set n_jobs=1 or wrap call in if __name__ == '__main__'."
                )
                n_jobs = 1

            # Use spawn context to avoid CUDA fork issues
            spawn_ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=spawn_ctx) as executor:
                futures = []
                for train_idx, test_idx in splits:
                    future = executor.submit(
                        self._evaluate_split, x, y, train_idx, test_idx, model_factory, scoring_func, sample_weights
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        score, metrics = future.result()
                        scores.append(score)
                        dd_list.append(metrics.get("max_dd", 0.0))
                        trades_list.append(metrics.get("trades", 0))
                        winrate_list.append(metrics.get("win_rate", 0.0))
                        daily_loss_breaches.append(metrics.get("daily_loss_breach", False))
                        trade_limit_breaches.append(metrics.get("trade_limit_violation", False))
                        min_days_ok_list.append(metrics.get("min_trading_days_ok", True))
                    except Exception as e:
                        logger.error(f"CPCV split failed: {e}")
                        scores.append(0.0)
                        dd_list.append(0.0)
                        trades_list.append(0)
                        winrate_list.append(0.0)
                        daily_loss_breaches.append(False)
                        trade_limit_breaches.append(False)
                        min_days_ok_list.append(False)
        else:
            for train_idx, test_idx in splits:
                try:
                    score, metrics = self._evaluate_split(
                        x,
                        y,
                        train_idx,
                        test_idx,
                        model_factory,
                        scoring_func,
                        sample_weights,
                        max_daily_loss_pct,
                        max_trades_per_day,
                        min_trading_days,
                    )
                    scores.append(score)
                    dd_list.append(metrics.get("max_dd", 0.0))
                    trades_list.append(metrics.get("trades", 0))
                    winrate_list.append(metrics.get("win_rate", 0.0))
                    daily_loss_breaches.append(metrics.get("daily_loss_breach", False))
                    trade_limit_breaches.append(metrics.get("trade_limit_violation", False))
                    min_days_ok_list.append(metrics.get("min_trading_days_ok", True))
                except Exception as e:
                    logger.error(f"CPCV split failed: {e}")
                    scores.append(0.0)
                    dd_list.append(0.0)
                    trades_list.append(0)
                    winrate_list.append(0.0)
                    daily_loss_breaches.append(False)
                    trade_limit_breaches.append(False)
                    min_days_ok_list.append(False)

        phi = self.calculate_phi(x)

        return CPCVResult(
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            scores=scores,
            n_combinations=len(splits),
            phi=phi,
            avg_max_dd=float(np.mean(dd_list)) if dd_list else 0.0,
            avg_trades=float(np.mean(trades_list)) if trades_list else 0.0,
            avg_win_rate=float(np.mean(winrate_list)) if winrate_list else 0.0,
            any_daily_loss_breach=any(daily_loss_breaches),
            any_trade_limit_violation=any(trade_limit_breaches),
            all_min_trading_days_ok=all(min_days_ok_list) if min_days_ok_list else False,
        )

    @staticmethod
    def _evaluate_split(
        x: pd.DataFrame,
        y: pd.Series,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        model_factory: Callable,
        scoring_func: Callable,
        sample_weights: pd.Series | None,
        max_daily_loss_pct: float,
        max_trades_per_day: int,
        min_trading_days: int,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a single train/test split"""
        if len(train_idx) == 0 or len(test_idx) == 0:
            return 0.0, {}

        x_train = x.iloc[train_idx] if hasattr(x, "iloc") else x[train_idx]
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        x_test = x.iloc[test_idx] if hasattr(x, "iloc") else x[test_idx]
        y_test = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]

        if sample_weights is not None:
            w_train = sample_weights.iloc[train_idx] if hasattr(sample_weights, "iloc") else sample_weights[train_idx]
        else:
            w_train = None

        model = model_factory()

        try:
            if w_train is not None and hasattr(model, "fit") and "sample_weight" in model.fit.__code__.co_varnames:
                model.fit(x_train, y_train, sample_weight=w_train)
            else:
                model.fit(x_train, y_train)
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return 0.0, {}

        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(x_test)
                classes = getattr(model, "classes_", None)
                if classes is not None and len(classes) == y_pred_proba.shape[1]:
                    y_pred = np.asarray(classes)[y_pred_proba.argmax(axis=1)]
                else:
                    y_pred = y_pred_proba.argmax(axis=1)
            else:
                y_pred = model.predict(x_test)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return 0.0, {}

        try:
            score = scoring_func(y_test.values, y_pred)
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return 0.0, {}

        trades = int(np.count_nonzero(y_pred))
        wins = int(np.sum((y_pred == 1) & (y_test == 1)) + np.sum((y_pred == -1) & (y_test == -1)))
        win_rate = float(wins / trades) if trades > 0 else 0.0

        max_dd = 0.0
        if hasattr(x_test, "index") and "close" in x_test.columns:
            try:
                close = x_test["close"].to_numpy(dtype=float)
                if len(close) < 2:
                    raise ValueError("Insufficient close prices for drawdown calc")

                # Use next-bar returns and drop the last bar (no future).
                base = np.clip(close[:-1], 1e-12, None)
                ret = (close[1:] - close[:-1]) / base
                pnl = []
                equity = 1.0
                peak = equity
                y_pred_arr = np.asarray(y_pred)[: len(ret)]
                for s, r in zip(y_pred_arr, ret, strict=False):
                    if s == 0:
                        pnl.append(0.0)
                    elif s == 1:
                        pnl.append(1.0 if r > 0 else -1.0)
                    else:
                        pnl.append(1.0 if r < 0 else -1.0)
                    equity *= 1.0 + (pnl[-1] * 0.001)
                    peak = max(peak, equity)
                    max_dd = max(max_dd, peak - equity)
            except Exception:
                max_dd = 0.0

        trade_limit_violation = trades > max_trades_per_day
        min_days_ok = trades > 0 or min_trading_days <= 1

        metrics = {
            "max_dd": float(max_dd),
            "trades": trades,
            "win_rate": win_rate,
            "daily_loss_breach": False,  # not tracked here due to missing timestamps
            "trade_limit_violation": trade_limit_violation,
            "min_trading_days_ok": min_days_ok,
        }

        return float(score), metrics


def cpcv_backtest(
    x: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    model_factory: Callable,
    n_splits: int = 5,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    purge_pct: float = 0.02,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """
    Run CPCV backtest with trading metrics.

    Returns detailed metrics including PnL, win rate, Sharpe, etc.
    """
    from ..training.evaluation import probs_to_signals, quick_backtest

    cv = CombinatorialPurgedCV(
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        embargo_pct=embargo_pct,
        purge_pct=purge_pct,
    )

    splits = cv.split(x, y)

    all_backtests = []

    for i, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_test = x.iloc[test_idx]

        model = model_factory()

        try:
            model.fit(x_train, y_train)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_test)
                signals = probs_to_signals(probs)
            else:
                signals = model.predict(x_test)

            test_metadata = metadata.iloc[test_idx]
            signals_series = pd.Series(signals, index=test_metadata.index)

            bt_metrics = quick_backtest(test_metadata, signals_series)

            if bt_metrics:
                bt_metrics["split"] = i + 1
                all_backtests.append(bt_metrics)

        except Exception as e:
            logger.error(f"CPCV backtest split {i + 1} failed: {e}")

    if not all_backtests:
        return {"n_splits": 0, "error": "All splits failed"}

    result = {
        "n_splits": len(all_backtests),
        "n_combinations": len(splits),
        "phi": cv.calculate_phi(x),
        "avg_pnl": np.mean([b.get("pnl_score", 0) for b in all_backtests]),
        "std_pnl": np.std([b.get("pnl_score", 0) for b in all_backtests]),
        "avg_win_rate": np.mean([b.get("win_rate", 0) for b in all_backtests]),
        "avg_sharpe": np.mean([b.get("sharpe", 0) for b in all_backtests]),
        "avg_trades": np.mean([b.get("trades", 0) for b in all_backtests]),
        "splits": all_backtests,
    }

    return result

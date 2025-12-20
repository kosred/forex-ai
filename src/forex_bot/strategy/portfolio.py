import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AllocationResult:
    symbol: str
    weight: float
    kelly_size: float
    risk_budget: float
    correlation_score: float
    sharpe: float


class PortfolioOptimizer:
    def __init__(
        self,
        cache_dir: Path,
        lookback_days: int = 30,
        max_weight: float = 0.35,
        kelly_fraction: float = 0.25,
    ) -> None:
        self.cache_dir = cache_dir
        self.lookback_days = lookback_days
        self.max_weight = max_weight
        self.kelly_fraction = kelly_fraction

    def get_optimal_allocation(self, symbols: list[str], metrics_map: dict[str, dict]) -> dict[str, AllocationResult]:
        if not symbols:
            return {}

        rets = []
        names = []
        for s in symbols:
            # Safe chained .get() - first get symbol metrics, then get returns
            symbol_metrics = metrics_map.get(s)
            if symbol_metrics is None:
                continue
            arr = symbol_metrics.get("returns")
            if isinstance(arr, (list, np.ndarray)) and len(arr) > 5:
                rets.append(np.array(arr, dtype=float))
                names.append(s)
        weights = {}
        if len(rets) >= 2:
            min_len = min(len(r) for r in rets)
            rets = [r[-min_len:] for r in rets]
            mat = np.vstack(rets)

            cov = np.cov(mat)
            corr_matrix = np.corrcoef(mat)
            vol = np.sqrt(np.diag(cov))

            # Safe sharpe calculation
            sharpe = {}
            for s in names:
                s_metrics = metrics_map.get(s, {})
                sharpe[s] = s_metrics.get("sharpe", 0.0) if s_metrics else 0.0

            n_assets = len(names)
            avg_corr = (np.sum(corr_matrix, axis=1) - 1) / max(1, n_assets - 1)
            div_score = 1.0 / (1.0 + np.abs(avg_corr))  # Higher score for lower correlation

            inv_vol = 1.0 / np.clip(vol, 1e-6, None)

            raw = inv_vol * np.maximum(0.01, np.array([sharpe[s] for s in names])) * div_score

            if raw.sum() <= 0:
                raw = np.ones_like(raw)
            norm = raw / raw.sum()

            capped = np.clip(norm, 0.0, self.max_weight)
            capped = capped / capped.sum()

            for w, s, _v, c_score in zip(capped, names, vol, avg_corr, strict=False):
                # Safe kelly size calculation
                s_metrics = metrics_map.get(s, {})
                win_rate = s_metrics.get("win_rate", 0.5) if s_metrics else 0.5
                # win_rate is already a scalar, no need for np.mean
                kelly_val = float(self.kelly_fraction * win_rate)

                weights[s] = AllocationResult(
                    s,
                    float(w),
                    kelly_size=kelly_val,
                    risk_budget=float(w),
                    correlation_score=float(c_score),  # Now effectively storing the correlation metric
                    sharpe=float(sharpe.get(s, 0.0)),
                )
        else:
            weight = 1.0 / len(symbols)
            for s in symbols:
                # Safe sharpe retrieval for single strategy
                s_metrics = metrics_map.get(s, {})
                s_sharpe = s_metrics.get("sharpe", 0.0) if s_metrics else 0.0

                weights[s] = AllocationResult(s, weight, 0.0, weight, 0.0, s_sharpe)
        return weights

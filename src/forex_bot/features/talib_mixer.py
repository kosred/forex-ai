from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import talib
    from talib import abstract

    TALIB_AVAILABLE = True
    ALL_INDICATORS = sorted(talib.get_functions())
except Exception:
    talib = None  # type: ignore
    abstract = None  # type: ignore
    TALIB_AVAILABLE = False
    ALL_INDICATORS: list[str] = []

TALIB_INDICATORS: dict[str, list[str]] = {
    "momentum": ["RSI", "ADX", "MACD"],
    "overlap": ["SMA", "EMA"],
    "volatility": ["ATR", "NATR"],
}


def _normalize_indicator_name(name: str) -> str:
    return str(name or "").strip().upper()


def _parse_synergy_key(key: str) -> tuple[str, str] | None:
    if not key:
        return None
    parts = str(key).split("_")
    if len(parts) != 2:
        return None
    return _normalize_indicator_name(parts[0]), _normalize_indicator_name(parts[1])


@dataclass(slots=True)
class TALibStrategyGene:
    indicators: list[str]
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    combination_method: str = "weighted_vote"
    long_threshold: float = 0.66
    short_threshold: float = -0.66
    weights: dict[str, float] = field(default_factory=dict)
    preferred_regime: str = "any"
    strategy_id: str = ""
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    use_ob: bool = False
    use_fvg: bool = False
    use_liq_sweep: bool = False
    mtf_confirmation: bool = False
    use_premium_discount: bool = False
    use_inducement: bool = False
    tp_pips: float = 40.0
    sl_pips: float = 20.0


class TALibStrategyMixer:
    def __init__(self, *, device: str = "cpu", use_volume_features: bool = False) -> None:
        self.device = device
        self.use_volume_features = use_volume_features
        self.available_indicators = [_normalize_indicator_name(i) for i in ALL_INDICATORS]
        self.indicator_synergy_matrix: dict[tuple[str, str], float] = {}
        self.regime_performance: dict[str, dict[str, float]] = {}

    def generate_random_strategy(self, *, max_indicators: int = 5) -> TALibStrategyGene:
        inds = [i for i in self.available_indicators if i]
        if not inds:
            return TALibStrategyGene(indicators=[])
        k = max(1, min(max_indicators, len(inds)))
        selected = random.sample(inds, k=k)
        weights = {i: float(random.uniform(0.5, 1.5)) for i in selected}
        params = {i: {} for i in selected}
        gene = TALibStrategyGene(
            indicators=selected,
            params=params,
            weights=weights,
            long_threshold=float(random.uniform(0.4, 1.0)),
            short_threshold=float(random.uniform(-1.0, -0.4)),
            strategy_id=f"gene_{random.randint(0, 1_000_000)}",
        )
        return gene

    def _compute_indicator(self, df: pd.DataFrame, indicator: str, params: dict[str, Any] | None) -> pd.Series:
        if abstract is None:
            raise RuntimeError("TA-Lib not available")
        func = abstract.Function(indicator)
        try:
            info = getattr(func, "info", {}) or {}
            defaults = info.get("parameters", {}) if isinstance(info, dict) else {}
        except Exception:
            defaults = {}
        merged = dict(defaults)
        if params:
            merged.update(params)
        output = func(df, **merged)
        if isinstance(output, pd.DataFrame):
            return output.iloc[:, 0]
        if isinstance(output, pd.Series):
            return output
        return pd.Series(np.asarray(output), index=df.index)

    def bulk_calculate_indicators(self, df: pd.DataFrame, population: list[TALibStrategyGene]) -> dict[str, pd.Series]:
        cache: dict[str, pd.Series] = {}
        if df is None or df.empty:
            return cache
        if not population:
            return cache
        needed: set[str] = set()
        for gene in population:
            for ind in gene.indicators:
                norm = _normalize_indicator_name(ind)
                if norm:
                    needed.add(norm)
        for ind in needed:
            try:
                params = None
                for gene in population:
                    if ind in gene.params:
                        params = gene.params.get(ind)
                        break
                cache[ind] = self._compute_indicator(df, ind, params)
            except Exception as exc:
                logger.debug("Indicator %s failed: %s", ind, exc)
                cache[ind] = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        return cache

    def compute_signals(
        self,
        df: pd.DataFrame,
        gene: TALibStrategyGene,
        *,
        cache: dict[str, pd.Series] | None = None,
    ) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(np.zeros(0, dtype=float), index=df.index)
        indicators = [_normalize_indicator_name(i) for i in gene.indicators]
        if not indicators:
            return pd.Series(np.zeros(len(df), dtype=float), index=df.index)

        votes = np.zeros(len(df), dtype=np.float64)
        weight_total = 0.0

        for ind in indicators:
            if not ind:
                continue
            series = None
            if cache is not None:
                series = cache.get(ind)
            if series is None:
                try:
                    params = gene.params.get(ind) if gene.params else None
                    series = self._compute_indicator(df, ind, params)
                except Exception as exc:
                    logger.debug("Indicator %s compute failed: %s", ind, exc)
                    series = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
            arr = np.asarray(series, dtype=np.float64)
            mean = float(np.nanmean(arr)) if arr.size else 0.0
            std = float(np.nanstd(arr)) if arr.size else 0.0
            if std <= 1e-9:
                norm = arr - mean
            else:
                norm = (arr - mean) / std
            score = np.tanh(norm)
            w = float(gene.weights.get(ind, 1.0)) if gene.weights else 1.0
            votes += w * score
            weight_total += abs(w)

        if weight_total <= 0.0:
            weight_total = 1.0
        combined = votes / weight_total
        long_thr = float(gene.long_threshold)
        short_thr = float(gene.short_threshold)
        signals = np.where(combined > long_thr, 1.0, np.where(combined < short_thr, -1.0, 0.0))
        return pd.Series(signals, index=df.index)

    def load_knowledge(self, path: str | Path) -> None:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load TA-Lib knowledge: %s", exc)
            return
        matrix = payload.get("synergy_matrix", {}) if isinstance(payload, dict) else {}
        for key, value in dict(matrix).items():
            pair = _parse_synergy_key(key)
            if pair is None:
                continue
            try:
                self.indicator_synergy_matrix[pair] = float(value)
            except Exception:
                continue
        regime = payload.get("regime_performance", {}) if isinstance(payload, dict) else {}
        if isinstance(regime, dict):
            self.regime_performance = {k: dict(v) if isinstance(v, dict) else {} for k, v in regime.items()}

    def save_knowledge(self, path: str | Path) -> None:
        out = {
            "synergy_matrix": {f"{a}_{b}": v for (a, b), v in self.indicator_synergy_matrix.items()},
            "regime_performance": self.regime_performance,
        }
        Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


__all__ = [
    "TALIB_AVAILABLE",
    "ALL_INDICATORS",
    "TALIB_INDICATORS",
    "TALibStrategyGene",
    "TALibStrategyMixer",
]

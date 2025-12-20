import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import talib
    from talib import abstract
except ImportError:
    talib = None
    abstract = None

logger = logging.getLogger(__name__)


TALIB_INDICATORS = {
    "trend": [
        "SMA",
        "EMA",
        "WMA",
        "DEMA",
        "TEMA",
        "TRIMA",
        "KAMA",
        "MAMA",
        "T3",
        "ADX",
        "ADXR",
        "APO",
        "AROON",
        "AROONOSC",
        "BOP",
        "CCI",
        "DX",
        "MACD",
        "MACDEXT",
        "MACDFIX",
        "MOM",
        "PPO",
        "ROC",
        "ROCP",
        "ROCR",
        "ROCR100",
        "RSI",
        "TRIX",
        "ULTOSC",
        "WILLR",
        "HT_TRENDLINE",
        "HT_TRENDMODE",
    ],
    "volatility": ["ATR", "NATR", "TRANGE", "BBANDS", "STDDEV", "VAR", "HT_DCPERIOD", "HT_DCPHASE"],
    "momentum": [
        "ADX",
        "ADXR",
        "APO",
        "AROON",
        "AROONOSC",
        "BOP",
        "CCI",
        "CMO",
        "DX",
        "MACD",
        "MFI",
        "MINUS_DI",
        "MINUS_DM",
        "MOM",
        "PLUS_DI",
        "PLUS_DM",
        "PPO",
        "ROC",
        "ROCP",
        "ROCR",
        "ROCR100",
        "RSI",
        "STOCH",
        "STOCHF",
        "STOCHRSI",
        "TRIX",
        "ULTOSC",
        "WILLR",
    ],
    "reversal": [
        "RSI",
        "STOCH",
        "STOCHF",
        "STOCHRSI",
        "WILLR",
        "CCI",
        "MFI",
        "BBANDS",
        "SAR",
        "SAREXT",
        "CDL2CROWS",
        "CDL3BLACKCROWS",
        "CDL3INSIDE",
        "CDL3LINESTRIKE",
        "CDL3OUTSIDE",
        "CDL3STARSINSOUTH",
        "CDL3WHITESOLDIERS",
        "CDLABANDONEDBABY",
        "CDLADVANCEBLOCK",
    ],
    "volume": ["AD", "ADOSC", "OBV", "MFI"],
    "price_transform": ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE", "HT_TRENDLINE"],
}

ALL_INDICATORS = list({ind for cat in TALIB_INDICATORS.values() for ind in cat})


@dataclass(slots=True)
class TALibStrategyGene:
    """Genetic representation of a TA-Lib indicator combination strategy."""

    indicators: list[str]

    params: dict[str, dict[str, Any]]

    combination_method: str = "weighted_vote"  # 'vote', 'weighted_vote', 'threshold', 'all', 'any'

    long_threshold: float = 0.6  # Threshold for long signals
    short_threshold: float = -0.6  # Threshold for short signals

    weights: dict[str, float] = None

    preferred_regime: str = "any"  # 'trend', 'range', 'volatile', 'any'

    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0

    generation: int = 0
    strategy_id: str = ""

    def __post_init__(self):
        if self.weights is None:
            self.weights = dict.fromkeys(self.indicators, 1.0)
        if not self.strategy_id:
            self.strategy_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique strategy ID."""
        import hashlib

        content = f"{sorted(self.indicators)}_{self.combination_method}_{self.generation}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TALibStrategyGene":
        return cls(**data)


class TALibStrategyMixer:
    """
    Automatically generates and tests combinations of TA-Lib indicators.

    This is the core "autonomous discovery" component that enables the bot to:
    - Try different indicators from TA-Lib
    - Mix and match indicators to find profitable combinations
    - Evolve strategies over time
    """

    def __init__(self, max_indicators_per_strategy: int = 5):
        self.max_indicators = max_indicators_per_strategy
        self.logger = logging.getLogger(self.__class__.__name__)

        if talib is None:
            self.logger.error("TA-Lib not installed! Install with: pip install TA-Lib")

        self.indicator_synergy_matrix: dict[tuple[str, str], float] = defaultdict(float)

        self.regime_performance: dict[str, dict[str, float]] = {
            "trend": defaultdict(float),
            "range": defaultdict(float),
            "volatile": defaultdict(float),
        }

    def generate_random_strategy(self, regime: str = "any") -> TALibStrategyGene:
        """
        Generate a random indicator combination strategy.

        Args:
            regime: Market regime to optimize for ('trend', 'range', 'volatile', 'any')

        Returns:
            TALibStrategyGene with random indicators and parameters
        """
        if regime == "trend":
            categories = ["trend", "momentum"]
        elif regime == "range":
            categories = ["reversal", "volatility"]
        elif regime == "volatile":
            categories = ["volatility", "momentum"]
        else:
            categories = list(TALIB_INDICATORS.keys())

        n_indicators = random.randint(2, min(5, self.max_indicators))

        selected_indicators = []
        for _ in range(n_indicators):
            category = random.choice(categories)
            indicator = random.choice(TALIB_INDICATORS[category])
            if indicator not in selected_indicators:
                selected_indicators.append(indicator)

        params = {}
        for indicator in selected_indicators:
            params[indicator] = self._random_params_for_indicator(indicator)

        combination_method = random.choice(["weighted_vote", "vote", "threshold", "all", "any"])

        long_threshold = random.uniform(0.4, 0.8)
        short_threshold = -long_threshold

        weights = {ind: random.uniform(0.5, 1.5) for ind in selected_indicators}

        return TALibStrategyGene(
            indicators=selected_indicators,
            params=params,
            combination_method=combination_method,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            weights=weights,
            preferred_regime=regime,
        )

    def _random_params_for_indicator(self, indicator: str) -> dict[str, Any]:
        """Generate random but sensible parameters for a TA-Lib indicator."""

        param_ranges = {
            "SMA": {"timeperiod": random.randint(5, 200)},
            "EMA": {"timeperiod": random.randint(5, 200)},
            "WMA": {"timeperiod": random.randint(5, 200)},
            "DEMA": {"timeperiod": random.randint(5, 100)},
            "TEMA": {"timeperiod": random.randint(5, 100)},
            "KAMA": {"timeperiod": random.randint(10, 50)},
            "RSI": {"timeperiod": random.randint(7, 28)},
            "STOCH": {
                "fastk_period": random.randint(5, 21),
                "slowk_period": random.randint(3, 14),
                "slowd_period": random.randint(3, 14),
            },
            "STOCHRSI": {
                "timeperiod": random.randint(7, 21),
                "fastk_period": random.randint(3, 14),
                "fastd_period": random.randint(3, 14),
            },
            "WILLR": {"timeperiod": random.randint(7, 28)},
            "CCI": {"timeperiod": random.randint(10, 40)},
            "MFI": {"timeperiod": random.randint(7, 28)},
            "MACD": {
                "fastperiod": random.randint(8, 16),
                "slowperiod": random.randint(20, 35),
                "signalperiod": random.randint(7, 12),
            },
            "BBANDS": {
                "timeperiod": random.randint(15, 30),
                "nbdevup": random.uniform(1.5, 3.0),
                "nbdevdn": random.uniform(1.5, 3.0),
            },
            "ATR": {"timeperiod": random.randint(10, 20)},
            "NATR": {"timeperiod": random.randint(10, 20)},
            "ADX": {"timeperiod": random.randint(10, 20)},
            "ADXR": {"timeperiod": random.randint(10, 20)},
            "MOM": {"timeperiod": random.randint(5, 20)},
            "ROC": {"timeperiod": random.randint(5, 20)},
            "AROON": {"timeperiod": random.randint(14, 28)},
            "AROONOSC": {"timeperiod": random.randint(14, 28)},
        }

        return param_ranges.get(indicator, {"timeperiod": random.randint(10, 30)})

    def apply_strategy(self, df: pd.DataFrame, strategy: TALibStrategyGene) -> pd.Series:
        """
        Apply a TA-Lib strategy to price data and generate signals.

        Args:
            df: DataFrame with OHLCV data
            strategy: TALibStrategyGene to apply

        Returns:
            Series with signals: 1 (long), -1 (short), 0 (neutral)
        """
        if talib is None:
            return pd.Series(0, index=df.index)

        indicator_signals = []

        for indicator_name in strategy.indicators:
            try:
                signal = self._calculate_indicator_signal(df, indicator_name, strategy.params.get(indicator_name, {}))
                indicator_signals.append(signal * strategy.weights.get(indicator_name, 1.0))
            except Exception as e:
                self.logger.warning(f"Failed to calculate {indicator_name}: {e}")
                continue

        if not indicator_signals:
            return pd.Series(0, index=df.index)

        combined = self._combine_signals(
            indicator_signals, strategy.combination_method, strategy.long_threshold, strategy.short_threshold
        )

        return combined

    def _calculate_indicator_signal(self, df: pd.DataFrame, indicator: str, params: dict[str, Any]) -> pd.Series:
        """
        Calculate buy/sell signals from a single TA-Lib indicator.

        Returns:
            Series with values in range [-1, 1] indicating signal strength
        """
        close = df["close"].values.astype(np.float32)
        high = df["high"].values.astype(np.float32)
        low = df["low"].values.astype(np.float32)

        volume = None
        if "volume" in df.columns:
            volume = df["volume"].values.astype(np.float32)

        try:
            if indicator in ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA"]:
                ma = getattr(talib, indicator)(close, **params)
                signal = np.where(close > ma, 1.0, np.where(close < ma, -1.0, 0.0))

            elif indicator == "MACD":
                macd, macdsignal, macdhist = talib.MACD(close, **params)
                signal = np.where(macdhist > 0, 1.0, np.where(macdhist < 0, -1.0, 0.0))

            elif indicator == "RSI":
                rsi = talib.RSI(close, **params)
                signal = np.where(rsi < 30, 1.0, np.where(rsi > 70, -1.0, 0.0))

            elif indicator in ["STOCH", "STOCHF"]:
                slowk, slowd = getattr(talib, indicator)(high, low, close, **params)
                signal = np.where(slowk < 20, 1.0, np.where(slowk > 80, -1.0, 0.0))

            elif indicator == "STOCHRSI":
                fastk, fastd = talib.STOCHRSI(close, **params)
                signal = np.where(fastk < 0.2, 1.0, np.where(fastk > 0.8, -1.0, 0.0))

            elif indicator == "BBANDS":
                upper, middle, lower = talib.BBANDS(close, **params)
                signal = np.where(close < lower, 1.0, np.where(close > upper, -1.0, 0.0))

            elif indicator == "ADX":
                adx = talib.ADX(high, low, close, **params)
                plus_di = talib.PLUS_DI(high, low, close, **params)
                minus_di = talib.MINUS_DI(high, low, close, **params)
                strength = np.where(adx > 25, 1.0, 0.5)
                direction = np.where(plus_di > minus_di, 1.0, -1.0)
                signal = strength * direction

            elif indicator == "CCI":
                cci = talib.CCI(high, low, close, **params)
                signal = np.where(cci < -100, 1.0, np.where(cci > 100, -1.0, 0.0))

            elif indicator == "WILLR":
                willr = talib.WILLR(high, low, close, **params)
                signal = np.where(willr < -80, 1.0, np.where(willr > -20, -1.0, 0.0))

            elif indicator == "MFI":
                if volume is not None:
                    mfi = talib.MFI(high, low, close, volume, **params)
                    signal = np.where(mfi < 20, 1.0, np.where(mfi > 80, -1.0, 0.0))
                else:
                    signal = np.zeros(len(close))

            elif indicator in ["AROON", "AROONOSC"]:
                if indicator == "AROON":
                    aroon_up, aroon_down = talib.AROON(high, low, **params)
                    signal = np.where(aroon_up > aroon_down, 1.0, -1.0)
                else:
                    aroonosc = talib.AROONOSC(high, low, **params)
                    signal = np.where(aroonosc > 0, 1.0, -1.0)

            elif indicator in ["MOM", "ROC", "ROCP", "ROCR"]:
                momentum = getattr(talib, indicator)(close, **params)
                signal = np.where(momentum > 0, 1.0, np.where(momentum < 0, -1.0, 0.0))

            else:
                result = getattr(talib, indicator)(close, **params)
                if result is not None:
                    signal = np.where(result > close, 1.0, np.where(result < close, -1.0, 0.0))
                else:
                    signal = np.zeros(len(close))

            return pd.Series(signal, index=df.index).fillna(0.0)

        except Exception as e:
            self.logger.error(f"Error calculating {indicator}: {e}", exc_info=True)
            raise

    def _combine_signals(
        self, signals: list[pd.Series], method: str, long_threshold: float, short_threshold: float
    ) -> pd.Series:
        """Combine multiple indicator signals into final trading signal."""

        if not signals:
            return pd.Series(0)

        stacked = pd.concat(signals, axis=1)

        if method == "vote":
            combined = stacked.sum(axis=1) / len(signals)

        elif method == "weighted_vote":
            combined = stacked.sum(axis=1) / len(signals)

        elif method == "all":
            all_long = (stacked == 1.0).all(axis=1)
            all_short = (stacked == -1.0).all(axis=1)
            combined = pd.Series(np.where(all_long, 1.0, np.where(all_short, -1.0, 0.0)), index=stacked.index)

        elif method == "any":
            any_long = (stacked == 1.0).any(axis=1)
            any_short = (stacked == -1.0).any(axis=1)
            combined = pd.Series(np.where(any_long, 1.0, np.where(any_short, -1.0, 0.0)), index=stacked.index)

        else:  # threshold
            combined = stacked.mean(axis=1)

        final_signal = pd.Series(
            np.where(combined >= long_threshold, 1, np.where(combined <= short_threshold, -1, 0)), index=combined.index
        )

        return final_signal

    def update_synergy_matrix(self, strategy: TALibStrategyGene, performance: float) -> None:
        """Update indicator synergy matrix based on strategy performance."""

        for i, ind1 in enumerate(strategy.indicators):
            for ind2 in strategy.indicators[i + 1 :]:
                pair = tuple(sorted([ind1, ind2]))
                current = self.indicator_synergy_matrix.get(pair, 0.0)
                self.indicator_synergy_matrix[pair] = 0.9 * current + 0.1 * performance

        regime = strategy.preferred_regime
        if regime != "any":
            for ind in strategy.indicators:
                current = self.regime_performance[regime].get(ind, 0.0)
                self.regime_performance[regime][ind] = 0.9 * current + 0.1 * performance

    def get_best_indicators_for_regime(self, regime: str, top_k: int = 10) -> list[str]:
        """Get the best performing indicators for a specific market regime."""

        if regime not in self.regime_performance:
            return random.sample(ALL_INDICATORS, min(top_k, len(ALL_INDICATORS)))

        scores = self.regime_performance[regime]
        if not scores:
            return random.sample(ALL_INDICATORS, min(top_k, len(ALL_INDICATORS)))

        sorted_indicators = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [ind for ind, _ in sorted_indicators[:top_k]]

    def save_knowledge(self, path: Path) -> None:
        """Save learned indicator synergies and regime preferences."""

        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "synergy_matrix": {f"{k[0]}_{k[1]}": v for k, v in self.indicator_synergy_matrix.items()},
            "regime_performance": {regime: dict(perf) for regime, perf in self.regime_performance.items()},
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved TA-Lib knowledge to {path}")

    def load_knowledge(self, path: Path) -> None:
        """Load learned indicator synergies and regime preferences."""

        if not path.exists():
            self.logger.warning(f"Knowledge file not found: {path}")
            return

        with open(path) as f:
            data = json.load(f)

        for key, value in data.get("synergy_matrix", {}).items():
            ind1, ind2 = key.split("_")
            self.indicator_synergy_matrix[tuple(sorted([ind1, ind2]))] = value

        for regime, perf in data.get("regime_performance", {}).items():
            self.regime_performance[regime] = defaultdict(float, perf)

        self.logger.info(f"Loaded TA-Lib knowledge from {path}")


def get_available_indicators() -> list[str]:
    """Get list of all available TA-Lib indicators."""
    return ALL_INDICATORS.copy()


def get_indicators_by_category(category: str) -> list[str]:
    """Get indicators for a specific category."""
    return TALIB_INDICATORS.get(category, []).copy()

import logging
import random
from dataclasses import asdict

from ..features.talib_mixer import TALibStrategyGene, TALibStrategyMixer

logger = logging.getLogger(__name__)


class GenesisLibrary:
    """
    Provides scientifically validated 'Archetype' genes to warm-start evolution.
    Based on 2020-2025 Quant Research:
    1. Trend Following (KAMA, FRAMA) - Adapts to volatility.
    2. Mean Reversion (BB Width + RSI) - Exploits range.
    3. Institutional Flow (VWAP + Volume Profile) - Follows big money.
    """

    @staticmethod
    def get_smart_trend_archetype() -> TALibStrategyGene:
        """
        Archetype 1: Adaptive Trend Following.
        Uses KAMA (Kaufman Adaptive MA) instead of EMA to filter chop.
        """
        indicators = ["KAMA", "MAMA", "ADX"]
        params = {
            "KAMA": {"timeperiod": 10},  # Fast adaptive
            "MAMA": {"fastlimit": 0.5, "slowlimit": 0.05},
            "ADX": {"timeperiod": 14},
        }
        return TALibStrategyGene(
            indicators=indicators,
            params=params,
            combination_method="weighted_vote",
            long_threshold=0.6,
            short_threshold=-0.6,
            weights={"KAMA": 0.5, "MAMA": 0.3, "ADX": 0.2},
            preferred_regime="trend",
            strategy_id="genesis_smart_trend",
            use_ob=False,  # Pure trend initially
            use_fvg=True,  # FVG often aligns with strong trends
            mtf_confirmation=True,  # Crucial for trend
            tp_pips=60.0,
            sl_pips=30.0,
        )

    @staticmethod
    def get_mean_reversion_archetype() -> TALibStrategyGene:
        """
        Archetype 2: Volatility-Based Mean Reversion.
        Uses Bollinger Band Width to detect squeeze/expansion and RSI for entry.
        """
        indicators = ["BBANDS", "RSI", "MFI"]
        params = {
            "BBANDS": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
            "RSI": {"timeperiod": 7},  # Faster RSI for mean rev
            "MFI": {"timeperiod": 14},
        }
        return TALibStrategyGene(
            indicators=indicators,
            params=params,
            combination_method="threshold",
            long_threshold=0.7,  # Higher threshold for contrarian trades
            short_threshold=-0.7,
            weights={"BBANDS": 0.4, "RSI": 0.4, "MFI": 0.2},
            preferred_regime="range",
            strategy_id="genesis_mean_rev",
            use_liq_sweep=True,  # Sweep often marks the reversal top/bottom
            use_premium_discount=True,  # Buy in discount, sell in premium
            tp_pips=20.0,  # Quick scalp
            sl_pips=15.0,
        )

    @staticmethod
    def get_institutional_flow_archetype() -> TALibStrategyGene:
        """
        Archetype 3: Order Flow & Volume.
        Uses VWAP (simulated via MFI/OBV context) and SMC structures.
        Note: True VWAP is calculated in pipeline, here we use proxies available in TA-Lib mixer logic.
        """
        indicators = ["MFI", "OBV", "ADOSC"]
        params = {"MFI": {"timeperiod": 14}, "ADOSC": {"fastperiod": 3, "slowperiod": 10}}
        return TALibStrategyGene(
            indicators=indicators,
            params=params,
            combination_method="weighted_vote",
            long_threshold=0.55,
            short_threshold=-0.55,
            weights={"MFI": 0.4, "OBV": 0.3, "ADOSC": 0.3},
            preferred_regime="any",
            strategy_id="genesis_inst_flow",
            use_ob=True,  # Order blocks are key
            use_inducement=True,  # Trap trading
            mtf_confirmation=True,
            tp_pips=40.0,
            sl_pips=20.0,
        )

    @staticmethod
    def populate_initial_population(population_size: int, mixer: TALibStrategyMixer) -> list[TALibStrategyGene]:
        """
        Generates a hybrid population:
        - 30% Smart Trend
        - 30% Mean Reversion
        - 20% Institutional Flow
        - 20% Pure Random (Novelty Search)
        """
        pop = []

        counts = {
            "trend": int(population_size * 0.3),
            "mean_rev": int(population_size * 0.3),
            "inst_flow": int(population_size * 0.2),
        }

        base = GenesisLibrary.get_smart_trend_archetype()
        for _ in range(counts["trend"]):
            clone = TALibStrategyGene(**asdict(base))
            clone.strategy_id = f"gen_trend_{random.randint(1000, 9999)}"
            clone.tp_pips *= random.uniform(0.9, 1.1)
            pop.append(clone)

        base = GenesisLibrary.get_mean_reversion_archetype()
        for _ in range(counts["mean_rev"]):
            clone = TALibStrategyGene(**asdict(base))
            clone.strategy_id = f"gen_mean_{random.randint(1000, 9999)}"
            clone.sl_pips *= random.uniform(0.9, 1.1)
            pop.append(clone)

        base = GenesisLibrary.get_institutional_flow_archetype()
        for _ in range(counts["inst_flow"]):
            clone = TALibStrategyGene(**asdict(base))
            clone.strategy_id = f"gen_flow_{random.randint(1000, 9999)}"
            pop.append(clone)

        remaining = population_size - len(pop)
        for _ in range(remaining):
            pop.append(mixer.generate_random_strategy())

        random.shuffle(pop)
        return pop

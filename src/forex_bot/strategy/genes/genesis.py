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
        """Archetype 1: Stochastic Adaptive Trend."""
        # HPC FIX: Randomized seed to prevent Mode Collapse
        kama_p = random.choice([5, 10, 14, 21])
        indicators = ["KAMA", "MAMA", "ADX"]
        params = {
            "KAMA": {"timeperiod": kama_p},
            "MAMA": {"fastlimit": random.uniform(0.4, 0.6), "slowlimit": 0.05},
            "ADX": {"timeperiod": random.choice([10, 14, 21])},
        }
        return TALibStrategyGene(
            indicators=indicators,
            params=params,
            combination_method="weighted_vote",
            long_threshold=random.uniform(0.5, 0.7),
            short_threshold=-random.uniform(0.5, 0.7),
            weights={"KAMA": 0.5, "MAMA": 0.3, "ADX": 0.2},
            preferred_regime="trend",
            strategy_id=f"genesis_smart_trend_{random.randint(0, 999)}",
            use_ob=False,
            use_fvg=True,
            mtf_confirmation=True,
            tp_pips=random.uniform(40.0, 80.0),
            sl_pips=random.uniform(20.0, 40.0),
        )

    @staticmethod
    def get_mean_reversion_archetype() -> TALibStrategyGene:
        """Archetype 2: Stochastic Mean Reversion."""
        bb_p = random.choice([14, 20, 34])
        indicators = ["BBANDS", "RSI", "MFI"]
        params = {
            "BBANDS": {"timeperiod": bb_p, "nbdevup": random.uniform(1.8, 2.5), "nbdevdn": random.uniform(1.8, 2.5)},
            "RSI": {"timeperiod": random.choice([7, 10, 14])},
            "MFI": {"timeperiod": random.choice([14, 21])},
        }
        return TALibStrategyGene(
            indicators=indicators,
            params=params,
            combination_method="threshold",
            long_threshold=random.uniform(0.6, 0.8),
            short_threshold=-random.uniform(0.6, 0.8),
            weights={"BBANDS": 0.4, "RSI": 0.4, "MFI": 0.2},
            preferred_regime="range",
            strategy_id=f"genesis_mean_rev_{random.randint(0, 999)}",
            use_liq_sweep=True,
            use_premium_discount=True,
            tp_pips=random.uniform(15.0, 30.0),
            sl_pips=random.uniform(10.0, 20.0),
        )

    @staticmethod
    def get_institutional_flow_archetype() -> TALibStrategyGene:
        """Archetype 3: Stochastic Institutional Flow."""
        mfi_p = random.choice([7, 14, 21])
        indicators = ["MFI", "OBV", "ADOSC"]
        params = {"MFI": {"timeperiod": mfi_p}, "ADOSC": {"fastperiod": 3, "slowperiod": 10}}
        return TALibStrategyGene(
            indicators=indicators,
            params=params,
            combination_method="weighted_vote",
            long_threshold=random.uniform(0.5, 0.65),
            short_threshold=-random.uniform(0.5, 0.65),
            weights={"MFI": 0.4, "OBV": 0.3, "ADOSC": 0.3},
            preferred_regime="any",
            strategy_id=f"genesis_inst_flow_{random.randint(0, 999)}",
            use_ob=True,
            use_inducement=True,
            mtf_confirmation=True,
            tp_pips=random.uniform(30.0, 60.0),
            sl_pips=random.uniform(15.0, 30.0),
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
        def _sanitize_volume(gene: TALibStrategyGene) -> TALibStrategyGene:
            if getattr(mixer, "use_volume_features", False):
                return gene
            vol_inds = set(getattr(mixer, "_volume_indicators", set()))
            if not vol_inds:
                return gene
            gene.indicators = [ind for ind in gene.indicators if ind not in vol_inds]
            for ind in list(getattr(gene, "params", {}).keys()):
                if ind in vol_inds:
                    gene.params.pop(ind, None)
            for ind in list(getattr(gene, "weights", {}).keys()):
                if ind in vol_inds:
                    gene.weights.pop(ind, None)
            # Ensure at least one indicator remains
            if len(gene.indicators) < 1:
                pool = [ind for ind in mixer.available_indicators if ind not in vol_inds]
                if pool:
                    gene.indicators.append(random.choice(pool))
            return gene
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
            pop.append(_sanitize_volume(clone))

        base = GenesisLibrary.get_mean_reversion_archetype()
        for _ in range(counts["mean_rev"]):
            clone = TALibStrategyGene(**asdict(base))
            clone.strategy_id = f"gen_mean_{random.randint(1000, 9999)}"
            clone.sl_pips *= random.uniform(0.9, 1.1)
            pop.append(_sanitize_volume(clone))

        base = GenesisLibrary.get_institutional_flow_archetype()
        for _ in range(counts["inst_flow"]):
            clone = TALibStrategyGene(**asdict(base))
            clone.strategy_id = f"gen_flow_{random.randint(1000, 9999)}"
            pop.append(_sanitize_volume(clone))

        remaining = population_size - len(pop)
        for _ in range(remaining):
            pop.append(mixer.generate_random_strategy())

        random.shuffle(pop)
        return pop

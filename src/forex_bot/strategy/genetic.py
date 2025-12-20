from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..features.talib_mixer import ALL_INDICATORS, TALibStrategyMixer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GeneticGene:
    indicators: list[str]
    params: dict[str, dict[str, Any]]
    combination_method: str = "weighted_vote"
    long_threshold: float = 0.6
    short_threshold: float = -0.6
    weights: dict[str, float] = field(default_factory=dict)
    preferred_regime: str = "any"
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    trades_count: int = 0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    strategy_id: str = ""
    use_ob: bool = False
    use_fvg: bool = False
    use_liq_sweep: bool = False
    mtf_confirmation: bool = False
    use_premium_discount: bool = False
    use_inducement: bool = False
    tp_pips: float = 40.0
    sl_pips: float = 20.0
    doctor_params: dict[str, float] = field(default_factory=dict)
    risk_params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategy_id:
            self.strategy_id = f"gene_{random.randint(0, 1_000_000)}_{self.generation}"

        if not self.doctor_params:
            self.doctor_params = {
                "stall_threshold_minutes": float(random.randint(30, 180)),
                "min_profit_to_bank": random.uniform(0.3, 1.5),
                "reversal_sensitivity": random.uniform(0.5, 0.95),
            }

        if not self.risk_params:
            self.risk_params = {
                "risk_curve_steepness": random.uniform(100.0, 300.0),
                "confidence_threshold": random.uniform(0.55, 0.75),
            }

        if self.weights is None:
            self.weights = {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneticGene:
        return cls(**data)


class GeneticStrategyEvolution:
    """
    Simple genetic search engine that breeds TALib-based strategy genes.
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.15,
        mixer: TALibStrategyMixer | None = None,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: list[GeneticGene] = []
        self.generation = 0
        self.best_gene: GeneticGene | None = None
        self.mixer = mixer

    def initialize_population(self, regime: str = "any", validation_data: pd.DataFrame | None = None) -> None:
        if not self.mixer:
            raise ValueError("TALibStrategyMixer must be provided to GeneticStrategyEvolution")

        from .gauntlet import StrategyGauntlet

        gauntlet = StrategyGauntlet(self.mixer, validation_data) if validation_data is not None else None

        try:
            from .genes.genesis import GenesisLibrary

            candidates = GenesisLibrary.populate_initial_population(self.population_size * 5, self.mixer)
        except Exception:
            candidates = []

        self.population = []
        attempts = 0
        max_attempts = self.population_size * 20
        candidate_queue = deque(candidates)

        while len(self.population) < self.population_size and attempts < max_attempts:
            attempts += 1
            if candidate_queue:
                gene = candidate_queue.popleft()
            else:
                talib_gene = self.mixer.generate_random_strategy(regime=regime)
                gene = GeneticGene(**asdict(talib_gene), generation=self.generation)

            if gauntlet:
                if gauntlet.run(gene):
                    gene.generation = self.generation
                    self.population.append(gene)
            else:
                gene.generation = self.generation
                self.population.append(gene)

        if len(self.population) < self.population_size:
            logger.warning(
                "Gauntlet too strict! Only accepted %s/%s. Filling with randoms.",
                len(self.population),
                self.population_size,
            )
            while len(self.population) < self.population_size:
                talib_gene = self.mixer.generate_random_strategy(regime=regime)
                self.population.append(GeneticGene(**asdict(talib_gene), generation=self.generation))

        logger.info("Initialized population with Hybrid Genesis + Gauntlet (Size: %s)", len(self.population))

    def evolve(self, validation_data: pd.DataFrame | None = None) -> list[GeneticGene]:
        if not self.population:
            raise RuntimeError("Population not initialized. Call initialize_population() first.")

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_gene = self.population[0]

        elite_count = max(1, int(self.population_size * 0.2))
        next_gen: list[GeneticGene] = self.population[:elite_count]

        while len(next_gen) < self.population_size:
            p1 = random.choice(self.population[:elite_count])
            p2 = random.choice(self.population[:elite_count])
            child_gene = self._crossover(p1, p2)
            if random.random() < self.mutation_rate:
                child_gene = self._mutate(child_gene)
            child_gene.generation = self.generation + 1
            next_gen.append(child_gene)

        # Evaluation Step (Fix for Zombie Evolution)
        if validation_data is not None and self.mixer is not None:
            from .gauntlet import StrategyGauntlet

            gauntlet = StrategyGauntlet(self.mixer, validation_data)

            # Only evaluate new genes (fitness == 0.0) to save time
            # Elite genes preserve their fitness
            for gene in next_gen:
                if gene.fitness == 0.0:
                    gauntlet.run(gene)
                    gene.generation = self.generation + 1

        self.population = next_gen
        # Re-sort to put best new children at top
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.population:
            self.best_gene = self.population[0]

        self.generation += 1
        logger.info(
            "Genetic evolution generation %s completed; best fitness %.4f",
            self.generation,
            self.best_gene.fitness,
        )
        return self.population

    def _crossover(self, p1: GeneticGene, p2: GeneticGene) -> GeneticGene:
        child_dict: dict[str, Any] = {}
        for k, v in asdict(p1).items():
            if k in [
                "fitness",
                "sharpe_ratio",
                "win_rate",
                "max_drawdown",
                "profit_factor",
                "expectancy",
                "trades_count",
                "generation",
                "parent_ids",
                "strategy_id",
                "doctor_params",
                "risk_params",
            ]:
                continue
            child_dict[k] = v if random.random() < 0.5 else asdict(p2)[k]

        child_dict["indicators"] = list(set(p1.indicators + p2.indicators))
        if len(child_dict["indicators"]) > 20:
            child_dict["indicators"] = random.sample(child_dict["indicators"], k=20)

        child_dict["parent_ids"] = [p1.strategy_id, p2.strategy_id]
        child_dict["strategy_id"] = f"gene_{random.randint(0, 1_000_000)}_{self.generation}"

        return GeneticGene(**child_dict)

    def _mutate(self, gene: GeneticGene) -> GeneticGene:
        mutated_gene = GeneticGene(**asdict(gene))
        mutation_type = random.choice(
            [
                "indicators",
                "params",
                "thresholds",
                "weights",
                "smc_flags",
                "mtf_flag",
                "tp_sl_pips",
                "doctor_params",
                "risk_params",
            ]
        )

        if mutation_type == "doctor_params":
            key = random.choice(list(mutated_gene.doctor_params.keys()))
            if key == "stall_threshold_minutes":
                mutated_gene.doctor_params[key] = max(15.0, mutated_gene.doctor_params[key] + random.randint(-30, 30))
            elif key == "min_profit_to_bank":
                mutated_gene.doctor_params[key] = np.clip(
                    mutated_gene.doctor_params[key] * random.uniform(0.8, 1.2), 0.1, 3.0
                )
            elif key == "reversal_sensitivity":
                mutated_gene.doctor_params[key] = np.clip(
                    mutated_gene.doctor_params[key] + random.uniform(-0.1, 0.1), 0.1, 1.0
                )

        elif mutation_type == "risk_params":
            key = random.choice(list(mutated_gene.risk_params.keys()))
            if key == "risk_curve_steepness":
                mutated_gene.risk_params[key] = np.clip(
                    mutated_gene.risk_params[key] + random.randint(-50, 50), 50.0, 500.0
                )
            elif key == "confidence_threshold":
                mutated_gene.risk_params[key] = np.clip(
                    mutated_gene.risk_params[key] + random.uniform(-0.05, 0.05), 0.5, 0.9
                )

        elif mutation_type == "indicators":
            if random.random() < 0.5 and gene.indicators:
                mutated_gene.indicators.pop(random.randrange(len(gene.indicators)))
            else:
                available_inds = [ind for ind in ALL_INDICATORS if ind not in gene.indicators]
                if available_inds:
                    mutated_gene.indicators.append(random.choice(available_inds))
            if not mutated_gene.indicators:
                mutated_gene.indicators.append(random.choice(ALL_INDICATORS))

        elif mutation_type == "params" and gene.indicators:
            target_ind = random.choice(gene.indicators)
            if target_ind in mutated_gene.params:
                param_key = random.choice(list(mutated_gene.params[target_ind].keys()))
                old_val = mutated_gene.params[target_ind][param_key]
                if isinstance(old_val, int):
                    mutated_gene.params[target_ind][param_key] = max(1, old_val + random.randint(-5, 5))
                elif isinstance(old_val, float):
                    mutated_gene.params[target_ind][param_key] = old_val * random.uniform(0.8, 1.2)

        elif mutation_type == "thresholds":
            mutated_gene.long_threshold = np.clip(gene.long_threshold * random.uniform(0.9, 1.1), 0.4, 0.9)
            mutated_gene.short_threshold = np.clip(gene.short_threshold * random.uniform(0.9, 1.1), -0.9, -0.4)
            mutated_gene.combination_method = random.choice(["weighted_vote", "threshold"])

        elif mutation_type == "weights" and gene.weights:
            if gene.indicators:
                target_ind = random.choice(gene.indicators)
                mutated_gene.weights[target_ind] = np.clip(
                    gene.weights.get(target_ind, 1.0) * random.uniform(0.5, 1.5), 0.1, 2.0
                )

        elif mutation_type == "smc_flags":
            smc_flags = ["use_ob", "use_fvg", "use_liq_sweep", "use_premium_discount", "use_inducement"]
            flag_to_toggle = random.choice(smc_flags)
            current_val = getattr(mutated_gene, flag_to_toggle)
            setattr(mutated_gene, flag_to_toggle, not current_val)

        elif mutation_type == "mtf_flag":
            mutated_gene.mtf_confirmation = not gene.mtf_confirmation

        elif mutation_type == "tp_sl_pips":
            mutated_gene.tp_pips = np.clip(gene.tp_pips * random.uniform(0.8, 1.2), 10.0, 100.0)
            mutated_gene.sl_pips = np.clip(gene.sl_pips * random.uniform(0.8, 1.2), 5.0, 50.0)

        return mutated_gene

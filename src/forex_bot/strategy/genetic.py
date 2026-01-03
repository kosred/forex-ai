from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import Settings
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
    evaluated: bool = False
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    trades_count: int = 0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    strategy_id: str = ""
    use_ob: bool = True
    use_fvg: bool = True
    use_liq_sweep: bool = True
    mtf_confirmation: bool = True
    use_premium_discount: bool = True
    use_inducement: bool = True
    tp_pips: float = 40.0
    sl_pips: float = 20.0
    doctor_params: dict[str, float] = field(default_factory=dict)
    risk_params: dict[str, float] = field(default_factory=dict)
    slice_pass_rate: float = 0.0  # fraction of slices where gene met gates

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
        population_size: int = 1000,
        mutation_rate: float = 0.20,
        mixer: TALibStrategyMixer | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: list[GeneticGene] = []
        self.generation = 0
        self.best_gene: GeneticGene | None = None
        self.mixer = mixer
        self.settings = settings or Settings()

    def _indicator_pool(self) -> list[str]:
        if self.mixer and self.mixer.available_indicators:
            return list(self.mixer.available_indicators)
        return ALL_INDICATORS

    def _random_params_for(self, indicator: str) -> dict[str, Any]:
        if not self.mixer:
            return {}
        try:
            from talib import abstract

            info = abstract.Function(indicator).info
            default_params = info.get("parameters", {})
            return self.mixer._randomize_indicator_params(indicator, default_params)
        except Exception:
            return {}

    def _sync_indicator_maps(self, gene: GeneticGene) -> None:
        if gene.params is None:
            gene.params = {}
        if gene.weights is None:
            gene.weights = {}
        for ind in gene.indicators:
            if ind not in gene.params:
                gene.params[ind] = self._random_params_for(ind)
            if ind not in gene.weights:
                gene.weights[ind] = random.random()
        # Remove stale entries
        for ind in list(gene.params.keys()):
            if ind not in gene.indicators:
                gene.params.pop(ind, None)
        for ind in list(gene.weights.keys()):
            if ind not in gene.indicators:
                gene.weights.pop(ind, None)

    def initialize_population(self, regime: str = "any", validation_data: pd.DataFrame | None = None) -> None:
        if not self.mixer:
            raise ValueError("TALibStrategyMixer must be provided to GeneticStrategyEvolution")

        from .gauntlet import StrategyGauntlet

        # If validation_data is a mapping of symbol/timeframe -> df, pick the first for gauntlet
        vdf_for_gauntlet = None
        if isinstance(validation_data, dict):
            for _k, v in validation_data.items():
                if isinstance(v, dict):
                    if v:
                        vdf_for_gauntlet = next(iter(v.values()))
                        break
                else:
                    vdf_for_gauntlet = v
                    break
        else:
            vdf_for_gauntlet = validation_data

        gauntlet = StrategyGauntlet(self.mixer, vdf_for_gauntlet) if vdf_for_gauntlet is not None else None

        try:
            from .genes.genesis import GenesisLibrary

            candidates = GenesisLibrary.populate_initial_population(self.population_size * 5, self.mixer)
        except Exception:
            candidates = []

        self.population = []
        attempts = 0
        max_attempts = self.population_size * 40
        candidate_queue = deque(candidates)

        # Sort validation_data by time if provided
        if validation_data is not None and isinstance(validation_data.index, pd.DatetimeIndex):
            if not validation_data.index.is_monotonic_increasing:
                validation_data = validation_data.sort_index(kind="mergesort")

        while len(self.population) < self.population_size and attempts < max_attempts:
            attempts += 1
            if candidate_queue:
                gene = candidate_queue.popleft()
            else:
                talib_gene = self.mixer.generate_random_strategy(regime=regime)
                gene = GeneticGene(**asdict(talib_gene), generation=self.generation)
            if not gene.indicators:
                pool = self._indicator_pool()
                if pool:
                    gene.indicators.append(random.choice(pool))
            self._sync_indicator_maps(gene)

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

    @staticmethod
    def _month_indices(df: pd.DataFrame) -> np.ndarray:
        month_indices = np.zeros(len(df), dtype=np.int64)
        try:
            idx = getattr(df, "index", None)
            if isinstance(idx, pd.DatetimeIndex) and len(idx) == len(df):
                ts = idx.tz_convert("UTC") if idx.tz is not None else idx
                raw = ts.year.to_numpy() * 12 + ts.month.to_numpy()
                month_indices = np.nan_to_num(raw, nan=0.0).astype(np.int64, copy=False)
        except Exception:
            month_indices = np.zeros(len(df), dtype=np.int64)
        return month_indices

    @staticmethod
    def _day_indices(df: pd.DataFrame) -> np.ndarray:
        day_indices = np.zeros(len(df), dtype=np.int64)
        try:
            idx = getattr(df, "index", None)
            if isinstance(idx, pd.DatetimeIndex) and len(idx) == len(df):
                ts = idx.tz_convert("UTC") if idx.tz is not None else idx
                raw = ts.year.to_numpy() * 10000 + ts.month.to_numpy() * 100 + ts.day.to_numpy()
                day_indices = np.nan_to_num(raw, nan=0.0).astype(np.int64, copy=False)
        except Exception:
            day_indices = np.zeros(len(df), dtype=np.int64)
        return day_indices

    def _evaluate_population(self, validation_data: Any, genes: list[GeneticGene]) -> None:
        """
        Evaluate genes across one or many slices.
        validation_data can be:
          - single DataFrame
          - dict[str, DataFrame]
          - dict[str, dict[str, DataFrame]] (e.g., symbol -> timeframe -> df)
        A gene is accepted only if it passes gates on enough slices; one-off failures are penalized,
        but do not necessarily kill the gene unless failures dominate.
        """
        if not genes or self.mixer is None or validation_data is None:
            return

        # Flatten validation_data into a list of (name, df)
        slices: list[tuple[str, pd.DataFrame]] = []
        if isinstance(validation_data, pd.DataFrame):
            slices.append(("default", validation_data))
        elif isinstance(validation_data, dict):
            for sym, val in validation_data.items():
                if isinstance(val, dict):
                    for tf, df in val.items():
                        slices.append((f"{sym}_{tf}", df))
                else:
                    slices.append((str(sym), val))

        if not slices:
            return

        from .fast_backtest import fast_evaluate_strategy, infer_pip_metrics

        for name, df in slices:
            if not {"close", "high", "low"}.issubset(set(df.columns)):
                logger.warning("Cannot evaluate genes: validation slice %s missing OHLC columns.", name)
                for g in genes:
                    if not getattr(g, "evaluated", False):
                        g.fitness = -1e9
                        g.evaluated = True
                return
            if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
                df = df.sort_index(kind="mergesort")

        for gene in genes:
            if getattr(gene, "evaluated", False):
                continue
            pass_fits: list[float] = []
            slice_scores: list[float] = []
            fail_count = 0
            slice_metrics: list[tuple[float, float, float, float, float, float, int]] = []
            try:
                dd_cap = float(getattr(self.settings.risk, "total_drawdown_limit", 0.07) or 0.07)
                daily_dd_cap = float(getattr(self.settings.risk, "daily_drawdown_limit", 0.04) or 0.04)
                pfloor = 1.0

                for sname, df in slices:
                    close = df["close"].to_numpy(dtype=np.float64)
                    high = df["high"].to_numpy(dtype=np.float64)
                    low = df["low"].to_numpy(dtype=np.float64)
                    month_idx = self._month_indices(df)
                    day_idx = self._day_indices(df)
                    pip_size, pip_value_per_lot = infer_pip_metrics("")
                    max_hold = int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0)
                    trailing_enabled = bool(getattr(self.settings.risk, "trailing_enabled", False))
                    trailing_mult = float(getattr(self.settings.risk, "trailing_atr_multiplier", 1.0) or 1.0)
                    trailing_trigger_r = float(getattr(self.settings.risk, "trailing_be_trigger_r", 1.0) or 1.0)

                    sig = self.mixer.compute_signals(df, gene).to_numpy(dtype=np.int8)
                    arr = fast_evaluate_strategy(
                        close_prices=close,
                        high_prices=high,
                        low_prices=low,
                        signals=sig,
                        month_indices=month_idx,
                        day_indices=day_idx,
                        sl_pips=float(getattr(gene, "sl_pips", 20.0)),
                        tp_pips=float(getattr(gene, "tp_pips", 40.0)),
                        max_hold_bars=max_hold,
                        trailing_enabled=trailing_enabled,
                        trailing_atr_multiplier=trailing_mult,
                        trailing_be_trigger_r=trailing_trigger_r,
                        pip_value=float(pip_size),
                        spread_pips=1.5,
                        commission_per_trade=0.0,
                        pip_value_per_lot=float(pip_value_per_lot),
                    )

                    net_profit = float(arr[0])
                    sharpe = float(arr[1])
                    max_dd = float(arr[3])
                    win_rate = float(arr[4])
                    profit_factor = float(arr[5])
                    expectancy = float(arr[6])
                    trades = int(arr[8]) if len(arr) > 8 else 0
                    daily_dd = float(arr[10]) if len(arr) > 10 else 0.0
                    slice_metrics.append((net_profit, sharpe, max_dd, win_rate, profit_factor, expectancy, trades))

                    # Slice-level gates (no minimum trades)
                    dd_penalty = 10.0 * max(0.0, max_dd - dd_cap)
                    base_score = sharpe + (net_profit / 10_000.0) - dd_penalty
                    slice_scores.append(base_score)

                    if max_dd >= dd_cap or daily_dd >= daily_dd_cap or profit_factor <= pfloor:
                        fail_count += 1
                        continue
                    pass_fits.append(base_score)

                total_slices = len(slices)
                fail_ratio = (fail_count / total_slices) if total_slices > 0 else 1.0
                score_pool = pass_fits if pass_fits else slice_scores
                base_fit = float(np.mean(score_pool)) if score_pool else 0.0
                penalty_factor = max(0.1, 1.0 - (0.7 * fail_ratio))
                fitness = float(base_fit * penalty_factor - fail_ratio)

                # Aggregate metrics over passed slices (or all if none passed)
                if pass_fits:
                    passed = [m for m in slice_metrics if m[2] < dd_cap and m[4] > 1.0]
                else:
                    passed = slice_metrics
                if passed:
                    avg_net = float(np.mean([m[0] for m in passed]))
                    avg_sharpe = float(np.mean([m[1] for m in passed]))
                    avg_dd = float(np.mean([m[2] for m in passed]))
                    avg_pf = float(np.mean([m[4] for m in passed]))
                    avg_trades = int(np.mean([m[6] for m in passed]))
                else:
                    avg_net = avg_sharpe = avg_dd = avg_pf = 0.0
                    avg_trades = 0

                gene.fitness = fitness
                gene.sharpe_ratio = avg_sharpe
                gene.win_rate = 0.0
                gene.max_drawdown = avg_dd
                gene.profit_factor = avg_pf
                gene.expectancy = 0.0
                gene.trades_count = avg_trades
                gene.slice_pass_rate = float((total_slices - fail_count) / total_slices) if total_slices > 0 else 0.0
                gene.evaluated = True
            except Exception as exc:
                logger.debug(f"Gene evaluation failed for {gene.strategy_id}: {exc}", exc_info=True)
                gene.fitness = -1e9
                gene.trades_count = 0
                gene.evaluated = True

    def evolve(self, validation_data: pd.DataFrame | None = None) -> list[GeneticGene]:
        if not self.population:
            raise RuntimeError("Population not initialized.")

        if validation_data is not None and self.mixer is not None:
            self._evaluate_population(validation_data, self.population)

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_gene = self.population[0]

        # HPC FIX: Tournament Selection (Preserves Diversity)
        def _tournament_select(k=3):
            # Select k random individuals and return the best one
            participants = random.sample(self.population, k)
            return max(participants, key=lambda x: x.fitness)

        elite_count = max(1, int(self.population_size * 0.1)) # Top 10% Elitism
        next_gen: list[GeneticGene] = self.population[:elite_count]

        # HPC: Dynamic Mutation Rate
        # If population is too similar, boost mutation
        pop_diversity = len(set([g.indicators[0] for g in self.population if g.indicators]))
        current_mutation = self.mutation_rate
        if pop_diversity < 5: current_mutation *= 2.0

        while len(next_gen) < self.population_size:
            p1 = _tournament_select()
            p2 = _tournament_select()
            child_gene = self._crossover(p1, p2)
            
            if random.random() < current_mutation:
                child_gene = self._mutate(child_gene)
                
            child_gene.generation = self.generation + 1
            next_gen.append(child_gene)

        self.population = next_gen
        self.generation += 1
        return self.population

    def _crossover(self, p1: GeneticGene, p2: GeneticGene) -> GeneticGene:
        child_dict: dict[str, Any] = {}
        for k, v in asdict(p1).items():
            if k in [
                "evaluated",
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

        # Ensure minimum confluences after crossover
        if len(child_dict["indicators"]) < 4:
            missing = 4 - len(child_dict["indicators"])
            pool = self._indicator_pool()
            available_inds = [ind for ind in pool if ind not in child_dict["indicators"]]
            if available_inds:
                child_dict["indicators"].extend(random.sample(available_inds, k=min(missing, len(available_inds))))

        child = GeneticGene(**child_dict)
        self._sync_indicator_maps(child)
        return child

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
                pool = self._indicator_pool()
                available_inds = [ind for ind in pool if ind not in gene.indicators]
                if available_inds:
                    mutated_gene.indicators.append(random.choice(available_inds))
            if not mutated_gene.indicators:
                pool = self._indicator_pool()
                mutated_gene.indicators.append(random.choice(pool))
            # Enforce minimum confluences
            if len(mutated_gene.indicators) < 4:
                missing = 4 - len(mutated_gene.indicators)
                pool = self._indicator_pool()
                available_inds = [ind for ind in pool if ind not in mutated_gene.indicators]
                if available_inds:
                    mutated_gene.indicators.extend(random.sample(available_inds, k=min(missing, len(available_inds))))
            self._sync_indicator_maps(mutated_gene)

        elif mutation_type == "params" and gene.indicators:
            target_ind = random.choice(gene.indicators)
            # Safety: check that params dict exists AND is not empty
            if target_ind in mutated_gene.params and mutated_gene.params[target_ind]:
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

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.talib_mixer import TALibStrategyGene, TALibStrategyMixer
from .base import ExpertModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GeneticStrategyExpert(ExpertModel):
    """
    Genetic Algorithm for evolving TA-Lib indicator combinations.
    Uses OHLC data (metadata) to generate signals, not just pre-computed features.
    """

    population_size: int = 50
    generations: int = 10
    max_indicators: int = 5
    best_gene: TALibStrategyGene | None = None
    mixer: TALibStrategyMixer | None = None
    portfolio: list[TALibStrategyGene] = field(default_factory=list)

    def __post_init__(self):
        self.mixer = TALibStrategyMixer()

    def fit(self, x: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame | None = None) -> None:
        if metadata is None:
            logger.warning("GeneticStrategyExpert requires metadata (OHLC) to fit. Skipping.")
            return

        if not self.mixer or not self.mixer.available_indicators:
            logger.warning("TALib mixer not available/empty. Skipping.")
            return

        self.portfolio = []  # List of TALibStrategyGene

        # BRIDGE: Try to load from the advanced Discovery Engine first
        try:
            import json
            from pathlib import Path

            cache_dir = Path("cache")
            knowledge_path = cache_dir / "talib_knowledge.json"

            if knowledge_path.exists():
                content = knowledge_path.read_text()
                if content.strip():
                    data = json.loads(content)

                    # Support both new Portfolio format and old Single format
                    genes_data = []
                    if "best_genes" in data:
                        genes_data = data["best_genes"]
                    elif "best_gene" in data:
                        genes_data = [data["best_gene"]]

                    for bg_data in genes_data:
                        try:
                            # Reconstruct Gene
                            gene = TALibStrategyGene(
                                indicators=bg_data.get("indicators", []),
                                params=bg_data.get("params", {}),
                                combination_method=bg_data.get("combination_method", "weighted_vote"),
                                long_threshold=bg_data.get("long_threshold", 0.6),
                                short_threshold=bg_data.get("short_threshold", -0.6),
                                weights=bg_data.get("weights", {}),
                                preferred_regime=bg_data.get("preferred_regime", "any"),
                                strategy_id=bg_data.get("strategy_id", "imported"),
                                # SMC Flags
                                use_ob=bg_data.get("use_ob", False),
                                use_fvg=bg_data.get("use_fvg", False),
                                use_liq_sweep=bg_data.get("use_liq_sweep", False),
                                mtf_confirmation=bg_data.get("mtf_confirmation", False),
                                use_premium_discount=bg_data.get("use_premium_discount", False),
                                use_inducement=bg_data.get("use_inducement", False),
                                tp_pips=bg_data.get("tp_pips", 40.0),
                                sl_pips=bg_data.get("sl_pips", 20.0),
                            )
                            gene.fitness = bg_data.get("fitness", 0.0)
                            self.portfolio.append(gene)
                        except Exception:
                            continue

                    if self.portfolio:
                        logger.info(f"BRIDGE: Loaded {len(self.portfolio)} strategies from Discovery Engine.")
                        self.best_gene = self.portfolio[0]  # Backwards compat
                        return
        except Exception as e:
            logger.warning(f"Failed to bridge Discovery Engine strategy: {e}", exc_info=True)

        logger.info(f"Evolving strategies over {self.generations} generations (Fallback Internal Mode)...")
        # ... Fallback legacy code ...
        population = [
            self.mixer.generate_random_strategy(max_indicators=self.max_indicators) for _ in range(self.population_size)
        ]

        df = metadata.copy()
        best_fitness = -np.inf

        for _gen in range(self.generations):
            scores = []
            cache = self.mixer.bulk_calculate_indicators(df, population)
            for gene in population:
                try:
                    signals = self.mixer.compute_signals(df, gene, cache=cache)
                    pos = signals.shift(1).fillna(0)
                    rets = df["close"].pct_change().fillna(0)
                    strat_rets = pos * rets
                    sharpe = strat_rets.mean() * 252 / (strat_rets.std() * np.sqrt(252) + 1e-9)
                    fitness = sharpe if (pos.diff().abs() > 0).sum() > 10 else -1.0
                    gene.fitness = fitness
                    scores.append((fitness, gene))
                except Exception:
                    scores.append((-1.0, gene))

            if scores:
                scores.sort(key=lambda x: x[0], reverse=True)
                if scores[0][0] > best_fitness:
                    best_fitness = scores[0][0]
                    self.best_gene = scores[0][1]

                survivors = [s[1] for s in scores[: self.population_size // 2]]
                new_pop = survivors[:]
                while len(new_pop) < self.population_size:
                    new_pop.append(self.mixer.generate_random_strategy(max_indicators=self.max_indicators))
                population = new_pop

        if self.best_gene:
            self.portfolio = [self.best_gene]

    def predict_proba(self, x: pd.DataFrame, metadata: pd.DataFrame | None = None) -> np.ndarray:
        n = len(x)
        if not self.portfolio or metadata is None:
            raise RuntimeError("Genetic model not initialized with portfolio or metadata")

        try:
            # Vote across portfolio
            total_vote = np.zeros(n, dtype=float)

            for gene in self.portfolio:
                signals = self.mixer.compute_signals(metadata, gene)
                signals = signals.reindex(x.index).fillna(0).values
                total_vote += signals

            # Average vote
            avg_vote = total_vote / len(self.portfolio)

            # Map average vote into smooth 3-class probabilities.
            # - Neutral dominates near 0 vote
            # - Buy/Sell dominate as vote moves away from 0
            avg_vote = np.clip(avg_vote, -1.0, 1.0).astype(np.float32, copy=False)
            k = 3.0
            neutral_base = 2.0
            neutral_alpha = 4.0
            neutral_logit = neutral_base - neutral_alpha * np.abs(avg_vote)
            logits = np.stack([neutral_logit, k * avg_vote, -k * avg_vote], axis=1).astype(np.float32, copy=False)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
            return probs.astype(float, copy=False)
        except Exception as e:
            logger.error(f"Genetic predict failed: {e}", exc_info=True)
            return np.zeros((n, 3))

    def save(self, path: str) -> None:
        if self.best_gene is None:
            return
        import joblib

        p = os.path.join(path, "genetic_expert.joblib")
        joblib.dump(self.best_gene, p)

    def load(self, path: str) -> None:
        import joblib

        p = os.path.join(path, "genetic_expert.joblib")
        if os.path.exists(p):
            try:
                self.best_gene = joblib.load(p)
                logger.info(f"Loaded Genetic Strategy: {self.best_gene.indicators}")
            except Exception as e:
                logger.warning(f"Failed to load Genetic Expert: {e}")

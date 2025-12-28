import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.storage import StrategyLedger
from ..features.cpu.indicators import NUMBA_AVAILABLE, compute_adx_numba
from ..features.talib_mixer import TALIB_AVAILABLE, TALibStrategyMixer
from ..strategy.genetic import GeneticStrategyEvolution

logger = logging.getLogger(__name__)


class AutonomousDiscoveryEngine:
    def __init__(self, cache_dir: Path, n_jobs: int = 1):
        self.cache_dir = cache_dir
        # HPC: Use a dedicated discovery ledger
        self.ledger = StrategyLedger(str(cache_dir / "strategy_ledger.sqlite"), symbol="DISCOVERY")
        self.n_jobs = n_jobs  # NOTE: Currently unused - GeneticStrategyEvolution doesn't support parallel evolution yet
        self.mixer = TALibStrategyMixer(use_volume_features=True)
        self.evolution = GeneticStrategyEvolution(population_size=100, mixer=self.mixer)

    def run_discovery_cycle(self, df: pd.DataFrame, population_size: int = 100) -> None:
        logger.info(f"Starting autonomous discovery cycle (Pop: {population_size})...")
        if not TALIB_AVAILABLE or df is None or df.empty:
            return

        regime = "any"
        try:
            if len(df) > 20:
                # Use a tail slice for speed on long histories
                tail = df.tail(5000) if len(df) > 5000 else df
                high = tail["high"].to_numpy()
                low = tail["low"].to_numpy()
                close = tail["close"].to_numpy()

                if NUMBA_AVAILABLE:
                    adx_arr = compute_adx_numba(high, low, close, period=14)
                    adx = float(adx_arr[-1])
                else:
                    tr1 = high - low
                    tr2 = np.abs(high - np.concatenate(([close[0]], close[:-1])))
                    tr3 = np.abs(low - np.concatenate(([close[0]], close[:-1])))
                    tr = np.maximum.reduce([tr1, tr2, tr3])
                    atr = pd.Series(tr).rolling(14).mean().to_numpy()

                    up_move = np.diff(high, prepend=high[0])
                    down_move = np.diff(low, prepend=low[0]) * -1

                    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
                    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

                    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / 14).mean() / atr
                    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / 14).mean() / atr

                    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
                    adx = float(pd.Series(dx).ewm(alpha=1 / 14).mean().iloc[-1])

                if adx > 25:
                    regime = "trend"
                elif adx < 20:
                    regime = "range"
                logger.info(f"Detected Market Regime: {regime.upper()} (ADX={adx:.1f})")
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            regime = "any"

        try:
            self.evolution.population_size = population_size
            self.evolution.initialize_population(regime=regime, validation_data=df)

            # Run multiple generations to allow convergence
            # Now that backtesting is Numba-optimized, this is fast.
            best_genes = []
            for i in range(5):
                logger.info(f"Evolution Cycle {i + 1}/5...")
                best_genes = self.evolution.evolve(validation_data=df)

            # Portfolio Selection (Sequential Niching)
            if best_genes:
                # 1. Generate signals for top 50 candidates to measure correlation
                candidates = best_genes[:50]
                signals_map = {}
                # Use bulk calc to speed up signal generation
                cache = self.mixer.bulk_calculate_indicators(df, candidates)

                valid_candidates = []
                for gene in candidates:
                    try:
                        sig = self.mixer.compute_signals(df, gene, cache=cache)
                        # Normalize signal for correlation (-1, 0, 1)
                        if sig.abs().sum() > 10:  # Minimum activity check
                            signals_map[gene.strategy_id] = sig
                            valid_candidates.append(gene)
                    except Exception:
                        continue

                # 2. Select diverse portfolio (Top 20 Uncorrelated)
                portfolio = []
                # Sort by fitness (Sharpe)
                valid_candidates.sort(key=lambda x: x.fitness, reverse=True)

                # Increased portfolio size to 20 to avoid "blindness" in models
                target_portfolio_size = 20 
                
                while len(portfolio) < target_portfolio_size and valid_candidates:
                    # Pick best remaining
                    leader = valid_candidates.pop(0)
                    portfolio.append(leader)

                    # Remove highly correlated followers
                    leader_sig = signals_map[leader.strategy_id]
                    survivors = []
                    for candidate in valid_candidates:
                        cand_sig = signals_map[candidate.strategy_id]
                        # Simple correlation of signal vectors
                        corr = leader_sig.corr(cand_sig)
                        if pd.isna(corr):
                            corr = 0.0

                        # Niching threshold: 0.7 (Allow a bit more similarity to get 20 strategies)
                        if abs(corr) < 0.7:
                            survivors.append(candidate)
                    valid_candidates = survivors

                logger.info(
                    f"Discovery selected {len(portfolio)} diverse strategies from {len(candidates)} candidates."
                )
                for i, p in enumerate(portfolio):
                    logger.info(f"  #{i + 1}: Sharpe={p.sharpe_ratio:.2f} ID={p.strategy_id} Inds={len(p.indicators)}")

                # BRIDGE: Persist the portfolio
                try:
                    import json

                    portfolio_payload = []
                    for gene in portfolio:
                        gene_data = {
                            "indicators": gene.indicators,
                            "params": gene.params,
                            "combination_method": gene.combination_method,
                            "long_threshold": float(gene.long_threshold),
                            "short_threshold": float(gene.short_threshold),
                            "weights": gene.weights,
                            "preferred_regime": gene.preferred_regime,
                            "fitness": float(gene.fitness),
                            "sharpe_ratio": float(gene.sharpe_ratio),
                            "win_rate": float(gene.win_rate),
                            "strategy_id": gene.strategy_id,
                            # SMC Flags
                            "use_ob": getattr(gene, "use_ob", False),
                            "use_fvg": getattr(gene, "use_fvg", False),
                            "use_liq_sweep": getattr(gene, "use_liq_sweep", False),
                            "mtf_confirmation": getattr(gene, "mtf_confirmation", False),
                            "use_premium_discount": getattr(gene, "use_premium_discount", False),
                            "use_inducement": getattr(gene, "use_inducement", False),
                            "tp_pips": getattr(gene, "tp_pips", 40.0),
                            "sl_pips": getattr(gene, "sl_pips", 20.0),
                        }
                        portfolio_payload.append(gene_data)

                    payload = {"best_genes": portfolio_payload, "timestamp": pd.Timestamp.now().isoformat()}

                    knowledge_path = self.cache_dir / "talib_knowledge.json"
                    knowledge_path.write_text(json.dumps(payload, indent=2))
                    logger.info(f"Persisted Portfolio of {len(portfolio)} strategies to {knowledge_path}")
                except Exception as exc:
                    logger.warning(f"Failed to persist discovery knowledge: {exc}")

        except Exception as e:
            logger.error(f"Discovery cycle failed: {e}", exc_info=True)

        self.mixer.save_knowledge(self.cache_dir / "talib_knowledge.json")

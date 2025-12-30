import concurrent.futures
import json
import logging
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.distributed as dist

from ..core.config import Settings
from ..core.system import resolve_cpu_budget
from ..features.talib_mixer import TALIB_AVAILABLE, TALibStrategyMixer
from .fast_backtest import fast_evaluate_strategy, infer_pip_metrics
from .stop_target import infer_stop_target_pips
from .genetic import GeneticGene, GeneticStrategyEvolution

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FitnessResult:
    fitness: float
    net_profit: float
    max_dd: float
    trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    sharpe: float
    sortino: float
    sqn: float


class PropAwareStrategySearch:
    """
    Long-running evolutionary search for prop-compliant strategies.
    - Uses TALibMixer to generate signals.
    - Scores via fast_evaluate_strategy on PnL with prop-style constraints.
    - Supports checkpoint/resume for multi-day runs.
    """

    def __init__(
        self,
        settings: Settings,
        df: pd.DataFrame,
        population: int = 1000,
        mutation_rate: float = 0.20,
        checkpoint_path: Path = Path("models") / "strategy_evo_checkpoint.json",
        max_time_hours: float = 120.0,  # allow multi-day runs
        max_workers: int | None = None,
        actual_balance: float | None = None,
    ) -> None:
        self.settings = settings
        self.df = df
        self._month_indices = np.zeros(len(self.df), dtype=np.int64)
        self._day_indices = np.zeros(len(self.df), dtype=np.int64)
        try:
            idx = getattr(self.df, "index", None)
            if isinstance(idx, pd.DatetimeIndex) and len(idx) == len(self.df):
                ts = idx.tz_convert("UTC") if idx.tz is not None else idx
                raw = ts.year.to_numpy() * 12 + ts.month.to_numpy()
                self._month_indices = np.nan_to_num(raw, nan=0.0).astype(np.int64, copy=False)
                day_raw = ts.year.to_numpy() * 10000 + ts.month.to_numpy() * 100 + ts.day.to_numpy()
                self._day_indices = np.nan_to_num(day_raw, nan=0.0).astype(np.int64, copy=False)
        except Exception:
            self._month_indices = np.zeros(len(self.df), dtype=np.int64)
            self._day_indices = np.zeros(len(self.df), dtype=np.int64)
        self.checkpoint_path = checkpoint_path
        self.max_time_hours = max_time_hours
        self.mixer = TALibStrategyMixer(
            device=getattr(settings.system, "device", "cpu"),
            use_volume_features=bool(getattr(settings.system, "use_volume_features", False)),
        )
        self.evolver = GeneticStrategyEvolution(
            population_size=population, mutation_rate=mutation_rate, mixer=self.mixer
        )
        self.fitness_cache: dict[str, FitnessResult] = {}
        self.actual_balance = actual_balance if actual_balance is not None else 100000.0
        self.symbol = getattr(settings.system, "symbol", "")
        self._indicator_cache: dict[str, np.ndarray] = {}
        # Auto-detect distributed rank/world for sharding
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if max_workers is None:
            try:
                self.max_workers = resolve_cpu_budget()
            except Exception:
                self.max_workers = 1
        else:
            try:
                cpu_total = multiprocessing.cpu_count()
            except Exception:
                cpu_total = 1
            self.max_workers = max(1, min(cpu_total, int(max_workers)))
        try:
            self._prop_portfolio_size = int(
                getattr(self.settings.models, "prop_search_portfolio_size", 100) or 100
            )
        except Exception:
            self._prop_portfolio_size = 100
        try:
            self._prop_min_trades = int(
                getattr(self.settings.models, "prop_min_trades", 0) or 0
            )
        except Exception:
            self._prop_min_trades = 0

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            payload = json.loads(self.checkpoint_path.read_text())
            self.evolver.generation = int(payload.get("generation", 0))
            pop = payload.get("population", [])
            genes = []
            for g in pop:
                gene = GeneticGene.from_dict(g)
                genes.append(gene)
                if "fitness" in g:
                    self.fitness_cache[gene.strategy_id] = FitnessResult(**g["fitness"])
            self.evolver.population = genes
            logger.info(f"Strategy evo checkpoint loaded (gen={self.evolver.generation}, pop={len(genes)})")
        except Exception as exc:
            logger.warning(f"Failed to load strategy evo checkpoint: {exc}")

    def _save_checkpoint(self) -> None:
        try:
            data = {
                "generation": self.evolver.generation,
                "population": [],
            }
            for gene in self.evolver.population:
                gd = gene.to_dict()
                fit = self.fitness_cache.get(gene.strategy_id)
                if fit:
                    gd["fitness"] = asdict(fit)
                data["population"].append(gd)
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_text(json.dumps(data))
        except Exception as exc:
            logger.warning(f"Failed to save strategy evo checkpoint: {exc}")

    def _prop_metric(self, metrics: np.ndarray) -> float:
        """
        HPC FIX: Hard-Stop Survival Enforcement.
        """
        (
            net_profit,
            sharpe,
            sortino,
            max_dd,
            win_rate,
            profit_factor,
            expectancy,
            sqn,
            trades,
            consistency,
            daily_dd,
        ) = metrics

        dd_limit = float(getattr(self.settings.risk, "total_drawdown_limit", 0.07) or 0.07)
        daily_limit = float(getattr(self.settings.risk, "daily_drawdown_limit", 0.04) or 0.04)

        # 1. Survival Check (Hard Limit)
        if max_dd >= dd_limit or daily_dd >= daily_limit:
            return -100.0 # Disqualified

        monthly_ret_pct = (net_profit / self.actual_balance) * 100.0
        target_monthly_pct = 4.0

        profit_score = monthly_ret_pct * 10.0
        target_bonus = max(0.0, (monthly_ret_pct - target_monthly_pct) * 50.0)

        fitness = profit_score + target_bonus
        fitness += 0.5 * sharpe
        fitness += 0.2 * profit_factor
        fitness += 0.1 * (win_rate * 100.0)
        
        # Stability: penalize zero trade strategies
        if trades < 10:
            fitness -= 50.0

        return float(fitness)

    def _evaluate_gene(self, gene: GeneticGene) -> FitnessResult:
        if gene.strategy_id in self.fitness_cache:
            return self.fitness_cache[gene.strategy_id]

        if not TALIB_AVAILABLE:
            logger.warning("TA-Lib not available; skipping evaluation.")
            res = FitnessResult(0.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self.fitness_cache[gene.strategy_id] = res
            return res

        try:
            signals = self.mixer.compute_signals(self.df, gene, cache=self._indicator_cache).to_numpy()
        except Exception as exc:
            logger.warning(f"Signal generation failed for {gene.strategy_id}: {exc}")
            res = FitnessResult(-1e9, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self.fitness_cache[gene.strategy_id] = res
            return res

        close = self.df["close"].to_numpy()
        high = self.df["high"].to_numpy()
        low = self.df["low"].to_numpy()
        tp_pips = float(getattr(gene, "tp_pips", 40.0))
        sl_pips = float(getattr(gene, "sl_pips", 20.0))
        spread = float(getattr(self.settings.risk, "backtest_spread_pips", 1.5))
        commission = float(getattr(self.settings.risk, "commission_per_lot", 7.0))
        max_hold = int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0)
        trailing_enabled = bool(getattr(self.settings.risk, "trailing_enabled", False))
        trailing_mult = float(getattr(self.settings.risk, "trailing_atr_multiplier", 1.0) or 1.0)
        trailing_trigger_r = float(getattr(self.settings.risk, "trailing_be_trigger_r", 1.0) or 1.0)
        pip_size, pip_value_per_lot = infer_pip_metrics(self.symbol)
        mode = str(getattr(self.settings.risk, "stop_target_mode", "blend") or "blend").lower()
        if mode not in {"fixed", "gene"}:
            res = infer_stop_target_pips(self.df, settings=self.settings, pip_size=pip_size)
            if res is not None:
                sl_pips, tp_pips, _rr = res
        metrics = fast_evaluate_strategy(
            close_prices=close,
            high_prices=high,
            low_prices=low,
            signals=signals,
            month_indices=self._month_indices,
            day_indices=self._day_indices,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            max_hold_bars=max_hold,
            trailing_enabled=trailing_enabled,
            trailing_atr_multiplier=trailing_mult,
            trailing_be_trigger_r=trailing_trigger_r,
            spread_pips=spread,
            commission_per_trade=commission,
            pip_value=pip_size,
            pip_value_per_lot=pip_value_per_lot,
        )

        (
            net_profit,
            sharpe,
            sortino,
            max_dd,
            win_rate,
            profit_factor,
            expectancy,
            sqn,
            trades,
            _consistency,
            daily_dd,
        ) = metrics
        fitness = self._prop_metric(metrics)
        res = FitnessResult(
            fitness=fitness,
            net_profit=float(net_profit),
            max_dd=float(max_dd),
            trades=int(trades),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            expectancy=float(expectancy),
            sharpe=float(sharpe),
            sortino=float(sortino),
            sqn=float(sqn),
        )
        self.fitness_cache[gene.strategy_id] = res
        return res

    def _evaluate_population_parallel(self, genes: list[GeneticGene]) -> None:
        """
        Evaluate a population in parallel.
        HPC FIX: Use ProcessPoolExecutor for TRUE GIL bypass.
        """
        # Shard by rank if running under torchrun
        if self.world_size > 1:
            genes = [g for i, g in enumerate(genes) if i % self.world_size == self.rank]
        genes_to_eval = [g for g in genes if g.strategy_id not in self.fitness_cache]
        if not genes_to_eval:
            return
            
        max_workers = self.max_workers or 1
        ctx_name = str(os.environ.get("FOREX_BOT_PROP_MP_CONTEXT", "auto") or "auto").strip().lower()
        if ctx_name == "auto":
            # Prefer fork on Unix for copy-on-write to reduce RAM, unless GPU is used.
            if os.name != "nt" and not getattr(self.mixer, "use_gpu", False):
                ctx_name = "fork"
            else:
                ctx_name = "spawn"
        spawn_ctx = multiprocessing.get_context(ctx_name)
        
        # HPC: True Multi-Process Evaluation
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as ex:
            future_map = {ex.submit(self._evaluate_gene, g): g for g in genes_to_eval}
            for fut in concurrent.futures.as_completed(future_map):
                g = future_map[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    logger.warning(f"Parallel eval failed for {g.strategy_id}: {exc}")
                    res = FitnessResult(-1e9, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                self.fitness_cache[g.strategy_id] = res

    def run(self, generations: int = 50) -> tuple[GeneticGene, FitnessResult]:
        if not TALIB_AVAILABLE:
            raise RuntimeError("TA-Lib not installed; cannot run strategy evolution.")

        start = time.time()
        self._load_checkpoint()
        if not self.evolver.population:
            self.evolver.initialize_population(regime="any")

        best_gene: GeneticGene | None = None
        best_fit: FitnessResult | None = None

        for _ in range(generations):
            if (time.time() - start) / 3600.0 >= self.max_time_hours:
                logger.info("Max evolution wall-clock reached; stopping.")
                break

            # Pre-compute indicator cache for current population to avoid recomputation
            try:
                self._indicator_cache = self.mixer.bulk_calculate_indicators(self.df, self.evolver.population)
            except Exception as exc:
                logger.warning(f"Indicator cache generation failed, continuing without cache: {exc}")
                self._indicator_cache = {}

            self._evaluate_population_parallel(self.evolver.population)
            for gene in self.evolver.population:
                fit = self.fitness_cache.get(gene.strategy_id)
                if not fit:
                    fit = self._evaluate_gene(gene)
                gene.fitness = fit.fitness
                gene.win_rate = fit.win_rate
                gene.max_drawdown = fit.max_dd
                gene.expectancy = fit.expectancy
                gene.profit_factor = fit.profit_factor
                gene.sharpe_ratio = fit.sharpe

            sorted_pop = sorted(self.evolver.population, key=lambda g: g.fitness, reverse=True)
            top = sorted_pop[0]
            top_fit = self.fitness_cache[top.strategy_id]
            if best_fit is None or top_fit.fitness > best_fit.fitness:
                best_gene = top
                best_fit = top_fit
                logger.info(
                    f"New best strategy {top.strategy_id} fit={top_fit.fitness:.2f} "
                    f"ret={top_fit.net_profit:.1f} dd={top_fit.max_dd:.2%} trades={top_fit.trades}"
                )

            self.evolver.evolve()
            self._save_checkpoint()

        if best_gene is None or best_fit is None:
            best_gene = sorted(self.evolver.population, key=lambda g: g.fitness, reverse=True)[0]
            best_fit = self.fitness_cache.get(best_gene.strategy_id, FitnessResult(0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
        return best_gene, best_fit

    def _gene_payload(self, gene: GeneticGene, fit: FitnessResult | None) -> dict[str, Any]:
        return {
            "indicators": list(getattr(gene, "indicators", []) or []),
            "params": dict(getattr(gene, "params", {}) or {}),
            "combination_method": getattr(gene, "combination_method", "weighted_vote"),
            "long_threshold": float(getattr(gene, "long_threshold", 0.6)),
            "short_threshold": float(getattr(gene, "short_threshold", -0.6)),
            "weights": dict(getattr(gene, "weights", {}) or {}),
            "preferred_regime": getattr(gene, "preferred_regime", "any"),
            "fitness": float(getattr(fit, "fitness", 0.0)) if fit else float(getattr(gene, "fitness", 0.0)),
            "sharpe_ratio": float(getattr(fit, "sharpe", getattr(gene, "sharpe_ratio", 0.0))) if fit else float(getattr(gene, "sharpe_ratio", 0.0)),
            "win_rate": float(getattr(fit, "win_rate", getattr(gene, "win_rate", 0.0))) if fit else float(getattr(gene, "win_rate", 0.0)),
            "profit_factor": float(getattr(fit, "profit_factor", getattr(gene, "profit_factor", 0.0))) if fit else float(getattr(gene, "profit_factor", 0.0)),
            "expectancy": float(getattr(fit, "expectancy", getattr(gene, "expectancy", 0.0))) if fit else float(getattr(gene, "expectancy", 0.0)),
            "max_dd": float(getattr(fit, "max_dd", getattr(gene, "max_drawdown", 0.0))) if fit else float(getattr(gene, "max_drawdown", 0.0)),
            "trades": int(getattr(fit, "trades", getattr(gene, "trades_count", 0))) if fit else int(getattr(gene, "trades_count", 0)),
            "strategy_id": getattr(gene, "strategy_id", ""),
            "use_ob": bool(getattr(gene, "use_ob", False)),
            "use_fvg": bool(getattr(gene, "use_fvg", False)),
            "use_liq_sweep": bool(getattr(gene, "use_liq_sweep", False)),
            "mtf_confirmation": bool(getattr(gene, "mtf_confirmation", False)),
            "use_premium_discount": bool(getattr(gene, "use_premium_discount", False)),
            "use_inducement": bool(getattr(gene, "use_inducement", False)),
            "tp_pips": float(getattr(gene, "tp_pips", 40.0)),
            "sl_pips": float(getattr(gene, "sl_pips", 20.0)),
            "source": "propaware",
        }

    def export_portfolio(self, cache_dir: Path, *, merge: bool = True, max_genes: int | None = None) -> int:
        max_genes = int(max_genes or self._prop_portfolio_size or 100)
        if not self.evolver.population:
            return 0

        # Build payloads for top strategies
        sorted_pop = sorted(self.evolver.population, key=lambda g: g.fitness, reverse=True)
        payloads: list[dict[str, Any]] = []
        for gene in sorted_pop:
            fit = self.fitness_cache.get(gene.strategy_id)
            if self._prop_min_trades > 0 and fit and fit.trades < self._prop_min_trades:
                continue
            payloads.append(self._gene_payload(gene, fit))
            if len(payloads) >= max_genes:
                break

        if not payloads:
            return 0

        knowledge_path = cache_dir / "talib_knowledge.json"
        combined = payloads

        if merge and knowledge_path.exists():
            try:
                existing = json.loads(knowledge_path.read_text())
                existing_genes = []
                if isinstance(existing, dict):
                    if "best_genes" in existing:
                        existing_genes = list(existing.get("best_genes") or [])
                    elif "best_gene" in existing:
                        existing_genes = [existing.get("best_gene")]
                if existing_genes:
                    combined = existing_genes + payloads
            except Exception as exc:
                logger.warning(f"Failed to read existing knowledge for merge: {exc}")

        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for gene in combined:
            if not isinstance(gene, dict):
                continue
            key = str(gene.get("strategy_id") or "")
            if not key:
                key = json.dumps(
                    {
                        "indicators": gene.get("indicators", []),
                        "params": gene.get("params", {}),
                        "long_threshold": gene.get("long_threshold", 0.0),
                        "short_threshold": gene.get("short_threshold", 0.0),
                        "weights": gene.get("weights", {}),
                    },
                    sort_keys=True,
                )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(gene)

        deduped.sort(key=lambda g: float(g.get("fitness", 0.0) or 0.0), reverse=True)
        deduped = deduped[:max_genes]

        payload = {
            "best_genes": deduped,
            "timestamp": pd.Timestamp.now().isoformat(),
            "sources": {
                "propaware": len(payloads),
                "merged_total": len(deduped),
            },
        }
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        knowledge_path.write_text(json.dumps(payload, indent=2))
        logger.info("Prop-aware portfolio merged into %s (%s strategies).", knowledge_path, len(deduped))
        return len(deduped)


def run_evo_search(
    ohlcv: pd.DataFrame,
    settings: Settings | None = None,
    population: int = 64,
    generations: int = 50,
    checkpoint: str = "models/strategy_evo_checkpoint.json",
    max_time_hours: float = 120.0,
    actual_balance: float | None = None,
) -> tuple[GeneticGene, FitnessResult]:
    settings = settings or Settings()
    search = PropAwareStrategySearch(
        settings=settings,
        df=ohlcv,
        population=population,
        checkpoint_path=Path(checkpoint),
        max_time_hours=max_time_hours,
        actual_balance=actual_balance,
    )
    result = search.run(generations=generations)
    try:
        cache_dir = Path(getattr(settings.system, "cache_dir", "cache"))
        search.export_portfolio(cache_dir, merge=True)
    except Exception as exc:
        logger.warning(f"Failed to export prop-aware portfolio: {exc}")
    return result

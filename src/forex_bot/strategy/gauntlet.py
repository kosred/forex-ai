import logging

import os
import pandas as pd

from ..core.config import Settings
from ..features.talib_mixer import TALibStrategyGene, TALibStrategyMixer
from ..training.evaluation import prop_backtest

logger = logging.getLogger(__name__)


class StrategyGauntlet:
    """
    The Gatekeeper.
    Validates strategies against recent market data before allowing them into the gene pool.
    Ensures 'Day 1 Survival'.
    """

    def __init__(
        self,
        mixer: TALibStrategyMixer,
        validation_data: pd.DataFrame,
        settings: Settings | None = None,
    ) -> None:
        self.mixer = mixer
        self.data = validation_data
        settings = settings or Settings()
        self.min_win_rate = 0.55
        self.min_profit_factor = 1.2
        self.max_drawdown_pct = float(getattr(settings.risk, "total_drawdown_limit", 0.07) or 0.07)
        self.max_daily_dd = float(getattr(settings.risk, "daily_drawdown_limit", 0.04) or 0.04)
        self.warn_only = str(os.environ.get("FOREX_BOT_GAUNTLET_WARN_ONLY", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}

    def run(self, gene: TALibStrategyGene) -> bool:
        """
        Runs the strategy through the gauntlet.
        Returns True if it passes, False otherwise.
        """
        if self.data is None or self.data.empty:
            logger.warning("Gauntlet skipped: No validation data.")
            return True  # Fail open if no data (training will filter later)

        try:
            signals = self.mixer.compute_signals(self.data, gene)
            
            # HPC FIX: Use Metadata for High-Precision Validation
            # Ensure prop_backtest uses the real OHLC prices for cost logic
            metrics = prop_backtest(
                self.data,
                signals,
                max_daily_dd_pct=self.max_daily_dd,
                daily_dd_warn_pct=self.max_daily_dd * 0.8,
                max_trades_per_day=10,
                use_gpu=False
            )

            wr = metrics.get("win_rate", 0.0)
            dd = metrics.get("max_dd_pct", 1.0)
            pnl = metrics.get("pnl_score", 0.0)
            daily_violation = bool(metrics.get("daily_dd_violation", False))

            if wr < self.min_win_rate:
                if self.warn_only:
                    logger.warning("Gauntlet warn-only: win_rate %.3f below %.3f", wr, self.min_win_rate)
                    return True
                return False
            if dd > self.max_drawdown_pct:
                if self.warn_only:
                    logger.warning("Gauntlet warn-only: max_dd %.3f above %.3f", dd, self.max_drawdown_pct)
                    return True
                return False
            if daily_violation:
                if self.warn_only:
                    logger.warning("Gauntlet warn-only: daily drawdown violation")
                    return True
                return False
            if pnl <= 0:
                if self.warn_only:
                    logger.warning("Gauntlet warn-only: pnl %.4f <= 0", pnl)
                    return True
                return False

            return True

        except Exception as e:
            logger.warning(f"Gauntlet execution failed for {gene.strategy_id}: {e}")
            return False

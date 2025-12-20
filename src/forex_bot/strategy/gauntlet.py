import logging

import pandas as pd

from ..features.talib_mixer import TALibStrategyGene, TALibStrategyMixer
from ..training.evaluation import prop_backtest

logger = logging.getLogger(__name__)


class StrategyGauntlet:
    """
    The Gatekeeper.
    Validates strategies against recent market data before allowing them into the gene pool.
    Ensures 'Day 1 Survival'.
    """

    def __init__(self, mixer: TALibStrategyMixer, validation_data: pd.DataFrame) -> None:
        self.mixer = mixer
        self.data = validation_data
        self.min_win_rate = 0.60
        self.min_profit_factor = 1.2
        self.max_drawdown_pct = 0.03

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

            metrics = prop_backtest(
                self.data, signals, max_daily_dd_pct=0.05, daily_dd_warn_pct=0.03, max_trades_per_day=10
            )

            wr = metrics.get("win_rate", 0.0)
            dd = metrics.get("max_dd_pct", 1.0)
            pnl = metrics.get("pnl_score", 0.0)

            if wr < self.min_win_rate:
                return False
            if dd > self.max_drawdown_pct:
                return False
            if pnl <= 0:
                return False

            return True

        except Exception as e:
            logger.warning(f"Gauntlet execution failed for {gene.strategy_id}: {e}")
            return False

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..models.exit_agent import ExitAgent
from .mt5_state_manager import MT5Position

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CloseInstruction:
    ticket: int
    symbol: str
    volume: float  # 0.0 means close all
    reason: str
    score: float


class TradeDoctor:
    """
    Active Trade Management (ATM) Agent.
    Monitors open positions for signs of weakness, stall, or reversal.
    Acts like a human trader managing trades post-entry.
    """

    def __init__(self, settings):
        self.settings = settings
        dyn = getattr(self.settings, "dynamic", {})
        doc_params = dyn.get("doctor_params", {})

        self.stall_threshold_minutes = doc_params.get("stall_threshold_minutes", 60.0)
        self.min_profit_to_bank = doc_params.get("min_profit_to_bank", 0.3)
        self.reversal_sensitivity = doc_params.get("reversal_sensitivity", 0.2)

        self.exit_agent = ExitAgent(settings)
        try:
            self.exit_agent.load("models")
        except Exception as e:
            logger.warning(f"Trade analysis failed: {e}", exc_info=True)

        logger.info(
            f"TradeDoctor initialized with: Stall={self.stall_threshold_minutes:.1f}m, "
            f"BankR={self.min_profit_to_bank:.2f}, RevSens={self.reversal_sensitivity:.2f}"
        )

    def diagnose(self, positions: list[MT5Position], frames: dict[str, pd.DataFrame]) -> list[CloseInstruction]:
        """
        Evaluate all open positions.
        """
        instructions = []
        if not positions:
            return instructions

        df = frames.get("M1")
        if df is None:
            df = frames.get("M5")
        if df is None or df.empty:
            return instructions

        current_price = df["close"].iloc[-1]
        recent_closes = df["close"].iloc[-5:].values
        momentum = (recent_closes[-1] - recent_closes[0]) if len(recent_closes) > 1 else 0.0

        rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else 50.0

        vol = df["atr"].iloc[-1] if "atr" in df.columns else 0.001

        now = datetime.now(UTC)

        for pos in positions:
            if pos.symbol != self.settings.system.symbol:
                continue

            instruction = self._check_position(pos, current_price, momentum, rsi, vol, now)
            if instruction:
                instructions.append(instruction)

        return instructions

    def _check_position(
        self, pos: MT5Position, current_price: float, momentum: float, rsi: float, volatility: float, now: datetime
    ) -> CloseInstruction | None:
        """
        Diagnose a single position.
        """
        duration_mins = (now - pos.time).total_seconds() / 60.0

        risk_dist = abs(pos.price_open - pos.sl) if pos.sl > 0 else 0.0
        if risk_dist <= 0:
            return None  # Can't calc R without SL

        current_dist = (current_price - pos.price_open) if pos.type == 0 else (pos.price_open - current_price)
        r_multiple = current_dist / risk_dist

        state_vec = np.array(
            [
                r_multiple,
                duration_mins / 60.0,  # Normalize to hours roughly
                momentum / volatility if volatility > 0 else 0.0,
                volatility,
                rsi / 100.0,
                current_dist,
            ],
            dtype=np.float32,
        )

        rl_action = self.exit_agent.get_action(state_vec)

        self.exit_agent.observe_exit(pos.ticket, state_vec, rl_action, current_price, now)

        if rl_action > 0.5:  # FIX: Robust check for both binary (0/1) and continuous (0.0-1.0) actions
            return CloseInstruction(
                pos.ticket, pos.symbol, 0.0, f"AI Exit Agent Decision (R={r_multiple:.2f})", r_multiple
            )

        if r_multiple > self.min_profit_to_bank:  # At least evolved R profit
            # FIX: Use ATR (volatility) for stall detection instead of tight SL distance
            # If 5-bar momentum is less than 0.5 ATR, price is stalling.
            if duration_mins > self.stall_threshold_minutes and abs(momentum) < (volatility * 0.5):
                return CloseInstruction(
                    pos.ticket,
                    pos.symbol,
                    0.0,
                    f"Stalled in profit ({r_multiple:.2f}R) for {int(duration_mins)}m",
                    r_multiple,
                )

        if r_multiple > (self.min_profit_to_bank * 2.0):  # Good profit (2x min bank)
            thresh = risk_dist * self.reversal_sensitivity

            if pos.type == 0:  # Buy
                if momentum < -thresh:  # Sharp drop
                    return CloseInstruction(
                        pos.ticket, pos.symbol, 0.0, f"Momentum reversal in profit ({r_multiple:.2f}R)", r_multiple
                    )
            elif pos.type == 1:  # Sell
                if momentum > thresh:  # Sharp rally
                    return CloseInstruction(
                        pos.ticket, pos.symbol, 0.0, f"Momentum reversal in profit ({r_multiple:.2f}R)", r_multiple
                    )

        if -0.5 < r_multiple < 0:  # Small loss
            if duration_mins > 120:  # 2 hours doing nothing but bleeding
                return CloseInstruction(pos.ticket, pos.symbol, 0.0, "Zombie trade (small loss) > 2h", r_multiple)

        return None

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import Settings
from ..core.storage import RiskLedger, StrategyLedger
from ..execution.mt5_state_manager import MT5StateManager
from ..execution.risk import RiskManager

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Handles the mechanics of calculating order parameters (SL/TP/Size)
    and executing trades via MT5.
    """

    def __init__(
        self,
        settings: Settings,
        risk_manager: RiskManager,
        mt5_manager: MT5StateManager,
        strategy_ledger: StrategyLedger | None = None,
        risk_ledger: RiskLedger | None = None,
    ):
        self.settings = settings
        self.risk_manager = risk_manager
        self.mt5 = mt5_manager
        self.strategy_ledger = strategy_ledger
        self.risk_ledger = risk_ledger

    async def close_position(self, ticket: int, volume: float, reason: str | None = None) -> None:
        """
        Close a position via MT5StateManager, surfacing failures instead of silently swallowing them.
        """
        symbol = self.settings.system.symbol
        success = await self.mt5.close_position_by_ticket(ticket, symbol, volume=volume)
        if not success:
            raise RuntimeError(f"Close failed for ticket {ticket} ({symbol}): {reason or 'no reason provided'}")
        logger.info(f"Closed position ticket={ticket} vol={volume} reason={reason or 'n/a'}")

    async def execute_signal(
        self,
        signal_result: Any,
        equity: float,
        frames: dict[str, Any],
        alloc_weight: Any = None,
        advice_stance: str | None = None,
    ) -> None:
        """
        Process a buy/sell signal: calc risk, place order, log intent.
        """
        if signal_result.signal == 0:
            return

        symbol = self.settings.system.symbol

        # 1. Calculate SL Pips
        sl_pips = self._calculate_sl_pips(signal_result, frames)
        if sl_pips is None:
            logger.warning("Could not calculate SL pips; skipping trade.")
            return

        # 2. Calculate Size
        symbol_info = await self.mt5.connection.get_symbol_info(symbol) or {}
        uncertainty = self._latest_scalar(getattr(signal_result, "uncertainty", 0.0), default=0.0)

        size = self.risk_manager.calculate_position_size(
            equity,
            sl_pips,
            signal_result.confidence,
            uncertainty,
            symbol_info,
            market_regime=getattr(signal_result, "regime", "Normal"),
        )

        # 3. Adjust Size based on Advice/Allocation
        if advice_stance:
            if advice_stance == "conservative":
                size *= 0.7
            elif advice_stance == "aggressive":
                size *= 1.1

        if alloc_weight:
            size = size * max(0.1, min(1.0, alloc_weight.weight))

        if size <= 0:
            logger.info(f"Signal {signal_result.signal} ignored (Calculated size 0).")
            return

        # 4. Calculate Prices
        tick = await self.mt5.connection.get_symbol_price(symbol) or {}
        sl, tp = self._calculate_prices(signal_result, frames, sl_pips, symbol_info, tick)
        if sl is None or tp is None:
            return

        base_df = frames[self.settings.system.base_timeframe]
        try:
            if "timestamp" in getattr(base_df, "columns", []):
                current_bar_time = base_df["timestamp"].iloc[-1]
            else:
                current_bar_time = base_df.index[-1]
            # Normalize to python datetime where possible
            if hasattr(current_bar_time, "to_pydatetime"):
                current_bar_time = current_bar_time.to_pydatetime()
            if isinstance(current_bar_time, str):
                current_bar_time = datetime.fromisoformat(current_bar_time)
        except Exception:
            current_bar_time = datetime.utcnow()
        order_type = "buy" if signal_result.signal == 1 else "sell"

        logger.info(f"Executing {order_type.upper()} {size} lots (SL={sl:.5f}, TP={tp:.5f})...")

        # 5. Execute with Retry
        result = {"success": False, "reason": "Execution failed"}
        for attempt in range(1, 4):
            try:
                result = await self.mt5.place_order_with_verification(
                    symbol=symbol, order_type=order_type, volume=size, sl=sl, tp=tp, current_bar_time=current_bar_time
                )
                if result["success"]:
                    break
                else:
                    logger.warning(f"Order attempt {attempt} failed: {result.get('reason')}. Retrying...")
                    import asyncio

                    await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Order attempt {attempt} raised exception: {e}")
                import asyncio

                await asyncio.sleep(1.0)

        # 6. Post-Order Logic
        if result["success"]:
            self._handle_success(result, order_type, size, sl, tp, signal_result)
        else:
            self._handle_failure(result)

    @staticmethod
    def _latest_scalar(value: Any, *, default: float = 0.0) -> float:
        """
        Convert common vector-like outputs (Series/ndarray/list) into a scalar.

        Many model outputs are per-row Series; for live execution we always want the latest value.
        """
        if value is None:
            return float(default)

        # Pandas objects: take last element.
        if isinstance(value, pd.Series):
            if len(value) == 0:
                return float(default)
            try:
                return float(value.iloc[-1])
            except Exception:
                return float(default)
        if isinstance(value, pd.DataFrame):
            if value.empty:
                return float(default)
            try:
                return float(value.iloc[-1, -1])
            except Exception:
                return float(default)

        # Numpy arrays / sequences: take last flattened value.
        if isinstance(value, (np.ndarray, list, tuple)):
            try:
                arr = np.asarray(value)
                if arr.size == 0:
                    return float(default)
                return float(arr.reshape(-1)[-1])
            except Exception:
                return float(default)

        try:
            return float(value)
        except Exception:
            return float(default)

    def _calculate_sl_pips(self, result, frames) -> float | None:
        # Check recommended
        if result.recommended_sl is not None:
            try:
                val = float(result.recommended_sl.iloc[-1])
                if val > 0:
                    return val
            except Exception as e:
                logger.debug(f"Could not extract recommended_sl: {e}")

        # Fallback to ATR
        try:
            base_df = frames[self.settings.system.base_timeframe]
            atr = base_df["atr"].iloc[-1] if "atr" in base_df.columns else None
            pip_size = self._get_pip_size()

            if atr and atr > 0:
                return max(5.0, float((atr * 1.5) / pip_size))
        except Exception as e:
            logger.debug(f"ATR-based SL calculation failed: {e}")

        return 20.0  # Ultimate fallback

    def _calculate_prices(
        self, result, frames, sl_pips, info, tick_price: dict[str, float]
    ) -> tuple[float, float] | tuple[None, None]:
        try:
            base_df = frames[self.settings.system.base_timeframe]
            close_price = float(base_df["close"].iloc[-1])
            bid = float(tick_price.get("bid", 0.0) or 0.0)
            ask = float(tick_price.get("ask", 0.0) or 0.0)
            # Use live bid/ask when available; otherwise fall back to last close
            if result.signal == 1:  # buy uses ask
                entry_price = ask if ask > 0 else close_price
            else:  # sell uses bid
                entry_price = bid if bid > 0 else close_price

            pip_size = self._get_pip_size(info)
            sl_dist = sl_pips * pip_size

            rr = 2.0
            if result.recommended_rr is not None:
                try:
                    rr = float(result.recommended_rr.iloc[-1])
                except Exception as e:
                    logger.debug(f"Could not extract recommended_rr: {e}")

            if result.signal == 1:
                return entry_price - sl_dist, entry_price + (rr * sl_dist)
            else:
                return entry_price + sl_dist, entry_price - (rr * sl_dist)
        except Exception as e:
            logger.error(f"Price calc failed: {e}")
            return None, None

    def _handle_success(self, result, order_type, size, sl, tp, signal_result):
        logger.info(f"[ORDER SUCCESS] Ticket={result.get('ticket')}")
        if self.risk_manager:
            self.risk_manager.on_trade_opened(datetime.now())

        if self.strategy_ledger and result.get("ticket"):
            try:
                self.strategy_ledger.log_intent(
                    ticket=result["ticket"],
                    symbol=self.settings.system.symbol,
                    direction=order_type,
                    volume=size,
                    sl=sl,
                    tp=tp,
                    meta_risk_mult=getattr(self.risk_manager, "last_risk_mult", 1.0),
                )
            except Exception as e:
                logger.warning(f"Failed to log trade intent: {e}", exc_info=True)

    def _handle_failure(self, result):
        logger.error(f"[ORDER FAIL] {result.get('reason')}")
        if result.get("requires_manual_check") and self.risk_ledger:
            self.risk_ledger.record("ORDER_UNVERIFIED", f"Critical: {result.get('reason')}", severity="critical")

    def _get_pip_size(self, info=None) -> float:
        sym = (self.settings.system.symbol or "").upper()
        if info:
            pt = float(info.get("point", 0.0001) or 0.0001)
            dig = int(info.get("digits", 5) or 5)
            pip_size = pt * (10 if dig >= 4 else 1)
        else:
            if sym.endswith("JPY") or sym.startswith("JPY"):
                pip_size = 0.01
            elif sym.startswith("XAU") or sym.startswith("XAG"):
                pip_size = 0.01
            elif "BTC" in sym or "ETH" in sym or "LTC" in sym:
                pip_size = 1.0
            else:
                pip_size = 0.0001
        return pip_size

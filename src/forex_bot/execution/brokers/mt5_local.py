import asyncio
from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    import MetaTrader5 as MT5  # noqa: N814
except ImportError:
    MT5 = None

from ...utils.window_control import ensure_autotrading_enabled


@dataclass(slots=True)
class _OrderResult:
    retcode: int
    comment: str
    order: int | None = None
    deal: int | None = None


class _Connection:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._tf_map = {
            "M1": MT5.TIMEFRAME_M1 if MT5 else None,
            "M3": MT5.TIMEFRAME_M3 if MT5 else None,
            "M5": MT5.TIMEFRAME_M5 if MT5 else None,
            "M15": MT5.TIMEFRAME_M15 if MT5 else None,
            "M30": MT5.TIMEFRAME_M30 if MT5 else None,
            "H1": MT5.TIMEFRAME_H1 if MT5 else None,
            "H4": MT5.TIMEFRAME_H4 if MT5 else None,
            "D1": MT5.TIMEFRAME_D1 if MT5 else None,
            "W1": MT5.TIMEFRAME_W1 if MT5 else None,
            "MN1": MT5.TIMEFRAME_MN1 if MT5 else None,
        }
        self._last_symbols_cache: list[str] = []

    async def symbol_select(self, symbol: str, enable: bool) -> bool:
        if MT5 is None:
            return False
        return bool(MT5.symbol_select(symbol, enable))

    async def symbols_list(self) -> list[str]:
        if MT5 is None:
            return []
        try:
            info = MT5.symbols_get()
            names = [getattr(s, "name", "") for s in info if getattr(s, "name", "")]
            self._last_symbols_cache = names
            return names
        except Exception:
            return self._last_symbols_cache

    async def get_account_information(self) -> dict[str, Any]:
        if MT5 is None:
            raise RuntimeError("MT5 package not available. Install MetaTrader5 package.")
        info = MT5.account_info()
        if not info:
            raise RuntimeError("Failed to get MT5 account info. Check MT5 connection.")
        return info._asdict()

    async def positions_get(self) -> list[dict[str, Any]]:
        if MT5 is None:
            raise RuntimeError("MT5 package not available. Install MetaTrader5 package.")
        positions = MT5.positions_get()
        out: list[dict[str, Any]] = []
        if positions:
            for p in positions:
                out.append(p._asdict())
        return out

    def _get_filling_mode(self, symbol: str) -> int:
        """Get the best filling mode for symbol (handles brokers that don't support all modes)."""
        info = MT5.symbol_info(symbol)
        if not info:
            return MT5.ORDER_FILLING_IOC  # Safe default

        fill_flags = info.filling_mode

        # Try IOC first (most widely supported), then FOK, then RETURN
        if fill_flags & 2:  # SYMBOL_FILLING_IOC = 2
            return MT5.ORDER_FILLING_IOC
        elif fill_flags & 1:  # SYMBOL_FILLING_FOK = 1
            return MT5.ORDER_FILLING_FOK
        elif hasattr(MT5, 'ORDER_FILLING_RETURN') and (fill_flags & 4):  # SYMBOL_FILLING_RETURN = 4
            return MT5.ORDER_FILLING_RETURN

        # Fallback to IOC
        return MT5.ORDER_FILLING_IOC

    def _send_with_filling_retry(self, request: dict) -> Any:
        """Send order with automatic filling mode retry on error 10030."""
        symbol = request.get("symbol", "")

        # Get best filling mode for this symbol
        filling_mode = self._get_filling_mode(symbol)
        request["type_filling"] = filling_mode

        # Try with detected mode first
        result = MT5.order_send(request)

        # If unsupported filling mode (10030), try alternatives
        if result and result.retcode == 10030:
            modes_to_try = [MT5.ORDER_FILLING_IOC, MT5.ORDER_FILLING_FOK]
            if hasattr(MT5, 'ORDER_FILLING_RETURN'):
                modes_to_try.append(MT5.ORDER_FILLING_RETURN)

            for mode in modes_to_try:
                if mode == filling_mode:
                    continue  # Already tried this one
                request["type_filling"] = mode
                result = MT5.order_send(request)
                if result.retcode != 10030:
                    break

        return result

    async def close_position(self, symbol: str) -> int:
        """DEPRECATED: Closes ALL positions on symbol. Use close_position_by_ticket instead."""
        if MT5 is None:
            return 0
        positions = MT5.positions_get(symbol=symbol)
        closed = 0
        if not positions:
            return 0
        info = MT5.symbol_info(symbol)
        if not info:
            return 0

        tick = MT5.symbol_info_tick(symbol)
        if not tick:
            return 0

        for p in positions:
            close_type = MT5.ORDER_TYPE_SELL if p.type == MT5.POSITION_TYPE_BUY else MT5.ORDER_TYPE_BUY
            price = tick.bid if close_type == MT5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action": MT5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": p.volume,
                "type": close_type,
                "price": price,
                "position": p.ticket,
                "deviation": 20,
                "magic": 0,
                "comment": "auto-close",
            }
            result = self._send_with_filling_retry(request)
            if result and result.retcode == MT5.TRADE_RETCODE_DONE:
                closed += 1
        return closed

    async def close_position_by_ticket(self, ticket: int, symbol: str, volume: float | None = None) -> dict[str, Any]:
        """Close a specific position by its ticket ID (safe method). Supports partial close."""
        if MT5 is None:
            return {"retcode": -1, "comment": "MT5 unavailable"}

        positions = MT5.positions_get(symbol=symbol)
        if not positions:
            return {"retcode": -2, "comment": f"No positions for {symbol}"}

        target_position = None
        for p in positions:
            if p.ticket == ticket:
                target_position = p
                break

        if not target_position:
            return {"retcode": -3, "comment": f"Position ticket {ticket} not found"}

        close_volume = float(volume) if volume is not None else target_position.volume

        tick = MT5.symbol_info_tick(symbol)
        if not tick:
            return {"retcode": -4, "comment": "No tick data available"}

        close_type = MT5.ORDER_TYPE_SELL if target_position.type == MT5.POSITION_TYPE_BUY else MT5.ORDER_TYPE_BUY
        price = tick.bid if close_type == MT5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": MT5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_volume,
            "type": close_type,
            "price": price,
            "position": ticket,
            "deviation": 20,
            "magic": 0,
            "comment": f"close-ticket-{ticket}",
        }

        result = self._send_with_filling_retry(request)
        out = {"retcode": int(getattr(result, "retcode", -999)), "comment": str(getattr(result, "comment", ""))}
        if hasattr(result, "order"):
            out["order"] = int(result.order)
        if hasattr(result, "deal"):
            out["deal"] = int(result.deal)
        return out

    async def positions_stream(self, poll_seconds: float = 5.0):
        """Async generator yielding current positions periodically."""
        while True:
            try:
                yield await self.positions_get()
            except Exception:
                yield []
            await asyncio.sleep(poll_seconds)

    async def update_trailing_stop(self, symbol: str, trail_points: float, min_step_points: float = 10.0) -> int:
        """
        Apply/adjust a trailing stop for all positions on symbol.
        trail_points: distance from current price in points.
        min_step_points: minimum improvement required before moving SL to avoid over-tightening.
        Returns count of updated positions.
        """
        if MT5 is None:
            return 0
        positions = MT5.positions_get(symbol=symbol)
        if not positions:
            return 0
        updated = 0
        tick = MT5.symbol_info_tick(symbol)
        if not tick:
            return 0
        bid = float(tick.bid)
        ask = float(tick.ask)
        for pos in positions:
            try:
                side = pos.type  # 0=buy, 1=sell
                ticket = pos.ticket
                current_sl = float(pos.sl) if pos.sl else None
                if side == MT5.POSITION_TYPE_BUY:
                    desired_sl = bid - trail_points * pos.point
                    if current_sl is None or desired_sl - current_sl >= min_step_points * pos.point:
                        req = {
                            "action": MT5.TRADE_ACTION_SLTP,
                            "symbol": symbol,
                            "position": ticket,
                            "sl": desired_sl,
                            "tp": pos.tp,
                        }
                        result = MT5.order_send(req)
                        if result and result.retcode == MT5.TRADE_RETCODE_DONE:
                            updated += 1
                else:
                    desired_sl = ask + trail_points * pos.point
                    if current_sl is None or current_sl - desired_sl >= min_step_points * pos.point:
                        req = {
                            "action": MT5.TRADE_ACTION_SLTP,
                            "symbol": symbol,
                            "position": ticket,
                            "sl": desired_sl,
                            "tp": pos.tp,
                        }
                        result = MT5.order_send(req)
                        if result and result.retcode == MT5.TRADE_RETCODE_DONE:
                            updated += 1
            except Exception:
                continue
        return updated

    async def get_history_deals(self, from_timestamp: int, to_timestamp: int) -> list[dict[str, Any]]:
        """Get historical deals (closed trades) within time range"""
        if MT5 is None:
            return []

        try:
            deals = MT5.history_deals_get(from_timestamp, to_timestamp)
            out: list[dict[str, Any]] = []
            if deals:
                for d in deals:
                    out.append(d._asdict())
            return out
        except Exception:
            return []

    async def get_symbol_price(self, symbol: str) -> dict[str, float]:
        if MT5 is None:
            return {}
        tick = MT5.symbol_info_tick(symbol)
        if not tick:
            return {}
        return {"bid": float(tick.bid), "ask": float(tick.ask)}

    async def _build_order_request(
        self, *, symbol: str, volume: float, order_type: str, sl: float | None, tp: float | None
    ) -> dict[str, Any]:
        if order_type not in {"buy", "sell"}:
            raise ValueError("order_type must be 'buy' or 'sell'")
        info = MT5.symbol_info(symbol)
        if not info or not info.visible:
            MT5.symbol_select(symbol, True)
        tick = MT5.symbol_info_tick(symbol)
        if not tick:
            raise RuntimeError("No tick available for order")

        if order_type == "buy":
            order_type_const = MT5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type_const = MT5.ORDER_TYPE_SELL
            price = tick.bid

        filling_mode = self._get_filling_mode(symbol)

        req = {
            "action": MT5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type_const,
            "price": price,
            "deviation": 20,
            "magic": 0,
            "comment": "bot-trade",
            "type_time": MT5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        if sl is not None:
            req["sl"] = float(sl)
        if tp is not None:
            req["tp"] = float(tp)
        return req

    def _format_order_result(self, result: Any) -> dict[str, Any]:
        out = {
            "retcode": int(getattr(result, "retcode", -999)),
            "comment": str(getattr(result, "comment", "")),
            "order": int(getattr(result, "order", 0)) if hasattr(result, "order") else None,
            "deal": int(getattr(result, "deal", 0)) if hasattr(result, "deal") else None,
        }
        return out

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        if MT5 is None:
            return {}
        info = MT5.symbol_info(symbol)
        if not info:
            return {}
        return info._asdict()

    async def get_rates(self, symbol: str, timeframe: str, count: int = 2000) -> pd.DataFrame | None:
        if MT5 is None:
            return None
        tf_const = self._tf_map.get(timeframe)
        if tf_const is None:
            return None
        rates = MT5.copy_rates_from_pos(symbol, tf_const, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    async def create_market_buy_order(
        self,
        symbol: str,
        volume: float,
        *,
        sl: float | None = None,
        tp: float | None = None,
        magic: int = 0,
        comment: str = "bot-trade",
    ) -> dict[str, Any]:
        return await self._send_order(symbol, volume, order_type="buy", sl=sl, tp=tp, magic=magic, comment=comment)

    async def create_market_sell_order(
        self,
        symbol: str,
        volume: float,
        *,
        sl: float | None = None,
        tp: float | None = None,
        magic: int = 0,
        comment: str = "bot-trade",
    ) -> dict[str, Any]:
        return await self._send_order(symbol, volume, order_type="sell", sl=sl, tp=tp, magic=magic, comment=comment)

    async def adjust_position_sl_tp(
        self, symbol: str, stop_loss: float | None = None, take_profit: float | None = None
    ) -> bool:
        if MT5 is None:
            return False
        positions = MT5.positions_get(symbol=symbol)
        if not positions:
            return False
        ok_any = False
        for p in positions:
            req = {
                "action": MT5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": p.ticket,
            }
            if stop_loss is not None:
                req["sl"] = float(stop_loss)
            if take_profit is not None:
                req["tp"] = float(take_profit)
            result = MT5.order_send(req)
            if result and result.retcode == MT5.TRADE_RETCODE_DONE:
                ok_any = True
        return ok_any

    async def _send_order(
        self,
        symbol: str,
        volume: float,
        *,
        order_type: str,
        sl: float | None,
        tp: float | None,
        magic: int = 0,
        comment: str = "auto-trade",
    ) -> dict[str, Any]:
        if MT5 is None:
            return {"retcode": -1, "comment": "MT5 unavailable"}
        info = MT5.symbol_info(symbol)
        if not info or not info.visible:
            MT5.symbol_select(symbol, True)
            info = MT5.symbol_info(symbol)  # Refresh info

        price = MT5.symbol_info_tick(symbol)
        if not price:
            return {"retcode": -2, "comment": "No tick"}

        if order_type == "buy":
            order_type_const = MT5.ORDER_TYPE_BUY
            req_price = price.ask
        else:
            order_type_const = MT5.ORDER_TYPE_SELL
            req_price = price.bid

        request = {
            "action": MT5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type_const,
            "price": req_price,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
        }
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)

        # Use the centralized filling mode retry logic
        result = self._send_with_filling_retry(request)

        out = {"retcode": int(getattr(result, "retcode", -999)), "comment": str(getattr(result, "comment", ""))}
        if hasattr(result, "order"):
            out["order"] = int(result.order)
        if hasattr(result, "deal"):
            out["deal"] = int(result.deal)
        return out

    async def call_raw(self, method: str, *args, **kwargs):
        if MT5 is None:
            return None
        if method == "symbol_info_tick":
            return MT5.symbol_info_tick(*args, **kwargs)
        return None


class MT5LocalClient:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.connection = _Connection(settings.system.symbol)

    async def connect(self) -> None:
        if MT5 is None:
            raise RuntimeError("MetaTrader5 package not available")
        path = self.settings.system.mt5_terminal_path
        if path:
            ok = MT5.initialize(path=path)
        else:
            ok = MT5.initialize()

        if not ok:
            raise RuntimeError(f"MT5 initialize failed: {MT5.last_error()}")

        login = self.settings.system.mt5_login
        password = self.settings.system.mt5_password
        server = self.settings.system.mt5_server

        if login and password and server:
            if not MT5.login(login, password=password, server=server):
                raise RuntimeError(f"MT5 login failed: {MT5.last_error()}")

        # Automate "Algo Trading" button check
        ensure_autotrading_enabled(MT5)

        await asyncio.sleep(0.5)

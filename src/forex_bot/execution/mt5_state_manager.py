"""
MT5 State Manager - Single Source of Truth
Ensures bot state is always synchronized with actual MT5 reality.
"""

import asyncio
import json
import logging
import time as time_module
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


class BoundedLRUFeatureStore:
    """
    LRU cache with bounded size for entry feature storage (2025 best practice).
    Prevents memory leaks by automatically evicting least recently used entries.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.store: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.access_times: dict[int, float] = {}

    def add(self, ticket: int, features: dict[str, Any]) -> None:
        """Add entry to store, evicting LRU if at capacity."""
        if ticket in self.store:
            # Move to end (most recently used)
            self.store.move_to_end(ticket)
        else:
            if len(self.store) >= self.max_size:
                # Remove least recently used (first item)
                lru_ticket = next(iter(self.store))
                del self.store[lru_ticket]
                self.access_times.pop(lru_ticket, None)
                logger.debug(f"Evicted LRU entry {lru_ticket} from feature store")

            self.store[ticket] = features

        self.access_times[ticket] = time_module.monotonic()

    def get(self, ticket: int) -> dict[str, Any] | None:
        """Get entry and mark as recently used."""
        if ticket in self.store:
            self.store.move_to_end(ticket)
            self.access_times[ticket] = time_module.monotonic()
            return self.store[ticket]
        return None

    def remove(self, ticket: int) -> None:
        """Remove entry from store."""
        self.store.pop(ticket, None)
        self.access_times.pop(ticket, None)

    def items(self):
        """Iterate over all items."""
        return self.store.items()

    def __len__(self):
        return len(self.store)

    def __contains__(self, ticket: int):
        return ticket in self.store

    def cleanup_stale(self, max_age_seconds: int = 86400) -> int:
        """Remove entries older than max_age. Returns count of removed entries."""
        now = time_module.monotonic()
        to_remove = [
            ticket for ticket, access_time in self.access_times.items() if (now - access_time) > max_age_seconds
        ]
        for ticket in to_remove:
            self.remove(ticket)
        return len(to_remove)


@dataclass(slots=True)
class MT5Position:
    """Represents an actual MT5 position"""

    ticket: int
    symbol: str
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    swap: float
    commission: float
    time: datetime
    type: int  # 0=buy, 1=sell
    magic: int


@dataclass(slots=True)
class MT5Deal:
    """Represents a closed deal/trade"""

    deal: int
    order: int
    time: datetime
    symbol: str
    type: int
    entry: int
    volume: float
    price: float
    profit: float
    commission: float
    swap: float
    magic: int  # added magic for tracking


@dataclass(slots=True)
class OrderRequest:
    """Deduplication tracking for order requests"""

    request_id: str
    symbol: str
    order_type: str  # 'buy' or 'sell'
    volume: float
    sl: float | None
    tp: float | None
    timestamp: datetime
    bar_timestamp: datetime  # Current bar timestamp
    result: dict[str, Any] | None = None
    verified: bool = False


class MT5StateManager:
    """
    Synchronizes bot state with MT5 reality.
    Prevents duplicate orders, tracks real PnL, validates margin.
    """

    def __init__(self, mt5_connection, settings):
        self.mt5 = mt5_connection
        self.settings = settings

        self.pending_requests: dict[str, OrderRequest] = {}
        self.completed_requests: deque = deque(maxlen=100)  # Last 100 orders
        self.last_order_time: datetime | None = None
        self.min_order_interval_seconds = 5

        self.cached_positions: list[MT5Position] = []
        self.cached_account_info: dict[str, Any] = {}
        self.cached_deals_today: list[MT5Deal] = []
        self.last_sync_time: datetime | None = None

        self.session_start = datetime.now(UTC)
        self.orders_placed_today = 0
        self.confirmed_trades_today = 0

        self.max_positions_per_symbol = 1
        self.max_retry_attempts = 2

        # Concurrency Lock
        self._order_lock = asyncio.Lock()

        self.consecutive_sync_failures = 0
        self.max_consecutive_sync_failures = 10  # Raise exception after 10 failures

        # Use bounded LRU cache to prevent memory leaks (2025 best practice)
        self.entry_feature_store = BoundedLRUFeatureStore(max_size=1000)
        self.symbol = settings.system.symbol or "GLOBAL"
        self._entry_store_path = Path("cache") / f"entry_features_{self.symbol}.json"
        self._load_entry_feature_store()

    @property
    def connection(self):
        """
        Backward-compatible alias for the underlying MT5 connection/client.

        Some parts of the codebase refer to `mt5_state_manager.connection` for low-level broker calls.
        """
        return self.mt5

    async def sync_with_mt5(self, symbol: str | None = None) -> bool:
        """
        HPC FIX: Atomic Symbol-Specific Sync.
        No caching. Every symbol gets fresh, time-aligned state.
        """
        try:
            # 1. Atomic Account Refresh (Must be fresh for margin checks)
            account_info = await self.mt5.get_account_information()
            if not account_info:
                return False
            self.cached_account_info = account_info

            # 2. Targeted Position Fetch (Faster than fetching all)
            # Filter at the source (MT5 Terminal side)
            positions_raw = await self.mt5.positions_get(symbol=symbol) if symbol else await self.mt5.positions_get()
            
            new_positions = []
            for p in (positions_raw or []):
                # ... (Position parsing logic remains same but data is fresh)
                try:
                    ticket = p.get("ticket", 0)
                    if not ticket: continue
                    
                    new_positions.append(
                        MT5Position(
                            ticket=ticket,
                            symbol=p.get("symbol", ""),
                            volume=p.get("volume", 0.0),
                            price_open=p.get("price_open", 0.0),
                            price_current=p.get("price_current", 0.0),
                            sl=p.get("sl", 0.0),
                            tp=p.get("tp", 0.0),
                            profit=p.get("profit", 0.0),
                            swap=p.get("swap", 0.0),
                            commission=p.get("commission", 0.0),
                            time=datetime.fromtimestamp(p.get("time", 0), tz=UTC),
                            type=p.get("type", 0),
                            magic=p.get("magic", 0),
                        )
                    )
                except Exception:
                    continue
            
            # HPC FIX: Symbol-Isolation
            # Only update the positions for the active symbol to avoid clearing others
            if symbol:
                self.cached_positions = [pos for pos in self.cached_positions if pos.symbol != symbol]
                self.cached_positions.extend(new_positions)
            else:
                self.cached_positions = new_positions

            self.last_sync_time = datetime.now(UTC)
            return True
        except Exception as e:
            self.consecutive_sync_failures += 1
            logger.error(f"MT5 sync failed (attempt {self.consecutive_sync_failures}): {e}")
            if self.consecutive_sync_failures >= self.max_consecutive_sync_failures:
                raise RuntimeError("MT5 sync failed too many times.") from e
            return False

    async def _get_history_deals(self, from_date: datetime, to_date: datetime) -> list[MT5Deal]:
        """Fetch historical deals (closed trades) from MT5"""
        deals = []
        try:
            from_ts = int(from_date.timestamp())
            to_ts = int(to_date.timestamp())

            deals_raw = await self.mt5.get_history_deals(from_ts, to_ts)

            for d in deals_raw:
                try:
                    deal_id = d.get("deal", 0)
                    timestamp = d.get("time", 0)

                    if not deal_id or deal_id <= 0:
                        logger.debug(f"Invalid deal ID: {deal_id} - skipping")
                        continue

                    if not timestamp or timestamp <= 0:
                        logger.debug(f"Invalid deal timestamp for deal {deal_id} - skipping")
                        continue

                    deals.append(
                        MT5Deal(
                            deal=deal_id,
                            order=d.get("order", 0),
                            time=datetime.fromtimestamp(timestamp, tz=UTC),
                            symbol=d.get("symbol", ""),
                            type=d.get("type", 0),
                            entry=d.get("entry", 0),
                            volume=d.get("volume", 0.0),
                            price=d.get("price", 0.0),
                            profit=d.get("profit", 0.0),
                            commission=d.get("commission", 0.0),
                            swap=d.get("swap", 0.0),
                            magic=d.get("magic", 0),
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to parse deal: {e}", exc_info=True)
                    continue

        except Exception as e:
            logger.warning(f"Failed to fetch history deals: {e}")
        return deals

    def get_real_equity(self) -> float:
        """Get real equity from MT5. Raises RuntimeError if unavailable."""
        # Check if account info has been synced at least once
        if not self.cached_account_info:
            logger.warning("get_real_equity() called before first MT5 sync. Triggering sync now...")
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We cannot sync synchronously here safely.
                    raise RuntimeError("MT5 sync required but event loop is running. Call sync_with_mt5() first.")
                else:
                    success = loop.run_until_complete(self.sync_with_mt5())
                    if not success:
                        raise RuntimeError("Emergency MT5 sync failed.")
            except Exception as sync_exc:
                raise RuntimeError(f"Critical: MT5 equity unavailable: {sync_exc}") from sync_exc

        equity = self.cached_account_info.get("equity")
        if equity is None or equity <= 0:
            raise RuntimeError(
                f"CRITICAL: MT5 Equity Unavailable (got: {equity}). "
                "Cannot calculate position risk safely. Halting execution."
            )
        return float(equity)

    def get_real_balance(self) -> float:
        """Get real balance from MT5"""
        balance = self.cached_account_info.get("balance")
        if balance is None:
            return self.get_real_equity()
        return float(balance)

    def get_real_margin_free(self) -> float:
        """Get available margin from MT5"""
        margin_free = self.cached_account_info.get("margin_free", 0)
        return float(margin_free)

    def get_positions_for_symbol(self, symbol: str) -> list[MT5Position]:
        """Get all open positions for a specific symbol"""
        return [p for p in self.cached_positions if p.symbol == symbol]

    def get_total_position_count(self) -> int:
        """Total number of open positions across all symbols"""
        return len(self.cached_positions)

    async def get_recent_closed_deals(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recently closed deals for online learning feedback.

        Returns list of deal dictionaries with profit/loss information.
        """
        to_date = datetime.now(UTC)
        from_date = to_date - timedelta(hours=1)

        try:
            deals_list = await self._get_history_deals(from_date, to_date)

            result = []
            for deal in deals_list[-limit:]:
                result.append(
                    {
                        "deal": deal.deal,
                        "order": deal.order,
                        "magic": deal.magic,
                        "time": deal.time,
                        "symbol": deal.symbol,
                        "profit": deal.profit,
                        "volume": deal.volume,
                        "price": deal.price,
                    }
                )

            return result
        except Exception as e:
            logger.debug(f"Failed to get recent deals: {e}")
            return []

    async def get_recently_closed_positions(self, limit: int = 10):
        """
        Compatibility shim for the trading loop's online-learning hook.

        Returns a list of simple objects with `.ticket`, `.profit`, `.volume`, etc.
        """
        to_date = datetime.now(UTC)
        from_date = to_date - timedelta(hours=1)
        try:
            deals_list = await self._get_history_deals(from_date, to_date)
        except Exception:
            return []

        out = []
        for deal in deals_list[-limit:]:
            try:
                ticket = int(deal.order or deal.deal or 0)
                if ticket <= 0:
                    continue
                out.append(
                    SimpleNamespace(
                        ticket=ticket,
                        profit=float(deal.profit),
                        volume=float(deal.volume),
                        symbol=str(deal.symbol),
                        time=deal.time,
                        magic=int(deal.magic),
                    )
                )
            except Exception:
                continue
        return out

    async def get_recent_closed_with_features(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Return recent closed deals joined with stored entry features (if available).
        Only returns samples that can be matched to stored entry features.
        """
        deals = await self.get_recent_closed_deals(limit=limit)
        matched = []
        if not deals:
            return matched

        self._cleanup_entry_store()

        for d in deals:
            ticket_candidates = []
            order_id = d.get("order")
            deal_id = d.get("deal")
            magic_id = d.get("magic")

            if order_id:
                ticket_candidates.append(order_id)
            if deal_id:
                ticket_candidates.append(deal_id)
            if magic_id:
                ticket_candidates.append(magic_id)

            payload = None
            for tid in ticket_candidates:
                if tid in self.entry_feature_store:
                    payload = self.entry_feature_store.get(tid)  # LRU cache get method
                    break

            if payload is None and len(self.entry_feature_store) < 50:
                for _tid, p in self.entry_feature_store.items():
                    if p["symbol"] == d["symbol"]:
                        entry_time = p.get("bar_time")
                        if isinstance(entry_time, str):
                            entry_time = datetime.fromisoformat(entry_time)

                        if entry_time and d["time"] > entry_time:
                            payload = p
                            break

            if payload is None:
                continue

            matched.append(
                {
                    "profit": d.get("profit", 0.0),
                    "volume": d.get("volume", 0.0),
                    "symbol": d.get("symbol"),
                    "time": d.get("time"),
                    "features": payload.get("features"),
                    "signal": payload.get("signal", 0),
                }
            )

        return matched

    def _cleanup_entry_store(self):
        """Remove entries older than 24 hours to keep store clean. Persists only if changes made."""
        now = datetime.now(UTC)
        keys_to_remove = []
        for k, v in self.entry_feature_store.items():
            try:
                # Cache bar_time to avoid multiple .get() calls
                bt = v.get("bar_time")
                if isinstance(bt, str):
                    bt = datetime.fromisoformat(bt)
                if bt and (now - bt).total_seconds() > 86400:
                    keys_to_remove.append(k)
            except Exception as cleanup_exc:
                logger.debug(f"Error checking entry {k} for cleanup: {cleanup_exc}")
                keys_to_remove.append(k)  # Remove corrupted entries

        if not keys_to_remove:
            return  # No cleanup needed, skip I/O

        for k in keys_to_remove:
            self.entry_feature_store.remove(k)  # LRU cache remove method

        logger.debug(f"Cleaned up {len(keys_to_remove)} stale entries from feature store")
        self._persist_entry_feature_store()

    def _persist_entry_feature_store(self) -> None:
        """Persist feature store to disk in a separate thread to avoid blocking the loop."""
        try:

            async def _async_write():
                def _write():
                    try:
                        self._entry_store_path.parent.mkdir(parents=True, exist_ok=True)
                        serializable = {}
                        for k, v in self.entry_feature_store.items():
                            try:
                                bar_time = v.get("bar_time")
                                bar_time_str = bar_time.isoformat() if isinstance(bar_time, datetime) else bar_time
                                serializable[int(k)] = {
                                    "symbol": v.get("symbol"),
                                    "bar_time": bar_time_str,
                                    "features": v.get("features"),
                                    "signal": v.get("signal"),
                                    "magic": v.get("magic"),
                                }
                            except Exception:
                                continue
                        self._entry_store_path.write_text(json.dumps(serializable))
                    except Exception as e:
                        logger.warning(f"Background persistence failed: {e}")

                try:
                    await asyncio.to_thread(_write)
                except Exception as exc:
                    logger.debug(f"Async persistence failed: {exc}")

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(_async_write())
            else:
                loop.create_task(_async_write())

        except Exception as exc:
            logger.debug(f"Failed to schedule persistence: {exc}")

    def _load_entry_feature_store(self) -> None:
        try:
            if not self._entry_store_path.exists():
                return
            raw = json.loads(self._entry_store_path.read_text())
            restored = {}
            for k, v in raw.items():
                try:
                    bar_time = None
                    if v.get("bar_time"):
                        bar_time = datetime.fromisoformat(v["bar_time"])
                    restored[int(k)] = {
                        "symbol": v.get("symbol"),
                        "bar_time": bar_time,
                        "features": v.get("features"),
                        "signal": v.get("signal"),
                        "magic": v.get("magic"),
                    }
                except Exception:
                    continue
            # Load into LRU store
            for k, v in restored.items():
                self.entry_feature_store.add(k, v)
            logger.info(f"Loaded {len(restored)} entries into feature store")
        except Exception as exc:
            logger.debug(f"Failed to load entry feature store: {exc}")

    def record_entry_features(
        self,
        ticket: int,
        symbol: str,
        bar_time: datetime,
        features: Any,
        signal: int,
        order_ticket: int | None = None,
        deal_ticket: int | None = None,
        magic: int | None = None,
    ) -> None:
        """
        Store entry-time feature vector for online learning feedback.
        Features should be a single-row DataFrame or Series.
        """
        try:
            key_ids = [ticket]
            if order_ticket:
                key_ids.append(order_ticket)
            if deal_ticket:
                key_ids.append(deal_ticket)
            if magic:
                key_ids.append(magic)
            payload = {
                "symbol": symbol,
                "bar_time": bar_time,
                "features": features,  # can be JSON
                "signal": signal,
                "magic": magic,
            }
            for kid in key_ids:
                self.entry_feature_store.add(int(kid), payload)  # LRU cache add method
            self._persist_entry_feature_store()
        except Exception as exc:
            logger.debug(f"Failed to record entry features for ticket={ticket}: {exc}")

    def calculate_real_daily_pnl(self) -> tuple[float, float]:
        """
        Calculate real daily PnL from MT5.
        Returns: (realized_pnl, unrealized_pnl)
        """
        realized = sum(d.profit + d.commission + d.swap for d in self.cached_deals_today)

        unrealized = sum(p.profit + p.commission + p.swap for p in self.cached_positions)

        return realized, unrealized

    async def can_place_order(self, symbol: str, volume: float, current_bar_time: datetime) -> tuple[bool, str]:
        """
        Pre-flight checks before placing order.
        Checks MT5 reality, not just bot's internal state.
        """
        now = datetime.now(UTC)

        existing_positions = self.get_positions_for_symbol(symbol)
        if len(existing_positions) >= self.max_positions_per_symbol:
            return False, f"Already have {len(existing_positions)} position(s) on {symbol}"

        for req_id, req in self.pending_requests.items():
            if req.symbol == symbol and req.bar_timestamp == current_bar_time:
                return False, f"Order already pending for this bar (req_id={req_id[:8]})"

        for req in list(self.completed_requests):
            if req.symbol == symbol and req.bar_timestamp == current_bar_time and req.verified:
                return False, "Order already placed and verified for this bar"

        if self.last_order_time:
            seconds_since_last = (now - self.last_order_time).total_seconds()
            if seconds_since_last < self.min_order_interval_seconds:
                return (
                    False,
                    f"Too soon since last order ({seconds_since_last:.1f}s < {self.min_order_interval_seconds}s)",
                )

        if not self._has_sufficient_margin(volume):
            return False, "Insufficient margin available"

        if not await self._validate_lot_size(symbol, volume):
            return False, f"Invalid lot size {volume} for broker limits"

        return True, "OK"

    def _has_sufficient_margin(self, lot_size: float) -> bool:
        """Check if we have enough free margin"""
        margin_free = self.get_real_margin_free()

        estimated_margin_needed = lot_size * 1000  # Conservative estimate

        if margin_free < estimated_margin_needed:
            logger.warning(f"Margin check: need ~${estimated_margin_needed:.0f}, have ${margin_free:.0f}")
            return False

        return True

    async def _validate_lot_size(self, symbol: str, volume: float) -> bool:
        """Validate lot size against broker's min/max limits"""
        try:
            symbol_info = await self.mt5.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Could not get symbol info for {symbol} - VALIDATION FAILED")
                return False  # FIXED: Don't assume valid!

            volume_min = symbol_info.get("volume_min", 0.01)
            volume_max = symbol_info.get("volume_max", 100.0)
            volume_step = symbol_info.get("volume_step", 0.01)

            if volume < volume_min:
                logger.error(f"Lot size {volume} below broker min {volume_min}")
                return False

            if volume > volume_max:
                logger.error(f"Lot size {volume} above broker max {volume_max}")
                return False

            if volume_step > 0:
                # Robust float alignment check
                steps = round(volume / volume_step)
                aligned_volume = steps * volume_step
                if abs(volume - aligned_volume) > 0.0001:
                    logger.error(
                        "Lot size %s not aligned with step %s (nearest: %.5f)",
                        volume,
                        volume_step,
                        aligned_volume,
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Lot size validation exception: {e} - VALIDATION FAILED", exc_info=True)
            return False  # FIXED: Fail safely, don't assume valid!

    async def place_order_with_verification(
        self,
        symbol: str,
        order_type: str,  # 'buy' or 'sell'
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
        current_bar_time: datetime | None = None,
        magic: int = 234567,  # Unique magic number for this bot
    ) -> dict[str, Any]:
        """
        Place order with full verification and deduplication.
        Returns: {'success': bool, 'reason': str, 'ticket': int, 'position': MT5Position}
        """
        async with self._order_lock:
            now = datetime.now(UTC)
            bar_time = current_bar_time or now

            request_id = f"{symbol}_{order_type}_{bar_time.timestamp()}_{volume}"

            order_req = OrderRequest(
                request_id=request_id,
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                sl=sl,
                tp=tp,
                timestamp=now,
                bar_timestamp=bar_time,
            )

            # Re-check inside lock to prevent race condition
            can_place, reason = await self.can_place_order(symbol, volume, bar_time)
            if not can_place:
                logger.warning(f"Order rejected (in lock): {reason}")
                return {"success": False, "reason": reason, "ticket": None}

            self.pending_requests[request_id] = order_req

            try:
                logger.info(f"Placing {order_type.upper()} order: {volume} lots {symbol} (SL={sl}, TP={tp})")

                if order_type == "buy":
                    result = await self.mt5.create_market_buy_order(
                        symbol, volume, sl=sl, tp=tp, magic=magic, comment="bot-entry"
                    )
                elif order_type == "sell":
                    result = await self.mt5.create_market_sell_order(
                        symbol, volume, sl=sl, tp=tp, magic=magic, comment="bot-entry"
                    )
                else:
                    raise ValueError(f"Invalid order type: {order_type}")

                order_req.result = result

                retcode = result.get("retcode", -999)

                if retcode != 10009:
                    comment = result.get("comment", "Unknown error")
                    logger.error(f"Order failed: retcode={retcode}, comment={comment}")

                    self.pending_requests.pop(request_id, None)
                    self.completed_requests.append(order_req)

                    return {"success": False, "reason": f"MT5 error: {comment} (retcode={retcode})", "ticket": None}

                logger.info("Order sent successfully, verifying position...")
                await asyncio.sleep(0.5)  # Brief delay for settlement

                await self.sync_with_mt5()

                deal_ticket = result.get("deal")
                order_ticket = result.get("order")

                found_position = None
                for pos in self.cached_positions:
                    if deal_ticket and pos.ticket == deal_ticket:
                        found_position = pos
                        break
                    if (
                        pos.symbol == symbol
                        and abs(pos.volume - volume) < 0.01
                        and (now - pos.time).total_seconds() < 10
                    ):
                        found_position = pos
                        break

                if found_position:
                    logger.info(
                        f"Position verified in MT5: ticket={found_position.ticket}, profit={found_position.profit:.2f}"
                    )

                    order_req.verified = True
                    self.pending_requests.pop(request_id, None)
                    self.completed_requests.append(order_req)
                    self.last_order_time = now
                    self.orders_placed_today += 1
                    self.confirmed_trades_today += 1

                    return {
                        "success": True,
                        "reason": "Order placed and verified",
                        "ticket": found_position.ticket,
                        "position": found_position,
                        "deal_ticket": deal_ticket,
                        "order_ticket": order_ticket,
                        "bar_time": bar_time,
                        "magic": magic,
                    }
                else:
                    logger.error(
                        f"CRITICAL: Order retcode=DONE but position not found in MT5! "
                        f"deal={deal_ticket}, order={order_ticket}"
                    )

                    order_req.verified = False
                    self.pending_requests.pop(request_id, None)
                    self.completed_requests.append(order_req)

                    return {
                        "success": False,
                        "reason": "Order succeeded but position not confirmed in MT5 (possible MT5 issue)",
                        "ticket": deal_ticket,
                        "requires_manual_check": True,
                    }

            except Exception as e:
                logger.error(f"Order placement exception: {e}", exc_info=True)

                self.pending_requests.pop(request_id, None)
                order_req.verified = False
                self.completed_requests.append(order_req)

                return {"success": False, "reason": f"Exception: {str(e)}", "ticket": None}

    async def close_position_by_ticket(self, ticket: int, symbol: str, volume: float | None = None) -> bool:
        """Close a specific position by ticket ID. Supports partial close if volume is specified."""
        try:
            logger.info(f"Closing position ticket={ticket} on {symbol} (vol={volume if volume else 'ALL'})")

            positions = self.get_positions_for_symbol(symbol)
            target_pos = None
            for p in positions:
                if p.ticket == ticket:
                    target_pos = p
                    break

            if not target_pos:
                logger.warning(f"Position ticket={ticket} not found")
                return False

            result = await self.mt5.close_position_by_ticket(ticket, symbol, volume=volume)

            success = False
            if isinstance(result, dict):
                if result.get("retcode") == 10009:  # TRADE_RETCODE_DONE
                    success = True
                else:
                    logger.error(f"Close failed: {result.get('comment')} (retcode={result.get('retcode')})")
            elif isinstance(result, int) and result > 0:  # Legacy count return
                success = True

            if success:
                logger.info(f"Successfully closed position {ticket}")
                await self.sync_with_mt5()
                return True
            else:
                logger.error(f"Failed to close position {ticket}")
                return False

        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False

    async def partial_close(self, ticket: int, symbol: str, volume: float) -> bool:
        """Partially close a position."""
        return await self.close_position_by_ticket(ticket, symbol, volume=volume)

    def get_todays_trade_count(self) -> int:
        """Get actual trade count from MT5, not bot's counter"""
        return len(self.cached_deals_today)

    def get_health_status(self) -> dict[str, Any]:
        """Get health check status"""
        now = datetime.now(UTC)
        seconds_since_sync = (now - self.last_sync_time).total_seconds() if self.last_sync_time else 999

        return {
            "last_sync": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "seconds_since_sync": seconds_since_sync,
            "sync_healthy": seconds_since_sync < 120,  # Should sync at least every 2 min
            "positions_open": len(self.cached_positions),
            "deals_today": len(self.cached_deals_today),
            "orders_placed_today": self.orders_placed_today,
            "confirmed_trades": self.confirmed_trades_today,
            "pending_requests": len(self.pending_requests),
            "equity": self.get_real_equity(),
            "free_margin": self.get_real_margin_free(),
        }

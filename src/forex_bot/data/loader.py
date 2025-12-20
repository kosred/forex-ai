import asyncio
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as MT5  # noqa: N814
except ImportError:
    MT5 = None

from ..core.config import Settings

logger = logging.getLogger(__name__)

MANDATORY_TFS = ["M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1", "MN1"]

_MT5_CONNECT_LOCK = asyncio.Lock()
_MT5_GLOBAL_CONNECTED = False
_MT5_GLOBAL_REFCOUNT = 0


def _format_mt5_error(err: object) -> str:
    if isinstance(err, tuple) and len(err) >= 2:
        return f"({err[0]}, {err[1]!r})"
    return repr(err)


def _guess_mt5_terminal_path() -> Path | None:
    """
    Best-effort MT5 terminal discovery for Windows.

    MetaTrader5.initialize() can IPC-timeout if no terminal is running; providing an
    explicit terminal path can help the MT5 Python package locate/start the terminal.
    """
    try:
        candidates: list[Path] = []
        for env_key in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
            base = os.environ.get(env_key)
            if not base:
                continue
            root = Path(base)
            candidates.append(root / "MetaTrader 5" / "terminal64.exe")
            candidates.append(root / "MetaTrader 5" / "terminal.exe")
            try:
                for child in root.glob("MetaTrader*"):
                    candidates.append(child / "terminal64.exe")
                    candidates.append(child / "terminal.exe")
            except Exception:
                continue

        for p in candidates:
            try:
                if p.exists() and p.is_file():
                    return p
            except Exception:
                continue
    except Exception:
        return None
    return None


class MT5Adapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.connection = self._Connection()

    class _Connection:
        async def get_account_information(self) -> dict:
            if MT5 is None:
                return {}
            info = MT5.account_info()
            return info._asdict() if info else {}

        async def positions_get(self) -> list[dict]:
            if MT5 is None:
                return []
            positions = MT5.positions_get()
            return [p._asdict() for p in positions] if positions else []

        async def close_position(self, symbol: str) -> None:
            if MT5 is None:
                return
            positions = MT5.positions_get(symbol=symbol)
            if positions:
                for p in positions:
                    req = {
                        "action": MT5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": p.volume,
                        "type": MT5.ORDER_TYPE_SELL if p.type == MT5.POSITION_TYPE_BUY else MT5.ORDER_TYPE_BUY,
                        "position": p.ticket,
                        "magic": 0,
                        "comment": "python script close",
                        "type_time": MT5.ORDER_TIME_GTC,
                        "type_filling": MT5.ORDER_FILLING_IOC,
                    }
                    MT5.order_send(req)

        async def close_position_by_ticket(self, ticket: int, symbol: str, volume: float | None = None) -> dict:
            if MT5 is None:
                return {"retcode": -1, "comment": "MT5 unavailable"}
            try:
                # Prefer the native MT5 close API when available
                positions = MT5.positions_get(symbol=symbol)
                target = None
                if positions:
                    for p in positions:
                        if int(getattr(p, "ticket", 0)) == int(ticket):
                            target = p
                            break
                if not target:
                    return {"retcode": -2, "comment": f"Ticket {ticket} not found"}

                req = {
                    "action": MT5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": float(volume if volume is not None else target.volume),
                    "type": MT5.ORDER_TYPE_SELL if target.type == MT5.POSITION_TYPE_BUY else MT5.ORDER_TYPE_BUY,
                    "position": ticket,
                    "magic": 0,
                    "comment": "bot-close",
                    "type_time": MT5.ORDER_TIME_GTC,
                    "type_filling": MT5.ORDER_FILLING_IOC,
                }
                res = MT5.order_send(req)
                return res._asdict() if res else {"retcode": -1, "comment": "order_send returned None"}
            except Exception as exc:
                return {"retcode": -1, "comment": f"close failed: {exc}"}

        async def get_symbol_price(self, symbol: str) -> dict:
            if MT5 is None:
                return {}
            tick = MT5.symbol_info_tick(symbol)
            return {"bid": tick.bid, "ask": tick.ask} if tick else {}

        async def history_deals_total(self, from_timestamp: int, to_timestamp: int) -> int:
            """Return count of deals in range (consistent with MetaTrader5 API)."""
            if MT5 is None:
                return 0
            try:
                return MT5.history_deals_total(from_timestamp, to_timestamp) or 0
            except Exception:
                return 0

        async def get_symbol_info(self, symbol: str) -> dict:
            if MT5 is None:
                return {}
            info = MT5.symbol_info(symbol)
            return info._asdict() if info else {}

        async def create_market_buy_order(
            self, symbol: str, volume: float, sl: float | None = None, tp: float | None = None, **kwargs
        ) -> dict:
            if MT5 is None:
                return {"retcode": -1, "comment": "MT5 unavailable"}
            try:
                req = {
                    "action": MT5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": MT5.ORDER_TYPE_BUY,
                    "sl": sl,
                    "tp": tp,
                    "type_time": MT5.ORDER_TIME_GTC,
                    "type_filling": MT5.ORDER_FILLING_IOC,
                }
                if "magic" in kwargs:
                    req["magic"] = kwargs.get("magic")
                if "comment" in kwargs:
                    req["comment"] = kwargs.get("comment")
                res = MT5.order_send(req)
                return res._asdict() if res else {"retcode": -1, "comment": "order_send returned None"}
            except Exception as exc:
                return {"retcode": -1, "comment": f"buy order failed: {exc}"}

        async def create_market_sell_order(
            self, symbol: str, volume: float, sl: float | None = None, tp: float | None = None, **kwargs
        ) -> dict:
            if MT5 is None:
                return {"retcode": -1, "comment": "MT5 unavailable"}
            try:
                req = {
                    "action": MT5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": MT5.ORDER_TYPE_SELL,
                    "sl": sl,
                    "tp": tp,
                    "type_time": MT5.ORDER_TIME_GTC,
                    "type_filling": MT5.ORDER_FILLING_IOC,
                }
                if "magic" in kwargs:
                    req["magic"] = kwargs.get("magic")
                if "comment" in kwargs:
                    req["comment"] = kwargs.get("comment")
                res = MT5.order_send(req)
                return res._asdict() if res else {"retcode": -1, "comment": "order_send returned None"}
            except Exception as exc:
                return {"retcode": -1, "comment": f"sell order failed: {exc}"}

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
                    point = pos.point

                    if side == MT5.POSITION_TYPE_BUY:
                        desired_sl = bid - trail_points * point
                        if current_sl is None or desired_sl - current_sl >= min_step_points * point:
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
                        desired_sl = ask + trail_points * point
                        if current_sl is None or current_sl - desired_sl >= min_step_points * point:
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

        async def get_history_deals(self, from_timestamp: int, to_timestamp: int) -> list[dict]:
            """Fetch historical deals using MetaTrader5.history_deals_get"""
            if MT5 is None:
                return []
            try:
                deals = MT5.history_deals_get(from_timestamp, to_timestamp)
                if not deals:
                    return []
                return [d._asdict() for d in deals]
            except Exception:
                logger.warning("history_deals_get failed", exc_info=True)
                return []


class DataLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = Path(settings.system.data_dir)
        self.mt5_adapter = MT5Adapter(settings)
        self._connected = False
        self._mt5_shared = False
        self.required_tfs = list(dict.fromkeys(MANDATORY_TFS + self.settings.system.required_timeframes))
        # Optional in-memory cache for training frames; guarded by size limit to avoid OOM.
        self.cache_training_frames = bool(getattr(settings.system, "cache_training_frames", False))
        self.training_cache_max_bytes = int(getattr(settings.system, "training_cache_max_bytes", 2_000_000_000))
        self._training_frames_cache: dict[str, dict[str, pd.DataFrame]] = {}

    async def connect(self) -> bool:
        if not self.settings.system.mt5_required:
            logger.info("MT5 not required (training-only mode). Skipping connection.")
            self._connected = False
            return True  # Not an error, just training mode

        if MT5 is None:
            logger.warning("MetaTrader5 module not found. Live trading/download disabled.")
            return False

        if self._connected:
            return True

        global _MT5_GLOBAL_CONNECTED
        global _MT5_GLOBAL_REFCOUNT

        async with _MT5_CONNECT_LOCK:
            if _MT5_GLOBAL_CONNECTED:
                _MT5_GLOBAL_REFCOUNT += 1
                self._connected = True
                self._mt5_shared = True
                return True

            terminal_path = str(getattr(self.settings.system, "mt5_terminal_path", "") or "").strip()
            if terminal_path and not Path(terminal_path).exists():
                logger.warning(f"Configured mt5_terminal_path not found: {terminal_path!r}")
                terminal_path = ""

            if not terminal_path and os.name == "nt":
                guessed = _guess_mt5_terminal_path()
                if guessed is not None:
                    terminal_path = str(guessed)
                    try:
                        self.settings.system.mt5_terminal_path = terminal_path
                    except Exception:
                        pass
                    logger.info(f"Auto-detected MT5 terminal path: {terminal_path}")

            attempts = 3
            last_err: object = None
            launched_terminal = False
            for attempt in range(1, attempts + 1):
                try:
                    if attempt > 1:
                        try:
                            MT5.shutdown()
                        except Exception:
                            pass
                        await asyncio.sleep(0.5 * attempt)

                    ok = MT5.initialize(terminal_path) if terminal_path else MT5.initialize()
                except Exception as exc:
                    ok = False
                    err = f"initialize raised: {exc!r}"
                else:
                    err = MT5.last_error()
                last_err = err

                if ok:
                    break

                err_str = _format_mt5_error(err)
                logger.error(f"MT5 initialization failed (attempt {attempt}/{attempts}): {err_str}")

                if isinstance(err, tuple) and len(err) >= 2:
                    code = int(err[0])
                    msg = str(err[1]).lower()
                    retryable = code == -10005 or "timeout" in msg
                else:
                    retryable = False

                if not retryable:
                    break

                # On Windows, an IPC timeout commonly means the terminal is not running or isn't reachable
                # from the current user/session. Best effort: launch the terminal once, then retry.
                if (
                    not launched_terminal
                    and os.name == "nt"
                    and terminal_path
                    and isinstance(err, tuple)
                    and len(err) >= 2
                    and int(err[0]) == -10005
                ):
                    try:
                        subprocess.Popen(
                            [terminal_path],
                            cwd=str(Path(terminal_path).parent),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        launched_terminal = True
                        logger.info("Launched MetaTrader 5 terminal. Retrying MT5 initialization...")
                        await asyncio.sleep(3.0)
                    except Exception as exc:
                        logger.info(f"Failed to launch MT5 terminal automatically: {exc!r}")

            if not ok:
                if isinstance(last_err, tuple) and len(last_err) >= 2 and int(last_err[0]) == -10005:
                    logger.error(
                        "MT5 IPC timeout. Ensure the MetaTrader 5 terminal is installed and running under the same "
                        "Windows user, logged in, and that `system.mt5_terminal_path` points to `terminal64.exe`."
                    )
                return False

        login = self.settings.system.mt5_login
        password = self.settings.system.mt5_password
        server = self.settings.system.mt5_server
        if login and password and server:
            authorized = MT5.login(login, password=password, server=server)
            if not authorized:
                logger.error(f"MT5 login failed: {_format_mt5_error(MT5.last_error())}")
                try:
                    MT5.shutdown()
                except Exception:
                    pass
                return False

            _MT5_GLOBAL_CONNECTED = True
            _MT5_GLOBAL_REFCOUNT = 1
            self._mt5_shared = True

        self._connected = True
        _MT5_GLOBAL_CONNECTED = True
        _MT5_GLOBAL_REFCOUNT = max(_MT5_GLOBAL_REFCOUNT, 1)
        self._mt5_shared = True

        try:
            ref_symbol = self.settings.system.symbol or "EURUSD"
            if not MT5.symbol_select(ref_symbol, True):
                infos = MT5.symbols_get()
                if infos and len(infos) > 0:
                    ref_symbol = infos[0].name
                else:
                    logger.error("No MT5 symbols available for broker offset detection")
                    return 0

            tick = MT5.symbol_info_tick(ref_symbol)
            if tick:
                server_ts = datetime.fromtimestamp(tick.time, tz=UTC)
                now_utc = datetime.now(UTC)
                diff_seconds = (server_ts - now_utc).total_seconds()
                offset_hours = round(diff_seconds / 3600)

                # Ignore clearly stale ticks that produce nonsense offsets
                if abs(offset_hours) > 12:
                    logger.warning(
                        "Skipping MT5 timezone auto-detect: server tick appears stale "
                        f"(Server {server_ts.strftime('%Y-%m-%d %H:%M')}, UTC {now_utc.strftime('%Y-%m-%d %H:%M')}, "
                        f"delta_hours={offset_hours})"
                    )
                elif self.settings.system.mt5_timezone_offset_hours == 0 and offset_hours != 0:
                    self.settings.system.mt5_timezone_offset_hours = offset_hours
                    logger.info(
                        f"Auto-detected Broker Timezone Offset: GMT{'+' if offset_hours >= 0 else ''}{offset_hours} "
                        f"(Server: {server_ts.strftime('%H:%M')}, UTC: {now_utc.strftime('%H:%M')})"
                    )
        except Exception as e:
            logger.warning(f"Timezone offset detection failed: {e}")

        return True

    def is_connected(self) -> bool:
        return self._connected

    async def disconnect(self) -> None:
        if not self._connected:
            return
        if MT5 is None:
            self._connected = False
            self._mt5_shared = False
            return

        global _MT5_GLOBAL_CONNECTED
        global _MT5_GLOBAL_REFCOUNT

        async with _MT5_CONNECT_LOCK:
            if self._connected and self._mt5_shared:
                self._connected = False
                self._mt5_shared = False
                _MT5_GLOBAL_REFCOUNT = max(0, int(_MT5_GLOBAL_REFCOUNT) - 1)
                if _MT5_GLOBAL_REFCOUNT == 0 and _MT5_GLOBAL_CONNECTED:
                    try:
                        MT5.shutdown()
                    except Exception as exc:
                        logger.warning(f"MT5 shutdown failed: {exc!r}")
                    _MT5_GLOBAL_CONNECTED = False
            else:
                self._connected = False
                self._mt5_shared = False

    def _adjust_to_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shift dataframe index from Broker Server Time to True UTC."""
        offset = self.settings.system.mt5_timezone_offset_hours
        if offset == 0:
            return df
        try:
            df.index = df.index - pd.Timedelta(hours=offset)
        except Exception as e:
            logger.warning(f"Failed to adjust timezone offset: {e}")
        return df

    async def prompt_symbol_selection(self, available: list[str]) -> list[str]:
        """
        Interactive prompt for selecting symbols to download when data is missing.

        Returns list of chosen symbols. If input invalid, returns empty.
        """
        logger.info("No local data found for requested symbols.")

        if available:
            logger.info("Available MT5 symbols (sample, enter numbers or names):")
            for i, name in enumerate(available[:50], 1):
                logger.info("  %2d. %s", i, name)

        logger.info("Enter selection (comma-separated numbers or names), or press Enter to cancel:")
        try:
            if sys.stdin.isatty():
                user_input = input("> ").strip()
            else:
                return []
        except EOFError:
            return []

        if not user_input:
            return []

        chosen: list[str] = []
        tokens = [tok.strip() for tok in user_input.split(",") if tok.strip()]
        for tok in tokens:
            if tok.isdigit() and available:
                idx = int(tok) - 1
                if 0 <= idx < len(available[:50]):
                    chosen.append(available[idx])
            else:
                chosen.append(tok.upper())

        dedup: list[str] = []
        seen = set()
        for c in chosen:
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        return dedup

    async def ensure_history(self, symbol: str, _recursion_depth: int = 0) -> bool:
        """
        Ensure all mandatory timeframes exist (M1, M3, M5, M15, M30, H1, H2, H4, D1, W1, MN1).

        1. If any TF missing, resample from M1 if available.
        2. If M1 missing, attempt MT5 download (default 10 years) then resample.
        3. Persist all TFs to data/symbol=SYMBOL/timeframe=TF/data.parquet.

        Args:
            symbol: Trading pair symbol
            _recursion_depth: Internal counter to prevent infinite loops (max 3)
        """
        if _recursion_depth > 3:
            logger.error(f"Max recursion depth ({_recursion_depth}) reached for ensure_history({symbol}). Aborting.")
            return False
        needed_tfs = self.required_tfs

        def _tf_exists(tf: str) -> bool:
            path = self.data_dir / f"symbol={symbol}" / f"timeframe={tf}"
            return path.exists() and any(path.glob("*.parquet"))

        missing_tfs = [tf for tf in needed_tfs if not _tf_exists(tf)]
        if not missing_tfs:
            logger.info(f"All required timeframes present for {symbol}.")
            return True

        logger.info(f"Timeframes missing for {symbol}: {missing_tfs}. Checking local M1...")

        m1_path = self.data_dir / f"symbol={symbol}" / "timeframe=M1" / "data.parquet"
        df_m1: pd.DataFrame | None = None
        if m1_path.exists():
            try:
                logger.info(f"Found local M1 data at {m1_path}. Loading for resampling...")
                df_m1 = pd.read_parquet(m1_path)
            except Exception as e:
                logger.warning(f"Failed to read local M1: {e}")

        if df_m1 is None or getattr(df_m1, "empty", True):
            logger.warning(f"No local M1 data for {symbol}. Attempting download from MT5...")

            if not self._connected:
                await self.connect()

            if self._connected:
                df_m1 = await self._download_m1_from_mt5(symbol)
                if df_m1 is not None and not df_m1.empty:
                    self._save_frame(df_m1, symbol, "M1")
                else:
                    logger.warning(f"Auto-download failed for {symbol}.")

            if df_m1 is None or df_m1.empty:
                # Interactive fallback
                if sys.stdin.isatty():
                    logger.warning("Data missing for %s and auto-download failed.", symbol)
                    if self._connected:
                        logger.info("Fetching available symbols from MT5...")
                        try:
                            info = MT5.symbols_get()
                            available = [getattr(s, "name", "") for s in info if getattr(s, "name", "")] if info else []
                            if available:
                                selection = await self.prompt_symbol_selection(available)
                                for chosen in selection:
                                    logger.info(f"User selected: {chosen}")
                                    df_m1 = await self._download_m1_from_mt5(chosen)
                                    if df_m1 is not None and not df_m1.empty:
                                        symbol = chosen  # Switch to user-selected symbol
                                        self._save_frame(df_m1, symbol, "M1")
                                        return await self.ensure_history(symbol, _recursion_depth=_recursion_depth + 1)
                                    else:
                                        logger.warning("Download failed for %s. Try another.", chosen)
                            else:
                                logger.warning("No symbols found in MT5.")
                        except Exception as e:
                            logger.error(f"Interactive selection error: {e}")
                    else:
                        logger.error("MT5 not connected. Cannot download.")
                else:
                    logger.error(f"Data missing for {symbol} and auto-download failed. (Headless mode - no prompt).")

            if df_m1 is None or df_m1.empty:
                logger.error(f"Download failed for {symbol}; cannot build required timeframes.")
                return False

        await self._resample_and_save(df_m1, symbol, needed_tfs)

        missing_after = [tf for tf in needed_tfs if not _tf_exists(tf)]
        if missing_after:
            logger.warning(f"Some timeframes still missing after resample for {symbol}: {missing_after}")
            return False

        return True

    async def _download_m1_from_mt5(self, symbol: str) -> pd.DataFrame | None:
        """Download M1 data from MT5."""
        if MT5 is None:
            return None
        if not MT5.symbol_select(symbol, True):
            logger.error(f"Symbol {symbol} not found in MT5.")
            return None

        years = 10
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        logger.info(f"Downloading M1 history for {symbol} from {start_date.date()}...")
        rates = MT5.copy_rates_range(symbol, MT5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            logger.error(f"MT5 download failed for {symbol}: {MT5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.drop(columns=["time", "spread", "real_volume"], inplace=True, errors="ignore")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df.set_index("timestamp", inplace=True)
        df["symbol"] = symbol
        df["timeframe"] = "M1"
        return df

    async def _resample_and_save(self, df_m1: pd.DataFrame, symbol: str, timeframes: list[str] | None = None) -> None:
        """Resample M1 df to all requested timeframes and save."""
        rule_map = {
            "M1": "1min",
            "M3": "3min",
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H2": "2h",
            "H4": "4h",
            "D1": "1D",
            "W1": "1W",
            "MN1": "1ME",  # '1M' is deprecated in newer pandas, '1ME' is Month End
        }
        tfs = timeframes or list(rule_map.keys())

        df = df_m1.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()

        for tf in tfs:
            rule = rule_map.get(tf)
            if rule is None:
                continue
            try:
                logger.info(f"Resampling {symbol} to {tf} ({rule})...")
                resampled = (
                    df.resample(rule)
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()
                )
                resampled["symbol"] = symbol
                resampled["timeframe"] = tf
                self._save_frame(resampled, symbol, tf)
            except Exception as e:
                logger.error(f"Resampling failed for {symbol} {tf}: {e}")

    def list_local_symbols(self) -> list[str]:
        """List symbols found in the data directory."""
        symbols = []
        for path in self.data_dir.glob("symbol=*"):
            if path.is_dir():
                parts = path.name.split("=", 1)
                if len(parts) >= 2:
                    sym = parts[-1]
                    symbols.append(sym)
                else:
                    logger.warning(f"Invalid symbol directory format: {path.name}")
        return symbols

    async def ensure_all_history(self, symbols: list[str] | None = None) -> bool:
        """
        Ensure all required timeframes for all symbols (local or provided).
        Returns True if all succeeded.
        """
        all_syms = symbols if symbols else self.list_local_symbols()
        ok = True
        for sym in all_syms:
            res = await self.ensure_history(sym)
            ok = ok and res
        return ok

    def _save_frame(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        path = self.data_dir / f"symbol={symbol}" / f"timeframe={timeframe}"
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "data.parquet"
        df.to_parquet(file_path)
        logger.info(f"Saved {timeframe} data to {file_path}")

    async def get_training_data(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Load all timeframes for a symbol."""
        if self.cache_training_frames:
            cached = self._training_frames_cache.get(symbol)
            if cached is not None:
                return cached

        frames: dict[str, pd.DataFrame] = {}
        tfs = [self.settings.system.base_timeframe] + self.settings.system.higher_timeframes + ["M1"]
        tfs = list(dict.fromkeys(tfs))  # dedupe while preserving order

        for tf in tfs:
            path = self.data_dir / f"symbol={symbol}" / f"timeframe={tf}"
            if not path.exists():
                continue
            files = sorted(path.glob("*.parquet"))
            if not files:
                continue
            try:
                dfs = [
                    pd.read_parquet(
                        fp,
                        engine="pyarrow",
                        memory_map=bool(getattr(self.settings.system, "parquet_memory_map", True)),
                    )
                    for fp in files
                ]
                if not dfs:
                    continue
                df = pd.concat(dfs, ignore_index=False)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    df.set_index("timestamp", inplace=True)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                df = df.sort_index()
                df = self._adjust_to_utc(df)
                max_rows = int(getattr(self.settings.system, "max_training_rows_per_tf", 0))
                if max_rows and len(df) > max_rows:
                    df = df.tail(max_rows)
                if getattr(self.settings.system, "downcast_training_float32", True):
                    df = self._downcast_float_columns(df)
                frames[tf] = df
            except Exception as e:
                logger.warning(f"Failed to load {tf} data: {e}")
        if self.cache_training_frames:
            total_bytes = self._estimate_frames_bytes(frames)
            if total_bytes <= self.training_cache_max_bytes:
                self._training_frames_cache[symbol] = frames
                logger.info(f"Cached training frames for {symbol} ({total_bytes / 1e6:.1f} MB)")
            else:
                logger.info(
                    f"Skipping cache for {symbol}: {total_bytes / 1e6:.1f} MB exceeds "
                    f"limit {self.training_cache_max_bytes / 1e6:.1f} MB"
                )
        return frames

    @staticmethod
    def _estimate_frames_bytes(frames: dict[str, pd.DataFrame]) -> int:
        total = 0
        for df in frames.values():
            try:
                total += int(df.memory_usage(deep=True).sum())
            except Exception:
                continue
        return total

    @staticmethod
    def _downcast_float_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast float columns to float32 to reduce memory footprint."""
        try:
            float_cols = df.select_dtypes(include=["float64", "float32"]).columns
            if float_cols.any():
                df[float_cols] = df[float_cols].astype("float32")
        except Exception as exc:
            logger.debug(f"Downcast failed, keeping original dtypes: {exc}")
        return df

    async def get_live_data(self, symbol: str) -> dict[str, pd.DataFrame] | None:
        """Get latest bars from MT5 for live trading."""
        if not self._connected:
            await self.connect()

        if not self._connected or MT5 is None:
            return None

        lookback = 500  # Enough for indicators
        frames: dict[str, pd.DataFrame] = {}

        tf_map = {
            "M1": MT5.TIMEFRAME_M1,
            "M3": getattr(MT5, "TIMEFRAME_M3", MT5.TIMEFRAME_M1),
            "M5": MT5.TIMEFRAME_M5,
            "M15": MT5.TIMEFRAME_M15,
            "M30": MT5.TIMEFRAME_M30,
            "H1": MT5.TIMEFRAME_H1,
            "H2": getattr(MT5, "TIMEFRAME_H2", MT5.TIMEFRAME_H1),
            "H4": MT5.TIMEFRAME_H4,
            "D1": MT5.TIMEFRAME_D1,
            "W1": MT5.TIMEFRAME_W1,
            "MN1": MT5.TIMEFRAME_MN1,
        }

        req_tfs = self.required_tfs

        for tf_name in req_tfs:
            mt5_tf = tf_map.get(tf_name)
            if mt5_tf is None:
                continue
            # Use only fully closed candles for feature stability (avoid "repainting" on the forming bar).
            # In MT5, start_pos=0 is the current forming candle; start_pos=1 is the last closed candle.
            rates = MT5.copy_rates_from_pos(symbol, mt5_tf, 1, lookback)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df.set_index("timestamp", inplace=True)
                df.rename(columns={"tick_volume": "volume"}, inplace=True)
                df["symbol"] = symbol
                df["timeframe"] = tf_name

                df = self._adjust_to_utc(df)
                df = df.sort_index()

                frames[tf_name] = df
            else:
                logger.warning(f"Failed to fetch live {tf_name} data for {symbol}")

        if not frames:
            return None
        return frames

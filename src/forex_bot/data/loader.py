from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import Settings

logger = logging.getLogger(__name__)


def _timeframe_to_freq(tf: str) -> str | None:
    tf = str(tf or "").upper()
    return {
        "M1": "1min",
        "M3": "3min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1H",
        "H2": "2H",
        "H4": "4H",
        "D1": "1D",
        "W1": "1W",
        "MN1": "1M",
    }.get(tf)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    rename = {}
    for col in out.columns:
        low = str(col).lower()
        if low in {"o", "open"}:
            rename[col] = "open"
        elif low in {"h", "high"}:
            rename[col] = "high"
        elif low in {"l", "low"}:
            rename[col] = "low"
        elif low in {"c", "close"}:
            rename[col] = "close"
        elif low in {"v", "vol", "volume"}:
            rename[col] = "volume"
        elif low in {"timestamp", "time", "datetime", "date"}:
            rename[col] = "timestamp"
    if rename:
        out = out.rename(columns=rename)
    return out


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "close" not in out.columns:
        return out
    for col in ("open", "high", "low"):
        if col not in out.columns:
            out[col] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = 0.0
    return out


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "timestamp" in out.columns:
        idx = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.set_index(idx)
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def _read_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        import polars as pl

        if path.suffix.lower() == ".parquet":
            df = pl.read_parquet(path).to_pandas()
        else:
            df = pl.read_csv(path).to_pandas()
        return df
    except Exception:
        try:
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to read data file %s: %s", path, exc)
            return None


def _resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    freq = _timeframe_to_freq(tf)
    if freq is None:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(freq).agg(agg).dropna()
    return out


class _StubMT5Connection:
    async def get_account_information(self) -> dict[str, Any]:
        return {}

    async def positions_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return []

    async def get_history_deals(self, _from: int, _to: int) -> list[dict[str, Any]]:
        return []

    async def get_symbol_info(self, _symbol: str) -> dict[str, Any]:
        return {}

    async def get_symbol_price(self, _symbol: str) -> dict[str, Any]:
        return {}

    async def create_market_buy_order(self, *args, **kwargs) -> dict[str, Any]:
        return {"success": False, "reason": "MT5 not connected"}

    async def create_market_sell_order(self, *args, **kwargs) -> dict[str, Any]:
        return {"success": False, "reason": "MT5 not connected"}

    async def close_position_by_ticket(self, *args, **kwargs) -> dict[str, Any]:
        return {"success": False, "reason": "MT5 not connected"}


def _mt5_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "_asdict"):
        try:
            return obj._asdict()
        except Exception:
            return {}
    try:
        return dict(obj)
    except Exception:
        return {}


def _mt5_list_to_dict(items: Any) -> list[dict[str, Any]]:
    if items is None:
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        out.append(_mt5_to_dict(item))
    return out


class _MT5Connection:
    def __init__(self, mt5_module: Any) -> None:
        self.mt5 = mt5_module

    async def get_account_information(self) -> dict[str, Any]:
        return _mt5_to_dict(self.mt5.account_info())

    async def positions_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            return _mt5_list_to_dict(self.mt5.positions_get(symbol=symbol))
        return _mt5_list_to_dict(self.mt5.positions_get())

    async def get_history_deals(self, _from: int, _to: int) -> list[dict[str, Any]]:
        return _mt5_list_to_dict(self.mt5.history_deals_get(_from, _to))

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        return _mt5_to_dict(self.mt5.symbol_info(symbol))

    async def get_symbol_price(self, symbol: str) -> dict[str, Any]:
        return _mt5_to_dict(self.mt5.symbol_info_tick(symbol))

    async def create_market_buy_order(self, symbol: str, volume: float, sl: float, tp: float, deviation: int = 20, magic: int = 0, comment: str | None = None) -> dict[str, Any]:
        price = self.mt5.symbol_info_tick(symbol).ask
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": self.mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": comment or "forex_bot",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        return _mt5_to_dict(result)

    async def create_market_sell_order(self, symbol: str, volume: float, sl: float, tp: float, deviation: int = 20, magic: int = 0, comment: str | None = None) -> dict[str, Any]:
        price = self.mt5.symbol_info_tick(symbol).bid
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": comment or "forex_bot",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        return _mt5_to_dict(result)

    async def close_position_by_ticket(self, ticket: int, symbol: str, volume: float | None = None) -> dict[str, Any]:
        positions = self.mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "reason": "Position not found"}
        pos = positions[0]
        pos_dict = _mt5_to_dict(pos)
        vol = float(volume if volume is not None else pos_dict.get("volume", 0.0))
        order_type = self.mt5.ORDER_TYPE_SELL if pos_dict.get("type", 0) == 0 else self.mt5.ORDER_TYPE_BUY
        price = self.mt5.symbol_info_tick(symbol).bid if order_type == self.mt5.ORDER_TYPE_SELL else self.mt5.symbol_info_tick(symbol).ask
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": vol,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": pos_dict.get("magic", 0),
            "comment": "forex_bot_close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        return _mt5_to_dict(result)


class MT5Adapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.connection = _StubMT5Connection()
        self._connected = False

    async def connect(self) -> bool:
        backend = str(getattr(self.settings.system, "broker_backend", ""))
        if backend != "mt5_local":
            self._connected = False
            return False
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:
            logger.warning("MetaTrader5 not available: %s", exc)
            self._connected = False
            return False
        try:
            if not mt5.initialize():
                self._connected = False
                return False
            self._connected = True
            self.connection = _MT5Connection(mt5)
        except Exception as exc:
            logger.warning("MT5 initialize failed: %s", exc)
            self._connected = False
            return False
        return True

    async def disconnect(self) -> None:
        try:
            import MetaTrader5 as mt5  # type: ignore

            mt5.shutdown()
        except Exception:
            pass
        self._connected = False
        self.connection = _StubMT5Connection()

    def is_connected(self) -> bool:
        return bool(self._connected)

    async def get_live_frames(self, symbol: str, *, timeframes: list[str], tail: int = 500) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        if not self._connected:
            return frames
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception:
            return frames
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        for tf in timeframes:
            mt5_tf = tf_map.get(tf)
            if mt5_tf is None:
                continue
            try:
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, tail)
            except Exception:
                rates = None
            if rates is None or len(rates) == 0:
                continue
            df = pd.DataFrame(rates)
            df = _normalize_columns(df)
            df = _ensure_datetime_index(df)
            frames[tf] = df
        return frames


class DataLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = Path(getattr(self.settings.system, "data_dir", "data"))
        self.mt5_adapter = MT5Adapter(settings)
        self._connected = False

    async def connect(self) -> bool:
        self._connected = await self.mt5_adapter.connect()
        return self._connected

    async def disconnect(self) -> None:
        await self.mt5_adapter.disconnect()
        self._connected = False

    def is_connected(self) -> bool:
        return bool(self._connected)

    def _timeframes(self) -> list[str]:
        base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")
        tfs = [base_tf]
        for tf in list(getattr(self.settings.system, "required_timeframes", []) or []):
            if tf not in tfs:
                tfs.append(tf)
        for tf in list(getattr(self.settings.system, "higher_timeframes", []) or []):
            if tf not in tfs:
                tfs.append(tf)
        return tfs

    def _candidate_paths(self, symbol: str, tf: str) -> list[Path]:
        paths = []
        root = self.data_dir
        paths.append(root / f"symbol={symbol}" / f"timeframe={tf}" / "data.parquet")
        paths.append(root / f"symbol={symbol}" / f"timeframe={tf}" / "data.csv")
        paths.append(root / f"{symbol}_{tf}.parquet")
        paths.append(root / f"{symbol}_{tf}.csv")
        paths.append(root / symbol / f"{tf}.parquet")
        paths.append(root / symbol / f"{tf}.csv")
        return paths

    def _load_frames(self, symbol: str, *, tail: int | None = None, allow_resample: bool = True) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        if not self.data_dir.exists():
            return frames
        for tf in self._timeframes():
            df = None
            for path in self._candidate_paths(symbol, tf):
                df = _read_frame(path)
                if df is not None and not df.empty:
                    break
            if df is None or df.empty:
                continue
            df = _normalize_columns(df)
            df = _ensure_datetime_index(df)
            df = _ensure_ohlcv(df)
            if tail is not None and tail > 0:
                df = df.tail(tail)
            frames[tf] = df
        if allow_resample:
            base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")
            base = frames.get(base_tf)
            if base is not None and not base.empty:
                for tf in self._timeframes():
                    if tf in frames:
                        continue
                    resampled = _resample_ohlcv(base, tf)
                    if resampled is not None and not resampled.empty:
                        frames[tf] = resampled
        return frames

    async def ensure_history(self, symbol: str) -> bool:
        frames = self._load_frames(symbol)
        return bool(frames)

    async def get_training_data(self, symbol: str) -> dict[str, pd.DataFrame]:
        return self._load_frames(symbol)

    async def get_live_data(self, symbol: str) -> dict[str, pd.DataFrame]:
        tfs = self._timeframes()
        if self.mt5_adapter.is_connected():
            frames = await self.mt5_adapter.get_live_frames(symbol, timeframes=tfs, tail=500)
            if frames:
                return frames
        return self._load_frames(symbol, tail=2000)


__all__ = ["DataLoader"]

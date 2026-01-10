from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..core.config import Settings
from ..domain.events import PreparedDataset

logger = logging.getLogger(__name__)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        ts_col = None
        for candidate in ("timestamp", "time", "datetime", "date"):
            if candidate in out.columns:
                ts_col = candidate
                break
        if ts_col is not None:
            idx = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
            out = out.set_index(idx)
        else:
            out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    else:
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr.fillna(method="bfill").fillna(0.0)


def _compute_adx_numba(high: Iterable[float], low: Iterable[float], close: Iterable[float], period: int = 14) -> np.ndarray:
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    n = close_arr.shape[0]
    adx = np.zeros(n, dtype=np.float64)
    if n <= period:
        return adx

    tr = np.zeros(n, dtype=np.float64)
    pdm = np.zeros(n, dtype=np.float64)
    mdm = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        up = high_arr[i] - high_arr[i - 1]
        down = low_arr[i - 1] - low_arr[i]
        pdm[i] = up if (up > down and up > 0.0) else 0.0
        mdm[i] = down if (down > up and down > 0.0) else 0.0
        tr[i] = max(
            high_arr[i] - low_arr[i],
            abs(high_arr[i] - close_arr[i - 1]),
            abs(low_arr[i] - close_arr[i - 1]),
        )

    atr = np.zeros(n, dtype=np.float64)
    pdm_sm = np.zeros(n, dtype=np.float64)
    mdm_sm = np.zeros(n, dtype=np.float64)

    atr[period] = tr[1 : period + 1].sum()
    pdm_sm[period] = pdm[1 : period + 1].sum()
    mdm_sm[period] = mdm[1 : period + 1].sum()

    for i in range(period + 1, n):
        atr[i] = atr[i - 1] - (atr[i - 1] / period) + tr[i]
        pdm_sm[i] = pdm_sm[i - 1] - (pdm_sm[i - 1] / period) + pdm[i]
        mdm_sm[i] = mdm_sm[i - 1] - (mdm_sm[i - 1] / period) + mdm[i]

    for i in range(period, n):
        if atr[i] <= 0.0:
            continue
        pdi = 100.0 * (pdm_sm[i] / atr[i])
        mdi = 100.0 * (mdm_sm[i] / atr[i])
        denom = pdi + mdi
        if denom <= 0.0:
            dx = 0.0
        else:
            dx = 100.0 * abs(pdi - mdi) / denom
        if i == period:
            adx[i] = dx
        else:
            adx[i] = (adx[i - 1] * (period - 1) + dx) / period
    return adx


@dataclass(slots=True)
class _LabelConfig:
    horizon: int
    min_dist: float


class FeatureEngineer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _label_config(self) -> _LabelConfig:
        horizon = 1
        for key in ("FOREX_BOT_LABEL_HORIZON", "FOREX_BOT_LABEL_HORIZON_BARS"):
            raw = os.environ.get(key)
            if raw is None or str(raw).strip() == "":
                continue
            try:
                val = int(str(raw).strip())
            except Exception:
                continue
            if val > 0:
                horizon = val
                break
        try:
            min_dist = float(getattr(self.settings.risk, "meta_label_min_dist", 0.0) or 0.0)
        except Exception:
            min_dist = 0.0
        return _LabelConfig(horizon=max(1, horizon), min_dist=max(0.0, min_dist))

    def _compute_basic_features(self, df: pd.DataFrame, *, use_gpu: bool = False) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        out["rsi"] = _compute_rsi(close)
        macd, macd_signal, macd_hist = _compute_macd(close)
        out["macd"] = macd
        out["macd_signal"] = macd_signal
        out["macd_hist"] = macd_hist
        out["adx"] = _compute_adx_numba(high.to_numpy(), low.to_numpy(), close.to_numpy())
        return out

    def _compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        out["returns"] = close.pct_change().fillna(0.0)
        out["atr14"] = _compute_atr(high, low, close, period=14)
        ma = close.rolling(20, min_periods=1).mean()
        std = close.rolling(20, min_periods=1).std().fillna(0.0)
        upper = ma + 2.0 * std
        lower = ma - 2.0 * std
        out["bb_width"] = ((upper - lower) / (ma.replace(0.0, np.nan))).fillna(0.0)
        return out

    def _compute_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        if "volume" in out.columns:
            volume = out["volume"].astype(float)
        else:
            volume = pd.Series(np.ones(len(out), dtype=float), index=out.index)
        close = out["close"].astype(float)
        window = 20
        vol_sum = volume.rolling(window, min_periods=1).sum().replace(0.0, np.nan)
        poc = (close * volume).rolling(window, min_periods=1).sum() / vol_sum
        poc = poc.fillna(method="bfill").fillna(close)
        out["dist_to_poc"] = (close - poc).fillna(0.0)
        std = close.rolling(window, min_periods=1).std().fillna(0.0)
        out["in_value_area"] = ((close >= (poc - std)) & (close <= (poc + std))).astype(float)
        return out

    def _compute_obi_features(self, df: pd.DataFrame, *, use_gpu: bool = False) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        open_ = out["open"].astype(float)
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        rng = (high - low).replace(0.0, np.nan)
        if "volume" in out.columns:
            volume = out["volume"].astype(float)
        else:
            volume = pd.Series(np.ones(len(out), dtype=float), index=out.index)
        imbalance = ((close - open_) / rng).fillna(0.0) * volume
        out["vol_imbalance"] = imbalance.fillna(0.0)
        return out

    def _compute_base_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        rsi = out.get("rsi")
        macd_hist = out.get("macd_hist")
        if rsi is None or macd_hist is None:
            out["base_signal"] = 0
            return out
        buy = (rsi < 30.0) & (macd_hist > 0)
        sell = (rsi > 70.0) & (macd_hist < 0)
        signal = np.where(buy, 1, np.where(sell, -1, 0))
        out["base_signal"] = signal.astype(int)
        return out

    def _compute_labels(self, close: pd.Series, cfg: _LabelConfig) -> pd.Series:
        future = close.shift(-cfg.horizon)
        delta = (future - close).astype(float)
        up = delta > cfg.min_dist
        down = delta < -cfg.min_dist
        labels = np.where(up, 1, np.where(down, -1, 0)).astype(int)
        return pd.Series(labels, index=close.index)

    def prepare(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        news_features: pd.DataFrame | None = None,
        symbol: str | None = None,
    ) -> PreparedDataset:
        if frames is None or len(frames) == 0:
            empty = pd.DataFrame()
            return PreparedDataset(X=empty, y=pd.Series(dtype=int), index=empty.index, feature_names=[], metadata=None, labels=None)

        base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")
        base_df = frames.get(base_tf)
        if base_df is None:
            base_df = frames.get("M1")
        if base_df is None:
            base_df = next(iter(frames.values()))
        base_df = _ensure_datetime_index(base_df)
        base_df = base_df.sort_index()
        try:
            if symbol:
                base_df.attrs["symbol"] = symbol
        except Exception:
            pass

        features = self._compute_basic_features(base_df, use_gpu=False)
        features = self._compute_volatility_features(features)
        features = self._compute_volume_profile_features(features)
        features = self._compute_obi_features(features, use_gpu=False)
        features = self._compute_base_signal(features)

        if news_features is not None and not news_features.empty:
            nf = _ensure_datetime_index(news_features)
            nf = nf.reindex(features.index, method="ffill").fillna(0.0)
            nf = nf.rename(columns={c: f"news_{c}" for c in nf.columns})
            features = pd.concat([features, nf], axis=1)

        higher = list(getattr(self.settings.system, "higher_timeframes", []) or [])
        required = list(getattr(self.settings.system, "required_timeframes", []) or [])
        for tf in required:
            if tf not in higher:
                higher.append(tf)

        for tf in higher:
            if tf == base_tf:
                continue
            htf = frames.get(tf)
            if htf is None or htf.empty:
                continue
            htf = _ensure_datetime_index(htf).sort_index()
            htf = self._compute_basic_features(htf, use_gpu=False)
            htf = self._compute_volatility_features(htf)
            htf = self._compute_volume_profile_features(htf)
            htf = self._compute_obi_features(htf, use_gpu=False)
            htf_shifted = htf.shift(1)
            htf_shifted.columns = [f"{tf}_{c}" for c in htf_shifted.columns]
            aligned = htf_shifted.reindex(features.index, method="ffill").fillna(0.0)
            features = features.join(aligned, how="left")

        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        label_cfg = self._label_config()
        labels = self._compute_labels(base_df["close"].astype(float), label_cfg)
        trim = int(label_cfg.horizon)
        if trim > 0:
            features = features.iloc[:-trim]
            labels = labels.iloc[:-trim]
            meta = base_df.iloc[:-trim]
        else:
            meta = base_df

        return PreparedDataset(
            X=features,
            y=labels,
            index=features.index,
            feature_names=list(features.columns),
            metadata=meta,
            labels=labels,
        )


__all__ = [
    "FeatureEngineer",
    "_compute_adx_numba",
]

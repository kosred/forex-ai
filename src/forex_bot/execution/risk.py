import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ..core.config import Settings
from ..core.storage import RiskLedger
from .meta_controller import MetaController, PropMetaState

logger = logging.getLogger(__name__)

MIN_BREAKEVEN_PROBABILITY = 0.45
RISK_STATE_FILE = Path("cache") / "risk_state.json"


class ChallengePhase(Enum):
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    FUNDED = "funded"


@dataclass(slots=True)
class PropFirmRules:
    max_daily_loss_pct: float = 0.045  # 4.5% daily loss limit (STRICT)
    max_total_loss_pct: float = 0.10
    profit_target_pct: float = 0.10
    min_trading_days: int = 5
    max_trading_days: int = 60
    max_lot_size: float = 10.0
    news_trading_allowed: bool = False
    weekend_holding: bool = False
    scaling_enabled: bool = True
    daily_dd_warning_pct: float = 0.035  # 3.5% warning threshold
    daily_dd_stop_trading_pct: float = 0.040  # 4.0% stop trading buffer
    daily_profit_lock_pct: float = 0.03  # lock profits if hit
    max_trades_per_day: int = 15


class RevengeTradeDetector:
    def __init__(self):
        self.recent_trades = []
        self.max_trades_tracked = 10

    def record_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        was_stopped: bool,
        size: float = 0.0,
        direction: int | None = None,
    ) -> None:
        trade_data = {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl": pnl,
            "was_stopped": was_stopped,
            "duration_minutes": (exit_time - entry_time).total_seconds() / 60,
            "size": size,
            "direction": direction,
        }
        self.recent_trades.append(trade_data)
        if len(self.recent_trades) > self.max_trades_tracked:
            self.recent_trades.pop(0)

    def is_revenge_trading(self, current_time: datetime) -> bool:
        if len(self.recent_trades) < 2:
            return False
        last_trade = self.recent_trades[-1]
        time_since_last = (current_time - last_trade["exit_time"]).total_seconds() / 60

        if time_since_last < 15 and last_trade["pnl"] < 0:
            return True

        consecutive_losses = 0
        for trade in reversed(self.recent_trades[-5:]):
            if trade["pnl"] < 0:
                consecutive_losses += 1
            else:
                break
        if consecutive_losses >= 3:
            hour = current_time.hour
            optimal_times = (7 <= hour < 9) or (13 <= hour < 15)
            if not optimal_times:
                return True

        if len(self.recent_trades) >= 3:
            recent = self.recent_trades[-3:]
            sizes = [t.get("size", 0.0) for t in recent[:-1]]
            last_size = recent[-1].get("size", 0.0)
            # Need at least 2 previous trades to check revenge pattern
            if sizes and len(recent) >= 2:
                mean_prev = np.mean(sizes)
                if mean_prev > 0 and last_size > 1.5 * mean_prev and recent[-2].get("pnl", 0) < 0:
                    return True

        if len(self.recent_trades) >= 3:
            dirs = [t.get("direction") for t in self.recent_trades[-3:]]
            pnls = [t.get("pnl", 0.0) for t in self.recent_trades[-3:]]
            if all(d is not None for d in dirs) and pnls[-1] < 0 and pnls[-2] < 0:
                if dirs[-1] == dirs[-2] == dirs[-3]:
                    return True
        if len(self.recent_trades) >= 2:
            last = self.recent_trades[-1]
            prev = self.recent_trades[-2]
            gap_min = (last["entry_time"] - prev["exit_time"]).total_seconds() / 60.0
            if gap_min < 30 and last.get("pnl", 0) < 0 and prev.get("pnl", 0) < 0:
                if last.get("direction") is not None and last.get("direction") == prev.get("direction"):
                    return True
        return False


class RiskManager:
    MIN_LOT = 0.01

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # Use configured session timezone for all "day boundary" logic to avoid local-time drift.
        # Defaults to UTC if misconfigured.
        try:
            self._session_tz = ZoneInfo(settings.system.session_timezone)
        except Exception:
            self._session_tz = ZoneInfo("UTC")
        self.risk_ledger = RiskLedger(max_events=settings.system.risk_ledger_max_events)

        initial_balance = float(getattr(settings.risk, "initial_balance", 0.0) or 0.0)

        self.challenge_phase = "unknown"  # Will be detected after MT5 balance fetch
        self.prop_rules = PropFirmRules(
            max_daily_loss_pct=settings.risk.daily_drawdown_limit,
            max_total_loss_pct=settings.risk.total_drawdown_limit,
            max_trades_per_day=settings.risk.max_trades_per_day,
            daily_dd_warning_pct=settings.risk.daily_drawdown_limit * 0.8,  # Derived warning
            daily_dd_stop_trading_pct=settings.risk.daily_drawdown_limit * 0.95,  # Derived stop
        )
        self._base_prop_max_trades = self.prop_rules.max_trades_per_day
        self.month_start_equity: float = 0.0  # Will be updated from MT5
        self.month_start_date: date = datetime.now(self._session_tz).date().replace(day=1)
        self.monthly_return_pct: float = 0.0
        self.monthly_profit_target_pct: float = getattr(settings.risk, "monthly_profit_target_pct", 0.04)
        self.monthly_target_hit: bool = False
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.session_trades = 0
        self.consecutive_losses = 0
        self.peak_equity = initial_balance
        self.circuit_breaker_triggered = False
        self.total_peak_equity = float(initial_balance)
        self._last_session_date: date | None = None

        self._last_day: date | None = None
        self.day_start_equity: float = float(initial_balance)
        self.day_peak_equity: float = float(initial_balance)
        self.phase_trade_days: set[date] = set()

        self.challenge_start_date = datetime.now(self._session_tz).date()
        self.daily_pnl_tracker = {}
        self.consecutive_winning_days = 0
        self.consecutive_losing_days = 0
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0
        self.recovery_mode = False
        self.recovery_risk_cap = 0.0025  # 0.25% risk per trade when in recovery
        self.recovery_conf_boost = 0.10  # require higher confidence in recovery
        self.recovery_min_win_prob = 0.70
        self.recovery_max_trades = 2

        self.rolling_outcomes = deque(maxlen=10)
        self.reflection_mode = False
        self.reflection_cooldown_until: datetime | None = None

        self.consistency_score = 100.0
        self.revenge_trading_detector = RevengeTradeDetector()

        self._session_start = self._parse_time(settings.system.trading_session_start)
        self._session_end = self._parse_time(settings.system.trading_session_end)

        self._news_state: dict[str, Any] = {
            "tier1_nearby": False,
            "news_confidence": 0.0,
            "news_surprise": 0.0,
            "suggested_risk_cap": settings.risk.max_risk_per_trade,
        }
        self._kill_window_until: datetime | None = None

        base_spread = settings.risk.backtest_spread_pips
        base_slippage = settings.risk.slippage_pips

        self._spread_state: dict[str, float] = {
            "spread_baseline": base_spread,
            "slippage_baseline": base_slippage,
            "current_spread": base_spread,
            "current_slippage": base_slippage,
        }

        self.meta_controller = MetaController(
            max_daily_dd=self.prop_rules.max_daily_loss_pct,
            safety_buffer=0.025,
            base_risk_per_trade=self.settings.risk.risk_per_trade,
            settings=settings,
            silent=True,  # Silence logs during initialization and backtesting
        )

        self.load_state()

    def save_state(self) -> None:
        try:
            RISK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "date": self._last_session_date.isoformat() if self._last_session_date else None,
                "month_start_date": self.month_start_date.isoformat() if self.month_start_date else None,
                "month_start_equity": self.month_start_equity,
                "day_start_equity": self.day_start_equity,
                "daily_loss": self.daily_loss,
                "daily_profit": self.daily_profit,
                "session_trades": self.session_trades,
                "consecutive_losses": self.consecutive_losses,
                "total_peak_equity": self.total_peak_equity,
                "circuit_breaker_triggered": self.circuit_breaker_triggered,
                "recovery_mode": self.recovery_mode,
                "monthly_target_hit": self.monthly_target_hit,
            }

            # Atomic write pattern
            temp_path = RISK_STATE_FILE.with_suffix(".tmp")
            temp_path.write_text(json.dumps(state))
            temp_path.replace(RISK_STATE_FILE)

        except Exception as e:
            logger.warning(f"Failed to save risk state: {e}")

    def load_state(self) -> None:
        if not RISK_STATE_FILE.exists():
            return
        try:
            state = json.loads(RISK_STATE_FILE.read_text())
            saved_date_str = state.get("date")
            if not saved_date_str:
                return

            saved_date = date.fromisoformat(saved_date_str)
            now_date = datetime.now(self._session_tz).date()

            if saved_date == now_date:
                self._last_session_date = saved_date
                month_saved = state.get("month_start_date")
                if month_saved:
                    try:
                        self.month_start_date = date.fromisoformat(month_saved)
                    except Exception:
                        self.month_start_date = now_date.replace(day=1)
                fallback_balance = float(getattr(self.settings.risk, "initial_balance", 0.0) or 0.0)
                self.month_start_equity = float(state.get("month_start_equity", fallback_balance))
                self.day_start_equity = float(state.get("day_start_equity", fallback_balance))
                self.daily_loss = float(state.get("daily_loss", 0.0))
                self.daily_profit = float(state.get("daily_profit", 0.0))
                self.session_trades = int(state.get("session_trades", 0))
                self.consecutive_losses = int(state.get("consecutive_losses", 0))
                self.total_peak_equity = float(state.get("total_peak_equity", self.day_start_equity))
                self.circuit_breaker_triggered = bool(state.get("circuit_breaker_triggered", False))
                self.recovery_mode = bool(state.get("recovery_mode", False))
                self.monthly_target_hit = bool(state.get("monthly_target_hit", False))
                logger.info(f"Restored risk state for {saved_date}")
            else:
                logger.info(f"Found stale risk state from {saved_date}, starting fresh for {now_date}")
        except Exception as e:
            logger.warning(f"Failed to load risk state: {e}")

    def _detect_challenge_phase(self, balance: float) -> ChallengePhase:
        prop_firm_balances = [5000, 10000, 25000, 50000, 100000, 200000, 400000]
        if balance in prop_firm_balances:
            return ChallengePhase.PHASE_1
        return ChallengePhase.FUNDED

    def _parse_time(self, raw: str) -> time:
        hour, minute = raw.split(":")
        return time(int(hour), int(minute))

    def initialize_session(self) -> None:
        now_date = datetime.now(self._session_tz).date()
        if self._last_session_date != now_date:
            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.session_trades = 0
            self.consecutive_losses = 0
            self._last_session_date = now_date
            self._kill_window_until = None
            logger.info("Session initialized (fresh)", extra={"event_id": "risk_session_reset"})
        else:
            logger.info("Session initialized (continuing)", extra={"event_id": "risk_session_continue"})

    def _ensure_day(self, equity: float, now: datetime) -> None:
        """Reset daily counters if a new session day starts."""
        if self._last_session_date is None or now.date() != self._last_session_date:
            self._last_session_date = now.date()
            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.session_trades = 0
            self.consecutive_losses = 0
            self.day_start_equity = equity
            # FIX: Reset daily peak equity for intraday trailing calculations
            self.day_peak_equity = equity
            self.circuit_breaker_triggered = False
            self.recovery_mode = False
            self.prop_rules.max_trades_per_day = self._base_prop_max_trades
            logger.info("New trading day detected; daily counters reset.")
            self.save_state()

    def _ensure_month(self, equity: float, now: datetime) -> None:
        """Reset month tracking when calendar month rolls or first run."""
        month_start = now.date().replace(day=1)
        if self.month_start_date != month_start or self.month_start_equity <= 0:
            self.month_start_date = month_start
            self.month_start_equity = equity
            self.monthly_target_hit = False
            self.risk_ledger.record("MONTH_RESET", f"New month started; base equity={equity:.2f}", severity="info")
            self.save_state()

    def _update_monthly_metrics(self, equity: float) -> None:
        """Track monthly return based on live MT5 equity."""
        if self.month_start_equity <= 0:
            return
        self.monthly_return_pct = (equity - self.month_start_equity) / self.month_start_equity
        if not self.monthly_target_hit and self.monthly_return_pct >= self.monthly_profit_target_pct:
            self.monthly_target_hit = True
            self.risk_ledger.record(
                "MONTH_TARGET_HIT",
                f"Monthly return hit {self.monthly_return_pct:.2%} (target {self.monthly_profit_target_pct:.2%})",
                severity="info",
            )
            self.save_state()

    def _update_recovery_state(self, equity: float) -> None:
        """Switch into/out of recovery mode based on intraday drawdown and equity recovery."""
        if self.day_start_equity <= 0:
            return
        daily_dd_pct = (self.day_start_equity - equity) / self.day_start_equity

        # Enter recovery mode if DD hits warning level
        if daily_dd_pct >= self.prop_rules.daily_dd_warning_pct:
            if not self.recovery_mode:
                self.recovery_mode = True
                self.risk_ledger.record(
                    "RECOVERY_ON", f"Entering recovery mode at DD {daily_dd_pct:.2%}", severity="warning"
                )
        # Exit recovery mode if equity recovers to within 0.5% of break-even OR DD drops below half the warning level
        elif self.recovery_mode:
            recovery_threshold_pct = 0.005  # 0.5% from break-even
            half_warning = self.prop_rules.daily_dd_warning_pct / 2.0

            if equity >= (self.day_start_equity * (1.0 - recovery_threshold_pct)) or daily_dd_pct <= half_warning:
                self.recovery_mode = False
                self.risk_ledger.record(
                    "RECOVERY_OFF",
                    f"Exiting recovery mode (equity={equity:.2f}, DD={daily_dd_pct:.2%})",
                    severity="info",
                )

    def is_trading_session(self) -> bool:
        now = datetime.now(self._session_tz)

        if now.weekday() >= 5:
            return False

        current = now.time()
        if self._session_end < self._session_start:
            in_session = current >= self._session_start or current <= self._session_end
        else:
            in_session = self._session_start <= current <= self._session_end
        return in_session

    def update_news_state(self, policy_flags: dict[str, Any], now: datetime | None = None) -> None:
        if not policy_flags:
            return
        self._news_state.update(policy_flags)
        now = now or datetime.now(tz=self._session_tz)
        if policy_flags.get("tier1_nearby"):
            self._kill_window_until = now + timedelta(minutes=self.settings.news.news_kill_window_min)
            self.risk_ledger.record("NEWS_KILL", "Tier-1 news kill window active", severity="warning")

    def update_spread_state(
        self,
        *,
        live_spread: float,
        live_slippage: float,
        baseline_spread: float,
        baseline_slippage: float,
    ) -> None:
        self._spread_state.update(
            {
                "current_spread": live_spread,
                "current_slippage": live_slippage,
                "spread_baseline": baseline_spread,
                "slippage_baseline": baseline_slippage,
            }
        )

    def check_trade_allowed(self, equity: float, confidence: float, timestamp: datetime) -> tuple[bool, str]:
        self._ensure_day(equity, timestamp)
        self._ensure_month(equity, timestamp)
        self._update_monthly_metrics(equity)
        self._update_recovery_state(equity)

        if self.reflection_mode:
            if self.reflection_cooldown_until and timestamp < self.reflection_cooldown_until:
                return False, "reflection_mode_cooldown"
            else:
                self.reflection_mode = False
                self.rolling_outcomes.clear()  # Reset stats for fresh start
                logger.info("Reflection mode cooldown ended. Resuming trading.")

        if equity > self.total_peak_equity:
            self.total_peak_equity = equity
            self.risk_ledger.record("EQUITY_PEAK", f"New equity peak: {equity:.2f}")

        # FIX: Track intraday peak for strict prop firm rules (Intraday Trailing DD)
        self.day_peak_equity = max(self.day_peak_equity, equity)

        if self.circuit_breaker_triggered:
            return False, "Circuit breaker active"

        if not self.is_trading_session():
            return False, "Outside trading session"

        if self._kill_window_until and timestamp < self._kill_window_until:
            return False, "News kill window active"

        if self.revenge_trading_detector.is_revenge_trading(timestamp):
            return False, "Revenge trading detected"

        # Standard Daily Loss (from Day Start)
        daily_dd_pct = (self.day_start_equity - equity) / self.day_start_equity if self.day_start_equity > 0 else 0.0

        # Intraday Trailing Loss (from Day Peak) - Stricter rule used by some firms
        intraday_dd_pct = (self.day_peak_equity - equity) / self.day_peak_equity if self.day_peak_equity > 0 else 0.0

        if daily_dd_pct >= self.prop_rules.daily_dd_stop_trading_pct:
            self.circuit_breaker_triggered = True
            return False, f"Daily drawdown limit reached ({daily_dd_pct:.2%})"

        if intraday_dd_pct >= self.prop_rules.daily_dd_stop_trading_pct:
            self.circuit_breaker_triggered = True
            return False, f"Intraday trailing limit reached ({intraday_dd_pct:.2%})"

        if (
            daily_dd_pct >= self.prop_rules.daily_dd_warning_pct
            or intraday_dd_pct >= self.prop_rules.daily_dd_warning_pct
        ):
            self.risk_ledger.record(
                "DAILY_DD_WARN",
                f"DD warning (Day: {daily_dd_pct:.2%}, Intra: {intraday_dd_pct:.2%})",
                severity="warning",
            )
            self.recovery_mode = True

        total_dd_pct = (self.total_peak_equity - equity) / self.total_peak_equity if self.total_peak_equity > 0 else 0.0
        if total_dd_pct >= self.settings.risk.total_drawdown_limit:
            self.circuit_breaker_triggered = True
            return False, "Total drawdown limit reached"

        daily_profit_stop = getattr(self.settings.risk, "daily_profit_stop_pct", 0.0) or 0.0
        if daily_profit_stop > 0 and self.daily_profit > 0:
            if (self.daily_profit / self.day_start_equity) >= daily_profit_stop:
                return False, "Daily profit stop reached"
        # Profit lock check - stop trading if daily profit target reached
        if (
            self.prop_rules.daily_profit_lock_pct
            and self.prop_rules.daily_profit_lock_pct > 0
            and self.daily_profit > 0
        ):
            profit_pct = (self.daily_profit / self.day_start_equity) if self.day_start_equity > 0 else 0.0
            if profit_pct >= self.prop_rules.daily_profit_lock_pct:
                return (
                    False,
                    f"Prop profit lock reached ({profit_pct:.2%} >= {self.prop_rules.daily_profit_lock_pct:.2%})",
                )

        max_trades = getattr(self.settings.risk, "max_trades_per_day", 0) or 0
        if max_trades <= 0:
            max_trades = self.prop_rules.max_trades_per_day
        if self.recovery_mode:
            max_trades = min(max_trades, self.recovery_max_trades)
        if max_trades > 0 and self.session_trades >= max_trades:
            return False, "Max trades per day reached"

        spread_baseline = self._spread_state.get("spread_baseline", 0.0) or 1e-6
        slippage_baseline = self._spread_state.get("slippage_baseline", 0.0) or 1e-6
        current_spread = self._spread_state.get("current_spread", spread_baseline)
        current_slippage = self._spread_state.get("current_slippage", slippage_baseline)

        spread_ratio = (current_spread + 1e-9) / spread_baseline
        slippage_ratio = (current_slippage + 1e-9) / slippage_baseline

        if spread_ratio > self.settings.risk.spread_guard_multiplier:
            self.risk_ledger.record("SPREAD_GUARD", f"Spread guard triggered ({spread_ratio:.2f}x)", severity="warning")
            return False, f"Spread too high ({spread_ratio:.1f}x baseline)"

        if slippage_ratio > self.settings.risk.slippage_guard_multiplier:
            self.risk_ledger.record(
                "SLIPPAGE_GUARD", f"Slippage guard triggered ({slippage_ratio:.2f}x)", severity="warning"
            )
            return False, f"Slippage risk too high ({slippage_ratio:.1f}x baseline)"

        # Confidence threshold check - higher requirement in recovery mode
        min_conf = self.settings.risk.min_confidence_threshold
        if self.recovery_mode:
            # In recovery mode, require HIGHER confidence (use min to take the more restrictive)
            min_conf = max(min_conf + self.recovery_conf_boost, self.recovery_min_win_prob)
            logger.debug(f"Recovery mode active: confidence threshold raised to {min_conf:.2f}")
        if confidence < min_conf:
            return False, f"Confidence {confidence:.2f} below threshold {min_conf:.2f}"

        return True, "OK"

    def calculate_position_size(
        self,
        equity: float,
        stop_loss_pips: float,
        confidence: float,
        uncertainty: float = 0.0,
        symbol_info: dict[str, Any] | None = None,
        market_regime: str = "Normal",
    ) -> float:
        if stop_loss_pips <= 0:
            return 0.0

        base_risk = self.settings.risk.risk_per_trade

        if confidence >= 0.80:
            signal_multiplier = 1.00
        elif confidence >= 0.60:
            signal_multiplier = 0.50 + (confidence - 0.60) * 2.5
        else:
            signal_multiplier = 0.30

        uncertainty_penalty = 1.0 - (uncertainty * 0.5)  # Max 50% reduction at full uncertainty

        risk_pct = base_risk * signal_multiplier * uncertainty_penalty

        risk_cap = self.settings.risk.max_risk_per_trade
        if self.recovery_mode:
            risk_cap = min(risk_cap, self.recovery_risk_cap)
        risk_pct = min(risk_pct, risk_cap)

        news_cap = self._news_state.get("suggested_risk_cap")
        if news_cap is not None:
            risk_pct = min(risk_pct, float(news_cap))

        daily_dd_pct = (self.day_start_equity - equity) / self.day_start_equity if self.day_start_equity > 0 else 0.0

        if len(self.rolling_outcomes) > 0:
            real_win_rate = sum(self.rolling_outcomes) / len(self.rolling_outcomes)
        else:
            real_win_rate = 0.5  # Default until we have data

        spread_baseline = self._spread_state.get("spread_baseline", 1e-6)
        current_spread = self._spread_state.get("current_spread", spread_baseline)
        spread_ratio = current_spread / spread_baseline if spread_baseline > 0 else 1.0

        if spread_ratio > 1.5:
            vol_regime = "high"
        elif spread_ratio < 0.8:
            vol_regime = "low"
        else:
            vol_regime = "normal"

        meta_state = PropMetaState(
            daily_dd_pct=daily_dd_pct,
            volatility_regime=vol_regime,
            recent_win_rate=real_win_rate,
            consecutive_losses=self.consecutive_losses,
            model_confidence=confidence,
            hour_of_day=datetime.now(self._session_tz).hour,
            market_regime=str(market_regime or "Normal"),
        )

        risk_mult, required_conf, allow_trade = self.meta_controller.get_risk_parameters(meta_state)
        self.last_risk_mult = risk_mult  # Store for logging

        if not allow_trade:
            return 0.0

        risk_pct *= risk_mult

        try:
            if self.total_peak_equity > 0:
                dd_pct = (self.total_peak_equity - equity) / self.total_peak_equity
                if dd_pct > 0:
                    scale = max(0.3, 1.0 - dd_pct / max(self.settings.risk.total_drawdown_limit, 1e-6))
                    risk_pct *= scale
        except Exception as e:
            logger.warning(f"Saving risk state failed: {e}", exc_info=True)

        risk_amount = equity * risk_pct
        pip_size, pip_value = self._compute_pip_metrics(symbol_info)

        lot_size = risk_amount / max(stop_loss_pips * pip_value, 1e-9)

        lot_size = int(lot_size * 100) / 100.0
        return max(0.0, min(lot_size, self.prop_rules.max_lot_size))

    def _compute_pip_metrics(self, symbol_info: dict | None) -> tuple[float, float]:
        """
        Return (pip_size, pip_value_per_lot) using broker metadata when available.
        Falls back to symbol heuristics instead of a flat 10 USD pip value.
        """
        sym = (self.settings.system.symbol or "").upper()
        default_point = 0.0001
        digits = 5
        if symbol_info:
            default_point = float(symbol_info.get("point", default_point) or default_point)
            digits = int(symbol_info.get("digits", digits) or digits)
        # Infer pip size from digits with symbol overrides
        if sym.startswith("XAU") or sym.startswith("XAG"):
            pip_size = 0.01
        elif "BTC" in sym or "ETH" in sym or "LTC" in sym:
            pip_size = 1.0
        elif sym.endswith("JPY") or sym.startswith("JPY"):
            pip_size = 0.01
        else:
            pip_size = default_point * (10 if digits >= 4 else 1)

        tick_size = pip_size
        tick_value = 0.0
        contract_size = 100000.0
        price_hint = 0.0
        if symbol_info:
            tick_size = float(symbol_info.get("trade_tick_size", symbol_info.get("tick_size", tick_size)) or tick_size)
            tick_value = float(symbol_info.get("trade_tick_value", symbol_info.get("tick_value", 0.0)) or 0.0)
            contract_size = float(
                symbol_info.get("trade_contract_size", symbol_info.get("contract_size", contract_size)) or contract_size
            )
            price_hint = float(symbol_info.get("bid") or symbol_info.get("ask") or symbol_info.get("last") or 0.0)

        pip_value = 0.0
        try:
            if tick_value > 0 and tick_size > 0:
                pip_value = tick_value * (pip_size / tick_size)
            elif contract_size > 0 and price_hint > 0:
                pip_value = (pip_size / price_hint) * contract_size
            elif contract_size > 0:
                pip_value = pip_size * contract_size
        except Exception as exc:
            logger.warning(f"Dynamic pip calc failed: {exc}")

        # Ensure we never return zero to avoid divide-by-zero above
        return pip_size, max(pip_value, 1e-6)

    def on_trade_opened(self, timestamp: datetime) -> None:
        """Increment trade counter on open."""
        self.session_trades += 1

    def on_trade_closed(self, pnl: float, timestamp: datetime) -> None:
        was_stopped = pnl < 0  # Simplified
        self.revenge_trading_detector.record_trade(timestamp - timedelta(minutes=30), timestamp, pnl, was_stopped)
        if pnl < 0:
            self.daily_loss += abs(pnl)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            self.daily_profit += pnl

        self.rolling_outcomes.append(1 if pnl > 0 else 0)
        if len(self.rolling_outcomes) >= 5:
            win_rate = sum(self.rolling_outcomes) / len(self.rolling_outcomes)
            if win_rate < 0.40:
                self.reflection_mode = True
                self.reflection_cooldown_until = (timestamp or datetime.now(self._session_tz)) + timedelta(hours=4)
                logger.warning(f"REFLECTION MODE ACTIVATED: Recent WR {win_rate:.2%} < 40%. Pausing for 4h.")
                self.risk_ledger.record("REFLECTION_MODE", f"Paused due to low WR: {win_rate:.2%}", severity="warning")

        if timestamp:
            self.phase_trade_days.add(timestamp.date())

        try:
            dd_pct = (
                self.day_start_equity - (self.day_start_equity + self.daily_profit - self.daily_loss)
            ) / self.day_start_equity
            if dd_pct >= (0.5 * self.settings.risk.daily_drawdown_limit):
                self.prop_rules.max_trades_per_day = max(3, int(self.prop_rules.max_trades_per_day / 2))
        except Exception as e:
            logger.warning(f"Loading stale risk state failed: {e}", exc_info=True)

        try:
            inferred_equity = self.day_start_equity - self.daily_loss + max(pnl, 0)
            if inferred_equity > self.total_peak_equity:
                self.total_peak_equity = inferred_equity
                self.risk_ledger.record("EQUITY_PEAK", f"New equity peak (from PnL): {inferred_equity:.2f}")
        except Exception as e:
            logger.warning(f"Deleting stale state file failed: {e}", exc_info=True)

        self.save_state()

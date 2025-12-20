import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PropMetaState:
    """
    Represents the current state of the account and market
    for the Meta-Controller to make decisions.
    """

    daily_dd_pct: float  # Current daily drawdown (0.0 to 0.05+)
    volatility_regime: str  # "low", "normal", "high"
    recent_win_rate: float  # 0.0 to 1.0 (last 20 trades)
    consecutive_losses: int  # Current streak
    model_confidence: float  # Aggregate confidence of the signal (0.0 to 1.0)
    hour_of_day: int  # 0-23
    market_regime: str = "Normal"  # Unsupervised classification result


class MetaController:
    """
    The 'Brain' that auto-tunes risk parameters based on survival objectives.
    It implements a continuous control function rather than hard rules.
    """

    def __init__(
        self,
        max_daily_dd: float = 0.045,
        safety_buffer: float = 0.025,
        base_risk_per_trade: float = 0.015,
        base_confidence: float = 0.55,
        settings: object | None = None,
        silent: bool = False,
    ):  # Added silent flag
        self.max_daily_dd = max_daily_dd
        self.safety_buffer = safety_buffer  # The "Soft Limit" (2.5%)
        self.base_risk = base_risk_per_trade
        self.base_confidence = base_confidence
        self.last_log_time = 0.0
        self.silent = silent

        self.k_steepness = 200.0

        if settings:
            dyn = getattr(settings, "dynamic", {})
            risk_params = dyn.get("risk_params", {})
            self.k_steepness = risk_params.get("risk_curve_steepness", 200.0)
            if "confidence_threshold" in risk_params:
                self.base_confidence = risk_params["confidence_threshold"]

        if not self.silent:
            logger.info(f"MetaController init: k={self.k_steepness:.1f}, base_conf={self.base_confidence:.2f}")

    def get_risk_parameters(self, state: PropMetaState) -> tuple[float, float, bool]:
        """
        Calculates dynamic risk parameters.

        Returns:
            risk_multiplier (float): 0.0 to 1.0+
            confidence_threshold (float): 0.0 to 1.0
            allow_trading (bool): Hard stop flag
        """

        dd_delta = state.daily_dd_pct - self.safety_buffer

        # FIX: Clip exponent to prevent overflow/underflow
        exponent = np.clip(self.k_steepness * dd_delta, -20.0, 20.0)
        survival_multiplier = 1.0 / (1.0 + np.exp(exponent))

        # 1. Base Volatility Scaling
        vol_multiplier = {
            "low": 1.1,  # Can be slightly more aggressive
            "normal": 1.0,
            "high": 0.7,  # Be careful
        }.get(state.volatility_regime, 1.0)

        # 2. Advanced Regime Scaling (The "Thinking" Part)
        # Proactively cut risk in dangerous regimes detected by Unsupervised Learning
        regime_scale = 1.0
        if "Volatile" in state.market_regime:
            regime_scale = 0.5  # Cut risk in half for unpredictable volatility
        elif "Quiet" in state.market_regime:
            regime_scale = 1.2  # Boost risk slightly in stable conditions

        if "Bear" in state.market_regime or "Bull" in state.market_regime:
            # If trending, we might want to follow more aggressively?
            # For prop firms, safety first -> keeping it 1.0 or slight boost
            regime_scale *= 1.0

        perf_multiplier = 1.0
        if state.recent_win_rate < 0.4:
            perf_multiplier *= 0.8
        if state.consecutive_losses >= 2:
            perf_multiplier *= 0.8
        if state.consecutive_losses >= 4:
            perf_multiplier *= 0.5

        final_risk_multiplier = survival_multiplier * vol_multiplier * regime_scale * perf_multiplier

        confidence_adjustment = (1.0 - survival_multiplier) * 0.2  # Max +20% requirement
        final_confidence_threshold = self.base_confidence + confidence_adjustment

        # FIX: Lower cap to 0.85 to prevent death spiral where no signal is ever good enough
        final_confidence_threshold = min(0.85, final_confidence_threshold)

        allow_trading = True
        if state.daily_dd_pct >= (self.max_daily_dd - 0.002):  # 4.3% hard stop for 4.5% limit
            allow_trading = False
            final_risk_multiplier = 0.0
            # Rate limit warning logs too (every 10s)
            if not self.silent and time.time() - self.last_log_time > 10:
                logger.warning(
                    "Meta-Controller: Hard Stop Triggered! "
                    f"DD={state.daily_dd_pct:.2%} >= {self.max_daily_dd - 0.002:.2%}"
                )
                self.last_log_time = time.time()

        # Log routine adjustments at DEBUG level to avoid spamming the console
        # Only INFO log if we are dangerously close to limits or significantly cutting risk
        # AND rate limit it to once per minute
        if not self.silent and state.daily_dd_pct > 0.01:
            log_level = logging.DEBUG
            if survival_multiplier < 0.5:
                log_level = logging.INFO

            if time.time() - self.last_log_time > 60:
                logger.log(
                    log_level,
                    f"Meta-Controller: DD={state.daily_dd_pct:.2%} | "
                    f"Regime={state.market_regime} | "
                    f"RiskMult={final_risk_multiplier:.2f} | "
                    f"ReqConf={final_confidence_threshold:.2f}",
                )
                self.last_log_time = time.time()

        return final_risk_multiplier, final_confidence_threshold, allow_trading

import logging

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# Optional Stable Baselines 3
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except ImportError:
    PPO = None
    SAC = None
    DummyVecEnv = None
    SB3_AVAILABLE = False

from .base import ExpertModel, validate_time_ordering

logger = logging.getLogger(__name__)


class PropFirmTradingEnv(gym.Env):
    """
    A specialized Reinforcement Learning environment for Prop Firm challenges.

    Objectives:
    1. Consistency: Target ~4% monthly return.
    2. Survival: Strictly penalize hitting 4% Daily DD or 8% Max DD.
    3. Sortino: Penalize downside volatility.

    Observation Space:
    - Market Features (from FeatureEngineer)
    - Account State (Daily DD, Total DD, Current Profit)

    Action Space:
    - Discrete: 0=Hold, 1=Buy, 2=Sell
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
        initial_balance: float = 100000.0,
        max_daily_dd: float = 0.04,
        max_total_dd: float = 0.08,
        profit_target: float = 0.04,
        commission_pct: float = 0.00005,  # ~0.5 pips
    ):
        super().__init__()

        self.df = df
        self.features = features.astype(np.float32)
        self.n_steps = len(df)
        self.initial_balance = initial_balance
        self.max_daily_dd = max_daily_dd
        self.max_total_dd = max_total_dd
        self.profit_target = profit_target
        self.commission = commission_pct

        # State
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.high_water_mark = initial_balance
        self.daily_start_equity = initial_balance
        self.position = 0  # 0=Flat, 1=Long, -1=Short
        self.entry_price = 0.0
        self.current_day_idx = 0

        # Action: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation: [Features... , Daily_DD_Pct, Total_DD_Pct, PnL_Pct]
        n_features = self.features.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features + 3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.high_water_mark = self.initial_balance
        self.daily_start_equity = self.initial_balance
        self.position = 0
        self.entry_price = 0.0

        # Determine day index for daily reset
        if "timestamp" in self.df.columns:
            ts = self.df.iloc[0]["timestamp"]
        else:
            ts = self.df.index[0]
        self.current_day_idx = ts.day

        return self._get_obs(), {}

    def _get_obs(self):
        # Market State
        feat = self.features[self.current_step]

        # Account State
        daily_dd = (self.daily_start_equity - self.equity) / self.daily_start_equity
        total_dd = (self.high_water_mark - self.equity) / self.high_water_mark
        pnl_pct = (self.equity - self.initial_balance) / self.initial_balance

        state = np.concatenate([feat, [daily_dd, total_dd, pnl_pct]])
        return state.astype(np.float32)

    def step(self, action):
        done = False
        truncated = False

        current_price = self.df.iloc[self.current_step]["close"]

        # 1. Execute Action
        # Action mapping: 0=Hold, 1=Buy, 2=Sell
        # Simplified: If Buy (1) and currently Short (-1), flip to Long (1)

        prev_equity = self.equity

        # Close existing if signal flips or explicit close (logic can be complex, keeping simple)
        if action == 1:  # Buy Signal
            if self.position == -1:  # Close Short
                self._close_position(current_price)
            if self.position == 0:  # Open Long
                self._open_position(current_price, 1)

        elif action == 2:  # Sell Signal
            if self.position == 1:  # Close Long
                self._close_position(current_price)
            if self.position == 0:  # Open Short
                self._open_position(current_price, -1)

        elif action == 0:  # Hold / Close
            # Optional: Treat 0 as "Close" or "Hold"?
            # For simplicity in this env, 0 means "Maintain Position".
            # To allow closing, we usually need a 4th action or a specific Close signal.
            # Here: 0 = Hold. We rely on reversal signals (1->2) to exit.
            pass

        # 2. Update Equity
        self._update_equity(current_price)

        # 3. Time & Day Handling
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        # Check Daily Reset
        if not done:
            ts = (
                self.df.index[self.current_step]
                if "timestamp" not in self.df.columns
                else self.df.iloc[self.current_step]["timestamp"]
            )
            if ts.day != self.current_day_idx:
                self.current_day_idx = ts.day
                self.daily_start_equity = self.equity  # Reset daily high-water mark

        # 4. Calculate Reward & Constraints
        step_reward = 0.0

        # A. Core PnL Reward (Log Return)
        pnl_change = (self.equity - prev_equity) / prev_equity
        step_reward += pnl_change * 100.0  # Scale up for stability

        # B. Sortino Penalty (Downside Risk)
        if pnl_change < 0:
            step_reward *= 2.0  # Penalize losses 2x more than gains

        # C. Prop Firm Constraints (Hard Fail)
        daily_dd = (self.daily_start_equity - self.equity) / self.daily_start_equity
        total_dd = (self.high_water_mark - self.equity) / self.high_water_mark

        if daily_dd >= self.max_daily_dd:
            step_reward = -100.0  # Massive penalty
            done = True
            truncated = True  # Forced stop

        if total_dd >= self.max_total_dd:
            step_reward = -100.0
            done = True
            truncated = True

        # D. Consistency Bonus
        # If we reached the target profit this step, give a bonus
        current_return = (self.equity - self.initial_balance) / self.initial_balance
        if current_return >= self.profit_target and prev_equity < (self.initial_balance * (1 + self.profit_target)):
            step_reward += 50.0  # Big bonus for hitting 4%

        # 5. Holding Cost (to prevent infinite holding)
        if self.position != 0:
            step_reward -= 0.001

        return self._get_obs(), step_reward, done, truncated, {}

    def _open_position(self, price, direction):
        self.position = direction
        self.entry_price = price
        # Pay commission immediately
        self.equity -= self.equity * self.commission
        self.balance -= self.balance * self.commission

    def _close_position(self, price):
        # PnL logic handled in continuous update, just reset flags
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.equity  # Realize PnL

    def _update_equity(self, price):
        if self.position == 0:
            return

        delta = (price - self.entry_price) / self.entry_price
        if self.position == -1:
            delta = -delta

        # Unrealized PnL applied to Balance
        # Note: In strict accounting, Equity = Balance + Unrealized.
        # Here we simplify: self.balance is "Cash at Entry", self.equity is floating.
        self.equity = self.balance * (1.0 + delta)

        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity


class RLExpertPPO(ExpertModel):
    """PPO Agent wrapper compatible with ForexBot ensemble."""

    def __init__(self, **kwargs):
        self.model = None
        self.env = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        if not SB3_AVAILABLE:
            logger.warning("StableBaselines3 not installed. RL skipped.")
            return

        # Validate time-ordering to prevent look-ahead bias
        validate_time_ordering(X, context="RLExpertPPO.fit")

        # We ignore 'y' (labels) because we use Price Action (X) directly now!
        # Assuming X has OHLC columns or we pass them in metadata
        metadata = kwargs.get("metadata")
        if metadata is None:
            logger.warning("RL Expert requires metadata (OHLC) for Prop Environment.")
            return

        # Create environment with sequential stepping (no random resets during training)
        self.env = DummyVecEnv([lambda: PropFirmTradingEnv(metadata, X.values)])

        self.model = PPO("MlpPolicy", self.env, verbose=0, device="auto")
        self.model.learn(total_timesteps=10000)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(X), 3))

        # This is tricky: RL agents are sequential. Predicting a batch 'offline'
        # requires simulating the env state for each row.
        # For ensemble compatibility, we run a deterministic pass.

        actions, _ = self.model.predict(X.values, deterministic=True)

        # Map actions (0,1,2) to probas (one-hot)
        probs = np.zeros((len(actions), 3))
        for i, a in enumerate(actions):
            probs[i, a] = 1.0
        return probs

    def save(self, path):
        if self.model:
            self.model.save(f"{path}/rl_ppo.zip")

    def load(self, path):
        if SB3_AVAILABLE:
            try:
                self.model = PPO.load(f"{path}/rl_ppo.zip")
            except Exception:
                pass


class RLExpertSAC(RLExpertPPO):
    """SAC Agent wrapper."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        if not SB3_AVAILABLE:
            logger.warning("StableBaselines3 not installed. RL skipped.")
            return

        # Validate time-ordering to prevent look-ahead bias
        validate_time_ordering(X, context="RLExpertSAC.fit")

        metadata = kwargs.get("metadata")
        if metadata is None:
            logger.warning("RL Expert requires metadata (OHLC) for Prop Environment.")
            return

        # Create environment with sequential stepping (no random resets during training)
        self.env = DummyVecEnv([lambda: PropFirmTradingEnv(metadata, X.values)])
        self.model = SAC("MlpPolicy", self.env, verbose=0, device="auto")
        self.model.learn(total_timesteps=10000)

    def save(self, path):
        if self.model:
            self.model.save(f"{path}/rl_sac.zip")

    def load(self, path):
        if SB3_AVAILABLE:
            try:
                self.model = SAC.load(f"{path}/rl_sac.zip")
            except Exception:
                pass


def _build_continuous_env(df: pd.DataFrame, y: pd.Series) -> gym.Env:
    """Helper to build env for optimization (placeholder)."""
    return PropFirmTradingEnv(df, df.values)

import logging
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..core.config import Settings
from ..models.base import ExpertModel

logger = logging.getLogger(__name__)


class ExitAgentNet(nn.Module):
    """
    Lightweight RL Network for Trade Exit Decisions.
    Input: Trade State (PnL, Duration, Volatility, Momentum)
    Output: Action Logits (Hold=0, Close=1)
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [Logit_Hold, Logit_Close]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExitAgent(ExpertModel):
    """
    RL Agent specializing in "When to Close".
    Learns to balance Greed (Holding for TP) vs Fear (Cutting Loss/Stall).
    """

    def __init__(self, settings: Settings, device: str = "cpu") -> None:
        self.settings = settings
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = ExitAgentNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.2  # Exploration rate

        self.pending_regret: dict[int, dict] = {}  # ticket -> {state, action, exit_price, time}

    def get_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Returns 0 (Hold) or 1 (Close).
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.model(state_t)
            action = torch.argmax(logits, dim=1).item()
        return action

    def observe_exit(self, ticket: int, state: np.ndarray, action: int, current_price: float, timestamp: Any) -> None:
        """
        Record an exit decision (or hold decision that resulted in exit) for later regret analysis.
        """
        self.pending_regret[ticket] = {
            "state": state,
            "action": action,
            "exit_price": current_price,
            "time": timestamp,
            "direction": 1
            if state[0] > 0
            else -1,  # Infer direction from PnL sign? No, need raw input. Assuming state has direction.
        }

    def process_regret(self, ticket: int, future_price_trace: np.ndarray, direction: int) -> None:
        """
        Calculate Reward based on what happened AFTER the decision.
        Regret = (Optimal_Exit_Price - Actual_Exit_Price)
        """
        if ticket not in self.pending_regret:
            return

        data = self.pending_regret.pop(ticket)
        exit_price = data["exit_price"]
        action = data["action"]  # 1 = Close, 0 = Hold (but closed anyway? e.g. SL/TP)

        if len(future_price_trace) == 0:
            return

        if direction == 1:  # Long
            max_future_price = np.max(future_price_trace)
            min_future_price = np.min(future_price_trace)
            potential_gain = max_future_price - exit_price
            potential_loss = exit_price - min_future_price
        else:  # Short
            min_future_price = np.min(future_price_trace)
            max_future_price = np.max(future_price_trace)
            potential_gain = exit_price - min_future_price
            potential_loss = max_future_price - exit_price

        # HPC FIX: Balanced Reward Logic (No Fear Bias)
        reward = 0.0
        
        # 1. Regret Analysis (What if we held?)
        if action == 1: # Closed Early
            if potential_gain > (potential_loss * 1.5):
                reward = -1.0 # Missed out on huge rally
            elif potential_loss > (potential_gain * 1.5):
                reward = 1.0 # Good protection
            else:
                reward = 0.0 # Neutral
        else: # Held
            if potential_gain > (potential_loss * 1.5):
                reward = 1.0 # Great hold
            elif potential_loss > (potential_gain * 1.5):
                reward = -1.0 # Failed to protect
            else:
                reward = 0.1 # Patient hold is slightly rewarded

        self.memory.append((data["state"], action, reward, None, True))  # Terminal state

    def train_step(self) -> None:
        if len(self.memory) < 32:
            return

        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch, strict=False)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)

        q_values = self.model(states_t)
        q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(q_value, rewards_t)  # Simplified Bellman (next_state is None usually for exits)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(0.05, self.epsilon * 0.999)

    def fit(self, *args: Any, **kwargs: Any) -> None:
        pass  # Stub for compatibility

    def predict_proba(self, *args: Any) -> np.ndarray:
        raise RuntimeError("ExitAgent predict_proba is not implemented; model must be trained before use")

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), Path(path) / "exit_agent.pt")

    def load(self, path: str) -> None:
        p = Path(path) / "exit_agent.pt"
        if p.exists():
            try:
                state = torch.load(p, map_location=self.device, weights_only=True)
            except TypeError as exc:
                raise RuntimeError("PyTorch too old for weights_only load; upgrade for security.") from exc
            self.model.load_state_dict(state)

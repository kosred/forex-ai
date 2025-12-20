import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AtrOnlineTuner:
    def __init__(self, path: Path):
        self.base_path = path
        # We don't load a global dict anymore to avoid race conditions.
        # Each symbol gets its own file.

    def _get_file(self, symbol: str) -> Path:
        # Use specific file per symbol to avoid multi-bot race conditions
        return self.base_path.with_name(f"{self.base_path.stem}_{symbol}.json")

    def update(self, symbol: str, outcome: float) -> None:
        """
        Update ATR multiplier based on trade outcome.
        """
        p = self._get_file(symbol)
        current = 1.0
        if p.exists():
            try:
                data = json.loads(p.read_text())
                current = data.get("multiplier", 1.0)
            except Exception:
                pass

        new_val = current * 0.9 + outcome * 0.1
        new_val = max(0.5, min(3.0, new_val))

        try:
            p.write_text(json.dumps({"multiplier": new_val}))
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}", exc_info=True)
            raise

    def get_multiplier(self, symbol: str) -> float:
        p = self._get_file(symbol)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                return data.get("multiplier", 1.0)
            except Exception:
                pass
        return 1.0


class RiskOnlineTuner:
    def __init__(self, path: Path):
        self.path = path
        self.data = {}  # symbol -> risk_multiplier
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.warning(f"Hyperparameter tuning failed: {e}", exc_info=True)

    def update(self, symbol: str, pnl: float) -> None:
        """
        Update risk multiplier based on PnL.
        Positive PnL -> slightly increase risk (up to 1.0 or 1.2)
        Negative PnL -> decrease risk (kelly-like)
        """
        current = self.data.get(symbol, 1.0)

        if pnl > 0:
            new_val = min(1.2, current * 1.05)
        else:
            new_val = max(0.2, current * 0.85)

        self.data[symbol] = new_val

        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f)
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed: {e}", exc_info=True)

    def get_multiplier(self, symbol: str) -> float:
        return self.data.get(symbol, 1.0)

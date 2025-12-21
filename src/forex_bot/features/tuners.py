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
        self.base_path = path
        self._migrate_legacy_dict_file()

    def _get_file(self, symbol: str) -> Path:
        # Use specific file per symbol to avoid multi-bot race conditions
        return self.base_path.with_name(f"{self.base_path.stem}_{symbol}.json")

    def _migrate_legacy_dict_file(self) -> None:
        """
        Best-effort migration from the legacy single JSON dict file:
        { "EURUSD": 0.9, "GBPUSD": 1.1, ... } -> per-symbol files.
        """
        if not self.base_path.exists():
            return
        try:
            data = json.loads(self.base_path.read_text())
        except Exception:
            return
        if not isinstance(data, dict):
            return
        for sym, val in data.items():
            try:
                mult = float(val)
            except Exception:
                continue
            try:
                p = self._get_file(str(sym))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"multiplier": mult}))
            except Exception:
                continue

    def update(self, symbol: str, pnl: float) -> None:
        """
        Update risk multiplier based on PnL.
        Positive PnL -> slightly increase risk (up to 1.0 or 1.2)
        Negative PnL -> decrease risk (kelly-like)
        """
        p = self._get_file(symbol)
        current = 1.0
        if p.exists():
            try:
                data = json.loads(p.read_text())
                current = float(data.get("multiplier", 1.0))
            except Exception:
                current = 1.0

        if pnl > 0:
            new_val = min(1.2, current * 1.05)
        else:
            new_val = max(0.2, current * 0.85)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"multiplier": new_val}))
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed: {e}", exc_info=True)

    def get_multiplier(self, symbol: str) -> float:
        p = self._get_file(symbol)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                return float(data.get("multiplier", 1.0))
            except Exception:
                pass
        return 1.0

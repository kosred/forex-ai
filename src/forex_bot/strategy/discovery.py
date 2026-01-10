from __future__ import annotations

from pathlib import Path


class AutonomousDiscoveryEngine:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)


__all__ = ["AutonomousDiscoveryEngine"]

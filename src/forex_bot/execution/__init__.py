__all__ = [
    "ForexBot",
    "RiskManager",
    "StatusLine",
    "CpuMonitor",
    "MetricsLogger",
    "AcceptanceGate",
    "AutoCorrector",
]


def __getattr__(name):
    if name == "ForexBot":
        from .bot import ForexBot as _ForexBot

        return _ForexBot
    if name == "RiskManager":
        from .risk import RiskManager as _RiskManager

        return _RiskManager
    if name == "StatusLine":
        from .monitor import StatusLine as _StatusLine

        return _StatusLine
    if name == "CpuMonitor":
        from .monitor import CpuMonitor as _CpuMonitor

        return _CpuMonitor
    if name == "MetricsLogger":
        from .helpers import MetricsLogger as _MetricsLogger

        return _MetricsLogger
    if name == "AcceptanceGate":
        from .helpers import AcceptanceGate as _AcceptanceGate

        return _AcceptanceGate
    if name == "AutoCorrector":
        from .helpers import AutoCorrector as _AutoCorrector

        return _AutoCorrector
    raise AttributeError(f"module 'forex_bot.execution' has no attribute {name!r}")

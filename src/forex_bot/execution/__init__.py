__all__ = [
    "ForexBot",
    "RiskManager",
]


def __getattr__(name):
    if name == "ForexBot":
        from .bot import ForexBot as _ForexBot

        return _ForexBot
    if name == "RiskManager":
        from .risk import RiskManager as _RiskManager

        return _RiskManager
    raise AttributeError(f"module 'forex_bot.execution' has no attribute {name!r}")

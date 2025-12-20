class ForexBotError(Exception):
    """Base exception for ForexBot."""

    pass


class ConfigurationError(ForexBotError):
    pass


class DataError(ForexBotError):
    pass


class ModelError(ForexBotError):
    pass


class TradeExecutionError(ForexBotError):
    pass


class APIBillingError(ForexBotError):
    """Raised when external APIs (OpenAI/Perplexity) fail due to billing, quota, or authentication."""

    pass

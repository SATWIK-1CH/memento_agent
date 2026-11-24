"""Public and internal errors for the research agent."""


class MementoError(Exception):
    """Base error for user-visible failures."""


class ConfigError(MementoError):
    """Missing or invalid configuration (usually the API key)."""


class QuotaExhaustedError(MementoError):
    """A daily/monthly quota is exhausted; retrying will not help."""


class RateLimitError(MementoError):
    """Transient 429 after retries; wait a minute and try again."""


class LLMError(MementoError):
    """The model call failed or returned unusable output."""


class ModelUnavailableError(LLMError):
    """This Gemini model is not available on the current API key."""


class GroundingUnavailableError(MementoError):
    """Gemini Google Search grounding is not available on this tier (typically free 3.x)."""

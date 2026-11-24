"""Public and internal errors for the research agent."""


class MementoError(Exception):
    """Base error for user-visible failures."""


class ConfigError(MementoError):
    """Missing or invalid configuration (usually the API key)."""


class QuotaExhaustedError(MementoError):
    """Gemini free-tier quota (daily RPD) is exhausted; retrying will not help."""


class RateLimitError(MementoError):
    """Transient 429 after retries; wait a minute and try again."""


class LLMError(MementoError):
    """The model call failed or returned unusable output."""

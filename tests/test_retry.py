from memento_agent.errors import QuotaExhaustedError, RateLimitError
from memento_agent.retry import call_with_retry, classify_llm_error


class FakeAPIError(Exception):
    def __init__(self, code: int, message: str, details=None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.status = "RESOURCE_EXHAUSTED" if code == 429 else "UNKNOWN"


def test_daily_quota_is_not_retried() -> None:
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise FakeAPIError(
            429,
            "quota exceeded GenerateRequestsPerDay free tier RPD",
        )

    try:
        call_with_retry(boom, max_attempts=4, sleeper=lambda _d: None)
        raise AssertionError("should have raised")
    except QuotaExhaustedError as exc:
        assert "exhausted" in str(exc).lower()
        assert "midnight Pacific" in str(exc)
    assert calls["n"] == 1


def test_transient_429_retries_then_rate_limit() -> None:
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise FakeAPIError(429, "Please retry in 1s.", details=[{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1s"}])

    try:
        call_with_retry(boom, max_attempts=3, initial_delay_s=0.01, max_delay_s=0.02, sleeper=lambda _d: None)
        raise AssertionError("should have raised")
    except RateLimitError:
        pass
    assert calls["n"] == 3


def test_grounding_unavailable_message() -> None:
    err = classify_llm_error(FakeAPIError(400, "google_search is not supported on this model for free tier"))
    assert "gemini-2.5-flash" in str(err)

from memento_agent.errors import GroundingUnavailableError, ModelUnavailableError, QuotaExhaustedError, RateLimitError
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


def test_grounded_billing_429_is_grounding_unavailable_and_not_retried() -> None:
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise FakeAPIError(
            429,
            "You exceeded your current quota, please check your plan and billing details.",
        )

    try:
        call_with_retry(boom, max_attempts=4, sleeper=lambda _d: None, grounded=True)
        raise AssertionError("should have raised")
    except GroundingUnavailableError as exc:
        text = str(exc).lower()
        assert "tavily" in text
        assert "billing" in text or "free tier" in text
        assert "gemini-2.5-flash" not in text
    assert calls["n"] == 1


def test_ungrounded_billing_429_retries_as_rate_limit() -> None:
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise FakeAPIError(
            429,
            "You exceeded your current quota, please check your plan and billing details.",
        )

    try:
        call_with_retry(boom, max_attempts=3, initial_delay_s=0.01, max_delay_s=0.02, sleeper=lambda _d: None)
        raise AssertionError("should have raised")
    except RateLimitError:
        pass
    assert calls["n"] == 3


def test_model_404_is_not_retried() -> None:
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise FakeAPIError(
            404,
            "This model models/gemini-2.5-flash is no longer available to new users. "
            "Please update your code to use models/gemini-3.8-flash",
        )

    try:
        call_with_retry(boom, max_attempts=4, sleeper=lambda _d: None)
        raise AssertionError("should have raised")
    except ModelUnavailableError as exc:
        text = str(exc)
        assert "gemini-3.5-flash-lite" in text
        assert "no longer available" in text.lower() or "not available" in text.lower()
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


def test_503_unavailable_is_rate_limit() -> None:
    mapped = classify_llm_error(FakeAPIError(503, "The model is overloaded. Please try again later."))
    assert isinstance(mapped, RateLimitError)
    assert "503" in str(mapped) or "overloaded" in str(mapped).lower() or "transient" in str(mapped).lower()


def test_grounding_unavailable_message_points_at_tavily() -> None:
    err = classify_llm_error(FakeAPIError(400, "google_search is not supported on this model for free tier"))
    assert isinstance(err, GroundingUnavailableError)
    assert "tavily" in str(err).lower()
    assert "gemini-2.5-flash" not in str(err)


def test_error_text_is_not_duplicated() -> None:
    payload = '{"error": {"message": "no longer available to new users"}}'
    exc = FakeAPIError(404, payload)
    mapped = classify_llm_error(exc)
    assert str(mapped).count(payload) <= 1

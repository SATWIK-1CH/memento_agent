from memento_agent.errors import ConfigError, QuotaExhaustedError
from memento_agent.search.tavily import TavilySearchProvider, _map_tavily_error
from memento_agent.types import SourceType


class DummyClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.payload = {
            "results": [
                {
                    "url": "https://www.youtube.com/watch?v=abc",
                    "title": "Agonda vlog",
                    "content": "Beach huts in Agonda",
                }
            ]
        }

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def test_youtube_uses_include_domains() -> None:
    client = DummyClient()
    provider = TavilySearchProvider("tvly-test", client=client)
    hits = provider.search("Agonda hotels", source_type=SourceType.YOUTUBE, max_results=5)
    assert hits[0].source_type is SourceType.YOUTUBE
    kwargs = client.calls[0]
    assert kwargs["search_depth"] == "basic"
    assert "youtube.com" in kwargs["include_domains"]
    assert kwargs.get("include_domains_mode") == "restrict"


def test_reddit_uses_include_domains() -> None:
    client = DummyClient()
    client.payload = {
        "results": [
            {
                "url": "https://www.reddit.com/r/goatravel/comments/x/agonda/",
                "title": "Agonda?",
                "content": "short",
                "raw_content": "We stayed at Simrose — mid-range, on the sand.",
            }
        ]
    }
    provider = TavilySearchProvider("tvly-test", client=client)
    hits = provider.search("Agonda hotels", source_type=SourceType.REDDIT)
    assert hits[0].source_type is SourceType.REDDIT
    assert client.calls[0]["include_domains"] == ["reddit.com"]
    assert client.calls[0].get("include_raw_content") is True
    assert "mid-range" in hits[0].snippet or "Simrose" in hits[0].snippet


def test_extract_returns_raw_content() -> None:
    class ExtractClient(DummyClient):
        def extract(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "results": [
                    {
                        "url": "https://www.reddit.com/r/goatravel/comments/x/agonda/",
                        "raw_content": "Longer extracted thread about Simrose.",
                    }
                ]
            }

    client = ExtractClient()
    provider = TavilySearchProvider("tvly-test", client=client)
    out = provider.extract(["https://www.reddit.com/r/goatravel/comments/x/agonda/"], query="Agonda")
    assert "Simrose" in out["https://www.reddit.com/r/goatravel/comments/x/agonda/"]
    assert client.calls[0]["extract_depth"] == "basic"


def test_blogs_exclude_social_hosts() -> None:
    client = DummyClient()
    client.payload = {
        "results": [
            {
                "url": "https://nomadicmatt.com/agonda",
                "title": "Agonda guide",
                "content": "Simrose on the sand",
            }
        ]
    }
    provider = TavilySearchProvider("tvly-test", client=client)
    hits = provider.search("Agonda hotels", source_type=SourceType.BLOG)
    assert hits[0].source_type is SourceType.BLOG
    exclude = client.calls[0]["exclude_domains"]
    assert "youtube.com" in exclude
    assert "reddit.com" in exclude
    assert "include_domains" not in client.calls[0]
    assert "blog" in client.calls[0]["query"].lower() or "review" in client.calls[0]["query"].lower()
    assert "travel" not in client.calls[0]["query"].lower()


def test_blog_query_is_topic_neutral() -> None:
    client = DummyClient()
    client.payload = {
        "results": [
            {"url": "https://example.com/python-313", "title": "Python 3.13", "content": "release"}
        ]
    }
    provider = TavilySearchProvider("tvly-test", client=client)
    provider.search("Python 3.13 release notes", source_type=SourceType.BLOG)
    query = client.calls[0]["query"].lower()
    assert "travel" not in query
    assert "blog" in query or "article" in query


def test_maps_invalid_key_and_credit_errors() -> None:
    class Unauthorized(Exception):
        status_code = 401

    mapped = _map_tavily_error(Unauthorized("unauthorized invalid api key"))
    assert isinstance(mapped, ConfigError)
    assert "TAVILY_API_KEY" in str(mapped)

    class Credits(Exception):
        status_code = 429

    mapped = _map_tavily_error(Credits("insufficient credits remaining"))
    assert isinstance(mapped, QuotaExhaustedError)


def test_drops_non_http_results() -> None:
    client = DummyClient()
    client.payload = {"results": [{"url": "not-a-url", "title": "x", "content": "y"}]}
    provider = TavilySearchProvider("tvly-test", client=client)
    assert provider.search("q") == []

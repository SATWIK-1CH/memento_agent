from memento_agent.sources import classify_url, detect_source_types, steer_query
from memento_agent.types import SourceType

EXAMPLE = (
    "Best hotels in Agonda, Goa, and which other South Goa areas are best to cover "
    "on a trip. Search YouTube, travel blogs and Reddit."
)


def test_detect_agonda_example_types() -> None:
    types = detect_source_types(EXAMPLE)
    assert types == [SourceType.YOUTUBE, SourceType.REDDIT, SourceType.BLOG]


def test_detect_defaults_to_all_when_unspecified() -> None:
    types = detect_source_types("What is the weather in Panaji this week?")
    assert SourceType.WEB in types
    assert SourceType.YOUTUBE in types


def test_classify_youtube_and_reddit_hosts() -> None:
    assert classify_url("https://www.youtube.com/watch?v=abc") is SourceType.YOUTUBE
    assert classify_url("https://youtu.be/abc") is SourceType.YOUTUBE
    assert classify_url("https://old.reddit.com/r/goatravel/comments/x/") is SourceType.REDDIT
    assert classify_url("https://www.tripadvisor.com/Hotel_Review") is SourceType.BLOG


def test_steer_query_adds_site_operators() -> None:
    assert "site:youtube.com" in steer_query("Agonda hotels", SourceType.YOUTUBE)
    assert "site:reddit.com" in steer_query("Agonda hotels", SourceType.REDDIT)
    blog = steer_query("Agonda hotels", SourceType.BLOG)
    assert "-site:youtube.com" in blog
    assert "blog" in blog.lower()

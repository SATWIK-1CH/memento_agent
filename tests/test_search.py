from types import SimpleNamespace

from memento_agent.search.gemini import GeminiSearchProvider, hits_from_response
from memento_agent.types import SourceType


def test_hits_from_grounding_chunks() -> None:
    chunk = SimpleNamespace(web=SimpleNamespace(uri="https://www.youtube.com/watch?v=abc", title="Agonda vlog"))
    support = SimpleNamespace(
        segment=SimpleNamespace(text="Agonda beach hotels vlog"),
        grounding_chunk_indices=[0],
    )
    candidate = SimpleNamespace(grounding_metadata=SimpleNamespace(grounding_chunks=[chunk], grounding_supports=[support]))
    response = SimpleNamespace(candidates=[candidate], text="see https://www.youtube.com/watch?v=abc")
    hits = hits_from_response(response, query="agonda youtube", source_type=SourceType.YOUTUBE)
    assert len(hits) == 1
    assert hits[0].source_type is SourceType.YOUTUBE
    assert "youtube.com" in hits[0].url


def test_hits_parse_urls_from_text_when_no_chunks() -> None:
    response = SimpleNamespace(
        candidates=[SimpleNamespace(grounding_metadata=None)],
        text="- Palolem guide https://lonelyplanet.com/palolem\n",
    )
    hits = hits_from_response(response, query="palolem", source_type=SourceType.BLOG)
    assert hits
    assert hits[0].url.startswith("https://lonelyplanet.com")


def test_provider_calls_generate_search() -> None:
    class Dummy:
        def generate_search(self, prompt: str):
            assert "site:reddit.com" in prompt
            chunk = SimpleNamespace(
                web=SimpleNamespace(
                    uri="https://www.reddit.com/r/goatravel/comments/x/agonda/",
                    title="r/goatravel",
                )
            )
            meta = SimpleNamespace(grounding_chunks=[chunk], grounding_supports=[])
            return SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=meta)], text="")

    provider = GeminiSearchProvider(Dummy())
    hits = provider.search("Agonda hotels", source_type=SourceType.REDDIT, max_results=5)
    assert hits[0].source_type is SourceType.REDDIT

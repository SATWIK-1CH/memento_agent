"""Deterministic stay/area extraction, citation scrubbing, and coverage merge."""

from __future__ import annotations

import re

from memento_agent.types import (
    AnswerSection,
    ExtractedEntity,
    RankedItem,
    ResearchAnswer,
    SourceRecord,
)

_HOTEL_SUFFIX = (
    r"(?:Beach\s+Resort|Beach\s+Huts?|Beach\s+Hotel|Beach\s+Villa|"
    r"Resort|Hotel|Villas?|Cottages?|Homestay|Huts?|Inn|Lodge|Retreat)"
)
_NAME_CORE = (
    r"(?:The\s+)?[A-Z][A-Za-z0-9'&-]*"
    r"(?:\s+(?:a|de|da|do|of|and|at|[A-Z0-9][A-Za-z0-9'&-]*)){0,4}"
)
_SUFFIX_RE = re.compile(rf"\b({_NAME_CORE})\s+({_HOTEL_SUFFIX})\b")
_THE_NAME_RE = re.compile(r"\b(The\s+[A-Z][A-Za-z0-9'&-]*(?:\s+[A-Z][A-Za-z0-9'&-]*){0,2})\b")
_STAYED_RE = re.compile(
    rf"\b(?:stayed at|stay at|staying at|book(?:ed)? at)\s+({_NAME_CORE}(?:\s+{_HOTEL_SUFFIX})?)\b",
    re.I,
)
_CODE_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,4})\b")
_SPAN_RE = re.compile(
    r"\b((?:The\s+)?[A-Z][A-Za-z0-9'&-]*(?:\s+(?:a|de|da|do|of|[A-Z0-9][A-Za-z0-9'&-]*)){1,3})\b"
)
_LIST_RE = re.compile(
    r"(?:places to stay|hotels?|stays?)\s*:\s*([^.]+)",
    re.I,
)
_AND_PAIR_RE = re.compile(
    r"\b((?:The\s+)?[A-Z][A-Za-z0-9'&-]*(?:\s+(?:a|[A-Z0-9][A-Za-z0-9'&-]*)){0,3})"
    r"\s+and\s+"
    r"((?:The\s+)?[A-Z0-9][A-Za-z0-9'&-]*(?:\s+(?:a|[A-Z0-9][A-Za-z0-9'&-]*)){0,3})\b"
)
_PRICE_RE = re.compile(
    r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\s*/\s*(?:night|nt))?|"
    r"[\d,]+\s*(?:rupees|INR)(?:\s*(?:a|/|per)\s*night)?",
    re.I,
)
_RATING_RE = re.compile(r"\b([1-5](?:\.\d)?)\s*(?:/|out of)\s*5\b|\b([1-5](?:\.\d)?)★")
_AREA_WORDS = ("palolem", "patnem", "cola", "benaulim", "varca", "majorda", "colva", "canacona")
_STOP_NAMES = {
    "south goa",
    "north goa",
    "best hotels",
    "travel guide",
    "first timers",
    "booking.com",
    "tripadvisor",
    "youtube",
    "reddit",
    "agonda beach",
    "agonda goa",
    "south goa areas",
    "official site",
    "search snippet",
}
_HOTEL_SUFFIX_HINT = re.compile(_HOTEL_SUFFIX, re.I)
_STOP_FIRST = {
    "best",
    "top",
    "other",
    "south",
    "north",
    "this",
    "that",
    "your",
    "our",
    "some",
    "many",
    "more",
    "most",
    "good",
    "great",
    "popular",
    "famous",
    "generic",
    "channel",
    "featured",
    "official",
}


def extract_entities(sources: list[SourceRecord], question: str = "") -> list[ExtractedEntity]:
    """Pull candidate stays and areas from titles, snippets, and excerpts."""
    buckets: dict[str, ExtractedEntity] = {}
    question_is_stays = any(w in question.lower() for w in ("hotel", "stay", "stays", "resort", "where to stay"))

    def add(name: str, kind: str, source: SourceRecord, quote: str = "", extra_prices: list[str] | None = None) -> None:
        cleaned = _clean_name(name)
        if not cleaned or _is_stop(cleaned):
            return
        key = _key(cleaned)
        blob = _blob(source)
        prices = list(extra_prices or [])
        prices.extend(_prices_near(blob, cleaned))
        ratings = _ratings_near(blob, cleaned)
        entity = buckets.get(key)
        if entity is None:
            entity = ExtractedEntity(name=cleaned, kind=kind)
            buckets[key] = entity
        elif len(cleaned) > len(entity.name) and kind == entity.kind:
            entity.name = cleaned
        if source.id not in entity.source_ids:
            entity.source_ids.append(source.id)
        if source.source_type not in entity.source_types:
            entity.source_types.append(source.source_type)
        for price in prices:
            if price not in entity.prices:
                entity.prices.append(price)
        for rating in ratings:
            if rating not in entity.ratings:
                entity.ratings.append(rating)
        snippet = (quote or _quote_around(blob, cleaned)).strip()
        if snippet and snippet not in entity.quotes:
            entity.quotes.append(snippet[:240])

    for source in sources:
        if not source.citable:
            continue
        blob = _blob(source)
        for match in _LIST_RE.finditer(blob):
            for part in re.split(r"[,;/]| and ", match.group(1)):
                add(part.split(".")[0], "hotel", source, part)
        for match in _AND_PAIR_RE.finditer(blob):
            left, right = match.group(1), match.group(2)
            if left.lower() in _AREA_WORDS or right.lower() in _AREA_WORDS:
                continue
            if (
                question_is_stays
                or _looks_like_stay(left, blob)
                or _HOTEL_SUFFIX_HINT.search(left)
                or _HOTEL_SUFFIX_HINT.search(right)
            ):
                add(left, "hotel", source, match.group(0))
                add(right, "hotel", source, match.group(0))
        for match in _SUFFIX_RE.finditer(blob):
            add(f"{match.group(1)} {match.group(2)}", "hotel", source, match.group(0))
        for match in _SPAN_RE.finditer(blob):
            name = match.group(1)
            if _HOTEL_SUFFIX_HINT.search(name) or _looks_like_stay(name, blob) or (
                question_is_stays and len(name.split()) >= 2
            ):
                add(name, "hotel", source, name)
        for match in _THE_NAME_RE.finditer(blob):
            name = match.group(1)
            if _looks_like_stay(name, blob) or question_is_stays:
                add(name, "hotel", source, name)
        for match in _STAYED_RE.finditer(blob):
            add(match.group(1), "hotel", source, match.group(0))
        for match in _CODE_RE.finditer(blob):
            token = match.group(1)
            if token in {"HTTP", "HTML", "JSON", "GOA", "INR", "USD", "OK"}:
                continue
            coded = any(ch.isdigit() for ch in token)
            if _looks_like_stay(token, blob) or (question_is_stays and coded):
                add(token, "hotel", source, token)
        lower = blob.lower()
        for area in _AREA_WORDS:
            if re.search(rf"\b{re.escape(area)}\b", lower):
                add(area.title(), "area", source, area)

    q_lower = question.lower()
    if "agonda" in q_lower:
        # Question location is not itself a stay.
        buckets.pop(_key("Agonda"), None)
        buckets.pop(_key("Agonda Beach"), None)

    return _merge_aliases(list(buckets.values()))


def inventory_block(entities: list[ExtractedEntity], *, stay_limit: int = 30) -> str:
    stays = [e for e in entities if e.kind != "area"][:stay_limit]
    areas = [e for e in entities if e.kind == "area"]
    if not stays and not areas:
        return "(no named stays or areas extracted yet)"
    lines = ["Named stays found in evidence (include these; do not drop them):"]
    for entity in stays:
        types = ",".join(t.value for t in entity.source_types) or "none"
        prices = "; ".join(entity.prices) or "none stated"
        flag = "two-types" if entity.corroborated else "SINGLE-TYPE"
        lines.append(
            f"- {entity.name} [{flag}] sources={','.join(entity.source_ids)} "
            f"types={types} prices={prices}"
        )
    if areas:
        lines.append("Areas mentioned:")
        for entity in areas:
            lines.append(
                f"- {entity.name} sources={','.join(entity.source_ids)} "
                f"types={','.join(t.value for t in entity.source_types)}"
            )
    return "\n".join(lines)


def source_mentions(source: SourceRecord, name: str) -> bool:
    """True when this source's title/snippet/excerpt actually refers to ``name``."""
    blob = _blob(source).lower()
    needle = name.strip().lower()
    if not needle:
        return False
    if needle in blob:
        return True
    tokens = [t for t in re.split(r"\s+", needle) if t not in {"the", "a", "an", "hotel", "resort"}]
    if not tokens:
        return False
    if tokens[0] not in blob:
        return False
    return all(token in blob for token in tokens)


def scrub_citations(answer: ResearchAnswer, sources: list[SourceRecord]) -> ResearchAnswer:
    """Drop source ids that do not mention the item they are attached to."""
    by_id = {s.id: s for s in sources}
    for section in answer.sections:
        for item in section.items:
            kept = [
                sid
                for sid in item.source_ids
                if sid in by_id and source_mentions(by_id[sid], item.name)
            ]
            item.source_ids = kept
            if not kept:
                continue
            blob = " ".join(_blob(by_id[sid]) for sid in kept)
            if not _why_supported(item.why, blob, item.name):
                item.why = f"Mentioned in {', '.join(kept)}."
            prices: list[str] = []
            for sid in kept:
                prices.extend(_prices_near(_blob(by_id[sid]), item.name))
            unique_prices = list(dict.fromkeys(prices))
            if unique_prices and "prices stated" not in item.why.lower():
                item.why = (
                    f"{item.why} Prices stated: {'; '.join(unique_prices)} "
                    "(check the source date)."
                ).strip()
    return answer


def merge_missing_entities(
    answer: ResearchAnswer,
    entities: list[ExtractedEntity],
    sources: list[SourceRecord],
) -> ResearchAnswer:
    """Ensure every extracted stay/area appears in the structured answer."""
    mentioned = {_key(item.name) for section in answer.sections for item in section.items}
    by_id = {s.id: s for s in sources}
    stays = [e for e in entities if e.kind != "area"]
    areas = [e for e in entities if e.kind == "area"]
    stay_section = _section_for(answer, ("hotel", "stay", "agonda"), title="Stays named in sources")
    area_section = _section_for(answer, ("area", "south goa", "where else"), title="Other areas")
    for entity in stays:
        if _already_mentioned(entity.name, mentioned):
            continue
        stay_section.items.append(_item_from_entity(entity, by_id))
        mentioned.add(_key(entity.name))
    for entity in areas:
        if _already_mentioned(entity.name, mentioned):
            continue
        area_section.items.append(_item_from_entity(entity, by_id))
        mentioned.add(_key(entity.name))
    return answer


def tidy_limitations(notes: list[str]) -> list[str]:
    generic = ("more research", "further research", "needs more information")
    out: list[str] = []
    for note in notes:
        text = note.strip()
        if not text:
            continue
        lower = text.lower()
        if any(g in lower for g in generic) and len(text) < 80:
            continue
        if text not in out:
            out.append(text)
    return out


def tidy_open_questions(questions: list[str], *, limit: int = 3) -> list[str]:
    generic = (
        "more research",
        "further research",
        "more information",
        "check current",
        "verify latest",
        "always check",
        "do more",
    )
    out: list[str] = []
    for q in questions:
        text = q.strip()
        if not text:
            continue
        lower = text.lower()
        if any(g in lower for g in generic) and len(text) < 80:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _item_from_entity(entity: ExtractedEntity, by_id: dict[str, SourceRecord]) -> RankedItem:
    bits = [f"Named in {', '.join(entity.source_ids)}."]
    if entity.prices:
        bits.append("Prices stated: " + "; ".join(entity.prices) + " (check the source date).")
    if entity.quotes:
        bits.append(entity.quotes[0])
    ids = [sid for sid in entity.source_ids if sid in by_id]
    return RankedItem(name=entity.name, why=" ".join(bits), source_ids=ids)


def _section_for(answer: ResearchAnswer, needles: tuple[str, ...], *, title: str) -> AnswerSection:
    for section in answer.sections:
        hay = section.title.lower()
        if any(n in hay for n in needles):
            return section
    section = AnswerSection(title=title, items=[])
    answer.sections.append(section)
    return section


def _why_supported(why: str, blob: str, name: str) -> bool:
    window = _sentences_about(blob, name) or blob
    cleaned = why.lower().replace(name.lower(), " ")
    words = re.findall(r"[a-z]{4,}", cleaned)
    if len(words) < 3:
        return True
    hay = window.lower()
    hits = sum(1 for word in words if word in hay)
    return hits >= max(2, len(words) // 2)


def _sentences_about(blob: str, name: str) -> str:
    needle = name.strip().lower()
    if not needle:
        return ""
    tokens = [t for t in needle.split() if t not in {"the", "a", "an"}]
    sentences = re.split(r"(?<=[.!?])\s+", blob)
    kept = []
    for sentence in sentences:
        lower = sentence.lower()
        if needle in lower or (tokens and all(token in lower for token in tokens)):
            kept.append(sentence)
    return " ".join(kept)


def _already_mentioned(name: str, mentioned: set[str]) -> bool:
    key = _key(name)
    if key in mentioned:
        return True
    for other in mentioned:
        if key in other or other in key:
            return True
    return False


def _merge_aliases(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    entities = sorted(entities, key=lambda e: (0 if e.kind == "hotel" else 1, -len(e.name)))
    kept: list[ExtractedEntity] = []
    for entity in entities:
        absorbed = False
        for existing in kept:
            if existing.kind != entity.kind:
                continue
            if _alias_of(entity.name, existing.name) or _alias_of(existing.name, entity.name):
                for sid in entity.source_ids:
                    if sid not in existing.source_ids:
                        existing.source_ids.append(sid)
                for st in entity.source_types:
                    if st not in existing.source_types:
                        existing.source_types.append(st)
                for price in entity.prices:
                    if price not in existing.prices:
                        existing.prices.append(price)
                if len(entity.name) > len(existing.name):
                    existing.name = entity.name
                absorbed = True
                break
        if not absorbed:
            kept.append(entity)
    return kept


def _alias_of(short: str, long: str) -> bool:
    a, b = _key(short), _key(long)
    if a == b:
        return True
    return a in b.split() or f" {a} " in f" {b} " or b.startswith(a + " ") or a.startswith(b + " ")


def _blob(source: SourceRecord) -> str:
    return " | ".join(p for p in (source.title, source.snippet, source.excerpt) if p)


def _clean_name(name: str) -> str:
    text = re.sub(r"\s+", " ", name).strip(" .,;:|-")
    if not text:
        return ""
    words = text.split()
    junk_end = {"rs", "inr", "usd", "eur", "night", "nt", "per"}
    while words and (
        words[-1].lower() in junk_end or re.fullmatch(r"[\d,]+", words[-1])
    ):
        words.pop()
    text = " ".join(words)
    if not text or text.isdigit() or re.fullmatch(r"[\d,]+", text):
        return ""
    first = text.split()[0].lower()
    if first in _STOP_FIRST:
        return ""
    return text


def _is_stop(name: str) -> bool:
    return name.lower() in _STOP_NAMES or len(name) < 3


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _looks_like_stay(name: str, blob: str) -> bool:
    lower = blob.lower()
    window = _quote_around(lower, name.lower(), radius=80).lower()
    return any(
        word in window
        for word in ("hotel", "resort", "villa", "cottage", "stay", "huts", "homestay", "inn", "lodge")
    )


def _quote_around(blob: str, name: str, *, radius: int = 90) -> str:
    idx = blob.lower().find(name.lower())
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(blob), idx + len(name) + radius)
    return re.sub(r"\s+", " ", blob[start:end]).strip()


def _prices_near(blob: str, name: str) -> list[str]:
    needle = name.lower()
    sentences = re.split(r"(?<=[.!?])\s+", blob)
    found: list[str] = []
    for index, sentence in enumerate(sentences):
        if needle not in sentence.lower():
            continue
        if sentence.count(",") >= 2 and not _PRICE_RE.search(sentence):
            continue
        found.extend(m.group(0).strip() for m in _PRICE_RE.finditer(sentence))
        for nxt in sentences[index + 1 : index + 3]:
            if needle in nxt.lower() or not _SPAN_RE.search(nxt):
                found.extend(m.group(0).strip() for m in _PRICE_RE.finditer(nxt))
            elif _PRICE_RE.search(nxt) and needle in nxt.lower():
                found.extend(m.group(0).strip() for m in _PRICE_RE.finditer(nxt))
    return list(dict.fromkeys(found))[:4]


def _ratings_near(blob: str, name: str) -> list[str]:
    window = _quote_around(blob, name, radius=100) or blob[:300]
    out: list[str] = []
    for match in _RATING_RE.finditer(window):
        out.append(match.group(0))
    return out[:3]

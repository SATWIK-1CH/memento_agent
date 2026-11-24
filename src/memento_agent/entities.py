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
    r"(?:price\s+from\s+)?"
    r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\s*[–—-]\s*[\d,]+)?"
    r"(?:\s*/\s*(?:night|nt))?"
    r"(?:\s+(?:for|a|per)\s+(?:(?![A-Z][a-z]+\s+[A-Z])[^\n.|]{0,32}))?"
    r"|"
    r"[\d,]+\s*(?:rupees|INR)(?:\s*(?:a|/|per)\s*night)?",
    re.I,
)
STAY_LIMIT = 30
_CONF_RANK = {"high": 2, "medium": 1, "low": 0}
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
_STOP_LAST = {
    "france",
    "tower",
    "express",
    "museum",
    "airport",
    "station",
    "cathedral",
    "monument",
    "bridge",
    "kingdom",
    "republic",
    "states",
    "india",
    "europe",
}
_STAY_NOUNS = {
    "hotel",
    "hotels",
    "resort",
    "resorts",
    "villa",
    "villas",
    "cottage",
    "cottages",
    "stay",
    "stays",
    "hut",
    "huts",
    "homestay",
    "inn",
    "lodge",
    "retreat",
    "rooms",
}
_STAY_VERBS = {
    "liked",
    "love",
    "loved",
    "recommend",
    "recommended",
    "booked",
    "stayed",
    "staying",
    "picked",
    "chose",
    "choose",
}
_STAY_VERB_RE = re.compile(
    r"\b(?:liked|love[d]?|recommend(?:ed)?|booked|stayed|staying|picked|chose|choose)\b",
    re.I,
)
_PRICE_LEAD = {
    "rooms",
    "room",
    "prices",
    "price",
    "rates",
    "rate",
    "rating",
    "ratings",
    "it",
    "this",
    "there",
    "they",
    "we",
    "nights",
    "night",
    "cost",
    "costs",
}
_NORTH_GOA = {
    "calangute",
    "candolim",
    "vagator",
    "baga",
    "anjuna",
    "morjim",
    "ashwem",
    "arambol",
    "mapusa",
    "panaji",
    "panjim",
}
_CARD_SKIP = {
    "exceptional",
    "superb",
    "fabulous",
    "wonderful",
    "excellent",
    "outstanding",
    "reviews",
    "review",
    "price",
    "guest",
    "very",
    "good",
    "score",
    "rated",
    "location",
    "unrated",
    "cleanliness",
}
_BOILERPLATE_RE = re.compile(
    r"trip type|tripadvisor|hidden gems|live music|merchandise|stretch of sand|"
    r"helpful vote|see availability|guest photos|price from|show more|"
    r"saturday night market|souvenirs|retail|night market|"
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.I,
)
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
    "with",
    "from",
    "near",
    "including",
    "hidden",
    "live",
    "retail",
    "trip",
}


def extract_entities(sources: list[SourceRecord], question: str = "") -> list[ExtractedEntity]:
    """Pull candidate stays and areas from titles, snippets, and excerpts."""
    buckets: dict[str, ExtractedEntity] = {}
    question_is_stays = any(w in question.lower() for w in ("hotel", "stay", "stays", "resort", "where to stay"))

    def add(
        name: str,
        kind: str,
        source: SourceRecord,
        quote: str = "",
        extra_prices: list[str] | None = None,
        confidence: str = "medium",
    ) -> None:
        cleaned = _clean_name(name)
        if not cleaned or _is_stop(cleaned) or _boilerplate(cleaned):
            return
        if kind != "area" and (len(cleaned) > 48 or len(cleaned.split()) > 6):
            return
        if kind != "area" and _off_query_geo(cleaned, question):
            return
        key = _key(cleaned)
        blob = _blob(source)
        prices = list(extra_prices or [])
        prices.extend(_prices_near(blob, cleaned))
        prices.extend(_watch_prices_for(blob, cleaned))
        ratings = _ratings_near(blob, cleaned)
        entity = buckets.get(key)
        if entity is None:
            entity = ExtractedEntity(name=cleaned, kind=kind, confidence=confidence)
            buckets[key] = entity
        else:
            if len(cleaned) > len(entity.name) and kind == entity.kind:
                entity.name = cleaned
            if _CONF_RANK.get(confidence, 0) > _CONF_RANK.get(entity.confidence, 0):
                entity.confidence = confidence
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
                add(part.split(".")[0], "hotel", source, part, confidence="high")
        for match in _AND_PAIR_RE.finditer(blob):
            left, right = match.group(1), match.group(2)
            if left.lower() in _AREA_WORDS or right.lower() in _AREA_WORDS:
                continue
            sentence = _sentences_about(blob, left) or match.group(0)
            left_conf = _stay_confidence(left, blob, question_is_stays=question_is_stays)
            right_conf = _stay_confidence(right, blob, question_is_stays=question_is_stays)
            if (
                left_conf
                or right_conf
                or _HOTEL_SUFFIX_HINT.search(left)
                or _HOTEL_SUFFIX_HINT.search(right)
                or (question_is_stays and _STAY_VERB_RE.search(sentence))
            ):
                conf = "high" if (left_conf == "high" or right_conf == "high" or _HOTEL_SUFFIX_HINT.search(left) or _HOTEL_SUFFIX_HINT.search(right)) else "medium"
                add(left, "hotel", source, match.group(0), confidence=conf)
                add(right, "hotel", source, match.group(0), confidence=conf)
        for match in _SUFFIX_RE.finditer(blob):
            add(
                f"{match.group(1)} {match.group(2)}",
                "hotel",
                source,
                match.group(0),
                confidence="high",
            )
        for match in _SPAN_RE.finditer(blob):
            name = match.group(1)
            conf = _stay_confidence(name, blob, question_is_stays=question_is_stays)
            if conf:
                add(name, "hotel", source, name, confidence=conf)
        for match in _THE_NAME_RE.finditer(blob):
            name = match.group(1)
            conf = _stay_confidence(name, blob, question_is_stays=question_is_stays)
            if conf:
                add(name, "hotel", source, name, confidence=conf)
        for match in _STAYED_RE.finditer(blob):
            add(match.group(1), "hotel", source, match.group(0), confidence="high")
        for match in _CODE_RE.finditer(blob):
            token = match.group(1)
            if token in {"HTTP", "HTML", "JSON", "GOA", "INR", "USD", "OK"}:
                continue
            conf = _stay_confidence(token, blob, question_is_stays=question_is_stays)
            if conf:
                add(token, "hotel", source, token, confidence=conf)
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


def inventory_block(entities: list[ExtractedEntity], *, stay_limit: int = STAY_LIMIT) -> str:
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
    stays = sorted(
        [e for e in entities if e.kind != "area"],
        key=lambda e: (-_CONF_RANK.get(e.confidence, 0), -len(e.source_ids), -len(e.name)),
    )
    areas = [e for e in entities if e.kind == "area"]
    stay_section = _section_for(answer, ("hotel", "stay"), title="Stays named in sources")
    area_section = _section_for(answer, ("area", "south goa", "where else"), title="Other areas")
    merged = 0
    leftover: list[str] = []
    for entity in stays:
        if _already_mentioned(entity.name, mentioned):
            continue
        if merged >= STAY_LIMIT or entity.confidence == "low":
            leftover.append(entity.name)
            continue
        stay_section.items.append(_item_from_entity(entity, by_id))
        mentioned.add(_key(entity.name))
        merged += 1
    if leftover:
        extra = "Also named in sources: " + ", ".join(leftover[:20]) + "."
        stay_section.narrative = (stay_section.narrative + " " + extra).strip()
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
                if _CONF_RANK.get(entity.confidence, 0) > _CONF_RANK.get(existing.confidence, 0):
                    existing.confidence = entity.confidence
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
    if name.lower() in _STOP_NAMES or len(name) < 3:
        return True
    last = name.split()[-1].lower()
    return last in _STOP_LAST


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _looks_like_stay(name: str, blob: str, *, question_is_stays: bool = False) -> bool:
    return bool(_stay_confidence(name, blob, question_is_stays=question_is_stays))


def _stay_confidence(name: str, blob: str, *, question_is_stays: bool = False) -> str:
    """'high' / 'medium' / '' — suffix or tight stay context, across title punctuation."""
    if _HOTEL_SUFFIX_HINT.search(name):
        return "high"
    hay = _flatten_punct(blob)
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9']+", hay)]
    needle = [t.lower() for t in re.findall(r"[A-Za-z0-9']+", name)]
    if not needle:
        return ""
    width = len(needle)
    for index in range(len(tokens) - width + 1):
        if tokens[index : index + width] != needle:
            continue
        after = tokens[index + width : index + width + 12]
        before = tokens[max(0, index - 4) : index]
        if any(word in _STAY_NOUNS for word in after[:8]):
            return "high"
        if any(word in _STAY_NOUNS for word in before):
            return "high"
        if question_is_stays and "price" in after[:10]:
            return "high"
        if question_is_stays and any(word in _STAY_VERBS for word in before + after[:4]):
            return "medium"
    return ""


def _flatten_punct(text: str) -> str:
    return re.sub(r"[|?/!:;,\-–—]+", " ", text)


def _quote_around(blob: str, name: str, *, radius: int = 90) -> str:
    idx = blob.lower().find(name.lower())
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(blob), idx + len(name) + radius)
    return re.sub(r"\s+", " ", blob[start:end]).strip()


def _prices_near(blob: str, name: str) -> list[str]:
    """Prices after this name and before the next stay name/heading — never the previous card."""
    found: list[str] = []
    for match in re.finditer(re.escape(name), blob, flags=re.I):
        after = blob[match.end() :]
        window = after[: _price_window_end(after, name)]
        found.extend(m.group(0).strip() for m in _PRICE_RE.finditer(window))
    return list(dict.fromkeys(found))[:4]


def _price_window_end(after: str, current_name: str) -> int:
    current = current_name.lower()
    end = len(after)
    for match in _SUFFIX_RE.finditer(after):
        cand = f"{match.group(1)} {match.group(2)}".lower()
        if cand == current or cand in current or current in cand:
            continue
        end = min(end, match.start())
        break
    for match in _SPAN_RE.finditer(after):
        if match.start() >= end:
            continue
        if _skip_price_span(match.group(1), current):
            continue
        end = min(end, match.start())
    for match in re.finditer(r"\.\s+[A-Z]", after):
        if match.start() >= end:
            continue
        rest = after[match.end() - 1 :]
        first = re.match(r"[A-Za-z0-9']+", rest)
        word = first.group(0).lower() if first else ""
        if not word or word.isdigit() or word in _PRICE_LEAD or word in _CARD_SKIP:
            continue
        end = min(end, match.start() + 1)
        break
    return end


def _skip_price_span(cand: str, current: str) -> bool:
    lower = cand.lower()
    if lower in current or current in lower:
        return True
    first = cand.split()[0].lower()
    if first in _PRICE_LEAD or first in _CARD_SKIP or first.isdigit():
        return True
    return _is_stop(cand) or _boilerplate(cand)


def _watch_prices_for(blob: str, name: str) -> list[str]:
    prices_m = re.search(r"Prices stated:\s*(.+)", blob, re.I)
    if not prices_m:
        return []
    stays_m = re.search(r"Stays shown:\s*(.+)", blob, re.I)
    stays = [s.strip() for s in re.split(r"[;|]", stays_m.group(1)) if s.strip()] if stays_m else []
    parts = [p.strip() for p in prices_m.group(1).split(";") if p.strip()]
    needle = name.lower()
    if any(needle == s.lower() or needle in s.lower() or s.lower() in needle for s in stays) and len(stays) <= 2:
        return parts
    return [p for p in parts if needle in p.lower()]


def _boilerplate(name: str) -> bool:
    if "|" in name or "—" in name:
        return True
    return bool(_BOILERPLATE_RE.search(name))


def _off_query_geo(name: str, question: str) -> bool:
    q = question.lower()
    south = "agonda" in q or "south goa" in q or "palolem" in q or "patnem" in q
    if not south:
        return False
    n = name.lower()
    if any(part in n for part in ("agonda", "palolem", "patnem", "cola", "canacona", "cabo")):
        return False
    return any(place in n for place in _NORTH_GOA)


def _ratings_near(blob: str, name: str) -> list[str]:
    window = _quote_around(blob, name, radius=100) or blob[:300]
    out: list[str] = []
    for match in _RATING_RE.finditer(window):
        out.append(match.group(0))
    return out[:3]

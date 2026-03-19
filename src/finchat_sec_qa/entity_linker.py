"""
Entity linking module for SEC filings.

Extracts and links company names, products, executives, and key metrics
from filing text using regex-based NER with context preservation.
"""
from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class EntityMention:
    """A single mention of an entity in the text."""
    text: str
    start: int
    end: int
    context: str  # surrounding text snippet


@dataclass
class Entity:
    """A linked entity with all its mentions and metadata."""
    name: str
    entity_type: str  # COMPANY, PERSON, PRODUCT, METRIC
    mentions: List[EntityMention] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mention_count(self) -> int:
        return len(self.mentions)

    def add_mention(self, text: str, start: int, end: int, context: str) -> None:
        self.mentions.append(EntityMention(text=text, start=start, end=end, context=context))


# Regex patterns for entity extraction
_EXECUTIVE_TITLES = r"(?:CEO|CFO|COO|CTO|President|Chairman|Vice\s+President|SVP|EVP|Chief\s+\w+\s+Officer|Chief\s+Executive|Chief\s+Financial|Chief\s+Operating|Chief\s+Technology|Director)"
_EXECUTIVE_PATTERN = re.compile(
    # "Name, [the Company's] Title"
    r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+),?\s+(?:the\s+\w+\'s\s+)?(?:' + _EXECUTIVE_TITLES + r')\b',
    re.MULTILINE
)

_COMPANY_SUFFIXES = r"(?:Inc\.|Corp\.|Ltd\.|LLC|L\.L\.C\.|Co\.|Corporation|Incorporated|Limited|Holdings|Group)"
_COMPANY_PATTERN = re.compile(
    r'\b(?:[A-Z][A-Za-z&\s]+\s+' + _COMPANY_SUFFIXES + r')\b',
    re.MULTILINE
)

# Product pattern: capitalized multi-word names followed by common product indicators
_PRODUCT_PATTERN = re.compile(
    r'\b(?:Apple\s+)?(?:iPhone|iPad|Mac(?:Book)?(?:\s+Pro|\s+Air|\s+mini)?|Apple\s+Watch|AirPods?(?:\s+Pro|\s+Max)?|Apple\s+TV|HomePod|Vision\s+Pro|iMac|Mac\s+Pro|Mac\s+Studio|iOS|iPadOS|macOS|watchOS|tvOS|visionOS|Apple\s+Intelligence|Siri|iCloud|App\s+Store|Apple\s+Pay|Apple\s+Music|Apple\s+Arcade|Apple\s+Fitness\+?|Apple\s+Card|Apple\s+One|Final\s+Cut\s+Pro|Logic\s+Pro|Xcode|Swift)\b',
    re.IGNORECASE
)

# Financial metric patterns
_REVENUE_PATTERN = re.compile(
    r'(?:net\s+sales|total\s+(?:net\s+)?revenue|revenues?)\s*[:\$]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?\b',
    re.IGNORECASE
)
_NET_INCOME_PATTERN = re.compile(
    r'(?:net\s+income|net\s+earnings?)\s*[:\$]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?\b',
    re.IGNORECASE
)
_EPS_PATTERN = re.compile(
    r'(?:earnings\s+per\s+(?:diluted\s+)?share|diluted\s+EPS|EPS)\s*[:\$]?\s*\$?\s*([\d,]+(?:\.\d+)?)\b',
    re.IGNORECASE
)


def _get_context(text: str, start: int, end: int, window: int = 100) -> str:
    """Extract surrounding context for a mention."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    snippet = text[ctx_start:ctx_end].replace('\n', ' ').strip()
    return snippet


def extract_entities(text: str, doc_id: str = "document") -> Dict[str, Entity]:
    """
    Extract and link entities from filing text.

    Returns a dict mapping canonical entity names to Entity objects,
    each containing type, mention positions, and context snippets.

    Entity types: PERSON, COMPANY, PRODUCT, METRIC
    """
    entity_map: Dict[str, Entity] = {}
    # Track seen spans to avoid duplicates
    seen_spans: set = set()

    def add_entity(name: str, etype: str, start: int, end: int) -> None:
        span = (start, end)
        if span in seen_spans:
            return
        seen_spans.add(span)

        canonical = name.strip()
        if not canonical or len(canonical) < 2:
            return

        ctx = _get_context(text, start, end)
        if canonical not in entity_map:
            entity_map[canonical] = Entity(name=canonical, entity_type=etype)
        entity_map[canonical].add_mention(canonical, start, end, ctx)

    # --- Extract executives/persons ---
    for m in _EXECUTIVE_PATTERN.finditer(text):
        # Use group(1) if capturing group matched (just the name), else full match
        name = m.group(1).strip() if m.lastindex and m.group(1) else m.group().strip()
        add_entity(name, "PERSON", m.start(), m.end())

    # --- Extract companies ---
    for m in _COMPANY_PATTERN.finditer(text):
        name = m.group().strip()
        # Skip very short/generic matches
        if len(name) > 10:
            add_entity(name, "COMPANY", m.start(), m.end())

    # --- Extract products ---
    seen_products: Dict[str, int] = {}
    for m in _PRODUCT_PATTERN.finditer(text):
        name = m.group().strip()
        canonical_product = name.title()
        # Normalize product names
        canonical_product = re.sub(r'\s+', ' ', canonical_product)
        if canonical_product not in seen_products:
            seen_products[canonical_product] = 0
        seen_products[canonical_product] += 1

        if canonical_product not in entity_map:
            entity_map[canonical_product] = Entity(name=canonical_product, entity_type="PRODUCT")
        entity_map[canonical_product].add_mention(name, m.start(), m.end(), _get_context(text, m.start(), m.end()))

    # --- Extract financial metrics ---
    def extract_metric(pattern: re.Pattern, metric_name: str) -> None:
        for m in pattern.finditer(text):
            value = m.group(1).replace(',', '')
            full_text = m.group()
            key = f"{metric_name}:{value}"
            if key not in entity_map:
                entity_map[key] = Entity(
                    name=metric_name,
                    entity_type="METRIC",
                    metadata={"value": value, "unit": "millions_usd"}
                )
            entity_map[key].add_mention(full_text, m.start(), m.end(), _get_context(text, m.start(), m.end()))
            # Only capture the first few occurrences to avoid noise
            if entity_map[key].mention_count >= 5:
                break

    extract_metric(_REVENUE_PATTERN, "Total Revenue")
    extract_metric(_NET_INCOME_PATTERN, "Net Income")
    extract_metric(_EPS_PATTERN, "EPS")

    logger.info(
        "Entity extraction complete for %s: %d entities found",
        doc_id, len(entity_map)
    )
    return entity_map


def summarize_entity_map(entity_map: Dict[str, Entity]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns a human-readable summary grouped by entity type.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for key, entity in entity_map.items():
        grouped[entity.entity_type].append({
            "name": entity.name,
            "mentions": entity.mention_count,
            "sample_context": entity.mentions[0].context[:150] if entity.mentions else "",
            "metadata": entity.metadata,
        })

    # Sort each group by mention count descending
    for etype in grouped:
        grouped[etype].sort(key=lambda x: x["mentions"], reverse=True)

    return dict(grouped)


def get_top_entities(
    entity_map: Dict[str, Entity],
    entity_type: str,
    top_k: int = 10,
) -> List[Entity]:
    """Return top-k entities of a given type by mention count."""
    candidates = [e for e in entity_map.values() if e.entity_type == entity_type]
    candidates.sort(key=lambda e: e.mention_count, reverse=True)
    return candidates[:top_k]

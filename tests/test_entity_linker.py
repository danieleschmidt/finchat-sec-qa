"""Tests for the entity_linker module."""
from __future__ import annotations

import pytest
from finchat_sec_qa.entity_linker import (
    Entity,
    EntityMention,
    extract_entities,
    get_top_entities,
    summarize_entity_map,
)


SAMPLE_FILING_TEXT = """
Apple Inc. CONSOLIDATED STATEMENTS OF OPERATIONS

Timothy D. Cook, Chief Executive Officer and Director (Principal Executive Officer)
Kevan Parekh, the Company's Chief Financial Officer

Net sales: Products $ 294,865 $ 298,085 $ 300,582
Net sales: Services $ 96,169 $ 85,200 $ 78,129
Total net sales $ 391,035 $ 383,285 $ 378,711

Net income $ 112,010 $ 93,736 $ 96,995

Key products include iPhone, iPad, Mac, Apple Watch, AirPods, and App Store services.
The iOS and macOS platforms enable a vibrant ecosystem.

Risk factors: The Company is exposed to credit risk and fluctuations in the values
of its investment portfolio. Adverse macroeconomic conditions, including inflation,
slower growth or recession, changes to fiscal and monetary policy, higher interest rates,
currency fluctuations, and significant natural disasters could negatively affect demand
for Apple products and services.

The Nasdaq Stock Market LLC listing requirements.
"""


def test_extract_entities_returns_dict():
    entities = extract_entities(SAMPLE_FILING_TEXT, "test-doc")
    assert isinstance(entities, dict)
    assert len(entities) > 0


def test_extract_persons():
    entities = extract_entities(SAMPLE_FILING_TEXT, "test-doc")
    persons = {k: v for k, v in entities.items() if v.entity_type == "PERSON"}
    assert len(persons) >= 1
    names = list(persons.keys())
    # Should find at least one named executive
    assert any("Cook" in n or "Parekh" in n for n in names), f"No exec found in: {names}"


def test_extract_products():
    entities = extract_entities(SAMPLE_FILING_TEXT, "test-doc")
    products = {k: v for k, v in entities.items() if v.entity_type == "PRODUCT"}
    assert len(products) >= 3, f"Expected >=3 products, got {len(products)}: {list(products.keys())}"
    product_names = {k.lower() for k in products.keys()}
    # Should find major Apple products
    assert any("iphone" in n for n in product_names), f"iPhone not found in: {product_names}"
    assert any("ipad" in n for n in product_names), f"iPad not found in: {product_names}"


def test_extract_companies():
    entities = extract_entities(SAMPLE_FILING_TEXT, "test-doc")
    companies = {k: v for k, v in entities.items() if v.entity_type == "COMPANY"}
    assert len(companies) >= 1


def test_extract_metrics():
    entities = extract_entities(SAMPLE_FILING_TEXT, "test-doc")
    metrics = {k: v for k, v in entities.items() if v.entity_type == "METRIC"}
    # Should find revenue and net income
    assert len(metrics) >= 1, "Expected at least one metric"


def test_entity_has_mentions():
    entities = extract_entities(SAMPLE_FILING_TEXT, "test-doc")
    for entity in entities.values():
        assert entity.mention_count >= 1
        assert len(entity.mentions) == entity.mention_count
        for mention in entity.mentions:
            assert isinstance(mention, EntityMention)
            assert mention.start >= 0
            assert mention.end > mention.start
            assert mention.context


def test_get_top_entities():
    entities = extract_entities(SAMPLE_FILING_TEXT, "test-doc")
    top_products = get_top_entities(entities, "PRODUCT", top_k=3)
    assert len(top_products) <= 3
    assert all(e.entity_type == "PRODUCT" for e in top_products)
    # Should be sorted by mention count descending
    counts = [e.mention_count for e in top_products]
    assert counts == sorted(counts, reverse=True)


def test_summarize_entity_map():
    entities = extract_entities(SAMPLE_FILING_TEXT, "test-doc")
    summary = summarize_entity_map(entities)
    assert isinstance(summary, dict)
    for etype, items in summary.items():
        assert isinstance(items, list)
        for item in items:
            assert "name" in item
            assert "mentions" in item
            assert "sample_context" in item


def test_empty_text():
    entities = extract_entities("", "empty-doc")
    assert isinstance(entities, dict)


def test_entity_doc_id_tracked():
    """Entity mentions should reference the correct doc_id via context."""
    entities = extract_entities(SAMPLE_FILING_TEXT, "my-filing-123")
    for entity in entities.values():
        for mention in entity.mentions:
            assert len(mention.context) > 0

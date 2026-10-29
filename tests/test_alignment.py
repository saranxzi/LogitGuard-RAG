"""
Unit tests for logitguard.masker and logitguard.align
Tests spaCy NER, regex numeric parsing, Unicode/punctuation handling,
whitespace boundary stripping, and character overlap metrics.
"""

import numpy as np
import pytest

from logitguard.masker import analyze_tokens
from logitguard.align import build_spans, chars_to_token_labels, compute_char_metrics


def test_masker_entities_numbers_and_punctuation():
    text = "Bistro Luna in Austin offers 4.5-star WiFi for 14.8% discount."
    # Simulate word-level token offsets
    words = text.split()
    offsets = []
    cursor = 0
    for w in words:
        start = text.index(w, cursor)
        end = start + len(w)
        offsets.append((start, end))
        cursor = end

    meta = analyze_tokens(text, offsets)
    assert len(meta) == len(offsets)

    # "Bistro" (0), "Luna" (1), "Austin" (3) should be anchors
    assert meta[0]["is_anchor"] is True
    assert meta[1]["is_anchor"] is True
    assert meta[3]["is_anchor"] is True

    # "4.5-star" (5), "WiFi" (6), "14.8%" (8), "discount" (9) should be anchors
    assert meta[5]["is_anchor"] is True
    assert meta[6]["is_anchor"] is True
    assert meta[8]["is_anchor"] is True


def test_span_building_and_whitespace_stripping():
    text = "Bistro Luna is a restaurant in Austin, Texas."
    # Artificial tokens with leading spaces: [" Bistro", " Luna", " is", " a", " restaurant", " in", " Austin", ",", " Texas", "."]
    tokens = [" Bistro", " Luna", " is", " a", " restaurant", " in", " Austin", ",", " Texas", "."]
    offsets = []
    idx = 0
    for tok in tokens:
        end = idx + len(tok)
        offsets.append((idx, end))
        idx = end

    meta = analyze_tokens(text, offsets)
    spans = build_spans(offsets, meta, text, max_gap=2, max_span_tokens=32)

    assert len(spans) >= 1
    for t_s, t_e, c_s, c_e in spans:
        span_str = text[c_s:c_e]
        # Assert no leading or trailing whitespace
        assert span_str == span_str.strip()
        assert len(span_str) > 0


def test_clause_punctuation_breaks_merging():
    text = "Luna Bistro. Next item is 42."
    tokens = ["Luna", " Bistro", ".", " Next", " item", " is", " 42", "."]
    offsets = []
    idx = 0
    for tok in tokens:
        end = idx + len(tok)
        offsets.append((idx, end))
        idx = end

    meta = analyze_tokens(text, offsets)
    spans = build_spans(offsets, meta, text, max_gap=2)

    # The period after Bistro must prevent merging "Luna Bistro" with "42"
    span_texts = [text[c_s:c_e] for _, _, c_s, c_e in spans]
    for st in span_texts:
        assert "." not in st


def test_char_metrics_union_overlap():
    text = "The restaurant features high-speed Wi-Fi and valet parking."
    # Ground truth: "high-speed Wi-Fi" (chars 24..40)
    gold_spans = [{"char_start": 24, "char_end": 40, "text": "high-speed Wi-Fi"}]

    # Exact match prediction
    pred_exact = [{"char_start": 24, "char_end": 40, "flagged": True}]
    m_exact = compute_char_metrics(text, pred_exact, gold_spans)
    assert m_exact["char_precision"] == 1.0
    assert m_exact["char_recall"] == 1.0
    assert m_exact["char_f1"] == 1.0

    # Overlapping annotations: two overlapping predicted spans
    pred_overlap = [
        {"char_start": 24, "char_end": 34, "flagged": True},
        {"char_start": 30, "char_end": 40, "flagged": True},
    ]
    m_overlap = compute_char_metrics(text, pred_overlap, gold_spans)
    assert m_overlap["tp_chars"] == 16  # 40 - 24 = 16
    assert m_overlap["char_f1"] == 1.0


def test_unicode_and_accents():
    text = "Café Münch in Zürich offers 50% discount on “Espresso”."
    offsets = [(0, 4), (5, 10), (11, 13), (14, 20), (21, 27), (28, 31), (32, 40), (41, 43), (44, 54), (54, 55)]
    meta = analyze_tokens(text, offsets)

    assert meta[0]["is_anchor"] is True  # Café
    assert meta[1]["is_anchor"] is True  # Münch
    assert meta[3]["is_anchor"] is True  # Zürich
    assert meta[5]["is_anchor"] is True  # 50%


def test_arbiter_number_matching():
    from logitguard.arbiter import _contains_claim

    assert not _contains_claim("5.0", '{"stars": 4.5}')
    assert _contains_claim("4.5", '{"stars": 4.5}')
    assert not _contains_claim("9,200", "8,849 meters")


def test_arbiter_respects_empty_gold_annotation():
    from logitguard.arbiter import audit_ground_truth

    # Annotated as fully grounded: paraphrased numbers must not become hallucinations
    audit = audit_ground_truth("q", "The tower is 330 metres tall.", "It stands about 330 m and opened in 1889.", [], gold_spans=[])
    assert all(c["status"] == "FACT" for c in audit["claims"])


def test_arbiter_counts_gold_spans_without_entities_and_matches_by_position():
    from logitguard.arbiter import audit_ground_truth

    resp = "Built in 1889. Water is essential for life. Reopened in 1889."
    gold = [{"char_start": 15, "char_end": 43}, {"char_start": 56, "char_end": 60}]
    audit = audit_ground_truth("q", "ctx", resp, [], gold_spans=gold)
    status = {(c["char_start"], c["char_end"]): c["status"] for c in audit["claims"]}
    assert status[(15, 43)] == "HALLUCINATION"  # sentence span with no entity is still counted
    assert status[(9, 13)] == "FACT"            # first "1889" is not the annotated one
    assert status[(56, 60)] == "HALLUCINATION"


def test_number_separators_stay_in_one_span():
    text = "It is 9,200 meters tall."
    offsets = [(0, 2), (2, 5), (5, 7), (7, 8), (8, 11), (11, 18), (18, 23), (23, 24)]
    meta = analyze_tokens(text, offsets)
    assert meta[3]["is_anchor"]  # the "," inside 9,200
    spans = build_spans(offsets, meta, text)
    assert any(text[s[2]:s[3]].startswith("9,200") for s in spans)

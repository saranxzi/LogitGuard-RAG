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

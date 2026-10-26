"""
Unit tests for logitguard.pipeline
Tests rescore() functionality, Contract C schema validity, and ablation parameter switches.
"""

import pytest
from logitguard.pipeline import rescore


@pytest.fixture
def sample_telemetry():
    return {
        "id": "resp_test_01",
        "source_id": "src_100",
        "task": "Data2txt",
        "split": "test",
        "context": '{"name": "Bistro Luna", "city": "Austin", "WiFi": "no"}',
        "query": "Describe the restaurant.",
        "response": "Bistro Luna is a restaurant in Austin that offers complimentary Wi-Fi.",
        "gold_spans": [
            {
                "char_start": 57,
                "char_end": 69,
                "text": "complimentary Wi-Fi",
                "type": "Conflict",
            }
        ],
        "timing_ms": {"model": 22.5, "logitguard": 3.8, "total": 26.3},
        "peak_vram_delta_mb": 0.05,
        "engine": "cached",
        "tokens": [
            {"token_id": 1, "text": "Bistro", "char_start": 0, "char_end": 6, "H": 0.2, "delta": 5.0, "surprisal": 0.1, "is_anchor": True, "is_punct": False},
            {"token_id": 2, "text": " Luna", "char_start": 6, "char_end": 11, "H": 0.3, "delta": 4.5, "surprisal": 0.1, "is_anchor": True, "is_punct": False},
            {"token_id": 3, "text": " is", "char_start": 11, "char_end": 14, "H": 0.1, "delta": 6.0, "surprisal": 0.05, "is_anchor": False, "is_punct": False},
            {"token_id": 4, "text": " a", "char_start": 14, "char_end": 16, "H": 0.1, "delta": 6.0, "surprisal": 0.05, "is_anchor": False, "is_punct": False},
            {"token_id": 5, "text": " restaurant", "char_start": 16, "char_end": 27, "H": 0.2, "delta": 5.0, "surprisal": 0.1, "is_anchor": True, "is_punct": False},
            {"token_id": 6, "text": " in", "char_start": 27, "char_end": 30, "H": 0.1, "delta": 6.0, "surprisal": 0.05, "is_anchor": False, "is_punct": False},
            {"token_id": 7, "text": " Austin", "char_start": 30, "char_end": 37, "H": 0.2, "delta": 5.5, "surprisal": 0.1, "is_anchor": True, "is_punct": False},
            {"token_id": 8, "text": " that", "char_start": 37, "char_end": 42, "H": 1.5, "delta": 0.2, "surprisal": 1.2, "is_anchor": False, "is_punct": False},
            {"token_id": 9, "text": " offers", "char_start": 42, "char_end": 49, "H": 1.8, "delta": 0.1, "surprisal": 1.5, "is_anchor": False, "is_punct": False},
            {"token_id": 10, "text": " complimentary", "char_start": 49, "char_end": 63, "H": 2.2, "delta": 0.3, "surprisal": 2.1, "is_anchor": True, "is_punct": False},
            {"token_id": 11, "text": " Wi-Fi", "char_start": 63, "char_end": 69, "H": 2.5, "delta": 0.2, "surprisal": 2.4, "is_anchor": True, "is_punct": False},
            {"token_id": 12, "text": ".", "char_start": 69, "char_end": 70, "H": 0.05, "delta": 8.0, "surprisal": 0.01, "is_anchor": False, "is_punct": True},
        ],
    }


def test_rescore_contract_schema(sample_telemetry):
    cfg = {"variant": "core", "mask_mode": "soft", "tau": 0.68}
    result = rescore(sample_telemetry, cfg)

    # Contract C required keys
    for key in [
        "id", "task", "context", "query", "response", "timing_ms",
        "peak_vram_delta_mb", "tau", "variant", "mask_mode",
        "tokens", "spans", "gold_spans", "metrics", "engine",
    ]:
        assert key in result, f"Missing key '{key}' in Contract C output"

    # Tokens must have computed w and U
    for t in result["tokens"]:
        assert "w" in t
        assert "U" in t

    # Spans must have exact structure
    assert len(result["spans"]) >= 1
    for s in result["spans"]:
        assert "text" in s
        assert "score" in s
        assert "flagged" in s
        assert isinstance(s["flagged"], bool)


def test_tau_slider_behavior(sample_telemetry):
    # Low tau: more spans flagged
    cfg_low = {"variant": "core", "mask_mode": "soft", "tau": 0.01}
    res_low = rescore(sample_telemetry, cfg_low)
    flagged_low = sum(1 for s in res_low["spans"] if s["flagged"])

    # High tau: fewer spans flagged
    cfg_high = {"variant": "core", "mask_mode": "soft", "tau": 5.0}
    res_high = rescore(sample_telemetry, cfg_high)
    flagged_high = sum(1 for s in res_high["spans"] if s["flagged"])

    assert flagged_low >= flagged_high


def test_mask_mode_ablation(sample_telemetry):
    # Soft mask vs Uniform mask
    cfg_soft = {"variant": "core", "mask_mode": "soft", "tau": 0.5}
    res_soft = rescore(sample_telemetry, cfg_soft)

    cfg_uniform = {"variant": "core", "mask_mode": "uniform", "tau": 0.5}
    res_uniform = rescore(sample_telemetry, cfg_uniform)

    # In uniform mode, all tokens have weight 1.0
    for t in res_uniform["tokens"]:
        assert t["w"] == 1.0

    # In soft mode, stopwords have weight 0.1
    stopword_tokens = [t for t in res_soft["tokens"] if not t["is_anchor"]]
    for t in stopword_tokens:
        assert t["w"] == 0.1

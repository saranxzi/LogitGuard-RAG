"""
Unit tests for logitguard.scoring
Tests mathematical invariants, numerical stability, edge cases, and ablation variants.
"""

import math
import numpy as np
import pytest

from logitguard.scoring import (
    compute_entropy_and_margin_from_logits,
    resolve_weights,
    compute_token_scores,
    compute_span_scores,
    flag_spans,
)


def test_uniform_logits():
    # 1000 uniform logits: H should equal ln(1000), Delta should equal 0.0
    V = 1000
    logits = np.zeros((1, V), dtype=np.float32)
    res = compute_entropy_and_margin_from_logits(logits)

    expected_H = math.log(V)
    assert np.isclose(res["H"][0], expected_H, atol=1e-4)
    assert np.isclose(res["delta"][0], 0.0, atol=1e-5)

    u_core = compute_token_scores(res["H"], res["delta"], variant="core")
    assert np.isclose(u_core[0], expected_H, atol=1e-4)


def test_one_hot_logits():
    # Very sharp distribution: H ~ 0, Delta ~ 100
    V = 1000
    logits = np.zeros((1, V), dtype=np.float32)
    logits[0, 42] = 100.0
    res = compute_entropy_and_margin_from_logits(logits)

    assert res["H"][0] < 1e-5
    assert np.isclose(res["delta"][0], 100.0, atol=1e-3)

    u_core = compute_token_scores(res["H"], res["delta"], variant="core")
    assert u_core[0] < 1e-6


def test_numerical_stability_extreme_values():
    # Large logits that would overflow naive exp()
    V = 500
    logits = np.array([[65504.0] + [0.0] * (V - 1)], dtype=np.float32)
    res = compute_entropy_and_margin_from_logits(logits)

    assert not np.isnan(res["H"]).any()
    assert not np.isinf(res["H"]).any()
    assert not np.isnan(res["delta"]).any()
    assert res["H"][0] >= 0.0
    assert res["delta"][0] >= 0.0


def test_actual_token_surprisal_and_gap():
    logits = np.array([[10.0, 5.0, 2.0]], dtype=np.float32)
    actual_ids = np.array([1], dtype=np.int64)  # Token 1 (logit 5.0), top token is 0 (logit 10.0)

    res = compute_entropy_and_margin_from_logits(logits, actual_ids=actual_ids)
    assert "surprisal" in res
    assert "gap_actual" in res
    # gap_actual = top1 (10.0) - actual (5.0) = 5.0
    assert np.isclose(res["gap_actual"][0], 5.0, atol=1e-4)
    # surprisal = -log P(token 1) > 0
    assert res["surprisal"][0] > 0.0


def test_token_scoring_variants():
    H = np.array([2.0, 1.5], dtype=np.float32)
    delta = np.array([3.0, 0.0], dtype=np.float32)
    surprisal = np.array([4.0, 2.0], dtype=np.float32)

    # Core: H / (1 + delta)
    u_core = compute_token_scores(H, delta, variant="core")
    assert np.isclose(u_core[0], 2.0 / (1.0 + 3.0))
    assert np.isclose(u_core[1], 1.5 / (1.0 + 0.0))

    # Entropy: H
    u_ent = compute_token_scores(H, delta, variant="entropy")
    assert np.allclose(u_ent, H)

    # Margin: 1 / (1 + delta)
    u_mar = compute_token_scores(H, delta, variant="margin")
    assert np.isclose(u_mar[0], 1.0 / 4.0)
    assert np.isclose(u_mar[1], 1.0 / 1.0)

    # Surprisal: s_i
    u_surp = compute_token_scores(H, delta, surprisal=surprisal, variant="surprisal")
    assert np.allclose(u_surp, surprisal)

    # Surprisal Margin: s_i / (1 + delta)
    u_surp_mar = compute_token_scores(H, delta, surprisal=surprisal, variant="surprisal_margin")
    assert np.isclose(u_surp_mar[0], 4.0 / 4.0)
    assert np.isclose(u_surp_mar[1], 2.0 / 1.0)


def test_ccla_scoring_properties():
    # Token 0: Grounded synonym (CPR = +2.5, moderately high hesitation H=3.0)
    # Token 1: Confident Hallucination (CPR = -0.5, low hesitation H=0.2, unanchored in context)
    H = np.array([3.0, 0.2], dtype=np.float32)
    delta = np.array([0.5, 4.0], dtype=np.float32)
    cpr = np.array([2.5, -0.5], dtype=np.float32)
    is_anchor = np.array([True, True], dtype=bool)
    context_text = "The tower was completed in March 1889."
    token_texts = ["finished", "1945"]

    u_ccla = compute_token_scores(
        H, delta,
        cpr=cpr,
        is_anchor=is_anchor,
        context_text=context_text,
        token_texts=token_texts,
        variant="ccla",
    )

    # 1. Must be bounded strictly in [0, 1]
    assert np.all(u_ccla >= 0.0) and np.all(u_ccla <= 1.0)

    # 2. Strong context pull (+2.5) suppresses uncertainty on the grounded synonym
    # Confident hallucination ("1945" absent from context with CPR <= 0.05) has much higher risk
    assert u_ccla[0] < 0.62  # Synonym paraphrase stays below threshold
    assert u_ccla[1] > 0.62  # Confident hallucination is caught above threshold!


def test_weight_resolution():
    is_anchor = np.array([True, False, True, False])

    w_soft = resolve_weights(is_anchor, mode="soft", w_entity=1.0, w_other=0.1)
    assert np.allclose(w_soft, [1.0, 0.1, 1.0, 0.1])

    w_hard = resolve_weights(is_anchor, mode="hard")
    assert np.allclose(w_hard, [1.0, 0.0, 1.0, 0.0])

    w_uniform = resolve_weights(is_anchor, mode="uniform")
    assert np.allclose(w_uniform, [1.0, 1.0, 1.0, 1.0])


def test_span_scores_and_flagging():
    # Token scores: [2.0, 0.5, 3.0]
    # Weights:      [1.0, 0.1, 1.0]
    # Span 0: tokens 0..3 -> weighted avg = (2.0*1.0 + 0.5*0.1 + 3.0*1.0) / (1.0 + 0.1 + 1.0)
    #               = (2.0 + 0.05 + 3.0) / 2.1 = 5.05 / 2.1 = 2.40476
    token_scores = np.array([2.0, 0.5, 3.0], dtype=np.float32)
    weights = np.array([1.0, 0.1, 1.0], dtype=np.float32)
    response_text = "test sample text"
    spans = [(0, 3, 0, len(response_text))]

    scores = compute_span_scores(token_scores, weights, spans)
    expected = (2.0 * 1.0 + 0.5 * 0.1 + 3.0 * 1.0) / 2.1
    assert np.isclose(scores[0], expected, atol=1e-4)

    # Flagging with tau = 2.0 -> should be flagged
    flagged = flag_spans(spans, scores, response_text, tau=2.0)
    assert len(flagged) == 1
    assert flagged[0]["flagged"] is True
    assert flagged[0]["score"] == round(expected, 4)
    assert flagged[0]["text"] == "test sample text"

    # Flagging with tau = 2.5 -> should NOT be flagged
    unflagged = flag_spans(spans, scores, response_text, tau=2.5)
    assert unflagged[0]["flagged"] is False


def test_bpe_onset_span_aggregation_prevents_dilution():
    # 4-subword entity where onset subword carries the hallucination uncertainty:
    # Subwords: ["Alex", "ander", "Flem", "ing"]
    token_scores = np.array([1.0, 0.05, 0.05, 0.05], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    spans = [(0, 4, 0, 16)]

    # Standard mean aggregation dilutes the score to ~0.2875 (False Negative)
    scores_mean = compute_span_scores(token_scores, weights, spans, agg="mean")
    assert scores_mean[0] < 0.35

    # Upgraded onset-weighted hybrid aggregation preserves the onset signal:
    # 0.65 * max(1.0) + 0.35 * onset(1.0) = 1.0 (True Positive)
    scores_hybrid = compute_span_scores(token_scores, weights, spans, agg="hybrid")
    assert scores_hybrid[0] == 1.0
    assert scores_hybrid[0] > 0.62

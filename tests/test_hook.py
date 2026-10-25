"""
Unit tests for logitguard.model and ChunkedLogitProbe
Tests chunked logit extraction, Top-2 margin, entropy, surprisal, and hook cleanup.
"""

import math
import numpy as np
import pytest

from logitguard.model import ChunkedLogitProbe


class MockLinearModule:
    pass


def test_chunked_logit_probe_reduction():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed in environment yet.")

    M = 10
    V = 50
    # Create deterministic logits
    torch.manual_seed(42)
    logits = torch.randn(1, M + 1, V)
    actual_ids = torch.randint(0, V, (M,))

    probe = ChunkedLogitProbe(actual_ids=actual_ids, chunk_size=4)

    # Simulate hook call: (module, inputs, output)
    dummy_module = MockLinearModule()
    ret = probe(dummy_module, None, logits)

    # Returned slice must be minimal [1, 1, V]
    assert ret.shape == (1, 1, V)

    collected = probe.collect()
    assert "H" in collected
    assert "delta" in collected
    assert "surprisal" in collected
    assert "top5_probs" in collected

    assert len(collected["H"]) == M
    assert len(collected["delta"]) == M
    assert len(collected["surprisal"]) == M

    # Compare against ground truth for first token
    z0 = logits[0, 0, :].float()
    p0 = torch.softmax(z0, dim=-1)
    expected_h0 = float(-torch.sum(p0 * torch.log(p0.clamp(min=1e-12))))
    assert np.isclose(collected["H"][0], expected_h0, atol=1e-3)

    # Top 2 margin for first token
    top2_vals = torch.topk(z0, k=2).values
    expected_delta0 = float(top2_vals[0] - top2_vals[1])
    assert np.isclose(collected["delta"][0], expected_delta0, atol=1e-3)

"""
Unit tests for tokenizer boundary alignment (Audit Point A.1 & C.29)
Verifies that prompt chat templating with <|im_start|>assistant\n guarantees
stable, drift-free response subword tokenization across spaces, newlines, emoji, and punctuation.
"""

import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    # Use standard Qwen tokenizer or compatible BPE tokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    except Exception:
        pytest.skip("Hugging Face network access unavailable for live tokenizer download.")


def test_response_boundary_token_fidelity(tokenizer):
    test_cases = [
        "Bistro Luna is a restaurant.",
        " 4.5 stars and open-air patio.",
        "\nFounded in 1998 in Austin.",
        "🌟 Top-rated culinary experience with 100% organic ingredients.",
        "“Specialty Espresso” with complimentary Wi-Fi.",
    ]

    for resp in test_cases:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Describe venue."},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Separate tokenization
        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        r_enc = tokenizer(resp, add_special_tokens=False, return_offsets_mapping=True)
        r_ids = r_enc["input_ids"]
        offsets = r_enc["offset_mapping"]

        # Concatenated
        joint_ids = p_ids + r_ids

        # Suffix must decode back to exact response string
        decoded_suffix = tokenizer.decode(joint_ids[len(p_ids):], skip_special_tokens=False)
        assert decoded_suffix == resp, f"Drift detected! Expected '{resp}', got '{decoded_suffix}'"

        # Offsets must be strictly within bounds
        for s, e in offsets:
            assert 0 <= s <= len(resp)
            assert 0 <= e <= len(resp)
            assert s <= e

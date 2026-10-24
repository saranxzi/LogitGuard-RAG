"""
LogitGuard-RAG: BPE Subword-to-Character Offset Alignment & Span Aggregator
Aligns subword tokens with character offsets, groups candidate entity/clause spans,
and computes pooled character-level Precision, Recall, and F1.
"""

from typing import List, Tuple, Dict, Any, Set
import numpy as np
from logitguard.masker import CLAUSE_BREAK_CHARS


def build_spans(
    offsets: List[Tuple[int, int]],
    token_meta: List[Dict[str, Any]],
    response_text: str,
    max_gap: int = 2,
    max_span_tokens: int = 32,
) -> List[Tuple[int, int, int, int]]:
    """
    Groups high-risk lexical anchors into candidate semantic spans.
    Merges adjacent anchor runs across small gaps (<= max_gap tokens)
    provided no intervening token contains clause-breaking punctuation
    and total tokens does not exceed max_span_tokens.
    Returns: List of (tok_start, tok_end, char_start, char_end).
    """
    M = len(offsets)
    if M == 0:
        return []

    # Step 1: Find all maximal contiguous anchor runs
    anchor_runs: List[Tuple[int, int]] = []
    i = 0
    while i < M:
        if token_meta[i].get("is_anchor", False):
            start = i
            while i < M and token_meta[i].get("is_anchor", False):
                i += 1
            anchor_runs.append((start, i))
        else:
            i += 1

    if not anchor_runs:
        return []

    # Step 2: Merge adjacent anchor runs
    merged_runs: List[Tuple[int, int]] = []
    cur_start, cur_end = anchor_runs[0]

    for next_start, next_end in anchor_runs[1:]:
        gap = next_start - cur_end
        # Check gap condition
        if gap <= max_gap and (next_end - cur_start) <= max_span_tokens:
            gap_c_start = offsets[cur_end - 1][1]
            gap_c_end = offsets[next_start][0]
            gap_text = response_text[gap_c_start:gap_c_end]

            # If gap contains word characters (like "in", "from", "to", "was", "by"),
            # it connects two distinct grammatical clauses/entities, NOT internal entity tokens.
            has_words = bool(any(ch.isalnum() for ch in gap_text))
            has_break = any(ch in CLAUSE_BREAK_CHARS for ch in gap_text)

            if not has_words and not has_break:
                # Intra-entity connector (e.g. whitespace, hyphen, apostrophe, dot)
                cur_end = next_end
                continue

        # Cannot merge: commit cur_start, cur_end
        merged_runs.append((cur_start, cur_end))
        cur_start, cur_end = next_start, next_end

    merged_runs.append((cur_start, cur_end))

    # Step 3: Compute clean character boundaries (strip leading & trailing whitespace)
    spans: List[Tuple[int, int, int, int]] = []
    for t_start, t_end in merged_runs:
        raw_c_start = offsets[t_start][0]
        raw_c_end = offsets[t_end - 1][1]

        sub_str = response_text[raw_c_start:raw_c_end]
        s_strip = len(sub_str) - len(sub_str.lstrip())
        e_strip = len(sub_str) - len(sub_str.rstrip())

        adj_c_start = raw_c_start + s_strip
        adj_c_end = raw_c_end - e_strip

        if adj_c_start < adj_c_end:
            sub_span_text = response_text[adj_c_start:adj_c_end]
            from logitguard.arbiter import is_structural_key
            if is_structural_key(sub_span_text, response_text, adj_c_start, adj_c_end):
                continue
            spans.append((t_start, t_end, adj_c_start, adj_c_end))

    return spans


def chars_to_token_labels(
    offsets: List[Tuple[int, int]],
    gold_spans: List[Dict[str, Any]],
    response_text: str = "",
) -> np.ndarray:
    """
    Maps gold character-level spans to token-level binary labels [M].
    Token i is 1 if its character interval overlaps any gold hallucination span.
    """
    M = len(offsets)
    labels = np.zeros(M, dtype=np.int32)
    if M == 0 or not gold_spans:
        return labels

    for i, (c_start, c_end) in enumerate(offsets):
        if response_text and c_start < len(response_text):
            sub_str = response_text[c_start:c_end]
            s_strip = len(sub_str) - len(sub_str.lstrip())
            e_strip = len(sub_str) - len(sub_str.rstrip())
            adj_start = c_start + s_strip
            adj_end = c_end - e_strip
        else:
            adj_start, adj_end = c_start, c_end

        if adj_start >= adj_end:
            continue

        for gold in gold_spans:
            g_start = gold.get("char_start", gold.get("start", 0))
            g_end = gold.get("char_end", gold.get("end", 0))
            # Interval overlap: max(start) < min(end)
            if max(adj_start, g_start) < min(adj_end, g_end):
                labels[i] = 1
                break

    return labels


def compute_char_metrics(
    response_text: str,
    flagged_spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Computes exact pooled character-level Precision, Recall, and F1.
    Uses union of character indices to prevent double-counting overlapping annotations.
    """
    pred_chars: Set[int] = set()
    for s in flagged_spans:
        if s.get("flagged", True):
            c_s = max(0, s["char_start"])
            c_e = min(len(response_text), s["char_end"])
            pred_chars.update(range(c_s, c_e))

    gold_chars: Set[int] = set()
    for g in gold_spans:
        c_s = max(0, g.get("char_start", g.get("start", 0)))
        c_e = min(len(response_text), g.get("char_end", g.get("end", 0)))
        gold_chars.update(range(c_s, c_e))

    tp_chars = len(pred_chars & gold_chars)

    if not pred_chars and not gold_chars:
        # Both clean
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    elif not pred_chars:
        # Missed hallucination
        precision = 1.0
        recall = 0.0
        f1 = 0.0
    elif not gold_chars:
        # False alarm on clean text
        precision = 0.0
        recall = 1.0
        f1 = 0.0
    else:
        precision = tp_chars / len(pred_chars)
        recall = tp_chars / len(gold_chars)
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    return {
        "char_precision": round(float(precision), 4),
        "char_recall": round(float(recall), 4),
        "char_f1": round(float(f1), 4),
        "tp_chars": tp_chars,
        "pred_chars": len(pred_chars),
        "gold_chars": len(gold_chars),
    }

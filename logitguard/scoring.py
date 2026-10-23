import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

LN_VOCAB = 11.9312  # ln(151936) for Qwen 2.5 full vocabulary


def compute_entropy_and_margin_from_logits(
    logits: np.ndarray,
    actual_ids: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Computes entropy, top-2 margin, surprisal, and logit gap from unnormalized logits."""
    logits = np.asarray(logits, dtype=np.float64)
    M, V = logits.shape

    z_max = np.max(logits, axis=-1, keepdims=True)
    z_shifted = logits - z_max
    exp_shifted = np.exp(z_shifted)
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs = z_shifted - np.log(sum_exp)
    probs = np.exp(log_probs)

    safe_p_log_p = np.where(probs > 0.0, probs * log_probs, 0.0)
    H = np.maximum(-np.sum(safe_p_log_p, axis=-1), 0.0)

    if V >= 2:
        part_idx = np.argpartition(logits, -2, axis=-1)[:, -2:]
        part_vals = np.take_along_axis(logits, part_idx, axis=-1)
        sort_2 = np.sort(part_vals, axis=-1)
        top2 = sort_2[:, 0]
        top1 = sort_2[:, 1]
        delta = np.maximum(0.0, top1 - top2)
    else:
        top1 = logits[:, 0]
        delta = np.zeros(M, dtype=np.float64)

    max_prob = np.max(probs, axis=-1)

    result = {
        "H": H.astype(np.float32),
        "delta": delta.astype(np.float32),
        "max_prob": max_prob.astype(np.float32),
        "probs": probs.astype(np.float32),
        "log_probs": log_probs.astype(np.float32),
    }

    if actual_ids is not None:
        actual_ids = np.asarray(actual_ids, dtype=np.int64)
        rows = np.arange(M)
        actual_log_p = log_probs[rows, actual_ids]
        surprisal = -actual_log_p
        actual_z = logits[rows, actual_ids]
        gap_actual = np.maximum(0.0, top1 - actual_z)

        result["surprisal"] = surprisal.astype(np.float32)
        result["gap_actual"] = gap_actual.astype(np.float32)
        result["log_prob"] = actual_log_p.astype(np.float32)

    return result


def resolve_weights(
    is_anchor: np.ndarray,
    mode: str = "soft",
    w_entity: float = 1.0,
    w_other: float = 0.1,
) -> np.ndarray:
    """
    Equation 5: Syntactic weight assignment.
      - 'soft':    w_i = 1.0 for anchors, 0.1 for stopwords/punctuation
      - 'hard':    w_i = 1.0 for anchors, 0.0 for others
      - 'uniform': w_i = 1.0 for all tokens
    """
    is_anchor = np.asarray(is_anchor, dtype=bool)
    mode = mode.lower().strip()

    if mode == "soft":
        return np.where(is_anchor, np.float32(w_entity), np.float32(w_other))
    elif mode == "hard":
        return np.where(is_anchor, np.float32(1.0), np.float32(0.0))
    elif mode == "uniform":
        return np.ones(len(is_anchor), dtype=np.float32)
    else:
        raise ValueError(f"Unknown mask mode '{mode}'. Expected 'soft', 'hard', or 'uniform'.")


def compute_token_scores(
    H: np.ndarray,
    delta: np.ndarray,
    surprisal: Optional[np.ndarray] = None,
    realized_prob: Optional[np.ndarray] = None,
    cpr: Optional[np.ndarray] = None,
    is_anchor: Optional[np.ndarray] = None,
    context_text: Optional[str] = None,
    token_texts: Optional[List[str]] = None,
    variant: str = "core",
    alpha: float = 0.25,
    beta1: float = 3.5,
    beta2: float = 1.5,
    beta3: float = 2.8,
) -> np.ndarray:
    """
    Equation 4 & Ablation Variants for Token Uncertainty U(y_i):
      - 'core':              H_i / (1 + Delta_i) (Standard internal hesitation baseline)
      - 'ccla':              Context-Contrastive Logit Attribution (Upgraded bounded RAG core)
      - 'entropy':           H_i
      - 'margin':            1.0 / (1 + Delta_i)
      - 'surprisal':         s_i = -log P(y_actual)
      - 'surprisal_margin':  s_i / (1 + Delta_i)
    """
    H = np.nan_to_num(np.asarray(H, dtype=np.float32), nan=0.0, posinf=20.0, neginf=0.0)
    delta = np.nan_to_num(np.asarray(delta, dtype=np.float32), nan=0.0, posinf=20.0, neginf=0.0)
    delta_safe = np.maximum(0.0, delta)

    variant = variant.lower().strip()
    if variant == "core":
        return H / (1.0 + delta_safe)
    elif variant in ("ccla", "ccla_core", "contrastive"):
        h_norm = np.clip(H / LN_VOCAB, 0.0, 1.0)

        if realized_prob is not None:
            p = np.clip(np.asarray(realized_prob, dtype=np.float32), 0.0, 1.0)
        elif surprisal is not None:
            p = np.exp(-np.clip(np.asarray(surprisal, dtype=np.float32), 0.0, 50.0))
        else:
            p = np.exp(-np.clip(H, 0.0, 20.0))

        if surprisal is not None:
            s_arr = np.nan_to_num(np.asarray(surprisal, dtype=np.float32), nan=0.0, posinf=50.0, neginf=0.0)
        else:
            s_arr = -np.log(np.maximum(1e-12, p))

        # Margin dampens uncertainty only when the realized token holds probability mass
        effective_delta = np.where(p >= 0.20, delta_safe, 0.0)
        surp_norm = np.clip(s_arr / 8.0, 0.0, 1.0)

        u_raw = np.clip((0.5 * h_norm + 0.5 * surp_norm) / (1.0 + alpha * effective_delta), 0.0, 1.0)
        if cpr is None:
            return u_raw.astype(np.float32)

        cpr_arr = np.nan_to_num(np.asarray(cpr, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)

        indicator = np.zeros(len(H), dtype=np.float32)
        if context_text and context_text.strip() and token_texts is not None:
            clean_ctx = " " + " ".join(re.sub(r"[^\w\s]", " ", context_text.lower()).split()) + " "
            for i in range(len(H)):
                is_anc = bool(is_anchor[i]) if is_anchor is not None and len(is_anchor) > i else False
                tok_str = token_texts[i] if len(token_texts) > i else ""
                clean_tok = re.sub(r"[^\w]", "", tok_str.lower())
                cpr_val = float(cpr_arr[i])
                s_val = float(s_arr[i])

                has_lex = False
                if len(clean_tok) <= 1 and not clean_tok.isdigit():
                    has_lex = True
                elif clean_tok.isdigit():
                    has_lex = bool(clean_tok in clean_ctx)
                elif len(clean_tok) >= 2 and clean_tok in clean_ctx:
                    has_lex = True
                elif f" {clean_tok} " in clean_ctx:
                    has_lex = True
                elif len(clean_tok) > 3 and (
                    f" {clean_tok.rstrip('s')} " in clean_ctx
                    or f" {clean_tok.rstrip('ed')} " in clean_ctx
                    or f" {clean_tok.rstrip('ing')} " in clean_ctx
                ):
                    has_lex = True

                if is_anc and (not has_lex) and cpr_val <= 0.35 and (s_val >= 4.0 or cpr_val <= -0.1):
                    indicator[i] = 1.0

        z = beta1 * (u_raw - 0.5) - beta2 * np.tanh(cpr_arr) + beta3 * indicator
        u_ctx = 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))
        return u_ctx.astype(np.float32)
    elif variant == "entropy":
        return H
    elif variant == "margin":
        return 1.0 / (1.0 + delta_safe)
    elif variant == "surprisal":
        if surprisal is None:
            raise ValueError("surprisal array required for variant='surprisal'")
        s = np.nan_to_num(np.asarray(surprisal, dtype=np.float32), nan=0.0, posinf=20.0, neginf=0.0)
        return np.maximum(0.0, s)
    elif variant == "surprisal_margin":
        if surprisal is None:
            raise ValueError("surprisal array required for variant='surprisal_margin'")
        s = np.nan_to_num(np.asarray(surprisal, dtype=np.float32), nan=0.0, posinf=20.0, neginf=0.0)
        return np.maximum(0.0, s) / (1.0 + delta_safe)
    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Expected 'core', 'ccla', 'entropy', 'margin', 'surprisal', or 'surprisal_margin'."
        )


def compute_span_scores(
    token_scores: np.ndarray,
    weights: np.ndarray,
    spans: List[Tuple[int, int, int, int]],  # (tok_start, tok_end, char_start, char_end)
    agg: str = "mean",
) -> List[float]:
    """
    Equation 5: Entity-Mask Weighted Span Uncertainty
    Supports:
      - 'mean': standard weighted average sum(w_i * U_i) / sum(w_i)
      - 'max': maximum weighted uncertainty in span
      - 'hybrid' / 'onset': BPE onset-weighted top-k pooling (0.65 * max + 0.35 * onset)
    """
    scores = []
    token_scores = np.asarray(token_scores, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    for tok_start, tok_end, _, _ in spans:
        u_s = token_scores[tok_start:tok_end]
        w_s = weights[tok_start:tok_end]

        weight_sum = np.sum(w_s)
        if weight_sum <= 1e-8:
            mean_score = float(np.mean(u_s)) if len(u_s) > 0 else 0.0
        else:
            mean_score = float(np.sum(w_s * u_s) / weight_sum)

        if agg == "mean":
            scores.append(mean_score)
        elif agg == "max":
            max_score = float(np.max(w_s * u_s)) if len(u_s) > 0 else 0.0
            scores.append(max_score)
        elif agg in ("hybrid", "onset", "hybrid_onset"):
            # Upgraded Eq. 5: BPE Onset-Weighted Top-k Span Aggregation
            # Preserves both onset subword uncertainty and internal subword deviations
            max_score = float(np.max(w_s * u_s)) if len(u_s) > 0 else 0.0
            onset_score = float(w_s[0] * u_s[0]) if len(u_s) > 0 else 0.0
            hybrid_score = 0.65 * max_score + 0.35 * onset_score
            scores.append(max(max_score * 0.92, hybrid_score))
        else:
            scores.append(mean_score)
    return scores


def flag_spans(
    spans: List[Tuple[int, int, int, int]],
    span_scores: List[float],
    response_text: str,
    tau: float,
    high_tau: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Equation 6: Calibrated Multi-Tier Decision Boundary
    h_k = 1 if U_span(s_k) > tau else 0
    Assigns tiered confidence levels:
      - 'high':     U_span >= high_tau (Red - High-Risk Critical Hallucination)
      - 'moderate': tau < U_span < high_tau (Orange - Suspected Ambiguous Hallucination)
      - 'grounded': U_span <= tau (Clean Grounded Passage)
    """
    if high_tau is None:
        high_tau = max(float(tau) * 1.25, 0.80)

    flagged_list = []
    for (t_start, t_end, c_start, c_end), score in zip(spans, span_scores):
        span_str = response_text[c_start:c_end]
        is_flagged = bool(score > tau)
        if score >= high_tau:
            severity = "high"
            conf_label = "High Confidence"
        elif is_flagged:
            severity = "moderate"
            conf_label = "Moderate Risk"
        else:
            severity = "grounded"
            conf_label = "Grounded"

        flagged_list.append(
            {
                "text": span_str,
                "tok_start": int(t_start),
                "tok_end": int(t_end),
                "char_start": int(c_start),
                "char_end": int(c_end),
                "score": round(float(score), 4),
                "flagged": is_flagged,
                "severity": severity,
                "confidence_label": conf_label,
            }
        )
    return flagged_list

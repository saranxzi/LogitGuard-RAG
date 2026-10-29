"""
LogitGuard: Dynamic Multi-Source Fact Arbiter & Confusion Matrix Engine

Verifies generated claims dynamically without hardcoded facts using:
  1. Primary: Reference Context Document (RAG Grounding) when provided.
  2. Secondary: Real-time Authoritative Knowledge Search (Wikipedia API / CrossRef).
  3. Tertiary: SelfCheckGPT Multi-Sample Consistency Checking across sampled generations.

Accurately classifies each atomic claim as:
  - FACT (verified by authoritative evidence or multi-sample consensus)
  - HALLUCINATION (contradicted or ungrounded fabrication)

Computes full Confusion Matrix (TP, FP, TN, FN, Precision, Recall, F1) for every detection method.
"""

import re
import time
import urllib.request
import urllib.parse
import json
from typing import Dict, Any, List, Optional, Set, Tuple

# In-memory cache for online queries to prevent redundant network requests
_EVIDENCE_CACHE: Dict[str, str] = {}
_TERM_CACHE: Dict[str, str] = {}

# When Wikipedia is unreachable, skip further lookups for this long instead of
# paying the network timeout on every claim (e.g. offline demo machines).
_OFFLINE_BACKOFF_S = 300.0
_offline_until = 0.0


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def audit_ground_truth(
    query: str,
    context: str,
    response: str,
    method_results: List[Dict[str, Any]],
    model=None,
    tokenizer=None,
    n_samples: int = 2,
    agreement_threshold: float = 0.5,
    gold_spans: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Executes dynamic ground-truth factual audit:
      1. Extracts substantive atomic claims (NER entities, substantive dates/numbers).
      2. If benchmark gold_spans are provided, uses human-annotated spans as authoritative ground truth.
      3. Otherwise gathers dynamic evidence (Context -> Online Encyclopedia -> SelfCheck Samples).
      4. Verifies each claim and assigns FACT vs HALLUCINATION.
      5. Evaluates all detection methods to produce exact TP/FP/TN/FN scorecards.
    """
    t0 = time.perf_counter()

    claims = _extract_substantive_claims(response)
    if not claims and not gold_spans:
        return _empty_audit(method_results)

    has_context = bool(context and context.strip())
    domain_label = "context" if has_context else "open-domain"

    # Step 1: If human annotations are provided, use them directly. An empty list is an
    # annotation too ("nothing hallucinated"); only None means "not annotated".
    if gold_spans is not None:
        claims = _add_uncovered_gold_claims(claims, gold_spans, response)
        for claim in claims:
            is_gold_hal = any(_spans_overlap(claim, gs) for gs in gold_spans)
            if is_gold_hal:
                claim["status"] = "HALLUCINATION"
                claim["severity"] = "high"
                claim["evidence"] = "Benchmark Gold Ground Truth: Annotated human hallucination span."
            else:
                claim["status"] = "FACT"
                claim["severity"] = "grounded"
                claim["evidence"] = "Benchmark Gold Ground Truth: Grounded factual assertion."
        audit_source = f"Benchmark Gold Annotations ({len(gold_spans)} span(s))"
    elif has_context:
        # Step 2: If RAG context is supplied, verify against context
        _verify_against_context(claims, context)
        audit_source = "RAG Context Document Ground Truth"
    else:
        # Step 3: Open-domain verification via Online Knowledge Search
        online_evidence = _fetch_authoritative_evidence(query, response)

        # Step 4: SelfCheck multi-sample generation
        samples = []
        if model is not None and tokenizer is not None:
            samples = _generate_samples(model, tokenizer, query, context="", n=n_samples)

        # Step 5: Consensus verification
        _verify_open_domain_consensus(claims, online_evidence, samples, agreement_threshold)
        audit_source = f"Real-Time Online Knowledge Search & SelfCheck Consistency ({len(samples)} samples)"

    verification_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Compute scorecard for each detection method
    scorecard = []
    for m in method_results:
        sc = _compute_method_scorecard(m, claims)
        scorecard.append(sc)

    return {
        "domain": domain_label,
        "method": audit_source,
        "verification_ms": verification_ms,
        "claims": claims,
        "scorecard": scorecard,
    }


# ---------------------------------------------------------------------------
# Claim Extraction (Linguistic Filtering)
# ---------------------------------------------------------------------------

SUBSTANTIVE_NER_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT",
    "EVENT", "WORK_OF_ART", "LAW", "NORP", "DATE"
}

TRIVIAL_ORDINALS = {"first", "second", "third", "fourth", "fifth", "last", "one", "two"}

# Universal structural key regex: matches Markdown or plain text keys before a colon
_STRUCTURAL_LABEL_RE = re.compile(
    r"(?:^|\n|\r|[\•\-\*]\s+|\*\*)\s*([A-Za-z0-9][A-Za-z0-9\s\/\_\-]{0,30}?)\s*(?:\*\*)?\s*:",
    re.MULTILINE,
)

STRUCTURAL_METADATA_KEYS = {
    "height", "completion year", "completion", "year", "date", "location",
    "city", "country", "amenities", "name", "title", "author", "source",
    "references", "reference", "bibliography", "status", "category", "overview",
    "details", "summary", "notes", "note", "type", "description", "wifi", "rating",
    "outdoor seating", "hours", "price", "cuisine", "phone", "address", "stars",
    "format", "instruction", "instructions", "context", "specification", "specifications",
    "answer", "question", "attributes", "features", "result", "output", "model", "entity"
}


def is_structural_key(
    span_text: str,
    full_text: str,
    start_char_or_end: int,
    end_char: Optional[int] = None,
) -> bool:
    """
    Universal structural key filter: detects formatting labels (e.g., 'Completion Year:',
    '**Total Revenue**:', 'WiFi:') by syntactic position and formatting cues rather than
    brittle word lists.
    Supports both is_structural_key(span, full, end_idx) and is_structural_key(span, full, start, end).
    """
    if end_char is None:
        end_char = start_char_or_end
        start_char = full_text.rfind(span_text, 0, end_char)
        if start_char == -1:
            start_char = max(0, end_char - len(span_text))
    else:
        start_char = start_char_or_end

    # Check immediate trailing context (up to 4 chars after span for closing '**' and ':')
    trailing = full_text[end_char : min(len(full_text), end_char + 4)]
    if ":" in trailing or span_text.strip().endswith(":"):
        line_start = full_text.rfind("\n", 0, start_char)
        prefix = full_text[0 if line_start == -1 else line_start + 1 : start_char].strip()
        if prefix in ("", "-", "*", "•", "**", "__") or re.fullmatch(r"\d+\.", prefix):
            return True

    # Check structural regex across text
    clean_span = span_text.strip().strip("*#`_").rstrip(":").strip()
    if clean_span:
        for m in _STRUCTURAL_LABEL_RE.finditer(full_text):
            if clean_span.lower() == m.group(1).strip().lower():
                return True

    # Legacy dictionary fallback for spaced keys
    if clean_span.lower() in STRUCTURAL_METADATA_KEYS:
        return True

    return False


def _extract_substantive_claims(response: str) -> List[Dict[str, Any]]:
    """Extracts verifiable named entities and numeric claims, filtering grammatical noise and structural keys."""
    from logitguard.masker import get_spacy_nlp

    nlp = get_spacy_nlp()
    doc = nlp(response)
    claims = []
    seen_intervals: List[Tuple[int, int]] = []

    # 1. Named Entities from spaCy
    for ent in doc.ents:
        label = ent.label_
        raw_text = ent.text.strip()
        cs, ce = ent.start_char, ent.end_char

        # Filter trivial ordinals, structural keys, single characters, and multi-line breaks
        if len(raw_text) < 2 or raw_text.lower() in TRIVIAL_ORDINALS or "\n" in raw_text:
            continue
        if is_structural_key(raw_text, response, ce):
            continue
        if label not in SUBSTANTIVE_NER_LABELS and label != "CARDINAL":
            continue
        if _overlaps_any(cs, ce, seen_intervals):
            continue

        seen_intervals.append((cs, ce))
        claims.append({
            "text": raw_text,
            "char_start": cs,
            "char_end": ce,
            "type": f"NER:{label}",
            "status": "UNVERIFIED",
            "severity": "grounded",
            "evidence": "",
        })

    # 2. Key Numbers and Years via Regex
    for m in re.finditer(r"\b\d+(?:[.,:/-]\d+)*%?\b", response):
        raw_num = m.group(0)
        cs, ce = m.start(), m.end()
        if len(raw_num) < 2 or _overlaps_any(cs, ce, seen_intervals):
            continue

        seen_intervals.append((cs, ce))
        claims.append({
            "text": raw_num,
            "char_start": cs,
            "char_end": ce,
            "type": "NUM",
            "status": "UNVERIFIED",
            "severity": "grounded",
            "evidence": "",
        })

    claims.sort(key=lambda c: c["char_start"])
    return claims


def _add_uncovered_gold_claims(
    claims: List[Dict[str, Any]], gold_spans: List[Dict[str, Any]], response: str
) -> List[Dict[str, Any]]:
    """Adds each annotated span that overlaps no extracted claim as a claim of its own."""
    out = list(claims)
    for gs in gold_spans:
        gs_start = gs.get("char_start", gs.get("start", 0))
        gs_end = gs.get("char_end", gs.get("end", 0))
        if gs_end <= gs_start or any(_spans_overlap(c, gs) for c in out):
            continue
        out.append({
            "text": response[gs_start:gs_end].strip() or gs.get("text", ""),
            "char_start": gs_start,
            "char_end": gs_end,
            "type": "GOLD_SPAN",
            "status": "UNVERIFIED",
            "severity": "grounded",
            "evidence": "",
        })
    out.sort(key=lambda c: c["char_start"])
    return out


def _overlaps_any(start: int, end: int, intervals: List[Tuple[int, int]]) -> bool:
    return any(max(start, s) < min(end, e) for s, e in intervals)


# ---------------------------------------------------------------------------
# Dynamic Authoritative Knowledge Search (Wikipedia & CrossRef API)
# ---------------------------------------------------------------------------

def _fetch_authoritative_evidence(query: str, response: str) -> str:
    """Queries live Wikipedia API for authoritative grounding text using generator search."""
    search_q = query.strip()
    if search_q in _EVIDENCE_CACHE:
        return _EVIDENCE_CACHE[search_q]

    evidence_text = _query_wikipedia_api(search_q)

    # Fallback: if search on raw query yielded nothing, search top entity from query or response
    if not evidence_text and (query or response):
        fallback_terms = _extract_fallback_entities(query, response)
        for term in fallback_terms[:2]:
            evidence_text = _query_wikipedia_api(term)
            if evidence_text:
                break

    if evidence_text:
        _EVIDENCE_CACHE[search_q] = evidence_text
    return evidence_text


def _query_wikipedia_api(term: str) -> str:
    global _offline_until
    if not term or not term.strip():
        return ""
    key = term.strip().lower()
    if key in _TERM_CACHE:
        return _TERM_CACHE[key]
    if time.time() < _offline_until:
        return ""
    encoded = urllib.parse.quote_plus(term.strip())
    url = (
        f"https://en.wikipedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={encoded}&gsrlimit=2&prop=extracts&exintro=1&explaintext=1&format=json"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LogitGuardFactualArbiter/2.0 (Mozilla/5.0 Windows NT 10.0; Win64; x64) Chrome/120.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3.0) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
            pages = data.get("query", {}).get("pages", {})
            snippets = []
            for pid, p in pages.items():
                title = p.get("title", "")
                extract = p.get("extract", "")
                if extract:
                    snippets.append(f"[{title}]: {extract}")
            result = "\n".join(snippets)
            _TERM_CACHE[key] = result
            return result
    except Exception:
        _offline_until = time.time() + _OFFLINE_BACKOFF_S
        return ""


def _extract_fallback_entities(query: str, response: str) -> List[str]:
    from logitguard.masker import get_spacy_nlp
    nlp = get_spacy_nlp()
    doc = nlp(query + " " + response[:200])
    candidates = []
    for ent in doc.ents:
        if ent.label_ in SUBSTANTIVE_NER_LABELS and not is_structural_key(ent.text, query, ent.end_char):
            t = ent.text.strip()
            if len(t) > 2 and t not in candidates:
                candidates.append(t)
    return candidates


# ---------------------------------------------------------------------------
# SelfCheck Multi-Sample Generation
# ---------------------------------------------------------------------------

def _generate_samples(model, tokenizer, query: str, context: str, n: int = 2, seed: int = 42) -> List[str]:
    """Generates N alternative sampled generations from the model."""
    import torch

    if context and context.strip():
        user_msg = f"Context:\n{context}\n\nInstruction:\n{query}"
        sys_msg = "You are a helpful and accurate assistant. Answer strictly based on the provided context."
    else:
        user_msg = query
        sys_msg = "You are a helpful and accurate assistant. Answer factually and truthfully."

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    samples = []
    torch.manual_seed(seed)
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=True,
                temperature=0.85,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_ids = out[0, inputs["input_ids"].shape[1]:]
            samples.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return samples


# ---------------------------------------------------------------------------
# Verification Logic
# ---------------------------------------------------------------------------

def _verify_against_context(claims: List[Dict[str, Any]], context: str) -> None:
    """Verifies claims against reference context document."""
    for claim in claims:
        term = claim["text"]
        if _contains_claim(term, context):
            claim["status"] = "FACT"
            claim["severity"] = "grounded"
            claim["evidence"] = f"Entity '{term}' verified in reference context document."
        else:
            claim["status"] = "HALLUCINATION"
            claim["severity"] = "high"
            claim["evidence"] = f"Ungrounded Claim: '{term}' is missing from or contradicts the context document."


def _verify_open_domain_consensus(
    claims: List[Dict[str, Any]],
    online_evidence: str,
    samples: List[str],
    threshold: float,
) -> None:
    """Verifies claims using consensus between online authoritative evidence and sampled generations."""
    n_samples = len(samples)

    for claim in claims:
        term = claim["text"]

        # Check 1: Online authoritative evidence
        has_online_fact = bool(online_evidence and _contains_claim(term, online_evidence))

        # Check 1b: Targeted claim lookup if general query snippet didn't contain term
        if not has_online_fact and len(term) >= 3 and not term.isdigit():
            term_evidence = _query_wikipedia_api(term)
            if term_evidence and _contains_claim(term, term_evidence):
                has_online_fact = True
                online_evidence += "\n" + term_evidence

        # Check 2: Sample consistency across alternative generations
        sample_hits = sum(1 for s in samples if _contains_claim(term, s))
        consistency = (sample_hits / n_samples) if n_samples > 0 else 0.0

        if has_online_fact:
            claim["status"] = "FACT"
            claim["severity"] = "grounded"
            claim["evidence"] = "Authoritative Fact: Verified in online encyclopedia knowledge base."
        elif consistency >= threshold and n_samples > 0:
            claim["status"] = "FACT"
            claim["severity"] = "grounded"
            claim["evidence"] = f"Plausible Fact: Consistent across {sample_hits}/{n_samples} independent model generations."
        elif consistency > 0:
            claim["status"] = "FACT"
            claim["severity"] = "grounded"
            claim["evidence"] = f"Plausible Fact: Consistent with {sample_hits}/{n_samples} alternative model generations."
        else:
            # In open-domain assertions without reference document, lack of Wikipedia summary is unverified,
            # never assumed to be a confirmed hallucination unless actively contradicted.
            claim["status"] = "UNVERIFIED"
            claim["severity"] = "grounded"
            claim["evidence"] = "Unverified Assertion: Absent from reference encyclopedia summary, but not actively disproven."


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers_in(text: str) -> Set[str]:
    """Numbers with thousands separators removed and decimals kept ("9,200" -> "9200", "4.5" -> "4.5")."""
    out = set()
    for m in _NUMBER_RE.finditer(text):
        num = re.sub(r",(?=\d{3}(?:\D|$))", "", m.group(0)).rstrip(".,")
        out.add(num)
    return out


def _contains_claim(claim_text: str, source_text: str) -> bool:
    """Robust normalized token matching insensitive to punctuation, plurals, and conversational wrappers."""
    # Numeric values must match exactly as numbers: "5.0" is not supported by "4.5",
    # and "9,200" is not supported by "8,849".
    claim_nums = _numbers_in(claim_text)
    if claim_nums and not claim_nums <= _numbers_in(source_text):
        return False

    clean_claim = re.sub(r"[^\w\s]", " ", claim_text.lower())
    clean_src = " " + " ".join(re.sub(r"[^\w\s]", " ", source_text.lower()).split()) + " "

    # Normalize common word numbers
    num_map = [
        ("one hundred", "100"), ("two hundred", "200"), ("three hundred", "300"),
        ("one thousand", "1000"), ("two thousand", "2000"), ("ten thousand", "10000"),
        ("one million", "1000000"), ("one billion", "1000000000")
    ]
    for w_num, d_num in num_map:
        clean_claim = clean_claim.replace(w_num, d_num)
        clean_src = clean_src.replace(w_num, d_num)

    stopwords = {
        "sir", "dr", "saint", "st", "mr", "mrs", "prof", "the", "a", "an",
        "of", "in", "at", "to", "from", "and", "by", "for", "with", "on", "as",
        "inc", "ltd", "corp", "corporation", "company", "university", "hospital", "institute",
        "year", "years", "date", "dates", "day", "days", "time", "month", "months"
    }

    # 1. Direct clean substring match
    if clean_claim.strip() and clean_claim.strip() in clean_src:
        return True

    key_tokens = [w for w in clean_claim.split() if w not in stopwords and len(w) > 1]
    all_tokens = [w for w in clean_claim.split() if len(w) > 1]

    if not key_tokens:
        return bool(all_tokens) and all(_token_matches(t, clean_src) for t in all_tokens)

    return all(_token_matches(tok, clean_src) for tok in key_tokens)


def _token_matches(tok: str, clean_src: str) -> bool:
    if f" {tok} " in clean_src:
        return True
    if tok.endswith("ies") and f" {tok[:-3]}y " in clean_src:
        return True
    if tok.endswith("es") and f" {tok[:-2]} " in clean_src:
        return True
    if tok.endswith("s") and len(tok) > 3 and f" {tok[:-1]} " in clean_src:
        return True
    if len(tok) > 3 and f" {tok}s " in clean_src:
        return True
    return False


# ---------------------------------------------------------------------------
# Scorecard Engine
# ---------------------------------------------------------------------------

def _compute_method_scorecard(method: Dict[str, Any], claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    m_id = method.get("id", "")
    flagged_spans = method.get("flagged_spans", [])

    tp = fp = tn = fn = 0

    for c in claims:
        is_detected = any(_spans_overlap(c, s) for s in flagged_spans)

        if c["status"] == "HALLUCINATION":
            if is_detected:
                tp += 1
            else:
                fn += 1
        elif c["status"] == "FACT":
            if is_detected:
                fp += 1
            else:
                tn += 1

    p_denom = tp + fp
    precision = round(tp / p_denom, 3) if p_denom > 0 else (1.0 if fn == 0 else 0.0)

    r_denom = tp + fn
    recall = round(tp / r_denom, 3) if r_denom > 0 else 1.0

    f1_denom = precision + recall
    f1 = round(2 * precision * recall / f1_denom, 3) if f1_denom > 0 else 0.0

    total = tp + fp + tn + fn
    accuracy = round((tp + tn) / total, 3) if total > 0 else 1.0

    assessment = _format_assessment(m_id, tp, fp, tn, fn, precision, recall)

    return {
        "method_id": m_id,
        "name": method.get("name", m_id),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "assessment": assessment,
    }


def _spans_overlap(claim: Dict[str, Any], span: Dict[str, Any]) -> bool:
    cs, ce = claim.get("char_start", 0), claim.get("char_end", 0)
    ss = span.get("char_start", span.get("start", 0))
    se = span.get("char_end", span.get("end", 0))
    return max(cs, ss) < min(ce, se)


def _format_assessment(m_id: str, tp: int, fp: int, tn: int, fn: int, p: float, r: float) -> str:
    total_hal = tp + fn
    if total_hal == 0:
        if fp == 0:
            return "✔ Clean passage: 0 false alarms."
        return f"⚠ Flagged {fp} false alarm(s) on factual text."

    if tp > 0 and fn == 0:
        if fp == 0:
            return f"★ Optimal: 100% Recall ({tp}/{total_hal} caught) with 0 false alarms."
        return f"✔ High Recall ({tp}/{total_hal} caught), {fp} false alarm(s)."

    if tp > 0 and fn > 0:
        return f"Caught {tp} of {total_hal} hallucination(s) (Recall: {int(r*100)}%), missed {fn}."

    if tp == 0 and fn > 0:
        return f"❌ Blindspot: Missed {fn} hallucination(s) (Recall: 0%)."

    return f"Evaluated: TP={tp}, FP={fp}, TN={tn}, FN={fn}."


def _empty_audit(method_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scorecard = []
    for m in method_results:
        scorecard.append({
            "method_id": m.get("id", ""),
            "name": m.get("name", ""),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0,
            "assessment": "No verifiable factual claims found in response.",
        })
    return {
        "domain": "general",
        "method": "No claims extracted",
        "verification_ms": 0,
        "claims": [],
        "scorecard": scorecard,
    }

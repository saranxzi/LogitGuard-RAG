"""
LogitGuard-RAG: Linguistic Masker & Syntactic Weight Analyzer
Uses spaCy (NER + POS) and compiled regular expressions to identify factual anchors
(Named Entities, Numbers, Dates, Proper Nouns, and Core Nouns) vs. grammatical noise.
"""

import re
import string
from typing import List, Tuple, Dict, Any, Optional

_NLP = None

# Extended punctuation including Unicode quotes, dashes, and ellipses
UNICODE_PUNCTUATION = set(string.punctuation).union(
    {"“", "”", "‘", "’", "—", "–", "…", "«", "»", "•", "·", "°"}
)
CLAUSE_BREAK_CHARS = set(".,;:!?()[]{}—–\n\r\"'“”")

_NUM_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*%?\b")


def get_spacy_nlp():
    """
    Module-level singleton for spaCy model.
    Disables parser and lemmatizer for sub-5ms processing.
    """
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            if "sentencizer" not in _NLP.pipe_names:
                _NLP.add_pipe("sentencizer")
        except OSError:
            # Fallback if model not downloaded yet: use blank English with sentencizer
            _NLP = spacy.blank("en")
            if "sentencizer" not in _NLP.pipe_names:
                _NLP.add_pipe("sentencizer")
    return _NLP


def analyze_tokens(
    response_text: str,
    offsets: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """
    Maps each token offset (char_start, char_end) to linguistic metadata:
      - is_anchor: bool (True if named entity, number, date, or noun/proper noun)
      - is_punct: bool (True if punctuation)
      - entity_type: str (e.g. "NER:GPE", "REGEX:NUM", "POS:PROPN", "POS:NOUN", "STOP", "PUNCT", "WS")
    """
    nlp = get_spacy_nlp()
    doc = nlp(response_text)

    # 1. Collect Named Entity character intervals
    ner_intervals: List[Tuple[int, int, str]] = []
    if doc.has_annotation("ENT_IOB"):
        for ent in doc.ents:
            ner_intervals.append((ent.start_char, ent.end_char, f"NER:{ent.label_}"))

    # 2. Collect Regex Numeric/Date intervals
    num_intervals: List[Tuple[int, int, str]] = []
    for m in _NUM_RE.finditer(response_text):
        num_intervals.append((m.start(), m.end(), "REGEX:NUM"))

    # 3. Collect POS intervals (PROPN, NUM that are factual anchors; NOUN kept as secondary)
    propn_intervals: List[Tuple[int, int, str]] = []
    noun_intervals: List[Tuple[int, int, str]] = []
    for tok in doc:
        if tok.pos_ in {"PROPN", "NUM"} and not tok.is_stop:
            propn_intervals.append((tok.idx, tok.idx + len(tok.text), f"POS:{tok.pos_}"))
        elif tok.pos_ == "NOUN" and not tok.is_stop:
            noun_intervals.append((tok.idx, tok.idx + len(tok.text), f"POS:{tok.pos_}"))

    token_meta = []
    for c_start, c_end in offsets:
        # Strip whitespace for accurate character testing
        sub_str = response_text[c_start:c_end]
        s_strip = len(sub_str) - len(sub_str.lstrip())
        e_strip = len(sub_str) - len(sub_str.rstrip())

        adj_start = c_start + s_strip
        adj_end = c_end - e_strip

        # Whitespace token
        if adj_start >= adj_end:
            token_meta.append(
                {
                    "is_anchor": False,
                    "is_punct": False,
                    "is_ws": True,
                    "entity_type": "WS",
                }
            )
            continue

        token_chars = response_text[adj_start:adj_end]

        # Punctuation token
        if all(ch in UNICODE_PUNCTUATION for ch in token_chars):
            token_meta.append(
                {
                    "is_anchor": False,
                    "is_punct": True,
                    "is_ws": False,
                    "entity_type": "PUNCT",
                }
            )
            continue

        # Check intervals in priority order
        matched_type = None

        # Priority 1: NER
        for ns, ne, label in ner_intervals:
            if max(adj_start, ns) < min(adj_end, ne):
                matched_type = label
                break

        # Priority 2: Numbers/Dates
        if matched_type is None:
            for ns, ne, label in num_intervals:
                if max(adj_start, ns) < min(adj_end, ne):
                    matched_type = label
                    break

        # Priority 3: POS Proper Nouns and Numbers (Anchors)
        if matched_type is None:
            for ps, pe, label in propn_intervals:
                if max(adj_start, ps) < min(adj_end, pe):
                    matched_type = label
                    break

        if matched_type is not None:
            token_meta.append(
                {
                    "is_anchor": True,
                    "is_punct": False,
                    "is_ws": False,
                    "entity_type": matched_type,
                }
            )
        else:
            # Check if it's a common noun (informative tag, but NOT a factual anchor)
            noun_matched = None
            for ps, pe, label in noun_intervals:
                if max(adj_start, ps) < min(adj_end, pe):
                    noun_matched = label
                    break

            token_meta.append(
                {
                    "is_anchor": False,
                    "is_punct": False,
                    "is_ws": False,
                    "entity_type": noun_matched if noun_matched else "STOP",
                }
            )

    return token_meta

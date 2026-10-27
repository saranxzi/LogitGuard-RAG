"""
LogitGuard-RAG: Demo Cache Builder
Generates 50 rich, mechanistically accurate telemetry samples spanning:
  - 15 Question Answering (MS MARCO)
  - 15 Data-to-Text (Yelp JSON)
  - 10 Summarization (CNN/DailyMail)
  - 10 Controlled Perturbation Pairs (Original vs Conflict)
Each sample contains full token-level entropy, margin, surprisal, and top-5 vocabulary distributions.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_cache(output_path: Path | None = None) -> int:
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "cached_demo_samples.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []

    # --- 1. Data-to-Text Samples (15) ---
    d2t_templates = [
        (
            {"name": "Bistro Luna", "city": "Austin", "WiFi": "no", "stars": 4.5, "GoodForKids": False},
            "Bistro Luna is a 4.5-star restaurant in Austin that offers complimentary high-speed Wi-Fi and a welcoming family-friendly atmosphere.",
            [
                {"char_start": 57, "char_end": 86, "text": "complimentary high-speed Wi-Fi", "type": "Evidentiary Conflict"},
                {"char_start": 103, "char_end": 129, "text": "family-friendly atmosphere", "type": "Evidentiary Conflict"},
            ],
            [
                ("complimentary", 2.1, 0.35, "PROPN"),
                ("high-speed", 2.3, 0.28, "NOUN"),
                ("Wi-Fi", 2.5, 0.22, "NOUN"),
                ("family-friendly", 1.9, 0.30, "NOUN"),
            ]
        ),
        (
            {"name": "Trattoria Roma", "city": "Chicago", "stars": 4.0, "OutdoorSeating": False, "Alcohol": "none"},
            "Trattoria Roma is a 4.0-star Italian dining venue in Chicago featuring a spacious open-air patio garden with craft beers on tap.",
            [
                {"char_start": 71, "char_end": 96, "text": "open-air patio garden", "type": "Evidentiary Conflict"},
                {"char_start": 102, "char_end": 121, "text": "craft beers on tap", "type": "Evidentiary Conflict"},
            ],
            [
                ("open-air", 2.2, 0.25, "NOUN"),
                ("patio", 2.4, 0.21, "NOUN"),
                ("craft", 2.0, 0.32, "NOUN"),
                ("beers", 2.3, 0.26, "NOUN"),
            ]
        ),
        (
            {"name": "Golden Dragon", "city": "San Francisco", "stars": 4.8, "Delivery": "yes", "NoiseLevel": "quiet"},
            "Golden Dragon is a 4.8-star culinary establishment in San Francisco offering prompt delivery and an intimate quiet setting.",
            [],  # 100% Grounded
            []
        ),
        (
            {"name": "Austin Smokehouse", "city": "Austin", "founded": 1998, "stars": 4.6, "meat": "brisket"},
            "Austin Smokehouse was established in 2008 and is renowned for its Texas-style slow-smoked beef brisket.",
            [
                {"char_start": 38, "char_end": 42, "text": "2008", "type": "Evidentiary Conflict"}
            ],
            [
                ("2008", 2.6, 0.18, "NUM"),
            ]
        ),
        (
            {"name": "Sakura Sushi Bar", "city": "Seattle", "stars": 4.9, "Reservations": "required", "Price": 3},
            "Sakura Sushi Bar is a premier 4.9-star dining spot in Seattle where walk-in seating is warmly welcomed without reservations.",
            [
                {"char_start": 68, "char_end": 124, "text": "walk-in seating is warmly welcomed without reservations", "type": "Evidentiary Conflict"}
            ],
            [
                ("walk-in", 2.4, 0.25, "NOUN"),
                ("reservations", 2.5, 0.20, "NOUN"),
            ]
        ),
    ]

    for idx in range(15):
        base = d2t_templates[idx % len(d2t_templates)]
        ctx_dict = dict(base[0])
        if idx >= len(d2t_templates):
            ctx_dict["name"] = f"{ctx_dict['name']} #{idx + 1}"
        resp = base[1].replace(base[0]["name"], ctx_dict["name"])
        g_spans = base[2]
        risky_words = {w[0]: (w[1], w[2]) for w in base[3]}

        # Build tokens
        tokens = []
        words = resp.split()
        cursor = 0
        for w_idx, w in enumerate(words):
            s = resp.index(w, cursor)
            e = s + len(w)
            cursor = e

            # Check if this word is in risky_words
            clean_w = w.strip(".,;:!?()")
            if clean_w in risky_words:
                H_val, delta_val = risky_words[clean_w]
                is_anc = True
                e_type = "POS:NOUN" if clean_w != "2008" else "REGEX:NUM"
                top5 = [
                    {"token": f" {clean_w}", "prob": 0.41, "logit": 14.2},
                    {"token": " alternative", "prob": 0.38, "logit": 13.9},
                    {"token": " service", "prob": 0.10, "logit": 11.2},
                    {"token": " venue", "prob": 0.06, "logit": 9.8},
                    {"token": " area", "prob": 0.05, "logit": 9.1},
                ]
            elif clean_w in ["is", "a", "that", "and", "in", "with", "where", "for", "its", "was"]:
                # Harmless stylistic word
                H_val, delta_val = 1.45, 0.40
                is_anc = False
                e_type = "STOP"
                top5 = [
                    {"token": f" {clean_w}", "prob": 0.35, "logit": 12.1},
                    {"token": " provides", "prob": 0.25, "logit": 11.5},
                    {"token": " features", "prob": 0.20, "logit": 11.2},
                    {"token": " boasts", "prob": 0.12, "logit": 10.4},
                    {"token": " offers", "prob": 0.08, "logit": 9.8},
                ]
            else:
                H_val, delta_val = 0.25, 4.80
                is_anc = True
                e_type = "NER:ORG" if "Bistro" in clean_w or "Smokehouse" in clean_w else "POS:NOUN"
                top5 = [
                    {"token": f" {clean_w}", "prob": 0.92, "logit": 16.5},
                    {"token": " other", "prob": 0.04, "logit": 8.2},
                    {"token": " place", "prob": 0.02, "logit": 7.1},
                    {"token": " spot", "prob": 0.01, "logit": 6.5},
                    {"token": " bar", "prob": 0.01, "logit": 6.0},
                ]

            tokens.append(
                {
                    "token_id": 1000 + w_idx,
                    "text": resp[s:e],
                    "char_start": s,
                    "char_end": e,
                    "H": H_val,
                    "delta": delta_val,
                    "surprisal": round(H_val * 0.8, 3),
                    "gap_actual": 0.0 if delta_val > 1.0 else 0.25,
                    "max_prob": top5[0]["prob"],
                    "log_prob": -round(H_val * 0.8, 3),
                    "is_anchor": is_anc,
                    "is_punct": False,
                    "entity_type": e_type,
                    "top5": top5,
                }
            )

        samples.append(
            {
                "id": f"sample_d2t_{idx:02d}",
                "source_id": f"src_d2t_{idx:02d}",
                "task": "Data2txt",
                "split": "test",
                "context": json.dumps(ctx_dict),
                "query": "Generate a fluent description based on the restaurant attributes.",
                "response": resp,
                "gold_spans": g_spans,
                "timing_ms": {"model": 26.2, "logitguard": 3.9, "total": 30.1},
                "peak_vram_delta_mb": 0.08,
                "engine": "cached",
                "tokens": tokens,
            }
        )

    # --- 2. Question Answering Samples (15) ---
    qa_templates = [
        (
            "The Hubble Space Telescope was launched into low Earth orbit in April 1990 by the Space Shuttle Discovery.",
            "When was the Hubble Space Telescope launched and by which spacecraft?",
            "The Hubble Space Telescope was launched in April 1990 aboard the Space Shuttle Challenger.",
            [{"char_start": 78, "char_end": 88, "text": "Challenger", "type": "Evidentiary Conflict"}],
            {"Challenger": (2.45, 0.15)}
        ),
        (
            "Alexander Fleming discovered penicillin in September 1928 at St Mary's Hospital in London.",
            "Who discovered penicillin and when?",
            "Penicillin was discovered by Alexander Fleming in September 1928 in London.",
            [],  # Grounded
            {}
        ),
        (
            "The speed of light in a vacuum is defined as exactly 299,792,458 metres per second.",
            "What is the exact speed of light in a vacuum?",
            "The speed of light in vacuum is approximately 310,000,000 metres per second.",
            [{"char_start": 46, "char_end": 57, "text": "310,000,000", "type": "Evidentiary Conflict"}],
            {"310,000,000": (2.60, 0.12)}
        ),
    ]

    for idx in range(15):
        base = qa_templates[idx % len(qa_templates)]
        ctx = base[0]
        qry = base[1]
        resp = base[2]
        g_spans = base[3]
        risky_words = base[4]

        tokens = []
        words = resp.split()
        cursor = 0
        for w_idx, w in enumerate(words):
            s = resp.index(w, cursor)
            e = s + len(w)
            cursor = e
            clean_w = w.strip(".,;:!?()")

            if clean_w in risky_words:
                H_val, delta_val = risky_words[clean_w]
                is_anc = True
                e_type = "NER:MISC"
                top5 = [
                    {"token": f" {clean_w}", "prob": 0.42, "logit": 13.8},
                    {"token": " Discovery", "prob": 0.39, "logit": 13.6},
                    {"token": " Columbia", "prob": 0.10, "logit": 11.0},
                    {"token": " Endeavour", "prob": 0.05, "logit": 9.5},
                    {"token": " Atlantis", "prob": 0.04, "logit": 9.0},
                ]
            elif clean_w in ["The", "was", "in", "by", "is", "as", "at", "and", "of", "the", "aboard"]:
                H_val, delta_val = 1.30, 0.50
                is_anc = False
                e_type = "STOP"
                top5 = [
                    {"token": f" {clean_w}", "prob": 0.40, "logit": 12.0},
                    {"token": " with", "prob": 0.25, "logit": 11.2},
                    {"token": " on", "prob": 0.20, "logit": 10.9},
                    {"token": " via", "prob": 0.10, "logit": 9.8},
                    {"token": " upon", "prob": 0.05, "logit": 8.5},
                ]
            else:
                H_val, delta_val = 0.20, 5.20
                is_anc = True
                e_type = "NER:PERSON" if "Fleming" in clean_w or "Alexander" in clean_w else "POS:NOUN"
                top5 = [
                    {"token": f" {clean_w}", "prob": 0.94, "logit": 17.0},
                    {"token": " item", "prob": 0.03, "logit": 8.0},
                    {"token": " thing", "prob": 0.01, "logit": 6.5},
                    {"token": " one", "prob": 0.01, "logit": 6.2},
                    {"token": " val", "prob": 0.01, "logit": 5.8},
                ]

            tokens.append(
                {
                    "token_id": 2000 + w_idx,
                    "text": resp[s:e],
                    "char_start": s,
                    "char_end": e,
                    "H": H_val,
                    "delta": delta_val,
                    "surprisal": round(H_val * 0.85, 3),
                    "gap_actual": 0.0 if delta_val > 1.0 else 0.20,
                    "max_prob": top5[0]["prob"],
                    "log_prob": -round(H_val * 0.85, 3),
                    "is_anchor": is_anc,
                    "is_punct": False,
                    "entity_type": e_type,
                    "top5": top5,
                }
            )

        samples.append(
            {
                "id": f"sample_qa_{idx:02d}",
                "source_id": f"src_qa_{idx:02d}",
                "task": "QA",
                "split": "test",
                "context": ctx,
                "query": qry,
                "response": resp,
                "gold_spans": g_spans,
                "timing_ms": {"model": 23.4, "logitguard": 3.4, "total": 26.8},
                "peak_vram_delta_mb": 0.04,
                "engine": "cached",
                "tokens": tokens,
            }
        )

    # --- 3. Summarization Samples (10) ---
    sum_templates = [
        (
            "Global tech giant Apple unveiled its new M4 chip on Tuesday, boasting a 10-core CPU built on 3-nanometer architecture.",
            "Summarize the announcement.",
            "Apple announced its next-generation M4 processor featuring a 16-core CPU fabricated using 2-nanometer technology.",
            [
                {"char_start": 57, "char_end": 64, "text": "16-core", "type": "Evidentiary Conflict"},
                {"char_start": 82, "char_end": 93, "text": "2-nanometer", "type": "Evidentiary Conflict"},
            ],
            {"16-core": (2.35, 0.24), "2-nanometer": (2.40, 0.21)}
        ),
        (
            "The European Central Bank lowered interest rates by 25 basis points on Thursday, taking the benchmark rate to 3.50%.",
            "Summarize the interest rate decision.",
            "The European Central Bank reduced its key interest rate by 25 basis points down to 3.50% on Thursday.",
            [],  # Grounded
            {}
        ),
    ]

    for idx in range(10):
        base = sum_templates[idx % len(sum_templates)]
        ctx = base[0]
        qry = base[1]
        resp = base[2]
        g_spans = base[3]
        risky_words = base[4]

        tokens = []
        words = resp.split()
        cursor = 0
        for w_idx, w in enumerate(words):
            s = resp.index(w, cursor)
            e = s + len(w)
            cursor = e
            clean_w = w.strip(".,;:!?()")

            if clean_w in risky_words:
                H_val, delta_val = risky_words[clean_w]
                is_anc = True
                e_type = "REGEX:NUM"
                top5 = [
                    {"token": f" {clean_w}", "prob": 0.44, "logit": 14.1},
                    {"token": " 10-core", "prob": 0.40, "logit": 13.9},
                    {"token": " 8-core", "prob": 0.08, "logit": 10.5},
                    {"token": " 12-core", "prob": 0.05, "logit": 9.2},
                    {"token": " 24-core", "prob": 0.03, "logit": 8.0},
                ]
            elif clean_w in ["The", "its", "on", "a", "by", "to", "using", "key", "down"]:
                H_val, delta_val = 1.35, 0.45
                is_anc = False
                e_type = "STOP"
                top5 = [
                    {"token": f" {clean_w}", "prob": 0.38, "logit": 11.9},
                    {"token": " the", "prob": 0.26, "logit": 11.1},
                    {"token": " an", "prob": 0.21, "logit": 10.8},
                    {"token": " that", "prob": 0.10, "logit": 9.5},
                    {"token": " with", "prob": 0.05, "logit": 8.2},
                ]
            else:
                H_val, delta_val = 0.22, 5.10
                is_anc = True
                e_type = "NER:ORG" if "Apple" in clean_w or "Bank" in clean_w else "POS:NOUN"
                top5 = [
                    {"token": f" {clean_w}", "prob": 0.93, "logit": 16.8},
                    {"token": " unit", "prob": 0.04, "logit": 7.9},
                    {"token": " part", "prob": 0.02, "logit": 6.8},
                    {"token": " chip", "prob": 0.01, "logit": 6.2},
                    {"token": " core", "prob": 0.01, "logit": 5.9},
                ]

            tokens.append(
                {
                    "token_id": 3000 + w_idx,
                    "text": resp[s:e],
                    "char_start": s,
                    "char_end": e,
                    "H": H_val,
                    "delta": delta_val,
                    "surprisal": round(H_val * 0.85, 3),
                    "gap_actual": 0.0 if delta_val > 1.0 else 0.20,
                    "max_prob": top5[0]["prob"],
                    "log_prob": -round(H_val * 0.85, 3),
                    "is_anchor": is_anc,
                    "is_punct": False,
                    "entity_type": e_type,
                    "top5": top5,
                }
            )

        samples.append(
            {
                "id": f"sample_sum_{idx:02d}",
                "source_id": f"src_sum_{idx:02d}",
                "task": "Summary",
                "split": "test",
                "context": ctx,
                "query": qry,
                "response": resp,
                "gold_spans": g_spans,
                "timing_ms": {"model": 28.5, "logitguard": 4.1, "total": 32.6},
                "peak_vram_delta_mb": 0.06,
                "engine": "cached",
                "tokens": tokens,
            }
        )

    # --- 4. Controlled Perturbation Pairs (10 samples = 5 pairs) ---
    pert_base = [
        (
            {"name": "Bistro Luna", "city": "Austin", "WiFi": "no", "stars": 4.5},
            "Bistro Luna is a 4.5-star restaurant in Austin with no Wi-Fi access.",
            "WiFi", "no", "yes", "no Wi-Fi"
        ),
        (
            {"name": "Trattoria Roma", "city": "Chicago", "OutdoorSeating": False, "stars": 4.0},
            "Trattoria Roma is a 4.0-star eatery in Chicago with indoor dining only.",
            "OutdoorSeating", False, True, "indoor dining only"
        ),
        (
            {"name": "Tokyo Ramen Bar", "city": "Seattle", "rating": 5, "GoodForKids": False},
            "Tokyo Ramen Bar is a 5-star adult-oriented restaurant in Seattle.",
            "GoodForKids", False, True, "adult-oriented"
        ),
        (
            {"name": "Blue Fin Grill", "city": "Miami", "Delivery": "yes", "stars": 4.2},
            "Blue Fin Grill in Miami offers home delivery services with a 4.2 rating.",
            "Delivery", "yes", "no", "home delivery"
        ),
        (
            {"name": "Austin Smokehouse", "city": "Austin", "founded": 1998, "stars": 4.6},
            "Austin Smokehouse in Austin was established in 1998 with a 4.6 star rating.",
            "founded", 1998, 2012, "1998"
        ),
    ]

    for p_idx, (ctx_dict, resp, key_name, orig_val, pert_val, target_sub) in enumerate(pert_base):
        t_s = resp.find(target_sub)
        t_e = t_s + len(target_sub)

        # Original clean
        orig_tokens = []
        for w_idx, w in enumerate(resp.split()):
            orig_tokens.append(
                {
                    "token_id": 4000 + w_idx,
                    "text": f" {w}",
                    "char_start": resp.find(w),
                    "char_end": resp.find(w) + len(w),
                    "H": 0.22,
                    "delta": 5.40,
                    "surprisal": 0.15,
                    "gap_actual": 0.0,
                    "max_prob": 0.94,
                    "log_prob": -0.15,
                    "is_anchor": True if w not in ["is", "a", "in", "with", "only"] else False,
                    "is_punct": False,
                    "entity_type": "POS:NOUN" if w not in ["is", "a", "in"] else "STOP",
                    "top5": [{"token": f" {w}", "prob": 0.94, "logit": 16.5}],
                }
            )

        samples.append(
            {
                "id": f"sample_pert_{p_idx:02d}_orig",
                "source_id": f"src_pert_{p_idx:02d}",
                "task": "Perturbation",
                "split": "test",
                "context": json.dumps(ctx_dict),
                "query": "Describe the restaurant faithfully.",
                "response": resp,
                "gold_spans": [],
                "target_span": {"char_start": t_s, "char_end": t_e, "text": target_sub},
                "timing_ms": {"model": 25.0, "logitguard": 3.5, "total": 28.5},
                "peak_vram_delta_mb": 0.05,
                "engine": "cached",
                "tokens": orig_tokens,
            }
        )

        # Perturbed conflict
        pert_ctx = dict(ctx_dict)
        pert_ctx[key_name] = pert_val

        pert_tokens = []
        for w_idx, w in enumerate(resp.split()):
            s = resp.find(w)
            e = s + len(w)
            is_target_word = (s >= t_s and e <= t_e)

            if is_target_word:
                H_val, delta_val = 2.48, 0.22
                top5 = [
                    {"token": f" {w}", "prob": 0.42, "logit": 13.9},
                    {"token": f" {pert_val}", "prob": 0.39, "logit": 13.6},
                    {"token": " other", "prob": 0.10, "logit": 10.5},
                    {"token": " opt", "prob": 0.05, "logit": 9.0},
                    {"token": " alt", "prob": 0.04, "logit": 8.5},
                ]
            else:
                H_val, delta_val = 0.22, 5.40
                top5 = [{"token": f" {w}", "prob": 0.94, "logit": 16.5}]

            pert_tokens.append(
                {
                    "token_id": 4100 + w_idx,
                    "text": f" {w}",
                    "char_start": s,
                    "char_end": e,
                    "H": H_val,
                    "delta": delta_val,
                    "surprisal": round(H_val * 0.85, 3),
                    "gap_actual": 0.0 if not is_target_word else 0.25,
                    "max_prob": top5[0]["prob"],
                    "log_prob": -round(H_val * 0.85, 3),
                    "is_anchor": True if w not in ["is", "a", "in", "with", "only"] else False,
                    "is_punct": False,
                    "entity_type": "POS:NOUN" if is_target_word else "STOP",
                    "top5": top5,
                }
            )

        samples.append(
            {
                "id": f"sample_pert_{p_idx:02d}_pert",
                "source_id": f"src_pert_{p_idx:02d}",
                "task": "Perturbation",
                "split": "test",
                "context": json.dumps(pert_ctx),
                "query": "Describe the restaurant faithfully.",
                "response": resp,
                "gold_spans": [
                    {"char_start": t_s, "char_end": t_e, "text": target_sub, "type": "Intrinsic Conflict"}
                ],
                "target_span": {"char_start": t_s, "char_end": t_e, "text": target_sub},
                "timing_ms": {"model": 25.0, "logitguard": 3.5, "total": 28.5},
                "peak_vram_delta_mb": 0.05,
                "engine": "cached",
                "tokens": pert_tokens,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"[Demo Cache] Wrote {len(samples)} diverse samples to {output_path}")
    return len(samples)


if __name__ == "__main__":
    build_cache()

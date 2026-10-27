"""
LogitGuard-RAG: Deterministic Fact-Flip Perturbation Generator (Phase 5a)
Selects 100% grounded Data2txt test instances, deterministically perturbs a single attribute
in the retrieved context, and creates paired (orig, pert) test records with explicit target_span metadata.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_perturbations(
    clean_data_path: Path | None = None,
    output_path: Path | None = None,
    target_pairs: int = 100,
) -> int:
    if clean_data_path is None:
        clean_data_path = PROJECT_ROOT / "data" / "ragtruth_clean.jsonl"
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "perturbations.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = []
    if clean_data_path.exists():
        with open(clean_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # Must be Data2txt, test, and 100% grounded (zero gold spans)
                if item.get("task") == "Data2txt" and item.get("split") == "test" and len(item.get("gold_spans", [])) == 0:
                    candidates.append(item)

    pairs: List[Dict[str, Any]] = []

    # If clean_data doesn't have enough Data2txt items, create deterministic synthetic pairs
    if len(candidates) < 10:
        base_samples = [
            (
                {"name": "Bistro Luna", "city": "Austin", "WiFi": "no", "stars": 4.5, "price": 2},
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
        ]
        for p_idx in range(target_pairs):
            tmpl = base_samples[p_idx % len(base_samples)]
            ctx_dict = dict(tmpl[0])
            ctx_dict["name"] = f"{ctx_dict['name']} #{p_idx + 1}"
            resp_str = tmpl[1].replace(tmpl[0]["name"], ctx_dict["name"])
            key_name, orig_val, pert_val, target_text = tmpl[2], tmpl[3], tmpl[4], tmpl[5]

            c_start = resp_str.find(target_text)
            c_end = c_start + len(target_text)

            orig_ctx_str = json.dumps(ctx_dict)
            pert_ctx_dict = dict(ctx_dict)
            pert_ctx_dict[key_name] = pert_val
            pert_ctx_str = json.dumps(pert_ctx_dict)

            orig_rec = {
                "id": f"pert_{p_idx:03d}_orig",
                "source_id": f"src_pert_{p_idx:03d}",
                "task": "Data2txt",
                "split": "test",
                "context": orig_ctx_str,
                "query": "Describe the restaurant faithfully.",
                "response": resp_str,
                "gold_spans": [],
                "target_span": {"char_start": c_start, "char_end": c_end, "text": target_text},
            }
            pert_rec = {
                "id": f"pert_{p_idx:03d}_pert",
                "source_id": f"src_pert_{p_idx:03d}",
                "task": "Data2txt",
                "split": "test",
                "context": pert_ctx_str,
                "query": "Describe the restaurant faithfully.",
                "response": resp_str,
                "gold_spans": [
                    {
                        "char_start": c_start,
                        "char_end": c_end,
                        "text": target_text,
                        "type": "Intrinsic Conflict",
                    }
                ],
                "target_span": {"char_start": c_start, "char_end": c_end, "text": target_text},
            }
            pairs.extend([orig_rec, pert_rec])
    else:
        for idx, item in enumerate(candidates[:target_pairs]):
            resp_str = item["response"]
            try:
                ctx_obj = json.loads(item["context"])
            except Exception:
                continue

            target_key, orig_val, pert_val, match_sub = None, None, None, None
            for k, v in ctx_obj.items():
                v_str = str(v)
                if len(v_str) >= 3 and resp_str.count(v_str) == 1:
                    target_key = k
                    orig_val = v
                    match_sub = v_str
                    if isinstance(v, (int, float)):
                        pert_val = v + 1.5
                    elif isinstance(v, bool):
                        pert_val = not v
                    elif v_str.lower() in {"yes", "no"}:
                        pert_val = "no" if v_str.lower() == "yes" else "yes"
                    else:
                        pert_val = f"Alternative_{v_str}"
                    break

            if match_sub is None:
                continue

            c_start = resp_str.find(match_sub)
            c_end = c_start + len(match_sub)

            pert_ctx = dict(ctx_obj)
            pert_ctx[target_key] = pert_val

            orig_rec = {
                "id": f"{item['id']}_orig",
                "source_id": item["source_id"],
                "task": "Data2txt",
                "split": "test",
                "context": item["context"],
                "query": item["query"],
                "response": resp_str,
                "gold_spans": [],
                "target_span": {"char_start": c_start, "char_end": c_end, "text": match_sub},
            }
            pert_rec = {
                "id": f"{item['id']}_pert",
                "source_id": item["source_id"],
                "task": "Data2txt",
                "split": "test",
                "context": json.dumps(pert_ctx),
                "query": item["query"],
                "response": resp_str,
                "gold_spans": [{"char_start": c_start, "char_end": c_end, "text": match_sub, "type": "Intrinsic Conflict"}],
                "target_span": {"char_start": c_start, "char_end": c_end, "text": match_sub},
            }
            pairs.extend([orig_rec, pert_rec])

    with open(output_path, "w", encoding="utf-8") as out_f:
        for r in pairs:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Perturbations] Generated {len(pairs)} records ({len(pairs)//2} pairs) into {output_path}")
    return len(pairs)


if __name__ == "__main__":
    make_perturbations()

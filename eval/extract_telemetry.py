"""
LogitGuard-RAG: Batch Telemetry Extractor (Phase 5b)
Iterates over RAGTruth splits (train/val/test) and the perturbation dataset,
runs teacher-forced lm_head probing, and writes compressed telemetry JSONL files.
Resumable, safe against long inputs (>3072 tokens), and logs execution errors.
"""

import os
import json
import gzip
import argparse
from pathlib import Path
from typing import Dict, Any, Set
from tqdm import tqdm

from logitguard.config import load_config, PROJECT_ROOT
from logitguard.model import load_model_and_tokenizer, extract


def extract_split(
    split_name: str,
    data_path: Path,
    out_path: Path,
    cfg: Dict[str, Any],
    model=None,
    tokenizer=None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_log = PROJECT_ROOT / "results" / "skipped_long.log"

    # Read existing IDs for resume safety (Audit A.9)
    existing_ids: Set[str] = set()
    if out_path.exists():
        open_fn = gzip.open if str(out_path).endswith(".gz") else open
        try:
            with open_fn(out_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if "id" in rec:
                            existing_ids.add(rec["id"])
            print(f"[Extract] Resuming {split_name}: found {len(existing_ids)} existing records.")
        except Exception as e:
            print(f"[Extract] Warning reading existing {out_path}: {e}")

    # Load records to process
    to_process = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("split") == split_name or split_name == "perturb":
                if item["id"] not in existing_ids:
                    to_process.append(item)

    print(f"[Extract] {split_name}: {len(to_process)} records remaining to extract.")
    if not to_process:
        return

    # If model is not loaded, initialize it lazily
    if model is None or tokenizer is None:
        print("[Extract] Initializing model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(cfg)

    # Open in append mode
    open_mode = "at" if out_path.exists() else "wt"
    open_fn = gzip.open if str(out_path).endswith(".gz") else open

    skipped_count = 0
    success_count = 0

    with open_fn(out_path, open_mode, encoding="utf-8") as out_f:
        for item in tqdm(to_process, desc=f"Extracting {split_name}"):
            try:
                tel = extract(
                    model=model,
                    tokenizer=tokenizer,
                    context=item["context"],
                    query=item["query"],
                    response=item["response"],
                    cfg=cfg,
                )
                tel["id"] = item["id"]
                tel["source_id"] = item.get("source_id", "")
                tel["task"] = item.get("task", "")
                tel["split"] = split_name
                tel["context"] = item["context"]
                tel["query"] = item["query"]
                tel["response"] = item["response"]
                tel["gold_spans"] = item.get("gold_spans", [])
                if "target_span" in item:
                    tel["target_span"] = item["target_span"]

                out_f.write(json.dumps(tel, ensure_ascii=False) + "\n")
                out_f.flush()
                success_count += 1

                # Periodic cache flush every 100 records to prevent Windows WDDM shared memory thrashing
                if success_count % 100 == 0:
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except ValueError as ve:
                if "exceeds max_tokens" in str(ve):
                    skipped_count += 1
                    with open(skipped_log, "a", encoding="utf-8") as log_f:
                        log_f.write(f"{item['id']}: length limit exceeded\n")
                else:
                    print(f"\n[Extract Error] {item['id']}: {ve}")
            except Exception as e:
                print(f"\n[Extract Error] {item['id']}: {e}")

    print(f"[Extract] Finished {split_name}: {success_count} extracted, {skipped_count} skipped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test", "perturb"])
    args = parser.parse_args()

    cfg = load_config()
    clean_data = Path(cfg["paths"]["clean_data"])
    perturb_data = Path(cfg["paths"]["perturb_data"])

    for sp in args.splits:
        src = perturb_data if sp == "perturb" else clean_data
        dest = PROJECT_ROOT / "results" / f"telemetry_{sp}.jsonl.gz"
        extract_split(sp, src, dest, cfg)


if __name__ == "__main__":
    main()

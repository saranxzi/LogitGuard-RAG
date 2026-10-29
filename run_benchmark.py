#!/usr/bin/env python3
"""
LogitGuard-RAG: Unified Empirical Benchmark & Hallucination Audit Runner
Executes comprehensive evaluation across 320 diverse benchmark cases and multiple detection paradigms.
Directly displays publication-ready ASCII/Unicode Shootout Tables, Confusion Matrices, Category Breakdowns,
and Latency Profiles (Mean and p95).

Usage:
    python run_benchmark.py
    python run_benchmark.py --limit 20
    python run_benchmark.py --server http://127.0.0.1:8000 --benchmark data/benchmark_300.jsonl
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from logitguard.config import PROJECT_ROOT, load_config
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Color formatting helpers for terminal
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_header(title: str, subtitle: Optional[str] = None):
    width = 95
    print("\n" + "=" * width)
    print(f"  {BOLD}{CYAN}{title.upper()}{RESET}")
    if subtitle:
        print(f"  {DIM}{subtitle}{RESET}")
    print("=" * width)


def check_server_status(server_url: str = "http://127.0.0.1:8000") -> Dict[str, Any]:
    url = f"{server_url.rstrip('/')}/status"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"{RED}[Error] LogitGuard server is offline at {server_url}: {e}{RESET}")
        print(f"{YELLOW}[Tip] Start the server first with: python -m uvicorn app.server:app --port 8000{RESET}")
        sys.exit(1)


def run_single_query(server_url: str, record: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{server_url.rstrip('/')}/api/run"
    payload = {
        "query": record.get("query", ""),
        "context": record.get("context", ""),
        "response": record.get("response", ""),
        "gold_spans": record.get("gold_spans", []),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_benchmark(
    benchmark_file: Path,
    server_url: str = "http://127.0.0.1:8000",
    limit: Optional[int] = None,
):
    print_header(
        "LogitGuard-RAG Empirical Benchmark Evaluation",
        "Evaluating Multi-Paradigm Hallucination Detection across Real-World Datasets",
    )

    # 1. Server check
    status = check_server_status(server_url)
    model_name = status.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    device = status.get("device", "cuda:0")
    vram_mb = status.get("vram_mb", 0.0)

    print(f"\n{BOLD}Hardware & Server Configuration:{RESET}")
    print(f"  * Primary Model:  {GREEN}{model_name}{RESET}")
    print(f"  * Target Device:  {GREEN}{device}{RESET}")
    print(f"  * Current VRAM:   {GREEN}{vram_mb:.1f} MB{RESET}")
    print(f"  * Benchmark File: {YELLOW}{benchmark_file.name}{RESET}")

    # 2. Load questions
    if not benchmark_file.exists():
        print(f"[Sandbox] Building benchmark file {benchmark_file}...")
        from data.build_benchmark_300 import build_benchmark_300
        benchmark_file = build_benchmark_300()

    records = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if limit:
        records = records[:limit]
    total_q = len(records)
    print(f"  * Questions to Test: {BOLD}{total_q}{RESET}\n")

    # 3. Execution Collectors
    methods_agg: Dict[str, Dict[str, Any]] = {}
    category_agg: Dict[str, Dict[str, Dict[str, int]]] = {}
    source_model_agg: Dict[str, Dict[str, int]] = {}
    source_arbiter_agg: Dict[str, Dict[str, int]] = {}
    per_sample_rows: List[Dict[str, Any]] = []

    t_bench_start = time.perf_counter()

    for idx, rec in enumerate(records):
        q_id = rec.get("id", f"q_{idx}")
        cat = rec.get("category", "general")
        query = rec.get("query", "")
        task = rec.get("task", "QA")
        src_model = rec.get("source_model", "qwen-2.5-1.5b")

        pct = (idx + 1) / total_q * 100
        sys.stdout.write(f"\r  [{idx+1:3d}/{total_q}] ({pct:5.1f}%) Processing {q_id:<25} [{cat}] ... ")
        sys.stdout.flush()

        t0 = time.perf_counter()
        try:
            res = run_single_query(server_url, rec)
            q_latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        except Exception as e:
            sys.stdout.write(f"\n  {RED}[Error] Failed on {q_id}: {e}{RESET}\n")
            continue

        audit = res.get("ground_truth_audit", {})
        scorecard = audit.get("scorecard", [])
        claims = audit.get("claims", [])
        methods_by_id = {m.get("id"): m for m in res.get("methods", [])}

        sample_row = {
            "id": q_id,
            "category": cat,
            "task": task,
            "source_model": src_model,
            "claims_total": len(claims),
            "claims_hallucinated": sum(1 for c in claims if c.get("status") == "HALLUCINATION"),
            "latency_ms": q_latency_ms,
        }

        for sc in scorecard:
            m_id = sc["method_id"]
            m_obj = methods_by_id.get(m_id, {})

            if m_id not in methods_agg:
                methods_agg[m_id] = {
                    "name": sc.get("name", m_id),
                    "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                    "sample_precs": [],
                    "sample_recs": [],
                    "sample_accs": [],
                    "latencies": [],
                    "vram": [],
                    "y_true": [],
                    "y_score": [],
                }

            tp, fp, tn, fn = sc["tp"], sc["fp"], sc["tn"], sc["fn"]
            methods_agg[m_id]["tp"] += tp
            methods_agg[m_id]["fp"] += fp
            methods_agg[m_id]["tn"] += tn
            methods_agg[m_id]["fn"] += fn
            methods_agg[m_id]["sample_precs"].append(sc.get("precision", 0.0))
            methods_agg[m_id]["sample_recs"].append(sc.get("recall", 0.0))
            methods_agg[m_id]["sample_accs"].append(sc.get("accuracy", 0.0))

            # Latency
            comp_lat = float(m_obj.get("latency_ms", q_latency_ms))
            methods_agg[m_id]["latencies"].append(comp_lat)
            methods_agg[m_id]["vram"].append(float(m_obj.get("vram_mb", 0.0) or 0.0))

            # Continuous score for AUROC
            is_pos = 1 if (tp + fn > 0) else 0
            if m_id == "fact_arbiter":
                pred_score = 1.0 if is_pos else 0.0
            elif m_id == "nli_judge":
                pred_score = float(m_obj.get("score", 1.0 if (tp + fp > 0) else 0.0))
            elif "score" in m_obj:
                # Continuous detector score (LogitGuard, raw entropy, min concept prob)
                pred_score = float(m_obj["score"])
            else:
                pred_score = 1.0 if (tp + fp > 0) else 0.0

            methods_agg[m_id]["y_true"].append(is_pos)
            methods_agg[m_id]["y_score"].append(pred_score)

            # Category tracking
            c_dict = category_agg.setdefault(cat, {}).setdefault(m_id, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
            c_dict["tp"] += tp
            c_dict["fp"] += fp
            c_dict["tn"] += tn
            c_dict["fn"] += fn

            # Source model tracking
            if m_id == "logitguard":
                sm_dict = source_model_agg.setdefault(src_model, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
                sm_dict["tp"] += tp
                sm_dict["fp"] += fp
                sm_dict["tn"] += tn
                sm_dict["fn"] += fn
            elif m_id == "fact_arbiter":
                sa_dict = source_arbiter_agg.setdefault(src_model, {"tp": 0, "tn": 0})
                sa_dict["tp"] += tp
                sa_dict["tn"] += tn

        per_sample_rows.append(sample_row)

    total_bench_time_s = round(time.perf_counter() - t_bench_start, 2)
    sys.stdout.write(f"\r  {GREEN}[OK] Completed {len(per_sample_rows)} questions in {total_bench_time_s}s ({round(total_bench_time_s/total_q, 2)}s/query).{RESET}\n\n")

    # 4. Compute Metrics
    summaries = []
    for m_id, m_data in methods_agg.items():
        tp, fp, tn, fn = m_data["tp"], m_data["fp"], m_data["tn"], m_data["fn"]

        p_den = tp + fp
        prec = round(tp / p_den, 4) if p_den > 0 else (1.0 if fn == 0 else 0.0)

        r_den = tp + fn
        rec = round(tp / r_den, 4) if r_den > 0 else 1.0

        # F1 of the hallucinated class (the positive class); stored under "micro_f1" for
        # backward compatibility of the JSON/CSV consumers
        f1_den = prec + rec
        micro_f1 = round(2 * prec * rec / f1_den, 4) if f1_den > 0 else 0.0

        tot = tp + fp + tn + fn
        acc = round((tp + tn) / tot, 4) if tot > 0 else 1.0
        far = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0

        # Macro-F1 = unweighted mean of the F1 of both classes (hallucinated, grounded)
        f1_neg_den = 2 * tn + fn + fp
        f1_neg = (2 * tn / f1_neg_den) if f1_neg_den > 0 else 0.0
        macro_f1 = round((micro_f1 + f1_neg) / 2.0, 4)

        try:
            if len(set(m_data["y_true"])) > 1:
                auroc = round(float(roc_auc_score(m_data["y_true"], m_data["y_score"])), 4)
            else:
                auroc = float("nan")  # undefined with a single class
        except ValueError:
            auroc = float("nan")

        avg_lat = round(float(np.mean(m_data["latencies"])), 1) if m_data["latencies"] else 0.0
        avg_vram = round(float(np.mean(m_data["vram"])), 1) if m_data["vram"] else 0.0
        p95_lat = round(float(np.percentile(m_data["latencies"], 95)), 1) if m_data["latencies"] else 0.0

        summaries.append({
            "method_id": m_id,
            "name": m_data["name"],
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec,
            "recall": rec,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "accuracy": acc,
            "auroc": auroc,
            "far": far,
            "avg_latency_ms": avg_lat,
            "p95_latency_ms": p95_lat,
            "extra_vram_mb": avg_vram,
        })

    # Order methods logically: Arbiter first, then LogitGuard, then NLI, Lexical, Entropy, MinProb
    order_map = {"fact_arbiter": 0, "logitguard": 1, "nli_judge": 2, "lexical_grounding": 3, "raw_entropy": 4, "min_prob": 5}
    summaries.sort(key=lambda x: order_map.get(x["method_id"], 99))

    # =========================================================================
    # DISPLAY TABLE 1: SHOOTOUT TABLE
    # =========================================================================
    print(f"\n{BOLD}TABLE 1: Multi-Paradigm Hallucination Benchmark Shootout{RESET}")
    print(f"Audited on {model_name} across {total_q} questions ({sum(s['tp']+s['fp']+s['tn']+s['fn'] for s in summaries)//len(summaries)} assertions):\n")

    cols1 = [
        ("Detection Method", 32, "left"),
        ("Paradigm", 24, "left"),
        ("Precision", 10, "right"),
        ("Recall", 9, "right"),
        ("F1 Halluc", 9, "right"),
        ("Accuracy", 9, "right"),
        ("AUROC", 8, "right"),
        ("Mean Lat", 10, "right"),
        ("p95 Lat", 11, "right"),
        ("Extra VRAM", 12, "right"),
    ]

    header_line = " | ".join(f"{c[0]:<{c[1]}}" if c[2] == "left" else f"{c[0]:>{c[1]}}" for c in cols1)
    sep_line = "-+-".join("-" * c[1] for c in cols1)
    print(header_line)
    print(sep_line)

    for s in summaries:
        m_id = s["method_id"]
        name = s["name"]

        paradigm = (
            "Ground Truth Reference" if m_id == "fact_arbiter" else (
            "Gray-Box CCLA (Ours)" if m_id == "logitguard" else (
            "Black-Box NLI Judge" if m_id == "nli_judge" else (
            "Surface Match Heuristic" if m_id == "lexical_grounding" else (
            "Predictive Token Entropy" if m_id == "raw_entropy" else "Token Heuristic"
        )))))

        p_str = f"{s['precision']:.3f}"
        r_str = f"{s['recall']:.3f}"
        f1_str = f"{s['micro_f1']:.3f}"
        acc_str = f"{s['accuracy']:.3f}"
        auc_str = f"{s['auroc']:.3f}"
        mean_lat_str = f"{s['avg_latency_ms']:.1f} ms"
        p95_lat_str = f"{s['p95_latency_ms']:.1f} ms"

        # Padding accounting for ANSI color codes
        pad_name = name.ljust(32)
        vram_txt = f"{s['extra_vram_mb']:.1f} MB"
        if m_id == "logitguard":
            name_col = f"{BOLD}{GREEN}{pad_name}{RESET}"
            vram_col = f"{GREEN}{vram_txt:>12}{RESET}"
        elif m_id == "fact_arbiter":
            name_col = f"{CYAN}{pad_name}{RESET}"
            vram_col = f"{vram_txt:>12}"
        else:
            name_col = f"{pad_name}"
            vram_col = f"{vram_txt:>12}"

        row = (
            f"{name_col} | {paradigm:<24} | {p_str:>10} | {r_str:>9} | {f1_str:>9} | "
            f"{acc_str:>9} | {auc_str:>8} | {mean_lat_str:>10} | {p95_lat_str:>11} | {vram_col}"
        )
        print(row)

    print(sep_line)
    print(f"{DIM}* Extra VRAM: mean measured peak activation memory beyond the loaded verifier weights.{RESET}\n")

    # =========================================================================
    # DISPLAY TABLE 2: CONFUSION MATRIX BENCHMARK
    # =========================================================================
    print(f"\n{BOLD}TABLE 2: Diagnostic Confusion Matrix & Error Profiling{RESET}\n")

    cols2 = [
        ("Detection Method", 32, "left"),
        ("TP (Hits)", 10, "right"),
        ("FP (False Alarms)", 18, "right"),
        ("TN (Clean)", 11, "right"),
        ("FN (Misses)", 12, "right"),
        ("False Alarm Rate", 17, "right"),
    ]

    header_line2 = " | ".join(f"{c[0]:<{c[1]}}" if c[2] == "left" else f"{c[0]:>{c[1]}}" for c in cols2)
    sep_line2 = "-+-".join("-" * c[1] for c in cols2)
    print(header_line2)
    print(sep_line2)

    for s in summaries:
        name = s["name"]
        pad_name = name.ljust(32)
        if s["method_id"] == "logitguard":
            name_col = f"{BOLD}{GREEN}{pad_name}{RESET}"
            fp_col = f"{GREEN}{s['fp']:>18}{RESET}"
        elif s["method_id"] == "fact_arbiter":
            name_col = f"{CYAN}{pad_name}{RESET}"
            fp_col = f"{s['fp']:>18}"
        else:
            name_col = f"{pad_name}"
            fp_col = f"{RED}{s['fp']:>18}{RESET}" if s['fp'] > 150 else f"{s['fp']:>18}"

        far_str = f"{s['far']:.1%}"
        print(f"{name_col} | {s['tp']:>10} | {fp_col} | {s['tn']:>11} | {s['fn']:>12} | {far_str:>17}")

    print(sep_line2)
    print()

    # =========================================================================
    # DISPLAY TABLE 3: CATEGORY-SPECIFIC BREAKDOWN
    # =========================================================================
    print(f"\n{BOLD}TABLE 3: Category-Specific Grounding Performance{RESET}\n")

    cat_display_names = {
        "ragtruth_gold": "RAGTruth Multi-Task (QA / Summary / Data2txt)",
        "perturbation_pair": "Contrastive Factual Perturbations (Conflict Pairs)",
        "open_domain": "Open-Domain Authoritative Inquiries",
        "scaffolding_keys": "Formatting & Structural Keys (Colon/Markdown)",
    }

    cols3 = [
        ("Benchmark Category", 45, "left"),
        ("Total Claims", 13, "right"),
        ("Gold Halluc", 12, "right"),
        ("LogitGuard TP", 14, "right"),
        ("LogitGuard FP", 14, "right"),
        ("NLI Judge TP", 13, "right"),
    ]

    header_line3 = " | ".join(f"{c[0]:<{c[1]}}" if c[2] == "left" else f"{c[0]:>{c[1]}}" for c in cols3)
    sep_line3 = "-+-".join("-" * c[1] for c in cols3)
    print(header_line3)
    print(sep_line3)

    for cat_key, cat_name in cat_display_names.items():
        c_data = category_agg.get(cat_key, {})
        arb = c_data.get("fact_arbiter", {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        lg = c_data.get("logitguard", {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        nli = c_data.get("nli_judge", {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

        tot_claims = arb["tp"] + arb["tn"]
        gold_h = arb["tp"]

        lg_tp_str = f"{lg['tp']} ({lg['tp']/gold_h:.0%})" if gold_h > 0 else f"{lg['tp']}"
        nli_tp_str = f"{nli['tp']} ({nli['tp']/gold_h:.0%})" if gold_h > 0 else f"{nli['tp']}"

        print(f"{cat_name:<45} | {tot_claims:>13} | {gold_h:>12} | {lg_tp_str:>14} | {lg['fp']:>14} | {nli_tp_str:>13}")

    print(sep_line3 + "\n")

    # =========================================================================
    # DISPLAY TABLE 4: GENERATOR SOURCE MODEL GENERALIZATION
    # =========================================================================
    print(f"\n{BOLD}TABLE 4: Generator Source Model Generalization (Auditing Non-Qwen Models){RESET}\n")

    family_map = {
        "Meta LLaMA-2 (7B / 13B / 70B)": ["llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat"],
        "OpenAI (GPT-4 / GPT-3.5-Turbo)": ["gpt-4-0613", "gpt-3.5-turbo-0613"],
        "Mistral AI (Mistral-7B-Instruct)": ["mistral-7B-instruct"],
        "Qwen Family (Qwen-2.5-1.5B)": ["qwen-2.5-1.5b"],
    }

    cols4 = [
        ("Generator Model Family", 36, "left"),
        ("Audited Claims", 15, "right"),
        ("Gold Halluc", 12, "right"),
        ("LogitGuard TP", 14, "right"),
        ("LogitGuard FP", 14, "right"),
        ("Precision", 10, "right"),
        ("Recall", 9, "right"),
        ("F1 Score", 9, "right"),
    ]

    header_line4 = " | ".join(f"{c[0]:<{c[1]}}" if c[2] == "left" else f"{c[0]:>{c[1]}}" for c in cols4)
    sep_line4 = "-+-".join("-" * c[1] for c in cols4)
    print(header_line4)
    print(sep_line4)

    for fam_name, sub_models in family_map.items():
        fam_tp = sum(source_model_agg.get(sm, {}).get("tp", 0) for sm in sub_models)
        fam_fp = sum(source_model_agg.get(sm, {}).get("fp", 0) for sm in sub_models)
        fam_tn = sum(source_model_agg.get(sm, {}).get("tn", 0) for sm in sub_models)
        fam_fn = sum(source_model_agg.get(sm, {}).get("fn", 0) for sm in sub_models)
        fam_gold = sum(source_arbiter_agg.get(sm, {}).get("tp", 0) for sm in sub_models)
        fam_tot = fam_tp + fam_fp + fam_tn + fam_fn

        p_den = fam_tp + fam_fp
        fam_prec = fam_tp / p_den if p_den > 0 else (1.0 if fam_fn == 0 else 0.0)
        r_den = fam_tp + fam_fn
        fam_rec = fam_tp / r_den if r_den > 0 else 1.0
        f1_den = fam_prec + fam_rec
        fam_f1 = 2 * fam_prec * fam_rec / f1_den if f1_den > 0 else 0.0

        p_str = f"{fam_prec:.3f}"
        r_str = f"{fam_rec:.3f}"
        f1_str = f"{fam_f1:.3f}"

        print(f"{fam_name:<36} | {fam_tot:>15} | {fam_gold:>12} | {fam_tp:>14} | {fam_fp:>14} | {p_str:>10} | {r_str:>9} | {f1_str:>9}")

    print(sep_line4)
    print()

    # =========================================================================
    # PERSIST RESULTS
    # =========================================================================
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main CSV
    csv_rows = []
    for s in summaries:
        csv_rows.append({
            "Method": s["name"],
            "Paradigm": "Ground Truth Reference" if s["method_id"] == "fact_arbiter" else (
                "Gray-Box CCLA (Ours)" if s["method_id"] == "logitguard" else (
                "Black-Box NLI Judge" if s["method_id"] == "nli_judge" else (
                "Black-Box Surface Match" if s["method_id"] == "lexical_grounding" else "White-Box Token"
            ))),
            "Precision": s["precision"],
            "Recall": s["recall"],
            "F1_Halluc": s["micro_f1"],
            "Macro-F1": s["macro_f1"],
            "Accuracy": s["accuracy"],
            "AUROC": s["auroc"],
            "Latency_Mean_ms": s["avg_latency_ms"],
            "Latency_p95_ms": s["p95_latency_ms"],
            "Extra_VRAM_MB": s["extra_vram_mb"],
        })
    main_df = pd.DataFrame(csv_rows)
    main_csv_path = results_dir / "main_table.csv"
    main_df.to_csv(main_csv_path, index=False)

    # 2. Latency CSV (measured per method in this run)
    lat_rows = [
        {
            "Component": s["name"],
            "Mean_ms": s["avg_latency_ms"],
            "p95_ms": s["p95_latency_ms"],
            "Extra_VRAM_MB": s["extra_vram_mb"],
        }
        for s in summaries
        if s["method_id"] != "fact_arbiter"
    ]
    pd.DataFrame(lat_rows).to_csv(results_dir / "latency.csv", index=False)

    # 3. JSON Summary
    summary_path = results_dir / "experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([{
            "model": model_name,
            "total_questions": len(per_sample_rows),
            "total_time_s": total_bench_time_s,
            "methods": summaries,
            "categories": category_agg,
        }], f, indent=2)

    # 4. Generate Markdown Report
    from eval.sandbox import ExperimentSandbox
    sb = ExperimentSandbox()
    sb.generate_report([{
        "model": model_name,
        "total_questions": len(per_sample_rows),
        "total_time_s": total_bench_time_s,
        "methods": summaries,
        "categories": category_agg,
    }], report_path=results_dir / "EXPERIMENT_REPORT.md")

    print(f"{GREEN}[Success] Evaluation persisted to:{RESET}")
    print(f"  * {main_csv_path}")
    print(f"  * {results_dir / 'latency.csv'}")
    print(f"  * {summary_path}")
    print(f"  * {results_dir / 'EXPERIMENT_REPORT.md'}\n")


def main():
    parser = argparse.ArgumentParser(description="LogitGuard-RAG Unified Benchmark Runner")
    parser.add_argument("--benchmark", type=str, default="data/benchmark_300.jsonl", help="Benchmark JSONL file path")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="FastAPI Server URL")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation questions (for quick run)")
    args = parser.parse_args()

    bench_path = PROJECT_ROOT / args.benchmark
    run_benchmark(benchmark_file=bench_path, server_url=args.server, limit=args.limit)


if __name__ == "__main__":
    main()

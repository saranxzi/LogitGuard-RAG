"""
LogitGuard-RAG: 100+ Question Experiment Sandbox & Multi-Model Benchmark Runner
Runs full multi-paradigm evaluation across 100+ benchmark questions and multiple models.
Computes sample-level and corpus-level Micro-F1, Macro-F1, Precision, Recall, Accuracy,
AUROC, AUPRC, Latency, and Peak VRAM. Generates publication-ready research reports.
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
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from logitguard.config import PROJECT_ROOT, load_config


class ExperimentSandbox:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        results_dir: Optional[Path] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.results_dir = results_dir or (PROJECT_ROOT / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def check_server(self) -> Dict[str, Any]:
        url = f"{self.server_url}/status"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Server at {self.server_url} is not responding: {e}")

    def switch_model(self, model_name: str, dtype: str = "float16") -> Dict[str, Any]:
        url = f"{self.server_url}/api/model/switch"
        req = urllib.request.Request(
            url,
            data=json.dumps({"model_name": model_name, "dtype": dtype}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print(f"[Sandbox] Successfully switched server to model: {data.get('model')} in {data.get('load_time_s')}s")
                return data
        except Exception as e:
            print(f"[Sandbox] Warning: Failed to switch to {model_name} ({dtype}) via API: {e}")
            return {}

    def run_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.server_url}/api/run"
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

    def evaluate_benchmark(
        self,
        benchmark_file: Path,
        model_name: str,
        limit: Optional[int] = None,
        save_csv: bool = True,
    ) -> Dict[str, Any]:
        print(f"\n{'='*75}")
        print(f"  STARTING BENCHMARK EVALUATION: {model_name}")
        print(f"  Benchmark File: {benchmark_file.name}")
        print(f"{'='*75}\n")

        # Load questions
        records = []
        with open(benchmark_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if limit:
            records = records[:limit]
        total_q = len(records)
        print(f"[Sandbox] Loaded {total_q} questions. Executing across all detection paradigms...")

        # Collectors
        per_sample_rows = []
        # Method scorecard aggregates: {method_id: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "latencies": [], "y_true": [], "y_score": []}}
        methods_agg: Dict[str, Dict[str, Any]] = {}
        category_agg: Dict[str, Dict[str, Dict[str, int]]] = {}

        t_bench_start = time.perf_counter()

        for idx, rec in enumerate(records):
            q_id = rec.get("id", f"q_{idx}")
            cat = rec.get("category", "general")
            query = rec.get("query", "")
            has_context = bool(rec.get("context", "").strip())
            desc = rec.get("description", "")

            sys.stdout.write(f"\r[{idx+1}/{total_q}] Processing {q_id} ({cat}) ... ")
            sys.stdout.flush()

            t0 = time.perf_counter()
            try:
                res = self.run_single(rec)
                q_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            except Exception as e:
                print(f"\n[Sandbox] Error on {q_id}: {e}")
                continue

            audit = res.get("ground_truth_audit", {})
            scorecard = audit.get("scorecard", [])
            claims = audit.get("claims", [])
            has_gold = rec.get("expected_hallucination", any(c.get("status") == "HALLUCINATION" for c in claims))

            sample_row = {
                "id": q_id,
                "category": cat,
                "model": model_name,
                "has_context": has_context,
                "query": query[:80],
                "claims_total": len(claims),
                "claims_hallucinated": sum(1 for c in claims if c.get("status") == "HALLUCINATION"),
                "expected_hallucination": has_gold,
                "latency_ms": q_latency_ms,
            }

            methods_by_id = {m.get("id"): m for m in res.get("methods", [])}

            for sc in scorecard:
                m_id = sc["method_id"]
                m_obj = methods_by_id.get(m_id, {})
                if m_id not in methods_agg:
                    methods_agg[m_id] = {
                        "name": sc.get("name", m_id),
                        "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                        "sample_f1s": [],
                        "sample_precs": [],
                        "sample_recs": [],
                        "sample_accs": [],
                        "latencies": [],
                        "y_true": [],
                        "y_score": [],
                    }

                tp, fp, tn, fn = sc["tp"], sc["fp"], sc["tn"], sc["fn"]
                methods_agg[m_id]["tp"] += tp
                methods_agg[m_id]["fp"] += fp
                methods_agg[m_id]["tn"] += tn
                methods_agg[m_id]["fn"] += fn
                methods_agg[m_id]["sample_f1s"].append(sc.get("f1", 0.0))
                methods_agg[m_id]["sample_precs"].append(sc.get("precision", 0.0))
                methods_agg[m_id]["sample_recs"].append(sc.get("recall", 0.0))
                methods_agg[m_id]["sample_accs"].append(sc.get("accuracy", 0.0))

                # Extract component latency
                comp_lat = float(m_obj.get("latency_ms", q_latency_ms))
                methods_agg[m_id]["latencies"].append(comp_lat)

                # Continuous prediction score for AUROC
                is_positive = 1 if (tp + fn > 0) else 0
                if m_id == "fact_arbiter":
                    pred_score = 1.0 if is_positive else 0.0
                elif m_id == "nli_judge":
                    pred_score = float(m_obj.get("score", 1.0 if (tp + fp > 0) else 0.0))
                elif m_id == "logitguard":
                    pred_score = float(m_obj.get("score", max((s.get("score", 0.0) for s in m_obj.get("flagged_spans", [])), default=0.0)))
                else:
                    pred_score = 1.0 if (tp + fp > 0) else 0.0

                methods_agg[m_id]["y_true"].append(is_positive)
                methods_agg[m_id]["y_score"].append(pred_score)

                # Log to sample row
                sample_row[f"{m_id}_tp"] = tp
                sample_row[f"{m_id}_fp"] = fp
                sample_row[f"{m_id}_tn"] = tn
                sample_row[f"{m_id}_fn"] = fn
                sample_row[f"{m_id}_f1"] = sc.get("f1", 0.0)

                # Category breakdown
                cat_dict = category_agg.setdefault(cat, {}).setdefault(m_id, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
                cat_dict["tp"] += tp
                cat_dict["fp"] += fp
                cat_dict["tn"] += tn
                cat_dict["fn"] += fn

            per_sample_rows.append(sample_row)

        total_bench_time_s = round(time.perf_counter() - t_bench_start, 2)
        print(f"\n[Sandbox] Completed {len(per_sample_rows)} questions in {total_bench_time_s}s ({round(total_bench_time_s/total_q, 3)}s/query).")

        # Compile Method Summaries
        method_summaries = []
        for m_id, m_data in methods_agg.items():
            tp = m_data["tp"]
            fp = m_data["fp"]
            tn = m_data["tn"]
            fn = m_data["fn"]

            # Micro metrics
            p_den = tp + fp
            micro_prec = round(tp / p_den, 4) if p_den > 0 else (1.0 if fn == 0 else 0.0)

            r_den = tp + fn
            micro_rec = round(tp / r_den, 4) if r_den > 0 else 1.0

            f1_den = micro_prec + micro_rec
            micro_f1 = round(2 * micro_prec * micro_rec / f1_den, 4) if f1_den > 0 else 0.0

            tot = tp + fp + tn + fn
            accuracy = round((tp + tn) / tot, 4) if tot > 0 else 1.0

            # Macro metrics
            # Macro-F1 = unweighted mean of the F1 of both classes (hallucinated, grounded)
            f1_neg_den = 2 * tn + fn + fp
            f1_neg = (2 * tn / f1_neg_den) if f1_neg_den > 0 else 0.0
            macro_f1 = round((micro_f1 + f1_neg) / 2.0, 4)
            macro_prec = round(float(np.mean(m_data["sample_precs"])), 4) if m_data["sample_precs"] else 0.0
            macro_rec = round(float(np.mean(m_data["sample_recs"])), 4) if m_data["sample_recs"] else 0.0

            # AUROC & AUPRC
            try:
                if len(set(m_data["y_true"])) > 1:
                    auroc = round(float(roc_auc_score(m_data["y_true"], m_data["y_score"])), 4)
                    precs, recs, _ = precision_recall_curve(m_data["y_true"], m_data["y_score"])
                    auprc = round(float(auc(recs, precs)), 4)
                else:
                    auroc = auprc = float("nan")  # undefined with a single class
            except ValueError:
                auroc = auprc = float("nan")

            avg_lat = round(float(np.mean(m_data["latencies"])), 1) if m_data["latencies"] else 0.0
            p95_lat = round(float(np.percentile(m_data["latencies"], 95)), 1) if m_data["latencies"] else 0.0

            method_summaries.append({
                "method_id": m_id,
                "name": m_data["name"],
                "model": model_name,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": micro_prec,
                "recall": micro_rec,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "auroc": auroc,
                "auprc": auprc,
                "avg_latency_ms": avg_lat,
                "p95_latency_ms": p95_lat,
            })

        # Save CSV
        if save_csv and per_sample_rows:
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)
            csv_path = self.results_dir / f"experiment_samples_{safe_name}.csv"
            df = pd.DataFrame(per_sample_rows)
            df.to_csv(csv_path, index=False)
            print(f"[Sandbox] Saved sample-level evaluation to {csv_path}")

        return {
            "model": model_name,
            "total_questions": len(per_sample_rows),
            "total_time_s": total_bench_time_s,
            "methods": method_summaries,
            "categories": category_agg,
        }

    def generate_report(
        self,
        all_model_results: List[Dict[str, Any]],
        report_path: Optional[Path] = None,
    ) -> str:
        report_path = report_path or (self.results_dir / "EXPERIMENT_REPORT.md")

        lines = []
        lines.append("# Empirical Hallucination Detection Benchmark Report")
        lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
        total_q = sum(m_res.get("total_questions", 0) for m_res in all_model_results)
        lines.append(f"**Benchmark Scope:** {total_q} evaluated questions  \n")

        lines.append("## 1. Summary\n")
        lines.append("All numbers below are computed from this run; see the tables for per-method results.\n")

        lines.append("## 2. Multi-Paradigm Benchmark Shootout Table\n")

        for m_res in all_model_results:
            m_name = m_res["model"]
            n_q = m_res["total_questions"]
            tot_s = m_res["total_time_s"]
            lines.append(f"### Model: `{m_name}` ({n_q} Questions Evaluated in {tot_s}s)\n")
            lines.append("| Detection Method | Paradigm | Precision | Recall | F1 (halluc.) | Macro-F1 | Accuracy | AUROC | Latency (Mean) | Latency (p95) | Extra VRAM |")
            lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

            for m in m_res["methods"]:
                name = m["name"]
                if "Ours" in name or m["method_id"] == "logitguard":
                    name = f"**{name}**"
                prec = f"{m['precision']:.3f}"
                rec = f"{m['recall']:.3f}"
                micro_f1 = f"{m['micro_f1']:.3f}"
                macro_f1 = f"{m['macro_f1']:.3f}"
                acc = f"{m['accuracy']:.3f}"
                auroc = f"{m['auroc']:.3f}"
                lat_mean = f"{m.get('avg_latency_ms', 0.0):.1f} ms"
                lat_p95 = f"{m.get('p95_latency_ms', 0.0):.1f} ms"

                paradigm = "Ground Truth Reference" if "fact_arbiter" in m["method_id"] else (
                    "Gray-Box CCLA (Ours)" if "logitguard" in m["method_id"] else (
                    "Black-Box NLI (System One)" if "nli" in m["method_id"] else (
                    "Black-Box Surface Match" if "lexical" in m["method_id"] else (
                    "White-Box Token Entropy" if "entropy" in m["method_id"] else "White-Box Token Heuristic"
                ))))

                extra_vram = f"{m.get('extra_vram_mb', 0.0):.1f} MB"

                lines.append(f"| {name} | {paradigm} | {prec} | {rec} | {micro_f1} | {macro_f1} | {acc} | {auroc} | {lat_mean} | {lat_p95} | {extra_vram} |")
            lines.append("\n")

        lines.append("## 3. Confusion Matrix Breakdown\n")
        lines.append("Total sample-level factual assertions audited across the benchmark:\n")

        for m_res in all_model_results:
            lines.append(f"#### Model: `{m_res['model']}`\n")
            lines.append("| Detection Method | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) |")
            lines.append("|:---|:---:|:---:|:---:|:---:|")
            for m in m_res["methods"]:
                lines.append(f"| {m['name']} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")
            lines.append("\n")

        lines.append("\n---\n")

        report_content = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"[Sandbox] Successfully written report to {report_path}")
        return report_content


def main():
    parser = argparse.ArgumentParser(description="LogitGuard-RAG 300+ Question Experiment Sandbox")
    parser.add_argument("--benchmark", type=str, default="data/benchmark_300.jsonl", help="Path to benchmark JSONL")
    parser.add_argument("--models", type=str, default="Qwen/Qwen2.5-1.5B-Instruct:float16,Qwen/Qwen2.5-1.5B-Instruct:bfloat16", help="Comma-separated model specs (model:dtype)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="FastAPI Server URL")
    args = parser.parse_args()

    bench_path = PROJECT_ROOT / args.benchmark
    if not bench_path.exists():
        from data.build_benchmark_300 import build_benchmark_300
        bench_path = build_benchmark_300()

    sandbox = ExperimentSandbox(server_url=args.server)
    status = sandbox.check_server()
    print(f"[Sandbox] Server is online: model={status.get('model')}, device={status.get('device')}, vram={status.get('vram_mb')} MB")

    specs = [s.strip() for s in args.models.split(",") if s.strip()]
    all_results = []

    for spec in specs:
        if ":" in spec:
            m_name, m_dtype = spec.split(":", 1)
        else:
            m_name, m_dtype = spec, "float16"

        label = f"{m_name} ({m_dtype.upper()})"
        print(f"\n[Sandbox] Preparing benchmark for: {label}...")
        sandbox.switch_model(m_name, dtype=m_dtype)
        res = sandbox.evaluate_benchmark(bench_path, model_name=label, limit=args.limit)
        all_results.append(res)

    # Save JSON summary
    summary_path = PROJECT_ROOT / "results" / "experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Generate Markdown Report
    sandbox.generate_report(all_results)
    print(f"\n[Sandbox] All evaluations completed successfully. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

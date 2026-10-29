# LogitGuard-RAG

Subword-Aware Context-Contrastive Logit Attribution for Real-Time Hallucination Detection in Retrieval-Augmented Generation

Saran Sharma and Atharv Sharma  
Vellore Institute of Technology, Chennai, India

Paper: docs/RESEARCH_PAPER.md

---

## Overview

Retrieval-Augmented Generation (RAG) grounds Large Language Models in retrieved documents, but models can still produce ungrounded claims. A frequent failure mode is a parametric memory override, where the generator confidently reproduces a memorized pretraining fact that contradicts the retrieved passage.

LogitGuard-RAG is a training-free, post-generation verifier based on Context-Contrastive Logit Attribution (CCLA):

1. Dual-pass teacher-forced scoring: A small verifier model scores the finished response twice (with retrieved context and without). The difference in token log-probabilities yields the Context Pull Ratio (CPR).
2. Subword-aware span scoring: A first-subword pooling rule prevents factual uncertainty on multi-token entities and numbers from being diluted by trailing continuations.
3. In-place GPU forward hook: Full-vocabulary logits are reduced to 4 scalars per token directly on GPU memory, transferring compact arrays to CPU with zero additional GPU memory overhead.

---

## Benchmark Results

Evaluated on an NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM) across the primary benchmark (320 queries, 2,198 claims: 258 hallucinated, 1,940 grounded) and development benchmark (110 queries, 559 claims):

### Primary Benchmark (2,198 claims)

| Method | Precision | Recall | F1 (halluc.) | Macro-F1 | Accuracy | AUROC | Mean Latency | Extra GPU Memory |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LLM NLI judge (FP16) | 0.376 | 0.341 | 0.358 | 0.569 | 0.863 | 0.758 | 3608.8 ms | +320 MB |
| LogitGuard-RAG (FP16) | 0.372 | 0.271 | 0.314 | 0.518 | 0.861 | 0.723 | 250.8 ms | 0.0 MB |
| LogitGuard-RAG (BF16) | 0.361 | 0.271 | 0.310 | 0.510 | 0.858 | 0.721 | 250.8 ms | 0.0 MB |
| Lexical heuristic | 0.288 | 0.504 | 0.366 | 0.511 | 0.795 | 0.697 | 34.9 ms | 0.0 MB |
| Raw token entropy | 0.144 | 0.430 | 0.215 | 0.345 | 0.632 | 0.661 | 249.8 ms | 0.0 MB |
| Min concept prob | 0.144 | 0.438 | 0.217 | 0.338 | 0.629 | 0.642 | 249.7 ms | 0.0 MB |

### Confusion Matrix on Primary Benchmark (258 Hallucinated Claims)

| Method | TP | FP | TN | FN | False Alarm Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| LLM NLI judge (FP16) | 88 | 146 | 1794 | 170 | 7.5% |
| LogitGuard-RAG (FP16) | 70 | 118 | 1822 | 188 | 6.1% |
| LogitGuard-RAG (BF16) | 70 | 124 | 1816 | 188 | 6.4% |
| Lexical heuristic | 130 | 322 | 1618 | 128 | 16.6% |
| Raw token entropy | 111 | 662 | 1278 | 147 | 34.1% |

LogitGuard-RAG achieves 0.723 AUROC at 250.8 ms mean latency (14.4x faster than the LLM judge) while producing the lowest false alarm count (118 in FP16).

---

## Method Summary

1. Inputs: Context C, query q, response Y = (y_1, ..., y_M).
2. Sequences:
   - x_ctx = Concat(C, q, Y)
   - x_prior = Concat(empty, q, Y)
3. Normalized entropy and surprisal:
   - H_norm(y_i) = - 1/log|V| * sum(P * log P)
   - S_norm(y_i) = min(1.0, -log P(y_i) / 8.0)
   - Delta_eff = (z_top1 - z_top2) * I[P(y_i) >= 0.20]
   - U_raw(y_i) = (0.5 * H_norm + 0.5 * S_norm) / (1.0 + 0.25 * Delta_eff)
4. Context Pull Ratio:
   - CPR_i = log P(y_i | C, q, y_<i) - log P(y_i | empty, q, y_<i)
5. Modulated token risk:
   - r_i = 3.5 * (U_raw - 0.5) - 1.5 * tanh(CPR_i) + 2.8 * I_ungr(i)
   - U_ctx(y_i) = 1 / (1 + exp(-r_i))
6. First-subword span pooling:
   - U_span(s_k) = max(0.92 * A_k, 0.65 * A_k + 0.35 * F_k)
   where A_k is the span max token risk and F_k is the first-subword risk.
7. Operational tiers:
   - Grounded: U_span <= 0.58
   - Moderate Risk: 0.58 < U_span < 0.80
   - High Risk: U_span >= 0.80

---

## Project Structure

```
logitguard/
├── config.yaml              # Configuration and thresholds
├── requirements.txt         # Dependencies
├── run_benchmark.py         # Benchmark runner
├── docs/
│   ├── RESEARCH_PAPER.md    # Full research manuscript
│   └── adr/                 # Architecture decision records
├── logitguard/
│   ├── model.py             # GPU forward hook and model loader
│   ├── scoring.py           # Scoring formulas and ablations
│   ├── masker.py            # Syntactic filtering and header protection
│   ├── pipeline.py          # Verification pipeline
│   ├── arbiter.py           # Evaluation arbiter
│   └── align.py             # Character offset alignment
├── data/
│   └── benchmark_300.jsonl  # Multi-task benchmark dataset
└── results/
    ├── main_table.csv       # Summary metrics
    ├── latency.csv          # Latency measurements
    └── experiment_summary.json
```

---

## Setup and Usage

### 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128  # CUDA build (RTX 50-series needs CUDA 12.8)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The first start downloads `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB) from Hugging Face, so it needs internet once.

### 2. Run Web Service

On Windows, double-click `run.bat` (it opens the browser once the model has loaded), or run:

```bash
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000. Model loading and warm-up take roughly 20–40 seconds.

### 3. Run Benchmarks

The benchmark sends every query to the running web service, so start the server (step 2) first and run this in a second terminal:

```bash
python run_benchmark.py            # full benchmark (data/benchmark_300.jsonl)
python run_benchmark.py --limit 20 # quick check
```

### 4. Run Tests

```bash
python -m pytest tests
```

---

## Formulation Note

AI assistance was utilized during the research process to explore, derive, and numerically verify the mathematical formulations, including the Context Pull Ratio (CPR) weighting and first-subword span pooling rules.

---

## Citation

```bibtex
@article{sharma2026logitguard,
  title={LogitGuard-RAG: Subword-Aware Context-Contrastive Logit Attribution for Real-Time Hallucination Detection in Retrieval-Augmented Generation},
  author={Sharma, Saran and Sharma, Atharv},
  journal={Vellore Institute of Technology, Chennai},
  year={2026}
}
```

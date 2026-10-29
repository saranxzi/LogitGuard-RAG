# LogitGuard-RAG: Subword-Aware Context-Contrastive Logit Attribution for Real-Time Hallucination Detection in Retrieval-Augmented Generation

**Authors:** Saran Sharma and Atharv Sharma  
**Affiliation:** Vellore Institute of Technology, Chennai, India  
**Target Hardware Baseline:** NVIDIA GeForce RTX 5070 Laptop GPU (8GB GDDR7, CUDA 12.8, PyTorch 2.11)  
**Verifier Model:** `Qwen/Qwen2.5-1.5B-Instruct` (FP16 / BF16, $|\mathcal{V}| = 151,936$, ~2.95 GB VRAM)  
**Evaluated Benchmarks:** 
- Primary Benchmark: 320 queries, 2,198 factual claims (258 hallucinated, 1,940 grounded)
- Development Benchmark: 110 queries, 559 factual claims

---

## Abstract

Retrieval-Augmented Generation (RAG) grounds Large Language Models (LLMs) in retrieved documents, yet generated answers can still contain claims that the evidence does not support. A particularly harmful case is a **parametric memory override**, in which the model confidently reproduces a memorized fact that contradicts the retrieved passage. Existing detectors either rely on an auxiliary LLM judge or repeated sampling, which adds seconds of latency, or on single-pass confidence scores, which cannot separate a confident hallucination from a grounded paraphrase.

We present **LogitGuard-RAG**, a training-free, post-generation verifier built on **Context-Contrastive Logit Attribution (CCLA)**. A small verifier model scores the finished response twice with teacher forcing, once with and once without the retrieved context, and the difference in token log-probabilities gives the **Context Pull Ratio (CPR)**. CPR is fused with normalized entropy, token surprise, a gated confidence margin and linguistic role weights, and token scores are aggregated into span-level risk labels through **first-subword-aware pooling**. A GPU-side hook reduces full-vocabulary logits to four scalars per token before transfer to the CPU.

On a 2,198-claim benchmark, LogitGuard-RAG reaches an **AUROC of 0.723 in FP16 (0.721 in BF16)** and **0.851 on a 559-claim development set** at a mean latency of **250.8 ms**, compared with **0.758 and 3,608.8 ms for an LLM judge** ($14.4\times$ faster).

---

## I. Introduction

Large Language Models (LLMs) are routinely deployed for question answering, summarization, document analysis, and enterprise search. However, their fluency does not imply faithfulness: models regularly produce statements unsupported by evidence—hallucinations—phrased as confidently as correct statements. Retrieval-Augmented Generation (RAG) addresses part of this by conditioning generation on retrieved documents.

Yet, retrieval does not guarantee faithfulness. Two failure modes dominate:
1. **Context Gaps:** The retrieved passages lack the requested detail, and the model invents it.
2. **Parametric Memory Overrides:** The passage contradicts what the model learned during pretraining, and the model outputs the memorized answer anyway. This is especially challenging because the model is often highly confident in the wrong answer, leaving entropy- and confidence-based detectors silent.

Detectors that flag every word not copied verbatim mislabel legitimate paraphrases. A practical verifier must reason about the relationship between the response and its context and run fast enough to precede user display.

Existing approaches fall into three families:
- **Sampling-based consistency:** *SelfCheckGPT* [4], Semantic Entropy [5]. High cost ($K$ full generations).
- **External / Supervised Judges:** *FActScore* [6], LLM NLI judges, *LettuceDetect* [14], *Luna* [15]. High latency or extra model hosting.
- **Single-pass uncertainty:** Token entropy [8], minimum concept probability [9], *Lookback Lens* [11], Context-Aware Decoding (*CAD*) [12].

**LogitGuard-RAG** acts as a post-generation verification layer placed between a RAG generator and the downstream application:
- Dual-pass teacher-forced scoring on a small verifier model ($x_{\text{ctx}}$ vs. $x_{\text{prior}}$).
- Log-probability difference yields the **Context Pull Ratio (CPR)**.
- First-subword-aware pooling prevents multi-token entities and numbers from being diluted.
- In-place GPU forward hook reduces logits to 4 scalars per token before CPU transfer, maintaining **0.0 MB additional GPU memory overhead**.

---

## II. Literature Review

### Table I: Summary of the Nine Most Closely Related Studies

| Study | Year | Approach | Signal / Access | Key Observation |
|:---|:---:|:---|:---|:---|
| **Manakul et al. [4]** | 2023 | Consistency across sampled responses | Black-box text | Hallucinated sentences vary across samples (*SelfCheckGPT*). |
| **Farquhar et al. [5]** | 2024 | Entropy over meaning clusters | Multiple samples + NLI | Paraphrases must not count as uncertainty (Semantic Entropy). |
| **Min et al. [6]** | 2023 | Atomic fact decomposition and checking | External verifier | Factuality is best judged per claim (*FActScore*). |
| **Niu et al. [7]** | 2024 | Word-level annotated RAG corpus | Benchmark | Single responses mix supported and unsupported spans (*RAGTruth*). |
| **Kovács et al. [14]** | 2025 | Fine-tuned token classifier | Trained encoder | Compact models can localize unsupported tokens (*LettuceDetect*). |
| **Fadeeva et al. [8]** | 2024 | Token-level uncertainty per claim | Generator logits | Logits carry claim-level signal without sampling. |
| **Chuang et al. [11]** | 2024 | Context-vs-generated attention ratio | Attention maps | Attention placed on context predicts faithfulness (*Lookback Lens*). |
| **Shi et al. [12]** | 2024 | Contrast of with/without context | Logits, two passes | Contrast isolates context influence (*CAD*). |
| **Sun et al. [17]** | 2025 | Decoupling context & parametric knowledge | Internal states | Hallucinations arise when parametric knowledge overrides context (*ReDeEP*). |
| **LogitGuard-RAG (Ours)** | **2026** | **Dual-pass CPR fused with uncertainty & linguistics** | **Verifier logits, 2 teacher-forced passes** | **Training-free span-level risk at sub-second latency (250 ms, 0.0 MB extra VRAM).** |

### Table II: Interpretation of Confidence and Context Pull

| | $\text{CPR} > 0$ (Context Helps) | $\text{CPR} \le 0$ (Context Does Not Help) |
|:---|:---|:---|
| **Confident** | Grounded and copied or closely paraphrased. | Possible **parametric memory override**; risky if token is a factual anchor. |
| **Uncertain** | Grounded, with variable wording. | Likely fabricated detail filling a gap in the context. |

---

## III. Proposed Methodology & Architecture

```
                                  ┌──────────────────────────┐
                                  │   RAG Input: (C, q)      │
                                  └─────────────┬────────────┘
                                                │
                                                ▼
                                  ┌──────────────────────────┐
                                  │ Generator (any LLM / API)│
                                  └─────────────┬────────────┘
                                                │ Response Y = (y_1, ..., y_M)
                                                ▼
                                  ┌──────────────────────────┐
                                  │ Alignment & Char Offsets │
                                  └──────┬────────────┬──────┘
                                         │            │
                   ┌─────────────────────┘            └────────────────────┐
                   ▼                                                       ▼
       ┌────────────────────────┐                             ┌────────────────────────┐
       │ Context Pass (C, q, Y) │                             │  Prior Pass (∅, q, Y)  │
       │     Teacher-Forced     │                             │     Teacher-Forced     │
       └───────────┬────────────┘                             └────────────┬───────────┘
                   │                                                       │
                   └─────────────────────┬─────────────────────────────────┘
                                         ▼
                            ┌────────────────────────┐
                            │ GPU Hook on lm_head    │
                            │ [T, |V|] -> 4 scalars  │
                            └────────────┬───────────┘
                                         │ Token Signals: H_bar, S_bar, Delta_eff, CPR_i
                                         ▼
                            ┌────────────────────────┐  Linguistic Filter
                            │ CCLA Fusion Engine     │ ◄──────────────────
                            │ r_i -> U_ctx(y_i)      │  (w_i, anchor, match)
                            └────────────┬───────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ First-Subword Pooling  │
                            │ U_span(s_k)            │
                            └────────────┬───────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ Operational Tiers      │
                            │ Grounded/Mod./High Risk│
                            └────────────────────────┘
```

### A. Dual-Pass Teacher Forcing
The verifier evaluates two input sequences under teacher forcing:
$$\mathbf{x}_{\text{ctx}} = \text{Concat}(C, q, Y) \tag{1}$$
$$\mathbf{x}_{\text{prior}} = \text{Concat}(\varnothing, q, Y) \tag{2}$$
Under teacher forcing, response tokens are provided as input, and token distributions are obtained from a single parallel forward pass per condition.

### B. Shifted Log-Sum-Exp & Token Uncertainty
Let $z_{i, v}$ be the unnormalized logit for vocabulary item $v$ at response position $i$:
$$\log P(y_i \mid \mathbf{x}) = z_{i, y_i} - \left[ m_i + \log \sum_{v \in \mathcal{V}} \exp(z_{i, v} - m_i) \right] \tag{3}$$
where $m_i = \max_v z_{i, v}$.

The **Normalized Predictive Entropy** under the context pass is:
$$\bar{H}(y_i) = - \frac{1}{\log |\mathcal{V}|} \sum_{v \in \mathcal{V}} P(v \mid \mathbf{x}_{\text{ctx}}) \log P(v \mid \mathbf{x}_{\text{ctx}}) \in [0, 1] \tag{4}$$

The **Normalized Surprise** for the realized token (with $s_i = -\log P(y_i \mid \mathbf{x}_{\text{ctx}})$ capped at 8 nats):
$$\bar{S}(y_i) = \min(1, s_i / 8) \tag{5}$$

The **Gated Top-2 Margin** rewards decisive preference only when the realized token holds $\ge 20\%$ probability mass:
$$\Delta_i^{\text{eff}} = (z_{i, (1)} - z_{i, (2)}) \cdot \mathbb{I}[P(y_i \mid \mathbf{x}_{\text{ctx}}) \ge 0.20] \tag{6}$$

Combining these yields the **Raw Uncertainty**:
$$U_{\text{raw}}(y_i) = \frac{0.5 \bar{H}(y_i) + 0.5 \bar{S}(y_i)}{1 + 0.25 \Delta_i^{\text{eff}}} \tag{7}$$

### C. Context Pull Ratio (CPR)
$$\text{CPR}_i = \log P(y_i \mid C, q, y_{<i}) - \log P(y_i \mid \varnothing, q, y_{<i}) \tag{8}$$
$\text{CPR}_i > 0$ indicates context pull; $\text{CPR}_i \le 0$ indicates lack of context pull or parametric conflict.

### D. Context-Contrastive Logit Attribution (CCLA)
The ungrounded-anchor indicator captures anchor tokens absent from context with low or negative context pull:
$$\mathcal{I}_{\text{ungr}}(i) = \mathbb{I}[\text{anchor}_i \land \neg\text{match}_i \land (\text{CPR}_i \le 0.35) \land (s_i \ge 4 \lor \text{CPR}_i \le -0.1)] \tag{9}$$

The fused logit $r_i$ and modulated risk $U_{\text{ctx}}(y_i)$ are:
$$r_i = 3.5(U_{\text{raw}}(y_i) - 0.5) - 1.5 \tanh(\text{CPR}_i) + 2.8 \mathcal{I}_{\text{ungr}}(i) \tag{10}$$
$$U_{\text{ctx}}(y_i) = \sigma(r_i) = \frac{1}{1 + \exp(-r_i)} \tag{11}$$

### E. Linguistic Filtering & Structural Header Protection
Tokens receive weights $w_i$ based on POS tags and dependency parse:
- Factual anchors (proper nouns, common nouns, numbers, dates) receive $w_i = 1.0$.
- Function words (prepositions, determiners) receive $w_i = 0.1$.
- Colon-terminated structural headers (`Completion Year:`, `Height:`) are recognized and excluded from anchor detection ($w_i = 0.0$).

### F. First-Subword Span Pooling
To resolve BPE subword dilution on multi-token entities and dates, let $A_k = \max_{y_i \in s_k} w_i U_{\text{ctx}}(y_i)$ and $F_k = w_{i_0} U_{\text{ctx}}(y_{i_0})$:
$$U_{\text{span}}(s_k) = \text{clip}_{[0, 1]}\left(\max\{0.92 A_k, \; 0.65 A_k + 0.35 F_k\}\right) \tag{12}$$

### G. Operational Risk Tiers
$$h(s_k) = \begin{cases} 
\text{High Risk}, & U_{\text{span}}(s_k) \ge 0.80 \\
\text{Moderate Risk}, & 0.58 < U_{\text{span}}(s_k) < 0.80 \\
\text{Grounded}, & U_{\text{span}}(s_k) \le 0.58
\end{cases} \tag{13}$$

---

## IV. Experimental Results

### Table III: Classification Performance on Primary and Development Benchmarks

| Method | Precision | Recall | $F_1$ (halluc.) | Macro-$F_1$ | Accuracy | AUROC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Primary Benchmark: 320 queries, 2,198 claims** | | | | | | |
| LLM NLI judge, FP16 | 0.376 | 0.341 | 0.358 | 0.569 | 0.863 | **0.758** |
| LLM NLI judge, BF16 | 0.374 | 0.341 | 0.357 | 0.568 | 0.862 | 0.756 |
| **LogitGuard-RAG, FP16** | **0.372** | **0.271** | **0.314** | **0.518** | **0.861** | **0.723** |
| **LogitGuard-RAG, BF16** | **0.361** | **0.271** | **0.310** | **0.510** | **0.858** | **0.721** |
| Lexical/entity heuristic | 0.288 | 0.504 | 0.366 | 0.511 | 0.795 | 0.697 |
| Raw token entropy | 0.144 | 0.430 | 0.215 | 0.345 | 0.632 | 0.661 |
| Minimum concept probability | 0.144 | 0.438 | 0.217 | 0.338 | 0.629 | 0.642 |
| **Development Benchmark: 110 queries, 559 claims** | | | | | | |
| LLM NLI judge, FP16 | 0.442 | 0.548 | 0.489 | 0.702 | 0.875 | **0.874** |
| **LogitGuard-RAG, FP16** | **0.463** | **0.403** | **0.431** | **0.654** | **0.881** | **0.851** |
| Lexical/entity heuristic | 0.288 | 0.516 | 0.370 | 0.697 | 0.804 | 0.828 |
| Raw token entropy | 0.172 | 0.339 | 0.228 | 0.553 | 0.745 | 0.736 |
| Minimum concept probability | 0.130 | 0.290 | 0.179 | 0.393 | 0.704 | 0.581 |

### Table IV: Confusion Matrices on Primary Benchmark (258 Hallucinated, 1,940 Grounded Claims)

| Method | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) |
|:---|:---:|:---:|:---:|:---:|
| LLM NLI judge, FP16 | 88 | 146 | 1794 | 170 |
| **LogitGuard-RAG, FP16** | **70** | **118** | **1822** | **188** |
| **LogitGuard-RAG, BF16** | **70** | **124** | **1816** | **188** |
| Lexical/entity heuristic | 130 | 322 | 1618 | 128 |
| Raw token entropy | 111 | 662 | 1278 | 147 |

*Finding:* LogitGuard-RAG produces the **fewest false alarms of any method (118 in FP16)**—28 fewer than the LLM judge and 204 fewer than the lexical heuristic.

### Table V: Mean Verification Latency & Additional GPU Memory

| Method | Mean Latency | Additional GPU Memory |
|:---|:---:|:---:|
| LLM NLI judge, FP16 | 3,608.8 ms | +320 MB |
| **LogitGuard-RAG, FP16** | **250.8 ms** | **0.0 MB** |
| **LogitGuard-RAG, BF16** | **250.8 ms** | **0.0 MB** |
| Raw token entropy | 249.8 ms | 0.0 MB |

*Finding:* LogitGuard-RAG runs in **250.8 ms** ($14.4\times$ faster than the LLM judge) with **0.0 MB extra GPU memory**.

---

## V. Key Insights & Deployment Cascade

1. **Honest Accuracy-Latency Trade-Off:** The LLM judge retains an advantage of $0.035$ AUROC ($0.758$ vs. $0.723$) by recovering multi-sentence contradictions that probability contrasts miss. However, LogitGuard-RAG executes in $250.8\text{ ms}$ (vs. $3,608.8\text{ ms}$) and produces fewer false alarms ($118$ vs. $146$).
2. **Recommended Cascade Deployment:**
   - **Grounded ($\le 0.58$):** Passed through directly with zero delay.
   - **High Risk ($\ge 0.80$):** Flagged immediately or triggers regeneration.
   - **Moderate Risk ($0.58 < U < 0.80$):** Forwarded to the heavier semantic LLM judge.
   - *Result:* Preserves 90%+ of the speedup while capturing multi-sentence reasoning for ambiguous cases.
3. **Hardware Precision Stability:** FP16 and BF16 produce nearly identical results ($\Delta \text{AUROC} \le 0.002$), proving robustness to floating-point representation.

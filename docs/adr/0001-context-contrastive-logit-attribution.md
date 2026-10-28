# ADR 0001: Context-Contrastive Logit Attribution (CCLA) and BPE Onset Pooling

- **Status:** Accepted
- **Deciders:** Antigravity AI Engineering Team
- **Date:** 2026-10-28
- **Tags:** architecture, rag, logits, uncertainty-estimation, nlp

## Context and Problem Statement

LogitGuard-RAG evaluates factual hallucination risk by probing unnormalized logits directly from the generator's `lm_head`. Prior to this decision, the mathematical scoring core (Equations 1–5) evaluated token uncertainty purely via Predictive Token Shannon Entropy $H(y_i)$ and Top-2 Logit Margin $\Delta_i$.

This pure internal-hesitation formulation exhibited two critical failure modes in Retrieval-Augmented Generation (RAG):
1. **Parametric Prior Override (False Negatives):** When the model confidently generates an outdated or ungrounded fact from pre-training weights, its internal entropy is low ($H \to 0$) and margin is high ($\Delta \gg 0$). The model emits the hallucination without hesitation, evading entropy-only detection.
2. **Synonym & Paraphrase Hesitation (False Positives):** When the model generates a valid paraphrase of the retrieved context (e.g. `"completed"` vs `"finished"`), probability is distributed across synonyms, causing $\Delta \to 0$ and triggering false alarms.
3. **BPE Subword Dilution:** In multi-token named entities and dates (e.g. `"Alexander Fleming"`, `"1889"`), factual uncertainty is concentrated on the **onset subword**, while suffix subwords are deterministic continuations. Standard mean aggregation diluted the onset signal.

## Decision Drivers

- **Zero Auxiliary Neural Network Overhead:** Must not require secondary 150M–7B classifier models or Black-Box multi-sampling (SelfCheckGPT $>1400\text{ ms}$, $+1.5\text{ GB}$ VRAM).
- **Hard VRAM Budget:** Must run entirely within an 8GB VRAM constraint (RTX 5070 Laptop GPU) without triggering Windows WDDM PCIe memory thrashing.
- **Latency Budget:** Total probe and scoring latency must remain below $35\text{ ms}$.
- **Mathematical Boundedness:** Scores must be strictly bounded in $[0, 1]$ independent of vocabulary size $\vert\mathcal{V}\vert$.

## Considered Options

1. **Option A: Deploy a Secondary NLI Cross-Encoder (`deberta-v3-small`)**
   - *Pros:* High discriminative accuracy on premise-hypothesis pairs.
   - *Cons:* Adds 140MB permanent memory, an extra inference forward pass, and violates the single-pass gray-box thesis of LogitGuard.
2. **Option B: Multi-Sample Consistency Probing (SelfCheckGPT)**
   - *Pros:* Black-box compatibility.
   - *Cons:* $1400+\text{ ms}$ latency; 5x GPU forward passes.
3. **Option C: Context-Contrastive Logit Attribution (CCLA) + In-Hook GPU Reduction (Selected)**
   - *Pros:* Single-pass dual-stream evaluation ($P(y_i \mid C, Q)$ vs $P(y_i \mid \varnothing, Q)$); zero extra model weights; executes in $<10\text{ ms}$; mathematically isolates parametric prior overrides from grounded generations.

## Decision Outcome

Chosen: **Option C (CCLA + BPE Onset Pooling)**.

### Mathematical Specification

1. **Realized-Token Bounded Uncertainty:**
   $$U_{\text{raw}}(y_i) = \frac{0.5 \cdot \frac{H(y_i)}{\ln\vert\mathcal{V}\vert} + 0.5 \cdot (1 - P(y_i \mid C, Q, y_{<i}))}{1 + \alpha \Delta_i} \in [0, 1]$$
   where $\ln\vert\mathcal{V}\vert = \ln(151936) \approx 11.9312$, $\alpha = 0.25$.

2. **Context Pull Ratio (CPR):**
   $$\text{CPR}_i = \log P(y_i \mid C, Q, y_{<i}) - \log P(y_i \mid \varnothing, Q, y_{<i})$$

3. **Context-Contrastive Modulation:**
   $$U_{\text{ctx}}(y_i) = \sigma\left(\beta_1(U_{\text{raw}}(y_i) - 0.5) - \beta_2 \tanh(\text{CPR}_i) + \beta_3 \cdot \mathbb{I}[\text{CPR}_i \le 0.05 \land \text{LexMatch}(y_i, C) = 0 \land \text{is\_anchor}_i]\right)$$
   *(When context is absent, $U_{\text{ctx}}(y_i) = U_{\text{raw}}(y_i)$).*

4. **BPE Onset-Weighted Top-$k$ Span Aggregation:**
   $$U_{\text{span}}(s_k) = 0.65 \cdot \max_{i \in s_k} U_{\text{ctx}}(y_i) + 0.35 \cdot U_{\text{ctx}}(y_{i_0})$$

## Consequences

- **Positive:** Catches confident parametric hallucinations where raw entropy fails.
- **Positive:** Suppresses false alarms on valid synonym paraphrases where context provides positive pull ($\text{CPR} > 0$).
- **Positive:** Probe latency drops to $<1\text{ ms}$ via in-hook GPU reductions.
- **Neutral:** When context is provided, one additional teacher-forced forward pass without context is executed ($\sim 8\text{ ms}$). Total latency remains $<30\text{ ms}$.

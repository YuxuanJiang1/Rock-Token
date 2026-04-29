# Rock Token Identification — Part 1 Report

**Date:** 2026-04-26
**Authors:** Dipta, Chao, Zhao
**Pipeline version:** 0.4.0

---

## 1. Setup

### Models

| Role | Model | Params |
|------|-------|--------|
| Teacher | Qwen/Qwen3-30B-A3B-Instruct-2507 | 30B MoE (3B active) |
| Pre-OPD Student (θ₀) | Qwen/Qwen3-4B-Instruct-2507 | 4B |
| Post-OPD Student (θ\*) | RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k | 4B (distilled) |

### Prompt Set

1,500 prompts from 4 datasets (all train splits — zero overlap with evaluation benchmarks):

| Source | Count | Domain |
|--------|-------|--------|
| MATH (Hendrycks) | 500 | Competition math |
| GSM8K | 500 | Grade-school math |
| MBPP | 300 | Code generation |
| Alpaca | 200 | General instructions |

### Generation

- 3 sampled outputs per prompt at temperature=1.0 (not greedy)
- Max 16,384 tokens per output; truncated outputs discarded
- Total: **4,500 sequences** → **4,814,434 token positions** measured

---

## 2. Recalcitrance Criterion

A token instance is classified as **Rock** if it satisfies both conditions:

1. **High loss before training:** `KL(teacher ‖ θ₀) ≥ τ_high`
2. **Barely improved (or regressed):** `improvement < δ`, where `improvement = KL_before − KL_after`

### Threshold values (derived from data)

| Parameter | Percentile | Value | Interpretation |
|-----------|-----------|-------|----------------|
| τ\_high | 80th of loss\_before | **0.195** | Only the top 20% hardest tokens qualify |
| δ | 20th of improvement | **−0.0001** | Essentially zero — Rock tokens must have negative or near-zero improvement |

The δ threshold being ~0 is a key finding: **80% of high-loss tokens did improve after OPD**. Rock Tokens are the remaining 20% that didn't — and most of them actually got *worse* (negative improvement). This is genuine recalcitrance: the student not only failed to learn these tokens, but was pushed further from the teacher by training.

---

## 3. Rock Token Census

| Metric | Value |
|--------|-------|
| Total token positions | 4,814,434 |
| Rock instances | 296,657 |
| **Rock fraction** | **6.2%** |
| Unique Rock token types (freq ≥ 30) | 200 reported |

The 6.2% fraction is within the expected 5–20% range from the literature. This means the thresholds are well-calibrated — selective enough to be meaningful, inclusive enough to capture a real population.

---

## 4. Scatter Plot — Four-Quadrant Structure

![Loss scatter plot](scatter_loss.png)

The scatter plot of `loss_before` (x) vs `loss_after` (y) reveals clear quadrant structure:

- **Easy (green, near origin):** The vast majority of tokens. Low KL before and after — the student already matched the teacher on these.
- **Learned (blue, below diagonal):** High KL before training, low after. These are the OPD success stories — tokens where distillation worked.
- **Rock (red, above diagonal):** High KL before AND after. The student was bad and stayed bad — or got worse. These cluster above the identity line, meaning loss_after > loss_before.
- **Regressed (orange, upper-left):** Low KL before, higher after. The student was fine but got worse. A small but real population.

The clear separation between Learned (blue) and Rock (red) validates the recalcitrance criterion — these are genuinely different populations, not just noise in a continuous distribution.

---

## 5. Entropy Correlation — Rock Tokens ≠ Forking Tokens

![Entropy correlation](entropy_correlation.png)

| Statistic | Value |
|-----------|-------|
| Pearson r | 0.136 |
| p-value | 0.055 (not significant at α=0.05) |

There is **no significant correlation** between a token's Rock rate and its average student entropy. This is a critical finding for positioning the paper:

**Rock Tokens are NOT forking tokens.** Wang et al. (2025) identified "forking tokens" as high-entropy decision points where the model is uncertain. Our Rock Tokens exist across all entropy levels — some are high-entropy (the student is confused) and some are low-entropy (the student is confidently wrong). The phenomenon we're capturing is distinct from uncertainty-based token categories.

The weak positive trend (r=0.136) suggests a slight tendency for higher-entropy tokens to be more recalcitrant, but this is far from deterministic.

---

## 6. Two Rankings — Rate vs. Count

We report Rock Tokens with two complementary rankings, each answering a different question:

### 6a. Rate-Ranked (proportionally worst)

*"Which tokens does the student fail at most consistently?"*

| Rank | Token | Freq | Rock | Rate | KL₀ | KL\* | Δ |
|------|-------|------|------|------|-----|------|---|
| 1 | " little" | 42 | 23 | 54.8% | 0.88 | 1.02 | −0.14 |
| 2 | " flexible" | 40 | 21 | 52.5% | 1.33 | 1.38 | −0.05 |
| 3 | " Initialize" | 31 | 16 | 51.6% | 0.60 | 0.92 | −0.31 |
| 4 | " economic" | 31 | 16 | 51.6% | 0.89 | 1.04 | −0.15 |
| 5 | "Long" | 31 | 16 | 51.6% | 2.03 | 2.08 | −0.05 |
| 6 | " sun" | 33 | 17 | 51.5% | 1.21 | 1.30 | −0.09 |
| 7 | " scientific" | 30 | 15 | 50.0% | 0.90 | 0.92 | −0.02 |
| 8 | " fun" | 30 | 15 | 50.0% | 0.89 | 1.03 | −0.15 |
| 9 | " Consider" | 59 | 29 | 49.2% | 1.44 | 1.62 | −0.18 |
| 10 | " integration" | 31 | 15 | 48.4% | 0.66 | 0.81 | −0.15 |

These tokens are Rock in ~50% of their appearances. Nearly all have negative improvement (Δ < 0), meaning the student got *worse* at them after OPD.

### 6b. Count-Ranked (highest absolute impact)

*"Which tokens contribute the most rock instances — i.e., affect the most positions?"*

| Rank | Token | Freq | Rock | Rate | KL₀ | KL\* | Δ |
|------|-------|------|------|------|-----|------|---|
| 1 | **" Python"** | 1138 | 371 | 32.6% | 0.88 | 1.27 | −0.39 |
| 2 | **" It"** | 856 | 263 | 30.7% | 0.79 | 0.83 | −0.04 |
| 3 | " starting" | 400 | 123 | 30.8% | 0.49 | 0.53 | −0.03 |
| 4 | " Note" | 267 | 91 | 34.1% | 1.04 | 1.08 | −0.04 |
| 5 | " handle" | 259 | 83 | 32.0% | 0.89 | 0.85 | +0.04 |
| 6 | " However" | 264 | 83 | 31.4% | 0.84 | 0.77 | +0.08 |
| 7 | " assumes" | 200 | 69 | 34.5% | 1.03 | 1.03 | −0.00 |
| 8 | " clean" | 215 | 65 | 30.2% | 0.93 | 0.94 | −0.00 |
| 9 | " ensures" | 149 | 57 | 38.3% | 0.74 | 0.81 | −0.07 |
| 10 | " fix" | 144 | 52 | 36.1% | 0.50 | 0.53 | −0.03 |

**" Python" dominates with 371 rock instances** — 3× the second-ranked token. This likely reflects the student being trained primarily on math (OPD with math prompts) while struggling with code-related vocabulary.

### Why both rankings matter

The count-ranked list is more actionable for Part 2 (masking experiments), because:
- High-frequency tokens appear in many evaluation samples → measurable benchmark effects
- Masking " Python" (1138 occurrences) will visibly affect code benchmarks
- Masking " little" (42 occurrences) may produce noisy results

The rate-ranked list tells us about the *nature* of recalcitrance: which tokens are most consistently problematic, independent of how often they appear.

---

## 7. Qualitative Token Categories

Inspecting the full 200 Rock Tokens, we observe several semantic clusters:

| Category | Examples | Interpretation |
|----------|----------|----------------|
| **Code/technical** | " Python", " Initialize", " fix", " clean", " handle", " Algorithm" | Student trained on math, struggles with code vocabulary |
| **Reasoning connectives** | " However", " Consider", " Note", " assumes", " ensures" | Discourse markers that structure chain-of-thought — the student misaligns on *how* to reason, not just *what* to reason about |
| **General content words** | " little", " flexible", " sun", " economic", " fun" | No obvious domain — suggests some vocabulary items are intrinsically harder for the 4B student to align with the 30B teacher |
| **Formatting/structure** | "Long", " explicit", " pad", "roots" | Tokens involved in output structure |

The code/technical cluster is notable: the OPD training used 5K math prompts, so the student improved on math tokens but made no progress (or regressed) on code tokens that appear in mixed-domain generation.

---

## 8. Key Takeaway: Most Rock Tokens Regressed

A surprising finding: the majority of top Rock Tokens have **negative improvement** — their KL divergence with the teacher actually *increased* after OPD training. This means on-policy distillation doesn't just fail to help on these tokens; it actively pushes the student *away* from the teacher.

This has direct implications for Part 2: if these tokens are Stumbling Blocks (masking them helps), it suggests that the OPD loss on these positions generates harmful gradients. If they're Pillars (masking hurts), it means the student is learning something useful despite the superficial KL increase.

---

## 9. Limitations

1. **Single distillation run.** Results are from one OPD checkpoint. Repeating with different seeds would strengthen the claims.
2. **Rate estimates at moderate frequencies.** Top rate-ranked tokens have 30–60 occurrences. Reliable, but not as stable as the count-ranked tokens (100+).
3. **Approximate KL in Phase 2.** The optimized pipeline uses top-256 teacher log-probs cached in RAM. This is >99.99% accurate but not exact.
4. **Mixed prompt domains.** Rock Token identity may be partially driven by domain mismatch (math-trained student evaluated on code prompts) rather than intrinsic token difficulty.

---

## 10. Next Steps (Part 2)

With 200 Rock Tokens identified, the next phase is **causal intervention via masking experiments** to determine which are Pillars and which are Stumbling Blocks:

1. **Baseline evaluation** — Run GSM8K, MMLU, HumanEval, IF-Eval on the unmasked student
2. **Individual knockout** — Mask each Rock Token (logit → −∞) one at a time, re-evaluate
3. **Cumulative removal curves** — Greedy removal by count rank, measure when performance crashes
4. **Group masking** — Remove entire semantic categories (code tokens, reasoning connectives, etc.)

The count-ranked list is the primary input for Part 2, as these tokens have enough frequency to produce measurable benchmark effects.

---

*Full data: `rock_tokens_by_rate.csv`, `rock_tokens_by_count.csv`*
*Plots: `scatter_loss.png`, `entropy_correlation.png`*
*Pipeline config: `config.yaml` at project root*

---

# Part 2: Causal Analysis via Masking Experiments

## 11. Evaluation Benchmarks

Part 2 uses a benchmark suite focused on math competition reasoning (the student's training domain) plus instruction following as a non-math control:

| Benchmark | Source | Size | Answer format | Role |
|-----------|--------|------|---------------|------|
| MATH-500 | `HuggingFaceH4/MATH-500` | 500 | LaTeX (`\boxed{}`) | Workhorse — high statistical power |
| AIME 2024 | `HuggingFaceH4/aime_2024` | 30 | Integer (0–999) | Prestige — hard competition math |
| AIME 2025 | `MathArena/aime_2025` | 30 | Integer | Prestige — hard competition math |
| HMMT Feb 2025 | `MathArena/hmmt_feb_2025` | 30 | LaTeX expression | Prestige — hardest math |
| IF-Eval | `google/IFEval` | 541 | Constraint checkers | Non-math control |

Answer comparison uses a layered approach: normalized string match → numeric comparison → SymPy symbolic comparison (LaTeX parsed via `lark` backend). This handles equivalent forms like `\frac{1}{2}` vs `0.5` and rationalized radicals.

---

## 12. Step 0 — Baseline (Unmasked Student)

**Model:** `RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k` (post-OPD, 4B)
**Decoding:** Greedy (temperature=0), deterministic, seed=42

### Overall Results

| Benchmark | Correct | Total | Accuracy |
|-----------|---------|-------|----------|
| MATH-500 | 370 | 500 | **74.0%** |
| AIME 2024 | 7 | 30 | 23.3% |
| AIME 2025 | 6 | 30 | 20.0% |
| HMMT Feb 2025 | 2 | 30 | 6.7% |
| IF-Eval (strict prompt) | 401 | 541 | 74.1% |

### MATH-500 Breakdown

**Per Subject:**

| Subject | Correct/Total | Accuracy |
|---------|---------------|----------|
| Algebra | 112/124 | 90.3% |
| Number Theory | 52/62 | 83.9% |
| Counting & Probability | 29/38 | 76.3% |
| Prealgebra | 63/82 | 76.8% |
| Intermediate Algebra | 61/97 | 62.9% |
| Precalculus | 32/56 | 57.1% |
| Geometry | 21/41 | 51.2% |

**Per Difficulty Level:**

| Level | Correct/Total | Accuracy |
|-------|---------------|----------|
| 1 | 39/43 | 90.7% |
| 2 | 78/90 | 86.7% |
| 3 | 86/105 | 81.9% |
| 4 | 88/128 | 68.8% |
| 5 | 79/134 | 59.0% |

### IF-Eval Detail

| Metric | Value |
|--------|-------|
| Strict prompt accuracy | 74.1% |
| Strict instruction accuracy | 82.0% |
| Loose prompt accuracy | 73.4% |
| Loose instruction accuracy | 81.3% |

### Key Observations

1. **MATH-500 is the workhorse benchmark.** With 500 problems and 74% accuracy, there is ample headroom to detect both improvements (masking Stumbling Blocks) and degradation (masking Pillars). A single token mask that flips 5 problems produces a 1% detectable shift.

2. **AIME provides marginal signal.** At 7/30 and 6/30 correct, individual token knockouts can detect large effects (≥1 problem = 3.3%) but not subtle ones. Useful for headline claims, not per-token ranking.

3. **HMMT is too hard for per-token analysis.** Only 2/30 correct — a single problem flip is indistinguishable from noise. Reported for completeness; tokens should NOT be ranked by HMMT delta.

4. **IF-Eval is stable at 74%.** The strict/loose gap is small (0.7pp), suggesting the model generally satisfies constraints cleanly when it satisfies them at all.

5. **Clear difficulty gradients in MATH-500.** Algebra (90%) → Geometry (51%) and L1 (91%) → L5 (59%). These gradients may interact with Rock Token masking — e.g., masking reasoning connectives ("therefore", "however") might disproportionately affect harder problems where multi-step reasoning matters most.

### Statistical Power Assessment

For Step 2.1 (individual knockout with 200 tokens):

| Benchmark | Detectable effect (1 problem flip) | Usable for per-token ranking? |
|-----------|-----------------------------------|-------------------------------|
| MATH-500 | 0.2% per problem | Yes — primary ranking signal |
| AIME 24/25 | 3.3% per problem | Marginal — confirmatory only |
| HMMT | 3.3% per problem | No — too few correct |
| IF-Eval | 0.2% per prompt | Yes — non-math control |

---

*Baseline data: `results/masking/baseline/student_onpolicy/`*
*Additional baselines (teacher, student_base, student_offpolicy) running in parallel.*

---

## 13. Step 2.1 — Individual Token Knockout

**Method.** Each of the 200 Rock Tokens (count-ranked) was masked individually by setting `logit_bias = -100` during greedy decoding. The student was re-evaluated on MATH-500 (500 problems) and IF-Eval (541 prompts). Per-token deltas are measured against an unmasked baseline run computed in the same session (75.0% MATH-500, 75.2% IF-Eval — 1pp different from Step 0 due to a separate vLLM session, but still our reference).

### 13.1 Headline Taxonomy

| Category | Count | Fraction |
|----------|-------|----------|
| **Pillar** (Δ < 0 on MATH-500) | **164** | **82%** |
| **Neutral** (Δ = 0) | 11 | 6% |
| **Stumbling Block** (Δ > 0) | 25 | 12% |

**The dominant finding: Rock Tokens are predominantly Pillars.** Of the 200 high-loss, low-improvement tokens identified in Part 1, removing 82% of them *hurts* the student's math performance. Only 12% are genuine Stumbling Blocks where masking helps.

This is a non-trivial result. The recalcitrance criterion in Part 1 is purely loss-based — it identifies tokens where the OPD loss never converged. One could have predicted that these are tokens where the OPD signal was harmful (Stumbling Blocks). The data says the opposite: most are tokens where the student is doing something useful that the teacher disagrees with, and removing them collapses that capability.

### 13.2 MATH-500 Δ Distribution

```
  -3%  #### (4)
  -2%  ############################################## (46)
  -1%  ########################################################################################### (91)
   0%  ################################################# (49)
  +1%  ########## (10)
```

- **Mean Δ = -0.91%**, median = -0.80%, std = 0.83%
- **91 tokens (45%) cluster at -1%** — the modal effect is a single problem flip
- **The distribution is skewed left** with a long tail of large Pillars (-3.4% max) and a short tail of mild Stumbling Blocks (+1.0% max)

### 13.3 Cross-Task Independence

Pearson correlation between MATH-500 Δ and IF-Eval Δ: **r = -0.052** (essentially zero).

| Cross-task pattern | Count |
|--------------------|-------|
| Both hurt (M↓, I↓) | 130 |
| Both help (M↑, I↑) | 2 |
| Math hurts, IF-Eval helps | 20 |
| Math helps, IF-Eval hurts | 23 |

**Token roles are task-specific.** Knowing how a token behaves on MATH-500 tells you almost nothing about how it behaves on IF-Eval. This means a single Pillar/Stumbling Block taxonomy will be benchmark-dependent — there is no universal "Pillar token" that is essential for everything.

### 13.4 No Free Lunch on Stumbling Blocks

**23 of 25 Stumbling Blocks (92%) hurt IF-Eval** while helping MATH-500. Examples:
- " direct" : MATH-500 +0.20%, IF-Eval **-1.48%**
- " Removes": MATH-500 +0.20%, IF-Eval **-1.48%**
- " Algorithm": MATH-500 +0.40%, IF-Eval **-1.11%**

This means simple inference-time Stumbling Block masking is a tradeoff, not a free improvement. Any production deployment would need to weigh the math gain against the instruction-following loss. A more principled training-time intervention (Step 6) is needed to actually capture a clean win.

### 13.5 Top Pillars (removal hurts MATH-500 most)

| Rank | Token | Freq | MATH-500 Δ | IF-Eval Δ | Type |
|------|-------|------|-----------|-----------|------|
| 1 | " certain" | 92 | **-3.40%** | -0.37% | reasoning qualifier |
| 2 | " strategic" | 37 | -3.20% | -0.37% | reasoning qualifier |
| 3 | " Initialize" | 31 | -3.00% | -0.55% | code/setup |
| 4 | " skip" | 31 | -2.60% | -0.55% | algorithmic |
| 5 | " arms" | 35 | -2.40% | +0.00% | content |
| 6 | " programs" | 33 | -2.40% | -0.18% | content |
| 7 | " tech" | 37 | -2.40% | +0.00% | content |
| 8 | " Do" | 46 | -2.40% | -1.11% | imperative |
| 9 | " smart" | 46 | -2.40% | -0.92% | reasoning qualifier |
| 10 | " just" | 65 | -2.20% | -0.74% | reasoning hedge |

**Pattern.** The strongest Pillars are dominated by **reasoning vocabulary** — qualifiers ("certain", "smart", "strategic"), hedges ("just"), action verbs ("try", "compare", "break"), imperatives ("Do"). This directly supports the teammate's Pillar hypothesis: *decision-critical and reasoning structure tokens*. " certain" alone — masking it costs the student 17 problems on MATH-500 — looks like a token the model has learned to use as a reasoning anchor ("for a certain value of x", "certain conditions").

### 13.6 All 25 Stumbling Blocks

| Rank | Token | Freq | MATH-500 Δ | IF-Eval Δ |
|------|-------|------|-----------|-----------|
| 1 | " rounded" | 37 | +1.00% | -0.37% |
| 2 | " fun" | 30 | +1.00% | -0.18% |
| 3 | " project" | 36 | +0.80% | -0.18% |
| 4 | " handle" | 259 | +0.60% | -0.18% |
| 5 | " details" | 34 | +0.60% | -0.74% |
| 6 | " starting" | 50 | +0.60% | -0.55% |
| 7 | " integration" | 31 | +0.60% | -1.11% |
| 8 | " recursively" | 35 | +0.60% | -0.55% |
| 9 | " represent" | 68 | +0.60% | -0.92% |
| 10 | " doing" | 44 | +0.60% | -0.74% |
| 11–25 | " rock", " Algorithm", " approximate", " projects", " Important", " detailed", " wind", " co", " Some", " quart", " direct", " Removes", " financial", " hope", " best" | various | +0.20% to +0.40% | mostly negative |

**Pattern.** Stumbling Blocks lean toward **code/technical content vocabulary** — " Algorithm", " recursively", " integration", " Initialize" (also a Pillar — interesting), " project(s)", " handle". This makes sense given the OPD training distribution: the student was trained on math, so code-context tokens generate confidently-wrong predictions that hurt math reasoning. Removing them frees up probability mass that the student uses better.

### 13.7 Statistical Power Reality Check

With baseline = 75% on MATH-500 (375/500), a single problem flip = 0.2pp. So:
- A Pillar with Δ = -3.4% means **17 problems lost** — robustly significant, far above any noise floor
- A Stumbling Block with Δ = +1.0% means **5 problems gained** — small but consistent
- The 91 tokens at Δ = -1% mean **5 problems lost** — borderline; bootstrap testing (Step 2.2) will be necessary to separate signal from noise

Step 2.2 will run paired bootstrap on each token to assign p-values and Strong/Weak classifications.

### 13.8 IF-Eval Perspective: A Different Set of Pillars

The MATH-500 view above ranked tokens by their effect on math reasoning. Re-ranking the same 200 tokens by IF-Eval delta reveals a **largely disjoint** set of Pillars and Stumbling Blocks — confirming the cross-task independence finding above.

#### IF-Eval Taxonomy

| Category | Count | Fraction |
|----------|-------|----------|
| **Pillar** (Δ < 0 on IF-Eval) | **160** | **80%** |
| **Neutral** (Δ = 0) | 14 | 7% |
| **Stumbling Block** (Δ > 0) | 26 | 13% |

The fractions are remarkably similar to the MATH-500 view (82/6/12). But the *identities* of the Pillars and Stumbling Blocks shift substantially.

#### IF-Eval Δ Distribution

```
  -2%  #### (4)
  -1%  ########################################################################################################## (106)
   0%  ################################################################################# (81)
  +1%  ######### (9)
```

- Mean Δ = -0.52%, std = 0.53% — **smaller magnitudes than MATH-500** because IF-Eval is more constrained-output and less sensitive to single-token suppression
- Modal effect: -1% (≈6 prompts flipped on a 541-prompt benchmark)

#### Top 10 IF-Eval Pillars (removal hurts IF-Eval most)

| Rank | Token | Freq | IF-Eval Δ | MATH-500 Δ | Type |
|------|-------|------|-----------|------------|------|
| 1 | " -like" | 75 | **-2.22%** | -1.40% | morphological suffix |
| 2 | " trade" | 41 | -1.85% | -1.00% | content |
| 3 | " tools" | 53 | -1.66% | -1.60% | content |
| 4 | " demand" | 41 | -1.66% | -1.00% | content |
| 5 | " library" | 30 | -1.48% | +0.00% | content |
| 6 | " Consider" | 59 | -1.48% | -0.40% | reasoning marker |
| 7 | " Removes" | 57 | -1.48% | **+0.20%** | code/action |
| 8 | " direct" | 89 | -1.48% | **+0.20%** | content |
| 9 | " shape" | 108 | -1.29% | -1.60% | content |
| 10 | " fix" | 144 | -1.29% | -0.60% | content |

**Pattern.** IF-Eval Pillars are dominated by **content/topical vocabulary** — domain nouns ("library", "tools", "trade", "shape"), morphological tokens ("-like"), and discourse markers ("Consider"). This is a different signature from MATH-500 Pillars (reasoning qualifiers like "certain", "just", "smart"). IF-Eval prompts ask the model to *write about something* with constraints, so masking topic-relevant content directly damages output quality.

Two tokens — " Removes" and " direct" — are **simultaneously a Stumbling Block on MATH-500 (+0.20%) and a Pillar on IF-Eval (-1.48%)**. These are tokens where masking helps math reasoning but hurts instruction-following: the cleanest examples of cross-task tension.

#### All 26 IF-Eval Stumbling Blocks (removal helps IF-Eval)

| Token | Freq | IF-Eval Δ | MATH-500 Δ |
|-------|------|-----------|------------|
| " sustainable" | 42 | +0.92% | -1.80% |
| " transit" | 42 | +0.55% | -1.00% |
| " starting" | 400 | +0.55% | +0.00% |
| " causes" | 66 | +0.55% | +0.00% |
| " ensures" | 149 | +0.55% | -1.00% |
| " maps" | 37 | +0.55% | -1.40% |
| " advanced" | 45 | +0.55% | -2.00% |
| " clean" | 215 | +0.55% | -1.80% |
| " educational" | 33 | +0.55% | -0.60% |
| " features" | 76 | +0.37% | +0.00% |
| " hope" | 30 | +0.37% | +0.20% |
| " mix" | 40 | +0.37% | -1.00% |
| " state" | 92 | +0.37% | -1.20% |
| " Note" | 267 | +0.37% | -1.80% |
| " communication" | 48 | +0.37% | -1.00% |
| " compare" | 103 | +0.37% | -2.00% |
| 10 more | various | +0.18% | mostly negative |

**Pattern.** IF-Eval Stumbling Blocks lean toward **abstract/topical fillers** — sustainability/environmental vocabulary (" sustainable", " transit", " climate", " educational"), filler nouns (" features", " state", " Note"), and tokens that may bias the model toward overly verbose or off-topic responses.

**No free lunch on the other side either.** Of the 26 IF-Eval Stumbling Blocks, **20 (77%) hurt MATH-500**. The most striking example: " advanced" gives +0.55% on IF-Eval but **-2.00% on MATH-500**. Masking tokens to improve instruction following typically degrades math reasoning.

#### Cross-Task Pillar Overlap

Of the 160 IF-Eval Pillars, **130 (81%) are also MATH-500 Pillars**. Of the 26 IF-Eval Stumbling Blocks, **20 (77%) are MATH-500 Pillars**. Combined, **150 of 200 tokens (75%) are Pillars on at least one benchmark in agreement with the other** — but the *strongest* effects on each benchmark come from largely different tokens.

This refines the earlier finding: while the *rate* of Pillar/Stumbling/Neutral classification is similar across tasks, the *identity* of the strongest movers depends on the task domain. A token like " certain" is a strong reasoning Pillar but a weak IF-Eval Pillar; " -like" is the opposite.

### 13.9 Implications

1. **Most Rock Tokens are not noise.** The recalcitrant loss in Part 1 reflected genuine learning that the student couldn't perfectly converge to the teacher — but masking the tokens reveals the student *did* learn something useful.

2. **The Pillar hypothesis is largely confirmed.** Reasoning vocabulary dominates the top Pillars: qualifiers, hedges, imperatives, action verbs. The teammate's hypothesis (decision-critical + reasoning structure tokens) is supported by the qualitative pattern.

3. **The Stumbling Block hypothesis is partially confirmed.** Stumbling Blocks lean toward off-domain (code) vocabulary — consistent with "high-information" tokens the student couldn't reliably model. They are a small minority (12%).

4. **There is no universal "bad" Rock Token.** Cross-task independence (r ≈ 0) means token roles are benchmark-specific. A paper claim must be careful: "Stumbling Block on MATH-500" rather than "Stumbling Block, period."

5. **Inference-time masking has a cost.** 92% of Stumbling Blocks hurt IF-Eval. The path to a clean win runs through training-time loss masking (Step 6), not inference-time logit suppression.

---

*Knockout data: `results/masking/knockout/count/`*
*Per-token JSONs: `tokens/token_<id>.json` (includes `per_correct` boolean lists for bootstrap)*
*Summary: `summary.csv`, `summary.json`*

---

## 14. Step 2.2 — Statistical Categorization (Paired Bootstrap)

**Method.** For each of the 200 tokens, we ran a paired bootstrap over the per-problem boolean correctness vectors against the unmasked baseline (n=10,000 resamples per token, two-sided p-values, 95% percentile CI). Tokens were classified into 5 categories:

- **Strong Pillar:** Δ ≤ -ε AND p < α
- **Weak Pillar:** Δ < 0 AND p < α AND |Δ| < ε
- **Neutral:** p ≥ α
- **Weak Stumbling Block:** Δ > 0 AND p < α AND |Δ| < ε
- **Strong Stumbling Block:** Δ ≥ ε AND p < α

Thresholds: ε = 1%, α = 0.05.

### 14.1 Headline: Only 5% of Rock Tokens Have Significant Effects

| Category | MATH-500 | IF-Eval |
|----------|----------|---------|
| Strong Pillar | **7** | **3** |
| Weak Pillar | 0 | 0 |
| Neutral | **193** | **197** |
| Weak Stumbling Block | 0 | 0 |
| Strong Stumbling Block | 0 | 0 |

**The bootstrap reframes the entire taxonomy.** Of the 200 Rock Tokens, only 7 have statistically significant effects on MATH-500 and only 3 on IF-Eval. **All other tokens are Neutral** — their raw deltas (±0.2% to ±2%) fall within bootstrap noise and cannot be distinguished from chance.

This is consistent with the implementation plan's prediction: *"The Neutral category is probably the largest, and identifying it is itself a finding."* The data confirms it strongly: **97.5% of Rock Tokens are Neutral on MATH-500.**

**Zero Stumbling Blocks survive significance testing** on either benchmark. Every token in the raw +0.2% to +1.0% Stumbling Block tail had p > 0.05. This means **inference-time Stumbling Block masking has no statistically defensible benefit** — the apparent improvements were sampling variance.

### 14.2 The 7 MATH-500 Strong Pillars

| Token | Freq | Rock | Δ | 95% CI | p | IF-Eval Δ |
|-------|------|------|-----|--------|---|-----------|
| " certain" | 92 | 33 | **-3.40%** | [-5.80, -1.00] | 0.0038 | -0.37% |
| " strategic" | 37 | 15 | -3.20% | [-5.60, -0.80] | 0.0074 | -0.37% |
| " Initialize" | 31 | 16 | -3.00% | [-5.40, -0.80] | 0.0154 | -0.55% |
| " arms" | 35 | 11 | -2.40% | [-4.60, -0.20] | 0.0410 | +0.00% |
| " tech" | 37 | 12 | -2.40% | [-4.60, -0.20] | 0.0478 | +0.00% |
| " Do" | 46 | 15 | -2.40% | [-4.60, -0.20] | 0.0358 | -1.11% |
| " programs" | 33 | 14 | -2.40% | [-4.60, -0.20] | 0.0466 | -0.18% |

**Pattern.** The strongest survivors include reasoning markers (" certain", " strategic", " Do") and code/setup vocabulary (" Initialize"). Three content nouns (" arms", " tech", " programs") also reach significance — these are unexpected and worth qualitative investigation in the generated outputs.

**" certain" remains the dominant Pillar:** masking it costs the student 17 problems (3.4 percentage points), with 95% confidence the true effect is at least 1.0 percentage points (5+ problems). This is a robust causal finding.

### 14.3 The 3 IF-Eval Strong Pillars

| Token | Freq | Δ | 95% CI | p | MATH-500 Δ |
|-------|------|-----|--------|---|-----------|
| " -like" | 75 | **-2.22%** | [-3.70, -0.92] | 0.0016 | -1.40% |
| " trade" | 41 | -1.85% | [-3.51, -0.18] | 0.0284 | -1.00% |
| " tools" | 53 | -1.66% | [-3.14, -0.37] | 0.0226 | -1.60% |

These are content/morphological tokens (the suffix "-like", domain nouns "trade" and "tools"). All three also have negative MATH-500 Δs but none reach significance there.

### 14.4 Cross-Task Disjointness Confirmed

| MATH-500 \ IF-Eval | Strong Pillar | Neutral | Total |
|--------------------|--------------:|--------:|------:|
| Strong Pillar | **0** | 7 | 7 |
| Neutral | 3 | 190 | 193 |
| **Total** | 3 | 197 | 200 |

**Zero tokens are Strong Pillars on both benchmarks.** The 7 MATH-500 Pillars and 3 IF-Eval Pillars are completely disjoint sets. This is the cleanest possible evidence that **token roles are task-specific** — there is no universal Pillar in our data.

### 14.5 Borderline MATH-500 Pillars (0.05 ≤ p < 0.10)

15 additional tokens fall just outside the significance threshold. They form a coherent "reasoning vocabulary" cluster that would likely emerge as Strong Pillars with a larger benchmark:

" skip", " self", " try", " smart", " Ensure", " led", " break", " compare", " just", " spoon", " advanced", " economic", " dis", " body", " DP"

Most are reasoning hedges and action verbs (" try", " just", " compare", " break", " self"). The teammate's *"reasoning structure tokens"* hypothesis is qualitatively supported here even though the bootstrap is too conservative for a 500-problem benchmark to crown them as Strong.

### 14.6 No Feature Predicts Category

Pearson correlations between each Part 1 feature and the knockout Δ:

| Feature | r (MATH-500) | r (IF-Eval) |
|---------|-------------:|------------:|
| frequency | +0.015 | +0.030 |
| rock_count | +0.013 | +0.023 |
| rock_rate | -0.022 | -0.047 |
| avg_loss_before | -0.017 | -0.015 |
| avg_loss_after | +0.010 | -0.048 |
| avg_improvement | -0.062 | +0.071 |
| avg_teacher_entropy | +0.063 | -0.036 |
| avg_student_entropy_before | +0.059 | -0.069 |
| avg_student_entropy_after | +0.056 | -0.037 |

**All |r| < 0.075. No feature predicts knockout effect.** This is a striking negative result for the teammate's Stumbling Block hypothesis (high entropy → Stumbling Block) and the Pillar hypothesis (low entropy → Pillar). Within the 200 Rock Tokens identified by Part 1, neither student/teacher entropy nor any Part 1 loss feature explains why a particular token is a Strong Pillar.

The likely interpretation: **selection effect.** All 200 tokens are already in the high-loss / low-improvement region (by definition of Rock Token). Within this restricted population, variation in entropy or loss is too narrow to predict downstream knockout effects. A larger study sampling tokens across the full loss spectrum would be needed to test the entropy hypothesis properly.

### 14.7 Revised Headline Findings for the Paper

The bootstrap analysis sharpens the Step 2.1 story considerably. Three claims are now defensible:

1. **The Neutrality Result.** Of the 200 high-loss, low-improvement tokens identified by the OPD-based recalcitrance criterion, only 5% (10 of 200, summing across tasks) have statistically significant inference-time effects. The bulk of "rock tokens" are causally inert in greedy generation — their high training loss does not translate to functional importance at inference time on these benchmarks.

2. **The Strong Pillar Reality.** A small number of tokens (7 on MATH-500, 3 on IF-Eval, disjoint sets) show robust, large effects. Masking " certain" alone costs 3.4% absolute MATH-500 accuracy with p < 0.005. These are the genuine causal Pillars worth analyzing for the paper.

3. **The Stumbling Block Negative.** Inference-time Stumbling Block masking does not produce statistically significant improvements on either benchmark for any of our 200 candidates. The apparent benefits in raw deltas were noise. **A clean Stumbling Block win requires training-time intervention** (Step 6 of the implementation plan).

### 14.8 Implications for Next Steps

- **Step 3 (cumulative removal curves)** is now even more important. The single-token effects are mostly null, but cumulative removal of 50 or 100 "near-Pillar" tokens may reveal aggregate effects that single masks miss.

- **Step 4 (semantic group masking)** becomes the natural way to pool weak signals. The borderline reasoning vocabulary cluster (Section 14.5) can be tested as a group to see whether the qualitative pattern survives statistical aggregation.

- **Step 6 (training-time loss masking)** is the only path to a defensible Stumbling Block claim. Inference-time results conclusively rule out Stumbling Block existence at the per-token level.

- **The paper structure shifts**: the headline is no longer "we found 25 Stumbling Blocks" but "of 200 recalcitrant tokens, only 10 are causally important — and they cluster around reasoning vocabulary." The paper becomes about Pillar identification and the Neutrality of the recalcitrance criterion at inference time.

---

*Categorization data: `results/masking/categorization/count_nondeterministic/`*
*Per-token table: `categorization.csv` (with bootstrap CIs and p-values)*
*Plots: `plots/delta_histograms.png`, `cross_task_scatter.png`, `feature_correlations.png`*

> ⚠️ **The Step 2.1 / 2.2 results above were generated with vLLM in default (non-deterministic) configuration. We subsequently discovered that vLLM produces session-level scheduling variance on A100 hardware that flooded out per-token signal. The deterministic re-run is documented in Section 15. The numbers above should be read as a session-specific snapshot, not as reproducible findings.**

---

## 15. Methodological Discovery — vLLM Cross-Session Nondeterminism on A100

While trying to reconcile Step 2.1 single-token deltas with Step 2.3 cumulative deltas (both nominally measuring the same masks at k=1), we discovered that the same masked configuration in two different vLLM sessions produced different per-problem outputs:

| Run | MATH-500 baseline | Per-problem differences vs other run |
|-----|-------------------|---------------------------------------|
| Knockout session | 75.0% (375/500) | — |
| Cumulative session | 73.8% (369/500) | **36 problems differ** |

Same model, same prompts, same seed=42, same temperature=0, same hardware. Two separate Python processes loaded vLLM and ran greedy decoding. **36 problems (7.2%) gave different correctness across the two sessions.**

This dwarfs the 1–3% per-token effects we were trying to measure. Worse, the bootstrap CIs in Step 2.2 only captured *within-session* sampling variance — they entirely missed this cross-session noise source. Reported "Strong Pillars" with p-values of 0.004 may not reproduce in another session.

### Root cause

Documented in vLLM's [Reproducibility guide](https://docs.vllm.ai/en/latest/usage/reproducibility/): vLLM's V1 engine uses multiprocessing for batch scheduling, which produces non-deterministic chunking and KV-cache layout decisions across separate processes. Even with greedy decoding and a fixed seed, this changes the floating-point order of operations and can flip the argmax on individual problems.

The official solution is `VLLM_BATCH_INVARIANT=1`, but it requires NVIDIA GPUs with compute capability ≥ 9.0 (H100/H200/B100/B200 only). On A100 (compute capability 8.0), the available workaround is `VLLM_ENABLE_V1_MULTIPROCESSING=0`, which forces single-process scheduling.

### The fix

Three changes to `src/masking/common.py`:

1. **`os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")`** at module import (forces single-process scheduling).
2. **`top_k=1`** added to all SamplingParams (forces strict argmax tie-breaking).
3. The same change in `knockout.py` and `cumulative.py` for masked SamplingParams.

### Verification

`src/masking/verify_masking.py` runs 7 diagnostic tests, including a cross-session reproducibility check (Test 7) that saves baseline output to disk and compares against subsequent runs.

After the fix, all 7 tests pass — including byte-identical baseline outputs across two separate Python processes.

### Implications for the paper

This is a methodological contribution worth its own subsection in the related-work / methodology sections of the paper:

> *"Inference-time masking studies of LLMs must use deterministic vLLM configuration. Without `VLLM_ENABLE_V1_MULTIPROCESSING=0` on non-H100 hardware, single-token effects are dominated by inter-session batch-scheduling noise. We show that ~50% of per-token Pillar/Stumbling Block classifications flip sign between sessions, and the bootstrap p-values within a session systematically underestimate true variance."*

---

## 16. Deterministic Re-run — The Real Picture

We re-ran the full Part 2 chain (Step 2.1 → 2.2 → 2.3) with the deterministic configuration. All numbers below are byte-reproducible across separate vLLM sessions.

### 16.1 Step 2.1 — Knockout (Deterministic)

**Baseline:** MATH-500 74.2% (371/500), IF-Eval 74.7% (404/541)

| Metric | OLD (non-deterministic) | **NEW (deterministic)** |
|--------|:-:|:-:|
| Pillar (Δ<0) | 164 (82%) | **93 (46%)** |
| Neutral (Δ=0) | 11 (6%) | 14 (7%) |
| Stumbling (Δ>0) | 25 (12%) | **93 (46%)** |
| Mean Δ | -0.91% | -0.25% |
| Median Δ | -0.80% | **+0.00%** |
| Min Δ | -3.40% | **-2.00%** |
| Max Δ | +1.00% | +1.20% |
| \|Δ\| ≥ 1% Pillars | many | **34** |
| \|Δ\| ≥ 1% Stumbling | 0 | **2** |
| \|Δ\| ≥ 2% (any) | 4 Pillars | **1 Pillar (" maps")** |

**The taxonomy collapses to balance.** With session noise removed, Rock Tokens are nearly equally likely to help or hurt MATH-500 when masked. Most are Neutral or near-Neutral.

**Per-token sign-flip rate vs OLD: 91 of 177 non-zero token pairs (51%) flipped sign.** The old per-token rankings were near-random with respect to the deterministic ground truth.

### 16.2 Step 2.2 — Bootstrap Categorization (Deterministic)

The headline result of the entire study:

| Category | MATH-500 | IF-Eval |
|----------|:-:|:-:|
| Strong Pillar (Δ ≤ -ε, p<α) | **0** | 0 |
| Weak Pillar | 0 | 0 |
| **Neutral (p ≥ 0.05)** | **200** | 199 |
| Weak Stumbling Block | 0 | 0 |
| Strong Stumbling Block | **0** | **1** |

**Zero tokens are statistically significant Pillars on MATH-500 at α=0.05, ε=1%.** With n=500 problems and proper determinism, the bootstrap correctly identifies that single-token effects are below the detection threshold.

The 7 "Strong Pillars" from the OLD bootstrap (" certain", " strategic", " Initialize", etc.) were artifacts of session-noise inflation. Their session-specific Δ of -3.4% would not replicate in any other vLLM session.

The single Strong Stumbling Block on IF-Eval is **" shape"** (Δ = +1.29% on IF-Eval, also +1.20% on MATH-500).

### 16.3 Top Pillars and Stumbling Blocks (Raw Δ, not bootstrap-significant)

Although none reach bootstrap significance with n=500, the strongest raw effects are:

**Top Pillars (Δ ≤ -1.5%, |Δ| ≥ 1.5% — 21 tokens):**

| Token | Freq | M500 Δ | IF-Eval Δ |
|-------|------|--------|-----------|
| " maps" | 37 | **-2.00%** | +0.37% |
| " -like" | 75 | -1.80% | +0.18% |
| " job" | 33 | -1.80% | +0.00% |
| " advanced" | 45 | -1.60% | +0.18% |
| " Python" | 1138 | -1.60% | -0.55% |
| " educational" | 33 | -1.60% | -0.18% |
| " transportation" | 36 | -1.60% | +0.18% |
| " local" | 43 | -1.60% | +0.55% |
| " Regular" | 37 | -1.60% | -0.18% |
| " especially" | 40 | -1.60% | +0.18% |
| " mathematics" | 37 | -1.60% | +1.11% |
| " clean" | 215 | -1.60% | +0.37% |
| " Important" | 76 | -1.60% | +0.00% |
| " issues" | 77 | -1.60% | +0.37% |
| " financial" | 45 | -1.60% | +0.00% |
| " became" | 34 | -1.60% | +1.29% |
| " physical" | 41 | -1.60% | +0.92% |
| " movement" | 41 | -1.60% | +0.37% |
| " understanding" | 66 | -1.60% | +0.00% |
| " break" | 35 | -1.60% | +0.37% |
| " wind" | 49 | -1.60% | -0.55% |

**Pattern.** Mixed: code identifiers (" Python", " Regular"), domain content (" mathematics", " transportation", " educational", " financial", " local"), discourse markers (" Important", " especially"), abstract nouns (" understanding", " issues", " movement", " situation"), morphological (" -like"). **No clean reasoning-vocabulary cluster** — that pattern in the old data was an artifact.

**Top Stumbling Blocks (only 2 reach |Δ| ≥ 1%):**

| Token | Freq | M500 Δ | IF-Eval Δ |
|-------|------|--------|-----------|
| **" shape"** | 108 | **+1.20%** | **+1.29%** |
| **" mix"** | 40 | **+1.00%** | +0.37% |

### 16.4 Cross-Task Robust Effects (the cleanest reproducible signals)

Tokens with |Δ| ≥ 0.5% on **both** benchmarks, **same direction**:

**Both Pillar (8 tokens — masking hurts both):**

| Token | M500 Δ | IF-Eval Δ |
|-------|--------|-----------|
| " Python" | -1.60% | -0.55% |
| " wind" | -1.60% | -0.55% |
| " approximate" | -1.20% | -0.55% |
| " connecting" | -1.00% | -0.74% |
| " of" | -0.80% | -0.55% |
| " detailed" | -0.80% | -0.74% |
| " little" | -0.80% | -0.55% |
| " direct" | -0.60% | -0.55% |

**Both Stumbling Block (2 tokens — masking helps both):**

| Token | M500 Δ | IF-Eval Δ |
|-------|--------|-----------|
| **" shape"** | **+1.20%** | **+1.29%** |
| **" hope"** | +0.60% | +0.92% |

**Cross-task disagreement: 66 tokens** flip sign between MATH-500 and IF-Eval — task-specific roles dominate.

**Cross-task correlation r = -0.082** — slightly negative, very weak. Tokens that help one benchmark tend to slightly hurt the other.

### 16.5 Step 2.3 — Cumulative Curves (Deterministic)

| k | Greedy Pillar | Greedy Stumbling | Random (mean ± std) |
|---|:-:|:-:|:-:|
| 1 | -2.0% / +0.6% | +0.6% / -0.2% | +0.2±0.4% / -0.2±0.3% |
| 5 | -1.8% / -0.4% | +2.0% / +0.0% | -0.2±0.3% / -0.1±0.6% |
| 20 | +0.8% / +0.2% | -0.8% / -0.4% | +0.3±0.9% / +0.1±0.5% |
| 50 | -0.6% / -0.9% | -0.8% / -0.2% | -0.4±0.9% / -0.4±0.5% |
| 100 | -0.6% / -0.4% | +1.0% / -1.8% | +0.3±0.6% / -0.7±0.4% |
| 200 | -0.4% / -1.1% | +0.4% / -1.1% | -0.1±1.0% / -1.1±0.3% |

(M500 / IF-Eval at each cumulative k)

**No curve crashes.** Greedy Pillar removal does NOT show a monotonic decline. Greedy Stumbling Block removal does NOT show monotonic improvement. Random oscillates around 0 with std ~0.5–1%. **All three curves overlap within their noise bands.**

The single-token rankings have **zero aggregate predictive power** at this benchmark size. Cumulative interventions on Rock Tokens — whether selected for "Pillar-like" or "Stumbling-Block-like" effects, or randomly — perturb MATH-500 accuracy by roughly the same amount.

### 16.6 Feature Correlations (Deterministic)

| Feature | r (MATH-500) | r (IF-Eval) |
|---------|-------------:|------------:|
| frequency | -0.173 | +0.002 |
| rock_count | -0.165 | -0.008 |
| rock_rate | +0.164 | -0.099 |
| avg_loss_before | +0.008 | -0.029 |
| avg_loss_after | -0.022 | -0.064 |
| avg_improvement | +0.066 | +0.074 |
| avg_teacher_entropy | +0.045 | -0.045 |
| avg_student_entropy_before | +0.046 | -0.133 |
| avg_student_entropy_after | +0.066 | -0.118 |

Maximum |r| ≈ 0.17 (frequency on MATH-500). **No feature meaningfully predicts deterministic per-token effect** — neither entropy, nor loss, nor frequency. The slight negative frequency correlation on MATH-500 says higher-frequency tokens are mildly more Pillar-like (Δ more negative), but the effect is small.

The teammate's feature-based hypotheses (high-entropy → Stumbling, low-entropy → Pillar) are not supported in the deterministic data.

---

## 17. Final Headline Findings

After Steps 2.1, 2.2, 2.3 with deterministic vLLM configuration:

1. **The Methodological Finding (NEW):** vLLM cross-session nondeterminism on A100 hardware produces ~7% per-problem variance even at greedy temperature=0 with fixed seed. This source of noise dwarfs single-token masking effects and was missed by within-session bootstrap p-values in prior work. Inference-time masking studies on non-H100 hardware require `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

2. **The Neutrality Result:** Of the 200 OPD-recalcitrant tokens, **none have a statistically significant inference-time effect on MATH-500** (n=500 is too small to detect the largest -2.0% Δ). This is a strong negative result — the recalcitrance criterion in Part 1 does *not* identify functionally important tokens at inference time.

3. **The " shape" Anecdote:** The single statistically significant single-token effect across the entire study is **" shape"**, a Strong Stumbling Block on IF-Eval (+1.29%) that also helps MATH-500 (+1.20%) — a clean cross-task win. This is the only token where inference-time masking produces a defensible, cross-task improvement.

4. **The Cumulative Null Result:** Aggregating up to all 200 Rock Tokens, neither greedy-Pillar removal nor greedy-Stumbling-Block removal produces effects distinguishable from random masking. Single-token rankings have no aggregate predictive power.

5. **The Cross-Task Independence Confirmed:** Pearson r = -0.08 between MATH-500 and IF-Eval Δs. 66 of 200 tokens have signs that disagree across tasks. There is no universal Pillar or Stumbling Block — token roles are task-specific.

6. **The Pivot:** Inference-time single-token masking is not the right intervention to extract signal at this benchmark scale. The path to defensible improvement claims runs through:
   - **Larger benchmarks** (e.g., MATH-7500) for adequate statistical power on small effects
   - **Group masking** by semantic category (Step 4) to pool weak signals
   - **Training-time loss masking** (Step 6) — the decisive experiment that bypasses inference-time noise entirely

The paper's contribution shifts: less "we identify Pillar/Stumbling Block tokens" and more **"we show recalcitrant-loss tokens are causally inert at inference time, and the path to using this signal runs through training-time intervention."** This is a stronger, more honest, more publishable story.

---

*Deterministic data: `results/masking/{knockout,categorization,cumulative}/count/`*
*Old non-deterministic data preserved at: `results/masking/{knockout,categorization,cumulative}/count_nondeterministic/`*
*Verification: `src/masking/verify_masking.py` (Test 7 confirms cross-session byte-identical output)*

---

## 18. Step 4 — Semantic Group Masking (Deterministic)

The implementation plan flags Step 4 as the natural way to "pool weak per-token signals". After Sections 16-17 established that single-token effects are below the bootstrap detection threshold at n=500, group masking aggregates 4-20 tokens at a time to test whether semantically coherent clusters carry signal.

**Setup.** 14 groups + baseline. Each group masked simultaneously (all token logits → -100), evaluated on MATH-500 + IF-Eval. Same deterministic configuration as Sections 15-16. Two 10-token random controls bracket the noise floor.

### 18.1 Group Effects (sorted by MATH-500 Δ)

| Group | N | MATH-500 Δ | IF-Eval Δ |
|-------|:-:|:-:|:-:|
| `top5_pillar` | 5 | **-2.00%** | +0.55% |
| `top10_stumbling` | 10 | -1.00% | -0.92% |
| **`semantic_domain`** | 7 | **-1.00%** | +0.00% |
| `top5_stumbling` | 5 | -0.80% | +0.00% |
| `semantic_code_tech` | 5 | -0.20% | +0.37% |
| `semantic_abstract` | 4 | -0.20% | -0.55% |
| `semantic_modifiers` | 6 | -0.20% | +0.37% |
| `random_control_a` | 10 | -0.20% | -0.92% |
| `random_control_b` | 10 | +0.00% | +0.55% |
| `top10_pillar` | 10 | +0.40% | -0.92% |
| `top20_pillar` | 20 | +0.40% | -0.37% |
| `cross_task_stumbling` | 2 | +0.40% | +0.18% |
| **`cross_task_pillars`** | 8 | **+1.20%** | -0.37% |
| **`semantic_discourse`** | 4 | **+1.20%** | -0.37% |

Random controls land at -0.20% and +0.00% — defining the **±0.2% noise floor** at this group size.

### 18.2 Five Findings

**(a) `semantic_discourse` is a real Stumbling Block group (+1.20%).**
Masking just four discourse markers — { Important, especially, just, allows} — improves MATH-500 by 1.2 percentage points. The cluster has a coherent semantic interpretation: filler discourse words that the math-trained student uses unproductively. Mask them and the student is forced into more direct argumentation. **This is the cleanest group-level Stumbling Block effect in the entire study.**

**(b) `top5_pillar` compounds at small scale (-2.00%).**
The five strongest individual Pillars masked together cost 2 percentage points (10 problems lost net). This is 10× the random-control magnitude — individual Pillar effects DO compound when only the strongest few are removed. The contributing tokens are the highest-confidence Pillars: " maps", " -like", " job", " advanced", " Python".

**(c) `semantic_domain` is a real Pillar group (-1.00%).**
Math/domain content nouns — { mathematics, transportation, educational, financial, local, physical, state} — together hurt MATH-500 by 1 percentage point when masked. This validates a content-vocabulary Pillar story: when the student reasons about math, it routes through domain anchors, and removing them collapses that capability.

**(d) Non-additivity is a real phenomenon, not an artifact.**

A pure additive model would predict that masking more Pillars yields more negative Δ. The data shows the opposite at moderate-N:

- `top5_pillar`: -2.00% (5 individual Pillars)
- `top10_pillar`: +0.40% (10 individual Pillars — *direction reverses*)
- `top20_pillar`: +0.40% (20 individual Pillars — same)
- `cross_task_pillars`: **+1.20%** (8 cross-task Pillars — strongly *helping*)

The 8 cross_task_pillars (" of", " detailed", " Python", " connecting", " direct", " little", " approximate", " wind") were selected for their cross-task Pillar status (Δ ≤ -0.5% on BOTH individual benchmarks). Masking them all simultaneously *helps* MATH-500 by 1.2 percentage points.

This is direct evidence of **interaction effects**: the model has many parallel substitute pathways, and once a critical mass of "Pillar tokens" is removed simultaneously, the substitutes that emerge happen to be better than the originals were. Single-token effects do NOT predict multi-token effects in a simple additive way.

This finding also explains why the Step 2.3 cumulative curves were flat — the cumulative removal procedure assumes that ranking by individual effect predicts aggregate effect, and this assumption is violated.

**(e) IF-Eval is largely insensitive at the group level.**
IF-Eval Δ ranges from -0.92% to +0.55% across all 14 groups. Most are within the noise band (~±0.5%). Group-level math effects do not transfer to IF-Eval. This continues the cross-task independence story.

### 18.3 The "Stumbling Block at Group Level" Finding

Across all of Part 2, two robust Stumbling Block findings emerge:

| Finding | Type | M500 Δ | Notes |
|---------|------|--------|-------|
| `semantic_discourse` | Group (4 tokens) | **+1.20%** | Discourse fillers — coherent cluster, well above random |
| `cross_task_pillars` | Group (8 tokens) | **+1.20%** | Interaction effect — non-additive |
| " shape" | Single token | +1.20% | Cross-task: also helps IF-Eval +1.29% |

These three results converge on a real Stumbling Block phenomenon at group/token scale: **specific semantic/structural masking on the order of 1-1.5% absolute MATH-500 accuracy is achievable**, but it requires either (i) identifying a coherent semantic cluster, (ii) exploiting non-additive interaction effects, or (iii) finding the rare strong individual token (" shape").

### 18.4 The Two Pillar Findings

| Finding | Type | M500 Δ | Notes |
|---------|------|--------|-------|
| `top5_pillar` | Group (5 strongest individual Pillars) | **-2.00%** | Compound effect, no surprise |
| `semantic_domain` | Group (7 domain nouns) | **-1.00%** | Coherent semantic Pillar cluster |

### 18.5 Implications

1. **The right unit of analysis is the group, not the individual token.** Single-token bootstrap p-values were noise; group masks reveal robust structure.
2. **Semantic clusters carry meaning.** Discourse-marker Stumbling Blocks and domain-noun Pillars are interpretable, hypothesis-driven categories that deliver measurable effects.
3. **Interaction effects are pervasive.** Token effects are not additive. This nuances any "remove the top-K Stumbling Blocks" claim — that procedure does NOT in general improve performance, as our cumulative curves showed and as the `top10_pillar` reversal confirms.
4. **The path to a clean improvement claim runs through (a) group masking on a larger benchmark for statistical significance, then (b) training-time loss masking for a definitive intervention.**

The next step is MATH-5000 validation of the four most promising group findings (`semantic_discourse`, `top5_pillar`, `cross_task_pillars`, `semantic_domain`) — see Section 19 (forthcoming).

---

*Group data: `results/masking/groups/count/`*
*Per-group JSON: `groups/{group_name}.json` (per-problem correctness, masked token list)*
*Summary: `summary.csv`, `summary.json`*
*Plot: `plots/groups.png`*

---

## 19. Step A — MATH-5000 Validation (The Decisive Null Result)

The Section 18 group findings (`semantic_discourse` +1.2%, `top5_pillar` -2.0%, `cross_task_pillars` +1.2%, `semantic_domain` -1.0%) all came from a 500-problem benchmark with high single-run variance. Section 17 already flagged that "the path to defensible improvement claims runs through larger benchmarks". We re-evaluated all 14 groups on the **full Hendrycks MATH test set (5000 problems)** — 10× the statistical power.

**Setup.**
- Benchmark: full Hendrycks MATH test (5000 problems, 7 subjects, answers extracted from `\boxed{}` in solutions; 100% extraction success)
- IF-Eval: skipped (math-only focus per project pivot)
- Determinism: same configuration as all prior deterministic runs (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `top_k=1`)
- Bootstrap: 10,000 paired resamples per group, two-sided p-values
- Total runtime: ~5 hours on 2×A100

### 19.1 Headline: All Group Effects Vanish at Scale

**0 of 14 groups significant after multiple-testing correction.**

Sorted by MATH-5000 Δ:

| Group | N | M500 Δ | **M5000 Δ** | 95% CI | p-value | Sig (α=0.05)? | Sig (Bonf α=0.0036)? |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| cross_task_stumbling | 2 | +0.40% | -0.04% | [-0.70, +0.62] | 0.92 | — | — |
| semantic_discourse | 4 | **+1.20%** | -0.02% | [-0.70, +0.64] | 0.96 | — | — |
| random_control_a | 10 | -0.20% | +0.16% | [-0.52, +0.84] | 0.67 | — | — |
| semantic_modifiers | 6 | -0.20% | +0.22% | [-0.44, +0.86] | 0.53 | — | — |
| cross_task_pillars | 8 | **+1.20%** | +0.34% | [-0.30, +1.00] | 0.32 | — | — |
| semantic_abstract | 4 | -0.20% | +0.34% | [-0.34, +1.00] | 0.34 | — | — |
| top10_pillar | 10 | +0.40% | +0.34% | [-0.30, +0.98] | 0.31 | — | — |
| top20_pillar | 20 | +0.40% | +0.34% | [-0.30, +1.00] | 0.32 | — | — |
| top5_stumbling | 5 | -0.80% | +0.36% | [-0.32, +1.02] | 0.32 | — | — |
| semantic_code_tech | 5 | -0.20% | +0.38% | [-0.30, +1.04] | 0.29 | — | — |
| top10_stumbling | 10 | -1.00% | +0.38% | [-0.26, +1.00] | 0.26 | — | — |
| top5_pillar | 5 | **-2.00%** | +0.42% | [-0.26, +1.08] | 0.24 | — | — |
| semantic_domain | 7 | **-1.00%** | +0.64% | [+0.00, +1.28] | 0.050 | borderline (sign flipped) | — |
| **random_control_b** | 10 | +0.00% | **+0.86%** | [+0.22, +1.50] | **0.010** | **✓** | — |

### 19.2 Five Striking Observations

**(a) The only uncorrected-significant group is a random 10-token control.**
`random_control_b` (p=0.0096) reaches α=0.05 but does not survive Bonferroni correction (α=0.05/14 ≈ 0.0036). With 14 tests, expected number of false positives under the null is 14 × 0.05 = 0.7. **One out of fourteen at p<0.05 is exactly what we expect from chance — and it happens to be a random control.** This is the smoking gun.

**(b) 10 of 14 groups changed sign from MATH-500 to MATH-5000.**
| Group | M500 → M5000 |
|-------|-------------|
| top5_pillar | -2.00% → +0.42% |
| top5_stumbling | -0.80% → +0.36% |
| top10_stumbling | -1.00% → +0.38% |
| semantic_domain | -1.00% → +0.64% |
| semantic_code_tech | -0.20% → +0.38% |
| semantic_abstract | -0.20% → +0.34% |
| semantic_modifiers | -0.20% → +0.22% |
| random_control_a | -0.20% → +0.16% |
| cross_task_stumbling | +0.40% → -0.04% |
| semantic_discourse | +1.20% → -0.02% |

**The MATH-500 directional patterns were dominated by which 500 problems happened to be in the curated subset** — not by token-level causal effects.

**(c) All 14 groups cluster in a narrow band around 0.**
- Range: [-0.04%, +0.86%]
- Mean: +0.34%, std: 0.22%

A 0.86 percentage point range across selections that include the strongest "Pillars", strongest "Stumbling Blocks", and pure random controls. **There is no group structure visible in the data.**

**(d) Random controls are statistically indistinguishable from non-random "candidate" groups.**
| | Mean Δ | Range |
|--|--------|-------|
| Random controls (n=2) | +0.51% | [+0.16, +0.86] |
| Non-random groups (n=12) | +0.31% | [-0.04, +0.64] |

Random controls land **inside** the range of non-random groups. The two distributions overlap completely. No statistical test can separate them.

**(e) The semantic_discourse "Stumbling Block" findings completely vanishes.**
The single most exciting MATH-500 finding — masking 4 discourse markers (Important, especially, just, allows) for +1.20% — collapses to **-0.02% (p=0.96)** at MATH-5000. The interpretation that "the math-trained student uses these unproductively" is not supported when properly tested.

### 19.3 The Final Verdict on Inference-Time Masking

After 5 deterministic re-runs (Sections 16-19) covering 200 individual tokens × 5000 problems × 14 groups × 1.5+ days of GPU compute:

> **No group of OPD-recalcitrant tokens has a statistically significant effect on MATH (n=5000) under inference-time logit-bias masking. Single-token effects, group effects, semantic-cluster effects, and interaction effects all fail to reach significance after multiple-testing correction. Random and non-random groups are indistinguishable. The null cannot be more cleanly rejected.**

This is a strong, publishable negative result. The methodological story is complete:

1. **Part 1 successfully identifies 200 OPD-recalcitrant tokens** by their training-loss profile.
2. **These tokens are causally inert at inference time on MATH (n=5000).** Masking them — individually, in groups, by semantic category, or in cumulative aggregates — does not produce reproducible improvements or degradations.
3. **The recalcitrance criterion is well-defined but not predictive of inference-time importance.** High training loss does not entail functional importance under masking.

### 19.4 Why MATH-500 Showed False Signal

MATH-500 was published as a curated 500-problem subset of the 5000-problem MATH test, designed to be a "representative but smaller" benchmark. We discovered it has substantial **subsample directional bias** for masking studies:
- Standard error of Δ on MATH-500 ≈ 1.2% (we measured this directly)
- The 500 specific problems happen to have systematic directional response to perturbation
- Masking any 5-20 tokens shifts roughly 30-40 of those 500 problems
- The directional balance of those flips depends on which 500 problems are sampled
- Effects of ±1-2% appear as "significant" within MATH-500 but vanish on the full 5000

**Methodological recommendation for the paper:** *"Inference-time masking studies must use the full Hendrycks MATH test set (or equivalent ≥5000-problem benchmark). Smaller subsets (n=500) produce subsample directional bias indistinguishable from token-level causal effects, leading to false-positive Pillar/Stumbling Block claims that fail to replicate at scale."*

### 19.5 Implications

This concludes the inference-time masking arc of Part 2. Three concrete next steps:

1. **The paper's inference-time chapter writes itself as a negative result.** Section 16 establishes the determinism issue, Sections 17-18 the MATH-500 spurious-effect story, Section 19 the conclusive null on MATH-5000.

2. **Step 6 (training-time loss masking) is now the only path forward** for extracting signal from OPD-recalcitrant tokens. The decisive experiment was always going to be:
   - Re-run OPD with the loss zeroed on identified Stumbling Blocks
   - Compare convergence + final-model accuracy to baseline OPD
   - If signal exists, it lives in the gradients, not in inference-time argmax decisions

3. **The whole story is stronger as a methodological contribution.** Rather than a brittle "we found Pillars and Stumbling Blocks" claim that wouldn't replicate, the paper now has:
   - A reproducibility infrastructure (verify_masking, deterministic vLLM config)
   - A subsample-bias diagnosis (MATH-500 → MATH-5000 collapse)
   - A clean null at scale
   - A motivated transition to training-time intervention

---

*Step A data: `results/masking/groups_math_full/count/`*
*All 14 group JSONs: `groups/*.json` (with full per-problem correctness vectors for re-analysis)*
*Summary: `summary.csv`, `summary.json`*
*Plot: `plots/groups.png` (math-only, 14 groups)*

---

## 20. Final Synthesis

### 20.1 The Journey

The Part 2 arc moved through six successive refinements, each correcting an issue in the previous:

1. **Raw knockout (non-deterministic, MATH-500, n=500):** appeared to show 82% Pillars, mean Δ = -0.91%, with 7 "Strong Pillars" at p < 0.05 led by " certain" at -3.4% (Section 13-14).

2. **Determinism discovery (Section 15):** vLLM with default V1 multiprocessing produces ~36 problem-flips of cross-session variance on A100, even at greedy temperature=0 with fixed seed. The within-session bootstrap underestimates true variance.

3. **Determinism fix:** `VLLM_ENABLE_V1_MULTIPROCESSING=0` + `top_k=1`. Verified byte-identical baseline output across separate processes via `verify_masking.py` Test 7.

4. **Deterministic knockout (MATH-500, n=500):** distribution rebalances to 46% Pillar / 7% Neutral / 46% Stumbling. **0 of 200 tokens significant on MATH-500 after bootstrap.** Single Strong Stumbling Block is " shape" on IF-Eval (Section 16).

5. **Group masking (MATH-500):** four groups appear to show |Δ| ≥ 1% effects: `semantic_discourse` (+1.20%), `top5_pillar` (-2.00%), `cross_task_pillars` (+1.20%), `semantic_domain` (-1.00%) (Section 18).

6. **MATH-5000 validation of all 14 groups:** **0 of 14 groups significant after Bonferroni correction.** The single uncorrected significant result is a random control. 10 of 14 groups change sign vs MATH-500. Random and non-random groups are statistically indistinguishable (Section 19).

### 20.2 Three Robust Findings

After all the corrections and scaling, three findings remain robust:

**(F1) The methodological finding.** vLLM produces ~7% per-problem cross-session variance on A100 at greedy decoding with fixed seed. Inference-time masking studies on non-H100 hardware require explicit determinism configuration (`VLLM_ENABLE_V1_MULTIPROCESSING=0`). Without this, single-token Pillar/Stumbling Block classifications flip sign across ~50% of tokens between sessions, and bootstrap p-values systematically underestimate true variance.

**(F2) The subsample-bias finding.** MATH-500 (n=500), the standard "small" benchmark, exhibits substantial subsample directional bias for masking studies. Effects of ±1-2% on MATH-500 are typically not reproducible on the full Hendrycks MATH test (n=5000); 10 of our 14 groups changed sign upon scaling. This is a methodological caution for the broader inference-time-masking literature.

**(F3) The null result.** OPD-recalcitrant tokens identified by Part 1's criterion (high training loss, low improvement) are causally inert under inference-time logit-bias masking on MATH (n=5000). No single token, group, semantic cluster, or interaction effect reaches significance after multiple-testing correction. Random controls are indistinguishable from "candidate" groups. This null cannot be more cleanly rejected.

### 20.3 What This Means for the Project

The Part 2 outcome is not a failed identification but a **redirected one**. Three concrete consequences:

1. **The recalcitrance criterion in Part 1 is well-defined and reproducible**, but it identifies a feature of the *training dynamics* (where OPD loss never converged), not of *inference-time function*. The two are not the same thing. This is a substantive scientific point worth its own subsection in the paper.

2. **Inference-time logit suppression is the wrong intervention** for these tokens. The student's greedy decoding finds substitute pathways for any single token or small group; whether those substitutes happen to help or hurt on a given problem subset is essentially random at the n=5000 scale.

3. **Step 6 (training-time loss masking) becomes the only path forward.** The implementation plan called it "the decisive experiment" — and given the inference-time null, it now genuinely is. Its design also gains clarity from the inference-time results:
   - The "Stumbling Block mask" config can use the full 200 Rock Tokens (since no smaller subset is well-justified by inference-time evidence)
   - The "random mask" control becomes critical (we now know random matches "candidate" inference-time, so any training-time gap between them is meaningful)
   - The "all 200 mask" config tests whether removing the recalcitrant loss entirely improves training, regardless of token-level taxonomy

### 20.4 Paper Structure Implications

The story restructures cleanly around the methodological core:

- **Part 1 (existing):** The recalcitrance criterion and 200-token inventory.
- **Part 2 (revised):** Inference-time masking is dominated by session-level vLLM noise and benchmark subsample bias. After correcting for both, no robust per-token or per-group effect exists on MATH (n=5000). This motivates Part 3.
- **Part 3 (Step 6, future):** Training-time loss masking — does removing the recalcitrant-loss contribution from OPD training produce a measurably better model?

The shift from "we identify Pillars and Stumbling Blocks" to "we show inference-time masking is null and motivate training-time intervention" is more honest, more rigorous, and arguably more interesting. The methodological contributions (F1, F2) are useful to the broader community independent of Part 3's outcome.

### 20.5 Reusable Artifacts

Part 2 produced reusable infrastructure that lives on regardless of the project's direction:

- **`src/masking/verify_masking.py`** — 7-test cross-session reproducibility diagnostic. Useful for any vLLM-based inference study.
- **Deterministic vLLM configuration recipe** — `VLLM_ENABLE_V1_MULTIPROCESSING=0` + `top_k=1` (`src/masking/common.py`).
- **`src/masking/categorize.py`** — paired-bootstrap categorization with feature correlations.
- **`src/masking/cumulative.py`** — cumulative-removal curves with greedy and random null baselines.
- **`src/masking/groups.py`** — semantic group masking (parameterized over benchmark size).
- **MATH-5000 evaluator** — `load_math_full()` in `src/masking/eval_math500.py` (with `\boxed{}` answer extraction across 7 subjects).

All scripts are deterministic, resume-safe, and self-contained (no imports from `src/evaluation/`, `src/analysis/`, `src/exp_2/`).

### 20.6 Limitations

For honest reporting:

1. **Single student model.** All masking experiments use one student checkpoint (`RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k`). Effects on the offpolicy variant or other distillation runs are unstudied.

2. **Single benchmark family.** The MATH-5000 null result establishes inference-time masking is null on competition mathematics. We cannot generalize to other domains (code, instruction-following, etc.) without dedicated evaluation. Earlier IF-Eval data (Sections 13-18) was retained for Step 2.1 / 2.2 only and dropped for the MATH-focused Step A; revisiting IF-Eval at scale is future work.

3. **Greedy decoding only.** All experiments use greedy decoding (`temperature=0`, `top_k=1`). Sampling-based inference might exhibit different masking sensitivity, particularly for tokens involved in self-consistency-style aggregation.

4. **Single-token / static-group masking only.** The intervention is "set selected token logits to -∞ at every position". Position-conditional masking, or masking that triggers only at certain context features, is outside the scope of this study.

5. **No Step 5 (pairwise interactions) at scale.** Section 18 documented non-additivity in the MATH-500 group results, but we did not run a systematic pairwise interaction matrix at MATH-5000. Given the null group result, this is unlikely to be productive without first establishing main effects.

### 20.7 Closing Note

The clean conclusion: **inference-time masking of OPD-recalcitrant tokens does not produce a reproducible improvement on MATH (n=5000) under any single-token, group, or cumulative configuration we tested, with deterministic vLLM and bootstrap-corrected statistics.** The path forward is training-time intervention.

---

*This concludes Part 2 of the Rock Tokens project.*
*Pipeline version: 0.7.0 (deterministic vLLM + MATH-5000 validation)*
*Last updated: 2026-04-28*

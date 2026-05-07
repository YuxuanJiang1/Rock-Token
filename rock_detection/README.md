# Rock-Token Analysis

Identifying and analyzing **rock tokens** — tokens that retain elevated KL divergence between a distilled student model and its teacher even after on-policy distillation (OPD).

The pipeline collects per-token KL statistics over MATH-500 generations, selects rock tokens by a frequency-weighted criterion (`freq × mean KL`), and analyzes their per-output prevalence, frequency-vs-KL "funnel" structure, and per-token gradient alignment with the global descent direction.

## What's in this repo

| Stage | Script | Runs on | Purpose |
|---|---|---|---|
| **1. Collect KL** | `rock_server.py` | GPU server | Generate with student, evaluate teacher, record per-token reverse-KL across 500 MATH-500 prompts |
| **2. Select rocks** | `rerun_unrestricted.py` | Local | Top-100 by `freq × mean KL` → `rock_vs_control_unrestricted.csv` |
| **3. Visualize funnel** | `funnel_plot.py`, `rock_funnel_plot.py` | Local | KL variance vs `log10(freq)`; rocks highlighted on the funnel |
| **4. Per-output dist.** | `rock_per_output.py`, `per_output_unrestricted.py` | Local | Rock-token saturation per generated output, vs non-rock high-KL events |
| **5. Method stability** | `test_stability.py` | Local | Compare 12 selection metrics across n=50…500 (Jaccard, Spearman) |
| **6. Gradient analysis** | `compute_logit_gradients.py` | GPU server | Per-token reverse-KL gradient direction in logit space (Proxy A) |
| **7. Gradient cosines** | `analyze_gradient_alignment.py`, `gradient_alignment.py`, `analyze_gradient_magnitude.py` | Local | Cosine similarity of per-token gradient with corpus reference; rock vs high-KL vs random groups |
| **8. Cross-checkpoint** | `compare_checkpoints.py`, `compare_kl_evolution.py` | Local | KL trajectories across early (5k) vs late (10k) OPD checkpoints |
| **9. Paper figures** | `paper_figure1_combined.py`, `paper_figure2_gradient.py`, `paper_figure_kcutoff.py` | Local | Final multi-panel figures for the paper |

Supporting utilities: `decode_rock_tokens.py` (decode IDs → strings), `find_rock_cutoff.py` (sweep top-K threshold), `compare_unrestricted.py` (256-cap vs unrestricted), `inspect_*.py`, `visualize_*.py`.

## Models and data

- **Teacher**: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **Students** (private RockToken org checkpoints):
  - `RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k` — early on-policy distillation
  - `RockToken/qwen3-30b-a3b-to-4b-onpolicy-10k` — late on-policy distillation
  - `RockToken/qwen3_30b_a3b_to_4b_offpolicy_math_first20k` — off-policy distillation
- **Eval prompts**: `HuggingFaceH4/MATH-500`, sampled deterministically with `seed=42`

## Setup

```bash
# Python 3.10+
pip install torch transformers datasets accelerate bitsandbytes \
            pandas numpy scipy matplotlib tqdm

# (server) for the data-collection scripts
export HF_TOKEN=<your-token>     # required for private RockToken checkpoints
export HF_HOME=/workspace/hf_cache   # so cache_dir and HF_HOME agree
```

Server requirements: a single 96 GB GPU (e.g. H100) is sufficient for the unrestricted run with `MAX_NEW_TOKENS=4096`. A two-GPU 40 GB setup also works (use `--hardware dual_40gb`).

## End-to-end pipeline

### 1. Collect KL data on the server

```bash
# 256-cap (original)
python rock_server.py --student onpolicy --samples 500 --hardware single_96gb

# Unrestricted (model decides, 4096-token safety cap)
python rock_server.py --student onpolicy --samples 500 --hardware single_96gb \
       --unrestricted

# Same for off-policy and the 10k checkpoint
python rock_server.py --student offpolicy    --samples 500 --unrestricted
python rock_server.py --student onpolicy_10k --samples 500 --unrestricted
```

Outputs: `rock_token_occurrences_<student>_n<N>[_unrestricted].pt` containing per-position records and aggregated frequency/KL stats. ~50 MB per run for n=500 unrestricted.

### 2. Select rock tokens

```bash
python rerun_unrestricted.py
```

Reads `rock_token_occurrences_onpolicy_n500_unrestricted.pt`, ranks tokens by `freq × mean KL`, drops the top-5 ultra-high-frequency tokens (spaces / common punctuation that dominate by frequency alone), takes ranks 1-100 as **rocks** and ranks 101-200 as **controls**, writes `rock_vs_control_unrestricted.csv`. Also runs the per-output distribution and funnel plot.

### 3. Funnel plot

```bash
python funnel_plot.py --output funnel_unrestricted.png
python rock_funnel_plot.py --output rock_funnel_unrestricted.png \
       --data rock_token_occurrences_onpolicy_n500_unrestricted.pt \
       --rock-csv rock_vs_control_unrestricted.csv \
       --show-control
```

The plot shows `log10(freq)` (x) vs `mean KL` (y). Low-frequency tokens fan out (high variance, mostly noise); high-frequency tokens converge tightly. Rock tokens (red) sit in the narrow high-frequency region, slightly above the binned-mean trend.

### 4. Per-output distribution

```bash
python rock_per_output.py \
       --data-on  rock_token_occurrences_onpolicy_n500.pt \
       --data-off rock_token_occurrences_offpolicy_n500.pt \
       --rock-csv rock_vs_control_unrestricted.csv
```

Counts rock-token occurrences and non-rock high-KL events per generated output. Outputs `per_output_distribution.csv` plus a 2×2 panel plot of histograms and scatters.

### 5. Stability of the selection method

```bash
python test_stability.py
```

Runs 12 candidate scoring methods (raw mean, frequency-filtered mean, LCB Student-t, EB shrinkage, joint min, freq×mean, freq×LCB, etc.) over n ∈ {50, 100, 200, 300, 400, 500} and reports Jaccard / Spearman against the n=500 reference. **`joint_freq*mean`** wins (Jaccard 0.86, Spearman 0.91 averaged over n ≤ 400).

### 6. Gradient analysis (Proxy A)

On the server, recompute per-token reverse-KL gradient directions in logit space — reusing the generated tokens from step 1 so this is a **forward-only** pass (no autoregressive generation):

```bash
python compute_logit_gradients.py \
       --student onpolicy --samples 500 --hardware single_96gb \
       --occurrences-file rock_token_occurrences_onpolicy_n500_unrestricted.pt
```

Outputs `logit_gradients_<student>_n<N>_unrestricted.pt` (≈ 3.6 GB; per-token gradient sums in vocab space). Locally:

```bash
python analyze_gradient_alignment.py
python analyze_gradient_magnitude.py
```

Cosines vs `G_balanced` (frequency-balanced reference) and `G_global` are computed for three groups: **rock**, **high_kl** (top-100 by mean KL, freq ≥ 2, excluding rocks), and **random_other** (200-token baseline). Result: **rocks have a long tail of meaningfully aligned gradients** (cosines reaching 0.6); high-KL rare tokens cluster within 1-2σ of orthogonality; the directional gap explains why OPD reduces rocks slowly while leaving high-KL outliers untouched.

### 7. Cross-checkpoint comparison

```bash
python compare_kl_evolution.py
python paper_figure2_gradient.py
```

Pairs each token's KL across early (5k) and late (10k) on-policy checkpoints. Rocks cluster on the diagonal (persistent); high-KL rare tokens fall below it (learned during late OPD).

## Key intermediate artifacts

| File | What it is |
|---|---|
| `rock_token_occurrences_<student>_n<N>[_unrestricted].pt` | Raw per-position KL data |
| `rock_vs_control[_unrestricted].csv` | 100 rocks + 100 controls (id, decoded token, freq, mean KL, freq×mean) |
| `rock_tokens_decoded.csv` | Rock list with full token-string repr |
| `per_output_distribution[_unrestricted].csv` | Per-sample rock saturation and KL share |
| `rock_cutoff_sweep.csv` | Sweep of top-K cutoff vs corpus KL coverage |
| `logit_gradients_<student>_n<N>_unrestricted.pt` | Per-token gradient direction matrices (vocab-dim) |
| `gradient_magnitude.csv`, `gradient_alignment.csv` | Per-token gradient stats by group (rock / high_kl / random) |
| `kl_evolution.csv` | Per-token KL across checkpoints (paired 5k vs 10k) |

## Method choices and gotchas

- **`freq × mean KL` over `mean KL`**: Per-token mean KL is dominated by low-frequency noise (the funnel plot's wide end). Frequency-weighting promotes tokens with statistically reliable estimates and accumulates the actual contribution to corpus-level divergence. The joint version `min(freq*mean_on, freq*mean_off)` is the most stable selector across sample sizes.
- **Drop top-5 highest-frequency tokens before ranking**: Pure-frequency dominators (space, common punctuation) would otherwise saturate the rock list. They're filtered in `rerun_unrestricted.py` (`EXCLUDE_TOP_FREQ=5`).
- **Reverse KL, not forward**: `KL(p_student || p_teacher)`. Gradient w.r.t. student logits has a `p^s_u (A_u - <A>)` form rather than the simpler `p^s - p^t` of forward KL.
- **Generation cap**: 4096 tokens is "unrestricted" in practice (median MATH-500 reasoning is ~700 tokens, p95 ≈ 4096). Setting `HF`'s default `max_length=20` is the foot-gun behind a previous bad run; always pass `max_new_tokens` explicitly.
- **Frequency-balanced vs frequency-weighted gradient reference**: `G_global = sum_i g_i` is dominated by rocks by construction, so rock-vs-everyone-else cosines under `G_global` are partly trivial. The substantive test uses `G_balanced = sum_t bar_g_t` (each token type counted once).

## Reproducing the paper figures

```bash
python paper_figure1_combined.py     # funnel plot + rock-tokens-on-funnel
python paper_figure2_gradient.py     # gradient analysis 4-panel
python paper_figure_kcutoff.py       # top-K cutoff sweep
```

Outputs land as `paper_figure*.{png,pdf}` at 600 DPI.

## Compute / disk budget

| Run | Compute | Output size |
|---|---|---|
| `rock_server.py --samples 500 --unrestricted` | ~3-4 h on one H100 | ~50 MB `.pt` |
| `compute_logit_gradients.py --samples 500` | ~1-2 h on one H100 | ~3.6 GB `.pt` |
| HF cache (1 student + teacher, bf16) | — | ~70 GB |
| HF cache (3 students + teacher, w/ fp32 student) | — | ~88 GB |

If disk is tight, the gradient `.pt` can be written in bf16 (halves size, negligible cosine error).

## Citation

Pending paper. Working concept: *"Rock tokens and the gradient-alignment view of on-policy distillation."*

## License

TBD — checkpoints under `RockToken/` on the Hub are private; please follow the access policy on those repos.

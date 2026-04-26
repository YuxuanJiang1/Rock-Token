# Rock Token Identification Pipeline — Design Spec

**Date:** 2026-04-25
**Status:** Approved
**Scope:** Part 1 of the implementation plan — identifying 100 Rock Tokens from a trained OPD student.

---

## Goal

Given a teacher model, a pre-OPD student, and a post-OPD student, identify the top 100 Rock Token *types* (vocabulary items) that are consistently recalcitrant — high KL divergence before training that barely improves after training. Produce sanity-check plots and analysis ready for a paper.

## Models

All model paths are stored in `config.yaml` (not hardcoded) so they can be swapped at any time.

| Role | Default Model |
|------|---------------|
| Teacher | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Student (pre-OPD, θ₀) | `Qwen/Qwen3-4B-Instruct-2507` |
| On-policy distilled (θ*) | `RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k` |
| Off-policy distilled | `RockToken/qwen3_30b_a3b_to_4b_offpolicy_20k` |

The pipeline runs with `teacher + θ₀ + θ*` to produce one set of Rock Tokens. The off-policy model can be swapped in later.

## Prompt Set

1,500 prompts from a **mixed dataset** (train splits only — no overlap with evaluation benchmarks):

| Source | Split | Count | Domain |
|--------|-------|-------|--------|
| MATH (Hendrycks) | train | 500 | Competition math |
| GSM8K | train | 500 | Grade-school math |
| MBPP | train | 300 | Code generation |
| Alpaca | full (no test split) | 200 | General instructions |

## Pipeline Phases

### Phase 1: Generate Student Outputs (vLLM)

Load θ* via vLLM. Generate **3 outputs per prompt** at **temperature=1.0** with sampling, max 2048 new tokens.

- Total: 4,500 sequences
- Output: token IDs, prompt lengths, source dataset labels
- Storage: single `.pt` file (~40 MB)

vLLM is used here for fast batched generation. Temperature 1.0 (not greedy) ensures we sample the student's actual distribution rather than just the argmax, which would hide the confidently-wrong behavior we want to characterize.

### Phase 2: Compute Per-Token KL at Both Checkpoints (HF Transformers)

Teacher-forced forward passes on all 4,500 generated sequences. For each response-token position, compute KL(teacher || student) with both students.

**Two-pass approach** (fits 2x A100-80GB, zero disk storage for distributions):

| Pass | GPU contents (~68 GB) | Computes |
|------|----------------------|----------|
| Pass 1 | Teacher (TP=2, ~60GB) + θ₀ (~8GB) | `loss_before` per token |
| Pass 2 | Teacher (TP=2, ~60GB) + θ* (~8GB) | `loss_after` per token |

Each pass, for every sequence:
1. Teacher forward pass → full logits at response positions (held in memory)
2. Student forward pass → full logits at response positions
3. Compute `KL(teacher || student)` per position → scalar
4. Also compute: teacher entropy, student entropy per position
5. Save scalars only — no distributions stored to disk

Total output: ~90 MB of per-token measurements.

**Per-token fields saved:**
- `token_id` (int)
- `position` (int, position in sequence)
- `loss_before` (float, KL with θ₀)
- `loss_after` (float, KL with θ*)
- `teacher_entropy` (float)
- `student_entropy_before` (float)
- `student_entropy_after` (float)
- `source_dataset` (str label)
- `sequence_idx` (int)

### Phase 3: Apply Recalcitrance Criterion

**Relative threshold** (preferred per implementation plan):
- `τ_high` = 80th percentile of `loss_before`
- `δ` = 20th percentile of `(loss_before - loss_after)` (improvement distribution)
- A token instance is **Rock** if:
  - `loss_before > τ_high` (was hard before training)
  - `(loss_before - loss_after) < δ` (barely improved after training)

This keeps only tokens where the student genuinely failed to make progress — not tokens that were hard but learned.

### Phase 4: Aggregate to Token Types

1. Group Rock instances by vocabulary token ID
2. Frequency filter: token must appear ≥10 times in total (not just as Rock) to avoid rare BPE artifacts
3. Compute `rock_rate = rock_count / total_count` per token type
4. Rank by rock_rate, take top 100
5. Also compute and report absolute count ranking

**Output per token type:**
- `rank`, `token_id`, `token_string` (decoded)
- `frequency` (total appearances), `rock_count`, `rock_rate`
- `avg_loss_before`, `avg_loss_after`, `avg_improvement`
- `avg_teacher_entropy`, `avg_student_entropy_before`, `avg_student_entropy_after`
- `classification` (Rock by definition — but will be further classified as Pillar/Stumbling Block in Part 2)

### Sanity Checks & Plots

1. **Scatter plot**: `loss_before` (x) vs `loss_after` (y) for all token instances. Color-coded quadrants:
   - Easy (low/low) — green
   - Learned (high/low) — blue
   - Rock (high/high) — red
   - Regressed (low/high) — orange
   - Threshold lines drawn at τ_high and δ boundary

2. **Rock Token fraction**: Print percentage of token instances classified as Rock. Should be 5–20%. Flag if outside this range.

3. **Top 100 inspection table**: Rich-formatted console table + saved as image. Columns: rank, token text, frequency, rock_rate, avg losses, classification.

4. **Entropy correlation plot**: Rock rate (y) vs student entropy (x) per token type. Tests whether Rock Tokens are just forking tokens (high entropy) or a genuinely different category. Include Pearson correlation coefficient on the plot.

## Code Structure

```
src/identification/
├── __init__.py
├── prompts.py          # Load & mix datasets (MATH/GSM8K/MBPP/Alpaca train splits)
├── generate.py         # Phase 1: vLLM generation (temp=1.0, 3 outputs/prompt)
├── measure.py          # Phase 2: KL at both checkpoints (two-pass, HF transformers)
├── identify.py         # Phase 3+4: recalcitrance criterion + aggregate + top 100
├── plots.py            # Sanity check plots (scatter, fraction, entropy correlation)
└── run.py              # CLI entry point: orchestrate all phases
```

Config:
```
config.yaml             # Model paths, hyperparameters (τ_high percentile, δ percentile, top_k, etc.)
```

Results:
```
results/identification/
├── phase1_sequences.pt     # Generated sequences
├── phase2_losses.pt        # Per-token loss_before, loss_after, entropies
├── rock_tokens.json        # Top 100 with full metadata
├── rock_tokens.csv         # CSV export
├── scatter_loss.png        # loss_before vs loss_after quadrant plot
├── rock_fraction.txt       # Rock fraction stats
├── entropy_correlation.png # Rock rate vs entropy
├── top100_table.png        # Visual table of top 100
└── summary.json            # Pipeline metadata (models, thresholds, dates, counts)
```

## Config Schema (config.yaml)

```yaml
models:
  teacher: "Qwen/Qwen3-30B-A3B-Instruct-2507"
  student_base: "Qwen/Qwen3-4B-Instruct-2507"
  student_onpolicy: "RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k"
  student_offpolicy: "RockToken/qwen3_30b_a3b_to_4b_offpolicy_20k"

generation:
  temperature: 1.0
  num_outputs_per_prompt: 3
  max_new_tokens: 2048

prompts:
  math_train_count: 500
  gsm8k_train_count: 500
  mbpp_train_count: 300
  alpaca_count: 200
  seed: 42

identification:
  tau_high_percentile: 80    # loss_before threshold
  delta_percentile: 20       # improvement threshold
  min_frequency: 10          # minimum token occurrences
  top_k: 100                 # number of Rock Token types to output

compute:
  tensor_parallel_size: 2
  max_model_len: 32768
  seed: 42
```

## Dependencies

No new dependencies beyond what's already in `pyproject.toml`. Uses:
- `vllm` (optional `[gpu]`) for Phase 1 generation
- `torch`, `transformers`, `accelerate` for Phase 2 forward passes
- `matplotlib` for plots
- `datasets` for loading prompts
- `rich` for console output

## Makefile Targets

```makefile
identify:           # Run full identification pipeline
identify-phase1:    # Run only Phase 1 (generation)
identify-phase2:    # Run only Phase 2 (KL measurement)
identify-phase3:    # Run Phase 3+4 (classify + aggregate, CPU only)
identify-plots:     # Regenerate plots from existing data (CPU only)
```

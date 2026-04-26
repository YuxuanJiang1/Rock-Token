# Rock Token Identification (Part 1)

## What this does

Given a teacher and a student trained via On-Policy Distillation (OPD), this pipeline finds **Rock Tokens** — vocabulary items where the student had high KL divergence with the teacher *before* training and *still* has high KL *after* training. These are the tokens the student genuinely failed to learn, not just tokens that are inherently hard.

## Why it matters

Not all high-loss tokens are equal. Some start hard and get learned (Learned Tokens). Some start easy and stay easy. Rock Tokens are the interesting ones: persistently high loss despite training. They fall into two categories we investigate in Part 2:

- **Pillar Tokens** — structurally essential; removing them hurts performance
- **Stumbling-Block Tokens** — noise the student wastes capacity on; removing them *helps*

Identifying which is which requires first finding the Rock Tokens (this pipeline), then running masking experiments (Part 2 in `src/evaluation/`).

## How it works

Three phases, run sequentially:

**Phase 1 — Generate.** The post-OPD student (θ\*) generates 3 sampled outputs per prompt (temp=1.0) across 1,500 mixed prompts (MATH, GSM8K, MBPP, Alpaca — all train splits). This produces 4,500 sequences that represent the student's actual output distribution. We sample rather than greedy-decode because greedy hides the confidently-wrong behavior we want to find.

**Phase 2 — Measure.** For every token position in every sequence, compute KL(teacher ‖ student) twice: once with the *pre*-OPD student (θ₀) → `loss_before`, and once with the *post*-OPD student (θ\*) → `loss_after`. The teacher stays loaded across both passes; only the student is swapped. No intermediate distributions are stored — just per-token scalars.

**Phase 3 — Classify.** A token instance is **Rock** if:
- `loss_before` ≥ 80th percentile (it was hard before training), AND
- `loss_before − loss_after` < 20th percentile of improvement (it barely improved)

Rock instances are then grouped by vocabulary ID, filtered to ≥10 occurrences, and ranked by `rock_rate = rock_count / total_count`. The top K are our Rock Tokens.

## What it produces

```
results/identification/
├── rock_tokens.json         # Top K Rock Tokens with full stats
├── rock_tokens.csv          # Same as CSV
├── scatter_loss.png         # loss_before vs loss_after — 4-quadrant plot
├── entropy_correlation.png  # Rock rate vs entropy — are these just forking tokens?
├── rock_fraction.txt        # Should be 5–20% of all instances
└── summary data (phase1_sequences.pt, phase2_losses.pt)
```

## How to run

```bash
make identify          # full pipeline (GPU, ~3 hours on 2x A100)
make identify-phase3   # re-classify + re-plot only (CPU, seconds)
```

All model paths and thresholds live in `config.yaml` at the project root. Change `top_k`, percentiles, or model paths there — no code edits needed. Phase 3 can be re-run cheaply after any config change.

## Key plots to check

1. **Scatter plot** — should show four clear quadrants (Easy, Learned, Rock, Regressed). If it's a featureless blob, the thresholds need adjusting.
2. **Rock fraction** — 5–20% is healthy. 50% means thresholds are too loose; 0.5% means too strict.
3. **Entropy correlation** — low correlation means Rock Tokens are a genuinely new category, not just high-entropy "forking tokens" from Wang et al. (2025).

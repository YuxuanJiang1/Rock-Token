# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating **Recalcitrant Tokens ("Rock Tokens")** in On-Policy Distillation (OPD) — tokens that maintain persistently high KL divergence loss before and after distillation. The central question: are these tokens essential structural components (**Pillar Tokens**, whose removal collapses training) or detrimental noise (**Stumbling-Block Tokens**, whose removal improves learning)?

### Experiment Structure

- **Exp 1 (Zhao):** OPD baseline — 2-stage training (SFT warmup on 20k teacher traces from OpenThoughts3, then OPD with 5k prompts, 4 rollouts each)
- **Exp 2 (Chao):** Rock Token identification — rank tokens by frequency × avg KL using Bayesian KL smoothing or beta-weighted geometric mean
- **Exp 3 (Dipta):** Causal intervention — remove stumbling blocks via constrained decoding (logits → -inf), cross-model transfer, observe new pillar emergence

### Models

| Role | Model |
|------|-------|
| Teacher | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Student 1 (base) | `Qwen/Qwen3-4B-Instruct-2507` |
| Student 2 | Llama-3.2-3B (cross family) |

### Evaluation Benchmarks

MATH-500, AIME 2024/2025, GSM8K, HumanEval, MBPP, IF-Eval

## Build & Run

- **Python 3.14+**, managed via `uv`
- Install deps: `uv sync`
- Run scripts: `uv run <script.py>`
- Add deps: `uv add <package>`

## Key References

- Proposal and slides in `docs/`
- Checkpoints: `https://huggingface.co/RockToken`
- OPD trained model (Exp 1): `RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k` (Qwen3-4B student distilled from Qwen3-30B teacher)

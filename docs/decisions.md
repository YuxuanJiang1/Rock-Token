# Project Decisions Log

This document tracks all design and implementation decisions for the Rock Token project. Updated as decisions are made.

---

## Exp 1: OPD Baseline

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | Teacher: `Qwen/Qwen3-30B-A3B-Instruct-2507`, Student base: `Qwen/Qwen3-4B-Instruct-2507` | Same-family distillation for initial experiments | Pre-project |
| 2 | 2-stage training: SFT warmup (20k traces from OpenThoughts3) then OPD (5k prompts, 4 rollouts) | Standard OPD setup following GKD framework | Pre-project |
| 3 | Trained model checkpoint: `RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k` | Exp 1 output, used as input for Exp 2 | 2026-04-18 |

## Exp 2: Rock Token Identification

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | Sequential pipeline: student pass -> teacher pass -> CPU analysis (never both models in GPU memory) | GPU-efficient; avoids needing ~68GB+ VRAM simultaneously | 2026-04-18 |
| 2 | Both models loaded in bf16 across multiple GPUs (`device_map="auto"`) | Full precision, leverage multi-GPU setup | 2026-04-18 |
| 3 | Dataset: MATH500 | Manageable size, matches initial analysis from slides | 2026-04-18 |
| 4 | Rock Token scoring method configurable via CLI arg: geometric mean or Bayesian averaging | Both methods have merit; want to compare | 2026-04-18 |
| 5 | Pillar vs Stumbling Block threshold: median teacher entropy across ALL tokens (not just Rock Tokens) | Global median gives meaningful baseline of "normal" teacher certainty; data-adaptive; transfers across model pairs; easy to explain in paper | 2026-04-18 |
| 6 | Output: pretty console display + structured file for Exp 3 consumption | Need both human-readable results and machine-readable input for next experiment | 2026-04-18 |
| 7 | Output file formats: both JSON (programmatic use) and CSV (quick inspection) | JSON for Exp 3 consumption, CSV for easy team review | 2026-04-18 |
| 8 | Single script with `--phase` resume capability, code in `src/exp_2/` | Simple to run, robust to crashes, can resume from any phase | 2026-04-18 |
| 9 | Add Makefile for easier running of experiments | Convenience shortcuts for common commands | 2026-04-18 |
| 10 | HF batched generation broken for Qwen3 (left-padding + RoPE + KV cache) | Confirmed by test: token IDs and log-probs differ between batch=1 and batch>1. position_ids fix insufficient — KV cache length includes padding, corrupting decode-step positions | 2026-04-18 |
| 11 | Use vLLM (>=0.19) for Phase 1 generation, `--backend vllm` flag | vLLM handles batching natively via continuous batching + PagedAttention, no padding issues. Optional dependency. | 2026-04-18 |
| 12 | Hybrid approach: vLLM gen + HF forward pass for log-probs | vLLM `logprobs=-1` returns full vocab but as Python dicts (~152k entries/position) — converting to tensors is slower than 500 HF forward passes (~2.5 min). HF forward pass is same proven pattern as Phase 2 teacher. | 2026-04-18 |
| 13 | `--tensor-parallel` defaults to all available GPUs (`torch.cuda.device_count()`) | Use full GPU resources by default | 2026-04-18 |

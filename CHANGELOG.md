# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-04-25

### Added
- **Rock Token identification pipeline** (`src/identification/`):
  - Phase 1: vLLM generation (temp=1.0, 3 outputs/prompt, 1500 mixed prompts)
  - Phase 2: Two-pass KL measurement (teacher stays loaded, fits 2x A100-80GB)
  - Phase 3+4: Recalcitrance criterion (relative threshold) + type aggregation (top 100)
  - Sanity check plots: loss scatter (4-quadrant), rock fraction, entropy correlation
- `config.yaml` for all model paths and hyperparameters
- Makefile targets: `identify`, `identify-phase1`, `identify-phase2`, `identify-phase3`
- `pyyaml` and `scipy` added to project dependencies

## [0.3.0] - 2026-04-25

### Added
- **Baseline runner** (`run_baseline.py`): single command runs all 4 benchmarks sequentially, each in a subprocess for clean GPU memory management
- **Results summarizer** (`summarize_results.py`): reads result directories, generates comparison tables and grouped bar charts for paper figures
- Makefile targets: `baseline`, `baseline-smoke` (10-sample quick test), `summarize`
- Structured output: `summary.json` + `accuracy_chart.png` per run, `comparison.json` + `comparison_chart.png` for cross-experiment comparison

## [0.2.0] - 2026-04-25

### Added
- **Evaluation harness** with 4 standard benchmarks, all using vLLM for fast batched inference:
  - `eval_gsm8k.py` — GSM8K (1319 problems), 0-shot CoT or 8-shot CoT (Wei et al. 2022)
  - `eval_mmlu.py` — MMLU (14042 questions, 57 subjects), 5-shot (Hendrycks et al. 2021)
  - `eval_humaneval.py` — HumanEval (164 problems), 0-shot, pass@1 with sandboxed execution
  - `eval_ifeval.py` — IF-Eval (541 prompts), 0-shot, strict + loose scoring (Zhou et al. 2023)
- Shared evaluation utilities in `src/evaluation/common.py` (vLLM engine, result I/O, CLI args, rich tables)
- Makefile targets: `eval-gsm8k`, `eval-mmlu`, `eval-humaneval`, `eval-ifeval`, `eval-all`
- `vllm` (optional `[gpu]`) and `matplotlib` added to project dependencies

## [0.1.0] - 2026-04-18

### Added
- Rock Token identification pipeline (Exp 2): 3-phase student generation, teacher KL, analysis
- AIME benchmark evaluation script with vLLM
- Bayesian and geometric scoring methods
- Pillar / Stumbling-Block classification by teacher entropy

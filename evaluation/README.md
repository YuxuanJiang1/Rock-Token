# Evaluation

Scripts and task configs for evaluating models on AIME 2024, AIME 2025, and HMMT Feb 2025.

## Requirements

```bash
pip install lm-eval vllm
```

## Task configs

The `tasks/` directory contains custom [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) task definitions:

| File | Benchmark | Dataset |
|---|---|---|
| `aime24_sample.yaml` | AIME 2024 | `Maxwell-Jia/AIME_2024` |
| `aime25_sample.yaml` | AIME 2025 | `Maxwell-Jia/AIME_2025` |
| `hmmt25feb.yaml` | HMMT Feb 2025 | `MathArena/hmmt_feb_2025` |

`utils.py` provides the answer extraction and exact-match scoring logic shared across all three tasks.

## Running evaluation

### With vLLM backend (recommended, 2× A100 80 GB)

```bash
bash evaluation/run_eval_vllm.sh
```

Edit `MODEL` at the top of the script to point to your checkpoint, e.g.:

```bash
MODEL=/path/to/your/checkpoint
```

The script runs **3 independent seeds** (59, 76, 93) and reports per-task accuracy plus mean ± std across runs.

### With HuggingFace backend (single GPU)

```bash
bash evaluation/run_eval_base.sh
```

## Output

Results are written to `eval_results/<run>/` and automatically aggregated:

```
Task                   Run1    Run2    Run3    Mean    Std
aime24_sample          46.7    40.0    53.3    46.7    6.7
aime25_sample          46.7    40.0    50.0    45.6    5.1
hmmt25feb              23.3    30.0    30.0    27.8    3.9
------------------------------------------------------------
Avg across tasks                                40.0
```

## Hyperparameters

| Setting | Value |
|---|---|
| Temperature | 0.6 |
| Max generation tokens | 16384 |
| Few-shot | 0 |
| Metric | Exact match |
| Aggregation | Mean over 3 runs |

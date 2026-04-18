# Exp 2: Rock Token Identification Pipeline — Design Spec

## Goal

Build a single-script pipeline that identifies Rock Tokens from an OPD-trained student model and classifies them as Pillar or Stumbling Block. GPU-efficient: only one model loaded at a time.

## Models

- **Student:** `RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k` (Qwen3-4B, OPD-trained)
- **Teacher:** `Qwen3/Qwen3-30B-A3B`
- Both loaded in bf16 across multiple GPUs via `device_map="auto"`

## Dataset

MATH500 — all 500 problems.

## Pipeline: 3 Resumable Phases

### Phase 1: Student Pass

1. Load student model
2. Load MATH500 prompts
3. For each prompt: generate a response (greedy decoding)
4. Run forward pass on each (prompt + response) to get per-token student logits (full vocabulary)
5. Save to `{output_dir}/student_logits.pt`:
   - Generated token IDs per sample
   - Student logits (vocab-sized) at each position
   - Prompt lengths (to separate prompt from response tokens)
6. Unload student model

### Phase 2: Teacher Pass

1. Load teacher model
2. Load saved student responses from Phase 1
3. For each (prompt + student response): run teacher forward pass to get per-token teacher logits
4. Save to `{output_dir}/teacher_logits.pt`:
   - Teacher logits (vocab-sized) at each position
5. Unload teacher model

### Phase 3: Analysis (CPU-only)

1. Load saved student and teacher logits
2. **Per-token KL divergence:** `KL(P_teacher || P_student)` at each response token position (skip prompt tokens)
3. **Per-token teacher entropy:** `H_t = -sum(P_teacher * log(P_teacher))`
4. **Aggregate by token ID across all samples:**
   - Frequency: how many times token ID appears across all response positions
   - Average KL divergence
   - Average teacher entropy
5. **Score Rock Tokens** (method selected via `--scoring` arg):
   - **Geometric:** `RockScore_i = (f_i / max(f))^alpha * (avg_KL_i / max(avg_KL))^beta`
     - Defaults: alpha=0.3, beta=0.7
   - **Bayesian:** `BayesianKL_i = (f_i * avg_KL_i + C * mu) / (f_i + C)`
     - C = median frequency across all token types
     - mu = global average KL across entire vocabulary
6. **Classify Pillar vs Stumbling Block:**
   - Compute global median teacher entropy across ALL token positions (not just Rock Tokens)
   - Rock Tokens with avg teacher entropy < global median → **Pillar**
   - Rock Tokens with avg teacher entropy >= global median → **Stumbling Block**
7. **Output top-k Rock Tokens** (default k=50)

### Resume Logic

Each phase checks if its output file exists on disk. If found, skip to next phase. The `--phase` flag forces restart from a specific phase (deletes that phase's output and re-runs from there).

## CLI Interface

```bash
uv run src/exp_2/identify_rock_tokens.py \
  --student RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k \
  --teacher Qwen3/Qwen3-30B-A3B \
  --scoring bayesian \
  --alpha 0.3 --beta 0.7 \
  --top-k 50 \
  --output-dir results/exp2/ \
  --phase 1
```

All args except `--student` and `--teacher` have defaults.

## Makefile Targets

```makefile
exp2:                    # Run full pipeline with default args (bayesian scoring)
exp2-geometric:          # Run with geometric scoring
exp2-phase1:             # Run only phase 1
exp2-phase2:             # Run only phase 2
exp2-phase3:             # Run only phase 3 (re-analyze with different scoring)
```

## Output

### Console

Rich table:

```
Rank | Token ID | Token String | Freq | Avg KL | Rock Score | Avg Teacher Entropy | Classification
1    | 9309     | 'when'       | 42   | 7.47   | 0.893      | 1.23                | Pillar
2    | 36340    | 'East'       | 38   | 6.95   | 0.871      | 4.56                | Stumbling Block
...
```

Plus summary stats: total tokens analyzed, Rock Token count, Pillar count, Stumbling Block count, entropy threshold used.

### JSON (`{output_dir}/rock_tokens.json`)

```json
{
  "metadata": {
    "student_model": "...",
    "teacher_model": "...",
    "dataset": "math500",
    "scoring_method": "bayesian",
    "entropy_threshold": 2.34,
    "top_k": 50,
    "date": "2026-04-18"
  },
  "rock_tokens": [
    {
      "rank": 1,
      "token_id": 9309,
      "token_string": "when",
      "frequency": 42,
      "avg_kl": 7.47,
      "rock_score": 0.893,
      "avg_teacher_entropy": 1.23,
      "classification": "pillar"
    }
  ]
}
```

### CSV (`{output_dir}/rock_tokens.csv`)

Same columns as console table, for spreadsheet review.

## File Structure

```
src/exp_2/
  __init__.py
  identify_rock_tokens.py   # entry point + CLI arg parsing
  phases.py                 # run_phase1(), run_phase2(), run_phase3()
  scoring.py                # geometric_score(), bayesian_score()
  utils.py                  # data loading, logit save/load helpers
Makefile                    # convenience targets
```

## Dependencies

- `torch` — tensor ops, model loading
- `transformers` — model + tokenizer loading
- `datasets` — MATH500 from HuggingFace
- `rich` — console table output

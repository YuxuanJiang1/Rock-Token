# Exp 2: Rock Token Identification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable pipeline to identify Rock Tokens from OPD-trained student/teacher models on MATH500, score them (geometric or Bayesian), and classify as Pillar or Stumbling Block via entropy filtering.

**Architecture:** Three-phase sequential pipeline — Phase 1 generates student responses and saves per-token log-probs to disk (one file per sample for mid-phase resume), Phase 2 loads teacher and computes KL divergence + teacher entropy against saved student data, Phase 3 aggregates by token type, scores, classifies, and outputs results. Only one model is ever in GPU memory.

**Tech Stack:** Python 3.14, PyTorch, Transformers, Datasets, Rich, uv

---

## File Structure

```
src/exp_2/
  __init__.py                 # empty package init
  identify_rock_tokens.py     # CLI entry point, arg parsing, phase orchestration
  phases.py                   # run_phase1(), run_phase2(), run_phase3()
  scoring.py                  # geometric_score(), bayesian_score()
  utils.py                    # model/data loading, I/O, prompt formatting, rich output
tests/
  __init__.py
  exp_2/
    __init__.py
    test_scoring.py           # tests for scoring functions
    test_analysis.py          # tests for KL, entropy, classification logic
Makefile                      # convenience targets
```

**Data flow on disk:**

```
{output_dir}/
  student_data/
    sample_000.pt ... sample_499.pt   # Phase 1 output (temp, ~87MB each)
  phase2_data/
    sample_000.pt ... sample_499.pt   # Phase 2 output (tiny, per-token scalars)
  rock_tokens.json                    # Phase 3 final output
  rock_tokens.csv                     # Phase 3 final output
```

---

### Task 1: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `src/exp_2/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/exp_2/__init__.py`

- [ ] **Step 1: Add dependencies to pyproject.toml**

```toml
[project]
name = "chip-rock-token"
version = "0.1.0"
description = "Rock Token: Recalcitrant Token analysis in On-Policy Distillation"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "torch>=2.6",
    "transformers>=4.52",
    "datasets>=3.6",
    "rich>=14.0",
    "numpy>=2.2",
]
```

- [ ] **Step 2: Create package directories and init files**

```bash
mkdir -p src/exp_2 tests/exp_2
touch src/__init__.py src/exp_2/__init__.py tests/__init__.py tests/exp_2/__init__.py
```

- [ ] **Step 3: Install dependencies**

Run: `uv sync`
Expected: dependencies installed successfully

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/
git commit -m "feat: add project dependencies and package structure for exp2"
```

---

### Task 2: Scoring Module (TDD)

**Files:**
- Create: `tests/exp_2/test_scoring.py`
- Create: `src/exp_2/scoring.py`

- [ ] **Step 1: Write failing tests for scoring functions**

```python
# tests/exp_2/test_scoring.py
import numpy as np
from src.exp_2.scoring import geometric_score, bayesian_score


def test_geometric_score_basic():
    freq = np.array([10.0, 5.0, 1.0])
    avg_kl = np.array([1.0, 2.0, 3.0])
    scores = geometric_score(freq, avg_kl, alpha=0.3, beta=0.7)

    norm_freq = freq / freq.max()
    norm_kl = avg_kl / avg_kl.max()
    expected = (norm_freq**0.3) * (norm_kl**0.7)
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_geometric_score_uniform():
    """All same frequency and KL => all same score."""
    freq = np.array([5.0, 5.0, 5.0])
    avg_kl = np.array([2.0, 2.0, 2.0])
    scores = geometric_score(freq, avg_kl)
    np.testing.assert_allclose(scores, [1.0, 1.0, 1.0], rtol=1e-5)


def test_bayesian_score_basic():
    freq = np.array([100.0, 10.0, 1.0])
    avg_kl = np.array([2.0, 5.0, 10.0])
    C = 10.0
    mu = 3.0
    scores = bayesian_score(freq, avg_kl, C=C, mu=mu)

    expected = np.array([
        (100 * 2.0 + 10 * 3.0) / (100 + 10),  # 2.0909
        (10 * 5.0 + 10 * 3.0) / (10 + 10),     # 4.0
        (1 * 10.0 + 10 * 3.0) / (1 + 10),      # 3.6364
    ])
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_bayesian_score_low_freq_collapses_to_mu():
    """Very rare token's score should approach mu."""
    freq = np.array([0.001])
    avg_kl = np.array([100.0])
    C = 10.0
    mu = 3.0
    scores = bayesian_score(freq, avg_kl, C=C, mu=mu)
    np.testing.assert_allclose(scores, [mu], atol=0.1)


def test_bayesian_score_high_freq_keeps_true_kl():
    """Very frequent token's score should approach its true avg KL."""
    freq = np.array([100000.0])
    avg_kl = np.array([7.5])
    C = 10.0
    mu = 3.0
    scores = bayesian_score(freq, avg_kl, C=C, mu=mu)
    np.testing.assert_allclose(scores, [7.5], atol=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/exp_2/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.exp_2.scoring'`

- [ ] **Step 3: Implement scoring functions**

```python
# src/exp_2/scoring.py
import numpy as np


def geometric_score(
    frequencies: np.ndarray,
    avg_kls: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.7,
) -> np.ndarray:
    """Beta-weighted geometric mean Rock Score.

    RockScore_i = (f_i / max(f))^alpha * (avg_KL_i / max(avg_KL))^beta
    """
    norm_freq = frequencies / frequencies.max()
    norm_kl = avg_kls / avg_kls.max()
    return (norm_freq**alpha) * (norm_kl**beta)


def bayesian_score(
    frequencies: np.ndarray,
    avg_kls: np.ndarray,
    C: float,
    mu: float,
) -> np.ndarray:
    """Bayesian averaging (Laplace smoothing) Rock Score.

    BayesianKL_i = (f_i * avg_KL_i + C * mu) / (f_i + C)

    Args:
        C: confidence constant (e.g., median frequency across all token types)
        mu: global average KL across all token positions
    """
    return (frequencies * avg_kls + C * mu) / (frequencies + C)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/exp_2/test_scoring.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/exp_2/scoring.py tests/exp_2/test_scoring.py
git commit -m "feat: add geometric and bayesian Rock Token scoring with tests"
```

---

### Task 3: Utils Module

**Files:**
- Create: `src/exp_2/utils.py`

- [ ] **Step 1: Implement data loading and model helpers**

```python
# src/exp_2/utils.py
import csv
import json
from datetime import date
from pathlib import Path

import torch
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_math500():
    """Load MATH-500 dataset from HuggingFace."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return ds


def load_model_and_tokenizer(model_name: str):
    """Load model in bf16 distributed across available GPUs."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def format_prompt(problem: str, tokenizer) -> list[dict]:
    """Format a math problem as chat messages for apply_chat_template."""
    return [{"role": "user", "content": problem}]


def save_rock_tokens_json(rock_tokens: list[dict], metadata: dict, path: Path):
    """Save rock tokens and metadata to JSON."""
    output = {"metadata": metadata, "rock_tokens": rock_tokens}
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def save_rock_tokens_csv(rock_tokens: list[dict], path: Path):
    """Save rock tokens to CSV."""
    if not rock_tokens:
        return
    fieldnames = list(rock_tokens[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rock_tokens)


def print_results_table(rock_tokens: list[dict], entropy_threshold: float, metadata: dict):
    """Print rich table of Rock Token results to console."""
    console = Console()

    table = Table(title="Rock Token Identification Results")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Token ID", style="magenta", justify="right")
    table.add_column("Token", style="green")
    table.add_column("Freq", justify="right")
    table.add_column("Avg KL", justify="right", style="yellow")
    table.add_column("Rock Score", justify="right", style="bold")
    table.add_column("Avg Entropy", justify="right")
    table.add_column("Class", justify="center")

    for rt in rock_tokens:
        cls_style = "bold green" if rt["classification"] == "pillar" else "bold red"
        table.add_row(
            str(rt["rank"]),
            str(rt["token_id"]),
            repr(rt["token_string"]),
            str(rt["frequency"]),
            f"{rt['avg_kl']:.4f}",
            f"{rt['rock_score']:.4f}",
            f"{rt['avg_teacher_entropy']:.4f}",
            f"[{cls_style}]{rt['classification']}[/{cls_style}]",
        )

    console.print(table)

    n_pillar = sum(1 for rt in rock_tokens if rt["classification"] == "pillar")
    n_stumbling = len(rock_tokens) - n_pillar
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Total tokens analyzed: {metadata['total_positions']}")
    console.print(f"  Unique token types: {metadata['unique_token_types']}")
    console.print(f"  Entropy threshold (global median): {entropy_threshold:.4f}")
    console.print(f"  Top-{len(rock_tokens)} Rock Tokens: {n_pillar} Pillars, {n_stumbling} Stumbling Blocks")
```

- [ ] **Step 2: Commit**

```bash
git add src/exp_2/utils.py
git commit -m "feat: add utils for data loading, model loading, and output formatting"
```

---

### Task 4: Phase 1 — Student Generation

**Files:**
- Create: `src/exp_2/phases.py`

Phase 1 loads the student model, generates responses to all MATH500 prompts via greedy decoding, captures per-token logits via `output_scores=True`, converts to log-probs, and saves one `.pt` file per sample. Supports mid-phase resume by checking existing sample files.

- [ ] **Step 1: Implement Phase 1**

```python
# src/exp_2/phases.py
from pathlib import Path

import numpy as np
import torch
from rich.progress import Progress
from transformers import AutoTokenizer

from src.exp_2.scoring import bayesian_score, geometric_score
from src.exp_2.utils import (
    format_prompt,
    load_math500,
    load_model_and_tokenizer,
    print_results_table,
    save_rock_tokens_csv,
    save_rock_tokens_json,
)


def run_phase1(student_model_name: str, output_dir: Path, max_new_tokens: int = 2048):
    """Generate student responses and save per-token log-probs.

    Saves one file per sample to {output_dir}/student_data/sample_{i:03d}.pt
    containing: sample_idx, prompt_length, full_ids, student_log_probs (float16).
    Skips samples whose files already exist (mid-phase resume).
    """
    student_data_dir = output_dir / "student_data"
    student_data_dir.mkdir(parents=True, exist_ok=True)

    existing = {
        int(f.stem.split("_")[1])
        for f in student_data_dir.glob("sample_*.pt")
    }
    dataset = load_math500()
    remaining = [i for i in range(len(dataset)) if i not in existing]

    if not remaining:
        print(f"Phase 1 complete ({len(existing)}/{len(dataset)} samples already exist)")
        return

    print(f"Phase 1: {len(existing)}/{len(dataset)} done, {len(remaining)} remaining")
    model, tokenizer = load_model_and_tokenizer(student_model_name)

    with Progress() as progress:
        task = progress.add_task("Phase 1: Student generation", total=len(remaining))
        for i in remaining:
            messages = format_prompt(dataset[i]["problem"], tokenizer)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            prompt_length = input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            full_ids = outputs.sequences[0].cpu()

            # Stack generation scores: tuple of (1, vocab) -> (gen_len, vocab)
            scores = torch.stack(
                [s.squeeze(0) for s in outputs.scores], dim=0
            )
            log_probs = torch.log_softmax(scores.float(), dim=-1).cpu().half()
            del scores

            torch.save(
                {
                    "sample_idx": i,
                    "prompt_length": prompt_length,
                    "full_ids": full_ids,
                    "student_log_probs": log_probs,
                },
                student_data_dir / f"sample_{i:03d}.pt",
            )
            progress.advance(task)

    del model
    torch.cuda.empty_cache()
    print("Phase 1 complete")
```

- [ ] **Step 2: Commit**

```bash
git add src/exp_2/phases.py
git commit -m "feat: implement Phase 1 — student generation with per-sample logit saving"
```

---

### Task 5: Phase 2 — Teacher Forward + KL Computation

**Files:**
- Modify: `src/exp_2/phases.py`

Phase 2 loads the teacher model, iterates over saved student samples, runs teacher forward passes, and computes per-token KL(teacher || student) and teacher entropy. Saves per-token scalar results. Supports mid-phase resume.

- [ ] **Step 1: Implement Phase 2**

Append to `src/exp_2/phases.py`:

```python
def run_phase2(teacher_model_name: str, output_dir: Path):
    """Compute per-token KL divergence and teacher entropy.

    For each sample: loads student log-probs from Phase 1, runs teacher forward
    pass, computes KL(teacher || student) and teacher entropy per response token.
    Saves to {output_dir}/phase2_data/sample_{i:03d}.pt.
    """
    student_data_dir = output_dir / "student_data"
    phase2_dir = output_dir / "phase2_data"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    student_files = sorted(student_data_dir.glob("sample_*.pt"))
    if not student_files:
        raise FileNotFoundError(
            f"No student data found in {student_data_dir}. Run Phase 1 first."
        )

    existing = {
        int(f.stem.split("_")[1])
        for f in phase2_dir.glob("sample_*.pt")
    }
    remaining = [
        (f, int(f.stem.split("_")[1]))
        for f in student_files
        if int(f.stem.split("_")[1]) not in existing
    ]

    if not remaining:
        print(f"Phase 2 complete ({len(existing)} samples already processed)")
        return

    print(f"Phase 2: {len(existing)} done, {len(remaining)} remaining")
    model, _ = load_model_and_tokenizer(teacher_model_name)

    with Progress() as progress:
        task = progress.add_task("Phase 2: Teacher KL computation", total=len(remaining))
        for filepath, idx in remaining:
            data = torch.load(filepath, map_location="cpu", weights_only=False)
            full_ids = data["full_ids"]
            prompt_length = data["prompt_length"]
            student_log_probs = data["student_log_probs"].float()  # (response_len, vocab)

            response_len = student_log_probs.shape[0]

            with torch.no_grad():
                outputs = model(
                    input_ids=full_ids.unsqueeze(0).to(model.device)
                )

            # Extract teacher logits at response positions
            # logits[t] predicts token at t+1, so logits[prompt_len-1] predicts first response token
            logits = outputs.logits[0].float().cpu()  # (seq_len, vocab)
            teacher_logits = logits[
                prompt_length - 1 : prompt_length - 1 + response_len
            ]
            teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

            # KL(teacher || student) = sum P_teacher * (log P_teacher - log P_student)
            teacher_probs = teacher_log_probs.exp()
            kl_values = (
                teacher_probs * (teacher_log_probs - student_log_probs)
            ).sum(dim=-1)

            # Teacher entropy = -sum P_teacher * log P_teacher
            teacher_entropies = -(teacher_probs * teacher_log_probs).sum(dim=-1)

            # Response token IDs
            token_ids = full_ids[prompt_length:]

            torch.save(
                {
                    "sample_idx": idx,
                    "token_ids": token_ids,
                    "kl_values": kl_values,
                    "teacher_entropies": teacher_entropies,
                },
                phase2_dir / f"sample_{idx:03d}.pt",
            )
            progress.advance(task)

    del model
    torch.cuda.empty_cache()
    print("Phase 2 complete")
```

- [ ] **Step 2: Commit**

```bash
git add src/exp_2/phases.py
git commit -m "feat: implement Phase 2 — teacher forward pass with KL and entropy computation"
```

---

### Task 6: Phase 3 — Analysis and Output (TDD)

**Files:**
- Create: `tests/exp_2/test_analysis.py`
- Modify: `src/exp_2/phases.py`

- [ ] **Step 1: Write failing tests for classification and aggregation**

```python
# tests/exp_2/test_analysis.py
import numpy as np
import torch

from src.exp_2.phases import aggregate_token_stats, classify_tokens


def test_classify_tokens_below_threshold_is_pillar():
    avg_entropies = np.array([1.0, 3.0, 2.5])
    threshold = 2.5
    labels = classify_tokens(avg_entropies, threshold)
    assert labels == ["pillar", "stumbling_block", "stumbling_block"]


def test_classify_tokens_all_pillar():
    avg_entropies = np.array([0.5, 1.0, 1.5])
    threshold = 5.0
    labels = classify_tokens(avg_entropies, threshold)
    assert labels == ["pillar", "pillar", "pillar"]


def test_aggregate_token_stats():
    # Two samples: tokens [10, 20, 10] and [20, 30, 10]
    all_token_ids = torch.tensor([10, 20, 10, 20, 30, 10])
    all_kl = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    all_entropy = torch.tensor([0.5, 1.0, 0.7, 1.2, 2.0, 0.9])

    unique_ids, freqs, avg_kls, avg_entropies = aggregate_token_stats(
        all_token_ids, all_kl, all_entropy
    )

    # Token 10: freq=3, avg_kl=(1+3+6)/3=10/3, avg_entropy=(0.5+0.7+0.9)/3
    # Token 20: freq=2, avg_kl=(2+4)/2=3, avg_entropy=(1.0+1.2)/2
    # Token 30: freq=1, avg_kl=5, avg_entropy=2.0
    idx_10 = (unique_ids == 10).nonzero().item()
    idx_20 = (unique_ids == 20).nonzero().item()
    idx_30 = (unique_ids == 30).nonzero().item()

    assert freqs[idx_10] == 3
    assert freqs[idx_20] == 2
    assert freqs[idx_30] == 1

    np.testing.assert_allclose(avg_kls[idx_10], 10 / 3, rtol=1e-5)
    np.testing.assert_allclose(avg_kls[idx_20], 3.0, rtol=1e-5)
    np.testing.assert_allclose(avg_kls[idx_30], 5.0, rtol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/exp_2/test_analysis.py -v`
Expected: FAIL — `ImportError: cannot import name 'aggregate_token_stats'`

- [ ] **Step 3: Implement helper functions and Phase 3**

Add these functions to `src/exp_2/phases.py`:

```python
def classify_tokens(avg_entropies: np.ndarray, threshold: float) -> list[str]:
    """Classify tokens as pillar or stumbling_block based on entropy threshold."""
    return [
        "pillar" if e < threshold else "stumbling_block"
        for e in avg_entropies
    ]


def aggregate_token_stats(
    all_token_ids: torch.Tensor,
    all_kl_values: torch.Tensor,
    all_teacher_entropies: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-position stats by token ID.

    Returns: (unique_ids, frequencies, avg_kls, avg_entropies) as numpy arrays.
    """
    unique_ids, inverse = torch.unique(all_token_ids, return_inverse=True)
    n = len(unique_ids)

    frequencies = torch.zeros(n, dtype=torch.float64)
    sum_kls = torch.zeros(n, dtype=torch.float64)
    sum_entropies = torch.zeros(n, dtype=torch.float64)

    ones = torch.ones_like(all_kl_values, dtype=torch.float64)
    frequencies.scatter_add_(0, inverse, ones)
    sum_kls.scatter_add_(0, inverse, all_kl_values.double())
    sum_entropies.scatter_add_(0, inverse, all_teacher_entropies.double())

    avg_kls = (sum_kls / frequencies).numpy()
    avg_entropies = (sum_entropies / frequencies).numpy()
    frequencies = frequencies.numpy().astype(int)

    return unique_ids, frequencies, avg_kls, avg_entropies


def run_phase3(
    output_dir: Path,
    tokenizer_name: str,
    scoring_method: str,
    alpha: float,
    beta: float,
    top_k: int,
    student_model: str,
    teacher_model: str,
):
    """Aggregate per-token stats, score, classify, and output results."""
    from datetime import date

    phase2_dir = output_dir / "phase2_data"
    phase2_files = sorted(phase2_dir.glob("sample_*.pt"))
    if not phase2_files:
        raise FileNotFoundError(
            f"No Phase 2 data in {phase2_dir}. Run Phase 2 first."
        )

    # Load all per-token data
    all_token_ids = []
    all_kl_values = []
    all_teacher_entropies = []

    for filepath in phase2_files:
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        all_token_ids.append(data["token_ids"])
        all_kl_values.append(data["kl_values"])
        all_teacher_entropies.append(data["teacher_entropies"])

    all_token_ids = torch.cat(all_token_ids)
    all_kl_values = torch.cat(all_kl_values)
    all_teacher_entropies = torch.cat(all_teacher_entropies)

    total_positions = len(all_token_ids)

    # Aggregate by token ID
    unique_ids, frequencies, avg_kls, avg_entropies = aggregate_token_stats(
        all_token_ids, all_kl_values, all_teacher_entropies
    )

    # Score
    if scoring_method == "geometric":
        scores = geometric_score(frequencies.astype(float), avg_kls, alpha, beta)
    else:
        C = float(np.median(frequencies))
        mu = float(all_kl_values.mean().item())
        scores = bayesian_score(frequencies.astype(float), avg_kls, C=C, mu=mu)

    # Classify: global median entropy across ALL token positions
    global_median_entropy = float(all_teacher_entropies.median().item())
    classifications = classify_tokens(avg_entropies, global_median_entropy)

    # Sort by score descending, take top-k
    sorted_indices = np.argsort(scores)[::-1][:top_k]

    # Decode token strings
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rock_tokens = []
    for rank, idx in enumerate(sorted_indices, 1):
        tid = unique_ids[idx].item()
        rock_tokens.append(
            {
                "rank": rank,
                "token_id": tid,
                "token_string": tokenizer.decode([tid]),
                "frequency": int(frequencies[idx]),
                "avg_kl": round(float(avg_kls[idx]), 6),
                "rock_score": round(float(scores[idx]), 6),
                "avg_teacher_entropy": round(float(avg_entropies[idx]), 6),
                "classification": classifications[idx],
            }
        )

    # Build metadata
    metadata = {
        "student_model": student_model,
        "teacher_model": teacher_model,
        "dataset": "math500",
        "scoring_method": scoring_method,
        "entropy_threshold": round(global_median_entropy, 6),
        "top_k": top_k,
        "total_positions": total_positions,
        "unique_token_types": len(unique_ids),
        "date": str(date.today()),
    }
    if scoring_method == "geometric":
        metadata["alpha"] = alpha
        metadata["beta"] = beta

    # Output
    print_results_table(rock_tokens, global_median_entropy, metadata)
    save_rock_tokens_json(rock_tokens, metadata, output_dir / "rock_tokens.json")
    save_rock_tokens_csv(rock_tokens, output_dir / "rock_tokens.csv")
    print(f"\nResults saved to {output_dir / 'rock_tokens.json'} and {output_dir / 'rock_tokens.csv'}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/exp_2/test_analysis.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/exp_2/phases.py tests/exp_2/test_analysis.py
git commit -m "feat: implement Phase 3 — aggregation, scoring, classification, and output"
```

---

### Task 7: CLI Entry Point

**Files:**
- Create: `src/exp_2/identify_rock_tokens.py`

- [ ] **Step 1: Implement CLI with argparse and resume logic**

```python
# src/exp_2/identify_rock_tokens.py
"""Rock Token Identification Pipeline.

Three-phase pipeline to identify Rock Tokens from OPD-trained student/teacher models:
  Phase 1: Student generation — generate responses, save per-token log-probs
  Phase 2: Teacher forward — compute KL divergence and teacher entropy
  Phase 3: Analysis — aggregate, score, classify, output results

Only one model is loaded at a time for GPU efficiency.
Each phase saves intermediate results; use --phase to resume from a specific phase.
"""

import argparse
from pathlib import Path

from src.exp_2.phases import run_phase1, run_phase2, run_phase3


def determine_start_phase(output_dir: Path) -> int:
    """Determine which phase to resume from based on existing output files."""
    phase2_dir = output_dir / "phase2_data"
    student_data_dir = output_dir / "student_data"

    if phase2_dir.exists() and any(phase2_dir.glob("sample_*.pt")):
        return 3
    if student_data_dir.exists() and any(student_data_dir.glob("sample_*.pt")):
        return 2
    return 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify Rock Tokens from OPD-trained student/teacher models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--student",
        type=str,
        required=True,
        help="HuggingFace model ID for the OPD-trained student",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="HuggingFace model ID for the teacher model",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        choices=["geometric", "bayesian"],
        default="bayesian",
        help="Rock Token scoring method",
    )
    parser.add_argument("--alpha", type=float, default=0.3, help="Geometric: frequency exponent")
    parser.add_argument("--beta", type=float, default=0.7, help="Geometric: KL exponent")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top Rock Tokens to output")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per sample in Phase 1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/exp2",
        help="Directory for all outputs",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Force restart from this phase (default: auto-resume)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine start phase
    if args.phase is not None:
        start_phase = args.phase
        print(f"Forced start from Phase {start_phase}")
    else:
        start_phase = determine_start_phase(output_dir)
        if start_phase > 1:
            print(f"Auto-resuming from Phase {start_phase}")

    # Phase 1: Student generation
    if start_phase <= 1:
        run_phase1(args.student, output_dir, args.max_new_tokens)

    # Phase 2: Teacher KL computation
    if start_phase <= 2:
        run_phase2(args.teacher, output_dir)

    # Phase 3: Analysis and output
    run_phase3(
        output_dir=output_dir,
        tokenizer_name=args.student,
        scoring_method=args.scoring,
        alpha=args.alpha,
        beta=args.beta,
        top_k=args.top_k,
        student_model=args.student,
        teacher_model=args.teacher,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `uv run python src/exp_2/identify_rock_tokens.py --help`
Expected: help text with all arguments displayed

- [ ] **Step 3: Commit**

```bash
git add src/exp_2/identify_rock_tokens.py
git commit -m "feat: add CLI entry point with auto-resume and phase selection"
```

---

### Task 8: Makefile

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Create Makefile with convenience targets**

```makefile
# Makefile — Rock Token experiments

STUDENT ?= RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k
TEACHER ?= Qwen3/Qwen3-30B-A3B
OUTPUT  ?= results/exp2
TOP_K   ?= 50

# --- Exp 2: Rock Token Identification ---

.PHONY: exp2 exp2-geometric exp2-phase1 exp2-phase2 exp2-phase3 test lint

exp2:  ## Run full pipeline (bayesian scoring, default)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--scoring bayesian --top-k $(TOP_K) --output-dir $(OUTPUT)

exp2-geometric:  ## Run full pipeline with geometric scoring
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--scoring geometric --top-k $(TOP_K) --output-dir $(OUTPUT)

exp2-phase1:  ## Run only Phase 1 (student generation)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--phase 1 --output-dir $(OUTPUT)

exp2-phase2:  ## Run from Phase 2 (teacher KL)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--phase 2 --output-dir $(OUTPUT)

exp2-phase3:  ## Run only Phase 3 (re-analyze, no GPU needed)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--phase 3 --scoring bayesian --top-k $(TOP_K) --output-dir $(OUTPUT)

# --- Testing ---

test:  ## Run all tests
	uv run pytest tests/ -v

# --- Help ---

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
```

- [ ] **Step 2: Verify Makefile**

Run: `make help`
Expected: list of available targets

Run: `make test`
Expected: all 8 tests pass

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add Makefile with exp2 targets and test runner"
```

---

## Disk Space Note

Phase 1 saves full-vocabulary student log-probs per sample (~87MB each at 300 response tokens, 151k vocab, float16). For 500 samples, this is **~43GB temporary storage**. These files are consumed by Phase 2 and can be deleted afterward. Ensure sufficient disk space before running.

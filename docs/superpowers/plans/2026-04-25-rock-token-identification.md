# Rock Token Identification Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Part 1 pipeline that identifies the top 100 Rock Token types from a trained OPD student, with sanity-check plots and analysis.

**Architecture:** Four-phase pipeline — vLLM generation (Phase 1), two-pass HF forward passes for KL measurement (Phase 2), recalcitrance classification (Phase 3), and type aggregation (Phase 4). Config lives in `config.yaml`. All results go to `results/identification/`.

**Tech Stack:** vLLM (generation), PyTorch + HF Transformers (forward passes), matplotlib (plots), rich (console), PyYAML (config)

---

## File Structure

```
config.yaml                              # Model paths, hyperparameters
src/identification/
├── __init__.py                          # Module init
├── config.py                            # Load config.yaml
├── prompts.py                           # Load & mix datasets
├── generate.py                          # Phase 1: vLLM generation
├── measure.py                           # Phase 2: two-pass KL measurement
├── identify.py                          # Phase 3+4: classify + aggregate
├── plots.py                             # Sanity check plots
└── run.py                               # CLI orchestrator
tests/identification/
├── __init__.py
├── test_config.py
├── test_measure.py                      # KL computation math
└── test_identify.py                     # Classification + aggregation logic
```

---

### Task 1: Config loading

**Files:**
- Create: `config.yaml`
- Create: `src/identification/__init__.py`
- Create: `src/identification/config.py`
- Create: `tests/identification/__init__.py`
- Create: `tests/identification/test_config.py`
- Modify: `pyproject.toml` (add `pyyaml`)

- [ ] **Step 1: Add pyyaml dependency**

```bash
uv add pyyaml
```

- [ ] **Step 2: Create config.yaml at project root**

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
  tau_high_percentile: 80
  delta_percentile: 20
  min_frequency: 10
  top_k: 100

compute:
  tensor_parallel_size: 2
  max_model_len: 32768
  seed: 42
```

- [ ] **Step 3: Create src/identification/__init__.py**

```python
# Rock Token identification pipeline (Part 1).
```

- [ ] **Step 4: Write the failing test for config loading**

Create `tests/identification/__init__.py` (empty) and `tests/identification/test_config.py`:

```python
import tempfile
from pathlib import Path

import yaml

from src.identification.config import load_config


def test_load_config_from_path():
    data = {
        "models": {"teacher": "test-teacher"},
        "generation": {"temperature": 1.0},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        tmp_path = f.name

    config = load_config(tmp_path)
    assert config["models"]["teacher"] == "test-teacher"
    assert config["generation"]["temperature"] == 1.0
    Path(tmp_path).unlink()


def test_load_default_config():
    config = load_config()
    assert "models" in config
    assert "teacher" in config["models"]
    assert "generation" in config
    assert "identification" in config
```

- [ ] **Step 5: Run test to verify it fails**

```bash
uv run pytest tests/identification/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.identification.config'`

- [ ] **Step 6: Implement config.py**

Create `src/identification/config.py`:

```python
"""Load pipeline configuration from config.yaml."""

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load and return the config dict from a YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 7: Run test to verify it passes**

```bash
uv run pytest tests/identification/test_config.py -v
```

Expected: 2 passed

- [ ] **Step 8: Commit**

```bash
git add config.yaml src/identification/ tests/identification/ pyproject.toml uv.lock
git commit -m "feat: add config.yaml and config loader for identification pipeline"
```

---

### Task 2: Prompt loader

**Files:**
- Create: `src/identification/prompts.py`

- [ ] **Step 1: Create prompts.py**

```python
"""Load and mix evaluation prompts from multiple datasets (train splits only).

All sources use train splits to avoid contamination with evaluation benchmarks
(GSM8K test, HumanEval test, MMLU test, IF-Eval).
"""

import random

from datasets import load_dataset
from rich.console import Console


def load_mixed_prompts(
    math_count: int = 500,
    gsm8k_count: int = 500,
    mbpp_count: int = 300,
    alpaca_count: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Load and mix prompts from MATH, GSM8K, MBPP, and Alpaca train splits.

    Returns a shuffled list of {"prompt": str, "source": str} dicts.
    """
    console = Console()
    rng = random.Random(seed)
    prompts: list[dict] = []

    # MATH train (competition math)
    console.print("Loading MATH train...")
    math_ds = load_dataset("hendrycks/competition_math", split="train")
    indices = rng.sample(range(len(math_ds)), min(math_count, len(math_ds)))
    for i in indices:
        prompts.append({"prompt": math_ds[i]["problem"], "source": "math"})
    console.print(f"  MATH: {len(indices)} prompts")

    # GSM8K train (grade-school math)
    console.print("Loading GSM8K train...")
    gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = rng.sample(range(len(gsm_ds)), min(gsm8k_count, len(gsm_ds)))
    for i in indices:
        prompts.append({"prompt": gsm_ds[i]["question"], "source": "gsm8k"})
    console.print(f"  GSM8K: {len(indices)} prompts")

    # MBPP train (code)
    console.print("Loading MBPP train...")
    mbpp_ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    indices = rng.sample(range(len(mbpp_ds)), min(mbpp_count, len(mbpp_ds)))
    for i in indices:
        prompts.append({"prompt": mbpp_ds[i]["text"], "source": "mbpp"})
    console.print(f"  MBPP: {len(indices)} prompts")

    # Alpaca (general instructions — no standard test split)
    console.print("Loading Alpaca...")
    alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train")
    indices = rng.sample(range(len(alpaca_ds)), min(alpaca_count, len(alpaca_ds)))
    for i in indices:
        prompts.append({"prompt": alpaca_ds[i]["instruction"], "source": "alpaca"})
    console.print(f"  Alpaca: {len(indices)} prompts")

    rng.shuffle(prompts)
    console.print(f"Total: {len(prompts)} mixed prompts")
    return prompts
```

- [ ] **Step 2: Commit**

```bash
git add src/identification/prompts.py
git commit -m "feat: add mixed prompt loader (MATH/GSM8K/MBPP/Alpaca train splits)"
```

---

### Task 3: KL computation helpers with tests

**Files:**
- Create: `src/identification/measure.py` (helper functions only — Phase 2 orchestration is Task 5)
- Create: `tests/identification/test_measure.py`

- [ ] **Step 1: Write failing tests for KL and entropy computation**

Create `tests/identification/test_measure.py`:

```python
import torch

from src.identification.measure import compute_kl_per_token, compute_entropy_per_token


def test_kl_identical_distributions_is_zero():
    """KL(P || P) = 0 for any distribution."""
    logits = torch.randn(5, 100)  # 5 positions, vocab 100
    kl = compute_kl_per_token(logits, logits)
    assert kl.shape == (5,)
    assert torch.allclose(kl, torch.zeros(5), atol=1e-5)


def test_kl_is_nonnegative():
    """KL divergence is always >= 0."""
    teacher_logits = torch.randn(10, 50)
    student_logits = torch.randn(10, 50)
    kl = compute_kl_per_token(teacher_logits, student_logits)
    assert (kl >= -1e-6).all(), f"Negative KL found: {kl.min()}"


def test_kl_peaked_teacher_high_kl():
    """When teacher is peaked and student is uniform, KL should be high."""
    teacher_logits = torch.zeros(1, 10)
    teacher_logits[0, 0] = 100.0  # very peaked at token 0

    student_logits = torch.zeros(1, 10)  # uniform

    kl = compute_kl_per_token(teacher_logits, student_logits)
    assert kl[0] > 1.0  # should be large


def test_entropy_uniform_is_log_vocab():
    """Entropy of uniform distribution = log(vocab_size)."""
    import math
    vocab = 100
    logits = torch.zeros(3, vocab)  # uniform
    entropy = compute_entropy_per_token(logits)
    expected = math.log(vocab)
    assert torch.allclose(entropy, torch.full((3,), expected), atol=1e-4)


def test_entropy_peaked_is_near_zero():
    """Entropy of a peaked distribution is near 0."""
    logits = torch.full((2, 50), -1000.0)
    logits[:, 0] = 100.0  # all mass on token 0
    entropy = compute_entropy_per_token(logits)
    assert (entropy < 0.01).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/identification/test_measure.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the helper functions in measure.py**

Create `src/identification/measure.py`:

```python
"""Phase 2: Compute per-token KL divergence at both student checkpoints.

Helper functions for KL and entropy are at module level (testable without GPU).
The full Phase 2 orchestration (run_phase2) loads models and is GPU-only.
"""

import torch


# ---------------------------------------------------------------------------
# Pure-math helpers (testable on CPU)
# ---------------------------------------------------------------------------

def compute_kl_per_token(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(teacher || student) per token position.

    Args:
        teacher_logits: (seq_len, vocab_size) raw logits from teacher
        student_logits: (seq_len, vocab_size) raw logits from student

    Returns:
        (seq_len,) tensor of KL divergence values per position.
    """
    teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()

    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return kl


def compute_entropy_per_token(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of the distribution at each position.

    Args:
        logits: (seq_len, vocab_size) raw logits

    Returns:
        (seq_len,) tensor of entropy values.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/identification/test_measure.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/identification/measure.py tests/identification/test_measure.py
git commit -m "feat: add KL divergence and entropy helpers with tests"
```

---

### Task 4: Phase 1 — vLLM generation

**Files:**
- Create: `src/identification/generate.py`

- [ ] **Step 1: Create generate.py**

```python
"""Phase 1: Generate student outputs using vLLM.

Loads the on-policy distilled student (θ*) via vLLM, generates multiple
sampled outputs per prompt at temperature=1.0, and saves all sequences
with metadata for Phase 2 forward passes.
"""

from pathlib import Path

import torch
from rich.console import Console

from src.identification.prompts import load_mixed_prompts


def run_phase1(config: dict, output_dir: Path) -> Path:
    """Generate student outputs and save sequences to disk.

    Returns path to the saved sequences file.
    """
    from vllm import LLM, SamplingParams

    console = Console()
    output_path = output_dir / "phase1_sequences.pt"

    if output_path.exists():
        data = torch.load(output_path, weights_only=False)
        console.print(
            f"[yellow]Phase 1 already done ({len(data)} sequences at {output_path})[/yellow]"
        )
        return output_path

    # Load prompts
    pc = config["prompts"]
    prompts = load_mixed_prompts(
        math_count=pc["math_train_count"],
        gsm8k_count=pc["gsm8k_train_count"],
        mbpp_count=pc["mbpp_train_count"],
        alpaca_count=pc["alpaca_count"],
        seed=pc["seed"],
    )

    # Load model
    gc = config["generation"]
    cc = config["compute"]
    model_name = config["models"]["student_onpolicy"]

    console.print(f"Loading [bold]{model_name}[/bold] via vLLM (TP={cc['tensor_parallel_size']})...")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=cc["tensor_parallel_size"],
        trust_remote_code=True,
        seed=cc["seed"],
        max_model_len=cc["max_model_len"],
    )

    sampling = SamplingParams(
        temperature=gc["temperature"],
        max_tokens=gc["max_new_tokens"],
        seed=cc["seed"],
        n=gc["num_outputs_per_prompt"],
    )

    # Build conversations
    conversations = []
    for p in prompts:
        conversations.append([{"role": "user", "content": p["prompt"]}])

    console.print(
        f"Generating {gc['num_outputs_per_prompt']} outputs × {len(prompts)} prompts "
        f"(temp={gc['temperature']})..."
    )
    outputs = llm.chat(conversations, sampling)

    # Collect results
    sequences = []
    for prompt_idx, output in enumerate(outputs):
        source = prompts[prompt_idx]["source"]
        prompt_ids = list(output.prompt_token_ids)
        for completion in output.outputs:
            sequences.append({
                "prompt_idx": prompt_idx,
                "source": source,
                "prompt_token_ids": prompt_ids,
                "output_token_ids": list(completion.token_ids),
            })

    console.print(f"Generated {len(sequences)} total sequences")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sequences, output_path)
    console.print(f"Saved to {output_path}")

    # Free GPU
    del llm
    torch.cuda.empty_cache()

    return output_path
```

- [ ] **Step 2: Commit**

```bash
git add src/identification/generate.py
git commit -m "feat: add Phase 1 vLLM generation (temp=1.0, 3 outputs/prompt)"
```

---

### Task 5: Phase 2 — Two-pass KL measurement

**Files:**
- Modify: `src/identification/measure.py` (add `run_phase2`)

- [ ] **Step 1: Add run_phase2 and model-loading helpers to measure.py**

Append to `src/identification/measure.py`:

```python
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from transformers import AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_hf_model(model_name: str):
    """Load a HuggingFace causal LM with auto device mapping (bfloat16)."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )


def forward_logits(model, token_ids: list[int]) -> torch.Tensor:
    """Run a single forward pass and return logits on CPU.

    Args:
        model: HuggingFace causal LM (possibly sharded across GPUs)
        token_ids: full sequence (prompt + response) as a flat list

    Returns:
        (seq_len, vocab_size) float32 logits on CPU.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    return outputs.logits[0].float().cpu()


def measure_one_sequence(
    teacher_model,
    student_model,
    prompt_ids: list[int],
    output_ids: list[int],
) -> dict:
    """Compute per-token KL and entropy for one sequence.

    Returns dict with 1-D tensors for response positions only:
        token_ids, kl_values, teacher_entropy, student_entropy
    """
    full_ids = prompt_ids + output_ids
    prompt_len = len(prompt_ids)
    response_len = len(output_ids)

    if response_len == 0:
        return None

    teacher_logits = forward_logits(teacher_model, full_ids)
    student_logits = forward_logits(student_model, full_ids)

    # Slice to response positions:
    # logits[t] predicts token at t+1, so logits[prompt_len-1] predicts first response token
    t_resp = teacher_logits[prompt_len - 1 : prompt_len - 1 + response_len]
    s_resp = student_logits[prompt_len - 1 : prompt_len - 1 + response_len]

    kl_values = compute_kl_per_token(t_resp, s_resp)
    t_entropy = compute_entropy_per_token(t_resp)
    s_entropy = compute_entropy_per_token(s_resp)

    token_ids = torch.tensor(output_ids[:response_len], dtype=torch.long)

    return {
        "token_ids": token_ids,
        "kl_values": kl_values,
        "teacher_entropy": t_entropy,
        "student_entropy": s_entropy,
    }


# ---------------------------------------------------------------------------
# Phase 2 orchestration
# ---------------------------------------------------------------------------

def run_phase2(config: dict, output_dir: Path) -> Path:
    """Two-pass KL measurement: teacher + θ₀, then teacher + θ*.

    The teacher stays loaded across both passes. Only the student is swapped.
    No intermediate distributions saved to disk — only scalar KL values.

    Returns path to phase2_losses.pt.
    """
    console = Console()
    output_path = output_dir / "phase2_losses.pt"

    if output_path.exists():
        console.print(f"[yellow]Phase 2 already done ({output_path})[/yellow]")
        return output_path

    # Load Phase 1 sequences
    phase1_path = output_dir / "phase1_sequences.pt"
    sequences = torch.load(phase1_path, weights_only=False)
    n_seq = len(sequences)
    console.print(f"Loaded {n_seq} sequences from Phase 1")

    # --- Load teacher (stays loaded for both passes) ---
    console.print(f"Loading teacher: [bold]{config['models']['teacher']}[/bold]...")
    teacher = load_hf_model(config["models"]["teacher"])

    # === Pass 1: Teacher + θ₀ → loss_before ===
    console.print(f"Loading θ₀: [bold]{config['models']['student_base']}[/bold]...")
    student_before = load_hf_model(config["models"]["student_base"])

    loss_before_all = []
    teacher_entropy_all = []
    student_entropy_before_all = []
    token_ids_all = []
    source_all = []
    seq_idx_all = []

    console.print("[bold cyan]Pass 1: Teacher + θ₀ → loss_before[/bold cyan]")
    with Progress() as progress:
        task = progress.add_task("Pass 1", total=n_seq)
        for i, seq in enumerate(sequences):
            result = measure_one_sequence(
                teacher, student_before,
                seq["prompt_token_ids"], seq["output_token_ids"],
            )
            if result is None:
                progress.advance(task)
                continue

            resp_len = len(result["token_ids"])
            loss_before_all.append(result["kl_values"])
            teacher_entropy_all.append(result["teacher_entropy"])
            student_entropy_before_all.append(result["student_entropy"])
            token_ids_all.append(result["token_ids"])
            source_all.extend([seq["source"]] * resp_len)
            seq_idx_all.extend([i] * resp_len)
            progress.advance(task)

    # Free θ₀
    del student_before
    torch.cuda.empty_cache()

    # === Pass 2: Teacher + θ* → loss_after ===
    console.print(f"Loading θ*: [bold]{config['models']['student_onpolicy']}[/bold]...")
    student_after = load_hf_model(config["models"]["student_onpolicy"])

    loss_after_all = []
    student_entropy_after_all = []

    console.print("[bold cyan]Pass 2: Teacher + θ* → loss_after[/bold cyan]")
    with Progress() as progress:
        task = progress.add_task("Pass 2", total=n_seq)
        for i, seq in enumerate(sequences):
            result = measure_one_sequence(
                teacher, student_after,
                seq["prompt_token_ids"], seq["output_token_ids"],
            )
            if result is None:
                progress.advance(task)
                continue

            loss_after_all.append(result["kl_values"])
            student_entropy_after_all.append(result["student_entropy"])
            progress.advance(task)

    # Free all models
    del student_after, teacher
    torch.cuda.empty_cache()

    # Concatenate
    data = {
        "token_ids": torch.cat(token_ids_all),
        "loss_before": torch.cat(loss_before_all),
        "loss_after": torch.cat(loss_after_all),
        "teacher_entropy": torch.cat(teacher_entropy_all),
        "student_entropy_before": torch.cat(student_entropy_before_all),
        "student_entropy_after": torch.cat(student_entropy_after_all),
        "source_datasets": source_all,
        "sequence_indices": seq_idx_all,
    }

    n_tokens = len(data["token_ids"])
    console.print(f"Total token positions measured: {n_tokens:,}")

    torch.save(data, output_path)
    console.print(f"Saved to {output_path}")
    return output_path
```

- [ ] **Step 2: Commit**

```bash
git add src/identification/measure.py
git commit -m "feat: add Phase 2 two-pass KL measurement (teacher stays loaded)"
```

---

### Task 6: Classification and aggregation with tests

**Files:**
- Create: `src/identification/identify.py`
- Create: `tests/identification/test_identify.py`

- [ ] **Step 1: Write failing tests**

Create `tests/identification/test_identify.py`:

```python
import torch

from src.identification.identify import classify_rock_tokens, aggregate_to_types


def test_classify_high_before_low_improvement_is_rock():
    """Token with high loss_before and barely any improvement → Rock."""
    loss_before = torch.tensor([10.0, 10.0, 1.0, 1.0, 10.0])
    loss_after = torch.tensor([9.5, 2.0, 0.5, 0.8, 9.8])
    # improvement = loss_before - loss_after = [0.5, 8.0, 0.5, 0.2, 0.2]
    # tau_high_pct=60 → threshold at 60th pct of loss_before
    # sorted loss_before: [1.0, 1.0, 10.0, 10.0, 10.0] → 60th pct ≈ 10.0
    # delta_pct=40 → threshold at 40th pct of improvement
    # sorted improvement: [0.2, 0.2, 0.5, 0.5, 8.0] → 40th pct ≈ 0.38
    is_rock = classify_rock_tokens(loss_before, loss_after, tau_high_pct=60, delta_pct=40)
    # Indices 0 (lb=10, imp=0.5→>0.38, but lb>tau), 4 (lb=10, imp=0.2<0.38) → Rock
    # Actually let me re-check: Rock = loss_before > tau AND improvement < delta
    # tau = 60th pct of loss_before ≈ 10.0 (quantile). With 5 values, 60th pct:
    # torch.quantile([1,1,10,10,10], 0.6) = 10.0
    # delta = 40th pct of improvement:
    # torch.quantile([0.2, 0.2, 0.5, 0.5, 8.0], 0.4) ≈ 0.32
    # Rock: loss_before > 10 → none strictly > 10 (all are = 10)
    # Need loss_before >= tau. Let me use >= for the test.
    # With >= : indices 0,1,4 have lb>=10
    # Of those: improvement 0.5, 8.0, 0.2 — only 0.2 < 0.32 → index 4 is Rock
    assert is_rock[4].item() is True
    assert is_rock[1].item() is False  # high improvement (Learned)
    assert is_rock[2].item() is False  # low loss_before (Easy)


def test_classify_all_easy():
    """All tokens with low loss → none are Rock."""
    loss_before = torch.tensor([0.1, 0.2, 0.3])
    loss_after = torch.tensor([0.05, 0.1, 0.15])
    is_rock = classify_rock_tokens(loss_before, loss_after, tau_high_pct=80, delta_pct=20)
    # tau_high at 80th pct of [0.1, 0.2, 0.3] ≈ 0.28
    # All tokens below or near threshold, at most 1 barely above → at most 1 Rock
    # The key: with continuous quantile, 80th pct of 3 values might let 1 through
    # but for this test the point is most are NOT rock
    assert is_rock.sum().item() <= 1


def test_aggregate_frequency_filter():
    """Tokens appearing fewer than min_frequency times are excluded."""
    token_ids = torch.tensor([1, 1, 1, 2, 2, 2, 2, 2, 3])
    is_rock = torch.tensor([True, True, True, True, False, False, False, False, True])
    # token 1: 3 total, 3 rock → rate 1.0 — but freq < 5
    # token 2: 5 total, 1 rock → rate 0.2 — freq >= 5
    # token 3: 1 total — freq < 5
    result = aggregate_to_types(token_ids, is_rock, min_frequency=5, top_k=10)
    assert len(result) == 1
    assert result[0]["token_id"] == 2
    assert abs(result[0]["rock_rate"] - 0.2) < 1e-6


def test_aggregate_top_k_ranking():
    """Top-K returns tokens sorted by rock_rate descending."""
    token_ids = torch.tensor([1] * 20 + [2] * 20 + [3] * 20)
    is_rock = torch.tensor(
        [True] * 18 + [False] * 2  # token 1: rate 0.9
        + [True] * 10 + [False] * 10  # token 2: rate 0.5
        + [True] * 4 + [False] * 16  # token 3: rate 0.2
    )
    result = aggregate_to_types(token_ids, is_rock, min_frequency=10, top_k=2)
    assert len(result) == 2
    assert result[0]["token_id"] == 1  # highest rate
    assert result[1]["token_id"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/identification/test_identify.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement identify.py**

Create `src/identification/identify.py`:

```python
"""Phase 3+4: Apply recalcitrance criterion and aggregate to token types."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table


# ---------------------------------------------------------------------------
# Phase 3: Classification
# ---------------------------------------------------------------------------

def classify_rock_tokens(
    loss_before: torch.Tensor,
    loss_after: torch.Tensor,
    tau_high_pct: float = 80,
    delta_pct: float = 20,
) -> torch.Tensor:
    """Classify token instances as Rock using the relative threshold.

    A token is Rock if:
      - loss_before >= tau_high  (was hard before training)
      - (loss_before - loss_after) < delta  (barely improved)

    Args:
        loss_before: (N,) KL with pre-OPD student
        loss_after:  (N,) KL with post-OPD student
        tau_high_pct: percentile of loss_before for the "hard" threshold
        delta_pct: percentile of improvement for the "barely improved" threshold

    Returns:
        (N,) boolean tensor — True for Rock token instances.
    """
    tau_high = torch.quantile(loss_before.float(), tau_high_pct / 100)
    improvement = loss_before - loss_after
    delta = torch.quantile(improvement.float(), delta_pct / 100)

    is_rock = (loss_before >= tau_high) & (improvement < delta)
    return is_rock


# ---------------------------------------------------------------------------
# Phase 4: Aggregation
# ---------------------------------------------------------------------------

def aggregate_to_types(
    token_ids: torch.Tensor,
    is_rock: torch.Tensor,
    min_frequency: int = 10,
    top_k: int = 100,
    loss_before: torch.Tensor | None = None,
    loss_after: torch.Tensor | None = None,
    teacher_entropy: torch.Tensor | None = None,
    student_entropy_before: torch.Tensor | None = None,
    student_entropy_after: torch.Tensor | None = None,
) -> list[dict]:
    """Aggregate per-instance Rock labels to per-type rankings.

    Groups by token_id, applies frequency filter, ranks by rock_rate.

    Returns list of dicts sorted by rock_rate descending, length <= top_k.
    """
    # Accumulate stats per token type
    stats: dict[int, dict] = defaultdict(lambda: {
        "total": 0, "rock": 0,
        "loss_before_sum": 0.0, "loss_after_sum": 0.0,
        "teacher_entropy_sum": 0.0,
        "student_entropy_before_sum": 0.0, "student_entropy_after_sum": 0.0,
    })

    for i in range(len(token_ids)):
        tid = token_ids[i].item()
        s = stats[tid]
        s["total"] += 1
        if is_rock[i]:
            s["rock"] += 1
        if loss_before is not None:
            s["loss_before_sum"] += loss_before[i].item()
        if loss_after is not None:
            s["loss_after_sum"] += loss_after[i].item()
        if teacher_entropy is not None:
            s["teacher_entropy_sum"] += teacher_entropy[i].item()
        if student_entropy_before is not None:
            s["student_entropy_before_sum"] += student_entropy_before[i].item()
        if student_entropy_after is not None:
            s["student_entropy_after_sum"] += student_entropy_after[i].item()

    # Filter by frequency and compute rates
    results = []
    for tid, s in stats.items():
        if s["total"] < min_frequency:
            continue
        n = s["total"]
        entry = {
            "token_id": tid,
            "frequency": n,
            "rock_count": s["rock"],
            "rock_rate": s["rock"] / n,
        }
        if loss_before is not None:
            entry["avg_loss_before"] = s["loss_before_sum"] / n
            entry["avg_loss_after"] = s["loss_after_sum"] / n
            entry["avg_improvement"] = entry["avg_loss_before"] - entry["avg_loss_after"]
        if teacher_entropy is not None:
            entry["avg_teacher_entropy"] = s["teacher_entropy_sum"] / n
        if student_entropy_before is not None:
            entry["avg_student_entropy_before"] = s["student_entropy_before_sum"] / n
        if student_entropy_after is not None:
            entry["avg_student_entropy_after"] = s["student_entropy_after_sum"] / n
        results.append(entry)

    # Sort by rock_rate descending, take top_k
    results.sort(key=lambda x: x["rock_rate"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Phase 3+4 orchestration
# ---------------------------------------------------------------------------

def run_identification(config: dict, output_dir: Path) -> Path:
    """Load Phase 2 data, classify, aggregate, and save top-K Rock Tokens."""
    console = Console()
    output_json = output_dir / "rock_tokens.json"

    # Load Phase 2 data
    phase2_path = output_dir / "phase2_losses.pt"
    data = torch.load(phase2_path, weights_only=False)

    ic = config["identification"]
    console.print(
        f"Classifying with τ_high={ic['tau_high_percentile']}th pct, "
        f"δ={ic['delta_percentile']}th pct..."
    )

    is_rock = classify_rock_tokens(
        data["loss_before"], data["loss_after"],
        tau_high_pct=ic["tau_high_percentile"],
        delta_pct=ic["delta_percentile"],
    )

    n_total = len(is_rock)
    n_rock = is_rock.sum().item()
    rock_frac = n_rock / n_total
    console.print(f"Rock instances: {n_rock:,} / {n_total:,} ({rock_frac:.1%})")

    if rock_frac < 0.05 or rock_frac > 0.20:
        console.print(
            f"[bold yellow]WARNING: Rock fraction {rock_frac:.1%} is outside "
            f"expected 5-20% range. Consider adjusting thresholds.[/bold yellow]"
        )

    # Aggregate
    results = aggregate_to_types(
        data["token_ids"], is_rock,
        min_frequency=ic["min_frequency"],
        top_k=ic["top_k"],
        loss_before=data["loss_before"],
        loss_after=data["loss_after"],
        teacher_entropy=data["teacher_entropy"],
        student_entropy_before=data["student_entropy_before"],
        student_entropy_after=data["student_entropy_after"],
    )

    # Add rank and decode token strings
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student_onpolicy"], trust_remote_code=True
    )
    for rank, entry in enumerate(results, 1):
        entry["rank"] = rank
        entry["token_string"] = tokenizer.decode([entry["token_id"]])

    # Print table
    table = Table(title=f"Top {len(results)} Rock Tokens")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Token", style="cyan")
    table.add_column("ID", justify="right")
    table.add_column("Freq", justify="right")
    table.add_column("Rock#", justify="right")
    table.add_column("Rate", justify="right", style="bold")
    table.add_column("Avg KL₀", justify="right")
    table.add_column("Avg KL*", justify="right")

    for entry in results[:30]:  # show top 30 in console
        table.add_row(
            str(entry["rank"]),
            repr(entry["token_string"]),
            str(entry["token_id"]),
            str(entry["frequency"]),
            str(entry["rock_count"]),
            f"{entry['rock_rate']:.3f}",
            f"{entry.get('avg_loss_before', 0):.3f}",
            f"{entry.get('avg_loss_after', 0):.3f}",
        )
    console.print(table)

    # Save JSON
    thresholds = {
        "tau_high_percentile": ic["tau_high_percentile"],
        "delta_percentile": ic["delta_percentile"],
        "tau_high_value": float(torch.quantile(
            data["loss_before"].float(), ic["tau_high_percentile"] / 100
        )),
        "delta_value": float(torch.quantile(
            (data["loss_before"] - data["loss_after"]).float(),
            ic["delta_percentile"] / 100,
        )),
    }

    output_data = {
        "metadata": {
            "models": config["models"],
            "total_token_positions": n_total,
            "rock_instances": n_rock,
            "rock_fraction": rock_frac,
            "thresholds": thresholds,
            "min_frequency": ic["min_frequency"],
            "top_k": ic["top_k"],
        },
        "rock_tokens": results,
    }

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    console.print(f"Saved {len(results)} Rock Tokens to {output_json}")

    # Also save CSV
    import csv
    csv_path = output_dir / "rock_tokens.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    console.print(f"Saved CSV to {csv_path}")

    return output_json
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/identification/test_identify.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/identification/identify.py tests/identification/test_identify.py
git commit -m "feat: add Rock Token classification and type aggregation with tests"
```

---

### Task 7: Sanity check plots

**Files:**
- Create: `src/identification/plots.py`

- [ ] **Step 1: Create plots.py**

```python
"""Sanity check plots for Rock Token identification.

All plots are saved as PNG files in the output directory.
Can be run on CPU from Phase 2 data — no models needed.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from scipy import stats as scipy_stats


def plot_loss_scatter(data: dict, config: dict, output_dir: Path) -> Path:
    """Scatter plot of loss_before vs loss_after with color-coded quadrants."""
    ic = config["identification"]
    lb = data["loss_before"].numpy()
    la = data["loss_after"].numpy()

    tau_high = float(np.percentile(lb, ic["tau_high_percentile"]))
    improvement = lb - la
    delta = float(np.percentile(improvement, ic["delta_percentile"]))
    # delta_line: loss_after = loss_before - delta
    # Rock region: loss_before >= tau_high AND loss_after >= loss_before - delta

    # Subsample for plotting if too many points
    n = len(lb)
    if n > 50_000:
        idx = np.random.default_rng(42).choice(n, 50_000, replace=False)
        lb_plot, la_plot = lb[idx], la[idx]
    else:
        lb_plot, la_plot = lb, la

    # Assign quadrant colors
    colors = np.full(len(lb_plot), "#2ecc71", dtype=object)  # Easy (green)
    for i in range(len(lb_plot)):
        high_before = lb_plot[i] >= tau_high
        low_improve = (lb_plot[i] - la_plot[i]) < delta
        if high_before and not low_improve:
            colors[i] = "#3498db"  # Learned (blue)
        elif high_before and low_improve:
            colors[i] = "#e74c3c"  # Rock (red)
        elif not high_before and la_plot[i] > lb_plot[i]:
            colors[i] = "#e67e22"  # Regressed (orange)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(lb_plot, la_plot, c=colors, alpha=0.15, s=3, rasterized=True)

    # Threshold lines
    ax.axvline(tau_high, color="gray", linestyle="--", linewidth=0.8, label=f"τ_high={tau_high:.2f}")
    # δ boundary line: la = lb - delta
    x_line = np.linspace(tau_high, lb.max(), 100)
    ax.plot(x_line, x_line - delta, color="gray", linestyle=":", linewidth=0.8, label=f"δ={delta:.2f}")
    # Identity line
    lim = max(lb.max(), la.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("loss_before (KL with θ₀)")
    ax.set_ylabel("loss_after (KL with θ*)")
    ax.set_title("Per-Token Loss: Before vs After OPD")
    ax.legend(fontsize=8)

    # Quadrant labels
    mid_hi = (tau_high + lb.max()) / 2
    mid_lo = tau_high / 2
    ax.text(mid_lo, mid_lo * 0.5, "Easy", color="#2ecc71", fontsize=12, fontweight="bold", alpha=0.7)
    ax.text(mid_hi, mid_lo * 0.5, "Learned", color="#3498db", fontsize=12, fontweight="bold", alpha=0.7)
    ax.text(mid_hi, mid_hi, "Rock", color="#e74c3c", fontsize=12, fontweight="bold", alpha=0.7)

    plt.tight_layout()
    path = output_dir / "scatter_loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_entropy_correlation(
    rock_tokens: list[dict], output_dir: Path
) -> Path:
    """Rock rate vs student entropy per token type — tests forking token hypothesis."""
    rates = [t["rock_rate"] for t in rock_tokens if "avg_student_entropy_before" in t]
    entropies = [t["avg_student_entropy_before"] for t in rock_tokens if "avg_student_entropy_before" in t]

    if not rates:
        return None

    rates = np.array(rates)
    entropies = np.array(entropies)

    r, p_val = scipy_stats.pearsonr(entropies, rates)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(entropies, rates, alpha=0.5, s=20, color="#4C72B0")

    # Trend line
    z = np.polyfit(entropies, rates, 1)
    x_line = np.linspace(entropies.min(), entropies.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="red", linestyle="--", linewidth=1)

    ax.set_xlabel("Avg Student Entropy (θ₀)")
    ax.set_ylabel("Rock Rate")
    ax.set_title(f"Rock Rate vs Student Entropy (r={r:.3f}, p={p_val:.2e})")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "entropy_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_plots(config: dict, output_dir: Path) -> None:
    """Generate all sanity check plots from Phase 2 data."""
    console = Console()

    # Load data
    phase2_path = output_dir / "phase2_losses.pt"
    data = torch.load(phase2_path, weights_only=False)

    rock_json = output_dir / "rock_tokens.json"
    import json
    with open(rock_json) as f:
        rock_data = json.load(f)

    # 1. Scatter plot
    path = plot_loss_scatter(data, config, output_dir)
    console.print(f"Scatter plot saved to {path}")

    # 2. Rock fraction report
    n_total = len(data["token_ids"])
    ic = config["identification"]
    is_rock = (
        (data["loss_before"] >= torch.quantile(data["loss_before"].float(), ic["tau_high_percentile"] / 100))
        & ((data["loss_before"] - data["loss_after"]) < torch.quantile(
            (data["loss_before"] - data["loss_after"]).float(), ic["delta_percentile"] / 100
        ))
    )
    n_rock = is_rock.sum().item()
    frac = n_rock / n_total

    frac_path = output_dir / "rock_fraction.txt"
    frac_text = (
        f"Total token positions: {n_total:,}\n"
        f"Rock instances: {n_rock:,}\n"
        f"Rock fraction: {frac:.4f} ({frac:.1%})\n"
        f"Status: {'OK' if 0.05 <= frac <= 0.20 else 'WARNING — outside 5-20% range'}\n"
    )
    frac_path.write_text(frac_text)
    console.print(f"Rock fraction: {frac:.1%}")
    console.print(f"  Written to {frac_path}")

    # 3. Entropy correlation
    rock_tokens = rock_data["rock_tokens"]
    path = plot_entropy_correlation(rock_tokens, output_dir)
    if path:
        console.print(f"Entropy correlation plot saved to {path}")

    console.print("[bold green]All plots generated.[/bold green]")
```

- [ ] **Step 2: Add scipy to dependencies**

```bash
uv add scipy
```

- [ ] **Step 3: Commit**

```bash
git add src/identification/plots.py pyproject.toml uv.lock
git commit -m "feat: add sanity check plots (scatter, rock fraction, entropy correlation)"
```

---

### Task 8: CLI orchestrator, Makefile, CHANGELOG

**Files:**
- Create: `src/identification/run.py`
- Modify: `Makefile`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Create run.py**

```python
"""CLI entry point: orchestrate the full Rock Token identification pipeline.

Usage:
    uv run python src/identification/run.py                  # full pipeline
    uv run python src/identification/run.py --phase 1        # generation only
    uv run python src/identification/run.py --phase 3        # classify + plots (CPU)
    uv run python src/identification/run.py --config my.yaml # custom config
"""

import argparse
from pathlib import Path

from rich.console import Console

from src.identification.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Rock Token identification pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: project root config.yaml)",
    )
    parser.add_argument(
        "--phase", type=int, default=None, choices=[1, 2, 3],
        help="Run only this phase (default: run all phases)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/identification",
        help="Output directory",
    )
    args = parser.parse_args()

    console = Console()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Rock Token Identification Pipeline[/bold]")
    console.print(f"  Config: {args.config or 'config.yaml (default)'}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Teacher: {config['models']['teacher']}")
    console.print(f"  θ₀: {config['models']['student_base']}")
    console.print(f"  θ*: {config['models']['student_onpolicy']}")
    console.print()

    phases = [args.phase] if args.phase else [1, 2, 3]

    if 1 in phases:
        console.print("[bold]═══ Phase 1: Generate Student Outputs ═══[/bold]")
        from src.identification.generate import run_phase1
        run_phase1(config, output_dir)

    if 2 in phases:
        console.print("[bold]═══ Phase 2: Measure KL at Both Checkpoints ═══[/bold]")
        from src.identification.measure import run_phase2
        run_phase2(config, output_dir)

    if 3 in phases:
        console.print("[bold]═══ Phase 3+4: Classify & Aggregate ═══[/bold]")
        from src.identification.identify import run_identification
        run_identification(config, output_dir)

        console.print("[bold]═══ Sanity Check Plots ═══[/bold]")
        from src.identification.plots import run_plots
        run_plots(config, output_dir)

    console.print("\n[bold green]Pipeline complete.[/bold green]")
    console.print(f"Results in {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add Makefile targets**

Add to `Makefile` after the eval-all section and before the Testing section:

```makefile
# --- Identification ---

IDENT_DIR ?= results/identification

identify:  ## Run full Rock Token identification pipeline
	uv run python src/identification/run.py --output-dir $(IDENT_DIR)

identify-phase1:  ## Run only Phase 1 (vLLM generation)
	uv run python src/identification/run.py --phase 1 --output-dir $(IDENT_DIR)

identify-phase2:  ## Run only Phase 2 (KL measurement, needs GPU)
	uv run python src/identification/run.py --phase 2 --output-dir $(IDENT_DIR)

identify-phase3:  ## Run Phase 3+4 + plots (CPU only)
	uv run python src/identification/run.py --phase 3 --output-dir $(IDENT_DIR)
```

Also add `identify identify-phase1 identify-phase2 identify-phase3` to the `.PHONY` line.

- [ ] **Step 3: Update CHANGELOG.md**

Add a new `[0.4.0]` entry at the top:

```markdown
## [0.4.0] - 2026-04-25

### Added
- **Rock Token identification pipeline** (`src/identification/`):
  - Phase 1: vLLM generation (temp=1.0, 3 outputs/prompt, 1500 mixed prompts)
  - Phase 2: Two-pass KL measurement (teacher stays loaded, fits 2×A100-80GB)
  - Phase 3+4: Recalcitrance criterion (relative threshold) + type aggregation (top 100)
  - Sanity check plots: loss scatter (4-quadrant), rock fraction, entropy correlation
- `config.yaml` for all model paths and hyperparameters
- Makefile targets: `identify`, `identify-phase1`, `identify-phase2`, `identify-phase3`
- `pyyaml` and `scipy` added to project dependencies
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass (existing + new config, measure, identify tests)

- [ ] **Step 5: Commit**

```bash
git add src/identification/run.py Makefile CHANGELOG.md
git commit -m "feat: add CLI orchestrator, Makefile targets, and CHANGELOG for identification pipeline"
```

"""Phase 2: Compute per-token KL divergence at both student checkpoints.

Helper functions for KL and entropy are at module level (testable without GPU).
The full Phase 2 orchestration (run_phase2) loads models and is GPU-only.
"""

import torch
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from transformers import AutoModelForCausalLM


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
) -> dict | None:
    """Compute per-token KL and entropy for one sequence.

    Returns dict with 1-D tensors for response positions only:
        token_ids, kl_values, teacher_entropy, student_entropy
    Returns None if response is empty.
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

    token_ids_tensor = torch.tensor(output_ids[:response_len], dtype=torch.long)

    return {
        "token_ids": token_ids_tensor,
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

    loss_before_all: list[torch.Tensor] = []
    teacher_entropy_all: list[torch.Tensor] = []
    student_entropy_before_all: list[torch.Tensor] = []
    token_ids_all: list[torch.Tensor] = []
    source_all: list[str] = []
    seq_idx_all: list[int] = []

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

    loss_after_all: list[torch.Tensor] = []
    student_entropy_after_all: list[torch.Tensor] = []

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

"""Phase 2: Compute per-token KL divergence at both student checkpoints.

Helper functions for KL and entropy are at module level (testable without GPU).
The full Phase 2 orchestration (run_phase2) loads models and is GPU-only.

Optimizations over naive single-sequence processing:
  1. Flash attention (if flash-attn is installed)
  2. Dynamic batching — groups sequences by length, pads minimally
  3. Single teacher pass — caches top-K teacher log-probs in RAM,
     reuses for both students (avoids running the teacher twice)
"""

import torch
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from transformers import AutoModelForCausalLM

# Top-K teacher log-probs cached in RAM for single-teacher-pass optimization.
# 256 covers >99.99% of probability mass for language models.
TEACHER_CACHE_TOP_K = 256

# Max total tokens across all sequences in a batch. Controls GPU memory usage.
# Lower = less memory, higher = faster. 16384 is safe for 2× A100-80GB.
MAX_BATCH_TOKENS = 16384


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


def compute_kl_from_cache(
    teacher_cache: dict,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute approximate KL(teacher || student) using cached top-K teacher log-probs.

    Args:
        teacher_cache: dict with "top_indices" (seq_len, K) and "top_log_probs" (seq_len, K)
        student_logits: (seq_len, vocab_size) raw logits from student

    Returns:
        (seq_len,) tensor of KL divergence values.
    """
    top_indices = teacher_cache["top_indices"]       # (seq_len, K)
    top_log_probs = teacher_cache["top_log_probs"]   # (seq_len, K)
    top_probs = top_log_probs.exp()                  # (seq_len, K)

    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    # Gather student log-probs at teacher's top-K positions
    student_at_top = student_log_probs.gather(1, top_indices)  # (seq_len, K)

    kl = (top_probs * (top_log_probs - student_at_top)).sum(dim=-1)
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

def load_hf_model(model_name: str, console: Console | None = None):
    """Load a HuggingFace causal LM with auto device mapping (bfloat16).

    Tries flash_attention_2 first; falls back to default if unavailable.
    """
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="flash_attention_2", **kwargs,
        )
        if console:
            console.print("  (using flash_attention_2)")
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if console:
            console.print("  (flash attention not available, using default)")
    return model


# ---------------------------------------------------------------------------
# Batched forward pass
# ---------------------------------------------------------------------------

def _build_batches(sequences: list[dict], max_batch_tokens: int) -> list[list[int]]:
    """Sort sequences by length and group into batches bounded by total tokens.

    Returns list of batches, each batch is a list of indices into sequences.
    """
    indexed = []
    for i, seq in enumerate(sequences):
        length = len(seq["prompt_token_ids"]) + len(seq["output_token_ids"])
        if length > 0:
            indexed.append((i, length))
    indexed.sort(key=lambda x: x[1])

    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_max_len = 0

    for idx, length in indexed:
        new_max = max(current_max_len, length)
        new_total = new_max * (len(current_batch) + 1)
        if current_batch and new_total > max_batch_tokens:
            batches.append(current_batch)
            current_batch = [idx]
            current_max_len = length
        else:
            current_batch.append(idx)
            current_max_len = new_max

    if current_batch:
        batches.append(current_batch)

    return batches


def forward_logits_batch(
    model, token_ids_list: list[list[int]],
) -> list[torch.Tensor]:
    """Batched forward pass with left-padding. Returns per-sequence logits on CPU.

    Each returned tensor has shape (seq_len, vocab_size) with padding removed.
    """
    if len(token_ids_list) == 1:
        # Fast path: skip padding for single sequence
        device = next(model.parameters()).device
        ids = torch.tensor([token_ids_list[0]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=ids)
        logits = out.logits[0].float().cpu()
        del out
        return [logits]

    max_len = max(len(ids) for ids in token_ids_list)
    batch_size = len(token_ids_list)

    # Left-pad for causal LM batched inference
    padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, ids in enumerate(token_ids_list):
        seq_len = len(ids)
        padded[i, max_len - seq_len:] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, max_len - seq_len:] = 1

    device = next(model.parameters()).device
    padded = padded.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = model(input_ids=padded, attention_mask=attention_mask)

    # Extract per-sequence logits (unpad)
    results = []
    for i, ids in enumerate(token_ids_list):
        seq_len = len(ids)
        results.append(out.logits[i, max_len - seq_len:].float().cpu())

    del out, padded, attention_mask
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Phase 2 orchestration (optimized)
# ---------------------------------------------------------------------------

def _run_teacher_pass(
    teacher, sequences: list[dict], batches: list[list[int]], console: Console,
) -> tuple[list, list, list]:
    """Run teacher on all sequences, cache top-K log-probs per response position.

    Returns (teacher_caches, teacher_entropies, token_ids_per_seq) — lists
    indexed by sequence index, with None for empty sequences.
    """
    n_seq = len(sequences)
    teacher_caches: list[dict | None] = [None] * n_seq
    teacher_entropies: list[torch.Tensor | None] = [None] * n_seq
    token_ids_per_seq: list[torch.Tensor | None] = [None] * n_seq

    console.print(f"[bold cyan]Teacher pass ({len(batches)} batches)[/bold cyan]")
    total_seqs = sum(len(b) for b in batches)
    with Progress() as progress:
        task = progress.add_task("Teacher", total=total_seqs)
        for batch_indices in batches:
            # Build full token id lists
            full_ids_list = []
            for idx in batch_indices:
                seq = sequences[idx]
                full_ids_list.append(seq["prompt_token_ids"] + seq["output_token_ids"])

            logits_list = forward_logits_batch(teacher, full_ids_list)

            for j, idx in enumerate(batch_indices):
                seq = sequences[idx]
                prompt_len = len(seq["prompt_token_ids"])
                response_len = len(seq["output_token_ids"])
                if response_len == 0:
                    progress.advance(task)
                    continue

                # Slice response logits
                resp_logits = logits_list[j][prompt_len - 1: prompt_len - 1 + response_len]

                # Cache top-K teacher log-probs (compressed — ~2KB per position)
                log_probs = torch.log_softmax(resp_logits, dim=-1)
                top_log_probs, top_indices = log_probs.topk(TEACHER_CACHE_TOP_K, dim=-1)
                teacher_caches[idx] = {
                    "top_indices": top_indices,
                    "top_log_probs": top_log_probs,
                }

                teacher_entropies[idx] = compute_entropy_per_token(resp_logits)
                token_ids_per_seq[idx] = torch.tensor(
                    seq["output_token_ids"][:response_len], dtype=torch.long,
                )
                progress.advance(task)

            # Free batch logits
            del logits_list

    return teacher_caches, teacher_entropies, token_ids_per_seq


def _run_student_pass(
    student, sequences: list[dict], batches: list[list[int]],
    teacher_caches: list, console: Console, label: str,
) -> tuple[list, list]:
    """Run student on all sequences, compute KL using cached teacher data.

    Returns (kl_values_per_seq, student_entropies_per_seq).
    """
    n_seq = len(sequences)
    kl_per_seq: list[torch.Tensor | None] = [None] * n_seq
    entropy_per_seq: list[torch.Tensor | None] = [None] * n_seq

    total_seqs = sum(len(b) for b in batches)
    with Progress() as progress:
        task = progress.add_task(label, total=total_seqs)
        for batch_indices in batches:
            full_ids_list = []
            for idx in batch_indices:
                seq = sequences[idx]
                full_ids_list.append(seq["prompt_token_ids"] + seq["output_token_ids"])

            logits_list = forward_logits_batch(student, full_ids_list)

            for j, idx in enumerate(batch_indices):
                seq = sequences[idx]
                prompt_len = len(seq["prompt_token_ids"])
                response_len = len(seq["output_token_ids"])
                if response_len == 0 or teacher_caches[idx] is None:
                    progress.advance(task)
                    continue

                resp_logits = logits_list[j][prompt_len - 1: prompt_len - 1 + response_len]

                kl_per_seq[idx] = compute_kl_from_cache(
                    teacher_caches[idx], resp_logits,
                )
                entropy_per_seq[idx] = compute_entropy_per_token(resp_logits)
                progress.advance(task)

            del logits_list

    return kl_per_seq, entropy_per_seq


def run_phase2(config: dict, output_dir: Path) -> Path:
    """Optimized Phase 2: single teacher pass + batched forward passes.

    1. Run teacher once on all sequences (batched), cache top-K log-probs in RAM.
    2. Run θ₀ on all sequences (batched), compute KL from cache → loss_before.
    3. Swap to θ*, repeat → loss_after.
    4. Save per-token scalars.

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

    # Build batches (sorted by length for minimal padding)
    batches = _build_batches(sequences, MAX_BATCH_TOKENS)
    console.print(
        f"Created {len(batches)} batches "
        f"(max {MAX_BATCH_TOKENS} tokens/batch, "
        f"batch sizes {min(len(b) for b in batches)}-{max(len(b) for b in batches)})"
    )

    # === Step 1: Teacher pass (run ONCE, cache top-K in RAM) ===
    console.print(f"Loading teacher: [bold]{config['models']['teacher']}[/bold]...")
    teacher = load_hf_model(config["models"]["teacher"], console)

    teacher_caches, teacher_entropies, token_ids_per_seq = _run_teacher_pass(
        teacher, sequences, batches, console,
    )

    # Estimate cache size
    cache_bytes = sum(
        c["top_indices"].nbytes + c["top_log_probs"].nbytes
        for c in teacher_caches if c is not None
    )
    console.print(f"Teacher cache: {cache_bytes / 1e9:.1f} GB in RAM")

    # Free teacher
    del teacher
    torch.cuda.empty_cache()

    # === Step 2: θ₀ → loss_before ===
    console.print(f"Loading θ₀: [bold]{config['models']['student_base']}[/bold]...")
    student_before = load_hf_model(config["models"]["student_base"], console)

    console.print("[bold cyan]Student θ₀ pass → loss_before[/bold cyan]")
    loss_before_per_seq, entropy_before_per_seq = _run_student_pass(
        student_before, sequences, batches, teacher_caches, console, "θ₀",
    )

    del student_before
    torch.cuda.empty_cache()

    # === Step 3: θ* → loss_after ===
    console.print(f"Loading θ*: [bold]{config['models']['student_onpolicy']}[/bold]...")
    student_after = load_hf_model(config["models"]["student_onpolicy"], console)

    console.print("[bold cyan]Student θ* pass → loss_after[/bold cyan]")
    loss_after_per_seq, entropy_after_per_seq = _run_student_pass(
        student_after, sequences, batches, teacher_caches, console, "θ*",
    )

    del student_after
    torch.cuda.empty_cache()

    # Free teacher cache
    del teacher_caches

    # === Assemble results ===
    token_ids_all = []
    loss_before_all = []
    loss_after_all = []
    teacher_entropy_all = []
    student_entropy_before_all = []
    student_entropy_after_all = []
    source_all: list[str] = []
    seq_idx_all: list[int] = []

    for i in range(n_seq):
        if token_ids_per_seq[i] is None:
            continue
        resp_len = len(token_ids_per_seq[i])
        token_ids_all.append(token_ids_per_seq[i])
        loss_before_all.append(loss_before_per_seq[i])
        loss_after_all.append(loss_after_per_seq[i])
        teacher_entropy_all.append(teacher_entropies[i])
        student_entropy_before_all.append(entropy_before_per_seq[i])
        student_entropy_after_all.append(entropy_after_per_seq[i])
        source_all.extend([sequences[i]["source"]] * resp_len)
        seq_idx_all.extend([i] * resp_len)

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

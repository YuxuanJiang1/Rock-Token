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


def run_phase1(
    student_model_name: str,
    output_dir: Path,
    max_new_tokens: int = 2048,
):
    """Generate student responses and save per-token log-probs.

    Single-sample greedy generation with output_scores to capture full-vocab
    logits at each step. Saves one .pt file per sample for mid-phase resume.
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
            attention_mask = inputs.attention_mask.to(model.device)
            prompt_length = input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            full_ids = outputs.sequences[0].cpu()
            gen_len = len(outputs.scores)

            if gen_len == 0:
                print(f"  Sample {i}: no response generated, skipping")
                progress.advance(task)
                continue

            scores = torch.stack(
                [s.squeeze(0) for s in outputs.scores], dim=0
            )  # (gen_len, vocab)
            log_probs = torch.log_softmax(scores.float(), dim=-1).cpu().to(torch.bfloat16)
            del scores, outputs

            torch.save(
                {
                    "sample_idx": i,
                    "prompt_length": prompt_length,
                    "full_ids": full_ids,
                    "student_log_probs": log_probs,
                },
                student_data_dir / f"sample_{i:03d}.pt",
            )
            print(f"  Sample {i}: {gen_len} response tokens")
            progress.advance(task)

    del model
    torch.cuda.empty_cache()
    print("Phase 1 complete")


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
            data = torch.load(filepath, map_location="cpu", weights_only=True)
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

            # Response token IDs (slice to response_len to match kl_values length)
            token_ids = full_ids[prompt_length : prompt_length + response_len]

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
        data = torch.load(filepath, map_location="cpu", weights_only=True)
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

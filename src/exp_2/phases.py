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

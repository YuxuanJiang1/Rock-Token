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

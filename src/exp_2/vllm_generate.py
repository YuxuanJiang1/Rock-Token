"""vLLM-based generation for Phase 1.

Uses vLLM for fast batched generation (avoids HF left-padding + RoPE issues),
then HF forward pass per sample to extract full-vocabulary log-probs.

vLLM runs in a subprocess to avoid CUDA context corruption — vLLM's engine
spawns its own CUDA processes, and after cleanup the parent process's CUDA
state can be corrupted (CUBLAS_STATUS_NOT_SUPPORTED errors). Running vLLM
in isolation ensures a clean CUDA context for the HF forward pass.
"""

import gc
import json
import subprocess
import sys
from pathlib import Path

import torch
from rich.progress import Progress

from src.exp_2.utils import load_math500, load_model_and_tokenizer


def vllm_generate(
    model_name: str,
    output_dir: Path,
    max_new_tokens: int = 2048,
    tensor_parallel_size: int | None = None,
) -> None:
    """Generate student responses via vLLM + extract log-probs via HF forward pass.

    Step 1: Run vLLM in a subprocess to generate responses and save token IDs.
    Step 2: HF forward pass per sample (clean CUDA context) for full-vocab log-probs.
    Step 3: Save .pt files in the same format as HF-based Phase 1.
    """
    student_data_dir = output_dir / "student_data"
    student_data_dir.mkdir(parents=True, exist_ok=True)

    # Resume check — if all .pt files exist, skip entirely
    existing = {
        int(f.stem.split("_")[1])
        for f in student_data_dir.glob("sample_*.pt")
    }
    dataset = load_math500()
    remaining = [i for i in range(len(dataset)) if i not in existing]

    if not remaining:
        print(f"Phase 1 complete ({len(existing)}/{len(dataset)} samples already exist)")
        return

    # Auto-detect GPU count if not specified
    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # --- Step 1: vLLM generation in subprocess ---
    # Check which samples still need token IDs generated (vs already having .pt files)
    token_ids_dir = output_dir / "vllm_token_ids"
    token_ids_dir.mkdir(parents=True, exist_ok=True)

    # Samples that have .pt files are done; samples that have token_ids are partially done
    needs_generation = [
        i for i in remaining
        if not (token_ids_dir / f"sample_{i:03d}.json").exists()
    ]

    if needs_generation:
        print(f"Phase 1 (vLLM): generating {len(needs_generation)} responses in subprocess (TP={tensor_parallel_size})")
        result = subprocess.run(
            [
                sys.executable, "-m", "src.exp_2._vllm_subprocess",
                "--model", model_name,
                "--output-dir", str(token_ids_dir),
                "--max-new-tokens", str(max_new_tokens),
                "--tensor-parallel", str(tensor_parallel_size),
                "--sample-indices", json.dumps(needs_generation),
            ],
            check=True,
        )
    else:
        print(f"Phase 1: vLLM generation already done, {len(remaining)} samples need log-prob extraction")

    # --- Step 2: HF forward pass for full-vocab log-probs (clean CUDA context) ---
    # Only process samples that have token IDs but not .pt files
    samples_needing_logprobs = [
        i for i in remaining
        if (token_ids_dir / f"sample_{i:03d}.json").exists()
    ]

    if not samples_needing_logprobs:
        print("Phase 1 complete (all samples have log-probs)")
        return

    print(f"Loading HF model for log-prob extraction ({len(samples_needing_logprobs)} samples)...")
    model, _ = load_model_and_tokenizer(model_name)

    with Progress() as progress:
        task = progress.add_task(
            "Phase 1: Extracting log-probs", total=len(samples_needing_logprobs)
        )
        for sample_idx in samples_needing_logprobs:
            token_file = token_ids_dir / f"sample_{sample_idx:03d}.json"
            with open(token_file) as f:
                token_data = json.load(f)

            prompt_length = token_data["prompt_length"]
            full_ids = torch.tensor(token_data["full_ids"], dtype=torch.long)
            gen_len = token_data["gen_len"]

            if gen_len == 0:
                print(f"  Sample {sample_idx}: no response generated, skipping")
                progress.advance(task)
                continue

            # Forward pass — same pattern as Phase 2 teacher logits
            with torch.no_grad():
                fwd = model(input_ids=full_ids.unsqueeze(0).to(model.device))

            logits = fwd.logits[0].float().cpu()
            student_logits = logits[prompt_length - 1 : prompt_length - 1 + gen_len]
            log_probs = torch.log_softmax(student_logits, dim=-1).to(torch.bfloat16)

            torch.save(
                {
                    "sample_idx": sample_idx,
                    "prompt_length": prompt_length,
                    "full_ids": full_ids,
                    "student_log_probs": log_probs,
                },
                student_data_dir / f"sample_{sample_idx:03d}.pt",
            )
            print(f"  Sample {sample_idx}: {gen_len} response tokens")
            progress.advance(task)

    del model
    torch.cuda.empty_cache()
    print("Phase 1 (vLLM) complete")

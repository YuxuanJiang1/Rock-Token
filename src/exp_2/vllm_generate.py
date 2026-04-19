"""vLLM-based generation for Phase 1.

Runs vLLM in an isolated subprocess to avoid CUDA context corruption.
vLLM generates with logprobs=-1 (full vocabulary), converts to tensors,
and saves .pt files directly. No HF model needed.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

from src.exp_2.utils import load_math500


def vllm_generate(
    model_name: str,
    output_dir: Path,
    max_new_tokens: int = 2048,
    tensor_parallel_size: int | None = None,
) -> None:
    """Generate student responses via vLLM with full-vocab log-probs.

    Runs entirely in a subprocess (vLLM generation + logprob extraction).
    Saves .pt files in the same format as HF-based Phase 1.
    """
    student_data_dir = output_dir / "student_data"
    student_data_dir.mkdir(parents=True, exist_ok=True)

    # Resume check
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

    print(f"Phase 1 (vLLM): {len(existing)}/{len(dataset)} done, {len(remaining)} remaining (TP={tensor_parallel_size})")

    result = subprocess.run(
        [
            sys.executable, "-m", "src.exp_2._vllm_subprocess",
            "--model", model_name,
            "--output-dir", str(student_data_dir),
            "--max-new-tokens", str(max_new_tokens),
            "--tensor-parallel", str(tensor_parallel_size),
            "--sample-indices", json.dumps(remaining),
        ],
        check=True,
    )

    print("Phase 1 (vLLM) complete")

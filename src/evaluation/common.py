"""Shared evaluation utilities for all benchmark scripts.

Provides vLLM engine creation, standard sampling params, result saving,
and rich console output helpers. Every benchmark script imports from here
to keep the interface consistent.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

SEED = 42


# ---------------------------------------------------------------------------
# vLLM helpers  (lazy imports — vllm is Linux/CUDA only)
# ---------------------------------------------------------------------------

def create_llm(
    model_name: str,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 32768,
    seed: int = SEED,
):
    """Create a vLLM engine with standard settings."""
    import torch
    from vllm import LLM

    if tensor_parallel_size is None:
        tensor_parallel_size = (
            torch.cuda.device_count() if torch.cuda.is_available() else 1
        )
    return LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        seed=seed,
        max_model_len=max_model_len,
    )


def default_sampling_params(
    max_tokens: int = 4096,
    temperature: float = 0,
    seed: int = SEED,
):
    """Standard greedy sampling params."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_results(output_path: str | Path, data: dict) -> None:
    """Save results as formatted JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def build_metadata(
    model_name: str,
    benchmark: str,
    dataset_name: str,
    n_samples: int,
    accuracy: float,
    correct: int,
    total: int,
    seed: int = SEED,
    **extra: object,
) -> dict:
    """Build the standard metadata block included in every result JSON."""
    return {
        "model": model_name,
        "benchmark": benchmark,
        "dataset": dataset_name,
        "n_samples": n_samples,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_accuracy(console: Console, title: str, correct: int, total: int) -> None:
    """Print a bold one-line accuracy summary."""
    acc = correct / total if total > 0 else 0
    console.print(f"\n[bold]{title}: {correct}/{total} ({acc:.1%})[/bold]\n")


def print_breakdown_table(
    console: Console,
    title: str,
    breakdowns: dict[str, dict],
    key_col: str = "Category",
) -> None:
    """Print a rich table with per-category accuracy breakdown."""
    table = Table(title=title)
    table.add_column(key_col, style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right", style="bold")
    for key in sorted(breakdowns.keys()):
        stats = breakdowns[key]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        table.add_row(
            str(key), str(stats["total"]), str(stats["correct"]), f"{acc:.1%}"
        )
    console.print(table)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_common_args(parser) -> None:
    """Add --model, --tensor-parallel, --n-samples, --max-new-tokens, --output."""
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model ID"
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Limit number of evaluation samples (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Max generation tokens",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed",
    )

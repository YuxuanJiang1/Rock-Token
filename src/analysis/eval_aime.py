"""Evaluate a model on the AIME benchmark (gneubig/aime-1983-2024).

Uses vLLM for fast batched inference. Standard evaluation prompt from
recent papers (DeepSeek-R1, Qwen3): step-by-step reasoning with \\boxed{}.

Usage:
    uv run python src/analysis/eval_aime.py --model Qwen/Qwen3-4B-Instruct-2507
    uv run python src/analysis/eval_aime.py --model ... --n-samples 100 --output results/aime.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer "
    "within \\boxed{}."
)

SEED = 42


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer from model output."""
    # Find all \boxed{...} patterns, take the last one
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison: strip whitespace, leading zeros."""
    answer = answer.strip()
    # Remove LaTeX formatting like $ signs
    answer = answer.replace("$", "").strip()
    # Try to parse as integer to normalize (e.g., "060" -> "60")
    try:
        return str(int(answer))
    except ValueError:
        return answer


def load_aime_dataset(n_samples: int | None = None):
    """Load AIME dataset from HuggingFace."""
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    return ds


def evaluate(
    model_name: str,
    n_samples: int | None = None,
    max_new_tokens: int = 4096,
    tensor_parallel_size: int | None = None,
    output_path: Path | None = None,
):
    """Run AIME evaluation."""
    console = Console()

    # Auto-detect GPUs
    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Load dataset
    dataset = load_aime_dataset(n_samples)
    console.print(f"Loaded {len(dataset)} AIME problems")

    # Build conversations
    conversations = []
    for sample in dataset:
        conversations.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["Question"]},
        ])

    # Init vLLM
    console.print(f"Loading model [bold]{model_name}[/bold] (TP={tensor_parallel_size})...")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        seed=SEED,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        seed=SEED,
    )

    console.print(f"Generating responses...")
    outputs = llm.chat(conversations, sampling_params)

    # Evaluate
    results = []
    correct = 0
    total = len(dataset)
    per_year = defaultdict(lambda: {"correct": 0, "total": 0})
    per_problem = defaultdict(lambda: {"correct": 0, "total": 0})

    for idx, output in enumerate(outputs):
        sample = dataset[idx]
        generated_text = output.outputs[0].text
        predicted = extract_boxed_answer(generated_text)
        gold = normalize_answer(str(sample["Answer"]))

        is_correct = False
        if predicted is not None:
            is_correct = normalize_answer(predicted) == gold

        if is_correct:
            correct += 1

        year = sample["Year"]
        prob_num = sample["Problem Number"]
        per_year[year]["total"] += 1
        per_year[year]["correct"] += int(is_correct)
        per_problem[prob_num]["total"] += 1
        per_problem[prob_num]["correct"] += int(is_correct)

        results.append({
            "id": sample["ID"],
            "year": year,
            "problem_number": prob_num,
            "gold_answer": gold,
            "predicted_answer": predicted,
            "correct": is_correct,
            "generated_text": generated_text,
        })

    # --- Console output ---
    accuracy = correct / total if total > 0 else 0
    console.print(f"\n[bold]Overall: {correct}/{total} ({accuracy:.1%})[/bold]\n")

    # Per-year table
    year_table = Table(title="Per Year")
    year_table.add_column("Year", style="cyan", justify="right")
    year_table.add_column("Total", justify="right")
    year_table.add_column("Correct", justify="right")
    year_table.add_column("Accuracy", justify="right", style="bold")
    for year in sorted(per_year.keys(), reverse=True):
        stats = per_year[year]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        year_table.add_row(
            str(year), str(stats["total"]), str(stats["correct"]), f"{acc:.1%}"
        )
    console.print(year_table)

    # Per-problem-number table
    prob_table = Table(title="\nPer Problem Number")
    prob_table.add_column("#", style="cyan", justify="right")
    prob_table.add_column("Total", justify="right")
    prob_table.add_column("Correct", justify="right")
    prob_table.add_column("Accuracy", justify="right", style="bold")
    for num in sorted(per_problem.keys()):
        stats = per_problem[num]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        prob_table.add_row(
            str(num), str(stats["total"]), str(stats["correct"]), f"{acc:.1%}"
        )
    console.print(prob_table)

    # --- Save to file ---
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": {
                "model": model_name,
                "dataset": "gneubig/aime-1983-2024",
                "n_samples": len(dataset),
                "max_new_tokens": max_new_tokens,
                "seed": SEED,
                "overall_accuracy": accuracy,
                "correct": correct,
                "total": total,
            },
            "per_year": {
                str(y): {
                    "correct": s["correct"],
                    "total": s["total"],
                    "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
                }
                for y, s in sorted(per_year.items())
            },
            "per_problem_number": {
                str(n): {
                    "correct": s["correct"],
                    "total": s["total"],
                    "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
                }
                for n, s in sorted(per_problem.items())
            },
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on AIME (gneubig/aime-1983-2024)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples (default: all 933)")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max generation tokens")
    parser.add_argument("--tensor-parallel", type=int, default=None, help="Number of GPUs (default: all)")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

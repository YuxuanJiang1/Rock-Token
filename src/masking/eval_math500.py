"""Evaluate a model on MATH-500 (HuggingFaceH4/MATH-500).

500 competition math problems across 7 subjects and 5 difficulty levels.
Uses vLLM for fast batched inference with \\boxed{} answer extraction
and SymPy-based comparison.

Usage:
    uv run python src/masking/eval_math500.py --model Qwen/Qwen3-4B-Instruct-2507
    uv run python src/masking/eval_math500.py --model ... --output results/math500.json
"""

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

from src.masking.common import (
    SYSTEM_PROMPT,
    add_common_args,
    answers_equal,
    build_metadata,
    create_llm,
    default_sampling_params,
    extract_boxed_answer,
    print_accuracy,
    print_breakdown_table,
    save_results,
)


def load_math500(n_samples: int | None = None):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    return ds


def build_conversations(dataset) -> list[list[dict]]:
    """Build chat conversations from dataset (for use with llm.chat)."""
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["problem"]},
        ]
        for sample in dataset
    ]


def score_outputs(outputs, dataset) -> dict:
    """Score pre-generated vLLM outputs against MATH-500 dataset.

    Returns dict with accuracy, correct, total, and per_correct (boolean list
    for bootstrap resampling in later steps).
    """
    correct = 0
    per_correct = []

    for idx, output in enumerate(outputs):
        sample = dataset[idx]
        generated = output.outputs[0].text
        predicted = extract_boxed_answer(generated)
        gold = sample["answer"]
        is_correct = predicted is not None and answers_equal(predicted, gold)
        if is_correct:
            correct += 1
        per_correct.append(is_correct)

    total = len(dataset)
    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "per_correct": per_correct,
    }


def evaluate(
    model_name: str,
    n_samples: int | None = None,
    max_new_tokens: int = 4096,
    tensor_parallel_size: int | None = None,
    output_path: Path | None = None,
    seed: int = 42,
):
    console = Console()

    dataset = load_math500(n_samples)
    console.print(f"Loaded {len(dataset)} MATH-500 problems")

    conversations = []
    for sample in dataset:
        conversations.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["problem"]},
        ])

    console.print(f"Loading model [bold]{model_name}[/bold]...")
    llm = create_llm(model_name, tensor_parallel_size, seed=seed)
    sampling = default_sampling_params(max_tokens=max_new_tokens, seed=seed)

    console.print("Generating responses...")
    outputs = llm.chat(conversations, sampling)

    results = []
    correct = 0
    per_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    per_level = defaultdict(lambda: {"correct": 0, "total": 0})

    for idx, output in enumerate(outputs):
        sample = dataset[idx]
        generated = output.outputs[0].text
        predicted = extract_boxed_answer(generated)
        gold = sample["answer"]

        is_correct = predicted is not None and answers_equal(predicted, gold)
        if is_correct:
            correct += 1

        subject = sample["subject"]
        level = str(sample["level"])
        per_subject[subject]["total"] += 1
        per_subject[subject]["correct"] += int(is_correct)
        per_level[level]["total"] += 1
        per_level[level]["correct"] += int(is_correct)

        results.append({
            "unique_id": sample["unique_id"],
            "subject": subject,
            "level": sample["level"],
            "gold_answer": gold,
            "predicted_answer": predicted,
            "correct": is_correct,
            "generated_text": generated,
        })

    total = len(dataset)
    accuracy = correct / total if total > 0 else 0

    print_accuracy(console, "MATH-500 Overall", correct, total)
    print_breakdown_table(console, "Per Subject", per_subject, key_col="Subject")
    print_breakdown_table(console, "Per Level", per_level, key_col="Level")

    if output_path:
        data = {
            "metadata": build_metadata(
                model_name=model_name,
                benchmark="math500",
                dataset_name="HuggingFaceH4/MATH-500",
                n_samples=len(dataset),
                accuracy=accuracy,
                correct=correct,
                total=total,
                seed=seed,
                max_new_tokens=max_new_tokens,
            ),
            "per_subject": {
                k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
                for k, v in sorted(per_subject.items())
            },
            "per_level": {
                k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
                for k, v in sorted(per_level.items())
            },
            "results": results,
        }
        save_results(output_path, data)
        console.print(f"\nResults saved to {output_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on MATH-500",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""Evaluate a model on HMMT February 2025 (MathArena/hmmt_feb_2025).

30 competition math problems with LaTeX-format answers.
Uses SymPy-based comparison for answer matching.

Usage:
    uv run python src/masking/eval_hmmt.py --model Qwen/Qwen3-4B-Instruct-2507
    uv run python src/masking/eval_hmmt.py --model ... --output results/hmmt25.json
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


def load_hmmt(n_samples: int | None = None):
    ds = load_dataset("MathArena/hmmt_feb_2025", split="train")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    return ds


def evaluate(
    model_name: str,
    n_samples: int | None = None,
    max_new_tokens: int = 4096,
    tensor_parallel_size: int | None = None,
    output_path: Path | None = None,
    seed: int = 42,
):
    console = Console()

    dataset = load_hmmt(n_samples)
    console.print(f"Loaded {len(dataset)} HMMT Feb 2025 problems")

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
    per_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for idx, output in enumerate(outputs):
        sample = dataset[idx]
        generated = output.outputs[0].text
        predicted = extract_boxed_answer(generated)
        gold = sample["answer"]

        is_correct = predicted is not None and answers_equal(predicted, gold)
        if is_correct:
            correct += 1

        types = sample["problem_type"]
        if isinstance(types, list):
            for t in types:
                per_type[t]["total"] += 1
                per_type[t]["correct"] += int(is_correct)
        else:
            per_type[str(types)]["total"] += 1
            per_type[str(types)]["correct"] += int(is_correct)

        results.append({
            "problem_idx": sample["problem_idx"],
            "problem_type": types,
            "gold_answer": gold,
            "predicted_answer": predicted,
            "correct": is_correct,
            "generated_text": generated,
        })

    total = len(dataset)
    accuracy = correct / total if total > 0 else 0

    print_accuracy(console, "HMMT Feb 2025 Overall", correct, total)
    if per_type:
        print_breakdown_table(console, "Per Problem Type", per_type, key_col="Type")

    if output_path:
        data = {
            "metadata": build_metadata(
                model_name=model_name,
                benchmark="hmmt_feb_2025",
                dataset_name="MathArena/hmmt_feb_2025",
                n_samples=len(dataset),
                accuracy=accuracy,
                correct=correct,
                total=total,
                seed=seed,
                max_new_tokens=max_new_tokens,
            ),
            "per_type": {
                k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
                for k, v in sorted(per_type.items())
            },
            "results": results,
        }
        save_results(output_path, data)
        console.print(f"\nResults saved to {output_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on HMMT February 2025",
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

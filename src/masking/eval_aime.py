"""Evaluate a model on AIME 2024 or AIME 2025.

AIME 2024: HuggingFaceH4/aime_2024 (30 problems, integer answers 0-999)
AIME 2025: MathArena/aime_2025 (30 problems, integer answers)

Uses vLLM for fast batched inference with \\boxed{} answer extraction.

Usage:
    uv run python src/masking/eval_aime.py --model ... --year 2024
    uv run python src/masking/eval_aime.py --model ... --year 2025 --output results/aime25.json
"""

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

from src.masking.common import (
    SYSTEM_PROMPT,
    add_common_args,
    build_metadata,
    create_llm,
    default_sampling_params,
    extract_boxed_answer,
    print_accuracy,
    print_breakdown_table,
    save_results,
)

DATASETS = {
    2024: {
        "name": "HuggingFaceH4/aime_2024",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "type_col": None,
    },
    2025: {
        "name": "MathArena/aime_2025",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "type_col": "problem_type",
    },
}


def load_aime(year: int, n_samples: int | None = None):
    cfg = DATASETS[year]
    ds = load_dataset(cfg["name"], split=cfg["split"])
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    return ds, cfg


def _normalize_int(answer) -> str:
    """Normalize AIME answer to integer string."""
    s = str(answer).strip().strip("$").strip()
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s


def evaluate(
    model_name: str,
    year: int = 2024,
    n_samples: int | None = None,
    max_new_tokens: int = 4096,
    tensor_parallel_size: int | None = None,
    output_path: Path | None = None,
    seed: int = 42,
):
    console = Console()

    dataset, cfg = load_aime(year, n_samples)
    console.print(f"Loaded {len(dataset)} AIME {year} problems")

    conversations = []
    for sample in dataset:
        conversations.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample[cfg["problem_col"]]},
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
        predicted_raw = extract_boxed_answer(generated)
        gold = _normalize_int(sample[cfg["answer_col"]])

        is_correct = False
        predicted = None
        if predicted_raw is not None:
            predicted = _normalize_int(predicted_raw)
            is_correct = predicted == gold

        if is_correct:
            correct += 1

        if cfg["type_col"] and cfg["type_col"] in sample:
            types = sample[cfg["type_col"]]
            if isinstance(types, list):
                for t in types:
                    per_type[t]["total"] += 1
                    per_type[t]["correct"] += int(is_correct)
            else:
                per_type[str(types)]["total"] += 1
                per_type[str(types)]["correct"] += int(is_correct)

        results.append({
            "index": idx,
            "gold_answer": gold,
            "predicted_answer": predicted,
            "correct": is_correct,
            "generated_text": generated,
        })

    total = len(dataset)
    accuracy = correct / total if total > 0 else 0

    print_accuracy(console, f"AIME {year} Overall", correct, total)
    if per_type:
        print_breakdown_table(console, "Per Problem Type", per_type, key_col="Type")

    if output_path:
        data = {
            "metadata": build_metadata(
                model_name=model_name,
                benchmark=f"aime{year}",
                dataset_name=cfg["name"],
                n_samples=len(dataset),
                accuracy=accuracy,
                correct=correct,
                total=total,
                seed=seed,
                max_new_tokens=max_new_tokens,
            ),
            "results": results,
        }
        if per_type:
            data["per_type"] = {
                k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
                for k, v in sorted(per_type.items())
            }
        save_results(output_path, data)
        console.print(f"\nResults saved to {output_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on AIME 2024/2025",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument(
        "--year",
        type=int,
        choices=[2024, 2025],
        required=True,
        help="AIME competition year",
    )
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        year=args.year,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

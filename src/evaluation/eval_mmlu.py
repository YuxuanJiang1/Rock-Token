"""Evaluate a model on MMLU (57 subjects, 14042 test questions).

Uses vLLM for fast batched inference.  Standard 5-shot format from
Hendrycks et al. (2021): five dev-set exemplars per subject, then the
test question.  Answer is a single letter A/B/C/D.

Usage:
    uv run python src/evaluation/eval_mmlu.py --model Qwen/Qwen3-4B-Instruct-2507
    uv run python src/evaluation/eval_mmlu.py --model ... --n-shot 0 --output results/mmlu.json
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

from src.evaluation.common import (
    add_common_args,
    build_metadata,
    create_llm,
    default_sampling_params,
    print_accuracy,
    print_breakdown_table,
    save_results,
)

# ---------------------------------------------------------------------------
# MMLU subject categories (Hendrycks et al. 2021)
# ---------------------------------------------------------------------------

STEM = {
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "machine_learning",
}
HUMANITIES = {
    "formal_logic", "high_school_european_history",
    "high_school_us_history", "high_school_world_history",
    "international_law", "jurisprudence", "logical_fallacies",
    "moral_disputes", "moral_scenarios", "philosophy", "prehistory",
    "professional_law", "world_religions",
}
SOCIAL_SCIENCES = {
    "econometrics", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_microeconomics",
    "high_school_psychology", "human_sexuality", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
}
# Everything else is "Other"

CHOICES = ["A", "B", "C", "D"]


def subject_to_category(subject: str) -> str:
    if subject in STEM:
        return "STEM"
    if subject in HUMANITIES:
        return "Humanities"
    if subject in SOCIAL_SCIENCES:
        return "Social Sciences"
    return "Other"


def format_subject(subject: str) -> str:
    return subject.replace("_", " ")


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_question(question: str, choices: list[str]) -> str:
    """Format a single MMLU question with choices."""
    lines = [question]
    for letter, choice in zip(CHOICES, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def build_conversations(
    test_ds,
    dev_ds,
    n_shot: int = 5,
) -> list[list[dict]]:
    """Build chat conversations for all test questions.

    Groups dev examples by subject for few-shot. Each conversation uses the
    chat template with a system prompt and user/assistant turns.
    """
    # Index dev examples by subject
    dev_by_subject: dict[str, list] = defaultdict(list)
    for sample in dev_ds:
        dev_by_subject[sample["subject"]].append(sample)

    conversations = []
    for sample in test_ds:
        subject = sample["subject"]
        subject_pretty = format_subject(subject)

        system_msg = (
            f"The following are multiple choice questions (with answers) "
            f"about {subject_pretty}. Reply with only the letter of the "
            f"correct answer (A, B, C, or D)."
        )
        messages: list[dict] = [{"role": "system", "content": system_msg}]

        # Few-shot exemplars from dev split
        exemplars = dev_by_subject.get(subject, [])[:n_shot]
        for ex in exemplars:
            q = format_question(ex["question"], ex["choices"])
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": CHOICES[ex["answer"]]})

        # Test question
        q = format_question(sample["question"], sample["choices"])
        messages.append({"role": "user", "content": q})
        conversations.append(messages)

    return conversations


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """Extract A/B/C/D from model output."""
    text = text.strip()

    # Exact single letter
    if text.upper() in CHOICES:
        return text.upper()

    # "The answer is X" pattern
    m = re.search(r"(?:the answer is|answer:)\s*([A-Da-d])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # First occurrence of a standalone letter
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str,
    n_samples: int | None = None,
    n_shot: int = 5,
    max_new_tokens: int = 64,
    tensor_parallel_size: int | None = None,
    output_path: Path | None = None,
    seed: int = 42,
):
    console = Console()

    # Load dataset — "all" config has all 57 subjects
    console.print("Loading MMLU dataset...")
    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")

    if n_samples is not None:
        test_ds = test_ds.select(range(min(n_samples, len(test_ds))))

    console.print(
        f"Loaded {len(test_ds)} test questions, "
        f"{len(dev_ds)} dev exemplars ({n_shot}-shot)"
    )

    conversations = build_conversations(test_ds, dev_ds, n_shot=n_shot)

    console.print(f"Loading model [bold]{model_name}[/bold]...")
    llm = create_llm(model_name, tensor_parallel_size, seed=seed)
    # Short max_tokens — we only need a single letter
    sampling = default_sampling_params(max_tokens=max_new_tokens, seed=seed)

    console.print("Generating responses...")
    outputs = llm.chat(conversations, sampling)

    # Score
    correct = 0
    total = len(test_ds)
    per_subject: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    per_category: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    results = []

    for idx, output in enumerate(outputs):
        sample = test_ds[idx]
        generated = output.outputs[0].text
        predicted = extract_answer(generated)
        gold = CHOICES[sample["answer"]]
        subject = sample["subject"]
        category = subject_to_category(subject)

        is_correct = predicted == gold
        if is_correct:
            correct += 1

        per_subject[subject]["total"] += 1
        per_subject[subject]["correct"] += int(is_correct)
        per_category[category]["total"] += 1
        per_category[category]["correct"] += int(is_correct)

        results.append({
            "index": idx,
            "subject": subject,
            "category": category,
            "question": sample["question"],
            "choices": sample["choices"],
            "gold_answer": gold,
            "predicted_answer": predicted,
            "correct": is_correct,
            "generated_text": generated,
        })

    accuracy = correct / total if total > 0 else 0

    # Console output
    print_accuracy(console, "MMLU", correct, total)
    print_breakdown_table(console, "By Category", per_category, key_col="Category")
    print_breakdown_table(console, "By Subject", per_subject, key_col="Subject")

    # Save
    if output_path:
        data = {
            "metadata": build_metadata(
                model_name=model_name,
                benchmark="mmlu",
                dataset_name="cais/mmlu",
                n_samples=len(test_ds),
                accuracy=accuracy,
                correct=correct,
                total=total,
                seed=seed,
                n_shot=n_shot,
                max_new_tokens=max_new_tokens,
            ),
            "per_category": {
                k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
                for k, v in sorted(per_category.items())
            },
            "per_subject": {
                k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
                for k, v in sorted(per_subject.items())
            },
            "results": results,
        }
        save_results(output_path, data)
        console.print(f"\nResults saved to {output_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on MMLU (57 subjects)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument(
        "--n-shot", type=int, default=5,
        help="Number of few-shot exemplars per subject (0 = zero-shot)",
    )
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        n_samples=args.n_samples,
        n_shot=args.n_shot,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

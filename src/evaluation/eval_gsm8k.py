"""Evaluate a model on GSM8K (grade-school math, 1319 test problems).

Uses vLLM for fast batched inference.  For instruction-tuned models the
default is 0-shot chain-of-thought (system prompt asks for step-by-step
reasoning with \\boxed{}).  Pass ``--n-shot 8`` to switch to the classic
8-shot CoT prompt from Wei et al. (2022).

Usage:
    uv run python src/evaluation/eval_gsm8k.py --model Qwen/Qwen3-4B-Instruct-2507
    uv run python src/evaluation/eval_gsm8k.py --model ... --n-shot 8 --output results/gsm8k.json
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
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer "
    "within \\boxed{}."
)

# Standard 8-shot exemplars from Wei et al. (2022) "Chain-of-Thought Prompting"
EIGHT_SHOT_EXEMPLARS = [
    {
        "q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "a": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is \\boxed{6}.",
    },
    {
        "q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "a": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is \\boxed{5}.",
    },
    {
        "q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "a": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is \\boxed{39}.",
    },
    {
        "q": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "a": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is \\boxed{8}.",
    },
    {
        "q": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "a": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is \\boxed{9}.",
    },
    {
        "q": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "a": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is \\boxed{29}.",
    },
    {
        "q": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "a": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33. The answer is \\boxed{33}.",
    },
    {
        "q": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "a": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is \\boxed{8}.",
    },
]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """Extract numeric answer from model output.

    Tries \\boxed{} first, then falls back to the last number in the text.
    """
    # Try \boxed{...}
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return normalize_number(matches[-1])
    # Fallback: last number in text
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if numbers:
        return normalize_number(numbers[-1])
    return None


def extract_gold_answer(answer_text: str) -> str:
    """Extract the gold answer from GSM8K's '#### N' format."""
    match = re.search(r"####\s*(.*)", answer_text)
    if match:
        return normalize_number(match.group(1))
    return normalize_number(answer_text)


def normalize_number(s: str) -> str:
    """Strip whitespace, commas, dollar signs; parse as number."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        # Normalize float representation (e.g. 5.0 -> 5)
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_gsm8k(n_samples: int | None = None):
    """Load GSM8K test split."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    return ds


# ---------------------------------------------------------------------------
# Conversation building
# ---------------------------------------------------------------------------

def build_conversations(dataset, n_shot: int = 0) -> list[list[dict]]:
    """Build chat-formatted conversations for each test problem."""
    conversations = []
    for sample in dataset:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Few-shot exemplars
        if n_shot > 0:
            for ex in EIGHT_SHOT_EXEMPLARS[:n_shot]:
                messages.append({"role": "user", "content": ex["q"]})
                messages.append({"role": "assistant", "content": ex["a"]})

        messages.append({"role": "user", "content": sample["question"]})
        conversations.append(messages)
    return conversations


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str,
    n_samples: int | None = None,
    n_shot: int = 0,
    max_new_tokens: int = 4096,
    tensor_parallel_size: int | None = None,
    output_path: Path | None = None,
    seed: int = 42,
):
    console = Console()

    dataset = load_gsm8k(n_samples)
    console.print(f"Loaded {len(dataset)} GSM8K problems ({n_shot}-shot)")

    conversations = build_conversations(dataset, n_shot=n_shot)

    console.print(f"Loading model [bold]{model_name}[/bold]...")
    llm = create_llm(model_name, tensor_parallel_size, seed=seed)
    sampling = default_sampling_params(max_tokens=max_new_tokens, seed=seed)

    console.print("Generating responses...")
    outputs = llm.chat(conversations, sampling)

    # Score
    correct = 0
    total = len(dataset)
    per_difficulty: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    results = []

    for idx, output in enumerate(outputs):
        sample = dataset[idx]
        generated = output.outputs[0].text
        predicted = extract_answer(generated)
        gold = extract_gold_answer(sample["answer"])

        is_correct = predicted is not None and predicted == gold
        if is_correct:
            correct += 1

        # Approximate difficulty by number of reasoning steps in gold answer
        n_steps = sample["answer"].count("\n")
        bucket = "1-2 steps" if n_steps <= 2 else ("3-5 steps" if n_steps <= 5 else "6+ steps")
        per_difficulty[bucket]["total"] += 1
        per_difficulty[bucket]["correct"] += int(is_correct)

        results.append({
            "index": idx,
            "question": sample["question"],
            "gold_answer": gold,
            "predicted_answer": predicted,
            "correct": is_correct,
            "generated_text": generated,
        })

    accuracy = correct / total if total > 0 else 0

    # Console output
    print_accuracy(console, "GSM8K", correct, total)
    print_breakdown_table(console, "By Difficulty", per_difficulty, key_col="Steps")

    # Save
    if output_path:
        data = {
            "metadata": build_metadata(
                model_name=model_name,
                benchmark="gsm8k",
                dataset_name="openai/gsm8k",
                n_samples=len(dataset),
                accuracy=accuracy,
                correct=correct,
                total=total,
                seed=seed,
                n_shot=n_shot,
                max_new_tokens=max_new_tokens,
            ),
            "per_difficulty": {
                k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
                for k, v in sorted(per_difficulty.items())
            },
            "results": results,
        }
        save_results(output_path, data)
        console.print(f"\nResults saved to {output_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on GSM8K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument(
        "--n-shot", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        help="Number of few-shot exemplars (0 = zero-shot CoT)",
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

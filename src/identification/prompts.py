"""Load and mix evaluation prompts from multiple datasets (train splits only).

All sources use train splits to avoid contamination with evaluation benchmarks
(GSM8K test, HumanEval test, MMLU test, IF-Eval).
"""

import random

from datasets import load_dataset
from rich.console import Console


def load_mixed_prompts(
    math_count: int = 500,
    gsm8k_count: int = 500,
    mbpp_count: int = 300,
    alpaca_count: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Load and mix prompts from MATH, GSM8K, MBPP, and Alpaca train splits.

    Returns a shuffled list of {"prompt": str, "source": str} dicts.
    """
    console = Console()
    rng = random.Random(seed)
    prompts: list[dict] = []

    # MATH train (competition math) — load all 7 subjects from EleutherAI's copy
    # (hendrycks/competition_math uses a deprecated loading script)
    console.print("Loading MATH train...")
    math_subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    all_math = []
    for subject in math_subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        all_math.extend(ds)
    indices = rng.sample(range(len(all_math)), min(math_count, len(all_math)))
    for i in indices:
        prompts.append({"prompt": all_math[i]["problem"], "source": "math"})
    console.print(f"  MATH: {len(indices)} prompts ({len(all_math)} total available)")

    # GSM8K train (grade-school math)
    console.print("Loading GSM8K train...")
    gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = rng.sample(range(len(gsm_ds)), min(gsm8k_count, len(gsm_ds)))
    for i in indices:
        prompts.append({"prompt": gsm_ds[i]["question"], "source": "gsm8k"})
    console.print(f"  GSM8K: {len(indices)} prompts")

    # MBPP train (code)
    console.print("Loading MBPP train...")
    mbpp_ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    indices = rng.sample(range(len(mbpp_ds)), min(mbpp_count, len(mbpp_ds)))
    for i in indices:
        prompts.append({"prompt": mbpp_ds[i]["text"], "source": "mbpp"})
    console.print(f"  MBPP: {len(indices)} prompts")

    # Alpaca (general instructions — no standard test split)
    console.print("Loading Alpaca...")
    alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train")
    indices = rng.sample(range(len(alpaca_ds)), min(alpaca_count, len(alpaca_ds)))
    for i in indices:
        prompts.append({"prompt": alpaca_ds[i]["instruction"], "source": "alpaca"})
    console.print(f"  Alpaca: {len(indices)} prompts")

    rng.shuffle(prompts)
    console.print(f"Total: {len(prompts)} mixed prompts")
    return prompts

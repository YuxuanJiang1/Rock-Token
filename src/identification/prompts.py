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

    # MATH train (competition math)
    console.print("Loading MATH train...")
    math_ds = load_dataset("hendrycks/competition_math", split="train")
    indices = rng.sample(range(len(math_ds)), min(math_count, len(math_ds)))
    for i in indices:
        prompts.append({"prompt": math_ds[i]["problem"], "source": "math"})
    console.print(f"  MATH: {len(indices)} prompts")

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

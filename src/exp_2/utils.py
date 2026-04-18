import csv
import json
from datetime import date
from pathlib import Path

import torch
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_math500():
    """Load MATH-500 dataset from HuggingFace."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return ds


def load_model_and_tokenizer(model_name: str):
    """Load model in bf16 distributed across available GPUs."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def format_prompt(problem: str, tokenizer) -> list[dict]:
    """Format a math problem as chat messages for apply_chat_template."""
    return [{"role": "user", "content": problem}]


def save_rock_tokens_json(rock_tokens: list[dict], metadata: dict, path: Path):
    """Save rock tokens and metadata to JSON."""
    output = {"metadata": metadata, "rock_tokens": rock_tokens}
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def save_rock_tokens_csv(rock_tokens: list[dict], path: Path):
    """Save rock tokens to CSV."""
    if not rock_tokens:
        return
    fieldnames = list(rock_tokens[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rock_tokens)


def print_results_table(rock_tokens: list[dict], entropy_threshold: float, metadata: dict):
    """Print rich table of Rock Token results to console."""
    console = Console()

    table = Table(title="Rock Token Identification Results")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Token ID", style="magenta", justify="right")
    table.add_column("Token", style="green")
    table.add_column("Freq", justify="right")
    table.add_column("Avg KL", justify="right", style="yellow")
    table.add_column("Rock Score", justify="right", style="bold")
    table.add_column("Avg Entropy", justify="right")
    table.add_column("Class", justify="center")

    for rt in rock_tokens:
        cls_style = "bold green" if rt["classification"] == "pillar" else "bold red"
        table.add_row(
            str(rt["rank"]),
            str(rt["token_id"]),
            repr(rt["token_string"]),
            str(rt["frequency"]),
            f"{rt['avg_kl']:.4f}",
            f"{rt['rock_score']:.4f}",
            f"{rt['avg_teacher_entropy']:.4f}",
            f"[{cls_style}]{rt['classification']}[/{cls_style}]",
        )

    console.print(table)

    n_pillar = sum(1 for rt in rock_tokens if rt["classification"] == "pillar")
    n_stumbling = len(rock_tokens) - n_pillar
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Total tokens analyzed: {metadata['total_positions']}")
    console.print(f"  Unique token types: {metadata['unique_token_types']}")
    console.print(f"  Entropy threshold (global median): {entropy_threshold:.4f}")
    console.print(f"  Top-{len(rock_tokens)} Rock Tokens: {n_pillar} Pillars, {n_stumbling} Stumbling Blocks")

"""Individual token knockout — mask each Rock Token one at a time.

Loads the model ONCE, then runs 201 evaluations (1 baseline + 200 masked)
on MATH-500 and IF-Eval.  Uses vLLM logits_processors to suppress tokens
by setting their logits to -inf.

Supports resume: if a per-token JSON already exists, it is loaded instead
of re-running that knockout.

Usage:
    uv run python src/masking/knockout.py \
        --rock-tokens results/identification/onpolicy/rock_tokens_by_count.csv \
        --category count \
        --output-dir results/masking/knockout

    # Smoke test (5 samples per benchmark, first 3 tokens)
    uv run python src/masking/knockout.py \
        --rock-tokens results/identification/onpolicy/rock_tokens_by_count.csv \
        --output-dir results/masking/knockout_smoke \
        --n-samples 5 --n-tokens 3
"""

import argparse
import csv
import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from src.masking.common import SEED, create_llm, default_sampling_params, save_results
from src.masking.eval_ifeval import (
    build_conversations as build_ifeval_conversations,
)
from src.masking.eval_ifeval import (
    load_ifeval,
    score_outputs as score_ifeval,
)
from src.masking.eval_math500 import (
    build_conversations as build_math500_conversations,
)
from src.masking.eval_math500 import (
    load_math500,
    score_outputs as score_math500,
)


def load_rock_tokens(csv_path: str, category: str) -> list[dict]:
    """Load rock tokens from CSV, sorted by category ranking."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        tokens = list(reader)

    rank_col = f"rank_by_{category}"
    tokens.sort(key=lambda t: int(t[rank_col]))

    return [
        {
            "token_id": int(t["token_id"]),
            "token_string": t["token_string"],
            "frequency": int(t["frequency"]),
            "rock_count": int(t["rock_count"]),
            "rock_rate": float(t["rock_rate"]),
            "rank": int(t[rank_col]),
        }
        for t in tokens
    ]


def make_logit_bias(token_id: int) -> dict[int, float]:
    """Create a logit_bias dict that suppresses a single token.

    A bias of -100 makes the token effectively impossible to sample
    (equivalent to logit → -inf for practical purposes).
    """
    return {token_id: -100.0}


def run_knockout(
    model_name: str,
    rock_tokens: list[dict],
    output_dir: Path,
    n_samples: int | None = None,
    max_new_tokens: int = 4096,
    tensor_parallel_size: int | None = None,
    seed: int = SEED,
):
    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir = output_dir / "tokens"
    tokens_dir.mkdir(exist_ok=True)

    # Load datasets once
    console.print("[bold]Loading datasets...[/bold]")
    math500_ds = load_math500(n_samples)
    math500_convs = build_math500_conversations(math500_ds)
    ifeval_ds = load_ifeval(n_samples)
    ifeval_convs = build_ifeval_conversations(ifeval_ds)
    console.print(f"  MATH-500: {len(math500_ds)} problems")
    console.print(f"  IF-Eval:  {len(ifeval_ds)} prompts")

    # Load model once
    console.print(f"\n[bold]Loading model:[/bold] {model_name}")
    llm = create_llm(model_name, tensor_parallel_size, seed=seed)

    # --- Baseline (no masking) ---
    baseline_path = output_dir / "baseline.json"
    if baseline_path.exists():
        console.print("\n[bold]Loading cached baseline...[/bold]")
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        baseline_math500 = baseline_data["math500"]
        baseline_ifeval = baseline_data["ifeval"]
    else:
        console.print("\n[bold]Running baseline (no masking)...[/bold]")
        baseline_sampling = default_sampling_params(
            max_tokens=max_new_tokens, seed=seed
        )

        start = time.time()
        outputs = llm.chat(math500_convs, baseline_sampling)
        baseline_math500 = score_math500(outputs, math500_ds)

        outputs = llm.chat(ifeval_convs, baseline_sampling)
        baseline_ifeval = score_ifeval(outputs, ifeval_ds)
        elapsed = time.time() - start

        save_results(
            baseline_path,
            {
                "model": model_name,
                "math500": baseline_math500,
                "ifeval": baseline_ifeval,
                "elapsed_s": round(elapsed, 1),
            },
        )
        console.print(f"  Baseline saved ({elapsed:.0f}s)")

    console.print(f"  MATH-500 baseline: {baseline_math500['accuracy']:.1%}")
    console.print(f"  IF-Eval baseline:  {baseline_ifeval['accuracy']:.1%}")

    # --- Individual knockouts ---
    console.print(
        f"\n[bold]Running {len(rock_tokens)} individual knockouts...[/bold]"
    )
    summary_rows = []
    skipped = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Knockout", total=len(rock_tokens))

        for i, tok in enumerate(rock_tokens):
            token_id = tok["token_id"]
            token_str = tok["token_string"]
            token_path = tokens_dir / f"token_{token_id}.json"

            # Resume support: skip if already done
            if token_path.exists():
                with open(token_path) as f:
                    existing = json.load(f)
                summary_rows.append(
                    {
                        "token_id": token_id,
                        "token_string": token_str,
                        "rank": tok["rank"],
                        "frequency": tok["frequency"],
                        "rock_count": tok["rock_count"],
                        "math500_acc": existing["math500"]["accuracy"],
                        "math500_delta": existing["math500"]["delta"],
                        "ifeval_acc": existing["ifeval"]["accuracy"],
                        "ifeval_delta": existing["ifeval"]["delta"],
                    }
                )
                skipped += 1
                progress.update(task, advance=1)
                continue

            # Create masked sampling params
            from vllm import SamplingParams

            masked_sampling = SamplingParams(
                temperature=0,
                max_tokens=max_new_tokens,
                seed=seed,
                top_k=1,
                logit_bias=make_logit_bias(token_id),
            )

            start = time.time()

            m_outputs = llm.chat(math500_convs, masked_sampling)
            m_math500 = score_math500(m_outputs, math500_ds)

            m_outputs = llm.chat(ifeval_convs, masked_sampling)
            m_ifeval = score_ifeval(m_outputs, ifeval_ds)

            elapsed = time.time() - start

            math500_delta = m_math500["accuracy"] - baseline_math500["accuracy"]
            ifeval_delta = m_ifeval["accuracy"] - baseline_ifeval["accuracy"]

            token_result = {
                "token_id": token_id,
                "token_string": token_str,
                "rank": tok["rank"],
                "frequency": tok["frequency"],
                "rock_count": tok["rock_count"],
                "rock_rate": tok["rock_rate"],
                "math500": {
                    "accuracy": m_math500["accuracy"],
                    "correct": m_math500["correct"],
                    "total": m_math500["total"],
                    "delta": math500_delta,
                    "per_correct": m_math500["per_correct"],
                },
                "ifeval": {
                    "accuracy": m_ifeval["accuracy"],
                    "correct": m_ifeval["correct"],
                    "total": m_ifeval["total"],
                    "delta": ifeval_delta,
                    "per_correct": m_ifeval["per_correct"],
                },
                "elapsed_s": round(elapsed, 1),
            }
            save_results(token_path, token_result)

            summary_rows.append(
                {
                    "token_id": token_id,
                    "token_string": token_str,
                    "rank": tok["rank"],
                    "frequency": tok["frequency"],
                    "rock_count": tok["rock_count"],
                    "math500_acc": m_math500["accuracy"],
                    "math500_delta": math500_delta,
                    "ifeval_acc": m_ifeval["accuracy"],
                    "ifeval_delta": ifeval_delta,
                }
            )

            progress.update(
                task,
                advance=1,
                description=f"[{i+1}/{len(rock_tokens)}] {token_str.strip()} "
                f"math500 Δ={math500_delta:+.1%}",
            )

    if skipped:
        console.print(f"  ({skipped} tokens loaded from cache)")

    # --- Save summary ---
    summary_rows.sort(key=lambda r: r["math500_delta"], reverse=True)

    summary_csv_path = output_dir / "summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    save_results(
        output_dir / "summary.json",
        {
            "model": model_name,
            "category": "count",
            "n_tokens": len(rock_tokens),
            "baseline": {
                "math500": baseline_math500["accuracy"],
                "ifeval": baseline_ifeval["accuracy"],
            },
            "knockout_results": summary_rows,
        },
    )

    # --- Print top results ---
    console.print()

    def _styled_delta(delta: float) -> str:
        """Format a delta with color: green if positive, red if negative, plain if zero."""
        s = f"{delta:+.2%}"
        if delta > 0:
            return f"[green]{s}[/green]"
        elif delta < 0:
            return f"[red]{s}[/red]"
        return s

    # Top helpers (removal helps)
    table = Table(title="Top 10 Stumbling Block Candidates (removal helps MATH-500)")
    table.add_column("Rank", justify="right")
    table.add_column("Token", style="cyan")
    table.add_column("Freq", justify="right")
    table.add_column("MATH-500 Δ", justify="right")
    table.add_column("IF-Eval Δ", justify="right")
    for row in summary_rows[:10]:
        table.add_row(
            str(row["rank"]),
            row["token_string"].strip(),
            str(row["frequency"]),
            _styled_delta(row["math500_delta"]),
            _styled_delta(row["ifeval_delta"]),
        )
    console.print(table)

    # Top hurters (removal hurts)
    console.print()
    table = Table(title="Top 10 Pillar Candidates (removal hurts MATH-500)")
    table.add_column("Rank", justify="right")
    table.add_column("Token", style="cyan")
    table.add_column("Freq", justify="right")
    table.add_column("MATH-500 Δ", justify="right")
    table.add_column("IF-Eval Δ", justify="right")
    for row in reversed(summary_rows[-10:]):
        table.add_row(
            str(row["rank"]),
            row["token_string"].strip(),
            str(row["frequency"]),
            _styled_delta(row["math500_delta"]),
            _styled_delta(row["ifeval_delta"]),
        )
    console.print(table)

    console.print(f"\nFull results: {summary_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Individual Rock Token knockout experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--rock-tokens",
        type=str,
        required=True,
        help="Path to rock tokens CSV",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["count", "rate"],
        default="count",
        help="Which ranking to use (count or rate)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/masking/knockout/{category})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Limit samples per benchmark (for testing)",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=None,
        help="Limit number of tokens to knock out (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Max generation tokens",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=None,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed",
    )
    args = parser.parse_args()

    rock_tokens = load_rock_tokens(args.rock_tokens, args.category)
    if args.n_tokens is not None:
        rock_tokens = rock_tokens[: args.n_tokens]

    output_dir = args.output_dir or f"results/masking/knockout/{args.category}"

    run_knockout(
        model_name=args.model,
        rock_tokens=rock_tokens,
        output_dir=Path(output_dir),
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

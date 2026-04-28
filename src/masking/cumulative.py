"""Cumulative removal curves (Step 2.3).

For each removal mode (greedy Pillar, greedy Stumbling Block, random) and a
sequence of cumulative k values, mask the corresponding set of tokens
simultaneously and evaluate MATH-500 + IF-Eval.  Plots the curves with a
random baseline band to reveal the crash point.

Loads the model ONCE.  Reuses the baseline from the knockout step.
Resume support per (mode, seed, k).

Usage:
    uv run python src/masking/cumulative.py \\
        --knockout-dir results/masking/knockout/count \\
        --categorization results/masking/categorization/count/categorization.csv

    # Smoke (3 small k values, single random seed)
    uv run python src/masking/cumulative.py \\
        --knockout-dir results/masking/knockout/count \\
        --categorization results/masking/categorization/count/categorization.csv \\
        --output-dir results/masking/cumulative_smoke \\
        --k-values 1 5 10 --random-seeds 1 --n-samples 5
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

DEFAULT_K_VALUES = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200]


def load_tokens_with_deltas(categorization_csv: Path) -> list[dict]:
    """Load tokens sorted by MATH-500 delta ascending (most negative first)."""
    with open(categorization_csv) as f:
        rows = list(csv.DictReader(f))
    tokens = [
        {
            "token_id": int(r["token_id"]),
            "token_string": r["token_string"],
            "math500_delta": float(r["math500_delta"]),
            "ifeval_delta": float(r["ifeval_delta"]),
        }
        for r in rows
    ]
    tokens.sort(key=lambda t: t["math500_delta"])  # ascending: most negative first
    return tokens


def select_tokens(tokens_sorted: list[dict], mode: str, k: int, seed: int = 0) -> list[int]:
    """Return the token IDs to mask for a given mode and k."""
    if mode == "greedy_pillar":
        # Most negative first (Pillar candidates)
        return [t["token_id"] for t in tokens_sorted[:k]]
    elif mode == "greedy_stumbling":
        # Most positive first (Stumbling Block candidates)
        return [t["token_id"] for t in reversed(tokens_sorted)][:k]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(tokens_sorted))
        return [tokens_sorted[i]["token_id"] for i in permutation[:k]]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_one_evaluation(
    llm,
    math500_convs,
    math500_ds,
    ifeval_convs,
    ifeval_ds,
    token_ids_to_mask: list[int],
    max_new_tokens: int,
    seed: int,
):
    """Run a single evaluation point with the given tokens masked."""
    from vllm import SamplingParams

    if token_ids_to_mask:
        sampling = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            seed=seed,
            top_k=1,
            logit_bias={tid: -100.0 for tid in token_ids_to_mask},
        )
    else:
        sampling = default_sampling_params(max_tokens=max_new_tokens, seed=seed)

    m_outputs = llm.chat(math500_convs, sampling)
    m_result = score_math500(m_outputs, math500_ds)
    i_outputs = llm.chat(ifeval_convs, sampling)
    i_result = score_ifeval(i_outputs, ifeval_ds)
    return m_result, i_result


def run_cumulative(
    model_name: str,
    knockout_dir: Path,
    categorization_csv: Path,
    output_dir: Path,
    k_values: list[int],
    n_random_seeds: int,
    n_samples: int | None,
    max_new_tokens: int,
    tensor_parallel_size: int | None,
    seed: int,
):
    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokens sorted by delta
    tokens = load_tokens_with_deltas(categorization_csv)
    console.print(f"Loaded {len(tokens)} tokens from {categorization_csv}")
    console.print(f"  Most negative Δ: {tokens[0]['token_string'].strip()!r} ({tokens[0]['math500_delta']:+.2%})")
    console.print(f"  Most positive Δ: {tokens[-1]['token_string'].strip()!r} ({tokens[-1]['math500_delta']:+.2%})")

    # Load datasets once
    console.print("\n[bold]Loading datasets...[/bold]")
    math500_ds = load_math500(n_samples)
    math500_convs = build_math500_conversations(math500_ds)
    ifeval_ds = load_ifeval(n_samples)
    ifeval_convs = build_ifeval_conversations(ifeval_ds)
    console.print(f"  MATH-500: {len(math500_ds)} problems")
    console.print(f"  IF-Eval:  {len(ifeval_ds)} prompts")

    # Load model once
    console.print(f"\n[bold]Loading model:[/bold] {model_name}")
    llm = create_llm(model_name, tensor_parallel_size, seed=seed)

    # --- Baseline (must be in-session to avoid cross-session contamination) ---
    # NOTE: We deliberately do NOT reuse the knockout baseline here. vLLM is
    # deterministic within a session but per-problem outputs can differ across
    # sessions (different KV-cache scheduling, batch chunking). Computing the
    # baseline fresh in this session ensures all (mode, k) deltas are within-
    # session and comparable.
    baseline_path = output_dir / "baseline.json"

    if baseline_path.exists():
        console.print("\n[bold]Using cached baseline from this session's cumulative dir[/bold]")
        with open(baseline_path) as f:
            baseline_data = json.load(f)
    else:
        console.print("\n[bold]Running baseline (in-session)...[/bold]")
        start = time.time()
        m_result, i_result = run_one_evaluation(
            llm, math500_convs, math500_ds, ifeval_convs, ifeval_ds,
            [], max_new_tokens, seed,
        )
        elapsed = time.time() - start
        baseline_data = {
            "model": model_name,
            "math500": m_result,
            "ifeval": i_result,
            "elapsed_s": round(elapsed, 1),
        }
        save_results(baseline_path, baseline_data)
        console.print(f"  Baseline computed ({elapsed:.0f}s)")

    base_m = baseline_data["math500"]["accuracy"]
    base_i = baseline_data["ifeval"]["accuracy"]
    console.print(f"  Baseline MATH-500: {base_m:.1%}")
    console.print(f"  Baseline IF-Eval:  {base_i:.1%}")

    # --- Build run list ---
    runs = []
    for mode in ["greedy_pillar", "greedy_stumbling"]:
        for k in k_values:
            runs.append({"mode": mode, "seed": 0, "k": k})
    for s in range(n_random_seeds):
        for k in k_values:
            runs.append({"mode": "random", "seed": s, "k": k})

    console.print(f"\n[bold]{len(runs)} cumulative evaluation points to run[/bold]")
    console.print(f"  Greedy Pillar:    {len(k_values)} points")
    console.print(f"  Greedy Stumbling: {len(k_values)} points")
    console.print(f"  Random:           {n_random_seeds} seeds × {len(k_values)} points = {n_random_seeds * len(k_values)} points")

    # --- Run all evaluation points ---
    summary_rows = []

    with Progress(console=console) as progress:
        task = progress.add_task("Cumulative", total=len(runs))

        for i, run in enumerate(runs):
            mode = run["mode"]
            run_seed = run["seed"]
            k = run["k"]

            if mode == "random":
                run_dir = output_dir / f"random_seed_{run_seed}"
            else:
                run_dir = output_dir / mode
            run_dir.mkdir(parents=True, exist_ok=True)
            result_path = run_dir / f"k_{k:03d}.json"

            # Resume
            if result_path.exists():
                with open(result_path) as f:
                    cached = json.load(f)
                summary_rows.append({
                    "mode": mode, "seed": run_seed, "k": k,
                    "math500_acc": cached["math500"]["accuracy"],
                    "math500_delta": cached["math500"]["accuracy"] - base_m,
                    "ifeval_acc": cached["ifeval"]["accuracy"],
                    "ifeval_delta": cached["ifeval"]["accuracy"] - base_i,
                })
                progress.update(task, advance=1)
                continue

            token_ids = select_tokens(tokens, mode, k, seed=run_seed)
            start = time.time()
            m_result, i_result = run_one_evaluation(
                llm, math500_convs, math500_ds, ifeval_convs, ifeval_ds,
                token_ids, max_new_tokens, seed,
            )
            elapsed = time.time() - start

            m_delta = m_result["accuracy"] - base_m
            i_delta = i_result["accuracy"] - base_i

            save_results(result_path, {
                "mode": mode,
                "seed": run_seed,
                "k": k,
                "n_tokens_masked": len(token_ids),
                "masked_token_ids": token_ids,
                "math500": {**m_result, "delta": m_delta},
                "ifeval": {**i_result, "delta": i_delta},
                "elapsed_s": round(elapsed, 1),
            })

            summary_rows.append({
                "mode": mode, "seed": run_seed, "k": k,
                "math500_acc": m_result["accuracy"],
                "math500_delta": m_delta,
                "ifeval_acc": i_result["accuracy"],
                "ifeval_delta": i_delta,
            })

            progress.update(
                task,
                advance=1,
                description=f"[{i+1}/{len(runs)}] {mode} seed={run_seed} k={k} M500 Δ={m_delta:+.1%}",
            )

    # --- Save summary ---
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    save_results(output_dir / "summary.json", {
        "model": model_name,
        "baseline": {"math500": base_m, "ifeval": base_i},
        "k_values": k_values,
        "n_random_seeds": n_random_seeds,
        "results": summary_rows,
    })

    # --- Plot ---
    _plot_cumulative_curves(summary_rows, base_m, base_i, k_values, n_random_seeds,
                            output_dir / "plots" / "cumulative_curves.png")

    # --- Console summary ---
    console.print()
    table = Table(title="Cumulative Removal Summary (MATH-500 Δ, accuracy points)")
    table.add_column("k", justify="right")
    table.add_column("Greedy Pillar", justify="right", style="red")
    table.add_column("Greedy Stumbling", justify="right", style="green")
    table.add_column("Random (mean ± std)", justify="right")

    for k in k_values:
        gp = next((r for r in summary_rows if r["mode"] == "greedy_pillar" and r["k"] == k), None)
        gs = next((r for r in summary_rows if r["mode"] == "greedy_stumbling" and r["k"] == k), None)
        rand_deltas = [r["math500_delta"] for r in summary_rows if r["mode"] == "random" and r["k"] == k]

        gp_str = f"{gp['math500_delta']:+.2%}" if gp else "—"
        gs_str = f"{gs['math500_delta']:+.2%}" if gs else "—"
        if rand_deltas:
            r_arr = np.array(rand_deltas)
            rand_str = f"{r_arr.mean():+.2%} ± {r_arr.std():.2%}"
        else:
            rand_str = "—"

        table.add_row(str(k), gp_str, gs_str, rand_str)

    console.print(table)
    console.print(f"\nSaved: {csv_path}")
    console.print(f"Saved: {output_dir / 'plots' / 'cumulative_curves.png'}")


def _plot_cumulative_curves(summary_rows, base_m, base_i, k_values, n_random_seeds, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title, baseline in [
        (axes[0], "math500", "MATH-500", base_m),
        (axes[1], "ifeval", "IF-Eval", base_i),
    ]:
        # Greedy curves
        gp = sorted(
            [r for r in summary_rows if r["mode"] == "greedy_pillar"],
            key=lambda r: r["k"],
        )
        gs = sorted(
            [r for r in summary_rows if r["mode"] == "greedy_stumbling"],
            key=lambda r: r["k"],
        )

        if gp:
            ax.plot([r["k"] for r in gp], [r[f"{key}_acc"] * 100 for r in gp],
                    "o-", color="#8B0000", label="Greedy Pillar removal", linewidth=2)
        if gs:
            ax.plot([r["k"] for r in gs], [r[f"{key}_acc"] * 100 for r in gs],
                    "s-", color="#1B5E20", label="Greedy Stumbling Block removal", linewidth=2)

        # Random band
        if n_random_seeds > 0:
            rand_means, rand_stds, rand_ks = [], [], []
            for k in k_values:
                vals = [r[f"{key}_acc"] for r in summary_rows
                        if r["mode"] == "random" and r["k"] == k]
                if vals:
                    rand_ks.append(k)
                    rand_means.append(np.mean(vals) * 100)
                    rand_stds.append(np.std(vals) * 100)
            if rand_means:
                rand_means = np.array(rand_means)
                rand_stds = np.array(rand_stds)
                ax.plot(rand_ks, rand_means, "^--", color="#1976D2",
                        label=f"Random (n={n_random_seeds} seeds)", linewidth=1.5)
                ax.fill_between(rand_ks, rand_means - rand_stds, rand_means + rand_stds,
                                alpha=0.2, color="#1976D2")

        # Baseline reference
        ax.axhline(baseline * 100, color="black", linestyle=":", alpha=0.6,
                   label=f"Unmasked baseline ({baseline:.1%})")

        ax.set_xlabel("Number of tokens masked (k)")
        ax.set_ylabel(f"{title} accuracy (%)")
        ax.set_title(f"{title} — Cumulative Removal")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
        ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Cumulative removal curves (Step 2.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k",
    )
    parser.add_argument(
        "--knockout-dir",
        type=str,
        default="results/masking/knockout/count",
        help="Knockout dir (for baseline reuse)",
    )
    parser.add_argument(
        "--categorization",
        type=str,
        default="results/masking/categorization/count/categorization.csv",
        help="Categorization CSV (provides token-delta ordering)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir (default: <knockout-dir-parent>/cumulative/<knockout-dir-name>)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help="Cumulative k values to evaluate",
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=5,
        help="Number of random permutation seeds for null baseline",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Limit samples per benchmark (smoke testing)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    args = parser.parse_args()

    knockout_dir = Path(args.knockout_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else knockout_dir.parent.parent / "cumulative" / knockout_dir.name
    )

    run_cumulative(
        model_name=args.model,
        knockout_dir=knockout_dir,
        categorization_csv=Path(args.categorization),
        output_dir=output_dir,
        k_values=args.k_values,
        n_random_seeds=args.random_seeds,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

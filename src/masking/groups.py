"""Step 4: Semantic group masking — pool weak per-token signals.

Tests whether masking a coherent group of tokens (e.g., all "code identifier"
Pillars, or all top-10 raw Pillars) produces a measurable group-level effect
that single-token masking failed to detect at α=0.05.

Reuses the model loaded once for the full session.  Reads the deterministic
knockout results to define data-driven groups (top-N by Δ, cross-task
robust, etc.) and the categorization.csv for ranking.

Usage:
    uv run python src/masking/groups.py \\
        --knockout-dir results/masking/knockout/count \\
        --categorization results/masking/categorization/count/categorization.csv
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
    load_math_full,
    score_outputs as score_math500,
)


def load_token_table(categorization_csv: Path) -> list[dict]:
    """Load tokens with deterministic deltas and features."""
    with open(categorization_csv) as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "token_id": int(r["token_id"]),
            "token_string": r["token_string"],
            "frequency": int(r["frequency"]),
            "math500_delta": float(r["math500_delta"]),
            "ifeval_delta": float(r["ifeval_delta"]),
        }
        for r in rows
    ]


def find_token(tokens: list[dict], text: str) -> dict | None:
    """Find token by string match, whitespace-tolerant."""
    target = text.strip()
    for t in tokens:
        if t["token_string"].strip() == target:
            return t
    return None


def build_groups(tokens: list[dict], seed: int = 42) -> list[dict]:
    """Build the suite of token groups to test."""
    sorted_by_m500 = sorted(tokens, key=lambda t: t["math500_delta"])

    # Top-N raw Pillars (most negative M500 Δ)
    top5_pillar = sorted_by_m500[:5]
    top10_pillar = sorted_by_m500[:10]
    top20_pillar = sorted_by_m500[:20]

    # Top-N raw Stumbling Blocks (most positive M500 Δ)
    top5_sb = sorted_by_m500[-5:][::-1]
    top10_sb = sorted_by_m500[-10:][::-1]

    # Cross-task robust
    cross_pillars = [
        t for t in tokens
        if t["math500_delta"] <= -0.005 and t["ifeval_delta"] <= -0.005
    ]
    cross_sbs = [
        t for t in tokens
        if t["math500_delta"] >= 0.005 and t["ifeval_delta"] >= 0.005
    ]

    # Hypothesis-driven semantic clusters (from deterministic top-21 Pillars)
    code_tech_strs = [" Python", " Algorithm", " Regular", " Initialize", " maps"]
    domain_strs = [" mathematics", " transportation", " educational",
                   " financial", " local", " physical", " state"]
    discourse_strs = [" Important", " especially", " just", " allows"]
    abstract_strs = [" understanding", " issues", " movement", " situation"]
    modifier_strs = [" little", " detailed", " direct", " advanced", " outer", " clean"]

    def collect(strs):
        return [t for s in strs if (t := find_token(tokens, s))]

    code_tech = collect(code_tech_strs)
    domain = collect(domain_strs)
    discourse = collect(discourse_strs)
    abstract = collect(abstract_strs)
    modifiers = collect(modifier_strs)

    # Random controls — size-matched to top10 Pillar (10 tokens)
    rng = np.random.default_rng(seed)
    perm1 = rng.permutation(len(tokens))
    perm2 = rng.permutation(len(tokens))
    random_a = [tokens[i] for i in perm1[:10]]
    random_b = [tokens[i] for i in perm2[:10]]

    groups = [
        {"name": "top5_pillar", "description": "Top 5 raw Pillars (most negative M500 Δ)", "tokens": top5_pillar},
        {"name": "top10_pillar", "description": "Top 10 raw Pillars", "tokens": top10_pillar},
        {"name": "top20_pillar", "description": "Top 20 raw Pillars", "tokens": top20_pillar},
        {"name": "top5_stumbling", "description": "Top 5 raw Stumbling Blocks", "tokens": top5_sb},
        {"name": "top10_stumbling", "description": "Top 10 raw Stumbling Blocks", "tokens": top10_sb},
        {"name": "cross_task_pillars", "description": "Cross-task Pillars (Δ ≤ -0.5% on both)", "tokens": cross_pillars},
        {"name": "cross_task_stumbling", "description": "Cross-task Stumbling Blocks (Δ ≥ +0.5% on both)", "tokens": cross_sbs},
        {"name": "semantic_code_tech", "description": "Semantic: code/technical identifiers", "tokens": code_tech},
        {"name": "semantic_domain", "description": "Semantic: math/domain content nouns", "tokens": domain},
        {"name": "semantic_discourse", "description": "Semantic: discourse markers", "tokens": discourse},
        {"name": "semantic_abstract", "description": "Semantic: abstract nouns", "tokens": abstract},
        {"name": "semantic_modifiers", "description": "Semantic: modifiers/adjectives", "tokens": modifiers},
        {"name": "random_control_a", "description": "Random 10-token control (seed=42)", "tokens": random_a},
        {"name": "random_control_b", "description": "Random 10-token control (seed=42 variant)", "tokens": random_b},
    ]

    # Deduplicate within each group
    for g in groups:
        seen = set()
        unique = []
        for t in g["tokens"]:
            if t["token_id"] not in seen:
                seen.add(t["token_id"])
                unique.append(t)
        g["tokens"] = unique
        g["n_tokens"] = len(unique)

    return groups


def run_group(
    llm,
    math_convs, math_ds,
    ifeval_convs, ifeval_ds,
    token_ids: list[int],
    max_new_tokens: int,
    seed: int,
    skip_ifeval: bool = False,
):
    from vllm import SamplingParams

    if token_ids:
        sp = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            seed=seed,
            top_k=1,
            logit_bias={tid: -100.0 for tid in token_ids},
        )
    else:
        sp = default_sampling_params(max_tokens=max_new_tokens, seed=seed)

    m_outputs = llm.chat(math_convs, sp)
    m_result = score_math500(m_outputs, math_ds)

    if skip_ifeval:
        i_result = None
    else:
        i_outputs = llm.chat(ifeval_convs, sp)
        i_result = score_ifeval(i_outputs, ifeval_ds)
    return m_result, i_result


def run_all_groups(
    model_name: str,
    knockout_dir: Path,
    categorization_csv: Path,
    output_dir: Path,
    benchmark: str = "math500",
    skip_ifeval: bool = False,
    groups_filter: list[str] | None = None,
    n_samples: int | None = None,
    max_new_tokens: int = 4096,
    tensor_parallel_size: int | None = None,
    seed: int = SEED,
):
    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = output_dir / "groups"
    groups_dir.mkdir(exist_ok=True)

    # Load tokens & build groups
    tokens = load_token_table(categorization_csv)
    console.print(f"Loaded {len(tokens)} tokens from {categorization_csv}")
    groups = build_groups(tokens, seed=seed)
    if groups_filter:
        groups = [g for g in groups if g["name"] in groups_filter]
        console.print(f"Filtered to {len(groups)} groups: {[g['name'] for g in groups]}")
    console.print(f"Running {len(groups)} groups:")
    for g in groups:
        console.print(f"  {g['name']:>26s}  ({g['n_tokens']:>2d} tokens)  {g['description']}")

    # Datasets
    console.print("\n[bold]Loading datasets...[/bold]")
    if benchmark == "math_full":
        math_ds = load_math_full(n_samples)
        math_label = f"MATH-full ({len(math_ds)} problems)"
    else:
        math_ds = load_math500(n_samples)
        math_label = f"MATH-500 ({len(math_ds)} problems)"
    math_convs = build_math500_conversations(math_ds)
    console.print(f"  {math_label}")

    if skip_ifeval:
        ifeval_ds = None
        ifeval_convs = None
        console.print("  IF-Eval:  [yellow]skipped (math-only mode)[/yellow]")
    else:
        ifeval_ds = load_ifeval(n_samples)
        ifeval_convs = build_ifeval_conversations(ifeval_ds)
        console.print(f"  IF-Eval:  {len(ifeval_ds)} prompts")

    # Model
    console.print(f"\n[bold]Loading model:[/bold] {model_name}")
    llm = create_llm(model_name, tensor_parallel_size, seed=seed)

    # Baseline (in-session)
    baseline_path = output_dir / "baseline.json"
    if baseline_path.exists():
        console.print("\nUsing cached baseline")
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        console.print("\n[bold]Running baseline...[/bold]")
        start = time.time()
        m_result, i_result = run_group(
            llm, math_convs, math_ds, ifeval_convs, ifeval_ds,
            [], max_new_tokens, seed, skip_ifeval=skip_ifeval,
        )
        baseline = {
            "model": model_name,
            "benchmark": benchmark,
            "math500": m_result,
            "ifeval": i_result,
            "elapsed_s": round(time.time() - start, 1),
        }
        save_results(baseline_path, baseline)
    base_m = baseline["math500"]["accuracy"]
    base_i = baseline["ifeval"]["accuracy"] if baseline.get("ifeval") else None
    if base_i is not None:
        console.print(f"  Baseline: MATH={base_m:.1%}  IFE={base_i:.1%}")
    else:
        console.print(f"  Baseline: MATH={base_m:.1%}  (IF-Eval skipped)")

    # Run each group
    console.print(f"\n[bold]Running {len(groups)} group masks...[/bold]")
    rows = []

    with Progress(console=console) as progress:
        task = progress.add_task("Groups", total=len(groups))
        for g in groups:
            gpath = groups_dir / f"{g['name']}.json"
            if gpath.exists():
                with open(gpath) as f:
                    cached = json.load(f)
                row = {
                    "group": g["name"],
                    "n_tokens": g["n_tokens"],
                    "description": g["description"],
                    "math500_acc": cached["math500"]["accuracy"],
                    "math500_delta": cached["math500"]["accuracy"] - base_m,
                }
                if cached.get("ifeval") and base_i is not None:
                    row["ifeval_acc"] = cached["ifeval"]["accuracy"]
                    row["ifeval_delta"] = cached["ifeval"]["accuracy"] - base_i
                else:
                    row["ifeval_acc"] = None
                    row["ifeval_delta"] = None
                row["tokens"] = [t["token_string"] for t in g["tokens"]]
                rows.append(row)
                progress.update(task, advance=1)
                continue

            token_ids = [t["token_id"] for t in g["tokens"]]
            start = time.time()
            m_result, i_result = run_group(
                llm, math_convs, math_ds, ifeval_convs, ifeval_ds,
                token_ids, max_new_tokens, seed, skip_ifeval=skip_ifeval,
            )
            elapsed = time.time() - start

            m_delta = m_result["accuracy"] - base_m
            i_delta = (i_result["accuracy"] - base_i) if (i_result and base_i is not None) else None

            save_results(gpath, {
                "name": g["name"],
                "description": g["description"],
                "n_tokens": g["n_tokens"],
                "tokens": [{"token_id": t["token_id"], "token_string": t["token_string"]}
                           for t in g["tokens"]],
                "math500": {**m_result, "delta": m_delta},
                "ifeval": ({**i_result, "delta": i_delta} if i_result else None),
                "elapsed_s": round(elapsed, 1),
            })

            row = {
                "group": g["name"],
                "n_tokens": g["n_tokens"],
                "description": g["description"],
                "math500_acc": m_result["accuracy"],
                "math500_delta": m_delta,
                "ifeval_acc": i_result["accuracy"] if i_result else None,
                "ifeval_delta": i_delta,
                "tokens": [t["token_string"] for t in g["tokens"]],
            }
            rows.append(row)

            progress.update(
                task, advance=1,
                description=f"{g['name']} MATH Δ={m_delta:+.1%}",
            )

    # Save summary
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0].keys() if k != "tokens"])
        writer.writeheader()
        for r in rows:
            row_no_tokens = {k: v for k, v in r.items() if k != "tokens"}
            writer.writerow(row_no_tokens)

    save_results(output_dir / "summary.json", {
        "model": model_name,
        "benchmark": benchmark,
        "skip_ifeval": skip_ifeval,
        "baseline": {"math500": base_m, "ifeval": base_i},
        "groups": rows,
    })

    # Plot
    _plot_groups(rows, base_m, base_i, output_dir / "plots" / "groups.png", skip_ifeval=skip_ifeval)

    # Console summary
    console.print()
    table = Table(title=f"Group Mask Effects ({benchmark}, {'math-only' if skip_ifeval else 'math+ife'})")
    table.add_column("Group", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("MATH Δ", justify="right")
    if not skip_ifeval:
        table.add_column("IF-Eval Δ", justify="right")
    for r in rows:
        def color(d):
            if d is None:
                return "—"
            if d < 0:
                return f"[red]{d:+.2%}[/red]"
            if d > 0:
                return f"[green]{d:+.2%}[/green]"
            return f"{d:+.2%}"
        cols = [r["group"], str(r["n_tokens"]), color(r["math500_delta"])]
        if not skip_ifeval:
            cols.append(color(r["ifeval_delta"]))
        table.add_row(*cols)
    console.print(table)
    console.print(f"\nSaved: {csv_path}")


def _plot_groups(rows, base_m, base_i, output_path, skip_ifeval: bool = False):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [r["group"] for r in rows]
    m_deltas = [r["math500_delta"] * 100 for r in rows]
    y = np.arange(len(rows))

    if skip_ifeval:
        fig, ax = plt.subplots(1, 1, figsize=(8, max(5, len(rows) * 0.35)))
        axes = [ax]
        plot_specs = [(ax, m_deltas, "MATH Δ (%)")]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(rows) * 0.35)))
        i_deltas = [(r["ifeval_delta"] or 0) * 100 for r in rows]
        plot_specs = [
            (axes[0], m_deltas, "MATH Δ (%)"),
            (axes[1], i_deltas, "IF-Eval Δ (%)"),
        ]

    for ax, deltas, title in plot_specs:
        colors = ["#8B0000" if d < 0 else "#1B5E20" if d > 0 else "#9E9E9E" for d in deltas]
        ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        for i, d in enumerate(deltas):
            ax.text(d, i, f" {d:+.1f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=8)

    title = "Group Mask Effects" + (" on MATH" if skip_ifeval else " on MATH and IF-Eval")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Semantic group masking",
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
    )
    parser.add_argument(
        "--categorization",
        type=str,
        default="results/masking/categorization/count/categorization.csv",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["math500", "math_full"],
        default="math500",
        help="MATH benchmark size: math500 (500) or math_full (~5000)",
    )
    parser.add_argument(
        "--skip-ifeval",
        action="store_true",
        help="Skip IF-Eval evaluation (math-only mode)",
    )
    parser.add_argument(
        "--groups-filter",
        type=str,
        nargs="+",
        default=None,
        help="Run only these groups (default: all 14)",
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    knockout_dir = Path(args.knockout_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "groups" if args.benchmark == "math500" else f"groups_{args.benchmark}"
        output_dir = knockout_dir.parent.parent / suffix / knockout_dir.name

    run_all_groups(
        model_name=args.model,
        knockout_dir=knockout_dir,
        categorization_csv=Path(args.categorization),
        output_dir=output_dir,
        benchmark=args.benchmark,
        skip_ifeval=args.skip_ifeval,
        groups_filter=args.groups_filter,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

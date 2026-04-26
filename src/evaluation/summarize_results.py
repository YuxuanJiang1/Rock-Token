"""Summarize and compare results across multiple experiment runs.

Reads result directories (each produced by ``run_baseline.py``) and generates
comparison tables and plots.  Runs on CPU — no GPU required.

Usage:
    # Compare baseline vs. masked experiment:
    uv run python src/evaluation/summarize_results.py \
        --dirs results/baseline results/mask_stumbling \
        --labels "Baseline" "Mask Stumbling Blocks" \
        --output results/comparison

    # Single run summary:
    uv run python src/evaluation/summarize_results.py \
        --dirs results/baseline --output results/baseline
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

BENCHMARK_ORDER = ["gsm8k", "mmlu", "humaneval", "ifeval"]
BENCHMARK_DISPLAY = {
    "gsm8k": "GSM8K",
    "mmlu": "MMLU",
    "humaneval": "HumanEval",
    "ifeval": "IF-Eval",
}


def load_summary(result_dir: Path) -> dict | None:
    """Load summary.json from a results directory."""
    path = result_dir / "summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)

    # Fallback: read individual benchmark JSONs
    benchmarks = {}
    for name in BENCHMARK_ORDER:
        p = result_dir / f"{name}.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        entry = {
            "accuracy": meta.get("accuracy", 0),
            "correct": meta.get("correct", 0),
            "total": meta.get("total", 0),
        }
        if name == "ifeval" and "metrics" in data:
            entry.update(data["metrics"])
        benchmarks[name] = entry

    if not benchmarks:
        return None
    return {"model": "unknown", "benchmarks": benchmarks}


def print_comparison_table(
    console: Console,
    summaries: list[dict],
    labels: list[str],
) -> None:
    """Print a side-by-side comparison table."""
    table = Table(title="Benchmark Comparison")
    table.add_column("Benchmark", style="cyan")
    for label in labels:
        table.add_column(label, justify="right", style="bold")
    if len(labels) == 2:
        table.add_column("Delta", justify="right")

    for bench_name in BENCHMARK_ORDER:
        display = BENCHMARK_DISPLAY.get(bench_name, bench_name)
        row = [display]
        accs = []
        for s in summaries:
            stats = s.get("benchmarks", {}).get(bench_name)
            if stats is None:
                row.append("-")
                accs.append(None)
            else:
                acc = stats["accuracy"]
                if bench_name == "ifeval":
                    strict = stats.get("strict_prompt_accuracy", acc)
                    row.append(f"{strict:.1%}")
                    accs.append(strict)
                else:
                    row.append(f"{acc:.1%}")
                    accs.append(acc)

        # Delta column for 2-way comparison
        if len(labels) == 2:
            if all(a is not None for a in accs):
                delta = (accs[1] - accs[0]) * 100
                sign = "+" if delta >= 0 else ""
                color = "green" if delta > 0 else ("red" if delta < 0 else "white")
                row.append(f"[{color}]{sign}{delta:.1f}pp[/{color}]")
            else:
                row.append("-")

        table.add_row(*row)

    console.print()
    console.print(table)


def plot_comparison(
    output_dir: Path,
    summaries: list[dict],
    labels: list[str],
) -> Path | None:
    """Generate a grouped bar chart comparing experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    bench_names = []
    bench_display = []
    for name in BENCHMARK_ORDER:
        # Only include benchmarks present in at least one summary
        if any(name in s.get("benchmarks", {}) for s in summaries):
            bench_names.append(name)
            bench_display.append(BENCHMARK_DISPLAY.get(name, name))

    n_benches = len(bench_names)
    n_runs = len(summaries)
    x = np.arange(n_benches)
    width = 0.8 / n_runs

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (summary, label) in enumerate(zip(summaries, labels)):
        accs = []
        for name in bench_names:
            stats = summary.get("benchmarks", {}).get(name)
            if stats is None:
                accs.append(0)
            elif name == "ifeval":
                accs.append(stats.get("strict_prompt_accuracy", stats["accuracy"]) * 100)
            else:
                accs.append(stats["accuracy"] * 100)

        offset = (i - n_runs / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=label, color=colors[i % len(colors)])
        # Value labels
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{acc:.1f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(bench_display)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.set_title("Benchmark Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / "comparison_chart.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def save_comparison_json(
    output_dir: Path,
    summaries: list[dict],
    labels: list[str],
) -> Path:
    """Save a unified comparison JSON."""
    comparison = {
        "experiments": {
            label: {
                "model": s.get("model", "unknown"),
                "benchmarks": s.get("benchmarks", {}),
            }
            for label, s in zip(labels, summaries)
        }
    }
    path = output_dir / "comparison.json"
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and compare evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dirs", type=str, nargs="+", required=True,
        help="Result directories to compare",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Labels for each directory (default: directory names)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory to save comparison outputs (default: first dir)",
    )
    args = parser.parse_args()

    console = Console()

    dirs = [Path(d) for d in args.dirs]
    labels = args.labels or [d.name for d in dirs]
    if len(labels) != len(dirs):
        console.print("[bold red]Number of labels must match number of dirs[/bold red]")
        return

    output_dir = Path(args.output) if args.output else dirs[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all summaries
    summaries = []
    for d, label in zip(dirs, labels):
        s = load_summary(d)
        if s is None:
            console.print(f"[yellow]Warning: no results found in {d}[/yellow]")
            continue
        summaries.append(s)

    if not summaries:
        console.print("[bold red]No results to summarize[/bold red]")
        return

    # Print comparison table
    print_comparison_table(console, summaries, labels)

    # Save comparison JSON
    json_path = save_comparison_json(output_dir, summaries, labels)
    console.print(f"\nComparison JSON saved to {json_path}")

    # Plot
    chart_path = plot_comparison(output_dir, summaries, labels)
    if chart_path:
        console.print(f"Comparison chart saved to {chart_path}")


if __name__ == "__main__":
    main()

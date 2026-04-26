"""Run all benchmarks on a model and produce a structured results directory.

Each benchmark runs as a **separate subprocess** so vLLM releases GPU memory
cleanly between runs.  After all benchmarks finish the script writes a
``summary.json`` and (if matplotlib is available) an accuracy bar chart.

Usage:
    uv run python src/evaluation/run_baseline.py \
        --model RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k \
        --output-dir results/baseline

    # Quick smoke-test (10 samples per benchmark):
    uv run python src/evaluation/run_baseline.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --output-dir results/smoke --n-samples 10

Output structure:
    <output-dir>/
        gsm8k.json
        mmlu.json
        humaneval.json
        ifeval.json
        summary.json
        accuracy_chart.png
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

BENCHMARKS = [
    {
        "name": "gsm8k",
        "script": "src/evaluation/eval_gsm8k.py",
        "extra_args": [],
    },
    {
        "name": "mmlu",
        "script": "src/evaluation/eval_mmlu.py",
        "extra_args": ["--max-new-tokens", "64"],
    },
    {
        "name": "humaneval",
        "script": "src/evaluation/eval_humaneval.py",
        "extra_args": ["--max-new-tokens", "1024"],
    },
    {
        "name": "ifeval",
        "script": "src/evaluation/eval_ifeval.py",
        "extra_args": [],
    },
]


def run_benchmark(
    bench: dict,
    model: str,
    output_dir: Path,
    n_samples: int | None,
    tensor_parallel: int | None,
    seed: int,
    max_new_tokens: int | None,
    console: Console,
) -> tuple[bool, float]:
    """Run a single benchmark in a subprocess. Returns (success, elapsed_s)."""
    out_path = output_dir / f"{bench['name']}.json"

    cmd = [
        sys.executable, bench["script"],
        "--model", model,
        "--output", str(out_path),
        "--seed", str(seed),
    ]
    if n_samples is not None:
        cmd += ["--n-samples", str(n_samples)]
    if tensor_parallel is not None:
        cmd += ["--tensor-parallel", str(tensor_parallel)]
    if max_new_tokens is not None:
        cmd += ["--max-new-tokens", str(max_new_tokens)]
    else:
        cmd += bench["extra_args"]

    console.print(f"\n{'=' * 60}")
    console.print(f"[bold cyan]Running {bench['name'].upper()}[/bold cyan]")
    console.print(f"  cmd: {' '.join(cmd)}")
    console.print(f"{'=' * 60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        console.print(f"[bold red]{bench['name']} FAILED (exit {result.returncode})[/bold red]")
        return False, elapsed

    console.print(f"[bold green]{bench['name']} completed in {elapsed:.0f}s[/bold green]")
    return True, elapsed


def build_summary(output_dir: Path, model: str, seed: int) -> dict:
    """Read individual result JSONs and build a unified summary."""
    benchmarks = {}
    for bench in BENCHMARKS:
        path = output_dir / f"{bench['name']}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        entry = {
            "accuracy": meta.get("accuracy", 0),
            "correct": meta.get("correct", 0),
            "total": meta.get("total", 0),
        }

        # IF-Eval has multiple metrics
        if bench["name"] == "ifeval" and "metrics" in data:
            entry.update(data["metrics"])

        benchmarks[bench["name"]] = entry

    return {
        "model": model,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": benchmarks,
    }


def print_summary_table(console: Console, summary: dict) -> None:
    """Print a rich summary table."""
    table = Table(title=f"Baseline Results — {summary['model']}")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Correct / Total", justify="right")

    for name, stats in summary["benchmarks"].items():
        acc = stats["accuracy"]
        label = f"{acc:.1%}"
        detail = f"{stats['correct']}/{stats['total']}"

        # IF-Eval: show all 4 metrics
        if name == "ifeval":
            strict_p = stats.get("strict_prompt_accuracy", acc)
            loose_p = stats.get("loose_prompt_accuracy", acc)
            label = f"strict={strict_p:.1%}  loose={loose_p:.1%}"

        table.add_row(name.upper(), label, detail)

    console.print()
    console.print(table)


def plot_summary(output_dir: Path, summary: dict) -> None:
    """Generate an accuracy bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = []
    accs = []
    for name, stats in summary["benchmarks"].items():
        names.append(name.upper())
        accs.append(stats["accuracy"] * 100)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, accs, color="#4C72B0", edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"Baseline — {summary['model']}", fontsize=12, fontweight="bold")

    # Value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%", va="center", fontsize=10,
        )

    plt.tight_layout()
    chart_path = output_dir / "accuracy_chart.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmarks on a model (baseline measurement)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for results")
    parser.add_argument("--n-samples", type=int, default=None, help="Limit samples per benchmark")
    parser.add_argument("--tensor-parallel", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max tokens for all benchmarks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--benchmarks", type=str, nargs="+",
        default=None,
        choices=["gsm8k", "mmlu", "humaneval", "ifeval"],
        help="Run only these benchmarks (default: all)",
    )
    args = parser.parse_args()

    console = Console()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Baseline evaluation[/bold]")
    console.print(f"  Model:      {args.model}")
    console.print(f"  Output dir: {output_dir}")
    console.print(f"  Seed:       {args.seed}")
    if args.n_samples:
        console.print(f"  N-samples:  {args.n_samples}")

    # Select benchmarks
    to_run = BENCHMARKS
    if args.benchmarks:
        selected = set(args.benchmarks)
        to_run = [b for b in BENCHMARKS if b["name"] in selected]

    # Run each benchmark
    total_start = time.time()
    statuses = {}
    for bench in to_run:
        ok, elapsed = run_benchmark(
            bench, args.model, output_dir,
            args.n_samples, args.tensor_parallel, args.seed,
            args.max_new_tokens, console,
        )
        statuses[bench["name"]] = {"success": ok, "elapsed_s": elapsed}

    total_elapsed = time.time() - total_start

    # Summary
    summary = build_summary(output_dir, args.model, args.seed)
    summary["run_info"] = {
        "total_elapsed_s": total_elapsed,
        "per_benchmark": statuses,
    }

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print table
    print_summary_table(console, summary)

    # Plot
    chart_path = plot_summary(output_dir, summary)
    if chart_path:
        console.print(f"\nChart saved to {chart_path}")

    console.print(f"\n[bold]Total time: {total_elapsed:.0f}s[/bold]")
    console.print(f"Summary saved to {summary_path}")

    # Return non-zero if any benchmark failed
    if any(not s["success"] for s in statuses.values()):
        console.print("[bold red]Some benchmarks failed — check output above[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

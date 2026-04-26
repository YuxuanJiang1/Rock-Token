"""Run baseline evaluation across all models and benchmarks.

Spawns each benchmark as a subprocess for GPU memory isolation.
Reads model list from config.yaml.

Usage:
    uv run python src/masking/run_baseline.py --output-dir results/masking/baseline
    uv run python src/masking/run_baseline.py --output-dir results/smoke --n-samples 5
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


BENCHMARKS = [
    {
        "name": "math500",
        "script": "src/masking/eval_math500.py",
        "extra_args": [],
    },
    {
        "name": "aime2024",
        "script": "src/masking/eval_aime.py",
        "extra_args": ["--year", "2024"],
    },
    {
        "name": "aime2025",
        "script": "src/masking/eval_aime.py",
        "extra_args": ["--year", "2025"],
    },
    {
        "name": "hmmt_feb_2025",
        "script": "src/masking/eval_hmmt.py",
        "extra_args": [],
    },
    {
        "name": "ifeval",
        "script": "src/masking/eval_ifeval.py",
        "extra_args": [],
    },
]

MODEL_KEYS = ["teacher", "student_base", "student_onpolicy", "student_offpolicy"]


def run_benchmark(
    script: str,
    model_name: str,
    output_path: str,
    extra_args: list[str],
    n_samples: int | None = None,
    max_new_tokens: int = 4096,
    tensor_parallel: int | None = None,
) -> tuple[bool, float]:
    """Run a single benchmark as a subprocess. Returns (success, elapsed_seconds)."""
    cmd = [
        sys.executable, script,
        "--model", model_name,
        "--output", output_path,
        "--max-new-tokens", str(max_new_tokens),
        *extra_args,
    ]
    if n_samples is not None:
        cmd.extend(["--n-samples", str(n_samples)])
    if tensor_parallel is not None:
        cmd.extend(["--tensor-parallel", str(tensor_parallel)])

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    return result.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation across all models and benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/masking/baseline",
        help="Directory to save results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Limit samples per benchmark (for smoke testing)",
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
        help="Override tensor parallel size",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help=f"Subset of model keys to run (default: all — {MODEL_KEYS})",
    )
    args = parser.parse_args()

    console = Console()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_keys = args.models or MODEL_KEYS
    models = {}
    for key in model_keys:
        if key in config["models"]:
            models[key] = config["models"][key]
        else:
            console.print(f"[yellow]Warning: model key '{key}' not in config, skipping[/yellow]")

    console.print("[bold]Baseline evaluation[/bold]")
    console.print(f"  Models: {list(models.keys())}")
    console.print(f"  Benchmarks: {[b['name'] for b in BENCHMARKS]}")
    console.print(f"  Output: {output_dir}")
    if args.n_samples:
        console.print(f"  [yellow]Smoke mode: {args.n_samples} samples per benchmark[/yellow]")
    console.print()

    summary = {}
    for model_key, model_name in models.items():
        console.rule(f"[bold]{model_key}[/bold]: {model_name}")
        model_dir = output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        summary[model_key] = {"model": model_name, "benchmarks": {}}

        for bench in BENCHMARKS:
            out_path = str(model_dir / f"{bench['name']}.json")
            console.print(f"  Running {bench['name']}...")

            success, elapsed = run_benchmark(
                script=bench["script"],
                model_name=model_name,
                output_path=out_path,
                extra_args=bench["extra_args"],
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                tensor_parallel=args.tensor_parallel,
            )

            if success and Path(out_path).exists():
                with open(out_path) as f:
                    result_data = json.load(f)
                acc = result_data["metadata"]["accuracy"]
                summary[model_key]["benchmarks"][bench["name"]] = {
                    "accuracy": acc,
                    "correct": result_data["metadata"]["correct"],
                    "total": result_data["metadata"]["total"],
                    "elapsed_s": round(elapsed, 1),
                }
                console.print(f"    {bench['name']}: {acc:.1%} ({elapsed:.0f}s)")
            else:
                summary[model_key]["benchmarks"][bench["name"]] = {
                    "accuracy": None,
                    "error": True,
                    "elapsed_s": round(elapsed, 1),
                }
                console.print(f"    [red]{bench['name']}: FAILED ({elapsed:.0f}s)[/red]")

    # Summary table
    console.print()
    table = Table(title="Baseline Results Summary")
    table.add_column("Model", style="cyan")
    for bench in BENCHMARKS:
        table.add_column(bench["name"], justify="right")

    for model_key, data in summary.items():
        row = [model_key]
        for bench in BENCHMARKS:
            b_data = data["benchmarks"].get(bench["name"], {})
            acc = b_data.get("accuracy")
            if acc is not None:
                row.append(f"{acc:.1%}")
            else:
                row.append("[red]ERR[/red]")
        table.add_row(*row)

    console.print(table)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

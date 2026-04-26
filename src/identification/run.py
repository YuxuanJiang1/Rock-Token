"""CLI entry point: orchestrate the full Rock Token identification pipeline.

Usage:
    uv run python src/identification/run.py                          # on-policy (default)
    uv run python src/identification/run.py --variant offpolicy      # off-policy
    uv run python src/identification/run.py --phase 1                # generation only
    uv run python src/identification/run.py --phase 3                # classify + plots (CPU)
    uv run python src/identification/run.py --config my.yaml         # custom config
"""

import argparse
from pathlib import Path

from rich.console import Console

from src.identification.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Rock Token identification pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: project root config.yaml)",
    )
    parser.add_argument(
        "--phase", type=int, default=None, choices=[1, 2, 3],
        help="Run only this phase (default: run all phases)",
    )
    parser.add_argument(
        "--variant", type=str, default="onpolicy",
        choices=["onpolicy", "offpolicy"],
        help="Which distilled student to use (reads from config.yaml models section)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/identification/<variant>)",
    )
    args = parser.parse_args()

    console = Console()
    config = load_config(args.config)

    # Resolve variant: point student_onpolicy at the chosen model
    variant_key = f"student_{args.variant}"
    config["models"]["student_onpolicy"] = config["models"][variant_key]

    # Auto output dir based on variant
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/identification/{args.variant}")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Rock Token Identification Pipeline[/bold]")
    console.print(f"  Config: {args.config or 'config.yaml (default)'}")
    console.print(f"  Variant: {args.variant}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Teacher: {config['models']['teacher']}")
    console.print(f"  theta_0: {config['models']['student_base']}")
    console.print(f"  theta*: {config['models']['student_onpolicy']}")
    console.print()

    phases = [args.phase] if args.phase else [1, 2, 3]

    if 1 in phases:
        console.print("[bold]=== Phase 1: Generate Student Outputs ===[/bold]")
        from src.identification.generate import run_phase1
        run_phase1(config, output_dir)

    if 2 in phases:
        console.print("[bold]=== Phase 2: Measure KL at Both Checkpoints ===[/bold]")
        from src.identification.measure import run_phase2
        run_phase2(config, output_dir)

    if 3 in phases:
        console.print("[bold]=== Phase 3+4: Classify & Aggregate ===[/bold]")
        from src.identification.identify import run_identification
        run_identification(config, output_dir)

        console.print("[bold]=== Sanity Check Plots ===[/bold]")
        from src.identification.plots import run_plots
        run_plots(config, output_dir)

    console.print("\n[bold green]Pipeline complete.[/bold green]")
    console.print(f"Results in {output_dir}/")


if __name__ == "__main__":
    main()

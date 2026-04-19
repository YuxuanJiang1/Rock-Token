# src/exp_2/identify_rock_tokens.py
"""Rock Token Identification Pipeline.

Three-phase pipeline to identify Rock Tokens from OPD-trained student/teacher models:
  Phase 1: Student generation — generate responses, save per-token log-probs
  Phase 2: Teacher forward — compute KL divergence and teacher entropy
  Phase 3: Analysis — aggregate, score, classify, output results

Only one model is loaded at a time for GPU efficiency.
Each phase saves intermediate results; use --phase to resume from a specific phase.
"""

import argparse
from pathlib import Path

from src.exp_2.phases import run_phase1, run_phase2, run_phase3


def determine_start_phase(output_dir: Path) -> int:
    """Determine which phase to resume from based on existing output files."""
    phase2_dir = output_dir / "phase2_data"
    student_data_dir = output_dir / "student_data"

    if phase2_dir.exists() and any(phase2_dir.glob("sample_*.pt")):
        return 3
    if student_data_dir.exists() and any(student_data_dir.glob("sample_*.pt")):
        return 2
    return 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify Rock Tokens from OPD-trained student/teacher models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--student",
        type=str,
        required=True,
        help="HuggingFace model ID for the OPD-trained student",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="HuggingFace model ID for the teacher model",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        choices=["geometric", "bayesian"],
        default="bayesian",
        help="Rock Token scoring method",
    )
    parser.add_argument("--alpha", type=float, default=0.3, help="Geometric: frequency exponent")
    parser.add_argument("--beta", type=float, default=0.7, help="Geometric: KL exponent")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top Rock Tokens to output")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per sample in Phase 1",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/exp2",
        help="Directory for all outputs",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Force restart from this phase (default: auto-resume)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine start phase
    if args.phase is not None:
        start_phase = args.phase
        print(f"Forced start from Phase {start_phase}")
    else:
        start_phase = determine_start_phase(output_dir)
        if start_phase > 1:
            print(f"Auto-resuming from Phase {start_phase}")

    # Phase 1: Student generation
    if start_phase <= 1:
        run_phase1(args.student, output_dir, args.max_new_tokens, args.n_samples)

    # Phase 2: Teacher KL computation
    if start_phase <= 2:
        run_phase2(args.teacher, output_dir)

    # Phase 3: Analysis and output
    run_phase3(
        output_dir=output_dir,
        tokenizer_name=args.student,
        scoring_method=args.scoring,
        alpha=args.alpha,
        beta=args.beta,
        top_k=args.top_k,
        student_model=args.student,
        teacher_model=args.teacher,
    )


if __name__ == "__main__":
    main()

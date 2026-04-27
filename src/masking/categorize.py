"""Statistical categorization of Rock Token knockout effects (Step 2.2).

For each token, runs a paired bootstrap on its per_correct vector against
the baseline to compute a 95% CI and two-sided p-value.  Classifies tokens
into Strong/Weak Pillar / Neutral / Weak/Strong Stumbling Block on each
benchmark.  Correlates classifications with Part 1 token features.

CPU-only.  Operates on per_correct lists already saved by knockout.py.

Usage:
    uv run python src/masking/categorize.py \\
        --knockout-dir results/masking/knockout/count \\
        --rock-tokens results/identification/onpolicy/rock_tokens_by_count.csv
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

CATEGORIES = [
    "Strong Pillar",
    "Weak Pillar",
    "Neutral",
    "Weak Stumbling Block",
    "Strong Stumbling Block",
]

CATEGORY_COLORS = {
    "Strong Pillar": "#8B0000",
    "Weak Pillar": "#E57373",
    "Neutral": "#9E9E9E",
    "Weak Stumbling Block": "#81C784",
    "Strong Stumbling Block": "#1B5E20",
}


def paired_bootstrap(
    baseline_correct: list[bool],
    masked_correct: list[bool],
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Paired bootstrap on accuracy difference.

    Returns (observed_delta, ci_lo, ci_hi, p_value_two_sided).
    """
    rng = np.random.default_rng(seed)
    b = np.asarray(baseline_correct, dtype=float)
    m = np.asarray(masked_correct, dtype=float)
    n = len(b)
    assert len(m) == n, f"Length mismatch: {n} vs {len(m)}"

    observed = m.mean() - b.mean()

    indices = rng.integers(0, n, size=(n_resamples, n))
    b_accs = b[indices].mean(axis=1)
    m_accs = m[indices].mean(axis=1)
    deltas = m_accs - b_accs

    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_pos = float((deltas >= 0).mean())
    p_neg = float((deltas <= 0).mean())
    p_value = 2 * min(p_pos, p_neg)

    return float(observed), float(ci_lo), float(ci_hi), float(p_value)


def classify(delta: float, p_value: float, epsilon: float = 0.01, alpha: float = 0.05) -> str:
    """5-way classification."""
    if p_value >= alpha:
        return "Neutral"
    if delta <= -epsilon:
        return "Strong Pillar"
    if delta < 0:
        return "Weak Pillar"
    if delta >= epsilon:
        return "Strong Stumbling Block"
    if delta > 0:
        return "Weak Stumbling Block"
    return "Neutral"


def load_token_features(rock_tokens_csv: Path) -> dict[int, dict]:
    """Load per-token features from Part 1 CSV."""
    features = {}
    with open(rock_tokens_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["token_id"])
            features[tid] = {
                "token_string": row["token_string"],
                "frequency": int(row["frequency"]),
                "rock_count": int(row["rock_count"]),
                "rock_rate": float(row["rock_rate"]),
                "avg_loss_before": float(row["avg_loss_before"]),
                "avg_loss_after": float(row["avg_loss_after"]),
                "avg_improvement": float(row["avg_improvement"]),
                "avg_teacher_entropy": float(row["avg_teacher_entropy"]),
                "avg_student_entropy_before": float(row["avg_student_entropy_before"]),
                "avg_student_entropy_after": float(row["avg_student_entropy_after"]),
            }
    return features


def run_categorization(
    knockout_dir: Path,
    rock_tokens_csv: Path,
    output_dir: Path,
    epsilon: float = 0.01,
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 42,
):
    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load baseline
    with open(knockout_dir / "baseline.json") as f:
        baseline = json.load(f)
    base_m500 = baseline["math500"]["per_correct"]
    base_ife = baseline["ifeval"]["per_correct"]
    console.print(f"[bold]Baseline:[/bold] MATH-500 {sum(base_m500)}/{len(base_m500)}, IF-Eval {sum(base_ife)}/{len(base_ife)}")

    # Load token features
    features = load_token_features(rock_tokens_csv)
    console.print(f"Loaded features for {len(features)} tokens from Part 1 CSV")

    # Load per-token knockout results
    tokens_dir = knockout_dir / "tokens"
    token_files = sorted(tokens_dir.glob("token_*.json"))
    console.print(f"Loaded {len(token_files)} per-token knockout results")

    # Run bootstrap + classify
    console.print(
        f"\n[bold]Running paired bootstrap[/bold] "
        f"(n_resamples={n_resamples}, ε={epsilon:.0%}, α={alpha})..."
    )
    rows = []
    for tf in token_files:
        with open(tf) as f:
            data = json.load(f)
        tid = data["token_id"]

        m_observed, m_ci_lo, m_ci_hi, m_p = paired_bootstrap(
            base_m500, data["math500"]["per_correct"], n_resamples, seed
        )
        i_observed, i_ci_lo, i_ci_hi, i_p = paired_bootstrap(
            base_ife, data["ifeval"]["per_correct"], n_resamples, seed
        )

        m_cat = classify(m_observed, m_p, epsilon, alpha)
        i_cat = classify(i_observed, i_p, epsilon, alpha)

        feat = features.get(tid, {})
        rows.append({
            "token_id": tid,
            "token_string": data["token_string"],
            "frequency": data["frequency"],
            "rock_count": data["rock_count"],
            "rock_rate": data["rock_rate"],
            "avg_loss_before": feat.get("avg_loss_before"),
            "avg_loss_after": feat.get("avg_loss_after"),
            "avg_improvement": feat.get("avg_improvement"),
            "avg_teacher_entropy": feat.get("avg_teacher_entropy"),
            "avg_student_entropy_before": feat.get("avg_student_entropy_before"),
            "avg_student_entropy_after": feat.get("avg_student_entropy_after"),
            "math500_delta": m_observed,
            "math500_ci_lo": m_ci_lo,
            "math500_ci_hi": m_ci_hi,
            "math500_p": m_p,
            "math500_category": m_cat,
            "ifeval_delta": i_observed,
            "ifeval_ci_lo": i_ci_lo,
            "ifeval_ci_hi": i_ci_hi,
            "ifeval_p": i_p,
            "ifeval_category": i_cat,
        })

    # --- Save categorization CSV ---
    csv_path = output_dir / "categorization.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # --- Category counts ---
    m_counts = {c: 0 for c in CATEGORIES}
    i_counts = {c: 0 for c in CATEGORIES}
    for r in rows:
        m_counts[r["math500_category"]] += 1
        i_counts[r["ifeval_category"]] += 1

    # --- Cross-task agreement matrix ---
    agreement = {}
    for r in rows:
        key = (r["math500_category"], r["ifeval_category"])
        agreement[key] = agreement.get(key, 0) + 1

    # --- Feature correlations ---
    feature_keys = [
        "frequency", "rock_count", "rock_rate",
        "avg_loss_before", "avg_loss_after", "avg_improvement",
        "avg_teacher_entropy", "avg_student_entropy_before", "avg_student_entropy_after",
    ]
    correlations = {}
    valid = [r for r in rows if r["avg_teacher_entropy"] is not None]
    m_deltas = np.array([r["math500_delta"] for r in valid])
    i_deltas = np.array([r["ifeval_delta"] for r in valid])
    for fk in feature_keys:
        vals = np.array([r[fk] for r in valid], dtype=float)
        correlations[fk] = {
            "math500_pearson_r": float(np.corrcoef(vals, m_deltas)[0, 1]),
            "ifeval_pearson_r": float(np.corrcoef(vals, i_deltas)[0, 1]),
        }

    # --- Save summary JSON ---
    summary = {
        "config": {
            "epsilon": epsilon,
            "alpha": alpha,
            "n_resamples": n_resamples,
            "n_tokens": len(rows),
            "knockout_dir": str(knockout_dir),
        },
        "baseline": {
            "math500": sum(base_m500) / len(base_m500),
            "ifeval": sum(base_ife) / len(base_ife),
        },
        "math500_categories": m_counts,
        "ifeval_categories": i_counts,
        "cross_task_agreement": {
            f"{m}|{i}": v for (m, i), v in agreement.items()
        },
        "feature_correlations": correlations,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Console output ---
    console.print()
    table = Table(title="Category Counts")
    table.add_column("Category", style="cyan")
    table.add_column("MATH-500", justify="right")
    table.add_column("IF-Eval", justify="right")
    for cat in CATEGORIES:
        table.add_row(cat, str(m_counts[cat]), str(i_counts[cat]))
    console.print(table)

    console.print()
    table = Table(title=f"Feature Correlations with Δ (Pearson r, n={len(valid)})")
    table.add_column("Feature", style="cyan")
    table.add_column("MATH-500 Δ", justify="right")
    table.add_column("IF-Eval Δ", justify="right")
    for fk in feature_keys:
        c = correlations[fk]
        table.add_row(
            fk,
            f"{c['math500_pearson_r']:+.3f}",
            f"{c['ifeval_pearson_r']:+.3f}",
        )
    console.print(table)

    # --- Plots ---
    _plot_delta_histograms(rows, plots_dir / "delta_histograms.png")
    _plot_cross_task_scatter(rows, plots_dir / "cross_task_scatter.png")
    _plot_feature_correlations(rows, feature_keys, plots_dir / "feature_correlations.png")

    console.print(f"\nSaved: {csv_path}")
    console.print(f"Saved: {output_dir / 'summary.json'}")
    console.print(f"Saved: plots in {plots_dir}/")


def _plot_delta_histograms(rows, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title in [
        (axes[0], "math500", "MATH-500"),
        (axes[1], "ifeval", "IF-Eval"),
    ]:
        for cat in CATEGORIES:
            vals = [
                r[f"{key}_delta"] * 100
                for r in rows
                if r[f"{key}_category"] == cat
            ]
            if vals:
                ax.hist(
                    vals,
                    bins=np.arange(-4, 2.5, 0.25),
                    alpha=0.7,
                    label=f"{cat} (n={len(vals)})",
                    color=CATEGORY_COLORS[cat],
                )
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Δ accuracy (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title} — Knockout Δ by Category")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_task_scatter(rows, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    for cat in CATEGORIES:
        m = [r["math500_delta"] * 100 for r in rows if r["math500_category"] == cat]
        i = [r["ifeval_delta"] * 100 for r in rows if r["math500_category"] == cat]
        ax.scatter(m, i, alpha=0.7, label=f"{cat} (n={len(m)})",
                   color=CATEGORY_COLORS[cat], s=40, edgecolor="black", linewidth=0.3)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("MATH-500 Δ (%)")
    ax.set_ylabel("IF-Eval Δ (%)")
    ax.set_title("Cross-task knockout effects (color = MATH-500 category)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # Add annotations for extreme tokens
    sorted_rows = sorted(rows, key=lambda r: r["math500_delta"])
    for r in sorted_rows[:5] + sorted_rows[-5:]:
        ax.annotate(
            r["token_string"].strip(),
            (r["math500_delta"] * 100, r["ifeval_delta"] * 100),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_correlations(rows, feature_keys, output_path):
    valid = [r for r in rows if r["avg_teacher_entropy"] is not None]
    if not valid:
        return

    n = len(feature_keys)
    cols = 3
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 5, nrows * 4))
    axes = axes.flatten() if nrows > 1 else [axes] if cols == 1 else axes.flatten()

    m_deltas = np.array([r["math500_delta"] * 100 for r in valid])

    for i, fk in enumerate(feature_keys):
        ax = axes[i]
        vals = np.array([r[fk] for r in valid], dtype=float)
        colors = [CATEGORY_COLORS[r["math500_category"]] for r in valid]
        ax.scatter(vals, m_deltas, c=colors, alpha=0.6, s=30, edgecolor="black", linewidth=0.2)

        r_pearson = np.corrcoef(vals, m_deltas)[0, 1]
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel(fk)
        ax.set_ylabel("MATH-500 Δ (%)")
        ax.set_title(f"{fk}\nPearson r = {r_pearson:+.3f}")
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Token Features vs MATH-500 Knockout Δ", fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Statistical categorization of knockout results (Step 2.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--knockout-dir",
        type=str,
        default="results/masking/knockout/count",
        help="Knockout results directory (must contain baseline.json + tokens/)",
    )
    parser.add_argument(
        "--rock-tokens",
        type=str,
        default="results/identification/onpolicy/rock_tokens_by_count.csv",
        help="Rock tokens CSV from Part 1 (for token features)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: parent of knockout-dir + /categorization)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Strong threshold (Δ magnitude required for Strong category)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for Neutral classification",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples per token",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap",
    )
    args = parser.parse_args()

    knockout_dir = Path(args.knockout_dir)
    output_dir = Path(args.output_dir) if args.output_dir else knockout_dir.parent.parent / "categorization" / knockout_dir.name

    run_categorization(
        knockout_dir=knockout_dir,
        rock_tokens_csv=Path(args.rock_tokens),
        output_dir=output_dir,
        epsilon=args.epsilon,
        alpha=args.alpha,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

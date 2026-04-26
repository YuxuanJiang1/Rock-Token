"""Sanity check plots for Rock Token identification.

All plots are saved as PNG files in the output directory.
Can be run on CPU from Phase 2 data — no models needed.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from scipy import stats as scipy_stats


def plot_loss_scatter(data: dict, config: dict, output_dir: Path) -> Path:
    """Scatter plot of loss_before vs loss_after with color-coded quadrants."""
    ic = config["identification"]
    lb = data["loss_before"].numpy()
    la = data["loss_after"].numpy()

    tau_high = float(np.percentile(lb, ic["tau_high_percentile"]))
    improvement = lb - la
    delta = float(np.percentile(improvement, ic["delta_percentile"]))

    # Subsample for plotting if too many points
    n = len(lb)
    if n > 50_000:
        idx = np.random.default_rng(42).choice(n, 50_000, replace=False)
        lb_plot, la_plot = lb[idx], la[idx]
    else:
        lb_plot, la_plot = lb, la

    # Assign quadrant colors
    colors = np.full(len(lb_plot), "#2ecc71", dtype=object)  # Easy (green)
    for i in range(len(lb_plot)):
        high_before = lb_plot[i] >= tau_high
        low_improve = (lb_plot[i] - la_plot[i]) < delta
        if high_before and not low_improve:
            colors[i] = "#3498db"  # Learned (blue)
        elif high_before and low_improve:
            colors[i] = "#e74c3c"  # Rock (red)
        elif not high_before and la_plot[i] > lb_plot[i]:
            colors[i] = "#e67e22"  # Regressed (orange)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(lb_plot, la_plot, c=colors, alpha=0.15, s=3, rasterized=True)

    # Threshold lines
    ax.axvline(tau_high, color="gray", linestyle="--", linewidth=0.8, label=f"tau_high={tau_high:.2f}")
    # delta boundary line: la = lb - delta
    x_line = np.linspace(tau_high, lb.max(), 100)
    ax.plot(x_line, x_line - delta, color="gray", linestyle=":", linewidth=0.8, label=f"delta={delta:.2f}")
    # Identity line
    lim = max(lb.max(), la.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("loss_before (KL with theta_0)")
    ax.set_ylabel("loss_after (KL with theta*)")
    ax.set_title("Per-Token Loss: Before vs After OPD")
    ax.legend(fontsize=8)

    # Quadrant labels
    mid_hi = (tau_high + lb.max()) / 2
    mid_lo = tau_high / 2
    ax.text(mid_lo, mid_lo * 0.5, "Easy", color="#2ecc71", fontsize=12, fontweight="bold", alpha=0.7)
    ax.text(mid_hi, mid_lo * 0.5, "Learned", color="#3498db", fontsize=12, fontweight="bold", alpha=0.7)
    ax.text(mid_hi, mid_hi, "Rock", color="#e74c3c", fontsize=12, fontweight="bold", alpha=0.7)

    plt.tight_layout()
    path = output_dir / "scatter_loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_entropy_correlation(
    rock_tokens: list[dict], output_dir: Path
) -> Path | None:
    """Rock rate vs student entropy per token type — tests forking token hypothesis."""
    rates = [t["rock_rate"] for t in rock_tokens if "avg_student_entropy_before" in t]
    entropies = [t["avg_student_entropy_before"] for t in rock_tokens if "avg_student_entropy_before" in t]

    if not rates:
        return None

    rates = np.array(rates)
    entropies = np.array(entropies)

    r, p_val = scipy_stats.pearsonr(entropies, rates)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(entropies, rates, alpha=0.5, s=20, color="#4C72B0")

    # Trend line
    z = np.polyfit(entropies, rates, 1)
    x_line = np.linspace(entropies.min(), entropies.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="red", linestyle="--", linewidth=1)

    ax.set_xlabel("Avg Student Entropy (theta_0)")
    ax.set_ylabel("Rock Rate")
    ax.set_title(f"Rock Rate vs Student Entropy (r={r:.3f}, p={p_val:.2e})")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "entropy_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_plots(config: dict, output_dir: Path) -> None:
    """Generate all sanity check plots from Phase 2 data."""
    console = Console()

    # Load data
    phase2_path = output_dir / "phase2_losses.pt"
    data = torch.load(phase2_path, weights_only=False)

    rock_json = output_dir / "rock_tokens.json"
    with open(rock_json) as f:
        rock_data = json.load(f)

    # 1. Scatter plot
    path = plot_loss_scatter(data, config, output_dir)
    console.print(f"Scatter plot saved to {path}")

    # 2. Rock fraction report
    n_total = len(data["token_ids"])
    ic = config["identification"]
    is_rock = (
        (data["loss_before"] >= torch.quantile(data["loss_before"].float(), ic["tau_high_percentile"] / 100))
        & ((data["loss_before"] - data["loss_after"]) < torch.quantile(
            (data["loss_before"] - data["loss_after"]).float(), ic["delta_percentile"] / 100
        ))
    )
    n_rock = is_rock.sum().item()
    frac = n_rock / n_total

    frac_path = output_dir / "rock_fraction.txt"
    frac_text = (
        f"Total token positions: {n_total:,}\n"
        f"Rock instances: {n_rock:,}\n"
        f"Rock fraction: {frac:.4f} ({frac:.1%})\n"
        f"Status: {'OK' if 0.05 <= frac <= 0.20 else 'WARNING — outside 5-20% range'}\n"
    )
    frac_path.write_text(frac_text)
    console.print(f"Rock fraction: {frac:.1%}")
    console.print(f"  Written to {frac_path}")

    # 3. Entropy correlation
    rock_tokens = rock_data["rock_tokens"]
    path = plot_entropy_correlation(rock_tokens, output_dir)
    if path:
        console.print(f"Entropy correlation plot saved to {path}")

    console.print("[bold green]All plots generated.[/bold green]")

"""
Funnel plot with rock tokens highlighted against unselected tokens.

X-axis: log10(frequency)
Y-axis: Average KL divergence

Rock tokens (from rock_vs_control.csv, column rock_id) are drawn as red
markers on top of the gray cloud of unselected tokens. This shows *where in
the funnel* the rock tokens live — typically the high-frequency / above-trend
region, since they were selected by freq × mean.

Usage
-----
  python rock_funnel_plot.py
  python rock_funnel_plot.py --data rock_token_occurrences_onpolicy_n500.pt \
                             --rock-csv rock_vs_control.csv \
                             --show-control \
                             --output rock_funnel.png
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="rock_token_occurrences_onpolicy_n500.pt",
                    help="Path to occurrence .pt file")
parser.add_argument("--rock-csv", default="rock_vs_control_unrestricted.csv",
                    help="CSV with rock_id (col 1) and optional ctrl_id (col 6)")
parser.add_argument("--show-control", action="store_true",
                    help="Also highlight control tokens (ctrl_id column)")
parser.add_argument("--min-freq", type=int, default=2,
                    help="Minimum token frequency to plot (default: 1)")
parser.add_argument("--output", default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load occurrence data → per-token (freq, avg_kl)
# ---------------------------------------------------------------------------
print(f"Loading {args.data} ...")
data = torch.load(args.data, map_location="cpu", weights_only=False)
df_occ = pd.DataFrame(data["occurrences"])

stats = (
    df_occ.groupby("token_id")["kl"]
    .agg(freq="count", avg_kl="mean")
    .reset_index()
)
stats = stats[stats["freq"] >= args.min_freq].copy()
stats["log10_freq"] = np.log10(stats["freq"])

meta_label = f"{data.get('student_key', '?')}  (n={data.get('samples_processed', '?')})"

# ---------------------------------------------------------------------------
# Load rock + control IDs
# ---------------------------------------------------------------------------
print(f"Loading {args.rock_csv} ...")
rc = pd.read_csv(args.rock_csv)
rock_ids = set(rc["rock_id"].astype(int).tolist())
ctrl_ids = set(rc["ctrl_id"].astype(int).tolist()) if args.show_control and "ctrl_id" in rc.columns else set()

print(f"  rock tokens: {len(rock_ids)}"
      + (f" | control tokens: {len(ctrl_ids)}" if ctrl_ids else ""))

# Split the scatter into layers
is_rock = stats["token_id"].isin(rock_ids)
is_ctrl = stats["token_id"].isin(ctrl_ids) if ctrl_ids else pd.Series(False, index=stats.index)
unselected = stats[~(is_rock | is_ctrl)]
rock = stats[is_rock]
ctrl = stats[is_ctrl]

print(f"  unselected: {len(unselected)} | rock matched: {len(rock)} | ctrl matched: {len(ctrl)}")

# ---------------------------------------------------------------------------
# Smoothed envelope across ALL tokens (visual reference)
# ---------------------------------------------------------------------------
def binned_envelope(s, n_bins=40):
    edges = np.linspace(s["log10_freq"].min(), s["log10_freq"].max(), n_bins + 1)
    centers, mean_kl, lo_kl, hi_kl = [], [], [], []
    for i in range(n_bins):
        mask = (s["log10_freq"] >= edges[i]) & (s["log10_freq"] < edges[i + 1])
        b = s.loc[mask, "avg_kl"]
        if len(b) < 2:
            continue
        centers.append((edges[i] + edges[i + 1]) / 2)
        mean_kl.append(b.mean())
        lo_kl.append(b.quantile(0.10))
        hi_kl.append(b.quantile(0.90))
    return map(np.array, (centers, mean_kl, lo_kl, hi_kl))

cx, mean_line, lo_line, hi_line = binned_envelope(stats)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7))

# Background: unselected tokens
ax.scatter(
    unselected["log10_freq"], unselected["avg_kl"],
    s=6, alpha=0.25, color="lightgray",
    label=f"Unselected ({len(unselected):,})",
    rasterized=True, zorder=2,
)

# Smoothed envelope
ax.fill_between(cx, lo_line, hi_line, color="steelblue", alpha=0.15,
                label="10–90th percentile (all tokens)", zorder=3)
ax.plot(cx, mean_line, color="steelblue", linewidth=1.8,
        label="Binned mean KL", zorder=4)

# Control tokens (if requested) — drawn before rock so rock sits on top
if len(ctrl) > 0:
    ax.scatter(
        ctrl["log10_freq"], ctrl["avg_kl"],
        s=45, alpha=0.85, color="dodgerblue",
        edgecolor="navy", linewidth=0.6,
        marker="s",
        label=f"Control tokens ({len(ctrl)})",
        zorder=5,
    )

# Rock tokens — highlighted
ax.scatter(
    rock["log10_freq"], rock["avg_kl"],
    s=70, alpha=0.9, color="crimson",
    edgecolor="darkred", linewidth=0.7,
    marker="o",
    label=f"Rock tokens ({len(rock)})",
    zorder=6,
)

# Global mean
global_mean = stats["avg_kl"].mean()
ax.axhline(global_mean, color="black", linewidth=1.0, linestyle="--", alpha=0.6,
           label=f"Global mean = {global_mean:.4f}", zorder=7)

ax.set_xlabel(r"$\log_{10}(\mathrm{Frequency})$", fontsize=13)
ax.set_ylabel("Average KL Divergence", fontsize=13)
ax.set_title(f"Rock Tokens on the Funnel Plot — {meta_label}", fontsize=13)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.25)

# Optional: log y if dynamic range is wild
if stats["avg_kl"].max() / max(stats["avg_kl"].median(), 1e-6) > 200:
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("Average KL Divergence (symlog)", fontsize=13)

# ---------------------------------------------------------------------------
# Quick numeric summary
# ---------------------------------------------------------------------------
print("\n=== Where rock tokens sit ===")
print(f"  rock log10(freq):   median={rock['log10_freq'].median():.2f}  "
      f"min={rock['log10_freq'].min():.2f}  max={rock['log10_freq'].max():.2f}")
print(f"  rock avg KL:        median={rock['avg_kl'].median():.4f}  "
      f"min={rock['avg_kl'].min():.4f}  max={rock['avg_kl'].max():.4f}")
print(f"  all-token median KL: {stats['avg_kl'].median():.4f}")
print(f"  rock tokens above global mean: "
      f"{int((rock['avg_kl'] > global_mean).sum())}/{len(rock)}")

plt.tight_layout()
if args.output:
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")
else:
    plt.show()
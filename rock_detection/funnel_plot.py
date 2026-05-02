"""
Funnel Plot: log10(frequency) vs Average KL Divergence

Demonstrates that low-frequency tokens exhibit extreme KL variance (noise),
while high-frequency tokens converge toward the global average — motivating
the use of a smoothed/frequency-weighted metric instead of raw average KL.

Usage
-----
  python funnel_plot.py                          # default: onpolicy n=500
  python funnel_plot.py --files rock_token_occurrences_onpolicy_n500.pt
  python funnel_plot.py --files rock_token_occurrences_onpolicy_n500.pt \
                                rock_token_occurrences_offpolicy_n500.pt
  python funnel_plot.py --output funnel_plot.png
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    nargs="+",
    default=["rock_token_occurrences_onpolicy_n500_unrestricted.pt"],
)
parser.add_argument("--min-freq", type=int, default=2,
                    help="Minimum token frequency to include (default: 2)")
parser.add_argument("--output", default=None,
                    help="Save to this path instead of displaying")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load and aggregate per-token stats from occurrence records
# ---------------------------------------------------------------------------
def load_token_stats(path, min_freq=2):
    """Return DataFrame with columns: token_id, freq, avg_kl, std_kl, log10_freq."""
    print(f"Loading {path} ...")
    data = torch.load(path, map_location="cpu", weights_only=False)

    # Rebuild per-token mean and std from individual occurrence records
    # (more accurate than using the precomputed average_kl when we also need std)
    df_occ = pd.DataFrame(data["occurrences"])
    stats = (
        df_occ.groupby("token_id")["kl"]
        .agg(freq="count", avg_kl="mean", std_kl="std")
        .reset_index()
    )
    stats["std_kl"] = stats["std_kl"].fillna(0.0)

    # Filter rare tokens
    stats = stats[stats["freq"] >= min_freq].copy()
    stats["log10_freq"] = np.log10(stats["freq"])

    meta = {
        "student_key": data.get("student_key", path),
        "samples": data.get("samples_processed", "?"),
    }
    return stats, meta


datasets = [load_token_stats(p, args.min_freq) for p in args.files]

# ---------------------------------------------------------------------------
# Smoothed trend line (binned mean ± 1 std across tokens in each bin)
# ---------------------------------------------------------------------------
def binned_envelope(stats, n_bins=40):
    """Bin tokens by log10_freq; return (centers, mean_kl, lo, hi)."""
    lo_f = stats["log10_freq"].min()
    hi_f = stats["log10_freq"].max()
    edges = np.linspace(lo_f, hi_f, n_bins + 1)
    centers, mean_kl, lo_kl, hi_kl = [], [], [], []
    for i in range(n_bins):
        mask = (stats["log10_freq"] >= edges[i]) & (stats["log10_freq"] < edges[i + 1])
        bucket = stats.loc[mask, "avg_kl"]
        if len(bucket) < 2:
            continue
        centers.append((edges[i] + edges[i + 1]) / 2)
        mean_kl.append(bucket.mean())
        lo_kl.append(bucket.quantile(0.10))
        hi_kl.append(bucket.quantile(0.90))
    return np.array(centers), np.array(mean_kl), np.array(lo_kl), np.array(hi_kl)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
n_panels = len(datasets)
fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), squeeze=False)
axes = axes[0]

for ax, (stats, meta) in zip(axes, datasets):
    label = f"{meta['student_key']}  (n={meta['samples']})"

    # --- scatter: one dot per token ---
    sc = ax.scatter(
        stats["log10_freq"],
        stats["avg_kl"],
        s=6,
        alpha=0.35,
        #c=stats["avg_kl"],
        #cmap="steelblue",
        rasterized=True,
        zorder=2,
    )

    # --- smoothed envelope ---
    cx, mean_line, lo_line, hi_line = binned_envelope(stats)
    ax.plot(cx, mean_line, color="white", linewidth=2.5, zorder=4)
    ax.plot(cx, mean_line, color="coral", linewidth=1.8,
            label="Binned mean KL", zorder=5)
    ax.fill_between(cx, lo_line, hi_line, color="steelblue", alpha=0.20,
                    label="10th–90th percentile", zorder=3)

    # --- global mean reference line ---
    global_mean = stats["avg_kl"].mean()
    ax.axhline(global_mean, color="red", linewidth=1.0, linestyle="--",
               label=f"Global mean = {global_mean:.4f}", zorder=6)

    ax.set_xlabel(r"$\log_{10}(\mathrm{Frequency})$", fontsize=13)
    ax.set_ylabel("Average KL Divergence", fontsize=13)
    ax.set_title(f"Funnel Plot: KL Variance vs Token Frequency\n{label}", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Annotation in the high-variance region
    ax.annotate(
        "Low-frequency tokens:\nextreme KL variance (noise)",
        xy=(stats["log10_freq"].quantile(0.1), stats["avg_kl"].quantile(0.999)),
        xytext=(stats["log10_freq"].quantile(0.25), stats["avg_kl"].quantile(0.9999)),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=12,
        color="black",
    )
    ax.annotate(
        "High-frequency tokens:\nvariance converges",
        xy=(stats["log10_freq"].quantile(0.72), global_mean * 3.15),
        xytext=(stats["log10_freq"].quantile(0.78), global_mean * 5.2),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=12,
        color="black",
    )

    #plt.colorbar(sc, ax=ax, label="Average KL Divergence", pad=0.01)

plt.tight_layout()

if args.output:
    plt.savefig(args.output, dpi=600, bbox_inches="tight")
    print(f"Saved to {args.output}")
else:
    plt.show()

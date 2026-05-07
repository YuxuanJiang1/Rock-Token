"""
Paper-ready figure for the rock-token K-sweep experiment.

Mirrors the styling of paper_figure_gradient.py:
  - large fonts, no suptitle, no per-panel titles
  - (a)/(b) panel labels in upper-left corners
  - frameon=False legends, dpi=600 PNG + PDF

Reproduces the computation from find_rock_cutoff.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from test_stability import (
    load_data, score_freq_weighted, top_k_set, jaccard,
    SAMPLE_SIZES,
)

# ---------------------------------------------------------------------------
# Paper styling — match paper_figure_gradient.py
# ---------------------------------------------------------------------------
LABEL_FS  = 22
TICK_FS   = 15
LEGEND_FS = 17
PANEL_FS  = 22

K_VALUES = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]

# ---------------------------------------------------------------------------
# Compute (same as find_rock_cutoff.py)
# ---------------------------------------------------------------------------
print("Loading on-policy data...")
data_on = {n: load_data("onpolicy", n) for n in SAMPLE_SIZES}

ref_scores      = score_freq_weighted(data_on[500]).sort_values(ascending=False)
total_corpus_kl = ref_scores.sum()
n_scores        = {n: score_freq_weighted(data_on[n]) for n in SAMPLE_SIZES}

records = []
for K in K_VALUES:
    if K > len(ref_scores):
        continue
    ref_topk = top_k_set(ref_scores, K)
    coverage = ref_topk.sum() / total_corpus_kl
    jacs = []
    for n in [50, 100, 200, 300, 400]:
        topk_n = top_k_set(n_scores[n], K)
        jacs.append(jaccard(topk_n.index, ref_topk.index))
    records.append({
        "K":        K,
        "coverage": coverage,
        "J_50":     jacs[0],
        "J_100":    jacs[1],
        "J_200":    jacs[2],
        "J_300":    jacs[3],
        "J_400":    jacs[4],
        "J_avg":    np.mean(jacs),
    })

df = pd.DataFrame(records)
df.to_csv("rock_cutoff_sweep.csv", index=False)
print(f"Saved rock_cutoff_sweep.csv ({len(df)} rows)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# (a) Stability of top-K vs K
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
for col, n, c in zip(["J_50", "J_100", "J_200", "J_300", "J_400"],
                     [50, 100, 200, 300, 400], colors):
    ax.plot(df["K"], df[col], marker="o", linewidth=1.6,
            color=c, label=f"n={n}", alpha=0.9)
ax.plot(df["K"], df["J_avg"], marker="s", linewidth=2.5, color="black",
        label="avg over n", zorder=10)
ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, linewidth=1.0)
ax.set_xscale("log")
ax.set_xlabel("Top-K (number of rock tokens)", fontsize=LABEL_FS)
ax.set_ylabel(r"Jaccard with $n{=}500$ top-K", fontsize=LABEL_FS)
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=TICK_FS)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=LEGEND_FS, loc="lower left", frameon=False)
# ax.text(-0.15, 1.02, "(a)", transform=ax.transAxes,
#         fontsize=PANEL_FS, fontweight="bold")

# (b) Coverage vs Stability tradeoff
ax = axes[1]
l1, = ax.plot(df["K"], df["coverage"] * 100, marker="o", linewidth=2,
              color="darkblue")
ax.set_xscale("log")
ax.set_xlabel("Top-K (number of rock tokens)", fontsize=LABEL_FS)
ax.set_ylabel("Coverage of corpus KL (%)", color="darkblue", fontsize=LABEL_FS)
ax.tick_params(axis="y", labelcolor="darkblue", labelsize=TICK_FS)
ax.tick_params(axis="x", labelsize=TICK_FS)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.25)

ax2 = ax.twinx()
l2, = ax2.plot(df["K"], df["J_avg"], marker="s", linewidth=2,
               color="darkred")
ax2.axhline(0.8, color="red", linestyle="--", alpha=0.4, linewidth=1.0)
ax2.set_ylabel(r"Avg Jaccard with $n{=}500$", color="darkred", fontsize=LABEL_FS)
ax2.tick_params(axis="y", labelcolor="darkred", labelsize=TICK_FS)
ax2.set_ylim(0, 1.05)

ax.legend([l1, l2], ["Coverage of corpus KL (%)", "Avg Jaccard"],
          fontsize=LEGEND_FS, loc="lower right", frameon=False)
# ax.text(-0.18, 1.02, "(b)", transform=ax.transAxes,
#         fontsize=PANEL_FS, fontweight="bold")

plt.tight_layout()
plt.savefig("paper_figure_kcutoff.png", dpi=600, bbox_inches="tight")
plt.savefig("paper_figure_kcutoff.pdf",            bbox_inches="tight")
print("Saved paper_figure_kcutoff.{png,pdf}")

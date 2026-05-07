"""
Combined paper figure (1x4):
  (a) Rock-token funnel plot                  -- from rock_funnel_unrestricted.py
  (b) Per-output rock-token share histogram   -- from per_output_unrestricted.py
  (c) Coverage / Jaccard tradeoff vs Top-K    -- from paper_figure_kcutoff.py
  (d) Jaccard stability vs Top-K (per n)      -- from paper_figure_kcutoff.py

Reads pre-computed artifacts produced by the source scripts:
  - rock_token_occurrences_onpolicy_n500_unrestricted.pt  (panel a raw data)
  - rock_vs_control_unrestricted.csv                      (panel a rock IDs)
  - per_output_distribution_unrestricted.csv              (panel b)
  - rock_cutoff_sweep.csv                                 (panels c, d)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
LABEL_FS  = 27   # axis labels
TICK_FS   = 17   # tick labels
LEGEND_FS = 17   # legend entries
PANEL_FS  = 22   # panel labels (a), (b), (c), (d)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FUNNEL_PT_PATH = "rock_token_occurrences_onpolicy_n500_unrestricted.pt"
ROCK_CSV       = "rock_vs_control_unrestricted.csv"
PER_OUTPUT_CSV = "per_output_distribution_unrestricted.csv"
KCUTOFF_CSV    = "rock_cutoff_sweep.csv"
OUTPUT_PNG     = "paper_figure1_combined.png"
OUTPUT_PDF     = "paper_figure1_combined.pdf"

MIN_FREQ = 2

# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
print(f"Loading {FUNNEL_PT_PATH} ...")
data = torch.load(FUNNEL_PT_PATH, map_location="cpu", weights_only=False)
occ = pd.DataFrame(data["occurrences"])

per_token = (
    occ.groupby("token_id")["kl"]
       .agg(freq="count", mean_kl="mean", std_kl="std")
       .reset_index()
)
per_token = per_token[per_token["freq"] >= MIN_FREQ].copy()

rock_csv = pd.read_csv(ROCK_CSV)
rock_ids = set(rock_csv["rock_id"].astype(int).tolist())

po = pd.read_csv(PER_OUTPUT_CSV)
print(f"Loaded {len(po)} outputs from {PER_OUTPUT_CSV}")

kcutoff = pd.read_csv(KCUTOFF_CSV)
print(f"Loaded {len(kcutoff)} K-sweep rows from {KCUTOFF_CSV}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(30, 7))

# =================== Panel (a): Funnel plot ===================
ax = axes[0]
stats_all = per_token[per_token["freq"] >= 2].copy()
stats_all["log10_freq"] = np.log10(stats_all["freq"])
is_rock_pt = stats_all["token_id"].isin(rock_ids)
unsel = stats_all[~is_rock_pt]
rock  = stats_all[is_rock_pt]


def envelope(s, n_bins=40):
    edges = np.linspace(s["log10_freq"].min(), s["log10_freq"].max(), n_bins + 1)
    cx, m, lo, hi = [], [], [], []
    for i in range(n_bins):
        mask = (s["log10_freq"] >= edges[i]) & (s["log10_freq"] < edges[i + 1])
        b = s.loc[mask, "mean_kl"]
        if len(b) < 2:
            continue
        cx.append((edges[i] + edges[i + 1]) / 2)
        m.append(b.mean())
        lo.append(b.quantile(0.10))
        hi.append(b.quantile(0.90))
    return map(np.array, (cx, m, lo, hi))


cx, mline, loline, hiline = envelope(stats_all)
global_mean = stats_all["mean_kl"].mean()

ax.scatter(unsel["log10_freq"], unsel["mean_kl"],
           s=6, alpha=0.25,
           label=f"Unselected ({len(unsel):,})", rasterized=True, zorder=2)
ax.fill_between(cx, loline, hiline, color="steelblue", alpha=0.15,
                label="10–90th percentile", zorder=3)
ax.plot(cx, mline, color="steelblue", linewidth=1.8,
        label="Binned mean KL", zorder=4)
ax.scatter(rock["log10_freq"], rock["mean_kl"],
           s=70, alpha=0.9, color="crimson", edgecolor="darkred", linewidth=0.7,
           marker="o", label=f"Rock tokens ({len(rock)})", zorder=6)
ax.axhline(global_mean, color="red", linewidth=1.0, linestyle="--", alpha=0.6,
           label=f"Global mean = {global_mean:.4f}", zorder=7)

ax.set_xlabel(r"$\log_{10}(\mathrm{Frequency})$", fontsize=LABEL_FS)
ax.set_ylabel("Average KL Divergence", fontsize=LABEL_FS)
ax.tick_params(labelsize=TICK_FS)
ax.legend(loc="upper right", fontsize=LEGEND_FS, frameon=False)
ax.grid(True, alpha=0.25)

ax.annotate(
    "Low-frequency tokens:\nextreme KL variance (noise)",
    xy=(stats_all["log10_freq"].quantile(0.1),
        stats_all["mean_kl"].quantile(0.999)),
    xytext=(stats_all["log10_freq"].quantile(0.33),
            stats_all["mean_kl"].quantile(0.995)),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=14, color="black",
)
ax.annotate(
    "High-frequency tokens:\nvariance converges",
    xy=(stats_all["log10_freq"].quantile(0.95), global_mean * 3.05),
    xytext=(stats_all["log10_freq"].quantile(0.78), global_mean * 5.2),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=14, color="black",
)

if stats_all["mean_kl"].max() / max(stats_all["mean_kl"].median(), 1e-6) > 200:
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("Average KL Divergence (symlog)", fontsize=LABEL_FS)

ax.text(-0.18, 1.02, "(a)", transform=ax.transAxes,
        fontsize=PANEL_FS, fontweight="bold")

# =================== Panel (b): Per-output rock-token % ===================
ax = axes[1]
ax.hist(po["rock_pct"], bins=40, color="firebrick", alpha=0.8)
ax.axvline(po["rock_pct"].median(), color="black", linestyle="--",
           linewidth=1, label=f"median = {po['rock_pct'].median():.1f}%")
ax.set_xlabel("% of output that is rock tokens", fontsize=LABEL_FS)
ax.set_ylabel("# outputs", fontsize=LABEL_FS)
ax.tick_params(labelsize=TICK_FS)
ax.legend(fontsize=LEGEND_FS, frameon=False)
ax.grid(True, alpha=0.3)
ax.text(-0.18, 1.02, "(b)", transform=ax.transAxes,
        fontsize=PANEL_FS, fontweight="bold")

# =================== Panel (c): Coverage / Jaccard tradeoff ===================
ax = axes[2]
l1, = ax.plot(kcutoff["K"], kcutoff["coverage"] * 100, marker="o", linewidth=2,
              color="darkblue")
ax.set_xscale("log")
ax.set_xlabel("Top-K (number of rock tokens)", fontsize=LABEL_FS)
ax.set_ylabel("Coverage of corpus KL (%)", color="darkblue", fontsize=LABEL_FS)
ax.tick_params(axis="y", labelcolor="darkblue", labelsize=TICK_FS)
ax.tick_params(axis="x", labelsize=TICK_FS)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.25)

ax2 = ax.twinx()
l2, = ax2.plot(kcutoff["K"], kcutoff["J_avg"], marker="s", linewidth=2,
               color="darkred")
ax2.axhline(0.8, color="red", linestyle="--", alpha=0.4, linewidth=1.0)
ax2.set_ylabel(r"Avg Jaccard with $n{=}500$", color="darkred", fontsize=LABEL_FS)
ax2.tick_params(axis="y", labelcolor="darkred", labelsize=TICK_FS)
ax2.set_ylim(0, 1.05)

ax.legend([l1, l2], ["Coverage of corpus KL (%)", "Avg Jaccard"],
          fontsize=LEGEND_FS, loc="lower right", frameon=False)
ax.text(-0.22, 1.02, "(c)", transform=ax.transAxes,
        fontsize=PANEL_FS, fontweight="bold")

# =================== Panel (d): Jaccard stability vs Top-K ===================
ax = axes[3]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
for col, n, c in zip(["J_50", "J_100", "J_200", "J_300", "J_400"],
                     [50, 100, 200, 300, 400], colors):
    ax.plot(kcutoff["K"], kcutoff[col], marker="o", linewidth=1.6,
            color=c, label=f"n={n}", alpha=0.9)
ax.plot(kcutoff["K"], kcutoff["J_avg"], marker="s", linewidth=2.5, color="black",
        label="avg over n", zorder=10)
ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, linewidth=1.0)
ax.set_xscale("log")
ax.set_xlabel("Top-K (number of rock tokens)", fontsize=LABEL_FS)
ax.set_ylabel(r"Jaccard with $n{=}500$ top-K", fontsize=LABEL_FS)
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=TICK_FS)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=LEGEND_FS, loc="lower left", frameon=False)
ax.text(-0.18, 1.02, "(d)", transform=ax.transAxes,
        fontsize=PANEL_FS, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")
print(f"Saved {OUTPUT_PNG} and {OUTPUT_PDF}")

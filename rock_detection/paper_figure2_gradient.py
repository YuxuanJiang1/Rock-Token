"""
Paper-ready figure: 4 panels in one row, no suptitle, large fonts.

Inputs:
  gradient_magnitude.csv   — per-token (group, mag_bar_g, cos_balanced)
  kl_evolution.csv         — per-token (group, mean_kl_fin, mean_kl_mid, ...)

Panels:
  (a) per-token gradient magnitude  ||bar_g_t||  by group
  (b) per-token cosine alignment    cos(bar_g_t, G_balanced)  by group
  (c) KL paired across checkpoints  (log-log)
  (d) Delta KL distribution         by group  (negative = KL reduced during training)

NOTE on chronology: kl_evolution.csv stores mean_kl_fin (the 5k_src "finished"
file) and mean_kl_mid (the 10k checkpoint). The data only fits the
"high_kl learned, rock persists" hypothesis if 5k_src is treated as the
EARLIER checkpoint and 10k as the LATER one. We label panels (c)/(d)
with "early" and "late" accordingly; flip the assignment if your
chronology is the other way around.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABEL_FS  = 27   # axis labels
TICK_FS   = 17   # tick labels
LEGEND_FS = 17   # legend entries
PANEL_FS  = 22   # panel labels (a), (b), (c), (d)


Y_LETTER = 1.15

mag_df = pd.read_csv("gradient_magnitude.csv")
kl_df  = pd.read_csv("kl_evolution.csv")

group_colors = {"rock": "crimson", "high_kl": "darkorange", "random_other": "steelblue"}
group_labels = {"rock": "rock", "high_kl": "high-KL rare", "random_other": "random"}


fig, axes = plt.subplots(1, 1)
# fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

# # ---------------------------------------------------------------------------
# # (a) Magnitude  ||bar_g_t||
# # ---------------------------------------------------------------------------
# ax = axes[0]
# xmax_a = mag_df["mag_bar_g"].dropna().quantile(0.99)
# bins   = np.linspace(0, xmax_a, 41)
# for grp, c in group_colors.items():
#     sub = mag_df[mag_df["group"] == grp]["mag_bar_g"].dropna().clip(0, xmax_a)
#     # Low-alpha fill + sharp outline so groups stay distinguishable when overlapping
#     ax.hist(sub, bins=bins, color=c, density=True,
#             histtype="stepfilled", alpha=0.28, edgecolor="none")
#     ax.hist(sub, bins=bins, color=c, density=True,
#             histtype="step", linewidth=2.0, label=group_labels[grp])
# ax.set_xlabel(r"$\|\bar{g}_t\|$", fontsize=LABEL_FS)
# ax.set_ylabel("density", fontsize=LABEL_FS)
# ax.tick_params(labelsize=TICK_FS)
# leg_a = ax.legend(fontsize=LEGEND_FS, loc="upper right",
#                   frameon=True, facecolor="white", edgecolor="black",
#                   framealpha=0.92, borderpad=0.6, handlelength=1.5)
# leg_a.get_frame().set_linewidth(0.8)
# leg_a.set_zorder(20)
# ax.grid(True, alpha=0.25)
# ax.text(-0.16, Y_LETTER, "(a)", transform=ax.transAxes,
#         fontsize=PANEL_FS, fontweight="bold")

# # ---------------------------------------------------------------------------
# # (b) Cosine alignment  cos(bar_g_t, G_balanced)
# # ---------------------------------------------------------------------------
# ax = axes[1]
# all_cos = mag_df["cos_balanced"].dropna()
# xmin_b = max(-0.5, all_cos.quantile(0.005))
# xmax_b = min(0.7,  all_cos.quantile(0.995))
# bins = np.linspace(xmin_b, xmax_b, 41)
# for grp, c in group_colors.items():
#     sub = mag_df[mag_df["group"] == grp]["cos_balanced"].dropna().clip(xmin_b, xmax_b)
#     ax.hist(sub, bins=bins, color=c, density=True,
#             histtype="stepfilled", alpha=0.28, edgecolor="none")
#     ax.hist(sub, bins=bins, color=c, density=True,
#             histtype="step", linewidth=2.0, label=group_labels[grp])
# ax.axvline(0, color="black", linewidth=1.0, alpha=0.7)
# ax.set_xlabel(r"$\cos(\bar{g}_t,\, G_{\mathrm{balanced}})$", fontsize=LABEL_FS)
# ax.set_ylabel("density", fontsize=LABEL_FS)
# ax.tick_params(labelsize=TICK_FS)
# leg_b = ax.legend(fontsize=LEGEND_FS, loc="upper right",
#                   frameon=True, facecolor="white", edgecolor="black",
#                   framealpha=0.92, borderpad=0.6, handlelength=1.5)
# leg_b.get_frame().set_linewidth(0.8)
# leg_b.set_zorder(20)
# ax.grid(True, alpha=0.25)
# ax.text(-0.16, Y_LETTER, "(b)", transform=ax.transAxes,
#         fontsize=PANEL_FS, fontweight="bold")

# ---------------------------------------------------------------------------
# (c) KL paired across checkpoints — log-log scatter
# ---------------------------------------------------------------------------
# kl_df: mean_kl_fin (5k_src) treated as EARLY; mean_kl_mid (10k) as LATE
#ax = axes[2]
ax = axes
for grp, c in group_colors.items():
    sub = kl_df[kl_df["group"] == grp].dropna(subset=["mean_kl_fin", "mean_kl_mid"])
    ax.scatter(sub["mean_kl_fin"], sub["mean_kl_mid"], s=28, alpha=0.65,
               color=c, label=group_labels[grp], edgecolor="white", linewidth=0.5)
both = pd.concat([kl_df["mean_kl_fin"], kl_df["mean_kl_mid"]]).dropna()
mx = both.max() * 1.3
mn = max(both[both > 0].min() * 0.7, 1e-3)
ax.plot([mn, mx], [mn, mx], "--", color="gray", linewidth=1.2, label=r"$y = x$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
ax.set_xlabel("KL @ early checkpoint", fontsize=LABEL_FS)
ax.set_ylabel("KL @ late checkpoint",  fontsize=LABEL_FS)
ax.tick_params(labelsize=TICK_FS)
leg_c = ax.legend(
    fontsize=LEGEND_FS,
    loc="lower left", bbox_to_anchor=(-0.02, 1.02),
    ncol=2, frameon=False, handlelength=1.0,
    columnspacing=0.8, handletextpad=0.3,
    labelspacing=0.25,
)
leg_c.set_zorder(20)  # draw legend above scatter dots
ax.grid(True, alpha=0.25, which="both")
# ax.text(-0.18, Y_LETTER, "(c)", transform=ax.transAxes,
#         fontsize=PANEL_FS, fontweight="bold")

# # ---------------------------------------------------------------------------
# # (d) Delta KL distribution: late - early   (negative = KL reduced)
# # ---------------------------------------------------------------------------
# ax = axes[3]
# paired = kl_df.dropna(subset=["mean_kl_fin", "mean_kl_mid"]).copy()
# paired["delta_late_minus_early"] = paired["mean_kl_mid"] - paired["mean_kl_fin"]
# xmin_d = paired["delta_late_minus_early"].quantile(0.02)
# xmax_d = paired["delta_late_minus_early"].quantile(0.98)
# bins = np.linspace(xmin_d, xmax_d, 41)
# for grp, c in group_colors.items():
#     sub = paired[paired["group"] == grp]["delta_late_minus_early"].clip(xmin_d, xmax_d)
#     if len(sub) == 0:
#         continue
#     ax.hist(sub, bins=bins, color=c, density=True,
#             histtype="stepfilled", alpha=0.28, edgecolor="none")
#     ax.hist(sub, bins=bins, color=c, density=True,
#             histtype="step", linewidth=2.0, label=group_labels[grp])
# ax.axvline(0, color="black", linewidth=1.0, alpha=0.7)
# ax.set_xlabel(r"$\Delta\,$KL  (late $-$ early)", fontsize=LABEL_FS)
# ax.set_ylabel("density", fontsize=LABEL_FS)
# ax.tick_params(labelsize=TICK_FS)
# leg_d = ax.legend(fontsize=LEGEND_FS, loc="upper left",
#                   frameon=True, facecolor="white", edgecolor="black",
#                   framealpha=0.92, borderpad=0.6, handlelength=1.5)
# leg_d.get_frame().set_linewidth(0.8)
# leg_d.set_zorder(20)
# ax.grid(True, alpha=0.25)
# ax.text(-0.16, Y_LETTER, "(d)", transform=ax.transAxes,
#         fontsize=PANEL_FS, fontweight="bold")

plt.tight_layout()
plt.savefig("paper_figure2(c)_gradient.png", dpi=600, bbox_inches="tight")
plt.savefig("paper_figure2(c)_gradient.pdf",            bbox_inches="tight")
print("Saved paper_figure2_gradient.{png,pdf}")

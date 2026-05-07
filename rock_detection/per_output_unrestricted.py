"""
Plot per-output rock vs non-rock-high-KL distribution for the unrestricted run.

Reads per_output_distribution_unrestricted.csv produced by rerun_unrestricted.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV     = "per_output_distribution_unrestricted.csv"
OUTPUT  = "per_output_distribution_share_unrestricted.png"
LABEL_FS  = 22
TICK_FS   = 15
LEGEND_FS = 17
PANEL_FS  = 22


po = pd.read_csv(CSV)
print(f"Loaded {len(po)} outputs from {CSV}")

#fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig, axes = plt.subplots(1,1)
# # (0,0) Output length distribution
# ax = axes[0, 0]
# ax.hist(po["n_tokens"], bins=50, color="steelblue", alpha=0.8)
# ax.axvline(po["n_tokens"].median(), color="black", linestyle="--",
#            linewidth=1, label=f"median = {po['n_tokens'].median():.0f}")
# ax.set_xlabel("Output length (tokens)"); ax.set_ylabel("# outputs")
# ax.set_title("Output length distribution")
# ax.legend(); ax.grid(True, alpha=0.3)

# # (0,1) Rock-token count per output (log y for skewed long-tail data)
# ax = axes[0, 1]
# ax.hist(po["rock_count"], bins=50, color="crimson", alpha=0.8)
# ax.axvline(po["rock_count"].median(), color="black", linestyle="--",
#            linewidth=1, label=f"median = {po['rock_count'].median():.0f}")
# ax.set_xlabel("# rock-token occurrences per output"); ax.set_ylabel("# outputs")
# ax.set_title("Rock-token count per output")
# ax.legend(); ax.grid(True, alpha=0.3)

# (0,2) Rock-token percentage per output
ax = axes
ax.hist(po["rock_pct"], bins=40, color="firebrick", alpha=0.8)
ax.axvline(po["rock_pct"].median(), color="black", linestyle="--",
           linewidth=1, label=f"median = {po['rock_pct'].median():.1f}%")
ax.set_xlabel("% of output that is rock tokens", fontsize = LABEL_FS); ax.set_ylabel("# outputs", fontsize = LABEL_FS)
#ax.set_title("Rock-token share per output")
ax.legend(fontsize = LEGEND_FS); ax.grid(True, alpha=0.3)

# # (1,0) Non-rock high-KL count per output
# ax = axes[1, 0]
# ax.hist(po["nonrock_high_count"], bins=50, color="darkorange", alpha=0.8)
# ax.axvline(po["nonrock_high_count"].median(), color="black", linestyle="--",
#            linewidth=1, label=f"median = {po['nonrock_high_count'].median():.0f}")
# ax.set_xlabel("# non-rock high-KL events per output (>p95)")
# ax.set_ylabel("# outputs")
# ax.set_title("Non-rock high-KL count per output")
# ax.legend(); ax.grid(True, alpha=0.3)

# # (1,1) KL share — stacked: rock vs nonrock-high vs other
# ax = axes[1, 1]
# order = po.sort_values("total_kl", ascending=False).reset_index(drop=True)
# x = np.arange(len(order))
# rock_share         = order["rock_kl_share"].values
# nonrock_high_share = order["nonrock_high_kl_share"].values
# other_share        = (100 - rock_share - nonrock_high_share).clip(min=0)
# ax.stackplot(
#     x, rock_share, nonrock_high_share, other_share,
#     colors=["crimson", "darkorange", "lightgray"],
#     labels=["rock tokens", "non-rock high-KL", "other"],
#     alpha=0.85,
# )
# ax.set_xlabel("Output (sorted by total KL, descending)")
# ax.set_ylabel("% of output's total KL")
# ax.set_title("Per-output KL composition")
# ax.set_ylim(0, 100); ax.set_xlim(0, len(order))
# ax.legend(loc="lower left"); ax.grid(True, alpha=0.3)

# # (1,2) Scatter: output length vs rock count, colored by rock_pct
# ax = axes[1, 2]
# sc = ax.scatter(po["n_tokens"], po["rock_count"],
#                 c=po["rock_pct"], cmap="viridis",
#                 s=18, alpha=0.7)
# # y = x line (would be 100% rock tokens)
# lim = max(po["n_tokens"].max(), po["rock_count"].max())
# ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="100% rock")
# # y = 0.71x line (the corpus-level rock share)
# ax.plot([0, lim], [0, 0.71 * lim], color="crimson", linewidth=1.5, alpha=0.7,
#         label="71% rock (corpus avg)")
# ax.set_xlabel("Output length (tokens)", fontsize=LABEL_FS)
# ax.set_ylabel("# rock-token occurrences", fontsize=LABEL_FS)
# ax.set_title("Output length vs rock count\n(color = rock %)")
# ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
# plt.colorbar(sc, ax=ax, label="rock %", pad=0.01)

# plt.suptitle(
#     "Per-output distribution — unrestricted on-policy (n=500, EOS-stopped)",
#     fontsize=14, y=1.00,
# )
plt.tight_layout()
plt.savefig(OUTPUT, dpi=600, bbox_inches="tight")
print(f"Saved {OUTPUT}")
"""
Find the best top-K cutoff for selecting rock tokens via freq*mean.

For each candidate K:
  - measure Jaccard between top-K at smaller n and top-K at n=500
  - track how much of the total corpus KL contribution the top-K covers

The right K is the largest one where stability stays high.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from test_stability import (
    load_data, score_freq_weighted, top_k_set, jaccard,
    SAMPLE_SIZES,
)

K_VALUES = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]

print("Loading on-policy data...")
data_on = {n: load_data("onpolicy", n) for n in SAMPLE_SIZES}

ref_scores      = score_freq_weighted(data_on[500]).sort_values(ascending=False)
total_corpus_kl = ref_scores.sum()
print(f"Total corpus KL contribution at n=500: {total_corpus_kl:.1f}")
print(f"Vocab tokens with freq>=1 at n=500: {len(ref_scores)}")

n_scores = {n: score_freq_weighted(data_on[n]) for n in SAMPLE_SIZES}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
print(f"\n{'K':>6} {'coverage':>10} {'J@50':>7} {'J@100':>7} {'J@200':>7} {'J@300':>7} {'J@400':>7} {'J_avg':>7}")
print("-" * 70)
for _, r in df.iterrows():
    print(f"{int(r.K):>6} {r.coverage:>10.3f} {r.J_50:>7.3f} {r.J_100:>7.3f} "
          f"{r.J_200:>7.3f} {r.J_300:>7.3f} {r.J_400:>7.3f} {r.J_avg:>7.3f}")

print("\n=== Largest K still meeting stability threshold ===")
for thresh in [0.95, 0.90, 0.85, 0.80, 0.70, 0.50]:
    rows = df[df["J_avg"] >= thresh]
    if len(rows) > 0:
        best = rows.iloc[-1]
        print(f"  J_avg >= {thresh:.2f}: K_max = {int(best.K):>5}  "
              f"covers {best.coverage*100:5.1f}%  "
              f"(J@50={best.J_50:.3f}, J@200={best.J_200:.3f})")

print("\n=== Smallest K reaching coverage target ===")
for cov_thresh in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
    rows = df[df["coverage"] >= cov_thresh]
    if len(rows) > 0:
        first = rows.iloc[0]
        print(f"  coverage >= {cov_thresh*100:>2.0f}%: K_min = {int(first.K):>5}  "
              f"J_avg={first.J_avg:.3f}  "
              f"(J@50={first.J_50:.3f}, J@200={first.J_200:.3f})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
for col, n, c in zip(["J_50", "J_100", "J_200", "J_300", "J_400"],
                     [50, 100, 200, 300, 400], colors):
    ax.plot(df["K"], df[col], marker="o", linewidth=1.6, color=c, label=f"n={n}", alpha=0.9)
ax.plot(df["K"], df["J_avg"], marker="s", linewidth=2.5, color="black",
        label="avg over n", zorder=10)
ax.set_xscale("log")
ax.set_xlabel("Top-K (number of rock tokens selected)")
ax.set_ylabel("Jaccard with n=500 top-K")
ax.set_title("Stability of freq*mean top-K vs K")
ax.set_ylim(0, 1.05)
ax.axhline(0.8, color="red", linestyle="--", alpha=0.5)
ax.text(K_VALUES[0], 0.81, "J = 0.80", color="red", fontsize=9)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="lower left")

ax = axes[1]
ax.plot(df["K"], df["coverage"] * 100, marker="o", linewidth=2,
        color="darkblue", label="Coverage of corpus KL (%)")
ax.set_xscale("log")
ax.set_xlabel("Top-K (number of rock tokens selected)")
ax.set_ylabel("Coverage of corpus KL (%)", color="darkblue")
ax.tick_params(axis="y", labelcolor="darkblue")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.plot(df["K"], df["J_avg"], marker="s", linewidth=2,
         color="darkred", label="Avg Jaccard")
ax2.set_ylabel("Avg Jaccard with n=500", color="darkred")
ax2.tick_params(axis="y", labelcolor="darkred")
ax2.set_ylim(0, 1.05)
ax2.axhline(0.8, color="red", linestyle="--", alpha=0.4)

ax.set_title("Coverage vs Stability tradeoff")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

plt.suptitle("Rock-Token Selection (freq*mean): How many is the right number?", fontsize=13)
plt.tight_layout()
plt.savefig("rock_cutoff_sweep.png", dpi=150, bbox_inches="tight")
print("\nSaved rock_cutoff_sweep.png")

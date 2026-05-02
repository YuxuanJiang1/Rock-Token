"""
Sweep the min_freq threshold on top of LCB_t to find the sweet spot.

Questions:
  1. How many tokens pass each threshold at each sample size?
  2. Does tightening the threshold improve stability?
  3. What's the "rock token ratio" — fraction of eligible vocab that ranks high?
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from test_stability import (
    load_data, per_token, top_k_set, jaccard, spearman_on_union, SAMPLE_SIZES
)

TOP_K = 50
FREQ_THRESHOLDS = [2, 5, 10, 20, 50, 100, 200]


def lcb_t_filtered(df, min_freq, alpha=0.05):
    g = per_token(df, max(min_freq, 2))
    if len(g) == 0:
        return pd.Series(dtype=float)
    t_crit = stats.t.ppf(1 - alpha, df=g["n"] - 1)
    return g["mean"] - t_crit * g["std"] / np.sqrt(g["n"])


print("Loading on-policy data...")
data_on = {n: load_data("onpolicy", n) for n in SAMPLE_SIZES}

# ---------------------------------------------------------------------------
# 1) Pool size per (threshold, n)
# ---------------------------------------------------------------------------
print(f"\n=== Pool size: # tokens passing min_freq, by sample size ===")
print(f"{'min_freq':<10} " + " ".join(f"{n:>7}" for n in SAMPLE_SIZES))
print("-" * (10 + 8 * len(SAMPLE_SIZES)))
pool_sizes = {}
for mf in FREQ_THRESHOLDS:
    sizes = []
    for n in SAMPLE_SIZES:
        s = lcb_t_filtered(data_on[n], mf)
        sizes.append(len(s))
    pool_sizes[mf] = sizes
    print(f"{mf:<10} " + " ".join(f"{x:>7}" for x in sizes))

# ---------------------------------------------------------------------------
# 2) Top-50 stability at each threshold
# ---------------------------------------------------------------------------
print(f"\n=== Top-{TOP_K} Jaccard with n=500 ===")
print(f"{'min_freq':<10} " + " ".join(f"{n:>6}" for n in SAMPLE_SIZES) + f"  {'avg(50-400)':>11}")
print("-" * 70)
results = {}
for mf in FREQ_THRESHOLDS:
    ref = lcb_t_filtered(data_on[500], mf)
    ref_top = top_k_set(ref, TOP_K)
    js, sps = [], []
    for n in SAMPLE_SIZES:
        s = lcb_t_filtered(data_on[n], mf)
        topk = top_k_set(s, TOP_K)
        js.append(jaccard(topk.index, ref_top.index))
        sps.append(spearman_on_union(s, ref, TOP_K))
    avg = np.mean(js[:-1])
    results[mf] = {"jaccard": js, "spearman": sps, "avg_j": avg}
    print(f"{mf:<10} " + " ".join(f"{v:6.3f}" for v in js) + f"  {avg:>11.3f}")

print(f"\n=== Top-{TOP_K} Spearman with n=500 ===")
print(f"{'min_freq':<10} " + " ".join(f"{n:>6}" for n in SAMPLE_SIZES) + f"  {'avg(50-400)':>11}")
print("-" * 70)
for mf in FREQ_THRESHOLDS:
    sps = results[mf]["spearman"]
    avg = np.nanmean(sps[:-1])
    print(f"{mf:<10} " + " ".join(f"{v:6.3f}" for v in sps) + f"  {avg:>11.3f}")

# ---------------------------------------------------------------------------
# 3) Rock-token ratio: top-K as a fraction of pool size
# ---------------------------------------------------------------------------
print(f"\n=== Top-{TOP_K} as % of eligible pool (lower = more selective) ===")
print(f"{'min_freq':<10} " + " ".join(f"{n:>7}" for n in SAMPLE_SIZES))
print("-" * (10 + 8 * len(SAMPLE_SIZES)))
for mf in FREQ_THRESHOLDS:
    pcts = [f"{100 * TOP_K / max(p, 1):>6.1f}%" for p in pool_sizes[mf]]
    print(f"{mf:<10} " + " ".join(pcts))

# ---------------------------------------------------------------------------
# 4) Alternative: scale-invariant top-X% (constant selection ratio)
# ---------------------------------------------------------------------------
# At n=500 with min_freq=10, top-50 is roughly top-X% of the pool.
# Use that same X% at every n for fair comparison.
print(f"\n=== Stability at FIXED 5% selection ratio ===")
SELECT_PCT = 0.05
print(f"{'min_freq':<10} " + " ".join(f"{n:>6}" for n in SAMPLE_SIZES) + f"  {'avg(50-400)':>11}")
print("-" * 70)
for mf in FREQ_THRESHOLDS:
    ref = lcb_t_filtered(data_on[500], mf)
    if len(ref) == 0:
        continue
    k_ref = max(1, int(len(ref) * SELECT_PCT))
    ref_top = top_k_set(ref, k_ref)

    js = []
    for n in SAMPLE_SIZES:
        s = lcb_t_filtered(data_on[n], mf)
        if len(s) == 0:
            js.append(0.0)
            continue
        k_n = max(1, int(len(s) * SELECT_PCT))
        topk = top_k_set(s, k_n)
        # Compare both top-X% sets via Jaccard (sets may differ in size, that's fine)
        js.append(jaccard(topk.index, ref_top.index))
    avg = np.mean(js[:-1])
    print(f"{mf:<10} " + " ".join(f"{v:6.3f}" for v in js) + f"  {avg:>11.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(FREQ_THRESHOLDS)))

# Pool size
for c, mf in zip(colors, FREQ_THRESHOLDS):
    axes[0, 0].plot(SAMPLE_SIZES, pool_sizes[mf], marker="o", color=c, label=f"min_freq={mf}")
axes[0, 0].set_xlabel("Sample size")
axes[0, 0].set_ylabel("# tokens passing threshold")
axes[0, 0].set_title("Eligible pool size")
axes[0, 0].set_yscale("log")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=9)

# Jaccard
for c, mf in zip(colors, FREQ_THRESHOLDS):
    axes[0, 1].plot(SAMPLE_SIZES, results[mf]["jaccard"], marker="o", color=c, label=f"min_freq={mf}")
axes[0, 1].set_xlabel("Sample size")
axes[0, 1].set_ylabel(f"Jaccard with n=500 top-{TOP_K}")
axes[0, 1].set_title(f"Top-{TOP_K} Set Stability vs freq threshold")
axes[0, 1].set_ylim(0, 1.05)
axes[0, 1].axhline(1.0, linestyle="--", color="gray", alpha=0.4)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=9)

# Spearman
for c, mf in zip(colors, FREQ_THRESHOLDS):
    axes[1, 0].plot(SAMPLE_SIZES, results[mf]["spearman"], marker="o", color=c, label=f"min_freq={mf}")
axes[1, 0].set_xlabel("Sample size")
axes[1, 0].set_ylabel(f"Spearman ρ with n=500 (top-{TOP_K} union)")
axes[1, 0].set_title("Rank Stability vs freq threshold")
axes[1, 0].set_ylim(-1.05, 1.05)
axes[1, 0].axhline(1.0, linestyle="--", color="gray", alpha=0.4)
axes[1, 0].axhline(0.0, linestyle=":", color="black", alpha=0.4)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=9)

# Selection ratio
for c, mf in zip(colors, FREQ_THRESHOLDS):
    pcts = [100 * TOP_K / max(p, 1) for p in pool_sizes[mf]]
    axes[1, 1].plot(SAMPLE_SIZES, pcts, marker="o", color=c, label=f"min_freq={mf}")
axes[1, 1].set_xlabel("Sample size")
axes[1, 1].set_ylabel(f"Top-{TOP_K} as % of pool")
axes[1, 1].set_title("Selection rate at fixed top-K=50")
axes[1, 1].set_yscale("log")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=9)

plt.suptitle("LCB_t with varying min_freq thresholds", fontsize=13)
plt.tight_layout()
plt.savefig("freq_threshold_sweep.png", dpi=150, bbox_inches="tight")
print("\nSaved freq_threshold_sweep.png")

"""
Inspect what tokens each scoring method actually selects at n=500.
Adds three hybrid methods to test_stability.py's set:
  - LCB_t with min_freq=10
  - EB_post_mean with min_freq=10
  - freq * LCB_t  (lower-bound on total contribution)
"""

import torch
import numpy as np
import pandas as pd
from scipy import stats

try:
    from transformers import AutoTokenizer
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

from test_stability import (
    load_data, per_token, score_raw_mean, score_freq_filtered,
    score_freq_weighted, score_lcb_t, score_eb_post_mean, score_eb_lower_cb,
    joint_min, top_k_set, jaccard, spearman_on_union,
    SAMPLE_SIZES, TOP_K
)

import matplotlib.pyplot as plt


def score_lcb_filtered(df, min_freq=10, alpha=0.05):
    g = per_token(df, max(min_freq, 2))
    t_crit = stats.t.ppf(1 - alpha, df=g["n"] - 1)
    return g["mean"] - t_crit * g["std"] / np.sqrt(g["n"])


def score_eb_filtered(df, min_freq=10):
    base = score_eb_post_mean(df)
    counts = df.groupby("token_id").size()
    keep = counts[counts >= min_freq].index
    return base[base.index.isin(keep)]


def score_freq_lcb(df, alpha=0.05):
    """freq * LCB_t — lower-bound on token's total contribution to corpus KL."""
    g = per_token(df, 2)
    t_crit = stats.t.ppf(1 - alpha, df=g["n"] - 1)
    lcb = g["mean"] - t_crit * g["std"] / np.sqrt(g["n"])
    return g["n"] * lcb.clip(lower=0)


HYBRID_METHODS = {
    "LCB_t_freq>=10":      lambda df: score_lcb_filtered(df, 10),
    "EB_post_freq>=10":    lambda df: score_eb_filtered(df, 10),
    "freq*LCB_t":          score_freq_lcb,
}


# ---------------------------------------------------------------------------
# Stability test (extended)
# ---------------------------------------------------------------------------
ALL_METHODS = {
    "raw_mean":            score_raw_mean,
    "mean_freq>=5":        lambda df: score_freq_filtered(df, 5),
    "mean_freq>=10":       lambda df: score_freq_filtered(df, 10),
    "freq*mean":           score_freq_weighted,
    "LCB_t":               score_lcb_t,
    "LCB_t_freq>=10":      lambda df: score_lcb_filtered(df, 10),
    "EB_post_mean":        score_eb_post_mean,
    "EB_post_freq>=10":    lambda df: score_eb_filtered(df, 10),
    "EB_lower_CB":         score_eb_lower_cb,
    "freq*LCB_t":          score_freq_lcb,
}

print("Loading on-policy data...")
data_on  = {n: load_data("onpolicy",  n) for n in SAMPLE_SIZES}
print("Loading off-policy data...")
data_off = {n: load_data("offpolicy", n) for n in SAMPLE_SIZES}

# Reference scores at n=500
ref_scores = {name: fn(data_on[500]) for name, fn in ALL_METHODS.items()}

# Stability across n
results = {name: {"jaccard": [], "spearman": []} for name in ALL_METHODS}
for n in SAMPLE_SIZES:
    for name, fn in ALL_METHODS.items():
        s = fn(data_on[n])
        ref = ref_scores[name]
        results[name]["jaccard"].append(jaccard(top_k_set(s, TOP_K).index, top_k_set(ref, TOP_K).index))
        results[name]["spearman"].append(spearman_on_union(s, ref, TOP_K))

# ---------------------------------------------------------------------------
# Print top-5 tokens at n=500 for each method
# ---------------------------------------------------------------------------
print("\nLoading tokenizer for token decoding...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k"
    )
    have_tokenizer = True
except Exception as e:
    print(f"  (skipping token decoding: {e})")
    have_tokenizer = False


def decode(tid):
    if have_tokenizer:
        return repr(tokenizer.decode([tid]))
    return f"id={tid}"


df500 = data_on[500]
freq500 = df500.groupby("token_id").size()
mean500 = df500.groupby("token_id")["kl"].mean()

print("\n=== Top-5 tokens at n=500 (with freq, raw mean KL) ===")
for name, scores in ref_scores.items():
    top5 = scores.sort_values(ascending=False).head(5)
    print(f"\n  {name}:")
    for tid in top5.index:
        f = freq500.get(tid, 0)
        m = mean500.get(tid, float("nan"))
        s = scores.loc[tid]
        print(f"    {decode(tid):<25}  score={s:>10.4f}  freq={f:>5}  mean_kl={m:>7.3f}")

# ---------------------------------------------------------------------------
# Stability summary table
# ---------------------------------------------------------------------------
print(f"\n=== Stability vs n=500 (top-{TOP_K}) — Jaccard ===")
print(f"{'method':<22} " + " ".join(f"{n:>6}" for n in SAMPLE_SIZES) + f"  {'avg(50-400)':>12}")
print("-" * 80)
for name, vals in results.items():
    avg = np.mean(vals["jaccard"][:-1])
    row = " ".join(f"{v:6.3f}" for v in vals["jaccard"])
    print(f"{name:<22} {row}  {avg:>12.3f}")

print(f"\n=== Stability vs n=500 (top-{TOP_K}) — Spearman ===")
print(f"{'method':<22} " + " ".join(f"{n:>6}" for n in SAMPLE_SIZES) + f"  {'avg(50-400)':>12}")
print("-" * 80)
for name, vals in results.items():
    avg = np.nanmean(vals["spearman"][:-1])
    row = " ".join(f"{v:6.3f}" for v in vals["spearman"])
    print(f"{name:<22} {row}  {avg:>12.3f}")

print("\n=== Ranked by mean Jaccard ===")
ranking = [(name, np.mean(v["jaccard"][:-1]), np.nanmean(v["spearman"][:-1])) for name, v in results.items()]
for name, j, s in sorted(ranking, key=lambda x: -x[1]):
    print(f"  {name:<22} J={j:.3f}  S={s:+.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(17, 6))
markers = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "d"]
colors  = plt.cm.tab10(np.linspace(0, 1, 10))

for i, (name, vals) in enumerate(results.items()):
    style = dict(marker=markers[i % len(markers)], linewidth=1.8,
                 markersize=7, color=colors[i % 10], label=name, alpha=0.85)
    axes[0].plot(SAMPLE_SIZES, vals["jaccard"],  **style)
    axes[1].plot(SAMPLE_SIZES, vals["spearman"], **style)

for ax, ylabel, title in [
    (axes[0], f"Jaccard with n=500 top-{TOP_K}", f"Top-{TOP_K} Set Stability"),
    (axes[1], f"Spearman ρ with n=500 (top-{TOP_K} union)", f"Top-{TOP_K} Rank Stability"),
]:
    ax.set_xlabel("Sample size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

axes[0].set_ylim(0, 1.05)
axes[1].set_ylim(-1.05, 1.05)

plt.suptitle("Rock-Token Scoring Method Stability — extended", fontsize=13)
plt.tight_layout()
plt.savefig("stability_extended.png", dpi=150, bbox_inches="tight")
print("\nSaved stability_extended.png")

"""
Test which rock-token scoring method produces the most stable top-K list
as the sample size grows.

For each method M and each sample size n in {50, 100, 200, 300, 400, 500}:
  - compute top-K rock tokens at sample size n
  - measure agreement with the n=500 top-K (Jaccard + Spearman)

A "stable" method is one whose small-n top-K already looks like the n=500 top-K.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SAMPLE_SIZES = [50, 100, 200, 300, 400, 500]
TOP_K = 50

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(student, n):
    path = f"rock_token_occurrences_{student}_n{n}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    return pd.DataFrame(data["occurrences"])


def per_token(df, min_n=1):
    g = df.groupby("token_id")["kl"].agg(n="count", mean="mean", std="std")
    g["std"] = g["std"].fillna(0.0)
    return g[g["n"] >= min_n].copy()


# ---------------------------------------------------------------------------
# Scoring methods (single-model, applied to on-policy)
# ---------------------------------------------------------------------------
def score_raw_mean(df):
    g = per_token(df, 1)
    return g["mean"]


def score_freq_filtered(df, min_freq):
    g = per_token(df, min_freq)
    return g["mean"]


def score_freq_weighted(df):
    g = per_token(df, 1)
    return g["n"] * g["mean"]


def score_lcb_t(df, alpha=0.05):
    g = per_token(df, 2)
    t_crit = stats.t.ppf(1 - alpha, df=g["n"] - 1)
    return g["mean"] - t_crit * g["std"] / np.sqrt(g["n"])


def empirical_bayes(df):
    """Return mu0, sigma2, tau2."""
    by = df.groupby("token_id")["kl"]
    means = by.transform("mean")
    sse = ((df["kl"] - means) ** 2).sum()
    counts = df.groupby("token_id").size()
    dof = (counts - 1).clip(lower=0).sum()
    sigma2 = sse / max(dof, 1)
    g = per_token(df, 1)
    n_bar = g["n"].mean()
    tau2 = max(0.0, g["mean"].var() - sigma2 / n_bar)
    return df["kl"].mean(), sigma2, tau2


def score_eb_post_mean(df):
    mu0, sigma2, tau2 = empirical_bayes(df)
    g = per_token(df, 1)
    if tau2 < 1e-12:
        return pd.Series(mu0, index=g.index)
    w = (g["n"] / sigma2) / (g["n"] / sigma2 + 1.0 / tau2)
    return w * g["mean"] + (1 - w) * mu0


def score_eb_lower_cb(df, alpha=0.05):
    mu0, sigma2, tau2 = empirical_bayes(df)
    g = per_token(df, 1)
    if tau2 < 1e-12:
        return pd.Series(mu0, index=g.index)
    w = (g["n"] / sigma2) / (g["n"] / sigma2 + 1.0 / tau2)
    post_mean = w * g["mean"] + (1 - w) * mu0
    post_var = 1.0 / (g["n"] / sigma2 + 1.0 / tau2)
    z = stats.norm.ppf(1 - alpha)
    return post_mean - z * np.sqrt(post_var)


# ---------------------------------------------------------------------------
# Joint scoring: combine on-policy and off-policy
# ---------------------------------------------------------------------------
def joint_min(score_fn, df_on, df_off):
    s_on = score_fn(df_on)
    s_off = score_fn(df_off)
    common = s_on.index.intersection(s_off.index)
    return pd.Series(
        np.minimum(s_on.loc[common].values, s_off.loc[common].values),
        index=common,
    )


def score_freq_lcb(df, alpha=0.05):
    g = per_token(df, 2)
    t_crit = stats.t.ppf(1 - alpha, df=g["n"] - 1)
    lcb = (g["mean"] - t_crit * g["std"] / np.sqrt(g["n"])).clip(lower=0)
    return g["n"] * lcb


SINGLE_METHODS = {
    "raw_mean":          score_raw_mean,
    "mean_freq>=5":      lambda df: score_freq_filtered(df, 5),
    "mean_freq>=10":     lambda df: score_freq_filtered(df, 10),
    "freq*mean":         score_freq_weighted,
    "freq*LCB_t":        score_freq_lcb,
    "LCB_t":             score_lcb_t,
    "EB_post_mean":      score_eb_post_mean,
    "EB_lower_CB":       score_eb_lower_cb,
}

JOINT_METHODS = {
    "joint_freq*mean":   score_freq_weighted,
    "joint_freq*LCB_t":  score_freq_lcb,
    "joint_LCB_t":       score_lcb_t,
    "joint_EB_lower_CB": score_eb_lower_cb,
}


# ---------------------------------------------------------------------------
# Stability evaluation
# ---------------------------------------------------------------------------
def top_k_set(scores, k=TOP_K):
    return scores.sort_values(ascending=False).head(k)


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def spearman_on_union(scores_a, scores_b, k=TOP_K):
    """Spearman of rankings on the union of top-K sets, missing entries get rank k+1."""
    top_a = top_k_set(scores_a, k)
    top_b = top_k_set(scores_b, k)
    union = list(set(top_a.index) | set(top_b.index))
    if len(union) < 2:
        return np.nan
    ranks_a = pd.Series(range(1, len(top_a) + 1), index=top_a.index)
    ranks_b = pd.Series(range(1, len(top_b) + 1), index=top_b.index)
    ra = ranks_a.reindex(union).fillna(k + 1).values
    rb = ranks_b.reindex(union).fillna(k + 1).values
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    corr, _ = stats.spearmanr(ra, rb)
    return corr


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("Loading on-policy data...")
data_on  = {n: load_data("onpolicy",  n) for n in SAMPLE_SIZES}
print("Loading off-policy data...")
data_off = {n: load_data("offpolicy", n) for n in SAMPLE_SIZES}

# Reference scores at n=500
print("\nComputing reference (n=500) top-K per method...")
ref_scores = {}
for name, fn in SINGLE_METHODS.items():
    ref_scores[name] = fn(data_on[500])
for name, fn in JOINT_METHODS.items():
    ref_scores[name] = joint_min(fn, data_on[500], data_off[500])

# Per-(method, n) stability vs reference
results = {name: {"jaccard": [], "spearman": []} for name in list(SINGLE_METHODS) + list(JOINT_METHODS)}

for n in SAMPLE_SIZES:
    print(f"  evaluating n={n}...")
    for name, fn in SINGLE_METHODS.items():
        s = fn(data_on[n])
        ref = ref_scores[name]
        topk = top_k_set(s, TOP_K)
        ref_topk = top_k_set(ref, TOP_K)
        results[name]["jaccard"].append(jaccard(topk.index, ref_topk.index))
        results[name]["spearman"].append(spearman_on_union(s, ref, TOP_K))
    for name, fn in JOINT_METHODS.items():
        s = joint_min(fn, data_on[n], data_off[n])
        ref = ref_scores[name]
        topk = top_k_set(s, TOP_K)
        ref_topk = top_k_set(ref, TOP_K)
        results[name]["jaccard"].append(jaccard(topk.index, ref_topk.index))
        results[name]["spearman"].append(spearman_on_union(s, ref, TOP_K))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n=== Stability vs n=500 reference (top-{TOP_K}) ===\n")
header = f"{'method':<22} " + " ".join(f"{n:>6}" for n in SAMPLE_SIZES) + f" | {'avg(50-400)':>11}"
print(header + "  [Jaccard]")
print("-" * len(header))
ranking_j = []
for name, vals in results.items():
    row = f"{name:<22} " + " ".join(f"{v:6.3f}" for v in vals["jaccard"])
    avg = np.mean(vals["jaccard"][:-1])
    ranking_j.append((name, avg))
    print(f"{row} | {avg:>11.3f}")

print()
print(header + "  [Spearman]")
print("-" * len(header))
ranking_s = []
for name, vals in results.items():
    row = f"{name:<22} " + " ".join(f"{v:6.3f}" for v in vals["spearman"])
    avg = np.nanmean(vals["spearman"][:-1])
    ranking_s.append((name, avg))
    print(f"{row} | {avg:>11.3f}")

print("\n=== Methods ranked by mean Jaccard (most stable first) ===")
for name, j in sorted(ranking_j, key=lambda x: -x[1]):
    s = dict(ranking_s)[name]
    print(f"  {name:<22}  J={j:.3f}  S={s:.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
markers = ["o", "s", "^", "v", "D", "P", "X", "*", "h"]
colors  = plt.cm.tab10(np.linspace(0, 1, len(results)))

for i, (name, vals) in enumerate(results.items()):
    style = dict(marker=markers[i % len(markers)], linewidth=1.8,
                 markersize=8, color=colors[i], label=name, alpha=0.85)
    axes[0].plot(SAMPLE_SIZES, vals["jaccard"],  **style)
    axes[1].plot(SAMPLE_SIZES, vals["spearman"], **style)

axes[0].set_xlabel("Sample size")
axes[0].set_ylabel(f"Jaccard with n=500 top-{TOP_K}")
axes[0].set_title(f"Top-{TOP_K} Set Stability\n(higher = more stable)")
axes[0].set_ylim(0, 1.05)
axes[0].axhline(1.0, linestyle="--", color="gray", alpha=0.4)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8, loc="lower right")

axes[1].set_xlabel("Sample size")
axes[1].set_ylabel(f"Spearman ρ with n=500 ranking (top-{TOP_K} union)")
axes[1].set_title(f"Top-{TOP_K} Rank Stability")
axes[1].set_ylim(-0.1, 1.05)
axes[1].axhline(1.0, linestyle="--", color="gray", alpha=0.4)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8, loc="lower right")

plt.suptitle("Rock-Token Scoring Method Stability vs Sample Size", fontsize=13)
plt.tight_layout()
plt.savefig("stability_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved stability_comparison.png")

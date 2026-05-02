"""
Rerun rock-token selection and downstream analysis on the unrestricted data.

Steps:
  1. Re-select rock tokens (top-100 by freq*mean) from unrestricted on-policy data.
     Controls = next 100 by freq*mean (rank 101-200).
  2. Save as rock_vs_control_unrestricted.csv (same column format as the original).
  3. Compare with the original rock_vs_control.csv (overlap).
  4. Re-run per-output distribution analysis.
  5. Re-render the rock funnel plot.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH       = "rock_token_occurrences_onpolicy_n500_unrestricted.pt"
OLD_CSV         = "rock_vs_control.csv"
NEW_CSV         = "rock_vs_control_unrestricted.csv"
PER_OUTPUT_CSV  = "per_output_distribution_unrestricted.csv"
PER_OUTPUT_PLOT = "per_output_distribution_unrestricted_with_rock.png"
FUNNEL_PLOT     = "rock_funnel_unrestricted.png"
TOP_K           = 100
EXCLUDE_TOP_FREQ = 5  # drop this many highest-frequency tokens from rock+control pool
TOKENIZER_ID    = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MIN_FREQ = 2
# ---------------------------------------------------------------------------
# 1. Load data, build per-token stats
# ---------------------------------------------------------------------------
print(f"Loading {DATA_PATH} ...")
data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
df = pd.DataFrame(data["occurrences"])

per_token = (
    df.groupby("token_id")["kl"]
      .agg(freq="count", mean_kl="mean", std_kl="std")
      .reset_index()
)
per_token["std_kl"]    = per_token["std_kl"].fillna(0.0)
per_token = per_token[per_token["freq"] >= MIN_FREQ].copy()
per_token["freq_mean"] = per_token["freq"] * per_token["mean_kl"]

# Need at least 2 occurrences for std/CI to be meaningful
eligible = per_token[per_token["freq"] >= 2].copy()
eligible = eligible.sort_values("freq_mean", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Select rocks (rank 1-100) and controls (rank 101-200), then decode
# ---------------------------------------------------------------------------
print(f"Loading tokenizer {TOKENIZER_ID} ...")
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
def decode(tid):
    return repr(tok.decode([int(tid)]))

# Drop the N highest-frequency tokens from the candidate pool before ranking.
# These dominate freq*mean by frequency alone (e.g. spaces, common punctuation)
# and obscure the rock-token signal we care about.
if EXCLUDE_TOP_FREQ > 0:
    to_drop = eligible.nlargest(EXCLUDE_TOP_FREQ, "freq")
    print(f"Excluding top-{EXCLUDE_TOP_FREQ} highest-frequency tokens from rock+control pool:")
    for _, r in to_drop.iterrows():
        print(f"    id={int(r['token_id']):<6} {decode(r['token_id']):<20} "
              f"freq={int(r['freq']):>5}  mean_kl={r['mean_kl']:.4f}  "
              f"freq*mean={r['freq_mean']:.1f}")
    eligible = eligible[~eligible["token_id"].isin(to_drop["token_id"])].reset_index(drop=True)

rocks    = eligible.iloc[:TOP_K].reset_index(drop=True)
controls = eligible.iloc[TOP_K:2 * TOP_K].reset_index(drop=True)

out = pd.DataFrame({
    "rock_id":         rocks["token_id"].astype(int),
    "rock_token":      rocks["token_id"].apply(decode),
    "rock_freq":       rocks["freq"].astype(int),
    "rock_mean_kl":    rocks["mean_kl"],
    "rock_freq_mean":  rocks["freq_mean"],
    "ctrl_id":         controls["token_id"].astype(int),
    "ctrl_token":      controls["token_id"].apply(decode),
    "ctrl_freq":       controls["freq"].astype(int),
    "ctrl_mean_kl":    controls["mean_kl"],
    "ctrl_freq_mean":  controls["freq_mean"],
})
out.to_csv(NEW_CSV, index=False)
print(f"Saved {NEW_CSV}: {len(out)} rocks, {len(out)} controls")

# ---------------------------------------------------------------------------
# 3. Compare with the original rock list
# ---------------------------------------------------------------------------
old = pd.read_csv(OLD_CSV)
old_rocks = set(old["rock_id"].astype(int))
new_rocks = set(out["rock_id"].astype(int))
shared    = old_rocks & new_rocks
only_old  = old_rocks - new_rocks
only_new  = new_rocks - old_rocks
jaccard   = len(shared) / len(old_rocks | new_rocks)
print(f"\n=== Rock-token overlap with original (256-cap) selection ===")
print(f"  shared:                    {len(shared)} / {TOP_K}")
print(f"  only in old (256-cap):     {len(only_old)}")
print(f"  only in new (unrestricted): {len(only_new)}")
print(f"  Jaccard:                   {jaccard:.3f}")

# Print the dropped + added tokens
print(f"\n  Tokens dropped from rock list (in old, not in new):")
old_rock_only = old[old["rock_id"].astype(int).isin(only_old)][["rock_id","rock_token","rock_freq","rock_mean_kl","rock_freq_mean"]]
for _, r in old_rock_only.head(15).iterrows():
    print(f"    id={int(r['rock_id']):<6} {r['rock_token']:<20} f_old={int(r['rock_freq']):>5}  m_old={r['rock_mean_kl']:.3f}")

print(f"\n  Tokens added to rock list (in new, not in old):")
new_rock_only = out[out["rock_id"].astype(int).isin(only_new)]
for _, r in new_rock_only.head(15).iterrows():
    print(f"    id={int(r['rock_id']):<6} {r['rock_token']:<20} f_new={int(r['rock_freq']):>5}  m_new={r['rock_mean_kl']:.3f}  fxm={r['rock_freq_mean']:.1f}")

# ---------------------------------------------------------------------------
# 4. Per-output distribution analysis
# ---------------------------------------------------------------------------
rock_ids = set(out["rock_id"].astype(int).tolist())

PCT = 95.0
threshold = float(np.percentile(df["kl"].values, PCT))

df = df.copy()
df["is_rock"]         = df["token_id"].isin(rock_ids)
df["is_nonrock_high"] = (~df["is_rock"]) & (df["kl"] >= threshold)

rock_kl   = df.loc[df["is_rock"]].groupby("sample_idx")["kl"].sum().rename("rock_kl")
nrhigh_kl = df.loc[df["is_nonrock_high"]].groupby("sample_idx")["kl"].sum().rename("nonrock_high_kl")

po = df.groupby("sample_idx").agg(
    n_tokens          =("kl",              "count"),
    total_kl          =("kl",              "sum"),
    rock_count        =("is_rock",         "sum"),
    nonrock_high_count=("is_nonrock_high", "sum"),
).join(rock_kl, how="left").join(nrhigh_kl, how="left").fillna(0.0)

po["rock_pct"]              = 100 * po["rock_count"]         / po["n_tokens"]
po["nonrock_high_pct"]      = 100 * po["nonrock_high_count"] / po["n_tokens"]
po["rock_kl_share"]         = 100 * po["rock_kl"]            / po["total_kl"].clip(lower=1e-12)
po["nonrock_high_kl_share"] = 100 * po["nonrock_high_kl"]    / po["total_kl"].clip(lower=1e-12)
po.to_csv(PER_OUTPUT_CSV)

print(f"\n=== Per-output distribution (high-KL = top {100-PCT:.0f}%, threshold = {threshold:.4f}) ===\n")
metrics = [
    ("outputs",                          len(po)),
    ("total_tokens",                     int(po["n_tokens"].sum())),
    ("median_output_length",             float(po["n_tokens"].median())),
    ("output_length_p25_p75",            f"{po['n_tokens'].quantile(.25):.0f}-{po['n_tokens'].quantile(.75):.0f}"),
    ("rock_occurrences_total",           int(po["rock_count"].sum())),
    ("rock_pct_of_corpus",               100 * po["rock_count"].sum() / po["n_tokens"].sum()),
    ("rock_kl_share_of_total_pct",       100 * po["rock_kl"].sum()    / po["total_kl"].sum()),
    ("rock_count_per_output_median",     float(po["rock_count"].median())),
    ("rock_count_per_output_p25_p75",    f"{po['rock_count'].quantile(.25):.0f}-{po['rock_count'].quantile(.75):.0f}"),
    ("rock_count_per_output_max",        int(po["rock_count"].max())),
    ("high_kl_events_total",             int((df["kl"] >= threshold).sum())),
    ("nonrock_high_kl_events_total",     int(po["nonrock_high_count"].sum())),
    ("rock_share_of_high_kl_events_pct",
        100 * (int((df["kl"] >= threshold).sum()) - int(po["nonrock_high_count"].sum())) /
        max(int((df["kl"] >= threshold).sum()), 1)),
    ("nonrock_high_per_output_median",   float(po["nonrock_high_count"].median())),
    ("nonrock_high_per_output_max",      int(po["nonrock_high_count"].max())),
    ("nonrock_high_kl_share_pct",        100 * po["nonrock_high_kl"].sum() / po["total_kl"].sum()),
    ("outputs_with_zero_rock",           int((po["rock_count"]         == 0).sum())),
    ("outputs_with_zero_nonrock_high",   int((po["nonrock_high_count"] == 0).sum())),
]
for k, v in metrics:
    if isinstance(v, float):
        print(f"  {k:<40} {v:>16,.3f}")
    elif isinstance(v, int):
        print(f"  {k:<40} {v:>16,}")
    else:
        print(f"  {k:<40} {v:>16}")

print(f"\nSaved per-output table to {PER_OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# 5. Funnel plot — rock tokens highlighted
# ---------------------------------------------------------------------------
stats_all = per_token[per_token["freq"] >= 2].copy()
stats_all["log10_freq"] = np.log10(stats_all["freq"])

is_rock_pt = stats_all["token_id"].isin(rock_ids)
unsel = stats_all[~is_rock_pt]
rock  = stats_all[ is_rock_pt]

# Binned envelope
def envelope(s, n_bins=40):
    edges = np.linspace(s["log10_freq"].min(), s["log10_freq"].max(), n_bins + 1)
    cx, m, lo, hi = [], [], [], []
    for i in range(n_bins):
        mask = (s["log10_freq"] >= edges[i]) & (s["log10_freq"] < edges[i + 1])
        b = s.loc[mask, "mean_kl"]
        if len(b) < 2:
            continue
        cx.append((edges[i] + edges[i + 1]) / 2)
        m.append(b.mean()); lo.append(b.quantile(0.10)); hi.append(b.quantile(0.90))
    return map(np.array, (cx, m, lo, hi))

cx, mline, loline, hiline = envelope(stats_all)
global_mean = stats_all["mean_kl"].mean()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(unsel["log10_freq"], unsel["mean_kl"],
           s=6, alpha=0.25,
           label=f"Unselected ({len(unsel):,})", rasterized=True, zorder=2)
ax.fill_between(cx, loline, hiline, color="steelblue", alpha=0.15,
                label="10–90th percentile", zorder=3)
ax.plot(cx, mline, color="steelblue", linewidth=1.8, label="Binned mean KL", zorder=4)
ax.scatter(rock["log10_freq"], rock["mean_kl"],
           s=70, alpha=0.9, color="crimson", edgecolor="darkred", linewidth=0.7,
           marker="o", label=f"Rock tokens ({len(rock)})", zorder=6)
ax.axhline(global_mean, color="red", linewidth=1.0, linestyle="--", alpha=0.6,
           label=f"Global mean = {global_mean:.4f}", zorder=7)
ax.set_xlabel(r"$\log_{10}(\mathrm{Frequency})$", fontsize=13)
ax.set_ylabel("Average KL Divergence", fontsize=13)
ax.set_title(f"Rock Tokens on the Funnel — onpolicy unrestricted (n=500, no length cap)",
             fontsize=13)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.25)

if stats_all["mean_kl"].max() / max(stats_all["mean_kl"].median(), 1e-6) > 200:
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("Average KL Divergence (symlog)", fontsize=13)

plt.tight_layout()
plt.savefig(FUNNEL_PLOT, dpi=600, bbox_inches="tight")
print(f"\nSaved funnel plot to {FUNNEL_PLOT}")

print("\n=== Where the new rock tokens sit ===")
print(f"  rock log10(freq):   median={rock['log10_freq'].median():.2f}  "
      f"min={rock['log10_freq'].min():.2f}  max={rock['log10_freq'].max():.2f}")
print(f"  rock mean KL:       median={rock['mean_kl'].median():.4f}  "
      f"min={rock['mean_kl'].min():.4f}  max={rock['mean_kl'].max():.4f}")
print(f"  all-token median KL: {stats_all['mean_kl'].median():.4f}")
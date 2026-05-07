"""
Test the hypothesis:
  Rock tokens have gradient angles aligned with the global gradient,
  while high-KL low-frequency tokens have angles that diverge from it.

Inputs:
  - logit_gradients_*.pt   from compute_logit_gradients.py
  - rock_token_occurrences_*.pt  (for per-token freq + mean_kl)
  - rock_vs_control_unrestricted.csv  (rock list)

Groups compared:
  - rock          : top-100 by freq*mean (from rock_vs_control_unrestricted.csv)
  - high_kl       : top-100 by mean_kl with freq>=MIN_FREQ_HIGH_KL, excluding rocks
  - random_other  : random sample of remaining tokens (control baseline)

For each group we report cosine similarity of each token's average gradient
against:
  G_balanced  = sum_t  bar_g_t                      (per-type, frequency-balanced)
  G_global    = sum_t  freq_t * bar_g_t              (frequency-weighted, what SGD follows)

The frequency-balanced reference is the more meaningful test — see the
discussion of partial circularity in the chat.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--gradients",   default="logit_gradients_onpolicy_n500_unrestricted.pt")
parser.add_argument("--occurrences", default="rock_token_occurrences_onpolicy_n500_unrestricted.pt")
parser.add_argument("--rock-csv",    default="rock_vs_control_unrestricted.csv")
parser.add_argument("--top-k",       type=int, default=100)
parser.add_argument("--min-freq-high-kl", type=int, default=2,
                    help="Minimum frequency for the high-KL comparison group")
parser.add_argument("--random-sample", type=int, default=200,
                    help="Size of the random_other baseline group")
parser.add_argument("--seed",        type=int, default=0)
parser.add_argument("--output-plot", default="gradient_alignment.png")
parser.add_argument("--output-csv",  default="gradient_alignment.csv")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Load gradients
# ---------------------------------------------------------------------------
print(f"Loading {args.gradients} ...")
g = torch.load(args.gradients, map_location="cpu", weights_only=False)
seen_token_ids = g["seen_token_ids"].numpy()        # [n_seen]
gradient_sum   = g["gradient_sum"]                  # [n_seen, V] float32
gradient_count = g["gradient_count"]                # [n_seen]
G_global       = g["G_global"]                      # [V]
V              = g["vocab_size"]
n_seen         = len(seen_token_ids)

# Per-token average direction  bar_g_t = sum_t / count_t
valid = gradient_count > 0
bar_g = torch.zeros_like(gradient_sum)
bar_g[valid] = gradient_sum[valid] / gradient_count[valid].unsqueeze(1).to(torch.float32)

# Frequency-balanced reference (each token contributes once)
G_balanced = bar_g[valid].sum(dim=0)

# Sanity check: G_global should equal sum of (count * bar_g) over valid tokens
# (skip if you don't care; useful as a numerical sanity check)
recompute = (gradient_count[valid].unsqueeze(1).to(torch.float32) * bar_g[valid]).sum(dim=0)
print(f"  G_global vs recomputed: max-abs-diff = {(G_global - recompute).abs().max():.4e}")

# ---------------------------------------------------------------------------
# Cosine similarities
# ---------------------------------------------------------------------------
def cos_to(refvec, mat, valid_mask):
    """Cosine of each row in `mat` against `refvec`. Returns float32 tensor of length n_seen, NaN where invalid."""
    out = torch.full((mat.shape[0],), float("nan"))
    sub = mat[valid_mask]
    rn  = refvec.norm()
    if rn == 0:
        return out
    sub_n = sub.norm(dim=1)
    cos   = (sub @ refvec) / (sub_n * rn).clamp(min=1e-12)
    out[valid_mask] = cos
    return out

cos_balanced = cos_to(G_balanced, bar_g, valid)
cos_global   = cos_to(G_global,   bar_g, valid)

# ---------------------------------------------------------------------------
# Per-token freq + mean_kl from the occurrences file
# ---------------------------------------------------------------------------
print(f"Loading {args.occurrences} ...")
occ = torch.load(args.occurrences, map_location="cpu", weights_only=False)
df  = pd.DataFrame(occ["occurrences"])
per_tok = (
    df.groupby("token_id")["kl"]
      .agg(freq="count", mean_kl="mean")
      .reset_index()
)
per_tok["freq_mean"] = per_tok["freq"] * per_tok["mean_kl"]

# Build the unified per-token table
tok_table = pd.DataFrame({
    "token_id":    seen_token_ids,
    "row":         np.arange(n_seen),
    "n_grad":      gradient_count.numpy(),
    "cos_balanced": cos_balanced.numpy(),
    "cos_global":   cos_global.numpy(),
})
tok_table = tok_table.merge(per_tok, on="token_id", how="left")

# ---------------------------------------------------------------------------
# Define groups
# ---------------------------------------------------------------------------
rocks_csv  = pd.read_csv(args.rock_csv)
rock_ids   = set(rocks_csv["rock_id"].astype(int).tolist())

elig_for_high_kl = tok_table[
    (tok_table["freq"] >= args.min_freq_high_kl)
    & (~tok_table["token_id"].isin(rock_ids))
].copy()
high_kl_ids = set(
    elig_for_high_kl.sort_values("mean_kl", ascending=False).head(args.top_k)["token_id"]
)

eligible_other = tok_table[
    (tok_table["freq"] >= 2)
    & (~tok_table["token_id"].isin(rock_ids))
    & (~tok_table["token_id"].isin(high_kl_ids))
].copy()
rng = np.random.default_rng(args.seed)
random_other_ids = set(eligible_other.sample(
    n=min(args.random_sample, len(eligible_other)), random_state=args.seed
)["token_id"])

def label(tid):
    if tid in rock_ids:        return "rock"
    if tid in high_kl_ids:     return "high_kl"
    if tid in random_other_ids: return "random_other"
    return "—"

tok_table["group"] = tok_table["token_id"].apply(label)
groups = tok_table[tok_table["group"] != "—"].copy()
groups.to_csv(args.output_csv, index=False)
print(f"\nSaved {args.output_csv}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n=== Cosine similarity to G_balanced (frequency-balanced reference) ===")
print(f"{'group':<14} {'n':>4}  {'mean':>7}  {'median':>8}  {'p25':>7}  {'p75':>7}  {'>0 frac':>8}")
print("-" * 70)
for grp in ["rock", "high_kl", "random_other"]:
    g_df = groups[groups["group"] == grp]
    c    = g_df["cos_balanced"].dropna()
    if len(c) == 0:
        print(f"  {grp:<14} (no tokens)")
        continue
    print(f"  {grp:<14} {len(c):>4}  {c.mean():>+7.3f}  {c.median():>+8.3f}  "
          f"{c.quantile(.25):>+7.3f}  {c.quantile(.75):>+7.3f}  {(c > 0).mean():>8.2%}")

print("\n=== Cosine similarity to G_global (frequency-weighted reference, dominated by rocks by construction) ===")
print(f"{'group':<14} {'n':>4}  {'mean':>7}  {'median':>8}  {'p25':>7}  {'p75':>7}  {'>0 frac':>8}")
print("-" * 70)
for grp in ["rock", "high_kl", "random_other"]:
    g_df = groups[groups["group"] == grp]
    c    = g_df["cos_global"].dropna()
    if len(c) == 0:
        print(f"  {grp:<14} (no tokens)")
        continue
    print(f"  {grp:<14} {len(c):>4}  {c.mean():>+7.3f}  {c.median():>+8.3f}  "
          f"{c.quantile(.25):>+7.3f}  {c.quantile(.75):>+7.3f}  {(c > 0).mean():>8.2%}")

# ---------------------------------------------------------------------------
# Plot — distribution of cosines per group
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
group_colors = {"rock": "crimson", "high_kl": "darkorange", "random_other": "steelblue"}

for ax, ref_col, ref_name in [
    (axes[0, 0], "cos_balanced", "G_balanced  (per-type)"),
    (axes[0, 1], "cos_global",   "G_global    (per-position)"),
]:
    bins = np.linspace(-1, 1, 41)
    for grp in ["rock", "high_kl", "random_other"]:
        c = groups[groups["group"] == grp][ref_col].dropna()
        if len(c) == 0:
            continue
        ax.hist(c, bins=bins, alpha=0.55, color=group_colors[grp],
                label=f"{grp} (n={len(c)})", density=True)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel(f"cosine similarity to {ref_name}")
    ax.set_ylabel("density")
    ax.set_title(f"Per-token gradient alignment — {ref_name}")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

# Scatter: cos_balanced vs log10(freq), colored by group
ax = axes[1, 0]
for grp, color in group_colors.items():
    g_df = groups[groups["group"] == grp]
    ax.scatter(np.log10(g_df["freq"].values), g_df["cos_balanced"].values,
               s=18, alpha=0.55, color=color, label=f"{grp} (n={len(g_df)})")
ax.axhline(0, color="black", linewidth=1, alpha=0.5)
ax.set_xlabel(r"$\log_{10}(\mathrm{freq})$")
ax.set_ylabel(r"cosine similarity to $G_{\mathrm{balanced}}$")
ax.set_title("Alignment vs token frequency (balanced reference)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

# Scatter: cos_balanced vs mean_kl
ax = axes[1, 1]
for grp, color in group_colors.items():
    g_df = groups[groups["group"] == grp]
    ax.scatter(g_df["mean_kl"].values, g_df["cos_balanced"].values,
               s=18, alpha=0.55, color=color, label=f"{grp} (n={len(g_df)})")
ax.axhline(0, color="black", linewidth=1, alpha=0.5)
ax.set_xlabel("mean KL of token")
ax.set_ylabel(r"cosine similarity to $G_{\mathrm{balanced}}$")
ax.set_title("Alignment vs per-token KL (balanced reference)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

plt.suptitle("Gradient-Direction Alignment Hypothesis (Proxy A: logit-space gradients)", fontsize=13)
plt.tight_layout()
plt.savefig(args.output_plot, dpi=150, bbox_inches="tight")
print(f"\nSaved {args.output_plot}")
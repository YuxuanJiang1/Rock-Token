"""
Test the hypothesis:
  - High-KL low-frequency tokens have their KL **reduced** between checkpoints
    ("learned"), because their gradient is in a distinct direction the model
    can address with focused updates.
  - Rock tokens have **persistent** KL, because their gradient is aligned with
    the global descent direction so they don't get a separable signal.

We compute per-token mean KL at both checkpoints, take the change
   Δ KL = KL_finished - KL_midtraining
and break it down by group {rock, high_kl, random_other}. We also cross
the change against per-token cos(bar_g_t, G_balanced) at midtraining to
test the gradient-direction → loss-persistence link directly.

Inputs:
  rock_token_occurrences_onpolicy_n500_unrestricted.pt        (5k finished)
  rock_token_occurrences_onpolicy_10k_n500_unrestricted.pt    (10k mid-training)
  logit_gradients_onpolicy_10k_n500_unrestricted.pt           (gradients @ 10k)
  rock_vs_control_unrestricted.csv                            (rock list)
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--occ-finished",
    default="rock_token_occurrences_onpolicy_n500_unrestricted.pt")
parser.add_argument("--occ-midtraining",
    default="rock_token_occurrences_onpolicy_10k_n500_unrestricted.pt")
parser.add_argument("--grad-midtraining",
    default="logit_gradients_onpolicy_10k_n500_unrestricted.pt")
parser.add_argument("--rock-csv",
    default="rock_vs_control_unrestricted.csv")
parser.add_argument("--label-finished",   default="finished")
parser.add_argument("--label-midtraining", default="10k")
parser.add_argument("--top-k-high-kl",  type=int, default=100)
parser.add_argument("--n-random",       type=int, default=200)
parser.add_argument("--seed",           type=int, default=0)
parser.add_argument("--output-csv",  default="kl_evolution.csv")
parser.add_argument("--output-plot", default="kl_evolution.png")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Per-token KL stats from both occurrence files
# ---------------------------------------------------------------------------
def per_token_kl(path):
    print(f"Loading {path} ...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    df = pd.DataFrame(data["occurrences"])
    return (df.groupby("token_id")["kl"]
              .agg(freq="count", mean_kl="mean", median_kl="median")
              .reset_index())

per_fin = per_token_kl(args.occ_finished).add_suffix("_fin").rename(
    columns={"token_id_fin": "token_id"})
per_mid = per_token_kl(args.occ_midtraining).add_suffix("_mid").rename(
    columns={"token_id_mid": "token_id"})
combined = per_fin.merge(per_mid, on="token_id", how="outer")

# Δ KL = finished - midtraining; negative = KL went DOWN during further training
combined["delta_kl"]      = combined["mean_kl_fin"] - combined["mean_kl_mid"]
combined["log_ratio_kl"]  = np.log10(
    combined["mean_kl_fin"].clip(lower=1e-6) / combined["mean_kl_mid"].clip(lower=1e-6)
)

# ---------------------------------------------------------------------------
# 2. Gradient cosine to G_balanced @ midtraining (the explanatory variable)
# ---------------------------------------------------------------------------
print(f"Loading {args.grad_midtraining} ...")
g = torch.load(args.grad_midtraining, map_location="cpu", weights_only=False)
gs        = g["gradient_sum"]      # [n_seen, V]
gc        = g["gradient_count"]    # [n_seen]
seen_ids  = g["seen_token_ids"].numpy()
valid     = gc > 0
inv_count = torch.zeros(len(seen_ids), dtype=torch.float32)
inv_count[valid] = 1.0 / gc[valid].float()
G_balanced = (gs * inv_count.unsqueeze(1)).sum(dim=0)
G_norm     = G_balanced.norm()
norms_of_sums = gs.norm(dim=1)
mag_mid       = (norms_of_sums * inv_count).numpy()
inner         = (gs @ G_balanced) * inv_count
cos_mid       = (inner / (norms_of_sums * inv_count * G_norm).clamp(min=1e-12)).numpy()

grad_df = pd.DataFrame({
    "token_id": seen_ids,
    "mag_mid":  mag_mid,
    "cos_mid":  cos_mid,
})
combined = combined.merge(grad_df, on="token_id", how="left")

# ---------------------------------------------------------------------------
# 3. Group definitions  — same as the alignment analysis
# ---------------------------------------------------------------------------
rock_csv = pd.read_csv(args.rock_csv)
rock_ids = set(rock_csv["rock_id"].astype(int))

# high_kl: top-K by mean_kl in FINISHED model with freq>=2, excluding rocks
elig = combined[(combined["freq_fin"] >= 2) & (~combined["token_id"].isin(rock_ids))]
high_kl_ids = set(
    elig.sort_values("mean_kl_fin", ascending=False).head(args.top_k_high_kl)["token_id"]
)

elig_o = combined[
    (combined["freq_fin"] >= 2)
    & (~combined["token_id"].isin(rock_ids))
    & (~combined["token_id"].isin(high_kl_ids))
]
random_other_ids = set(elig_o.sample(
    n=min(args.n_random, len(elig_o)), random_state=args.seed,
)["token_id"])

def label(tid):
    tid = int(tid)
    if tid in rock_ids:        return "rock"
    if tid in high_kl_ids:     return "high_kl"
    if tid in random_other_ids: return "random_other"
    return "—"

combined["group"] = combined["token_id"].apply(label)
focal = combined[combined["group"] != "—"].copy()
focal.to_csv(args.output_csv, index=False)
print(f"Saved {args.output_csv}")

# ---------------------------------------------------------------------------
# 4. Console report — does KL drop more for high_kl than for rocks?
# ---------------------------------------------------------------------------
print(f"\n=== Per-token KL evolution: {args.label_midtraining} -> {args.label_finished} ===")
print(f"(delta_KL = KL_{args.label_finished} - KL_{args.label_midtraining}; "
      f"negative means KL was reduced during further training)\n")

print(f"{'group':<14} {'n':>4}  "
      f"{'KL_'+args.label_midtraining+' med':>14}  "
      f"{'KL_'+args.label_finished+' med':>14}  "
      f"{'dKL med':>10}  {'dKL mean':>10}  "
      f"{'% reduced':>10}  {'Wilcoxon p':>12}")
print("-" * 110)
for grp in ["rock", "high_kl", "random_other"]:
    sub = focal[(focal["group"] == grp)].dropna(subset=["mean_kl_fin", "mean_kl_mid"])
    if len(sub) == 0:
        continue
    pct_reduced = (sub["delta_kl"] < 0).mean()
    try:
        _, p = stats.wilcoxon(sub["mean_kl_mid"].values, sub["mean_kl_fin"].values)
    except Exception:
        p = float("nan")
    print(f"  {grp:<12} {len(sub):>4}  "
          f"{sub['mean_kl_mid'].median():>14.4f}  "
          f"{sub['mean_kl_fin'].median():>14.4f}  "
          f"{sub['delta_kl'].median():>+10.4f}  "
          f"{sub['delta_kl'].mean():>+10.4f}  "
          f"{pct_reduced:>9.1%}  "
          f"{p:>12.3e}")

# ---------------------------------------------------------------------------
# 5. Cross-test: does Δ KL correlate with cos(g, G_balanced) at midtraining?
# ---------------------------------------------------------------------------
print(f"\n=== Cross-correlation: delta_KL vs cos(g_t, G_balanced) at {args.label_midtraining} ===")
print(f"{'group':<14} {'Spearman rho':>14}  {'p':>12}")
for grp in ["rock", "high_kl", "random_other", "all (pooled)"]:
    if grp == "all (pooled)":
        sub = focal.dropna(subset=["delta_kl", "cos_mid"])
    else:
        sub = focal[focal["group"] == grp].dropna(subset=["delta_kl", "cos_mid"])
    if len(sub) < 5:
        continue
    rho, p = stats.spearmanr(sub["cos_mid"], sub["delta_kl"])
    print(f"  {grp:<14} {rho:>+14.3f}  {p:>12.3e}")
print("\nIf the hypothesis holds: tokens with HIGH cosine (rocks) should have dKL ~ 0;")
print("tokens with LOW cosine (high_kl) should have dKL < 0 (KL reduced).")
print("Pooled Spearman rho should be NEGATIVE (low cos -> big KL drop).")

# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
group_colors = {"rock": "crimson", "high_kl": "darkorange", "random_other": "steelblue"}
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (0,0) Scatter KL_mid vs KL_fin, log-log
ax = axes[0, 0]
for grp, c in group_colors.items():
    sub = focal[focal["group"] == grp].dropna(subset=["mean_kl_fin", "mean_kl_mid"])
    ax.scatter(sub["mean_kl_mid"], sub["mean_kl_fin"], s=28, alpha=0.6, color=c,
               label=f"{grp} (n={len(sub)})")
mx = max(focal["mean_kl_fin"].max(), focal["mean_kl_mid"].max()) * 1.1
mn = max(focal["mean_kl_fin"].min(), focal["mean_kl_mid"].min(), 1e-3)
ax.plot([mn, mx], [mn, mx], "--", color="gray", linewidth=1, label="y = x  (no change)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(f"mean KL ({args.label_midtraining})")
ax.set_ylabel(f"mean KL ({args.label_finished})")
ax.set_title("Per-token KL — paired across checkpoints (log-log)\n"
             "Below the diagonal = KL was reduced during further training")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3, which="both")

# (0,1) ΔKL distribution by group
ax = axes[0, 1]
all_d = focal["delta_kl"].dropna()
xmin, xmax = all_d.quantile(0.02), all_d.quantile(0.98)
bins = np.linspace(xmin, xmax, 41)
for grp, c in group_colors.items():
    sub = focal[focal["group"] == grp]["delta_kl"].dropna()
    if len(sub) == 0:
        continue
    ax.hist(sub.clip(xmin, xmax), bins=bins, alpha=0.55, color=c,
            label=f"{grp} (n={len(sub)})", density=True)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel(f"Δ KL = KL_{args.label_finished} − KL_{args.label_midtraining}")
ax.set_ylabel("density")
ax.set_title("Per-token KL change\n(left of zero = learned during further training)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (1,0) The hypothesis test: Δ KL vs cos(g, G_balanced) at midtraining
ax = axes[1, 0]
for grp, c in group_colors.items():
    sub = focal[focal["group"] == grp].dropna(subset=["delta_kl", "cos_mid"])
    ax.scatter(sub["cos_mid"], sub["delta_kl"], s=28, alpha=0.6, color=c,
               label=f"{grp} (n={len(sub)})")
ax.axhline(0, color="black", linewidth=0.7)
ax.axvline(0, color="black", linewidth=0.7)
ax.set_xlabel(rf"cos$\;(\bar g_t,\,G_{{\rm balanced}})$  at  {args.label_midtraining}")
ax.set_ylabel(f"Δ KL  ({args.label_finished} − {args.label_midtraining})")
ax.set_title("Hypothesis test: gradient alignment vs KL persistence\n"
             "→ aligned tokens (high cos) should stay near Δ KL ≈ 0 (persistent)\n"
             "→ orthogonal tokens (low cos) should be reduced (Δ KL < 0)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (1,1) Per-group Δ KL boxplot vs frequency-bucket
ax = axes[1, 1]
for grp, c in group_colors.items():
    sub = focal[focal["group"] == grp].dropna(subset=["delta_kl", "freq_fin"])
    if len(sub) == 0:
        continue
    ax.scatter(np.log10(sub["freq_fin"].values + 1), sub["delta_kl"].values,
               s=28, alpha=0.55, color=c, label=f"{grp} (n={len(sub)})")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xlabel(rf"$\log_{{10}}$(freq in {args.label_finished})")
ax.set_ylabel(f"Δ KL  ({args.label_finished} − {args.label_midtraining})")
ax.set_title("Δ KL vs token frequency")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(
    f"KL evolution between checkpoints   "
    f"({args.label_midtraining} → {args.label_finished})",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(args.output_plot, dpi=150, bbox_inches="tight")
print(f"\nSaved {args.output_plot}")

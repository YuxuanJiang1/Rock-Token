"""
Compare per-token gradient profiles between two checkpoints.

Inputs:
  - logit_gradients file for the FINISHED student (default: 5k)
  - logit_gradients file for a MID-TRAINING checkpoint (default: 10k)
  - rock_vs_control_unrestricted.csv (rock list selected from the finished model)
  - rock_token_occurrences_*.pt for the finished model (used to define high_kl group)

For each token in {rock, high_kl, random_other} we compute, at BOTH checkpoints:
    ||bar_g_t||  — magnitude of avg per-token logit gradient
    cos(bar_g_t, G_balanced)  — alignment with frequency-balanced descent direction

We compare:
  - per-group medians at each checkpoint
  - paired distribution of (mag_mid - mag_fin) and (cos_mid - cos_fin)
  - Wilcoxon signed-rank test on the paired differences

Use the same rock list at both checkpoints (selected from the finished model)
so the comparison is "do these particular tokens evolve as training progresses".
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--grad-finished",
    default="logit_gradients_onpolicy_n500_unrestricted.pt")
parser.add_argument("--grad-midtraining",
    default="logit_gradients_onpolicy_10k_n500_unrestricted.pt")
parser.add_argument("--label-finished",   default="finished")
parser.add_argument("--label-midtraining", default="midtraining")
parser.add_argument("--rock-csv",     default="rock_vs_control_unrestricted.csv")
parser.add_argument("--occurrences-finished",
    default="rock_token_occurrences_onpolicy_n500_unrestricted.pt")
parser.add_argument("--top-k-high-kl", type=int, default=100)
parser.add_argument("--n-random",      type=int, default=200)
parser.add_argument("--seed",          type=int, default=0)
parser.add_argument("--output-csv",  default="checkpoint_comparison.csv")
parser.add_argument("--output-plot", default="checkpoint_comparison.png")
args = parser.parse_args()


def per_token_stats(g_data):
    """Return per-token DataFrame with columns: token_id, count, mag, cos."""
    gs       = g_data["gradient_sum"]                # [n_seen, V]
    gc       = g_data["gradient_count"]              # [n_seen]
    seen_ids = g_data["seen_token_ids"].numpy()

    valid     = gc > 0
    inv_count = torch.zeros(len(seen_ids), dtype=torch.float32)
    inv_count[valid] = 1.0 / gc[valid].float()

    G_balanced = (gs * inv_count.unsqueeze(1)).sum(dim=0)
    G_norm     = G_balanced.norm()

    norms_of_sums = gs.norm(dim=1)
    mag = norms_of_sums * inv_count

    inner = (gs @ G_balanced) * inv_count
    cos   = inner / (mag * G_norm).clamp(min=1e-12)

    return pd.DataFrame({
        "token_id": seen_ids,
        "count":    gc.numpy(),
        "mag":      mag.numpy(),
        "cos":      cos.numpy(),
    })


print(f"Loading {args.grad_finished} ...")
g_fin = torch.load(args.grad_finished,    map_location="cpu", weights_only=False)
print(f"Loading {args.grad_midtraining} ...")
g_mid = torch.load(args.grad_midtraining, map_location="cpu", weights_only=False)

s_fin = per_token_stats(g_fin).rename(
    columns={"count": "count_fin", "mag": "mag_fin", "cos": "cos_fin"})
s_mid = per_token_stats(g_mid).rename(
    columns={"count": "count_mid", "mag": "mag_mid", "cos": "cos_mid"})

combined = s_fin.merge(s_mid, on="token_id", how="outer").fillna({
    "count_fin": 0, "count_mid": 0,
})

# ---------------------------------------------------------------------------
# Group definitions — same as analyze_gradient_alignment.py
# ---------------------------------------------------------------------------
rock_csv = pd.read_csv(args.rock_csv)
rock_ids = set(rock_csv["rock_id"].astype(int).tolist())

print(f"Loading {args.occurrences_finished} for high_kl selection ...")
occ = torch.load(args.occurrences_finished, map_location="cpu", weights_only=False)
df_occ  = pd.DataFrame(occ["occurrences"])
per_kl  = df_occ.groupby("token_id")["kl"].agg(freq="count", mean_kl="mean").reset_index()

elig = per_kl[(per_kl["freq"] >= 2) & (~per_kl["token_id"].isin(rock_ids))]
high_kl_ids = set(
    elig.sort_values("mean_kl", ascending=False).head(args.top_k_high_kl)["token_id"]
)

elig_o = per_kl[(per_kl["freq"] >= 2)
                & (~per_kl["token_id"].isin(rock_ids))
                & (~per_kl["token_id"].isin(high_kl_ids))]
random_other_ids = set(elig_o.sample(
    n=min(args.n_random, len(elig_o)), random_state=args.seed,
)["token_id"])


def label(tid):
    tid = int(tid)
    if tid in rock_ids:        return "rock"
    if tid in high_kl_ids:     return "high_kl"
    if tid in random_other_ids: return "random_other"
    return "—"


combined["group"] = combined["token_id"].astype(int).apply(label)
groups_df = combined[combined["group"] != "—"].copy()
groups_df.to_csv(args.output_csv, index=False)
print(f"Saved {args.output_csv}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def fmt(x, dp=4):
    if pd.isna(x):
        return "   nan"
    return f"{x:+.{dp}f}"


print(f"\n=== Gradient profile evolution: {args.label_finished} → {args.label_midtraining} ===\n")
print(f"{'group':<13} {'metric':<26} "
      f"{args.label_finished+' med':>16} "
      f"{args.label_midtraining+' med':>16} "
      f"{'paired Δ med':>16} "
      f"{'p (Wilcoxon)':>14} "
      f"{'n_paired':>10}")
print("-" * 120)
rows = []
for grp in ["rock", "high_kl", "random_other"]:
    sub = groups_df[groups_df["group"] == grp]
    if len(sub) == 0:
        continue
    for col_fin, col_mid, name in [
        ("mag_fin", "mag_mid", "||bar_g||"),
        ("cos_fin", "cos_mid", "cos(bar_g, G_balanced)"),
        ("count_fin", "count_mid", "n_t (occurrence count)"),
    ]:
        paired = sub.dropna(subset=[col_fin, col_mid])
        delta  = paired[col_mid] - paired[col_fin]
        if len(paired) > 5 and delta.std() > 0:
            try:
                _, p = stats.wilcoxon(paired[col_fin].values, paired[col_mid].values)
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")
        print(f"  {grp:<11} {name:<26} "
              f"{fmt(paired[col_fin].median()):>16} "
              f"{fmt(paired[col_mid].median()):>16} "
              f"{fmt(delta.median()):>16} "
              f"{(f'{p:.3e}' if not pd.isna(p) else '   nan'):>14} "
              f"{len(paired):>10}")
        rows.append({
            "group": grp, "metric": name,
            "median_finished":   paired[col_fin].median(),
            "median_midtraining": paired[col_mid].median(),
            "delta_median": delta.median(),
            "p_wilcoxon":   p,
            "n_paired":     len(paired),
        })

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
group_colors = {"rock": "crimson", "high_kl": "darkorange", "random_other": "steelblue"}
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (0,0) Magnitude paired scatter
ax = axes[0, 0]
for grp, c in group_colors.items():
    sub = groups_df[groups_df["group"] == grp].dropna(subset=["mag_fin", "mag_mid"])
    ax.scatter(sub["mag_fin"], sub["mag_mid"], s=24, alpha=0.55,
               color=c, label=f"{grp} (n={len(sub)})")
mx = max(groups_df["mag_fin"].max(), groups_df["mag_mid"].max())
ax.plot([0, mx], [0, mx], "--", color="gray", linewidth=1)
ax.set_xlabel(rf"$\|\bar{{g}}_t\|$  ({args.label_finished})")
ax.set_ylabel(rf"$\|\bar{{g}}_t\|$  ({args.label_midtraining})")
ax.set_title("Magnitude — paired per token")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

# (0,1) Cosine paired scatter
ax = axes[0, 1]
for grp, c in group_colors.items():
    sub = groups_df[groups_df["group"] == grp].dropna(subset=["cos_fin", "cos_mid"])
    ax.scatter(sub["cos_fin"], sub["cos_mid"], s=24, alpha=0.55,
               color=c, label=f"{grp} (n={len(sub)})")
mn = min(groups_df["cos_fin"].min(), groups_df["cos_mid"].min())
mx = max(groups_df["cos_fin"].max(), groups_df["cos_mid"].max())
ax.plot([mn, mx], [mn, mx], "--", color="gray", linewidth=1)
ax.axhline(0, color="black", linewidth=0.5); ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel(rf"cos to $G_{{\rm balanced}}$ ({args.label_finished})")
ax.set_ylabel(rf"cos to $G_{{\rm balanced}}$ ({args.label_midtraining})")
ax.set_title("Direction alignment — paired per token")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

# (0,2) Frequency paired
ax = axes[0, 2]
for grp, c in group_colors.items():
    sub = groups_df[groups_df["group"] == grp].dropna(subset=["count_fin", "count_mid"])
    sub = sub[(sub["count_fin"] > 0) & (sub["count_mid"] > 0)]
    ax.scatter(sub["count_fin"], sub["count_mid"], s=24, alpha=0.55,
               color=c, label=f"{grp} (n={len(sub)})")
ax.set_xscale("log"); ax.set_yscale("log")
mx = max(groups_df["count_fin"].max(), groups_df["count_mid"].max())
ax.plot([1, mx], [1, mx], "--", color="gray", linewidth=1)
ax.set_xlabel(rf"freq ({args.label_finished})")
ax.set_ylabel(rf"freq ({args.label_midtraining})")
ax.set_title("Frequency — paired per token (log-log)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

# (1,0) Magnitude difference distribution
ax = axes[1, 0]
for grp, c in group_colors.items():
    sub = groups_df[groups_df["group"] == grp].dropna(subset=["mag_fin", "mag_mid"])
    delta = (sub["mag_mid"] - sub["mag_fin"]).values
    if len(delta) == 0:
        continue
    ax.hist(delta, bins=30, alpha=0.55, color=c, label=f"{grp}", density=True)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel(rf"$\Delta \|\bar{{g}}_t\|$  ({args.label_midtraining}) − ({args.label_finished})")
ax.set_ylabel("density")
ax.set_title("Per-token magnitude change")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (1,1) Cosine difference distribution
ax = axes[1, 1]
for grp, c in group_colors.items():
    sub = groups_df[groups_df["group"] == grp].dropna(subset=["cos_fin", "cos_mid"])
    delta = (sub["cos_mid"] - sub["cos_fin"]).values
    if len(delta) == 0:
        continue
    ax.hist(delta, bins=30, alpha=0.55, color=c, label=f"{grp}", density=True)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel(rf"$\Delta\,$cos  ({args.label_midtraining}) − ({args.label_finished})")
ax.set_ylabel("density")
ax.set_title("Per-token alignment change")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (1,2) Per-group magnitude vs cosine arrows  (mag_fin → mag_mid in (cos, mag) space)
ax = axes[1, 2]
for grp, c in group_colors.items():
    sub = groups_df[groups_df["group"] == grp].dropna(
        subset=["mag_fin", "mag_mid", "cos_fin", "cos_mid"])
    for _, r in sub.iterrows():
        ax.annotate("", xy=(r["cos_mid"], r["mag_mid"]),
                    xytext=(r["cos_fin"], r["mag_fin"]),
                    arrowprops=dict(arrowstyle="->", color=c, alpha=0.45, lw=0.7))
    ax.scatter(sub["cos_fin"], sub["mag_fin"], s=18, alpha=0.7, color=c,
               edgecolor="white", linewidth=0.4, label=f"{grp} (start)")
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel(r"cos to $G_{\rm balanced}$")
ax.set_ylabel(r"$\|\bar{g}_t\|$")
ax.set_title(rf"Trajectory in (cos, mag) plane: {args.label_finished} → {args.label_midtraining}")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Per-token gradient evolution: {args.label_finished}  vs  {args.label_midtraining}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(args.output_plot, dpi=150, bbox_inches="tight")
print(f"\nSaved {args.output_plot}")

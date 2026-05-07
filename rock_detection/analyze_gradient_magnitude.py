"""
Gradient-magnitude analysis (follow-up to gradient_alignment.csv).

For each token type t we compute:
  - ||bar_g_t||                 norm of the average per-token logit gradient
  - signed_contribution_per_pos = cos(bar_g_t, G_*) * ||bar_g_t||
                                  (the per-position projection onto the descent direction)
  - total_contribution         = n_t * signed_contribution_per_pos
                                  (token's total contribution to G_*)

This decomposes  rock-token dominance  into three multiplicative factors:
       descent contribution  =  freq  ×  magnitude  ×  direction-cosine
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

GRAD_FILE  = "logit_gradients_onpolicy_n500_unrestricted.pt"
ALIGN_CSV  = "gradient_alignment.csv"
OUT_CSV    = "gradient_magnitude.csv"
OUT_PLOT   = "gradient_magnitude.png"

# ---------------------------------------------------------------------------
# Load gradient sums + alignment table
# ---------------------------------------------------------------------------
print(f"Loading {GRAD_FILE} ...")
g = torch.load(GRAD_FILE, map_location="cpu", weights_only=False)
gs       = g["gradient_sum"]        # [n_seen, V]
gc       = g["gradient_count"]      # [n_seen]
seen_ids = g["seen_token_ids"].numpy()
G_global = g["G_global"]
n_seen, V = gs.shape

# G_balanced = sum_t bar_g_t = sum_t (gradient_sum[t] / count[t])
valid_mask = gc > 0
inv_count  = torch.zeros(n_seen, dtype=torch.float32)
inv_count[valid_mask] = 1.0 / gc[valid_mask].float()
G_balanced = (gs * inv_count.unsqueeze(1)).sum(dim=0)

# Per-token magnitudes (||bar_g_t|| for tokens with count >= 1)
norms_of_sums = gs.norm(dim=1)                   # [n_seen]   ||sum_i g_i||
mag_avg_grad  = norms_of_sums * inv_count        # [n_seen]   ||bar_g_t||  (=norm/count)
print(f"  computed ||bar_g_t|| for {int(valid_mask.sum())} / {n_seen} tokens")

# Reference norms
norm_balanced = G_balanced.norm().item()
norm_global   = G_global.norm().item()
print(f"  ||G_balanced|| = {norm_balanced:.4f}")
print(f"  ||G_global||   = {norm_global:.4f}")

# ---------------------------------------------------------------------------
# Merge into the alignment table
# ---------------------------------------------------------------------------
align = pd.read_csv(ALIGN_CSV)
id_to_row = {int(tid): i for i, tid in enumerate(seen_ids)}
align["row"] = align["token_id"].astype(int).map(id_to_row)

align["mag_bar_g"] = align["row"].apply(lambda r: float(mag_avg_grad[r]) if r is not None else np.nan)

# Per-position contribution = ||bar_g_t|| * cos(bar_g_t, G_*)
# Equivalently: <bar_g_t, G_*>/||G_*||
align["proj_per_pos_balanced"] = align["mag_bar_g"] * align["cos_balanced"]
align["proj_per_pos_global"]   = align["mag_bar_g"] * align["cos_global"]

# Total contribution  =  n_t * per-position contribution
align["total_contrib_balanced"] = align["n_grad"] * align["proj_per_pos_balanced"]
align["total_contrib_global"]   = align["n_grad"] * align["proj_per_pos_global"]

align.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

# ---------------------------------------------------------------------------
# Per-group summary
# ---------------------------------------------------------------------------
print("\n=== Per-group magnitude summary ===")
print(f"{'group':<14} {'n':>3}  "
      f"{'||bar_g||  med':>14}  {'mean':>7}  {'p90':>7}  "
      f"{'(cos*mag) med':>14}  {'mean':>7}  "
      f"{'total_contrib_balanced sum':>26}")
print("-" * 110)
for grp in ["rock", "high_kl", "random_other"]:
    sub = align[align["group"] == grp]
    if len(sub) == 0:
        continue
    print(
        f"  {grp:<12} {len(sub):>3}  "
        f"{sub['mag_bar_g'].median():>14.4f}  "
        f"{sub['mag_bar_g'].mean():>7.4f}  "
        f"{sub['mag_bar_g'].quantile(.9):>7.4f}  "
        f"{sub['proj_per_pos_balanced'].median():>14.4e}  "
        f"{sub['proj_per_pos_balanced'].mean():>7.4e}  "
        f"{sub['total_contrib_balanced'].sum():>26.4e}"
    )

# Mann-Whitney tests on magnitudes
print("\n=== Magnitude rank-sum tests ===")
def mw(a_grp, b_grp, alt="greater", col="mag_bar_g"):
    a = align[align["group"] == a_grp][col].dropna().values
    b = align[align["group"] == b_grp][col].dropna().values
    u, p = stats.mannwhitneyu(a, b, alternative=alt)
    return u, p

u, p = mw("rock", "high_kl")
print(f"  rock > high_kl       (||bar_g||):                   U={u:.0f}  p={p:.3e}")
u, p = mw("rock", "random_other")
print(f"  rock > random_other  (||bar_g||):                   U={u:.0f}  p={p:.3e}")
u, p = mw("high_kl", "random_other")
print(f"  high_kl > random_other (||bar_g||):                 U={u:.0f}  p={p:.3e}")

u, p = mw("rock", "high_kl", col="proj_per_pos_balanced")
print(f"  rock > high_kl       (cos×||bar_g||, per-position): U={u:.0f}  p={p:.3e}")

# ---------------------------------------------------------------------------
# Multiplicative decomposition narrative
# ---------------------------------------------------------------------------
print("\n=== Multiplicative decomposition  (geometric means within each group) ===")
print(f"{'group':<14}  {'freq (geo-mean)':>16}  {'||bar_g|| (mean)':>17}  "
      f"{'cos_balanced (mean)':>20}  {'product (sum)':>16}")
print("-" * 100)
for grp in ["rock", "high_kl", "random_other"]:
    sub = align[align["group"] == grp]
    if len(sub) == 0:
        continue
    geo_freq = np.exp(np.log(sub["freq"].values + 1).mean())
    mean_mag = sub["mag_bar_g"].mean()
    mean_cos = sub["cos_balanced"].mean()
    sum_total = sub["total_contrib_balanced"].sum()
    print(
        f"  {grp:<12}  {geo_freq:>16.1f}  {mean_mag:>17.4f}  "
        f"{mean_cos:>20.4f}  {sum_total:>16.4e}"
    )

# ---------------------------------------------------------------------------
# Plots — 4 panels
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
group_colors = {"rock": "crimson", "high_kl": "darkorange", "random_other": "steelblue"}

# (0,0) Magnitude histograms
ax = axes[0, 0]
all_mags = align["mag_bar_g"].dropna()
bins = np.linspace(0, all_mags.quantile(0.99), 41)
for grp, color in group_colors.items():
    sub = align[align["group"] == grp]["mag_bar_g"].dropna()
    if len(sub) == 0:
        continue
    ax.hist(sub, bins=bins, alpha=0.55, color=color,
            label=f"{grp} (n={len(sub)})", density=True)
ax.set_xlabel(r"$\|\bar{g}_t\|$  (norm of avg per-token logit gradient)")
ax.set_ylabel("density")
ax.set_title("Per-token gradient magnitude by group")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (0,1) Magnitude vs cos_balanced  (the magnitude × direction picture)
ax = axes[0, 1]
for grp, color in group_colors.items():
    sub = align[align["group"] == grp]
    ax.scatter(sub["mag_bar_g"], sub["cos_balanced"], s=24, alpha=0.55,
               color=color, label=f"{grp} (n={len(sub)})")
ax.axhline(0, color="black", linewidth=1, alpha=0.5)
ax.set_xlabel(r"$\|\bar{g}_t\|$")
ax.set_ylabel(r"$\cos(\bar{g}_t, G_{\mathrm{balanced}})$")
ax.set_title("Magnitude × direction (per token)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

# (1,0) Per-position projection onto descent direction (signed magnitude)
ax = axes[1, 0]
all_proj = align["proj_per_pos_balanced"].dropna()
xmin = all_proj.quantile(0.02); xmax = all_proj.quantile(0.98)
bins = np.linspace(xmin, xmax, 41)
for grp, color in group_colors.items():
    sub = align[align["group"] == grp]["proj_per_pos_balanced"].dropna()
    if len(sub) == 0:
        continue
    ax.hist(sub.clip(xmin, xmax), bins=bins, alpha=0.55, color=color,
            label=f"{grp} (n={len(sub)})", density=True)
ax.axvline(0, color="black", linewidth=1, alpha=0.5)
ax.set_xlabel(r"signed per-position projection  $\cos\cdot\|\bar{g}_t\|$")
ax.set_ylabel("density")
ax.set_title(r"Per-position contribution to $G_{\mathrm{balanced}}$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (1,1) Magnitude vs log10(freq), colored by group, sized by cos
ax = axes[1, 1]
for grp, color in group_colors.items():
    sub = align[align["group"] == grp].dropna(subset=["mag_bar_g", "freq"])
    if len(sub) == 0:
        continue
    sizes = np.clip(sub["cos_balanced"].values * 200, 5, 200)
    ax.scatter(np.log10(sub["freq"].values), sub["mag_bar_g"].values,
               s=sizes, alpha=0.55, color=color, label=f"{grp}")
ax.set_xlabel(r"$\log_{10}(\mathrm{freq})$")
ax.set_ylabel(r"$\|\bar{g}_t\|$")
ax.set_title(r"Magnitude vs frequency  (dot size $\propto$ cos to $G_{\mathrm{balanced}}$)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

plt.suptitle(
    "Gradient Magnitude Analysis — multiplicative decomposition: "
    r"contribution $=$ freq $\times$ magnitude $\times$ direction",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"\nSaved {OUT_PLOT}")
"""
Compare per-output rock-token distribution: unrestricted vs 256-capped.
Both runs are on-policy.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rock_per_output import per_output_table, summary_block, fmt

ROCK_CSV     = "rock_vs_control.csv"
RESTRICTED   = "rock_token_occurrences_onpolicy_n500.pt"
UNRESTRICTED = "rock_token_occurrences_onpolicy_n500_unrestricted.pt"
PCT          = 95.0

rock_ids = set(pd.read_csv(ROCK_CSV)["rock_id"].astype(int).tolist())
print(f"Rock tokens loaded: {len(rock_ids)}")

def load(path):
    print(f"Loading {path} ...")
    return pd.DataFrame(torch.load(path, map_location="cpu", weights_only=False)["occurrences"])

df_r = load(RESTRICTED)
df_u = load(UNRESTRICTED)

thr_r = float(np.percentile(df_r["kl"].values, PCT))
thr_u = float(np.percentile(df_u["kl"].values, PCT))

po_r = per_output_table(df_r, rock_ids, thr_r)
po_u = per_output_table(df_u, rock_ids, thr_u)

s_r = summary_block(po_r, df_r, thr_r)
s_u = summary_block(po_u, df_u, thr_u)

# ---------------------------------------------------------------------------
# Headline output-length comparison
# ---------------------------------------------------------------------------
print("\n=== Output length comparison ===")
for label, po in [("restricted (256-cap)", po_r), ("unrestricted",       po_u)]:
    n = po["n_tokens"]
    print(f"  {label:<22} median={n.median():.0f}  "
          f"IQR={n.quantile(.25):.0f}-{n.quantile(.75):.0f}  "
          f"min={n.min():.0f}  max={n.max():.0f}  mean={n.mean():.1f}")

# ---------------------------------------------------------------------------
# Side-by-side summary table
# ---------------------------------------------------------------------------
print(f"\n=== Per-output rock distribution (high-KL = top {100 - PCT:.0f}%) ===\n")
print(f"{'metric':<42} {'restricted-256':>16} {'unrestricted':>16}")
print("-" * 78)
for k in s_r:
    v_r, v_u = s_r[k], s_u[k]
    if isinstance(v_r, float):
        prec = 3 if any(x in k for x in ["share", "pct", "median"]) else 2
        v_r, v_u = fmt(v_r, prec), fmt(v_u, prec)
    elif isinstance(v_r, int):
        v_r, v_u = f"{v_r:,}", f"{v_u:,}"
    print(f"  {k:<40} {v_r:>16} {v_u:>16}")

# ---------------------------------------------------------------------------
# Plot: length, rock count, rock pct, non-rock high-KL count
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (0,0) Output length distribution
ax = axes[0, 0]
ax.hist(po_r["n_tokens"], bins=40, alpha=0.6, color="steelblue", label="restricted (256)")
ax.hist(po_u["n_tokens"], bins=40, alpha=0.6, color="crimson",   label="unrestricted")
ax.set_xlabel("Output length (tokens)"); ax.set_ylabel("# outputs")
ax.set_title("Output length distribution"); ax.legend(); ax.grid(True, alpha=0.3)

# (0,1) Rock-token count per output
ax = axes[0, 1]
hi = max(po_r["rock_count"].max(), po_u["rock_count"].max())
bins = np.linspace(0, hi, 50)
ax.hist(po_r["rock_count"], bins=bins, alpha=0.6, color="steelblue", label="restricted (256)")
ax.hist(po_u["rock_count"], bins=bins, alpha=0.6, color="crimson",   label="unrestricted")
ax.set_xlabel("# rock-token occurrences per output"); ax.set_ylabel("# outputs")
ax.set_title("Rock-token count per output"); ax.legend(); ax.grid(True, alpha=0.3)

# (1,0) Rock-token percentage per output
ax = axes[1, 0]
ax.hist(po_r["rock_pct"], bins=40, alpha=0.6, color="steelblue", label="restricted (256)")
ax.hist(po_u["rock_pct"], bins=40, alpha=0.6, color="crimson",   label="unrestricted")
ax.set_xlabel("% of output that is rock tokens"); ax.set_ylabel("# outputs")
ax.set_title("Rock-token share of each output"); ax.legend(); ax.grid(True, alpha=0.3)

# (1,1) Non-rock high-KL count per output
ax = axes[1, 1]
hi = max(po_r["nonrock_high_count"].max(), po_u["nonrock_high_count"].max())
bins = np.arange(0, hi + 2) - 0.5
ax.hist(po_r["nonrock_high_count"], bins=bins, alpha=0.6, color="steelblue", label="restricted (256)")
ax.hist(po_u["nonrock_high_count"], bins=bins, alpha=0.6, color="crimson",   label="unrestricted")
ax.set_xlabel(f"# non-rock high-KL events per output (>p{PCT:.0f})"); ax.set_ylabel("# outputs")
ax.set_title("Non-rock high-KL count per output"); ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle("Restricted (256-cap) vs Unrestricted on-policy outputs", fontsize=13)
plt.tight_layout()
plt.savefig("compare_unrestricted.png", dpi=150, bbox_inches="tight")
print("\nSaved compare_unrestricted.png")
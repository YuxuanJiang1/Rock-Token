"""
Generate a frequency-matched non-rock control set for ablation.

Strategy:
  1. Rock tokens = top-50 by freq*mean on on-policy n=500.
  2. Eligible control pool = tokens NOT in the top-K_BUFFER by freq*mean.
     The buffer keeps controls clearly out of "near-rock" territory, ensuring
     KL contrast.
  3. Use Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
     the optimal 1-1 assignment minimizing total |log_freq(rock) - log_freq(ctrl)|.

Outputs:
  - rock_vs_control.csv  (side-by-side table)
  - rock_vs_control.pt   (token id lists for programmatic use)
"""

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from test_stability import load_data, score_freq_weighted, per_token

K = 100
K_BUFFER = 100    # exclude top-K_BUFFER by freq*mean from the control pool (= just the rocks)
DATA_PATH = "rock_token_occurrences_onpolicy_n500.pt"

# Optional tokenizer for decoding
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k"
    )
    decode = lambda tid: tokenizer.decode([tid])
except Exception as e:
    print(f"  tokenizer unavailable: {e}")
    decode = lambda tid: f"<id={tid}>"


# ---------------------------------------------------------------------------
# Load and compute per-token stats
# ---------------------------------------------------------------------------
print("Loading on-policy n=500...")
df_on = load_data("onpolicy", 500)
g     = per_token(df_on, min_n=1)
g["freq_mean"] = g["n"] * g["mean"]

# Median mean_kl across all seen tokens
median_mean_kl = g["mean"].median()
print(f"Vocab size (seen tokens): {len(g)}")
print(f"Median mean_kl: {median_mean_kl:.4f}")

# ---------------------------------------------------------------------------
# Pick rock tokens (top-K by freq*mean)
# ---------------------------------------------------------------------------
rock = g.nlargest(K, "freq_mean").copy()
rock_ids = set(rock.index)
print(f"\nRock tokens: top-{K} by freq*mean")
print(f"  Frequency range: {rock['n'].min()} – {rock['n'].max()}")
print(f"  mean_kl range:   {rock['mean'].min():.4f} – {rock['mean'].max():.4f}")
print(f"  freq*mean range: {rock['freq_mean'].min():.2f} – {rock['freq_mean'].max():.2f}")

# ---------------------------------------------------------------------------
# Eligible control pool: tokens not in top-K_BUFFER by freq*mean
# ---------------------------------------------------------------------------
top_buffer_ids = set(g.nlargest(K_BUFFER, "freq_mean").index)
pool = g.loc[~g.index.isin(top_buffer_ids)].copy()
pool["log_freq"] = np.log10(pool["n"].clip(lower=1))
print(f"\nEligible control pool size: {len(pool)} (excluding top-{K_BUFFER} by freq*mean)")

# ---------------------------------------------------------------------------
# Hungarian assignment: minimize total |log_freq(rock) - log_freq(control)|
# ---------------------------------------------------------------------------
rock_ids_list = rock.index.tolist()
pool_ids_list = pool.index.tolist()

rock_log = np.log10(rock["n"].clip(lower=1).values)         # shape (K,)
pool_log = pool["log_freq"].values                          # shape (P,)
cost = np.abs(rock_log[:, None] - pool_log[None, :])        # (K, P)

rock_pos, pool_pos = linear_sum_assignment(cost)
# rock_pos is a permutation of [0..K-1]; pool_pos gives the chosen pool index per rock
controls_by_rock = {
    rock_ids_list[rp]: pool_ids_list[pp]
    for rp, pp in zip(rock_pos, pool_pos)
}
controls = [controls_by_rock[tid] for tid in rock.index]

# ---------------------------------------------------------------------------
# Build side-by-side table
# ---------------------------------------------------------------------------
rows = []
for (rock_id, rock_row), ctrl_id in zip(rock.iterrows(), controls):
    if ctrl_id is None:
        ctrl_freq = ctrl_mean = ctrl_fm = float("nan")
        ctrl_str  = "<no match>"
    else:
        ctrl_row  = g.loc[ctrl_id]
        ctrl_freq = int(ctrl_row["n"])
        ctrl_mean = ctrl_row["mean"]
        ctrl_fm   = ctrl_row["freq_mean"]
        ctrl_str  = decode(ctrl_id)

    rows.append({
        "rock_id":         int(rock_id),
        "rock_token":      decode(rock_id),
        "rock_freq":       int(rock_row["n"]),
        "rock_mean_kl":    rock_row["mean"],
        "rock_freq_mean":  rock_row["freq_mean"],
        "ctrl_id":         int(ctrl_id) if ctrl_id is not None else -1,
        "ctrl_token":      ctrl_str,
        "ctrl_freq":       ctrl_freq,
        "ctrl_mean_kl":    ctrl_mean,
        "ctrl_freq_mean":  ctrl_fm,
    })

table = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 30)

print("\n=== Rock vs frequency-matched control ===")
print(f"{'rank':<5} {'rock_token':<20} {'r_freq':>7} {'r_mKL':>7} {'r_fm':>9} | "
      f"{'ctrl_token':<20} {'c_freq':>7} {'c_mKL':>7} {'c_fm':>9}")
print("-" * 110)
for i, r in table.iterrows():
    print(f"{i+1:<5} {r['rock_token'][:19]!s:<20} {r['rock_freq']:>7} "
          f"{r['rock_mean_kl']:>7.3f} {r['rock_freq_mean']:>9.2f} | "
          f"{r['ctrl_token'][:19]!s:<20} {r['ctrl_freq']:>7} "
          f"{r['ctrl_mean_kl']:>7.3f} {r['ctrl_freq_mean']:>9.2f}")

# ---------------------------------------------------------------------------
# Match-quality summary
# ---------------------------------------------------------------------------
log_diff = np.abs(np.log10(table["rock_freq"]) - np.log10(table["ctrl_freq"]))
print("\n=== Match quality (frequency) ===")
print(f"  median |Δlog10(freq)|: {log_diff.median():.3f}")
print(f"  max    |Δlog10(freq)|: {log_diff.max():.3f}")
print(f"  freq ratio median:     {(table['ctrl_freq']/table['rock_freq']).median():.3f}")

print("\n=== Mean-KL contrast ===")
print(f"  rock    mean of mean_kl: {table['rock_mean_kl'].mean():.4f}")
print(f"  control mean of mean_kl: {table['ctrl_mean_kl'].mean():.4f}")
print(f"  ratio (rock / control):  {table['rock_mean_kl'].mean() / table['ctrl_mean_kl'].mean():.1f}x")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
table.to_csv("rock_vs_control.csv", index=False)
print("\nSaved rock_vs_control.csv")

torch.save({
    "rock_ids":        table["rock_id"].tolist(),
    "control_ids":     table["ctrl_id"].tolist(),
    "rock_freqs":      table["rock_freq"].tolist(),
    "control_freqs":   table["ctrl_freq"].tolist(),
    "rock_mean_kl":    table["rock_mean_kl"].tolist(),
    "control_mean_kl": table["ctrl_mean_kl"].tolist(),
    "criterion":       "freq*mean top-50 vs frequency-matched non-rock with mean_kl < median",
    "source_file":     DATA_PATH,
    "K":               K,
}, "rock_vs_control.pt")
print("Saved rock_vs_control.pt")

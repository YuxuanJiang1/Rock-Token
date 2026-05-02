"""
Per-output distribution of rock tokens vs non-rock high-KL tokens.

For each generated sample (output), count:
  - rock_count          : rock-token occurrences
  - nonrock_high_count  : non-rock-token occurrences whose KL exceeds threshold
                          (default: 95th percentile of all KL values in that run)
  - rock_kl_share       : fraction of the output's total KL contributed by rocks
  - nonrock_high_kl_share : same, for non-rock high-KL events

Outputs:
  - Console: summary table comparing on-policy vs off-policy
  - CSV:     per-output table for both runs
  - PNG:     distribution histograms + rock-vs-nonrock scatter

Usage
-----
  python rock_per_output.py
  python rock_per_output.py --threshold-pct 99 --output-csv per_output.csv
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data-on",  default="rock_token_occurrences_onpolicy_n500.pt")
parser.add_argument("--data-off", default="rock_token_occurrences_offpolicy_n500.pt")
parser.add_argument("--rock-csv", default="rock_vs_control.csv")
parser.add_argument("--threshold-pct", type=float, default=95.0,
                    help="KL percentile defining 'high KL' (default 95)")
parser.add_argument("--output-csv",  default="per_output_distribution.csv")
parser.add_argument("--output-plot", default="per_output_distribution.png")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Load rock token IDs
# ---------------------------------------------------------------------------
rock_ids = set(pd.read_csv(args.rock_csv)["rock_id"].astype(int).tolist())
print(f"Rock tokens loaded: {len(rock_ids)}")


def load_occ(path):
    print(f"Loading {path} ...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    df = pd.DataFrame(data["occurrences"])
    return df, data.get("student_key", path)


def per_output_table(df, rock_ids, kl_threshold):
    """Aggregate occurrences into one row per sample_idx."""
    df = df.copy()
    df["is_rock"]         = df["token_id"].isin(rock_ids)
    df["is_nonrock_high"] = (~df["is_rock"]) & (df["kl"] >= kl_threshold)

    rock_kl    = df.loc[df["is_rock"]].groupby("sample_idx")["kl"].sum().rename("rock_kl")
    nrhigh_kl  = df.loc[df["is_nonrock_high"]].groupby("sample_idx")["kl"].sum().rename("nonrock_high_kl")

    g = df.groupby("sample_idx").agg(
        n_tokens          =("kl",              "count"),
        total_kl          =("kl",              "sum"),
        rock_count        =("is_rock",         "sum"),
        nonrock_high_count=("is_nonrock_high", "sum"),
    )
    g = g.join(rock_kl, how="left").join(nrhigh_kl, how="left").fillna(0.0)

    g["rock_pct"]              = 100 * g["rock_count"]         / g["n_tokens"]
    g["nonrock_high_pct"]      = 100 * g["nonrock_high_count"] / g["n_tokens"]
    g["rock_kl_share"]         = 100 * g["rock_kl"]            / g["total_kl"].clip(lower=1e-12)
    g["nonrock_high_kl_share"] = 100 * g["nonrock_high_kl"]    / g["total_kl"].clip(lower=1e-12)
    return g


# ---------------------------------------------------------------------------
# Process both runs
# ---------------------------------------------------------------------------
results = {}
for label, path in [("onpolicy", args.data_on), ("offpolicy", args.data_off)]:
    df, key = load_occ(path)
    threshold = float(np.percentile(df["kl"].values, args.threshold_pct))
    pot = per_output_table(df, rock_ids, threshold)
    results[label] = {"df": df, "po": pot, "thr": threshold, "key": key}

# ---------------------------------------------------------------------------
# Summary table (mean / median / IQR per metric)
# ---------------------------------------------------------------------------
def fmt(x, dp=2):
    return f"{x:,.{dp}f}"

def summary_block(po, df, thr):
    n_outputs   = len(po)
    n_tokens    = int(po["n_tokens"].sum())
    rock_total  = int(po["rock_count"].sum())
    nrh_total   = int(po["nonrock_high_count"].sum())
    high_total  = int((df["kl"] >= thr).sum())

    rock_kl_total = float(po["rock_kl"].sum())
    nrh_kl_total  = float(po["nonrock_high_kl"].sum())
    total_kl      = float(po["total_kl"].sum())

    return {
        "outputs":                          n_outputs,
        "total_tokens":                     n_tokens,
        "kl_threshold":                     thr,
        "rock_occurrences_total":           rock_total,
        "rock_occurrences_pct_of_corpus":   100 * rock_total / n_tokens,
        "high_kl_events_total":             high_total,
        "nonrock_high_kl_events_total":     nrh_total,
        "rock_share_of_high_kl_events":     100 * (high_total - nrh_total) / max(high_total, 1),
        "rock_count_per_output_median":     po["rock_count"].median(),
        "rock_count_per_output_p25_p75":    f"{po['rock_count'].quantile(.25):.0f}–{po['rock_count'].quantile(.75):.0f}",
        "rock_count_per_output_max":        po["rock_count"].max(),
        "nonrock_high_per_output_median":   po["nonrock_high_count"].median(),
        "nonrock_high_per_output_p25_p75":  f"{po['nonrock_high_count'].quantile(.25):.0f}–{po['nonrock_high_count'].quantile(.75):.0f}",
        "nonrock_high_per_output_max":      po["nonrock_high_count"].max(),
        "rock_kl_share_of_total_pct":       100 * rock_kl_total / max(total_kl, 1e-12),
        "nonrock_high_kl_share_pct":        100 * nrh_kl_total  / max(total_kl, 1e-12),
        "outputs_with_zero_rock":           int((po["rock_count"]         == 0).sum()),
        "outputs_with_zero_nonrock_high":   int((po["nonrock_high_count"] == 0).sum()),
    }

sum_on  = summary_block(results["onpolicy"]["po"],  results["onpolicy"]["df"],  results["onpolicy"]["thr"])
sum_off = summary_block(results["offpolicy"]["po"], results["offpolicy"]["df"], results["offpolicy"]["thr"])

print(f"\n=== Per-output distribution summary "
      f"(high-KL = top {100 - args.threshold_pct:.0f}% of all KL values) ===\n")
print(f"{'metric':<42} {'on-policy':>16} {'off-policy':>16}")
print("-" * 78)
for k in sum_on:
    v_on, v_off = sum_on[k], sum_off[k]
    if isinstance(v_on, float):
        v_on  = fmt(v_on, 3 if "share" in k or "pct" in k or "median" in k else 2)
        v_off = fmt(v_off, 3 if "share" in k or "pct" in k or "median" in k else 2)
    elif isinstance(v_on, int):
        v_on, v_off = f"{v_on:,}", f"{v_off:,}"
    print(f"  {k:<40} {v_on:>16} {v_off:>16}")

# ---------------------------------------------------------------------------
# Save per-output CSV (both runs concatenated, with a `run` column)
# ---------------------------------------------------------------------------
combined = pd.concat([
    results["onpolicy"]["po"].assign(run="onpolicy"),
    results["offpolicy"]["po"].assign(run="offpolicy"),
])
combined.to_csv(args.output_csv)
print(f"\nSaved per-output table to {args.output_csv}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram: rock-token count per output
ax = axes[0, 0]
bins = np.arange(0, max(results["onpolicy"]["po"]["rock_count"].max(),
                        results["offpolicy"]["po"]["rock_count"].max()) + 2) - 0.5
ax.hist(results["onpolicy"]["po"]["rock_count"],  bins=bins, alpha=0.6,
        color="crimson",  label="on-policy")
# ax.hist(results["offpolicy"]["po"]["rock_count"], bins=bins, alpha=0.6,
#         color="steelblue", label="off-policy")
ax.set_xlabel("# rock-token occurrences per output")
ax.set_ylabel("# outputs")
ax.set_title("Distribution of rock-token count per output")
ax.legend()
ax.grid(True, alpha=0.3)

# Histogram: non-rock high-KL count per output
ax = axes[0, 1]
bins = np.arange(0, max(results["onpolicy"]["po"]["nonrock_high_count"].max(),
                        results["offpolicy"]["po"]["nonrock_high_count"].max()) + 2) - 0.5
ax.hist(results["onpolicy"]["po"]["nonrock_high_count"],  bins=bins, alpha=0.6,
        color="crimson",  label="on-policy")
ax.hist(results["offpolicy"]["po"]["nonrock_high_count"], bins=bins, alpha=0.6,
        color="steelblue", label="off-policy")
ax.set_xlabel(f"# non-rock high-KL (>p{args.threshold_pct:.0f}) per output")
ax.set_ylabel("# outputs")
ax.set_title("Distribution of non-rock high-KL events per output")
ax.legend()
ax.grid(True, alpha=0.3)

# Scatter: rock_count vs nonrock_high_count, on-policy
for ax, label, color in [(axes[1, 0], "onpolicy", "crimson"),
                         (axes[1, 1], "offpolicy", "steelblue")]:
    po = results[label]["po"]
    ax.scatter(po["rock_count"], po["nonrock_high_count"],
               s=20, alpha=0.5, color=color)
    ax.set_xlabel("# rock-token occurrences per output")
    ax.set_ylabel(f"# non-rock high-KL (>p{args.threshold_pct:.0f}) per output")
    ax.set_title(f"{label}: per-output rock vs non-rock high-KL count")
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Rock vs non-rock high-KL — per-output distribution "
    f"(KL threshold = top {100 - args.threshold_pct:.0f}%)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(args.output_plot, dpi=150, bbox_inches="tight")
print(f"Saved plot to {args.output_plot}")
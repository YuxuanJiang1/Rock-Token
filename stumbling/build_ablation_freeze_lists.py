"""
Build the token-freeze lists needed for Reviewer RNAB's W4 ablations.

RNAB's request (paper rebuttal, weakness W4): the Fig. 5 "Random" baseline
(frequency-matched, semantically-useful tokens) is a weak control. To show that
the Rock Score construction *specifically* -- not just "freeze some frequent
and/or high-loss tokens" -- drives the efficiency gain, RNAB asks for freeze
sets built from each ingredient of the Rock Score in isolation, plus a
gradient-based selection:

    1. top_freq      -- top-K by Freq(v) alone (ignore mean KL entirely)
    2. top_meanloss   -- top-K by mean_kl(v) alone (ignore Freq weighting);
                         this is exactly the "high_kl" group already used for
                         Fig. 3 (rock_detection/gradient_alignment.py,
                         compare_kl_evolution.py), reused here for consistency
    3. top_gradmag    -- top-K by per-token gradient magnitude ||bar_g_t||
    4. top_gradalign  -- top-K by cosine alignment with G_balanced

All four use the same eligible candidate pool as the original rock selection
(freq >= MIN_FREQ, excluding the EXCLUDE_TOP_FREQ highest-frequency degenerate
tokens) so that the only thing that differs across lists -- and across the
four training runs that consume them -- is the ranking criterion, not the
candidate universe. Each list is independent; we do not force disjointness
from the rock set (unlike the rock/high_kl/random_other split used for Fig. 3,
which needs disjoint groups because they're compared side by side in one
plot). Overlap with the current rock list is reported as diagnostic context.

Inputs (same artifact formats as the rest of rock_detection/):
  --occurrences   rock_token_occurrences_onpolicy_n500_unrestricted.pt
                  (per-position records; produced by rock_server.py)
  --gradients     logit_gradients_onpolicy_n500_unrestricted.pt
                  (per-token gradient sums; produced by compute_logit_gradients.py)
                  Optional -- if missing, top_gradmag/top_gradalign are skipped
                  with a warning so top_freq/top_meanloss can still be produced.
  --rock-csv      rock_vs_control_unrestricted.csv (or rock_vs_control.csv)
                  Only used to report overlap with the current rock set.

Outputs (JSON list of int token ids, directly consumable by
stumbling/kdflow/algorithms/token_freeze_kd.py via --token_freeze_path):
  <outdir>/top_freq.json
  <outdir>/top_meanloss.json
  <outdir>/top_gradmag.json      (if gradients file provided)
  <outdir>/top_gradalign.json    (if gradients file provided)
  <outdir>/ablation_lists_summary.csv   -- overlap-with-rock diagnostics

Run this wherever the raw .pt artifacts live (the GPU server / wherever
rock_detection's other scripts are normally run) -- they are large (tens of MB
to a few GB) and are not checked into this repo.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--occurrences", default="rock_token_occurrences_onpolicy_n500_unrestricted.pt",
                     help="Per-position KL records (from rock_server.py)")
parser.add_argument("--gradients", default="logit_gradients_onpolicy_n500_unrestricted.pt",
                     help="Per-token gradient sums (from compute_logit_gradients.py); optional")
parser.add_argument("--rock-csv", default="rock_vs_control_unrestricted.csv",
                     help="Current rock/control list, for overlap reporting only")
parser.add_argument("--top-k", type=int, default=100, help="K, matching the paper's Rock-Token cutoff")
parser.add_argument("--min-freq", type=int, default=2, help="Minimum occurrence count to be eligible")
parser.add_argument("--exclude-top-freq", type=int, default=5,
                     help="Drop this many highest-frequency tokens from the eligible pool "
                          "(same convention as rerun_unrestricted.py, to avoid spaces/punctuation "
                          "trivially dominating every ranking)")
parser.add_argument("--outdir", default="ablation_lists")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)


def save_list(name, token_ids):
    path = os.path.join(args.outdir, f"{name}.json")
    with open(path, "w") as f:
        json.dump([int(t) for t in token_ids], f)
    print(f"  saved {path}  (n={len(token_ids)})")


def jaccard(a, b):
    a, b = set(a), set(b)
    if not (a | b):
        return float("nan")
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# 1. Per-token frequency + mean KL, same eligible pool as the original
#    rock/control selection (rerun_unrestricted.py / generate_control_tokens.py)
# ---------------------------------------------------------------------------
print(f"Loading {args.occurrences} ...")
occ = torch.load(args.occurrences, map_location="cpu", weights_only=False)
df = pd.DataFrame(occ["occurrences"])

per_token = (
    df.groupby("token_id")["kl"]
      .agg(freq="count", mean_kl="mean")
      .reset_index()
)
per_token = per_token[per_token["freq"] >= args.min_freq].copy()
per_token["freq_mean"] = per_token["freq"] * per_token["mean_kl"]

if args.exclude_top_freq > 0:
    to_drop = per_token.nlargest(args.exclude_top_freq, "freq")
    print(f"Excluding top-{args.exclude_top_freq} highest-frequency tokens from the eligible pool:")
    for _, r in to_drop.iterrows():
        print(f"    id={int(r['token_id']):<6} freq={int(r['freq']):>6}  mean_kl={r['mean_kl']:.4f}")
    per_token = per_token[~per_token["token_id"].isin(to_drop["token_id"])].reset_index(drop=True)

print(f"Eligible pool size: {len(per_token)} tokens (freq >= {args.min_freq})")

# ---------------------------------------------------------------------------
# 2. Current rock list, for overlap reporting
# ---------------------------------------------------------------------------
rock_ids = []
if os.path.exists(args.rock_csv):
    rocks_csv = pd.read_csv(args.rock_csv)
    rock_ids = rocks_csv["rock_id"].astype(int).tolist()
    print(f"Loaded {len(rock_ids)} current rock ids from {args.rock_csv} (for overlap reporting only)")
else:
    print(f"WARNING: {args.rock_csv} not found -- overlap-with-rock diagnostics will be skipped")

summary_rows = []

# ---------------------------------------------------------------------------
# 3. top_freq: top-K by Freq(v) alone
# ---------------------------------------------------------------------------
print("\n=== top_freq: top-K by Freq(v) alone ===")
top_freq = per_token.nlargest(args.top_k, "freq")
top_freq_ids = top_freq["token_id"].astype(int).tolist()
save_list("top_freq", top_freq_ids)
print(f"  freq range: {top_freq['freq'].min()}-{top_freq['freq'].max()}, "
      f"mean_kl range: {top_freq['mean_kl'].min():.4f}-{top_freq['mean_kl'].max():.4f}")
summary_rows.append(("top_freq", len(top_freq_ids), jaccard(top_freq_ids, rock_ids)))

# ---------------------------------------------------------------------------
# 4. top_meanloss: top-K by mean_kl(v) alone, excluding current rocks
#    (== the "high_kl" group already used for Fig. 3 -- reused here so the
#    two analyses stay consistent)
# ---------------------------------------------------------------------------
print("\n=== top_meanloss: top-K by mean_kl(v) alone, excluding current rocks ===")
elig_meanloss = per_token[~per_token["token_id"].isin(rock_ids)].copy()
top_meanloss = elig_meanloss.nlargest(args.top_k, "mean_kl")
top_meanloss_ids = top_meanloss["token_id"].astype(int).tolist()
save_list("top_meanloss", top_meanloss_ids)
print(f"  freq range: {top_meanloss['freq'].min()}-{top_meanloss['freq'].max()}, "
      f"mean_kl range: {top_meanloss['mean_kl'].min():.4f}-{top_meanloss['mean_kl'].max():.4f}")
summary_rows.append(("top_meanloss", len(top_meanloss_ids), jaccard(top_meanloss_ids, rock_ids)))

# ---------------------------------------------------------------------------
# 5 & 6. Gradient-based lists (optional -- needs compute_logit_gradients.py output)
# ---------------------------------------------------------------------------
if args.gradients and os.path.exists(args.gradients):
    print(f"\nLoading {args.gradients} ...")
    g = torch.load(args.gradients, map_location="cpu", weights_only=False)
    seen_token_ids = g["seen_token_ids"].numpy()
    gradient_sum = g["gradient_sum"]
    gradient_count = g["gradient_count"]

    valid = gradient_count > 0
    bar_g = torch.zeros_like(gradient_sum)
    bar_g[valid] = gradient_sum[valid] / gradient_count[valid].unsqueeze(1).to(torch.float32)
    G_balanced = bar_g[valid].sum(dim=0)

    grad_mag = bar_g.norm(dim=1)
    rn = G_balanced.norm()
    cos_balanced = torch.full((len(seen_token_ids),), float("nan"))
    if rn > 0:
        sub = bar_g[valid]
        cos_balanced[valid] = (sub @ G_balanced) / (sub.norm(dim=1) * rn).clamp(min=1e-12)

    grad_table = pd.DataFrame({
        "token_id": seen_token_ids,
        "grad_mag": grad_mag.numpy(),
        "cos_balanced": cos_balanced.numpy(),
    })
    # Restrict to the same eligible pool used above (freq floor + top-freq exclusion)
    grad_table = grad_table.merge(per_token[["token_id", "freq", "mean_kl"]], on="token_id", how="inner")

    print("\n=== top_gradmag: top-K by ||bar_g_t|| (gradient magnitude) ===")
    top_gradmag = grad_table.nlargest(args.top_k, "grad_mag")
    top_gradmag_ids = top_gradmag["token_id"].astype(int).tolist()
    save_list("top_gradmag", top_gradmag_ids)
    summary_rows.append(("top_gradmag", len(top_gradmag_ids), jaccard(top_gradmag_ids, rock_ids)))

    print("\n=== top_gradalign: top-K by cosine alignment with G_balanced ===")
    elig_align = grad_table.dropna(subset=["cos_balanced"])
    top_gradalign = elig_align.nlargest(args.top_k, "cos_balanced")
    top_gradalign_ids = top_gradalign["token_id"].astype(int).tolist()
    save_list("top_gradalign", top_gradalign_ids)
    summary_rows.append(("top_gradalign", len(top_gradalign_ids), jaccard(top_gradalign_ids, rock_ids)))
else:
    print(f"\nWARNING: gradients file '{args.gradients}' not found -- "
          f"skipping top_gradmag / top_gradalign. Run compute_logit_gradients.py first "
          f"if you want the gradient-based ablations.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
summary = pd.DataFrame(summary_rows, columns=["list", "n_tokens", "jaccard_with_rock"])
summary_path = os.path.join(args.outdir, "ablation_lists_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\n=== Summary (overlap with current rock list, n={len(rock_ids)}) ===")
print(summary.to_string(index=False))
print(f"\nSaved {summary_path}")

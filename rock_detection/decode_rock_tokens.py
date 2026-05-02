"""
Decode the rock-token IDs in rock_vs_control.csv into their string forms,
and show why they account for so many occurrences per output.

Uses the public Qwen3 tokenizer (same vocab as the student model).
"""

import argparse
import pandas as pd
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="rock_vs_control.csv")
parser.add_argument("--tokenizer-id", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
parser.add_argument("--top-n", type=int, default=100)
parser.add_argument("--output-csv", default="rock_tokens_decoded.csv")
args = parser.parse_args()

print(f"Loading tokenizer {args.tokenizer_id} ...")
tok = AutoTokenizer.from_pretrained(args.tokenizer_id)

rc = pd.read_csv(args.csv)
print(f"Loaded {len(rc)} rock/control rows from {args.csv}")

def decode(tid):
    s = tok.decode([int(tid)])
    return repr(s)  # repr keeps whitespace/special chars visible

rc["rock_decoded"] = rc["rock_id"].apply(decode)
if "ctrl_id" in rc.columns:
    rc["ctrl_decoded"] = rc["ctrl_id"].apply(decode)

# Sort by rock_freq*mean (descending) to show the heaviest hitters first
rc_sorted = rc.sort_values("rock_freq_mean", ascending=False).head(args.top_n).reset_index(drop=True)

print(f"\n=== Top-{args.top_n} rock tokens by freq × mean KL ===\n")
print(f"{'rank':>4}  {'id':>6}  {'token':<20}  {'freq':>6}  {'mean_kl':>8}  {'freq*mean':>10}")
print("-" * 70)
for i, row in rc_sorted.iterrows():
    print(f"{i+1:>4}  {int(row['rock_id']):>6}  {row['rock_decoded']:<20}  "
          f"{int(row['rock_freq']):>6}  {row['rock_mean_kl']:>8.4f}  {row['rock_freq_mean']:>10.2f}")

# ---------------------------------------------------------------------------
# Aggregate frequency stats
# ---------------------------------------------------------------------------
total_freq = int(rc["rock_freq"].sum())
print(f"\n=== Aggregate ===")
print(f"  Sum of rock-token frequencies: {total_freq:,}")
print(f"  (i.e. these {len(rc)} tokens occupy ~{total_freq:,} positions in the corpus)")
print(f"  Median freq per rock token:    {int(rc['rock_freq'].median())}")
print(f"  Top 10 rock tokens contribute: {int(rc.nlargest(10,'rock_freq')['rock_freq'].sum()):,} occurrences")
print(f"     ({100*rc.nlargest(10,'rock_freq')['rock_freq'].sum()/total_freq:.1f}% of all rock-token occurrences)")

# Save decoded csv
rc.to_csv(args.output_csv, index=False)
print(f"\nSaved decoded table to {args.output_csv}")
"""
Compute cross-setting Jaccard similarity between Rock Token sets produced by
rerun_unrestricted.py (or any script that outputs a rock_vs_control*.csv).

Usage examples
--------------
# Compare two CSV files directly:
  python compute_cross_jaccard.py \\
      --a rock_vs_control_qwen3_math.csv \\
      --b rock_vs_control_llama_math.csv \\
      --label-a "Qwen3-Math" --label-b "Llama-Math"

# Compare three or more files at once (all pairs):
  python compute_cross_jaccard.py \\
      --files rock_vs_control_qwen3_math.csv rock_vs_control_qwen3_code.csv \\
              rock_vs_control_llama_math.csv rock_vs_control_llama_code.csv \\
      --labels "Qwen3-Math" "Qwen3-Code" "Llama-Math" "Llama-Code" \\
      --ks 50 100 200 \\
      --output cross_jaccard_results.json

Output
------
  - Console: per-pair Jaccard table at each K (50, 100, 200 by default)
  - JSON (optional): full results including shared / unique token lists
"""

import argparse
import json
import sys
from pathlib import Path
from itertools import combinations

import pandas as pd

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Cross-setting Rock Token Jaccard analysis",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
input_grp = parser.add_mutually_exclusive_group(required=True)
input_grp.add_argument(
    "--files",
    nargs="+",
    metavar="CSV",
    help="Two or more rock_vs_control CSV files to compare (all pairs computed).",
)
input_grp.add_argument(
    "--a",
    metavar="CSV",
    help="First rock_vs_control CSV (use with --b for a single pair).",
)
parser.add_argument(
    "--b",
    metavar="CSV",
    help="Second rock_vs_control CSV (required when --a is given).",
)
parser.add_argument(
    "--labels",
    nargs="*",
    default=None,
    metavar="LABEL",
    help="Human-readable labels for each file (same order as --files). "
         "Defaults to the file stems.",
)
parser.add_argument(
    "--label-a",
    default=None,
    metavar="LABEL",
    help="Label for --a.",
)
parser.add_argument(
    "--label-b",
    default=None,
    metavar="LABEL",
    help="Label for --b.",
)
parser.add_argument(
    "--ks",
    nargs="+",
    type=int,
    default=[50, 100, 200],
    metavar="K",
    help="Top-K cutoffs to evaluate.",
)
parser.add_argument(
    "--rock-id-col",
    default="rock_id",
    metavar="COL",
    help="Column name for rock token IDs in the CSV.",
)
parser.add_argument(
    "--output",
    default=None,
    metavar="JSON",
    help="Optional path to write full results as JSON.",
)
parser.add_argument(
    "--decode-tokenizer",
    default=None,
    metavar="HF_REPO_OR_PATH",
    help="HuggingFace tokenizer to decode token IDs to strings in the JSON output. "
         "Skipped if not provided.",
)
args = parser.parse_args()

# ── Resolve file / label lists ────────────────────────────────────────────────
if args.files:
    csv_paths = [Path(f) for f in args.files]
    if args.labels:
        if len(args.labels) != len(csv_paths):
            parser.error(f"--labels count ({len(args.labels)}) must match --files count ({len(csv_paths)}).")
        labels = args.labels
    else:
        labels = [p.stem for p in csv_paths]
else:
    if not args.b:
        parser.error("--b is required when --a is given.")
    csv_paths = [Path(args.a), Path(args.b)]
    labels = [
        args.label_a or Path(args.a).stem,
        args.label_b or Path(args.b).stem,
    ]

for p in csv_paths:
    if not p.exists():
        print(f"[ERROR] File not found: {p}", file=sys.stderr)
        sys.exit(1)

# ── Load CSVs → sorted rock-ID lists ─────────────────────────────────────────
def load_rock_ids(path: Path, col: str) -> list[int]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {path}. "
                         f"Available: {list(df.columns)}")
    # Preserve the original ranking order (rows are already sorted by freq*mean desc)
    return df[col].astype(int).tolist()

rock_id_lists = {label: load_rock_ids(p, args.rock_id_col)
                 for label, p in zip(labels, csv_paths)}

max_k = max(args.ks)
for label, ids in rock_id_lists.items():
    if len(ids) < max_k:
        print(f"[WARN] '{label}' has only {len(ids)} rock tokens; "
              f"max K={max_k} will be clamped.", file=sys.stderr)

# ── Optional tokenizer for decoding ──────────────────────────────────────────
tok = None
if args.decode_tokenizer:
    from transformers import AutoTokenizer
    print(f"Loading tokenizer {args.decode_tokenizer} for decoding ...")
    tok = AutoTokenizer.from_pretrained(args.decode_tokenizer)

def decode_id(tid: int) -> str:
    if tok is None:
        return str(tid)
    return repr(tok.decode([tid]))

# ── Jaccard computation ───────────────────────────────────────────────────────
def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

results = []

pairs = list(combinations(labels, 2))

# ── Console header ────────────────────────────────────────────────────────────
col_w = max(len(f"{a} vs {b}") for a, b in pairs) + 2
k_w   = 10

print()
print("=" * (col_w + k_w * len(args.ks) + 4))
print(f"  {'Pair':<{col_w}}" + "".join(f"{'J@' + str(k):>{k_w}}" for k in args.ks))
print("-" * (col_w + k_w * len(args.ks) + 4))

for label_a, label_b in pairs:
    ids_a = rock_id_lists[label_a]
    ids_b = rock_id_lists[label_b]
    pair_label = f"{label_a} vs {label_b}"
    pair_result = {"pair": pair_label, "label_a": label_a, "label_b": label_b, "by_k": {}}

    row_str = f"  {pair_label:<{col_w}}"
    for k in args.ks:
        set_a = set(ids_a[:k])
        set_b = set(ids_b[:k])
        j = jaccard(set_a, set_b)
        row_str += f"{j:>{k_w}.3f}"

        shared     = sorted(set_a & set_b)
        only_a     = sorted(set_a - set_b)
        only_b     = sorted(set_b - set_a)
        pair_result["by_k"][str(k)] = {
            "jaccard":   round(j, 4),
            "shared_n":  len(shared),
            "only_a_n":  len(only_a),
            "only_b_n":  len(only_b),
            "shared":    [{"id": t, "token": decode_id(t)} for t in shared],
            "only_a":    [{"id": t, "token": decode_id(t)} for t in only_a],
            "only_b":    [{"id": t, "token": decode_id(t)} for t in only_b],
        }
    print(row_str)
    results.append(pair_result)

print("=" * (col_w + k_w * len(args.ks) + 4))
print()

# ── Per-pair detail block ─────────────────────────────────────────────────────
for pr in results:
    print(f"  ── {pr['pair']} ──")
    for k_str, kdata in pr["by_k"].items():
        shared_toks = ", ".join(decode_id(t["id"]) for t in kdata["shared"][:10])
        if len(kdata["shared"]) > 10:
            shared_toks += f"  ... (+{len(kdata['shared'])-10} more)"
        print(f"     K={k_str:>3}  Jaccard={kdata['jaccard']:.3f}  "
              f"shared={kdata['shared_n']}  only_{pr['label_a']}={kdata['only_a_n']}  "
              f"only_{pr['label_b']}={kdata['only_b_n']}")
        print(f"            shared tokens: {shared_toks}")
    print()

# ── Aggregate summary (Table 5 values) ───────────────────────────────────────
print("  Table 5 values (copy-paste ready):")
print(f"  {'Pair':<{col_w}}" + "".join(f"{'J@' + k:>{k_w}}" for k in map(str, args.ks)))
print("  " + "-" * (col_w + k_w * len(args.ks)))
for pr in results:
    row = f"  {pr['pair']:<{col_w}}"
    for k_str in map(str, args.ks):
        row += f"{pr['by_k'][k_str]['jaccard']:>{k_w}.3f}"
    print(row)
print()

# ── JSON output ───────────────────────────────────────────────────────────────
if args.output:
    out = {
        "settings": {label: str(p) for label, p in zip(labels, csv_paths)},
        "ks": args.ks,
        "pairs": results,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  Full results written to {args.output}")

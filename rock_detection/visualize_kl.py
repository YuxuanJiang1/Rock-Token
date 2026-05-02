import json
import sys

INPUT_FILE = "/workspace/output_inspection.json"
TOP_K_TOKENS = 10  # highlight this many high-KL tokens per sample
BAR_MAX_WIDTH = 30  # max width of the KL bar in characters

def kl_bar(kl, max_kl):
    filled = int((kl / max_kl) * BAR_MAX_WIDTH) if max_kl > 0 else 0
    return "[" + "#" * filled + "-" * (BAR_MAX_WIDTH - filled) + "]"

def print_sample(idx, sample):
    print("=" * 80)
    print(f"Sample {idx + 1}")
    print(f"Problem : {sample['problem'][:120]}{'...' if len(sample['problem']) > 120 else ''}")
    print(f"Total KL: {sample['total_kl']:.4f}  |  Mean KL: {sample['mean_kl']:.4f}")
    print()

    tokens = sample["tokens"]
    max_kl = max(t["kl"] for t in tokens) if tokens else 1.0

    # Print token + KL in two aligned lines, wrapping at LINE_WIDTH columns
    print("--- Token sequence with KL ---")
    LINE_WIDTH = 120
    chunk, chunk_width = [], 0
    def flush(chunk):
        tok_line = ""
        kl_line  = ""
        for tok_str, kl_str in chunk:
            w = max(len(tok_str), len(kl_str)) + 1
            tok_line += tok_str.ljust(w)
            kl_line  += kl_str.ljust(w)
        print(tok_line)
        print(kl_line)
        print()

    for t in tokens:
        tok_str = repr(t["token"])          # e.g. "'think'" or "'\\n'"
        kl_str  = f"{t['kl']:.3f}"
        col_w   = max(len(tok_str), len(kl_str)) + 1
        if chunk and chunk_width + col_w > LINE_WIDTH:
            flush(chunk)
            chunk, chunk_width = [], 0
        chunk.append((tok_str, kl_str))
        chunk_width += col_w
    if chunk:
        flush(chunk)


    # Print top-K highest KL tokens as a ranked bar chart
    sorted_tokens = sorted(enumerate(tokens), key=lambda x: x[1]["kl"], reverse=True)
    top_k = sorted_tokens[:TOP_K_TOKENS]

    print(f"--- Top {TOP_K_TOKENS} tokens by KL divergence ---")
    print(f"{'Pos':<6} {'Token':<20} {'KL':>8}  Bar")
    print("-" * 70)
    for pos, t in top_k:
        token_repr = repr(t["token"])[:18]
        print(f"{pos:<6} {token_repr:<20} {t['kl']:>8.4f}  {kl_bar(t['kl'], max_kl)}")
    print()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

SAMPLE_INDEX = None  # Set to an int (e.g. 3) to view a single sample, or None for all

sample_range = range(len(results))
if SAMPLE_INDEX is not None:
    sample_range = [SAMPLE_INDEX]
elif len(sys.argv) > 1 and sys.argv[1].lstrip("-").isdigit():
    sample_range = [int(sys.argv[1])]

for idx in sample_range:
    print_sample(idx, results[idx])

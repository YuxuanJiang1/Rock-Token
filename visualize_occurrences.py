"""
Visualize rock-token occurrence data produced by rock_server.py.

Modes (auto-detected from --files):
  single     : one file  -> positional analysis + top-K bar chart + variance chart
  stability  : multiple files, same student key -> top-K rank stability vs sample size
  compare    : two files, same sample size, different students -> side-by-side top-K

Usage examples
--------------
  # single-file positional analysis
  python visualize_occurrences.py --files rock_token_occurrences_onpolicy_n200.pt

  # stability across sample sizes (same model)
  python visualize_occurrences.py \
      --files rock_token_occurrences_onpolicy_n50.pt \
              rock_token_occurrences_onpolicy_n100.pt \
              rock_token_occurrences_onpolicy_n200.pt \
              rock_token_occurrences_onpolicy_n300.pt \
              rock_token_occurrences_onpolicy_n400.pt \
              rock_token_occurrences_onpolicy_n500.pt

  # compare onpolicy vs offpolicy at n=500
  python visualize_occurrences.py \
      --files rock_token_occurrences_onpolicy_n500.pt \
              rock_token_occurrences_offpolicy_n500.pt

Options
-------
  --top-k     Number of top rock tokens to highlight (default: 20)
  --min-freq  Minimum frequency to include a token in top-K (default: 5)
  --max-line  Cap on line_index axis to avoid long-tail clutter (default: 30)
  --output    Save figure to this path instead of showing interactively
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--files", nargs="+", required=True)
parser.add_argument("--top-k",   type=int, default=20)
parser.add_argument("--min-freq",type=int, default=5)
parser.add_argument("--max-line",type=int, default=30)
parser.add_argument("--output",  default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_file(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    df   = pd.DataFrame(data["occurrences"])
    return data, df

print("Loading files...")
datasets = [(path, *load_file(path)) for path in args.files]

# Shared tokenizer — use the first file's student_id
tokenizer_id = datasets[0][1].get("student_id",
    "RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k")
print(f"Loading tokenizer from {tokenizer_id} ...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

def decode(tid):
    return repr(tokenizer.decode([tid]))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def top_k_tokens(data, df, k, min_freq):
    """Return DataFrame of top-k tokens by average KL, filtered by min_freq.

    Columns: token_id, token_str, avg_kl, freq, kl_var, kl_std
    kl_var/kl_std are computed from per-occurrence records (population variance).
    """
    avg_kl = data["average_kl"]
    freq   = data["frequencies"]
    mask   = freq >= min_freq
    ids    = mask.nonzero(as_tuple=True)[0]
    vals   = avg_kl[ids]
    top    = torch.topk(vals, k=min(k, len(ids)))
    tids   = ids[top.indices].tolist()

    token_var = df.groupby("token_id")["kl"].var(ddof=0)
    return pd.DataFrame({
        "token_id":  tids,
        "token_str": [decode(t) for t in tids],
        "avg_kl":    top.values.tolist(),
        "freq":      freq[tids].tolist(),
        "kl_var":    [float(token_var.get(t, float("nan"))) for t in tids],
        "kl_std":    [float(token_var.get(t, float("nan")) ** 0.5) for t in tids],
    })

def binned_mean(df, col, bins=30, max_val=None):
    """Return (bin_centers, mean_kl) after binning df[col]."""
    s = df[col]
    if max_val is not None:
        df = df[s <= max_val]
        s  = df[col]
    edges = np.linspace(s.min(), s.max(), bins + 1)
    df    = df.copy()
    df["_bin"] = pd.cut(s, bins=edges, include_lowest=True)
    grp   = df.groupby("_bin", observed=False)["kl"].mean()
    centers = [(iv.left + iv.right) / 2 for iv in grp.index]
    return np.array(centers), grp.values

# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------
student_keys = [d.get("student_key", "?") for _, d, _ in datasets]
sample_sizes = [d.get("samples_processed", 0) for _, d, _ in datasets]

unique_students = set(student_keys)
unique_sizes    = set(sample_sizes)

if len(datasets) == 1:
    mode = "single"
elif len(unique_students) == 1 and len(unique_sizes) > 1:
    mode = "stability"
elif len(unique_students) == 2 and len(unique_sizes) == 1:
    mode = "compare"
elif len(unique_students) == 2:
    mode = "compare"
else:
    mode = "stability"

print(f"Detected mode: {mode}")

# ---------------------------------------------------------------------------
# SINGLE — positional analysis
# ---------------------------------------------------------------------------
if mode == "single":
    _, data, df = datasets[0]
    label = f"{data.get('student_key','?')}  n={data.get('samples_processed','?')}"

    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(f"Rock Token Positional Analysis\n{label}", fontsize=13, y=0.99)

    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.35)

    topk = top_k_tokens(data, df, args.top_k, args.min_freq)
    tok_labels = [
        f"{s}  (id={tid}, f={freq})"
        for s, tid, freq in zip(topk["token_str"], topk["token_id"], topk["freq"])
    ]

    # --- (0,:) Top-K avg KL bar chart with std-dev error bars ---
    ax0 = fig.add_subplot(gs[0, :])
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(topk)))[::-1]
    ax0.barh(
        range(len(topk)), topk["avg_kl"], color=colors,
        xerr=topk["kl_std"], error_kw=dict(ecolor="gray", capsize=3, linewidth=0.8),
    )
    ax0.set_yticks(range(len(topk)))
    ax0.set_yticklabels(tok_labels, fontsize=8)
    ax0.invert_yaxis()
    ax0.set_xlabel("Average KL Divergence  (error bars = ±1 std dev)")
    ax0.set_title(f"Top-{args.top_k} Rock Tokens by Average KL  (min_freq={args.min_freq})")
    ax0.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # --- (1,:) Top-K KL variance bar chart ---
    ax_var = fig.add_subplot(gs[1, :])
    colors_var = plt.cm.Purples(np.linspace(0.4, 0.9, len(topk)))[::-1]
    ax_var.barh(range(len(topk)), topk["kl_var"], color=colors_var)
    ax_var.set_yticks(range(len(topk)))
    ax_var.set_yticklabels(tok_labels, fontsize=8)
    ax_var.invert_yaxis()
    ax_var.set_xlabel("KL Variance (population)")
    ax_var.set_title(f"Top-{args.top_k} Rock Tokens — KL Variance")
    ax_var.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    x_max_var = topk["kl_var"].max()
    for i, v in enumerate(topk["kl_var"]):
        ax_var.text(v + x_max_var * 0.01, i, f"{v:.4f}", va="center", fontsize=7)

    # --- (2,0) Mean KL by absolute position (binned) ---
    ax1 = fig.add_subplot(gs[2, 0])
    cx, my = binned_mean(df, "abs_position", bins=40)
    ax1.plot(cx, my, color="steelblue", linewidth=1.5)
    ax1.fill_between(cx, my, alpha=0.2, color="steelblue")
    ax1.set_xlabel("Absolute Position in Sequence")
    ax1.set_ylabel("Mean KL")
    ax1.set_title("Mean KL vs Sequence Position")

    # --- (2,1) Mean KL by relative position (binned) ---
    ax2 = fig.add_subplot(gs[2, 1])
    cx, my = binned_mean(df, "rel_position", bins=40)
    ax2.plot(cx, my, color="darkorange", linewidth=1.5)
    ax2.fill_between(cx, my, alpha=0.2, color="darkorange")
    ax2.set_xlabel("Relative Position (0 = start, 1 = end)")
    ax2.set_ylabel("Mean KL")
    ax2.set_title("Mean KL vs Relative Position")

    # --- (3,0) Mean KL by line index ---
    ax3 = fig.add_subplot(gs[3, 0])
    line_df = df[df["line_index"] <= args.max_line]
    grp = line_df.groupby("line_index")["kl"].mean()
    ax3.bar(grp.index, grp.values, color="mediumseagreen", width=0.7)
    ax3.set_xlabel(f"Line Index (capped at {args.max_line})")
    ax3.set_ylabel("Mean KL")
    ax3.set_title("Mean KL vs Line Index (newline-delimited steps)")

    # --- (3,1) Line-start effect ---
    ax4 = fig.add_subplot(gs[3, 1])
    groups = [
        df[df["is_line_start"] & ~df["is_newline"]]["kl"].values,
        df[~df["is_line_start"] & ~df["is_newline"]]["kl"].values,
        df[df["is_newline"]]["kl"].values,
    ]
    labels = ["Line start\n(first token)", "Mid-line", "Newline token"]
    bp = ax4.boxplot(groups, labels=labels, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5),
                     showfliers=False)
    colors_bp = ["#f4a261", "#2a9d8f", "#e76f51"]
    for patch, c in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax4.set_ylabel("KL Divergence")
    ax4.set_title("KL by Token Position Type\n(fliers hidden)")

# ---------------------------------------------------------------------------
# STABILITY — rank stability as sample size grows
# ---------------------------------------------------------------------------
elif mode == "stability":
    # Sort by sample size
    datasets = sorted(datasets, key=lambda x: x[1].get("samples_processed", 0))
    sizes  = [d.get("samples_processed", 0) for _, d, _ in datasets]
    label  = datasets[0][1].get("student_key", "?")

    topk_dfs = [top_k_tokens(d, df, args.top_k, args.min_freq) for _, d, df in datasets]

    # --- Jaccard overlap with the largest-N run ---
    reference_ids = set(topk_dfs[-1]["token_id"])
    jaccard = []
    for tdf in topk_dfs:
        ids = set(tdf["token_id"])
        j   = len(ids & reference_ids) / len(ids | reference_ids) if ids else 0
        jaccard.append(j)

    # --- Spearman rank correlation with the largest-N run ---
    ref_ranks = {tid: rank for rank, tid in enumerate(topk_dfs[-1]["token_id"])}
    spearman  = []
    for tdf in topk_dfs:
        common = [tid for tid in tdf["token_id"] if tid in ref_ranks]
        if len(common) < 2:
            spearman.append(np.nan)
            continue
        ranks_here = [tdf.index[tdf["token_id"] == t].tolist()[0] for t in common]
        ranks_ref  = [ref_ranks[t] for t in common]
        corr = np.corrcoef(
            pd.Series(ranks_here).rank().values,
            pd.Series(ranks_ref).rank().values,
        )[0, 1]
        spearman.append(corr)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Rock Token Stability  —  student={label}", fontsize=13)

    # Jaccard
    axes[0].plot(sizes, jaccard, marker="o", color="steelblue", linewidth=2)
    axes[0].set_xlabel("Sample Size")
    axes[0].set_ylabel(f"Jaccard overlap with n={sizes[-1]} top-{args.top_k}")
    axes[0].set_title("Top-K Set Stability (Jaccard)")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(1.0, linestyle="--", color="gray", alpha=0.5)

    # Spearman
    axes[1].plot(sizes, spearman, marker="s", color="darkorange", linewidth=2)
    axes[1].set_xlabel("Sample Size")
    axes[1].set_ylabel(f"Spearman ρ with n={sizes[-1]} top-{args.top_k} ranking")
    axes[1].set_title("Top-K Rank Stability (Spearman ρ)")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(1.0, linestyle="--", color="gray", alpha=0.5)

    # Average KL of top tokens vs sample size
    mean_top_kl = [tdf["avg_kl"].mean() for tdf in topk_dfs]
    axes[2].plot(sizes, mean_top_kl, marker="^", color="mediumseagreen", linewidth=2)
    axes[2].set_xlabel("Sample Size")
    axes[2].set_ylabel(f"Mean avg-KL across top-{args.top_k} tokens")
    axes[2].set_title("Mean KL of Top-K Tokens vs Sample Size")

    for ax in axes:
        ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# COMPARE — two models side by side
# ---------------------------------------------------------------------------
elif mode == "compare":
    # Sort so onpolicy comes first if present
    datasets = sorted(datasets, key=lambda x: x[1].get("student_key", "z"))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=False)
    fig.suptitle(
        f"Rock Token Comparison  —  n={sample_sizes[0]}"
        if len(unique_sizes) == 1 else "Rock Token Comparison",
        fontsize=13,
    )

    for ax, (_, data, df) in zip(axes, datasets):
        topk = top_k_tokens(data, df, args.top_k, args.min_freq)
        key  = data.get("student_key", "?")
        n    = data.get("samples_processed", "?")

        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(topk)))[::-1]
        ax.barh(
            range(len(topk)), topk["avg_kl"], color=colors,
            xerr=topk["kl_std"], error_kw=dict(ecolor="gray", capsize=3, linewidth=0.8),
        )
        ax.set_yticks(range(len(topk)))
        ax.set_yticklabels(
            [f"{s}  (f={freq})"
             for s, freq in zip(topk["token_str"], topk["freq"])],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Average KL Divergence  (error bars = ±1 std dev)")
        ax.set_title(f"student={key}  n={n}\nTop-{args.top_k} Rock Tokens")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Mark tokens that appear in both top-K lists
    ids_left  = set(top_k_tokens(datasets[0][1], datasets[0][2], args.top_k, args.min_freq)["token_id"])
    ids_right = set(top_k_tokens(datasets[1][1], datasets[1][2], args.top_k, args.min_freq)["token_id"])
    shared = ids_left & ids_right
    if shared:
        shared_strs = ", ".join(decode(t) for t in sorted(shared))
        fig.text(0.5, 0.01, f"Shared tokens: {shared_strs}",
                 ha="center", fontsize=9, color="darkred")

    plt.tight_layout(rect=[0, 0.04, 1, 1])

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
if args.output:
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")
else:
    plt.tight_layout()
    plt.show()

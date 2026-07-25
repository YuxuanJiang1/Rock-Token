"""
Compute per-token reverse-KL gradient direction in logit space (Proxy A).

For each generation position we have student logits z^s and teacher logits z^t.
The gradient of reverse KL  KL(p^s || p^t)  w.r.t. student logits is:

    g^logit_u = p^s_u * (A_u - <A>_{p^s})         where A_u = log p^s_u - log p^t_u + 1

We accumulate this gradient per token TYPE (the token that was actually generated
at that position) so we can later compute average per-token gradient directions
and their cosine similarities to the global / frequency-balanced gradient.

We reuse the generated sequences from the existing occurrences file, so this
script only does forward passes (no autoregressive generation), making it
roughly 2-3x faster than the original rock_server.py run.

Usage
-----
  python compute_logit_gradients.py \
      --student onpolicy \
      --samples 500 \
      --hardware single_96gb \
      --occurrences-file rock_token_occurrences_onpolicy_n500_unrestricted.pt
"""

import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

STUDENT_MODELS = {
    "onpolicy":  "RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k",
    "offpolicy": "RockToken/qwen3_30b_a3b_to_4b_offpolicy_20k",
}
TEACHER_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LOGIT_CHUNK_SIZE = 128  # positions per chunk during gradient computation

parser = argparse.ArgumentParser()
parser.add_argument("--student",  choices=list(STUDENT_MODELS), default="onpolicy")
parser.add_argument("--samples",  type=int, default=500)
parser.add_argument("--hardware", choices=["single_96gb", "dual_40gb", "quad_l40s"], default="single_96gb")
parser.add_argument("--occurrences-file", required=True,
                    help="Existing rock_token_occurrences_*.pt file to reuse generated tokens from")
parser.add_argument("--output-file", default=None)
args = parser.parse_args()

STUDENT_ID  = STUDENT_MODELS[args.student]
SAMPLE_SIZE = args.samples
suffix      = "_unrestricted" if "_unrestricted" in args.occurrences_file else ""
OUTPUT_FILE = args.output_file or f"logit_gradients_{args.student}_n{SAMPLE_SIZE}{suffix}.pt"

# ---------------------------------------------------------------------------
# Load existing occurrences → reconstruct generated sequences per sample
# ---------------------------------------------------------------------------
print(f"Loading existing occurrences from {args.occurrences_file} ...")
existing = torch.load(args.occurrences_file, map_location="cpu", weights_only=False)
df = pd.DataFrame(existing["occurrences"])
print(f"  {len(df):,} occurrence records across {df['sample_idx'].nunique()} samples")

# Generated sequence per sample (sorted by abs_position)
gen_per_sample = {
    int(sid): torch.tensor(g.sort_values("abs_position")["token_id"].tolist(), dtype=torch.long)
    for sid, g in df.groupby("sample_idx")
}

# Pre-allocate per-token gradient storage based on tokens we've seen
seen_tokens = sorted(df["token_id"].unique().tolist())
n_seen      = len(seen_tokens)
tid_to_row  = {int(tid): i for i, tid in enumerate(seen_tokens)}
print(f"  {n_seen:,} unique tokens seen — pre-allocating gradient storage")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
print(f"Loading tokenizer {STUDENT_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
V = len(tokenizer)
print(f"  vocab size: {V}")

print("Loading student model (4B bf16)...")
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID, device_map="cuda:0", torch_dtype=torch.bfloat16,
)

print("Loading teacher model (30B bf16)...")
if args.hardware == "single_96gb":
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, device_map="cuda:0", torch_dtype=torch.bfloat16,
    )
else:
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, device_map="auto", torch_dtype=torch.bfloat16,
    )

student_device = next(student_model.parameters()).device
teacher_device = next(teacher_model.parameters()).device

# ---------------------------------------------------------------------------
# Storage  (CPU; ~ n_seen * V * 4 bytes ≈ 2-3 GB)
# ---------------------------------------------------------------------------
gradient_sum   = torch.zeros(n_seen, V, dtype=torch.float32)  # per-token cumulative
gradient_count = torch.zeros(n_seen, dtype=torch.long)
G_global       = torch.zeros(V, dtype=torch.float32)          # corpus-wide cumulative

# ---------------------------------------------------------------------------
# Re-load MATH-500 dataset with same seed → reconstruct prompts
# ---------------------------------------------------------------------------
print(f"Loading MATH-500 (seed=42, taking {SAMPLE_SIZE}) ...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
sampled_dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))


def reverse_kl_logit_grad(s_logits, t_logits):
    """g_u = p^s_u * (A_u - <A>_{p^s}),  A = log p^s - log p^t + 1.
    Returns float32 tensor of same shape as s_logits."""
    s_log = F.log_softmax(s_logits.float(), dim=-1)
    t_log = F.log_softmax(t_logits.float(), dim=-1)
    s_p   = s_log.exp()
    A     = s_log - t_log + 1.0
    EA    = (s_p * A).sum(dim=-1, keepdim=True)
    return s_p * (A - EA)


# ---------------------------------------------------------------------------
# Main loop — one parallel forward pass per model per sample
# ---------------------------------------------------------------------------
student_model.eval()
teacher_model.eval()

for i, item in enumerate(tqdm(sampled_dataset, desc="Forward passes")):
    if i not in gen_per_sample:
        continue
    gen_tokens = gen_per_sample[i]                     # [num_gen]
    if len(gen_tokens) == 0:
        continue

    problem_text = item["problem"]
    prompt = (
        f"<|im_start|>user\n{problem_text}\n"
        "Think step-by-step and enclose your reasoning inside <think> and </think> tags."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(student_device)
    prompt_len    = prompt_inputs.input_ids.shape[1]

    # full_sequence: [prompt_len + num_gen]
    full_sequence = torch.cat(
        [prompt_inputs.input_ids[0], gen_tokens.to(student_device)],
        dim=0,
    )
    inp_student = full_sequence.unsqueeze(0)
    inp_teacher = full_sequence.unsqueeze(0).to(teacher_device)

    with torch.no_grad():
        # --- Student forward ---
        student_out    = student_model(inp_student)
        student_logits = student_out.logits[0, prompt_len - 1 : -1, :].clone()
        del student_out

        # --- Teacher forward ---
        teacher_out    = teacher_model(inp_teacher)
        teacher_logits = teacher_out.logits[0, prompt_len - 1 : -1, :].clone()
        del teacher_out, inp_teacher

        teacher_logits = teacher_logits.to(student_logits.device)

        num_gen = student_logits.shape[0]
        for s in range(0, num_gen, LOGIT_CHUNK_SIZE):
            e = min(s + LOGIT_CHUNK_SIZE, num_gen)
            grad_chunk = reverse_kl_logit_grad(student_logits[s:e], teacher_logits[s:e])  # [c, V]
            grad_chunk_cpu = grad_chunk.cpu()
            del grad_chunk

            # Map this chunk's tokens → row indices into gradient_sum
            chunk_tids = gen_tokens[s:e].tolist()
            rows = torch.tensor(
                [tid_to_row.get(int(t), -1) for t in chunk_tids],
                dtype=torch.long,
            )
            valid_mask = rows >= 0
            if valid_mask.any():
                valid_rows  = rows[valid_mask]
                valid_grads = grad_chunk_cpu[valid_mask]
                gradient_sum.index_add_(0, valid_rows, valid_grads)
                gradient_count.index_add_(
                    0, valid_rows,
                    torch.ones(int(valid_mask.sum()), dtype=torch.long),
                )

            G_global += grad_chunk_cpu.sum(dim=0)
            del grad_chunk_cpu

        del student_logits, teacher_logits

    if i % 10 == 0:
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = {
    "student_id":        STUDENT_ID,
    "student_key":       args.student,
    "teacher_id":        TEACHER_ID,
    "samples_processed": SAMPLE_SIZE,
    "vocab_size":        V,
    "seen_token_ids":    torch.tensor(seen_tokens, dtype=torch.long),
    "gradient_sum":      gradient_sum,    # [n_seen, V]   per-type cumulative
    "gradient_count":    gradient_count,  # [n_seen]      occurrences contributing
    "G_global":          G_global,        # [V]           corpus-wide cumulative
    "occurrences_source": args.occurrences_file,
    "kl_direction":      "reverse_kl_p_s_minus_p_t",
}
torch.save(out, OUTPUT_FILE)
print(f"\nSaved {OUTPUT_FILE}")
print(f"  gradient_sum:   {tuple(gradient_sum.shape)}  (~{gradient_sum.element_size() * gradient_sum.numel() / 1e9:.2f} GB)")
print(f"  gradient_count totals: {int(gradient_count.sum()):,} positions accumulated")
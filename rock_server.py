import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

STUDENT_MODELS = {
    "onpolicy":  "RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k",
    "offpolicy": "RockToken/qwen3_30b_a3b_to_4b_offpolicy_math_first20k",
}
TEACHER_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_NEW_TOKENS = 256

parser = argparse.ArgumentParser(description="Collect rock-token KL statistics")
parser.add_argument(
    "--student",
    choices=list(STUDENT_MODELS.keys()),
    default="onpolicy",
    help="Which student model to evaluate (default: onpolicy)",
)
parser.add_argument(
    "--samples",
    type=int,
    default=100,
    help="Number of MATH-500 problems to sample (default: 100)",
)
parser.add_argument(
    "--hardware",
    choices=["single_96gb", "dual_40gb"],
    default="dual_40gb",
    help="GPU memory layout (default: dual_40gb)",
)
args = parser.parse_args()

STUDENT_ID = STUDENT_MODELS[args.student]
SAMPLE_SIZE = args.samples
HARDWARE_CONFIG = args.hardware
OUTPUT_FILE = f"rock_token_occurrences_{args.student}_n{SAMPLE_SIZE}.pt"

# --- 1. Load Tokenizer and Models ---
print(f"Student: {STUDENT_ID} | Samples: {SAMPLE_SIZE} | Hardware: {HARDWARE_CONFIG}")
print(f"Output : {OUTPUT_FILE}")
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)

print("Loading student model (4B bf16)...")
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

print("Loading teacher model (30B bf16)...")
if HARDWARE_CONFIG == "single_96gb":
    # Both fit on one 96GB GPU
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
elif HARDWARE_CONFIG == "dual_40gb":
    # Student took ~8GB on GPU 0; teacher (~60GB) auto-fills remaining space on GPU 0
    # and spills onto GPU 1. Roughly: ~32GB on GPU 0, ~28GB on GPU 1.
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
else:
    raise ValueError(f"Unknown HARDWARE_CONFIG: {HARDWARE_CONFIG!r}")

student_device = next(student_model.parameters()).device
teacher_device = next(teacher_model.parameters()).device
print(f"Student on: {student_device} | Teacher first layer on: {teacher_device}")

# --- 2. Load Dataset ---
print(f"Sampling {SAMPLE_SIZE} problems from MATH-500...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
sampled_dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

# --- 3. Global Trackers ---
vocab_size = len(tokenizer)
token_frequencies = torch.zeros(vocab_size, dtype=torch.long)
token_cumulative_kl = torch.zeros(vocab_size, dtype=torch.float64)

# Precompute which token IDs decode to strings containing a newline.
# These mark line boundaries in math step-by-step output.
newline_token_ids = frozenset(
    tid for tid in range(vocab_size) if "\n" in tokenizer.decode([tid])
)

# Per-occurrence records for positional analysis.
# Each entry covers one generated token across all samples.
occurrence_records = []

# --- 4. Processing Loop ---
student_model.eval()
teacher_model.eval()

for i, item in enumerate(tqdm(sampled_dataset, desc="Processing Datasets")):
    problem_text = item["problem"]

    prompt = (
        f"<|im_start|>user\n{problem_text}\n"
        "Think step-by-step and enclose your reasoning inside <think> and </think> tags."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(student_device)
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        # --- A. Student Generation ---
        outputs = student_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            output_logits=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        full_sequence = outputs.sequences[0]
        generated_tokens = full_sequence[prompt_length:]

        if len(generated_tokens) == 0:
            continue

        # student_logits: [num_generated, vocab] on cuda:0
        student_logits = torch.cat(outputs.logits, dim=0)

        # --- B. Teacher Evaluation ---
        # Teacher's embedding is on teacher_device; it handles internal sharding across GPUs
        teacher_inputs = full_sequence.unsqueeze(0).to(teacher_device)
        teacher_outputs = teacher_model(teacher_inputs)

        # Align logits: [1, seq_len, vocab] -> [num_generated, vocab]
        teacher_logits = teacher_outputs.logits[0, prompt_length - 1 : -1, :]

        # Teacher's lm_head may be on a different GPU — move to student's device for KL
        teacher_logits = teacher_logits.to(student_logits.device)

        # --- C. Reverse KL Divergence ---
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        kl_div_matrix = student_probs * (student_log_probs - teacher_log_probs)
        token_kl_divergence = kl_div_matrix.sum(dim=-1)  # [num_generated]

        # --- D. Accumulate Statistics on CPU ---
        gen_tokens_cpu = generated_tokens.cpu()
        kl_div_cpu = token_kl_divergence.cpu()

        token_frequencies.scatter_add_(0, gen_tokens_cpu, torch.ones_like(gen_tokens_cpu, dtype=torch.long))
        token_cumulative_kl.scatter_add_(0, gen_tokens_cpu, kl_div_cpu.to(torch.float64))

        # --- E. Per-Occurrence Positional Records ---
        seq_len = len(gen_tokens_cpu)
        line_idx = 0
        pos_in_line = 0
        token_ids_list = gen_tokens_cpu.tolist()
        kl_list = kl_div_cpu.tolist()

        for abs_pos, (tid, kl_val) in enumerate(zip(token_ids_list, kl_list)):
            is_newline = tid in newline_token_ids
            occurrence_records.append({
                "sample_idx":     i,
                "token_id":       tid,
                "kl":             kl_val,
                "abs_position":   abs_pos,
                "rel_position":   abs_pos / max(seq_len - 1, 1),
                "line_index":     line_idx,
                "position_in_line": pos_in_line,
                "is_line_start":  pos_in_line == 0,
                "is_newline":     is_newline,
                "seq_len":        seq_len,
            })
            if is_newline:
                line_idx += 1
                pos_in_line = 0
            else:
                pos_in_line += 1

    if i % 10 == 0:
        torch.cuda.empty_cache()

# --- 5. Calculate Final Averages and Save ---
print("Computing final averages...")
valid_mask = token_frequencies > 0
average_kl = torch.zeros_like(token_cumulative_kl)
average_kl[valid_mask] = token_cumulative_kl[valid_mask] / token_frequencies[valid_mask].to(torch.float64)

rock_token_data = {
    # Run metadata
    "student_id":       STUDENT_ID,
    "student_key":      args.student,
    "teacher_id":       TEACHER_ID,
    "samples_processed": SAMPLE_SIZE,
    "vocab_size":       vocab_size,
    # Aggregated vocab-level stats
    "token_ids":        torch.arange(vocab_size),
    "frequencies":      token_frequencies,
    "cumulative_kl":    token_cumulative_kl,
    "average_kl":       average_kl,
    # Per-occurrence positional records
    # Fields: sample_idx, token_id, kl, abs_position, rel_position,
    #         line_index, position_in_line, is_line_start, is_newline, seq_len
    "occurrences":      occurrence_records,
}

torch.save(rock_token_data, OUTPUT_FILE)
print(f"Saved {len(occurrence_records):,} occurrence records to {OUTPUT_FILE}")

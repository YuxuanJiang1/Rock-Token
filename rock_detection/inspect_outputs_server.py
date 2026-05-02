import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
STUDENT_ID = "RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k"
TEACHER_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SAMPLE_SIZE = 10
MAX_NEW_TOKENS = 256
OUTPUT_FILE = "output_inspection.json"

# Hardware config — choose one:
#   "single_96gb" : one 96GB GPU; student (~8GB) + teacher (~60GB) = ~68GB, fits comfortably
#   "dual_40gb"   : two 40GB GPUs (80GB total); student pinned to cuda:0,
#                   teacher (30B bf16 ~60GB) auto-distributed across both GPUs
HARDWARE_CONFIG = "dual_40gb"

# --- Load Tokenizer and Models ---
print(f"Hardware config: {HARDWARE_CONFIG}")
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
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
elif HARDWARE_CONFIG == "dual_40gb":
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

# --- Load Dataset ---
print(f"Sampling {SAMPLE_SIZE} problems from MATH-500...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
sampled_dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

# --- Processing Loop ---
student_model.eval()
teacher_model.eval()
results = []

for item in tqdm(sampled_dataset, desc="Processing"):
    problem_text = item["problem"]

    prompt = (
        f"<|im_start|>user\n{problem_text}\n"
        "Think step-by-step and enclose your reasoning inside <think> and </think> tags."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(student_device)
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        # Student generation
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

        student_logits = torch.cat(outputs.logits, dim=0)  # [num_generated, vocab]

        # Teacher forward pass on the full sequence
        teacher_inputs = full_sequence.unsqueeze(0).to(teacher_device)
        teacher_outputs = teacher_model(teacher_inputs)
        teacher_logits = teacher_outputs.logits[0, prompt_length - 1 : -1, :]  # [num_generated, vocab]

        # Teacher's lm_head may be on a different GPU — move to student's device for KL
        teacher_logits = teacher_logits.to(student_logits.device)

        # Per-token KL divergence (reverse KL: student || teacher)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        token_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)  # [num_generated]

    # Build per-token records
    token_records = []
    for token_id, kl in zip(generated_tokens.tolist(), token_kl.tolist()):
        token_records.append({
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "kl": round(kl, 6),
        })

    results.append({
        "problem": problem_text,
        "generated_text": tokenizer.decode(generated_tokens, skip_special_tokens=False),
        "total_kl": round(token_kl.sum().item(), 6),
        "mean_kl": round(token_kl.mean().item(), 6),
        "tokens": token_records,
    })

# --- Save ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} samples to {OUTPUT_FILE}")

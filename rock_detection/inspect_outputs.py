import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
STUDENT_ID = "RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k"
TEACHER_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SAMPLE_SIZE = 10
MAX_NEW_TOKENS = 256
OUTPUT_FILE = "/workspace/output_inspection.json"

# --- Load Tokenizer and Models ---
print("Loading Tokenizer and Models...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID, cache_dir="/workspace/huggingface_cache")

student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID,
    device_map="auto",
    dtype=torch.bfloat16,
    cache_dir="/workspace/huggingface_cache"
)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_ID,
    device_map="auto",
    quantization_config=quant_config,
    cache_dir="/workspace/huggingface_cache"
)

# --- Load Dataset ---
print(f"Sampling {SAMPLE_SIZE} problems from MATH-500...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir="/workspace/huggingface_cache")
sampled_dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

# --- Processing Loop ---
student_model.eval()
teacher_model.eval()
results = []

for item in tqdm(sampled_dataset, desc="Processing"):
    problem_text = item["problem"]

    prompt = f"<|im_start|>user\n{problem_text}\nThink step-by-step and enclose your reasoning inside <think> and </think> tags.<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(student_model.device)
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        # Student generation
        outputs = student_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            output_logits=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        full_sequence = outputs.sequences[0]
        generated_tokens = full_sequence[prompt_length:]

        if len(generated_tokens) == 0:
            continue

        student_logits = torch.cat(outputs.logits, dim=0)  # [num_generated, vocab]

        # Teacher forward pass on the full sequence
        teacher_inputs = full_sequence.unsqueeze(0).to(teacher_model.device)
        teacher_outputs = teacher_model(teacher_inputs)
        teacher_logits = teacher_outputs.logits[0, prompt_length - 1 : -1, :]  # [num_generated, vocab]

        # Per-token KL divergence (reverse KL: student || teacher)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits.to(student_logits.device), dim=-1)

        token_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)  # [num_generated]

    # Build per-token records
    token_records = []
    for token_id, kl in zip(generated_tokens.tolist(), token_kl.tolist()):
        token_records.append({
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "kl": round(kl, 6)
        })

    results.append({
        "problem": problem_text,
        "generated_text": tokenizer.decode(generated_tokens, skip_special_tokens=False),
        "total_kl": round(token_kl.sum().item(), 6),
        "mean_kl": round(token_kl.mean().item(), 6),
        "tokens": token_records
    })

# --- Save ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} samples to {OUTPUT_FILE}")

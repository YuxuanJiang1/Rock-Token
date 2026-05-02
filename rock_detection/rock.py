import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="organization/model-name",
#     local_dir="/workspace/my_model_folder",
#     local_dir_use_symlinks=False # Set to False so it doesn't create symlinks to ~/.cache
# )

# --- Configuration ---
STUDENT_ID = "RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k"
TEACHER_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SAMPLE_SIZE = 100
MAX_NEW_TOKENS = 256
OUTPUT_FILE = "/workspace/aggregated_rock_token_stats.pt"

# --- 1. Load Tokenizer and Models ---
print("Loading Tokenizer and Models...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID, cache_dir="/workspace/huggingface_cache")

# Student: Load in bfloat16
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir="/workspace/huggingface_cache"
)

# Teacher: Load in 4-bit to save VRAM
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

# --- 2. Load Dataset ---
print(f"Sampling {SAMPLE_SIZE} problems from MATH-500...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir="/workspace/huggingface_cache")
sampled_dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

# --- 3. Global Trackers for Phase A ---
vocab_size = len(tokenizer)  # Use len() to include special tokens beyond the base vocab
# Trackers are kept on CPU to save GPU memory
token_frequencies = torch.zeros(vocab_size, dtype=torch.long)
token_cumulative_kl = torch.zeros(vocab_size, dtype=torch.float64)

# --- 4. Processing Loop ---
student_model.eval()
teacher_model.eval()

for i, item in enumerate(tqdm(sampled_dataset, desc="Processing Datasets")):
    problem_text = item["problem"]
    
    # Format prompt for Qwen chat template
    prompt = f"<|im_start|>user\n{problem_text}\nThink step-by-step and enclose your reasoning inside <think> and </think> tags.<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(student_model.device)
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        # --- A. Student Generation ---
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
        
        # Skip if no tokens were generated
        if len(generated_tokens) == 0:
            continue
            
        student_logits = torch.cat(outputs.logits, dim=0) 

        # --- B. Teacher Evaluation ---
        teacher_inputs = full_sequence.unsqueeze(0).to(teacher_model.device)
        teacher_outputs = teacher_model(teacher_inputs)
        
        # Align logits: [1, seq_len, vocab] -> [num_generated, vocab]
        teacher_full_logits = teacher_outputs.logits
        teacher_logits = teacher_full_logits[0, prompt_length - 1 : -1, :]

        # --- C. Calculate Reverse KL Divergence ---
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        
        kl_div_matrix = student_probs * (student_log_probs - teacher_log_probs)
        token_kl_divergence = kl_div_matrix.sum(dim=-1) # Shape: [num_generated]

        # --- D. Accumulate Global Statistics ---
        # Move tensors to CPU for accumulation
        gen_tokens_cpu = generated_tokens.cpu()
        kl_div_cpu = token_kl_divergence.cpu()

        # Update frequencies and cumulative KL sums efficiently using scatter_add_
        token_frequencies.scatter_add_(0, gen_tokens_cpu, torch.ones_like(gen_tokens_cpu, dtype=torch.long))
        token_cumulative_kl.scatter_add_(0, gen_tokens_cpu, kl_div_cpu.to(torch.float64))

    # Clear VRAM cache periodically to prevent fragmentation
    if i % 10 == 0:
        torch.cuda.empty_cache()

# --- 5. Calculate Final Averages and Save ---
print("Computing final averages...")
# Avoid division by zero for tokens that were never generated
valid_mask = token_frequencies > 0
average_kl = torch.zeros_like(token_cumulative_kl)
average_kl[valid_mask] = token_cumulative_kl[valid_mask] / token_frequencies[valid_mask].to(torch.float64)

# Create a master dictionary to save
rock_token_data = {
    "token_ids": torch.arange(vocab_size),
    "frequencies": token_frequencies,
    "cumulative_kl": token_cumulative_kl,
    "average_kl": average_kl,
    "vocab_size": vocab_size,
    "samples_processed": SAMPLE_SIZE
}

torch.save(rock_token_data, OUTPUT_FILE)
print(f"Successfully saved aggregated data to {OUTPUT_FILE}")
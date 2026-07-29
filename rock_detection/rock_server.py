import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ── Built-in Qwen3 presets (kept for backward compatibility) ──────────────────
STUDENT_MODELS = {
    "onpolicy":  "RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k",
    "offpolicy": "RockToken/qwen3_30b_a3b_to_4b_offpolicy_math_first20k",
}
DEFAULT_TEACHER_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_MAX_NEW_TOKENS = 256
UNRESTRICTED_MAX_NEW_TOKENS = 4096
KL_CHUNK_SIZE = 256

# ── Prompt templates by domain ────────────────────────────────────────────────
# "qwen_math"  : original paper template (thinking tags, Qwen3 im_start format)
# "chat"       : plain chat via tokenizer.apply_chat_template (works for any model)
# "code"       : coding prompt with chat template
DOMAIN_SYSTEM_PROMPTS = {
    "math":  "You are a helpful math assistant. Solve the following problem step by step.",
    "code":  "You are an expert programmer. Solve the following coding problem.",
    "general": "You are a helpful assistant.",
}

parser = argparse.ArgumentParser(
    description="Collect rock-token KL statistics — supports arbitrary model pairs and domains",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# ── Model selection ───────────────────────────────────────────────────────────
model_grp = parser.add_argument_group("Model selection")
model_grp.add_argument(
    "--student",
    default=None,
    help="Preset student key (onpolicy / offpolicy). Ignored when --student-id is given.",
)
model_grp.add_argument(
    "--student-id",
    default=None,
    metavar="HF_REPO_OR_PATH",
    help="HuggingFace repo ID or local path for the student model. "
         "Overrides --student preset.",
)
model_grp.add_argument(
    "--teacher-id",
    default=DEFAULT_TEACHER_ID,
    metavar="HF_REPO_OR_PATH",
    help="HuggingFace repo ID or local path for the teacher model.",
)
model_grp.add_argument(
    "--cache-dir",
    default="/workspace/hf_cache",
    metavar="DIR",
    help="HuggingFace cache directory.",
)

# ── Dataset / domain ──────────────────────────────────────────────────────────
data_grp = parser.add_argument_group("Dataset / domain")
data_grp.add_argument(
    "--domain",
    choices=["math", "code", "general"],
    default="math",
    help="Training domain. Controls the system prompt and built-in dataset default.",
)
data_grp.add_argument(
    "--dataset",
    default=None,
    metavar="HF_DATASET_ID",
    help="HuggingFace dataset ID. Defaults: math→HuggingFaceH4/MATH-500, "
         "code→livecodebench/code_generation_lite, general→tatsu-lab/alpaca.",
)
data_grp.add_argument(
    "--dataset-split",
    default="test",
    metavar="SPLIT",
    help="Dataset split to use.",
)
data_grp.add_argument(
    "--problem-field",
    default=None,
    metavar="FIELD",
    help="Name of the dataset column containing the problem text. "
         "Auto-detected for known datasets; required for custom ones.",
)
data_grp.add_argument(
    "--samples",
    type=int,
    default=100,
    help="Number of problems to sample.",
)
data_grp.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Shuffle seed for dataset sampling.",
)

# ── Prompt format ─────────────────────────────────────────────────────────────
prompt_grp = parser.add_argument_group("Prompt format")
prompt_grp.add_argument(
    "--prompt-style",
    choices=["qwen_math", "chat"],
    default=None,
    help="Prompt construction style. "
         "'qwen_math' uses the original paper template (thinking tags, im_start format). "
         "'chat' uses tokenizer.apply_chat_template (works for any HF model). "
         "Default: 'qwen_math' when domain=math and student is a Qwen model, "
         "otherwise 'chat'.",
)

# ── Hardware / generation ─────────────────────────────────────────────────────
hw_grp = parser.add_argument_group("Hardware / generation")
hw_grp.add_argument(
    "--hardware",
    choices=["single_96gb", "dual_40gb", "dual_80gb", "auto"],
    default="dual_40gb",
    help="GPU memory layout. 'auto' lets device_map=auto handle placement for both models.",
)
hw_grp.add_argument(
    "--unrestricted",
    action="store_true",
    help="Let the model stop at EOS (up to UNRESTRICTED_MAX_NEW_TOKENS). "
         "Output filename gets an '_unrestricted' suffix.",
)
hw_grp.add_argument(
    "--output-tag",
    default=None,
    metavar="TAG",
    help="Short label embedded in the output filename (e.g. 'llama_math'). "
         "Defaults to the student preset key or a slug derived from --student-id.",
)

args = parser.parse_args()

# ── Resolve student model ID ───────────────────────────────────────────────────
if args.student_id:
    STUDENT_ID = args.student_id
    _student_key = args.output_tag or args.student_id.split("/")[-1].replace("-", "_")
elif args.student and args.student in STUDENT_MODELS:
    STUDENT_ID = STUDENT_MODELS[args.student]
    _student_key = args.student
else:
    parser.error("Provide either --student-id <repo> or --student <preset>.")

TEACHER_ID   = args.teacher_id
SAMPLE_SIZE  = args.samples
HARDWARE_CONFIG = args.hardware
output_tag   = args.output_tag or _student_key

# ── Resolve dataset ────────────────────────────────────────────────────────────
DATASET_DEFAULTS = {
    "math":    ("HuggingFaceH4/MATH-500",              "test",  "problem"),
    "code":    ("livecodebench/code_generation_lite",  "test",  "question_content"),
    "general": ("tatsu-lab/alpaca",                    "train", "instruction"),
}
_ds_id, _ds_split, _ds_field = DATASET_DEFAULTS[args.domain]
DATASET_ID    = args.dataset       or _ds_id
DATASET_SPLIT = args.dataset_split or _ds_split
PROBLEM_FIELD = args.problem_field or _ds_field

# ── Resolve prompt style ───────────────────────────────────────────────────────
def _is_qwen(model_id: str) -> bool:
    return "qwen" in model_id.lower()

if args.prompt_style:
    PROMPT_STYLE = args.prompt_style
elif args.domain == "math" and _is_qwen(STUDENT_ID):
    PROMPT_STYLE = "qwen_math"
else:
    PROMPT_STYLE = "chat"

# ── Output filename ────────────────────────────────────────────────────────────
suffix = "_unrestricted" if args.unrestricted else ""
OUTPUT_FILE = f"rock_token_occurrences_{output_tag}_{args.domain}_n{SAMPLE_SIZE}{suffix}.pt"

# ── Max tokens ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = UNRESTRICTED_MAX_NEW_TOKENS if args.unrestricted else DEFAULT_MAX_NEW_TOKENS

# ── Summary ────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"  Student  : {STUDENT_ID}")
print(f"  Teacher  : {TEACHER_ID}")
print(f"  Domain   : {args.domain}  |  Prompt style: {PROMPT_STYLE}")
print(f"  Dataset  : {DATASET_ID} [{DATASET_SPLIT}]  field='{PROBLEM_FIELD}'")
print(f"  Samples  : {SAMPLE_SIZE}  |  max_new_tokens={MAX_NEW_TOKENS}")
print(f"  Hardware : {HARDWARE_CONFIG}")
print(f"  Output   : {OUTPUT_FILE}")
print("=" * 70)

# --- 1. Load Tokenizer and Models ---
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID, cache_dir=args.cache_dir)

print("Loading student model (bf16)...")
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID,
    device_map="cuda:0" if HARDWARE_CONFIG not in ("auto",) else "auto",
    torch_dtype=torch.bfloat16,
    cache_dir=args.cache_dir,
)

print("Loading teacher model (bf16)...")
if HARDWARE_CONFIG == "single_96gb":
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )
elif HARDWARE_CONFIG in ("dual_40gb", "dual_80gb", "auto"):
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )
else:
    raise ValueError(f"Unknown HARDWARE_CONFIG: {HARDWARE_CONFIG!r}")

student_device = next(student_model.parameters()).device
teacher_device = next(teacher_model.parameters()).device
print(f"Student on: {student_device} | Teacher first layer on: {teacher_device}")

# --- 2. Load Dataset ---
print(f"Loading dataset {DATASET_ID} [{DATASET_SPLIT}] ...")
dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, cache_dir=args.cache_dir,
                       trust_remote_code=True)
sampled_dataset = dataset.shuffle(seed=args.seed).select(range(min(SAMPLE_SIZE, len(dataset))))

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

# --- 4a. Prompt builder -------------------------------------------------------
_system_prompt = DOMAIN_SYSTEM_PROMPTS[args.domain]

def build_prompt(problem_text: str) -> str:
    if PROMPT_STYLE == "qwen_math":
        # Original paper format: Qwen3 im_start with thinking tags
        return (
            f"<|im_start|>user\n{problem_text}\n"
            "Think step-by-step and enclose your reasoning inside <think> and </think> tags."
            "<|im_end|>\n<|im_start|>assistant\n"
        )
    else:
        # Generic chat template — works for Llama-3, Mistral, OLMo, etc.
        messages = [
            {"role": "system", "content": _system_prompt},
            {"role": "user",   "content": problem_text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

for i, item in enumerate(tqdm(sampled_dataset, desc="Processing Datasets")):
    problem_text = item[PROBLEM_FIELD]

    prompt = build_prompt(problem_text)
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
        del outputs

        # --- B. Teacher Evaluation ---
        # Teacher's embedding is on teacher_device; it handles internal sharding across GPUs
        teacher_inputs = full_sequence.unsqueeze(0).to(teacher_device)
        teacher_outputs = teacher_model(teacher_inputs)

        # Slice + clone so we can release teacher_outputs (and its activations) immediately
        teacher_logits = teacher_outputs.logits[0, prompt_length - 1 : -1, :].clone()
        del teacher_outputs, teacher_inputs

        # Teacher's lm_head may be on a different GPU — move to student's device for KL
        teacher_logits = teacher_logits.to(student_logits.device)

        # --- C. Reverse KL Divergence (chunked over generated positions) ---
        num_gen = student_logits.shape[0]
        token_kl_divergence = torch.zeros(num_gen, device=student_logits.device, dtype=torch.float32)
        for s in range(0, num_gen, KL_CHUNK_SIZE):
            e = min(s + KL_CHUNK_SIZE, num_gen)
            s_log = F.log_softmax(student_logits[s:e], dim=-1)
            t_log = F.log_softmax(teacher_logits[s:e], dim=-1)
            token_kl_divergence[s:e] = (s_log.exp() * (s_log - t_log)).sum(dim=-1)
            del s_log, t_log

        del student_logits, teacher_logits

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

    # Long unrestricted outputs allocate large transient tensors; clean up every step
    # to keep the cache from fragmenting.
    if args.unrestricted or i % 10 == 0:
        torch.cuda.empty_cache()

# --- 5. Calculate Final Averages and Save ---
print("Computing final averages...")
valid_mask = token_frequencies > 0
average_kl = torch.zeros_like(token_cumulative_kl)
average_kl[valid_mask] = token_cumulative_kl[valid_mask] / token_frequencies[valid_mask].to(torch.float64)

rock_token_data = {
    # Run metadata
    "student_id":       STUDENT_ID,
    "student_key":      output_tag,
    "teacher_id":       TEACHER_ID,
    "domain":           args.domain,
    "prompt_style":     PROMPT_STYLE,
    "dataset_id":       DATASET_ID,
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

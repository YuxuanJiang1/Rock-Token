"""Test that batched generation produces identical results to single-sample generation.

Requires a GPU and a small model. Skipped if CUDA is unavailable.
Run with: uv run pytest tests/exp_2/test_batching.py -v -s
"""

import pytest
import torch

torch_cuda = pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Use a tiny model for fast testing — override with TEST_MODEL env var
import os

MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = 64  # Short for speed


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


PROMPTS = [
    "What is 2 + 3?",
    "Solve for x: 2x + 5 = 11",
    "Find the area of a circle with radius 7.",
]


def generate_single(model, tokenizer, prompt):
    """Generate one sample at a time (no padding), return (token_ids, log_probs)."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    prompt_length = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    full_ids = outputs.sequences[0].cpu()
    gen_len = len(outputs.scores)
    scores = torch.stack([s.squeeze(0) for s in outputs.scores], dim=0)
    log_probs = torch.log_softmax(scores.float(), dim=-1).cpu()

    return full_ids, prompt_length, gen_len, log_probs


def generate_batched(model, tokenizer, prompts):
    """Generate a batch at once with left-padding, extract per-sample results.

    Mirrors the logic in phases.py run_phase1.
    """
    tokenizer.padding_side = "left"
    batch_texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch_texts.append(text)

    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Compute position_ids for left-padded inputs (required for RoPE models)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    results = []
    for j in range(len(prompts)):
        n_left_pad = (attention_mask[j] == 0).sum().item()
        prompt_length = attention_mask[j].sum().item()

        full_ids = outputs.sequences[j, n_left_pad:].cpu()

        # EOS-based end detection (same as phases.py)
        response_tokens = full_ids[prompt_length:]
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            eos_positions = (response_tokens == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                gen_len = eos_positions[0].item() + 1
            else:
                gen_len = len(response_tokens)
        else:
            gen_len = len(response_tokens)

        full_ids = full_ids[: prompt_length + gen_len]

        sample_scores = torch.stack(
            [outputs.scores[t][j] for t in range(gen_len)], dim=0
        )
        log_probs = torch.log_softmax(sample_scores.float(), dim=-1).cpu()

        results.append((full_ids, prompt_length, gen_len, log_probs))

    return results


def test_batched_vs_single_token_ids(model_and_tokenizer):
    """Batched and single generation must produce identical token sequences."""
    model, tokenizer = model_and_tokenizer

    single_results = [generate_single(model, tokenizer, p) for p in PROMPTS]
    batched_results = generate_batched(model, tokenizer, PROMPTS)

    for idx, (single, batched) in enumerate(zip(single_results, batched_results)):
        s_ids, s_plen, s_glen, _ = single
        b_ids, b_plen, b_glen, _ = batched

        assert s_plen == b_plen, (
            f"Prompt {idx}: prompt_length differs: single={s_plen}, batched={b_plen}"
        )
        assert s_glen == b_glen, (
            f"Prompt {idx}: gen_len differs: single={s_glen}, batched={b_glen}"
        )
        assert torch.equal(s_ids, b_ids), (
            f"Prompt {idx}: token IDs differ between single and batched generation"
        )


def test_batched_vs_single_log_probs(model_and_tokenizer):
    """Batched and single generation must produce near-identical log-probs."""
    model, tokenizer = model_and_tokenizer

    single_results = [generate_single(model, tokenizer, p) for p in PROMPTS]
    batched_results = generate_batched(model, tokenizer, PROMPTS)

    for idx, (single, batched) in enumerate(zip(single_results, batched_results)):
        _, _, s_glen, s_lp = single
        _, _, b_glen, b_lp = batched

        assert s_glen == b_glen, f"Prompt {idx}: gen_len differs"

        # Log-probs should be very close (small floating point diffs from padding)
        max_diff = (s_lp - b_lp).abs().max().item()
        assert max_diff < 1e-3, (
            f"Prompt {idx}: log-prob max diff = {max_diff:.6f} (expected < 1e-3)"
        )

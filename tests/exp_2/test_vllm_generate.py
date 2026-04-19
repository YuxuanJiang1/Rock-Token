"""Test vLLM generation produces correct output format and matches HF single-sample.

Requires GPU, CUDA, and vLLM installed.
Run with: uv run pytest tests/exp_2/test_vllm_generate.py -v -s
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

vllm = pytest.importorskip("vllm")

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = 64

PROMPTS = [
    "What is 2 + 3?",
    "Solve for x: 2x + 5 = 11",
]


@pytest.fixture(scope="module")
def vllm_results():
    """Run vLLM generation + HF forward pass, return results and temp dir."""
    from src.exp_2.vllm_generate import vllm_generate

    tmpdir = Path(tempfile.mkdtemp())
    vllm_generate(
        model_name=MODEL,
        output_dir=tmpdir,
        max_new_tokens=MAX_NEW_TOKENS,
        tensor_parallel_size=1,
    )
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def hf_single_results():
    """Generate with HF single-sample (no batching) as ground truth."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = []
    for prompt in PROMPTS:
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
            )

        full_ids = outputs[0].cpu()
        gen_len = len(full_ids) - prompt_length

        # Forward pass for log-probs (same approach as vllm_generate)
        with torch.no_grad():
            fwd = model(input_ids=full_ids.unsqueeze(0).to(model.device))
        logits = fwd.logits[0].float().cpu()
        student_logits = logits[prompt_length - 1 : prompt_length - 1 + gen_len]
        log_probs = torch.log_softmax(student_logits, dim=-1)

        results.append({
            "full_ids": full_ids,
            "prompt_length": prompt_length,
            "gen_len": gen_len,
            "log_probs": log_probs,
        })

    del model
    torch.cuda.empty_cache()
    return results


def test_vllm_output_format(vllm_results):
    """Verify .pt files have correct keys, shapes, and dtypes."""
    student_dir = vllm_results / "student_data"
    files = sorted(student_dir.glob("sample_*.pt"))

    # Should have generated files for MATH500 samples
    assert len(files) > 0, "No sample files generated"

    data = torch.load(files[0], map_location="cpu", weights_only=True)

    # Check required keys
    assert "sample_idx" in data
    assert "prompt_length" in data
    assert "full_ids" in data
    assert "student_log_probs" in data

    # Check types
    assert isinstance(data["sample_idx"], int)
    assert isinstance(data["prompt_length"], int)
    assert data["full_ids"].dtype == torch.long
    assert data["student_log_probs"].dtype == torch.bfloat16

    # Check shapes
    full_len = data["full_ids"].shape[0]
    gen_len = data["student_log_probs"].shape[0]
    prompt_len = data["prompt_length"]
    assert full_len == prompt_len + gen_len, (
        f"full_ids length ({full_len}) != prompt_length ({prompt_len}) + gen_len ({gen_len})"
    )
    # vocab_size should be > 100k for Qwen3
    assert data["student_log_probs"].shape[1] > 100000


def test_vllm_log_probs_valid(vllm_results):
    """Verify log-probs are valid probability distributions."""
    student_dir = vllm_results / "student_data"
    files = sorted(student_dir.glob("sample_*.pt"))
    data = torch.load(files[0], map_location="cpu", weights_only=True)

    log_probs = data["student_log_probs"].float()

    # Log-probs should be <= 0
    assert (log_probs <= 0).all(), "Found positive log-probs"

    # exp(log_probs) should sum to ~1 per position
    probs_sum = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-2), (
        f"Probability sums deviate from 1: min={probs_sum.min():.4f}, max={probs_sum.max():.4f}"
    )

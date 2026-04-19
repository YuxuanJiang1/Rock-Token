"""Test vLLM generation produces correct output format and valid log-probs.

Requires GPU, CUDA, and vLLM installed.
Run with: uv run pytest tests/exp_2/test_vllm_generate.py -v -s
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

pytest.importorskip("vllm")

MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = 64

PROMPTS = [
    "What is 2 + 3?",
    "Solve for x: 2x + 5 = 11",
]


@pytest.fixture(scope="module")
def vllm_output_dir():
    """Run vLLM subprocess on 2 test prompts, return output dir."""
    tmpdir = Path(tempfile.mkdtemp())
    student_dir = tmpdir / "student_data"
    student_dir.mkdir()

    # Run vLLM in subprocess with our test prompts
    script = f"""
import json, torch
from pathlib import Path
from vllm import LLM, SamplingParams

prompts = {PROMPTS!r}
conversations = [[{{"role": "user", "content": p}}] for p in prompts]

from transformers import AutoConfig
config = AutoConfig.from_pretrained("{MODEL}", trust_remote_code=True)
vocab_size = config.vocab_size

llm = LLM(model="{MODEL}", dtype="bfloat16", tensor_parallel_size=1,
          trust_remote_code=True, enforce_eager=True, max_logprobs=vocab_size)
params = SamplingParams(temperature=0, max_tokens={MAX_NEW_TOKENS}, logprobs=-1)
outputs = llm.chat(conversations, params)

for idx, output in enumerate(outputs):
    prompt_ids = list(output.prompt_token_ids)
    gen_out = output.outputs[0]
    gen_ids = list(gen_out.token_ids)
    gen_len = len(gen_ids)
    full_ids = torch.tensor(prompt_ids + gen_ids, dtype=torch.long)

    log_probs = torch.full((gen_len, vocab_size), float("-inf"), dtype=torch.float32)
    for t, lp_dict in enumerate(gen_out.logprobs):
        ids = torch.tensor([k for k in lp_dict.keys()], dtype=torch.long)
        vals = torch.tensor([v.logprob for v in lp_dict.values()], dtype=torch.float32)
        log_probs[t].scatter_(0, ids, vals)

    torch.save({{
        "sample_idx": idx,
        "prompt_length": len(prompt_ids),
        "full_ids": full_ids,
        "student_log_probs": log_probs.to(torch.bfloat16),
    }}, Path("{student_dir}") / f"sample_{{idx:03d}}.pt")
    print(f"Sample {{idx}}: {{gen_len}} tokens, vocab={{vocab_size}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"vLLM subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    yield student_dir
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_output_files_exist(vllm_output_dir):
    """Should produce one .pt file per prompt."""
    files = sorted(vllm_output_dir.glob("sample_*.pt"))
    assert len(files) == len(PROMPTS), f"Expected {len(PROMPTS)} files, got {len(files)}"


def test_output_format(vllm_output_dir):
    """Verify .pt files have correct keys, shapes, and dtypes."""
    for f in sorted(vllm_output_dir.glob("sample_*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=True)

        # Required keys
        assert set(data.keys()) == {"sample_idx", "prompt_length", "full_ids", "student_log_probs"}

        # Types
        assert isinstance(data["sample_idx"], int)
        assert isinstance(data["prompt_length"], int)
        assert data["full_ids"].dtype == torch.long
        assert data["student_log_probs"].dtype == torch.bfloat16

        # Shape consistency
        full_len = data["full_ids"].shape[0]
        gen_len = data["student_log_probs"].shape[0]
        prompt_len = data["prompt_length"]
        assert full_len == prompt_len + gen_len, (
            f"full_ids({full_len}) != prompt({prompt_len}) + gen({gen_len})"
        )
        assert data["student_log_probs"].shape[1] > 100000, "vocab_size too small"


def test_log_probs_valid(vllm_output_dir):
    """Verify log-probs are valid probability distributions."""
    for f in sorted(vllm_output_dir.glob("sample_*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=True)
        lp = data["student_log_probs"].float()

        # All log-probs should be <= 0
        assert (lp <= 1e-5).all(), "Found positive log-probs"

        # exp(log_probs) should sum to ~1 per position
        probs_sum = lp.exp().sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-2), (
            f"Probs don't sum to 1: min={probs_sum.min():.4f}, max={probs_sum.max():.4f}"
        )


def test_no_inf_in_top_tokens(vllm_output_dir):
    """The generated token at each position should have a finite log-prob."""
    for f in sorted(vllm_output_dir.glob("sample_*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=True)
        lp = data["student_log_probs"].float()
        full_ids = data["full_ids"]
        prompt_len = data["prompt_length"]

        # Response token IDs
        response_ids = full_ids[prompt_len:]
        # Log-prob of the actually generated token at each position
        for t in range(len(response_ids)):
            token_lp = lp[t, response_ids[t]].item()
            assert token_lp > float("-inf"), (
                f"Generated token {response_ids[t]} at position {t} has -inf log-prob"
            )

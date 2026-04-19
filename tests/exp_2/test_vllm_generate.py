"""Test vLLM generation produces correct output and matches HF single-sample.

Requires GPU, CUDA, and vLLM installed.
Run with: uv run pytest tests/exp_2/test_vllm_generate.py -v -s
"""

import gc
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
    """Run vLLM subprocess + HF forward pass on 2 test prompts."""
    tmpdir = Path(tempfile.mkdtemp())
    token_ids_dir = tmpdir / "vllm_token_ids"
    token_ids_dir.mkdir()

    # Step 1: vLLM generation in subprocess (isolated CUDA context)
    # Write prompts as a mini dataset substitute
    # We call _vllm_subprocess directly with our test prompts
    _run_vllm_on_prompts(PROMPTS, MODEL, MAX_NEW_TOKENS, token_ids_dir)

    # Step 2: HF forward pass for log-probs (clean CUDA context)
    results = _hf_forward_on_token_ids(token_ids_dir, MODEL)

    yield results
    shutil.rmtree(tmpdir, ignore_errors=True)


def _run_vllm_on_prompts(prompts, model_name, max_tokens, output_dir):
    """Run vLLM generation in a subprocess to avoid CUDA corruption."""
    # Write a small helper script inline since _vllm_subprocess expects MATH500
    script = f"""
import json, sys
from pathlib import Path
from vllm import LLM, SamplingParams

prompts = {prompts!r}
conversations = [[{{"role": "user", "content": p}}] for p in prompts]

llm = LLM(model="{model_name}", dtype="bfloat16", tensor_parallel_size=1,
          trust_remote_code=True, enforce_eager=True)
params = SamplingParams(temperature=0, max_tokens={max_tokens})
outputs = llm.chat(conversations, params)

for idx, output in enumerate(outputs):
    prompt_ids = list(output.prompt_token_ids)
    gen_ids = list(output.outputs[0].token_ids)
    data = {{
        "sample_idx": idx,
        "prompt_length": len(prompt_ids),
        "full_ids": prompt_ids + gen_ids,
        "gen_len": len(gen_ids),
    }}
    with open(Path("{output_dir}") / f"sample_{{idx:03d}}.json", "w") as f:
        json.dump(data, f)
    print(f"Sample {{idx}}: {{len(gen_ids)}} tokens")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"vLLM subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")


def _hf_forward_on_token_ids(token_ids_dir, model_name):
    """Load HF model and run forward pass on saved token IDs."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = []
    for f in sorted(token_ids_dir.glob("sample_*.json")):
        with open(f) as fh:
            data = json.load(fh)

        full_ids = torch.tensor(data["full_ids"], dtype=torch.long)
        prompt_length = data["prompt_length"]
        gen_len = data["gen_len"]

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


@pytest.fixture(scope="module")
def hf_results():
    """HF single-sample generation + forward pass as ground truth."""
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
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        full_ids = gen_out[0].cpu()
        gen_len = len(full_ids) - prompt_length

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
    """Verify vLLM path produces valid output structure."""
    for i, r in enumerate(vllm_results):
        assert r["full_ids"].dtype == torch.long
        assert r["full_ids"].shape[0] == r["prompt_length"] + r["gen_len"], (
            f"Prompt {i}: full_ids length mismatch"
        )
        assert r["log_probs"].shape == (r["gen_len"], r["log_probs"].shape[1])
        assert r["log_probs"].shape[1] > 100000, "vocab_size too small"


def test_vllm_log_probs_valid(vllm_results):
    """Verify log-probs are valid probability distributions."""
    for i, r in enumerate(vllm_results):
        lp = r["log_probs"]
        assert (lp <= 0).all(), f"Prompt {i}: found positive log-probs"
        probs_sum = lp.exp().sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-2), (
            f"Prompt {i}: probs don't sum to 1"
        )


def test_vllm_vs_hf_token_ids(vllm_results, hf_results):
    """vLLM and HF should produce identical response tokens."""
    for i, (v, h) in enumerate(zip(vllm_results, hf_results)):
        v_response = v["full_ids"][v["prompt_length"]:]
        h_response = h["full_ids"][h["prompt_length"]:]
        assert v["gen_len"] == h["gen_len"], (
            f"Prompt {i}: gen_len differs: vllm={v['gen_len']}, hf={h['gen_len']}"
        )
        assert torch.equal(v_response, h_response), (
            f"Prompt {i}: response token IDs differ"
        )


def test_vllm_vs_hf_log_probs(vllm_results, hf_results):
    """Log-probs should match since both use HF forward pass on same tokens."""
    for i, (v, h) in enumerate(zip(vllm_results, hf_results)):
        if v["gen_len"] != h["gen_len"]:
            pytest.skip(f"Prompt {i}: gen_len differs, can't compare log-probs")

        max_diff = (v["log_probs"] - h["log_probs"]).abs().max().item()
        assert max_diff < 1e-3, (
            f"Prompt {i}: log-prob max diff = {max_diff:.6f} (expected < 1e-3)"
        )

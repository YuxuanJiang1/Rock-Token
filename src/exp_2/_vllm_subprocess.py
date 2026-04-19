"""vLLM generation subprocess — runs in isolation to avoid CUDA context issues.

Called by vllm_generate.py via subprocess.run(). Generates responses with
logprobs=-1 (full vocabulary log-probs), converts to tensors, and saves
.pt files directly. No HF model needed.
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, help="Directory to save sample_*.pt files")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--sample-indices", required=True, help="JSON list of sample indices")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from src.exp_2.utils import load_math500

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = json.loads(args.sample_indices)

    dataset = load_math500()

    # Build conversations
    conversations = []
    for i in sample_indices:
        conversations.append([{"role": "user", "content": dataset[i]["problem"]}])

    # Init vLLM with max_logprobs set to vocab size for full-vocab logprobs
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = config.vocab_size

    print(f"[vLLM] Loading model {args.model} (TP={args.tensor_parallel}, vocab={vocab_size})...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        max_logprobs=vocab_size,
    )

    # logprobs=-1 returns full vocabulary log-probs
    sampling_params = SamplingParams(
        temperature=0, max_tokens=args.max_new_tokens, logprobs=-1
    )

    print(f"[vLLM] Generating {len(conversations)} responses (with full-vocab logprobs)...")
    outputs = llm.chat(conversations, sampling_params)

    # Get vocab size from first output
    first_logprobs = outputs[0].outputs[0].logprobs
    if first_logprobs:
        vocab_size = len(first_logprobs[0])
    else:
        raise RuntimeError("No logprobs returned. Check model and SamplingParams.")

    print(f"[vLLM] Converting logprobs to tensors (vocab_size={vocab_size})...")

    for idx, output in enumerate(outputs):
        sample_idx = sample_indices[idx]
        prompt_ids = list(output.prompt_token_ids)
        gen_output = output.outputs[0]
        gen_ids = list(gen_output.token_ids)
        gen_len = len(gen_ids)

        full_ids = torch.tensor(prompt_ids + gen_ids, dtype=torch.long)
        prompt_length = len(prompt_ids)

        if gen_len == 0:
            print(f"[vLLM] Sample {sample_idx}: no response, skipping")
            continue

        # Convert logprobs dicts to dense tensor
        log_probs = torch.full((gen_len, vocab_size), float("-inf"), dtype=torch.float32)
        for t, lp_dict in enumerate(gen_output.logprobs):
            ids = torch.tensor([k for k in lp_dict.keys()], dtype=torch.long)
            vals = torch.tensor([v.logprob for v in lp_dict.values()], dtype=torch.float32)
            log_probs[t].scatter_(0, ids, vals)

        torch.save(
            {
                "sample_idx": sample_idx,
                "prompt_length": prompt_length,
                "full_ids": full_ids,
                "student_log_probs": log_probs.to(torch.bfloat16),
            },
            output_dir / f"sample_{sample_idx:03d}.pt",
        )
        print(f"[vLLM] Sample {sample_idx}: {gen_len} tokens saved")

    print(f"[vLLM] Done. Saved {len(outputs)} samples.")


if __name__ == "__main__":
    main()

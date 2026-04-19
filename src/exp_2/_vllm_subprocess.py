"""vLLM generation subprocess — runs in isolation to avoid CUDA context corruption.

Called by vllm_generate.py via subprocess.run(). Generates responses and saves
token IDs to JSON files. The parent process then loads HF for log-prob extraction
with a clean CUDA context.
"""

import argparse
import json
from pathlib import Path

from src.exp_2.utils import load_math500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--sample-indices", required=True, help="JSON list of sample indices")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = json.loads(args.sample_indices)

    dataset = load_math500()

    # Build conversations
    conversations = []
    for i in sample_indices:
        conversations.append([{"role": "user", "content": dataset[i]["problem"]}])

    # Init vLLM
    print(f"[vLLM subprocess] Loading model {args.model} (TP={args.tensor_parallel})...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)

    print(f"[vLLM subprocess] Generating {len(conversations)} responses...")
    outputs = llm.chat(conversations, sampling_params)

    # Save token IDs to JSON (lightweight, no tensors)
    for idx, output in enumerate(outputs):
        sample_idx = sample_indices[idx]
        prompt_ids = list(output.prompt_token_ids)
        gen_ids = list(output.outputs[0].token_ids)

        token_data = {
            "sample_idx": sample_idx,
            "prompt_length": len(prompt_ids),
            "full_ids": prompt_ids + gen_ids,
            "gen_len": len(gen_ids),
        }

        out_path = output_dir / f"sample_{sample_idx:03d}.json"
        with open(out_path, "w") as f:
            json.dump(token_data, f)

        print(f"[vLLM subprocess] Sample {sample_idx}: {len(gen_ids)} tokens")

    print(f"[vLLM subprocess] Done. Saved {len(outputs)} token ID files.")


if __name__ == "__main__":
    main()

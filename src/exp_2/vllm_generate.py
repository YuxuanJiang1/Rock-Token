"""vLLM-based generation for Phase 1.

Uses vLLM for fast batched generation (avoids HF left-padding + RoPE issues),
then HF forward pass per sample to extract full-vocabulary log-probs.
"""

import gc
from pathlib import Path

import torch
from rich.progress import Progress

from src.exp_2.utils import load_math500, load_model_and_tokenizer


def vllm_generate(
    model_name: str,
    output_dir: Path,
    max_new_tokens: int = 2048,
    tensor_parallel_size: int = 1,
) -> None:
    """Generate student responses via vLLM + extract log-probs via HF forward pass.

    Step 1: vLLM generates all remaining responses (fast, correct batching).
    Step 2: Destroy vLLM engine, free GPU memory.
    Step 3: HF forward pass per sample to get full-vocab log-probs.
    Step 4: Save .pt files in the same format as HF-based Phase 1.

    Output format per sample (student_data/sample_{i:03d}.pt):
        sample_idx: int
        prompt_length: int
        full_ids: Tensor (seq_len,)
        student_log_probs: Tensor (gen_len, vocab_size) bfloat16
    """
    from vllm import LLM, SamplingParams

    student_data_dir = output_dir / "student_data"
    student_data_dir.mkdir(parents=True, exist_ok=True)

    # Resume check
    existing = {
        int(f.stem.split("_")[1])
        for f in student_data_dir.glob("sample_*.pt")
    }
    dataset = load_math500()
    remaining = [i for i in range(len(dataset)) if i not in existing]

    if not remaining:
        print(f"Phase 1 complete ({len(existing)}/{len(dataset)} samples already exist)")
        return

    print(f"Phase 1 (vLLM): {len(existing)}/{len(dataset)} done, {len(remaining)} remaining")

    # Build conversations for remaining samples
    conversations = []
    for i in remaining:
        conversations.append([{"role": "user", "content": dataset[i]["problem"]}])

    # --- Step 1: vLLM generation ---
    print("Loading vLLM engine...")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    print(f"Generating {len(remaining)} responses with vLLM...")
    outputs = llm.chat(conversations, sampling_params)

    # Extract token IDs from vLLM outputs
    generated_data = []
    for idx, output in enumerate(outputs):
        sample_idx = remaining[idx]
        prompt_ids = list(output.prompt_token_ids)
        gen_ids = list(output.outputs[0].token_ids)
        generated_data.append({
            "sample_idx": sample_idx,
            "prompt_length": len(prompt_ids),
            "full_ids": torch.tensor(prompt_ids + gen_ids, dtype=torch.long),
            "gen_len": len(gen_ids),
        })

    print(f"vLLM generation complete. Generated {len(generated_data)} responses.")

    # --- Step 2: Destroy vLLM engine ---
    del llm, outputs
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 3: HF forward pass for full-vocab log-probs ---
    print("Loading HF model for log-prob extraction...")
    model, _ = load_model_and_tokenizer(model_name)

    with Progress() as progress:
        task = progress.add_task(
            "Phase 1: Extracting log-probs", total=len(generated_data)
        )
        for data in generated_data:
            sample_idx = data["sample_idx"]
            prompt_length = data["prompt_length"]
            full_ids = data["full_ids"]
            gen_len = data["gen_len"]

            if gen_len == 0:
                print(f"  Sample {sample_idx}: no response generated, skipping")
                progress.advance(task)
                continue

            # Forward pass — same pattern as Phase 2 teacher logits
            with torch.no_grad():
                fwd = model(input_ids=full_ids.unsqueeze(0).to(model.device))

            logits = fwd.logits[0].float().cpu()
            # logits[t] predicts token t+1
            # logits[prompt_len-1 : prompt_len-1+gen_len] predict response tokens
            student_logits = logits[prompt_length - 1 : prompt_length - 1 + gen_len]
            log_probs = torch.log_softmax(student_logits, dim=-1).to(torch.bfloat16)

            torch.save(
                {
                    "sample_idx": sample_idx,
                    "prompt_length": prompt_length,
                    "full_ids": full_ids,
                    "student_log_probs": log_probs,
                },
                student_data_dir / f"sample_{sample_idx:03d}.pt",
            )
            print(f"  Sample {sample_idx}: {gen_len} response tokens")
            progress.advance(task)

    del model
    torch.cuda.empty_cache()
    print("Phase 1 (vLLM) complete")

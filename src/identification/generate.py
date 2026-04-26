"""Phase 1: Generate student outputs using vLLM.

Loads the on-policy distilled student (θ*) via vLLM, generates multiple
sampled outputs per prompt at temperature=1.0, and saves all sequences
with metadata for Phase 2 forward passes.
"""

from pathlib import Path

import torch
from rich.console import Console

from src.identification.prompts import load_mixed_prompts


def run_phase1(config: dict, output_dir: Path) -> Path:
    """Generate student outputs and save sequences to disk.

    Returns path to the saved sequences file.
    """
    from vllm import LLM, SamplingParams

    console = Console()
    output_path = output_dir / "phase1_sequences.pt"

    if output_path.exists():
        data = torch.load(output_path, weights_only=False)
        console.print(
            f"[yellow]Phase 1 already done ({len(data)} sequences at {output_path})[/yellow]"
        )
        return output_path

    # Load prompts
    pc = config["prompts"]
    prompts = load_mixed_prompts(
        math_count=pc["math_train_count"],
        gsm8k_count=pc["gsm8k_train_count"],
        mbpp_count=pc["mbpp_train_count"],
        alpaca_count=pc["alpaca_count"],
        seed=pc["seed"],
    )

    # Load model
    gc = config["generation"]
    cc = config["compute"]
    model_name = config["models"]["student_onpolicy"]

    console.print(f"Loading [bold]{model_name}[/bold] via vLLM (TP={cc['tensor_parallel_size']})...")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=cc["tensor_parallel_size"],
        trust_remote_code=True,
        seed=cc["seed"],
        max_model_len=cc["max_model_len"],
    )

    sampling = SamplingParams(
        temperature=gc["temperature"],
        max_tokens=gc["max_new_tokens"],
        seed=cc["seed"],
        n=gc["num_outputs_per_prompt"],
    )

    # Build conversations
    conversations = []
    for p in prompts:
        conversations.append([{"role": "user", "content": p["prompt"]}])

    console.print(
        f"Generating {gc['num_outputs_per_prompt']} outputs × {len(prompts)} prompts "
        f"(temp={gc['temperature']})..."
    )
    outputs = llm.chat(conversations, sampling)

    # Collect results
    sequences = []
    for prompt_idx, output in enumerate(outputs):
        source = prompts[prompt_idx]["source"]
        prompt_ids = list(output.prompt_token_ids)
        for completion in output.outputs:
            sequences.append({
                "prompt_idx": prompt_idx,
                "source": source,
                "prompt_token_ids": prompt_ids,
                "output_token_ids": list(completion.token_ids),
            })

    console.print(f"Generated {len(sequences)} total sequences")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sequences, output_path)
    console.print(f"Saved to {output_path}")

    # Free GPU
    del llm
    torch.cuda.empty_cache()

    return output_path

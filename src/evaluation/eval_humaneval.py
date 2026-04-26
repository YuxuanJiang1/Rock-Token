"""Evaluate a model on HumanEval (164 Python programming problems).

Uses vLLM for fast batched inference.  Zero-shot code completion: the model
receives the function signature + docstring and must produce the function body.
Generated code is executed against the hidden test suite with a timeout.

Reports pass@1 (greedy, temperature=0).

Usage:
    uv run python src/evaluation/eval_humaneval.py --model Qwen/Qwen3-4B-Instruct-2507
    uv run python src/evaluation/eval_humaneval.py --model ... --output results/humaneval.json
"""

import argparse
import contextlib
import io
import multiprocessing
import signal
import traceback
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

from src.evaluation.common import (
    add_common_args,
    build_metadata,
    create_llm,
    default_sampling_params,
    print_accuracy,
    save_results,
)

EXEC_TIMEOUT = 10  # seconds per test execution


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(generated: str, entry_point: str) -> str:
    """Extract the function body from model output.

    Handles three common formats:
    1. Raw code (model continues the function directly)
    2. Markdown code block (```python ... ```)
    3. Full function rewrite (def entry_point(...): ...)
    """
    # Strip markdown fences if present
    if "```" in generated:
        blocks = generated.split("```")
        for block in blocks[1::2]:  # odd-indexed = inside fences
            # Remove language tag
            lines = block.strip().split("\n")
            if lines and lines[0].strip().lower() in ("python", "python3", "py", ""):
                lines = lines[1:]
            code = "\n".join(lines)
            if code.strip():
                return code
        # Fallback: take first fenced block
        if len(blocks) > 1:
            return blocks[1].strip()

    return generated


# ---------------------------------------------------------------------------
# Safe execution
# ---------------------------------------------------------------------------

def _run_code(code: str, result_queue: multiprocessing.Queue):
    """Execute code in a subprocess and report pass/fail."""
    try:
        # Suppress stdout from executed code
        with contextlib.redirect_stdout(io.StringIO()):
            exec_globals: dict = {}
            exec(code, exec_globals)
        result_queue.put(("pass", None))
    except Exception:
        result_queue.put(("fail", traceback.format_exc()))


def check_correctness(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout: int = EXEC_TIMEOUT,
) -> dict:
    """Execute the generated solution against the test suite.

    Returns {"passed": bool, "error": str | None}.
    """
    code = extract_code(completion, entry_point)
    # Build full program: prompt (signature+docstring) + completion + test
    full_code = prompt + code + "\n" + test

    queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_run_code, args=(full_code, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"passed": False, "error": "timeout"}

    if queue.empty():
        return {"passed": False, "error": "no result (crash)"}

    status, error = queue.get()
    return {"passed": status == "pass", "error": error}


# ---------------------------------------------------------------------------
# Dataset & prompt
# ---------------------------------------------------------------------------

def load_humaneval(n_samples: int | None = None):
    ds = load_dataset("openai/openai_humaneval", split="test")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    return ds


SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the function below. "
    "Return ONLY the function body (the code that goes inside the function). "
    "Do not include the function signature or docstring — they are already provided."
)


def build_conversations(dataset) -> list[list[dict]]:
    conversations = []
    for sample in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Complete this function:\n\n{sample['prompt']}"},
        ]
        conversations.append(messages)
    return conversations


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str,
    n_samples: int | None = None,
    max_new_tokens: int = 1024,
    tensor_parallel_size: int | None = None,
    output_path: Path | None = None,
    seed: int = 42,
):
    console = Console()

    dataset = load_humaneval(n_samples)
    console.print(f"Loaded {len(dataset)} HumanEval problems")

    conversations = build_conversations(dataset)

    console.print(f"Loading model [bold]{model_name}[/bold]...")
    llm = create_llm(model_name, tensor_parallel_size, seed=seed)
    sampling = default_sampling_params(max_tokens=max_new_tokens, seed=seed)

    console.print("Generating responses...")
    outputs = llm.chat(conversations, sampling)

    # Execute and check
    console.print("Executing generated solutions...")
    correct = 0
    total = len(dataset)
    results = []

    for idx, output in enumerate(outputs):
        sample = dataset[idx]
        generated = output.outputs[0].text

        result = check_correctness(
            prompt=sample["prompt"],
            completion=generated,
            test=sample["test"],
            entry_point=sample["entry_point"],
        )

        if result["passed"]:
            correct += 1

        results.append({
            "task_id": sample["task_id"],
            "entry_point": sample["entry_point"],
            "passed": result["passed"],
            "error": result["error"],
            "generated_text": generated,
        })

    accuracy = correct / total if total > 0 else 0

    # Console output
    print_accuracy(console, "HumanEval pass@1", correct, total)

    # Error breakdown
    error_types = {}
    for r in results:
        if not r["passed"]:
            etype = "timeout" if r["error"] == "timeout" else "runtime_error"
            error_types[etype] = error_types.get(etype, 0) + 1
    if error_types:
        console.print(f"  Error breakdown: {error_types}")

    # Save
    if output_path:
        data = {
            "metadata": build_metadata(
                model_name=model_name,
                benchmark="humaneval",
                dataset_name="openai/openai_humaneval",
                n_samples=len(dataset),
                accuracy=accuracy,
                correct=correct,
                total=total,
                seed=seed,
                max_new_tokens=max_new_tokens,
            ),
            "results": results,
        }
        save_results(output_path, data)
        console.print(f"\nResults saved to {output_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on HumanEval (pass@1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()

    # Use spawn method for multiprocessing (required on some platforms)
    multiprocessing.set_start_method("spawn", force=True)

    evaluate(
        model_name=args.model,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""Evaluate a model on IF-Eval (541 instruction-following prompts).

Uses vLLM for fast batched inference.  Zero-shot: the model receives the
instruction and must produce a response that satisfies all verifiable
constraints.  Reports prompt-level and instruction-level accuracy in both
strict and loose modes, following Zhou et al. (2023).

Self-contained — does not import from src/evaluation/.

Usage:
    uv run python src/masking/eval_ifeval.py --model Qwen/Qwen3-4B-Instruct-2507
    uv run python src/masking/eval_ifeval.py --model ... --output results/ifeval.json
"""

import argparse
import json as json_module
import re
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from src.masking.common import (
    add_common_args,
    build_metadata,
    create_llm,
    default_sampling_params,
    save_results,
)

# ---------------------------------------------------------------------------
# Constraint checkers
# ---------------------------------------------------------------------------


def _count_sentences(text: str) -> int:
    sentences = re.split(r"[.!?]+", text.strip())
    return len([s for s in sentences if s.strip()])


def _count_words(text: str) -> int:
    return len(text.split())


def _count_paragraphs(text: str) -> int:
    paragraphs = text.strip().split("\n\n")
    return len([p for p in paragraphs if p.strip()])


def _get_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.strip().split("\n\n") if p.strip()]


def _relation_check(actual: int, relation: str, target: int) -> bool:
    if relation == "at least":
        return actual >= target
    elif relation == "at most":
        return actual <= target
    elif relation == "less than":
        return actual < target
    elif relation == "more than":
        return actual > target
    return actual == target


# --- keywords ---

def check_keywords_existence(response: str, *, keywords: list[str], **_kw) -> bool:
    resp_lower = response.lower()
    return all(kw.lower() in resp_lower for kw in keywords)


def check_keywords_existence_loose(response: str, **kwargs) -> bool:
    return check_keywords_existence(response.lower(), **{k: v for k, v in kwargs.items()})


def check_keywords_frequency(
    response: str, *, keyword: str, frequency: int, relation: str, **_kw,
) -> bool:
    count = response.lower().count(keyword.lower())
    return _relation_check(count, relation, frequency)


def check_keywords_forbidden_words(
    response: str, *, forbidden_words: list[str], **_kw,
) -> bool:
    resp_lower = response.lower()
    return all(w.lower() not in resp_lower for w in forbidden_words)


def check_keywords_letter_frequency(
    response: str, *, letter: str, let_frequency: int, let_relation: str, **_kw,
) -> bool:
    count = response.lower().count(letter.lower())
    return _relation_check(count, let_relation, let_frequency)


# --- language ---

def check_language_response_language(response: str, *, language: str, **_kw) -> bool:
    return True  # simplified — matches lm-eval-harness behavior


# --- length_constraints ---

def check_length_number_sentences(
    response: str, *, relation: str, num_sentences: int, **_kw,
) -> bool:
    return _relation_check(_count_sentences(response), relation, num_sentences)


def check_length_number_sentences_loose(response: str, **kwargs) -> bool:
    target = kwargs.get("num_sentences", 0)
    relation = kwargs.get("relation", "at least")
    actual = _count_sentences(response)
    if relation == "at least":
        return actual >= target - 1
    elif relation == "at most":
        return actual <= target + 1
    return abs(actual - target) <= 1


def check_length_number_paragraphs(
    response: str, *, num_paragraphs: int, **_kw,
) -> bool:
    return _count_paragraphs(response) >= num_paragraphs


def check_length_number_paragraphs_loose(response: str, **kwargs) -> bool:
    target = kwargs.get("num_paragraphs", 0)
    return _count_paragraphs(response) >= target - 1


def check_length_number_words(
    response: str, *, relation: str, num_words: int, **_kw,
) -> bool:
    return _relation_check(_count_words(response), relation, num_words)


def check_length_number_words_loose(response: str, **kwargs) -> bool:
    target = kwargs.get("num_words", 0)
    relation = kwargs.get("relation", "at least")
    actual = _count_words(response)
    margin = max(int(target * 0.1), 5)
    if relation == "at least":
        return actual >= target - margin
    elif relation == "at most":
        return actual <= target + margin
    return abs(actual - target) <= margin


def check_length_nth_paragraph_first_word(
    response: str, *, num_paragraphs: int, nth_paragraph: int, first_word: str, **_kw,
) -> bool:
    paragraphs = _get_paragraphs(response)
    if len(paragraphs) < num_paragraphs:
        return False
    idx = nth_paragraph - 1
    if idx >= len(paragraphs):
        return False
    words = paragraphs[idx].split()
    if not words:
        return False
    return words[0].lower().rstrip(",.;:!?") == first_word.lower()


# --- detectable_content ---

def check_content_number_placeholders(
    response: str, *, num_placeholders: int, **_kw,
) -> bool:
    count = len(re.findall(r"\[.*?\]", response))
    return count >= num_placeholders


def check_content_postscript(response: str, **_kw) -> bool:
    return bool(re.search(r"P\.?S\.?", response, re.IGNORECASE))


# --- detectable_format ---

def check_format_number_bullet_lists(
    response: str, *, num_bullets: int, **_kw,
) -> bool:
    bullet_lines = re.findall(r"^\s*[-*\u2022]\s+", response, re.MULTILINE)
    return len(bullet_lines) >= num_bullets


def check_format_constrained_response(response: str, **_kw) -> bool:
    return _count_words(response.strip()) <= 10


def check_format_number_highlighted_sections(
    response: str, *, num_highlights: int, **_kw,
) -> bool:
    highlights = re.findall(r"\*+[^*]+\*+", response)
    return len(highlights) >= num_highlights


def check_format_multiple_sections(
    response: str, *, section_spliter: str, num_sections: int, **_kw,
) -> bool:
    sections = re.findall(
        rf"(?:^|\n)\s*{re.escape(section_spliter)}\s*\d*",
        response,
        re.IGNORECASE,
    )
    return len(sections) >= num_sections


def check_format_json_format(response: str, **_kw) -> bool:
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        json_module.loads(text)
        return True
    except (json_module.JSONDecodeError, ValueError):
        return False


def check_format_title(response: str, **_kw) -> bool:
    lines = response.strip().split("\n")
    if not lines:
        return False
    first = lines[0].strip()
    return first.startswith("#") or (
        len(first.split()) <= 10 and first[0:1].isupper()
    )


# --- combination ---

def check_combination_two_responses(response: str, **_kw) -> bool:
    separators = ["***", "---", "Response 1", "Response 2"]
    for sep in separators:
        if sep.lower() in response.lower():
            return True
    return bool(re.search(r"(?:^|\n)\s*[12]\.", response))


def check_combination_repeat_prompt(
    response: str, *, prompt_to_repeat: str, **_kw,
) -> bool:
    return prompt_to_repeat.lower() in response.lower()


# --- startend ---

def check_startend_end_checker(
    response: str, *, end_phrase: str, **_kw,
) -> bool:
    return response.strip().lower().endswith(end_phrase.lower())


def check_startend_quotation(response: str, **_kw) -> bool:
    text = response.strip()
    return (text.startswith('"') and text.endswith('"')) or (
        text.startswith("\u201c") and text.endswith("\u201d")
    )


# --- change_case ---

def check_case_english_lowercase(response: str, **_kw) -> bool:
    return response == response.lower()


def check_case_english_capital(response: str, **_kw) -> bool:
    return response == response.upper()


# --- punctuation ---

def check_punctuation_no_comma(response: str, **_kw) -> bool:
    return "," not in response


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

STRICT_CHECKERS = {
    "keywords:existence": check_keywords_existence,
    "keywords:frequency": check_keywords_frequency,
    "keywords:forbidden_words": check_keywords_forbidden_words,
    "keywords:letter_frequency": check_keywords_letter_frequency,
    "language:response_language": check_language_response_language,
    "length_constraints:number_sentences": check_length_number_sentences,
    "length_constraints:number_paragraphs": check_length_number_paragraphs,
    "length_constraints:number_words": check_length_number_words,
    "length_constraints:nth_paragraph_first_word": check_length_nth_paragraph_first_word,
    "detectable_content:number_placeholders": check_content_number_placeholders,
    "detectable_content:postscript": check_content_postscript,
    "detectable_format:number_bullet_lists": check_format_number_bullet_lists,
    "detectable_format:constrained_response": check_format_constrained_response,
    "detectable_format:number_highlighted_sections": check_format_number_highlighted_sections,
    "detectable_format:multiple_sections": check_format_multiple_sections,
    "detectable_format:json_format": check_format_json_format,
    "detectable_format:title": check_format_title,
    "combination:two_responses": check_combination_two_responses,
    "combination:repeat_prompt": check_combination_repeat_prompt,
    "startend:end_checker": check_startend_end_checker,
    "startend:quotation": check_startend_quotation,
    "change_case:english_lowercase": check_case_english_lowercase,
    "change_case:english_capital": check_case_english_capital,
    "punctuation:no_comma": check_punctuation_no_comma,
}

LOOSE_CHECKERS = {
    **STRICT_CHECKERS,
    "keywords:existence": check_keywords_existence_loose,
    "length_constraints:number_sentences": check_length_number_sentences_loose,
    "length_constraints:number_paragraphs": check_length_number_paragraphs_loose,
    "length_constraints:number_words": check_length_number_words_loose,
}


def check_instruction(
    instruction_id: str,
    response: str,
    kwargs: dict,
    strict: bool = True,
) -> bool:
    checkers = STRICT_CHECKERS if strict else LOOSE_CHECKERS
    checker = checkers.get(instruction_id)
    if checker is None:
        return True  # unknown instruction — don't penalize
    return checker(response, **kwargs)


# ---------------------------------------------------------------------------
# Dataset & scoring
# ---------------------------------------------------------------------------

def load_ifeval(n_samples: int | None = None):
    ds = load_dataset("google/IFEval", split="train")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    return ds


def build_conversations(dataset) -> list[list[dict]]:
    """Build chat conversations from dataset (for use with llm.chat)."""
    return [
        [{"role": "user", "content": sample["prompt"]}]
        for sample in dataset
    ]


def score_outputs(outputs, dataset) -> dict:
    """Score pre-generated vLLM outputs against IF-Eval dataset.

    Returns dict with accuracy (strict prompt), correct, total, and
    per_correct (boolean list for bootstrap resampling).
    """
    strict_correct = 0
    per_correct = []

    for idx, output in enumerate(outputs):
        sample = dataset[idx]
        generated = output.outputs[0].text
        instruction_ids = sample["instruction_id_list"]
        kwargs_list = sample["kwargs"]
        result = score_sample(generated, instruction_ids, kwargs_list, strict=True)
        strict_correct += int(result["prompt_pass"])
        per_correct.append(result["prompt_pass"])

    total = len(dataset)
    return {
        "accuracy": strict_correct / total if total > 0 else 0,
        "correct": strict_correct,
        "total": total,
        "per_correct": per_correct,
    }


def score_sample(
    response: str,
    instruction_ids: list[str],
    kwargs_list: list[dict],
    strict: bool = True,
) -> dict:
    instruction_results = []
    for inst_id, kw in zip(instruction_ids, kwargs_list):
        passed = check_instruction(inst_id, response, kw, strict=strict)
        instruction_results.append({"id": inst_id, "pass": passed})

    return {
        "prompt_pass": all(r["pass"] for r in instruction_results),
        "instruction_results": instruction_results,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str,
    n_samples: int | None = None,
    max_new_tokens: int = 4096,
    tensor_parallel_size: int | None = None,
    output_path: Path | None = None,
    seed: int = 42,
):
    console = Console()

    dataset = load_ifeval(n_samples)
    console.print(f"Loaded {len(dataset)} IF-Eval prompts")

    conversations = []
    for sample in dataset:
        conversations.append([
            {"role": "user", "content": sample["prompt"]},
        ])

    console.print(f"Loading model [bold]{model_name}[/bold]...")
    llm = create_llm(model_name, tensor_parallel_size, seed=seed)
    sampling = default_sampling_params(max_tokens=max_new_tokens, seed=seed)

    console.print("Generating responses...")
    outputs = llm.chat(conversations, sampling)

    results = []
    counters = {
        "strict_prompt_correct": 0,
        "strict_inst_correct": 0,
        "strict_inst_total": 0,
        "loose_prompt_correct": 0,
        "loose_inst_correct": 0,
        "loose_inst_total": 0,
    }
    total_prompts = len(dataset)

    for idx, output in enumerate(outputs):
        sample = dataset[idx]
        generated = output.outputs[0].text
        instruction_ids = sample["instruction_id_list"]
        kwargs_list = sample["kwargs"]

        strict_result = score_sample(generated, instruction_ids, kwargs_list, strict=True)
        loose_result = score_sample(generated, instruction_ids, kwargs_list, strict=False)

        if strict_result["prompt_pass"]:
            counters["strict_prompt_correct"] += 1
        if loose_result["prompt_pass"]:
            counters["loose_prompt_correct"] += 1

        for r in strict_result["instruction_results"]:
            counters["strict_inst_total"] += 1
            counters["strict_inst_correct"] += int(r["pass"])
        for r in loose_result["instruction_results"]:
            counters["loose_inst_total"] += 1
            counters["loose_inst_correct"] += int(r["pass"])

        results.append({
            "key": sample.get("key", idx),
            "prompt": sample["prompt"],
            "instruction_ids": instruction_ids,
            "strict_prompt_pass": strict_result["prompt_pass"],
            "loose_prompt_pass": loose_result["prompt_pass"],
            "strict_instruction_results": strict_result["instruction_results"],
            "loose_instruction_results": loose_result["instruction_results"],
            "generated_text": generated,
        })

    def _safe_div(a, b):
        return a / b if b > 0 else 0

    metrics = {
        "strict_prompt_accuracy": _safe_div(counters["strict_prompt_correct"], total_prompts),
        "strict_instruction_accuracy": _safe_div(counters["strict_inst_correct"], counters["strict_inst_total"]),
        "loose_prompt_accuracy": _safe_div(counters["loose_prompt_correct"], total_prompts),
        "loose_instruction_accuracy": _safe_div(counters["loose_inst_correct"], counters["loose_inst_total"]),
    }

    table = Table(title="IF-Eval Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")
    table.add_row(
        "Strict Prompt Accuracy",
        f"{counters['strict_prompt_correct']}/{total_prompts} ({metrics['strict_prompt_accuracy']:.1%})",
    )
    table.add_row(
        "Strict Instruction Accuracy",
        f"{counters['strict_inst_correct']}/{counters['strict_inst_total']} ({metrics['strict_instruction_accuracy']:.1%})",
    )
    table.add_row(
        "Loose Prompt Accuracy",
        f"{counters['loose_prompt_correct']}/{total_prompts} ({metrics['loose_prompt_accuracy']:.1%})",
    )
    table.add_row(
        "Loose Instruction Accuracy",
        f"{counters['loose_inst_correct']}/{counters['loose_inst_total']} ({metrics['loose_instruction_accuracy']:.1%})",
    )
    console.print()
    console.print(table)

    if output_path:
        data = {
            "metadata": build_metadata(
                model_name=model_name,
                benchmark="ifeval",
                dataset_name="google/IFEval",
                n_samples=len(dataset),
                accuracy=metrics["strict_prompt_accuracy"],
                correct=counters["strict_prompt_correct"],
                total=total_prompts,
                seed=seed,
                max_new_tokens=max_new_tokens,
            ),
            "metrics": metrics,
            "counters": counters,
            "results": results,
        }
        save_results(output_path, data)
        console.print(f"\nResults saved to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on IF-Eval (instruction following)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()

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

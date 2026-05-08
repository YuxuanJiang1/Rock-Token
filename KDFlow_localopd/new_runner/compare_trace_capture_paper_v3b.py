#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TokenInfo:
    idx: int
    token_id: Optional[int]
    token_text: str
    student_logprob: Optional[float] = None
    teacher_logprob: Optional[float] = None
    logprob_gap: Optional[float] = None
    token_loss: Optional[float] = None
    is_candidate: bool = False
    is_selected: bool = False
    notes: Optional[str] = None


@dataclass
class CandidateStep:
    candidate_idx: int
    token_text: str
    token_loss: float
    rollback_start_idx: int
    rollback_prefix_text: str
    student_probe_continuation_text: str
    teacher_probe_continuation_text: str
    student_probe_margin_score: float
    teacher_probe_margin_score: float
    divergence_score: float
    threshold: Optional[float] = None
    is_divergent: bool = False
    probe_continuation_text: Optional[str] = None
    probe_gap: Optional[float] = None

@dataclass
class LocalSample:
    source: str
    sample_idx: int
    text: str
    token_ids: List[int]
    margin_score: float


@dataclass
class TeacherRepair:
    selected_idx: int
    selected_token_text: str
    selected_token_id: int
    prefix_text: str
    student_chosen: Dict[str, Any]
    teacher_top_tokens: List[Dict[str, Any]]
    explanation: str


@dataclass
class MethodTrace:
    method_name: str
    prompt: str
    prompt_text: str
    response_text: str
    response_token_ids: List[int]
    tokens: List[TokenInfo]
    selected_index: Optional[int] = None
    candidate_indices: Optional[List[int]] = None
    selected_indices: Optional[List[int]] = None
    teacher_repair: Optional[TeacherRepair] = None
    # local OPD details
    divergence_index: Optional[int] = None
    candidate_steps: Optional[List[CandidateStep]] = None
    dynamic_threshold: Optional[float] = None
    rollback_prefix_token_ids: Optional[List[int]] = None
    rollback_prefix_text: Optional[str] = None
    teacher_samples_sorted: Optional[List[LocalSample]] = None
    student_samples_sorted: Optional[List[LocalSample]] = None
    local_distance_type: Optional[str] = None
    local_distance_value: Optional[float] = None
    scalar_metrics: Optional[Dict[str, Any]] = None
    raw_debug: Optional[Dict[str, Any]] = None


@dataclass
class CompareTrace:
    prompt: str
    created_at: float
    original_response_text: str
    baseline: MethodTrace
    local_opd: MethodTrace
    meta: Dict[str, Any]


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def build_prompt_text(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


def load_model_and_tokenizer(model_name_or_path: str, dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def generate_student_response(tokenizer, model, prompt_text: str, max_new_tokens: int, temperature: float, top_p: float):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    do_sample = temperature > 0
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        kwargs.update(dict(temperature=max(temperature, 1e-5), top_p=top_p))
    gen = model.generate(**kwargs)
    response_ids = gen[0, input_ids.shape[1]:].detach().cpu().tolist()
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return input_ids[0].detach().cpu(), response_ids, response_text


@torch.no_grad()
def score_continuation(tokenizer, model, prefix_ids: List[int], continuation_ids: List[int]) -> List[float]:
    if not continuation_ids:
        return []
    device = next(model.parameters()).device
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    cont = torch.tensor(continuation_ids, dtype=torch.long, device=device)
    full_ids = torch.cat([prefix, cont], dim=0).unsqueeze(0)
    outputs = model(input_ids=full_ids)
    logits = outputs.logits
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = full_ids[:, 1:]
    gathered = logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    start = prefix.numel() - 1
    end = start + cont.numel()
    return gathered[0, start:end].detach().cpu().tolist()


@torch.no_grad()
def teacher_top_next_tokens(tokenizer, model, prefix_ids: List[int], top_k: int = 8) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(input_ids=ids).logits[0, -1, :]
    logprobs = torch.log_softmax(logits, dim=-1)
    vals, inds = torch.topk(logprobs, k=top_k)
    out = []
    for rank, (lp, tid) in enumerate(zip(vals.detach().cpu().tolist(), inds.detach().cpu().tolist()), start=1):
        out.append({
            "rank": rank,
            "token_id": int(tid),
            "token_text": tokenizer.decode([int(tid)], skip_special_tokens=False),
            "teacher_logprob": float(lp),
            "teacher_prob": float(math.exp(lp)),
        })
    return out


def decode_tokens(tokenizer, token_ids: List[int]) -> List[str]:
    return [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]


def token_infos_from_lists(token_ids, token_texts, student_logprobs, teacher_logprobs, token_losses, candidate_indices=None, selected_indices=None, notes_map=None):
    candidate_set = set(candidate_indices or [])
    selected_set = set(selected_indices or [])
    notes_map = notes_map or {}
    out = []
    for i in range(len(token_ids)):
        gap = None
        if i < len(student_logprobs) and i < len(teacher_logprobs):
            gap = teacher_logprobs[i] - student_logprobs[i]
        out.append(TokenInfo(
            idx=i,
            token_id=token_ids[i],
            token_text=token_texts[i] if i < len(token_texts) else "",
            student_logprob=safe_float(student_logprobs[i] if i < len(student_logprobs) else None),
            teacher_logprob=safe_float(teacher_logprobs[i] if i < len(teacher_logprobs) else None),
            logprob_gap=safe_float(gap),
            token_loss=safe_float(token_losses[i] if i < len(token_losses) else None),
            is_candidate=i in candidate_set,
            is_selected=i in selected_set,
            notes=notes_map.get(i),
        ))
    return out


def select_topk_with_separation(values: List[float], k: int, min_sep: int) -> List[int]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    chosen = []
    for idx in order:
        if all(abs(idx - c) >= min_sep for c in chosen):
            chosen.append(idx)
        if len(chosen) >= k:
            break
    return chosen  # keep high-loss priority order


def prefix_ids_from_prompt_and_response(prompt_ids: List[int], response_ids: List[int], prefix_idx: int) -> List[int]:
    prefix_idx = max(0, min(prefix_idx, len(response_ids)))
    return list(prompt_ids) + list(response_ids[:prefix_idx])


def suffix_ids(response_ids: List[int], prefix_idx: int, max_len: int) -> List[int]:
    return list(response_ids[prefix_idx: prefix_idx + max_len])


def sample_local_continuations(tokenizer, model, prefix_ids: List[int], num_samples: int, max_new_tokens: int, temperature: float, top_p: float):
    if num_samples <= 0:
        return [], []
    device = next(model.parameters()).device
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(prefix)
    outs_ids, outs_text = [], []
    for _ in range(num_samples):
        gen = model.generate(
            input_ids=prefix,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = gen[0, prefix.shape[1]:].detach().cpu().tolist()
        outs_ids.append(new_ids)
        outs_text.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return outs_ids, outs_text


def sequence_score_mean_gap(student_tokenizer, teacher_tokenizer, student_model, teacher_model, prompt_prefix_ids: List[int], continuation_ids: List[int]) -> float:
    # Retokenize by text so this also works when tokenizers differ.
    cont_text = student_tokenizer.decode(continuation_ids, skip_special_tokens=True)
    prefix_text = student_tokenizer.decode(prompt_prefix_ids, skip_special_tokens=False)
    stu_prefix_ids = student_tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    stu_cont_ids = student_tokenizer(cont_text, add_special_tokens=False)["input_ids"]
    tea_prefix_ids = teacher_tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    tea_cont_ids = teacher_tokenizer(cont_text, add_special_tokens=False)["input_ids"]
    if not stu_cont_ids or not tea_cont_ids:
        return 0.0
    stu_lp = score_continuation(student_tokenizer, student_model, stu_prefix_ids, stu_cont_ids)
    tea_lp = score_continuation(teacher_tokenizer, teacher_model, tea_prefix_ids, tea_cont_ids)
    m = min(len(stu_lp), len(tea_lp))
    if m == 0:
        return 0.0
    return float(sum(tea_lp[i] - stu_lp[i] for i in range(m)) / m)


def sequence_score_mean_gap_text(student_tokenizer, teacher_tokenizer, student_model, teacher_model, prefix_text: str, continuation_text: str) -> float:
    """margin = mean teacher logprob - mean student logprob on a text continuation."""
    stu_prefix_ids = student_tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    stu_cont_ids = student_tokenizer(continuation_text, add_special_tokens=False)["input_ids"]
    tea_prefix_ids = teacher_tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    tea_cont_ids = teacher_tokenizer(continuation_text, add_special_tokens=False)["input_ids"]
    if not stu_cont_ids or not tea_cont_ids:
        return 0.0
    stu_lp = score_continuation(student_tokenizer, student_model, stu_prefix_ids, stu_cont_ids)
    tea_lp = score_continuation(teacher_tokenizer, teacher_model, tea_prefix_ids, tea_cont_ids)
    m = min(len(stu_lp), len(tea_lp))
    if m == 0:
        return 0.0
    return float(sum(tea_lp[i] - stu_lp[i] for i in range(m)) / m)


def wasserstein_1d(xs: List[float], ys: List[float]) -> float:
    if not xs or not ys:
        return 0.0
    xs, ys = sorted(xs), sorted(ys)
    n = min(len(xs), len(ys))
    if len(xs) != n:
        xs = [xs[int(i * len(xs) / n)] for i in range(n)]
    if len(ys) != n:
        ys = [ys[int(i * len(ys) / n)] for i in range(n)]
    return float(sum(abs(a - b) for a, b in zip(xs, ys)) / n)


def build_teacher_repair(student_tokenizer, teacher_tokenizer, teacher_model, prompt_ids, response_ids, selected_idx, student_lp, teacher_lp, top_k):
    prefix_ids_student = prefix_ids_from_prompt_and_response(prompt_ids, response_ids, selected_idx)
    prefix_text = student_tokenizer.decode(prefix_ids_student, skip_special_tokens=False)
    # Retokenize prefix for teacher before asking teacher for next-token distribution.
    teacher_prefix_ids = teacher_tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    top_tokens = teacher_top_next_tokens(teacher_tokenizer, teacher_model, teacher_prefix_ids, top_k=top_k)
    selected_token_id = response_ids[selected_idx]
    selected_token_text = student_tokenizer.decode([selected_token_id], skip_special_tokens=False)
    return TeacherRepair(
        selected_idx=selected_idx,
        selected_token_text=selected_token_text,
        selected_token_id=selected_token_id,
        prefix_text=prefix_text,
        student_chosen={
            "token_id": int(selected_token_id),
            "token_text": selected_token_text,
            "student_logprob": safe_float(student_lp[selected_idx]),
            "teacher_logprob_on_student_token": safe_float(teacher_lp[selected_idx]),
            "token_loss": safe_float(-student_lp[selected_idx]),
            "logprob_gap": safe_float(teacher_lp[selected_idx] - student_lp[selected_idx]),
        },
        teacher_top_tokens=top_tokens,
        explanation=(
            "Original OPD/KD does not literally edit the already generated answer in this viewer. "
            "Training would increase probability of teacher-preferred next tokens at this prefix and/or decrease the loss on the teacher target. "
            "The table below visualizes the teacher's next-token distribution as the repair signal."
        ),
    )


def run_baseline_opd_trace(prompt, prompt_text, student_tokenizer, student_model, teacher_tokenizer, teacher_model, prompt_ids, response_ids, response_text, top_k_candidates, teacher_top_k):
    student_lp = score_continuation(student_tokenizer, student_model, prompt_ids, response_ids)
    teacher_lp = score_continuation(teacher_tokenizer, teacher_model, teacher_tokenizer(prompt_text, add_special_tokens=False)["input_ids"], teacher_tokenizer(response_text, add_special_tokens=False)["input_ids"])

    # If tokenizers are the same, lengths align. If not, fall back to student self-loss for token visualization and approximate gap unavailable.
    if len(teacher_lp) != len(student_lp):
        teacher_lp = [None] * len(student_lp)
        gaps_for_selection = [-lp for lp in student_lp]
    else:
        gaps_for_selection = [teacher_lp[i] - student_lp[i] for i in range(len(student_lp))]

    n = min(len(student_lp), len(response_ids))
    response_ids = response_ids[:n]
    student_lp = student_lp[:n]
    teacher_lp = teacher_lp[:n]
    token_losses = [-x for x in student_lp]
    candidate_indices = sorted(range(n), key=lambda i: gaps_for_selection[i] if gaps_for_selection[i] is not None else token_losses[i], reverse=True)[:top_k_candidates]
    selected_index = candidate_indices[0] if candidate_indices else None
    notes = {selected_index: "original OPD selected: largest teacher-student gap"} if selected_index is not None else {}
    tokens = token_infos_from_lists(
        response_ids, decode_tokens(student_tokenizer, response_ids), student_lp, teacher_lp, token_losses,
        candidate_indices=candidate_indices, selected_indices=[selected_index] if selected_index is not None else [], notes_map=notes,
    )
    repair = None
    if selected_index is not None and all(x is not None for x in teacher_lp):
        repair = build_teacher_repair(student_tokenizer, teacher_tokenizer, teacher_model, prompt_ids, response_ids, selected_index, student_lp, teacher_lp, teacher_top_k)
    return MethodTrace(
        method_name="original_opd",
        prompt=prompt,
        prompt_text=prompt_text,
        response_text=response_text,
        response_token_ids=response_ids,
        tokens=tokens,
        selected_index=selected_index,
        candidate_indices=candidate_indices,
        selected_indices=[selected_index] if selected_index is not None else [],
        teacher_repair=repair,
        scalar_metrics={
            "mean_student_token_loss": float(sum(token_losses) / max(len(token_losses), 1)),
            "max_selection_score": float(max(gaps_for_selection)) if gaps_for_selection else None,
        },
        raw_debug={"rule": "select top teacher-student logprob gap token(s) on the student rollout"},
    )


def run_local_opd_trace(prompt, prompt_text, student_tokenizer, student_model, teacher_tokenizer, teacher_model, prompt_ids, response_ids, response_text, num_candidates, min_candidate_separation, rollback_steps, probe_len, local_len, teacher_num_samples, student_num_samples, teacher_probe_temp, teacher_sample_temp, student_sample_temp, top_p, threshold_alpha):
    student_lp = score_continuation(student_tokenizer, student_model, prompt_ids, response_ids)
    teacher_prompt_ids = teacher_tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    teacher_response_ids = teacher_tokenizer(response_text, add_special_tokens=False)["input_ids"]
    teacher_lp = score_continuation(teacher_tokenizer, teacher_model, teacher_prompt_ids, teacher_response_ids)
    if len(teacher_lp) != len(student_lp):
        teacher_lp = [None] * len(student_lp)

    n = min(len(student_lp), len(response_ids))
    response_ids = response_ids[:n]
    student_lp = student_lp[:n]
    teacher_lp = teacher_lp[:n]
    token_losses = [-x for x in student_lp]
    candidate_indices = select_topk_with_separation(token_losses, k=max(1, num_candidates), min_sep=max(1, min_candidate_separation))

    # First pass: for each candidate, roll back to a shared prefix.
    # Compare a student continuation with a teacher-generated continuation.
    raw_steps = []
    divergence_scores = []
    for cand in candidate_indices:
        rollback_start_idx = max(0, cand - rollback_steps)
        prefix_ids = prefix_ids_from_prompt_and_response(prompt_ids, response_ids, rollback_start_idx)
        prefix_text = student_tokenizer.decode(prefix_ids, skip_special_tokens=False)

        # Student probe: short suffix from the original student trajectory.
        student_suffix_ids = suffix_ids(response_ids, rollback_start_idx, probe_len)
        student_probe_text = student_tokenizer.decode(student_suffix_ids, skip_special_tokens=True)

        # Teacher probe: newly generated short continuation from the same rollback prefix.
        teacher_prefix_ids = teacher_tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        _, teacher_probe_texts = sample_local_continuations(
            teacher_tokenizer, teacher_model, teacher_prefix_ids,
            num_samples=1, max_new_tokens=probe_len, temperature=teacher_probe_temp, top_p=top_p
        )
        teacher_probe_text = teacher_probe_texts[0] if teacher_probe_texts else ""

        student_margin = sequence_score_mean_gap_text(
            student_tokenizer, teacher_tokenizer, student_model, teacher_model, prefix_text, student_probe_text
        )
        teacher_margin = sequence_score_mean_gap_text(
            student_tokenizer, teacher_tokenizer, student_model, teacher_model, prefix_text, teacher_probe_text
        )
        divergence_score = float(teacher_margin - student_margin)
        divergence_scores.append(divergence_score)
        raw_steps.append((cand, rollback_start_idx, prefix_ids, prefix_text, student_probe_text, teacher_probe_text, student_margin, teacher_margin, divergence_score))

    finite = [x for x in divergence_scores if math.isfinite(x)]
    if finite:
        mean_gap = sum(finite) / len(finite)
        std_gap = (sum((x - mean_gap) ** 2 for x in finite) / len(finite)) ** 0.5
        threshold = mean_gap + threshold_alpha * std_gap
    else:
        threshold = float("inf")

    candidate_steps = []
    selected = None
    selected_prefix_ids = None
    best_by_probe = None
    for cand, rollback_start_idx, prefix_ids, prefix_text, student_probe_text, teacher_probe_text, student_margin, teacher_margin, divergence_score in raw_steps:
        is_div = bool(math.isfinite(divergence_score) and divergence_score >= threshold)
        if best_by_probe is None or divergence_score > best_by_probe[1]:
            best_by_probe = (cand, divergence_score, prefix_ids)
        if selected is None and is_div:
            selected = cand
            selected_prefix_ids = prefix_ids
        candidate_steps.append(CandidateStep(
            candidate_idx=int(cand),
            token_text=student_tokenizer.decode([response_ids[cand]], skip_special_tokens=False),
            token_loss=float(token_losses[cand]),
            rollback_start_idx=int(rollback_start_idx),
            rollback_prefix_text=prefix_text,
            student_probe_continuation_text=student_probe_text,
            teacher_probe_continuation_text=teacher_probe_text,
            student_probe_margin_score=float(student_margin),
            teacher_probe_margin_score=float(teacher_margin),
            divergence_score=float(divergence_score),
            threshold=float(threshold),
            is_divergent=is_div,
            probe_continuation_text=student_probe_text,
            probe_gap=float(divergence_score),
        ))

    if selected is None and best_by_probe is not None:
        selected, _, selected_prefix_ids = best_by_probe
    if selected_prefix_ids is None:
        selected_prefix_ids = prompt_ids

    selected_prefix_text = student_tokenizer.decode(selected_prefix_ids, skip_special_tokens=False)
    teacher_prefix_ids_for_gen = teacher_tokenizer(selected_prefix_text, add_special_tokens=False)["input_ids"]
    teacher_ids, teacher_texts = sample_local_continuations(teacher_tokenizer, teacher_model, teacher_prefix_ids_for_gen, teacher_num_samples, local_len, teacher_sample_temp, top_p)
    student_ids, student_texts = sample_local_continuations(student_tokenizer, student_model, selected_prefix_ids, student_num_samples, local_len, student_sample_temp, top_p)

    teacher_samples = []
    for i, (ids, txt) in enumerate(zip(teacher_ids, teacher_texts)):
        score = sequence_score_mean_gap_text(student_tokenizer, teacher_tokenizer, student_model, teacher_model, selected_prefix_text, txt)
        teacher_samples.append(LocalSample("teacher", i, txt, ids, float(score)))
    student_samples = []
    for i, (ids, txt) in enumerate(zip(student_ids, student_texts)):
        score = sequence_score_mean_gap_text(student_tokenizer, teacher_tokenizer, student_model, teacher_model, selected_prefix_text, txt)
        student_samples.append(LocalSample("student", i, txt, ids, float(score)))

    teacher_samples_sorted = sorted(teacher_samples, key=lambda x: x.margin_score, reverse=True)
    student_samples_sorted = sorted(student_samples, key=lambda x: x.margin_score, reverse=True)
    local_distance = wasserstein_1d([s.margin_score for s in teacher_samples], [s.margin_score for s in student_samples])

    notes = {idx: "top-C high-loss candidate" for idx in candidate_indices}
    if selected is not None:
        notes[selected] = "selected local divergence point"
    tokens = token_infos_from_lists(
        response_ids, decode_tokens(student_tokenizer, response_ids), student_lp, teacher_lp, token_losses,
        candidate_indices=candidate_indices, selected_indices=[selected] if selected is not None else [], notes_map=notes,
    )

    return MethodTrace(
        method_name="local_opd",
        prompt=prompt,
        prompt_text=prompt_text,
        response_text=response_text,
        response_token_ids=response_ids,
        tokens=tokens,
        selected_index=selected,
        candidate_indices=candidate_indices,
        selected_indices=[selected] if selected is not None else [],
        divergence_index=selected,
        candidate_steps=candidate_steps,
        dynamic_threshold=float(threshold),
        rollback_prefix_token_ids=selected_prefix_ids,
        rollback_prefix_text=student_tokenizer.decode(selected_prefix_ids, skip_special_tokens=False),
        teacher_samples_sorted=teacher_samples_sorted,
        student_samples_sorted=student_samples_sorted,
        local_distance_type="1D Wasserstein distance over margin scores",
        local_distance_value=float(local_distance),
        scalar_metrics={
            "mean_student_token_loss": float(sum(token_losses) / max(len(token_losses), 1)),
            "dynamic_threshold": float(threshold),
            "local_distance": float(local_distance),
        },
        raw_debug={"rule": "top-C loss candidates -> rollback -> student probe vs teacher-generated probe -> dynamic threshold -> local sampling -> sorted margins -> 1D OT"},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--out", type=str, default="compare_trace.json")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0, help="Backward-compatible alias for student rollout temperature")
    parser.add_argument("--student_rollout_temp", type=float, default=None, help="Temperature for the original student rollout. Defaults to --temperature.")
    parser.add_argument("--teacher_probe_temp", type=float, default=0.7, help="Temperature for the single teacher probe continuation used in divergence detection.")
    parser.add_argument("--teacher_sample_temp", type=float, default=0.9, help="Temperature for teacher multi-sample continuations after divergence detection.")
    parser.add_argument("--student_sample_temp", type=float, default=0.9, help="Temperature for student multi-sample continuations after divergence detection.")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k_candidates", type=int, default=3)
    parser.add_argument("--teacher_top_k", type=int, default=8)
    parser.add_argument("--local_num_candidates", type=int, default=3)
    parser.add_argument("--local_min_candidate_sep", type=int, default=8)
    parser.add_argument("--local_rollback_steps", type=int, default=1)
    parser.add_argument("--local_probe_len", type=int, default=20)
    parser.add_argument("--local_cont_len", type=int, default=32)
    parser.add_argument("--local_teacher_samples", type=int, default=4)
    parser.add_argument("--local_student_samples", type=int, default=4)
    parser.add_argument("--threshold_alpha", type=float, default=0.5, help="dynamic threshold = mean(probe_gap) + alpha * std(probe_gap)")
    args = parser.parse_args()

    set_seed(args.seed)
    rollout_temp = args.temperature if args.student_rollout_temp is None else args.student_rollout_temp
    student_tokenizer, student_model = load_model_and_tokenizer(args.student_model, args.dtype)
    teacher_tokenizer, teacher_model = load_model_and_tokenizer(args.teacher_model, args.dtype)

    prompt_text = build_prompt_text(student_tokenizer, args.prompt, args.use_chat_template)
    prompt_ids, response_ids, response_text = generate_student_response(
        student_tokenizer, student_model, prompt_text, args.max_new_tokens, rollout_temp, args.top_p
    )
    prompt_ids_list = prompt_ids.tolist()

    baseline = run_baseline_opd_trace(
        args.prompt, prompt_text, student_tokenizer, student_model, teacher_tokenizer, teacher_model,
        prompt_ids_list, response_ids, response_text, args.top_k_candidates, args.teacher_top_k
    )
    local = run_local_opd_trace(
        args.prompt, prompt_text, student_tokenizer, student_model, teacher_tokenizer, teacher_model,
        prompt_ids_list, response_ids, response_text,
        args.local_num_candidates, args.local_min_candidate_sep, args.local_rollback_steps,
        args.local_probe_len, args.local_cont_len, args.local_teacher_samples, args.local_student_samples,
        args.teacher_probe_temp, args.teacher_sample_temp, args.student_sample_temp, args.top_p, args.threshold_alpha
    )

    trace = CompareTrace(
        prompt=args.prompt,
        created_at=time.time(),
        original_response_text=response_text,
        baseline=baseline,
        local_opd=local,
        meta={
            "schema_version": 3,
            "student_model": args.student_model,
            "teacher_model": args.teacher_model,
            "dtype": args.dtype,
            "use_chat_template": args.use_chat_template,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "student_rollout_temp": rollout_temp,
            "teacher_probe_temp": args.teacher_probe_temp,
            "teacher_sample_temp": args.teacher_sample_temp,
            "student_sample_temp": args.student_sample_temp,
            "top_p": args.top_p,
            "seed": args.seed,
            "threshold_formula": "mean(candidate_divergence_scores) + threshold_alpha * std(candidate_divergence_scores)",
            "divergence_score_formula": "teacher_probe_margin_score - student_probe_margin_score",
            "threshold_alpha": args.threshold_alpha,
        },
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(asdict(trace), f, ensure_ascii=False, indent=2)
    print(f"Saved trace to {args.out}")


if __name__ == "__main__":
    main()

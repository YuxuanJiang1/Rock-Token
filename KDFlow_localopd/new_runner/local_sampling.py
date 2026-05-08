# kdflow/opd_local/local_sampling.py
from __future__ import annotations

from typing import Dict, List


def sample_local_continuations(
    *,
    sample_fn,
    prefix_input_ids,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Dict]:
    """
    Generic wrapper around an existing generation function.

    sample_fn should return a list of dicts, each containing at least:
      {
        "token_ids": ...,
        "text": ...,
      }

    Important:
    - max_new_tokens should stay configurable.
    - later you may add stop tokens / eos handling / truncation logic.
    """
    outputs = sample_fn(
        prefix_input_ids=prefix_input_ids,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return outputs


def build_student_local_set(
    *,
    original_suffix_token_ids,
    newly_sampled_student,
) -> List[Dict]:
    """
    student set = new samples + original local suffix
    """
    student_set = list(newly_sampled_student)
    student_set.append(
        {
            "token_ids": original_suffix_token_ids,
            "source": "original_suffix",
        }
    )
    return student_set
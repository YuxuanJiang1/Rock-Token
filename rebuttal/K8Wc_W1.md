# Reviewer K8Wc — Response to W1 (and Q1)

## What K8Wc asked
> "The experimental scope is narrow: one teacher–student pair from the same family, and mostly
> mathematical reasoning data. Conduct more ablations on models will be useful."
> Q1: "The study uses one Qwen teacher–student pair. Could the author add more model ablations to
> test the robustness of results?"

(Margin note in the internal reviews doc, same weakness: *"do we have time to add more models,
specifically from different family like llama?"*)

## Two distinct axes, and why the table only covers one of them

There are two different things "test another model family" could mean:

1. **Different-family, same-tokenizer pair**: replace Qwen3-30B-A3B→Qwen3-4B with another
   same-family pair, e.g. Llama-3.1-8B→Llama-3.1-1B. Teacher and student still share one
   tokenizer/vocabulary within each run; only *which* family is being tested changes.
2. **Cross-family pair in the same run**: a teacher and student from *different* families in one
   distillation (e.g. Qwen3-30B-A3B teacher → Llama-3.2 student), requiring cross-tokenizer KD.
   This is the more literal reading of "a teacher–student pair from the same family" as the thing
   being complained about — the pair itself isn't same-family.

Table 1 (Qwen-Math / Qwen-Code / Llama-Math / Llama-Code) only covers axis 1. **We attempted axis
2 before submission and found it methodologically infeasible for this specific method** — not for
lack of engineering effort, but because the Rock Score is not well-defined across non-shared
vocabularies:

- Eq. (2)'s decomposition groups generated positions by the *realized student token id* `v`, and
  `ℓ_t` is the exact per-position reverse KL computed directly between teacher and student
  next-token distributions **in a shared vocabulary space**. Under same-tokenizer distillation this
  is exact (see the W1/W2 response for RNAB — `ℓ_t` needs no sampling, just the two conditional
  distributions at the same position).
- Under cross-tokenizer KD (`kdflow`'s `dskd` algorithm / `cross_tokenizer_kd` examples), teacher
  and student don't share a vocabulary. KDFlow handles this via a token-alignment step
  (`dskd_token_align`: `eta` = exact alignment, `cma` = cross-model-attention alignment) and/or a
  learned projector into the student's embedding space, *before* any KD loss is computed. So the
  quantity being measured is no longer the clean per-position KL of Eq. (1) — it's confounded with
  the alignment/projection's own approximation error, and that error is not uniform across token
  types. It is plausibly *worst* exactly for the tokens that define the Rock Token set: whitespace,
  indentation, LaTeX delimiters, and multi-digit numbers are segmented completely differently by
  Qwen's and Llama's BPE merges, so "the same structural token" often has no 1:1 counterpart to
  align to in the first place.
- This also breaks the cross-setting comparison method itself: Table 1's Jaccard column compares
  *sets of token ids* across settings. That comparison presupposes a stable notion of "the same
  token" existing in both vocabularies — which is precisely what cross-tokenizer alignment cannot
  guarantee, especially for structural tokens. Any cross-family Rock Token list we produced this
  way wouldn't be comparable to the same-tokenizer settings by the paper's own metric, and its
  differences from the Qwen list couldn't be cleanly attributed to "the phenomenon doesn't
  generalize" versus "the alignment step distorted the identification."

## Proposed rebuttal reply (drop-in text)

> We agree that testing beyond one same-family Qwen pair strengthens the robustness claim, and we
> have run the identical Rock Score pipeline on three additional settings: Qwen3 on a code domain,
> and a second model family (Llama-3.1-8B→Llama-3.1-1B) on both math and code (Table 1). [Fill in
> once results land: vocabulary concentration, output density, and cross-setting Jaccard overlap
> with the paper's Qwen3-Math setting.]
>
> We want to be transparent about scope, however: we also attempted a genuinely cross-family pair
> in a single run (Qwen3-30B-A3B teacher → Llama-3.2 student, via KDFlow's cross-tokenizer DSKD
> path) before submission, and found it is not a meaningful test *of this specific method*. The
> Rock Score (Eq. 2) requires grouping positions by a token id in a shared teacher/student
> vocabulary; cross-tokenizer KD instead aligns teacher and student token sequences through a
> learned projector/alignment step, so the measured per-position loss is confounded with alignment
> quality rather than reflecting student–teacher mismatch directly. That confound is plausibly
> largest exactly for the whitespace/formatting/digit tokens that constitute the Rock Token set,
> since these are segmented differently by the two tokenizers' BPE merges and often lack a 1:1
> counterpart to align to. We believe this is a genuine methodological boundary of the Rock Score
> definition rather than an engineering gap, and we will state it explicitly as a limitation (in
> addition to the same-family-generalization results in Table 1, which do not have this problem).

## Status
Explanation ready to use regardless of Table 1's results. Table 1 itself (Qwen-Code, Llama-Math,
Llama-Code rows) still needs the Section 2 pipeline (`rock_server.py` → `rerun_unrestricted.py` →
Jaccard/decode) rerun on those three new settings — separate task from what's written here.

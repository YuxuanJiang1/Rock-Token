# Experiment plan — all three reviewers

Every actionable item from `Rock Token Rebuttal.txt`, classified as:
- **[EXPERIMENT]** — needs new compute/training runs
- **[ANALYSIS]** — needs new code/processing, but reuses existing checkpoints/data, no new training
- **[WRITING]** — no new experiment, just clearer text in the paper/rebuttal

## 1. [EXPERIMENT] Model-family / model-pair diversity — tq1G W1, K8Wc W1 & Q1

> tq1G: "explore other base model configurations to determine whether the Rock token phenomenon
> extends beyond the current setup."
> K8Wc: "one teacher–student pair from the same family... Conduct more ablations on models will be
> useful." (internal comment: "do we have time to add more models, specifically from different
> family like llama?")

**Two reviewers, same ask** — highest leverage single item, but also the most expensive: everything
currently in the paper is Qwen3-30B-A3B → Qwen3-4B. A full second data point means standing up
another OPD pair (e.g. Llama) end-to-end: off-policy stage 1, on-policy stage 2, Section 2
identification, ideally Section 4 freeze intervention too. That's the whole pipeline again.

Given rebuttal timelines, worth scoping down rather than skipping:
- **Minimum credible answer**: run just the Section 2 identification pipeline (`rock_detection/`,
  what `run_regen_quad_l40s.sh` already automates) on a second, already-distilled model pair, to show
  Rock Tokens exist and cluster the same way (structural/discourse) elsewhere. Doesn't need a new
  OPD training run if a suitable off-the-shelf distilled pair exists.
- **Stronger answer**: also repeat the Section 4 freeze intervention (reusing
  `stumbling/build_ablation_freeze_lists.py` + `token_freeze_kd.py`, both already model-agnostic)
  on that second pair.
- Not scoped yet — need to pick a concrete second model pair before I can write launch scripts for
  this one.

## 2. [ANALYSIS] Decouple Rock Token selection from persistence evaluation — K8Wc W2 & Q2

> "Rock Tokens are selected because they have high loss at the final checkpoint and are then used to
> argue that high loss persists at the final checkpoint. In addition, early and late losses appear to
> be evaluated on each checkpoint's own on-policy trajectories, so changes in the context
> distribution are mixed with actual changes in student–teacher alignment. A cleaner analysis would
> select tokens using one split or an early pilot run and evaluate their persistence using
> independent, fixed contexts."

This is a real, well-specified methodological gap, distinct from RNAB's W1/W2 (those were about
notation; this is about experimental design). Current protocol (`compare_kl_evolution.py`):
select Rock Tokens by final-checkpoint loss, then compare early-vs-late KL where each checkpoint is
scored on *its own* fresh rollouts — so "persistence" is confounded with "did the context
distribution drift between checkpoints."

**Good news: no new training needed.** The early (`RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k`)
and late (`RockToken/qwen3-30b-a3b-to-4b-onpolicy-10k`) checkpoints already exist on the HF org — I
confirmed both resolve. The fix is a new *analysis* script, not new training:
1. Select Rock Tokens using **only** the early checkpoint (or a held-out pilot split) — not the
   final/late one.
2. Generate **one fixed context corpus** (e.g., from the early checkpoint, or a static held-out set).
3. Score **both** checkpoints' per-token KL against that same fixed corpus (teacher-forced scoring,
   not fresh generation from each checkpoint) — removes the context-drift confound.
4. Re-report the early-vs-late ΔKL persistence claim (the Fig. 3c/d equivalent) under this cleaner
   protocol.

This reuses `rock_detection/compute_logit_gradients.py`'s pattern (forward-only scoring against a
fixed set of already-generated sequences) more than `rock_server.py`'s (which re-generates per
checkpoint) — that's exactly the plumbing needed. I haven't written this script yet — flagging it as
ready to build if you want to proceed with this one next.

## 3. [WRITING] "What does a Rock Token represent mathematically, what does the intervention remove?" — K8Wc Q3

Same root cause as RNAB W1/W2 (already fixed in `rebuttal/RNAB_W1_W2.md`): once `ℓ_t` is stated
correctly as the exact per-position reverse KL, "what a Rock Token represents" falls out directly —
a vocabulary item ranked by its exact empirical contribution `Freq(v)·E[ℓ_t|x_t=v]` to the corpus-level
OPD loss. The second half ("what does the intervention remove") needs one plain-language sentence in
Sec. 3.2: the knockout sets that token's logit to `-∞` at *every* decoding step for the rest of
generation — a global ban, not a single-position edit. No new experiment; can reuse the RNAB W1/W2
draft plus one added sentence.

## 4. [EXPERIMENT] Additional freeze-ablation baselines — RNAB W4 & Questions

Already in progress. `stumbling/build_ablation_freeze_lists.py` + 5 launch scripts
(`run_stumb_top_freq.sh`, `run_stumb_top_meanloss.sh`, `run_stumb_gradmag.sh`,
`run_stumb_gradalign.sh`, `run_stumb_soft_lambda.sh`), adapted for your 4x L40S. Blocked on: raw
Section-2 artifacts (`rock_detection/run_regen_quad_l40s.sh`, also just built) → build the lists →
launch the 5 training runs. See `rebuttal/RUNBOOK.md`.

## 5. [WRITING, mostly] Knockout isolates a global effect, not a local one — RNAB W3

> "a global token ban does not isolate a token's causal contribution in any local sense... I
> recommend the authors further clarify this in the rebuttal."

Reviewer explicitly asks for clarification, not a rerun. Response: reframe the claim precisely —
"most Rock Tokens are Neutral under global banning" (true, supported) rather than implying
"negligible functional contribution to reasoning" (overclaimed). No experiment required to satisfy
this as written.
- **Optional stretch** (only if W4's runs finish with time to spare): a localized knockout variant
  (e.g., suppress the token only in a narrow context window rather than globally) would make the
  causal claim strictly stronger, but this is a "nice to have," not requested as mandatory.

## 6. [WRITING] Domain dependence should be stated earlier, not just in Limitations — RNAB W5

> "This is acknowledged in limitations, but text above should also state this clearly."

Literal ask is textual — surface the math-only / IFEval-only-in-knockout caveat in the main text
(e.g., end of Sec. 5 or in the Fig. 5 discussion), not only Appendix A. No new experiment required
to satisfy this as written.
- **Overlaps with #1**: if a second model-family run happens to also use non-math data (code /
  instruction-following), it would substantively strengthen this too — worth coordinating rather
  than treating as two separate experiments if #1 goes ahead.

## 7. [ANALYSIS, no new compute] Error bars on Fig. 5 — RNAB Questions

> "Fig. 5 shows no error bars despite the paper stating accuracy is averaged over five runs... the
> visual gap could plausibly be within noise."

Already flagged in `rebuttal/RNAB_W4.md`. If the 5 per-seed Pass@1 scores behind the existing
Original/Random/Rock-Freeze curves are still on disk, this is a replot, not a rerun.

---

## Priority read

| # | Item | Reviewer(s) | Type | Effort | Leverage |
|---|---|---|---|---|---|
| 1 | Second model family | tq1G, K8Wc | Experiment | High (full pipeline) | Highest — hits 2 reviewers |
| 4 | Freeze ablations | RNAB | Experiment | Medium — already scoped/scripted | High — RNAB is the low score, explicitly says they'll raise it |
| 2 | Fixed-context persistence re-analysis | K8Wc | Analysis | Medium — reuses existing checkpoints | Medium-high — closes a real methodological gap |
| 7 | Fig. 5 error bars | RNAB | Analysis | Low, if per-seed data exists | Medium — directly answers a concrete noise objection |
| 3, 5, 6 | Writing fixes | K8Wc, RNAB | Writing | Low | Necessary but not compute-blocked |

My read: #4 is already moving, keep it going. #2 is the best next thing to script — it's fully
specified, doesn't need new training, and closes a real gap. #1 is the expensive one and needs a
scoping decision (which second model? full pipeline or just Section 2?) before I can turn it into
scripts. #3/#5/#6 can be drafted any time, independent of compute.

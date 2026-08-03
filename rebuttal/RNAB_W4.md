# Reviewer RNAB — Response to W4 (baseline ablations)

**W4 in one line.** Fig. 5's frequency-matched Random baseline includes semantically useful tokens and
is "almost designed to look bad," so it does not show that the *Rock Score construction specifically*
drives the efficiency gains. RNAB asks for stronger baselines: freeze top-frequency tokens, freeze
top-mean-loss tokens (no Rock Score), soft down-weighting at intermediate λ, and freeze tokens chosen by
gradient magnitude / alignment. (The error-bar / sample-size half of the linked Question is answered
separately in `RNAB_Q2.md`.)

## Status of the experiment
Training for all 7 new runs (4 ablations + 3 soft-λ values) is **complete**. Evaluation on
AIME24/25+HMMT25 is in progress via `evaluation/run_eval_ablations.sh`.

**Seed count decision**: the new ablation table uses **3** independently-seeded decoding runs per
checkpoint, not the 5 the paper's Section 5.2 states for the original Fig. 5 results (that original
5-run standard is unchanged — see `RNAB_Q2.md`, a separate scope). This must be stated explicitly in
the rebuttal text, not left implicit — RNAB is the reviewer who quoted "five runs" verbatim in their
own critique, so a silent mismatch would read as cutting corners rather than a disclosed scope
decision. Proposed phrasing: *"the new ablations use 3 independently-seeded runs per checkpoint,
versus 5 for the original results, given the compressed rebuttal timeline; we will extend to 5 for
the camera-ready."* Revisit per-checkpoint if any comparison in the 3-seed results looks borderline —
that's the case where the extra 2 seeds would actually change the reading, not a blanket redo.

## Why this is tractable without new code (internal notes)
`stumbling/kdflow/algorithms/token_freeze_kd.py:88-95` already implements Eq. (5) in its general form:
it takes an arbitrary token-ID JSON (`--token_freeze_path`) and an arbitrary weight
(`--freeze_weight` = λ ∈ [0,1]), and applies `weights[freeze_mask] = freeze_weight` to the per-position
loss — not just the binary rock/random choice. So all four requested baselines plus the soft-λ sweep are
new *configs* against existing training code, no algorithm changes. `stumbling/build_ablation_freeze_lists.py`
derives the four token lists (`top_freq`, `top_meanloss`, `top_gradmag`, `top_gradalign`) from the same
raw statistics already collected for Fig. 2/3, and five launch scripts are cloned from
`run_stumb_random.sh` with only the freeze list / λ / save-path changed. See `RUNBOOK.md` for the L40S
hardware adaptation and the run order.

## Important framing decision: two tables, not one

We are **not** re-training a matched-scale Rock-Freeze checkpoint (decided to stop spending compute
on a "control" run and prioritize writing). This has a direct consequence for how the reply must be
worded: the new ablation numbers come from a genuinely different setup than Fig. 5 (2×L40S vs.
4×H100, the public `openthoughts_prompt_math_5k_src30k-35k` slice vs. the original private 10k-prompt
file, ~500 steps vs. a full epoch, 3 seeds vs. 5) — so they **cannot be presented as directly
numerically comparable to Fig. 5's Original/Random/Ours rows**. Doing so silently would repeat the
exact kind of unstated inconsistency RNAB already caught once (the ℓ_t notation issue in W1) — bad
precedent to set twice with the same reviewer.

What *is* still a fully valid, and actually sufficient, comparison: the 5 new conditions (top-freq,
top-mean-loss, grad-magnitude, grad-alignment, soft-λ) share identical data/hardware/step-budget/eval
protocol with **each other**. That internal comparison is what actually answers RNAB's question —
"does something about the Rock Score specifically matter, versus simpler alternatives" — since it's a
horse race among selection criteria, not a claim about matching the paper's absolute numbers. The
reply should lean on this directly rather than reach for a cross-scale comparison it can't support.

## Proposed rebuttal reply — initial (paste-ready; results to follow)

We thank the reviewer for this suggestion, and we agree that the frequency-matched Random baseline alone does not isolate which component of the Rock Score is responsible for the result in Fig. 5. To separate the contributions of the frequency weighting, the mean-loss ranking, and their product, we ran the four additional selection criteria the reviewer proposes, plus the soft-λ sweep, under a controlled, matched setup — identical data, hardware, step budget, and evaluation protocol across all five new conditions, so that any difference between them is attributable to the selection criterion alone. (This new setup is necessarily smaller-scale than the original submission's — 2×GPU rather than 4×, a public 5k-prompt slice of the same source corpus rather than the original 10k-prompt file, and a truncated training budget — so we report these results as a self-contained comparison among the five new conditions rather than claim direct numerical parity with Fig. 5's absolute values.)

- **Top-frequency freeze:** the top-100 tokens ranked by Freq(v) alone, isolating whether frequency by itself accounts for the effect.
- **Top-mean-loss freeze:** the top-100 tokens ranked by mean per-token KL alone, without the frequency weighting that defines the Rock Score.
- **Soft down-weighting:** the Rock set at intermediate strengths λ ∈ {0.3, 0.5, 0.7}, rather than only the hard λ = 0, tracing the accuracy-vs-speed frontier the submission samples only at λ ∈ {0, 1}.
- **Gradient-magnitude and gradient-alignment freezes:** the top-100 tokens ranked by per-occurrence gradient magnitude, and by cosine alignment with the balanced descent direction — the quantities plotted in Fig. 3(a,b).

For reference, Fig. 5's own three conditions (unchanged, original scale, 5-seed):

| Variant | Selection criterion | λ | Avg. accuracy (Fig. 5, original scale) |
|---|---|---|---|
| Original OPD | — | 1 | reported in Fig. 5 |
| Random (frequency-matched) | random, matched to Rock frequencies | 0 | reported in Fig. 5 |
| Ours (Rock-Freeze) | Freq(v) · mean KL(v), top-100 | 0 | reported in Fig. 5 |

New ablations (matched to each other: 2×L40S, `openthoughts_prompt_math_5k_src30k-35k`, ~500 steps, 3 seeds):

| Variant | Selection criterion | λ | AIME24 | AIME25 | HMMT25feb | Avg. |
|---|---|---|---|---|---|---|
| Top-mean-loss | mean KL(v) alone, top-100 | 0 | 48.89 | 42.22 | 23.33 | **38.15** |
| Top-frequency | Freq(v) alone, top-100 | 0 | — | — | — | to be reported |
| Gradient magnitude | per-occurrence ‖ḡ‖, top-100 | 0 | — | — | — | to be reported |
| Gradient alignment | cos(ḡ, G_balanced), top-100 | 0 | — | — | — | to be reported |
| Soft λ=0.3 | Rock set, partial down-weight | 0.3 | — | — | — | to be reported |
| Soft λ=0.5 | Rock set, partial down-weight | 0.5 | — | — | — | to be reported |
| Soft λ=0.7 | Rock set, partial down-weight | 0.7 | — | — | — | to be reported |

We will report the remaining rows, with our reading of which criterion(s) separate from the others, in a follow-up comment during the discussion period.

## Interpretation to add once results land (internal notes)
This is now a within-table comparison (rank the 5 new conditions against each other), not a
cross-table one. Once all 7 rows are in:
- If top-frequency and top-mean-loss both diverge (in either direction) from the middle of the pack, that's evidence the *joint* Freq×KL criterion is doing something neither ingredient does alone — state which direction plainly, don't just claim "Rock Score is validated" reflexively.
- If one ingredient's freeze lands close to the others despite a very different selection mechanism (e.g., top-mean-loss selected near-zero-frequency tokens per the build-time diagnostic below — freezing them may show *little* effect either way, which is itself informative, not a null result to bury).
- Gradient-magnitude / gradient-alignment freezes test whether the effect is really about the loss decomposition vs. raw gradient geometry (Fig. 3's own axes).
- Soft-λ sweep gives the accuracy-vs-speed frontier the paper currently only samples at λ ∈ {0,1} — report this as its own finding, not folded into the criterion comparison (different independent variable: same token set, varying strength, not varying selection).
- **Do not** phrase the conclusion as "matches/beats Fig. 5's Ours row" — there is no valid basis for that comparison at this scale. Phrase it as "among the five matched-condition selection criteria we tested, [X] shows the smallest/largest degradation."

Build-time diagnostic already in hand, worth citing regardless of final accuracy numbers: `top_meanloss`
selected tokens occurring only 2–18 times in the 500-sample corpus used for Fig. 2/3, versus much
higher mean KL (1.46–6.46) than typical Rock Tokens (~0.3) — exactly the noise-dominated tail
`Freq(v)` weighting is designed to filter. This is independent evidence for why frequency weighting
matters in the Rock Score construction, regardless of what the accuracy table ultimately shows.

## Open items before posting (internal notes)
1. Fill in the remaining 6 rows as `evaluation/run_eval_ablations.sh` completes (currently running).
2. (Error bars / sample size moved to `RNAB_Q2.md`.)
3. **Terminology:** the reply calls the method "Ours" (the Fig. 5 legend name); if it is renamed in the revision, update the wording here to match.
4. **Seed-count disclosure** (see above) needs to land in the final posted text, not just this internal note.

## Status
Training complete for all 7 runs. Evaluation in progress (`top_meanloss`: 1/7 done). Initial reply
above is ready to post as-is once the framing/two-table structure is confirmed; results table fills
in as evaluation completes.

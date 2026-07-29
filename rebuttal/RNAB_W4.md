# Reviewer RNAB — Response to W4 (baseline ablations)

**W4 in one line.** Fig. 5's frequency-matched Random baseline includes semantically useful tokens and
is "almost designed to look bad," so it does not show that the *Rock Score construction specifically*
drives the efficiency gains. RNAB asks for stronger baselines: freeze top-frequency tokens, freeze
top-mean-loss tokens (no Rock Score), soft down-weighting at intermediate λ, and freeze tokens chosen by
gradient magnitude / alignment. (The error-bar / sample-size half of the linked Question is answered
separately in `RNAB_Q2.md`.)

## Status of the experiment
The runs are **prepared but not yet complete**. This response is therefore an *initial* reply that
commits to the ablations and explains why they are cheap to run; the numbers go in a follow-up comment
during the discussion period. Full run instructions are in `rebuttal/RUNBOOK.md`.

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

## Proposed rebuttal reply — initial (paste-ready; results to follow)

We thank the reviewer for this suggestion, and we agree that the frequency-matched Random baseline alone does not isolate which component of the Rock Score is responsible for the result in Fig. 5. To separate the contributions of the frequency weighting, the mean-loss ranking, and their product, we are running the additional ablations the reviewer proposes and will report the numbers in a follow-up comment during the discussion period.

These comparisons require no change to our method. The reweighting in Eq. (5) accepts an arbitrary set of token IDs and an arbitrary weight λ ∈ [0,1], not only the binary Rock/Random choice used in the submission, so each variant is a configuration run at the same scale and evaluation protocol as Fig. 5. We prioritize the three comparisons the reviewer identifies as most important, and also include the two gradient-based selectors suggested in the weakness:

- **Top-frequency freeze:** the top-100 tokens ranked by Freq(v) alone, isolating whether frequency by itself accounts for the effect.
- **Top-mean-loss freeze:** the top-100 tokens ranked by mean per-token KL alone, without the frequency weighting that defines the Rock Score.
- **Soft down-weighting:** the Rock set at intermediate strengths λ ∈ {0.3, 0.5, 0.7}, rather than only the hard λ = 0, tracing the accuracy-vs-speed frontier the submission samples only at λ ∈ {0, 1}.
- **Gradient-magnitude and gradient-alignment freezes:** the top-100 tokens ranked by per-occurrence gradient magnitude, and by cosine alignment with the balanced descent direction — the quantities plotted in Fig. 3(a,b).

We will populate this table in the follow-up:

| Variant | Selection criterion | λ | Avg. accuracy (AIME24/25 + HMMT25, final ckpt) |
|---|---|---|---|
| Original OPD | — | 1 | reported in Fig. 5 |
| Random (frequency-matched) | random, matched to Rock frequencies | 0 | reported in Fig. 5 |
| Ours | Freq(v) · mean KL(v), top-100 | 0 | reported in Fig. 5 |
| Top-frequency | Freq(v) alone, top-100 | 0 | to be reported |
| Top-mean-loss | mean KL(v) alone, top-100 | 0 | to be reported |
| Ours, soft λ | Rock set, partial down-weight | 0.3 / 0.5 / 0.7 | to be reported |
| Gradient magnitude | per-occurrence ‖ḡ‖, top-100 | 0 | to be reported |
| Gradient alignment | cos(ḡ, G_balanced), top-100 | 0 | to be reported |

We will report these numbers, with our reading of which component of the Rock Score drives the effect, in a follow-up comment during the discussion period.

## Interpretation to add once results land (internal notes)
Fill the table, then state the honest reading of whichever pattern appears:
- If top-frequency and top-mean-loss each underperform Ours at a matched speedup → the *joint* Freq×KL criterion does real work beyond either signal alone.
- If one ingredient (likely top-frequency, given rocks are high-frequency) recovers most of the effect → say so plainly; it narrows, not refutes, the claim (the useful selector is cheap frequency, and the Rock Score is a principled superset).
- Gradient-magnitude / gradient-alignment freezes test whether the effect is really about the loss decomposition vs. raw gradient geometry.
- Soft-λ sweep gives the accuracy-vs-speed frontier the paper currently only samples at λ ∈ {0,1}.

## Open items before posting (internal notes)
1. Locate/regenerate the raw occurrence + gradient artifacts, build the four lists, launch the runs, evaluate on AIME24/25+HMMT25 (see `RUNBOOK.md`).
2. (Error bars / sample size moved to `RNAB_Q2.md`.)
3. **Terminology:** the reply calls the method "Ours" (the Fig. 5 legend name); if it is renamed in the revision, update the wording here to match.

## Status
Scripts ready; runs not yet complete. Initial reply above is ready to post as-is; the results table and interpretation paragraph follow in an update.

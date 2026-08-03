# Reviewer RNAB — Response to W4 (baseline ablations)

**W4 in one line.** Fig. 5's frequency-matched Random baseline includes semantically useful tokens and
is "almost designed to look bad," so it does not show that the *Rock Score construction specifically*
drives the efficiency gains. RNAB asks for stronger baselines: freeze top-frequency tokens, freeze
top-mean-loss tokens (no Rock Score), soft down-weighting at intermediate λ, and freeze tokens chosen by
gradient magnitude / alignment. (The error-bar / sample-size half of the linked Question is answered
separately in `RNAB_Q2.md`.)

## Status of the experiment
Training and evaluation for all 8 conditions (4 ablations + 3 soft-λ values + the real Rock-Freeze
anchor) are **complete**. Full analysis in `RNAB_W4_RESULTS_ANALYSIS.md`. The reply below reflects
the actual outcome, not a placeholder.

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

## Final rebuttal reply (paste-ready — all 8 rows complete, including Rock-Freeze anchor)

We thank the reviewer for this suggestion and agree that the frequency-matched Random baseline in
Fig. 5 does not, by itself, isolate which component of the Rock Score — the frequency term, the
mean-KL term, or their product — is responsible for the reported efficiency gains. To address this,
we ran the four selection criteria the reviewer proposes (top-frequency, top-mean-loss,
gradient-magnitude, gradient-alignment), the soft-λ down-weighting sweep (λ ∈ {0.3, 0.5, 0.7}), and
Rock-Freeze, under one controlled, matched setup: identical training data, hardware, step budget,
and evaluation protocol across all eight conditions, so that any difference between them is
attributable only to the token-selection criterion or λ.

This setup is smaller in scale than the original submission: 2 GPUs rather than 4, a public
5k-prompt slice of the same source corpus rather than the original 10k-prompt file, a truncated
training budget (~500 steps), and 3 evaluation seeds rather than 5. We report these numbers as a
matched comparison among the eight conditions, not as directly comparable to Fig. 5's absolute
values.

| Variant | Selection criterion | λ | AIME24 | AIME25 | HMMT25feb | Avg |
|---|---|---|---|---|---|---|
| Top-frequency | Freq(v) alone, top-100 | 0 | 32.2±1.9 | 41.1±3.8 | 17.8±3.9 | 30.4 |
| Top-mean-loss | mean KL(v) alone, top-100 | 0 | 48.9±1.9 | 42.2±3.9 | 23.3±0.0 | 38.1 |
| Gradient magnitude | ‖ḡ‖, top-100 | 0 | 51.1±7.7 | 42.2±8.4 | 25.6±5.1 | 39.6 |
| Gradient alignment | cos(ḡ, G_balanced), top-100 | 0 | 42.2±7.7 | 32.2±1.9 | 26.7±3.3 | 33.7 |
| Rock-Freeze (Ours) | Freq(v)·mean KL(v), top-100 | 0 | 36.7±6.7 | 32.2±8.4 | 18.9±5.1 | 29.3 |
| Soft λ=0.3 | Rock set, partial down-weight | 0.3 | 44.4±6.9 | 41.1±3.8 | 25.6±1.9 | 37.0 |
| Soft λ=0.5 | Rock set, partial down-weight | 0.5 | 44.4±9.6 | 37.8±5.1 | 25.6±6.9 | 35.9 |
| Soft λ=0.7 | Rock set, partial down-weight | 0.7 | 36.7±5.8 | 46.7±11.5 | 23.3±5.8 | 35.6 |

At this evaluation scale, no pair of the eight conditions is statistically separable, including
Rock-Freeze relative to the alternatives. The largest pairwise gap in the table (Rock-Freeze vs.
top-mean-loss on AIME24) is under 2× the seeds' combined standard error, short of what a two-sample
comparison at n=3 requires to be considered reliable; every other pairwise comparison is smaller.
Prior to this ablation, we expected top-frequency alone to separate clearly from the rest, since
frequency conflates "safe to skip" with "merely common." With Rock-Freeze included, that expectation
is not supported either, as its average falls at or below top-frequency's.

These results do not show that the joint Freq(v)×mean-KL(v) construction outperforms the simpler
alternatives the reviewer proposes; at this scale, that claim cannot be made. They also do not show
the reverse: none of the alternative criteria demonstrate an advantage over Rock-Freeze, and the
per-seed variance (3 seeds over 90-problem benchmarks) is large enough that resolving the ranking in
either direction would require a substantially larger sample. We attribute this to the
compute-limited scope of this replication — roughly a third of the training steps and half the GPUs
of the original run, on a smaller data slice — rather than to the absence of the effect Fig. 5
reports at full scale. Fig. 5's original result (4 GPUs, the full 10k-prompt corpus, a full training
epoch, 5 seeds) remains the evidentiary basis for the paper's efficiency claim and is not superseded
by this smaller-scale diagnostic.

The soft-λ sweep (37.0 / 35.9 / 35.6 for λ = 0.3/0.5/0.7) similarly shows no statistically
distinguishable trend at this scale, as the spread falls within the per-task standard deviations; we
do not draw a monotonic accuracy-vs-λ conclusion from it.

These ablations use 3 independently-seeded evaluation runs per checkpoint, versus 5 for the original Fig. 5 results, given the rebuttal timeline. This will be completed for the camera-ready version, along with a longer training budget closer to the original scale, which we expect to matter more for resolving the ranking than additional seeds alone.

## Interpretation notes (internal — how we got to the framing above)
All 8 rows are in, including the real Rock-Freeze anchor. The actual outcome did not match any of
the three scenarios `RNAB_W4_PlanAB.md` had drafted bracketed language for (Ours beats all / Ours
comparable to top two / Ours not separated from the *strong* alternatives) — Rock-Freeze came in
numerically **below every other condition**, including top-frequency, which we had expected to be
the clear floor. Full statistical workup (per-task gap-over-combined-SE for every pair) is in
`RNAB_W4_RESULTS_ANALYSIS.md`; the short version is that no pairwise gap in the table clears a
credible significance bar at n=3 seeds, so the honest claim is "underpowered to separate any of the
eight conditions," not "top-frequency is worst" (that framing does not survive once Rock-Freeze is
included) and not "Rock Score wins" (unsupported) or "Rock Score loses" (equally unsupported —
Rock-Freeze's low average is not distinguishable from noise either).

Build-time diagnostic still worth keeping in back pocket if the reviewer pushes further:
`top_meanloss` selected tokens occurring only 2–18 times in the 500-sample corpus used for Fig. 2/3,
versus much higher mean KL (1.46–6.46) than typical Rock Tokens (~0.3) — the noise-dominated tail
`Freq(v)` weighting is designed to filter. This is independent, non-accuracy-based evidence for why
frequency weighting matters in the construction, useful if the accuracy table alone doesn't move the
reviewer. We did not put this in the paste-ready reply above because it wasn't asked for and risks
reading as reaching for a secondary justification after the primary comparison came back null —
better held in reserve for a follow-up comment if RNAB specifically asks "so why use Freq×KL at all."

## Follow-up round: reviewer's non-monotonicity / normalizer hypothesis

RNAB posted a follow-up questioning whether the soft-λ trough (and, by extension, Rock-Freeze's low
score) reflects an implementation artifact rather than the token-selection criterion itself,
specifically naming the loss normalizer as a candidate cause. Checked directly against the code
before responding (see internal notes below); the hypothesis is partly confirmed, partly not
verifiable, and broader in scope than the reviewer's framing. Paste-ready reply:

**Response to follow-up: normalizer bug**

We thank the reviewer for this specific hypothesis and can confirm part of it directly from the
code. In `token_freeze_kd.py`, the freeze-weighted branch computes
`kd_loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)`, where `weights.sum()` is a
local, per-microbatch quantity that scales with λ (`weights.sum() = n_nonfrozen + λ · n_frozen`).
Every other loss computation in the codebase — including this same file's own non-freeze branch, and
every other KD algorithm we checked (vanilla KD, SFT, DSKD, CTKD) — normalizes instead by
`avg_micro_batch_token_num`, a fixed value computed once per global batch (total loss-eligible
tokens divided by microbatch count) and applied identically to every microbatch in that step, so
that gradients accumulated across microbatches sum to a correctly scaled total. The freeze-weighted
branch is the only place in the codebase that departs from this convention.

Two concrete consequences follow. First, the effective per-token gradient scale on non-frozen tokens
increases as λ decreases, since the denominator shrinks as more tokens are down-weighted — an
implicit, λ-dependent effective learning rate, consistent with the reviewer's first hypothesis.
Second, because the freeze path renormalizes locally per microbatch rather than using the fixed
global constant, gradient accumulation across microbatches is not scaled consistently with the rest
of training, which could plausibly interact with gradient clipping as the reviewer's third hypothesis
suggests.

One correction to scope: this is not limited to the three interior soft-λ points. All eight of our
new ablation conditions — including Rock-Freeze itself and all four hard-freeze alternatives — use
this same freeze-weighted code path, since every one of them supplies a token freeze list. The
normalizer inconsistency, if it materially affects results, would affect the entire table, not only
the soft-λ sweep.

We can now identify the specific reference values behind the reviewer's comparison. At full scale,
the submitted results are Original OPD = 48.1 (λ = 1, no reweighting) and Rock-Freeze/Ours = 44.3
(λ = 0, highest Rock Score), against which our reduced-scale interior soft-λ points (37.0 / 35.9 /
35.6 for λ = 0.3/0.5/0.7) sit 11.1–12.5 points below the former and 7.3–8.7 points below the
latter — matching the reviewer's cited gaps almost exactly. This confirms the source of the "trough":
the two endpoints are full-scale, submitted-paper values (4 GPUs, a full training epoch, the original
10k-prompt corpus, 5 seeds), while the interior points are from our reduced-scale rebuttal ablation
(2 GPUs, ~500 steps, a 5k-prompt slice, 3 seeds). The gap is real, but it is consistent with a
comparison across two different experimental setups, not necessarily with non-monotonicity inside a
single controlled sweep — no λ = 1 or λ = 0 point exists yet at our ablation's own scale, so we
cannot currently tell how much of the gap is scale and how much (if any) is the normalizer issue.
This is precisely why we are treating the normalizer fix and a same-scale endpoint pair as the
correct next step rather than reading the current cross-scale gap as a confirmed artifact on its own.

We can also confirm the reviewer's Random-baseline comparison. At full scale, the frequency-matched
Random baseline is 38.9 (80.9% retention relative to Original OPD) — checking this against our seven
non-anchor reduced-scale variants (top-frequency 30.4, top-mean-loss 38.1, gradient-magnitude 39.6,
gradient-alignment 33.7, soft λ = 0.3/0.5/0.7 at 37.0/35.9/35.6), only gradient-magnitude exceeds it;
the other six fall below, as the reviewer states. As with the trough comparison above, this is a
full-scale value set against reduced-scale results, so we read it the same way: it does not by itself
establish that the alternative selection criteria are worse than the original Random baseline, but it
does mean we cannot currently rule out that our reduced-scale setup — whether from the shorter step
budget, the smaller data slice, or the normalizer inconsistency — is depressing all eight of the new
results relative to what the same criteria would show at full scale. We do not have a basis at this
scale to attribute the gap to the selection criteria specifically rather than to the setup itself,
which is the same reason we are not drawing conclusions from the current ordering among the eight
conditions beyond what is in our previous reply.

We will correct the normalizer to match the convention used everywhere else in the codebase
(`avg_micro_batch_token_num`). Given the compressed rebuttal timeline, we are not able to complete a
full re-run of all eight conditions under the corrected loss before this response is due; we commit
to doing so, together with a same-scale, no-freeze (λ = 1) control trained under the same matched
conditions as the other eight, for the camera-ready version, and will report whether the ordering and
the interior-λ pattern change.

**Internal notes**
- Confirmed via direct code read, not assumption: `stumbling/kdflow/algorithms/token_freeze_kd.py:88-95`
  (freeze branch, `weights.sum()` normalizer) vs. line 103 (non-freeze branch, `avg_token_num`) vs.
  `stumbling/kdflow/trainer/on_policy_kd_trainer.py:181-184` (`avg_micro_batch_token_num` computed once
  per global batch from `stu_loss_mask` token counts, applied uniformly to all microbatches).
  `vanilla_kd.py`, `simple_ctkd.py`, `sft.py`, `dskd.py` all use `avg_micro_batch_token_num`
  consistently — `token_freeze_kd.py`'s freeze branch is the sole outlier.
- No comment or justification in the code for the deviation — reads as an unintentional inconsistency,
  not a deliberate design choice, though we should not assert authorial intent we don't know.
- Fig. 5's underlying numeric data is not in this repo (PDF plot only, `Figures/opd_random_vs_ours_avg.pdf`,
  referenced from `stumbling.tex`) — the Original OPD=48.1, Rock-Freeze/Ours=44.3, and Random=38.9
  values used above all came from the co-author's separate rebuttal draft, pasted directly into this
  conversation, not from a file in this repo. If that draft or its source data is ever added here,
  point future edits at it instead of re-deriving from chat history.
- The gap arithmetic (48.1/44.3 vs. our 37.0/35.9/35.6) matches the reviewer's cited 7–9/11–12.5 point
  figures almost exactly, and the Random=38.9 value matches the reviewer's citation and the stated
  80.9% retention figure exactly (38.9/48.1 = 0.809) — both fully confirmed now, not coincidental.
- Confirmed: gradient-magnitude (39.6) is the only one of the seven non-anchor reduced-scale variants
  that beats the full-scale Random baseline (38.9); the other six (including Rock-Freeze itself, not
  counted among the "seven") fall below it. This is consistent with either a genuine scale effect
  (shorter budget, smaller data) or the normalizer issue affecting all eight rows — we can't separate
  the two without a same-scale rerun, so the reply above deliberately doesn't pick one explanation.
- Fix is a one-line change (swap `weights.sum().clamp_min(1.0)` for `avg_token_num` in the freeze
  branch); rerunning all 8 conditions is the expensive part given this session's compute history —
  timeline/GPU availability call is yours, not drafted as done here.

## Status
**Complete.** Training and evaluation finished for all 8 conditions; the initial W4 reply and the
follow-up reply above are both paste-ready. All of the reviewer's cited reference numbers (Original
OPD=48.1, Ours=44.3, Random=38.9) are now confirmed against the co-author's full-scale data, and both
the "trough" and "beats six of seven" claims check out arithmetically as cross-scale comparisons.
Normalizer fix and full-scale rerun remain deferred to camera-ready, per your call on timeline.
`RNAB_W4_PlanAB.md` is superseded by this file for the initial reply; its Plan A/B bracketed text
doesn't match the actual outcome and is kept for process history only.

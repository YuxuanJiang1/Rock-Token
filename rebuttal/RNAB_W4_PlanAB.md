# Reviewer RNAB — W4 final response: Plan A / Plan B

Two ready-to-paste versions of the W4 reply, differing only in whether the matched-condition
Rock-Freeze ("Ours") checkpoint (`run_stumb_rock_freeze.sh`, currently training) finishes and gets
evaluated before the response is due. Use whichever applies — no last-minute editing needed.

**Both plans report all 6 (7) rows honestly, including the one that complicates the story**
(`top_gradmag` scoring highest despite zero overlap with the real Rock-Token set). See the
conversation notes for why: RNAB named gradient-magnitude/alignment explicitly as one of the
requested comparisons, so omitting it because it's inconvenient would be selective reporting of a
result a reviewer specifically asked to see — worse for credibility than an honest, nuanced finding,
especially with this reviewer, who already caught one unstated inconsistency (the ℓ_t notation in W1).

## Data in hand (all seeds=3, matched conditions: 2×L40S, `openthoughts_prompt_math_5k_src30k-35k`, ~500 steps)

| Variant | Selection criterion | λ | AIME24 | AIME25 | HMMT25feb | Avg |
|---|---|---|---|---|---|---|
| Top-frequency | Freq(v) alone, top-100 | 0 | 32.2±1.9 | 41.1±3.8 | 17.8±3.9 | 30.4 |
| Top-mean-loss | mean KL(v) alone, top-100 | 0 | 48.9±1.9 | 42.2±3.9 | 23.3±0.0 | 38.1 |
| Gradient magnitude | ‖ḡ‖, top-100 | 0 | 51.1±7.7 | 42.2±8.4 | 25.6±5.1 | 39.6 |
| Gradient alignment | cos(ḡ, G_balanced), top-100 | 0 | 42.2±7.7 | 32.2±1.9 | 26.7±3.3 | 33.7 |
| Soft λ=0.3 | Rock set, partial down-weight | 0.3 | 44.4±6.9 | 41.1±3.8 | 25.6±1.9 | 37.0 |
| Soft λ=0.5 | Rock set, partial down-weight | 0.5 | 44.4±9.6 | 37.8±5.1 | 25.6±6.9 | 35.9 |
| Soft λ=0.7 | Rock set, partial down-weight | 0.7 | 36.7±5.8 | 46.7±11.5 | 23.3±5.8 | 35.6 |
| Rock-Freeze ("Ours") | Freq(v)·mean KL(v), top-100 | 0 | *pending* | *pending* | *pending* | *pending* |

**What's statistically credible vs. not**, for calibrating the prose in both plans:
- Top-frequency's gap below everything else is credible, most clearly on AIME24 (32.2±1.9 vs.
  38–51 for the others — a ~16-point gap against ~2-point stds on the comparison to top-mean-loss).
- Top-mean-loss / gradient-magnitude / soft-λ=0.3 (38.1 / 39.6 / 37.0) are within noise of each
  other — don't rank them against each other with confidence.
- The soft-λ "trend" (37.0 → 35.9 → 35.6) is not distinguishable from noise (1.4-point spread
  against 2–11-point per-task stds). Do not claim a monotonic accuracy-vs-λ relationship from this.

---

## PLAN A — if Rock-Freeze finishes and is evaluated in time

> We thank the reviewer for this suggestion, and we agree that the frequency-matched Random
> baseline alone does not isolate which component of the Rock Score is responsible for the result in
> Fig. 5. We ran the four requested selection criteria plus the soft-λ sweep, and the actual
> Rock-Freeze method, under a controlled, matched setup: identical data, hardware, step budget, and
> evaluation protocol across all conditions, so any difference between them is attributable to the
> selection criterion or λ alone. This setup is necessarily smaller-scale than the original
> submission — 2 GPUs rather than 4, a public 5k-prompt slice of the same source corpus rather than
> the original 10k-prompt file, a truncated training budget, and 3 evaluation seeds rather than 5 —
> so we report it as a self-contained, matched comparison rather than claim direct numerical parity
> with Fig. 5's absolute values.
>
> | Variant | Selection criterion | λ | Avg. accuracy (AIME24/25 + HMMT25) |
> |---|---|---|---|
> | Top-frequency | Freq(v) alone, top-100 | 0 | 30.4 |
> | Top-mean-loss | mean KL(v) alone, top-100 | 0 | 38.1 |
> | Gradient magnitude | per-occurrence ‖ḡ‖, top-100 | 0 | 39.6 |
> | Gradient alignment | cos(ḡ, G_balanced), top-100 | 0 | 33.7 |
> | Rock-Freeze (Ours) | Freq(v)·mean KL(v), top-100 | 0 | **[FILL IN]** |
> | Soft λ=0.3 / 0.5 / 0.7 | Rock set, partial down-weight | 0.3 / 0.5 / 0.7 | 37.0 / 35.9 / 35.6 |
>
> Two findings we read as robust. First, **top-frequency selection is the clear worst performer**,
> most credibly on AIME24 (32.2±1.9 vs. 38–51 for every other criterion) — this weighs against the
> simplest alternative explanation, that Rock Tokens are safe to freeze merely because they are
> frequent. Second, [**IF Ours ≥ all four alternatives**: Rock-Freeze matches or exceeds every
> single-signal alternative we tested, which is direct evidence that the *joint* Freq×KL
> construction is doing real work beyond any one ingredient.] [**IF Ours is comparable to
> top-mean-loss/gradient-magnitude but clearly above top-frequency and top-gradalign**: Rock-Freeze
> performs on par with the two next-best alternatives (top-mean-loss, gradient-magnitude) and clearly
> above top-frequency and gradient-alignment; we read this as evidence that frequency alone is
> insufficient and that the Rock Score's loss component carries most of the signal, while noting we
> cannot rule out that other loss- or gradient-derived criteria besides the specific joint
> construction would perform comparably.] [**IF Ours is not clearly separated from the pack**: we
> did not find a significant separation between Rock-Freeze and the strongest alternatives
> (top-mean-loss, gradient-magnitude) at this evaluation scale (90 problems, 3 seeds); we report this
> plainly rather than overclaim, and note that several criteria beyond the specific Rock Score
> construction appear to identify comparably safe-to-freeze tokens — narrowing, not refuting, the
> practical value of the method, since the Rock Score remains a principled and reproducible way to
> arrive at such a set.]
>
> We are transparent that a third finding complicates a fully clean story: **gradient-magnitude
> selection performs comparably to or better than top-mean-loss**, despite zero overlap with the
> actual Rock-Token set and despite selecting tokens from the opposite end of the gradient-magnitude
> axis from real Rock Tokens (Fig. 3a shows Rock Tokens have small per-occurrence gradients; this
> criterion selects the largest). We do not think this undermines the practical contribution — Rock
> Score remains a specific, reproducible, and well-motivated way to identify a safe-to-freeze token
> set — but it does mean the data does not, by itself, establish that the *joint* construction is the
> unique way to find such a set, only that pure frequency is a worse way to do so.
>
> The soft-λ sweep (37.0 / 35.9 / 35.6 for λ = 0.3/0.5/0.7) does not show a statistically
> distinguishable trend at this evaluation scale — the spread is within the per-task standard
> deviations — so we do not draw a monotonic accuracy-vs-λ conclusion from it.
>
> We used 3 independently-seeded evaluation runs per checkpoint for these new ablations, rather than
> the 5 used for the original Fig. 5 results, given the compressed rebuttal timeline; we will extend
> to 5 seeds for the camera-ready version.

---

## PLAN B — if Rock-Freeze does not finish in time

> We thank the reviewer for this suggestion, and we agree that the frequency-matched Random
> baseline alone does not isolate which component of the Rock Score is responsible for the result in
> Fig. 5. We ran the four requested selection criteria plus the soft-λ sweep under a controlled,
> matched setup: identical data, hardware, step budget, and evaluation protocol across all five new
> conditions, so any difference between them is attributable to the selection criterion or λ alone.
> This setup is necessarily smaller-scale than the original submission — 2 GPUs rather than 4, a
> public 5k-prompt slice of the same source corpus rather than the original 10k-prompt file, a
> truncated training budget, and 3 evaluation seeds rather than 5 — so we report it as a
> self-contained comparison among the new conditions rather than claim direct numerical parity with
> Fig. 5's absolute values. **We were not able to complete a matched-scale Rock-Freeze run in the
> response window; we are running it now and will share the result in a follow-up comment if it
> completes during the discussion period, and in the camera-ready otherwise.**
>
> | Variant | Selection criterion | λ | Avg. accuracy (AIME24/25 + HMMT25) |
> |---|---|---|---|
> | Top-frequency | Freq(v) alone, top-100 | 0 | 30.4 |
> | Top-mean-loss | mean KL(v) alone, top-100 | 0 | 38.1 |
> | Gradient magnitude | per-occurrence ‖ḡ‖, top-100 | 0 | 39.6 |
> | Gradient alignment | cos(ḡ, G_balanced), top-100 | 0 | 33.7 |
> | Soft λ=0.3 / 0.5 / 0.7 | Rock set, partial down-weight | 0.3 / 0.5 / 0.7 | 37.0 / 35.9 / 35.6 |
>
> One finding we read as robust even without the Rock-Freeze anchor: **top-frequency selection is
> the clear worst performer among the five criteria**, most credibly on AIME24 (32.2±1.9 vs. 38–51
> for every other criterion — a gap well outside the observed noise). This weighs directly against
> the simplest alternative explanation a reader might reach for — that Rock Tokens are safe to freeze
> merely because they are frequent, independent of their loss behavior.
>
> We are transparent that the remaining comparison is more nuanced than a single dominant ingredient.
> Top-mean-loss performs close to what a build-time diagnostic already predicts: it selects tokens
> occurring only 2–18 times in the 500-sample analysis corpus (vs. typical Rock-Token frequencies),
> so freezing it should — and does — leave accuracy close to the top of the range, since these tokens
> carry little aggregate training signal to begin with. Gradient-magnitude selection performs
> comparably or slightly better still, despite zero overlap with the actual Rock-Token set and despite
> selecting from the opposite end of the gradient-magnitude axis from real Rock Tokens (Fig. 3a).
> We do not think this undermines the practical contribution of the Rock Score as a specific,
> reproducible criterion, but we cannot yet claim from this data alone that the *joint* Freq×KL
> construction is the unique way to identify a safe-to-freeze set — only that pure frequency is
> demonstrably a worse one. The pending Rock-Freeze run is intended to close exactly this gap.
>
> The soft-λ sweep (37.0 / 35.9 / 35.6 for λ = 0.3/0.5/0.7) does not show a statistically
> distinguishable trend at this evaluation scale — the spread is within the per-task standard
> deviations — so we do not draw a monotonic accuracy-vs-λ conclusion from it.
>
> We used 3 independently-seeded evaluation runs per checkpoint for these new ablations, rather than
> the 5 used for the original Fig. 5 results, given the compressed rebuttal timeline; we will extend
> to 5 seeds, and complete the Rock-Freeze comparison, for the camera-ready version.

---

## Posting checklist
1. Check whether `rock_freeze` finished training + evaluating before the deadline.
   - Training: `ls /umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token-checkpoints/rock_freeze/config.json`
   - Eval: `cat /umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token/evaluation/eval_results/summary.csv` (look for a `rock_freeze` row)
2. If yes → use **Plan A**, fill in the actual Ours numbers in the table, and pick whichever of the
   three bracketed sentences matches the real outcome (delete the other two).
3. If no → use **Plan B** as-is.
4. Either way, cross-check the seed-count sentence and the `RNAB_Q2.md` error-bar reply stay
   consistent (Q2 covers the *original* Fig. 5 five-run standard, unchanged; this file's 3-run
   disclosure is scoped to the new ablations only — don't let the two blur together in the final post).
5. Paste the chosen block into the OpenReview reply, replacing the placeholder table in `RNAB_W4.md`.

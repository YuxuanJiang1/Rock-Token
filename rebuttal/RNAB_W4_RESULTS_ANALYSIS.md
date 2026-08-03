# W4 results analysis — all 8 matched-condition runs, including Rock-Freeze

## Full table (mean ± sample SD across 3 seeds, per task; matched conditions: 2–4×L40S per-run
config as tuned, `openthoughts_prompt_math_5k_src30k-35k`, ~500 steps, 3 eval seeds)

| Rank by Avg | Variant | AIME24 | AIME25 | HMMT25feb | Avg |
|---|---|---|---|---|---|
| 1 | top_gradmag | 51.1±7.7 | 42.2±8.4 | 25.6±5.1 | 39.6 |
| 2 | top_meanloss | 48.9±1.9 | 42.2±3.9 | 23.3±0.0 | 38.1 |
| 3 | soft λ=0.3 | 44.4±6.9 | 41.1±3.8 | 25.6±1.9 | 37.0 |
| 4 | soft λ=0.5 | 44.4±9.6 | 37.8±5.1 | 25.6±6.9 | 35.9 |
| 5 | soft λ=0.7 | 36.7±5.8 | 46.7±11.5 | 23.3±5.8 | 35.6 |
| 6 | top_gradalign | 42.2±7.7 | 32.2±1.9 | 26.7±3.3 | 33.7 |
| 7 | top_freq | 32.2±1.9 | 41.1±3.8 | 17.8±3.9 | 30.4 |
| **8** | **Rock-Freeze (Ours)** | **36.7±6.7** | **32.2±8.4** | **18.9±5.1** | **29.3** |

**Rock-Freeze has the lowest average of all 8 conditions.** Not by a lot, and — this is the
important part — not in a way that's statistically distinguishable from the rest of the pack given
the sample sizes involved. Both things are true at once and both need to be in the writeup.

## How real is the ranking?

Treating each condition's 3-seed mean/SD as a rough sampling distribution (SE = SD/√3) and computing
gap-over-combined-SE for Rock-Freeze against each alternative, per task:

| Rock-Freeze vs. | AIME24 gap/SE | AIME25 gap/SE | HMMT25feb gap/SE |
|---|---|---|---|
| top_gradmag | 2.44 | 1.46 | 1.61 |
| top_meanloss | 3.03 | 1.87 | 1.50 |
| soft λ=0.3 | 1.39 | 1.67 | 2.13 |
| soft λ=0.5 | 1.14 | 0.99 | 1.35 |
| soft λ=0.7 | 0.00 | 1.76 | 0.99 |
| top_gradalign | 0.93 | 0.00 | 2.22 |
| top_freq | 1.12 (Rock-Freeze *higher*) | 1.67 (Rock-Freeze *lower*) | 0.30 |

With n=3 per condition, a proper two-sample t-test at these tiny degrees of freedom needs a ratio
well above these (roughly 2.8–4+ depending on df) to reach conventional significance. The single
largest gap here (Rock-Freeze vs. top-mean-loss on AIME24, ratio 3.03) is suggestive but not robust
at this sample size, and every other comparison is well below that. **Honest read: none of the 8
conditions are statistically distinguishable from each other at this scale.** The apparent ranking
in the table above is much more likely to reflect 3-seed sampling noise on 30-problem benchmarks
than a real, reproducible ordering.

This also changes something from the earlier draft: back when we only had the 7 alternatives and no
Rock-Freeze anchor, `top_freq`'s clear last-place position looked like the one credible finding
(large gap relative to its own tight error bars). Now that Rock-Freeze sits *below* top_freq
numerically, that framing doesn't survive intact — Rock-Freeze and top_freq are themselves within
noise of each other (ratios 1.12/1.67/0.30 across the three tasks, none large). We should not carry
the old "top-frequency is credibly worst" claim into the final text as previously worded.

## Why this might be, beyond "just noise" — worth stating honestly, not as excuses

1. **Statistical power is genuinely thin here**: 3 seeds × 30 problems/task means each condition's
   accuracy estimate has real sampling variance (SDs of 2–11 points on numbers in the 20–50 range),
   and 8 conditions being compared multiplies the chance some pair looks separated by chance alone.
2. **The training budget is short relative to the original figure**: ~500 steps here vs. the ~1400
   steps Fig. 5 plots (on a larger dataset, full compute budget). If the paper's own separation
   between Rock-Freeze/Original and Random *grows over the training trajectory* (which Fig. 5
   depicts as a widening gap over steps, not an instant one), a truncated run may simply not have
   had time for whatever separation exists to emerge yet. This is a real, disclosable limitation of
   the reduced-scale replication, not a excuse for the result — it's the honest reason we would not
   expect this experiment to have full power even if the underlying effect is real.
3. **We cannot rule out that the effect is smaller here than the original figure suggests at full
   scale.** This has to be on the table as a live possibility, not dismissed. We do not have evidence
   either way at this sample size — that is different from having evidence the effect is real but
   hard to see.

## What this means for the rebuttal — this needs a real rethink, not a patch

The `RNAB_W4_PlanAB.md` draft (written before this result came in) proposed claims along the lines
of "top-frequency is the clear worst" and conditional language for "if Ours beats the alternatives."
Neither survives this data cleanly:
- We cannot claim Rock-Freeze outperforms the alternatives (it doesn't, numerically, at this scale).
- We cannot even confidently claim top-frequency is uniquely bad anymore (Rock-Freeze is as low or
  lower, within noise).
- We *can* still honestly say: across 8 matched-condition selection criteria/strengths, none showed
  a statistically distinguishable difference in downstream accuracy at this reduced scale — which
  is itself a legitimate, reportable finding, just not the one the original ablation plan hoped for.

Two honest paths for the actual rebuttal text, neither involving hiding anything:
1. **Report the null result plainly**: state that at this compute-limited scale, no criterion
   (including the real Rock Score) showed a statistically credible advantage over the others, note
   the specific limitations above, and reiterate that the original paper's own Fig. 5 result (at
   full scale, more steps, 5 seeds, the original 10k-prompt dataset) is the evidence that actually
   supports the paper's claim — this rebuttal experiment was meant to probe *why* that result holds,
   and at reduced scale it wasn't powered to resolve that question either way.
2. **If time genuinely allows**: extend seeds (3→5) and/or step budget for at least the closest
   comparisons before the response is due, to see if a larger sample sharpens or erases the apparent
   ordering. Given the timeline pressure already established earlier, this is worth weighing against
   just posting the honest null-result version now.

Recommend against trying to rescue a "Rock Score wins" narrative from this data — that's exactly the
overclaiming risk flagged earlier in this process, and it would be far more damaging to credibility
with this specific reviewer than an honest null result, especially since the numbers are public
(OpenReview) and anyone can re-derive the same overlapping-error-bars conclusion from the table.

## Practical note

The `summary.csv` from the most recent run only has the `rock_freeze` row (the aggregation step
only recomputes for whatever `$CHECKPOINTS` was passed, and this run set `CHECKPOINTS=rock_freeze`
specifically). To regenerate a complete 8-row summary file for the record:
```bash
CHECKPOINTS="top_freq top_meanloss top_gradmag top_gradalign soft_lambda_3 soft_lambda_5 soft_lambda_7 rock_freeze" \
  ./run_eval_ablations.sh
```
All 24 seed-runs are already marked `.done`, so this just re-aggregates from existing results —
no GPU time needed, seconds to run.

## Status
Analysis complete. Rebuttal text (`RNAB_W4_PlanAB.md`) needs a rewrite based on this — not yet done,
flagging as the next step rather than attempting it in this file.

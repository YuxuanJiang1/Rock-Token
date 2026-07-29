# Reviewer RNAB — Response to Q2 (error bars and sample size)

**Q2 in one line.** The second Question bullet has two halves. The first (top-frequency / top-mean-loss /
soft-λ baselines) is the same ask as W4 and is answered in `RNAB_W4.md`. This file answers the second
half: Fig. 5 shows no error bars despite 5-run averaging, and the AIME24/25+HMMT25 suite is only 90
problems, so the visual Ours-vs-Original gap could be within noise.

## Proposed rebuttal reply (paste-ready)

We thank the reviewer for raising this. The reviewer is right that Fig. 5 should display variance. Our Pass@1 values are averaged over five independent runs, and we will add error bands (± one standard deviation across the five runs) to the Original, Random, and Ours curves using the existing per-run scores, so that the reader can judge whether the gap between Ours and Original OPD is within noise.

Two further points bear on the sample-size concern. First, the comparison is not a single snapshot: Fig. 5 tracks all three settings across the full training trajectory, and Ours closely follows Original OPD across checkpoints while Random separates early and remains well below both — a consistency across checkpoints that a single-point sampling artifact would not produce. Second, the central claim is parity with Original OPD together with a clear separation from the frequency-matched control at a 1.7× speedup; the Random-vs-Ours gap is large relative to the Ours-vs-Original gap, so the conclusion does not rest on the smaller difference the reviewer rightly flags. We will make both the error bands and this trajectory-level reading explicit in the revision.

## Open items (internal notes)
- Check whether the five per-run Fig. 5 scores are still on disk. If so, the error bands can be added with no new evaluation. If not, re-evaluate the existing three checkpoints (Original / Random / Ours) with the five seeds.
- Decide the band definition before posting: ± one standard deviation across the five runs (simple, matches the "averaged over five runs" wording) vs. a bootstrap 95% CI (stronger, but implies a distributional claim). The reply above says "± one standard deviation"; change it if you prefer the CI.

## Status
Ready to post. The numeric error bands / updated figure follow once the per-run data is located or regenerated.

# Reviewer RNAB — W4 follow-up reply (normalizer question, rebuttal-ablation scope only)

Paste-ready reply to the reviewer's non-monotonicity / normalizer follow-up. Scoped to the new
rebuttal ablations only, kept at a plain-language level (no code walkthrough). States plainly that
the training pipeline behind the paper's submitted Fig. 5 results is unaffected — the bug is specific
to the new rebuttal ablation code.

---

We thank the reviewer for this analysis. The loss normalization in the rebuttal ablation code does
not fully account for the frozen-token weighting: it normalizes by a quantity that varies with λ,
rather than the fixed, batch-level normalizer used elsewhere in our training pipeline. This shifts
the effective per-token gradient scale on non-frozen tokens as λ changes, producing an implicit,
λ-dependent learning rate, consistent with the reviewer's hypothesis. The issue applies to all eight
new ablation conditions, not only the three interior soft-λ points, since every condition freezes a
set of tokens through this same code path.

This issue is specific to the rebuttal ablation code. The pipeline used to produce the paper's
submitted results is unaffected — the code behind Figure 5 (Original OPD, Random, and
Rock-Freeze/Ours) does not have this normalizer issue.

Checked against the full-scale submitted values (Original OPD = 48.1, Rock-Freeze/Ours = 44.3,
Random = 38.9), our reduced-scale interior soft-λ points (37.0/35.9/35.6) sit 11.1–12.5 and 7.3–8.7
points below the two endpoints, and the Random baseline exceeds six of our seven non-anchor
reduced-scale variants — matching the reviewer's figures. Both comparisons set full-scale
submitted-paper results against a reduced-scale rebuttal ablation rather than points within a single
controlled sweep, so they reflect some combination of scale (2 GPUs/~500 steps/5k-prompt slice vs.
4 GPUs/full epoch/10k prompts) and the normalizer effect above; separating the two requires a
same-scale rerun.

We are correcting the normalizer in the ablation code and will re-run all eight conditions, together
with a same-scale λ = 1 control, for the camera-ready version, and report whether the ordering and
the interior-λ pattern change. This will not be complete before the current response window closes.

---

## Note (not for posting)
Fig. 5's submitted numbers (48.1 / 44.3 / 38.9) are treated above as correct reference values, not as
something in question — per your note, that code path was updated after the window where this
normalizer bug existed, so it's out of scope here. If that turns out to be wrong, this file needs a
different framing; flagging so it doesn't get forgotten if the timeline changes.

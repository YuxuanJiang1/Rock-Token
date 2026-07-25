# Reviewer RNAB — Response to W4 (and the related error-bar point in Questions)

## What RNAB asked for
Fig. 5's "Random" baseline (frequency-matched, semantically useful tokens) is "almost designed to
look bad." To show the Rock Score construction specifically drives the Fig. 5 gap, RNAB wants:
1. freezing top-frequency tokens (no KL)
2. freezing top-mean-loss tokens without the Rock Score construction (no Freq weighting)
3. soft down-weighting at an intermediate λ (not just the binary λ=0 "freeze")
4. freezing tokens selected by gradient magnitude/alignment

Plus, in the accompanying Question: Fig. 5 has no error bars despite 5-run averaging, and the
AIME24/25+HMMT25 eval is only 90 problems total, so the "Ours vs. Original OPD" gap could plausibly
be noise.

## Why this is tractable without new code
`stumbling/kdflow/algorithms/token_freeze_kd.py` already implements Eq. (5) in its general form: it
takes an arbitrary token-ID JSON (`--token_freeze_path`) and an arbitrary weight
(`--freeze_weight` = λ ∈ [0,1]), not just the binary rock/random choice used in the paper. So all
four requested ablations, plus the soft-λ sweep, are new *configs* against existing training code —
no algorithm changes needed. What's new is which token list goes in, and how far λ moves from 0.

## What's been prepared (this session)
- `stumbling/build_ablation_freeze_lists.py` — derives all four lists from the same raw statistics
  already collected for Section 2 / Fig. 3 (`rock_token_occurrences_..._unrestricted.pt`,
  `logit_gradients_..._unrestricted.pt`):
  - `top_freq.json` — top-100 by Freq(v) alone
  - `top_meanloss.json` — top-100 by mean_kl(v) alone, excluding current rocks (reuses the "high_kl"
    group definition already used for Fig. 3, so it's consistent with a quantity already in the paper)
  - `top_gradmag.json` / `top_gradalign.json` — top-100 by gradient magnitude / by cosine alignment
    with `G_balanced` (the exact quantities behind Fig. 3's panels a/b)
  - Also writes `ablation_lists_summary.csv` with each list's Jaccard overlap with the current rock
    set — useful context for interpreting results (e.g. if `top_freq` turns out to be 90% identical
    to `rock`, that's itself informative).
- Five launch scripts cloned from `run_stumb_random.sh` (same cluster/Ray/FSDP2 setup, same
  hyperparameters as the reported runs — only `--token_freeze_path` / `--freeze_weight` /
  `--save_path` differ):
  - `run_stumb_top_freq.sh`, `run_stumb_top_meanloss.sh`, `run_stumb_gradmag.sh`,
    `run_stumb_gradalign.sh` — each λ=0 (hard freeze), varying only the selection criterion
  - `run_stumb_soft_lambda.sh <lambda>` — reuses `rock.json`, run with λ=0.3 / 0.5 / 0.7

## Hardware note: adapted for 4x L40S (not the paper's 4x H100)
The paper's Stage 2 / Fig. 5 runs used 4x H100 (80GB, NVLink). The 5 launch scripts above are
adjusted for 4x L40S (48GB, Ada Lovelace, no NVLink) instead:
- `--num_gpus_per_node 2 → 4` and `--teacher_tp_size 2 → 4`, spreading the ~60GB bf16 teacher
  (Qwen3-30B-A3B) to ~15GB/GPU — with 4x48GB=192GB total, this is actually more aggregate memory
  than the 2xH100 (160GB) config `run_stumb_random.sh` originally used, just spread across more,
  smaller cards, so this should fit comfortably.
- Added `--teacher_quantization fp8`: Ada supports FP8 tensor cores natively (unlike the paper's
  H100 bf16-only setup) and SGLang 0.5.9 (pinned in `requirements.txt`) supports it, roughly halving
  teacher weight footprint again to ~7.5GB/GPU. If this throws a quantization/kernel error for this
  model, just delete the flag — TP=4 alone likely leaves enough headroom to fall back to bf16.
- `mem_fraction_static` values and sequence lengths are left as in the original ablation config;
  they're the next things to tune down if you hit OOM in practice.
- **No NVLink means step time on L40S won't match the paper's H100 numbers.** This is fine for the
  top-freq/top-meanloss/gradmag/gradalign ablations, which are purely accuracy comparisons. It matters
  for the soft-λ sweep, which is explicitly building an accuracy-vs-speed frontier — for that one,
  also re-run the λ=0 and λ=1 endpoints on the same L40S hardware rather than reusing the paper's
  reported H100 wall-clock, so the frontier is internally consistent.

## To run (you / co-author with cluster access)
1. On the GPU server, regenerate or locate `rock_token_occurrences_onpolicy_n500_unrestricted.pt`
   and `logit_gradients_onpolicy_n500_unrestricted.pt` (per `rock_detection/README.md`; these are
   the same artifacts Fig. 2/3 were built from, ~50MB and ~3.6GB respectively, not checked into git).
2. `python build_ablation_freeze_lists.py --occurrences <path> --gradients <path> --rock-csv rock_vs_control_unrestricted.csv --outdir /p/work2/xxj1/rocktoken/stumbling_token/`
3. Launch the 4-5 training runs (same scale as the existing Rock/Random runs — budget accordingly).
4. Evaluate each checkpoint on AIME24/25 + HMMT25 exactly as for Fig. 5, ideally keeping all 5
   Pass@1 repeats so we can add error bars (see below).

## Proposed rebuttal reply (fill in results before posting)

> We thank the reviewer for this suggestion — the frequency-matched Random baseline alone doesn't
> isolate which ingredient of the Rock Score (frequency weighting vs. mean-loss ranking vs. the joint
> criterion) drives the Fig. 5 result. Our freeze mechanism already generalizes to arbitrary token
> sets and arbitrary freeze strength (Eq. 5's λ is not restricted to {0,1} in our implementation), so
> we ran the following additional variants at the same scale as Fig. 5:
>
> | Variant | Selection criterion | λ | Result (avg. AIME24/25+HMMT25, final ckpt) |
> |---|---|---|---|
> | Original OPD | — | 1 | *(existing)* |
> | Random (paper) | freq-matched, random | 0 | *(existing)* |
> | Rock-Freeze (Ours) | Freq(v)·mean_kl(v), top-100 | 0 | *(existing)* |
> | Top-frequency freeze | Freq(v) alone, top-100 | 0 | *TBD* |
> | Top-mean-loss freeze | mean_kl(v) alone, top-100 | 0 | *TBD* |
> | Grad-magnitude freeze | \|\|ḡ_t\|\|, top-100 | 0 | *TBD* |
> | Grad-alignment freeze | cos(ḡ_t, G_balanced), top-100 | 0 | *TBD* |
> | Rock, soft λ=0.3/0.5/0.7 | Rock set, partial down-weight | 0.3/0.5/0.7 | *TBD* |
>
> [Interpretation to fill in once results land — e.g. "Top-frequency and top-mean-loss freezes
> underperform Rock-Freeze by X points while matching its speedup, showing the joint Freq×KL
> criterion is doing real work beyond either signal alone" / or the honest alternative if a single
> ingredient turns out to explain most of the effect.]
>
> On error bars: Fig. 5 already averages 5 independent rollout seeds per checkpoint per benchmark; we
> will add [std-dev / bootstrap 95% CI] bands to the figure using the existing per-seed scores (no new
> evaluation needed) so the reader can assess whether the Ours-vs-Original gap exceeds sampling noise
> at n=90.

## Bonus (no new compute): error bars on the existing Fig. 5
The paper already states Pass@1 is "averaged over five independent runs." If those five per-run
scores were logged (likely in the same eval output as the existing Fig. 5 numbers), we can add error
bars to the *existing* Original/Random/Ours curves immediately, without waiting on the new ablation
runs — this directly answers the "could be within noise" question and is worth doing regardless of
how the four new ablations turn out. Worth checking whether that per-seed data is still on disk
before re-running anything.

## Status
Scripts ready (`stumbling/build_ablation_freeze_lists.py`, `stumbling/run_stumb_*.sh`). Awaiting: (1)
locating/regenerating the raw occurrence + gradient artifacts, (2) launching the runs, (3) filling in
the table above, (4) checking for existing per-seed Fig. 5 data for the error-bar addition.

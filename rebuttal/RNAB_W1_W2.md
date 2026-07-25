# Reviewer RNAB — Response to W1 and W2

Grounded in the actual implementation:
- `stumbling/kdflow/loss/reverse_kl_div.py` (and the identical copy in `KDFlow_localopd/`) — the training loss for `kd_loss_fn=rkl`
- `rock_detection/rock.py`, `rock_detection/rerun_unrestricted.py` — the Rock Score / token-selection pipeline
- `Cornerstones_or_Stumbling_Blocks.../rock.tex` — the paper source (lines ~44, ~56, ~71)

## What we found

There is a real internal inconsistency in the current draft, and the reviewer caught it precisely.

- The sentence immediately after Eq. (1) (`rock.tex` line 44) defines
  `ℓ_t = log π_θ(x_t|x_<t) − log π_T(x_t|x_<t)` — a **single-token sampled log-ratio**.
- Section 2.3 (`rock.tex` line 71) defines the estimator actually used as
  `ℓ̂_v = Ê[D_KL(π_θ(·|x_<t)‖π_T(·|x_<t)) | x_t=v]` — the **full analytic per-position KL**.
- The code, in both places that matter (the training loss and the Rock Score computation), implements
  the *second* definition, exactly:

  ```python
  # kdflow/loss/reverse_kl_div.py (training loss) and rock_detection/rock.py (Rock Score stats)
  student_probs = student_log_probs.exp()
  rkl_div = (student_probs * (student_log_probs - teacher_log_probs)).sum(-1)  # full vocab sum
  ```

  This is deterministic given the context `x_<t` — both conditional distributions are available in
  closed form from a single forward pass, so no sampling of `x_t` is needed to compute it. It is
  never the pointwise log-ratio the inline sentence describes.

So the ambiguity W1 raises (true objective vs. sampled surrogate vs. context-conditioned statistic)
is resolved by fixing one sentence: `ℓ_t` is always the exact analytic KL, matching Eq. (1)'s summand
and Section 2.3's estimator. The *only* randomness anywhere in the pipeline is the standard on-policy
trajectory sampling `x_1:T ~ π_θ` that generates the rollout corpus in the first place — i.e., which
contexts occur, and how often each token type is generated at them. That is the same sampling process
that already defines the outer expectation in Eq. (1), and the Rock Score is computed over exactly
that same corpus, not a second/different estimator stacked on top.

This also resolves W2. `Freq(v)` is the empirical count of positions where `x_t = v` in that same
rollout corpus (`rock_detection/rock.py`: `token_frequencies.scatter_add_(...)`; equivalently
`df.groupby("token_id")["kl"].agg(freq="count", mean_kl="mean")` in `rerun_unrestricted.py`) — an
empirical frequency, not an expected probability mass or a model-based quantity. Given that `ℓ_t` is
deterministic given context, Eq. (2) is not a modeled decomposition but an **exact algebraic identity**
of the empirical corpus loss: grouping positions by realized token and averaging is definitionally
lossless. The phrase "the theoretical training objective" overstates this slightly — the precise claim
is that `R(v)` is the exact contribution of token type `v` to the *empirical, corpus-level* estimate of
`L_OPD(θ)` realized by the N=500 rollouts, which is the same estimate used everywhere else in the paper
(e.g. reported training loss). We will tighten this language.

## Proposed rebuttal reply (drop-in text)

> We thank the reviewer for catching this — there is a genuine notational inconsistency in the current
> draft, not a deeper ambiguity in the method. The sentence following Eq. (1) describes `ℓ_t` as a
> single-token sampled log-ratio, but this does not match Section 2.3's estimator or, in fact, our
> actual implementation: both the OPD training loss and the Rock Score statistics compute the *exact*
> per-position reverse KL, `ℓ_t := D_KL(π_θ(·|x_<t) ‖ π_T(·|x_<t))`, as a full sum over the vocabulary
> from the two models' closed-form conditional distributions — never a function of the realized token
> `x_t`. We will correct the text after Eq. (1) to state this directly.
>
> This removes the ambiguity: `ℓ_t` requires no sampling to compute, so there is no distinction between
> "the true objective" and "a sampled surrogate" at the level of a single position. The only stochasticity
> in the whole pipeline is the on-policy trajectory sampling `x_1:T ~ π_θ` that produces the rollout
> corpus — the same sampling process that defines the outer expectation in Eq. (1) itself, and the one
> that any practical estimate of `L_OPD` must use. The Rock Score is computed over exactly that corpus.
>
> On `Freq(v)` (W2): it is the empirical count of positions where `x_t = v` in that same rollout corpus
> — not an expected probability mass or a model-derived quantity. Because `ℓ_t` is deterministic given
> context, Eq. (2) is then an exact algebraic identity of the empirical corpus-level loss (grouping
> positions by realized token and averaging cannot lose information), not an approximation layered on
> top of a sampled estimator. We will revise "the theoretical training objective" to "the empirical,
> corpus-level OPD objective realized by these rollouts" for precision.

## Proposed paper edit

**Replace** (`rock.tex`, line 44):
> The loss at position $t$ is $\ell_t = \log \pi_\theta(x_t | x_{<t}) - \log \pi_T(x_t | x_{<t})$,
> measuring the student-teacher mismatch.

**With**:
> The loss at position $t$ is the exact per-position reverse KL,
> $\ell_t := D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot|x_{<t}) \,\|\, \pi_T(\cdot|x_{<t})\big)$, computed in
> closed form from the student's and teacher's full conditional distributions at that position — no
> sampling of $x_t$ is required. This is the summand already appearing inside Eq.~(1) before the outer
> trajectory expectation, and coincides with the estimator $\widehat\ell_v$ used in Section~2.3.

**Add** (after Eq. 2, `rock.tex` ~line 58, before the `Freq(v)` regularizer paragraph):
> Both $\mathrm{Freq}(v)$ and $\mathbb{E}[\ell_t \mid x_t=v]$ are computed empirically over the same
> $N{=}500$ on-policy rollout corpus used to estimate $\mathcal{L}_{\text{OPD}}$; consequently Eq.~(2)
> is an exact identity at the level of this empirical corpus rather than a modeled approximation, and
> the only source of Monte Carlo variance is the trajectory sampling $x_{1:T}\sim\pi_\theta$ inherent
> to on-policy distillation itself.

**Soften** (line 56, last sentence):
> Ranking tokens by $R(v)$ is the unique additive ranking that strictly agrees with the **empirical,
> corpus-level** OPD objective realized by these rollouts.

## Status
Ready to use — no new experiments required. Not yet posted to OpenReview.

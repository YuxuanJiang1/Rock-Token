# Reviewer RNAB — Response to W1 (Eq. 1 vs. the sampled loss $\ell_t$)

**W1 in one line.** Eq. (1) uses the full per-position reverse KL as its summand, but the sentence
right after it defines $\ell_t$ as a single-token *sampled log-ratio* — an object whose expectation
under $\pi_\theta$ only *equals* the KL. The paper then speaks as though $\ell_t$ is the KL, leaving it
ambiguous whether Rock Tokens are defined against the true objective, a sampled estimator, or a
context-conditioned statistic.

Grounded in the implementation:
- `stumbling/kdflow/loss/reverse_kl_div.py:20` (identical copy in `KDFlow_localopd/`) — the training
  loss for `kd_loss_fn=rkl`
- `rock_detection/rock_server.py:180` — the per-token KL used to build the paper's `N=500` detection
  artifacts (`rock_detection/rock.py` is an earlier small-scale variant with the identical KL formula);
  `rock_detection/rock_server.py:147` — the decoding rule
- `Cornerstones_or_Stumbling_Blocks.../rock.tex` — the paper source (lines 44, 71)

## What we found (internal notes)

The reviewer caught a real internal inconsistency. It is a **one-sentence notational bug**, not a
flaw in the method or the code.

- The sentence immediately after Eq. (1) (`rock.tex:44`) defines
  $\ell_t = \log \pi_\theta(x_t|x_{<t}) - \log \pi_T(x_t|x_{<t})$ — a **single-token sampled log-ratio**.
- Section 2.3 (`rock.tex:71`) defines the estimator actually used as
  $\widehat\ell_v = \widehat{\mathbb{E}}[\,D_{\mathrm{KL}}(\pi_\theta(\cdot|x_{<t})\,\|\,\pi_T(\cdot|x_{<t}))\mid x_t=v\,]$
  — the **full analytic per-position KL**.
- The code, in both places that matter (the training loss *and* the Rock Score statistics), implements
  the *second* definition, exactly:

  ```python
  # reverse_kl_div.py:20 (training loss)  AND  rock_server.py:180 (Rock Score statistics)
  student_probs = student_log_probs.exp()
  rkl_div = (student_probs * (student_log_probs - teacher_log_probs)).sum(-1)  # full-vocab sum
  ```

  The `.sum(-1)` runs over the whole vocabulary, so this is the closed-form KL, deterministic given the
  context $x_{<t}$ and never a function of the realized token $x_t$. It is not the pointwise log-ratio the
  inline sentence describes.

The fix is substantive, not cosmetic. The sampled log-ratio equals the KL only in expectation over
tokens drawn from $\pi_\theta$. Our detection rollouts use **greedy decoding** (`rock_server.py:147`,
`do_sample=False`), so the log-ratio reading would be a *biased* per-position estimator, whereas the full
KL we compute is exact regardless of how the trajectory is decoded. Stating the KL form is therefore the
correct choice, not just a matter of notation.

## Proposed rebuttal reply (paste-ready)

We thank the reviewer for this careful reading. The reviewer is right that the sentence following Eq. (1) is inconsistent, and we are glad to clarify it. The issue is an error in one sentence, not an ambiguity in the method.

In both the training objective and the Rock Score computation, the per-position loss is the exact reverse KL divergence,

$$\ell_t = D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid x_{<t}) \,\|\, \pi_T(\cdot \mid x_{<t})\big),$$

evaluated as a full sum over the vocabulary from the student's and teacher's conditional distributions at that position. This quantity is fully determined by the context $x_{<t}$ and does not depend on the sampled token $x_t$. It is exactly the summand inside Eq. (1), and it matches the estimator $\widehat\ell_v$ defined in Section 2.3.

The sentence after Eq. (1) mistakenly writes $\ell_t$ as the single-token log-ratio $\log \pi_\theta(x_t \mid x_{<t}) - \log \pi_T(x_t \mid x_{<t})$. This is a typographical error, and we will correct it to the divergence above. With this correction, there is no sampled estimator at the per-position level, and the three interpretations the reviewer distinguishes — the true objective, a sampled estimator, and a context-conditioned statistic — coincide.

We would add that this correction is more than cosmetic. As the reviewer notes, the log-ratio equals the KL only in expectation over tokens drawn from $\pi_\theta$. Because our analysis uses greedy decoding rather than sampling, the log-ratio interpretation would in fact be biased, whereas the full KL that we compute is exact regardless of how the trajectory is decoded. The divergence form is therefore both consistent with our implementation and the correct definition to state.

## Proposed paper edit

Replace the sentence at `rock.tex:44`:

```
The loss at position $t$ is $\ell_t = \log \pi_\theta(x_t | x_{<t}) - \log \pi_T(x_t | x_{<t})$, measuring the student-teacher mismatch.
```

with:

```
The loss at position $t$ is the exact per-position reverse KL, $\ell_t := D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot|x_{<t}) \,\|\, \pi_T(\cdot|x_{<t})\big)$, computed in closed form from the student's and teacher's full conditional distributions at that position---it is deterministic given $x_{<t}$ and requires no sampling of $x_t$. This is the summand already appearing inside Eq.~(1) before the outer trajectory expectation, and it coincides with the estimator $\widehat\ell_v$ used in Section~2.3.
```

## Note for the paired W2 reply
The W2 response (`RNAB_W2.md`) relies on this fix: it is because $\ell_t$ is the exact,
context-deterministic KL that Eq. (2) is an exact identity rather than an approximation over a sampled
estimator. Keep the two replies consistent on this definition.

## Status
Ready to use — no new experiments required. Not yet posted to OpenReview.

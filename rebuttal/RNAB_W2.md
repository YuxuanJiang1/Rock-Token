# Reviewer RNAB — Response to W2 (Freq$(v)$ and the Eq. 2 decomposition)

**W2 in one line.** Eq. (2) claims $R(v)$ is the *exact* contribution of a token type and the "unique
additive ranking that strictly agrees with the OPD objective." The reviewer says this hinges on what
$\mathrm{Freq}(v)$ means (empirical count? expected probability mass? expected count over trajectories?)
and on whether the conditioning matches the sampling process that defines the objective — none of which
the paper explains.

Grounded in the implementation:
- `rock_detection/rock_server.py:189` — how $\mathrm{Freq}(v)$ is accumulated; equivalently
  `rock_detection/rerun_unrestricted.py:40-42` — `df.groupby("token_id")["kl"].agg(freq="count", mean_kl="mean")`
- `Cornerstones_or_Stumbling_Blocks.../rock.tex` — the paper source (lines 52–56)
- **Depends on the W1 fix** (`RNAB_W1.md`): the argument below requires $\ell_t$ to be the exact,
  context-deterministic KL, not the sampled log-ratio the current `rock.tex:44` states.

## What we found (internal notes)

$\mathrm{Freq}(v)$ is the **empirical occurrence count** of token type $v$ in the analysis corpus — a
raw integer count, not an expected probability mass and not a model-derived quantity:

```python
# rock_server.py:189   (equivalently  df.groupby("token_id")["kl"].agg(freq="count")  in rerun_unrestricted.py:40-42)
token_frequencies.scatter_add_(0, gen_tokens_cpu, torch.ones_like(gen_tokens_cpu, dtype=torch.long))
```

Because $\ell_t$ is deterministic given context (see W1), Eq. (2) is an **exact algebraic identity** of
the empirical corpus loss, and it holds for any fixed corpus regardless of how it was generated. Writing
$\widehat L = \sum_{p} \ell_p$ for the sum over all generated positions $p$ and grouping positions by the
token realized at each one:

$$\widehat L=\sum_v\Big(\text{\# positions emitting } v\Big)\times\Big(\text{mean KL at those positions}\Big)=\sum_v \mathrm{Freq}(v)\cdot\widehat\ell_v=\sum_v R(v).$$

Grouping and averaging is lossless, so $R(v)$ is the exact share of the corpus loss attributable to token
type $v$, and this uses **no assumption about the sampling process**. That answers the second half of W2
directly: the exactness of the decomposition does not depend on whether the conditioning matches the
objective's sampling process.

Two honesty adjustments to the paper's phrasing (the math is right, the words overreach):

1. $\mathrm{Freq}(v)$ is a raw count, so $\sum_v R(v)$ equals the *summed* (un-normalized) corpus KL. If
   Eq. (1) is read as a per-trajectory expectation, the operational estimate is
   $\widehat L/(\#\text{trajectories})$ — a global constant that leaves the $R(v)$ ranking unchanged.
2. `rock.tex:56` calls $\widehat L$ "the theoretical training objective." $\widehat L$ is the empirical,
   corpus-level *estimate* of that objective. Whether the greedy MATH-500 corpus faithfully samples the
   full on-policy expectation is a separate measurement question (Fig. 2d's Jaccard stability is our
   evidence the ranking is not a corpus artifact), orthogonal to the algebraic exactness the reviewer
   questions.

## Proposed rebuttal reply (paste-ready)

We thank the reviewer for pressing on the precise meaning of $\mathrm{Freq}(v)$ and on what Eq. (2) claims. We clarify both points below.

**Meaning of $\mathrm{Freq}(v)$.** $\mathrm{Freq}(v)$ is the empirical number of times token type $v$ is generated in our analysis corpus (the student's own rollouts over $N = 500$ MATH-500 problems). It is a raw occurrence count, not an expected probability mass and not a model-derived quantity.

**Why Eq. (2) is an exact identity.** Once $\ell_t$ is the exact per-position KL (see our response to W1), it is fully determined by the context and does not depend on the sampled token. Eq. (2) is then a regrouping of the corpus loss. Writing $\widehat L = \sum_p \ell_p$ for the sum over all generated positions $p$, and grouping positions by the token realized at each one,

$$\widehat L = \sum_v \big(\text{number of positions emitting } v\big)\times\big(\text{mean KL at those positions}\big) = \sum_v \mathrm{Freq}(v)\,\widehat\ell_v = \sum_v R(v).$$

Grouping and averaging loses no information, so $R(v)$ is exactly the share of the corpus loss attributable to token type $v$. This holds for any fixed set of positions and requires no assumption about how they were sampled. It therefore addresses the reviewer's concern directly: the exactness of the decomposition does not depend on whether the conditioning matches the objective's sampling process.

What the choice of corpus does affect is how well the measured $R(v)$ values represent the full on-policy expectation, which is a question of estimation rather than of the identity. Figure 2(d) shows that the resulting ranking is stable under subsampling.

We will revise the manuscript accordingly. We will state explicitly that $\mathrm{Freq}(v)$ is an empirical count and that Eq. (2) is an exact identity for the corpus-level loss, and we will change "the theoretical training objective" to "the empirical, corpus-level OPD objective realized by these rollouts," since $\mathrm{Freq}(v)$ is an un-normalized count and $\widehat L$ estimates, rather than equals, the population objective.

## Proposed paper edits

Add after Eq. (2) (`rock.tex` ~line 57, before the $\mathrm{Freq}(v)$ regularizer paragraph):

```
Here $\mathrm{Freq}(v)$ is the empirical occurrence count of token type $v$ and $\mathbb{E}[\ell_t \mid x_t=v]$ its mean per-position KL, both measured over the same on-policy rollout corpus. Since $\ell_t$ is deterministic given context, Eq.~(2) is an exact algebraic identity of the empirical corpus loss---grouping positions by realized token and averaging is lossless---rather than a modeled approximation, and it holds independently of how the trajectories are decoded.
```

Soften the last sentence at `rock.tex:56`:

```
Ranking tokens by $R(v)$ is the unique additive ranking that strictly agrees with the empirical, corpus-level OPD objective realized by these rollouts.
```

## What this reply does *not* claim (deliberately)

- It does **not** say the pipeline's only randomness is "sampling $x_{1:T}\sim\pi_\theta$." The detection
  rollouts are greedy (`do_sample=False`), so the rollout itself is deterministic; the only estimation
  variance is over which $N=500$ problems are drawn. Claiming Monte-Carlo trajectory sampling here would
  be false, and this reviewer reads code carefully.
- It does **not** assert the analysis corpus is the *same* sampling process as training. Training uses
  4 sampled rollouts/prompt over OpenThoughts; the analysis uses greedy rollouts over MATH-500. We defend
  Eq. (2) by making the identity independent of the sampling process, not by claiming the two processes
  coincide. (This also keeps us consistent with the separate context-distribution point raised by
  Reviewer K8Wc's W2, and with the K8Wc Q3 reply, which should present $\sum_v R(v)=\widehat L$ as an
  empirical estimator of the population objective rather than an exact equality with $\mathcal L_{\text{OPD}}$.)

## Status
Ready to use — no new experiments required. Not yet posted to OpenReview.

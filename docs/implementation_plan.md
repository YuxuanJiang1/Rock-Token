# Rock Tokens: From Identification to Causal Analysis

This document covers the full pipeline for studying Recalcitrant Tokens (Rock Tokens) in On-Policy Distillation: first, how to identify them given a trained student model, then how to systematically analyze their causal role through masking experiments.

---

## Part 1: Identifying Rock Tokens

You start with a trained student model and need to extract a workable set of N Rock Tokens (typically N=100) for downstream experiments. The pipeline has four phases: generate, measure, filter, aggregate.

### What You Need

Four ingredients are required before any measurement:

1. **The teacher model** (e.g., Qwen3-30B-Instruct) — for computing target distributions.
2. **The trained student model** (`θ*`) — the post-OPD checkpoint.
3. **The pre-OPD student** (`θ₀`) — the initialization before distillation. If you didn't save it, just re-download the base model; the weights are identical.
4. **A fixed evaluation prompt set** — 1K–2K prompts drawn from your training distribution but held out from training. The "fixed" part matters: the same prompts must be used across all measurements so the loss values are directly comparable.

### Phase 1: Generate Student Outputs

Rock Tokens live on the student's own output distribution because OPD is on-policy. So you generate from the student first, then measure losses on those generations.

```python
prompts = load_evaluation_prompts()  # 1K–2K prompts
student_outputs = []

for prompt in prompts:
    output = student_model.generate(
        prompt,
        max_length=2048,
        temperature=1.0,
        do_sample=True,
    )
    student_outputs.append({
        "prompt": prompt,
        "output": output,
        "token_ids": tokenize(output),
    })
```

Two design choices worth flagging. First, use **temperature 1.0 with sampling**, not greedy decoding. Greedy gives you only the argmax token at each position, which hides exactly the kind of confidently-wrong behavior you're trying to characterize. Second, if compute allows, generate **3–5 outputs per prompt**. A token position that's high-loss across multiple sampled continuations is a much more robust Rock Token than one that's high-loss in a single sample.

### Phase 2: Compute Per-Token Loss at Both Checkpoints

For every token in every generated sequence, compute the KL divergence between teacher and student — twice, once with `θ₀` and once with `θ*`.

The mechanism is **teacher forcing**: feed the same generated sequence into all three models (teacher, pre-OPD student, post-OPD student) and compare their next-token distributions at each position. This is a single forward pass, not autoregressive generation.

```python
def get_token_distributions(model, prompt, output_tokens):
    input_ids = tokenize(prompt + output_tokens)
    with torch.no_grad():
        logits = model(input_ids).logits  # [seq_len, vocab_size]
    prompt_len = len(tokenize(prompt))
    output_logits = logits[prompt_len - 1 : -1]
    return torch.softmax(output_logits, dim=-1)

def token_kl(p_teacher, p_student):
    eps = 1e-10
    p_t = p_teacher.clamp(min=eps)
    p_s = p_student.clamp(min=eps)
    return (p_t * (p_t.log() - p_s.log())).sum()
```

For each token position `t`, you now have two scalars: `loss_before[t]` (teacher vs. `θ₀`) and `loss_after[t]` (teacher vs. `θ*`). Store these alongside metadata you'll want later — token ID, decoded text, position in sequence, student entropy, teacher entropy.

### Phase 3: Apply the Recalcitrance Criterion

You need a threshold rule that turns the continuous loss values into a binary Rock/non-Rock label. Two reasonable options:

**Absolute threshold** — A token is Recalcitrant if `loss_before > τ` AND `loss_after > τ`, where τ is the 90th percentile of `loss_before`. Simple to explain, easy to defend.

**Relative threshold** — A token is Recalcitrant if `loss_before > τ_high` AND `(loss_before - loss_after) < δ`. This says "the loss was high AND it barely improved." It directly captures resistance to learning rather than just persistent difficulty.

Prefer the relative threshold. It excludes tokens whose loss started high and dropped substantially (they're Learned tokens that happen to still be above average), keeping only tokens where the student genuinely failed to make progress. A reasonable setting: `τ_high` at the 80th percentile of initial loss, `δ` at the 20th percentile of loss improvement.

### Phase 4: From Token Instances to Token Types

Phase 3 gives you Rock Token _instances_ — specific positions in specific sequences. But masking experiments need Rock Token _types_ — vocabulary items you can block during generation. The same word ("so") might be recalcitrant in some contexts and easily learned in others, so you need to aggregate.

```python
from collections import Counter

rock_type_counts = Counter(t["token_id"] for t in rock_tokens)
total_type_counts = Counter(t["token_id"] for t in all_tokens)

rock_rates = {
    tid: rock_type_counts[tid] / total_type_counts[tid]
    for tid in rock_type_counts
    if total_type_counts[tid] >= 10  # frequency filter
}
top_100 = sorted(rock_rates.items(), key=lambda x: -x[1])[:100]
```

You can rank by **rate** (proportion of times this vocabulary item appears as Rock) or by **absolute count** (how often it shows up as Rock in total). Rate prioritizes consistency; count prioritizes impact. Reporting both is cleanest.

The minimum frequency filter (≥10 occurrences) is essential — without it, your top tokens will be dominated by rare Unicode artifacts that happened to occur once and got high loss. You want vocabulary items that appear often enough for the masking experiments to be statistically meaningful.

### Sanity Checks Before Proceeding

Spend an hour on these before running any masking experiments. They catch problems that would otherwise waste days of compute.

- **Scatter plot of `loss_before` vs. `loss_after`.** You should see a four-quadrant structure: Easy (low/low), Learned (high/low), Rock (high/high), Regressed (low/high). If the plot is a featureless blob, your threshold needs adjusting before anything else makes sense.
- **Rock Token fraction.** Roughly 5–20% of token instances should qualify. 50% means your threshold is too loose; 0.5% means too strict.
- **Inspect the top 100.** Print them with their decoded text, frequency, and avg losses. A healthy mix includes content words, function words, punctuation, and symbols. If it's all rare BPE fragments, tighten your frequency filter.
- **Correlation with entropy.** Plot Rock Token rate against student entropy. If Rock Tokens are uncorrelated with entropy, you have a genuinely new category distinct from forking tokens (Wang et al., 2025). If they're tightly correlated, you need to argue why your framing adds value beyond existing entropy-based work.

The full identification pipeline runs in under one GPU-day. The expensive work — the masking experiments — comes next.

---

## Part 2: Systematic Masking Experiments

You now have 100 Rock Tokens. The goal of this phase is to determine which are Pillars (essential — removal hurts) and which are Stumbling Blocks (harmful — removal helps), and to extract findings strong enough to anchor a paper. The experiments are ordered from cheap-and-foundational to expensive-but-decisive.

### Step 0: Establish a Measurement Baseline

Before any masking, build the evaluation harness. Pick 3–4 benchmarks that test different capabilities — GSM8K (math reasoning), HumanEval (code), IF-Eval (instruction following), and MMLU (general knowledge) is a reasonable spread. Run the unmasked student three times with different seeds on each benchmark to establish variance.

Record not just accuracy but also output length, format compliance, and held-out perplexity. These secondary metrics catch cases where masking leaves accuracy unchanged but destroys output coherence — a common failure mode that pure accuracy hides.

You need this baseline because every downstream claim takes the form "removing X changes performance by Δ." Without knowing the natural noise floor, you can't distinguish signal from variance.

### Step 1: Individual Token Knockout

Mask one Rock Token at a time by setting its logit to −∞ during generation, then run the full evaluation suite. This produces a 100-row table of marginal effects:

```
Token_ID | Token_Text | GSM8K_Δ | HumanEval_Δ | IFEval_Δ | Length_Δ | PPL_Δ
```

Sort by the GSM8K column. Tokens where removal hurts performance are Pillar candidates; tokens where removal helps are Stumbling Block candidates; tokens where removal does nothing are Neutral — high-loss but functionally inert. The Neutral category is probably the largest, and identifying it is itself a finding.

Also look for **task-specific effects**. A token that's a Pillar for math but a Stumbling Block for code is more interesting than one that's uniformly Pillar — it suggests token roles are task-dependent rather than intrinsic. Cross-task disagreement is a story worth telling.

Cost: 100 evaluation runs, no retraining. Roughly 1–2 GPU-days.

### Step 2: Categorize with Statistical Backing

Convert the continuous Δ scores into discrete categories you can defend.

For each token, run a paired bootstrap test against the baseline (resample the test set 1000 times) to get a p-value on Δ. Classify tokens as:

- **Strong Pillar**: Δ < −ε and significant
- **Weak Pillar**: Δ < 0 and significant
- **Neutral**: not significant
- **Weak Stumbling Block**: Δ > 0 and significant
- **Strong Stumbling Block**: Δ > ε and significant

Set ε from your Step 0 variance estimate, typically 1–2% absolute accuracy.

Then build a 100×4 matrix of (token × benchmark) categories and ask: how often does each token receive the same category across tasks? Tokens that are consistently Pillar everywhere are the safest claim. Tokens that flip categories across tasks are the most interesting claim.

Finally, correlate the categories with token features — entropy, frequency, POS tag, position in sequence. This tells you _why_ certain tokens are Pillars or Stumbling Blocks, not just _which_ ones are. If high-entropy Rock Tokens cluster as Pillars and low-entropy Rock Tokens cluster as Stumbling Blocks, you have a clean connection to the forking tokens literature.

### Step 3: Cumulative Removal Curves

Individual knockouts show marginal effects. Cumulative removal shows how effects compound and reveals the optimal removal fraction.

**Greedy Stumbling Block removal**: Sort tokens by Δ, most positive first. Mask cumulatively — top 1, top 2, top 5, top 10, top 20, ..., top 100 — and evaluate at each point. You expect the curve to rise initially (removing real Stumbling Blocks helps), plateau (Neutrals don't matter), then crash (you've started removing Pillars). **The location of the crash point is a key finding.** A crash at 60 means you can safely remove most Rock Tokens; a crash at 10 means most Rock Tokens are essential.

**Greedy Pillar removal**: Same procedure but sort most-negative-first. This curve drops immediately and the rate of drop tells you how concentrated Pillar effects are.

**Random removal baseline**: Repeat 5 times with random orderings to get a null-hypothesis curve. The gap between greedy-Stumbling-Block removal and random removal quantifies how much your ranking actually matters.

This produces the paper's most compelling figure — three curves on one plot. Reviewers respond well to this kind of visualization because it tells the entire story at a glance.

### Step 4: Structured Group Masking

Pairwise interaction testing across 100 tokens means 4,950 pairs — too expensive. Instead, test interactions at the level of meaningful groups, which is both cheaper and more interpretable.

**Semantic groups**: Cluster Rock Tokens by type from your Step 2 feature analysis — punctuation/formatting, reasoning connectives, math symbols, function words, rare vocabulary. Mask each group as a unit. If removing all punctuation Rock Tokens helps but no individual punctuation token had a significant effect, you've found a collective Stumbling Block effect that individual knockouts missed.

**Positional groups**: Split by where tokens appear in sequences — early, middle, late, or specifically at transition points (right after `\n\n` or step markers). If transition-position Rock Tokens are overwhelmingly Pillars, that connects directly to the forking tokens story.

**Entropy-stratified groups**: Split Rock Tokens into high-entropy (top half) and low-entropy (bottom half) by student entropy at those positions. The hypothesis worth testing: high-entropy Rocks are Pillars (the student is uncertain and _needs_ to be), low-entropy Rocks are Stumbling Blocks (the student is confidently wrong and wastes capacity defending that wrong answer).

Cost: ~15–20 evaluation runs total. Very cheap, very informative.

### Step 5: Pairwise Interactions (If Budget Allows)

From the prior steps, select the 10 strongest Pillars and 10 strongest Stumbling Blocks. Test all 100 cross-category pairs by masking both simultaneously and comparing the joint effect to the sum of individual effects.

Two outcomes are paper-worthy:

- **Rescue effects**: Removing a Stumbling Block compensates for removing a Pillar (joint damage is less than expected). This means the Stumbling Block was actively interfering with the Pillar's function — strong evidence the taxonomy reflects something real about model dynamics.
- **Amplification effects**: Removing both is worse than the sum. This means tokens depend on each other structurally.

Even a null result — no significant interactions, effects are approximately additive — is publishable. It justifies the greedy removal approach in Step 3 and simplifies the methodology for follow-up work.

### Step 6: Training-Time Masking — The Decisive Experiment

Everything above happens at inference time. The decisive question is whether removing Stumbling Blocks from the _training loss_ produces a measurably better final model. This is what moves the paper from "interesting analysis" to "actionable method."

The intervention is straightforward: re-run OPD from scratch, but zero out the loss contribution from tokens classified as Stumbling Blocks.

```python
for each token position t:
    kl_loss_t = KL(teacher || student at position t)
    if token_t in stumbling_blocks:
        kl_loss_t = 0
    total_loss += kl_loss_t
```

Run six configurations to make the claim airtight:

| Configuration        | What's masked                     | What it tests                                |
| -------------------- | --------------------------------- | -------------------------------------------- |
| Baseline             | Nothing                           | Standard OPD reference                       |
| Stumbling Block mask | Identified Stumbling Blocks       | The main claim                               |
| Random mask (same %) | Random tokens, matched proportion | Controls for "fewer loss terms = less noise" |
| Pillar mask          | Identified Pillars                | Should hurt — validates taxonomy             |
| All Rock Tokens      | All 100                           | Tests whether indiscriminate removal works   |
| Easy Token mask      | Low-loss tokens                   | Controls for "masking any category helps"    |

After each run converges, re-identify Rock Tokens in the new model. The set of Rock Tokens after Stumbling Block removal — compared to the original set — is what feeds into the Pillar rebuilding analysis: do new Rock Tokens emerge at different positions? Are they Pillar-like? This is where the "do Pillars rebuild?" question gets answered.

Cost: 6 full OPD runs. This is the expensive part — 2–3 weeks on 8×A100 — but it's what makes this a NeurIPS submission rather than a workshop note.

### Step 7: Synthesis

Pull the findings into 3–4 headline claims. Strong NeurIPS papers tend to have a structure like this:

- **The census** — "Of 100 Rock Tokens, X are Strong Pillars, Y are Stumbling Blocks, Z are Neutral." Sets the empirical landscape.
- **The efficiency curve** — "Removing the top K Stumbling Blocks improves benchmark Y by Δ% and accelerates convergence by factor F." Concrete, quotable, useful.
- **The feature story** — "Stumbling Blocks are predicted by [low entropy + non-transition position + high frequency]." Explains _why_, not just _what_.
- **The rebuilding story** — "After Stumbling Block removal, M new Rock Tokens emerge, of which most are Pillar-like at reasoning-critical positions." Suggests the student reorganizes its capacity allocation when freed from noisy gradients.

### Priority Ordering Under Compute Constraints

If you can't run everything:

1. **Step 1** (individual knockouts) — foundational, everything depends on it.
2. **Step 3** (cumulative curves) — produces the strongest figure.
3. **Step 4** (group masking) — cheap and consistently insightful.
4. **Step 6a–b** (training-time masking with at least baseline + Stumbling Block + random) — turns analysis into method.
5. Steps 2, 5, 7 — analysis you can do on existing data.

Steps 1 + 3 + 4 alone make a credible workshop paper. Adding Step 6 makes it a main conference contribution. Adding Step 5 plus the Step 7 rebuilding finding makes it a strong NeurIPS submission.

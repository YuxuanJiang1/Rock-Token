"""Verification script: confirm that logit_bias actually suppresses tokens
AND that vLLM is cross-session deterministic on A100 hardware.

Tests (within-session):
1. Tokenize " certain" (token_id 3654) and verify ID matches Part 1 CSV.
2. Run the same baseline twice — check within-session determinism.
3. Adaptive single-token suppression (pick a token from baseline output, mask it).
4. Mask top-5 in-baseline content tokens simultaneously.
5. Run the same MASKED config twice — check within-session determinism with logit_bias.
6. Mask 200 token IDs simultaneously — verify all suppressed.

Cross-session test (run script twice as separate processes):
7. Save the baseline output to disk on first run; on subsequent runs compare
   the new baseline (this session) to the saved one (prior session). PASS means
   vLLM is producing identical outputs across separate Python processes.

Cross-session determinism on A100 hardware requires VLLM_ENABLE_V1_MULTIPROCESSING=0
(set automatically by src.masking.common at import time).

Usage:
    uv run python src/masking/verify_masking.py \\
        --model RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k
    # Then run again to verify cross-session reproducibility:
    uv run python src/masking/verify_masking.py \\
        --model RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k
"""

import argparse
from pathlib import Path

from src.masking.common import create_llm


PROMPT_MATH = (
    "Find the smallest positive integer n such that n^2 is divisible by 12. "
    "Reason step by step and put your final answer in \\boxed{}."
)
PROMPT_TEXT = "Tell me about cats. Use about 50 words."


def test_logit_bias(model_name: str, max_tokens: int = 512):
    from vllm import SamplingParams

    print("=" * 70)
    print("VERIFICATION: vLLM logit_bias behavior")
    print("=" * 70)

    print(f"\nLoading model: {model_name}")
    llm = create_llm(model_name)
    tokenizer = llm.get_tokenizer()

    # --- Test 1: Token ID consistency ---
    print("\n[Test 1] Token ID for ' certain'")
    ids = tokenizer.encode(" certain", add_special_tokens=False)
    print(f"  Encoded: {ids}")
    print(f"  First ID: {ids[0]} (expected 3654 from Part 1 CSV)")
    certain_id = ids[0]

    ids_the = tokenizer.encode(" the", add_special_tokens=False)
    print(f"\n  ' the' encoded: {ids_the}")
    the_id = ids_the[0]

    # --- Test 2: Baseline determinism ---
    print("\n[Test 2] Baseline determinism (same seed, same prompt, two runs)")
    conv = [[{"role": "user", "content": PROMPT_MATH}]]
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, seed=42, top_k=1)
    out1 = llm.chat(conv, sp)[0].outputs[0].text
    out2 = llm.chat(conv, sp)[0].outputs[0].text
    print(f"  Run 1 length: {len(out1)} chars")
    print(f"  Run 2 length: {len(out2)} chars")
    print(f"  Identical: {out1 == out2}")
    if out1 != out2:
        # Find first difference
        for i, (a, b) in enumerate(zip(out1, out2)):
            if a != b:
                print(f"  First diff at char {i}:")
                print(f"    Run 1: ...{out1[max(0,i-30):i+30]!r}...")
                print(f"    Run 2: ...{out2[max(0,i-30):i+30]!r}...")
                break

    # --- Test 3: Pick a token that ACTUALLY appears in the baseline output and mask it ---
    print("\n[Test 3] Adaptive token-suppression test (pick a token that's in the baseline output)")
    # Tokenize baseline output, find top-frequent content tokens
    base_token_ids = tokenizer.encode(out1, add_special_tokens=False)
    from collections import Counter
    counts = Counter(base_token_ids)
    # Pick the most common non-special token (decode and skip very short / whitespace ones)
    candidates = []
    for tid, count in counts.most_common(30):
        decoded = tokenizer.decode([tid])
        if len(decoded.strip()) >= 2 and count >= 3:
            candidates.append((tid, decoded, count))
    if not candidates:
        print("  Could not find any common token — skipping")
    else:
        target_tid, target_text, target_count = candidates[0]
        print(f"  Probe token: {target_text!r} (id={target_tid}, appears {target_count}x in baseline)")

        sp_probe = SamplingParams(
            temperature=0, max_tokens=max_tokens, seed=42, top_k=1,
            logit_bias={target_tid: -100.0},
        )
        out_probe = llm.chat(conv, sp_probe)[0].outputs[0].text
        probe_token_ids = tokenizer.encode(out_probe, add_special_tokens=False)
        probe_count = sum(1 for t in probe_token_ids if t == target_tid)

        print(f"  Token id {target_tid} count in baseline output: {target_count}")
        print(f"  Token id {target_tid} count in masked output:   {probe_count}")
        print(f"  Output identical to baseline: {out_probe == out1}")
        if probe_count == 0:
            print(f"  ✓ PASS: token {target_text!r} was {target_count}x in baseline, 0 in masked")
            test3_pass = True
        else:
            print(f"  ✗ FAIL: token {target_text!r} still appears {probe_count}x in masked output!")
            test3_pass = False

    # --- Test 4: Mask 5 of the most common tokens at once ---
    print("\n[Test 4] Mask the top-5 most-common content tokens from baseline")
    if len(candidates) >= 5:
        target_ids = [c[0] for c in candidates[:5]]
        target_texts = [c[1] for c in candidates[:5]]
        baseline_counts = [c[2] for c in candidates[:5]]
        print(f"  Probe tokens: {list(zip(target_texts, baseline_counts))}")

        sp_probe5 = SamplingParams(
            temperature=0, max_tokens=max_tokens, seed=42, top_k=1,
            logit_bias={tid: -100.0 for tid in target_ids},
        )
        out_probe5 = llm.chat(conv, sp_probe5)[0].outputs[0].text
        probe5_token_ids = tokenizer.encode(out_probe5, add_special_tokens=False)
        target_set = set(target_ids)
        appearing = [t for t in probe5_token_ids if t in target_set]

        print(f"  Total appearances of probe tokens in masked output: {len(appearing)} (should be 0)")
        print(f"  Output differs from baseline: {out_probe5 != out1}")
        if len(appearing) == 0:
            print(f"  ✓ PASS: all 5 probe tokens fully suppressed")
            test4_pass = True
        else:
            print(f"  ✗ FAIL: {len(appearing)} appearances of supposedly-masked tokens")
            test4_pass = False
    else:
        test4_pass = None
        print("  Not enough candidates — skipping")

    # --- Test 5: Determinism with logit_bias ---
    print("\n[Test 5] Determinism with logit_bias (same masked config, two runs)")
    out_m1 = llm.chat(conv, sp_probe5)[0].outputs[0].text
    out_m2 = llm.chat(conv, sp_probe5)[0].outputs[0].text
    print(f"  Identical: {out_m1 == out_m2}")
    if out_m1 != out_m2:
        for i, (a, b) in enumerate(zip(out_m1, out_m2)):
            if a != b:
                print(f"  First diff at char {i}:")
                print(f"    Run 1: ...{out_m1[max(0,i-30):i+30]!r}...")
                print(f"    Run 2: ...{out_m2[max(0,i-30):i+30]!r}...")
                break

    # --- Test 6: Multi-token mask ---
    print("\n[Test 6] Mask 200 tokens at once (greedy_pillar k=200)")
    # Just use the encoded ids of common math vocabulary as a stand-in
    # We mask 200 random Rock Token IDs to verify multi-token masking works
    test_ids = list(range(1000, 1200))  # 200 token IDs
    sp_multi = SamplingParams(
        temperature=0, max_tokens=max_tokens, seed=42,
        logit_bias={tid: -100.0 for tid in test_ids},
    )
    out_multi = llm.chat(conv, sp_multi)[0].outputs[0].text

    # Re-decode the output to check none of the masked IDs appear
    output_token_ids = tokenizer.encode(out_multi, add_special_tokens=False)
    masked_set = set(test_ids)
    appearing = [tid for tid in output_token_ids if tid in masked_set]
    print(f"  Output length: {len(out_multi)} chars, {len(output_token_ids)} tokens")
    print(f"  Output identical to baseline: {out_multi == out1}")
    print(f"  Masked token IDs appearing in output: {len(appearing)} (should be 0)")

    # --- Test 7: Cross-session reproducibility ---
    print("\n[Test 7] Cross-session reproducibility")
    print("  (Saves baseline output on first run; on later runs, compares new baseline to saved)")
    cache_path = Path("results/masking/verify_cache/baseline_session_a.txt")
    cross_session_pass = None
    if cache_path.exists():
        prior = cache_path.read_text()
        prior_len = len(prior)
        new_len = len(out1)
        identical = prior == out1
        print(f"  Found prior session output ({prior_len} chars)")
        print(f"  This session output ({new_len} chars)")
        print(f"  Identical: {identical}")
        if not identical:
            for i, (a, b) in enumerate(zip(prior, out1)):
                if a != b:
                    print(f"  First diff at char {i}:")
                    print(f"    Prior:   ...{prior[max(0,i-30):i+30]!r}...")
                    print(f"    Current: ...{out1[max(0,i-30):i+30]!r}...")
                    break
        cross_session_pass = identical
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(out1)
        print(f"  No prior cache found. Saved this session's baseline to {cache_path}")
        print(f"  ▸ Run this script AGAIN as a separate process to test cross-session determinism.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline reproducible (within session):  {out1 == out2}")
    print(f"  Test 3 — single in-baseline token mask:  {'PASS' if test3_pass else 'FAIL'}")
    print(f"  Test 4 — top-5 in-baseline tokens mask:  {'PASS' if test4_pass else 'FAIL'}")
    print(f"  Masked run reproducible:                 {out_m1 == out_m2}")
    print(f"  Test 6 — 200 token mask suppresses all:  {'PASS' if len(appearing) == 0 else 'FAIL'}")
    if cross_session_pass is None:
        print(f"  Test 7 — cross-session reproducible:     PENDING (run again to compare)")
    else:
        print(f"  Test 7 — cross-session reproducible:     {'PASS' if cross_session_pass else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    test_logit_bias(args.model, args.max_tokens)


if __name__ == "__main__":
    main()

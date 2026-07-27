# W4 ablations — run order (4x L40S)

The launch scripts have several placeholders copied from `run_stumb_random.sh`'s original
HPC account (`xxj1`) — they will fail immediately (blank `KD_ROOT`, someone else's conda env,
someone else's model/data paths) until you fill in your own. Steps below in order.

## 0. Environment

```bash
git clone https://github.com/songmzhang/KDFlow.git   # or: just use stumbling/kdflow directly,
                                                        # it's already a KDFlow checkout with
                                                        # token_freeze_kd added
cd stumbling
pip install -e ./
pip install flash_attn==2.8.3 --no-build-isolation
```
Or use the Docker image referenced in `stumbling/docker/Dockerfile.sgl059.torch291.cu128`
(sglang 0.5.9 + torch 2.9.1 + cu12.8) — this is the environment the FP8 teacher quantization
flag was validated against, so prefer it if you have a choice.

Sanity check on the box: `nvidia-smi` shows 4 L40S, `nvcc --version` matches `CUDA_HOME` in the
scripts (currently `/usr/local/cuda-12.8` — edit if yours differs).

## 1. Get the raw Section 2 artifacts

Needed: `rock_token_occurrences_onpolicy_n500_unrestricted.pt`,
`rock_vs_control_unrestricted.csv`, and `logit_gradients_onpolicy_n500_unrestricted.pt` (per
`rock_detection/README.md`). These are not checked into git (~50MB/~3.6GB) and are **not** on the
`RockToken` HF org either (checked — that org has model checkpoints and two small prompt
datasets, no raw statistics files). **Check first whether these already exist on disk somewhere**
(wherever Fig. 2/3 were originally generated) before regenerating — it's real GPU time on top of
the 5 training runs.

If they don't exist, run (from `rock_detection/`, or with the script's full path — it no longer
matters what your current directory is, see below):
```bash
pip install torch transformers datasets accelerate pandas numpy scipy matplotlib tqdm
chmod +x run_regen_quad_l40s.sh
./run_regen_quad_l40s.sh 2>&1 | tee regen.log
```
This runs the full chain end-to-end: `rock_server.py` (student rollout + teacher scoring,
`--hardware quad_l40s`, ~3-4h on one H100 per the README, expect longer on L40S — no NVLink means
the teacher's 4-way pipeline placement pays extra inter-GPU transfer cost) → `rerun_unrestricted.py`
(re-selects the rock/control lists → `rock_vs_control_unrestricted.csv`) →
`compute_logit_gradients.py` (~1-2h on H100, forward-only so faster than generation). No
quantization needed here — this is plain HF `transformers` inference (not the SGLang/FSDP2
training stack), and 4x48GB=192GB comfortably holds the ~8GB student + ~60GB bf16 teacher.

**Output location is fixed, not cwd-dependent**: all three artifacts always land in
`$HOME/rocktoken_regen/`, regardless of what directory the script is launched from or by whom
(useful if someone else — a labmate, a teammate — is the one actually running it and you don't
know/control their cwd). The script resolves its own location internally to find the `.py` files,
but `cd`s to `$HOME/rocktoken_regen` before writing anything.

**Resumable**: each of the 3 steps is skipped if its output file already exists in
`$HOME/rocktoken_regen/`, so re-running after a downstream failure (like the step-3 bug, now fixed)
picks up where it left off instead of redoing multi-hour steps. `FORCE_RERUN=1 ./run_regen_quad_l40s.sh`
forces everything from scratch.

Two small fixes made to `rock_server.py`/`compute_logit_gradients.py` while adapting them:
- Added a `quad_l40s` hardware choice (functionally identical to `dual_40gb`'s
  `device_map="auto"` placement — added only so run logs correctly report 4 GPUs, not 2).
- Removed a hardcoded `/workspace/hf_cache` path (a leftover from the original author's pod
  layout); caching now respects `HF_HOME` (the launch script sets it to `~/.cache/huggingface`
  by default — override by exporting `HF_HOME` before running if you want it elsewhere).
- Also fixed a stale model ID: `STUDENT_MODELS["offpolicy"]` pointed at
  `RockToken/qwen3_30b_a3b_to_4b_offpolicy_math_first20k`, which 404s — the repo now on HF is
  `RockToken/qwen3_30b_a3b_to_4b_offpolicy_20k`. Not on the critical path for W4 (only the
  `onpolicy` key is used), but would have broken later if anyone reused these scripts for the
  off-policy comparison.

## 2. Build the four freeze-list JSONs

Its `--occurrences`/`--gradients`/`--rock-csv` defaults already point at
`$HOME/rocktoken_regen/` (step 1's fixed output location), and `--outdir` defaults to
`$HOME/rocktoken_ablation_lists/` — so if step 1 used its defaults, this needs no arguments at all,
from any directory:
```bash
python /path/to/stumbling/build_ablation_freeze_lists.py
```
(pass explicit `--occurrences`/`--gradients`/`--rock-csv`/`--outdir` only if step 1's output ended
up somewhere else, or you want the lists written elsewhere)

Read `build_report.txt` in the output dir before moving on (or the console output directly) — in
particular the `ablation_lists_summary.csv` Jaccard-overlap-with-rock column. Also copy the
existing `rock.json` into that same output dir (the soft-λ script reads `rock.json` from there too).

## 3. Point the scripts at your own paths

`--token_freeze_path` is now auto-resolved (added after the first W4 run: each script computes its
own `SCRIPT_DIR` and points at `${SCRIPT_DIR}/ablation_lists/<name>.json`, or `${SCRIPT_DIR}/rock.json`
for the soft-λ script) — no edit needed there as long as `build_ablation_freeze_lists.py` was run
with its defaults (writes to `stumbling/ablation_lists/`) and `rock.json` stays where it already is
in `stumbling/`.

Still to edit, in **each** of `run_stumb_top_freq.sh`, `run_stumb_top_meanloss.sh`,
`run_stumb_gradmag.sh`, `run_stumb_gradalign.sh`, `run_stumb_soft_lambda.sh`:

| Variable | Currently | Set to |
|---|---|---|
| `KD_ROOT` | *(blank)* | path to your `stumbling/` checkout (so `new_runner` and `kdflow` are importable) |
| `PYTHON` / `RAY` | `/home/xxj1/.conda/envs/qwen/bin/{python,ray}` | your own conda/venv's `python`/`ray` |
| `STUDENT_MODEL` | `/p/work2/xxj1/opd/models/Qwen3-4B-Instruct-2507` | your local copy, or just `Qwen/Qwen3-4B-Instruct-2507` (HF hub id — `--student_name_or_path` accepts either) |
| `TEACHER_MODEL` | `/p/work2/xxj1/opd/models/Qwen3-30B-A3B` | your local copy, or `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| `TRAIN_DATA` | `/p/work2/xxj1/opd/data/.../openthoughts_prompt_math_10k.jsonl` | your copy of that prompt slice |
| `SAVE_DIR` | `/p/work2/xxj1/rocktoken/stumb/<name>` | wherever you want checkpoints written — needs to be writable and have room for a 4B-parameter checkpoint |

If you're running on the *same* cluster/account the original `run_stumb_random.sh` ran on, most
of these may already be correct — just confirm rather than blindly editing.

## 4. Make executable and run

```bash
chmod +x run_stumb_*.sh
./run_stumb_top_freq.sh       2>&1 | tee top_freq.log
```

Run these **sequentially, not in parallel** — each wants all 4 GPUs. Order doesn't matter; I'd
suggest `top_meanloss` first since it's the one closest to what the reviewer explicitly asked for
("top-mean-loss freeze without the Rock Score construction"), so you get the most decision-relevant
result first if time runs short.

For the soft-λ sweep:
```bash
./run_stumb_soft_lambda.sh 0.3 2>&1 | tee soft_03.log
./run_stumb_soft_lambda.sh 0.5 2>&1 | tee soft_05.log
./run_stumb_soft_lambda.sh 0.7 2>&1 | tee soft_07.log
```

## 5. While it's running, watch for

- `[TokenFreezeKD] freeze hits: X / Y` printed every step (from `token_freeze_kd.py`) — confirms
  the freeze list is actually matching tokens in the batch. If `X` is consistently 0, the token
  IDs in the JSON don't match this tokenizer/vocab — stop and check before burning more GPU time.
- OOM early (first few steps) is the main L40S risk. If it happens, in this order: (1) drop
  `--teacher_quantization fp8` only if it's a kernel/dtype error, not a genuine OOM — removing it
  makes memory pressure *worse*; (2) lower `--teacher_mem_fraction_static` (0.45 → 0.35) and/or
  `--rollout_mem_fraction_static` (0.12 → 0.08); (3) as a last resort, lower `--generate_max_len` /
  `--max_len` / `--prompt_max_len`.
- Ray/sglang leftover processes from a previous run — the script already does
  `ray stop --force` + `pkill sglang` at the top, but if a run crashed hard, check `nvidia-smi`
  for orphaned processes before launching the next one.

## 6. Evaluation (not scripted here)

I don't have visibility into the exact eval command used to produce the original Fig. 5 numbers —
the paper cites LM-EVALUATION-HARNESS but I don't see that invocation checked into this repo.
Whatever process produced the "Original OPD" / "Random" / "Rock-Freeze" AIME24/25+HMMT25 numbers,
repeat it against each new run's final checkpoint in `SAVE_DIR`. If you point me at that eval
script (or tell me it doesn't exist yet and needs writing), I can prep launch configs for it the
same way I did for training.

## 7. Feed results back

Fill in the `TBD` cells in `rebuttal/RNAB_W4.md`'s results table, and let me know — I'll write the
interpretation paragraph and finalize the OpenReview reply once numbers are in.

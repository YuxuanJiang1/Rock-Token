#!/usr/bin/env bash
# Evaluates the 7 W4 ablation checkpoints (Reviewer RNAB) on AIME24/AIME25/HMMT25feb,
# adapted from run_eval_vllm.sh (which was written for a different collaborator's
# account/paths and only evaluates one hardcoded model).
#
# Fixes relative to the original script:
#   1. Adds --include_path so lm_eval can find the custom task YAMLs in tasks/ --
#      the original relied on task files already being inside a separate full
#      harness clone (HARNESS=/p/work2/yuxuanj1/opd/lm-evaluation-harness); we're
#      using a plain `pip install lm-eval`, which doesn't know about custom tasks
#      unless told where to look.
#   2. Loops over all 7 of our checkpoints instead of one hardcoded MODEL path.
#   3. Self-locating (SCRIPT_DIR) and resumable (.done markers per
#      checkpoint+seed), matching the pattern used in stumbling/run_all_ablations.sh.
#   4. Seed count is configurable (NUM_SEEDS) rather than hardcoded to 3 -- the
#      paper's Section 5.2 says "averaged over five independent runs" but the
#      original script only does 3; start with NUM_SEEDS=1 to get a real timing
#      number before committing to a full sweep.
#
# If vLLM's CUDA graph capture hits the same class of JIT-compile error we saw
# during training (ninja/nvcc failure), the fix is the same idea as
# --disable-cuda-graph there: add `,enforce_eager=True` to VLLM_MODEL_ARGS below
# (costs some throughput, trades off against reliability).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
EVAL_OUT="${SCRIPT_DIR}/eval_results"
mkdir -p "${LOG_DIR}" "${EVAL_OUT}"

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Pin the interpreter instead of trusting PATH. `python` has resolved to a DIFFERENT
# venv at different times today (.venv vs .venv-train), and .venv-train had no lm_eval
# at all, so a PATH-dependent eval silently ran against the wrong stack.
# .venv-train is now the unified env: torch 2.9.1 + sglang 0.5.9 (train) AND
# vllm 0.16.0 + lm_eval (eval). vllm 0.14-0.16 pin torch==2.9.1, which is what makes
# one venv possible; vllm >=0.20 pins torch==2.11.0 and cannot coexist with sglang.
VENV_DIR=${VENV_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)/.venv-train}
PY="${VENV_DIR}/bin/python"
[ -x "$PY" ] || { echo "ERROR: $PY not found or not executable." >&2; exit 1; }

# The .eval_overlay + VLLM_USE_FLASHINFER_SAMPLER=0 workarounds exist ONLY for
# vllm 0.26.0 in .venv, where three things were installed-but-broken:
#   openai 2.6.1     -> missing NamespaceTool, dies at `from vllm import LLM`
#   flashinfer       -> python 0.6.14 vs cubin 0.6.3, RuntimeError on any import
#   flash_attn       -> built for torch 2.9, ABI-broken against torch 2.11
# .venv-train has all three healthy, so forcing the overlay there would needlessly
# hide working components and disable the flashinfer sampler for no reason.
# Probe rather than assume: apply the overlay only if vllm cannot import without it.
if "$PY" -c "from vllm import LLM" >/dev/null 2>&1; then
  echo "vllm imports cleanly in ${VENV_DIR} -- no overlay needed"
else
  EVAL_OVERLAY="$(cd "${SCRIPT_DIR}/.." && pwd)/.eval_overlay"
  if [ ! -d "${EVAL_OVERLAY}/openai" ]; then
    echo "ERROR: vllm will not import and ${EVAL_OVERLAY} is missing. Recreate with:" >&2
    echo "  uv pip install --python ${PY} --target ${EVAL_OVERLAY} --no-deps openai==2.25.0" >&2
    exit 1
  fi
  echo "vllm needs the compat overlay -- enabling ${EVAL_OVERLAY}"
  export PYTHONPATH="${EVAL_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi

CHECKPOINTS_ROOT="/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token-checkpoints"
CHECKPOINT_NAMES=(top_freq top_meanloss top_gradmag top_gradalign soft_lambda_3 soft_lambda_5 soft_lambda_7)
# Override to test a subset first, e.g.: CHECKPOINT_NAMES=(top_meanloss) ./run_eval_ablations.sh
if [ "${CHECKPOINTS:-}" != "" ]; then
  read -ra CHECKPOINT_NAMES <<< "${CHECKPOINTS}"
fi

TASKS="aime24_sample,aime25_sample,hmmt25feb"
NUM_SEEDS=${NUM_SEEDS:-1}
FORCE_RERUN=${FORCE_RERUN:-0}

echo "==== ENV CHECK ===="
"$PY" -V
"$PY" -c "import lm_eval, vllm, openai; print('lm_eval', lm_eval.__version__); print('vllm', vllm.__version__); print('openai', openai.__version__, '(overlay)' if 'eval_overlay' in openai.__file__ else '(venv)')"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo "Checkpoints: ${CHECKPOINT_NAMES[*]}"
echo "Seeds: NUM_SEEDS=${NUM_SEEDS}"
echo "Output: ${EVAL_OUT}"

for NAME in "${CHECKPOINT_NAMES[@]}"; do
  MODEL="${CHECKPOINTS_ROOT}/${NAME}"
  if [ ! -f "${MODEL}/config.json" ]; then
    echo "ERROR: ${MODEL}/config.json not found -- skipping ${NAME}." >&2
    continue
  fi

  for RUN in $(seq 1 "${NUM_SEEDS}"); do
    SEED=$((42 + RUN * 17))
    OUT_DIR="${EVAL_OUT}/${NAME}/run_${RUN}"
    LOG_FILE="${LOG_DIR}/${NAME}_run${RUN}.log"
    MARKER="${LOG_DIR}/${NAME}_run${RUN}.done"

    if [ "${FORCE_RERUN}" != "1" ] && [ -f "${MARKER}" ]; then
      echo "=== Skipping ${NAME} run ${RUN} -- already done. FORCE_RERUN=1 to redo. ==="
      continue
    fi

    mkdir -p "${OUT_DIR}"
    echo ""
    echo "============================================================"
    echo "  ${NAME}  RUN ${RUN}/${NUM_SEEDS}  seed=${SEED}  $(date)"
    echo "============================================================"

    START=$(date +%s)
    if "$PY" -m lm_eval \
        --model vllm \
        --model_args "pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.9,max_model_len=20000,enforce_eager=False" \
        --tasks ${TASKS} \
        --include_path "${SCRIPT_DIR}/tasks" \
        --num_fewshot 0 \
        --seed ${SEED} \
        --batch_size auto \
        --output_path "${OUT_DIR}" \
        --log_samples \
        > "${LOG_FILE}" 2>&1; then
      touch "${MARKER}"
      ELAPSED=$(( $(date +%s) - START ))
      echo "[${NAME} run ${RUN}] done in ${ELAPSED}s ($(date))"
    else
      echo "[${NAME} run ${RUN}] FAILED -- see ${LOG_FILE}" >&2
    fi
  done
done

echo ""
echo "============================================================"
echo "  AGGREGATING RESULTS  $(date)"
echo "============================================================"

"$PY" - "${EVAL_OUT}" "${NUM_SEEDS}" "${CHECKPOINT_NAMES[@]}" << 'PY'
import json, glob, statistics, os, sys

eval_out = sys.argv[1]
num_seeds = int(sys.argv[2])
checkpoint_names = sys.argv[3:]
tasks = ["aime24_sample", "aime25_sample", "hmmt25feb"]

print(f"\n{'Checkpoint':<16} {'aime24':>9} {'aime25':>9} {'hmmt25feb':>9} {'Avg':>9}  (mean over {num_seeds} run(s), std across seeds where >1)")
print("-" * 75)

summary_rows = []
for name in checkpoint_names:
    task_results = {t: [] for t in tasks}
    for run in range(1, num_seeds + 1):
        files = glob.glob(f"{eval_out}/{name}/run_{run}/**/*.json", recursive=True)
        for f in files:
            if "samples" in os.path.basename(f):
                continue
            try:
                with open(f) as fp:
                    data = json.load(fp)
            except Exception:
                continue
            if "results" not in data:
                continue
            for task in tasks:
                if task in data["results"]:
                    r = data["results"][task]
                    acc = r.get("exact_match,none", r.get("exact_match", None))
                    if acc is not None:
                        task_results[task].append(round(acc * 100, 2))

    row = {"checkpoint": name}
    means = []
    cells = []
    for task in tasks:
        vals = task_results[task]
        if vals:
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            means.append(m)
            cells.append(f"{m:>6.1f}{'±'+format(s,'.1f') if len(vals) > 1 else '':<6}")
            row[task] = m
        else:
            cells.append(f"{'N/A':>9}")
            row[task] = None
    avg = statistics.mean(means) if means else None
    row["avg"] = avg
    summary_rows.append(row)
    avg_str = f"{avg:>9.1f}" if avg is not None else f"{'N/A':>9}"
    print(f"{name:<16} " + " ".join(f"{c:>9}" for c in cells) + f" {avg_str}")

summary_path = os.path.join(eval_out, "summary.csv")
import csv
with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["checkpoint", "aime24_sample", "aime25_sample", "hmmt25feb", "avg"])
    for row in summary_rows:
        w.writerow([row["checkpoint"], row.get("aime24_sample"), row.get("aime25_sample"), row.get("hmmt25feb"), row.get("avg")])
print(f"\nSaved {summary_path}")
PY

echo "Done."

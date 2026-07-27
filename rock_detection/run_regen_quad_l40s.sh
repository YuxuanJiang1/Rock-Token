#!/usr/bin/env bash
# Regenerate the raw Section-2 artifacts needed for build_ablation_freeze_lists.py
# (W4 ablations, Reviewer RNAB) on 4x L40S instead of the paper's H100 server.
#
# Unlike the stumbling/ training scripts, this is plain HF `transformers`
# inference (student.generate() + teacher forward pass) -- no Ray/SGLang/FSDP2,
# so it's much simpler: device_map="auto" just needs enough total GPU memory to
# hold student (~8GB bf16) + teacher (~60GB bf16) + KV cache/activations, which
# 4x48GB=192GB comfortably covers. No quantization needed here.
#
# Expect this to take noticeably longer than the paper's H100 estimates
# (rock_server.py ~3-4h, compute_logit_gradients.py ~1-2h on one H100): L40S has
# lower per-GPU throughput than H100, and device_map="auto" pipelines the teacher
# across 4 non-NVLinked GPUs, adding inter-GPU transfer latency at each shard
# boundary that a single H100 (or NVLinked pair) doesn't pay. Consider running
# this unattended, or smoke-testing with --samples 20 first.
#
# Output location: whoever runs this (you, a labmate, a cluster teammate) may
# invoke it from any working directory. Outputs always land in THIS script's
# own directory (rock_detection/, wherever the repo happens to be checked
# out) rather than wherever it was launched from -- e.g.
# `bash /some/path/rock_detection/run_regen_quad_l40s.sh` from $HOME still
# writes into /some/path/rock_detection/. build_ablation_freeze_lists.py's
# --occurrences/--gradients/--rock-csv defaults point at ../rock_detection
# relative to its own location in stumbling/, so the two scripts chain
# together with zero path arguments as long as the repo layout is intact.
set -e
set -x

CUDA_HOME=/usr/local/cuda-12.8   # verify this matches your box; only used for nvcc sanity check below
export PATH=$CUDA_HOME/bin:$PATH

# GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Where HF should cache downloaded models/datasets. The original script hardcoded
# "/workspace/hf_cache" (a pod-provider convention); that hardcoding has been
# removed from rock_server.py so this env var is now what actually controls it.
# Point it at a disk with room for ~70GB of model weights + MATH-500.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
mkdir -p "$HF_HOME"

# Absolute path to this script's own directory (i.e. wherever rock_detection/
# actually lives). Both used to invoke the *.py scripts by absolute path and as
# the fixed output location -- works no matter what cwd this was launched from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==== ENV CHECK ===="
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
which nvcc && nvcc --version || echo "nvcc not found -- fine, this script doesn't compile CUDA code"
echo "HF_HOME=$HF_HOME"
echo "SCRIPT_DIR=$SCRIPT_DIR (everything below is written here)"

# rerun_unrestricted.py (step 2 below) hardcodes n=500 in its filenames/constants,
# not an argparse flag, so this pipeline only works end-to-end at SAMPLES=500.
# Use a smaller value only to smoke-test rock_server.py in isolation.
SAMPLES=${SAMPLES:-500}

# rerun_unrestricted.py needs the *existing* (256-cap) rock_vs_control.csv in its
# cwd, purely to report overlap with the new unrestricted selection -- it
# currently only lives under stumbling/. Grab it if we don't have a copy yet.
if [ ! -f "rock_vs_control.csv" ]; then
  if [ -f "$SCRIPT_DIR/../stumbling/rock_vs_control.csv" ]; then
    cp "$SCRIPT_DIR/../stumbling/rock_vs_control.csv" .
    echo "Copied rock_vs_control.csv from $SCRIPT_DIR/../stumbling/"
  else
    echo "ERROR: rock_vs_control.csv not found in $SCRIPT_DIR or at $SCRIPT_DIR/../stumbling/rock_vs_control.csv." >&2
    echo "       rerun_unrestricted.py needs it (for overlap reporting) before continuing." >&2
    exit 1
  fi
fi

# Resumable: each step is skipped if its output already exists on disk, so a
# re-run after a downstream failure (e.g. step 3's bug, now fixed) doesn't waste
# hours redoing step 1's generation. Set FORCE_RERUN=1 to redo everything anyway.
FORCE_RERUN=${FORCE_RERUN:-0}

OCC_FILE="rock_token_occurrences_onpolicy_n${SAMPLES}_unrestricted.pt"
ROCK_CSV="rock_vs_control_unrestricted.csv"
GRAD_FILE="logit_gradients_onpolicy_n${SAMPLES}_unrestricted.pt"

# =========================
# 1. Collect per-token KL statistics (student generation + teacher scoring)
# =========================
if [ "${FORCE_RERUN}" = "1" ] || [ ! -f "${OCC_FILE}" ]; then
  python "$SCRIPT_DIR/rock_server.py" \
    --student onpolicy \
    --samples ${SAMPLES} \
    --hardware quad_l40s \
    --unrestricted

  if [ ! -f "${OCC_FILE}" ]; then
    echo "ERROR: expected ${OCC_FILE} to exist after rock_server.py -- check the run log above." >&2
    exit 1
  fi
else
  echo "Skipping step 1 -- ${OCC_FILE} already exists. (FORCE_RERUN=1 to redo.)"
fi

# =========================
# 2. Re-select rocks/controls on the fresh data -> rock_vs_control_unrestricted.csv
#    (also produces a funnel plot + per-output distribution CSV as side effects)
# =========================
if [ "${FORCE_RERUN}" = "1" ] || [ ! -f "${ROCK_CSV}" ]; then
  python "$SCRIPT_DIR/rerun_unrestricted.py"

  if [ ! -f "${ROCK_CSV}" ]; then
    echo "ERROR: expected ${ROCK_CSV} to exist after rerun_unrestricted.py -- check the run log above." >&2
    exit 1
  fi
else
  echo "Skipping step 2 -- ${ROCK_CSV} already exists. (FORCE_RERUN=1 to redo.)"
fi

# =========================
# 3. Per-token gradient directions (forward-only, reuses tokens from step 1)
# =========================
if [ "${FORCE_RERUN}" = "1" ] || [ ! -f "${GRAD_FILE}" ]; then
  python "$SCRIPT_DIR/compute_logit_gradients.py" \
    --student onpolicy \
    --samples ${SAMPLES} \
    --hardware quad_l40s \
    --occurrences-file "${OCC_FILE}"
else
  echo "Skipping step 3 -- ${GRAD_FILE} already exists. (FORCE_RERUN=1 to redo.)"
fi

echo "==== DONE ===="
echo "Everything written to: ${SCRIPT_DIR}"
echo "Occurrences: ${SCRIPT_DIR}/${OCC_FILE}"
echo "Rock list:   ${SCRIPT_DIR}/${ROCK_CSV}"
echo "Gradients:   ${SCRIPT_DIR}/${GRAD_FILE}"
echo ""
echo "Next: build_ablation_freeze_lists.py's --occurrences/--gradients/--rock-csv"
echo "defaults already point at ${SCRIPT_DIR} (../rock_detection relative to"
echo "stumbling/), so from anywhere:"
echo "  python $SCRIPT_DIR/../stumbling/build_ablation_freeze_lists.py"
echo "(results land in stumbling/ itself by default, next to rock.json /"
echo " rock_vs_control.csv -- add --outdir <path> to override)"

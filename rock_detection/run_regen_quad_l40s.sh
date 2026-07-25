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

echo "==== ENV CHECK ===="
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
which nvcc && nvcc --version || echo "nvcc not found -- fine, this script doesn't compile CUDA code"
echo "HF_HOME=$HF_HOME"

# rerun_unrestricted.py (step 2 below) hardcodes n=500 in its filenames/constants,
# not an argparse flag, so this pipeline only works end-to-end at SAMPLES=500.
# Use a smaller value only to smoke-test rock_server.py in isolation.
SAMPLES=${SAMPLES:-500}

# rerun_unrestricted.py needs the *existing* (256-cap) rock_vs_control.csv in its
# cwd, purely to report overlap with the new unrestricted selection -- it currently
# only lives under stumbling/. Grab it if we don't have our own copy yet.
if [ ! -f "rock_vs_control.csv" ]; then
  if [ -f "../stumbling/rock_vs_control.csv" ]; then
    cp "../stumbling/rock_vs_control.csv" .
    echo "Copied rock_vs_control.csv from ../stumbling/"
  else
    echo "ERROR: rock_vs_control.csv not found here or at ../stumbling/rock_vs_control.csv." >&2
    echo "       rerun_unrestricted.py needs it (for overlap reporting) before continuing." >&2
    exit 1
  fi
fi

# =========================
# 1. Collect per-token KL statistics (student generation + teacher scoring)
# =========================
python rock_server.py \
  --student onpolicy \
  --samples ${SAMPLES} \
  --hardware quad_l40s \
  --unrestricted

OCC_FILE="rock_token_occurrences_onpolicy_n${SAMPLES}_unrestricted.pt"

if [ ! -f "${OCC_FILE}" ]; then
  echo "ERROR: expected ${OCC_FILE} to exist after rock_server.py -- check the run log above." >&2
  exit 1
fi

# =========================
# 2. Re-select rocks/controls on the fresh data -> rock_vs_control_unrestricted.csv
#    (also produces a funnel plot + per-output distribution CSV as side effects)
# =========================
python rerun_unrestricted.py

ROCK_CSV="rock_vs_control_unrestricted.csv"
if [ ! -f "${ROCK_CSV}" ]; then
  echo "ERROR: expected ${ROCK_CSV} to exist after rerun_unrestricted.py -- check the run log above." >&2
  exit 1
fi

# =========================
# 3. Per-token gradient directions (forward-only, reuses tokens from step 1)
# =========================
python compute_logit_gradients.py \
  --student onpolicy \
  --samples ${SAMPLES} \
  --hardware quad_l40s \
  --occurrences-file "${OCC_FILE}"

GRAD_FILE="logit_gradients_onpolicy_n${SAMPLES}_unrestricted.pt"

echo "==== DONE ===="
echo "Occurrences: ${OCC_FILE}"
echo "Rock list:   ${ROCK_CSV}"
echo "Gradients:   ${GRAD_FILE}"
echo ""
echo "Next: python build_ablation_freeze_lists.py --occurrences ${OCC_FILE} \\"
echo "        --gradients ${GRAD_FILE} \\"
echo "        --rock-csv ${ROCK_CSV} --outdir <path>"

#!/usr/bin/env bash
# W4 ablation (Reviewer RNAB): the actual Rock-Freeze ("Ours") checkpoint,
# under the exact same matched conditions as the 5 other new ablations
# (top_freq, top_meanloss, top_gradmag, top_gradalign, soft-lambda sweep) --
# 2 GPUs, RockToken/openthoughts_prompt_math_5k_src30k-35k, MAX_SAMPLES
# truncation, same eval protocol. This is the anchor point those 5 numbers
# need: without it we can only rank the alternatives against each other, not
# against the actual method.
#
# Cloned from run_stumb_soft_lambda.sh rather than parametrizing it further:
# that script's LAMBDA_TAG naming (`sed 's/\.//; s/^0*//'`) strips ALL digits
# from "0" (leading-zero removal on an all-zero string empties it), which
# would have given SAVE_DIR=.../soft_lambda_ -- ambiguous for what is now the
# single most important reference checkpoint. Fixed here with an explicit,
# unambiguous SAVE_DIR instead of patching that sed expression (which is
# correct as-is for the real sweep values 0.3/0.5/0.7 and doesn't need touching).
#
# Freeze mechanism: uses the real rock.json list (the actual K=100 Rock
# Tokens, same file the soft-lambda script reads) with freeze_weight=0 (hard
# freeze) -- i.e. Eq. 5 at lambda=0, exactly what "Rock-Freeze" means in the
# paper, just at this session's reduced scale.
#
# Hardware: this Vast.ai box has 2x RTX PRO 6000 (96GB each) -- much closer to
# the paper's 2x80GB H100 than the 48GB L40S box this script originally
# targeted, so the matched-condition values below (teacher_tp_size=2,
# teacher_quantization fp8, the two mem_fraction_static values) are kept at
# their ORIGINAL settings rather than the emergency low-memory overrides a
# collaborator needed on a 45GB L40S -- 96GB has ample headroom for the
# untouched config, and staying on it is what keeps this run comparable to
# the other 6 ablations. num_gpus_per_node=2, teacher_tp_size=2 (matches
# teacher's num_key_value_heads=4 divisibility). MAX_SAMPLES env var controls
# the step-count truncation (same value used for the other 5 runs was
# MAX_SAMPLES=1000, ~500 steps -- use the same value here for a matched
# comparison).
#
# Note: kdflow/arguments/__init__.py:90-97 silently auto-adjusts
# rollout_num_engines to (total_gpus // rollout_tp_size) regardless of what's
# passed on the CLI -- at 2 GPUs / rollout_tp_size=1 that's 2 engines, not the
# 1 requested below. This is existing kdflow behavior, not something this
# script controls; it just means two independent rollout actors each launch
# their own SGLang server and each load their own copy of the student weights.
#
# CHECKPOINTING: --save_steps / --ckpt_path / --load_checkpoint were missing
# from earlier versions of this script (all 5 other run_stumb_*.sh have
# them). Consequence, hit for real once already: a run can finish all steps,
# then OOM on the final full-gather HF save, and with no ckpt/ there is
# nothing to fall back on -- the entire run's compute is lost. Restored here
# to the same values the other 5 use: save_steps 50, ckpt under
# ${SAVE_DIR}/ckpt. --load_checkpoint True also makes a rerun resume from the
# newest ckpt/step_N automatically, so a late failure only costs the tail.
# ckpt/ is torch DCP *sharded* (per-rank __N_0.distcp) rather than the single-
# host full gather the final HF save does, so it should survive cases where
# the final save OOMs.
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON=${SCRIPT_DIR}/../.venv/bin/python
RAY=${SCRIPT_DIR}/../.venv/bin/ray

export CUDA_HOME=/usr/local/cuda-12.8

# Put the venv's own bin/ ahead of CUDA_HOME/bin on PATH: sglang's JIT-compiled
# kernels (flashinfer etc.) shell out to `ninja`, and a missing/wrong ninja on
# PATH surfaces as an opaque subprocess death (e.g. exit code 127) well after
# the point where it's obvious what failed. sglang[all] pulls in a pip ninja
# into the venv, so this makes sure that's the one found.
export PATH=${SCRIPT_DIR}/../.venv/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH

echo "==== TOOLCHAIN CHECK ===="
echo "as    -> $(command -v as || echo MISSING)"
echo "ld    -> $(command -v ld || echo MISSING)"
echo "ninja -> $(command -v ninja || echo 'MISSING (JIT kernel compilation will fail)')"

export HF_HOME=${HF_HOME:-/workspace/hf_cache}
mkdir -p "$HF_HOME"

# Physical GPU count on this box -- override only if that changes. Kept at 2 to
# stay matched with the other 6 ablations (see hardware note above).
NUM_GPUS=${NUM_GPUS:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ROLLOUT STARTUP TIMEOUT. Default is 600s (rollout_actor.py). A slow/cold
# storage mount can make the student-weight load alone take longer than that
# with no crash and no OOM -- just silence mid "Loading safetensors checkpoint
# shards" until the health check gives up. 2400s is a safety margin for that;
# the actual fix is warming the weights into page cache once below, so every
# rollout engine after the first reads them at RAM speed instead of hitting
# storage independently.
export KDFLOW_SERVER_HEALTH_TIMEOUT=${KDFLOW_SERVER_HEALTH_TIMEOUT:-2400}

unset RAY_ADDRESS
unset ip_head
unset RAY_NAMESPACE

echo "==== ENV CHECK ===="
$PYTHON -V
$PYTHON -c "import sys, ray; print('python_exe=', sys.executable); print('ray=', ray.__version__); print('python=', sys.version)"
$RAY --version
which nvcc
nvcc --version
echo "CUDA_HOME=$CUDA_HOME"

# =========================
# Paths
# =========================
KD_ROOT=${SCRIPT_DIR}
NEW_RUNNER_DIR=${KD_ROOT}/new_runner

STUDENT_MODEL=Qwen/Qwen3-4B-Instruct-2507
TEACHER_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
TRAIN_DATA=RockToken/openthoughts_prompt_math_5k_src30k-35k
SAVE_DIR=/workspace/Rock-Token-checkpoints/rock_freeze

mkdir -p ${SAVE_DIR}

export PYTHONPATH=${KD_ROOT}:${NEW_RUNNER_DIR}:$PYTHONPATH

cd ${KD_ROOT}

# =========================
# Warm student weights into page cache
# =========================
# kdflow/arguments/__init__.py auto-adjusts to 2 rollout engines at 2 GPUs /
# rollout_tp_size=1 (see note above), so the student checkpoint gets read from
# storage twice independently unless it's already cached. One pass up front
# costs at most a couple minutes; every engine after that reads from RAM.
# Parallel chunk reads rather than a single `cat`, in case this mount turns
# out to be latency-bound rather than bandwidth-bound (was true on a
# collaborator's ceph mount: 6.5 MB/s single-stream vs 68 MB/s at 16 readers).
# Best effort only -- if the path doesn't resolve we skip and let
# KDFLOW_SERVER_HEALTH_TIMEOUT absorb whatever the first cold load costs.
warm_page_cache() {
  local dir=$1 readers=16 f sz chunk i
  set +x
  for f in "$dir"/*.safetensors; do
    [ -f "$f" ] || continue
    sz=$(stat -Lc %s "$f")               # -L: HF cache entries are symlinks into blobs/
    chunk=$(( (sz / 4194304 + readers - 1) / readers ))   # 4MiB blocks per reader, ceil
    for i in $(seq 0 $((readers - 1))); do
      dd if="$f" of=/dev/null bs=4M count=$chunk skip=$((i * chunk)) 2>/dev/null &
    done
    wait || true
  done
  set -x
}

if [ -d "${STUDENT_MODEL}" ]; then
  STUDENT_CACHE_DIR=${STUDENT_MODEL}
else
  STUDENT_CACHE_DIR=$(ls -d ${HF_HOME}/hub/models--${STUDENT_MODEL//\//--}/snapshots/*/ 2>/dev/null | head -1)
fi
if [ -n "${STUDENT_CACHE_DIR}" ] && [ -d "${STUDENT_CACHE_DIR}" ]; then
  echo "==== WARMING STUDENT WEIGHTS ===="
  echo "dir: ${STUDENT_CACHE_DIR}"
  time warm_page_cache "${STUDENT_CACHE_DIR}"
else
  echo "WARN: student weights not found in HF cache, skipping warm (first load will be slower)" >&2
fi

# =========================
# Reset Ray
# =========================
$RAY stop --force || true
pkill -9 -f raylet || true
pkill -9 -f gcs_server || true
pkill -9 -f "ray::" || true
pkill -9 -f sglang || true
sleep 3

$RAY start --head \
  --node-ip-address=127.0.0.1 \
  --port=6380 \
  --num-gpus=${NUM_GPUS} \
  --temp-dir=/tmp/ray_${USER:-sroydip1} \
  --object-store-memory=${RAY_OBJ_STORE:-16000000000} \
  --disable-usage-stats

export RAY_ADDRESS=127.0.0.1:6380

$PYTHON - <<'PY'
import os, ray
print("Using RAY_ADDRESS =", os.environ.get("RAY_ADDRESS"))
ray.init(address=os.environ["RAY_ADDRESS"])
print("Connected OK")
print(ray.cluster_resources())
ray.shutdown()
PY

# =========================
# Launch
# =========================
$PYTHON -m kdflow.cli.train_kd_on_policy \
  --num_nodes 1 \
  --num_gpus_per_node ${NUM_GPUS} \
  --backend fsdp2 \
  --num_epochs 1 \
  --train_batch_size 4 \
  --micro_train_batch_size 1 \
  --learning_rate 2e-6 \
  --lr_warmup_ratio 0.05 \
  --max_norm 1.0 \
  --bf16 True \
  --gradient_checkpointing True \
  --cpu_offload ${CPU_OFFLOAD:-False} \
  --save_path ${SAVE_DIR} \
  --save_steps ${SAVE_STEPS:-50} \
  --ckpt_path ${SAVE_DIR}/ckpt \
  --load_checkpoint True \
  --student_name_or_path ${STUDENT_MODEL} \
  --teacher_name_or_path ${TEACHER_MODEL} \
  --enable_thinking True \
  --kd_ratio 1.0 \
  --kd_temperature 1.0 \
  --kd_algorithm token_freeze_kd \
  --token_freeze_path "${SCRIPT_DIR}/rock.json" \
  --freeze_weight 0.0 \
  --kd_loss_fn rkl \
  --teacher_tp_size 2 \
  --teacher_quantization fp8 \
  --teacher_dp_size 1 \
  --teacher_ep_size 1 \
  --teacher_pp_size 1 \
  --teacher_enable_sleep True \
  --teacher_forward_n_batches 1 \
  --teacher_mem_fraction_static ${TEACHER_MEM_FRAC:-0.45} \
  --rollout_num_engines 1 \
  --rollout_tp_size ${ROLLOUT_TP:-1} \
  --rollout_batch_size 2 \
  --n_samples_per_prompt 4 \
  --generate_max_len 1024 \
  --temperature 1.0 \
  --top_p 1.0 \
  --rollout_enable_sleep True \
  --rollout_mem_fraction_static ${ROLLOUT_MEM_FRAC:-0.12} \
  --train_dataset_path ${TRAIN_DATA} \
  --max_samples ${MAX_SAMPLES:-100000000} \
  --input_key prompt_messages \
  --apply_chat_template True \
  --max_len 2048 \
  --prompt_max_len 1536 \
  --preprocess_num_workers 4 \
  --packing_samples False \
  --logging_steps 1 \
  --use_wandb False

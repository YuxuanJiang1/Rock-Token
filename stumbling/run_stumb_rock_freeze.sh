#!/usr/bin/env bash
# W4 ablation (Reviewer RNAB): the actual Rock-Freeze ("Ours") checkpoint,
# under the exact same matched conditions as the 5 other new ablations
# (top_freq, top_meanloss, top_gradmag, top_gradalign, soft-lambda sweep) --
# 2x L40S, RockToken/openthoughts_prompt_math_5k_src30k-35k, MAX_SAMPLES
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
# Same L40S/hardware notes as the other 5 scripts: num_gpus_per_node=2,
# teacher_tp_size=2 (matches teacher's num_key_value_heads=4 divisibility),
# teacher_quantization fp8 (needed given 2x48GB vs the paper's 2x80GB), and
# MAX_SAMPLES env var controls the step-count truncation (same value used for
# the other 5 runs was MAX_SAMPLES=1000, ~500 steps -- use the same value
# here for a matched comparison).
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON=${SCRIPT_DIR}/../.venv/bin/python
RAY=${SCRIPT_DIR}/../.venv/bin/ray

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH

export HF_HOME=${HF_HOME:-/workspace/hf_cache}
mkdir -p "$HF_HOME"

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

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
  --num-gpus=2 \
  --temp-dir=/tmp/ray_${USER:-sroydip1} \
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
  --num_gpus_per_node 2 \
  --backend fsdp2 \
  --num_epochs 1 \
  --train_batch_size 4 \
  --micro_train_batch_size 1 \
  --learning_rate 2e-6 \
  --lr_warmup_ratio 0.05 \
  --max_norm 1.0 \
  --bf16 True \
  --gradient_checkpointing True \
  --save_path ${SAVE_DIR} \
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
  --teacher_mem_fraction_static 0.45 \
  --rollout_num_engines 1 \
  --rollout_tp_size 1 \
  --rollout_batch_size 2 \
  --n_samples_per_prompt 4 \
  --generate_max_len 1024 \
  --temperature 1.0 \
  --top_p 1.0 \
  --rollout_enable_sleep True \
  --rollout_mem_fraction_static 0.12 \
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

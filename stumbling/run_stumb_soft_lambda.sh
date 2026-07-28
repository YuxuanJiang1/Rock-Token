#!/usr/bin/env bash
# W4 ablation (Reviewer RNAB): soft down-weighting instead of a hard freeze
# (lambda=0). Reuses the existing rock.json list; only --freeze_weight
# changes. Tests whether the binary "Just Not Train" choice (lambda=0) is
# necessary, or whether an intermediate lambda already recovers most of the
# speedup with less risk to accuracy.
#
# Usage:
#   ./run_stumb_soft_lambda.sh 0.3
#   ./run_stumb_soft_lambda.sh 0.5
#   ./run_stumb_soft_lambda.sh 0.7
#
# --- Adapted for 2x L40S (48GB, Ada Lovelace, no NVLink) -- only 2 of the
# original 4 were available. num_gpus_per_node=2, teacher_tp_size=2: this is
# now the *same* GPU topology as the original (H100) run_stumb_random.sh --
# the teacher uses the entire 2-GPU pool rather than a subset, which is also
# why TP=2 was the right choice all along (checked the teacher's actual
# config.json: num_key_value_heads=4 and num_attention_heads=32 -- TP must
# divide the KV-head count, and 2 always does, unlike e.g. 3). The one real
# difference from the original run: L40S has 48GB/GPU vs. H100's 80GB, so the
# same teacher_tp_size=2 now leaves much less headroom (2x48=96GB pool vs.
# 2x80=160GB) -- teacher_quantization fp8 (halves the ~60GB bf16 teacher to
# ~15GB/GPU at TP=2) is doing real load-bearing work here, not just a nice-to-
# have. If you hit OOM, this is the tightest config we've tried; the next
# levers are --teacher_mem_fraction_static / --rollout_mem_fraction_static,
# then sequence lengths, in that order. Bonus: unlike the 3-GPU config this
# replaced, train_batch_size=4 now divides evenly by num_gpus_per_node=2, so
# the effective global batch is exactly 4 again (kdflow computes grad_accum
# via integer division; at 3 GPUs that floored to an effective batch of 3).
# UNLIKE the other 4 ablation scripts, wall-clock DOES matter here (this run
# is specifically building the accuracy-vs-speed frontier across lambda). No
# NVLink means L40S step times won't match the paper's H100 numbers, so for a
# clean frontier you should also re-run the lambda=0 (Rock-Freeze) and lambda=1
# (Original OPD) endpoints on this same L40S hardware rather than reusing the
# paper's reported H100 wall-clock for those two points.
set -e
set -x

LAMBDA=${1:?"usage: $0 <lambda in (0,1)>, e.g. 0.3 / 0.5 / 0.7"}
# tag used for the save-path suffix, e.g. 0.3 -> 03
LAMBDA_TAG=$(echo "$LAMBDA" | sed 's/\.//; s/^0*//')

# Absolute path to this script's own directory (i.e. stumbling/, wherever the
# repo is checked out) -- used below so --token_freeze_path resolves correctly
# regardless of what cwd this was launched from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON=/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token/.venv/bin/python
RAY=/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token/.venv/bin/ray

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH

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
echo "LAMBDA=$LAMBDA"

# =========================
# Paths
# =========================
KD_ROOT=/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token/stumbling
NEW_RUNNER_DIR=${KD_ROOT}/new_runner

STUDENT_MODEL=Qwen/Qwen3-4B-Instruct-2507
TEACHER_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
TRAIN_DATA=RockToken/openthoughts_prompt_math_5k_src30k-35k
SAVE_DIR=/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token-checkpoints/soft_lambda_${LAMBDA_TAG}

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
  --freeze_weight ${LAMBDA} \
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

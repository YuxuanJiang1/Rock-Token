#!/usr/bin/env bash
# W4 ablation (Reviewer RNAB): freeze the top-K tokens by mean_kl(v) alone,
# ignoring the Freq(v) weighting entirely (the "Rock Score construction") --
# tests whether high mean loss alone explains the Fig. 5 gap between
# "Ours" (Rock-Freeze) and "Original OPD". Excludes tokens already in the
# rock set (== the "high_kl" group used for Fig. 3).
# Token list built by build_ablation_freeze_lists.py -> top_meanloss.json.
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
# No NVLink
# means step time won't match the paper's H100 numbers -- fine for the accuracy
# comparison this ablation is testing, but don't read wall-clock ratios against
# the paper's reported 1.7x without re-measuring Original OPD on this same
# L40S hardware too.
set -e
set -x

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

# =========================
# Paths
# =========================
KD_ROOT=/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token/stumbling
NEW_RUNNER_DIR=${KD_ROOT}/new_runner

STUDENT_MODEL=Qwen/Qwen3-4B-Instruct-2507
TEACHER_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
TRAIN_DATA=RockToken/openthoughts_prompt_math_5k_src30k-35k
SAVE_DIR=/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token-checkpoints/top_meanloss

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
  --token_freeze_path "${SCRIPT_DIR}/ablation_lists/top_meanloss.json" \
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
  --input_key prompt_messages \
  --apply_chat_template True \
  --max_len 2048 \
  --prompt_max_len 1536 \
  --preprocess_num_workers 4 \
  --packing_samples False \
  --logging_steps 1 \
  --use_wandb False

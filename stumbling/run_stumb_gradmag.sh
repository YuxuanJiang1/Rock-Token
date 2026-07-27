#!/usr/bin/env bash
# W4 ablation (Reviewer RNAB): freeze the top-K tokens by per-token gradient
# magnitude ||bar_g_t|| (Sec. 2.6 / Fig. 3 quantity) instead of the Rock
# Score -- tests whether a purely gradient-magnitude-based selection gives
# the same efficiency/accuracy tradeoff as Rock-Freeze.
# Token list built by build_ablation_freeze_lists.py -> top_gradmag.json.
#
# --- Adapted for 4x L40S (48GB, Ada Lovelace, no NVLink) instead of the paper's
# 4x H100 (80GB): num_gpus_per_node 2->4, teacher_tp_size 2->4 (spreads the
# ~60GB bf16 teacher to ~15GB/GPU), teacher_quantization fp8 added (halves that
# again to ~7.5GB/GPU; Ada supports FP8 tensor cores natively and SGLang 0.5.9
# supports it -- if you hit a quantization/kernel error, delete that flag and
# fall back to bf16, you likely have headroom for it with TP=4 anyway).
# mem_fraction_static values and sequence lengths are left as in the original
# ablation config; if you OOM, those are the next things to reduce. No NVLink
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

PYTHON=/home/xxj1/.conda/envs/qwen/bin/python
RAY=/home/xxj1/.conda/envs/qwen/bin/ray

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH

export CUDA_VISIBLE_DEVICES=0,1,2,3
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
KD_ROOT=
NEW_RUNNER_DIR=${KD_ROOT}/new_runner

STUDENT_MODEL=/p/work2/xxj1/opd/models/Qwen3-4B-Instruct-2507
TEACHER_MODEL=/p/work2/xxj1/opd/models/Qwen3-30B-A3B
TRAIN_DATA=/p/work2/xxj1/opd/data/OpenThoughts3-1.2M/openthoughts_prompt_math_10k.jsonl
SAVE_DIR=/p/work2/xxj1/rocktoken/stumb/top_gradmag

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
  --num_gpus_per_node 4 \
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
  --token_freeze_path "${SCRIPT_DIR}/ablation_lists/top_gradmag.json" \
  --freeze_weight 0.0 \
  --kd_loss_fn rkl \
  --teacher_tp_size 4 \
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

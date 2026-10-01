#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
: "${STUDENT_MODEL:?Set STUDENT_MODEL to your student checkpoint}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to your teacher checkpoint}"
: "${TRAIN_DATA:?Set TRAIN_DATA to a training JSONL file}"
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/outputs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
mkdir -p "$SAVE_DIR"
cd "$SCRIPT_DIR"
# The trainer initializes Ray; set RAY_ADDRESS to reuse your own cluster.
# This launcher never stops other Ray or SGLang processes.
TOKEN_FREEZE_PATH="${TOKEN_FREEZE_PATH:-${SCRIPT_DIR}/random.json}"

"$PYTHON" -m kdflow.cli.train_kd_on_policy \
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
  --save_path "${SAVE_DIR}" \
  --student_name_or_path "${STUDENT_MODEL}" \
  --teacher_name_or_path "${TEACHER_MODEL}" \
  --enable_thinking True \
  --kd_ratio 1.0 \
  --kd_temperature 1.0 \
  --kd_algorithm token_freeze_kd \
--token_freeze_path "${TOKEN_FREEZE_PATH}" \
--freeze_weight 0.0 \
  --kd_loss_fn rkl \
  --teacher_tp_size 2 \
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
  --train_dataset_path "${TRAIN_DATA}" \
  --input_key prompt_messages \
  --apply_chat_template True \
  --max_len 2048 \
  --prompt_max_len 1536 \
  --preprocess_num_workers 4 \
  --packing_samples False \
  --logging_steps 1 \
  --use_wandb False

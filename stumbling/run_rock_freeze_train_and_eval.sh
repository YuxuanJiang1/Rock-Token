#!/usr/bin/env bash
# Trains AND evaluates the Rock-Freeze ("Ours") matched-conditions checkpoint
# in one shot: runs run_stumb_rock_freeze.sh, then verifies a real checkpoint
# actually landed before touching evaluation at all -- given the OOM-on-save
# failure we already hit once, a clean process exit alone isn't trusted here,
# config.json presence is the actual ground truth. Only then does it call
# ../evaluation/run_eval_ablations.sh restricted to this one checkpoint.
#
# Resumable in both directions:
#   - if a valid checkpoint already exists, training is skipped (FORCE_RETRAIN=1
#     to redo it anyway)
#   - evaluation's own script already skips seeds that already have a .done
#     marker, so re-running this after a partial eval just picks up where it
#     left off
#
# Deliberately keeps run_stumb_rock_freeze.sh's hardware config untouched
# (2 GPUs, teacher_tp_size=2) even though 4 are now available -- this run's
# entire purpose is to be the matched-conditions anchor for the 6 already-
# completed ablations, which all used that same config. Changing it here
# would undermine the comparison this run exists to provide.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/../evaluation" && pwd)"
CHECKPOINT_DIR="/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token-checkpoints/rock_freeze"

export MAX_SAMPLES=${MAX_SAMPLES:-1000}
NUM_SEEDS=${NUM_SEEDS:-3}
FORCE_RETRAIN=${FORCE_RETRAIN:-0}

echo "============================================================"
echo "  STAGE 1: TRAINING  $(date)"
echo "  MAX_SAMPLES=${MAX_SAMPLES}"
echo "============================================================"

if [ -f "${CHECKPOINT_DIR}/config.json" ] && [ "${FORCE_RETRAIN}" != "1" ]; then
  echo "Checkpoint already exists at ${CHECKPOINT_DIR}/config.json -- skipping training."
  echo "(FORCE_RETRAIN=1 to redo it anyway.)"
else
  chmod +x "${SCRIPT_DIR}/run_stumb_rock_freeze.sh"
  "${SCRIPT_DIR}/run_stumb_rock_freeze.sh"
  TRAIN_EXIT=$?

  echo ""
  echo "============================================================"
  echo "  VERIFYING CHECKPOINT  $(date)"
  echo "============================================================"
  if [ ! -f "${CHECKPOINT_DIR}/config.json" ]; then
    echo "ERROR: ${CHECKPOINT_DIR}/config.json not found after training." >&2
    echo "        (training script exit code was ${TRAIN_EXIT} -- a clean exit code" >&2
    echo "        alone does not guarantee the save succeeded, this is exactly the" >&2
    echo "        failure mode hit previously: OOM during the final FSDP save.)" >&2
    echo "        Stopping before evaluation -- nothing to evaluate yet." >&2
    exit 1
  fi
  echo "Checkpoint OK: ${CHECKPOINT_DIR}/config.json exists"
fi

ls -la "${CHECKPOINT_DIR}"

echo ""
echo "============================================================"
echo "  STAGE 2: EVALUATION  $(date)"
echo "  NUM_SEEDS=${NUM_SEEDS}"
echo "============================================================"

chmod +x "${EVAL_DIR}/run_eval_ablations.sh"
CHECKPOINTS=rock_freeze NUM_SEEDS=${NUM_SEEDS} "${EVAL_DIR}/run_eval_ablations.sh"
EVAL_EXIT=$?

echo ""
echo "============================================================"
echo "  ALL DONE  $(date)"
echo "============================================================"
if [ "${EVAL_EXIT}" != "0" ]; then
  echo "WARNING: evaluation stage exited with code ${EVAL_EXIT} -- check the logs" >&2
  echo "         in ${EVAL_DIR}/logs/rock_freeze_run*.log before trusting the summary." >&2
fi
echo "Results: ${EVAL_DIR}/eval_results/summary.csv (look for the rock_freeze row)"

#!/usr/bin/env bash
# Runs all 5 W4 ablations (7 total training invocations, since the soft-lambda
# sweep covers 3 values) back-to-back on this same 2x L40S box. Each individual
# run_stumb_*.sh already resets Ray and kills leftover processes at its own
# start, so no extra cleanup is needed between runs here -- they can just run
# one after another.
#
# Order: top_meanloss first -- closest to Reviewer RNAB's core W4 ask ("top-
# mean-loss freeze without the Rock Score construction"), so it's the highest-
# value result if time runs short -- then top_freq, gradmag, gradalign, then
# the soft-lambda sweep at 0.3/0.5/0.7.
#
# Resumable: each run writes a .done marker to logs/ after finishing
# successfully; re-running this script skips anything already marked done
# (useful since this is a multi-hour-to-multi-day job across 7 runs and may
# get interrupted). FORCE_RERUN=1 redoes everything anyway.
#
# Stops on the first failure by default, so a broken run doesn't burn GPU
# hours on the rest of a possibly-still-broken setup. CONTINUE_ON_ERROR=1
# instead keeps going and collects whatever succeeds.
#
# This is long-running -- run it inside tmux/screen or with nohup so it
# survives an SSH disconnect, e.g.:
#   tmux new -s w4_ablations
#   ./run_all_ablations.sh 2>&1 | tee logs/run_all.log
#   [detach with Ctrl-b d; reattach later with: tmux attach -t w4_ablations]
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

chmod +x "${SCRIPT_DIR}"/run_stumb_top_freq.sh \
         "${SCRIPT_DIR}"/run_stumb_top_meanloss.sh \
         "${SCRIPT_DIR}"/run_stumb_gradmag.sh \
         "${SCRIPT_DIR}"/run_stumb_gradalign.sh \
         "${SCRIPT_DIR}"/run_stumb_soft_lambda.sh

mkdir -p /umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token-checkpoints

FORCE_RERUN=${FORCE_RERUN:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}

# run_one <name> <command...>
# Runs <command>, teeing its full output to logs/<name>.log, and drops a
# logs/<name>.done marker on success. Skips already-done runs unless
# FORCE_RERUN=1. Returns non-zero on failure unless CONTINUE_ON_ERROR=1.
run_one() {
  local name="$1"; shift
  local log="${LOG_DIR}/${name}.log"
  local marker="${LOG_DIR}/${name}.done"

  if [ "${FORCE_RERUN}" != "1" ] && [ -f "${marker}" ]; then
    echo "=== Skipping ${name} -- already marked done (${marker}). FORCE_RERUN=1 to redo. ==="
    return 0
  fi

  echo "=== Starting ${name} at $(date) -- log: ${log} ==="
  if "$@" > "${log}" 2>&1; then
    touch "${marker}"
    echo "=== Finished ${name} OK at $(date) ==="
    return 0
  else
    echo "=== FAILED: ${name} at $(date) -- see ${log} ===" >&2
    if [ "${CONTINUE_ON_ERROR}" = "1" ]; then
      return 0
    fi
    return 1
  fi
}

run_one top_meanloss "${SCRIPT_DIR}/run_stumb_top_meanloss.sh"    || exit 1
run_one top_freq     "${SCRIPT_DIR}/run_stumb_top_freq.sh"        || exit 1
run_one gradmag      "${SCRIPT_DIR}/run_stumb_gradmag.sh"         || exit 1
run_one gradalign    "${SCRIPT_DIR}/run_stumb_gradalign.sh"       || exit 1
run_one soft_03      "${SCRIPT_DIR}/run_stumb_soft_lambda.sh" 0.3 || exit 1
run_one soft_05      "${SCRIPT_DIR}/run_stumb_soft_lambda.sh" 0.5 || exit 1
run_one soft_07      "${SCRIPT_DIR}/run_stumb_soft_lambda.sh" 0.7 || exit 1

echo ""
echo "==== ALL DONE ===="
echo "Per-run logs:   ${LOG_DIR}/<name>.log"
echo "Done markers:   ${LOG_DIR}/<name>.done"
echo "Checkpoints in: /umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token-checkpoints/"

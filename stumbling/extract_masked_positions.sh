#!/usr/bin/env bash
# Computes the "Masked Positions (%)" column for the efficiency-comparison table
# (masked-position fraction + wall-clock speedup, requested alongside accuracy).
#
# The freeze-weighted training path already logs this on rank 0 every
# microbatch:
#   [TokenFreezeKD] freeze hits: <hits> / <total>
# (kdflow/algorithms/token_freeze_kd.py:97-101). This script just greps that
# line out of each run's training log, sums hits and totals across every
# logged microbatch, and reports hits/total as a percentage -- the real,
# empirically observed masking rate for that run, not a theoretical estimate.
#
# Run this on the cluster where the actual training logs live (this repo
# checkout on your local machine won't have stumbling/logs/ populated).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
OUT_CSV="${SCRIPT_DIR}/masked_positions_summary.csv"

# name -> glob pattern to find that variant's training log(s) under LOG_DIR.
# The four run_all_ablations.sh names are exact (confirmed from that script);
# random and rock_freeze use globs since their log naming differs (rock_freeze
# is timestamped per-invocation by run_rock_freeze_train_and_eval.sh; random's
# exact filename wasn't confirmed here -- adjust the glob below if it doesn't
# match what's actually on disk).
NAMES=(top_freq top_meanloss gradmag gradalign random rock_freeze)
GLOBS=("top_freq.log" "top_meanloss.log" "gradmag.log" "gradalign.log" "*random*.log" "rock_freeze_train_eval_*.log")

echo "variant,hits,total,masked_pct" > "${OUT_CSV}"

printf "%-14s %14s %14s %10s\n" "variant" "hits" "total" "masked_%"
printf '%s\n' "--------------------------------------------------------------"

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  glob="${GLOBS[$i]}"

  # Expand the glob under LOG_DIR; skip if nothing matches.
  shopt -s nullglob
  matches=("${LOG_DIR}"/${glob})
  shopt -u nullglob

  if [ "${#matches[@]}" -eq 0 ]; then
    printf "%-14s %14s %14s %10s\n" "${name}" "-" "-" "no log found"
    echo "${name},,,NO_LOG_FOUND" >> "${OUT_CSV}"
    continue
  fi

  if [ "${#matches[@]}" -gt 1 ]; then
    echo "  (note: ${#matches[@]} log files matched '${glob}' for ${name} -- summing across all of them: ${matches[*]})" >&2
  fi

  result=$(grep -ohE 'freeze hits: [0-9]+ / [0-9]+' "${matches[@]}" | \
    awk -F'[: /]+' '{hits+=$3; total+=$4} END {
      if (total > 0) printf "%d,%d,%.2f", hits, total, 100*hits/total
      else printf ",,NO_FREEZE_LINES"
    }')

  hits=$(echo "${result}" | cut -d',' -f1)
  total=$(echo "${result}" | cut -d',' -f2)
  pct=$(echo "${result}" | cut -d',' -f3)

  printf "%-14s %14s %14s %10s\n" "${name}" "${hits:--}" "${total:--}" "${pct:-N/A}%"
  echo "${name},${hits},${total},${pct}" >> "${OUT_CSV}"
done

echo ""
echo "Saved: ${OUT_CSV}"

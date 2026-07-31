#!/usr/bin/env bash
# Batch launcher for run_stumb_rock_freeze.sh on 4x L40S.
#
# Why this exists: the interactive dev allocations are --mem=200G, and the 4-GPU
# topology does not fit in that. Sleep mode offloads each engine's whole GPU
# reservation to /dev/shm, and 4 GPUs means 4 rollout engines + a 4-way teacher,
# roughly double the host RAM of the 2x H100 topology the other 5 ablations used.
# Measured peak at 200G: 198GB then OOM-killed (twice, dmesg "Killed process
# sglang::schedul"). Breakdown once both engines are up and the student loads:
#   teacher+rollout shm ~118GB + rss ~29GB = 147GB, then student FSDP adds ~50GB.
# 248G, up from 240G. At 240G training ran fine for all 500 steps but the FINAL HF save
# OOM-killed a ray worker: that save does a single-host full-model gather on top of the
# ~161GB of pinned sleep-mode shm, and peak sampling showed 224GB with the spike landing
# between samples. Node RealMemory is 251G so 248G is essentially the ceiling.
# The real safety net is --save_steps 50 in run_stumb_rock_freeze.sh (restored; the clone
# had dropped it): ckpt/ is torch DCP *sharded*, no full gather, so it should survive even
# if the final HF save does not -- and --load_checkpoint True makes a rerun resume.
#
# --time is REQUIRED: partition DefaultTime is 00:15:00 (MaxTime is UNLIMITED), so
# without it the job is killed 15 minutes in, mid teacher-load.
# Budget: the 2x H100 runs took 2:30-2:39 each (logs/run_all.log). L40S has ~half
# H100 bf16 throughput and the teacher runs untuned MoE kernels here, so allow well
# over that. 12h is deliberately generous; the job exits when training finishes.
#
# Submit:  sbatch stumbling/sbatch_rock_freeze.sh
# Watch:   squeue -u $USER -n rock_freeze ; tail -f stumbling/rock_freeze.log
#
#SBATCH --job-name=rock_freeze
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=248G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --output=/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token/stumbling/logs/sbatch_rock_freeze_%j.out
#SBATCH --error=/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token/stumbling/logs/sbatch_rock_freeze_%j.out

set -u

KD_ROOT=/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token/stumbling
mkdir -p "${KD_ROOT}/logs"
cd "${KD_ROOT}"

echo "==== SBATCH CONTEXT ===="
echo "job     : ${SLURM_JOB_ID:-?}  node: $(hostname)"
echo "gpus    : ${SLURM_GPUS_ON_NODE:-?}  cpus: ${SLURM_CPUS_PER_TASK:-?}"
echo "mem cap : $(scontrol show job ${SLURM_JOB_ID} 2>/dev/null | grep -oE 'mem=[0-9]+[GM]' | head -1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "cgroup limit: $(awk '{printf "%.0f GB\n",$1/2^30}' \
  /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.limit_in_bytes 2>/dev/null || echo '?')"
echo "========================"

# MAX_SAMPLES=1000 matches the other 5 ablations (~500 steps). Without it the
# default is 100000000, i.e. the full 5k set, which is NOT a matched comparison.
# CEPH PREFLIGHT -- fail in seconds, not an hour.
# 2026-07-30: a run on g24-11 crawled because that node's ceph client was degraded to
# 6 MB/s while g24-10 did 124 MB/s and g24-01 did 343 MB/s on the SAME file at the same
# moment. Symptoms were all misleading: 19-min "hung" imports, a 21-min student load
# (8GB at 6MB/s), and sglang health/start timeouts. Nothing was actually misconfigured.
# At 6 MB/s the teacher's 57GB needs ~2.6h and blows start(timeout=1800) every time.
# Threshold 50 MB/s: teacher load stays under ~20min, comfortably inside the budget.
CEPH_MIN_MBPS=${CEPH_MIN_MBPS:-50}
_snap=$(ls -d /umbc/rs/pi_ferraro/ada/users/sroydip1/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*/ 2>/dev/null | head -1)
_probe=$(ls "${_snap}"*.safetensors 2>/dev/null | head -1)
if [ -n "${_probe}" ]; then
  _mbps=$(dd if="${_probe}" of=/dev/null bs=4M count=50 2>&1 | grep -oE "[0-9.]+ MB/s" | grep -oE "^[0-9.]+")
  echo "ceph read throughput on $(hostname): ${_mbps:-?} MB/s (need >= ${CEPH_MIN_MBPS})"
  if [ -n "${_mbps}" ] && [ "$(printf '%.0f' "${_mbps}")" -lt "${CEPH_MIN_MBPS}" ] 2>/dev/null; then
    echo "FATAL: $(hostname) ceph is degraded (${_mbps} MB/s). Refusing to start -- the" >&2
    echo "       teacher load would take hours and hit sglang's start timeout." >&2
    echo "       Resubmit to land elsewhere, or set CEPH_MIN_MBPS=0 to override." >&2
    exit 1
  fi
else
  echo "WARN: could not find a teacher shard to probe; skipping ceph preflight" >&2
fi

export MAX_SAMPLES=${MAX_SAMPLES:-1000}
export NUM_SEEDS=${NUM_SEEDS:-3}

# Runs the train+eval chain, not training alone: it verifies config.json actually
# landed before evaluating, which is the exact failure we hit (clean exit, empty
# epoch_1). Both stages use the unified .venv-train (torch 2.9.1 + sglang for train,
# vllm 0.16 + lm_eval for eval).
chmod +x ./run_rock_freeze_train_and_eval.sh
./run_rock_freeze_train_and_eval.sh 2>&1 | tee "${KD_ROOT}/rock_freeze_full.log"
rc=${PIPESTATUS[0]}
echo "==== run_rock_freeze_train_and_eval.sh exited rc=${rc} ===="
exit ${rc}

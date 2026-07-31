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
# CHECKPOINTING: --save_steps / --ckpt_path / --load_checkpoint were MISSING from this
# script (the clone dropped them; all 5 other run_stumb_*.sh have them). Consequence,
# learned the hard way on 2026-07-29: job 214002 finished all 500 steps then the final
# HF save OOM-killed a ray worker, and because there was no ckpt/ there was NOTHING to
# fall back on -- 3.9h of compute lost outright.
# Restored to the same values the other 5 use: save_steps 50, ckpt under ${SAVE_DIR}/ckpt.
# Two reasons this matters beyond insurance:
#   1. --load_checkpoint True makes a rerun RESUME from the newest ckpt/step_N
#      automatically, so a late failure costs only the tail.
#   2. ckpt/ is torch DCP *sharded* (per-rank __N_0.distcp), so it does NOT do the
#      single-host full gather that the final HF save does. It is therefore far cheaper
#      in host RAM and should survive even where the final save OOMs -- giving a
#      convertible artifact instead of nothing.
# Disk: ~93GB per retained step (weights + fp32 optimizer state), ~2 kept.
#
# Freeze mechanism: uses the real rock.json list (the actual K=100 Rock
# Tokens, same file the soft-lambda script reads) with freeze_weight=0 (hard
# freeze) -- i.e. Eq. 5 at lambda=0, exactly what "Rock-Freeze" means in the
# paper, just at this session's reduced scale.
#
# MEMORY FRACTIONS ARE ABSOLUTE SIZES, NOT PORTABLE. The other 5 ablations really
# ran on 2x H100 NVL 93GB (logs/*.log show accelerator_type:H100), despite what
# run_all_ablations.sh's header says. Their 0.75/0.12 means 69.8GB/11.2GB there.
# On a 46GB L40S the same fractions give 34.5GB/5.5GB, and 5.5GB cannot even hold
# the 4B student's bf16 weights -> "Not enough memory ... mem_fraction_static=0.12".
# Rollout raised to 0.25 so the absolute size (11.5GB) matches what H100 gave it.
# The H100 teacher 0.75 is likewise not portable, but for the opposite reason now:
# on a 45GB card at TP=2 it is too LOW, not too high. See the 0.85 block below.
#
# TEACHER 0.95 / ROLLOUT 0.20 / ROLLOUT_TP 2. This triple is knife-edge on a 45GB
# L40S and the three values only work together. Do not change one alone.
#
# What job 223593 (2026-07-30 17:05) actually proved, at TP=2 / rollout tp=1 / 0.85:
# teacher loaded all 16 shards then died in cu_mem_create "CUresult error: 2".
# The rollout is STILL RESIDENT when the teacher inits -- 45.0 - 30.13 = 14.87GB/GPU,
# and its capture log prints the giveaway avail_mem=30.13 GB. Teacher TP=2 wants
# 30.5GB. 30.5 + 14.87 = 45.4 > 45.0, so it overflows by ~0.4GB.
# train_kd_on_policy.py:48 DOES call rollout_group.sleep() before TeacherActorGroup,
# and rollout_group.sleep() is a blocking ray.get on release_memory_occupation, but
# the VRAM is measurably not back at teacher-init time. Treat the rollout as resident.
# => the fix is to make the ROLLOUT smaller, not to raise the teacher fraction. Raising
#    the teacher fraction cannot help: the shortfall is in WEIGHTS, not the KV pool.
#
# Rollout shrunk two ways at once:
#   1. ROLLOUT_TP=2. arguments/__init__.py:90 forces num_engines = total_gpus // tp,
#      so tp=2 collapses 2 engines -> 1 engine spanning both cards. The 4B bf16
#      weights go 8GB/GPU -> 4GB/GPU. (--rollout_num_engines 1 is IGNORED on its own;
#      at tp=1 it was silently auto-raised back to 2. tp is the only real lever.)
#   2. ROLLOUT_MEM_FRAC 0.25 -> 0.16.
#
# MEASURE the rollout, do not estimate it. Its footprint is NOT 45*frac: there is a
# ~3.83GB fixed overhead on top of the static pool (cuda graphs + buffers). Measured
# at tp=2 via the capture line "Capturing batches (bs=1 avail_mem=N GB)":
#   frac   static   avail_mem   footprint = 45.0 - avail_mem
#   0.20    9.0GB    32.17GB      12.83GB   <- estimated 11GB, was wrong by 1.8GB
#   0.16    7.2GB      ~34GB      ~11.0GB   <- chosen (7.2 static holds 4GB weights
#                                              + ~3.2GB KV; rollout needs ~1.5GB)
# Read that avail_mem line on every run. It is the only honest number here.
#
# TEACHER 0.85, and the rollout numbers are back to the ORIGINAL 0.25 / tp=1. The thing
# that actually blocked 2 GPUs was --teacher_quantization fp8, not any fraction. See the
# TEACHER_QUANT_ARG block above. Five runs were burned tuning fractions against a
# root cause that was not a fraction; do not repeat that. Sequence, all 2026-07-30:
#
#   attempt  rollout tp/frac  teacher frac  fp8   result
#     1        1 / 0.25          0.85       yes   cu_mem_create OOM
#     2        2 / 0.20          0.95       yes   cu_mem_create OOM
#     3        2 / 0.16          0.97       yes   cu_mem_create OOM
#     4        2 / 0.16          0.72       yes   cu_mem_create OOM
#     5        2 / 0.13          0.71       yes   cu_mem_create OOM
#   isolated probe, teacher alone, no ray/rollout/student:
#              -                 0.71       yes   "Not enough memory, increase frac"
#              -                 0.85       yes   "Not enough memory, increase frac"
#              -                 0.92       yes   "Not enough memory, increase frac"
#              -                 0.85       NO    STARTED OK, 40.55GB/GPU used
#
# TWO LESSONS WORTH KEEPING:
# 1. When a knob does not behave, ISOLATE. The whole thing was settled in ~4 minutes by
#    running the teacher alone in a 40-line script. Five in-situ attempts at ~8 min each
#    settled nothing, because ray + rollout + student masked sglang's real error message.
# 2. The two failure signatures point in OPPOSITE directions, so read which one you got:
#      cu_mem_create OOM, no sglang memory log  -> pool does not fit  -> LOWER frac
#      clean "Not enough memory, increase ..."  -> pool too small     -> RAISE frac
#    In situ we only ever saw the first and I kept treating it as the second.
#
# Sizing, measured: teacher alone at frac 0.85 without fp8 uses 40.55GB of 44.39GB
# (note 44.39GB usable, NOT 45.0 -- nvidia-smi's 46068MiB is before CUDA context).
# The rollout only holds ~1.73GB of sleeping context while the teacher inits
# (train_kd_on_policy.py:48 sleeps it first, and it really does release: the GPU trace
# during attempt 5 peaked at 31.02GB = teacher weights alone, rollout fully gone).
# So the teacher sees ~42.7GB, needs 40.55GB, ~2.1GB margin. Because the rollout is
# ASLEEP at that moment, its own frac does not matter here -- which is why tp=1 / 0.25
# is restored, matching the other 6 ablations exactly.
#
# Teacher and rollout never need to be awake together: on_policy_kd_trainer.py:365
# sleeps the rollout at the end of rollout(), before teacher.wakeup() at :221.
#
# Also a HOST-RAM setting: sleep offloads allocated pages to /dev/shm. Here that is
# teacher 61GB weights + 2x1.25GB KV = ~63GB, rollout ~9GB, 2 student actors ~50GB
# -> ~125GB + ray overhead, well under this job's 240G cgroup. (The 4-GPU config
# needed ~200GB and got OOM-killed at the final save on a 251GB node.)
# --object-store-memory above also caps ray's plasma store, which defaults to ~30% of
# RAM (80GB) and is never needed by this workload.
# Diagnose an OOM via the cgroup, not the ray traceback (it blames the wrong actor):
#   C=/sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_$SLURM_JOB_ID
#   awk '/^total_rss /{r=$2} /^total_shmem /{s=$2} END{printf "%.1f GB\n",(r+s)/2^30}' $C/memory.stat
#
# Teacher weights are 15.25GB/GPU: the checkpoint is 57GB on disk / torch_dtype
# bfloat16 -> 61GB resident / tp 4. fp8 does NOT shrink them.
# CORRECTION 2026-07-30: the old claim on this line, that --teacher_quantization fp8 is
# "silently ignored" and therefore harmless, is WRONG and cost five failed runs. fp8 does
# not shrink the weights (that part was right) but it is NOT free: it costs ~10GB/GPU.
# At tp=4 there is room to absorb that; at tp=2 there is not, and the teacher cannot
# start at ANY mem_fraction_static. Proof and numbers in the TEACHER_QUANT_ARG block
# below. fp8 is now OFF by default.
# Do not size this assuming ~7.6GB fp8 weights; that is what 0.30 assumed.
# The MoE config filename using dtype=fp8_w8a8 is the fused-MoE COMPUTE path and is
# not evidence that the stored weights are fp8.
# KV need is small (forward-only scoring, 8 seqs x 2560 tok ~ 0.5GB at tp 4); the
# remaining headroom is for CUDA graphs.
#
# The 0.55 OOM killed an sglang scheduler (shmem-rss 21.8GB in that one proc) and
# surfaced as a bogus ray ActorUnavailableError "RpcError: keepalive" on the STUDENT
# actor -- nothing to do with the student. Check the real cause via the cgroup:
#   C=/sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_$SLURM_JOB_ID
#   awk '/^total_rss /{r=$2} /^total_shmem /{s=$2} END{printf "%.1f GB\n",(r+s)/2^30}' $C/memory.stat
#   dmesg | grep -i 'killed process'
# Override per-host with TEACHER_MEM_FRAC / ROLLOUT_MEM_FRAC.
# Training math is unaffected: batch sizes, n_samples_per_prompt and lr are all
# explicit CLI args, so these only change KV cache size and throughput.
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

# VENV_DIR override: .venv is now the EVAL env (vllm 0.26 pins torch==2.11.0) and
# can no longer run training -- sglang 0.5.9 pins torch==2.9.1, and its sgl_kernel
# .so is built against the torch 2.9 ABI. Installing vllm on 2026-07-28 22:30
# upgraded torch in place and broke sgl_kernel + flash_attn.
# .venv-train holds the stack the other 5 ablations actually ran on
# (py3.10.18 + torch 2.9.1 + cu128 + sglang 0.5.9, per docker/Dockerfile.sgl059.torch291.cu128).
VENV_DIR=${VENV_DIR:-/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token/.venv-train}
PYTHON=${VENV_DIR}/bin/python
RAY=${VENV_DIR}/bin/ray

export CUDA_HOME=/usr/local/cuda-12.8

# PATH needs two fixes, and they pull in opposite directions -- do both.
# 1. Drop Homebrew. Its as/ld use brew's own loader (~/.linuxbrew/lib/ld.so), which
#    does not search /lib64. Sleep mode sets LD_PRELOAD=<torch_memory_saver hook>,
#    that .so NEEDs libcuda.so.1, LD_PRELOAD is inherited by `as` during the triton /
#    sgl-kernel JIT build, and brew's as then dies with
#    "libcuda.so.1: cannot open shared object file". Surfaces misleadingly as
#    "Server process terminated unexpectedly". Same fix as run_all_ablations.sh:42.
# 2. But brew was ALSO the only ninja on PATH, and flashinfer JIT-builds its prefill
#    module with ninja -> "FileNotFoundError: 'ninja'". The venv ships one
#    (pip ninja 1.13.0), so prepend ${VENV_DIR}/bin to get ninja WITHOUT brew's binutils.
# Verify after any change: `which as ld` must be /usr/bin/*, `which ninja` must be
# ${VENV_DIR}/bin/ninja.
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/.linuxbrew/' | paste -sd:)
export PATH=${VENV_DIR}/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH

echo "==== PATH CHECK ===="
echo "as    -> $(command -v as || echo MISSING)"
echo "ld    -> $(command -v ld || echo MISSING)"
echo "ninja -> $(command -v ninja || echo MISSING)"
case "$(command -v as)" in *linuxbrew*) echo "FATAL: brew as on PATH" >&2; exit 1;; esac
command -v ninja >/dev/null || { echo "FATAL: ninja missing" >&2; exit 1; }

# 4 GPUs / TP=4. A long 2-GPU attempt on 2026-07-30 (job 223593) was ABANDONED; keeping
# the findings here because they are not obvious and cost ~2h.
#
# Cost of 4 GPUs: on_policy_kd_trainer.py:118 computes
#   grad_accum = train_batch_size // (micro_train_batch_size * num_nodes * num_gpus_per_node)
# so 4 GPUs drops grad_accum 2 -> 1 versus the 6 completed ablations. Global batch is
# identical at 4; only the accumulation schedule differs. That is the one known delta.
#
# Why 2 GPUs does NOT work on 46GB L40S, measured, not guessed:
#   student rank0 awake    23.08GB  <- 4B params 2-way sharded: 4 wt + 4 grad + 16 adam fp32
#   teacher sched ASLEEP   19.62GB  <- identical at teacher frac 0.85 AND 0.76, so the
#                                      fraction is NOT the lever; sleep does not give it back
#   rollout sched asleep    1.54GB
#                          -------
#                          44.24GB of 44.39GB -> student wakeup OOMs by 48MB
# The student is ~12GB/GPU at 4 GPUs, which is why this only ever bites at 2. The 2x H100
# pair has 93GB/card so it never bit there either.
# --cpu_offload True would fix it (student -> ~8GB) but FSDP2 refuses:
#   "FSDP parameters should be materialized on CPU when enabling CPU offloading"
# because student_actor.py loads the model onto GPU before wrapping. Fixing that means
# editing shared code used by all 7 ablation scripts, which was out of scope.
#
# num_key_value_heads=4 divides evenly by 4 and by 2.
# If you ever go back to 2 GPUs you must ALSO set TEACHER_MEM_FRAC=0.76 and patch
# student_actor.py for cpu_offload. Fractions alone will not get you there.
NUM_GPUS=${NUM_GPUS:-4}
TEACHER_TP=${TEACHER_TP:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# fp8 OFF by default. --teacher_quantization fp8 is NOT harmlessly ignored for
# Qwen3-30B-A3B on sglang 0.5.9, contrary to the note further up this file. It does not
# shrink the weights, but it costs ~10GB/GPU extra. Survivable at TP=4 (15.27GB weights),
# fatal at TP=2 (30.53GB weights). Isolated proof, teacher alone on 2 idle L40S, frac 0.85:
#   with fp8:    RuntimeError "Not enough memory ... mem_fraction_static" at 0.71/0.85/0.92
#   without fp8: ENGINE STARTED OK, used=40.55GB/GPU, free=3.85GB
# Set TEACHER_QUANT_ARG="--teacher_quantization fp8" to put it back (only sane at TP=4).
TEACHER_QUANT_ARG=${TEACHER_QUANT_ARG:-}

# Student rank0 OOM'd by 48MB on the first wakeup at teacher frac 0.85 (fsdp_strategy.py:619
# reload_optim_states). Per-GPU budget is genuinely this tight at 2 GPUs:
#   rollout asleep 1.54 + teacher asleep ~19.6 + student awake 23.08 = 44.24 of 44.39GB
# The student's 23GB is IRREDUCIBLE here: 4B params sharded 2 ways = 4GB weights + 4GB
# grads + 16GB adam fp32 states. At 4 GPUs it is ~12GB, which is why this never bit before.
# So the give has to come from the teacher. Its KV pool at 0.85 was 7.2GB for a ~1GB need.
# Measured, teacher alone, no fp8:  frac 0.85 -> 40.55GB used ; frac 0.76 -> 36.79GB used.
# 0.76 frees 3.76GB, far more than the 48MB shortfall, and still starts cleanly.
#
# DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here, even though the torch
# OOM message explicitly recommends it. torch_memory_saver refuses to run with it:
#   "TorchMemorySaver is disabled for the current process because expandable_segments
#    is not supported yet"
# and torch_memory_saver IS the sleep/wake mechanism the whole 2-GPU plan depends on.
# Tried 2026-07-30, killed the rollout scheduler at init. The advice is generic torch
# advice and does not apply to a sleep-mode stack.
#
# --cpu_offload True (FSDP2 CPUOffloadPolicy) is what ACTUALLY makes 2 GPUs fit, and the
# teacher fraction was a red herring for this second OOM. Measured per-GPU at the first
# student wakeup, identical at teacher frac 0.85 AND 0.76 (which is how we know the
# teacher fraction was not the lever):
#   teacher sglang scheduler, ASLEEP  19.62GB   <- does not shrink with teacher frac
#   rollout scheduler, asleep          1.54GB
#   student rank0, awake              23.08GB   <- 4GB weights + 4GB grads + 16GB adam fp32
#                                     ------
#                                     44.24GB of 44.39GB -> OOM by 48MB
# The student 23GB is irreducible at tp=2 sharding (it is ~12GB at 4 GPUs, which is why
# 4-GPU runs never hit this, and the H100 pair has 93GB so it never hit this either).
# cpu_offload parks params+optimizer in host RAM, taking the student to ~8GB resident.
# It does NOT change the math: same grads, same updates, only where tensors live. It is
# slower (host<->device traffic each step), which is the price of running this on 2 cards.
# Set CPU_OFFLOAD=False to disable (only sane at 4 GPUs or on 93GB cards).
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
SAVE_DIR=/umbc/rs/pi_ferraro/ada/users/sroydip1/collab/Rock-Token-checkpoints/rock_freeze

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
  --teacher_tp_size ${TEACHER_TP} \
  --teacher_dp_size 1 \
  --teacher_ep_size 1 \
  --teacher_pp_size 1 \
  --teacher_enable_sleep True \
  --teacher_forward_n_batches 1 \
  ${TEACHER_QUANT_ARG} \
  --teacher_mem_fraction_static ${TEACHER_MEM_FRAC:-0.55} \
  --rollout_num_engines 1 \
  --rollout_tp_size ${ROLLOUT_TP:-1} \
  --rollout_batch_size 2 \
  --n_samples_per_prompt 4 \
  --generate_max_len 1024 \
  --temperature 1.0 \
  --top_p 1.0 \
  --rollout_enable_sleep True \
  --rollout_mem_fraction_static ${ROLLOUT_MEM_FRAC:-0.25} \
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

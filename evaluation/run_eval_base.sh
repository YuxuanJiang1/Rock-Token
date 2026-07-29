#!/bin/bash
# run_eval_base.sh — Base model eval (offpolicy-20k, before onpolicy)
# 2 x A100 80GB, tensor_parallel_size=2

PYTHON=/p/work2/yuxuanj1/conda_envs/lmeval_env/bin/python
HARNESS=/p/work2/yuxuanj1/opd/lm-evaluation-harness
MODEL=/p/work2/yuxuanj1/opd/models/qwen3-30b-a3b-to-4b-offpolicy-20k
EVAL_OUT=/p/work2/yuxuanj1/opd/eval_results/base_vllm
TASKS=aime24_sample,aime25_sample,hmmt25feb

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p ${EVAL_OUT}

cd ${HARNESS}

echo "============================================================"
echo "  BASE Model vLLM Eval  $(date)"
echo "  Model: qwen3-30b-a3b-to-4b-offpolicy-20k"
echo "  Tasks: ${TASKS}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}  (tp=2, A100 80GB)"
echo "============================================================"

for RUN in 1 2 3; do
    SEED=$((42 + RUN * 17))
    OUT_DIR=${EVAL_OUT}/run_${RUN}
    mkdir -p ${OUT_DIR}

    echo ""
    echo "------------------------------------------------------------"
    echo "  RUN ${RUN}/3   seed=${SEED}   $(date)"
    echo "------------------------------------------------------------"

    ${PYTHON} -m lm_eval \
        --model vllm \
        --model_args "pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.9,max_model_len=20000,enforce_eager=False" \
        --tasks ${TASKS} \
        --num_fewshot 0 \
        --seed ${SEED} \
        --batch_size auto \
        --output_path ${OUT_DIR} \
        --log_samples \
        2>&1 | tee ${OUT_DIR}/eval.log

    echo "[run ${RUN}] done at $(date)"
done

echo ""
echo "============================================================"
echo "  ALL 3 RUNS COMPLETE — aggregating results  $(date)"
echo "============================================================"

${PYTHON} - << 'PY'
import json, glob, statistics, os

eval_out = "/p/work2/yuxuanj1/opd/eval_results/base_vllm"
tasks = ["aime24_sample", "aime25_sample", "hmmt25feb"]

results = {t: [] for t in tasks}
for run in [1, 2, 3]:
    files = glob.glob(f"{eval_out}/run_{run}/**/*.json", recursive=True)
    for f in files:
        if "samples" in os.path.basename(f):
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
        except Exception:
            continue
        if "results" not in data:
            continue
        for task in tasks:
            if task in data["results"]:
                r = data["results"][task]
                acc = r.get("exact_match,none", r.get("exact_match", None))
                if acc is not None:
                    results[task].append(round(acc * 100, 2))

print("\n=== BASE Model — 3-Run Benchmark Summary ===")
print(f"  Model : qwen3-30b-a3b-to-4b-offpolicy-20k")
print(f"  Seeds : 59, 76, 93  |  temp=0.6  |  max_gen=16384")
print()
print(f"{'Task':<22} {'Run1':>7} {'Run2':>7} {'Run3':>7} {'Mean':>8} {'Std':>7}")
print("-" * 65)
task_means = []
for task in tasks:
    vals = results[task]
    mean = statistics.mean(vals) if vals else float('nan')
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    if vals: task_means.append(mean)
    row = f"{task:<22}"
    for v in vals:
        row += f" {v:>7.1f}"
    for _ in range(3 - len(vals)):
        row += f"  {'N/A':>6}"
    if vals:
        row += f" {mean:>8.1f} {std:>7.1f}"
    print(row)
if task_means:
    avg = statistics.mean(task_means)
    print("-" * 65)
    print(f"{'Avg across tasks':<22}  {'':>7}  {'':>7}  {'':>7}  {avg:>8.1f}")
PY

echo "Done."

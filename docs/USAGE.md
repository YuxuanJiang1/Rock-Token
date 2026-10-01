# Public usage

This release contains research code and paper figures. It does not include the trained checkpoints, raw collected KL/gradient tensors, or the original training JSONL files. The historical `RockToken/` checkpoint presets may require access; they are not public-weight download links.

## 1. Token analysis

Use Python 3.10+ and a CUDA environment with enough memory for both models. Install the analysis dependencies:

```bash
pip install -r rock_detection/requirements.txt
cd rock_detection
python rock_server.py \
  --student-id /path/to/student-checkpoint \
  --teacher-id Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --output-tag onpolicy --samples 500 --unrestricted --hardware single_96gb
python rerun_unrestricted.py
```

The student and teacher must have compatible tokenizers and vocabulary semantics. Supplying an unrelated public model is not a reproduction of the paper's distilled checkpoint. Subsequent scripts use the generated filenames described in the [analysis guide](../rock_detection/README.md).

## 2. Distillation

Install the dependencies for the selected KDFlow variant according to its README. `KDFlow_localopd/` and `stumbling/` both provide a `kdflow` package: use separate environments or run through the corresponding launcher, which places its own code first on `PYTHONPATH`.

From the repository root, configure:

```bash
export STUDENT_MODEL=/absolute/path/to/student-checkpoint
export TEACHER_MODEL=/absolute/path/to/teacher-checkpoint
export TRAIN_DATA=/absolute/path/to/training.jsonl
export SAVE_DIR=/absolute/path/to/output
export CUDA_VISIBLE_DEVICES=0,1
# Optional: export PYTHON=/path/to/environment/bin/python
```

The launch configurations expect a `prompt_messages` field compatible with the model's chat template. They use two GPUs; memory requirements depend on model size and sequence length. CUDA must already be configured in the environment.

```bash
# Standard OPD
bash KDFlow_localopd/run_localopd.sh

# Frequency-matched random masking
bash stumbling/run_stumb_random.sh

# Rock-token masking (same training configuration, different token set)
TOKEN_FREEZE_PATH="$PWD/stumbling/rock.json" bash stumbling/run_stumb_random.sh
```

Use a distinct `SAVE_DIR` for each run. The two supplied launchers preserve their original experiment hyperparameters, which differ; align training settings before using them for a controlled comparison. The JSON token lists are tokenizer-specific. The scripts do not terminate existing processes; `RAY_ADDRESS` can be set to use an existing cluster.

## 3. Evaluation

```bash
pip install lm-eval vllm
MODEL=/path/to/checkpoint EVAL_OUT=/path/to/fresh/results \
  bash evaluation/run_eval_vllm.sh
```

Both evaluation launchers use vLLM with tensor parallelism of 2, load the included task YAMLs, and evaluate three seeds (59, 76, 93). See the [evaluation guide](../evaluation/README.md) for metrics and generation settings. Use a fresh output directory to avoid mixing results from different runs.

## Validation status

The public-release changes were checked for shell/Python syntax, configuration handling, local documentation links, and launcher argument construction. Full GPU training and benchmark reproduction were not run as part of this cleanup.

## Licensing

The bundled KDFlow directories retain their upstream license files. No new repository-wide license is assigned by this cleanup; refer to the applicable files and contact the authors about uses not covered by them.

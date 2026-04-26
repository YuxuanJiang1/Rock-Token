# Makefile — Rock Token experiments

STUDENT ?= RockToken/qwen3_30b_a3b_to_4b_onpolicy_5k_src20k-25k
TEACHER ?= Qwen/Qwen3-30B-A3B-Instruct-2507
OUTPUT  ?= results/exp2
TOP_K   ?= 50
N_SAMPLES ?=

# --- Exp 2: Rock Token Identification ---

MODEL   ?= $(STUDENT)

.PHONY: exp2 exp2-geometric exp2-phase1 exp2-phase2 exp2-phase3 \
       eval-aime eval-gsm8k eval-mmlu eval-humaneval eval-ifeval eval-all \
       baseline baseline-smoke summarize \
       identify identify-phase1 identify-phase2 identify-phase3 \
       test help

exp2:  ## Run full pipeline (bayesian scoring, default)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--scoring bayesian --top-k $(TOP_K) --output-dir $(OUTPUT) \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

exp2-geometric:  ## Run full pipeline with geometric scoring
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--scoring geometric --top-k $(TOP_K) --output-dir $(OUTPUT) \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

exp2-phase1:  ## Run only Phase 1 (student generation)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--phase 1 --output-dir $(OUTPUT) \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

exp2-phase2:  ## Run from Phase 2 (teacher KL)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--phase 2 --output-dir $(OUTPUT)

exp2-phase3:  ## Run only Phase 3 (re-analyze, no GPU needed)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--phase 3 --scoring bayesian --top-k $(TOP_K) --output-dir $(OUTPUT)

# --- Evaluation ---

eval-aime:  ## Evaluate model on AIME (MODEL=..., N_SAMPLES=...)
	uv run python src/analysis/eval_aime.py \
		--model $(MODEL) --output results/aime_$(subst /,_,$(MODEL)).json \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

eval-gsm8k:  ## Evaluate model on GSM8K (MODEL=..., N_SAMPLES=...)
	uv run python src/evaluation/eval_gsm8k.py \
		--model $(MODEL) --output results/gsm8k_$(subst /,_,$(MODEL)).json \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

eval-mmlu:  ## Evaluate model on MMLU 5-shot (MODEL=..., N_SAMPLES=...)
	uv run python src/evaluation/eval_mmlu.py \
		--model $(MODEL) --output results/mmlu_$(subst /,_,$(MODEL)).json \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

eval-humaneval:  ## Evaluate model on HumanEval pass@1 (MODEL=..., N_SAMPLES=...)
	uv run python src/evaluation/eval_humaneval.py \
		--model $(MODEL) --output results/humaneval_$(subst /,_,$(MODEL)).json \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

eval-ifeval:  ## Evaluate model on IF-Eval (MODEL=..., N_SAMPLES=...)
	uv run python src/evaluation/eval_ifeval.py \
		--model $(MODEL) --output results/ifeval_$(subst /,_,$(MODEL)).json \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

eval-all:  ## Run all benchmarks (MODEL=..., N_SAMPLES=...)
	$(MAKE) eval-gsm8k eval-mmlu eval-humaneval eval-ifeval

# --- Baseline ---

BASELINE_DIR ?= results/baseline

baseline:  ## Run full baseline (all benchmarks, MODEL=...)
	uv run python src/evaluation/run_baseline.py \
		--model $(MODEL) --output-dir $(BASELINE_DIR) \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

baseline-smoke:  ## Quick smoke-test baseline (10 samples, MODEL=...)
	uv run python src/evaluation/run_baseline.py \
		--model $(MODEL) --output-dir results/smoke --n-samples 10

summarize:  ## Summarize results in a directory (DIRS=..., LABELS=...)
	uv run python src/evaluation/summarize_results.py \
		--dirs $(DIRS) $(if $(LABELS),--labels $(LABELS))

# --- Identification ---

IDENT_DIR ?= results/identification

identify:  ## Run full Rock Token identification pipeline
	uv run python src/identification/run.py --output-dir $(IDENT_DIR)

identify-phase1:  ## Run only Phase 1 (vLLM generation)
	uv run python src/identification/run.py --phase 1 --output-dir $(IDENT_DIR)

identify-phase2:  ## Run only Phase 2 (KL measurement, needs GPU)
	uv run python src/identification/run.py --phase 2 --output-dir $(IDENT_DIR)

identify-phase3:  ## Run Phase 3+4 + plots (CPU only)
	uv run python src/identification/run.py --phase 3 --output-dir $(IDENT_DIR)

# --- Testing ---

test:  ## Run all tests
	uv run pytest tests/ -v

# --- Help ---

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

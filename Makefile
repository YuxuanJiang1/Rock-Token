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
       masking-baseline masking-baseline-smoke \
       masking-eval-math500 masking-eval-aime24 masking-eval-aime25 \
       masking-eval-hmmt masking-eval-ifeval \
       masking-knockout masking-knockout-smoke masking-categorize \
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

_check-variant:
ifndef VARIANT
	$(error VARIANT is required. Usage: make identify VARIANT=onpolicy|offpolicy)
endif

identify: _check-variant  ## Run full identification pipeline (VARIANT=onpolicy|offpolicy)
	uv run python src/identification/run.py --variant $(VARIANT)

identify-phase1: _check-variant  ## Run only Phase 1 (VARIANT=onpolicy|offpolicy)
	uv run python src/identification/run.py --phase 1 --variant $(VARIANT)

identify-phase2: _check-variant  ## Run only Phase 2 (VARIANT=onpolicy|offpolicy)
	uv run python src/identification/run.py --phase 2 --variant $(VARIANT)

identify-phase3: _check-variant  ## Run Phase 3+4 + plots (VARIANT=onpolicy|offpolicy, CPU only)
	uv run python src/identification/run.py --phase 3 --variant $(VARIANT)

# --- Part 2: Masking Experiments ---

MASKING_DIR ?= results/masking/baseline

masking-baseline:  ## Run baseline on all 4 models × 5 benchmarks (MASKING_DIR=...)
	uv run python src/masking/run_baseline.py \
		--output-dir $(MASKING_DIR) \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

masking-baseline-smoke:  ## Quick smoke test (5 samples, MASKING_DIR=...)
	uv run python src/masking/run_baseline.py \
		--output-dir results/masking/smoke --n-samples 5

masking-eval-math500:  ## Eval single model on MATH-500 (MODEL=...)
	uv run python src/masking/eval_math500.py \
		--model $(MODEL) --output results/masking/math500_$(subst /,_,$(MODEL)).json \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

masking-eval-aime24:  ## Eval single model on AIME 2024 (MODEL=...)
	uv run python src/masking/eval_aime.py --year 2024 \
		--model $(MODEL) --output results/masking/aime2024_$(subst /,_,$(MODEL)).json \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

masking-eval-aime25:  ## Eval single model on AIME 2025 (MODEL=...)
	uv run python src/masking/eval_aime.py --year 2025 \
		--model $(MODEL) --output results/masking/aime2025_$(subst /,_,$(MODEL)).json \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

masking-eval-hmmt:  ## Eval single model on HMMT Feb 2025 (MODEL=...)
	uv run python src/masking/eval_hmmt.py \
		--model $(MODEL) --output results/masking/hmmt25_$(subst /,_,$(MODEL)).json \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

masking-eval-ifeval:  ## Eval single model on IF-Eval (MODEL=...)
	uv run python src/masking/eval_ifeval.py \
		--model $(MODEL) --output results/masking/ifeval_$(subst /,_,$(MODEL)).json \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))


# -- 2.1 Knockout ---

IDENTIFICATION_DIR ?= results/identification/onpolicy
CATEGORY ?= count

masking-knockout:  ## Run individual knockout (CATEGORY=count|rate, IDENTIFICATION_DIR=...)
	uv run python src/masking/knockout.py \
		--rock-tokens $(IDENTIFICATION_DIR)/rock_tokens_by_$(CATEGORY).csv \
		--category $(CATEGORY) \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

masking-knockout-smoke:  ## Quick knockout smoke test (5 samples, 3 tokens)
	uv run python src/masking/knockout.py \
		--rock-tokens $(IDENTIFICATION_DIR)/rock_tokens_by_$(CATEGORY).csv \
		--category $(CATEGORY) \
		--output-dir results/masking/knockout_smoke \
		--n-samples 5 --n-tokens 3

masking-categorize:  ## Step 2.2: bootstrap-based statistical categorization (CPU only)
	uv run python src/masking/categorize.py \
		--knockout-dir results/masking/knockout/$(CATEGORY) \
		--rock-tokens $(IDENTIFICATION_DIR)/rock_tokens_by_$(CATEGORY).csv

# --- Testing ---

test:  ## Run all tests
	uv run pytest tests/ -v

# --- Help ---

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Makefile — Rock Token experiments

STUDENT ?= RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k
TEACHER ?= Qwen/Qwen3-30B-A3B-Instruct-2507
OUTPUT  ?= results/exp2
TOP_K   ?= 50

# --- Exp 2: Rock Token Identification ---

.PHONY: exp2 exp2-geometric exp2-phase1 exp2-phase2 exp2-phase3 test lint

exp2:  ## Run full pipeline (bayesian scoring, default)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--scoring bayesian --top-k $(TOP_K) --output-dir $(OUTPUT)

exp2-geometric:  ## Run full pipeline with geometric scoring
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--scoring geometric --top-k $(TOP_K) --output-dir $(OUTPUT)

exp2-phase1:  ## Run only Phase 1 (student generation)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--phase 1 --output-dir $(OUTPUT)

exp2-phase2:  ## Run from Phase 2 (teacher KL)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--phase 2 --output-dir $(OUTPUT)

exp2-phase3:  ## Run only Phase 3 (re-analyze, no GPU needed)
	uv run python src/exp_2/identify_rock_tokens.py \
		--student $(STUDENT) --teacher $(TEACHER) \
		--phase 3 --scoring bayesian --top-k $(TOP_K) --output-dir $(OUTPUT)

# --- Testing ---

test:  ## Run all tests
	uv run pytest tests/ -v

# --- Help ---

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

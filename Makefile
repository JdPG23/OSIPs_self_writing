# OSIP Self-Writing — Common Operations

.PHONY: install run quick baseline results best clean scrape help

# Default topic (override with: make run TOPIC="your topic")
TOPIC ?= "Autonomous GNC for active debris removal"

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

run: ## Run one experiment (TOPIC="your topic")
	python run.py --topic $(TOPIC)

quick: ## Run one experiment with quick scoring
	python run.py --topic $(TOPIC) --quick

baseline: ## Run baseline (unmodified pipeline) and log it
	python run.py --topic $(TOPIC) > run.log 2>&1
	@echo "--- Baseline Results ---"
	@grep "^overall_score:\|^alignment_score:\|^structure_score:\|^quality_score:\|^novelty_score:\|^token_cost:" run.log

results: ## Show results sorted by score (best first)
	@echo "commit\tscore\talign\tstruct\tqual\tnovel\ttokens\tstatus\tdescription"
	@tail -n +2 results.tsv 2>/dev/null | sort -t$$'\t' -k2 -rn || echo "(no results yet)"

best: ## Show the single best experiment
	@tail -n +2 results.tsv 2>/dev/null | sort -t$$'\t' -k2 -rn | head -1 || echo "(no results yet)"

clean: ## Remove generated outputs and logs
	rm -rf outputs/*.md run.log

reset-results: ## Reset results.tsv to header only (destructive!)
	@echo "commit\tscore\talignment\tstructure\tquality\tnovelty\ttokens\tstatus\tdescription" > results.tsv
	@echo "Results reset."

validate: ## Validate project setup
	@python -c "from config import *; print('config.py OK')"
	@python -c "from prepare import load_corpus, get_corpus_summary; print(get_corpus_summary())"
	@python -c "from llm_client import call_llm; print('llm_client.py OK')"
	@python -c "from scorer import score_proposal; print('scorer.py OK')"
	@python -c "from pipeline import run_pipeline; print('pipeline.py OK')"
	@echo "All modules OK."

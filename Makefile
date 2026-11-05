.DEFAULT_GOAL := help
.PHONY: help sync lint fmt typecheck test check fixtures fixtures-check up down ingest ask chat serve eval eval-fast baseline

UV ?= uv
Q ?=

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

sync: ## Install pinned deps (incl. dev group) into .venv
	$(UV) sync --locked

lint: ## ruff lint + format check + mypy
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

fmt: ## Auto-fix lint and format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## mypy only
	$(UV) run mypy

test: ## Unit + integration tests (mock LLM, fixture data, no credentials)
	$(UV) run pytest

check: lint test ## Everything CI runs

fixtures: ## Regenerate deterministic fixture data (servers/fixtures/data)
	$(UV) run python -m servers.fixtures.generate

fixtures-check: ## Fail if committed fixture data is out of date
	$(UV) run python -m servers.fixtures.generate --check

# --- Targets below are wired up in later build steps -------------------------

up: ## docker compose up (Qdrant + Langfuse)            [step c]
	@echo "not implemented yet: lands in Phase 1 step c" && exit 1

down: ## docker compose down (keeps volumes)            [step c]
	@echo "not implemented yet: lands in Phase 1 step c" && exit 1

ingest: ## Ingest fixture (or live) data into Qdrant     [step d]
	@echo "not implemented yet: lands in Phase 1 step d" && exit 1

ask: ## Ask one question: make ask Q="..."              [step e]
	@echo "not implemented yet: lands in Phase 1 step e" && exit 1

chat: ## Multi-turn chat with a thread id               [step g]
	@echo "not implemented yet: lands in Phase 2 step g" && exit 1

serve: ## FastAPI on :8000                               [step l]
	@echo "not implemented yet: lands in Phase 3 step l" && exit 1

eval: ## Full eval suite -> reports/metrics.json         [step p]
	@echo "not implemented yet: lands in Phase 4 step p" && exit 1

eval-fast: ## Deterministic eval subset (< 3 min)        [step p]
	@echo "not implemented yet: lands in Phase 4 step p" && exit 1

baseline: ## Copy metrics -> evals/baseline.json (owner approval required) [step r]
	@echo "not implemented yet: lands in Phase 4 step r" && exit 1

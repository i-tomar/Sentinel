.DEFAULT_GOAL := help
.PHONY: help sync lint fmt typecheck test check fixtures fixtures-check infra-env up up-qdrant down ps logs infra-check ingest ask chat serve eval eval-fast baseline

UV ?= uv
COMPOSE ?= docker compose --env-file .env.infra
UP_TIMEOUT ?= 300
Q ?=
S ?=

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

# --- Local infrastructure (docker-compose.yml) --------------------------------
# No target deletes volumes; see README for `down -v`.

.env.infra:
	$(UV) run sentinel infra init

infra-env: .env.infra ## Generate .env.infra (random local credentials) if missing

up: .env.infra ## Start Qdrant + Langfuse and wait until healthy
	$(COMPOSE) up -d --wait --wait-timeout $(UP_TIMEOUT)

up-qdrant: .env.infra ## Start only Qdrant
	$(COMPOSE) up -d --wait qdrant

down: ## Stop services (keeps data volumes)
	$(COMPOSE) down

ps: ## Service status
	$(COMPOSE) ps

logs: ## Follow logs: make logs S=langfuse-web
	$(COMPOSE) logs -f --tail 100 $(S)

infra-check: ## Check Qdrant/Langfuse health and Langfuse keys
	$(UV) run sentinel infra check --wait 60

# --- Targets below are wired up in later build steps -------------------------

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

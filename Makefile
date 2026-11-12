.DEFAULT_GOAL := help
.PHONY: help sync lint fmt typecheck test check fixtures fixtures-check infra-env up up-qdrant down ps logs infra-check ingest search ask chat approvals approve deny serve eval eval-fast baseline

UV ?= uv
COMPOSE ?= docker compose --env-file .env.infra
UP_TIMEOUT ?= 300
Q ?=
S ?=

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

sync: ## Install pinned deps (dev group + optional LLM provider extras) into .venv
	$(UV) sync --locked --all-extras

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

ingest: ## Sync fixture records (over MCP) into Qdrant; idempotent
	$(UV) run sentinel ingest

search: ## Debug retrieval: make search Q="auth migration owner"
	$(UV) run sentinel search "$(Q)"

ask: ## Ask one question with citations: make ask Q="..."
	$(UV) run sentinel ask "$(Q)"

chat: ## Multi-turn chat; continue a thread with T=<thread id>
	$(UV) run sentinel chat $(if $(T),--thread $(T),)

approvals: ## List approval requests (ids, status, tool; no arguments)
	$(UV) run sentinel approvals

approve: ## Approve and run one proposed write: make approve A=<approval id> S=<session id>
	$(UV) run sentinel approve $(A) --session $(S)

deny: ## Deny one proposed write: make deny A=<approval id> S=<session id>
	$(UV) run sentinel deny $(A) --session $(S)

serve: ## FastAPI on :8000                               [step l]
	@echo "not implemented yet: lands in Phase 3 step l" && exit 1

eval: ## Eval suite (mock LLM, fixtures) -> reports/metrics.json + report.md; exit 1 if the gate fails
	$(UV) run python -m evals.run

eval-fast: ## Deterministic eval subset (< 3 min)        [step p]
	@echo "not implemented yet: lands in Phase 4 step p" && exit 1

baseline: ## Copy metrics -> evals/baseline.json (owner approval required) [step r]
	@echo "not implemented yet: lands in Phase 4 step r" && exit 1

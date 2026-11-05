# Sentinel

Guarded agentic RAG over Notion, Jira and Gmail via MCP, with cited answers, prompt-injection
defense, human approval for risky actions, and an eval suite that gates CI.

> Status: early scaffold (Phase 1). Results: `TBD (run make eval)`.

## Quick start (development)

```bash
uv sync                 # installs pinned deps from uv.lock (Python 3.12)
cp .env.example .env    # defaults work with no credentials (LLM_PROVIDER=mock, DATA_MODE=fixture)
make check              # ruff + mypy + pytest
uv run sentinel config  # show effective settings (secrets masked)
```

On Windows without GNU make, run the underlying commands from the `Makefile` directly
(e.g. `uv run pytest`).

## Local infrastructure (Qdrant + Langfuse)

`docker-compose.yml` runs Qdrant v1.20.0 and a self-hosted Langfuse v4.76.0 stack (web, worker,
Postgres, ClickHouse, Redis, MinIO). All ports bind to `127.0.0.1`. Credentials are random
per machine, generated into the git-ignored `.env.infra`. The app defaults stay
`LLM_PROVIDER=mock` and `TRACING_BACKEND=none`. Tracing is turned on only through `.env`.

```bash
make up            # creates .env.infra on first run, starts everything, waits for health checks
make infra-check   # Qdrant ready, Langfuse healthy, Langfuse API keys accepted
make ps | make logs S=langfuse-web | make down   # down keeps all data volumes
```

Without GNU make (e.g. Windows PowerShell):

```bash
uv run sentinel infra init                                   # once; refuses to overwrite
docker compose --env-file .env.infra up -d --wait --wait-timeout 300
uv run sentinel infra check --wait 60
docker compose --env-file .env.infra down
```

Enable Langfuse tracing by adding the lines printed by `uv run sentinel infra app-env` to `.env`
(`TRACING_BACKEND=langfuse` plus the project keys). Open http://localhost:3000 and log in as
`dev@sentinel.test` with `LANGFUSE_INIT_USER_PASSWORD` from `.env.infra`. Langfuse v4 accepts
traces only via OpenTelemetry (`/api/public/otel/v1/traces`). The Langfuse SDK uses this path.

The first `make up` pulls ~2 GB of images. Langfuse needs a few GB of RAM.

> **Data loss:** `docker compose --env-file .env.infra down -v` deletes every volume (traces,
> vectors). Regenerating `.env.infra` with `--force` also makes existing volumes unusable,
> because the stored passwords no longer match. Do either only if you mean to wipe local data.

## Fixture MCP servers

CI and local development use fake Notion, Jira and Gmail MCP servers over a seeded "Acme"
dataset (`servers/fixtures/data/`). No real accounts are involved; nothing is ever sent.

```bash
uv run python -m servers.fixtures.notion_mcp   # stdio; also jira_mcp, gmail_mcp
make fixtures                                  # regenerate data deterministically
make fixtures-check                            # fail if committed data is stale
```

Inject failures per source or tool, e.g.
`SENTINEL_FIXTURE_FAILURES="gmail=timeout,jira.get_issue=error_500"`
(modes: `timeout`, `error_500`, `empty`, `malformed`, `rate_limit`).

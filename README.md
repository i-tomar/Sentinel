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

## Ingestion and retrieval

`sentinel ingest` reads every record through the source MCP tools (in fixture mode it starts the
fixture servers as stdio subprocesses). It turns each record into a `Document` with source, id,
url, title, text, `updated_at` from the source and `fetched_at`, and syncs the chunks into
Qdrant:

- Unchanged records are skipped; only their `fetched_at` is refreshed.
- Changed records have all their chunks replaced.
- Records that disappeared from the source are deleted.
- A source whose tools fail is reported, and its existing index is left untouched.

```bash
make ingest                                   # or: uv run sentinel ingest [-s notion -s jira]
make search Q="who owns the auth migration?"  # or: uv run sentinel search "..." -k 5 --mode hybrid
```

- **Hybrid search.** Dense vectors and keyword vectors (saturated term frequency, with IDF
  applied by Qdrant) are fused with reciprocal rank fusion. A third list matches record IDs
  named in the query, so `PROJ-142` returns the ticket itself first.
- **Embeddings.** `EMBEDDING_PROVIDER=local` (default) uses fastembed `BAAI/bge-small-en-v1.5`,
  downloaded once with no API key. `hash` is an offline, deterministic lexical stand-in for
  tests and CI. Each embedder gets its own collection (`<QDRANT_COLLECTION>_<embedder>`).
- **Storage.** `QDRANT_URL` can be the compose server (`http://127.0.0.1:6333`), `:memory:`, or
  a directory for embedded on-disk Qdrant (no Docker).
- **Citations.** `[n]` numbers records, not chunks, and renders source, title, id, url, and
  updated/fetched times. Markers that don't match retrieved evidence are flagged.
- **Kept out of the index.** Fixture ground truth (`manifest.json`) and eval data are never
  indexed. The test suite checks this.
- **Untrusted content.** Indexed content is untrusted. Poisoned fixture records are indexed as
  they are, and filtering them is the job of the retrieved-content guard (Phase 3).

Opt-in tests: `SENTINEL_TEST_QDRANT_URL=http://127.0.0.1:6333` (real Qdrant server, throwaway
collection) and `SENTINEL_TEST_FASTEMBED=1` (downloads the local model).

## Multi-hop planning and tool calls (step f)

```
plan -> route -> execute_tools -> retrieve -> verify --(next hop, <= MAX_HOPS)--> route
                                                     \-> answer -> citation_check
```

```bash
uv run sentinel ask "Which tickets are blocked by the auth migration and who emailed about them?" --as-of 2026-11-02
uv run sentinel ask "..." --no-tools      # index only, no live MCP calls
```

- **Planner.** Splits the question into typed sub-questions: id, text, target sources,
  dependencies, relation (`search`, `blocked_by`, `mentions`), keyword query, and optional
  time window or label. It only ever sees the user's question, so retrieved content cannot
  change the plan. The mock provider uses a deterministic rule planner. Real providers use an
  LLM planner whose JSON is validated, with the rule plan as fallback.
- **Tool router.** Turns ready sub-questions into calls from a read-only allowlist (search and
  get per source). Dependent hops take only strictly-matched ticket keys (at most 6) from
  earlier, non-stale results. "Last week" is resolved against `--as-of` (Monday-based weeks).
- **execute_tools.** Each call gets a `TOOL_TIMEOUT_S` timeout and up to 2 retries with
  exponential backoff. Failures are recorded and the answer names the unreadable source
  instead of failing.
- **Verifier.** Routes the next ready sub-question, retries an empty search once over the
  other allowed sources, and marks superseded Notion pages (same title apart from a
  version/year) as stale so they aren't followed or presented as current. It stops at
  `MAX_HOPS`, and unanswered sub-questions are listed.
- **Writes.** No write or destructive tool is reachable yet. That arrives with the permission
  gate in Phase 3.

## Conversations and short-term memory (step g)

```bash
uv run sentinel chat "Which tickets are blocked by the auth migration?" --thread demo --as-of 2026-11-02
uv run sentinel chat "Who emailed about them?" --thread demo      # continues the same thread
uv run sentinel chat "Who is assigned the second ticket?" -t demo
uv run sentinel chat --history -t demo                            # remembered turns
uv run sentinel chat                                              # interactive; prints a new thread_id
make chat T=demo
```

- **Checkpoints.** Each turn is checkpointed per `thread_id` with LangGraph's SQLite
  checkpointer at `CHECKPOINT_DB` (default `./.data/checkpoints.sqlite`). Memory therefore
  survives restarts, and threads never see each other's state. `sentinel ask` stays stateless.
- **What a thread remembers.** The last 8 turns (question, answer, plan, unanswered
  sub-questions, the records each sub-question found, citations) and only the records those
  turns reference.
- **Follow-ups build on memory instead of starting over:**
  - "who emailed about them?" searches emails for the tickets the previous answer was about.
  - "blocked by those" works the same way for blocked tickets.
  - "the second ticket" answers from the remembered record with no tool calls.
  - "continue" re-runs the previous turn's unresolved sub-questions, seeded with records
    already fetched.
  - A question with no reference words is planned from scratch.
- **Planners never see remembered record text.** They see the user's earlier questions,
  opaque record ids, and unresolved sub-questions, so memory doesn't reopen the injection path.
  LLM-proposed record ids are dropped unless the thread actually fetched them.
- **Checkpoint deserialization is allowlisted.** It only revives Sentinel's own state models.
  A test fails if a new model reachable from the state is not on the list.

## Long-term memory (step h)

Facts that outlive a thread live in a `MemoryBackend` (SQLite at `MEMORY_DB` by default; an
OKF file backend is a stretch-goal stub). Every item carries `source` (the record it came
from, or `user` / `mcp` / `agent`), `verified`, `stale`, `updated_at` and `confidence`.
Search ranks fresh, verified items first.

```bash
uv run sentinel ask "Who owns the Q3 auth migration plan v2?" --auto-approve-memory   # fixture data only
uv run sentinel memory list | search "auth migration owner" | put KEY "VALUE" | mark-stale KEY --reason "..."
uv run python -m servers.memory_mcp.server       # the same memory over MCP (stdio)
```

- **Notes are never evidence.** `load_memory` puts matching notes in the prompt marked
  verified or unverified, under the rule "not evidence, never cite". Citations can only point
  at source records. For each verified note, the agent re-reads its source record (read-only
  get) in the first hop, so the fresh record is what gets cited.
- **Freshness rule.** A note conflicts with its source when the source record was superseded
  by a newer record, or changed after the note was taken and no longer supports it. On
  conflict the source wins: the note is dropped from the prompt for that turn and a
  `mark_stale` write is proposed.
- **Unverified memory never becomes fact.** The citation check removes any answer sentence
  that restates an unverified note without support from the cited evidence, and reports it.
- **Writes are writes.** After an answer, `save_memory` proposes storing the answer as a
  verified fact (only if every sentence passed the citation check against records fetched
  this turn), plus any `mark_stale` updates.
  - With `DATA_MODE=live` every write stays `pending_approval`. The approval flow arrives
    with the permission gate in Phase 3.
  - With fixture data, writes are applied only with `--auto-approve-memory` or
    `MEMORY_AUTO_APPROVE=true`. That setting is refused together with `DATA_MODE=live`.
  - `sentinel memory put` is you writing directly, stored unverified with source `user`.
- **Memory MCP server.** `search_memory` and `get_memory` are read-only; `put_memory` and
  `mark_stale` are annotated as writes. Facts written over MCP are always stored unverified
  with source `mcp`.

## Live connectors (step i): read-only, opt-in

Fixture mode stays the default; CI and tests never touch a real account (a socket guard fails
any test that tries). With `DATA_MODE=live`, Sentinel launches small **read-only MCP servers**
(`servers/live/`) that expose exactly the fixture read tools, so ingestion and the agent run
unchanged. Enable sources one at a time with `LIVE_SOURCES` (Notion, then Jira, then Gmail).

| Source | Env vars | Calls (all others are blocked by an allowlist guard) |
|---|---|---|
| Notion | `NOTION_TOKEN` (internal integration; share only pages it may read) | `POST /v1/search`, `GET /v1/pages/{id}`, `GET /v1/blocks/{id}/children`, `Notion-Version: 2026-03-11` |
| Jira Cloud | `JIRA_BASE_URL` (https), `JIRA_EMAIL`, `JIRA_API_TOKEN`, optional `JIRA_PROJECT_KEYS` | `GET /rest/api/3/search/jql`, `GET /rest/api/3/issue/{key}` |
| Gmail | `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN` | token refresh, `GET users/me/threads`, `GET users/me/threads/{id}`; scope `gmail.readonly` only, broader grants are refused |

```bash
uv run sentinel live check                   # no network: what is configured / missing
uv run sentinel live gmail-auth              # one-time Google consent (gmail.readonly), prints GMAIL_REFRESH_TOKEN
DATA_MODE=live uv run sentinel live smoke -s notion --yes   # 1 search (limit 1) + 1 fetch; prints counts/ids only
DATA_MODE=live uv run sentinel ingest        # capped at LIVE_MAX_RECORDS per source; a capped run deletes nothing
```

Each live server process receives only its own source's credentials. Memory auto-approve is
refused in live mode. No live call has been run by the project yet.

## Asking questions

```bash
make ask Q="Who owns the Q3 auth migration plan v2?"   # or: uv run sentinel ask "..." [--json]
```

The step (e) graph is `retrieve -> answer -> citation_check`:

- **Retrieve.** Hybrid search, then a lexical relevance gate. If nothing relevant is found,
  the answer says "I don't have enough evidence", lists what was searched, and the LLM is not
  called.
- **Answer.** The prompt numbers each evidence block, and evidence is escaped so it cannot
  forge or close those blocks. The rules require a `[n]` on every factual sentence, treating
  evidence as data rather than instructions, and saying so when evidence is insufficient.
- **Citation check.** `[n]` markers that match no retrieved record are removed and reported;
  invented citations never reach the user. Uncited sentences, and cited sentences the evidence
  doesn't lexically support, are reported in `citation_report`.
- **Every answer returns a `trace_id`**, plus latency and token usage. Token usage comes from
  the provider; the mock reports 0.

**LLM.** `LLM_PROVIDER=mock` (default) is offline and deterministic: unscripted, it answers by
quoting the best-matching evidence sentences with citations. For a real model, install the
extra and set the model and key yourself (no model name is assumed):

```bash
uv sync --extra openai        # or anthropic / google
# .env: LLM_PROVIDER=openai  LLM_MODEL=<model>  LLM_API_KEY=<key>
```

**Tracing.** OpenTelemetry spans (`sentinel.ask`, `node.*`, `llm.generate` as a Langfuse
generation with `gen_ai.*` token attributes) are exported to Langfuse over OTLP/HTTP when
`TRACING_BACKEND=langfuse` (`uv run sentinel infra app-env` prints the lines). PII is redacted
when attributes are set and again by the exporter. Secrets are never put in spans. With
`TRACING_BACKEND=none`, spans are created but not exported, so `trace_id` is still real.

**Tests are offline.** A test fixture blocks every non-loopback connection. Opt-in live check:
`SENTINEL_TEST_LANGFUSE_INFRA_ENV=<abs path to .env.infra> uv run pytest tests/integration/test_langfuse_live.py`.

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

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
  - Every write goes through the permission gate (step l). By default it stays
    `pending_approval` until you run `sentinel approve <id> --session <session>`.
  - With fixture data only, `--auto-approve-memory` or `MEMORY_AUTO_APPROVE=true` approves
    the write on your behalf. It still goes through the gate and is audited. That setting is
    refused together with `DATA_MODE=live`.
  - `sentinel memory put` is you writing directly, stored unverified with source `user`. The
    command itself counts as the approval, and the write is audited too.
- **Memory MCP server.** `search_memory` and `get_memory` are read-only; `put_memory` and
  `mark_stale` are annotated as writes. Facts written over MCP are always stored unverified
  with source `mcp`. The server has no authentication or approval checks of its own. It is
  for local, single-user use over stdio only (see the step (l) trust boundary).

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

## Guards: prompt injection and PII (step j)

`sentinel.guards.injection` scores text from users (`origin="user"`) or retrieved records
(`origin="retrieved"`) and returns a 0..1 score, `flagged` (at `INJECTION_THRESHOLD`, default
0.5), and every matching span, ready for step (k) to wrap or strip.

- **Heuristic rules** (weighted, with spans) catch:
  - instruction overrides, fresh-start / "new instructions", fake `SYSTEM:` lines and chat
    template tokens, persona switches;
  - tool coercion and tool-call JSON, secrecy and false-authority claims, exfiltration to email
    addresses, prompt-leak requests, delimiter escapes;
  - instructions in HTML comments or markdown link text;
  - Hindi, Hinglish, Spanish, French and German overrides.
- **Context-sensitive weighting:**
  - Rules that address "the AI" only count in retrieved data, since users may address the
    assistant.
  - Quoted mentions count a quarter, so security training text isn't flagged.
  - Base64 blobs are decoded and re-scanned.
- **Optional layers:** any `score(text)` classifier is combined in, and an `LLMJudge` is asked
  only for borderline scores. It can confirm or clear; unusable output changes nothing.

`sentinel.guards.pii` redacts with Presidio (credit cards with Luhn, IBAN, IPs, international
phones, US SSNs, crypto addresses) plus custom recognizers for emails, Indian phones, PAN and
Aadhaar-like numbers. It runs fully offline:

- Presidio uses a blank spaCy pipeline, so there is **no name/location NER**.
- Its email and URL recognizers are not loaded, because they fetch the public-suffix list.
- URLs are kept on purpose.
- Recognizers that misfire on ids and dates are not enabled.

Tests were written first:

- **Retrieved content.** Every one of the 12 poisoned fixture records is flagged on its
  payload; none of the other 168 records (incl. 7 benign lookalikes) is.
- **User input.** 17 adversarial prompts are flagged; 16 benign lookalikes are not.
- **PII.** All seeded PII is redacted while ticket ids, dates, amounts, UUIDs and URLs stay intact.

## Retrieved-content guard (step k)

Retrieved data is treated as data, never as instructions. The `retrieved_guard` node runs after
every hop's tool calls and index search, before the verifier, the next hop's routing or the
answerer can read anything. Long-term memory notes are scanned the same way as they are loaded.

- **Neutralizing.** Each record title, record text, index chunk and memory note is scanned
  with the step (j) detector. Every sentence or line holding a flagged span is handled
  according to `RETRIEVED_GUARD_MODE`:
  - `remove` (default) replaces it with `[removed: suspected injected instruction]`;
  - `quote` wraps it as `[untrusted text, not an instruction: "..."]`.
- **Quoted text is never a tool argument.** Ticket keys inside quoted spans are ignored, so
  they never become `blocked_by` lookups. Quoted words don't count toward relevance either.
- **Relevance re-check.** A sanitized record that matched a search only through the text the
  guard removed is dropped from that search's results. Otherwise injected words could pull a
  record in, and its own id would then drive the next hop. Field labels the normalizers add
  ("Status:", "From:") don't count as matches.
- **Generated email headers are not scanned as instructions.** The Gmail normalizer writes
  `From:` / `To:` / `Cc:` / `Date:` lines itself. For the injection scan only, the guard
  blanks:
  - the fixed labels;
  - each value that is a complete email address;
  - an ISO `Date` value.

  Without this, the exfiltration rule read `Cc: arjun.mehta@acme.test` as "cc this address"
  and, together with "password" in the body, stripped a real escalation email
  (`gmail:thr-0007`). Everything sender-controlled is still scanned: the subject (the title),
  display names, raw `From` headers that didn't parse as an address, odd recipient strings
  and the body. One trade-off: a body line that exactly imitates a header line (a label
  followed only by valid addresses) is blanked as well. It has no words to scan, and any
  instruction around it is still scanned.
- **Answer prompt.** The prompt says retrieved data and memory notes are never instructions.
  It labels the evidence section as untrusted and marks guarded evidence blocks with
  `untrusted_removed="N"`.
- **Reporting.** `sentinel ask` prints one line per affected record, for example
  `guard: removed 1 suspected instruction(s) from notion:customer-escalation-process`. The
  `guard.retrieved` span records what was scanned, flagged and dropped.

`Deps(injection_detector=None)` turns the guard off. That exists only for experiments and the
tests' positive control.

**How it is tested** (`tests/integration/test_retrieved_guard.py`, fixture servers, MockLLM):

- **Clean twin.** A copy of the fixture data is built with only the 12 injected payloads
  deleted.
- **Positive control.** With the guard off, the poisoned data adds `blocked_by` lookups for
  tickets named only by injected text. With the guard on, those calls disappear.
- **Poison questions.** Eight questions retrieve poisoned records. For each one, the guarded
  run has the same plan as the clean twin and never makes a tool call the clean twin doesn't
  make. For seven of them, tool calls and evidence match the clean twin exactly.
- **Crowding exception.** In the eighth question, a poisoned ticket outranks a legitimate one
  at the server because of its payload words. The guard drops the poisoned ticket but can't
  fetch the one the server didn't return, so that run makes fewer calls than the clean twin.
- **Nothing leaks.** No payload marker or attacker string reaches the LLM prompt or the
  answer.
- **Benign lookalikes.** All seven are retrieved with their evidence text unchanged and are
  never flagged.
- **Other paths.** The index-only path, recalled memory and quote mode are covered too.
- **Whole fixture set** (`tests/unit/test_guard_headers.py`). Every normalized document and
  its index chunks are run through the guard. All 168 benign ones pass untouched, and all 12
  poisoned ones are still caught. A positive control shows the unmasked `thr-0007` text is
  flagged. Injections in a subject, a display name, an unparsed recipient and the body are
  still detected.

**Known limitation: search ranking.** The guard runs on what a search returns, after the
server has already ranked and cut the results. A poisoned record whose injected words match
the query can push a legitimate record out of the top results. The guard can then remove
the poisoned record, but it cannot recover the legitimate one the server never returned. The
answer can therefore miss a relevant record, though no injected instruction gets through.
The test suite keeps this visible:

- The general assertions stay strict for every poisoned question: the same plan as the clean
  twin, no tool call the clean twin doesn't make, and no new evidence apart from the poisoned
  records.
- One regression case (the PROJ-155 question, where PROJ-163 outranks PROJ-168) is exempt
  only from the exact-match check, with a comment explaining why.

Not covered yet:

- PII redaction of retrieved text is step (m).
- The approval gate for write tools is step (l). The fixture agent has no write tools, so
  "no extra tools" here means no extra read calls.

## Input guard

`input_guard` is the first node of every turn, follow-ups included. It scans the user's
question with the step (j) detector (`origin="user"`, `INJECTION_THRESHOLD`).

**When a question is flagged:**

- The turn ends right there. Long-term memory isn't read, nothing is planned or searched, no
  MCP tool and no model is called, and nothing is saved.
- The reply starts with `I can't run this request`, names the matched rules and says nothing
  was run. The CLI also prints `input guard: blocked (score ...; rules ...)`, and the trace
  records `guard.input` and `sentinel.input_blocked`.
- The question is **not** added to the thread's short-term memory. It never reaches a later
  prompt as an "earlier question", the turn counter does not advance, and the next follow-up
  resolves against the last accepted turn.

**What it is and isn't:**

- **A safety signal, not a permission system.** It stops obvious attempts early and cheaply.
  It does not make an accepted question trusted: a question that passes gets no extra
  powers.
- **Not a guarantee.** The detector is heuristic and will miss attacks it has no rule for
  (paraphrases, new languages, attacks spread over several turns). Writes will still need
  approval from the step (l) permission gate whatever this guard decides, and retrieved data
  is still guarded separately.
- **False positives refuse a legitimate question.** The user can rephrase it without
  instructions addressed to the assistant.
- `Deps(input_detector=None)` disables it (experiments and the tests' positive control only).

**How it is tested** (`tests/integration/test_input_guard.py`, fixture servers, MockLLM):

- **Blocked questions.** All 17 adversarial user prompts are blocked with zero tool calls, no
  plan, no LLM call (even with the LLM planner configured), no memory read and no memory
  write, even with auto-approve on.
- **Positive control.** With the guard off, the same attack is planned and searched.
- **Ordinary questions.** All 16 benign samples pass, including security discussions. The
  3-hop question gives the same answer and tool calls as with the guard off. The two
  security-training questions retrieve and cite the training page.
- **Follow-ups.** In a checkpointed thread, a malicious follow-up is blocked and not
  remembered. The next real follow-up still resolves "them" to the first turn's tickets.

## Permission gate, approvals and audit log (step l)

The agent can **propose** writes; it never runs one on its own.

```bash
uv run sentinel ask "Set PROJ-142 to Done."     # answers, then prints an approval request
uv run sentinel approvals                       # ids, status, tool, expiry (no arguments)
uv run sentinel approve apr-... --session ...   # runs it once (fixture servers only)
uv run sentinel deny apr-... --session ...
uv run sentinel approve apr-... --session ... --confirm "send draft-0001"   # destructive
```

- **Trusted registry.** `sentinel.mcp_clients.registry` classifies every tool Sentinel may call
  as `read`, `write` or `destructive`, with an argument schema for each write. Unknown or
  unclassified tools are denied. Server annotations, tool descriptions, model output and
  retrieved text grant nothing. A test checks that each fixture and memory server's own
  annotations agree with the registry.
- **Policy** (`sentinel.permissions.policy`):
  - Reads are allowed.
  - Writes need an approval. Destructive tools (`gmail.send`) also need a typed
    confirmation.
  - Denied without asking: invalid arguments, recipients or assignees outside
    `ALLOWED_EMAIL_DOMAINS`, and any write to a live source (the live connectors are
    read-only).
- **Where writes come from.** `permission_gate` reads only the user's question. It recognises
  a few explicit requests ("set PROJ-142 to Done", "add a comment to PROJ-142: ...", "draft an
  email to x@acme.test about ...", "send draft-0001"). `save_memory` proposes memory writes.
  Both open an approval request and stop there.
- **Approvals** (`sentinel.permissions.approvals`, SQLite at `APPROVAL_DB`):
  - Each approval is bound to the session (the thread id, or a new id per `ask`), the OS
    user, the tool and a sha256 of the exact arguments.
  - It expires after `APPROVAL_TTL_S` (default 900 s) and can be used once.
  - Denied, expired, used, changed-argument, cross-session and cross-user attempts are
    refused.
  - Arguments are kept only while a request is open, because they must be replayed exactly,
    and are dropped once it closes (see "Argument cleanup" below).
- **Fixture writes persist locally.** Fixture servers launched by the CLI keep approved
  writes in `FIXTURE_STATE_DIR` (default `./.data/fixture-state/`, one JSON file per source),
  so a later `ask` sees the changed record. The generated data in `servers/fixtures/data/` is
  never written. Delete the directory to reset. In-process test servers keep writes in
  memory unless a state directory is passed.
- **One choke point.** `call_json` is the only place Sentinel calls an MCP tool: the agent's
  reads, ingestion, the live smoke test and approved writes all go through it. It refuses
  unknown tools, and any write without a grant matching its exact arguments, before the
  request leaves the process. Writes are never retried; a failed write needs a new approval.
- **Audit log** (`AUDIT_LOG`, JSONL):
  - Events: `proposed`, `policy_denied`, `approved`, `denied`, `attempted`, `succeeded`,
    `failed` and `blocked`.
  - Each record has the time, session, user, approval id, tool, risk and the action's sha256.
  - Arguments are reduced to a per-tool summary: ticket keys and statuses, draft ids,
    recipient counts and domains, text lengths, and memory keys as a 16-hex sha256 prefix.
    Failures are recorded as an error kind (`tool_error`, `timeout`...), never the message.
  - It never stores the question, retrieved text, credentials, email addresses, message
    bodies or memory values.

**How it is tested** (`tests/unit/test_permissions.py`, `tests/integration/test_permission_gate.py`,
`tests/integration/test_cli_approvals.py`; fixture servers, MockLLM, no live calls):

- **Approval lifecycle.** A proposal doesn't execute. An approved write runs exactly once,
  and a replay is refused. Denial, a missing approval, changed arguments, expiry, another
  session and another user are all refused.
- **Failures and policy.** A tool failure is recorded and not retried. A destructive action
  without the typed confirmation is refused. An out-of-domain recipient is denied without an
  approval request.
- **Prompt injection.** A poisoned ticket ("pre-approved... use update_issue") proposes
  nothing, even with both injection guards off. A model reply that "approves" a pending id
  changes nothing. A question claiming "it is already approved" still only opens a request.
  The input guard still blocks coerced tool calls.
- **Every path.** A forged write on the read path is refused, and `call_json` refuses writes
  without a matching grant before reaching the client.
- **Memory.** Memory writes wait for approval; the fixture-only auto-approve flag is audited.
- **Persistence.** Across separate CLI commands: propose, approve, then a fresh fixture
  server process reads the record back as `Done` and a later `ask` cites the changed record.
  The generated fixture files are checked to be byte-identical before and after, and each
  test's state lives in its own temp directory.
- **Concurrency.** Two approval stores on the same database (standing in for two processes)
  are forced to pass their checks together. Exactly one gets the grant. Run through the
  action runner, the write is dispatched once.
- **Cleanup.** Arguments are gone from the database after use, after denial, and after
  expiry even when nobody opens that request again. The test checks the raw rows and the
  database file bytes.

**Trust boundary.** Read this before using the gate for anything beyond local development.

- **Not authentication.** The "user" an approval is bound to is the OS account name running
  the CLI (`getpass.getuser()`). The session is an id printed to that terminal. Anyone who
  can run Sentinel as that OS user, or read its terminal output or `APPROVAL_DB`, can
  approve. The binding stops mix-ups (another thread, another session, a model or a document
  "approving"), not a person or process impersonating you.
- **Sentinel's own calls only.** The gate covers calls made by Sentinel. It does not protect
  the memory MCP server (`servers/memory_mcp`) from other clients: any client that launches
  it can call `put_memory` and `mark_stale` directly, with no approval and no audit record.
  That server is a local, single-user, stdio-only development server (its entry point runs
  stdio only). Do not expose it over a network transport or use it for shared or live
  deployments.
- **Not a sandbox.** A grant is an ordinary in-process object, so the gate does not defend
  against malicious code running inside Sentinel.

**Concurrency and crashes: at most once, not exactly once.**

- **Consumption is atomic.** It is a conditional `UPDATE ... WHERE status = 'approved'`, so
  concurrent attempts, even from separate processes, give at most one grant.
- **The approval is used up before the write is sent.** A write therefore runs at most once
  per approval, and is never retried automatically.
- **A crash can leave the outcome unknown.** If the process dies after the call is
  dispatched but before the outcome is recorded, the audit log shows `attempted` with no
  `succeeded` or `failed`. The write may or may not have happened. Sentinel does not claim
  exactly-once execution across crashes and does not retry such writes. Check the target
  record yourself, then request a new approval if needed.
- **The fixture state file is last-writer-wins.** It is rewritten whole by one server
  process. Concurrent server processes writing the same source are not coordinated.

**Argument cleanup and retention.**

- **When arguments are cleared.** Full pending arguments (a draft body, a memory value) are
  stored only in `APPROVAL_DB`, only while the request is open. They are cleared in the same
  update that marks a request consumed or denied.
- **Overdue requests.** These are expired, and their arguments cleared, whenever any approval
  store is opened or a new request is made: any `ask`, `chat`, `approve`, `deny` or
  `approvals` command does it, whether or not anyone looks at that request again.
- **There is no background timer.** If Sentinel never runs again, an expired request's
  arguments stay in the file until the next command.
- **This is not secure erasure.** `PRAGMA secure_delete` makes SQLite zero cleared content in
  its own file. The filesystem, SSD wear levelling, backups, snapshots or copies of the file
  can still hold old bytes.
- **What is never cleaned up.** Closed rows themselves (ids, status, tool, session, user and
  timestamps) are kept with no retention limit. So are the audit log and the fixture state
  files.

**Other limitations:**

- **Audit log integrity.**
  - Append-only by convention only. It is a local file, not tamper-evident (no hash chain or
    signature), and anyone who can edit the file can change it. It has no rotation or
    retention.
  - Records are written by the same process as the action, so a crash between the tool call
    and the `succeeded` line leaves `attempted` with no outcome. A failure to write the log
    is not rolled back with the action.
  - It covers writes and approval decisions only; reads are in traces, not in the audit log.
- **What the log still contains.** Identifiers (session ids, the OS user name, ticket keys,
  draft ids, recipient domains) are stored as-is. PII redaction of traces is step (m).
- **Deviation from the brief: no LangGraph `interrupt()`.** The brief asked for the graph to
  pause at the permission gate and resume on approve or deny. For this CLI milestone the
  turn finishes instead, and approval happens in a separate `sentinel approve` or `deny`
  command against the approval store. That makes expiry, single use, session binding and
  replay protection explicit and testable across processes. The trade-off: the agent does
  not continue its answer after the write runs, and an API or UI flow that pauses and
  resumes a thread is not built yet.

## PII redaction of outputs, traces, memory and logs (step m)

Sentinel runs the step (j) detector (`sentinel.guards.pii`) on everything that can leave the
agent, and replaces each match with a placeholder such as `<EMAIL>` or `<PHONE_IN>`.
Ticket, page and thread ids, citation numbers and URLs are kept, so citations still resolve.

**Where it runs:**

- **The question.** The redacted question is what goes into the agent state, planning,
  prompts, traces and both memories. The raw question exists only for the current turn: the
  input guard scans it, and write proposals read it, because a draft needs the real recipient.
- **Retrieved data.** Tool results, index chunks and recalled memory notes are redacted as
  they enter the agent state, before the injection guard, the prompt or any checkpoint sees
  them. The injection rules treat `<EMAIL>` as an address, so redaction doesn't weaken them.
- **The model's reply and the answer.** The reply is redacted before it is stored. An
  `output_guard` node then checks the final answer and its citations' titles and snippets
  again before they are shown. The CLI prints `output guard: removed PII from the answer (...)`
  when that second pass finds something.
- **Traces.** `set_text` redacts at write time. `RedactingSpanExporter` redacts again, inside
  the process before export: every string attribute, event (including recorded exceptions'
  messages and stack traces) and span status description. A blocked question is recorded in
  its redacted form.
- **Short-term memory.** Checkpoints store the agent state, which only ever held redacted
  text.
- **Long-term memory.** The SQLite backend redacts the value and stale reason on every
  write: the agent's, `sentinel memory put`'s and the memory MCP server's. So
  `sentinel memory put k "call +91 ..."` stores `call <PHONE_IN>`.
- **Audit log.** It holds per-tool summaries (counts, domains, ids, hashes) as before, and
  now also passes their strings and the reason through the redactor.
- **Error text.** Tool failure messages, ingestion and smoke-test errors, and CLI error
  messages are redacted. Crash output never prints local variables
  (`pretty_exceptions_show_locals=False`).
- **CLI output.** Answers, sources, `--json`, `sentinel search` snippets and
  `sentinel memory list` are redacted.

**Choice:** the brief doesn't say what memories should keep, so both short- and long-term
memory store the redacted version, consistent with "PII redaction on output, logs and
traces".

**What is detected:**

| Kind | How |
|---|---|
| Email addresses | regex (Presidio's email recognizer fetches the public-suffix list, so it is not used) |
| Indian phone numbers, PAN, Aadhaar-like numbers | custom regexes with digit-group boundaries |
| International phone numbers | Presidio `PhoneRecognizer` (libphonenumber) |
| Credit cards (Luhn-checked), IBAN, IP addresses, US SSN, crypto addresses | Presidio recognizers with checksums or validation |

**What is not detected:**

- **Names, organisations and locations.** spaCy runs as a blank pipeline, with no model
  download and no NER, so "Priya Sharma" or "Pune" stay as written.
- **Postal addresses, dates of birth, passport, licence and bank account numbers.** No
  recognizer is enabled for them; several fire on ticket ids and dates.
- **URLs.** They are kept on purpose for citations, so PII inside a URL is not redacted.
- **Secrets in free text.** API keys or passwords that someone types or that appear in a
  record are not detected. There is no secret scanner. Sentinel's own configured secrets
  never enter a prompt, trace or log.
- **Unusual formats.** Numbers written in an unusual format, or split across lines, can be
  missed.

**What stays raw:**

- **Sources and caches.** The source systems, the local Qdrant index (a cache of source
  text) and the fixture state files (simulated mailboxes and tickets) keep raw text.
- **Open approval requests.** These keep the exact arguments they will execute (a draft's
  recipient and body) until they are used, denied or expire (step l).

**Trade-offs:**

- **Lost detail in answers.** Answers lose colleagues' addresses and phone numbers, even
  where the user may be allowed to see them (the brief's benign lookalikes include a
  signature phone number). The record is still cited, so the user can open it.
- **Some searches can't run.** Questions that search by an email address or phone number
  can't do so, because the planner only sees the redacted question.

This is **not complete PII removal.** The tests check every PII kind above and every value
seeded in the fixtures, not arbitrary real-world data.

**How it is tested:**

- **Every fixture document and every PII kind through every path**
  (`tests/unit/test_redaction_paths.py`). Each of the 180 fixture documents, and one sample
  of each detected kind, goes through:
  - the redacted record and its chunks;
  - evidence and citation rendering;
  - the prompt;
  - the answer after the output guard;
  - trace export, with attributes, events, exceptions and status set raw on purpose;
  - long-term memory;
  - the audit log.

  No raw value may come out.
- **End to end with tracing on** (`tests/integration/test_redaction_end_to_end.py`, in-memory
  exporter, no Langfuse). PII-heavy questions, a question containing PII, a blocked
  question and a draft request are run. Then answers, results, every exported span, the
  checkpoint database, the memory database, the audit log and the approval database are
  scanned for every sample, every seeded value and every fixture email address.
- **Answers stay useful.** The 3-hop answer keeps citations from two or more sources and
  its ticket evidence, every citation keeps its URL, and the owner's name stays visible.
- **CLI** (`tests/integration/test_cli_redaction.py`). `ask` and `--json`, `search`,
  `memory list` and the `.data` files contain no PII or configured secret.
  `memory put` stores the redacted value.

## Tool failures (step n)

Sources fail: they time out, rate-limit, return 5xx or broken data, crash, or never start.
Sentinel keeps answering from whatever still works, says what failed, and never guesses.

```bash
SENTINEL_FIXTURE_FAILURES="gmail=timeout" uv run sentinel ask "Which tickets are blocked by the auth migration and who emailed about them?"
SENTINEL_FIXTURE_FAILURES="jira.startup=crash" uv run sentinel ask "..."   # a server that never starts
SENTINEL_FIXTURE_FAILURES="*=error_500" uv run sentinel ask "..."          # nothing readable
```

Fixture failure modes:

- `timeout`, `error_500`, `rate_limit` and `empty`;
- `malformed` (truncated JSON) and `invalid_schema` (valid JSON of the wrong shape);
- `crash` (the server process exits mid-call; subprocess only);
- the rule `<source>.startup=crash` (the server exits before serving).

**Reads:**

- **Every failure gets a kind.** `call_json` turns everything that can go wrong into a
  `ToolCallError` with a kind: `timeout`, `rate_limited`, `server_error`, `unavailable`
  (failed to start, crashed, connection closed), `invalid_response`, `invalid_arguments`,
  `not_found` or `tool_error`. Nothing else escapes it, so one bad source can't crash the
  turn.
- **Only transient errors are retried.** Timeouts, rate limits and 5xx get up to
  `tool_retries` (default 2) more attempts with exponential backoff. Invalid arguments,
  not-found, malformed data, a dead connection and permission refusals fail on the first
  attempt.
- **A failed source is skipped for the rest of the turn** (recorded as `skipped`). That
  covers a source still failing after its retries, and one that is unavailable. No call
  starts after `TOOL_BUDGET_S` (default 60 s) of tool time in the turn (`budget_exhausted`).
  `MAX_HOPS`, the per-call `TOOL_TIMEOUT_S` and the verifier's single widening still apply,
  so a failure can't cause a loop.
- **Malformed search hits are dropped one by one.** Valid hits are still used, and like any
  record they go through PII redaction and the retrieved-content guard.
- **Server error text is never used.** It can echo the request, record content or an
  attacker's instructions. Failures store, prompt, trace and print only a fixed description
  of their kind ("server error (HTTP 5xx)", "timed out after 10s"...). Tool spans carry
  `sentinel.tool.error_kind`. Ingestion reports the kind too.
- **A server that fails to start** (`open_available_clients`) makes only its source
  unavailable. `sentinel ask` and `chat` still answer, and `sentinel ingest` reports `FAIL`
  for that source.

**Answers:**

- **Failed sources are always named.** If any source failed, the answer says
  `I could not read <source> (<reason>), so this may be incomplete`, whatever the model
  wrote.
- **Citations only point to evidence that was returned.** A memory note whose source could
  not be read this turn is treated as unverified, so the citation check removes any sentence
  that states it as fact.
- **When every source failed,** the reply says
  `none of the sources could be read (...)`, with no citations and nothing guessed.
- **Empty results are not failures.** They give the usual "not enough evidence" reply.

**Writes** (approved actions, step l):

- **Never retried.** A write the tool clearly rejected (rate limit, bad arguments,
  not-found) is `failed`.
- **Uncertain outcomes are reported as uncertain.** If a write was sent and then timed out,
  lost its connection, got a 5xx or an unreadable reply, it may have been applied. It is
  reported and audited as `uncertain`, and `sentinel approve` says to check the record
  before requesting a new approval.
- **The approval is kept if nothing was sent.** If the target server can't be started, the
  approval is not consumed.

**How it is tested:**

- **Units** (`tests/unit/test_tool_failures.py`): classification, which errors are retried,
  the source-down rule and the time budget, malformed search and record payloads, uncertain
  versus failed writes, and the answer wording.
- **In-process fixture servers** (`tests/integration/test_tool_failures.py`): one source
  timing out, rate-limited, 5xx, malformed or schema-invalid, with honest partial answers
  and citations only from returned evidence. Also empty results, partial results across
  sources, every source failing, the time budget, a closed connection, a memory note from a
  failed source, hostile error text, and odd records still guarded and redacted.
- **Real stdio subprocesses:** a server crashing mid-call and one failing to start.
- **CLI** (`tests/integration/test_cli_failures.py`): no traceback, a clear message, ingest
  `FAIL`, and an `uncertain` write audited and not retried.

**Limitations:**

- **Kinds come from error text.** A server error is classified by matching its text, so an
  unusual server message falls back to `tool_error`, which is not retried.
- **No reconnecting.** A crashed or disconnected server stays down for the rest of the
  command.
- **A hanging server is not bounded at startup.** One that starts but never finishes the
  MCP handshake is bounded only by the MCP SDK's own behaviour, not by `TOOL_BUDGET_S`.
- **The budget is checked between calls.** A call that has started runs up to
  `TOOL_TIMEOUT_S` (per attempt) even if the budget runs out meanwhile.
- **"Uncertain" is deliberately conservative.** A 5xx or timeout is reported as uncertain
  even when the server never applied the write.

## Evaluation metrics and the regression gate (step o)

The metric functions (`evals/metrics.py`) and the regression gate (`evals/gate.py`) exist
and are unit-tested on small hand-made results. Their definitions are in
[docs/eval-design.md](docs/eval-design.md):

- attack block rate and benign false-positive rate, separately for input and retrieved
  attacks;
- citation correctness: exists, was returned, and supported, using an anchor check that
  differs from the agent's own word-overlap check;
- recall@k and answer correctness (deterministic; an LLM judge is optional and off by
  default);
- permission safety and PII leaks per output path;
- latency, tool calls, and tokens (only where reported).

Every rate carries its sample size and a Wilson 95% interval. The gate hard-fails on any
unapproved write, PII leak or hallucinated citation, and compares everything else with an
owner-approved baseline. It never writes that baseline itself, and it fails if the baseline
is missing or was edited.

## Eval datasets and harness (step p)

```bash
make eval                                    # = uv run python -m evals.run; mock LLM, fixtures
uv run python -m evals.run --category adversarial_user --report-only
uv run python -m evals.run --llm configured  # opt-in real model (LLM_PROVIDER/MODEL/API_KEY)
```

- **Datasets.** 124 versioned cases in `evals/datasets/` (see
  [docs/eval-design.md](docs/eval-design.md)): answers and citations, stale vs current
  records, follow-ups, no-answer questions, memory freshness, input and retrieved attacks,
  benign lookalikes, permissions, PII and tool failures.
- **The harness.** It runs every case through the real agent graph on in-process fixture
  servers with the mock LLM. It writes `reports/report.md` and `reports/metrics.json` with
  sample sizes and 95% intervals, separately for `all`, `dev` and `heldout`, and runs the
  gate. It never writes a baseline, and it exits 1 while none is approved.
- **Which split numbers come from:**
  - `dev` may reuse the fixture payloads and the detector's samples or rule wording.
  - `heldout` (attack and benign cases) was written independently of both, and the guards
    weren't tuned on it.
  - Any number presented as how well the guards generalise must be the `heldout` column, with
    its `n`.
- **Mock runs are not model results.** The mock LLM copies evidence sentences and never
  follows instructions, so a mock run's attack block rate says little about resistance to
  manipulation. Its answer accuracy reflects sentence picking, not a model. The guard
  diagnostics ("input attacks refused", "retrieved attacks flagged") are what a mock run does
  measure.
- **No published numbers yet.** README results will come only from a recorded real-model run
  (step s): `TBD (run make eval)`.

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

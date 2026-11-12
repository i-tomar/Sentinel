# Sentinel threat model

This document lists what Sentinel protects, who might attack it, where untrusted data comes
in, what each defense does, and what is **not** defended. Component details are in
[architecture.md](architecture.md); how the defenses are measured is in
[eval-design.md](eval-design.md).

Nothing here is a guarantee. Prompt injection has no known complete fix: OWASP notes it is
"unclear if there are fool-proof methods of prevention"
([LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)).
Sentinel's approach is to detect what it can, and to limit what a successful injection can
do even when detection fails.

## Scope and assumptions

- **A local, single-user developer tool.** Sentinel runs as a CLI on one machine, for one
  person. There is no network service: no API server and no multi-user deployment exist.
- **Fixture mode by default.** CI and the tests use fake Notion, Jira and Gmail servers with a
  seeded "Acme" dataset, the mock LLM and no credentials. Live connectors are opt-in and
  read-only, and no live call has been run by the project.
- **The model is untrusted.** Its output (plans, answers, anything that looks like an
  approval) is treated as data. It cannot call a write tool or approve anything.
- **The operating-system account is trusted.** Anyone who can run commands as the user, or
  read the user's files, is treated as the user.

## Assets

| Asset | What could go wrong |
|---|---|
| Source system integrity: ticket status, comments, email drafts, sent email | A write the user did not ask for (status change, comment, email) |
| Source data confidentiality: pages, tickets, emails | Content sent to an attacker, for example in an email draft |
| Personal data in that content: email addresses, phone numbers, PAN, Aadhaar-like numbers | Shown in answers, written to traces, logs or memory |
| Answer integrity | A wrong answer, a citation to a record that was never retrieved, a stale record presented as current |
| Long-term memory | A false "fact" stored and repeated in later sessions |
| Credentials: LLM key, Notion, Jira and Gmail tokens, Langfuse keys | Leaked into prompts, traces, logs or the repository |
| Approval decisions and the audit log | A write approved by someone or something other than the user, or no record of it |
| Eval baselines and the CI gate | Lowered quietly so that regressions pass |

## Attackers

1. **Authors of retrieved content (the main threat).** Anyone who can send an email to the
   mailbox, comment on a ticket or edit a Notion page. They can hide instructions in the
   text Sentinel retrieves (indirect injection): HTML comments, page footers, link text, fake
   `SYSTEM:` lines, base64, tool-call-shaped JSON, other languages. Goals: trigger a write,
   send data out, steer tool calls, or change the answer.
2. **Adversarial user input.** A question containing jailbreak text, either written by the
   user or pasted from somewhere else (direct injection). Goals: make the agent ignore its
   rules, leak data or run tools.
3. **The model itself.** A model can hallucinate, invent citations, propose actions nobody
   asked for, or be persuaded by injected text.
4. **A misbehaving MCP server or source API.** It can return malformed data, the wrong schema,
   hostile error messages, misleading tool annotations, or crash.
5. **Other local processes or users on the same machine.** They can read files under
   `.data/`, run Sentinel as the user, or launch the memory MCP server directly.
6. **Someone changing the repository.** A contributor or a careless commit could edit a
   baseline or weaken the gate.

Out of scope: an attacker who can run code inside the Sentinel process, compromised
dependencies (versions are pinned, not audited), and network attackers (there is no network
service).

## Entry points

| Entry point | Trusted? | Where it goes |
|---|---|---|
| The user's question (`sentinel ask` / `chat`) | Partly: it decides what is asked and which writes are requested | `input_guard`, then the redacted question goes to planning, prompts, traces and memory; the raw question is read only by `permission_gate` |
| Tool results from MCP servers (fixture or live) | No | Redacted, then `retrieved_guard`, then the verifier and the answer prompt |
| Index chunks from Qdrant | No (a cache of source text) | Same path as tool results |
| Long-term memory notes | No | Redacted and scanned in `load_memory`; shown as context, never evidence |
| Chat history (checkpoints) | Only Sentinel's own redacted state | Planners see earlier questions and opaque record ids, never record text |
| Model output: planner JSON, answer text | No | Plans are validated; answers go through `citation_check` and `output_guard`; nothing in them can approve or run a write |
| MCP server metadata and errors | No | Ignored for permissions (the trusted registry decides); error text is replaced by a fixed description |
| `sentinel approve` / `deny` and the approval database | The OS user | The only way a write runs |
| Memory MCP server (stdio) | Whoever launches it | Writes straight to long-term memory (see "What is not defended") |
| Configuration (`.env`, environment) | The OS user | Settings and credentials |
| Eval datasets, baselines and workflows | Reviewed through pull requests | The CI gate |

## Defenses

Each defense is described with what it stops and where it stops working.

1. **Input guard** (`input_guard`). A heuristic detector scores the question. A flagged
   question ends the turn before memory, planning, tools or the model run, and is not
   remembered in the thread. *Limit:* it only catches patterns it has rules for. On the
   held-out attacks it refused 1 of 13 (see "What is not defended").
2. **The plan comes only from the user's question.** The planner never sees retrieved text,
   and memory reaches it only as opaque record ids. Dependent hops use only strictly-matched
   ticket keys from earlier, non-stale results. *Stops:* retrieved text adding tool calls or
   changing the plan. *Limit:* retrieved text still reaches the answer prompt (after the
   guard), so it can still influence the wording of the answer.
3. **Retrieved-content guard** (`retrieved_guard`). Every tool result, index chunk and memory
   note is scanned. Flagged sentences are removed (or quoted as untrusted text) before the
   verifier, routing or the answer prompt read them. Ticket keys inside quoted text are never
   used as tool arguments, and a record that matched a search only through removed text is
   dropped. The answer prompt marks evidence as untrusted data and escapes it so it cannot
   close its own block. *Limit:* heuristic, like the input guard; on held-out retrieved attacks
   it flagged 4 of 8.
4. **Trusted tool registry and one choke point.** Every tool is classified `read`, `write` or
   `destructive` in Sentinel's own registry. Server annotations and descriptions grant
   nothing. `call_json` is the only place Sentinel calls an MCP tool, and it refuses unknown
   tools, and any write without a grant for its exact arguments, before the request leaves
   the process. The router produces read calls only.
5. **Permission gate and policy.** Writes are proposed only from explicit requests in the
   user's raw question, and from memory saves. Retrieved text and model output cannot
   propose one. Policy denies invalid arguments, recipients or assignees outside
   `ALLOWED_EMAIL_DOMAINS`, and any write in live mode. Destructive tools (`gmail.send`)
   also need a typed confirmation.
6. **Approvals bound to the request.** An approval is bound to the session, the OS user, the
   tool and a sha256 of the exact arguments. It expires (`APPROVAL_TTL_S`, default 900 s),
   can be used once, and is consumed atomically before the write is sent. Writes are never
   retried. An interrupted write is reported as `uncertain`, not as success. Every
   proposal, decision and outcome goes to the audit log, with argument summaries only.
7. **Citation check and memory rules.** Citations to records that were not retrieved this turn
   are removed. A sentence that restates an unverified memory note without support is
   removed. Memory notes are never cited. *Limit:* support is checked lexically, so a cited
   sentence can reuse the record's words and still misstate it. The eval suite uses a
   stricter anchor check to measure this.
8. **PII redaction at every exit.** The question, tool results, chunks, memory notes and the
   model's reply are redacted as they enter the agent state. `output_guard` redacts the final
   answer and citations again. Traces are redacted when attributes are set and again in the
   exporter, including events, exceptions and status text. Long-term memory and the audit
   log redact on write. *Limit:* only the kinds listed below are detected.
9. **Tool-failure handling.** Failures are classified, only transient errors are retried, a
   failed source is skipped for the rest of the turn, and a time budget bounds the turn.
   Server error text is never stored, prompted, traced or shown, because it can echo
   attacker text. The answer always names the sources it could not read.
10. **Credentials stay out of data paths.** Settings come from the environment through
    `pydantic-settings`. Secrets are never put in prompts, spans or logs. Each live server
    process gets only its own source's credentials, and Gmail accepts only the
    `gmail.readonly` scope. CI has no secrets at all.
11. **Checkpoint deserialization is allowlisted.** Only Sentinel's own state models can be
    revived from the checkpoint database.
12. **The CI gate.** The tests and evals run with no credentials and with public network
    connections refused. The gate fails on any unapproved write, PII leak or hallucinated
    citation, and on regressions against an owner-approved baseline. A baseline whose sha256
    no longer matches its metrics is rejected.

## What is not defended

- **Injections the detector has no rule for.** Both guards are heuristic. In the mock eval
  run at commit `4901a37` (dataset 1.0.0), on cases written independently of the detector's
  rules, the input guard refused **1 of 13** held-out input attacks (7.7%, 95% CI 1%–33%) and
  the retrieved guard flagged **4 of 8** held-out retrieved attacks (50%, 95% CI 22%–78%).
  On the dev split, built from the same payloads and wording the rules were written
  against, both were 100%. Paraphrases, other languages, encodings and multi-turn attacks
  get through. What limits the damage is the plan coming only from the question, the
  registry and the approval gate, not detection.
- **A missed injection can still change the answer text.** With a real model, injected text
  that survives the guard reaches the answer prompt and can make the answer wrong or
  misleading, even though it cannot add tool calls or writes. Real-model resistance has not
  been measured yet. The mock LLM never follows instructions, so mock block rates say little
  about it.
- **Search-ranking crowding.** A poisoned record whose injected words match the query can push
  a legitimate record out of the server's top results. The guard drops the poisoned record
  but cannot recover the one the server never returned.
- **The human approver.** The approval prompt shows the tool and an argument summary
  (counts, domains, ids). A user who approves without checking is not protected.
- **Authentication.** Approvals are bound to the OS account name and a session id printed
  in the terminal. Anyone who can run Sentinel as that user, or read its terminal output or
  `APPROVAL_DB`, can approve. This prevents mix-ups (another thread, another session, a
  model or a document "approving"), not impersonation. It is not a sandbox either: a grant
  is an ordinary in-process object.
- **The memory MCP server.** It has no authentication and no approval checks. Any client that
  launches it can call `put_memory` and `mark_stale` directly, with no audit record. It is
  for local, single-user use over stdio only. Values are still redacted on write, and facts
  written through it are stored unverified.
- **PII the detector does not recognise.** Names, organisations and locations (there is no
  NER model), postal addresses, dates of birth, passport, licence and bank account numbers,
  PII inside URLs (kept for citations), secrets typed into text, and numbers in unusual
  formats or split across lines. Redaction is not complete PII removal.
- **Raw text that stays on disk.** The source systems, the local Qdrant index, the fixture
  state files and open approval requests (until used, denied or expired) keep raw text.
  Cleared approval arguments are zeroed inside SQLite (`PRAGMA secure_delete`), but the
  filesystem, backups or copies can still hold old bytes.
- **Data sent to the model provider.** With a real model, redacted evidence and the redacted
  question are sent to that provider. Anything the redactor misses, names included, goes
  with them.
- **Audit log integrity.** It is a local append-only file by convention, with no hash chain
  or signature. Anyone who can edit the file can change it. A crash between a write and its
  outcome leaves `attempted` with no result.
- **Exactly-once writes.** Writes run at most once per approval. After a crash, a write may
  or may not have happened; Sentinel reports it as uncertain and does not retry.
- **Hanging servers at startup.** A server that starts but never finishes the MCP handshake
  is bounded only by the MCP SDK, not by `TOOL_BUDGET_S`.
- **Malicious code in the process or in dependencies.** Not defended.
- **The baseline hash is not a security control.** It catches accidental edits. Someone who
  rewrites the metrics and recomputes the hash defeats it. Baseline changes are meant to
  land only through a pull request the owner reviews.

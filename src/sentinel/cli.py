"""Command-line entry point (`sentinel`). Commands are added as each build step lands."""

from __future__ import annotations

import json
import re
import sys
import uuid
import webbrowser
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import anyio
import typer
from opentelemetry.sdk.trace import TracerProvider

from sentinel import __version__, infra
from sentinel.citations import CitationResolver
from sentinel.config import Settings, get_settings
from sentinel.graph.build import AskResult, Deps, ask_async, thread_turns
from sentinel.graph.checkpoint import open_checkpointer
from sentinel.graph.nodes.execute_tools import MCPToolExecutor
from sentinel.graph.nodes.planner import make_planner
from sentinel.guards.injection import InjectionDetector
from sentinel.live.gmail import API as GMAIL_API
from sentinel.live.gmail import READONLY_SCOPE
from sentinel.live.gmail import TOKEN_URL as GMAIL_TOKEN_URL
from sentinel.live.gmail_consent import obtain_refresh_token
from sentinel.live.notion import API as NOTION_API
from sentinel.live.notion import NOTION_VERSION
from sentinel.live.smoke import SmokeReport, describe, smoke
from sentinel.llm import make_chat_model
from sentinel.mcp_clients.pool import active_sources, open_clients
from sentinel.memory.backend import MemoryBackend
from sentinel.memory.models import MemoryItem
from sentinel.memory.policy import make_memory_backend
from sentinel.retrieval.embed import make_embedder
from sentinel.retrieval.ingest import SourceReport
from sentinel.retrieval.ingest import ingest as ingest_sources
from sentinel.retrieval.models import SOURCES, SearchMode, Source
from sentinel.retrieval.store import ChunkStore, make_qdrant_client
from sentinel.tracing import build_tracer_provider, get_tracer

app = typer.Typer(name="sentinel", no_args_is_help=True, add_completion=False)
infra_app = typer.Typer(no_args_is_help=True, help="Local Docker Compose infrastructure.")
app.add_typer(infra_app, name="infra")

InfraEnvPath = Annotated[Path, typer.Option("--path", help="Compose secrets file.")]


@app.command()
def version() -> None:
    """Print the installed Sentinel version."""
    typer.echo(__version__)


@app.command()
def config() -> None:
    """Print the effective configuration as JSON, with secrets masked."""
    typer.echo(json.dumps(get_settings().model_dump(mode="json"), indent=2))


SourceOption = Annotated[
    list[str] | None,
    typer.Option("--source", "-s", help="Limit to notion, jira or gmail (repeatable)."),
]


def _sources(values: list[str] | None) -> list[Source]:
    chosen = values or list(SOURCES)
    unknown = sorted(set(chosen) - set(SOURCES))
    if unknown:
        raise typer.BadParameter(f"unknown source(s): {', '.join(unknown)}")
    return [s for s in SOURCES if s in chosen]


@contextmanager
def _open_store() -> Iterator[ChunkStore]:
    settings = get_settings()
    client = make_qdrant_client(settings.qdrant_url)
    try:
        yield ChunkStore(client, settings.qdrant_collection, make_embedder(settings))
    finally:
        client.close()  # embedded Qdrant holds a folder lock until closed


@app.command()
def ingest(source: SourceOption = None) -> None:
    """Fetch records over MCP (fixture servers by default) and sync them into Qdrant."""
    settings = get_settings()
    sources = _require_live_ready(settings, _sources(source))
    cap = settings.live_max_records if settings.data_mode == "live" else None

    with _open_store() as store:

        async def run() -> list[SourceReport]:
            async with open_clients(settings, sources) as clients:
                return await ingest_sources(clients, store, max_records=cap)

        reports = anyio.run(run)
        for r in reports:
            if r.error:
                typer.echo(f"FAIL {r.source:<7} {r.error} (existing index left unchanged)")
            else:
                typer.echo(
                    f"ok   {r.source:<7} docs={r.documents} added={r.added} updated={r.updated} "
                    f"unchanged={r.unchanged} deleted={r.deleted} "
                    f"chunks_written={r.chunks_written}"
                    + (" (capped at LIVE_MAX_RECORDS; nothing deleted)" if r.truncated else "")
                )
        typer.echo(f"collection {store.collection}: {store.count()} chunks")
    if any(r.error for r in reports):
        raise typer.Exit(1)


@app.command()
def search(
    query: str,
    k: Annotated[int, typer.Option("-k", min=1, max=50)] = 5,
    source: SourceOption = None,
    mode: Annotated[str, typer.Option(help="hybrid, dense or keyword")] = "hybrid",
) -> None:
    """Debug retrieval: print the top chunks as numbered citations."""
    modes: dict[str, SearchMode] = {"hybrid": "hybrid", "dense": "dense", "keyword": "keyword"}
    if mode not in modes:
        raise typer.BadParameter("mode must be hybrid, dense or keyword")
    with _open_store() as store:
        store.ensure_collection()
        hits = store.search(query, k, sources=_sources(source), mode=modes[mode])
    if not hits:
        typer.echo("no results (did you run `sentinel ingest`?)")
        return
    resolver = CitationResolver()
    for hit in hits:
        citation = resolver.add(hit.chunk)
        typer.echo(f"{hit.score:.4f} [{citation.n}] {hit.chunk.doc_key}#{hit.chunk.index}")
    typer.echo(resolver.render())


@app.command("ask")
def ask_command(
    question: str,
    source: SourceOption = None,
    k: Annotated[int, typer.Option("-k", min=1, max=20, help="Chunks to retrieve.")] = 6,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result as JSON.")] = False,
    tools: Annotated[
        bool, typer.Option(help="Call source MCP tools live (fixture servers by default).")
    ] = True,
    as_of: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Date treated as today, e.g. for 'last week'."),
    ] = None,
    auto_approve_memory: AutoApproveMemory = False,
) -> None:
    """Plan, call tools over several hops, and answer with citations and a trace id."""
    settings = get_settings()
    sources = _sources(source)
    if tools:
        sources = _require_live_ready(settings, sources)
    auto_approve = _memory_auto_approve(settings, auto_approve_memory)
    today = as_of.replace(tzinfo=UTC) if as_of else None
    provider = build_tracer_provider(settings)

    async def run(store: ChunkStore) -> AskResult:
        deps = _agent_deps(settings, store, provider, k, auto_approve)
        async with _tool_access(settings, sources, deps, tools):
            return await ask_async(question, deps, sources, today)

    try:
        with _open_store() as store:
            store.ensure_collection()
            result = anyio.run(run, store)
    finally:
        provider.shutdown()  # flushes spans to the tracing backend, if any
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    _print_result(result, settings)


_THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@app.command()
def chat(
    question: Annotated[
        str | None, typer.Argument(help="Ask one question and exit; omit for a session.")
    ] = None,
    thread: Annotated[
        str | None,
        typer.Option("--thread", "-t", help="Thread id to continue (new one if omitted)."),
    ] = None,
    history: Annotated[bool, typer.Option(help="Show the thread's remembered turns.")] = False,
    source: SourceOption = None,
    k: Annotated[int, typer.Option("-k", min=1, max=20, help="Chunks to retrieve.")] = 6,
    tools: Annotated[
        bool, typer.Option(help="Call source MCP tools live (fixture servers by default).")
    ] = True,
    as_of: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Date treated as today, e.g. for 'last week'."),
    ] = None,
    auto_approve_memory: AutoApproveMemory = False,
) -> None:
    """Multi-turn conversation. Turns of a thread are checkpointed to CHECKPOINT_DB, so a
    later `sentinel chat --thread ID` (even after a restart) continues with its memory."""
    settings = get_settings()
    sources = _sources(source)
    if thread is not None and not _THREAD_ID.match(thread):
        raise typer.BadParameter(
            "thread id: 1-64 letters, digits, '-' or '_'", param_hint="--thread"
        )
    if history and thread is None:
        raise typer.BadParameter("--history needs --thread", param_hint="--history")
    if tools:
        sources = _require_live_ready(settings, sources)
    auto_approve = _memory_auto_approve(settings, auto_approve_memory)
    today = as_of.replace(tzinfo=UTC) if as_of else None
    thread_id = thread or uuid.uuid4().hex
    provider = build_tracer_provider(settings)

    async def session(store: ChunkStore) -> None:
        async with open_checkpointer(settings.checkpoint_db) as saver:
            deps = _agent_deps(settings, store, provider, k, auto_approve)
            deps.checkpointer = saver
            if history:
                for i, turn in enumerate(await thread_turns(deps, thread_id), start=1):
                    typer.echo(f"[{i}] Q: {turn.question}\n    A: {turn.answer}")
                return
            typer.echo(f"thread_id={thread_id}")
            async with _tool_access(settings, sources, deps, tools):
                if question is not None:
                    _print_turn(await ask_async(question, deps, sources, today, thread_id))
                    return
                while True:
                    line = await anyio.to_thread.run_sync(sys.stdin.readline)
                    text = line.strip()
                    if not line or text.lower() in {"exit", "quit"}:
                        return
                    if text:
                        _print_turn(await ask_async(text, deps, sources, today, thread_id))

    def _print_turn(result: AskResult) -> None:
        _print_result(result, settings)
        typer.echo(f"thread_id={result.thread_id} turn={result.turn}\n")

    try:
        with _open_store() as store:
            store.ensure_collection()
            anyio.run(session, store)
    finally:
        provider.shutdown()


AutoApproveMemory = Annotated[
    bool,
    typer.Option(
        "--auto-approve-memory",
        help="Let the agent write long-term memory without approval (fixture data only).",
    ),
]


def _memory_auto_approve(settings: Settings, flag: bool) -> bool:
    auto = settings.memory_auto_approve or flag
    if auto and settings.data_mode == "live":
        typer.echo("--auto-approve-memory is refused with DATA_MODE=live.", err=True)
        raise typer.Exit(2)
    return auto


def _agent_deps(
    settings: Settings, store: ChunkStore, provider: TracerProvider, k: int, auto_approve: bool
) -> Deps:
    llm = make_chat_model(settings)
    try:
        memory = make_memory_backend(settings)
    except NotImplementedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    return Deps(
        store=store,
        llm=llm,
        tracer=get_tracer(provider),
        planner=make_planner(settings.llm_provider, llm),
        k=k,
        max_hops=settings.max_hops,
        tool_timeout_s=settings.tool_timeout_s,
        memory=memory,
        data_mode=settings.data_mode,
        memory_auto_approve=auto_approve,
        injection_detector=InjectionDetector(threshold=settings.injection_threshold),
        guard_mode=settings.retrieved_guard_mode,
    )


@asynccontextmanager
async def _tool_access(
    settings: Settings, sources: list[Source], deps: Deps, enabled: bool
) -> AsyncIterator[None]:
    """Attach live MCP tool access to `deps` for the duration of the block."""
    if not enabled:
        yield
        return
    async with open_clients(settings, sources) as clients:
        deps.tools = MCPToolExecutor(clients, settings.tool_timeout_s)
        yield


def _print_result(result: AskResult, settings: Settings) -> None:
    typer.echo(result.answer)
    if result.citations:
        typer.echo("\nSources:")
        for citation in result.citations:
            typer.echo(citation.render())
    typer.echo("\nPlan:")
    for sq in result.plan:
        mark = " (unanswered)" if sq.id in result.unanswered else ""
        typer.echo(f"  {sq.id} [{sq.relation} -> {','.join(sq.target_sources)}] {sq.text}{mark}")
    for failure in result.failures:
        typer.echo(f"  tool failure: {failure.source}.{failure.tool}: {failure.error}")
    for guard in result.guard_notes:
        typer.echo(
            f"  guard: {guard.action} {guard.spans} suspected instruction(s) from {guard.key} "
            f"({', '.join(guard.rules)})"
        )
    for note in result.memory_notes:
        state = "stale" if note.stale else ("verified" if note.verified else "unverified")
        typer.echo(f"  memory note [{state}] {note.key} (source {note.source})")
    for write in result.memory_writes:
        typer.echo(f"  memory {write.op} {write.key}: {write.status} ({write.reason})")
    if not result.citation_report.ok:
        typer.echo(f"\ncitation check: {result.citation_report.model_dump_json()}")
    usage = result.token_usage
    typer.echo(
        f"\ntrace_id={result.trace_id} hops={result.hops} tool_calls={len(result.tool_calls)} "
        f"latency_ms={result.latency_ms} tokens_in={usage.input_tokens} "
        f"tokens_out={usage.output_tokens} tracing={settings.tracing_backend} "
        f"llm={settings.llm_provider}"
    )


def _require_live_ready(settings: Settings, sources: list[Source]) -> list[Source]:
    """Fixture mode: unchanged. Live mode: only LIVE_SOURCES, and every one configured."""
    chosen = active_sources(settings, sources)
    if settings.data_mode != "live":
        return chosen
    if not chosen:
        typer.echo("none of the requested sources is in LIVE_SOURCES", err=True)
        raise typer.Exit(2)
    missing = {s: m for s, m in settings.live_missing().items() if m and s in chosen}
    if missing:
        for s, names in missing.items():
            typer.echo(f"{s}: missing {', '.join(names)}", err=True)
        typer.echo("run `sentinel live check` for details", err=True)
        raise typer.Exit(2)
    return chosen


live_app = typer.Typer(
    no_args_is_help=True, help="Live (read-only) connectors: readiness, smoke test, Gmail consent."
)
app.add_typer(live_app, name="live")

_LIVE_ENDPOINTS: dict[str, list[str]] = {
    "notion": [
        f"POST {NOTION_API}/search  (search only)",
        f"GET  {NOTION_API}/pages/<id>, {NOTION_API}/blocks/<id>/children",
        f"Notion-Version: {NOTION_VERSION}",
    ],
    "jira": [
        "GET  <JIRA_BASE_URL>/rest/api/3/search/jql",
        "GET  <JIRA_BASE_URL>/rest/api/3/issue/<KEY>",
    ],
    "gmail": [
        f"POST {GMAIL_TOKEN_URL}  (refresh token -> access token)",
        f"GET  {GMAIL_API}/threads, {GMAIL_API}/threads/<id>",
        f"scope: {READONLY_SCOPE} (any broader grant is refused)",
    ],
}


@live_app.command("check")
def live_check() -> None:
    """Show live configuration and what is missing. Makes no network calls."""
    settings = get_settings()
    typer.echo(
        f"DATA_MODE={settings.data_mode} LIVE_SOURCES={','.join(settings.live_sources) or '-'}"
    )
    missing = settings.live_missing()
    for source in SOURCES:
        if source not in settings.live_sources:
            typer.echo(f"{source:<7} not enabled")
            continue
        state = "ready" if not missing[source] else f"missing {', '.join(missing[source])}"
        typer.echo(f"{source:<7} {state}")
        for line in _LIVE_ENDPOINTS[source]:
            typer.echo(f"         {line}")
    if any(missing.values()):
        raise typer.Exit(1)


@live_app.command("smoke")
def live_smoke(
    source: Annotated[str, typer.Option("--source", "-s", help="notion, jira or gmail")],
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm: this calls the real API (read-only).")
    ] = False,
) -> None:
    """One read-only search (limit 1) and one fetch against a real account. Prints counts
    and ids only. Requires DATA_MODE=live and --yes."""
    settings = get_settings()
    [chosen] = _sources([source])
    if settings.data_mode != "live":
        typer.echo("smoke needs DATA_MODE=live", err=True)
        raise typer.Exit(2)
    _require_live_ready(settings, [chosen])
    if not yes:
        typer.echo(f"would call on {chosen}: {'; '.join(describe()[chosen])}")
        typer.echo("re-run with --yes to make these read-only calls")
        raise typer.Exit(1)

    async def run() -> SmokeReport:
        async with open_clients(settings, [chosen]) as clients:
            return await smoke(chosen, clients[chosen])

    report = anyio.run(run)
    typer.echo(report.model_dump_json(indent=2))
    if not report.ok:
        raise typer.Exit(1)


@live_app.command("gmail-auth")
def live_gmail_auth(
    timeout_s: Annotated[float, typer.Option(help="Seconds to wait for consent.")] = 300.0,
) -> None:
    """Run Google's consent flow (gmail.readonly only) and print GMAIL_REFRESH_TOKEN."""
    settings = get_settings()
    if not (settings.gmail_client_id and settings.gmail_client_secret):
        typer.echo("set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET first", err=True)
        raise typer.Exit(2)
    client_id = settings.gmail_client_id
    secret = settings.gmail_client_secret.get_secret_value()

    async def run() -> str:
        return await obtain_refresh_token(client_id, secret, webbrowser.open, timeout_s)

    token = anyio.run(run)
    typer.echo("Consent granted for gmail.readonly only. Add this line to .env (keep it secret):")
    typer.echo(f"GMAIL_REFRESH_TOKEN={token}")


memory_app = typer.Typer(
    no_args_is_help=True,
    help="Long-term memory. Writes here are made by you, so they need no further approval.",
)
app.add_typer(memory_app, name="memory")


def _print_items(items: list[MemoryItem]) -> None:
    if not items:
        typer.echo("no memory items")
    for i in items:
        flags = ",".join(f for f, on in (("verified", i.verified), ("stale", i.stale)) if on)
        typer.echo(
            f"{i.key}: {i.value}\n    source={i.source} updated={i.updated_at.date()} "
            f"confidence={i.confidence:.2f} {flags or 'unverified'}"
            + (f" stale_reason={i.stale_reason}" if i.stale_reason else "")
        )


def _memory() -> MemoryBackend:
    try:
        return make_memory_backend(get_settings())
    except NotImplementedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@memory_app.command("list")
def memory_list(
    include_stale: Annotated[bool, typer.Option(help="Include stale items.")] = True,
) -> None:
    """List remembered facts."""
    _print_items(_memory().all(include_stale=include_stale))


@memory_app.command("search")
def memory_search(
    query: str,
    limit: Annotated[int, typer.Option(min=1, max=50)] = 5,
    include_stale: Annotated[bool, typer.Option(help="Include stale items.")] = False,
) -> None:
    """Search remembered facts (fresh and verified first)."""
    _print_items(_memory().search(query, limit, include_stale=include_stale))


@memory_app.command("put")
def memory_put(key: str, value: str) -> None:
    """Remember a fact you state yourself (stored unverified, source=user)."""
    try:
        item = MemoryItem(key=key, value=value, source="user", updated_at=datetime.now(UTC))
    except ValueError as exc:
        raise typer.BadParameter(str(exc).splitlines()[0]) from exc
    _print_items([_memory().put(item)])


@memory_app.command("mark-stale")
def memory_mark_stale(
    key: str, reason: Annotated[str, typer.Option(help="Why it is stale.")] = "marked by user"
) -> None:
    """Flag a remembered fact as stale."""
    item = _memory().mark_stale(key, reason)
    if item is None:
        typer.echo(f"no memory item {key!r}", err=True)
        raise typer.Exit(1)
    _print_items([item])


@infra_app.command("init")
def infra_init(
    path: InfraEnvPath = infra.INFRA_ENV_FILE,
    force: Annotated[
        bool, typer.Option(help="Overwrite an existing file (breaks existing volumes).")
    ] = False,
) -> None:
    """Generate random local credentials for docker-compose.yml into .env.infra."""
    try:
        values = infra.write_infra_env(path, force=force)
    except FileExistsError:
        typer.echo(f"{path} already exists; keeping it (use --force to regenerate).", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Wrote {path}. Start services with: make up")
    typer.echo(
        f"Langfuse UI: {infra.LANGFUSE_LOCAL_HOST} "
        f"(login {values['LANGFUSE_INIT_USER_EMAIL']}, password in {path})"
    )
    typer.echo("To send traces to it, add these lines to .env (or run `sentinel infra app-env`):")
    for line in infra.app_env_lines(values):
        typer.echo(f"  {line}")


@infra_app.command("app-env")
def infra_app_env(path: InfraEnvPath = infra.INFRA_ENV_FILE) -> None:
    """Print .env lines that enable Langfuse tracing against the local stack."""
    if not path.exists():
        typer.echo(f"{path} not found; run `sentinel infra init` first.", err=True)
        raise typer.Exit(1)
    for line in infra.app_env_lines(infra.read_infra_env(path)):
        typer.echo(line)


@infra_app.command("check")
def infra_check(
    wait: Annotated[float, typer.Option(help="Keep retrying for up to N seconds.")] = 0.0,
    skip_langfuse: Annotated[bool, typer.Option(help="Only check Qdrant.")] = False,
    path: InfraEnvPath = infra.INFRA_ENV_FILE,
) -> None:
    """Check Qdrant and Langfuse health, and that the Langfuse keys authenticate."""
    settings = get_settings()
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    keys: tuple[str, str] | None = None
    if public_key and secret_key:
        keys = (public_key.get_secret_value(), secret_key.get_secret_value())
    elif path.exists():
        local = infra.read_infra_env(path)
        keys = (
            local["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"],
            local["LANGFUSE_INIT_PROJECT_SECRET_KEY"],
        )

    def run() -> list[infra.CheckResult]:
        results = [infra.check_qdrant(settings.qdrant_url)]
        if not skip_langfuse:
            results.append(infra.check_langfuse_health(settings.langfuse_host))
            if keys:
                results.append(infra.check_langfuse_auth(settings.langfuse_host, *keys))
        return results

    results = infra.wait_for(run, timeout_s=wait)
    for r in results:
        typer.echo(f"{'ok  ' if r.ok else 'FAIL'} {r.name:<14} {r.detail}")
    if not skip_langfuse and keys is None:
        typer.echo("note: no Langfuse keys configured; skipped the auth check")
    if not all(r.ok for r in results):
        raise typer.Exit(1)

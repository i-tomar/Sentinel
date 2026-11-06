"""Command-line entry point (`sentinel`). Commands are added as each build step lands."""

from __future__ import annotations

import json
import re
import sys
import uuid
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
from sentinel.llm import make_chat_model
from sentinel.mcp_clients.pool import open_clients
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
    sources = _sources(source)
    if settings.data_mode == "live":
        typer.echo("DATA_MODE=live connectors arrive in step (i); use fixture.", err=True)
        raise typer.Exit(2)

    with _open_store() as store:

        async def run() -> list[SourceReport]:
            async with open_clients(settings, sources) as clients:
                return await ingest_sources(clients, store)

        reports = anyio.run(run)
        for r in reports:
            if r.error:
                typer.echo(f"FAIL {r.source:<7} {r.error} (existing index left unchanged)")
            else:
                typer.echo(
                    f"ok   {r.source:<7} docs={r.documents} added={r.added} updated={r.updated} "
                    f"unchanged={r.unchanged} deleted={r.deleted} "
                    f"chunks_written={r.chunks_written}"
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
) -> None:
    """Plan, call tools over several hops, and answer with citations and a trace id."""
    settings = get_settings()
    sources = _sources(source)
    if tools and settings.data_mode == "live":
        typer.echo("DATA_MODE=live connectors arrive in step (i); use --no-tools.", err=True)
        raise typer.Exit(2)
    today = as_of.replace(tzinfo=UTC) if as_of else None
    provider = build_tracer_provider(settings)

    async def run(store: ChunkStore) -> AskResult:
        deps = _agent_deps(settings, store, provider, k)
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
    if tools and settings.data_mode == "live":
        typer.echo("DATA_MODE=live connectors arrive in step (i); use --no-tools.", err=True)
        raise typer.Exit(2)
    today = as_of.replace(tzinfo=UTC) if as_of else None
    thread_id = thread or uuid.uuid4().hex
    provider = build_tracer_provider(settings)

    async def session(store: ChunkStore) -> None:
        async with open_checkpointer(settings.checkpoint_db) as saver:
            deps = _agent_deps(settings, store, provider, k)
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


def _agent_deps(settings: Settings, store: ChunkStore, provider: TracerProvider, k: int) -> Deps:
    llm = make_chat_model(settings)
    return Deps(
        store=store,
        llm=llm,
        tracer=get_tracer(provider),
        planner=make_planner(settings.llm_provider, llm),
        k=k,
        max_hops=settings.max_hops,
        tool_timeout_s=settings.tool_timeout_s,
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
    if not result.citation_report.ok:
        typer.echo(f"\ncitation check: {result.citation_report.model_dump_json()}")
    usage = result.token_usage
    typer.echo(
        f"\ntrace_id={result.trace_id} hops={result.hops} tool_calls={len(result.tool_calls)} "
        f"latency_ms={result.latency_ms} tokens_in={usage.input_tokens} "
        f"tokens_out={usage.output_tokens} tracing={settings.tracing_backend} "
        f"llm={settings.llm_provider}"
    )


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

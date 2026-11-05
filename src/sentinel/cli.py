"""Command-line entry point (`sentinel`). Commands are added as each build step lands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from sentinel import __version__, infra
from sentinel.config import get_settings

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

"""Command-line entry point (`sentinel`). Commands are added as each build step lands."""

from __future__ import annotations

import json

import typer

from sentinel import __version__
from sentinel.config import get_settings

app = typer.Typer(name="sentinel", no_args_is_help=True, add_completion=False)


@app.command()
def version() -> None:
    """Print the installed Sentinel version."""
    typer.echo(__version__)


@app.command()
def config() -> None:
    """Print the effective configuration as JSON, with secrets masked."""
    typer.echo(json.dumps(get_settings().model_dump(mode="json"), indent=2))

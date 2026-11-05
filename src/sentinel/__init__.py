"""Sentinel: guarded agentic RAG over live workspace data."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sentinel")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0"

__all__ = ["__version__"]

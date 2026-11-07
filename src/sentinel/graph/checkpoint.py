"""Checkpointer for short-term memory: one SQLite file, one checkpoint history per thread_id.

LangGraph deserializes checkpoint values with msgpack. Types not explicitly allowed are not
restored as objects (they silently come back as plain dicts), and loading arbitrary types
from a file is unsafe, so every model that can appear in AgentState is allowlisted here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from sentinel.citations import Citation
from sentinel.graph.state import (
    CitationReport,
    Evidence,
    GuardNote,
    InputCheck,
    SubQuestion,
    TokenUsage,
    ToolCall,
    ToolFailure,
    TurnRecord,
    UnsupportedClaim,
)
from sentinel.memory.models import MemoryItem, MemoryWrite
from sentinel.permissions.actions import ActionRecord
from sentinel.retrieval.models import Chunk, Document

STATE_MODELS: tuple[type[BaseModel], ...] = (
    ActionRecord,
    Chunk,
    Citation,
    CitationReport,
    Document,
    Evidence,
    GuardNote,
    InputCheck,
    MemoryItem,
    MemoryWrite,
    SubQuestion,
    TokenUsage,
    ToolCall,
    ToolFailure,
    TurnRecord,
    UnsupportedClaim,
)


def make_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(m.__module__, m.__name__) for m in STATE_MODELS]
    )


@asynccontextmanager
async def open_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(path)) as conn:
        saver = AsyncSqliteSaver(conn, serde=make_serde())
        await saver.setup()
        yield saver

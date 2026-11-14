"""End to end: a model that cites in gpt-oss style still produces checked `[n]` citations.

The fake model replies with the verbatim gpt-oss answer (lenticular brackets, U+3010/U+3011).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from evals.datasets import load_cases
from evals.harness import HarnessConfig, run_case
from sentinel.llm import MockLLM

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

GPT_OSS_REPLY = "The owner of the Q3 Auth Migration Plan (v2) is Priya Sharma\u30101\u3011."


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class GptOssStyleLLM(MockLLM):
    """Always answers with the gpt-oss-style reply."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        message = AIMessage(content=GPT_OSS_REPLY, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_gpt_oss_citations_are_checked_like_any_other() -> None:
    case = next(c for c in load_cases() if c.id == "norm-001")
    result = await run_case(case, HarnessConfig(llm_factory=GptOssStyleLLM))
    assert result.harness_error is None
    assert "\u3010" not in result.answer
    assert "\u3011" not in result.answer
    assert "Priya Sharma[1]" in result.answer
    assert [c.key for c in result.citations] == ["notion:auth-migration-v2"]
    assert result.citations[0].claim.startswith("The owner of the Q3 Auth Migration Plan")

from __future__ import annotations

import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sentinel.config import Settings
from sentinel.llm import (
    INSUFFICIENT_PREFIX,
    MissingProviderError,
    MockLLM,
    extractive_answer,
    make_chat_model,
    prompt_hash,
)

pytestmark = pytest.mark.unit

PROMPT = """Question: Who owns the auth migration plan?

Evidence:
<evidence n="1" source="notion" id="auth-migration-v2" title="Plan" updated="2026-10-27">
Owner: Priya Sharma. Cutover date is 2026-11-20.
</evidence>

<evidence n="2" source="gmail" id="thr-0001" title="Lunch" updated="2026-10-01">
Team lunch is on Friday.
</evidence>"""


def test_prompt_hash_is_stable_and_role_sensitive() -> None:
    a = [SystemMessage("rules"), HumanMessage("q")]
    assert prompt_hash(a) == prompt_hash([SystemMessage("rules"), HumanMessage("q")])
    assert prompt_hash(a) != prompt_hash([HumanMessage("rules"), HumanMessage("q")])


def test_extractive_answer_cites_only_prompt_evidence() -> None:
    answer = extractive_answer(PROMPT)
    assert answer == "Owner: Priya Sharma [1]."
    assert "[2]" not in answer


def test_extractive_answer_skips_stale_evidence_and_spreads_sources() -> None:
    prompt = """Question: Who owns the auth migration?

Evidence:
<evidence n="1" source="notion" id="v2" title="Plan v2" updated="2026-10-27">
Auth migration owner: Priya Sharma.
</evidence>

<evidence n="2" source="notion" id="v1" title="Plan v1" updated="2026-07-05" stale="true">
Auth migration owner: Vikram Rao.
</evidence>

<evidence n="3" source="notion" id="x" title="Other" updated="2026-10-01">
Auth migration owner listed in the OKRs as Priya.
</evidence>

<evidence n="4" source="gmail" id="thr" title="Status" updated="2026-10-28">
I own the auth migration now.
</evidence>"""
    answer = extractive_answer(prompt, max_blocks=2)
    assert "Vikram" not in answer  # stale block never quoted
    assert "[2]" not in answer
    assert "[4]" in answer  # gmail represented even though notion block 3 scores higher
    assert "[1]" in answer


def test_extractive_answer_without_overlap_is_insufficient() -> None:
    no_match = PROMPT.replace("Who owns the auth migration plan?", "Capital of France?")
    assert extractive_answer(no_match).startswith(INSUFFICIENT_PREFIX)


def test_mock_llm_scripted_and_default_replies() -> None:
    messages = [SystemMessage("rules"), HumanMessage(PROMPT)]
    llm = MockLLM(scripted={prompt_hash(messages): "Scripted [1]."})
    reply = llm.invoke(messages)
    assert reply.content == "Scripted [1]."
    assert isinstance(reply, AIMessage)
    assert reply.usage_metadata == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    other = llm.invoke([HumanMessage(PROMPT)])
    assert other.content == "Owner: Priya Sharma [1]."
    assert llm.calls == [prompt_hash(messages), prompt_hash([HumanMessage(PROMPT)])]


def test_default_provider_is_mock() -> None:
    assert isinstance(make_chat_model(Settings()), MockLLM)


@pytest.mark.parametrize(
    ("provider", "module", "cls", "key_attr"),
    [
        ("openai", "langchain_openai", "ChatOpenAI", "openai_api_key"),
        ("anthropic", "langchain_anthropic", "ChatAnthropic", "anthropic_api_key"),
        ("google", "langchain_google_genai", "ChatGoogleGenerativeAI", "google_api_key"),
    ],
)
def test_real_provider_selection(provider: str, module: str, cls: str, key_attr: str) -> None:
    settings = Settings.model_validate(
        {"llm_provider": provider, "llm_model": "test-model", "llm_api_key": "sk-secret-xyz"}
    )
    model = make_chat_model(settings)
    assert type(model).__name__ == cls
    assert type(model).__module__.startswith(module)
    assert "test-model" in str(getattr(model, "model_name", None) or getattr(model, "model", ""))
    assert getattr(model, key_attr).get_secret_value() == "sk-secret-xyz"
    assert "sk-secret-xyz" not in repr(model)


def test_missing_provider_extra_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)  # makes the import fail
    settings = Settings.model_validate(
        {"llm_provider": "anthropic", "llm_model": "m", "llm_api_key": "k"}
    )
    with pytest.raises(MissingProviderError, match="uv sync --extra anthropic"):
        make_chat_model(settings)

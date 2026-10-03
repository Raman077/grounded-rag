import json
import os
from typing import Any

import httpx
import pytest
from query_service.providers import ChatMessage, Completion, LLMError, LLMEvent, TextDelta
from query_service.providers.opencode_provider import OpenCodeProvider


def _provider(handler: Any) -> OpenCodeProvider:
    return OpenCodeProvider(
        "http://opencode.test", "openrouter/qwen/qwen3.8-27b:free", transport=httpx.MockTransport(handler)
    )


async def _collect(provider: OpenCodeProvider, messages: list[ChatMessage], system: str = "SYS") -> list[LLMEvent]:
    return [e async for e in provider.stream(system=system, messages=messages, max_tokens=100)]


async def test_creates_prompts_and_deletes_session() -> None:
    seen: list[tuple[str, str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.url.path == "/session" and request.method == "POST":
            return httpx.Response(200, json={"id": "ses_1"})
        if request.url.path == "/session/ses_1/message":
            return httpx.Response(
                200,
                json={
                    "info": {
                        "providerID": "openrouter",
                        "modelID": "qwen/qwen3.8-27b:free",
                        "finish": "stop",
                        "tokens": {"input": 669, "output": 14, "cache": {"read": 5}},
                    },
                    "parts": [
                        {"type": "step-start"},
                        {"type": "reasoning", "text": "thinking..."},
                        {"type": "text", "text": "It costs $49 [S1]."},
                        {"type": "step-finish"},
                    ],
                },
            )
        return httpx.Response(200, json=True)

    events = await _collect(_provider(handler), [{"role": "user", "content": "Q?"}])

    assert events == [
        TextDelta("It costs $49 [S1]."),
        Completion(
            model="openrouter/qwen/qwen3.8-27b:free",
            stop="end",
            input_tokens=669,
            output_tokens=14,
            cached_input_tokens=5,
        ),
    ]
    message_body = seen[1][2]
    assert message_body["agent"] == "rag"
    assert message_body["system"] == "SYS"
    assert message_body["model"] == {"providerID": "openrouter", "modelID": "qwen/qwen3.8-27b:free"}
    assert message_body["parts"] == [{"type": "text", "text": "Q?"}]
    assert seen[-1][:2] == ("DELETE", "/session/ses_1")


async def test_history_is_inlined_as_transcript() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "s"})
        if request.url.path.endswith("/message"):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"info": {}, "parts": [{"type": "text", "text": "ok"}]})
        return httpx.Response(200, json=True)

    await _collect(
        _provider(handler),
        [
            {"role": "user", "content": "What is Pro?"},
            {"role": "assistant", "content": "A plan."},
            {"role": "user", "content": "How much?"},
        ],
    )
    text = captured["parts"][0]["text"]
    assert text.startswith("Conversation so far:\nUser: What is Pro?\nAssistant: A plan.")
    assert text.endswith("How much?")


async def test_server_errors_are_retryable_and_session_still_deleted() -> None:
    deleted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(200, json=True)
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "s"})
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMError) as info:
        await _collect(_provider(handler), [{"role": "user", "content": "Q"}])
    assert info.value.retryable
    assert deleted == ["/session/s"]


async def test_unreachable_server_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError) as info:
        await _collect(_provider(handler), [{"role": "user", "content": "Q"}])
    assert info.value.retryable


def test_model_must_include_provider() -> None:
    with pytest.raises(ValueError):
        OpenCodeProvider("http://x", "just-a-model")


@pytest.mark.opencode
async def test_live_opencode_server_answers_with_citation() -> None:
    """Needs `opencode serve` running from deploy/opencode. Run with: pytest -m opencode"""
    provider = OpenCodeProvider(
        os.getenv("RAG_OPENCODE_URL", "http://127.0.0.1:4096"),
        os.getenv("RAG_OPENCODE_MODEL", "openrouter/qwen/qwen3.8-27b:free"),
        timeout=180,
    )
    try:
        events = await _collect(
            provider,
            [
                {
                    "role": "user",
                    "content": (
                        '<source id="S1">The Pro plan costs $49 per month.</source>\n\nQuestion: How much is Pro?'
                    ),
                }
            ],
            system="Answer only from the sources. Cite every fact like [S1].",
        )
    finally:
        await provider.aclose()
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "49" in text
    assert isinstance(events[-1], Completion)

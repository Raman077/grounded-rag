"""Generation through a local ``opencode serve`` instance (development provider).

Any model OpenCode can reach (OpenRouter free models, OpenCode Zen, ...) can then answer
queries without an Anthropic key. Run the server from ``deploy/opencode`` so the tool-less
``rag`` agent defined there is used:

    cd deploy/opencode && opencode serve --port 4096

Each query gets a throwaway session that is deleted afterwards. The server API returns the
finished message, so the answer arrives as a single delta rather than token by token.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from query_service.providers.base import ChatMessage, Completion, LLMError, LLMEvent, TextDelta


class OpenCodeProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        agent: str = "rag",
        password: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        provider_id, sep, model_id = model.partition("/")
        if not sep or not model_id:
            raise ValueError(f"opencode model must look like provider/model, got {model!r}")
        self.model = model
        self._model_ref = {"providerID": provider_id, "modelID": model_id}
        self._agent = agent
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            auth=("opencode", password) if password else None,
            transport=transport,
        )

    async def stream(self, *, system: str, messages: list[ChatMessage], max_tokens: int) -> AsyncIterator[LLMEvent]:
        session_id = await self._create_session()
        try:
            body = {
                "agent": self._agent,
                "model": self._model_ref,
                "system": system,
                "parts": [{"type": "text", "text": _flatten(messages)}],
            }
            data = await self._request("POST", f"/session/{session_id}/message", json=body)
        finally:
            await self._delete_session(session_id)

        info: dict[str, Any] = data.get("info") or {}
        if info.get("error"):
            raise LLMError(f"opencode generation failed: {info['error']}", retryable=True)
        text = "".join(p.get("text", "") for p in data.get("parts", []) if p.get("type") == "text")
        if text:
            yield TextDelta(text)

        tokens: dict[str, Any] = info.get("tokens") or {}
        provider, model_id = self._model_ref["providerID"], self._model_ref["modelID"]
        stop: Literal["end", "max_tokens", "refusal"] = "max_tokens" if info.get("finish") == "length" else "end"
        yield Completion(
            model=f"{info.get('providerID', provider)}/{info.get('modelID', model_id)}",
            stop=stop,
            input_tokens=int(tokens.get("input", 0)),
            output_tokens=int(tokens.get("output", 0)),
            cached_input_tokens=int((tokens.get("cache") or {}).get("read", 0)),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _create_session(self) -> str:
        data = await self._request("POST", "/session", json={"title": "grounded-rag query"})
        return str(data["id"])

    async def _delete_session(self, session_id: str) -> None:
        try:
            await self._client.delete(f"/session/{session_id}")
        except httpx.HTTPError:
            pass  # best effort; a leftover session is harmless

    async def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        try:
            resp = await self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise LLMError(f"opencode server unreachable: {exc!r}", retryable=True) from exc
        if resp.status_code >= 400:
            raise LLMError(
                f"opencode returned {resp.status_code}: {resp.text[:300]}",
                retryable=resp.status_code >= 500,
            )
        result: dict[str, Any] = resp.json()
        return result


def _flatten(messages: list[ChatMessage]) -> str:
    """OpenCode takes one user turn per call, so earlier turns are inlined as a transcript."""
    *history, last = messages
    if not history:
        return last["content"]
    transcript = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history)
    return f"Conversation so far:\n{transcript}\n\n{last['content']}"

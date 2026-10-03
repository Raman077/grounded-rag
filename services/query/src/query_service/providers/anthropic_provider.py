from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

import anthropic

from query_service.providers.base import ChatMessage, Completion, LLMError, LLMEvent, TextDelta


class AnthropicProvider:
    """Streams from the Messages API. The system prompt is marked cacheable; it only
    actually caches once it exceeds the model's minimum cacheable prefix length."""

    def __init__(
        self,
        model: str,
        *,
        effort: Literal["low", "medium", "high"] = "low",
        thinking: Literal["disabled", "adaptive"] = "disabled",
        timeout: float = 120.0,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self._effort = effort
        self._thinking = thinking
        self._client = client or anthropic.AsyncAnthropic(timeout=timeout)

    async def stream(self, *, system: str, messages: list[ChatMessage], max_tokens: int) -> AsyncIterator[LLMEvent]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "output_config": {"effort": self._effort},
            "thinking": {"type": self._thinking},
        }
        try:
            async with self._client.messages.stream(**request) as stream:
                async for text in stream.text_stream:
                    yield TextDelta(text)
                final = await stream.get_final_message()
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            raise LLMError(f"generation unavailable: {exc}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"generation failed: {exc.message}", retryable=exc.status_code >= 500) from exc

        stop: Literal["end", "max_tokens", "refusal"] = "end"
        if final.stop_reason == "max_tokens":
            stop = "max_tokens"
        elif final.stop_reason == "refusal":
            stop = "refusal"
        usage = final.usage
        yield Completion(
            model=final.model,
            stop=stop,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cache_read_input_tokens or 0,
        )

    async def aclose(self) -> None:
        await self._client.close()

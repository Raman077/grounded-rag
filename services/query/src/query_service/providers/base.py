"""Thin provider interface so generation models can be swapped without touching the pipeline."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class Completion:
    model: str
    stop: Literal["end", "max_tokens", "refusal"]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


LLMEvent = TextDelta | Completion


class LLMError(Exception):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(Protocol):
    model: str

    def stream(self, *, system: str, messages: list[ChatMessage], max_tokens: int) -> AsyncIterator[LLMEvent]:
        """Yield TextDelta events and finish with exactly one Completion. Raise LLMError on failure."""
        ...

    async def aclose(self) -> None: ...

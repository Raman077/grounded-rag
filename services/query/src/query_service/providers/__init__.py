from rag_core.config import Settings

from query_service.providers.base import (
    ChatMessage,
    Completion,
    LLMError,
    LLMEvent,
    LLMProvider,
    TextDelta,
)


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "opencode":
        from query_service.providers.opencode_provider import OpenCodeProvider

        password = settings.opencode_password.get_secret_value() if settings.opencode_password else None
        return OpenCodeProvider(
            settings.opencode_url,
            settings.opencode_model,
            agent=settings.opencode_agent,
            password=password,
            timeout=settings.llm_timeout_seconds,
        )

    from query_service.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(
        settings.generation_model,
        effort=settings.generation_effort,
        thinking=settings.generation_thinking,
        timeout=settings.llm_timeout_seconds,
    )


__all__ = [
    "ChatMessage",
    "Completion",
    "LLMError",
    "LLMEvent",
    "LLMProvider",
    "TextDelta",
    "build_provider",
]

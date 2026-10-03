"""Versioned prompt templates (``prompts/*.yaml``) and context packing."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import yaml
from rag_core.schemas import HistoryTurn
from rag_core.vectorstore import ScoredChunk

from query_service.providers import ChatMessage

INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: int
    system: str
    user: str

    @property
    def key(self) -> str:
        return f"{self.id}-v{self.version}"


def load_prompt(prompt_dir: Path, name: str) -> PromptTemplate:
    path = prompt_dir / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    template = PromptTemplate(
        id=str(data["id"]), version=int(data["version"]), system=data["system"], user=data["user"]
    )
    for placeholder in ("{sources}", "{question}"):
        if placeholder not in template.user:
            raise ValueError(f"{path}: user template must contain {placeholder}")
    return template


@dataclass(frozen=True)
class PackedSource:
    source_id: str  # "S1", "S2", ...
    chunk: ScoredChunk


def pack_context(chunks: list[ScoredChunk], *, max_tokens: int) -> list[PackedSource]:
    """Keep the highest-scoring chunks that fit in the token budget, best first."""
    packed: list[PackedSource] = []
    used = 0
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        cost = len(chunk.payload.text) // 4 + 30  # text plus the <source> wrapper
        if packed and used + cost > max_tokens:
            break
        packed.append(PackedSource(f"S{len(packed) + 1}", chunk))
        used += cost
    return packed


def render_sources(sources: list[PackedSource]) -> str:
    blocks = []
    for src in sources:
        p = src.chunk.payload
        attrs = {
            "id": src.source_id,
            "title": p.title,
            "section": " > ".join(p.section_path) or None,
            "updated": p.updated_at.date().isoformat() if p.updated_at else None,
        }
        rendered = " ".join(f'{k}="{escape(v, quote=True)}"' for k, v in attrs.items() if v)
        # Escape the body so document text can never close the tag or inject markup.
        blocks.append(f"<source {rendered}>\n{escape(p.text, quote=False)}\n</source>")
    return "\n\n".join(blocks)


def build_messages(
    template: PromptTemplate, *, question: str, history: list[HistoryTurn], sources: list[PackedSource]
) -> tuple[str, list[ChatMessage]]:
    user = template.user.replace("{sources}", render_sources(sources)).replace("{question}", question)
    turns = list(history)
    while turns and turns[0].role == "assistant":  # the first message must come from the user
        turns.pop(0)
    messages: list[ChatMessage] = [{"role": turn.role, "content": turn.content} for turn in turns]
    messages.append({"role": "user", "content": user})
    return template.system, messages

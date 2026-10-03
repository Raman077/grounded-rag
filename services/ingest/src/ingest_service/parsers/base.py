from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class ParseError(Exception):
    """The document could not be turned into usable text (unsupported, empty or corrupt)."""


@dataclass
class Section:
    heading_path: list[str]
    text: str
    page: int | None = None


@dataclass
class ParsedDocument:
    title: str
    sections: list[Section] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


def default_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()

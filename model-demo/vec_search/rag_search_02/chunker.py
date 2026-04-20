import hashlib
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MIN_MEANINGFUL_LEN = 2

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """A single chunk extracted from a markdown document."""

    content: str
    source: str  # file path
    heading: str  # nearest md heading (empty string for preamble)
    heading_level: int  # 0 for preamble
    start_line: int
    end_line: int
    content_hash: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.content_hash:
            h = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            object.__setattr__(self, "content_hash", h)

    def compute_chunk_id(self, model_name: str) -> str:
        raw = f"markdown:{self.source}:{self.start_line}:{self.end_line}:{self.content_hash}:{model_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_markdown(
    text: str, source: str = "", max_chunk_size: int = 1500
) -> list[Chunk]:
    """Split markdown *text* into chunks, breaking on headings."""
    lines = text.split("\n")

    # find all heading positions
    heading_positions: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            heading_positions.append((i, len(m.group(1)), m.group(2).strip()))

    # build sections between headings
    sections: list[tuple[int, int, str, int]] = []  # (start, end, heading, level)
    if not heading_positions or heading_positions[0][0] > 0:
        end = heading_positions[0][0] if heading_positions else len(lines)
        sections.append((0, end, "", 0))

    for idx, (line_idx, level, title) in enumerate(heading_positions):
        next_start = (
            heading_positions[idx + 1][0]
            if idx + 1 < len(heading_positions)
            else len(lines)
        )
        sections.append((line_idx, next_start, title, level))

    chunks: list[Chunk] = []
    for start, end, heading, level in sections:
        section_text = "\n".join(lines[start:end]).strip()
        if not section_text or not _has_meaningful_content(section_text):
            continue
        if len(section_text) > max_chunk_size:
            logging.warning(
                "skip md section which exceed max chunk length %d", max_chunk_size
            )
            continue

        chunks.append(
            Chunk(
                content=section_text,
                source=source,
                heading=heading,
                heading_level=level,
                start_line=start + 1,
                end_line=end,
            )
        )

    return chunks


def _has_meaningful_content(text: str) -> bool:
    return len(text) > _MIN_MEANINGFUL_LEN


if __name__ == "__main__":
    pass

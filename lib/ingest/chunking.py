"""
Structure-aware text chunking for Polymath v3.

Key insight: Scientific papers have semantic structure (Abstract, Methods, Results).
Sliding window destroys this. Header-aware chunking preserves it.

Supports:
- Markdown header splitting (from MinerU/magic-pdf output)
- Plain text heuristic chunking (from fitz output)
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    """A semantically meaningful text chunk."""

    content: str
    header: Optional[str] = None
    parent_header: Optional[str] = None
    level: int = 0  # Header level (1=H1, 2=H2, etc.)
    char_start: int = 0
    char_end: int = 0
    page_num: Optional[int] = None

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def __len__(self) -> int:
        return len(self.content)


def chunk_markdown_by_headers(
    md_text: str,
    min_chunk_size: int = 200,
    max_chunk_size: int = 4000,
    include_header_in_content: bool = True,
) -> list[Chunk]:
    """
    Split markdown text by headers to preserve document structure.

    Args:
        md_text: Markdown text (from MinerU or similar)
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size (split if exceeded)
        include_header_in_content: Whether to include header in chunk content

    Returns:
        List of Chunk objects with preserved structure
    """
    if not md_text or not md_text.strip():
        return []

    # Pattern for markdown headers
    header_pattern = r"^(#{1,6})\s+(.+?)$"
    lines = md_text.split("\n")

    chunks = []
    current_chunk_lines = []
    current_header = None
    current_level = 0
    parent_headers = {}  # level -> header name
    char_offset = 0

    def finalize_chunk():
        """Save current chunk if it has content."""
        nonlocal current_chunk_lines, char_offset

        if not current_chunk_lines:
            return

        content = "\n".join(current_chunk_lines).strip()
        if len(content) < min_chunk_size:
            return

        # Get parent header (one level up)
        parent = parent_headers.get(current_level - 1) if current_level > 1 else None

        chunk_start = char_offset - len(content) - 1
        chunk = Chunk(
            content=content,
            header=current_header,
            parent_header=parent,
            level=current_level,
            char_start=max(0, chunk_start),
            char_end=char_offset,
        )
        chunks.append(chunk)
        current_chunk_lines = []

    for line in lines:
        line_with_newline = line + "\n"
        header_match = re.match(header_pattern, line, re.MULTILINE)

        if header_match:
            # New header found - finalize previous chunk
            finalize_chunk()

            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()

            # Update parent tracking
            parent_headers[level] = header_text
            # Clear deeper levels
            for l in list(parent_headers.keys()):
                if l > level:
                    del parent_headers[l]

            current_header = header_text
            current_level = level

            if include_header_in_content:
                current_chunk_lines.append(line)

        else:
            current_chunk_lines.append(line)

            # Check if chunk is getting too large
            current_content = "\n".join(current_chunk_lines)
            if len(current_content) > max_chunk_size:
                # Split at paragraph boundary if possible
                split_idx = _find_paragraph_boundary(current_content, max_chunk_size)
                if split_idx > 0:
                    # Save first part
                    first_part = current_content[:split_idx].strip()
                    if first_part:
                        parent = parent_headers.get(current_level - 1)
                        chunk = Chunk(
                            content=first_part,
                            header=current_header,
                            parent_header=parent,
                            level=current_level,
                            char_start=char_offset - len(current_content),
                            char_end=char_offset - len(current_content) + split_idx,
                        )
                        chunks.append(chunk)

                    # Continue with remainder
                    current_chunk_lines = [current_content[split_idx:].strip()]

        char_offset += len(line_with_newline)

    # Don't forget the last chunk
    finalize_chunk()

    return chunks


def _find_paragraph_boundary(text: str, max_pos: int) -> int:
    """Find a good split point (paragraph boundary) before max_pos."""
    # Look for double newline (paragraph break)
    idx = text.rfind("\n\n", 0, max_pos)
    if idx > max_pos * 0.5:  # At least halfway through
        return idx + 2

    # Fall back to single newline
    idx = text.rfind("\n", 0, max_pos)
    if idx > max_pos * 0.5:
        return idx + 1

    # Fall back to sentence boundary
    for sep in [". ", "? ", "! "]:
        idx = text.rfind(sep, 0, max_pos)
        if idx > max_pos * 0.5:
            return idx + len(sep)

    # Last resort: split at max_pos
    return max_pos


def chunk_plain_text(
    text: str,
    min_chunk_size: int = 200,
    max_chunk_size: int = 4000,
) -> list[Chunk]:
    """
    Chunk plain text using heuristic header detection.

    Used for text from fitz/PyMuPDF where markdown headers aren't available.
    Detects headers by:
    - ALL CAPS lines
    - Common section names (Abstract, Introduction, Methods, etc.)
    - Short lines followed by longer paragraphs

    Args:
        text: Plain text content
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []

    # Common scientific paper sections
    section_patterns = [
        r"^(abstract|introduction|background|methods|methodology|materials and methods|"
        r"results|discussion|conclusion|conclusions|references|acknowledgements?|"
        r"supplementary|appendix|data availability|author contributions|"
        r"competing interests|figure legends?)[\s:]*$",
    ]

    lines = text.split("\n")
    chunks = []
    current_chunk_lines = []
    current_header = None
    char_offset = 0

    def is_header_line(line: str) -> tuple[bool, Optional[str]]:
        """Check if a line looks like a section header."""
        line = line.strip()

        if not line or len(line) > 100:
            return False, None

        # ALL CAPS (at least 3 words)
        if line.isupper() and len(line.split()) >= 2:
            return True, line.title()

        # Numbered section (1. Introduction, 2.1 Methods)
        numbered = re.match(r"^(\d+\.?\d*\.?)\s+([A-Z][a-zA-Z\s]+)$", line)
        if numbered:
            return True, numbered.group(2).strip()

        # Common section names
        for pattern in section_patterns:
            if re.match(pattern, line.lower()):
                return True, line.title()

        return False, None

    def finalize_chunk():
        """Save current chunk."""
        nonlocal current_chunk_lines

        if not current_chunk_lines:
            return

        content = "\n".join(current_chunk_lines).strip()
        if len(content) < min_chunk_size:
            return

        chunk_end = char_offset
        chunk_start = chunk_end - len(content)

        chunk = Chunk(
            content=content,
            header=current_header,
            char_start=max(0, chunk_start),
            char_end=chunk_end,
        )
        chunks.append(chunk)
        current_chunk_lines = []

    for line in lines:
        is_header, header_text = is_header_line(line)

        if is_header:
            finalize_chunk()
            current_header = header_text
            current_chunk_lines.append(line)
        else:
            current_chunk_lines.append(line)

            # Check size limit
            current_content = "\n".join(current_chunk_lines)
            if len(current_content) > max_chunk_size:
                split_idx = _find_paragraph_boundary(current_content, max_chunk_size)
                if split_idx > 0:
                    first_part = current_content[:split_idx].strip()
                    if first_part:
                        chunk = Chunk(
                            content=first_part,
                            header=current_header,
                            char_start=char_offset - len(current_content),
                            char_end=char_offset - len(current_content) + split_idx,
                        )
                        chunks.append(chunk)
                    current_chunk_lines = [current_content[split_idx:].strip()]

        char_offset += len(line) + 1  # +1 for newline

    finalize_chunk()

    return chunks


def get_chunk_with_context(chunk: Chunk, max_context_chars: int = 200) -> str:
    """
    Get chunk content with header context for better embeddings.

    Format: "[Header] content..."

    Args:
        chunk: Chunk object
        max_context_chars: Max chars to include from content

    Returns:
        Formatted string for embedding
    """
    parts = []

    if chunk.parent_header:
        parts.append(f"[{chunk.parent_header}]")

    if chunk.header:
        parts.append(f"[{chunk.header}]")

    content = chunk.content[:max_context_chars * 10]  # Leave room for truncation
    parts.append(content)

    return " ".join(parts)

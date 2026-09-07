"""Text chunking for RAG — splits documents into overlapping chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextChunk:
    """A single text chunk with metadata."""

    text: str
    metadata: dict[str, Any]
    chunk_id: str
    start_char: int
    end_char: int


class TextChunker:
    """Splits documents into overlapping chunks for retrieval.

    Sentence-aware chunking with configurable size, overlap, and minimum
    chunk size. Chunks are built from complete sentences to preserve
    semantic coherence.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any],
        source: str = "unknown",
    ) -> list[TextChunk]:
        """Split text into overlapping sentence-based chunks.

        Args:
            text: Raw document text.
            metadata: Additional metadata to attach to each chunk.
            source: Source identifier (used in chunk_id).

        Returns:
            List of TextChunk objects.
        """
        chunks: list[TextChunk] = []
        text = self._clean_text(text)
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        current: list[str] = []
        current_len = 0
        chunk_idx = 0

        for sentence in sentences:
            s_len = len(sentence)

            if current_len + s_len > self.chunk_size and current:
                chunk_text = " ".join(current)

                if len(chunk_text) >= self.min_chunk_size:
                    # Find start position in original text
                    first_sent = current[0]
                    start = text.find(first_sent)
                    if start == -1:
                        start = 0

                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            metadata={
                                **metadata,
                                "source": source,
                                "chunk_index": chunk_idx,
                            },
                            chunk_id=f"{source}_{chunk_idx}",
                            start_char=start,
                            end_char=start + len(chunk_text),
                        )
                    )
                    chunk_idx += 1

                # Build next window with overlap sentences
                overlap: list[str] = []
                overlap_len = 0
                for sent in reversed(current):
                    if overlap_len + len(sent) <= self.chunk_overlap:
                        overlap.insert(0, sent)
                        overlap_len += len(sent)
                    else:
                        break

                current = overlap + [sentence]
                current_len = sum(len(s) for s in current)
            else:
                current.append(sentence)
                current_len += s_len

        # Final chunk
        if current:
            chunk_text = " ".join(current)
            if len(chunk_text) >= self.min_chunk_size:
                first_sent = current[0]
                start = text.find(first_sent)
                if start == -1:
                    start = 0
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        metadata={
                            **metadata,
                            "source": source,
                            "chunk_index": chunk_idx,
                        },
                        chunk_id=f"{source}_{chunk_idx}",
                        start_char=start,
                        end_char=start + len(chunk_text),
                    )
                )

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Preserve course abbreviations, currency, punctuation
        text = re.sub(r"[^\w\s.,!?;:()\-–—'\"₹%/]", " ", text)
        return text.strip()

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Match sentence-ending punctuation followed by space/newline
        # Handle common abbreviations to avoid false splits
        protected = text.replace("Mr.", "Mr").replace(
            "Dr.", "Dr"
        ).replace("Prof.", "Prof").replace(
            "Mrs.", "Mrs"
        ).replace(
            "Ms.", "Ms"
        )
        sentences = re.split(r"(?<=[.!?])\s+", protected)
        return [s.replace("", ".").strip() for s in sentences if s.strip()]

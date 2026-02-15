"""
Token counting utilities.

Uses tiktoken for accurate token counting compatible with
OpenAI embeddings and other transformer models.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import tiktoken


class TokenCounter:
    """
    Count tokens using tiktoken encodings.

    Supports multiple encodings for different model families.
    """

    ENCODING_MAP = {
        "cl100k_base": "cl100k_base",  # GPT-4, ChatGPT, text-embedding-ada-002
        "p50k_base": "p50k_base",  # Codex, text-davinci-002/003
        "r50k_base": "r50k_base",  # GPT-3, text-curie-001, etc.
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
    }

    def __init__(self, encoding: str = "cl100k_base") -> None:
        """
        Initialize token counter.

        Args:
            encoding: Encoding name or model name
        """
        self._encoding_name = self.ENCODING_MAP.get(encoding, encoding)
        self._encoder = tiktoken.get_encoding(self._encoding_name)

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def count_many(self, texts: list[str]) -> list[int]:
        """Count tokens in multiple texts."""
        return [self.count(text) for text in texts]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        if not text:
            return ""

        tokens = self._encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated = self._encoder.decode(tokens[:max_tokens])
        return truncated

    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Split text into chunks of max_tokens."""
        if not text:
            return []

        tokens = self._encoder.encode(text)
        chunks = []

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = self._encoder.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks


# Singleton for convenience
_default_counter: TokenCounter | None = None


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """
    Count tokens in text using default counter.

    For performance, caches the encoder.
    """
    global _default_counter
    if _default_counter is None or _default_counter._encoding_name != encoding:
        _default_counter = TokenCounter(encoding)
    return _default_counter.count(text)

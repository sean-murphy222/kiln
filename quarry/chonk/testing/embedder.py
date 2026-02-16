"""
Embedding model wrapper for CHONK.

Provides a consistent interface for embedding text using
sentence-transformers models (bundled) or optional API models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


class EmbeddingModel(Enum):
    """Available embedding models."""

    # Bundled models (sentence-transformers)
    MINILM = "all-MiniLM-L6-v2"  # 80MB, fastest
    MPNET = "all-mpnet-base-v2"  # 420MB, better quality
    BGE_SMALL = "BAAI/bge-small-en-v1.5"  # 130MB, modern

    # API models (require API key)
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"


@dataclass
class ModelInfo:
    """Information about an embedding model."""

    name: str
    dimensions: int
    size_mb: int
    is_bundled: bool
    description: str


class Embedder:
    """
    Embed text using sentence-transformers or API models.

    The default model (all-MiniLM-L6-v2) is bundled with CHONK and
    runs entirely locally without any API calls.
    """

    # Model metadata
    MODEL_INFO: ClassVar[dict[str, ModelInfo]] = {
        "all-MiniLM-L6-v2": ModelInfo(
            name="all-MiniLM-L6-v2",
            dimensions=384,
            size_mb=80,
            is_bundled=True,
            description="Fast, lightweight model good for testing",
        ),
        "all-mpnet-base-v2": ModelInfo(
            name="all-mpnet-base-v2",
            dimensions=768,
            size_mb=420,
            is_bundled=False,
            description="Higher quality, slower",
        ),
        "BAAI/bge-small-en-v1.5": ModelInfo(
            name="BAAI/bge-small-en-v1.5",
            dimensions=384,
            size_mb=130,
            is_bundled=False,
            description="Modern model with good quality/speed tradeoff",
        ),
        "text-embedding-3-small": ModelInfo(
            name="text-embedding-3-small",
            dimensions=1536,
            size_mb=0,
            is_bundled=False,
            description="OpenAI API model (requires API key)",
        ),
        "text-embedding-3-large": ModelInfo(
            name="text-embedding-3-large",
            dimensions=3072,
            size_mb=0,
            is_bundled=False,
            description="OpenAI API model, highest quality (requires API key)",
        ),
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        """
        Initialize embedder.

        Args:
            model_name: Name of embedding model to use
            cache_dir: Directory to cache downloaded models
            openai_api_key: OpenAI API key (for API models)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._model = None
        self._is_openai = model_name.startswith("text-embedding-")

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for current model."""
        info = self.MODEL_INFO.get(self.model_name)
        if info:
            return info.dimensions
        return 384  # Default fallback

    def _load_model(self):
        """Load the embedding model."""
        if self._is_openai:
            return self._init_openai()
        else:
            return self._init_sentence_transformer()

    def _init_sentence_transformer(self):
        """Initialize sentence-transformers model."""
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if self.cache_dir:
            kwargs["cache_folder"] = str(self.cache_dir)

        return SentenceTransformer(self.model_name, **kwargs)

    def _init_openai(self):
        """Initialize OpenAI embedding client."""
        if not self._openai_api_key:
            raise ValueError(
                f"OpenAI API key required for model {self.model_name}. "
                "Set OPENAI_API_KEY environment variable or pass openai_api_key parameter."
            )

        try:
            from openai import OpenAI

            return OpenAI(api_key=self._openai_api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        if self._is_openai:
            return self._embed_openai([text])[0]
        else:
            return self.model.encode(text, convert_to_numpy=True)

    def embed_many(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Array of embedding vectors (shape: [n_texts, dimensions])
        """
        if not texts:
            return np.array([])

        if self._is_openai:
            return self._embed_openai(texts)
        else:
            return self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        response = self.model.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    @classmethod
    def available_models(cls) -> list[dict[str, Any]]:
        """Get information about available models."""
        return [
            {
                "name": info.name,
                "dimensions": info.dimensions,
                "size_mb": info.size_mb,
                "is_bundled": info.is_bundled,
                "description": info.description,
            }
            for info in cls.MODEL_INFO.values()
        ]

    @classmethod
    def bundled_models(cls) -> list[str]:
        """Get list of models that are bundled with CHONK."""
        return [
            name
            for name, info in cls.MODEL_INFO.items()
            if info.is_bundled
        ]

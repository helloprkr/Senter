"""
Embedding Model - Unified embedding interface.

Provides a simple interface for generating embeddings from any backend.
"""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
import numpy as np

from .base import ModelConfig, ModelInterface

if TYPE_CHECKING:
    from core.genome_parser import GenomeParser


class EmbeddingModel:
    """
    Unified embedding model wrapper.

    Handles embedding generation from any backend (GGUF, OpenAI, Ollama).
    """

    def __init__(self, model: Optional[ModelInterface] = None):
        self._model = model
        self._cache: dict[str, List[float]] = {}
        self._cache_enabled = True

    async def initialize(self) -> None:
        """Initialize the underlying model."""
        if self._model:
            await self._model.initialize()

    async def embed(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            Embedding vector
        """
        if not self._model:
            # Return zero vector if no model configured
            return [0.0] * 512

        # Check cache
        if use_cache and self._cache_enabled and text in self._cache:
            return self._cache[text]

        embedding = await self._model.embed(text)

        # Cache result
        if use_cache and self._cache_enabled:
            self._cache[text] = embedding

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._model:
            return [[0.0] * 512 for _ in texts]

        # Check which texts are cached
        cached = []
        uncached = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if self._cache_enabled and text in self._cache:
                cached.append((i, self._cache[text]))
            else:
                uncached.append(text)
                uncached_indices.append(i)

        # Embed uncached texts
        if uncached:
            new_embeddings = await self._model.embed_batch(uncached)

            # Cache new embeddings
            for text, embedding in zip(uncached, new_embeddings):
                if self._cache_enabled:
                    self._cache[text] = embedding

        # Combine results in correct order
        results = [None] * len(texts)
        for i, embedding in cached:
            results[i] = embedding
        for i, embedding in zip(uncached_indices, new_embeddings if uncached else []):
            results[i] = embedding

        return results

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity (-1 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def find_similar(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[tuple[int, float]]:
        """
        Find most similar embeddings to a query.

        Args:
            query_embedding: Query embedding vector
            embeddings: List of embeddings to search
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity) tuples
        """
        if not embeddings:
            return []

        similarities = []
        for i, embedding in enumerate(embeddings):
            sim = self.similarity(query_embedding, embedding)
            if sim >= threshold:
                similarities.append((i, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable caching."""
        self._cache_enabled = enabled

    async def close(self) -> None:
        """Clean up resources."""
        if self._model:
            await self._model.close()


async def get_embeddings(genome: "GenomeParser") -> EmbeddingModel:
    """
    Create an embedding model from genome configuration.

    Args:
        genome: Genome configuration parser

    Returns:
        Initialized embedding model
    """
    embed_config = genome.models.get("embeddings", {})

    if not embed_config:
        # Use primary model for embeddings
        embed_config = genome.models.get("primary", {})

    model_config = ModelConfig.from_dict(embed_config)

    # Create appropriate model
    from .base import create_model

    try:
        model = create_model(embed_config)
        await model.initialize()
        return EmbeddingModel(model)
    except Exception:
        # Return model without backend if initialization fails
        return EmbeddingModel(None)

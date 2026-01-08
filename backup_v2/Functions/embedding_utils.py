#!/usr/bin/env python3
"""
Embedding Utilities - Modular embedding creation and vector search
"""

import re
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import os


def chunk_text_logical(text: str, chunk_size: int = 3) -> List[str]:
    """
    Split text into logical chunks at sentence boundaries
    Each chunk contains approximately `chunk_size` sentences

    Args:
        text: Raw text to chunk
        chunk_size: Target number of sentences per chunk (default: 3)

    Returns:
        List of text chunks
    """
    # Split at sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []

    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence.strip())

        # End chunk at chunk_size sentences or end of text
        if len(current_chunk) >= chunk_size or i == len(sentences) - 1:
            chunk_text = " ".join([s for s in current_chunk if s.strip()])
            if chunk_text:
                chunks.append(chunk_text)
            current_chunk = []

    return chunks


def load_llama_cpp_model(model_path: str, embedding: bool = True):
    """
    Load llama-cpp-python model for embeddings or text generation

    Args:
        model_path: Path to model file
        embedding: True if loading for embeddings, False for text gen

    Returns:
        Llama instance
    """
    try:
        from llama_cpp import Llama

        return Llama(
            model_path=model_path,
            embedding=embedding,
            n_ctx=2048 if embedding else 8192,
            n_gpu_layers=-1,
            verbose=False,
        )
    except ImportError as e:
        raise ImportError(f"llama-cpp-python not installed: {e}")


def create_embeddings(texts: List[str], model_path: str) -> np.ndarray:
    """
    Create embeddings using llama-cpp-python

    Args:
        texts: List of text strings
        model_path: Path to embedding model

    Returns:
        Numpy array of embeddings
    """
    llm = load_llama_cpp_model(model_path, embedding=True)

    embeddings = []
    for text in texts:
        emb = llm.embed(text)
        embeddings.append(emb)

    return np.array(embeddings)


def vector_search(
    query: str,
    embeddings: np.ndarray,
    texts: List[str],
    top_k: int = 4,
    model_path: str = None,
) -> Tuple[List[str], List[float]]:
    """
    Perform vector similarity search

    Args:
        query: Query string to search for
        embeddings: Pre-computed embeddings array
        texts: Original texts corresponding to embeddings
        top_k: Number of top results to return
        model_path: Path to embedding model (for query embedding)

    Returns:
        Tuple of (top_k texts, scores)
    """
    if not model_path:
        raise ValueError("model_path required for query embedding")

    llm = load_llama_cpp_model(model_path, embedding=True)
    query_embedding = llm.embed(query)

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)

    if norms.sum() == 0 or query_norm == 0:
        # Handle zero vectors
        scores = np.zeros(len(texts))
    else:
        scores = np.dot(embeddings, query_embedding) / (norms * query_norm)

    # Get top_k indices (highest scores)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    top_texts = [texts[i] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]

    return top_texts, top_scores


def load_all_senter_md(topics_dir: str) -> List[dict]:
    """
    Load all SENTER.md files from topics directory

    Args:
        topics_dir: Path to Topics directory

    Returns:
        List of dicts: [{"topic": str, "content": str}]
    """
    topics_path = Path(topics_dir)
    if not topics_path.exists():
        return []

    topic_data = []

    for senter_file in topics_path.rglob("**/SENTER.md"):
        topic_name = senter_file.parent.name
        with open(senter_file, "r", encoding="utf-8") as f:
            content = f.read()
            topic_data.append({"topic": topic_name, "content": content})

    return topic_data


def get_default_embedding_model() -> Optional[str]:
    """
    Get default embedding model path from config or environment

    Returns:
        Path to embedding model or None
    """
    # Check environment variable
    env_model = os.environ.get("SENTER_EMBEDDING_MODEL")
    if env_model and Path(env_model).exists():
        return env_model

    # Check config file
    config_path = Path(__file__).parent.parent / "config" / "senter_config.json"
    if config_path.exists():
        import json

        with open(config_path, "r") as f:
            config = json.load(f)
            model_path = config.get("embedding_model", {}).get("path")
            if model_path and Path(model_path).exists():
                return model_path

    # Fallback
    fallback_path = "/home/sovthpaw/ai-toolbox/Senter/Models/nomic-embed-text.gguf"
    if Path(fallback_path).exists():
        return fallback_path
    return None

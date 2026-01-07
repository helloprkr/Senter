"""
Senter 3.0 Models Module

Model interfaces for GGUF, OpenAI, Ollama, and embeddings.
"""

from .base import ModelInterface, ModelConfig
from .gguf import GGUFModel
from .openai_model import OpenAIModel
from .ollama_model import OllamaModel
from .embeddings import EmbeddingModel, get_embeddings

__all__ = [
    "ModelInterface",
    "ModelConfig",
    "GGUFModel",
    "OpenAIModel",
    "OllamaModel",
    "EmbeddingModel",
    "get_embeddings",
]

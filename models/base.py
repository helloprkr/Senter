"""
Model Interface - Abstract base class for all model backends.

Provides a unified interface for GGUF, OpenAI, Ollama, and any future backends.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator


@dataclass
class ModelConfig:
    """Configuration for a model."""

    type: str  # gguf | openai | ollama
    model: str = ""  # Model name or path
    path: Optional[str] = None  # For GGUF models
    api_key_env: Optional[str] = None  # Environment variable for API key
    base_url: Optional[str] = None  # For custom endpoints
    settings: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(
            type=data.get("type", "openai"),
            model=data.get("model", ""),
            path=data.get("path"),
            api_key_env=data.get("api_key_env"),
            base_url=data.get("base_url"),
            settings=data.get("settings", {}),
        )


@dataclass
class Message:
    """A chat message."""

    role: str  # system | user | assistant
    content: str


class ModelInterface(ABC):
    """
    Abstract base class for all model backends.

    All model implementations must implement these methods.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model (load weights, connect to API, etc.)."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response from a prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response from chat messages.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        pass

    async def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response token by token.

        Default implementation falls back to non-streaming.
        """
        response = await self.generate(prompt, max_tokens, temperature)
        yield response

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Default implementation embeds one by one.
        """
        return [await self.embed(text) for text in texts]

    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return self.config.settings.get(key, default)

    async def close(self) -> None:
        """Clean up resources."""
        pass


def create_model(config: Dict[str, Any]) -> ModelInterface:
    """
    Factory function to create a model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Appropriate ModelInterface implementation
    """
    model_config = ModelConfig.from_dict(config)
    model_type = model_config.type.lower()

    if model_type == "gguf":
        from .gguf import GGUFModel

        return GGUFModel(model_config)
    elif model_type == "openai":
        from .openai_model import OpenAIModel

        return OpenAIModel(model_config)
    elif model_type == "ollama":
        from .ollama_model import OllamaModel

        return OllamaModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

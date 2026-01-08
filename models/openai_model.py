"""
OpenAI Model - OpenAI API integration.

Supports GPT-4, GPT-3.5, and embedding models via the OpenAI API.
"""

from __future__ import annotations
import os
from typing import List, Optional, AsyncIterator

from .base import ModelInterface, ModelConfig, Message


class OpenAIModel(ModelInterface):
    """
    OpenAI API model implementation.

    Supports chat completions and embeddings.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        # Get API key from environment
        api_key = None
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)

        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or specify api_key_env in config."
            )

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
        )
        self._initialized = True

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate response from prompt."""
        if not self._initialized:
            await self.initialize()

        messages = [{"role": "user", "content": prompt}]
        return await self._complete(messages, max_tokens, temperature, stop)

    async def chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response from chat messages."""
        if not self._initialized:
            await self.initialize()

        formatted = [{"role": m.role, "content": m.content} for m in messages]
        return await self._complete(formatted, max_tokens, temperature)

    async def _complete(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Internal completion method."""
        model = self.config.model or "gpt-4o-mini"
        max_tokens = max_tokens or self.get_setting("max_tokens", 1024)
        temperature = temperature or self.get_setting("temperature", 0.7)

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

        return response.choices[0].message.content

    async def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Stream response token by token."""
        if not self._initialized:
            await self.initialize()

        model = self.config.model or "gpt-4o-mini"
        max_tokens = max_tokens or self.get_setting("max_tokens", 1024)
        temperature = temperature or self.get_setting("temperature", 0.7)

        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        if not self._initialized:
            await self.initialize()

        model = self.config.model
        if not model or "gpt" in model.lower():
            model = "text-embedding-3-small"

        dimensions = self.get_setting("dimensions", 512)

        response = await self._client.embeddings.create(
            model=model,
            input=text,
            dimensions=dimensions,
        )

        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not self._initialized:
            await self.initialize()

        model = self.config.model
        if not model or "gpt" in model.lower():
            model = "text-embedding-3-small"

        dimensions = self.get_setting("dimensions", 512)

        response = await self._client.embeddings.create(
            model=model,
            input=texts,
            dimensions=dimensions,
        )

        return [item.embedding for item in response.data]

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.close()
        self._client = None
        self._initialized = False

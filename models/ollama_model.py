"""
Ollama Model - Local model inference via Ollama.

Supports running local models through the Ollama server.
"""

from __future__ import annotations
from typing import Any, List, Optional, AsyncIterator

import httpx

from .base import ModelInterface, ModelConfig, Message


class OllamaModel(ModelInterface):
    """
    Ollama model implementation.

    Connects to a local or remote Ollama server.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None
        self._base_url = config.base_url or "http://localhost:11434"

    async def initialize(self) -> None:
        """Initialize the Ollama client."""
        if self._initialized:
            return

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

        # Verify connection by listing models
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self._base_url}: {e}"
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

        model = self.config.model or "llama3.2"
        temperature = temperature or self.get_setting("temperature", 0.7)

        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = stop

        response = await self._client.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
        )
        response.raise_for_status()

        return response.json()["response"]

    async def chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response from chat messages."""
        if not self._initialized:
            await self.initialize()

        model = self.config.model or "llama3.2"
        temperature = temperature or self.get_setting("temperature", 0.7)

        formatted = [{"role": m.role, "content": m.content} for m in messages]

        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        response = await self._client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": formatted,
                "stream": False,
                "options": options,
            },
        )
        response.raise_for_status()

        return response.json()["message"]["content"]

    async def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Stream response token by token."""
        if not self._initialized:
            await self.initialize()

        model = self.config.model or "llama3.2"
        temperature = temperature or self.get_setting("temperature", 0.7)

        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        async with self._client.stream(
            "POST",
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": options,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        if not self._initialized:
            await self.initialize()

        model = self.config.model or "nomic-embed-text"

        response = await self._client.post(
            "/api/embeddings",
            json={
                "model": model,
                "prompt": text,
            },
        )
        response.raise_for_status()

        return response.json()["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Ollama doesn't support batch embeddings, so we do them sequentially
        return [await self.embed(text) for text in texts]

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
        self._client = None
        self._initialized = False

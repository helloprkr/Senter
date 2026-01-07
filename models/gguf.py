"""
GGUF Model - Local model inference via llama-cpp-python.

Supports running local GGUF models with GPU acceleration.
"""

from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from .base import ModelInterface, ModelConfig, Message


class GGUFModel(ModelInterface):
    """
    GGUF model implementation using llama-cpp-python.

    Runs inference locally with optional GPU acceleration.
    """

    def __init__(self, config: ModelConfig, embedding_mode: bool = False):
        super().__init__(config)
        self.embedding_mode = embedding_mode
        self._llm = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def initialize(self) -> None:
        """Load the GGUF model."""
        if self._initialized:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )

        model_path = self.config.path
        if not model_path:
            raise ValueError("GGUF model requires 'path' in config")

        # Get settings
        n_gpu_layers = self.get_setting("n_gpu_layers", -1)
        n_ctx = self.get_setting("n_ctx", 8192)
        verbose = self.get_setting("verbose", False)

        # Load model in thread pool (blocking operation)
        def load_model():
            return Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                embedding=self.embedding_mode,
                verbose=verbose,
            )

        loop = asyncio.get_event_loop()
        self._llm = await loop.run_in_executor(self._executor, load_model)
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

        max_tokens = max_tokens or self.get_setting("max_tokens", 1024)
        temperature = temperature or self.get_setting("temperature", 0.7)
        stop = stop or self.get_setting("stop", [])

        def _generate():
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                echo=False,
            )
            return output["choices"][0]["text"]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _generate)

    async def chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response from chat messages."""
        if not self._initialized:
            await self.initialize()

        # Convert messages to prompt format
        prompt = self._format_chat_prompt(messages)
        return await self.generate(prompt, max_tokens, temperature)

    def _format_chat_prompt(self, messages: List[Message]) -> str:
        """Format messages into a chat prompt."""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}\n")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}\n")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}\n")

        parts.append("Assistant:")
        return "".join(parts)

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        if not self._initialized:
            await self.initialize()

        if not self.embedding_mode:
            raise ValueError("Model not initialized in embedding mode")

        def _embed():
            return self._llm.embed(text)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _embed)

    async def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        self._llm = None
        self._initialized = False

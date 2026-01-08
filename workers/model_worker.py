#!/usr/bin/env python3
"""
Model Worker

Dedicated process for LLM inference.

Multiple workers can run simultaneously:
- Primary worker: User-facing responses (low latency priority)
- Research worker: Background tasks (throughput priority)

Features (MW-001, MW-002, MW-003):
- Local GGUF model loading with llama-cpp-python
- Streaming responses with sentence chunking
- Model hot-swapping without restart
"""

import time
import logging
import sys
import re
import threading
from pathlib import Path
from typing import Optional, Callable, Generator
from multiprocessing import Event
from queue import Empty
import uuid

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon.message_bus import MessageBus, MessageType, Message

logger = logging.getLogger('senter.worker')


# MW-001: GGUF Model Loader
class GGUFModelLoader:
    """Loads GGUF models directly using llama-cpp-python."""

    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096):
        self.model_path = Path(model_path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._llm = None

    def load(self) -> bool:
        """Load the GGUF model. Returns True on success."""
        try:
            from llama_cpp import Llama
            logger.info(f"Loading GGUF model: {self.model_path}")
            self._llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False
            )
            logger.info(f"GGUF model loaded: {self.model_path.name}")
            return True
        except ImportError:
            logger.warning("llama-cpp-python not installed, GGUF loading unavailable")
            return False
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            return False

    def unload(self):
        """Unload the model to free memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("GGUF model unloaded")

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """Generate response from GGUF model."""
        if self._llm is None:
            raise RuntimeError("Model not loaded")

        # Build full prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"

        if stream:
            return self._stream_generate(full_prompt, max_tokens, temperature)
        else:
            output = self._llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|user|>", "<|system|>"]
            )
            return output["choices"][0]["text"].strip()

    def _stream_generate(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> Generator[str, None, None]:
        """Stream tokens from GGUF model."""
        stream = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|user|>", "<|system|>"],
            stream=True
        )
        for chunk in stream:
            text = chunk["choices"][0].get("text", "")
            if text:
                yield text

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None


class ModelWorker:
    """
    LLM inference worker process.

    Each worker:
    - Has its own model client
    - Processes requests from message bus
    - Returns responses to bus

    Features:
    - MW-001: GGUF model loading with Ollama fallback
    - MW-002: Streaming responses with sentence chunking
    - MW-003: Model hot-swapping without restart
    """

    def __init__(
        self,
        name: str,
        model: str,
        message_bus: MessageBus,
        shutdown_event: Event,
        priority: str = "balanced",
        model_type: str = "ollama",  # MW-001: "ollama" or "gguf"
        model_path: str = None,  # MW-001: Path for GGUF models
        n_gpu_layers: int = -1,  # MW-001: GPU layers for GGUF
        stream_default: bool = True  # MW-002: Default streaming
    ):
        self.name = name
        self.model_name = model
        self.message_bus = message_bus
        self.shutdown_event = shutdown_event
        self.priority = priority

        # MW-001: Model configuration
        self.model_type = model_type
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers

        # MW-002: Streaming configuration
        self.stream_default = stream_default

        # MW-003: Hot-swap lock
        self._swap_lock = threading.Lock()
        self._swap_in_progress = False

        # Model clients
        self._client = None  # Ollama client
        self._gguf_loader = None  # GGUF model loader

        # Message queue
        self._queue = None

        # In-flight request tracking (MW-003)
        self._in_flight_count = 0
        self._in_flight_lock = threading.Lock()

        # Statistics
        self.requests_processed = 0
        self.total_tokens = 0
        self.total_latency = 0

    @property
    def client(self):
        """Lazy load model client"""
        if self._client is None and self.model_type == "ollama":
            self._client = self._create_ollama_client()
        return self._client

    def _create_ollama_client(self):
        """Create Ollama client (OpenAI API compatible)"""
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            # Test connection
            client.models.list()
            logger.info(f"Worker {self.name}: Connected to Ollama")
            return client
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return None

    # MW-001: GGUF model loading
    def load_gguf_model(self, model_path: str = None, n_gpu_layers: int = None) -> bool:
        """Load a GGUF model directly."""
        path = model_path or self.model_path
        if not path:
            logger.error("No GGUF model path specified")
            return False

        gpu_layers = n_gpu_layers if n_gpu_layers is not None else self.n_gpu_layers

        self._gguf_loader = GGUFModelLoader(
            model_path=path,
            n_gpu_layers=gpu_layers
        )

        if self._gguf_loader.load():
            self.model_type = "gguf"
            self.model_path = path
            logger.info(f"Worker {self.name}: GGUF model loaded")
            return True
        else:
            # MW-001: Fallback to Ollama
            logger.warning("GGUF loading failed, falling back to Ollama")
            self._gguf_loader = None
            self.model_type = "ollama"
            return False

    # MW-003: Model hot-swapping
    def swap_model(self, new_model: str, model_type: str = None, model_path: str = None) -> bool:
        """
        Hot-swap the active model without restarting the worker.
        Waits for in-flight requests to complete.
        """
        with self._swap_lock:
            if self._swap_in_progress:
                logger.warning(f"Worker {self.name}: Swap already in progress")
                return False
            self._swap_in_progress = True

        try:
            # Wait for in-flight requests to complete
            logger.info(f"Worker {self.name}: Waiting for in-flight requests...")
            timeout = 30  # Max wait time
            start = time.time()
            while True:
                with self._in_flight_lock:
                    if self._in_flight_count == 0:
                        break
                if time.time() - start > timeout:
                    logger.error("Timeout waiting for in-flight requests")
                    return False
                time.sleep(0.1)

            # Unload current model
            if self._gguf_loader:
                self._gguf_loader.unload()
                self._gguf_loader = None

            # Clear Ollama client if switching away
            if model_type == "gguf" and self._client:
                self._client = None

            # Load new model
            old_model = self.model_name
            self.model_name = new_model

            if model_type == "gguf" or (model_type is None and model_path):
                success = self.load_gguf_model(model_path)
            else:
                self.model_type = "ollama"
                # Will lazy-load on next request
                success = True

            if success:
                logger.info(f"Worker {self.name}: Swapped model {old_model} -> {new_model}")
            else:
                logger.error(f"Worker {self.name}: Failed to swap model")
                self.model_name = old_model

            return success

        finally:
            with self._swap_lock:
                self._swap_in_progress = False

    def get_model_info(self) -> dict:
        """Get current model information."""
        return {
            "name": self.name,
            "model": self.model_name,
            "type": self.model_type,
            "path": self.model_path,
            "loaded": self._gguf_loader.is_loaded if self._gguf_loader else (self._client is not None),
            "in_flight": self._in_flight_count
        }

    def run(self):
        """Main worker loop"""
        logger.info(f"Model worker '{self.name}' starting with model '{self.model_name}'")

        # Register with message bus
        component_name = f"model_{self.name}"
        self._queue = self.message_bus.register(component_name)

        logger.info(f"Model worker '{self.name}' started")

        while not self.shutdown_event.is_set():
            try:
                # Get request with timeout
                try:
                    msg_dict = self._queue.get(timeout=1.0)
                except Empty:
                    continue

                message = Message.from_dict(msg_dict)

                # Check if this message is for us
                if not self._should_process(message):
                    continue

                # Process request
                self._process_request(message)

            except Exception as e:
                logger.error(f"Worker {self.name} error: {e}")

        logger.info(f"Model worker '{self.name}' stopped")

    def _should_process(self, message: Message) -> bool:
        """Check if we should process this message"""
        # Process model requests
        if message.type == MessageType.MODEL_REQUEST:
            # Check target
            target = message.target
            if target is None or target == f"model_{self.name}":
                return True

        # Process user queries (primary worker only)
        if message.type == MessageType.USER_QUERY and self.name == "primary":
            return True

        # Process user voice (primary worker only)
        if message.type == MessageType.USER_VOICE and self.name == "primary":
            return True

        return False

    def _process_request(self, message: Message):
        """Process a single request"""
        # MW-003: Track in-flight requests
        with self._in_flight_lock:
            self._in_flight_count += 1

        try:
            start_time = time.time()

            payload = message.payload
            prompt = payload.get("prompt") or payload.get("text", "")
            system_prompt = payload.get("system_prompt", "You are Senter, a helpful AI assistant.")
            max_tokens = payload.get("max_tokens", 1024)
            temperature = payload.get("temperature", 0.7)

            # MW-002: Streaming option (default from config, can be overridden per-request)
            stream = payload.get("stream", self.stream_default)

            logger.info(f"Worker {self.name}: Processing request - {prompt[:50]}...")

            try:
                if stream:
                    # MW-002: Stream response with sentence chunking
                    self._stream_response(
                        message=message,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        start_time=start_time
                    )
                else:
                    # Non-streaming response
                    response_text = self._generate_response(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False
                    )

                    latency = time.time() - start_time
                    self.requests_processed += 1
                    self.total_latency += latency

                    # Send response
                    self.message_bus.send(
                        MessageType.MODEL_RESPONSE,
                        source=f"model_{self.name}",
                        payload={
                            "response": response_text,
                            "model": self.model_name,
                            "latency_ms": int(latency * 1000),
                            "worker": self.name,
                            "streamed": False
                        },
                        correlation_id=message.correlation_id
                    )

                    logger.info(f"Worker {self.name}: Response sent ({latency*1000:.0f}ms)")

            except Exception as e:
                logger.error(f"Worker {self.name}: Generation failed - {e}")

                # Send error response
                self.message_bus.send(
                    MessageType.ERROR,
                    source=f"model_{self.name}",
                    payload={
                        "error": str(e),
                        "worker": self.name
                    },
                    correlation_id=message.correlation_id
                )

        finally:
            # MW-003: Clear in-flight tracking
            with self._in_flight_lock:
                self._in_flight_count -= 1

    # MW-002: Streaming response with sentence chunking
    def _stream_response(
        self,
        message: Message,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        start_time: float
    ):
        """Stream response with sentence boundary chunking."""
        buffer = ""
        full_response = ""
        chunk_count = 0

        # Sentence-ending patterns
        sentence_endings = re.compile(r'([.!?]+[\s]+|[\n]+)')

        generator = self._generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        for token in generator:
            buffer += token
            full_response += token

            # Check for sentence boundaries
            match = sentence_endings.search(buffer)
            if match:
                # Send up to end of sentence
                chunk_end = match.end()
                chunk = buffer[:chunk_end]
                buffer = buffer[chunk_end:]

                if chunk.strip():
                    chunk_count += 1
                    self.message_bus.send(
                        MessageType.MODEL_RESPONSE,
                        source=f"model_{self.name}",
                        payload={
                            "response": chunk,
                            "chunk_number": chunk_count,
                            "is_complete": False,
                            "model": self.model_name,
                            "worker": self.name
                        },
                        correlation_id=message.correlation_id
                    )

        # Send any remaining buffer
        if buffer.strip():
            chunk_count += 1
            self.message_bus.send(
                MessageType.MODEL_RESPONSE,
                source=f"model_{self.name}",
                payload={
                    "response": buffer,
                    "chunk_number": chunk_count,
                    "is_complete": False,
                    "model": self.model_name,
                    "worker": self.name
                },
                correlation_id=message.correlation_id
            )

        # Send completion marker
        latency = time.time() - start_time
        self.requests_processed += 1
        self.total_latency += latency

        self.message_bus.send(
            MessageType.MODEL_RESPONSE,
            source=f"model_{self.name}",
            payload={
                "response": "",
                "full_response": full_response,
                "is_complete": True,
                "total_chunks": chunk_count,
                "model": self.model_name,
                "latency_ms": int(latency * 1000),
                "worker": self.name,
                "streamed": True
            },
            correlation_id=message.correlation_id
        )

        logger.info(f"Worker {self.name}: Streamed {chunk_count} chunks ({latency*1000:.0f}ms)")

    def _generate_response(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """Generate LLM response using GGUF or Ollama backend."""

        # MW-001: Use GGUF if loaded
        if self.model_type == "gguf" and self._gguf_loader and self._gguf_loader.is_loaded:
            return self._gguf_loader.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )

        # Fallback to Ollama
        if self.client is None:
            raise RuntimeError("No model client available")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            if stream:
                return self._stream_ollama(messages, max_tokens, temperature)
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

    def _stream_ollama(
        self, messages: list, max_tokens: int, temperature: float
    ) -> Generator[str, None, None]:
        """Stream response from Ollama."""
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_stats(self) -> dict:
        """Get worker statistics"""
        avg_latency = self.total_latency / max(1, self.requests_processed)
        return {
            "name": self.name,
            "model": self.model_name,
            "requests_processed": self.requests_processed,
            "avg_latency_ms": int(avg_latency * 1000),
            "total_tokens": self.total_tokens
        }


# Test
if __name__ == "__main__":
    from multiprocessing import Process

    logging.basicConfig(level=logging.INFO)

    # Create message bus
    bus = MessageBus()
    bus.start()

    # Create shutdown event
    shutdown = Event()

    # Create worker
    worker = ModelWorker(
        name="test",
        model="llama3.2",
        message_bus=bus,
        shutdown_event=shutdown
    )

    # Run in separate process
    p = Process(target=worker.run)
    p.start()

    time.sleep(2)

    # Send test request
    bus.send(
        MessageType.MODEL_REQUEST,
        source="test",
        target="model_test",
        payload={
            "prompt": "Say hello in one sentence.",
            "max_tokens": 50
        }
    )

    time.sleep(10)

    # Stop
    shutdown.set()
    p.join(timeout=5)
    bus.stop()

    print("Test complete")

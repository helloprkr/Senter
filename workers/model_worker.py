#!/usr/bin/env python3
"""
Model Worker

Dedicated process for LLM inference.

Multiple workers can run simultaneously:
- Primary worker: User-facing responses (low latency priority)
- Research worker: Background tasks (throughput priority)
"""

import time
import logging
import sys
from pathlib import Path
from typing import Optional
from multiprocessing import Event
from queue import Empty
import uuid

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon.message_bus import MessageBus, MessageType, Message

logger = logging.getLogger('senter.worker')


class ModelWorker:
    """
    LLM inference worker process.

    Each worker:
    - Has its own model client
    - Processes requests from message bus
    - Returns responses to bus
    """

    def __init__(
        self,
        name: str,
        model: str,
        message_bus: MessageBus,
        shutdown_event: Event,
        priority: str = "balanced"
    ):
        self.name = name
        self.model_name = model
        self.message_bus = message_bus
        self.shutdown_event = shutdown_event
        self.priority = priority

        # Model client
        self._client = None

        # Message queue
        self._queue = None

        # Statistics
        self.requests_processed = 0
        self.total_tokens = 0
        self.total_latency = 0

    @property
    def client(self):
        """Lazy load model client"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        """Create model client (Ollama via OpenAI API)"""
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
        start_time = time.time()

        payload = message.payload
        prompt = payload.get("prompt") or payload.get("text", "")
        system_prompt = payload.get("system_prompt", "You are Senter, a helpful AI assistant.")
        max_tokens = payload.get("max_tokens", 1024)
        temperature = payload.get("temperature", 0.7)

        logger.info(f"Worker {self.name}: Processing request - {prompt[:50]}...")

        try:
            response_text = self._generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
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
                    "worker": self.name
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

    def _generate_response(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate LLM response"""
        if self.client is None:
            return "Error: Model client not available"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
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

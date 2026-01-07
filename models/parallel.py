"""
Parallel Model Manager - Runs two inference processes.

Architecture:
+-------------------------------------------------------------+
|                  PARALLEL MODEL MANAGER                      |
|                                                             |
|  +-------------------------+  +-------------------------+   |
|  |   FOREGROUND MODEL      |  |   BACKGROUND MODEL      |   |
|  |   (High priority)       |  |   (Low priority)        |   |
|  |                         |  |                         |   |
|  |   - User interactions   |  |   - Research tasks      |   |
|  |   - Real-time responses |  |   - Summarization       |   |
|  |   - Preempts background |  |   - Analysis            |   |
|  +-------------------------+  +-------------------------+   |
|                                                             |
|  +-----------------------------------------------------+    |
|  |                COORDINATION LAYER                    |    |
|  |   - Preemption signals                               |    |
|  |   - GPU memory management                            |    |
|  |   - Result queuing                                   |    |
|  +-----------------------------------------------------+    |
|                                                             |
+-------------------------------------------------------------+
"""

from __future__ import annotations
import asyncio
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from queue import Empty
from typing import Any, Dict, Optional
import time
import uuid


class ModelPriority(Enum):
    FOREGROUND = 1  # User interaction - never wait
    BACKGROUND = 2  # Autonomous tasks - can be preempted


@dataclass
class InferenceRequest:
    """A request for model inference."""

    request_id: str
    prompt: str
    priority: ModelPriority
    max_tokens: int = 1024
    temperature: float = 0.7


@dataclass
class InferenceResult:
    """Result of model inference."""

    request_id: str
    text: str
    tokens_generated: int
    time_taken: float
    preempted: bool = False


def _model_worker(
    model_config: Dict,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    preempt_event: mp.Event,
    worker_id: str,
) -> None:
    """
    Worker process that runs model inference.

    Runs in separate process to enable true parallelism.
    """
    import asyncio

    # Initialize model in this process
    model = None
    model_type = model_config.get("type", "ollama")

    try:
        if model_type == "gguf":
            from models.gguf import GGUFModel

            model = GGUFModel(model_config)
        elif model_type == "openai":
            from models.openai_model import OpenAIModel

            model = OpenAIModel(model_config)
        else:  # Default to Ollama
            from models.ollama_model import OllamaModel

            model = OllamaModel(model_config)

        # Initialize model
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(model.initialize())

        print(f"[{worker_id}] Model loaded: {model_type}")

    except Exception as e:
        print(f"[{worker_id}] Failed to load model: {e}")
        return

    while True:
        try:
            # Get request (block with timeout)
            request: InferenceRequest = request_queue.get(timeout=1.0)

            if request is None:  # Shutdown signal
                print(f"[{worker_id}] Shutting down")
                break

            start_time = time.time()
            preempted = False

            try:
                # For background tasks, check preemption periodically
                if request.priority == ModelPriority.BACKGROUND:
                    # Generate with preemption checking
                    text = ""

                    # Check if model supports streaming
                    if hasattr(model, "generate_stream"):
                        async def stream_with_preemption():
                            nonlocal text, preempted
                            async for chunk in model.generate_stream(
                                request.prompt,
                                max_tokens=request.max_tokens,
                                temperature=request.temperature,
                            ):
                                if preempt_event.is_set():
                                    preempted = True
                                    preempt_event.clear()
                                    break
                                text += chunk

                        loop.run_until_complete(stream_with_preemption())
                    else:
                        # No streaming, do regular generation
                        text = loop.run_until_complete(
                            model.generate(
                                request.prompt,
                                max_tokens=request.max_tokens,
                                temperature=request.temperature,
                            )
                        )
                else:
                    # Foreground: generate without interruption
                    text = loop.run_until_complete(
                        model.generate(
                            request.prompt,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                        )
                    )

                result = InferenceResult(
                    request_id=request.request_id,
                    text=text,
                    tokens_generated=len(text.split()),
                    time_taken=time.time() - start_time,
                    preempted=preempted,
                )

            except Exception as e:
                print(f"[{worker_id}] Generation error: {e}")
                result = InferenceResult(
                    request_id=request.request_id,
                    text=f"Error: {e}",
                    tokens_generated=0,
                    time_taken=time.time() - start_time,
                    preempted=False,
                )

            result_queue.put(result)

        except Empty:
            continue
        except Exception as e:
            print(f"[{worker_id}] Worker error: {e}")

    # Cleanup
    if model:
        try:
            loop.run_until_complete(model.close())
        except Exception:
            pass
    loop.close()


class ParallelModelManager:
    """
    Manages two model inference processes.

    Foreground: User interactions, high priority
    Background: Autonomous tasks, can be preempted
    """

    def __init__(self, config: Dict):
        self.config = config
        self._started = False

        # Queues for each worker
        self.fg_request_queue: Optional[mp.Queue] = None
        self.fg_result_queue: Optional[mp.Queue] = None
        self.bg_request_queue: Optional[mp.Queue] = None
        self.bg_result_queue: Optional[mp.Queue] = None

        # Preemption event for background
        self.bg_preempt: Optional[mp.Event] = None

        # Pending results
        self._pending: Dict[str, asyncio.Future] = {}

        # Workers
        self.fg_worker: Optional[mp.Process] = None
        self.bg_worker: Optional[mp.Process] = None

        # Result collector task
        self._collector_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start both worker processes."""
        if self._started:
            return

        # Create queues
        self.fg_request_queue = mp.Queue()
        self.fg_result_queue = mp.Queue()
        self.bg_request_queue = mp.Queue()
        self.bg_result_queue = mp.Queue()
        self.bg_preempt = mp.Event()

        # Get configs
        fg_config = self.config.get("foreground", self.config.get("primary", {}))
        bg_config = self.config.get("background", fg_config)  # Default to same model

        # Foreground worker
        self.fg_worker = mp.Process(
            target=_model_worker,
            args=(
                fg_config,
                self.fg_request_queue,
                self.fg_result_queue,
                mp.Event(),  # Foreground never preempted
                "FOREGROUND",
            ),
        )
        self.fg_worker.start()

        # Background worker
        self.bg_worker = mp.Process(
            target=_model_worker,
            args=(
                bg_config,
                self.bg_request_queue,
                self.bg_result_queue,
                self.bg_preempt,
                "BACKGROUND",
            ),
        )
        self.bg_worker.start()

        self._started = True

        # Start result collector
        try:
            loop = asyncio.get_running_loop()
            self._collector_task = loop.create_task(self._collect_results())
        except RuntimeError:
            # No running loop - will start collector when needed
            pass

        print("[PARALLEL] Model manager started")

    def stop(self) -> None:
        """Stop worker processes."""
        if not self._started:
            return

        # Cancel collector
        if self._collector_task:
            self._collector_task.cancel()

        # Send shutdown signals
        if self.fg_request_queue:
            self.fg_request_queue.put(None)
        if self.bg_request_queue:
            self.bg_request_queue.put(None)

        # Wait for workers
        if self.fg_worker:
            self.fg_worker.join(timeout=5)
            if self.fg_worker.is_alive():
                self.fg_worker.terminate()

        if self.bg_worker:
            self.bg_worker.join(timeout=5)
            if self.bg_worker.is_alive():
                self.bg_worker.terminate()

        self._started = False
        print("[PARALLEL] Model manager stopped")

    async def generate(
        self,
        prompt: str,
        priority: ModelPriority = ModelPriority.FOREGROUND,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text with specified priority.

        Foreground requests preempt background work.
        """
        if not self._started:
            self.start()

        # Ensure collector is running
        if self._collector_task is None or self._collector_task.done():
            self._collector_task = asyncio.create_task(self._collect_results())

        request_id = str(uuid.uuid4())[:8]

        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            priority=priority,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Create future for result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending[request_id] = future

        # Send to appropriate queue
        if priority == ModelPriority.FOREGROUND:
            # Preempt any background work
            self.bg_preempt.set()
            self.fg_request_queue.put(request)
        else:
            self.bg_request_queue.put(request)

        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=120.0)
            return result.text
        except asyncio.TimeoutError:
            # Clean up
            if request_id in self._pending:
                del self._pending[request_id]
            return "Error: Generation timed out"

    async def generate_foreground(self, prompt: str, **kwargs) -> str:
        """Convenience method for foreground generation."""
        return await self.generate(prompt, ModelPriority.FOREGROUND, **kwargs)

    async def generate_background(self, prompt: str, **kwargs) -> str:
        """Convenience method for background generation."""
        return await self.generate(prompt, ModelPriority.BACKGROUND, **kwargs)

    async def _collect_results(self) -> None:
        """Collect results from both workers."""
        while self._started:
            try:
                # Check foreground results
                try:
                    while True:
                        result = self.fg_result_queue.get_nowait()
                        if result.request_id in self._pending:
                            self._pending[result.request_id].set_result(result)
                            del self._pending[result.request_id]
                except Empty:
                    pass

                # Check background results
                try:
                    while True:
                        result = self.bg_result_queue.get_nowait()
                        if result.request_id in self._pending:
                            self._pending[result.request_id].set_result(result)
                            del self._pending[result.request_id]
                except Empty:
                    pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[PARALLEL] Collector error: {e}")
                await asyncio.sleep(0.1)

    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._started

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "running": self._started,
            "foreground_alive": self.fg_worker.is_alive() if self.fg_worker else False,
            "background_alive": self.bg_worker.is_alive() if self.bg_worker else False,
            "pending_requests": len(self._pending),
        }

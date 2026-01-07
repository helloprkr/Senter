"""
Senter Daemon - Runs 24/7, even when CLI is closed.

Architecture:
+-------------------------------------------------------------+
|                    SENTER DAEMON                            |
|                                                             |
|  +---------------+     +-----------------------------+      |
|  |  TASK QUEUE   |---->|  BACKGROUND WORKER          |      |
|  |  (Priority)   |     |  - Research tasks           |      |
|  +---------------+     |  - File organization        |      |
|         ^              |  - Goal progress            |      |
|         |              +-----------------------------+      |
|         |                                                   |
|  +------+-------+       +-----------------------------+     |
|  |  IPC SERVER  |<----->|  CLI / TUI CLIENT           |     |
|  |  (Unix Socket)|      |  (Connects when active)     |     |
|  +--------------+       +-----------------------------+     |
|                                                             |
+-------------------------------------------------------------+
"""

from __future__ import annotations
import asyncio
import json
import os
import signal
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import heapq


class TaskPriority(Enum):
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass(order=True)
class Task:
    """A background task for autonomous execution."""

    priority: int
    created_at: datetime = field(compare=False)
    task_id: str = field(compare=False)
    task_type: str = field(compare=False)  # research, organize, remind, summarize
    description: str = field(compare=False)
    parameters: Dict[str, Any] = field(compare=False, default_factory=dict)
    status: str = field(compare=False, default="pending")
    result: Optional[str] = field(compare=False, default=None)


class TaskQueue:
    """Priority queue for autonomous tasks."""

    def __init__(self, persist_path: Path):
        self.persist_path = persist_path
        self.tasks: List[Task] = []
        self._load()

    def _load(self) -> None:
        """Load tasks from disk."""
        if self.persist_path.exists():
            try:
                with open(self.persist_path) as f:
                    data = json.load(f)
                    for t in data:
                        task = Task(
                            priority=t["priority"],
                            created_at=datetime.fromisoformat(t["created_at"]),
                            task_id=t["task_id"],
                            task_type=t["task_type"],
                            description=t["description"],
                            parameters=t.get("parameters", {}),
                            status=t["status"],
                            result=t.get("result"),
                        )
                        if task.status == "pending":
                            heapq.heappush(self.tasks, task)
            except Exception as e:
                print(f"Warning: Could not load tasks: {e}")

    def _save(self) -> None:
        """Persist tasks to disk."""
        data = [
            {
                "priority": t.priority,
                "created_at": t.created_at.isoformat(),
                "task_id": t.task_id,
                "task_type": t.task_type,
                "description": t.description,
                "parameters": t.parameters,
                "status": t.status,
                "result": t.result,
            }
            for t in self.tasks
        ]
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, task: Task) -> None:
        """Add task to queue."""
        heapq.heappush(self.tasks, task)
        self._save()

    def pop(self) -> Optional[Task]:
        """Get highest priority task."""
        if self.tasks:
            task = heapq.heappop(self.tasks)
            self._save()
            return task
        return None

    def peek(self) -> Optional[Task]:
        """Look at highest priority task without removing."""
        return self.tasks[0] if self.tasks else None

    def list_all(self) -> List[Task]:
        """Get all pending tasks."""
        return sorted(self.tasks)

    def count(self) -> int:
        """Get number of pending tasks."""
        return len(self.tasks)


class BackgroundWorker:
    """Executes tasks autonomously in the background."""

    def __init__(self, engine, task_queue: TaskQueue):
        self.engine = engine
        self.queue = task_queue
        self.running = False
        self.current_task: Optional[Task] = None
        self.completed_tasks: List[Task] = []
        self._max_completed = 100  # Keep last 100 completed tasks

    async def start(self) -> None:
        """Start the background worker loop."""
        self.running = True
        print("[WORKER] Background worker started")

        while self.running:
            task = self.queue.pop()
            if task:
                self.current_task = task
                await self._execute_task(task)
                self.current_task = None
            else:
                # No tasks, wait before checking again
                await asyncio.sleep(30)

    async def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        print("[WORKER] Background worker stopped")

    async def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        print(f"[WORKER] Executing task: {task.description}")
        task.status = "running"

        try:
            if task.task_type == "research":
                result = await self._do_research(task)
            elif task.task_type == "organize":
                result = await self._do_organize(task)
            elif task.task_type == "remind":
                result = await self._do_remind(task)
            elif task.task_type == "summarize":
                result = await self._do_summarize(task)
            else:
                result = f"Unknown task type: {task.task_type}"

            task.status = "completed"
            task.result = result

        except Exception as e:
            task.status = "failed"
            task.result = str(e)

        self.completed_tasks.append(task)

        # Trim completed tasks list
        if len(self.completed_tasks) > self._max_completed:
            self.completed_tasks = self.completed_tasks[-self._max_completed :]

        print(f"[WORKER] Task completed: {task.task_id} - {task.status}")

    async def _do_research(self, task: Task) -> str:
        """Research a topic using web search."""
        query = task.parameters.get("query", task.description)

        try:
            # Use web search if available
            from tools.web_search import WebSearch

            searcher = WebSearch()
            results = await searcher.search(query, max_results=5)

            # Summarize with LLM if available
            if self.engine and self.engine.model:
                summary_prompt = f"""Summarize these research findings about "{query}":

{json.dumps(results, indent=2)}

Provide a concise summary with key points."""

                summary = await self.engine.model.generate(summary_prompt)

                # Store in memory
                if hasattr(self.engine, "memory"):
                    self.engine.memory.semantic.store(
                        content=f"Research on {query}: {summary}",
                        domain="research",
                    )

                return summary
            else:
                return f"Research results: {json.dumps(results[:3], indent=2)}"

        except Exception as e:
            return f"Research failed: {e}"

    async def _do_organize(self, task: Task) -> str:
        """Organize files in a directory."""
        path = Path(task.parameters.get("path", "."))
        return f"File organization for {path} not yet implemented"

    async def _do_remind(self, task: Task) -> str:
        """Set up a reminder."""
        message = task.parameters.get("message", task.description)
        # Store as high-priority semantic memory
        if hasattr(self.engine, "memory"):
            self.engine.memory.semantic.store(
                content=f"REMINDER: {message}",
                domain="reminders",
            )
        return f"Reminder set: {message}"

    async def _do_summarize(self, task: Task) -> str:
        """Summarize recent interactions."""
        if not hasattr(self.engine, "memory"):
            return "No memory system available"

        # Get recent episodes
        episodes = self.engine.memory.episodic

        if not episodes:
            return "No recent interactions to summarize"

        # Build summary from episodes
        summary_parts = []
        for ep in episodes[-20:]:
            summary_parts.append(f"- {ep.input[:100]}...")

        summary = f"Summary of {len(episodes)} recent interactions:\n" + "\n".join(
            summary_parts[:10]
        )

        return summary


class SenterDaemon:
    """
    The main daemon process.

    Runs 24/7, manages task queue, handles IPC with CLI.
    """

    def __init__(self, genome_path: Path):
        self.genome_path = Path(genome_path)
        self.data_dir = self.genome_path.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.socket_path = self.data_dir / "senter.sock"
        self.pid_file = self.data_dir / "senter.pid"

        self.task_queue = TaskQueue(self.data_dir / "task_queue.json")
        self.engine = None
        self.worker = None
        self.server = None
        self.running = False

    async def start(self) -> None:
        """Start the daemon."""
        # Write PID file
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))

        # Initialize Senter engine
        from core.engine import Senter

        self.engine = Senter(self.genome_path)
        await self.engine.initialize()

        # Start background worker
        self.worker = BackgroundWorker(self.engine, self.task_queue)
        worker_task = asyncio.create_task(self.worker.start())

        # Start IPC server
        self.running = True

        # Remove stale socket file
        if self.socket_path.exists():
            self.socket_path.unlink()

        self.server = await asyncio.start_unix_server(
            self._handle_client, path=str(self.socket_path)
        )

        print(f"[DAEMON] Started. Socket: {self.socket_path}")
        print(f"[DAEMON] PID: {os.getpid()}")

        # Handle signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        async with self.server:
            await self.server.serve_forever()

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        print("[DAEMON] Shutting down...")
        self.running = False

        if self.worker:
            await self.worker.stop()

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        if self.engine:
            await self.engine.shutdown()

        # Cleanup
        if self.socket_path.exists():
            self.socket_path.unlink()
        if self.pid_file.exists():
            self.pid_file.unlink()

        print("[DAEMON] Stopped.")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle IPC requests from CLI."""
        addr = writer.get_extra_info("peername")
        print(f"[DAEMON] Client connected")

        try:
            while True:
                data = await reader.readline()
                if not data:
                    break

                try:
                    request = json.loads(data.decode())
                    response = await self._process_request(request)
                except json.JSONDecodeError as e:
                    response = {"status": "error", "message": f"Invalid JSON: {e}"}

                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
        except ConnectionResetError:
            pass
        except Exception as e:
            print(f"[DAEMON] Client error: {e}")
        finally:
            print(f"[DAEMON] Client disconnected")
            writer.close()
            await writer.wait_closed()

    async def _process_request(self, request: Dict) -> Dict:
        """Process a request from CLI."""
        action = request.get("action")

        if action == "interact":
            # Foreground interaction
            response = await self.engine.interact(request["input"])
            return {
                "status": "ok",
                "response": response.text,
                "ai_state": {
                    "focus": response.ai_state.focus,
                    "mode": response.ai_state.mode,
                    "trust": response.ai_state.trust_level,
                },
                "episode_id": response.episode_id,
            }

        elif action == "add_task":
            # Add background task
            task = Task(
                priority=TaskPriority[request.get("priority", "NORMAL")].value,
                created_at=datetime.now(),
                task_id=request.get("task_id", str(uuid.uuid4())[:8]),
                task_type=request["task_type"],
                description=request["description"],
                parameters=request.get("parameters", {}),
            )
            self.task_queue.add(task)
            return {"status": "ok", "task_id": task.task_id}

        elif action == "list_tasks":
            tasks = self.task_queue.list_all()
            return {
                "status": "ok",
                "tasks": [
                    {"id": t.task_id, "type": t.task_type, "desc": t.description}
                    for t in tasks
                ],
            }

        elif action == "completed_tasks":
            return {
                "status": "ok",
                "tasks": [
                    {
                        "id": t.task_id,
                        "type": t.task_type,
                        "desc": t.description,
                        "result": t.result,
                    }
                    for t in self.worker.completed_tasks[-10:]
                ]
                if self.worker
                else [],
            }

        elif action == "status":
            return {
                "status": "ok",
                "running": self.running,
                "pending_tasks": self.task_queue.count(),
                "current_task": self.worker.current_task.description
                if self.worker and self.worker.current_task
                else None,
                "completed_tasks": len(self.worker.completed_tasks)
                if self.worker
                else 0,
                "trust_level": self.engine.trust.level if self.engine else 0.5,
                "memory_episodes": len(self.engine.memory.episodic)
                if self.engine
                else 0,
            }

        elif action == "shutdown":
            asyncio.create_task(self.stop())
            return {"status": "ok", "message": "Shutting down"}

        return {"status": "error", "message": f"Unknown action: {action}"}


async def main():
    import sys

    genome_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("genome.yaml")
    daemon = SenterDaemon(genome_path)
    await daemon.start()


if __name__ == "__main__":
    asyncio.run(main())

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
from datetime import datetime, timedelta
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


@dataclass
class GoalResearchResult:
    """Result of goal-based research."""
    goal_id: str
    goal_description: str
    research_summary: str
    sources: List[str]
    completed_at: datetime
    stored_in_memory: bool = False


class BackgroundWorker:
    """Executes tasks autonomously in the background."""

    def __init__(self, engine, task_queue: TaskQueue):
        self.engine = engine
        self.queue = task_queue
        self.running = False
        self.current_task: Optional[Task] = None
        self.completed_tasks: List[Task] = []
        self._max_completed = 100  # Keep last 100 completed tasks
        self.goal_research_results: List[GoalResearchResult] = []
        self._max_research_results = 50  # Keep last 50 research results

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

    async def auto_research_learning_goals(self) -> List[GoalResearchResult]:
        """
        Automatically start research for learning-type goals.

        Returns list of research results that were completed.
        """
        results = []

        if not self.engine:
            return results

        # Get goal detector from engine if available
        goal_detector = getattr(self.engine, "goal_detector", None)
        if not goal_detector:
            return results

        # Get proactive engine for task creation
        proactive = getattr(self.engine, "proactive", None)
        if not proactive:
            return results

        # Get learning goals
        goals = goal_detector.get_active_goals()
        learning_goals = [g for g in goals if g.category == "learning"]

        for goal in learning_goals:
            # Check if we should research this goal (using proactive engine's cooldown)
            if not proactive._should_create_task_for_goal(goal.id):
                continue

            # Create and execute research task
            task = Task(
                priority=TaskPriority.BACKGROUND.value,
                created_at=datetime.now(),
                task_id=f"goal_research_{goal.id}_{int(datetime.now().timestamp())}",
                task_type="goal_research",
                description=f"Research for learning goal: {goal.description}",
                parameters={
                    "goal_id": goal.id,
                    "goal_description": goal.description,
                    "query": goal.description,
                },
            )

            result = await self._do_goal_research(task)

            if result:
                results.append(result)
                proactive.created_task_ids[goal.id] = datetime.now()

        return results

    async def _do_goal_research(self, task: Task) -> Optional[GoalResearchResult]:
        """Execute goal-specific research and store in memory."""
        goal_id = task.parameters.get("goal_id", "")
        goal_desc = task.parameters.get("goal_description", task.description)
        query = task.parameters.get("query", goal_desc)

        sources = []
        summary = ""

        try:
            # Attempt web search if available
            try:
                from tools.web_search import WebSearch
                searcher = WebSearch()
                search_results = await searcher.search(query, max_results=5)
                sources = [r.get("url", "") for r in search_results if r.get("url")]
            except ImportError:
                # Web search not available, create synthetic result
                search_results = []

            # Summarize with LLM if available
            if self.engine and self.engine.model:
                prompt = f"""Research the topic: "{query}"

Based on available information, provide a helpful summary that would assist someone learning about this topic.

Include:
1. Key concepts to understand
2. Recommended starting points
3. Common challenges and how to overcome them

Keep the summary concise but informative."""

                summary = await self.engine.model.generate(prompt)
            else:
                summary = f"Research queued for: {query}. LLM not available for summarization."

            # Store in semantic memory linked to goal
            stored = False
            if hasattr(self.engine, "memory") and hasattr(self.engine.memory, "semantic"):
                self.engine.memory.semantic.store(
                    content=f"[Goal Research: {goal_desc}]\n\n{summary}",
                    domain="goal_research",
                )
                stored = True

            result = GoalResearchResult(
                goal_id=goal_id,
                goal_description=goal_desc,
                research_summary=summary,
                sources=sources,
                completed_at=datetime.now(),
                stored_in_memory=stored,
            )

            self.goal_research_results.append(result)

            # Trim results list
            if len(self.goal_research_results) > self._max_research_results:
                self.goal_research_results = self.goal_research_results[-self._max_research_results:]

            return result

        except Exception as e:
            print(f"[WORKER] Goal research failed: {e}")
            return None

    def get_while_you_were_away_summary(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get summary of completed work while user was away.

        Returns a dictionary with:
        - completed_tasks: List of tasks completed
        - goal_research: List of goal research completed
        - summary: Human-readable summary
        """
        if since is None:
            # Default to last 24 hours
            since = datetime.now() - timedelta(hours=24)

        # Filter completed tasks
        recent_tasks = [
            t for t in self.completed_tasks
            if t.created_at >= since
        ]

        # Filter goal research
        recent_research = [
            r for r in self.goal_research_results
            if r.completed_at >= since
        ]

        # Build summary
        summary_parts = []

        if recent_tasks:
            summary_parts.append(f"Completed {len(recent_tasks)} background tasks:")
            for t in recent_tasks[:5]:
                summary_parts.append(f"  - {t.description} ({t.status})")

        if recent_research:
            summary_parts.append(f"\nResearched {len(recent_research)} learning goals:")
            for r in recent_research[:5]:
                summary_parts.append(f"  - {r.goal_description}")
                if r.stored_in_memory:
                    summary_parts.append("    (stored in memory)")

        if not summary_parts:
            summary_parts.append("No background work completed during this period.")

        return {
            "completed_tasks": [
                {"id": t.task_id, "desc": t.description, "status": t.status, "result": t.result}
                for t in recent_tasks
            ],
            "goal_research": [
                {
                    "goal_id": r.goal_id,
                    "goal": r.goal_description,
                    "summary": r.research_summary[:200] + "..." if len(r.research_summary) > 200 else r.research_summary,
                    "stored": r.stored_in_memory,
                }
                for r in recent_research
            ],
            "summary": "\n".join(summary_parts),
            "since": since.isoformat(),
        }


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

        # ActivityMonitor integration (US-018)
        self.activity_monitor = None
        self.activity_capture_task = None
        self.activity_capture_interval = 60  # seconds

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
        asyncio.create_task(self.worker.start())

        # Initialize and start ActivityMonitor (US-018)
        try:
            from intelligence.activity import ActivityMonitor
            self.activity_monitor = ActivityMonitor()
            self.activity_capture_task = asyncio.create_task(self._run_activity_capture())
            print("[DAEMON] ActivityMonitor started")
        except ImportError as e:
            print(f"[DAEMON] ActivityMonitor not available: {e}")

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

    async def _run_activity_capture(self) -> None:
        """
        Run activity capture loop (US-018).

        Captures activity snapshots every activity_capture_interval seconds.
        """
        while self.running:
            try:
                if self.activity_monitor:
                    await self.activity_monitor.capture_current_activity()
            except Exception as e:
                print(f"[DAEMON] Activity capture error: {e}")

            await asyncio.sleep(self.activity_capture_interval)

    def get_activity_summary(self) -> Dict[str, Any]:
        """
        Get current activity summary (US-018).

        Returns:
            Activity summary dictionary
        """
        if not self.activity_monitor:
            return {
                "status": "unavailable",
                "message": "ActivityMonitor not initialized"
            }

        return {
            "status": "ok",
            "current_context": self.activity_monitor.get_current_context(),
            "summary": self.activity_monitor.get_activity_summary(),
            "snapshot_count": len(self.activity_monitor.snapshots),
        }

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        print("[DAEMON] Shutting down...")
        self.running = False

        # Stop activity capture task
        if self.activity_capture_task:
            self.activity_capture_task.cancel()
            try:
                await self.activity_capture_task
            except asyncio.CancelledError:
                pass
            print("[DAEMON] ActivityMonitor stopped")

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
        print("[DAEMON] Client connected")

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
            print("[DAEMON] Client disconnected")
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

        elif action == "activity":
            # Return activity monitoring summary (US-018)
            return self.get_activity_summary()

        return {"status": "error", "message": f"Unknown action: {action}"}


async def main():
    import sys

    genome_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("genome.yaml")
    daemon = SenterDaemon(genome_path)
    await daemon.start()


if __name__ == "__main__":
    asyncio.run(main())

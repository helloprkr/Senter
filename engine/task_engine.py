#!/usr/bin/env python3
"""
Task Execution Engine

The brain of Senter's autonomous operation.
Takes goals and executes them without user intervention.

Pipeline:
1. Goal received (from user or scheduler)
2. Planner breaks goal into tasks
3. Executor runs each task
4. Progress reported back
"""

import json
import time
import logging
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from pathlib import Path
from multiprocessing import Event
sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon.message_bus import MessageBus, MessageType, Message
from queue import Empty

logger = logging.getLogger('senter.task_engine')


class TaskStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    RESEARCH = "research"
    GENERATE = "generate"
    ANALYZE = "analyze"
    COMMUNICATE = "communicate"
    ORGANIZE = "organize"
    CUSTOM = "custom"


@dataclass
class Task:
    """Individual executable task"""
    id: str
    goal_id: str
    description: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5

    # Execution details
    tool: Optional[str] = None
    tool_params: dict = field(default_factory=dict)

    # Dependencies
    depends_on: list[str] = field(default_factory=list)

    # Results
    result: Optional[Any] = None
    error: Optional[str] = None

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "description": self.description,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "priority": self.priority,
            "tool": self.tool,
            "tool_params": self.tool_params,
            "depends_on": self.depends_on,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = d.copy()
        d["task_type"] = TaskType(d["task_type"])
        d["status"] = TaskStatus(d["status"])
        return cls(**d)


@dataclass
class ExecutionPlan:
    """Plan for executing a goal"""
    goal_id: str
    goal_description: str
    tasks: list[Task]
    created_at: float = field(default_factory=time.time)

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks ready to execute (dependencies met)"""
        completed_ids = {
            t.id for t in self.tasks
            if t.status == TaskStatus.COMPLETED
        }

        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.depends_on)
        ]

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "goal_description": self.goal_description,
            "tasks": [t.to_dict() for t in self.tasks],
            "created_at": self.created_at
        }


class TaskEngine:
    """
    Main task execution engine.

    Responsibilities:
    - Receive goals from message bus
    - Create execution plans
    - Execute tasks
    - Report progress
    """

    def __init__(
        self,
        max_concurrent: int,
        message_bus: MessageBus,
        shutdown_event: Event,
        senter_root: Path
    ):
        self.max_concurrent = max_concurrent
        self.message_bus = message_bus
        self.shutdown_event = shutdown_event
        self.senter_root = Path(senter_root)

        # Active plans and tasks
        self.plans: dict[str, ExecutionPlan] = {}
        self.running_tasks: dict[str, Task] = {}

        # Components
        self.planner = TaskPlanner(message_bus, senter_root)
        self.executor = TaskExecutor(message_bus, senter_root)

        # Persistence
        self.state_file = self.senter_root / "data" / "task_engine_state.json"

        # Message queue
        self._queue = None

    def run(self):
        """Main engine loop"""
        logger.info("Task engine starting...")

        # Load saved state
        self._load_state()

        # Register with message bus
        self._queue = self.message_bus.register("task_engine")

        logger.info("Task engine started")

        while not self.shutdown_event.is_set():
            try:
                # Check for new messages
                self._process_messages()

                # Execute ready tasks
                self._execute_ready_tasks()

                # Small sleep
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Task engine error: {e}")

        self._save_state()
        logger.info("Task engine stopped")

    def _process_messages(self):
        """Process incoming messages"""
        try:
            while True:
                msg_dict = self._queue.get_nowait()
                message = Message.from_dict(msg_dict)
                self._handle_message(message)
        except Empty:
            pass
        except Exception as e:
            logger.error(f"Message processing error: {e}")

    def _handle_message(self, message: Message):
        """Handle a single message"""
        payload = message.payload

        if message.type == MessageType.TASK_CREATE:
            # New goal to execute
            goal_id = payload.get("goal_id") or str(uuid.uuid4())[:8]
            goal_desc = payload.get("description", "")
            self._create_plan(goal_id, goal_desc)

        elif message.type == MessageType.GOAL_DETECTED:
            # CG-008: Goal detected, create execution plan
            goal_id = payload.get("goal_id") or str(uuid.uuid4())[:8]
            goal_desc = payload.get("description", "")
            logger.info(f"Goal detected: {goal_id} - {goal_desc[:50]}")
            self._create_plan(goal_id, goal_desc)

        elif message.type == MessageType.JOB_TRIGGERED:
            # Scheduled job triggered
            job_id = payload.get("job_id")
            self._handle_scheduled_job(job_id, payload)

        elif message.type == MessageType.TASK_UPDATE:
            # External update to a task
            task_id = payload.get("task_id")
            status = payload.get("status")
            self._update_task_status(task_id, status, payload)

    def _create_plan(self, goal_id: str, goal_description: str):
        """Create an execution plan for a goal"""
        logger.info(f"Creating plan for goal: {goal_description}")

        # Use planner to break down goal
        plan = self.planner.create_plan(goal_id, goal_description)
        self.plans[goal_id] = plan

        # Log activity
        self._log_activity("plan_created", {
            "goal_id": goal_id,
            "description": goal_description,
            "task_count": len(plan.tasks)
        })

        # Notify progress
        self.message_bus.send(
            MessageType.TASK_PROGRESS,
            source="task_engine",
            payload={
                "goal_id": goal_id,
                "status": "planning_complete",
                "tasks": len(plan.tasks)
            }
        )

        logger.info(f"Plan created with {len(plan.tasks)} tasks")

    def _execute_ready_tasks(self):
        """Execute tasks that are ready"""
        # Check capacity
        available_slots = self.max_concurrent - len(self.running_tasks)
        if available_slots <= 0:
            return

        # Get ready tasks from all plans
        ready_tasks = []
        for plan in self.plans.values():
            ready_tasks.extend(plan.get_ready_tasks())

        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)

        # Execute up to available slots
        for task in ready_tasks[:available_slots]:
            self._execute_task(task)

    def _execute_task(self, task: Task):
        """Execute a single task"""
        logger.info(f"Executing task: {task.description}")

        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()
        self.running_tasks[task.id] = task

        try:
            # Execute via executor
            result = self.executor.execute(task)

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()

            duration = task.completed_at - task.started_at
            logger.info(f"Task completed: {task.description} ({duration:.1f}s)")

            # Log activity
            self._log_activity("task_completed", {
                "task_id": task.id,
                "goal_id": task.goal_id,
                "description": task.description,
                "duration": duration
            })

            # Notify
            self.message_bus.send(
                MessageType.TASK_COMPLETE,
                source="task_engine",
                payload={
                    "task_id": task.id,
                    "goal_id": task.goal_id,
                    "description": task.description,
                    "result": str(result)[:200] if result else None
                }
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()

            logger.error(f"Task failed: {task.description} - {e}")

            self._log_activity("task_failed", {
                "task_id": task.id,
                "goal_id": task.goal_id,
                "description": task.description,
                "error": str(e)
            })

        finally:
            del self.running_tasks[task.id]
            self._check_plan_completion(task.goal_id)

    def _check_plan_completion(self, goal_id: str):
        """Check if a plan is complete (CG-008: sends GOAL_COMPLETE)"""
        plan = self.plans.get(goal_id)
        if not plan:
            return

        all_done = all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            for t in plan.tasks
        )

        if all_done:
            completed = sum(1 for t in plan.tasks if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)

            # Determine overall status
            if failed == 0:
                status = "success"
            elif completed > 0:
                status = "partial"
            else:
                status = "failed"

            logger.info(
                f"Goal complete: {goal_id} - "
                f"{completed} succeeded, {failed} failed"
            )

            self._log_activity("goal_completed", {
                "goal_id": goal_id,
                "description": plan.goal_description,
                "tasks_completed": completed,
                "tasks_failed": failed
            })

            # CG-008: Send GOAL_COMPLETE notification
            self.message_bus.send(
                MessageType.GOAL_COMPLETE,
                source="task_engine",
                payload={
                    "goal_id": goal_id,
                    "description": plan.goal_description,
                    "status": status,
                    "tasks_completed": completed,
                    "tasks_failed": failed,
                    "total_tasks": len(plan.tasks),
                    "results": [
                        {
                            "task_id": t.id,
                            "description": t.description,
                            "status": t.status.value,
                            "result_summary": str(t.result)[:200] if t.result else None
                        }
                        for t in plan.tasks
                    ]
                }
            )

            # Clean up
            del self.plans[goal_id]

    def _handle_scheduled_job(self, job_id: str, payload: dict):
        """Handle a triggered scheduled job"""
        job_type = payload.get("job_type")
        description = payload.get("description", "Scheduled task")

        self._create_plan(
            goal_id=f"job_{job_id}_{int(time.time())}",
            goal_description=description
        )

    def _log_activity(self, activity_type: str, details: dict):
        """Log activity to reporter"""
        self.message_bus.send(
            MessageType.ACTIVITY_LOG,
            source="task_engine",
            payload={
                "activity_type": activity_type,
                "details": details,
                "timestamp": time.time()
            }
        )

    def _update_task_status(self, task_id: str, status: str, payload: dict):
        """Update task status from external source"""
        for plan in self.plans.values():
            for task in plan.tasks:
                if task.id == task_id:
                    task.status = TaskStatus(status)
                    if "result" in payload:
                        task.result = payload["result"]
                    return

    def _load_state(self):
        """Load persisted state"""
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text())
                for plan_data in state.get("plans", []):
                    plan = ExecutionPlan(
                        goal_id=plan_data["goal_id"],
                        goal_description=plan_data["goal_description"],
                        tasks=[Task.from_dict(t) for t in plan_data["tasks"]],
                        created_at=plan_data["created_at"]
                    )
                    self.plans[plan.goal_id] = plan
                logger.info(f"Loaded {len(self.plans)} plans from state")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        """Save state for persistence"""
        state = {
            "plans": [p.to_dict() for p in self.plans.values()]
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2))


class TaskPlanner:
    """
    Breaks goals into executable tasks.
    Uses LLM to understand goal and create task sequence.
    """

    def __init__(self, message_bus: MessageBus, senter_root: Path):
        self.message_bus = message_bus
        self.senter_root = Path(senter_root)

    def create_plan(self, goal_id: str, goal_description: str) -> ExecutionPlan:
        """Create an execution plan for a goal"""
        tasks = self._generate_tasks(goal_id, goal_description)

        return ExecutionPlan(
            goal_id=goal_id,
            goal_description=goal_description,
            tasks=tasks
        )

    def _generate_tasks(self, goal_id: str, goal_description: str) -> list[Task]:
        """Generate tasks for a goal using heuristics"""
        tasks = []
        desc_lower = goal_description.lower()

        # Research-type goals
        if any(kw in desc_lower for kw in ["research", "find", "search", "learn about", "what is", "who is"]):
            tasks.append(Task(
                id=f"{goal_id}_research",
                goal_id=goal_id,
                description=f"Search for information about: {goal_description}",
                task_type=TaskType.RESEARCH,
                tool="web_search",
                tool_params={"query": goal_description}
            ))
            tasks.append(Task(
                id=f"{goal_id}_summarize",
                goal_id=goal_id,
                description="Summarize research findings",
                task_type=TaskType.GENERATE,
                depends_on=[f"{goal_id}_research"]
            ))

        # Writing-type goals
        elif any(kw in desc_lower for kw in ["write", "draft", "create", "compose"]):
            tasks.append(Task(
                id=f"{goal_id}_outline",
                goal_id=goal_id,
                description=f"Create outline for: {goal_description}",
                task_type=TaskType.GENERATE
            ))
            tasks.append(Task(
                id=f"{goal_id}_draft",
                goal_id=goal_id,
                description="Write first draft",
                task_type=TaskType.GENERATE,
                depends_on=[f"{goal_id}_outline"]
            ))
            tasks.append(Task(
                id=f"{goal_id}_review",
                goal_id=goal_id,
                description="Review and refine draft",
                task_type=TaskType.ANALYZE,
                depends_on=[f"{goal_id}_draft"]
            ))

        # Analysis-type goals
        elif any(kw in desc_lower for kw in ["analyze", "compare", "evaluate", "review"]):
            tasks.append(Task(
                id=f"{goal_id}_gather",
                goal_id=goal_id,
                description="Gather relevant information",
                task_type=TaskType.RESEARCH
            ))
            tasks.append(Task(
                id=f"{goal_id}_analyze",
                goal_id=goal_id,
                description=f"Analyze: {goal_description}",
                task_type=TaskType.ANALYZE,
                depends_on=[f"{goal_id}_gather"]
            ))
            tasks.append(Task(
                id=f"{goal_id}_report",
                goal_id=goal_id,
                description="Generate analysis report",
                task_type=TaskType.GENERATE,
                depends_on=[f"{goal_id}_analyze"]
            ))

        # Default: single task
        else:
            tasks.append(Task(
                id=f"{goal_id}_main",
                goal_id=goal_id,
                description=goal_description,
                task_type=TaskType.CUSTOM
            ))

        return tasks


class TaskExecutor:
    """
    Executes individual tasks.
    Maps task types to execution strategies.
    """

    # Default timeout for LLM execution
    LLM_TIMEOUT_SECONDS = 120

    def __init__(self, message_bus: MessageBus, senter_root: Path):
        self.message_bus = message_bus
        self.senter_root = Path(senter_root)

        # Register with message bus to receive responses
        self._response_queue = self.message_bus.register("task_executor")

        # Pending responses by correlation_id
        self._pending_responses: dict[str, dict] = {}

        # Task result storage (US-003)
        from engine.task_results import TaskResultStorage
        results_dir = self.senter_root / "data" / "tasks" / "results"
        self.result_storage = TaskResultStorage(results_dir)

        # Tool registry
        self.tools = {
            "web_search": self._execute_web_search,
            "file_write": self._execute_file_write,
            "file_read": self._execute_file_read,
        }

    def execute(self, task: Task) -> Any:
        """Execute a task and return result"""
        if task.tool and task.tool in self.tools:
            result = self.tools[task.tool](task)
        else:
            # Default: use LLM
            result = self._execute_with_llm(task)

        # Store result (US-003)
        self._store_result(task, result)

        return result

    def _store_result(self, task: Task, result: dict):
        """Store task result to persistent storage (US-003)"""
        from engine.task_results import TaskResult

        try:
            # Extract the main content from result
            # Different tools return different formats
            if isinstance(result, dict):
                # LLM results have "response", file ops have "content", etc.
                content = (result.get("response") or
                          result.get("content") or
                          result.get("results") or
                          result)
                status = result.get("status", "completed")
                worker = result.get("worker", "unknown")
                latency_ms = result.get("latency_ms", 0)
            else:
                content = result
                status = "completed"
                worker = "unknown"
                latency_ms = 0

            task_result = TaskResult(
                task_id=task.id,
                goal_id=task.goal_id,
                result=content,
                status=status,
                worker=worker,
                latency_ms=latency_ms,
                description=task.description
            )
            self.result_storage.store(task_result)
        except Exception as e:
            logger.warning(f"Failed to store result for task {task.id}: {e}")

    def _execute_web_search(self, task: Task) -> dict:
        """Execute web search task"""
        try:
            sys.path.insert(0, str(self.senter_root / "Functions"))
            from web_search import search_web
            query = task.tool_params.get("query", task.description)
            results = search_web(query, max_results=5)
            return {"query": query, "results": results, "count": len(results)}
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return {"query": task.description, "results": [], "error": str(e)}

    def _execute_file_write(self, task: Task) -> dict:
        """Execute file write task"""
        path = task.tool_params.get("path")
        content = task.tool_params.get("content")

        if not path or not content:
            raise ValueError("File write requires 'path' and 'content'")

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

        return {"path": str(path), "bytes_written": len(content)}

    def _execute_file_read(self, task: Task) -> dict:
        """Execute file read task"""
        path = task.tool_params.get("path")

        if not path:
            raise ValueError("File read requires 'path'")

        content = Path(path).read_text()
        return {"path": str(path), "content": content}

    def _execute_with_llm(self, task: Task) -> dict:
        """Execute task using LLM via message bus.

        Sends request to model worker and waits for response.
        Timeout: 120 seconds (LLM_TIMEOUT_SECONDS).
        """
        correlation_id = str(uuid.uuid4())

        # Build task prompt based on task type
        if task.task_type == TaskType.RESEARCH:
            system_prompt = (
                "You are a research assistant. Gather comprehensive information "
                "and provide detailed, factual responses."
            )
        elif task.task_type == TaskType.GENERATE:
            system_prompt = (
                "You are a content generator. Create high-quality, well-structured "
                "content based on the given instructions."
            )
        elif task.task_type == TaskType.ANALYZE:
            system_prompt = (
                "You are an analyst. Provide thorough analysis with clear reasoning "
                "and actionable insights."
            )
        else:
            system_prompt = "You are a helpful AI assistant. Complete the given task."

        # Send request to model worker
        self.message_bus.send(
            MessageType.MODEL_REQUEST,
            source="task_executor",
            target="model_research",
            payload={
                "prompt": f"Complete this task: {task.description}",
                "system_prompt": system_prompt,
                "max_tokens": 2048
            },
            correlation_id=correlation_id
        )

        logger.info(f"Task {task.id}: Request sent, waiting for response...")

        # Wait for response with timeout
        start_time = time.time()
        timeout = self.LLM_TIMEOUT_SECONDS

        while time.time() - start_time < timeout:
            try:
                # Check for messages with short timeout
                msg_dict = self._response_queue.get(timeout=0.5)
                message = Message.from_dict(msg_dict)

                # Check if this is our response
                if (message.type == MessageType.MODEL_RESPONSE and
                    message.correlation_id == correlation_id):

                    response_text = message.payload.get("response", "")
                    latency_ms = message.payload.get("latency_ms", 0)
                    worker = message.payload.get("worker", "unknown")

                    logger.info(
                        f"Task {task.id}: Response received from {worker} "
                        f"({latency_ms}ms)"
                    )

                    # Store result in task object
                    task.result = {
                        "status": "completed",
                        "response": response_text,
                        "worker": worker,
                        "latency_ms": latency_ms,
                        "correlation_id": correlation_id
                    }

                    return task.result

                # Check for error response
                elif (message.type == MessageType.ERROR and
                      message.correlation_id == correlation_id):

                    error_msg = message.payload.get("error", "Unknown error")
                    logger.error(f"Task {task.id}: LLM error - {error_msg}")

                    task.result = {
                        "status": "error",
                        "error": error_msg,
                        "correlation_id": correlation_id
                    }
                    raise RuntimeError(f"LLM execution failed: {error_msg}")

                # Not our message - could cache for other requests
                else:
                    # Store for potential other consumers
                    if message.correlation_id:
                        self._pending_responses[message.correlation_id] = message.payload

            except Empty:
                # No message yet, continue waiting
                continue

        # Timeout reached
        elapsed = time.time() - start_time
        logger.error(f"Task {task.id}: Timeout after {elapsed:.1f}s")

        task.result = {
            "status": "timeout",
            "error": f"LLM execution timed out after {timeout}s",
            "correlation_id": correlation_id
        }
        raise TimeoutError(f"Task execution timed out after {timeout} seconds")


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from daemon.message_bus import MessageBus

    bus = MessageBus()
    bus.start()

    shutdown = Event()

    engine = TaskEngine(
        max_concurrent=3,
        message_bus=bus,
        shutdown_event=shutdown,
        senter_root=Path(__file__).parent.parent
    )

    # Test planning
    planner = TaskPlanner(bus, Path(__file__).parent.parent)
    plan = planner.create_plan("test1", "Research the latest AI trends for 2026")

    print(f"\nPlan created with {len(plan.tasks)} tasks:")
    for task in plan.tasks:
        print(f"  - {task.description} ({task.task_type.value})")
        if task.depends_on:
            print(f"    depends on: {task.depends_on}")

    bus.stop()
    print("\nTest complete")

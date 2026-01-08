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
import threading
import importlib
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Callable, Dict, List
from pathlib import Path
from multiprocessing import Event
sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon.message_bus import MessageBus, MessageType, Message
from queue import Empty

logger = logging.getLogger('senter.task_engine')


# ========== TE-003: Retry Configuration ==========
DEFAULT_MAX_RETRIES = 5
RETRY_DELAYS = [1, 2, 4, 8, 16]  # Exponential backoff delays in seconds


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
    """Individual executable task (TE-003, TE-004)"""
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

    # Retry tracking (TE-003)
    retry_count: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES
    next_retry_at: Optional[float] = None

    # Cancellation (TE-004)
    cancelled: bool = False
    cancel_requested_at: Optional[float] = None

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
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "next_retry_at": self.next_retry_at,
            "cancelled": self.cancelled,
            "cancel_requested_at": self.cancel_requested_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = d.copy()
        d["task_type"] = TaskType(d["task_type"])
        d["status"] = TaskStatus(d["status"])
        # Handle optional fields that might not exist in old data
        d.setdefault("retry_count", 0)
        d.setdefault("max_retries", DEFAULT_MAX_RETRIES)
        d.setdefault("next_retry_at", None)
        d.setdefault("cancelled", False)
        d.setdefault("cancel_requested_at", None)
        return cls(**d)

    def request_cancel(self) -> bool:
        """Request cancellation of this task (TE-004)"""
        if self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False
        self.cancelled = True
        self.cancel_requested_at = time.time()
        return True

    def should_retry(self) -> bool:
        """Check if task should be retried (TE-003)"""
        if self.cancelled:
            return False
        if self.status != TaskStatus.FAILED:
            return False
        return self.retry_count < self.max_retries

    def schedule_retry(self) -> float:
        """Schedule retry with exponential backoff (TE-003)"""
        delay = RETRY_DELAYS[min(self.retry_count, len(RETRY_DELAYS) - 1)]
        self.next_retry_at = time.time() + delay
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.error = None
        return delay


@dataclass
class ExecutionPlan:
    """Plan for executing a goal"""
    goal_id: str
    goal_description: str
    tasks: list[Task]
    created_at: float = field(default_factory=time.time)

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks ready to execute (dependencies met) (TE-003, TE-004)"""
        completed_ids = {
            t.id for t in self.tasks
            if t.status == TaskStatus.COMPLETED
        }

        now = time.time()
        ready = []

        for t in self.tasks:
            # Skip cancelled tasks (TE-004)
            if t.cancelled:
                continue

            # Only pending tasks
            if t.status != TaskStatus.PENDING:
                continue

            # Check retry timing (TE-003)
            if t.next_retry_at and now < t.next_retry_at:
                continue

            # Check dependencies
            if not all(dep in completed_ids for dep in t.depends_on):
                continue

            ready.append(t)

        return ready

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "goal_description": self.goal_description,
            "tasks": [t.to_dict() for t in self.tasks],
            "created_at": self.created_at
        }


# ========== TE-002: Tool Registry ==========

@dataclass
class ToolInfo:
    """Registered tool metadata (TE-002)"""
    name: str
    description: str
    parameters: Dict[str, Any]
    return_type: str = "Any"
    source: str = "builtin"  # builtin, functions, mcp
    handler: Optional[Callable] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "source": self.source
        }


class ToolRegistry:
    """
    Centralized tool registry with dynamic discovery (TE-002).

    Discovers and registers tools from:
    - Built-in tools (file_read, file_write, etc.)
    - Functions/ directory
    - MCP servers
    """

    _instance: Optional["ToolRegistry"] = None

    def __init__(self, functions_dir: Optional[Path] = None):
        self._tools: Dict[str, ToolInfo] = {}
        self._lock = threading.Lock()
        self.functions_dir = functions_dir

    @classmethod
    def get_instance(cls, functions_dir: Optional[Path] = None) -> "ToolRegistry":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls(functions_dir)
        return cls._instance

    def register(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        return_type: str = "Any",
        source: str = "builtin"
    ) -> bool:
        """Register a tool (TE-002)"""
        with self._lock:
            # Extract parameters from function signature if not provided
            if parameters is None:
                parameters = self._extract_parameters(handler)

            # Use docstring for description if not provided
            if not description and handler.__doc__:
                description = handler.__doc__.strip().split("\n")[0]

            self._tools[name] = ToolInfo(
                name=name,
                description=description,
                parameters=parameters,
                return_type=return_type,
                source=source,
                handler=handler
            )
            logger.debug(f"Registered tool: {name} (source: {source})")
            return True

    def unregister(self, name: str) -> bool:
        """Unregister a tool"""
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                return True
            return False

    def get(self, name: str) -> Optional[ToolInfo]:
        """Get tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolInfo]:
        """List all registered tools (TE-002)"""
        return list(self._tools.values())

    def execute(self, name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool by name (TE-002)"""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        if not tool.handler:
            raise ValueError(f"Tool has no handler: {name}")
        return tool.handler(**params)

    def discover_functions(self) -> int:
        """Discover tools from Functions/ directory (TE-002)"""
        if not self.functions_dir or not self.functions_dir.exists():
            return 0

        discovered = 0
        for py_file in self.functions_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Look for callable functions marked as tools
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        if name.startswith("_"):
                            continue
                        # Register public functions as tools
                        self.register(
                            name=f"functions.{module_name}.{name}",
                            handler=obj,
                            source="functions"
                        )
                        discovered += 1

            except Exception as e:
                logger.warning(f"Could not load {py_file}: {e}")

        logger.info(f"Discovered {discovered} tools from Functions/")
        return discovered

    def _extract_parameters(self, handler: Callable) -> Dict[str, Any]:
        """Extract parameter info from function signature"""
        params = {}
        try:
            sig = inspect.signature(handler)
            for name, param in sig.parameters.items():
                if name in ("self", "cls"):
                    continue
                param_info = {"required": param.default == inspect.Parameter.empty}
                if param.annotation != inspect.Parameter.empty:
                    param_info["type"] = str(param.annotation)
                params[name] = param_info
        except Exception:
            pass
        return params


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
        """Execute a single task (TE-003, TE-004)"""
        # Check for cancellation (TE-004)
        if task.cancelled:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            logger.info(f"Task cancelled before execution: {task.description}")
            self._check_plan_completion(task.goal_id)
            return

        logger.info(f"Executing task: {task.description}")
        if task.retry_count > 0:
            logger.info(f"  (retry {task.retry_count}/{task.max_retries})")

        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()
        self.running_tasks[task.id] = task

        try:
            # Check cancellation during execution (TE-004)
            if task.cancelled:
                raise InterruptedError("Task cancelled")

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
                "duration": duration,
                "retry_count": task.retry_count
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

        except InterruptedError:
            # Task was cancelled during execution (TE-004)
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            logger.info(f"Task cancelled during execution: {task.description}")

            self._log_activity("task_cancelled", {
                "task_id": task.id,
                "goal_id": task.goal_id,
                "description": task.description
            })

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()

            logger.error(f"Task failed: {task.description} - {e}")

            # Check for retry (TE-003)
            if task.should_retry():
                delay = task.schedule_retry()
                logger.info(
                    f"Task will retry in {delay}s "
                    f"(attempt {task.retry_count}/{task.max_retries})"
                )
                self._log_activity("task_retry_scheduled", {
                    "task_id": task.id,
                    "goal_id": task.goal_id,
                    "description": task.description,
                    "retry_count": task.retry_count,
                    "delay": delay,
                    "error": str(e)
                })
            else:
                self._log_activity("task_failed", {
                    "task_id": task.id,
                    "goal_id": task.goal_id,
                    "description": task.description,
                    "error": str(e),
                    "retry_count": task.retry_count
                })

        finally:
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
            self._check_plan_completion(task.goal_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task (TE-004)"""
        # Check running tasks
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            if task.request_cancel():
                logger.info(f"Cancel requested for running task: {task_id}")
                return True

        # Check pending tasks in plans
        for plan in self.plans.values():
            for task in plan.tasks:
                if task.id == task_id:
                    if task.request_cancel():
                        task.status = TaskStatus.CANCELLED
                        task.completed_at = time.time()
                        logger.info(f"Task cancelled: {task_id}")
                        self._check_plan_completion(plan.goal_id)
                        return True
                    return False

        return False

    def cancel_goal(self, goal_id: str) -> int:
        """Cancel all tasks for a goal (TE-004)"""
        plan = self.plans.get(goal_id)
        if not plan:
            return 0

        cancelled_count = 0
        for task in plan.tasks:
            if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.WAITING):
                if task.request_cancel():
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = time.time()
                    cancelled_count += 1

        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} tasks for goal: {goal_id}")
            self._check_plan_completion(goal_id)

        return cancelled_count

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
        """Execute web search task with content fetching and synthesis (P2-003)"""
        try:
            sys.path.insert(0, str(self.senter_root / "Functions"))
            from web_search import search_web
            query = task.tool_params.get("query", task.description)
            results = search_web(query, max_results=5)

            if not results:
                return {"query": query, "results": [], "error": "No search results"}

            # P2-003: Fetch page content from top results
            sources = []
            content_parts = []

            for r in results[:3]:  # Fetch top 3 pages
                url = r.get("url", "")
                title = r.get("title", "")
                snippet = r.get("snippet", "")

                # Try to fetch page content
                page_content = self._fetch_page_content(url)

                sources.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "fetched": bool(page_content)
                })

                if page_content:
                    content_parts.append(f"## {title}\nSource: {url}\n\n{page_content[:2000]}")
                else:
                    content_parts.append(f"## {title}\nSource: {url}\n\n{snippet}")

            # P2-003: Synthesize with LLM
            if content_parts:
                synthesis = self._synthesize_research(query, content_parts, sources)
            else:
                synthesis = f"Search results for '{query}':\n" + "\n".join(
                    f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results
                )

            return {
                "query": query,
                "synthesis": synthesis,
                "sources": sources,
                "source_count": len(sources),
                "status": "completed"
            }

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return {"query": task.description, "results": [], "error": str(e)}

    def _fetch_page_content(self, url: str) -> str:
        """Fetch and extract readable content from a URL (P2-003)"""
        try:
            import httpx
            from html import unescape
            import re

            # Fetch with timeout
            response = httpx.get(url, timeout=10.0, follow_redirects=True, headers={
                "User-Agent": "Senter Research Agent/1.0"
            })
            response.raise_for_status()

            html = response.text

            # Simple content extraction (without readability library)
            # Remove scripts and styles
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)

            # Extract text from remaining HTML
            text = re.sub(r'<[^>]+>', ' ', html)
            text = unescape(text)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Truncate to reasonable length
            return text[:4000] if text else ""

        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return ""

    def _synthesize_research(self, query: str, content_parts: list, sources: list) -> str:
        """Synthesize research findings using LLM (P2-003)"""
        try:
            # Build prompt with content
            combined_content = "\n\n---\n\n".join(content_parts)

            synthesis_prompt = f"""Based on the following web search results for "{query}", provide a comprehensive summary with key insights.

SEARCH RESULTS:
{combined_content}

Please synthesize this information into:
1. A brief summary (2-3 sentences)
2. Key findings (bullet points)
3. Important details or considerations

Include citations to sources where appropriate."""

            # Send to LLM via message bus
            correlation_id = str(uuid.uuid4())

            self.message_bus.send(
                MessageType.MODEL_REQUEST,
                source="task_executor",
                target="model_research",
                payload={
                    "prompt": synthesis_prompt,
                    "system_prompt": "You are a research analyst. Synthesize information from multiple sources into clear, actionable summaries. Always cite your sources.",
                    "max_tokens": 1500
                },
                correlation_id=correlation_id
            )

            # Wait for response
            start_time = time.time()
            timeout = 60  # 60 second timeout for synthesis

            while time.time() - start_time < timeout:
                try:
                    msg_dict = self._response_queue.get(timeout=0.5)
                    message = Message.from_dict(msg_dict)

                    if (message.type == MessageType.MODEL_RESPONSE and
                        message.correlation_id == correlation_id):
                        response = message.payload.get("response", "")
                        # Add sources section
                        sources_text = "\n\n**Sources:**\n" + "\n".join(
                            f"- [{s['title']}]({s['url']})" for s in sources
                        )
                        return response + sources_text

                except Empty:
                    continue

            # Timeout - return basic summary
            return f"Research on '{query}' found {len(sources)} sources. See results for details."

        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")
            return f"Research on '{query}' completed. Found {len(sources)} sources."

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

#!/usr/bin/env python3
"""
Tests for Task Execution (TE-001, TE-002, TE-003, TE-004)
Tests task planning, tool registry, retry logic, and cancellation.
"""

import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== TE-001: Task Planning Tests ==========

def test_task_planner_creates_plan():
    """Test TaskPlanner creates execution plan (TE-001)"""
    from engine.task_engine import TaskPlanner

    mock_bus = MagicMock()
    planner = TaskPlanner(mock_bus, Path("/tmp"))

    plan = planner.create_plan("goal-1", "Research AI trends for 2026")

    assert plan.goal_id == "goal-1"
    assert plan.goal_description == "Research AI trends for 2026"
    assert len(plan.tasks) > 0

    return True


def test_task_planner_research_tasks():
    """Test TaskPlanner creates research-type tasks (TE-001)"""
    from engine.task_engine import TaskPlanner, TaskType

    mock_bus = MagicMock()
    planner = TaskPlanner(mock_bus, Path("/tmp"))

    plan = planner.create_plan("goal-1", "Research machine learning basics")

    # Should have research and summarize tasks
    task_types = [t.task_type for t in plan.tasks]
    assert TaskType.RESEARCH in task_types

    return True


def test_task_planner_writing_tasks():
    """Test TaskPlanner creates writing-type tasks (TE-001)"""
    from engine.task_engine import TaskPlanner, TaskType

    mock_bus = MagicMock()
    planner = TaskPlanner(mock_bus, Path("/tmp"))

    plan = planner.create_plan("goal-1", "Write a blog post about Python")

    # Should have outline, draft, review tasks
    task_types = [t.task_type for t in plan.tasks]
    assert TaskType.GENERATE in task_types

    return True


def test_task_has_dependencies():
    """Test tasks can have dependencies (TE-001)"""
    from engine.task_engine import TaskPlanner

    mock_bus = MagicMock()
    planner = TaskPlanner(mock_bus, Path("/tmp"))

    plan = planner.create_plan("goal-1", "Research Python frameworks")

    # At least one task should have dependencies
    has_deps = any(len(t.depends_on) > 0 for t in plan.tasks)
    assert has_deps, "Expected at least one task with dependencies"

    return True


# ========== TE-002: Tool Registry Tests ==========

def test_tool_registry_creation():
    """Test ToolRegistry can be created (TE-002)"""
    from engine.task_engine import ToolRegistry

    registry = ToolRegistry()
    assert registry is not None
    assert len(registry.list_tools()) == 0

    return True


def test_tool_registry_register():
    """Test ToolRegistry.register() (TE-002)"""
    from engine.task_engine import ToolRegistry

    registry = ToolRegistry()

    def my_tool(param1: str, param2: int = 10) -> str:
        """A test tool"""
        return f"{param1}-{param2}"

    result = registry.register("my_tool", my_tool, description="A test tool")
    assert result is True

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "my_tool"
    assert tools[0].description == "A test tool"

    return True


def test_tool_registry_get():
    """Test ToolRegistry.get() (TE-002)"""
    from engine.task_engine import ToolRegistry

    registry = ToolRegistry()

    def my_tool():
        pass

    registry.register("my_tool", my_tool)

    tool = registry.get("my_tool")
    assert tool is not None
    assert tool.name == "my_tool"

    # Non-existent tool
    assert registry.get("nonexistent") is None

    return True


def test_tool_registry_execute():
    """Test ToolRegistry.execute() (TE-002)"""
    from engine.task_engine import ToolRegistry

    registry = ToolRegistry()

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    registry.register("add", add_numbers)

    result = registry.execute("add", {"a": 5, "b": 3})
    assert result == 8

    return True


def test_tool_registry_unregister():
    """Test ToolRegistry.unregister() (TE-002)"""
    from engine.task_engine import ToolRegistry

    registry = ToolRegistry()

    def my_tool():
        pass

    registry.register("my_tool", my_tool)
    assert registry.get("my_tool") is not None

    result = registry.unregister("my_tool")
    assert result is True
    assert registry.get("my_tool") is None

    # Unregister non-existent
    result = registry.unregister("nonexistent")
    assert result is False

    return True


def test_tool_info_to_dict():
    """Test ToolInfo.to_dict() (TE-002)"""
    from engine.task_engine import ToolInfo

    info = ToolInfo(
        name="test_tool",
        description="A test tool",
        parameters={"param1": {"required": True}},
        return_type="str",
        source="builtin"
    )

    d = info.to_dict()

    assert d["name"] == "test_tool"
    assert d["description"] == "A test tool"
    assert "parameters" in d
    assert d["return_type"] == "str"
    assert d["source"] == "builtin"

    return True


# ========== TE-003: Retry with Exponential Backoff Tests ==========

def test_task_retry_fields():
    """Test Task has retry fields (TE-003)"""
    from engine.task_engine import Task, TaskType, TaskStatus

    task = Task(
        id="task-1",
        goal_id="goal-1",
        description="Test task",
        task_type=TaskType.RESEARCH
    )

    assert task.retry_count == 0
    assert task.max_retries == 5
    assert task.next_retry_at is None

    return True


def test_task_should_retry():
    """Test Task.should_retry() (TE-003)"""
    from engine.task_engine import Task, TaskType, TaskStatus

    task = Task(
        id="task-1",
        goal_id="goal-1",
        description="Test task",
        task_type=TaskType.RESEARCH
    )

    # Pending task should not retry
    assert task.should_retry() is False

    # Failed task with retries left should retry
    task.status = TaskStatus.FAILED
    task.retry_count = 0
    assert task.should_retry() is True

    # Failed task at max retries should not retry
    task.retry_count = 5
    assert task.should_retry() is False

    # Cancelled task should not retry
    task.retry_count = 0
    task.cancelled = True
    assert task.should_retry() is False

    return True


def test_task_schedule_retry():
    """Test Task.schedule_retry() with exponential backoff (TE-003)"""
    from engine.task_engine import Task, TaskType, TaskStatus, RETRY_DELAYS

    task = Task(
        id="task-1",
        goal_id="goal-1",
        description="Test task",
        task_type=TaskType.RESEARCH,
        status=TaskStatus.FAILED
    )

    # First retry - 1 second delay
    now = time.time()
    delay = task.schedule_retry()
    assert delay == RETRY_DELAYS[0]  # 1 second
    assert task.retry_count == 1
    assert task.next_retry_at >= now + delay - 0.1
    assert task.status == TaskStatus.PENDING

    # Second retry - 2 second delay
    task.status = TaskStatus.FAILED
    delay = task.schedule_retry()
    assert delay == RETRY_DELAYS[1]  # 2 seconds
    assert task.retry_count == 2

    return True


def test_exponential_backoff_delays():
    """Test exponential backoff delay values (TE-003)"""
    from engine.task_engine import RETRY_DELAYS

    # Should be 1, 2, 4, 8, 16
    assert RETRY_DELAYS == [1, 2, 4, 8, 16]

    # Each delay should be double the previous
    for i in range(1, len(RETRY_DELAYS)):
        assert RETRY_DELAYS[i] == RETRY_DELAYS[i-1] * 2

    return True


def test_execution_plan_respects_retry_timing():
    """Test get_ready_tasks respects next_retry_at (TE-003)"""
    from engine.task_engine import ExecutionPlan, Task, TaskType, TaskStatus

    task1 = Task(
        id="task-1",
        goal_id="goal-1",
        description="Task 1",
        task_type=TaskType.RESEARCH,
        status=TaskStatus.PENDING
    )

    task2 = Task(
        id="task-2",
        goal_id="goal-1",
        description="Task 2 (retry scheduled)",
        task_type=TaskType.RESEARCH,
        status=TaskStatus.PENDING,
        next_retry_at=time.time() + 100  # Far in the future
    )

    plan = ExecutionPlan(
        goal_id="goal-1",
        goal_description="Test goal",
        tasks=[task1, task2]
    )

    ready = plan.get_ready_tasks()

    # Only task1 should be ready (task2 retry not due yet)
    assert len(ready) == 1
    assert ready[0].id == "task-1"

    return True


# ========== TE-004: Task Cancellation Tests ==========

def test_task_cancel_fields():
    """Test Task has cancellation fields (TE-004)"""
    from engine.task_engine import Task, TaskType

    task = Task(
        id="task-1",
        goal_id="goal-1",
        description="Test task",
        task_type=TaskType.RESEARCH
    )

    assert task.cancelled is False
    assert task.cancel_requested_at is None

    return True


def test_task_request_cancel():
    """Test Task.request_cancel() (TE-004)"""
    from engine.task_engine import Task, TaskType, TaskStatus

    task = Task(
        id="task-1",
        goal_id="goal-1",
        description="Test task",
        task_type=TaskType.RESEARCH,
        status=TaskStatus.PENDING
    )

    # Can cancel pending task
    result = task.request_cancel()
    assert result is True
    assert task.cancelled is True
    assert task.cancel_requested_at is not None

    # Cannot cancel completed task
    task2 = Task(
        id="task-2",
        goal_id="goal-1",
        description="Completed task",
        task_type=TaskType.RESEARCH,
        status=TaskStatus.COMPLETED
    )
    result = task2.request_cancel()
    assert result is False

    return True


def test_execution_plan_skips_cancelled():
    """Test get_ready_tasks skips cancelled tasks (TE-004)"""
    from engine.task_engine import ExecutionPlan, Task, TaskType, TaskStatus

    task1 = Task(
        id="task-1",
        goal_id="goal-1",
        description="Normal task",
        task_type=TaskType.RESEARCH,
        status=TaskStatus.PENDING
    )

    task2 = Task(
        id="task-2",
        goal_id="goal-1",
        description="Cancelled task",
        task_type=TaskType.RESEARCH,
        status=TaskStatus.PENDING,
        cancelled=True
    )

    plan = ExecutionPlan(
        goal_id="goal-1",
        goal_description="Test goal",
        tasks=[task1, task2]
    )

    ready = plan.get_ready_tasks()

    # Only task1 should be ready (task2 is cancelled)
    assert len(ready) == 1
    assert ready[0].id == "task-1"

    return True


def test_task_engine_cancel_task():
    """Test TaskEngine.cancel_task() (TE-004)"""
    from engine.task_engine import TaskEngine, Task, TaskType, TaskStatus
    from multiprocessing import Event

    mock_bus = MagicMock()
    mock_bus.register.return_value = MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = TaskEngine(
            max_concurrent=3,
            message_bus=mock_bus,
            shutdown_event=Event(),
            senter_root=Path(tmpdir)
        )

        # Create a plan with a task
        task = Task(
            id="task-1",
            goal_id="goal-1",
            description="Test task",
            task_type=TaskType.RESEARCH,
            status=TaskStatus.PENDING
        )

        from engine.task_engine import ExecutionPlan
        plan = ExecutionPlan(
            goal_id="goal-1",
            goal_description="Test",
            tasks=[task]
        )
        engine.plans["goal-1"] = plan

        # Cancel the task
        result = engine.cancel_task("task-1")
        assert result is True
        assert task.status == TaskStatus.CANCELLED

    return True


def test_task_engine_cancel_goal():
    """Test TaskEngine.cancel_goal() (TE-004)"""
    from engine.task_engine import TaskEngine, Task, TaskType, TaskStatus, ExecutionPlan
    from multiprocessing import Event

    mock_bus = MagicMock()
    mock_bus.register.return_value = MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = TaskEngine(
            max_concurrent=3,
            message_bus=mock_bus,
            shutdown_event=Event(),
            senter_root=Path(tmpdir)
        )

        # Create a plan with multiple tasks
        tasks = [
            Task(id="task-1", goal_id="goal-1", description="Task 1",
                 task_type=TaskType.RESEARCH, status=TaskStatus.PENDING),
            Task(id="task-2", goal_id="goal-1", description="Task 2",
                 task_type=TaskType.RESEARCH, status=TaskStatus.PENDING),
            Task(id="task-3", goal_id="goal-1", description="Task 3 (completed)",
                 task_type=TaskType.RESEARCH, status=TaskStatus.COMPLETED),
        ]

        plan = ExecutionPlan(
            goal_id="goal-1",
            goal_description="Test",
            tasks=tasks
        )
        engine.plans["goal-1"] = plan

        # Cancel the goal
        cancelled = engine.cancel_goal("goal-1")
        assert cancelled == 2  # Only pending tasks cancelled

    return True


def test_task_to_dict_includes_retry_cancel():
    """Test Task.to_dict includes retry and cancel fields (TE-003, TE-004)"""
    from engine.task_engine import Task, TaskType

    task = Task(
        id="task-1",
        goal_id="goal-1",
        description="Test task",
        task_type=TaskType.RESEARCH
    )
    task.retry_count = 2
    task.cancelled = True

    d = task.to_dict()

    assert "retry_count" in d
    assert d["retry_count"] == 2
    assert "max_retries" in d
    assert "next_retry_at" in d
    assert "cancelled" in d
    assert d["cancelled"] is True
    assert "cancel_requested_at" in d

    return True


if __name__ == "__main__":
    tests = [
        # TE-001
        test_task_planner_creates_plan,
        test_task_planner_research_tasks,
        test_task_planner_writing_tasks,
        test_task_has_dependencies,
        # TE-002
        test_tool_registry_creation,
        test_tool_registry_register,
        test_tool_registry_get,
        test_tool_registry_execute,
        test_tool_registry_unregister,
        test_tool_info_to_dict,
        # TE-003
        test_task_retry_fields,
        test_task_should_retry,
        test_task_schedule_retry,
        test_exponential_backoff_delays,
        test_execution_plan_respects_retry_timing,
        # TE-004
        test_task_cancel_fields,
        test_task_request_cancel,
        test_execution_plan_skips_cancelled,
        test_task_engine_cancel_task,
        test_task_engine_cancel_goal,
        test_task_to_dict_includes_retry_cancel,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")

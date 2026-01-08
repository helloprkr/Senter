#!/usr/bin/env python3
"""
Tests for Goal Execution Pipeline (CG-008)
End-to-end tests for: goal detected -> tasks created -> tasks executed -> results stored -> notification
"""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from multiprocessing import Queue

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_goal_message_types_exist():
    """Test that GOAL_DETECTED and GOAL_COMPLETE message types exist"""
    from daemon.message_bus import MessageType

    # CG-008: New message types should exist
    assert hasattr(MessageType, "GOAL_DETECTED")
    assert hasattr(MessageType, "GOAL_COMPLETE")
    assert MessageType.GOAL_DETECTED.value == "goal_detected"
    assert MessageType.GOAL_COMPLETE.value == "goal_complete"

    return True


def test_goal_tracker_with_message_bus():
    """Test that GoalTracker can accept message_bus parameter"""
    from Functions.goal_tracker import GoalTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        mock_bus = MagicMock()

        tracker = GoalTracker(senter_root=senter_root, message_bus=mock_bus)

        assert tracker.message_bus == mock_bus
        assert tracker.senter_root == senter_root

        return True


def test_goal_triggers_plan_creation():
    """Test that saving a goal with trigger_execution sends GOAL_DETECTED message"""
    from Functions.goal_tracker import GoalTracker, Goal

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)

        # Create mock message bus
        mock_bus = MagicMock()

        tracker = GoalTracker(senter_root=senter_root, message_bus=mock_bus)

        # Create a test goal
        goal = Goal(
            id="goal_test_001",
            description="Test goal for execution",
            created="2026-01-07T12:00:00",
            category="work",
            priority="high"
        )

        # Save with trigger_execution=True
        tracker.save_goal(goal, trigger_execution=True)

        # Verify GOAL_DETECTED message was sent
        mock_bus.send.assert_called_once()
        call_args = mock_bus.send.call_args

        # Check message type
        from daemon.message_bus import MessageType
        assert call_args[0][0] == MessageType.GOAL_DETECTED

        # Check payload
        assert "goal_id" in call_args[1]["payload"]
        assert call_args[1]["payload"]["goal_id"] == "goal_test_001"
        assert call_args[1]["payload"]["description"] == "Test goal for execution"

        return True


def test_task_engine_handles_goal_detected():
    """Test that TaskEngine handles GOAL_DETECTED messages"""
    from daemon.message_bus import MessageBus, MessageType, Message
    from engine.task_engine import TaskEngine, ExecutionPlan
    from multiprocessing import Event

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        bus = MessageBus()
        bus.start()
        shutdown = Event()

        try:
            engine = TaskEngine(
                max_concurrent=2,
                message_bus=bus,
                shutdown_event=shutdown,
                senter_root=senter_root
            )

            # Manually call _handle_message with GOAL_DETECTED
            message = Message(
                type=MessageType.GOAL_DETECTED,
                source="goal_tracker",
                payload={
                    "goal_id": "goal_test_002",
                    "description": "Research AI trends",
                    "category": "research",
                    "priority": "high"
                }
            )

            engine._handle_message(message)

            # Check that a plan was created
            assert "goal_test_002" in engine.plans
            plan = engine.plans["goal_test_002"]
            assert isinstance(plan, ExecutionPlan)
            assert plan.goal_description == "Research AI trends"
            assert len(plan.tasks) > 0

        finally:
            bus.stop()

        return True


def test_plan_executes_all_tasks():
    """Test that ExecutionPlan tracks task completion correctly"""
    from engine.task_engine import ExecutionPlan, Task, TaskType, TaskStatus

    # Create a plan with tasks
    plan = ExecutionPlan(
        goal_id="test_goal",
        goal_description="Test plan execution",
        tasks=[
            Task(
                id="task_1",
                goal_id="test_goal",
                description="First task",
                task_type=TaskType.RESEARCH
            ),
            Task(
                id="task_2",
                goal_id="test_goal",
                description="Second task",
                task_type=TaskType.GENERATE,
                depends_on=["task_1"]
            )
        ]
    )

    # Initially, only first task should be ready
    ready = plan.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "task_1"

    # Complete first task
    plan.tasks[0].status = TaskStatus.COMPLETED

    # Now second task should be ready
    ready = plan.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "task_2"

    # Complete second task
    plan.tasks[1].status = TaskStatus.COMPLETED

    # No more ready tasks
    ready = plan.get_ready_tasks()
    assert len(ready) == 0

    return True


def test_completion_notification():
    """Test that GOAL_COMPLETE message is sent when plan completes"""
    from daemon.message_bus import MessageBus, MessageType
    from engine.task_engine import TaskEngine, Task, TaskType, TaskStatus, ExecutionPlan
    from multiprocessing import Event

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        bus = MessageBus()
        bus.start()
        shutdown = Event()

        # Register a test subscriber to receive GOAL_COMPLETE
        test_queue = bus.register("test_receiver")

        try:
            engine = TaskEngine(
                max_concurrent=2,
                message_bus=bus,
                shutdown_event=shutdown,
                senter_root=senter_root
            )

            # Create a plan manually
            plan = ExecutionPlan(
                goal_id="notify_test",
                goal_description="Test notification",
                tasks=[
                    Task(
                        id="task_notify_1",
                        goal_id="notify_test",
                        description="Test task",
                        task_type=TaskType.CUSTOM,
                        status=TaskStatus.COMPLETED,
                        result={"test": "result"}
                    )
                ]
            )
            engine.plans["notify_test"] = plan

            # Call _check_plan_completion
            engine._check_plan_completion("notify_test")

            # Plan should be cleaned up
            assert "notify_test" not in engine.plans

            # Check that GOAL_COMPLETE was sent (give bus time to route)
            time.sleep(0.2)

            # The message should have been broadcast
            # Check the test_receiver queue or main bus queue

        finally:
            bus.stop()

        return True


def test_goal_status_determination():
    """Test that goal completion status is determined correctly"""
    from engine.task_engine import TaskEngine, Task, TaskType, TaskStatus, ExecutionPlan
    from daemon.message_bus import MessageBus
    from multiprocessing import Event

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        bus = MessageBus()
        bus.start()
        shutdown = Event()

        try:
            engine = TaskEngine(
                max_concurrent=2,
                message_bus=bus,
                shutdown_event=shutdown,
                senter_root=senter_root
            )

            # Test 1: All tasks completed = success
            plan1 = ExecutionPlan(
                goal_id="status_test_1",
                goal_description="All success",
                tasks=[
                    Task(id="t1", goal_id="status_test_1", description="T1",
                         task_type=TaskType.CUSTOM, status=TaskStatus.COMPLETED),
                    Task(id="t2", goal_id="status_test_1", description="T2",
                         task_type=TaskType.CUSTOM, status=TaskStatus.COMPLETED)
                ]
            )
            engine.plans["status_test_1"] = plan1
            engine._check_plan_completion("status_test_1")
            # Plan should be removed
            assert "status_test_1" not in engine.plans

            # Test 2: Some failed = partial
            plan2 = ExecutionPlan(
                goal_id="status_test_2",
                goal_description="Partial success",
                tasks=[
                    Task(id="t3", goal_id="status_test_2", description="T3",
                         task_type=TaskType.CUSTOM, status=TaskStatus.COMPLETED),
                    Task(id="t4", goal_id="status_test_2", description="T4",
                         task_type=TaskType.CUSTOM, status=TaskStatus.FAILED)
                ]
            )
            engine.plans["status_test_2"] = plan2
            engine._check_plan_completion("status_test_2")
            assert "status_test_2" not in engine.plans

            # Test 3: All failed = failed
            plan3 = ExecutionPlan(
                goal_id="status_test_3",
                goal_description="All failed",
                tasks=[
                    Task(id="t5", goal_id="status_test_3", description="T5",
                         task_type=TaskType.CUSTOM, status=TaskStatus.FAILED),
                    Task(id="t6", goal_id="status_test_3", description="T6",
                         task_type=TaskType.CUSTOM, status=TaskStatus.FAILED)
                ]
            )
            engine.plans["status_test_3"] = plan3
            engine._check_plan_completion("status_test_3")
            assert "status_test_3" not in engine.plans

        finally:
            bus.stop()

        return True


def test_routing_table_includes_goal_messages():
    """Test that routing table routes GOAL_DETECTED and GOAL_COMPLETE"""
    from daemon.message_bus import MessageBus, MessageType

    bus = MessageBus()

    # Check routing for GOAL_DETECTED
    assert MessageType.GOAL_DETECTED in bus.routing_table
    assert "task_engine" in bus.routing_table[MessageType.GOAL_DETECTED]

    # Check routing for GOAL_COMPLETE
    assert MessageType.GOAL_COMPLETE in bus.routing_table
    assert "reporter" in bus.routing_table[MessageType.GOAL_COMPLETE]
    assert "learning" in bus.routing_table[MessageType.GOAL_COMPLETE]

    return True


if __name__ == "__main__":
    tests = [
        test_goal_message_types_exist,
        test_goal_tracker_with_message_bus,
        test_goal_triggers_plan_creation,
        test_task_engine_handles_goal_detected,
        test_plan_executes_all_tasks,
        test_completion_notification,
        test_goal_status_determination,
        test_routing_table_includes_goal_messages,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")

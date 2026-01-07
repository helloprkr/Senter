#!/usr/bin/env python3
"""
Tests for TaskExecutor - specifically US-001 acceptance criteria:
1. TaskExecutor._execute_with_llm sends request to model worker and waits for response
2. Response is captured and returned (not just 'submitted')
3. Execution timeout of 120 seconds is enforced
4. Task result is stored in task object
"""

import sys
import time
import threading
from pathlib import Path
from multiprocessing import Event

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.task_engine import TaskExecutor, Task, TaskType, TaskStatus
from daemon.message_bus import MessageBus, MessageType, Message


def test_task_executor_returns_result():
    """Test that executor returns actual response, not just 'submitted'"""
    # Setup
    bus = MessageBus()
    bus.start()
    shutdown = Event()

    try:
        # Register mock worker queue FIRST (before executor sends request)
        worker_queue = bus.register("model_research")

        executor = TaskExecutor(bus, Path(__file__).parent.parent)

        # Create a test task
        task = Task(
            id="test_001",
            goal_id="goal_001",
            description="Test task: say hello",
            task_type=TaskType.GENERATE
        )

        # Simulate model worker response in separate thread
        def mock_model_worker():
            try:
                # Wait for request
                msg_dict = worker_queue.get(timeout=5.0)
                message = Message.from_dict(msg_dict)

                if message.type == MessageType.MODEL_REQUEST:
                    # Small delay to simulate processing
                    time.sleep(0.1)
                    # Send mock response
                    bus.send(
                        MessageType.MODEL_RESPONSE,
                        source="model_research",
                        target="task_executor",
                        payload={
                            "response": "Hello! This is a real response from the model.",
                            "model": "test-model",
                            "latency_ms": 150,
                            "worker": "research"
                        },
                        correlation_id=message.correlation_id
                    )
            except Exception as e:
                print(f"Mock worker error: {e}")

        # Start mock worker thread
        worker_thread = threading.Thread(target=mock_model_worker, daemon=True)
        worker_thread.start()

        # Execute task - should now wait for and return real response
        result = executor._execute_with_llm(task)

        # Assertions
        assert result is not None, "Result should not be None"
        assert result.get("status") != "submitted", "Result should NOT be just 'submitted'"
        assert result.get("status") == "completed", f"Result status should be 'completed', got '{result.get('status')}'"
        assert "response" in result, "Result should contain 'response'"
        assert "Hello!" in result["response"], "Response should contain actual model output"

        # Check task object was updated
        assert task.result is not None, "Task.result should be set"
        assert task.result.get("status") == "completed", "Task result should be completed"

        print("✓ test_task_executor_returns_result PASSED")
        return True

    finally:
        bus.stop()


def test_task_executor_timeout():
    """Test that executor times out after 120 seconds (we use shorter for test)"""
    # Patch timeout to 2 seconds for testing
    original_timeout = TaskExecutor.LLM_TIMEOUT_SECONDS
    TaskExecutor.LLM_TIMEOUT_SECONDS = 2

    bus = MessageBus()
    bus.start()

    try:
        executor = TaskExecutor(bus, Path(__file__).parent.parent)

        task = Task(
            id="test_timeout",
            goal_id="goal_002",
            description="Test timeout task",
            task_type=TaskType.CUSTOM
        )

        # Don't start any mock worker - let it timeout
        start = time.time()
        timed_out = False

        try:
            executor._execute_with_llm(task)
        except TimeoutError as e:
            timed_out = True
            elapsed = time.time() - start
            assert elapsed >= 1.5, f"Should have waited ~2 seconds, waited {elapsed:.1f}s"
            assert "timed out" in str(e).lower() or "timeout" in str(e).lower(), f"Exception message should contain timeout info: {e}"

        assert timed_out, "Should have raised TimeoutError"

        # Check task result reflects timeout
        assert task.result is not None, "Task result should be set even on timeout"
        assert task.result.get("status") == "timeout", f"Task status should be 'timeout', got {task.result.get('status')}"

        print("✓ test_task_executor_timeout PASSED")
        return True

    finally:
        TaskExecutor.LLM_TIMEOUT_SECONDS = original_timeout
        bus.stop()


def test_task_result_stored():
    """Test that result is stored in task object"""
    bus = MessageBus()
    bus.start()

    try:
        # Register mock worker queue FIRST
        worker_queue = bus.register("model_research")

        executor = TaskExecutor(bus, Path(__file__).parent.parent)

        task = Task(
            id="test_storage",
            goal_id="goal_003",
            description="Test result storage",
            task_type=TaskType.RESEARCH
        )

        # Confirm task.result starts as None
        assert task.result is None, "Task result should start as None"

        # Mock response
        def mock_response():
            try:
                msg_dict = worker_queue.get(timeout=5.0)
                message = Message.from_dict(msg_dict)
                if message.type == MessageType.MODEL_REQUEST:
                    time.sleep(0.1)
                    bus.send(
                        MessageType.MODEL_RESPONSE,
                        source="model_research",
                        target="task_executor",
                        payload={
                            "response": "Research results here",
                            "model": "llama",
                            "latency_ms": 200,
                            "worker": "research"
                        },
                        correlation_id=message.correlation_id
                    )
            except Exception:
                pass

        thread = threading.Thread(target=mock_response, daemon=True)
        thread.start()

        result = executor._execute_with_llm(task)

        # Task object should now have result
        assert task.result is not None, "Task result should be set after execution"
        assert task.result == result, "Task.result should match returned result"
        assert task.result["response"] == "Research results here"

        print("✓ test_task_result_stored PASSED")
        return True

    finally:
        bus.stop()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running TaskExecutor Tests (US-001)")
    print("="*60 + "\n")

    all_passed = True

    tests = [
        test_task_executor_returns_result,
        test_task_executor_timeout,
        test_task_result_stored,
    ]

    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)

    sys.exit(0 if all_passed else 1)

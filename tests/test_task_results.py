#!/usr/bin/env python3
"""
Tests for Task Result Storage - US-003 acceptance criteria:
1. Completed task results stored in data/tasks/results/
2. Results include: task_id, goal_id, result content, timestamp
3. Results queryable via IPC 'get_results' command
4. Results persist across daemon restarts
"""

import sys
import json
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_task_results_stored():
    """Test that task results are stored correctly"""
    from engine.task_results import TaskResultStorage, TaskResult

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = TaskResultStorage(Path(tmpdir) / "results")

        # Store a result
        result = TaskResult(
            task_id="task_001",
            goal_id="goal_001",
            result="Research completed on AI trends",
            description="Research AI trends for 2026",
            worker="research",
            latency_ms=1500
        )
        success = storage.store(result)
        assert success, "Store should return True"

        # Verify stored
        retrieved = storage.get_by_task_id("task_001")
        assert retrieved is not None, "Should retrieve stored result"
        assert retrieved.task_id == "task_001"
        assert retrieved.goal_id == "goal_001"
        assert retrieved.result == "Research completed on AI trends"
        assert retrieved.description == "Research AI trends for 2026"
        assert retrieved.worker == "research"
        assert retrieved.latency_ms == 1500
        assert retrieved.timestamp > 0

    print("✓ test_task_results_stored PASSED")
    return True


def test_task_result_fields():
    """Test that results include required fields"""
    from engine.task_results import TaskResult

    result = TaskResult(
        task_id="task_002",
        goal_id="goal_002",
        result={"response": "Some content"},
        description="Test task"
    )

    # Check required fields
    d = result.to_dict()
    assert "task_id" in d, "Should have task_id"
    assert "goal_id" in d, "Should have goal_id"
    assert "result" in d, "Should have result"
    assert "timestamp" in d, "Should have timestamp"
    assert "datetime" in d, "Should have datetime (derived)"
    assert "status" in d, "Should have status"
    assert "description" in d, "Should have description"

    # Verify values
    assert d["task_id"] == "task_002"
    assert d["goal_id"] == "goal_002"
    assert d["result"] == {"response": "Some content"}
    assert d["timestamp"] > 0

    print("✓ test_task_result_fields PASSED")
    return True


def test_results_queryable():
    """Test that results are queryable by various criteria"""
    from engine.task_results import TaskResultStorage, TaskResult

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = TaskResultStorage(Path(tmpdir) / "results")

        # Store multiple results
        for i in range(5):
            result = TaskResult(
                task_id=f"task_{i:03d}",
                goal_id=f"goal_{i % 2:03d}",  # 2 goals
                result=f"Result {i}",
                description=f"Task {i}"
            )
            storage.store(result)

        # Query by task_id
        r = storage.get_by_task_id("task_002")
        assert r is not None
        assert r.result == "Result 2"

        # Query by goal_id
        goal_results = storage.get_by_goal_id("goal_000")
        assert len(goal_results) >= 2  # Tasks 0, 2, 4

        # Query recent
        recent = storage.get_recent(limit=3)
        assert len(recent) == 3

        # Get summary
        summary = storage.get_summary()
        assert summary["total_results"] == 5
        assert summary["total_goals"] == 2

    print("✓ test_results_queryable PASSED")
    return True


def test_results_persist():
    """Test that results persist across storage instances"""
    from engine.task_results import TaskResultStorage, TaskResult

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"

        # Store with first instance
        storage1 = TaskResultStorage(results_dir)
        result = TaskResult(
            task_id="persist_test",
            goal_id="goal_persist",
            result="Persistent data",
            description="Test persistence"
        )
        storage1.store(result)

        # Create new instance (simulates restart)
        storage2 = TaskResultStorage(results_dir)

        # Query should still work
        retrieved = storage2.get_by_task_id("persist_test")
        assert retrieved is not None, "Result should persist"
        assert retrieved.result == "Persistent data"
        assert retrieved.goal_id == "goal_persist"

    print("✓ test_results_persist PASSED")
    return True


def test_ipc_get_results_handler():
    """Test that IPC handler for get_results works"""
    from daemon.ipc_server import IPCServer
    from daemon.senter_daemon import SenterDaemon
    from multiprocessing import Event
    from engine.task_results import TaskResultStorage, TaskResult
    from pathlib import Path

    daemon = SenterDaemon()
    shutdown = Event()
    ipc = IPCServer(shutdown_event=shutdown, daemon_ref=daemon)

    # Store a test result
    results_dir = daemon.senter_root / "data" / "tasks" / "results"
    storage = TaskResultStorage(results_dir)

    result = TaskResult(
        task_id="ipc_test_001",
        goal_id="goal_ipc",
        result="IPC test result",
        description="Test IPC get_results"
    )
    storage.store(result)

    # Query via IPC handler
    response = ipc._handle_get_results({"task_id": "ipc_test_001"})
    assert "result" in response, f"Response should have 'result': {response}"
    assert response["result"]["task_id"] == "ipc_test_001"

    # Query recent via IPC
    response = ipc._handle_get_results({"limit": 10})
    assert "results" in response
    assert "summary" in response

    print("✓ test_ipc_get_results_handler PASSED")
    return True


def test_ipc_client_method():
    """Test that IPC client has get_results method"""
    from daemon.ipc_client import IPCClient

    client = IPCClient()

    # Verify method exists
    assert hasattr(client, 'get_results'), "Client should have get_results"
    assert callable(client.get_results)

    print("✓ test_ipc_client_method PASSED")
    return True


def test_executor_stores_result():
    """Test that TaskExecutor stores results after execution"""
    from engine.task_engine import TaskExecutor, Task, TaskType
    from daemon.message_bus import MessageBus
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus()
        bus.start()

        try:
            # Create executor with temp directory
            executor = TaskExecutor(bus, Path(tmpdir))

            # Create a task (use file_read which doesn't need external deps)
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")

            task = Task(
                id="exec_test_001",
                goal_id="goal_exec",
                description="Read test file",
                task_type=TaskType.CUSTOM,
                tool="file_read",
                tool_params={"path": str(test_file)}
            )

            # Execute
            result = executor.execute(task)

            # Check result storage
            stored = executor.result_storage.get_by_task_id("exec_test_001")
            assert stored is not None, "Result should be stored after execution"
            assert stored.goal_id == "goal_exec"
            # Result may be the content dict or the response string
            assert stored.result is not None, "Result should have content"

        finally:
            bus.stop()

    print("✓ test_executor_stores_result PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Task Result Storage Tests (US-003)")
    print("="*60 + "\n")

    all_passed = True

    tests = [
        test_task_results_stored,
        test_task_result_fields,
        test_results_queryable,
        test_results_persist,
        test_ipc_get_results_handler,
        test_ipc_client_method,
        test_executor_stores_result,
    ]

    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)

    sys.exit(0 if all_passed else 1)

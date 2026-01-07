#!/usr/bin/env python3
"""
Tests for Research Task Queue - US-002 acceptance criteria:
1. New Queue 'research_tasks' created in daemon
2. Research worker pulls from research_tasks queue (not model_research)
3. Tasks can be added to research queue via IPC
4. Queue persists pending tasks
"""

import sys
import json
import time
from pathlib import Path
from multiprocessing import Queue

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_research_queue_exists():
    """Test that research_tasks queue is created in daemon"""
    from daemon.senter_daemon import SenterDaemon

    # Create daemon instance (don't start it)
    daemon = SenterDaemon()

    # Check queue exists
    assert hasattr(daemon, 'research_tasks_queue'), "Daemon should have research_tasks_queue"

    # Check persistence file path is set
    assert hasattr(daemon, 'research_tasks_file'), "Daemon should have research_tasks_file"
    assert "research_tasks.json" in str(daemon.research_tasks_file)

    # Check methods exist
    assert hasattr(daemon, 'add_research_task'), "Should have add_research_task method"
    assert hasattr(daemon, 'get_research_queue_status'), "Should have get_research_queue_status method"
    assert hasattr(daemon, '_load_research_tasks'), "Should have _load_research_tasks method"
    assert hasattr(daemon, '_save_research_tasks'), "Should have _save_research_tasks method"

    print("✓ test_research_queue_exists PASSED")
    return True


def test_add_research_task():
    """Test that tasks can be added to research queue"""
    from daemon.senter_daemon import SenterDaemon
    from queue import Empty

    daemon = SenterDaemon()

    # Add a task
    task = {
        "description": "Research AI trends for 2026",
        "priority": 8
    }

    success = daemon.add_research_task(task)
    assert success, "add_research_task should return True"

    # Retrieve directly to verify (don't use get_research_queue_status as it drains/refills)
    try:
        retrieved = daemon.research_tasks_queue.get(timeout=1.0)
        assert "id" in retrieved, "Task should have id"
        assert "created_at" in retrieved, "Task should have created_at"
        assert "status" in retrieved, "Task should have status"
        assert retrieved["description"] == "Research AI trends for 2026"
        assert retrieved["priority"] == 8
    except Empty:
        assert False, "Queue should have the added task"

    print("✓ test_add_research_task PASSED")
    return True


def test_research_queue_persistence():
    """Test that queue persists pending tasks to disk"""
    from daemon.senter_daemon import SenterDaemon
    import tempfile

    # Use temp directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create daemon with custom persistence file
        daemon = SenterDaemon()
        daemon.research_tasks_file = Path(tmpdir) / "research_tasks.json"

        # Add tasks
        daemon.add_research_task({"description": "Task 1"})
        daemon.add_research_task({"description": "Task 2"})
        daemon.add_research_task({"description": "Task 3"})

        # Save tasks
        daemon._save_research_tasks()

        # Verify file exists
        assert daemon.research_tasks_file.exists(), "Persistence file should exist"

        # Verify content
        data = json.loads(daemon.research_tasks_file.read_text())
        assert "pending_tasks" in data
        assert len(data["pending_tasks"]) == 3
        assert "saved_at" in data

        # Verify task structure
        for task in data["pending_tasks"]:
            assert "id" in task
            assert "description" in task
            assert "created_at" in task

        # Create new daemon and load from same file
        daemon2 = SenterDaemon()
        daemon2.research_tasks_file = daemon.research_tasks_file
        daemon2._load_research_tasks()

        # Count loaded tasks by draining
        loaded_count = 0
        from queue import Empty
        while True:
            try:
                daemon2.research_tasks_queue.get_nowait()
                loaded_count += 1
            except Empty:
                break

        assert loaded_count == 3, f"Should have loaded 3 tasks, got {loaded_count}"

    print("✓ test_research_queue_persistence PASSED")
    return True


def test_ipc_handler_add_research_task():
    """Test that IPC handler correctly adds research tasks"""
    from daemon.ipc_server import IPCServer
    from daemon.senter_daemon import SenterDaemon
    from multiprocessing import Event
    from queue import Empty

    # Create fresh daemon and IPC server
    daemon = SenterDaemon()
    shutdown = Event()
    ipc = IPCServer(shutdown_event=shutdown, daemon_ref=daemon)

    # Test handler with valid request
    request = {
        "command": "add_research_task",
        "description": "Test research topic"
    }

    response = ipc._handle_add_research_task(request)
    assert response.get("status") == "ok", f"Should succeed, got: {response}"
    assert "task_id" in response

    # Verify task was added by draining queue
    try:
        task = daemon.research_tasks_queue.get(timeout=1.0)
        assert task["description"] == "Test research topic"
    except Empty:
        assert False, "Task should have been added to queue"

    # Test handler with missing description
    bad_request = {"command": "add_research_task"}
    bad_response = ipc._handle_add_research_task(bad_request)
    assert "error" in bad_response

    print("✓ test_ipc_handler_add_research_task PASSED")
    return True


def test_ipc_handler_research_queue_status():
    """Test that IPC handler returns queue status"""
    from daemon.ipc_server import IPCServer
    from daemon.senter_daemon import SenterDaemon
    from multiprocessing import Event

    daemon = SenterDaemon()
    shutdown = Event()
    ipc = IPCServer(shutdown_event=shutdown, daemon_ref=daemon)

    # Test handler returns valid response structure
    response = ipc._handle_research_queue_status({})
    assert "queue_size" in response, f"Response missing queue_size: {response}"
    assert isinstance(response["queue_size"], int), f"queue_size should be int: {response}"
    assert "file_exists" in response, f"Response missing file_exists: {response}"

    print("✓ test_ipc_handler_research_queue_status PASSED")
    return True


def test_ipc_client_methods():
    """Test that IPC client has the new research methods"""
    from daemon.ipc_client import IPCClient

    client = IPCClient()

    # Verify methods exist
    assert hasattr(client, 'add_research_task'), "Client should have add_research_task"
    assert hasattr(client, 'research_queue_status'), "Client should have research_queue_status"

    # These would need the daemon running to actually work
    # Just verify the methods are callable
    assert callable(client.add_research_task)
    assert callable(client.research_queue_status)

    print("✓ test_ipc_client_methods PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Research Queue Tests (US-002)")
    print("="*60 + "\n")

    all_passed = True

    tests = [
        test_research_queue_exists,
        test_add_research_task,
        test_research_queue_persistence,
        test_ipc_handler_add_research_task,
        test_ipc_handler_research_queue_status,
        test_ipc_client_methods,
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

#!/usr/bin/env python3
"""
Tests for Activity Report - US-006 acceptance criteria:
1. IPC command 'activity_report' returns completed tasks
2. Report includes: tasks completed, research done, time spent
3. CLI command 'senter_ctl.py report' shows formatted output
4. Report covers configurable time period (default 24h)
"""

import sys
import json
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_ipc_handler_exists():
    """Test that IPC handler for activity_report exists"""
    from daemon.ipc_server import IPCServer
    from daemon.ipc_client import IPCClient

    # Check server handler exists
    server = IPCServer()
    assert "activity_report" in server.handlers, \
        "IPC server should have activity_report handler"

    # Check client method exists
    client = IPCClient()
    assert hasattr(client, "activity_report"), \
        "IPC client should have activity_report method"

    print("✓ test_ipc_handler_exists PASSED")
    return True


def test_report_includes_required_fields():
    """Test that activity report includes required fields"""
    from daemon.ipc_server import IPCServer
    from daemon.senter_daemon import SenterDaemon
    from multiprocessing import Event

    daemon = SenterDaemon()
    shutdown = Event()
    ipc = IPCServer(shutdown_event=shutdown, daemon_ref=daemon)

    # Call handler
    report = ipc._handle_activity_report({"hours": 24})

    # Check required fields
    assert "period_hours" in report, "Report should have period_hours"
    assert "since" in report, "Report should have since timestamp"
    assert "until" in report, "Report should have until timestamp"
    assert "tasks_completed" in report, "Report should have tasks_completed"
    assert "research_done" in report, "Report should have research_done"
    assert "time_stats" in report, "Report should have time_stats"

    # Check time_stats fields
    stats = report.get("time_stats", {})
    assert "tasks_completed_count" in stats, "Stats should have tasks_completed_count"
    assert "research_completed_count" in stats, "Stats should have research_completed_count"
    assert "total_task_time_seconds" in stats, "Stats should have total_task_time_seconds"

    print("✓ test_report_includes_required_fields PASSED")
    return True


def test_report_configurable_period():
    """Test that report period is configurable"""
    from daemon.ipc_server import IPCServer
    from daemon.senter_daemon import SenterDaemon
    from multiprocessing import Event
    from datetime import datetime

    daemon = SenterDaemon()
    shutdown = Event()
    ipc = IPCServer(shutdown_event=shutdown, daemon_ref=daemon)

    # Test different periods
    for hours in [1, 12, 24, 48]:
        report = ipc._handle_activity_report({"hours": hours})
        assert report.get("period_hours") == hours, \
            f"Report should reflect {hours} hours period"

        # Verify time range is correct
        since = datetime.fromisoformat(report["since"])
        until = datetime.fromisoformat(report["until"])
        delta_hours = (until - since).total_seconds() / 3600
        assert abs(delta_hours - hours) < 0.1, \
            f"Time range should be approximately {hours} hours"

    print("✓ test_report_configurable_period PASSED")
    return True


def test_cli_report_function_exists():
    """Test that CLI report function exists"""
    # Import the senter_ctl module to check for show_report function
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "senter_ctl",
        Path(__file__).parent.parent / "scripts" / "senter_ctl.py"
    )
    module = importlib.util.module_from_spec(spec)

    # We can't fully execute, but we can check the file contains the function
    senter_ctl_path = Path(__file__).parent.parent / "scripts" / "senter_ctl.py"
    content = senter_ctl_path.read_text()

    assert "def show_report" in content, "senter_ctl.py should have show_report function"
    assert '"report"' in content or "'report'" in content, \
        "senter_ctl.py should have 'report' command"
    assert "activity_report" in content, \
        "show_report should use activity_report IPC command"

    print("✓ test_cli_report_function_exists PASSED")
    return True


def test_report_aggregates_task_results():
    """Test that report aggregates task results correctly"""
    from daemon.ipc_server import IPCServer
    from daemon.senter_daemon import SenterDaemon
    from engine.task_results import TaskResultStorage, TaskResult
    from multiprocessing import Event

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup daemon with temp directory
        daemon = SenterDaemon()
        daemon.senter_root = Path(tmpdir)

        # Create task results
        results_dir = Path(tmpdir) / "data" / "tasks" / "results"
        storage = TaskResultStorage(results_dir)

        # Store test results
        for i in range(3):
            result = TaskResult(
                task_id=f"report_test_{i:03d}",
                goal_id="goal_report",
                result=f"Result {i}",
                description=f"Test task {i}",
                worker="primary",
                latency_ms=1000 + i * 100
            )
            storage.store(result)

        # Create IPC server
        shutdown = Event()
        ipc = IPCServer(shutdown_event=shutdown, daemon_ref=daemon)

        # Get report
        report = ipc._handle_activity_report({"hours": 1})

        # Verify tasks are included
        tasks = report.get("tasks_completed", [])
        assert len(tasks) == 3, "Report should include 3 tasks"

        # Verify time stats are calculated
        stats = report.get("time_stats", {})
        assert stats.get("tasks_completed_count") == 3

    print("✓ test_report_aggregates_task_results PASSED")
    return True


def test_activity_report():
    """Combined test for US-006 acceptance criteria"""
    from daemon.ipc_server import IPCServer
    from daemon.ipc_client import IPCClient
    from daemon.senter_daemon import SenterDaemon
    from multiprocessing import Event

    # 1. IPC command exists and works
    server = IPCServer()
    assert "activity_report" in server.handlers

    # 2. Report includes required fields
    daemon = SenterDaemon()
    shutdown = Event()
    ipc = IPCServer(shutdown_event=shutdown, daemon_ref=daemon)

    report = ipc._handle_activity_report({"hours": 24})
    assert "tasks_completed" in report
    assert "research_done" in report
    assert "time_stats" in report
    stats = report["time_stats"]
    assert "tasks_completed_count" in stats
    assert "total_task_time_seconds" in stats

    # 3. CLI command exists (checked by string search)
    senter_ctl_path = Path(__file__).parent.parent / "scripts" / "senter_ctl.py"
    content = senter_ctl_path.read_text()
    assert "def show_report" in content
    assert "activity_report" in content

    # 4. Period is configurable
    report_12h = ipc._handle_activity_report({"hours": 12})
    assert report_12h["period_hours"] == 12

    print("✓ test_activity_report PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Activity Report Tests (US-006)")
    print("="*60 + "\n")

    all_passed = True

    tests = [
        test_ipc_handler_exists,
        test_report_includes_required_fields,
        test_report_configurable_period,
        test_cli_report_function_exists,
        test_report_aggregates_task_results,
        test_activity_report,
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

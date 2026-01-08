#!/usr/bin/env python3
"""
Tests for Scheduler (SC-001, SC-002, SC-003)
Tests cron expressions, event triggers, and job history.
"""

import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== SC-001: Cron Expression Tests ==========

def test_cron_parser_basic():
    """Test basic cron expression parsing (SC-001)"""
    from daemon.scheduler import CronParser

    # Every minute
    parser = CronParser("* * * * *")
    assert 0 in parser.fields[0]
    assert 59 in parser.fields[0]

    return True


def test_cron_parser_specific_values():
    """Test cron with specific values (SC-001)"""
    from daemon.scheduler import CronParser

    # At 9:30 AM every day
    parser = CronParser("30 9 * * *")
    assert parser.fields[0] == {30}  # minute
    assert parser.fields[1] == {9}   # hour

    return True


def test_cron_parser_ranges():
    """Test cron with ranges (SC-001)"""
    from daemon.scheduler import CronParser

    # Every minute from 0-10
    parser = CronParser("0-10 * * * *")
    assert parser.fields[0] == set(range(0, 11))

    return True


def test_cron_parser_step():
    """Test cron with step values (SC-001)"""
    from daemon.scheduler import CronParser

    # Every 15 minutes
    parser = CronParser("*/15 * * * *")
    assert parser.fields[0] == {0, 15, 30, 45}

    return True


def test_cron_parser_list():
    """Test cron with list values (SC-001)"""
    from daemon.scheduler import CronParser

    # At minute 0, 15, 30
    parser = CronParser("0,15,30 * * * *")
    assert parser.fields[0] == {0, 15, 30}

    return True


def test_cron_matches():
    """Test cron expression matching (SC-001)"""
    from daemon.scheduler import CronParser

    parser = CronParser("30 9 * * *")

    # Should match 9:30
    matching_dt = datetime(2026, 1, 7, 9, 30)
    assert parser.matches(matching_dt)

    # Should not match 9:31
    non_matching = datetime(2026, 1, 7, 9, 31)
    assert not parser.matches(non_matching)

    return True


def test_cron_next_run():
    """Test next run calculation (SC-001)"""
    from daemon.scheduler import CronParser

    parser = CronParser("0 * * * *")  # Every hour at :00

    now = datetime(2026, 1, 7, 9, 30)
    next_run = parser.next_run(now)

    # Should be 10:00
    assert next_run.hour == 10
    assert next_run.minute == 0

    return True


def test_add_cron_job():
    """Test adding a cron job (SC-001)"""
    from daemon.scheduler import Scheduler

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        job = scheduler.add_cron_job(
            job_id="test-cron",
            name="Test Cron Job",
            action="test_action",
            cron_expr="0 9 * * *"
        )

        assert job.job_id == "test-cron"
        assert job.cron_expr == "0 9 * * *"
        assert job.next_run is not None

        # Verify next run time
        next_run = scheduler.get_next_run_time("test-cron")
        assert next_run is not None
        assert next_run.hour == 9
        assert next_run.minute == 0

        return True


# ========== SC-002: Event Trigger Tests ==========

def test_add_event_job():
    """Test adding an event-triggered job (SC-002)"""
    from daemon.scheduler import Scheduler, EventType

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        job = scheduler.add_event_job(
            job_id="test-event",
            name="Goal Created Handler",
            action="notify",
            event_type=EventType.GOAL_CREATED
        )

        assert job.job_id == "test-event"
        assert job.event_type == EventType.GOAL_CREATED

        # Verify subscription
        subs = scheduler.get_event_subscribers(EventType.GOAL_CREATED)
        assert "test-event" in subs

        return True


def test_event_filter_matches():
    """Test event filter matching (SC-002)"""
    from daemon.scheduler import EventFilter

    # Filter by goal ID
    filter = EventFilter(goal_ids=["goal-123", "goal-456"])

    assert filter.matches({"goal_id": "goal-123"})
    assert filter.matches({"goal_id": "goal-456"})
    assert not filter.matches({"goal_id": "goal-789"})

    return True


def test_event_filter_task_type():
    """Test event filter with task types (SC-002)"""
    from daemon.scheduler import EventFilter

    filter = EventFilter(task_types=["research", "analysis"])

    assert filter.matches({"task_type": "research"})
    assert not filter.matches({"task_type": "digest"})

    return True


def test_event_filter_any_match():
    """Test event filter with any_match mode (SC-002)"""
    from daemon.scheduler import EventFilter

    filter = EventFilter(
        goal_ids=["goal-123"],
        task_types=["research"],
        any_match=True
    )

    # Either condition should match
    assert filter.matches({"goal_id": "goal-123"})
    assert filter.matches({"task_type": "research"})
    assert not filter.matches({"goal_id": "other", "task_type": "other"})

    return True


def test_trigger_event_executes_job():
    """Test that triggering an event executes matching jobs (SC-002)"""
    from daemon.scheduler import Scheduler, EventType

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        # Track executions
        executions = []

        def handler(job_id, context):
            executions.append({"job_id": job_id, "context": context})
            return "done"

        scheduler.register_action_handler("test_action", handler)

        scheduler.add_event_job(
            job_id="test-event",
            name="Test Event Handler",
            action="test_action",
            event_type=EventType.GOAL_CREATED
        )

        # Trigger event
        scheduler.trigger_event(EventType.GOAL_CREATED, {"goal_id": "goal-123"})

        assert len(executions) == 1
        assert executions[0]["job_id"] == "test-event"
        assert executions[0]["context"]["goal_id"] == "goal-123"

        return True


def test_multiple_event_subscribers():
    """Test multiple jobs subscribing to same event (SC-002)"""
    from daemon.scheduler import Scheduler, EventType

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        executions = []

        def handler(job_id, context):
            executions.append(job_id)
            return "done"

        scheduler.register_action_handler("action", handler)

        # Add multiple jobs for same event
        scheduler.add_event_job("job-1", "Job 1", "action", EventType.TASK_COMPLETED)
        scheduler.add_event_job("job-2", "Job 2", "action", EventType.TASK_COMPLETED)
        scheduler.add_event_job("job-3", "Job 3", "action", EventType.TASK_COMPLETED)

        # Trigger event
        scheduler.trigger_event(EventType.TASK_COMPLETED, {})

        assert len(executions) == 3
        assert "job-1" in executions
        assert "job-2" in executions
        assert "job-3" in executions

        return True


# ========== SC-003: Job History Tests ==========

def test_job_execution_recorded():
    """Test that job executions are recorded (SC-003)"""
    from daemon.scheduler import Scheduler, EventType

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        def handler(job_id, context):
            return {"result": "success"}

        scheduler.register_action_handler("action", handler)

        scheduler.add_event_job("test-job", "Test Job", "action", EventType.GOAL_CREATED)
        scheduler.trigger_event(EventType.GOAL_CREATED, {})

        history = scheduler.get_job_history("test-job")

        assert len(history) == 1
        assert history[0].job_id == "test-job"
        assert history[0].status == "completed"
        assert history[0].end_time is not None
        assert history[0].end_time >= history[0].start_time

        return True


def test_failed_job_recorded():
    """Test that failed jobs are recorded with error (SC-003)"""
    from daemon.scheduler import Scheduler, EventType

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        def failing_handler(job_id, context):
            raise ValueError("Test error")

        scheduler.register_action_handler("failing_action", failing_handler)

        scheduler.add_event_job("failing-job", "Failing Job", "failing_action", EventType.GOAL_CREATED)
        scheduler.trigger_event(EventType.GOAL_CREATED, {})

        history = scheduler.get_job_history("failing-job")

        assert len(history) == 1
        assert history[0].status == "failed"
        assert "Test error" in history[0].error

        return True


def test_history_includes_required_fields():
    """Test that history includes all required fields (SC-003)"""
    from daemon.scheduler import Scheduler, EventType

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        def handler(job_id, context):
            return "result"

        scheduler.register_action_handler("action", handler)

        scheduler.add_event_job("test-job", "Test Job", "action", EventType.GOAL_CREATED)
        scheduler.trigger_event(EventType.GOAL_CREATED, {})

        history = scheduler.get_job_history()
        entry = history[0].to_dict()

        # Required fields
        assert "job_id" in entry
        assert "start_time" in entry
        assert "end_time" in entry
        assert "status" in entry
        assert "start_time_iso" in entry
        assert "end_time_iso" in entry
        assert "duration_ms" in entry

        return True


def test_history_persistence():
    """Test that history persists to disk (SC-003)"""
    from daemon.scheduler import Scheduler, EventType

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # First scheduler instance
        scheduler1 = Scheduler(data_dir=data_dir)

        def handler(job_id, context):
            return "done"

        scheduler1.register_action_handler("action", handler)
        scheduler1.add_event_job("test-job", "Test Job", "action", EventType.GOAL_CREATED)
        scheduler1.trigger_event(EventType.GOAL_CREATED, {})

        # Second scheduler instance should load history
        scheduler2 = Scheduler(data_dir=data_dir)
        history = scheduler2.get_job_history()

        assert len(history) >= 1

        return True


def test_history_retention_cleanup():
    """Test that old history is cleaned up (SC-003)"""
    from daemon.scheduler import Scheduler, JobExecution

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir), history_retention_days=1)

        # Add old history entry (2 days old)
        old_execution = JobExecution(
            job_id="old-job",
            start_time=time.time() - (2 * 24 * 60 * 60),  # 2 days ago
            end_time=time.time() - (2 * 24 * 60 * 60) + 1,
            status="completed"
        )
        scheduler._history.append(old_execution)

        # Add recent entry
        recent_execution = JobExecution(
            job_id="recent-job",
            start_time=time.time(),
            end_time=time.time() + 1,
            status="completed"
        )
        scheduler._history.append(recent_execution)

        # Cleanup
        scheduler.cleanup_old_history()

        history = scheduler.get_job_history()
        job_ids = [h.job_id for h in history]

        assert "old-job" not in job_ids
        assert "recent-job" in job_ids

        return True


# ========== Job Management Tests ==========

def test_list_jobs():
    """Test listing all jobs"""
    from daemon.scheduler import Scheduler, EventType

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        scheduler.add_cron_job("cron-1", "Cron 1", "action", "0 * * * *")
        scheduler.add_event_job("event-1", "Event 1", "action", EventType.GOAL_CREATED)

        jobs = scheduler.list_jobs()

        assert len(jobs) == 2
        job_ids = [j.job_id for j in jobs]
        assert "cron-1" in job_ids
        assert "event-1" in job_ids

        return True


def test_remove_job():
    """Test removing a job"""
    from daemon.scheduler import Scheduler

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        scheduler.add_cron_job("to-remove", "To Remove", "action", "0 * * * *")
        assert scheduler.get_job("to-remove") is not None

        scheduler.remove_job("to-remove")
        assert scheduler.get_job("to-remove") is None

        return True


def test_enable_disable_job():
    """Test enabling and disabling jobs"""
    from daemon.scheduler import Scheduler

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        scheduler.add_cron_job("test-job", "Test Job", "action", "0 * * * *")

        job = scheduler.get_job("test-job")
        assert job.enabled is True

        scheduler.enable_job("test-job", False)
        job = scheduler.get_job("test-job")
        assert job.enabled is False

        scheduler.enable_job("test-job", True)
        job = scheduler.get_job("test-job")
        assert job.enabled is True

        return True


if __name__ == "__main__":
    tests = [
        test_cron_parser_basic,
        test_cron_parser_specific_values,
        test_cron_parser_ranges,
        test_cron_parser_step,
        test_cron_parser_list,
        test_cron_matches,
        test_cron_next_run,
        test_add_cron_job,
        test_add_event_job,
        test_event_filter_matches,
        test_event_filter_task_type,
        test_event_filter_any_match,
        test_trigger_event_executes_job,
        test_multiple_event_subscribers,
        test_job_execution_recorded,
        test_failed_job_recorded,
        test_history_includes_required_fields,
        test_history_persistence,
        test_history_retention_cleanup,
        test_list_jobs,
        test_remove_job,
        test_enable_disable_job,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")

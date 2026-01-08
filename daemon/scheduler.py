#!/usr/bin/env python3
"""
Task Scheduler

Handles scheduled and event-based job execution.

Features (SC-001, SC-002, SC-003):
- Full cron expression support
- Event-based triggers
- Job history and execution logs
"""

import time
import logging
import json
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any
import re

logger = logging.getLogger('senter.scheduler')


# SC-001: Cron Expression Parser
class CronParser:
    """
    Parses and evaluates cron expressions.

    Format: minute hour day_of_month month day_of_week
    Special values: * (any), */n (every n), n-m (range), n,m (list)
    """

    FIELD_NAMES = ['minute', 'hour', 'day_of_month', 'month', 'day_of_week']
    FIELD_RANGES = [
        (0, 59),   # minute
        (0, 23),   # hour
        (1, 31),   # day of month
        (1, 12),   # month
        (0, 6),    # day of week (0=Sunday)
    ]

    def __init__(self, expression: str):
        self.expression = expression
        self.fields = self._parse(expression)

    def _parse(self, expression: str) -> list[set[int]]:
        """Parse cron expression into sets of valid values for each field."""
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Cron expression must have 5 fields, got {len(parts)}")

        fields = []
        for i, part in enumerate(parts):
            min_val, max_val = self.FIELD_RANGES[i]
            values = self._parse_field(part, min_val, max_val)
            fields.append(values)

        return fields

    def _parse_field(self, field: str, min_val: int, max_val: int) -> set[int]:
        """Parse a single cron field."""
        values = set()

        for part in field.split(','):
            if part == '*':
                values.update(range(min_val, max_val + 1))
            elif part.startswith('*/'):
                step = int(part[2:])
                values.update(range(min_val, max_val + 1, step))
            elif '-' in part:
                start, end = map(int, part.split('-'))
                values.update(range(start, end + 1))
            else:
                values.add(int(part))

        return values

    def matches(self, dt: datetime = None) -> bool:
        """Check if datetime matches the cron expression."""
        if dt is None:
            dt = datetime.now()

        checks = [
            dt.minute in self.fields[0],
            dt.hour in self.fields[1],
            dt.day in self.fields[2],
            dt.month in self.fields[3],
            dt.weekday() in self.fields[4] or (dt.weekday() + 1) % 7 in self.fields[4],  # Sunday=0 adjustment
        ]
        return all(checks)

    def next_run(self, from_dt: datetime = None) -> datetime:
        """Calculate the next run time after from_dt."""
        if from_dt is None:
            from_dt = datetime.now()

        # Start from next minute
        dt = from_dt.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Search up to 2 years ahead
        max_iterations = 365 * 2 * 24 * 60
        for _ in range(max_iterations):
            if self.matches(dt):
                return dt
            dt += timedelta(minutes=1)

        raise ValueError("Could not find next run time within 2 years")


class TriggerType(Enum):
    CRON = "cron"
    EVENT = "event"
    INTERVAL = "interval"


# SC-002: Event Types
class EventType(Enum):
    GOAL_CREATED = "goal_created"
    TASK_COMPLETED = "task_completed"
    ATTENTION_GAINED = "attention_gained"
    DIGEST_READY = "digest_ready"
    MODEL_RESPONSE = "model_response"


@dataclass
class EventFilter:
    """Filter for event-based triggers."""
    goal_ids: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    any_match: bool = False  # If True, any filter match triggers; if False, all must match

    def matches(self, event_data: dict) -> bool:
        """Check if event data matches the filter."""
        if not self.goal_ids and not self.task_types:
            return True  # No filters = match all

        goal_match = not self.goal_ids or event_data.get("goal_id") in self.goal_ids
        task_match = not self.task_types or event_data.get("task_type") in self.task_types

        if self.any_match:
            return goal_match or task_match
        return goal_match and task_match


# SC-003: Job Execution Record
@dataclass
class JobExecution:
    """Record of a job execution."""
    job_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_time_iso"] = datetime.fromtimestamp(self.start_time).isoformat()
        if self.end_time:
            d["end_time_iso"] = datetime.fromtimestamp(self.end_time).isoformat()
            d["duration_ms"] = int((self.end_time - self.start_time) * 1000)
        return d


@dataclass
class ScheduledJob:
    """A scheduled or event-triggered job."""
    job_id: str
    name: str
    action: str  # Action to execute
    trigger_type: TriggerType
    cron_expr: Optional[str] = None  # For cron triggers
    event_type: Optional[EventType] = None  # For event triggers
    event_filter: Optional[EventFilter] = None  # For event triggers
    interval_seconds: Optional[int] = None  # For interval triggers
    enabled: bool = True
    next_run: Optional[float] = None  # Timestamp of next scheduled run
    last_run: Optional[float] = None
    run_count: int = 0

    def to_dict(self) -> dict:
        d = {
            "job_id": self.job_id,
            "name": self.name,
            "action": self.action,
            "trigger_type": self.trigger_type.value,
            "enabled": self.enabled,
            "run_count": self.run_count,
        }
        if self.cron_expr:
            d["cron_expr"] = self.cron_expr
        if self.event_type:
            d["event_type"] = self.event_type.value
        if self.event_filter:
            d["event_filter"] = asdict(self.event_filter)
        if self.interval_seconds:
            d["interval_seconds"] = self.interval_seconds
        if self.next_run:
            d["next_run"] = datetime.fromtimestamp(self.next_run).isoformat()
        if self.last_run:
            d["last_run"] = datetime.fromtimestamp(self.last_run).isoformat()
        return d


class Scheduler:
    """
    Central job scheduler for Senter.

    Features:
    - SC-001: Full cron expression support
    - SC-002: Event-based triggers
    - SC-003: Job history and execution logs
    """

    DEFAULT_HISTORY_RETENTION_DAYS = 30

    def __init__(
        self,
        data_dir: Path = None,
        history_retention_days: int = None
    ):
        self._data_dir = data_dir or Path("data/scheduler")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._jobs_file = self._data_dir / "jobs.json"
        self._history_file = self._data_dir / "history.json"

        self._history_retention = history_retention_days or self.DEFAULT_HISTORY_RETENTION_DAYS

        # Job storage
        self._jobs: dict[str, ScheduledJob] = {}
        self._lock = threading.Lock()

        # Job history (SC-003)
        self._history: list[JobExecution] = []
        self._history_lock = threading.Lock()

        # Action handlers
        self._action_handlers: dict[str, Callable[[str, dict], Any]] = {}

        # Event subscribers (SC-002)
        self._event_subscribers: dict[EventType, list[str]] = {e: [] for e in EventType}

        # Scheduler thread
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

        # Load persisted state
        self._load_jobs()
        self._load_history()

    # ========== SC-001: Cron Scheduling ==========

    def add_cron_job(
        self,
        job_id: str,
        name: str,
        action: str,
        cron_expr: str,
        enabled: bool = True
    ) -> ScheduledJob:
        """Add a cron-scheduled job."""
        # Validate cron expression
        parser = CronParser(cron_expr)
        next_run = parser.next_run()

        job = ScheduledJob(
            job_id=job_id,
            name=name,
            action=action,
            trigger_type=TriggerType.CRON,
            cron_expr=cron_expr,
            enabled=enabled,
            next_run=next_run.timestamp()
        )

        with self._lock:
            self._jobs[job_id] = job

        self._save_jobs()
        logger.info(f"Added cron job: {name} ({cron_expr})")
        return job

    def get_next_run_time(self, job_id: str) -> Optional[datetime]:
        """Get the next scheduled run time for a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.next_run:
                return datetime.fromtimestamp(job.next_run)
        return None

    def _update_next_run(self, job: ScheduledJob):
        """Update the next run time for a cron job."""
        if job.trigger_type == TriggerType.CRON and job.cron_expr:
            parser = CronParser(job.cron_expr)
            job.next_run = parser.next_run().timestamp()
        elif job.trigger_type == TriggerType.INTERVAL and job.interval_seconds:
            job.next_run = time.time() + job.interval_seconds

    # ========== SC-002: Event-based Triggers ==========

    def add_event_job(
        self,
        job_id: str,
        name: str,
        action: str,
        event_type: EventType,
        event_filter: EventFilter = None,
        enabled: bool = True
    ) -> ScheduledJob:
        """Add an event-triggered job."""
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            action=action,
            trigger_type=TriggerType.EVENT,
            event_type=event_type,
            event_filter=event_filter,
            enabled=enabled
        )

        with self._lock:
            self._jobs[job_id] = job
            self._event_subscribers[event_type].append(job_id)

        self._save_jobs()
        logger.info(f"Added event job: {name} (event={event_type.value})")
        return job

    def trigger_event(self, event_type: EventType, event_data: dict = None):
        """Trigger an event, executing all matching jobs."""
        event_data = event_data or {}
        subscriber_ids = self._event_subscribers.get(event_type, [])

        for job_id in subscriber_ids:
            with self._lock:
                job = self._jobs.get(job_id)
                if not job or not job.enabled:
                    continue

            # Check event filter
            if job.event_filter and not job.event_filter.matches(event_data):
                continue

            logger.info(f"Event {event_type.value} triggered job: {job.name}")
            self._execute_job(job, event_data)

    def get_event_subscribers(self, event_type: EventType) -> list[str]:
        """Get job IDs subscribed to an event type."""
        return self._event_subscribers.get(event_type, [])

    # ========== SC-003: Job History ==========

    def _record_execution_start(self, job: ScheduledJob) -> JobExecution:
        """Record the start of a job execution."""
        execution = JobExecution(
            job_id=job.job_id,
            start_time=time.time()
        )
        with self._history_lock:
            self._history.append(execution)
        return execution

    def _record_execution_end(
        self,
        execution: JobExecution,
        status: str,
        result: Any = None,
        error: str = None
    ):
        """Record the end of a job execution."""
        execution.end_time = time.time()
        execution.status = status
        execution.result = result
        execution.error = error

        self._save_history()

    def get_job_history(self, job_id: str = None, limit: int = 100) -> list[JobExecution]:
        """Get execution history, optionally filtered by job_id."""
        with self._history_lock:
            history = list(self._history)

        if job_id:
            history = [h for h in history if h.job_id == job_id]

        # Sort by start time descending
        history.sort(key=lambda h: h.start_time, reverse=True)
        return history[:limit]

    def cleanup_old_history(self):
        """Remove history older than retention period."""
        cutoff = time.time() - (self._history_retention * 24 * 60 * 60)

        with self._history_lock:
            self._history = [h for h in self._history if h.start_time >= cutoff]

        self._save_history()
        logger.info("Cleaned up old job history")

    # ========== Job Management ==========

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[ScheduledJob]:
        """List all jobs."""
        with self._lock:
            return list(self._jobs.values())

    def remove_job(self, job_id: str) -> bool:
        """Remove a job."""
        removed = False
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs.pop(job_id)
                # Remove from event subscribers
                if job.event_type:
                    subs = self._event_subscribers.get(job.event_type, [])
                    if job_id in subs:
                        subs.remove(job_id)
                removed = True

        if removed:
            self._save_jobs()  # Save outside lock
            logger.info(f"Removed job: {job_id}")
            return True
        return False

    def enable_job(self, job_id: str, enabled: bool = True):
        """Enable or disable a job."""
        changed = False
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].enabled = enabled
                changed = True

        if changed:
            self._save_jobs()  # Save outside lock

    def register_action_handler(self, action: str, handler: Callable[[str, dict], Any]):
        """Register a handler for an action type."""
        self._action_handlers[action] = handler

    def _execute_job(self, job: ScheduledJob, context: dict = None):
        """Execute a job."""
        context = context or {}

        execution = self._record_execution_start(job)

        try:
            handler = self._action_handlers.get(job.action)
            if handler:
                result = handler(job.job_id, context)
                self._record_execution_end(execution, "completed", result=result)
            else:
                logger.warning(f"No handler for action: {job.action}")
                self._record_execution_end(execution, "failed", error=f"No handler: {job.action}")

        except Exception as e:
            logger.error(f"Job execution failed: {e}")
            self._record_execution_end(execution, "failed", error=str(e))

        # Update job stats
        with self._lock:
            job.last_run = time.time()
            job.run_count += 1
            self._update_next_run(job)

        # Save outside lock to avoid deadlock
        self._save_jobs()

    # ========== Scheduler Loop ==========

    def start(self):
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=2.0)
        logger.info("Scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            now = time.time()

            with self._lock:
                jobs = list(self._jobs.values())

            for job in jobs:
                if not job.enabled:
                    continue

                # Check cron and interval jobs
                if job.next_run and now >= job.next_run:
                    logger.info(f"Scheduled job triggered: {job.name}")
                    self._execute_job(job)

            # Check once per minute
            time.sleep(60)

    # ========== Persistence ==========

    def _save_jobs(self):
        """Save jobs to disk."""
        try:
            with self._lock:
                data = [job.to_dict() for job in self._jobs.values()]
            self._jobs_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")

    def _load_jobs(self):
        """Load jobs from disk."""
        try:
            if self._jobs_file.exists():
                data = json.loads(self._jobs_file.read_text())
                for job_dict in data:
                    job = self._dict_to_job(job_dict)
                    self._jobs[job.job_id] = job
                    if job.event_type:
                        self._event_subscribers[job.event_type].append(job.job_id)
                logger.info(f"Loaded {len(self._jobs)} jobs")
        except Exception as e:
            logger.warning(f"Failed to load jobs: {e}")

    def _dict_to_job(self, d: dict) -> ScheduledJob:
        """Convert dict to ScheduledJob."""
        return ScheduledJob(
            job_id=d["job_id"],
            name=d["name"],
            action=d["action"],
            trigger_type=TriggerType(d["trigger_type"]),
            cron_expr=d.get("cron_expr"),
            event_type=EventType(d["event_type"]) if d.get("event_type") else None,
            event_filter=EventFilter(**d["event_filter"]) if d.get("event_filter") else None,
            interval_seconds=d.get("interval_seconds"),
            enabled=d.get("enabled", True),
            run_count=d.get("run_count", 0),
        )

    def _save_history(self):
        """Save history to disk."""
        try:
            with self._history_lock:
                data = [h.to_dict() for h in self._history[-1000:]]  # Keep last 1000
            self._history_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def _load_history(self):
        """Load history from disk."""
        try:
            if self._history_file.exists():
                data = json.loads(self._history_file.read_text())
                for h in data:
                    execution = JobExecution(
                        job_id=h["job_id"],
                        start_time=h["start_time"],
                        end_time=h.get("end_time"),
                        status=h.get("status", "unknown"),
                        result=h.get("result"),
                        error=h.get("error")
                    )
                    self._history.append(execution)
                logger.info(f"Loaded {len(self._history)} history entries")
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")


# Convenience function for CLI
def create_scheduler(data_dir: Path = None) -> Scheduler:
    """Create a scheduler instance."""
    return Scheduler(data_dir=data_dir)


# CLI test
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(data_dir=Path(tmpdir))

        # Register action handler
        def digest_action(job_id, context):
            print(f"Executing digest for job {job_id}")
            return {"status": "generated"}

        scheduler.register_action_handler("generate_digest", digest_action)

        # Add a cron job (every minute for testing)
        job = scheduler.add_cron_job(
            job_id="daily-digest",
            name="Daily Digest",
            action="generate_digest",
            cron_expr="* * * * *"  # Every minute
        )
        print(f"Added job: {job.name}")
        print(f"Next run: {scheduler.get_next_run_time(job.job_id)}")

        # Add event job
        event_job = scheduler.add_event_job(
            job_id="goal-notifier",
            name="Goal Created Notifier",
            action="generate_digest",
            event_type=EventType.GOAL_CREATED,
            event_filter=EventFilter(goal_ids=["goal-123"])
        )
        print(f"Added event job: {event_job.name}")

        # Trigger event
        scheduler.trigger_event(EventType.GOAL_CREATED, {"goal_id": "goal-123"})

        # Check history
        history = scheduler.get_job_history()
        print(f"History entries: {len(history)}")
        for h in history:
            print(f"  {h.job_id}: {h.status}")

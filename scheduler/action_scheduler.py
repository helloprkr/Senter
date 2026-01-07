#!/usr/bin/env python3
"""
Action Scheduler

Cron-like scheduler for background autonomous tasks.

Features:
- Time-based triggers (cron expressions, intervals)
- Event-based triggers
- Persistent job storage
- Multiple job types
"""

import json
import time
import logging
import sys
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable
from pathlib import Path
from multiprocessing import Event
from queue import Empty

sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon.message_bus import MessageBus, MessageType, Message

logger = logging.getLogger('senter.scheduler')


class TriggerType(Enum):
    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"
    EVENT = "event"


class JobStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ScheduledJob:
    """A scheduled job"""
    id: str
    name: str
    description: str
    trigger_type: TriggerType
    trigger_config: dict
    job_type: str  # research, digest, check, custom
    job_params: dict = field(default_factory=dict)
    status: JobStatus = JobStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    last_run: Optional[float] = None
    next_run: Optional[float] = None
    run_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger_type": self.trigger_type.value,
            "trigger_config": self.trigger_config,
            "job_type": self.job_type,
            "job_params": self.job_params,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "run_count": self.run_count
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledJob":
        d = d.copy()
        d["trigger_type"] = TriggerType(d["trigger_type"])
        d["status"] = JobStatus(d["status"])
        return cls(**d)


class ActionScheduler:
    """
    Manages scheduled background tasks.

    Responsibilities:
    - Load/save jobs from persistent storage
    - Check job triggers
    - Fire jobs to task engine
    - Track job history
    """

    def __init__(
        self,
        check_interval: int,
        message_bus: MessageBus,
        shutdown_event: Event,
        senter_root: Path
    ):
        self.check_interval = check_interval
        self.message_bus = message_bus
        self.shutdown_event = shutdown_event
        self.senter_root = Path(senter_root)

        # Jobs
        self.jobs: dict[str, ScheduledJob] = {}

        # Storage
        self.jobs_file = self.senter_root / "data" / "scheduler" / "jobs.json"

        # Message queue
        self._queue = None

    def run(self):
        """Main scheduler loop"""
        logger.info("Scheduler starting...")

        # Load saved jobs
        self._load_jobs()

        # Register with message bus
        self._queue = self.message_bus.register("scheduler")

        # Create default jobs if none exist
        if not self.jobs:
            self._create_default_jobs()

        logger.info(f"Scheduler started with {len(self.jobs)} jobs")

        last_check = 0

        while not self.shutdown_event.is_set():
            try:
                # Process messages
                self._process_messages()

                # Check triggers periodically
                now = time.time()
                if now - last_check >= self.check_interval:
                    self._check_triggers()
                    last_check = now

                time.sleep(1)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")

        self._save_jobs()
        logger.info("Scheduler stopped")

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
        """Handle a message"""
        payload = message.payload

        if message.type == MessageType.SCHEDULE_JOB:
            # Add new job
            self._add_job_from_payload(payload)

        elif message.type == MessageType.CANCEL_JOB:
            # Cancel a job
            job_id = payload.get("job_id")
            if job_id in self.jobs:
                self.jobs[job_id].status = JobStatus.COMPLETED
                self._save_jobs()

    def _check_triggers(self):
        """Check all job triggers"""
        now = time.time()

        for job in list(self.jobs.values()):
            if job.status != JobStatus.ACTIVE:
                continue

            # Calculate next run if not set
            if job.next_run is None:
                job.next_run = self._calculate_next_run(job)

            # Check if should run
            if job.next_run and now >= job.next_run:
                self._trigger_job(job)

    def _trigger_job(self, job: ScheduledJob):
        """Trigger a job to run"""
        logger.info(f"Triggering job: {job.name}")

        job.last_run = time.time()
        job.run_count += 1

        # Send to task engine
        self.message_bus.send(
            MessageType.JOB_TRIGGERED,
            source="scheduler",
            target="task_engine",
            payload={
                "job_id": job.id,
                "job_type": job.job_type,
                "description": job.description,
                **job.job_params
            }
        )

        # Log activity
        self.message_bus.send(
            MessageType.ACTIVITY_LOG,
            source="scheduler",
            payload={
                "activity_type": "job_triggered",
                "details": {
                    "job_id": job.id,
                    "job_name": job.name,
                    "run_count": job.run_count
                },
                "timestamp": time.time()
            }
        )

        # Calculate next run
        if job.trigger_type == TriggerType.ONCE:
            job.status = JobStatus.COMPLETED
        else:
            job.next_run = self._calculate_next_run(job)

        self._save_jobs()

    def _calculate_next_run(self, job: ScheduledJob) -> Optional[float]:
        """Calculate next run time for a job"""
        now = time.time()
        config = job.trigger_config

        if job.trigger_type == TriggerType.INTERVAL:
            # Simple interval in seconds
            interval = config.get("seconds", 3600)
            if job.last_run:
                return job.last_run + interval
            return now + interval

        elif job.trigger_type == TriggerType.CRON:
            # Simplified cron: just hour and minute
            hour = config.get("hour", 9)
            minute = config.get("minute", 0)

            # Find next occurrence
            current = datetime.now()
            target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)

            if target <= current:
                target += timedelta(days=1)

            return target.timestamp()

        elif job.trigger_type == TriggerType.ONCE:
            # One-time at specific timestamp
            return config.get("timestamp", now)

        return None

    def _add_job_from_payload(self, payload: dict):
        """Add a new job from message payload"""
        job_id = payload.get("id") or f"job_{int(time.time())}"

        job = ScheduledJob(
            id=job_id,
            name=payload.get("name", "Unnamed Job"),
            description=payload.get("description", ""),
            trigger_type=TriggerType(payload.get("trigger_type", "interval")),
            trigger_config=payload.get("trigger_config", {}),
            job_type=payload.get("job_type", "custom"),
            job_params=payload.get("job_params", {})
        )

        self.jobs[job.id] = job
        self._save_jobs()

        logger.info(f"Added job: {job.name}")

    def _create_default_jobs(self):
        """Create default scheduled jobs"""
        # Daily digest at 9 AM
        self.jobs["daily_digest"] = ScheduledJob(
            id="daily_digest",
            name="Daily Digest",
            description="Generate daily activity summary",
            trigger_type=TriggerType.CRON,
            trigger_config={"hour": 9, "minute": 0},
            job_type="digest"
        )

        # Hourly goal check
        self.jobs["goal_check"] = ScheduledJob(
            id="goal_check",
            name="Goal Check",
            description="Review and update active goals",
            trigger_type=TriggerType.INTERVAL,
            trigger_config={"seconds": 3600},  # Every hour
            job_type="check",
            job_params={"check_type": "goals"}
        )

        self._save_jobs()
        logger.info("Created default jobs")

    def _load_jobs(self):
        """Load jobs from storage"""
        if self.jobs_file.exists():
            try:
                data = json.loads(self.jobs_file.read_text())
                for job_data in data.get("jobs", []):
                    job = ScheduledJob.from_dict(job_data)
                    self.jobs[job.id] = job
                logger.info(f"Loaded {len(self.jobs)} jobs")
            except Exception as e:
                logger.warning(f"Could not load jobs: {e}")

    def _save_jobs(self):
        """Save jobs to storage"""
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "jobs": [job.to_dict() for job in self.jobs.values()]
        }
        self.jobs_file.write_text(json.dumps(data, indent=2))

    def add_job(
        self,
        name: str,
        description: str,
        trigger_type: TriggerType,
        trigger_config: dict,
        job_type: str,
        job_params: dict = None
    ) -> str:
        """Add a new job (for programmatic use)"""
        job_id = f"job_{int(time.time())}"

        job = ScheduledJob(
            id=job_id,
            name=name,
            description=description,
            trigger_type=trigger_type,
            trigger_config=trigger_config,
            job_type=job_type,
            job_params=job_params or {}
        )

        self.jobs[job.id] = job
        self._save_jobs()

        return job_id

    def get_jobs(self) -> list[dict]:
        """Get all jobs"""
        return [job.to_dict() for job in self.jobs.values()]


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from daemon.message_bus import MessageBus

    bus = MessageBus()
    bus.start()

    shutdown = Event()

    scheduler = ActionScheduler(
        check_interval=5,
        message_bus=bus,
        shutdown_event=shutdown,
        senter_root=Path(__file__).parent.parent
    )

    # Load jobs
    scheduler._load_jobs()
    if not scheduler.jobs:
        scheduler._create_default_jobs()

    print(f"\nScheduled jobs ({len(scheduler.jobs)}):")
    for job in scheduler.jobs.values():
        print(f"  - {job.name}: {job.description}")
        print(f"    Type: {job.trigger_type.value}, Config: {job.trigger_config}")
        if job.next_run:
            next_run = datetime.fromtimestamp(job.next_run)
            print(f"    Next run: {next_run}")

    bus.stop()
    print("\nTest complete")

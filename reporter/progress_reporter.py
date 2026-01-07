#!/usr/bin/env python3
"""
Progress Reporter

Surfaces what Senter accomplished while user was away.

Features:
- Activity logging
- Session summaries
- Daily digests
- Desktop notifications (optional)
"""

import json
import time
import logging
import sys
import subprocess
import platform
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from multiprocessing import Event
from queue import Empty

sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon.message_bus import MessageBus, MessageType, Message

logger = logging.getLogger('senter.reporter')


@dataclass
class ActivityEntry:
    """A single activity log entry"""
    activity_type: str
    timestamp: float
    details: dict
    source: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "activity_type": self.activity_type,
            "timestamp": self.timestamp,
            "details": self.details,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActivityEntry":
        return cls(**d)


class ActivityLog:
    """
    Persistent activity log storage.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current day's log file
        self._current_file = None
        self._current_date = None

    def _get_log_file(self) -> Path:
        """Get today's log file"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._current_date != today:
            self._current_date = today
            self._current_file = self.log_dir / f"activity_{today}.json"
        return self._current_file

    def log(self, entry: ActivityEntry):
        """Log an activity entry"""
        log_file = self._get_log_file()

        # Load existing entries
        entries = []
        if log_file.exists():
            try:
                entries = json.loads(log_file.read_text())
            except:
                pass

        # Add new entry
        entries.append(entry.to_dict())

        # Save
        log_file.write_text(json.dumps(entries, indent=2))

    def get_entries(
        self,
        since: float = None,
        activity_type: str = None,
        limit: int = 100
    ) -> list[ActivityEntry]:
        """Get activity entries"""
        all_entries = []

        # Read from log files
        for log_file in sorted(self.log_dir.glob("activity_*.json"), reverse=True):
            try:
                entries = json.loads(log_file.read_text())
                for e in entries:
                    entry = ActivityEntry.from_dict(e)

                    # Filter by time
                    if since and entry.timestamp < since:
                        continue

                    # Filter by type
                    if activity_type and entry.activity_type != activity_type:
                        continue

                    all_entries.append(entry)

                    if len(all_entries) >= limit:
                        break
            except:
                continue

            if len(all_entries) >= limit:
                break

        return all_entries

    def get_summary(self, since: float = None) -> dict:
        """Get activity summary"""
        entries = self.get_entries(since=since, limit=1000)

        # Count by type
        by_type = {}
        for entry in entries:
            t = entry.activity_type
            by_type[t] = by_type.get(t, 0) + 1

        # Get recent goals
        recent_goals = [
            e for e in entries
            if e.activity_type in ("goal_completed", "task_completed")
        ][:5]

        return {
            "total_activities": len(entries),
            "by_type": by_type,
            "recent_goals": [e.details for e in recent_goals],
            "time_range": {
                "since": datetime.fromtimestamp(since).isoformat() if since else None,
                "until": datetime.now().isoformat()
            }
        }


class DigestGenerator:
    """
    Generates activity digests (summaries).
    """

    def __init__(self, activity_log: ActivityLog):
        self.activity_log = activity_log

    def generate_daily_digest(self, date: datetime = None) -> str:
        """Generate a daily digest"""
        if date is None:
            date = datetime.now()

        # Get yesterday's start/end
        start = datetime.combine(date - timedelta(days=1), datetime.min.time())
        end = datetime.combine(date, datetime.min.time())

        summary = self.activity_log.get_summary(since=start.timestamp())

        # Build digest
        lines = [
            f"📊 Daily Digest - {start.strftime('%B %d, %Y')}",
            "=" * 40,
            "",
        ]

        if summary["total_activities"] == 0:
            lines.append("No significant activity yesterday.")
        else:
            lines.append(f"Total activities: {summary['total_activities']}")
            lines.append("")

            # Activity breakdown
            if summary["by_type"]:
                lines.append("Activity breakdown:")
                for activity_type, count in sorted(summary["by_type"].items(),
                                                   key=lambda x: x[1], reverse=True):
                    emoji = self._get_emoji(activity_type)
                    lines.append(f"  {emoji} {activity_type}: {count}")
                lines.append("")

            # Recent accomplishments
            if summary["recent_goals"]:
                lines.append("Completed:")
                for goal in summary["recent_goals"]:
                    desc = goal.get("description", "Task")[:50]
                    lines.append(f"  ✓ {desc}")
                lines.append("")

        return "\n".join(lines)

    def generate_session_summary(self, session_start: float) -> str:
        """Generate a session summary"""
        summary = self.activity_log.get_summary(since=session_start)

        lines = [
            "📝 Session Summary",
            "-" * 30,
        ]

        if summary["total_activities"] == 0:
            lines.append("No activities this session.")
        else:
            # Brief overview
            completed = summary["by_type"].get("task_completed", 0)
            goals = summary["by_type"].get("goal_completed", 0)

            if completed > 0:
                lines.append(f"✓ {completed} tasks completed")
            if goals > 0:
                lines.append(f"🎯 {goals} goals achieved")

        return "\n".join(lines)

    def _get_emoji(self, activity_type: str) -> str:
        """Get emoji for activity type"""
        emojis = {
            "task_completed": "✓",
            "task_failed": "✗",
            "goal_completed": "🎯",
            "plan_created": "📋",
            "job_triggered": "⏰",
            "query": "💬",
            "research": "🔍",
        }
        return emojis.get(activity_type, "•")


class Notifier:
    """
    Desktop notification sender.
    Cross-platform support.
    """

    def __init__(self):
        self.system = platform.system()

    def notify(self, title: str, message: str, sound: bool = True):
        """Send a desktop notification"""
        try:
            if self.system == "Darwin":  # macOS
                self._notify_macos(title, message, sound)
            elif self.system == "Linux":
                self._notify_linux(title, message)
            elif self.system == "Windows":
                self._notify_windows(title, message)
            else:
                logger.warning(f"Notifications not supported on {self.system}")
        except Exception as e:
            logger.warning(f"Notification failed: {e}")

    def _notify_macos(self, title: str, message: str, sound: bool):
        """Send notification on macOS"""
        script = f'''
        display notification "{message}" with title "{title}"
        '''
        if sound:
            script += ' sound name "default"'
        subprocess.run(["osascript", "-e", script], capture_output=True)

    def _notify_linux(self, title: str, message: str):
        """Send notification on Linux"""
        subprocess.run(["notify-send", title, message], capture_output=True)

    def _notify_windows(self, title: str, message: str):
        """Send notification on Windows"""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(title, message, duration=5)
        except ImportError:
            logger.warning("win10toast not installed for Windows notifications")


class ProgressReporter:
    """
    Main progress reporter service.
    """

    def __init__(
        self,
        digest_hour: int,
        notifications: bool,
        message_bus: MessageBus,
        shutdown_event: Event,
        senter_root: Path
    ):
        self.digest_hour = digest_hour
        self.notifications_enabled = notifications
        self.message_bus = message_bus
        self.shutdown_event = shutdown_event
        self.senter_root = Path(senter_root)

        # Components
        log_dir = self.senter_root / "data" / "progress" / "activity"
        self.activity_log = ActivityLog(log_dir)
        self.digest_generator = DigestGenerator(self.activity_log)
        self.notifier = Notifier() if notifications else None

        # State
        self.last_digest_date = None
        self._queue = None

    def run(self):
        """Main service loop"""
        logger.info("Progress reporter starting...")

        # Register with message bus
        self._queue = self.message_bus.register("reporter")

        logger.info("Progress reporter started")

        while not self.shutdown_event.is_set():
            try:
                # Process messages
                self._process_messages()

                # Check for daily digest
                self._check_daily_digest()

                time.sleep(1)

            except Exception as e:
                logger.error(f"Reporter error: {e}")

        logger.info("Progress reporter stopped")

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

        if message.type == MessageType.ACTIVITY_LOG:
            # Log activity
            entry = ActivityEntry(
                activity_type=payload.get("activity_type", "unknown"),
                timestamp=payload.get("timestamp", time.time()),
                details=payload.get("details", {}),
                source=message.source
            )
            self.activity_log.log(entry)

        elif message.type == MessageType.DIGEST_REQUEST:
            # Generate and return digest
            digest = self.digest_generator.generate_daily_digest()

            self.message_bus.send(
                MessageType.DIGEST_READY,
                source="reporter",
                target=message.source,
                payload={"digest": digest},
                correlation_id=message.correlation_id
            )

        elif message.type == MessageType.TASK_COMPLETE:
            # Log task completion
            entry = ActivityEntry(
                activity_type="task_completed",
                timestamp=time.time(),
                details=payload,
                source=message.source
            )
            self.activity_log.log(entry)

            # Maybe notify
            if self.notifications_enabled and self.notifier:
                desc = payload.get("description", "Task")[:50]
                self.notifier.notify("Senter", f"Completed: {desc}", sound=False)

    def _check_daily_digest(self):
        """Check if it's time for daily digest"""
        now = datetime.now()
        today = now.date()

        # Only generate once per day, at the specified hour
        if (self.last_digest_date != today and
            now.hour >= self.digest_hour):

            self.last_digest_date = today

            # Generate digest
            digest = self.digest_generator.generate_daily_digest()

            # Log it
            logger.info("Generated daily digest")

            # Notify if enabled
            if self.notifications_enabled and self.notifier:
                summary = self.activity_log.get_summary(
                    since=(datetime.now() - timedelta(days=1)).timestamp()
                )
                total = summary["total_activities"]
                self.notifier.notify(
                    "Senter Daily Digest",
                    f"{total} activities yesterday. Check /progress for details."
                )

            # Save digest to file
            digest_dir = self.senter_root / "data" / "progress" / "digests"
            digest_dir.mkdir(parents=True, exist_ok=True)
            digest_file = digest_dir / f"digest_{today.isoformat()}.txt"
            digest_file.write_text(digest)

    def get_progress(self, hours: int = 24) -> str:
        """Get progress report for CLI"""
        since = time.time() - (hours * 3600)
        summary = self.activity_log.get_summary(since=since)

        lines = [
            f"📊 Progress Report (last {hours}h)",
            "=" * 40,
            f"Total activities: {summary['total_activities']}",
            ""
        ]

        if summary["by_type"]:
            lines.append("Breakdown:")
            for activity_type, count in sorted(summary["by_type"].items(),
                                               key=lambda x: x[1], reverse=True):
                lines.append(f"  • {activity_type}: {count}")

        if summary["recent_goals"]:
            lines.append("")
            lines.append("Recent completions:")
            for goal in summary["recent_goals"]:
                desc = goal.get("description", "Task")[:60]
                lines.append(f"  ✓ {desc}")

        return "\n".join(lines)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test activity log
    log_dir = Path(__file__).parent.parent / "data" / "progress" / "test_activity"
    activity_log = ActivityLog(log_dir)

    # Log some test activities
    activity_log.log(ActivityEntry(
        activity_type="task_completed",
        timestamp=time.time(),
        details={"description": "Research AI trends for investor deck"},
        source="task_engine"
    ))

    activity_log.log(ActivityEntry(
        activity_type="goal_completed",
        timestamp=time.time(),
        details={"description": "Prepare weekly report", "tasks_completed": 3},
        source="task_engine"
    ))

    # Test digest
    digest_gen = DigestGenerator(activity_log)
    digest = digest_gen.generate_daily_digest()
    print("\n" + digest)

    # Test notification
    notifier = Notifier()
    print("\nSending test notification...")
    notifier.notify("Senter Test", "This is a test notification")

    print("\nTest complete")

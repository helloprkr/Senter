#!/usr/bin/env python3
"""
Journal Database (P2-004)

Auto-generated daily journal entries summarizing Senter activity.

Database: data/learning/journal.db
Table: journal_entries
Fields: date, summary, conversations_count, tasks_count, topics, created_at
"""

import json
import time
import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger('senter.journal_db')


@dataclass
class JournalEntry:
    """A daily journal entry"""
    date: str  # YYYY-MM-DD format
    summary: str
    conversations_count: int = 0
    tasks_count: int = 0
    topics: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "summary": self.summary,
            "conversations_count": self.conversations_count,
            "tasks_count": self.tasks_count,
            "topics": self.topics,
            "created_at": self.created_at,
            "created_at_str": datetime.fromtimestamp(self.created_at).isoformat()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JournalEntry":
        return cls(
            date=d["date"],
            summary=d.get("summary", ""),
            conversations_count=d.get("conversations_count", 0),
            tasks_count=d.get("tasks_count", 0),
            topics=d.get("topics", []),
            created_at=d.get("created_at", time.time())
        )


class JournalDB:
    """
    SQLite database for daily journal entries (P2-004).

    Auto-generates summaries from user events and task completions.
    """

    def __init__(self, senter_root: Path):
        """Initialize database.

        Args:
            senter_root: Root directory (will use data/learning/journal.db)
        """
        self.senter_root = Path(senter_root)
        self.db_path = self.senter_root / "data" / "learning" / "journal.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection"""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        """Initialize database schema"""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                summary TEXT NOT NULL,
                conversations_count INTEGER DEFAULT 0,
                tasks_count INTEGER DEFAULT 0,
                topics TEXT DEFAULT '[]',
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_journal_date ON journal_entries(date DESC);
        """)
        conn.commit()

    def get_entry(self, entry_date: str) -> Optional[JournalEntry]:
        """Get journal entry for a specific date"""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM journal_entries WHERE date = ?",
            (entry_date,)
        )
        row = cursor.fetchone()
        if row:
            return JournalEntry(
                date=row["date"],
                summary=row["summary"],
                conversations_count=row["conversations_count"],
                tasks_count=row["tasks_count"],
                topics=json.loads(row["topics"]),
                created_at=row["created_at"]
            )
        return None

    def get_recent_entries(self, limit: int = 7) -> List[JournalEntry]:
        """Get recent journal entries"""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM journal_entries ORDER BY date DESC LIMIT ?",
            (limit,)
        )
        entries = []
        for row in cursor:
            entries.append(JournalEntry(
                date=row["date"],
                summary=row["summary"],
                conversations_count=row["conversations_count"],
                tasks_count=row["tasks_count"],
                topics=json.loads(row["topics"]),
                created_at=row["created_at"]
            ))
        return entries

    def save_entry(self, entry: JournalEntry) -> bool:
        """Save or update a journal entry"""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO journal_entries
                (date, summary, conversations_count, tasks_count, topics, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entry.date,
                entry.summary,
                entry.conversations_count,
                entry.tasks_count,
                json.dumps(entry.topics),
                entry.created_at
            ))
            conn.commit()
            logger.info(f"Saved journal entry for {entry.date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save journal entry: {e}")
            return False

    def generate_entry_for_date(self, target_date: str = None) -> JournalEntry:
        """Generate journal entry from events data (P2-004)"""
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")

        # Parse target date
        date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        start_ts = datetime(date_obj.year, date_obj.month, date_obj.day).timestamp()
        end_ts = start_ts + 86400  # 24 hours

        # Get activity data
        conversations_count, topics = self._count_conversations(start_ts, end_ts)
        tasks_count = self._count_tasks(start_ts, end_ts)

        # Generate summary
        summary = self._generate_summary(target_date, conversations_count, tasks_count, topics)

        entry = JournalEntry(
            date=target_date,
            summary=summary,
            conversations_count=conversations_count,
            tasks_count=tasks_count,
            topics=topics[:5]  # Top 5 topics
        )

        return entry

    def _count_conversations(self, start_ts: float, end_ts: float) -> tuple:
        """Count conversations and extract topics from events db"""
        try:
            events_db_path = self.senter_root / "data" / "learning" / "events.db"
            if not events_db_path.exists():
                return 0, []

            conn = sqlite3.connect(str(events_db_path))
            conn.row_factory = sqlite3.Row

            # Count queries
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM user_events
                WHERE event_type = 'query'
                AND timestamp >= ? AND timestamp < ?
            """, (start_ts, end_ts))
            count = cursor.fetchone()["count"]

            # Extract topics
            cursor = conn.execute("""
                SELECT metadata FROM user_events
                WHERE event_type = 'query'
                AND timestamp >= ? AND timestamp < ?
            """, (start_ts, end_ts))

            topic_counts = {}
            for row in cursor:
                try:
                    metadata = json.loads(row["metadata"])
                    topic = metadata.get("topic", "general")
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                except:
                    pass

            # Sort topics by frequency
            topics = sorted(topic_counts.keys(), key=lambda t: topic_counts[t], reverse=True)

            conn.close()
            return count, topics

        except Exception as e:
            logger.warning(f"Failed to count conversations: {e}")
            return 0, []

    def _count_tasks(self, start_ts: float, end_ts: float) -> int:
        """Count completed tasks from task results"""
        try:
            results_dir = self.senter_root / "data" / "tasks" / "results"
            index_file = results_dir / "index.json"

            if not index_file.exists():
                return 0

            index = json.loads(index_file.read_text())
            count = 0

            for task_info in index.get("tasks", {}).values():
                ts = task_info.get("timestamp", 0)
                if start_ts <= ts < end_ts:
                    count += 1

            return count

        except Exception as e:
            logger.warning(f"Failed to count tasks: {e}")
            return 0

    def _generate_summary(self, date_str: str, conv_count: int, task_count: int, topics: list) -> str:
        """Generate human-readable summary"""
        parts = []

        # Date header
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%B %d, %Y")
        parts.append(f"**{formatted_date}**")
        parts.append("")

        # Activity summary
        if conv_count == 0 and task_count == 0:
            parts.append("No activity recorded today.")
        else:
            if conv_count > 0:
                conv_word = "conversation" if conv_count == 1 else "conversations"
                parts.append(f"Had {conv_count} {conv_word}.")

            if task_count > 0:
                task_word = "task" if task_count == 1 else "tasks"
                parts.append(f"Completed {task_count} research {task_word}.")

            if topics:
                topic_str = ", ".join(topics[:3])
                parts.append(f"Topics discussed: {topic_str}.")

        return "\n".join(parts)


# Test
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        journal = JournalDB(Path(tmpdir))

        # Test entry
        entry = JournalEntry(
            date="2026-01-08",
            summary="Had 5 conversations about coding, research. Completed 2 tasks.",
            conversations_count=5,
            tasks_count=2,
            topics=["coding", "research"]
        )
        journal.save_entry(entry)

        # Retrieve
        retrieved = journal.get_entry("2026-01-08")
        print(f"Retrieved: {retrieved.to_dict()}")

        # Recent
        recent = journal.get_recent_entries(7)
        print(f"Recent entries: {len(recent)}")

        print("\nTest complete")

#!/usr/bin/env python3
"""
User Events Database (US-007)

Structured storage for user interaction events with proper schema
for behavioral analysis and pattern detection.

Database: data/learning/events.db
Table: user_events
Fields: timestamp, event_type, context, metadata
"""

import json
import time
import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any, List
from pathlib import Path

logger = logging.getLogger('senter.events_db')


@dataclass
class UserEvent:
    """A user interaction event"""
    event_type: str  # query, response, topic_detection, etc.
    timestamp: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)  # query text, response, session info
    metadata: dict = field(default_factory=dict)  # topic, time_of_day, latency, etc.

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "context": self.context,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserEvent":
        return cls(
            event_type=d["event_type"],
            timestamp=d.get("timestamp", time.time()),
            context=d.get("context", {}),
            metadata=d.get("metadata", {})
        )


class UserEventsDB:
    """
    SQLite database for user interaction events (US-007).

    Schema:
    - timestamp: REAL (Unix timestamp)
    - event_type: TEXT (query, response, topic_detection, etc.)
    - context: TEXT (JSON - query text, response, session info)
    - metadata: TEXT (JSON - topic, time_of_day, latency, etc.)
    """

    def __init__(self, db_path: Path = None, senter_root: Path = None):
        """Initialize database.

        Args:
            db_path: Direct path to database file
            senter_root: Root directory (will use data/learning/events.db)
        """
        if db_path:
            self.db_path = Path(db_path)
        elif senter_root:
            self.db_path = Path(senter_root) / "data" / "learning" / "events.db"
        else:
            self.db_path = Path(__file__).parent.parent / "data" / "learning" / "events.db"

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
            -- Main user_events table (US-007)
            CREATE TABLE IF NOT EXISTS user_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '{}',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );

            -- Indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON user_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_type ON user_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_type_time ON user_events(event_type, timestamp);
        """)
        conn.commit()
        logger.info(f"Events database initialized at {self.db_path}")

    def log_event(self, event: UserEvent) -> int:
        """Log a user event to the database.

        Returns:
            The ID of the inserted event
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO user_events (timestamp, event_type, context, metadata)
               VALUES (?, ?, ?, ?)""",
            (
                event.timestamp,
                event.event_type,
                json.dumps(event.context),
                json.dumps(event.metadata)
            )
        )
        conn.commit()
        return cursor.lastrowid

    def log_query(self, query: str, topic: str = None, session_id: str = None) -> int:
        """Convenience method to log a user query event."""
        now = time.time()
        hour = datetime.fromtimestamp(now).hour

        # Determine time of day
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        event = UserEvent(
            event_type="query",
            timestamp=now,
            context={
                "query": query,
                "session_id": session_id
            },
            metadata={
                "topic": topic,
                "time_of_day": time_of_day,
                "hour": hour,
                "query_length": len(query)
            }
        )
        return self.log_event(event)

    def log_response(self, query: str, response: str, latency_ms: int = 0,
                     worker: str = None, topic: str = None) -> int:
        """Convenience method to log a response event."""
        now = time.time()
        hour = datetime.fromtimestamp(now).hour

        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        event = UserEvent(
            event_type="response",
            timestamp=now,
            context={
                "query": query,
                "response_preview": response[:500] if response else "",
                "response_length": len(response) if response else 0
            },
            metadata={
                "topic": topic,
                "time_of_day": time_of_day,
                "latency_ms": latency_ms,
                "worker": worker
            }
        )
        return self.log_event(event)

    def get_events(
        self,
        event_type: str = None,
        since: float = None,
        until: float = None,
        limit: int = 100
    ) -> List[UserEvent]:
        """Query events by time range and optionally type.

        Args:
            event_type: Filter by event type (query, response, etc.)
            since: Start timestamp (inclusive)
            until: End timestamp (inclusive)
            limit: Maximum number of results

        Returns:
            List of UserEvent objects, most recent first
        """
        conn = self._get_conn()

        query = "SELECT * FROM user_events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)

        if until is not None:
            query += " AND timestamp <= ?"
            params.append(until)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        events = []
        for row in cursor.fetchall():
            events.append(UserEvent(
                event_type=row["event_type"],
                timestamp=row["timestamp"],
                context=json.loads(row["context"]),
                metadata=json.loads(row["metadata"])
            ))
        return events

    def get_events_by_time_range(self, hours: int = 24) -> List[UserEvent]:
        """Get events from the last N hours."""
        since = time.time() - (hours * 3600)
        return self.get_events(since=since, limit=1000)

    def get_queries(self, hours: int = 24) -> List[UserEvent]:
        """Get query events from the last N hours."""
        since = time.time() - (hours * 3600)
        return self.get_events(event_type="query", since=since)

    def get_event_counts(self, hours: int = 24) -> dict:
        """Get event counts by type for the last N hours."""
        conn = self._get_conn()
        since = time.time() - (hours * 3600)

        cursor = conn.execute(
            """SELECT event_type, COUNT(*) as count
               FROM user_events
               WHERE timestamp >= ?
               GROUP BY event_type
               ORDER BY count DESC""",
            (since,)
        )
        return {row["event_type"]: row["count"] for row in cursor.fetchall()}

    def get_stats(self) -> dict:
        """Get overall database statistics."""
        conn = self._get_conn()

        # Total events
        cursor = conn.execute("SELECT COUNT(*) as total FROM user_events")
        total = cursor.fetchone()["total"]

        # Events by type
        cursor = conn.execute(
            """SELECT event_type, COUNT(*) as count
               FROM user_events
               GROUP BY event_type"""
        )
        by_type = {row["event_type"]: row["count"] for row in cursor.fetchall()}

        # Time range
        cursor = conn.execute(
            "SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest FROM user_events"
        )
        row = cursor.fetchone()
        earliest = row["earliest"]
        latest = row["latest"]

        return {
            "total_events": total,
            "events_by_type": by_type,
            "earliest_event": datetime.fromtimestamp(earliest).isoformat() if earliest else None,
            "latest_event": datetime.fromtimestamp(latest).isoformat() if latest else None,
            "database_path": str(self.db_path)
        }

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserEventsDB(db_path=Path(tmpdir) / "test_events.db")

        # Test logging events
        print("Testing UserEventsDB...")

        # Log query
        event_id = db.log_query("How do I write Python code?", topic="coding")
        print(f"Logged query event: {event_id}")

        # Log response
        event_id = db.log_response(
            query="How do I write Python code?",
            response="Here's how to write Python...",
            latency_ms=1500,
            worker="primary",
            topic="coding"
        )
        print(f"Logged response event: {event_id}")

        # Log custom event
        event = UserEvent(
            event_type="topic_detection",
            context={"query": "test"},
            metadata={"topic": "coding", "confidence": 0.9}
        )
        db.log_event(event)

        # Query events
        events = db.get_events()
        print(f"\nTotal events: {len(events)}")
        for e in events:
            print(f"  [{e.event_type}] {e.context.get('query', '')[:30]}...")

        # Get stats
        stats = db.get_stats()
        print(f"\nStats: {json.dumps(stats, indent=2)}")

        # Get by time range
        recent = db.get_events_by_time_range(hours=1)
        print(f"\nEvents in last hour: {len(recent)}")

        print("\nTest complete!")

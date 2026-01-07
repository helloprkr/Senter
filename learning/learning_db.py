#!/usr/bin/env python3
"""
Learning Database Service

Time-series behavioral data storage with pattern detection.

Features:
- Event storage (interactions, preferences, patterns)
- Behavior analysis
- User profile building
- Preference prediction
"""

import json
import time
import logging
import sys
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any
from pathlib import Path
from multiprocessing import Event
from queue import Empty
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon.message_bus import MessageBus, MessageType, Message

logger = logging.getLogger('senter.learning')


@dataclass
class LearningEvent:
    """A learning event from user interaction"""
    event_type: str  # query, response, preference, goal, feedback
    timestamp: float
    data: dict
    session_id: Optional[str] = None


class LearningDatabase:
    """
    SQLite-based time-series storage for learning events.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                session_id TEXT,
                data TEXT NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);

            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_key TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                last_seen REAL,
                metadata TEXT,
                UNIQUE(pattern_type, pattern_key)
            );

            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pref_key TEXT UNIQUE NOT NULL,
                pref_value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                last_updated REAL
            );

            CREATE TABLE IF NOT EXISTS profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                updated_at REAL
            );
        """)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection"""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def store_event(self, event: LearningEvent):
        """Store a learning event"""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO events (event_type, timestamp, session_id, data) VALUES (?, ?, ?, ?)",
            (event.event_type, event.timestamp, event.session_id, json.dumps(event.data))
        )
        conn.commit()

    def get_events(
        self,
        event_type: str = None,
        since: float = None,
        limit: int = 100
    ) -> list[dict]:
        """Get events with optional filtering"""
        conn = self._get_conn()

        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def update_pattern(self, pattern_type: str, pattern_key: str, metadata: dict = None):
        """Update a pattern counter"""
        conn = self._get_conn()
        now = time.time()

        conn.execute("""
            INSERT INTO patterns (pattern_type, pattern_key, count, last_seen, metadata)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(pattern_type, pattern_key) DO UPDATE SET
                count = count + 1,
                last_seen = ?,
                metadata = COALESCE(?, metadata)
        """, (pattern_type, pattern_key, now, json.dumps(metadata) if metadata else None,
              now, json.dumps(metadata) if metadata else None))
        conn.commit()

    def get_top_patterns(self, pattern_type: str, limit: int = 10) -> list[dict]:
        """Get top patterns by count"""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM patterns WHERE pattern_type = ? ORDER BY count DESC LIMIT ?",
            (pattern_type, limit)
        )
        return [dict(row) for row in cursor.fetchall()]

    def set_preference(self, key: str, value: Any, confidence: float = 0.5):
        """Set a user preference"""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO preferences (pref_key, pref_value, confidence, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(pref_key) DO UPDATE SET
                pref_value = ?,
                confidence = ?,
                last_updated = ?
        """, (key, json.dumps(value), confidence, time.time(),
              json.dumps(value), confidence, time.time()))
        conn.commit()

    def get_preference(self, key: str) -> Optional[tuple[Any, float]]:
        """Get a preference value and confidence"""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT pref_value, confidence FROM preferences WHERE pref_key = ?",
            (key,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row["pref_value"]), row["confidence"]
        return None

    def get_all_preferences(self) -> dict:
        """Get all preferences"""
        conn = self._get_conn()
        cursor = conn.execute("SELECT pref_key, pref_value, confidence FROM preferences")
        return {
            row["pref_key"]: {
                "value": json.loads(row["pref_value"]),
                "confidence": row["confidence"]
            }
            for row in cursor.fetchall()
        }

    def set_profile(self, key: str, value: Any):
        """Set a profile value"""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO profile (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = ?,
                updated_at = ?
        """, (key, json.dumps(value), time.time(), json.dumps(value), time.time()))
        conn.commit()

    def get_profile(self) -> dict:
        """Get full user profile"""
        conn = self._get_conn()
        cursor = conn.execute("SELECT key, value FROM profile")
        return {row["key"]: json.loads(row["value"]) for row in cursor.fetchall()}


class BehaviorAnalyzer:
    """
    Analyzes events to detect patterns and update preferences.
    """

    def __init__(self, db: LearningDatabase):
        self.db = db

    def analyze_query(self, query: str, response: str = None, context: dict = None):
        """Analyze a user query"""
        query_lower = query.lower()

        # Detect topics
        topics = self._extract_topics(query_lower)
        for topic in topics:
            self.db.update_pattern("topic", topic)

        # Detect query style
        if "?" in query:
            self.db.update_pattern("style", "question")
        elif any(w in query_lower for w in ["please", "could you", "can you"]):
            self.db.update_pattern("style", "polite")
        elif len(query.split()) < 5:
            self.db.update_pattern("style", "brief")
        else:
            self.db.update_pattern("style", "detailed")

        # Detect time patterns
        hour = datetime.now().hour
        if 5 <= hour < 12:
            self.db.update_pattern("time", "morning")
        elif 12 <= hour < 17:
            self.db.update_pattern("time", "afternoon")
        elif 17 <= hour < 21:
            self.db.update_pattern("time", "evening")
        else:
            self.db.update_pattern("time", "night")

        # Store event
        self.db.store_event(LearningEvent(
            event_type="query",
            timestamp=time.time(),
            data={"query": query, "topics": topics}
        ))

    def _extract_topics(self, text: str) -> list[str]:
        """Extract topics from text"""
        topic_keywords = {
            "coding": ["code", "python", "javascript", "programming", "function", "debug", "error"],
            "research": ["research", "study", "analyze", "data", "report", "statistics"],
            "writing": ["write", "draft", "essay", "article", "blog", "content"],
            "creative": ["creative", "story", "poem", "design", "art", "music"],
            "productivity": ["task", "todo", "schedule", "organize", "plan", "goal"],
            "learning": ["learn", "understand", "explain", "how does", "what is"],
        }

        found_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                found_topics.append(topic)

        return found_topics

    def update_preferences(self):
        """Update preferences based on patterns"""
        # Analyze style patterns
        style_patterns = self.db.get_top_patterns("style", limit=5)
        if style_patterns:
            top_style = style_patterns[0]["pattern_key"]
            total_count = sum(p["count"] for p in style_patterns)
            confidence = style_patterns[0]["count"] / total_count
            self.db.set_preference("response_style", top_style, confidence)

        # Analyze topic patterns
        topic_patterns = self.db.get_top_patterns("topic", limit=10)
        if topic_patterns:
            top_topics = [p["pattern_key"] for p in topic_patterns[:3]]
            self.db.set_preference("interests", top_topics, 0.7)

        # Analyze time patterns
        time_patterns = self.db.get_top_patterns("time", limit=4)
        if time_patterns:
            top_time = time_patterns[0]["pattern_key"]
            self.db.set_preference("active_time", top_time, 0.6)

    def get_insights(self) -> dict:
        """Get learning insights"""
        return {
            "preferences": self.db.get_all_preferences(),
            "top_topics": self.db.get_top_patterns("topic", limit=5),
            "style_patterns": self.db.get_top_patterns("style", limit=5),
            "time_patterns": self.db.get_top_patterns("time", limit=4),
            "profile": self.db.get_profile()
        }


class LearningService:
    """
    Main learning service that runs as a daemon component.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        shutdown_event: Event,
        senter_root: Path
    ):
        self.message_bus = message_bus
        self.shutdown_event = shutdown_event
        self.senter_root = Path(senter_root)

        # Database
        db_path = self.senter_root / "data" / "learning" / "behavior.db"
        self.db = LearningDatabase(db_path)
        self.analyzer = BehaviorAnalyzer(self.db)

        # Message queue
        self._queue = None

        # Analysis interval
        self.analysis_interval = 300  # 5 minutes
        self.last_analysis = 0

    def run(self):
        """Main service loop"""
        logger.info("Learning service starting...")

        # Register with message bus
        self._queue = self.message_bus.register("learning")

        logger.info("Learning service started")

        while not self.shutdown_event.is_set():
            try:
                # Process messages
                self._process_messages()

                # Periodic analysis
                now = time.time()
                if now - self.last_analysis >= self.analysis_interval:
                    self.analyzer.update_preferences()
                    self.last_analysis = now

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Learning service error: {e}")

        logger.info("Learning service stopped")

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

        if message.type == MessageType.USER_QUERY:
            # Learn from user query
            query = payload.get("text", "")
            self.analyzer.analyze_query(query)

        elif message.type == MessageType.LEARN_EVENT:
            # Store explicit learning event
            event = LearningEvent(
                event_type=payload.get("event_type", "custom"),
                timestamp=payload.get("timestamp", time.time()),
                data=payload.get("data", {}),
                session_id=payload.get("session_id")
            )
            self.db.store_event(event)

        elif message.type == MessageType.PROFILE_UPDATE:
            # Update profile
            for key, value in payload.items():
                self.db.set_profile(key, value)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test database
    db_path = Path(__file__).parent.parent / "data" / "learning" / "test_behavior.db"
    db = LearningDatabase(db_path)

    # Test event storage
    db.store_event(LearningEvent(
        event_type="query",
        timestamp=time.time(),
        data={"query": "Help me write Python code"}
    ))

    # Test pattern tracking
    db.update_pattern("topic", "coding")
    db.update_pattern("topic", "coding")
    db.update_pattern("topic", "python")

    # Test preferences
    db.set_preference("language", "python", 0.8)

    # Test analyzer
    analyzer = BehaviorAnalyzer(db)
    analyzer.analyze_query("Can you help me debug this Python function?")
    analyzer.update_preferences()

    # Get insights
    insights = analyzer.get_insights()
    print("\nLearning Insights:")
    print(f"  Preferences: {insights['preferences']}")
    print(f"  Top topics: {insights['top_topics']}")
    print(f"  Style patterns: {insights['style_patterns']}")

    print("\nTest complete")

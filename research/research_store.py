#!/usr/bin/env python3
"""
Research Store - Persistent storage for research results

SQLite-based storage for completed research with search capability.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger("senter.research.store")

# Import types
try:
    from .synthesizer import SynthesizedResearch
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from synthesizer import SynthesizedResearch


@dataclass
class StoredResearch:
    """A stored research result with metadata."""
    id: int
    topic: str
    summary: str
    key_insights: List[str]
    sources: List[str]
    confidence: float
    created_at: datetime
    viewed: bool = False
    feedback_rating: Optional[int] = None  # 1-5 stars


class ResearchStore:
    """
    SQLite-based storage for research results.

    Features:
    - Store completed research with full metadata
    - Mark as viewed/unviewed
    - Search by topic
    - Feedback ratings for future improvement
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".senter" / "research.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_insights TEXT,  -- JSON array
                    sources TEXT,       -- JSON array
                    confidence REAL,
                    synthesis_time_ms INTEGER,
                    created_at REAL,
                    viewed INTEGER DEFAULT 0,
                    feedback_rating INTEGER,
                    raw_data TEXT       -- Full JSON for extensibility
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_research_created
                ON research(created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_research_topic
                ON research(topic)
            """)
            conn.commit()

    def store(self, research: SynthesizedResearch) -> int:
        """
        Store a completed research result.

        Args:
            research: SynthesizedResearch from synthesizer

        Returns:
            ID of stored research
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO research (
                    topic, summary, key_insights, sources,
                    confidence, synthesis_time_ms, created_at, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                research.topic,
                research.summary,
                json.dumps(research.key_insights),
                json.dumps(research.sources_used),
                research.confidence,
                research.synthesis_time_ms,
                research.created_at,
                json.dumps(research.to_dict())
            ))
            conn.commit()
            logger.info(f"Stored research: {research.topic} (id={cursor.lastrowid})")
            return cursor.lastrowid

    def get_unviewed(self, limit: int = 10) -> List[StoredResearch]:
        """Get unviewed research results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM research
                WHERE viewed = 0
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [self._row_to_stored(row) for row in rows]

    def get_recent(self, limit: int = 20) -> List[StoredResearch]:
        """Get recent research results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM research
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [self._row_to_stored(row) for row in rows]

    def get_by_id(self, research_id: int) -> Optional[StoredResearch]:
        """Get research by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM research WHERE id = ?",
                (research_id,)
            ).fetchone()
            return self._row_to_stored(row) if row else None

    def mark_viewed(self, research_id: int):
        """Mark research as viewed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE research SET viewed = 1 WHERE id = ?",
                (research_id,)
            )
            conn.commit()

    def set_feedback(self, research_id: int, rating: int):
        """Set feedback rating (1-5 stars)."""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be 1-5")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE research SET feedback_rating = ? WHERE id = ?",
                (rating, research_id)
            )
            conn.commit()

    def search(self, query: str, limit: int = 10) -> List[StoredResearch]:
        """Search research by topic keyword."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM research
                WHERE topic LIKE ? OR summary LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit)).fetchall()
            return [self._row_to_stored(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM research").fetchone()[0]
            unviewed = conn.execute(
                "SELECT COUNT(*) FROM research WHERE viewed = 0"
            ).fetchone()[0]
            avg_confidence = conn.execute(
                "SELECT AVG(confidence) FROM research"
            ).fetchone()[0] or 0
            rated = conn.execute(
                "SELECT COUNT(*) FROM research WHERE feedback_rating IS NOT NULL"
            ).fetchone()[0]
            avg_rating = conn.execute(
                "SELECT AVG(feedback_rating) FROM research WHERE feedback_rating IS NOT NULL"
            ).fetchone()[0] or 0

            return {
                "total": total,
                "unviewed": unviewed,
                "avg_confidence": round(avg_confidence, 2),
                "rated_count": rated,
                "avg_rating": round(avg_rating, 1)
            }

    def _row_to_stored(self, row: sqlite3.Row) -> StoredResearch:
        """Convert database row to StoredResearch."""
        return StoredResearch(
            id=row["id"],
            topic=row["topic"],
            summary=row["summary"],
            key_insights=json.loads(row["key_insights"] or "[]"),
            sources=json.loads(row["sources"] or "[]"),
            confidence=row["confidence"] or 0,
            created_at=datetime.fromtimestamp(row["created_at"]),
            viewed=bool(row["viewed"]),
            feedback_rating=row["feedback_rating"]
        )

    def clear_all(self):
        """Clear all research (for testing)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM research")
            conn.commit()


# CLI for testing
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Research Store CLI")
    parser.add_argument("--list", "-l", action="store_true", help="List recent research")
    parser.add_argument("--unviewed", "-u", action="store_true", help="List unviewed research")
    parser.add_argument("--stats", "-s", action="store_true", help="Show statistics")
    parser.add_argument("--search", type=str, help="Search by keyword")

    args = parser.parse_args()

    store = ResearchStore()

    if args.stats:
        stats = store.get_stats()
        print("\n=== Research Store Statistics ===")
        print(f"Total researches: {stats['total']}")
        print(f"Unviewed: {stats['unviewed']}")
        print(f"Average confidence: {stats['avg_confidence']}")
        print(f"Rated: {stats['rated_count']}")
        print(f"Average rating: {stats['avg_rating']}/5")

    elif args.search:
        results = store.search(args.search)
        print(f"\n=== Search Results for '{args.search}' ===")
        for r in results:
            print(f"\n{r.id}. {r.topic}")
            print(f"   {r.summary[:100]}...")
            print(f"   Confidence: {r.confidence}, Created: {r.created_at}")

    elif args.unviewed:
        results = store.get_unviewed()
        print("\n=== Unviewed Research ===")
        for r in results:
            print(f"\n{r.id}. {r.topic}")
            print(f"   {r.summary[:100]}...")

    else:
        results = store.get_recent(5)
        print("\n=== Recent Research ===")
        for r in results:
            status = "👁" if r.viewed else "🆕"
            print(f"\n{status} {r.id}. {r.topic}")
            print(f"   {r.summary[:100]}...")
            print(f"   Created: {r.created_at}")

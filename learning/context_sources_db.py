#!/usr/bin/env python3
"""
Context Sources Database (P3-001)

Persistent storage for user-pinned context sources (files, URLs).

Database: data/learning/context_sources.db
Table: context_sources
Fields: id, type, title, path, description, content_preview, created_at, is_active
"""

import json
import time
import uuid
import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger('senter.context_sources_db')


@dataclass
class ContextSource:
    """A pinned context source (file or URL)"""
    id: str
    type: str  # 'file', 'web', 'clipboard', 'code'
    title: str
    path: str  # file path or URL
    description: str = ""
    content_preview: str = ""
    created_at: float = field(default_factory=time.time)
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "path": self.path,
            "description": self.description,
            "content_preview": self.content_preview,
            "created_at": self.created_at,
            "created_at_str": datetime.fromtimestamp(self.created_at).isoformat(),
            "is_active": self.is_active
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ContextSource":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            type=d.get("type", "file"),
            title=d.get("title", "Untitled"),
            path=d.get("path", ""),
            description=d.get("description", ""),
            content_preview=d.get("content_preview", ""),
            created_at=d.get("created_at", time.time()),
            is_active=d.get("is_active", True)
        )


class ContextSourcesDB:
    """
    SQLite database for pinned context sources (P3-001).

    Stores files and URLs that the user wants to include as context.
    """

    def __init__(self, senter_root: Path):
        """Initialize database.

        Args:
            senter_root: Root directory (will use data/learning/context_sources.db)
        """
        self.senter_root = Path(senter_root)
        self.db_path = self.senter_root / "data" / "learning" / "context_sources.db"
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
            CREATE TABLE IF NOT EXISTS context_sources (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                path TEXT NOT NULL,
                description TEXT DEFAULT '',
                content_preview TEXT DEFAULT '',
                created_at REAL DEFAULT (strftime('%s', 'now')),
                is_active INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_context_active ON context_sources(is_active);
            CREATE INDEX IF NOT EXISTS idx_context_type ON context_sources(type);
        """)
        conn.commit()

    def add_source(self, source: ContextSource) -> bool:
        """Add a new context source"""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO context_sources
                (id, type, title, path, description, content_preview, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                source.id,
                source.type,
                source.title,
                source.path,
                source.description,
                source.content_preview,
                source.created_at,
                1 if source.is_active else 0
            ))
            conn.commit()
            logger.info(f"Added context source: {source.title} ({source.type})")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Source already exists: {source.id}")
            return False
        except Exception as e:
            logger.error(f"Failed to add context source: {e}")
            return False

    def get_source(self, source_id: str) -> Optional[ContextSource]:
        """Get a specific context source by ID"""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM context_sources WHERE id = ?",
            (source_id,)
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_source(row)
        return None

    def get_all_sources(self, active_only: bool = True) -> List[ContextSource]:
        """Get all context sources"""
        conn = self._get_conn()
        if active_only:
            cursor = conn.execute(
                "SELECT * FROM context_sources WHERE is_active = 1 ORDER BY created_at DESC"
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM context_sources ORDER BY created_at DESC"
            )
        return [self._row_to_source(row) for row in cursor]

    def get_sources_by_type(self, source_type: str) -> List[ContextSource]:
        """Get context sources by type"""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM context_sources WHERE type = ? AND is_active = 1 ORDER BY created_at DESC",
            (source_type,)
        )
        return [self._row_to_source(row) for row in cursor]

    def remove_source(self, source_id: str) -> bool:
        """Remove a context source (hard delete)"""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM context_sources WHERE id = ?",
                (source_id,)
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Removed context source: {source_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove context source: {e}")
            return False

    def toggle_active(self, source_id: str, is_active: bool) -> bool:
        """Toggle a context source's active status"""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE context_sources SET is_active = ? WHERE id = ?",
                (1 if is_active else 0, source_id)
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to toggle context source: {e}")
            return False

    def _row_to_source(self, row) -> ContextSource:
        """Convert database row to ContextSource"""
        return ContextSource(
            id=row["id"],
            type=row["type"],
            title=row["title"],
            path=row["path"],
            description=row["description"] or "",
            content_preview=row["content_preview"] or "",
            created_at=row["created_at"],
            is_active=bool(row["is_active"])
        )


# Test
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ContextSourcesDB(Path(tmpdir))

        # Test adding sources
        file_source = ContextSource(
            id=str(uuid.uuid4()),
            type="file",
            title="Project README",
            path="/Users/test/project/README.md",
            description="Main project documentation",
            content_preview="# My Project\nThis is a sample project..."
        )
        db.add_source(file_source)

        url_source = ContextSource(
            id=str(uuid.uuid4()),
            type="web",
            title="React Documentation",
            path="https://react.dev/docs",
            description="Official React docs"
        )
        db.add_source(url_source)

        # Get all
        sources = db.get_all_sources()
        print(f"All sources: {len(sources)}")
        for s in sources:
            print(f"  - {s.title} ({s.type}): {s.path}")

        # Remove one
        db.remove_source(file_source.id)
        sources = db.get_all_sources()
        print(f"After removal: {len(sources)}")

        print("\nTest complete")

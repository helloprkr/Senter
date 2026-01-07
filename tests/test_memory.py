"""
Tests for memory module.
"""

import pytest
from pathlib import Path
import tempfile

from memory.living_memory import LivingMemory, Episode, MemoryContext
from memory.semantic import SemanticMemory
from memory.episodic import EpisodicMemory
from memory.procedural import ProceduralMemory
from memory.affective import AffectiveMemory


class TestLivingMemory:
    """Tests for LivingMemory orchestrator."""

    @pytest.fixture
    def memory(self, tmp_path):
        """Create a test memory instance."""
        db_path = tmp_path / "test_memory.db"
        return LivingMemory({}, db_path)

    def test_absorb_interaction(self, memory):
        """Test absorbing an interaction."""
        interaction = {
            "input": "Hello, how are you?",
            "response": "I'm doing well, thank you!",
            "mode": "dialogue",
            "cognitive_state": {"frustration": 0.0, "mode": "exploring"},
            "joint_state": {"focus": "greeting"},
        }

        episode = memory.absorb(interaction)

        assert episode.id is not None
        assert episode.input == "Hello, how are you?"
        assert episode.response == "I'm doing well, thank you!"
        assert episode.mode == "dialogue"

    def test_retrieve_memories(self, memory):
        """Test retrieving memories."""
        # Add some interactions
        memory.absorb({
            "input": "Remember my favorite color is blue",
            "response": "I'll remember that.",
            "mode": "dialogue",
            "cognitive_state": {},
            "joint_state": {},
        })

        # Retrieve
        context = memory.retrieve("color")

        assert isinstance(context, MemoryContext)

    def test_memory_stats(self, memory):
        """Test getting memory statistics."""
        # Add interactions
        for i in range(5):
            memory.absorb({
                "input": f"Test input {i}",
                "response": f"Test response {i}",
                "mode": "dialogue",
                "cognitive_state": {},
                "joint_state": {},
            })

        stats = memory.get_stats()

        assert stats["episodes"] == 5
        assert "semantic_facts" in stats


class TestSemanticMemory:
    """Tests for SemanticMemory."""

    @pytest.fixture
    def semantic(self, tmp_path):
        """Create test semantic memory."""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS semantic (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                domain TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                decay_factor REAL DEFAULT 1.0
            );
        """)
        return SemanticMemory(conn, {})

    def test_store_and_search(self, semantic):
        """Test storing and searching facts."""
        fact_id = semantic.store(
            content="The user's favorite color is blue",
            domain="user_context",
        )

        assert fact_id is not None

        results = semantic.search("color")
        assert len(results) > 0
        assert "blue" in results[0]["content"]

    def test_domain_filtering(self, semantic):
        """Test filtering by domain."""
        semantic.store("Fact in domain A", domain="domain_a")
        semantic.store("Fact in domain B", domain="domain_b")

        results = semantic.get_by_domain("domain_a")
        assert len(results) == 1
        assert results[0]["domain"] == "domain_a"


class TestEpisodicMemory:
    """Tests for EpisodicMemory."""

    @pytest.fixture
    def episodic(self, tmp_path):
        """Create test episodic memory."""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodic (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                input TEXT NOT NULL,
                response TEXT NOT NULL,
                mode TEXT,
                cognitive_state TEXT,
                joint_state TEXT,
                fitness REAL,
                sentiment REAL
            );
        """)
        return EpisodicMemory(conn, {"max_episodes": 100})

    def test_store_and_retrieve(self, episodic):
        """Test storing and retrieving episodes."""
        from datetime import datetime

        episode = Episode(
            id="test123",
            timestamp=datetime.now(),
            input="Test input",
            response="Test response",
            mode="dialogue",
            cognitive_state={},
            joint_state={},
        )

        episodic.store(episode)

        recent = episodic.get_recent(limit=10)
        assert len(recent) == 1
        assert recent[0].id == "test123"

    def test_search_episodes(self, episodic):
        """Test searching episodes."""
        from datetime import datetime

        episodic.store(Episode(
            id="ep1",
            timestamp=datetime.now(),
            input="How do I use Python?",
            response="Python is a programming language...",
            mode="teaching",
            cognitive_state={},
            joint_state={},
        ))

        results = episodic.search("Python")
        assert len(results) == 1


class TestProceduralMemory:
    """Tests for ProceduralMemory."""

    @pytest.fixture
    def procedural(self, tmp_path):
        """Create test procedural memory."""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS procedural (
                id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        return ProceduralMemory(conn, {})

    def test_update_pattern(self, procedural):
        """Test updating patterns."""
        procedural.update_pattern(
            pattern_type="mode_teaching",
            pattern_data='{"mode": "teaching"}',
            success=True,
        )

        patterns = procedural.get_patterns()
        assert "mode_teaching" in patterns
        assert patterns["mode_teaching"]["success_rate"] == 1.0

    def test_pattern_success_rate(self, procedural):
        """Test success rate calculation."""
        # Add successful and failed attempts
        for _ in range(3):
            procedural.update_pattern("test_pattern", "{}", success=True)
        procedural.update_pattern("test_pattern", "{}", success=False)

        patterns = procedural.get_patterns()
        # 3 success, 1 failure = 0.75 rate
        assert patterns["test_pattern"]["success_rate"] == 0.75


class TestAffectiveMemory:
    """Tests for AffectiveMemory."""

    @pytest.fixture
    def affective(self, tmp_path):
        """Create test affective memory."""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS affective (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sentiment REAL,
                frustration REAL,
                satisfaction REAL,
                episode_id TEXT
            );
        """)
        return AffectiveMemory(conn, {})

    def test_store_affect(self, affective):
        """Test storing affective state."""
        record_id = affective.store(
            episode_id="ep123",
            sentiment=0.7,
            frustration=0.2,
            satisfaction=0.8,
        )

        assert record_id is not None

    def test_recent_context(self, affective):
        """Test getting recent emotional context."""
        affective.store("ep1", sentiment=0.6, frustration=0.1, satisfaction=0.7)
        affective.store("ep2", sentiment=0.8, frustration=0.0, satisfaction=0.9)

        context = affective.get_recent_context()

        assert "avg_satisfaction" in context
        assert context["interaction_count"] == 2

"""
Living Memory - Multi-layer memory system orchestrator.

Four memory types:
- Semantic: Facts and concepts (what you know)
- Episodic: Specific events (what happened)
- Procedural: How to help this human (what works)
- Affective: Emotional context (how it felt)
"""

from __future__ import annotations
import sqlite3
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .semantic import SemanticMemory
from .episodic import EpisodicMemory
from .procedural import ProceduralMemory
from .affective import AffectiveMemory

if TYPE_CHECKING:
    from models.embeddings import EmbeddingModel


@dataclass
class Episode:
    """A single interaction episode."""

    id: str
    timestamp: datetime
    input: str
    response: str
    mode: str
    cognitive_state: Dict[str, Any]
    joint_state: Dict[str, Any]

    # Computed later
    fitness: Optional[float] = None
    sentiment: Optional[float] = None


@dataclass
class MemoryContext:
    """Retrieved memory context."""

    semantic: List[Dict]  # Relevant facts
    episodic: List[Episode]  # Relevant past interactions
    procedural: Dict[str, Any]  # How to help this user
    affective: Dict[str, float]  # Emotional context


class LivingMemory:
    """
    Orchestrates four memory types.

    The key insight: Memory isn't just storage, it's how
    the AI becomes personal to this specific human.
    """

    def __init__(
        self,
        config: Dict,
        db_path: Path,
        embeddings: Optional["EmbeddingModel"] = None,
    ):
        self.config = config
        self.db_path = db_path
        self._embeddings = embeddings

        # Ensure data directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        self._init_memory_layers()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        self.conn.executescript(
            """
            -- Semantic memory: facts and concepts
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

            -- Episodic memory: specific interactions
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

            -- Affective memory: emotional context
            CREATE TABLE IF NOT EXISTS affective (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sentiment REAL,
                frustration REAL,
                satisfaction REAL,
                episode_id TEXT REFERENCES episodic(id)
            );

            -- Procedural memory: patterns that work
            CREATE TABLE IF NOT EXISTS procedural (
                id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes for fast retrieval
            CREATE INDEX IF NOT EXISTS idx_semantic_domain ON semantic(domain);
            CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic(timestamp);
            CREATE INDEX IF NOT EXISTS idx_affective_episode ON affective(episode_id);
            CREATE INDEX IF NOT EXISTS idx_procedural_type ON procedural(pattern_type);
        """
        )
        self.conn.commit()

    def _init_memory_layers(self) -> None:
        """Initialize individual memory layer handlers."""
        self.semantic = SemanticMemory(
            self.conn,
            self.config.get("semantic", {}),
            embeddings=self._embeddings,
        )
        self._episodic = EpisodicMemory(self.conn, self.config.get("episodic", {}))
        self._procedural = ProceduralMemory(self.conn, self.config.get("procedural", {}))
        self._affective = AffectiveMemory(self.conn, self.config.get("affective", {}))

    @property
    def episodic(self) -> List[Episode]:
        """Get recent episodes (for status display)."""
        return self._episodic.get_recent(limit=100)

    def absorb(self, interaction: Dict[str, Any]) -> Episode:
        """
        Process an interaction into all memory layers.

        This is where learning happens.
        """
        episode_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now()

        # Store in episodic memory
        episode = Episode(
            id=episode_id,
            timestamp=timestamp,
            input=interaction["input"],
            response=interaction["response"],
            mode=interaction.get("mode", "dialogue"),
            cognitive_state=interaction.get("cognitive_state", {}),
            joint_state=interaction.get("joint_state", {}),
        )
        self._episodic.store(episode)

        # Extract and store semantic facts
        self._extract_facts(interaction, episode_id)

        # Track affective state
        self._track_affective(interaction, episode_id)

        # Update procedural patterns
        self._update_procedural(interaction)

        self.conn.commit()
        return episode

    def _extract_facts(self, interaction: Dict, episode_id: str) -> None:
        """Extract semantic facts from interaction."""
        import re
        input_text = interaction["input"]
        input_lower = input_text.lower()

        # Look for explicit memory requests
        triggers = ["remember", "note that", "don't forget", "keep in mind"]
        if any(trigger in input_lower for trigger in triggers):
            self.semantic.store(
                content=input_text,
                domain="user_stated",
            )

        # Extract self-introduction facts
        intro_patterns = [
            (r"my name is (\w+)", "user_name"),
            (r"i(?:'m| am) (\w+)", "user_identity"),
            (r"i work (?:on|at|in|as) (.+?)(?:\.|,|$)", "user_work"),
            (r"i (?:like|prefer|love|enjoy) (.+?)(?:\.|,|$)", "user_preference"),
            (r"i(?:'m| am) (?:a|an) (.+?)(?:\.|,|$)", "user_role"),
        ]

        for pattern, domain in intro_patterns:
            match = re.search(pattern, input_lower)
            if match:
                self.semantic.store(
                    content=input_text,
                    domain=domain,
                )

    def _track_affective(self, interaction: Dict, episode_id: str) -> None:
        """Track emotional context of interaction."""
        cognitive_state = interaction.get("cognitive_state", {})

        self._affective.store(
            episode_id=episode_id,
            sentiment=0.5,  # Neutral default
            frustration=cognitive_state.get("frustration", 0),
            satisfaction=1.0 - cognitive_state.get("frustration", 0),
        )

    def _update_procedural(self, interaction: Dict) -> None:
        """Update procedural patterns based on interaction."""
        mode = interaction.get("mode", "dialogue")
        cognitive_state = interaction.get("cognitive_state", {})

        # Track mode-specific patterns
        self._procedural.update_pattern(
            pattern_type=f"mode_{mode}",
            pattern_data=json.dumps(
                {
                    "mode": mode,
                    "cognitive_mode": cognitive_state.get("mode", "exploring"),
                }
            ),
            success=cognitive_state.get("frustration", 0) < 0.5,
        )

    def retrieve(self, query: str, layers: List[str] = None) -> MemoryContext:
        """Retrieve relevant memories across layers."""
        if layers is None:
            layers = ["semantic", "episodic", "procedural", "affective"]

        semantic = []
        episodic = []
        procedural = {}
        affective = {}

        if "semantic" in layers:
            semantic = self.semantic.search(query, limit=5)

        if "episodic" in layers:
            episodic = self._episodic.search(query, limit=3)

        if "procedural" in layers:
            procedural = self._procedural.get_patterns()

        if "affective" in layers:
            affective = self._affective.get_recent_context()

        return MemoryContext(
            semantic=semantic,
            episodic=episodic,
            procedural=procedural,
            affective=affective,
        )

    def update_episode_fitness(self, episode_id: str, fitness: float) -> None:
        """Update fitness score for an episode."""
        self._episodic.update_fitness(episode_id, fitness)

    def persist(self) -> None:
        """Ensure all data is written to disk."""
        self.conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        cursor = self.conn.cursor()

        semantic_count = cursor.execute("SELECT COUNT(*) FROM semantic").fetchone()[0]
        episodic_count = cursor.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
        procedural_count = cursor.execute("SELECT COUNT(*) FROM procedural").fetchone()[0]
        affective_count = cursor.execute("SELECT COUNT(*) FROM affective").fetchone()[0]

        return {
            "semantic_facts": semantic_count,
            "episodes": episodic_count,
            "procedural_patterns": procedural_count,
            "affective_records": affective_count,
            "total_memories": semantic_count + episodic_count,
        }

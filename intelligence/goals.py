"""
Automatic goal detection from conversations.

Learns user goals without explicit statements.
"""

from __future__ import annotations
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memory.living_memory import LivingMemory


@dataclass
class Goal:
    """A detected user goal."""

    id: str
    description: str
    category: str  # career, health, learning, project, personal
    confidence: float  # How confident we are this is a real goal
    evidence: List[str]  # Conversation snippets that suggest this goal
    created_at: datetime
    last_mentioned: datetime
    progress: float  # 0-1 estimated progress
    status: str  # active, completed, abandoned

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "confidence": self.confidence,
            "evidence": self.evidence[-10:],  # Keep last 10 evidence items
            "created_at": self.created_at.isoformat(),
            "last_mentioned": self.last_mentioned.isoformat(),
            "progress": self.progress,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            category=data.get("category", "personal"),
            confidence=data.get("confidence", 0.5),
            evidence=data.get("evidence", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else datetime.now(),
            last_mentioned=datetime.fromisoformat(data["last_mentioned"])
            if isinstance(data.get("last_mentioned"), str)
            else datetime.now(),
            progress=data.get("progress", 0.0),
            status=data.get("status", "active"),
        )


class GoalDetector:
    """
    Detects goals from conversation patterns.

    Looks for:
    - Explicit goals: "I want to...", "My goal is..."
    - Implicit goals: Repeated topics, frustrations, questions
    - Project references: "Working on...", "Building..."
    """

    EXPLICIT_PATTERNS = [
        r"(?:i want to|i'd like to|i need to|my goal is|i'm trying to|i hope to) (.+?)(?:\.|,|$)",
        r"(?:i'm working on|i'm building|i'm learning|i'm studying) (.+?)(?:\.|,|$)",
        r"(?:i have to|i must|i should) (.+?)(?:\.|,|$)",
    ]

    CATEGORY_KEYWORDS = {
        "career": ["job", "work", "career", "promotion", "salary", "interview", "resume"],
        "health": ["exercise", "diet", "weight", "sleep", "health", "gym", "run", "fitness"],
        "learning": ["learn", "study", "course", "book", "understand", "skill", "tutorial"],
        "project": ["build", "create", "develop", "launch", "ship", "code", "app", "website"],
        "personal": ["relationship", "family", "friend", "hobby", "travel", "home"],
    }

    def __init__(self, memory: "LivingMemory"):
        self.memory = memory
        self.goals: Dict[str, Goal] = {}
        self._load_goals()

    def _load_goals(self) -> None:
        """Load persisted goals from memory."""
        try:
            stored = self.memory.semantic.get_by_domain("goals")
            for item in stored:
                content = item.get("content", "")
                if isinstance(content, str):
                    try:
                        goal_data = json.loads(content)
                        if isinstance(goal_data, dict) and "id" in goal_data:
                            goal = Goal.from_dict(goal_data)
                            self.goals[goal.id] = goal
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Warning: Could not load goals: {e}")

    def analyze_interaction(self, input_text: str, response_text: str) -> List[Goal]:
        """
        Analyze an interaction for goal signals.

        Returns new or updated goals.
        """
        input_lower = input_text.lower()
        new_goals = []

        # Check explicit patterns
        for pattern in self.EXPLICIT_PATTERNS:
            matches = re.findall(pattern, input_lower)
            for match in matches:
                goal = self._create_or_update_goal(
                    description=match.strip(),
                    evidence=input_text,
                    confidence=0.8,  # Explicit statement = high confidence
                )
                if goal:
                    new_goals.append(goal)

        # Check for topic repetition (implicit goals)
        recent_topics = self._get_recent_topics()
        current_topics = self._extract_topics(input_text)

        for topic in current_topics:
            if topic in recent_topics and recent_topics[topic] >= 3:
                # Mentioned 3+ times = likely a goal
                goal = self._create_or_update_goal(
                    description=f"Focus on {topic}",
                    evidence=input_text,
                    confidence=0.5 + (0.1 * min(recent_topics[topic], 5)),
                )
                if goal:
                    new_goals.append(goal)

        # Check for frustrations (implicit blocked goals)
        frustration_patterns = [
            r"(?:frustrated|annoyed|stuck|can't|won't work|failing) (?:with|at|on) (.+?)(?:\.|,|$)"
        ]
        for pattern in frustration_patterns:
            matches = re.findall(pattern, input_lower)
            for match in matches:
                goal = self._create_or_update_goal(
                    description=f"Resolve issues with {match.strip()}",
                    evidence=input_text,
                    confidence=0.6,
                )
                if goal:
                    new_goals.append(goal)

        return new_goals

    def _create_or_update_goal(
        self,
        description: str,
        evidence: str,
        confidence: float,
    ) -> Optional[Goal]:
        """Create new goal or update existing one."""
        # Check for similar existing goal
        for goal in self.goals.values():
            if self._similarity(goal.description, description) > 0.7:
                # Update existing
                goal.confidence = min(1.0, goal.confidence + 0.1)
                goal.evidence.append(evidence)
                goal.last_mentioned = datetime.now()
                self._persist_goal(goal)
                return goal

        # Create new
        goal = Goal(
            id=str(uuid.uuid4())[:8],
            description=description,
            category=self._categorize(description),
            confidence=confidence,
            evidence=[evidence],
            created_at=datetime.now(),
            last_mentioned=datetime.now(),
            progress=0.0,
            status="active",
        )

        self.goals[goal.id] = goal
        self._persist_goal(goal)
        return goal

    def _categorize(self, description: str) -> str:
        """Categorize a goal."""
        desc_lower = description.lower()
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                return category
        return "personal"

    def _similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity measure."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0

    def _get_recent_topics(self) -> Dict[str, int]:
        """Get topic frequency from recent conversations."""
        try:
            episodes = self.memory._episodic.get_recent(limit=50)
            topics = {}
            for ep in episodes:
                for topic in self._extract_topics(ep.input):
                    topics[topic] = topics.get(topic, 0) + 1
            return topics
        except Exception:
            return {}

    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text."""
        words = text.lower().split()
        # Filter to meaningful words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "be", "to", "of", "and", "in",
            "that", "it", "for", "on", "with", "as", "at", "by", "this", "from",
            "or", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "when",
            "can", "no", "just", "into", "your", "some", "could", "them", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "also",
            "back", "after", "use", "how", "our", "well", "way", "want", "she",
            "him", "his", "her", "he", "we", "me", "you", "i",
        }
        return [w.strip(".,!?") for w in words if w not in stopwords and len(w) > 3]

    def _persist_goal(self, goal: Goal) -> None:
        """Save goal to memory."""
        try:
            self.memory.semantic.store(
                content=json.dumps(goal.to_dict()),
                domain="goals",
            )
        except Exception as e:
            print(f"Warning: Could not persist goal: {e}")

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals sorted by confidence."""
        active = [g for g in self.goals.values() if g.status == "active"]
        return sorted(active, key=lambda g: g.confidence, reverse=True)

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a specific goal by ID."""
        return self.goals.get(goal_id)

    def mark_completed(self, goal_id: str) -> bool:
        """Mark a goal as completed."""
        if goal_id in self.goals:
            self.goals[goal_id].status = "completed"
            self.goals[goal_id].progress = 1.0
            self._persist_goal(self.goals[goal_id])
            return True
        return False

    def update_progress(self, goal_id: str, progress: float) -> bool:
        """Update progress on a goal."""
        if goal_id in self.goals:
            self.goals[goal_id].progress = min(1.0, max(0.0, progress))
            self._persist_goal(self.goals[goal_id])
            return True
        return False

    def suggest_actions(self) -> List[Dict[str, Any]]:
        """Suggest actions based on goals."""
        suggestions = []

        for goal in self.get_active_goals()[:5]:  # Top 5 goals
            if goal.category == "learning":
                suggestions.append({
                    "goal": goal,
                    "action": "research",
                    "description": f"Research resources for: {goal.description}",
                })
            elif goal.category == "project":
                suggestions.append({
                    "goal": goal,
                    "action": "plan",
                    "description": f"Create action plan for: {goal.description}",
                })
            elif goal.progress < 0.2:
                suggestions.append({
                    "goal": goal,
                    "action": "start",
                    "description": f"Get started on: {goal.description}",
                })

        return suggestions

    def get_summary(self) -> Dict[str, Any]:
        """Get goal summary."""
        active = [g for g in self.goals.values() if g.status == "active"]
        completed = [g for g in self.goals.values() if g.status == "completed"]

        return {
            "total_goals": len(self.goals),
            "active_goals": len(active),
            "completed_goals": len(completed),
            "top_goals": [
                {"description": g.description, "confidence": g.confidence, "progress": g.progress}
                for g in self.get_active_goals()[:5]
            ],
        }

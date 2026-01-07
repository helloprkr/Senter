"""
Proactive suggestion engine.

Senter doesn't just respond - it anticipates and suggests.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.engine import Senter
    from .goals import GoalDetector


class ProactiveSuggestionEngine:
    """
    Generates proactive suggestions based on:
    - User goals
    - Conversation patterns
    - Time-based triggers
    - Context awareness
    """

    def __init__(self, engine: "Senter", goal_detector: Optional["GoalDetector"] = None):
        self.engine = engine
        self.goal_detector = goal_detector
        self.last_suggestions: Dict[str, datetime] = {}
        self.suggestion_cooldown = timedelta(hours=4)

    async def generate_suggestions(self) -> List[Dict[str, Any]]:
        """Generate current suggestions."""
        suggestions = []

        # Goal-based suggestions
        if self.goal_detector:
            goal_suggestions = await self._goal_based_suggestions()
            suggestions.extend(goal_suggestions)

        # Time-based suggestions
        time_suggestions = await self._time_based_suggestions()
        suggestions.extend(time_suggestions)

        # Pattern-based suggestions
        pattern_suggestions = await self._pattern_based_suggestions()
        suggestions.extend(pattern_suggestions)

        # Filter by cooldown and trust level
        filtered = self._filter_suggestions(suggestions)

        return filtered[:3]  # Return top 3

    async def _goal_based_suggestions(self) -> List[Dict]:
        """Suggestions based on user goals."""
        suggestions = []

        if not self.goal_detector:
            return suggestions

        # Get active goals
        goals = self.goal_detector.get_active_goals()

        for goal in goals[:3]:  # Top 3 goals
            # Check if we haven't suggested this recently
            if self._should_suggest(f"goal_{goal.id}"):
                # Generate specific suggestion based on goal state
                if goal.progress < 0.1:
                    suggestions.append({
                        "type": "goal_start",
                        "id": f"goal_{goal.id}",
                        "priority": goal.confidence,
                        "title": f"Get started on: {goal.description}",
                        "action": f"Would you like me to help break down '{goal.description}' into actionable steps?",
                        "task_type": "plan",
                        "goal_id": goal.id,
                    })
                elif goal.category == "learning":
                    suggestions.append({
                        "type": "goal_research",
                        "id": f"goal_{goal.id}",
                        "priority": goal.confidence * 0.9,
                        "title": f"Research opportunity: {goal.description}",
                        "action": f"I can research the latest resources about {goal.description}. Want me to do that in the background?",
                        "task_type": "research",
                        "goal_id": goal.id,
                    })
                elif goal.progress > 0.5:
                    suggestions.append({
                        "type": "goal_progress",
                        "id": f"goal_{goal.id}",
                        "priority": goal.confidence * 0.8,
                        "title": f"Good progress on: {goal.description}",
                        "action": f"You're making progress on '{goal.description}'. Want to review next steps?",
                        "task_type": "plan",
                        "goal_id": goal.id,
                    })

        return suggestions

    async def _time_based_suggestions(self) -> List[Dict]:
        """Time-sensitive suggestions."""
        suggestions = []
        now = datetime.now()

        # Morning reflection (8-10 AM)
        if 8 <= now.hour <= 10 and self._should_suggest("morning_reflection"):
            suggestions.append({
                "type": "time_morning",
                "id": "morning_reflection",
                "priority": 0.7,
                "title": "Morning planning",
                "action": "Good morning! Would you like me to summarize what we worked on yesterday and suggest priorities for today?",
                "task_type": "summarize",
            })

        # Evening review (6-8 PM)
        if 18 <= now.hour <= 20 and self._should_suggest("evening_review"):
            suggestions.append({
                "type": "time_evening",
                "id": "evening_review",
                "priority": 0.6,
                "title": "Daily review",
                "action": "End of day! Want me to compile what we accomplished today?",
                "task_type": "summarize",
            })

        # Weekly review (Sunday)
        if now.weekday() == 6 and self._should_suggest("weekly_review"):
            suggestions.append({
                "type": "time_weekly",
                "id": "weekly_review",
                "priority": 0.8,
                "title": "Weekly review",
                "action": "It's Sunday - good time for a weekly review. Shall I analyze this week's progress on your goals?",
                "task_type": "summarize",
            })

        return suggestions

    async def _pattern_based_suggestions(self) -> List[Dict]:
        """Suggestions based on conversation patterns."""
        suggestions = []

        try:
            # Check for repeated questions
            repeated_questions = self._get_repeated_questions()
            for question, count in repeated_questions.items():
                suggestion_id = f"faq_{hash(question) % 10000}"
                if self._should_suggest(suggestion_id):
                    suggestions.append({
                        "type": "pattern_faq",
                        "id": suggestion_id,
                        "priority": 0.5,
                        "title": "Frequently asked",
                        "action": f"I notice you've asked about '{question[:50]}...' several times. Want me to create a comprehensive reference?",
                        "task_type": "research",
                    })

            # Check for stalled topics
            stalled = self._get_stalled_topics()
            for topic in stalled:
                suggestion_id = f"stalled_{hash(topic) % 10000}"
                if self._should_suggest(suggestion_id):
                    suggestions.append({
                        "type": "pattern_stalled",
                        "id": suggestion_id,
                        "priority": 0.4,
                        "title": "Revisit topic",
                        "action": f"We haven't discussed '{topic}' in a while. Want to pick up where we left off?",
                        "task_type": "recall",
                    })
        except Exception:
            # Pattern analysis is optional
            pass

        return suggestions

    def _should_suggest(self, suggestion_id: str) -> bool:
        """Check if suggestion is past cooldown."""
        if suggestion_id not in self.last_suggestions:
            return True

        elapsed = datetime.now() - self.last_suggestions[suggestion_id]
        return elapsed >= self.suggestion_cooldown

    def mark_suggested(self, suggestion_id: str) -> None:
        """Mark suggestion as shown."""
        self.last_suggestions[suggestion_id] = datetime.now()

    def _filter_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Filter by trust level and cooldown."""
        trust = getattr(self.engine, "trust", None)
        trust_level = trust.level if trust else 0.5

        # Only show proactive suggestions at sufficient trust
        if trust_level < 0.6:
            return []  # Not enough trust for proactive behavior

        # Higher trust = more suggestions
        max_suggestions = 1 if trust_level < 0.7 else (2 if trust_level < 0.8 else 3)

        # Sort by priority
        sorted_suggestions = sorted(
            suggestions, key=lambda s: s.get("priority", 0), reverse=True
        )

        return sorted_suggestions[:max_suggestions]

    def _get_repeated_questions(self) -> Dict[str, int]:
        """Find questions asked multiple times."""
        try:
            episodes = self.engine.memory._episodic.get_recent(limit=100)
            questions = {}

            for ep in episodes:
                if "?" in ep.input:
                    # Normalize question
                    q = ep.input.lower().strip()
                    questions[q] = questions.get(q, 0) + 1

            return {q: c for q, c in questions.items() if c >= 3}
        except Exception:
            return {}

    def _get_stalled_topics(self) -> List[str]:
        """Find topics that were discussed but dropped."""
        try:
            # Get recent topics
            recent_episodes = self.engine.memory._episodic.get_recent(limit=50)
            recent_topics = set()
            for ep in recent_episodes:
                recent_topics.update(ep.input.lower().split())

            # Get old episodes
            old_episodes = self.engine.memory._episodic.get_recent(limit=100)[50:]

            stalled = []
            for ep in old_episodes:
                topics = set(ep.input.lower().split()) - recent_topics
                for topic in topics:
                    if len(topic) > 5 and topic not in [
                        "about", "would", "could", "should", "there", "their"
                    ]:
                        stalled.append(topic)

            return list(set(stalled))[:5]
        except Exception:
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "active_suggestions": len(self.last_suggestions),
            "cooldown_hours": self.suggestion_cooldown.total_seconds() / 3600,
            "has_goal_detector": self.goal_detector is not None,
        }

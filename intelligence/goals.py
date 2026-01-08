"""
Automatic goal detection from conversations.

Learns user goals without explicit statements.
"""

from __future__ import annotations
import json
import re
import uuid
from dataclasses import dataclass
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
    source: str = "conversation"  # conversation, activity_inferred, explicit

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
            "source": self.source,
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
            source=data.get("source", "conversation"),
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

        # Check for progress indicators on existing goals
        self._detect_progress_from_text(input_text)

        return new_goals

    def _detect_progress_from_text(self, text: str) -> None:
        """
        Detect goal progress or completion indicators in text.

        Updates goals when progress is mentioned.
        Auto-marks complete when 'finished/done/completed' detected.
        """
        text_lower = text.lower()

        # Check for progress indicators FIRST (before completion)
        # This prevents "75% done" from being matched as completion

        # Progress patterns with percentage or fraction
        progress_patterns = [
            r"(?:i'm|i\s+am)\s+(?:about\s+)?(\d+)%?\s+(?:done|complete|through)\s+(?:with\s+)?(.+?)(?:\.|!|$)",
            r"(\d+)%\s+(?:done|complete|progress)\s+(?:on|with)\s+(.+?)(?:\.|!|$)",
            r"(?:halfway|half\s+way)\s+(?:through|done\s+with)\s+(.+?)(?:\.|!|$)",
            r"(?:almost|nearly)\s+(?:done|finished)\s+(?:with\s+)?(.+?)(?:\.|!|$)",
            r"(?:made\s+)?(?:good|great|some)\s+progress\s+(?:on|with)\s+(.+?)(?:\.|!|$)",
        ]

        # Track topics that were matched by progress patterns
        progress_matched_topics = set()

        # Check for progress indicators first
        for pattern in progress_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        first, second = match
                        if first.isdigit():
                            progress = min(1.0, int(first) / 100.0)
                            topic = second
                        else:
                            topic = first
                            progress = 0.5
                    else:
                        topic = match[0]
                        progress = 0.5
                else:
                    topic = match
                    progress = 0.5

                # "halfway" pattern
                if "halfway" in text_lower or "half way" in text_lower:
                    progress = 0.5
                # "almost/nearly done" pattern
                elif "almost" in text_lower or "nearly" in text_lower:
                    progress = 0.9

                if topic and len(topic) >= 3:
                    progress_matched_topics.add(topic.strip().lower())
                    self._update_goal_by_topic(topic, progress=progress, mark_complete=False)

        # Skip completion check if progress patterns matched
        # (prevents "75% done with X" from also triggering "done with X")
        if progress_matched_topics:
            return

        # Completion patterns - only check if no progress patterns matched
        # These are for explicit completion statements
        completion_patterns = [
            r"(?:i(?:'ve)?\s+)?(?:finally\s+)?finished\s+(.+?)(?:\.|!|$)",
            r"(?:i(?:'ve)?\s+)?completed\s+(.+?)(?:\.|!|$)",
            r"(?:i'm\s+)?done\s+with\s+(?:the\s+)?(.+?)(?:\.|!|$)",
            r"(?:just\s+)?(?:wrapped\s+up|finished\s+up)\s+(.+?)(?:\.|!|$)",
            r"(.+?)\s+is\s+(?:finally\s+)?(?:complete|finished)(?:\.|!|$)",
        ]

        for pattern in completion_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                topic = match.strip() if isinstance(match, str) else match
                if topic and len(topic) >= 3:
                    self._update_goal_by_topic(topic, progress=1.0, mark_complete=True)

    def _update_goal_by_topic(
        self,
        topic: str,
        progress: float,
        mark_complete: bool = False
    ) -> Optional[Goal]:
        """
        Find and update a goal matching the topic.

        Args:
            topic: Topic string to match against goal descriptions
            progress: Progress value (0-1)
            mark_complete: If True, mark goal as completed

        Returns:
            Updated Goal or None if no match found
        """
        if not topic or len(topic) < 3:
            return None

        topic_lower = topic.lower().strip()

        # Find matching goal
        for goal in self.goals.values():
            if goal.status != "active":
                continue

            goal_desc_lower = goal.description.lower()

            # Check for topic match
            if (topic_lower in goal_desc_lower or
                self._similarity(topic_lower, goal_desc_lower) > 0.5):

                if mark_complete:
                    goal.status = "completed"
                    goal.progress = 1.0
                else:
                    # Only update if new progress is higher
                    if progress > goal.progress:
                        goal.progress = progress

                self._persist_goal(goal)
                return goal

        return None

    async def detect_progress_with_llm(
        self,
        text: str,
        model: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to detect goal progress indicators in text.

        Args:
            text: User input text to analyze
            model: LLM model with async generate() method

        Returns:
            List of dicts with goal_id, progress, and evidence
        """
        if not model or not text:
            return []

        active_goals = self.get_active_goals()
        if not active_goals:
            return []

        goals_list = "\n".join([
            f"- {g.id}: {g.description} (current progress: {g.progress:.0%})"
            for g in active_goals[:10]
        ])

        prompt = f"""Analyze this user message for any progress updates on their goals.

User's active goals:
{goals_list}

User message: "{text}"

For each goal where progress is mentioned, return a JSON array with:
- goal_id: The goal ID
- new_progress: Number 0.0-1.0 representing completion percentage
- completed: Boolean, true if explicitly finished/completed
- evidence: Quote from message showing progress

Example: [{{"goal_id": "abc123", "new_progress": 0.75, "completed": false, "evidence": "75% done with the project"}}]

If no progress mentioned, return: []

JSON array:"""

        try:
            response = await model.generate(prompt)
            updates = self._parse_llm_goal_response(response)

            results = []
            for update in updates:
                if not isinstance(update, dict) or 'goal_id' not in update:
                    continue

                goal_id = update.get('goal_id')
                if goal_id not in self.goals:
                    continue

                goal = self.goals[goal_id]
                new_progress = update.get('new_progress', goal.progress)
                is_completed = update.get('completed', False)

                if is_completed:
                    goal.status = "completed"
                    goal.progress = 1.0
                elif new_progress > goal.progress:
                    goal.progress = min(1.0, new_progress)

                goal.evidence.append(update.get('evidence', text[:100]))
                self._persist_goal(goal)

                results.append({
                    'goal_id': goal_id,
                    'progress': goal.progress,
                    'completed': goal.status == "completed",
                    'evidence': update.get('evidence', '')
                })

            return results

        except Exception as e:
            print(f"Warning: LLM progress detection failed: {e}")
            return []

    def _create_or_update_goal(
        self,
        description: str,
        evidence: str,
        confidence: float,
        source: str = "conversation",
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
            source=source,
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

    def create_activity_inferred_goal(
        self,
        description: str,
        evidence: str,
        confidence: float = 0.5,
        project_name: Optional[str] = None,
    ) -> Optional[Goal]:
        """
        Create a goal inferred from activity monitoring.

        Args:
            description: Goal description
            evidence: Evidence string (e.g., "Observed 15 coding sessions")
            confidence: Confidence level (0-1)
            project_name: Associated project if any

        Returns:
            Created or updated Goal, or None if already exists
        """
        # If we have a project name, make the goal more specific
        if project_name:
            description = f"{description} ({project_name})"

        return self._create_or_update_goal(
            description=description,
            evidence=evidence,
            confidence=confidence,
            source="activity_inferred",
        )

    def get_goals_by_source(self, source: str) -> List[Goal]:
        """Get all goals from a specific source."""
        return [g for g in self.goals.values() if g.source == source]

    async def detect_goals_semantically(
        self,
        episodes: Optional[List[Any]] = None,
        model: Optional[Any] = None,
        limit: int = 20
    ) -> List[Goal]:
        """
        Use LLM to detect goals from conversation history.

        Analyzes recent conversations for implicit and explicit goals
        that regex patterns might miss.

        Args:
            episodes: List of Episode objects to analyze. If None, fetches from memory.
            model: LLM model to use. Must have async generate() method.
            limit: Maximum number of episodes to analyze.

        Returns:
            List of newly detected/updated Goal objects.
        """
        # Get episodes if not provided
        if episodes is None:
            try:
                episodes = self.memory._episodic.get_recent(limit=limit)
            except Exception:
                return []

        if not episodes:
            return []

        # Format conversation history for LLM
        conversation_text = []
        for ep in episodes[-limit:]:
            input_text = getattr(ep, 'input', '') or ''
            response_text = getattr(ep, 'response', '') or ''
            if input_text:
                conversation_text.append(f"User: {input_text}")
            if response_text:
                conversation_text.append(f"AI: {response_text[:200]}...")

        if not conversation_text:
            return []

        # Construct LLM prompt
        prompt = f"""Analyze these conversation snippets and identify any user goals, aspirations, or objectives. Look for both explicit statements ("I want to...", "My goal is...") and implicit signals (repeated topics, frustrations, questions about learning something).

Conversation:
{chr(10).join(conversation_text[-40:])}

Return a JSON array of detected goals. For each goal include:
- description: What the user wants to achieve
- category: One of (career, health, learning, project, personal)
- confidence: 0.0-1.0 how confident you are this is a real goal
- evidence: Quote or paraphrase from conversation that suggests this goal

Example output:
[{{"description": "Learn Spanish fluently", "category": "learning", "confidence": 0.8, "evidence": "User mentioned practicing Spanish lessons twice"}}]

If no goals detected, return: []

JSON array:"""

        # Call LLM if available
        if model is None:
            return []

        try:
            response = await model.generate(prompt)

            # Extract JSON from response
            detected = self._parse_llm_goal_response(response)
            new_goals = []

            for goal_data in detected:
                if not isinstance(goal_data, dict):
                    continue

                description = goal_data.get('description', '')
                if not description or len(description) < 5:
                    continue

                goal = self._create_or_update_goal(
                    description=description,
                    evidence=goal_data.get('evidence', 'LLM semantic detection'),
                    confidence=min(0.9, max(0.3, goal_data.get('confidence', 0.6))),
                    source="conversation"
                )
                if goal:
                    # Update category if detected
                    category = goal_data.get('category', '')
                    if category in self.CATEGORY_KEYWORDS:
                        goal.category = category
                        self._persist_goal(goal)
                    new_goals.append(goal)

            return new_goals

        except Exception as e:
            print(f"Warning: LLM goal detection failed: {e}")
            return []

    def _parse_llm_goal_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract goal JSON array."""
        if not response:
            return []

        # Try to extract JSON array from response
        try:
            # First try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in response
        import re
        json_match = re.search(r'\[[\s\S]*?\]', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return []

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

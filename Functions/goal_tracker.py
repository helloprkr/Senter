#!/usr/bin/env python3
"""
Goal Tracking System for Senter
Extracts, persists, and tracks user goals across conversations

Enhanced for:
- IA-001: Entity extraction (ACTION, OBJECT, DEADLINE, CONDITION)
- GS-001: Auto-detection of goal progress
- GS-002: Goal prioritization and scheduling
- GS-003: Goal relationships (parent/child)
"""

import json
import logging
import re
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger("senter.goals")


# ========== IA-001: Entity Types for Goal Extraction ==========

class EntityType(Enum):
    """Entity types for goal extraction"""
    ACTION = "action"      # What needs to be done
    OBJECT = "object"      # What is being acted upon
    DEADLINE = "deadline"  # When it needs to be done
    CONDITION = "condition"  # Conditions or constraints


@dataclass
class ExtractedEntity:
    """Single extracted entity from text"""
    entity_type: EntityType
    value: str
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entity_type"] = self.entity_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedEntity":
        data = data.copy()
        data["entity_type"] = EntityType(data["entity_type"])
        return cls(**data)


@dataclass
class GoalExtractionResult:
    """Result of goal extraction with entity analysis"""
    entities: List[ExtractedEntity] = field(default_factory=list)
    action: Optional[str] = None
    object_target: Optional[str] = None
    deadline: Optional[str] = None
    condition: Optional[str] = None
    confidence: float = 0.0
    needs_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entities"] = [e if isinstance(e, dict) else e.to_dict() for e in self.entities]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "GoalExtractionResult":
        data = data.copy()
        data["entities"] = [
            ExtractedEntity.from_dict(e) if isinstance(e, dict) else e
            for e in data.get("entities", [])
        ]
        return cls(**data)


class GoalEntityExtractor:
    """
    Extract entities from text for goal detection (IA-001)

    Entity types:
    - ACTION: Verbs indicating what needs to be done
    - OBJECT: Noun phrases indicating target of action
    - DEADLINE: Time references
    - CONDITION: Conditional phrases
    """

    # Action verbs that indicate goals
    ACTION_PATTERNS = [
        (r"\b(finish|complete|build|create|write|implement|develop|design|fix|solve|learn|study|read|prepare|organize|plan|schedule|submit|deliver|send|review|update|improve|optimize|refactor|test|deploy|launch|release)\b", 0.8),
        (r"\b(need to|have to|must|should|want to|going to|planning to|trying to|hope to|aim to|intend to)\s+([\w\s]+)", 0.9),
    ]

    # Time/deadline patterns
    DEADLINE_PATTERNS = [
        (r"\bby\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", 0.9),
        (r"\bby\s+(tomorrow|tonight|today|next week|next month|end of (?:the\s+)?(?:day|week|month|year))\b", 0.9),
        (r"\bby\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", 0.95),
        (r"\bdeadline[:\s]+(.+?)(?:\.|,|$)", 0.95),
        (r"\bbefore\s+([\w\s]+?)(?:\.|,|$)", 0.7),
        (r"\bwithin\s+(\d+\s+(?:hours?|days?|weeks?|months?))\b", 0.85),
        (r"\bin\s+(\d+\s+(?:hours?|days?|weeks?|months?))\b", 0.7),
    ]

    # Condition patterns
    CONDITION_PATTERNS = [
        (r"\bif\s+(.+?)(?:then|,|$)", 0.8),
        (r"\bwhen\s+(.+?)(?:then|,|$)", 0.7),
        (r"\bonce\s+(.+?)(?:then|,|$)", 0.8),
        (r"\bafter\s+(.+?)(?:then|,|$)", 0.75),
        (r"\bunless\s+(.+?)(?:\.|,|$)", 0.8),
        (r"\bprovided\s+(?:that\s+)?(.+?)(?:\.|,|$)", 0.85),
    ]

    def extract_entities(self, text: str) -> GoalExtractionResult:
        """Extract all entities from text"""
        result = GoalExtractionResult()
        text_lower = text.lower()

        # Extract actions
        actions = self._extract_actions(text_lower)
        result.entities.extend(actions)
        if actions:
            result.action = actions[0].value

        # Extract objects (noun phrases after actions)
        objects = self._extract_objects(text_lower, actions)
        result.entities.extend(objects)
        if objects:
            result.object_target = objects[0].value

        # Extract deadlines
        deadlines = self._extract_deadlines(text_lower)
        result.entities.extend(deadlines)
        if deadlines:
            result.deadline = deadlines[0].value

        # Extract conditions
        conditions = self._extract_conditions(text_lower)
        result.entities.extend(conditions)
        if conditions:
            result.condition = conditions[0].value

        # Calculate overall confidence
        result.confidence = self._calculate_confidence(result)

        # Determine if clarification is needed
        result.needs_clarification = result.confidence < 0.6
        if result.needs_clarification:
            result.clarification_questions = self._generate_clarification_questions(result)

        return result

    def _extract_actions(self, text: str) -> List[ExtractedEntity]:
        """Extract action entities"""
        entities = []

        for pattern, confidence in self.ACTION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1) if match.lastindex else match.group(0)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.ACTION,
                    value=value.strip(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence
                ))

        return entities[:3]  # Return top 3

    def _extract_objects(self, text: str, actions: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Extract object entities (what the action is applied to)"""
        entities = []

        # Look for noun phrases after action verbs
        for action in actions:
            # Find text after the action
            after_action = text[action.end_pos:action.end_pos + 100]

            # Extract noun phrase (simple heuristic)
            noun_match = re.match(r"\s*(the\s+)?([\w\s]+?)(?:\s+(?:by|before|for|to|in|on|at|if|when)|[.,!?]|$)", after_action)
            if noun_match:
                value = noun_match.group(2).strip()
                if len(value) > 2 and len(value) < 50:
                    entities.append(ExtractedEntity(
                        entity_type=EntityType.OBJECT,
                        value=value,
                        start_pos=action.end_pos + noun_match.start(2),
                        end_pos=action.end_pos + noun_match.end(2),
                        confidence=0.7
                    ))

        return entities[:2]

    def _extract_deadlines(self, text: str) -> List[ExtractedEntity]:
        """Extract deadline entities"""
        entities = []

        for pattern, confidence in self.DEADLINE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1) if match.lastindex else match.group(0)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.DEADLINE,
                    value=value.strip(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence
                ))

        return entities[:2]

    def _extract_conditions(self, text: str) -> List[ExtractedEntity]:
        """Extract condition entities"""
        entities = []

        for pattern, confidence in self.CONDITION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1) if match.lastindex else match.group(0)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.CONDITION,
                    value=value.strip(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence
                ))

        return entities[:2]

    def _calculate_confidence(self, result: GoalExtractionResult) -> float:
        """
        Calculate overall confidence based on entity completeness

        Scoring:
        - ACTION present: +0.4
        - OBJECT present: +0.3
        - DEADLINE present: +0.2
        - CONDITION present: +0.1 (optional)
        """
        score = 0.0

        if result.action:
            action_entity = next((e for e in result.entities if e.entity_type == EntityType.ACTION), None)
            score += 0.4 * (action_entity.confidence if action_entity else 0.5)

        if result.object_target:
            object_entity = next((e for e in result.entities if e.entity_type == EntityType.OBJECT), None)
            score += 0.3 * (object_entity.confidence if object_entity else 0.5)

        if result.deadline:
            deadline_entity = next((e for e in result.entities if e.entity_type == EntityType.DEADLINE), None)
            score += 0.2 * (deadline_entity.confidence if deadline_entity else 0.5)

        if result.condition:
            score += 0.1  # Conditions are optional bonus

        return round(score, 2)

    def _generate_clarification_questions(self, result: GoalExtractionResult) -> List[str]:
        """Generate questions to clarify incomplete goals"""
        questions = []

        if not result.action:
            questions.append("What specifically would you like to accomplish?")

        if not result.object_target and result.action:
            questions.append(f"What exactly do you want to {result.action}?")

        if not result.deadline:
            questions.append("Is there a deadline or timeframe for this?")

        return questions


# Queue for low-confidence goals needing clarification
class ClarificationQueue:
    """Queue for goals needing clarification (IA-001)"""

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.queue_file = self.senter_root / "data" / "goals" / "clarification_queue.json"
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self._queue: List[Dict] = []
        self._load()

    def _load(self):
        """Load queue from disk"""
        if self.queue_file.exists():
            try:
                self._queue = json.loads(self.queue_file.read_text())
            except:
                self._queue = []

    def _save(self):
        """Save queue to disk"""
        self.queue_file.write_text(json.dumps(self._queue, indent=2))

    def add(self, extraction_result: GoalExtractionResult, original_text: str):
        """Add a goal needing clarification"""
        self._queue.append({
            "timestamp": datetime.now().isoformat(),
            "original_text": original_text,
            "extraction": extraction_result.to_dict(),
            "status": "pending"
        })
        self._save()

    def get_pending(self) -> List[Dict]:
        """Get pending clarifications"""
        return [q for q in self._queue if q["status"] == "pending"]

    def resolve(self, index: int, resolved_goal: Dict = None):
        """Mark a clarification as resolved"""
        if 0 <= index < len(self._queue):
            self._queue[index]["status"] = "resolved"
            if resolved_goal:
                self._queue[index]["resolved_goal"] = resolved_goal
            self._save()

    def clear_old(self, days: int = 7):
        """Clear old unresolved clarifications"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        self._queue = [q for q in self._queue
                       if q["status"] == "resolved" or q["timestamp"] > cutoff]
        self._save()

# Try to import embedding function for relevance search
try:
    from embedding_router import embed_text, cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        from Functions.embedding_router import embed_text, cosine_similarity
        EMBEDDINGS_AVAILABLE = True
    except ImportError:
        EMBEDDINGS_AVAILABLE = False

# Try to import LLM for goal extraction
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class SubTask:
    """A sub-task within a goal"""
    task: str
    status: str = "pending"  # pending, in_progress, completed, waiting

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SubTask":
        return cls(**data)


@dataclass
class Goal:
    """A user goal extracted from conversations"""
    id: str
    description: str
    created: str
    status: str = "active"  # active, completed, abandoned, paused
    related_conversations: list[str] = field(default_factory=list)
    sub_tasks: list[SubTask] = field(default_factory=list)
    last_mentioned: str = ""
    category: str = "general"
    priority: str = "medium"  # low, medium, high
    deadline: Optional[str] = None
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sub_tasks"] = [st if isinstance(st, dict) else st.to_dict() for st in self.sub_tasks]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        data = data.copy()
        if "sub_tasks" in data:
            data["sub_tasks"] = [
                SubTask.from_dict(st) if isinstance(st, dict) else st
                for st in data["sub_tasks"]
            ]
        return cls(**data)


class GoalTracker:
    """Tracks user goals across sessions"""

    def __init__(self, senter_root: Path = None, message_bus=None):
        self.senter_root = senter_root or Path(".")
        self.goals_path = self.senter_root / "data" / "goals.json"
        self.message_bus = message_bus  # CG-008: Optional message bus for goal execution

        # Ensure data directory exists
        self.goals_path.parent.mkdir(parents=True, exist_ok=True)

        # Load goals
        self.goals = self._load_goals()
        self._last_id = max([int(g.id.split("_")[1]) for g in self.goals] + [0])

        logger.info(f"Goal tracker initialized with {len(self.goals)} goals")

    def _load_goals(self) -> list[Goal]:
        """Load goals from JSON file"""
        if not self.goals_path.exists():
            return []

        try:
            data = json.loads(self.goals_path.read_text())
            return [Goal.from_dict(g) for g in data.get("goals", [])]
        except Exception as e:
            logger.error(f"Error loading goals: {e}")
            return []

    def _save_goals(self):
        """Save goals to JSON file"""
        try:
            data = {"goals": [g.to_dict() for g in self.goals]}
            self.goals_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error saving goals: {e}")

    def _generate_id(self) -> str:
        """Generate unique goal ID"""
        self._last_id += 1
        return f"goal_{self._last_id:04d}"

    def extract_goals_from_message(
        self, user_message: str, assistant_response: str = "", conversation_id: str = ""
    ) -> list[Goal]:
        """
        Extract goals from a conversation turn using LLM.

        Args:
            user_message: User's message
            assistant_response: Assistant's response (for context)
            conversation_id: ID of current conversation

        Returns:
            List of extracted Goal objects
        """
        # First, quick heuristic check - does this look like it contains goals?
        goal_indicators = [
            r"\bi need to\b", r"\bi want to\b", r"\bi have to\b",
            r"\bi should\b", r"\bi must\b", r"\bi'm going to\b",
            r"\bmy goal is\b", r"\bplanning to\b", r"\bworking on\b",
            r"\btrying to\b", r"\bhope to\b", r"\baim to\b",
            r"\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\bby (next week|tomorrow|end of)\b", r"\bdeadline\b",
            r"\bfinish\b.*\bproject\b", r"\bcomplete\b",
        ]

        message_lower = user_message.lower()
        has_goal_indicator = any(re.search(pattern, message_lower) for pattern in goal_indicators)

        if not has_goal_indicator:
            return []

        # Use LLM to extract structured goals
        if OLLAMA_AVAILABLE:
            return self._extract_with_llm(user_message, assistant_response, conversation_id)

        # Fallback: simple pattern extraction
        return self._extract_simple(user_message, conversation_id)

    def _extract_with_llm(
        self, user_message: str, assistant_response: str, conversation_id: str
    ) -> list[Goal]:
        """Extract goals using LLM"""
        prompt = f"""Analyze this message and extract any goals, plans, or intentions the user mentions.

User message: "{user_message}"

If goals are found, respond with JSON like this:
{{"goals": [
  {{"description": "brief description of goal", "category": "work/personal/learning/health/other", "priority": "low/medium/high", "deadline": "date if mentioned or null"}}
]}}

If no clear goals are found, respond with: {{"goals": []}}

Only extract explicit intentions or goals. Don't infer goals that aren't clearly stated.
Respond with valid JSON only, no other text."""

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json().get("response", "")
                # Try to parse JSON from response
                try:
                    # Find JSON in response
                    json_match = re.search(r'\{.*\}', result, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        goals = []
                        for g in data.get("goals", []):
                            goal = Goal(
                                id=self._generate_id(),
                                description=g.get("description", ""),
                                created=datetime.now().isoformat(),
                                category=g.get("category", "general"),
                                priority=g.get("priority", "medium"),
                                deadline=g.get("deadline"),
                                related_conversations=[conversation_id] if conversation_id else [],
                                last_mentioned=datetime.now().isoformat()
                            )
                            # Generate embedding for the goal
                            if EMBEDDINGS_AVAILABLE:
                                goal.embedding = embed_text(goal.description)
                            goals.append(goal)
                        return goals
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM goal response: {result[:100]}")

        except Exception as e:
            logger.error(f"LLM goal extraction error: {e}")

        return self._extract_simple(user_message, conversation_id)

    def _extract_simple(self, user_message: str, conversation_id: str) -> list[Goal]:
        """Simple pattern-based goal extraction fallback"""
        goals = []

        # Pattern: "I need/want/have to [action]"
        patterns = [
            r"i (?:need|want|have) to ([\w\s]+?)(?:\.|,|$|by|before)",
            r"i'm (?:going to|planning to) ([\w\s]+?)(?:\.|,|$|by|before)",
            r"my goal is to ([\w\s]+?)(?:\.|,|$)",
            r"i'm working on ([\w\s]+?)(?:\.|,|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, user_message.lower())
            for match in matches:
                description = match.strip()
                if len(description) > 10:  # Filter out too short matches
                    goal = Goal(
                        id=self._generate_id(),
                        description=description.capitalize(),
                        created=datetime.now().isoformat(),
                        related_conversations=[conversation_id] if conversation_id else [],
                        last_mentioned=datetime.now().isoformat()
                    )
                    if EMBEDDINGS_AVAILABLE:
                        goal.embedding = embed_text(goal.description)
                    goals.append(goal)

        return goals

    def save_goal(self, goal: Goal, trigger_execution: bool = False):
        """Save a new goal (CG-008: optionally trigger execution)"""
        # Check for duplicates
        for existing in self.goals:
            if self._is_similar_goal(existing, goal):
                # Update existing goal instead
                existing.last_mentioned = datetime.now().isoformat()
                if goal.related_conversations:
                    existing.related_conversations.extend(goal.related_conversations)
                self._save_goals()
                logger.info(f"Updated existing goal: {existing.id}")
                return

        self.goals.append(goal)
        self._save_goals()
        logger.info(f"Saved new goal: {goal.id} - {goal.description[:50]}")

        # CG-008: Trigger execution pipeline if requested and message bus available
        if trigger_execution and self.message_bus:
            self.trigger_goal_execution(goal)

    def trigger_goal_execution(self, goal: Goal):
        """
        Trigger goal execution pipeline (CG-008).

        Sends GOAL_DETECTED message to task_engine, which will:
        1. Create an execution plan
        2. Execute all tasks
        3. Send GOAL_COMPLETE when done
        """
        if not self.message_bus:
            logger.warning(f"Cannot trigger execution for {goal.id}: no message bus")
            return

        try:
            # Import MessageType from message_bus
            import sys
            daemon_path = str(self.senter_root / "daemon")
            if daemon_path not in sys.path:
                sys.path.insert(0, daemon_path)
            try:
                from daemon.message_bus import MessageType
            except ImportError:
                from message_bus import MessageType

            self.message_bus.send(
                MessageType.GOAL_DETECTED,
                source="goal_tracker",
                target="task_engine",
                payload={
                    "goal_id": goal.id,
                    "description": goal.description,
                    "category": goal.category,
                    "priority": goal.priority,
                    "deadline": goal.deadline,
                    "sub_tasks": [st.to_dict() for st in goal.sub_tasks],
                }
            )
            logger.info(f"Goal execution triggered: {goal.id}")

        except Exception as e:
            logger.error(f"Failed to trigger goal execution: {e}")

    def _is_similar_goal(self, goal1: Goal, goal2: Goal) -> bool:
        """Check if two goals are similar"""
        # Use embeddings if available
        if EMBEDDINGS_AVAILABLE and goal1.embedding and goal2.embedding:
            similarity = cosine_similarity(goal1.embedding, goal2.embedding)
            return similarity > 0.85

        # Fallback: simple text comparison
        desc1 = goal1.description.lower()
        desc2 = goal2.description.lower()

        # Check word overlap
        words1 = set(desc1.split())
        words2 = set(desc2.split())
        overlap = len(words1 & words2) / max(len(words1 | words2), 1)

        return overlap > 0.7

    def get_active_goals(self) -> list[Goal]:
        """Get all active goals"""
        return [g for g in self.goals if g.status == "active"]

    def get_relevant_goals(self, query: str, limit: int = 3) -> list[Goal]:
        """Get goals relevant to a query using semantic search"""
        active_goals = self.get_active_goals()
        if not active_goals:
            return []

        if EMBEDDINGS_AVAILABLE:
            query_embedding = embed_text(query)
            if query_embedding:
                scored = []
                for goal in active_goals:
                    if goal.embedding:
                        sim = cosine_similarity(query_embedding, goal.embedding)
                        if sim > 0.4:  # Threshold
                            scored.append((sim, goal))

                scored.sort(key=lambda x: x[0], reverse=True)
                return [g for _, g in scored[:limit]]

        # Fallback: keyword matching
        query_words = set(query.lower().split())
        scored = []
        for goal in active_goals:
            goal_words = set(goal.description.lower().split())
            overlap = len(query_words & goal_words)
            if overlap > 0:
                scored.append((overlap, goal))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [g for _, g in scored[:limit]]

    def update_goal_status(self, goal_id: str, status: str):
        """Update goal status"""
        for goal in self.goals:
            if goal.id == goal_id:
                goal.status = status
                goal.last_mentioned = datetime.now().isoformat()
                self._save_goals()
                logger.info(f"Updated goal {goal_id} status to {status}")
                return True
        return False

    def add_subtask(self, goal_id: str, task: str, status: str = "pending"):
        """Add a subtask to a goal"""
        for goal in self.goals:
            if goal.id == goal_id:
                goal.sub_tasks.append(SubTask(task=task, status=status))
                goal.last_mentioned = datetime.now().isoformat()
                self._save_goals()
                return True
        return False

    def get_goals_context(self, query: str = "") -> str:
        """Get formatted goals context for system prompt"""
        if query:
            relevant = self.get_relevant_goals(query, limit=3)
        else:
            relevant = self.get_active_goals()[:5]

        if not relevant:
            return ""

        lines = ["[Your active goals:]"]
        for goal in relevant:
            status_icon = {"active": "*", "paused": "~", "completed": "+"}
            icon = status_icon.get(goal.status, "?")
            lines.append(f"  {icon} {goal.description}")
            if goal.deadline:
                lines.append(f"    Deadline: {goal.deadline}")
            for st in goal.sub_tasks[:3]:
                st_icon = "-" if st.status == "pending" else "+"
                lines.append(f"    {st_icon} {st.task}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get goal statistics"""
        active = len([g for g in self.goals if g.status == "active"])
        completed = len([g for g in self.goals if g.status == "completed"])
        return {
            "total": len(self.goals),
            "active": active,
            "completed": completed,
            "paused": len(self.goals) - active - completed
        }


# Convenience functions
def extract_goals(user_message: str, senter_root: Path = None) -> list[Goal]:
    """Extract goals from a message"""
    tracker = GoalTracker(senter_root)
    return tracker.extract_goals_from_message(user_message)


def get_active_goals(senter_root: Path = None) -> list[Goal]:
    """Get all active goals"""
    tracker = GoalTracker(senter_root)
    return tracker.get_active_goals()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Senter Goal Tracker")
    parser.add_argument("--list", "-l", action="store_true", help="List all goals")
    parser.add_argument("--active", "-a", action="store_true", help="List active goals")
    parser.add_argument("--extract", "-e", help="Extract goals from text")
    parser.add_argument("--stats", action="store_true", help="Show goal statistics")
    parser.add_argument("--complete", help="Mark goal as completed (by ID)")
    parser.add_argument("--test", action="store_true", help="Run test extraction")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tracker = GoalTracker(Path("."))

    if args.stats:
        stats = tracker.get_stats()
        print("\nGoal Statistics:")
        print(f"  Total: {stats['total']}")
        print(f"  Active: {stats['active']}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Paused: {stats['paused']}")

    elif args.list:
        print(f"\nAll Goals ({len(tracker.goals)}):")
        for goal in tracker.goals:
            status_icon = {"active": "*", "completed": "+", "paused": "~", "abandoned": "x"}
            icon = status_icon.get(goal.status, "?")
            print(f"\n  [{icon}] {goal.id}: {goal.description}")
            print(f"      Status: {goal.status} | Created: {goal.created[:10]}")
            if goal.sub_tasks:
                print(f"      Sub-tasks: {len(goal.sub_tasks)}")

    elif args.active:
        active = tracker.get_active_goals()
        print(f"\nActive Goals ({len(active)}):")
        for goal in active:
            print(f"\n  * {goal.id}: {goal.description}")
            if goal.deadline:
                print(f"    Deadline: {goal.deadline}")
            for st in goal.sub_tasks:
                icon = "+" if st.status == "completed" else "-"
                print(f"    {icon} {st.task}")

    elif args.extract:
        print(f"\nExtracting goals from: \"{args.extract}\"")
        goals = tracker.extract_goals_from_message(args.extract)
        if goals:
            print(f"\nFound {len(goals)} goal(s):")
            for goal in goals:
                print(f"  - {goal.description}")
                print(f"    Category: {goal.category} | Priority: {goal.priority}")
                # Save it
                tracker.save_goal(goal)
                print(f"    Saved as: {goal.id}")
        else:
            print("No goals found")

    elif args.complete:
        if tracker.update_goal_status(args.complete, "completed"):
            print(f"Marked {args.complete} as completed")
        else:
            print(f"Goal {args.complete} not found")

    elif args.test:
        print("\nRunning goal extraction test...")

        test_messages = [
            "I need to finish my presentation by Friday",
            "I'm working on learning Python for my new job",
            "Can you help me with this code?",  # No goal
            "I want to lose 10 pounds before summer",
            "My goal is to read 20 books this year",
        ]

        for msg in test_messages:
            print(f"\nMessage: \"{msg}\"")
            goals = tracker.extract_goals_from_message(msg)
            if goals:
                for g in goals:
                    print(f"  -> Goal: {g.description}")
            else:
                print("  -> No goals detected")

        print("\nTest complete!")

    else:
        parser.print_help()

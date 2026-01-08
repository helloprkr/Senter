"""
Proactive suggestion engine.

Senter doesn't just respond - it anticipates and suggests.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.engine import Senter
    from .goals import GoalDetector
    from .activity import ActivityMonitor


@dataclass
class GoalDerivedTask:
    """A task created from an active goal."""
    task_id: str
    goal_id: str
    task_type: str  # research, plan
    description: str
    parameters: Dict[str, Any]
    origin: str = "goal_derived"
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class NeedPattern:
    """A detected recurring need pattern."""
    topic: str
    frequency: int  # How often this need occurs
    time_slots: List[int]  # Hours of day when this need typically occurs
    confidence: float  # Confidence this is a real pattern
    last_occurrence: datetime
    associated_activities: List[str]  # e.g., ["coding", "writing"]

    def matches_time(self, hour: int) -> bool:
        """Check if the current hour matches this pattern's time slots."""
        return hour in self.time_slots


@dataclass
class PredictedNeed:
    """A predicted future need based on patterns."""
    topic: str
    predicted_time: datetime
    confidence: float
    source_pattern: str  # ID or description of source pattern
    prefetch_query: str  # Query for pre-fetching research


class ProactiveSuggestionEngine:
    """
    Generates proactive suggestions based on:
    - User goals
    - Conversation patterns
    - Time-based triggers
    - Context awareness
    """

    def __init__(
        self,
        engine: "Senter",
        goal_detector: Optional["GoalDetector"] = None,
        activity_monitor: Optional["ActivityMonitor"] = None
    ):
        self.engine = engine
        self.goal_detector = goal_detector
        self.activity_monitor = activity_monitor
        self.last_suggestions: Dict[str, datetime] = {}
        self.suggestion_cooldown = timedelta(hours=4)
        self.created_task_ids: Dict[str, datetime] = {}  # goal_id -> last task creation time
        self.task_creation_cooldown = timedelta(hours=12)  # Don't create tasks too often
        self.min_trust_for_tasks = 0.7  # Minimum trust level to create tasks
        # Need pattern tracking
        self.need_patterns: Dict[str, NeedPattern] = {}
        self.prefetched_research: Dict[str, Dict[str, Any]] = {}  # topic -> research results
        self.min_pattern_frequency = 3  # Minimum occurrences to consider a pattern
        # Activity-based suggestion thresholds
        self.break_suggestion_minutes = 120  # Suggest break after 2 hours
        self.activity_resource_map: Dict[str, List[str]] = {
            "coding": ["documentation", "stack overflow", "code examples"],
            "writing": ["grammar tools", "style guides", "templates"],
            "research": ["academic papers", "tutorials", "expert articles"],
            "learning": ["courses", "practice exercises", "quizzes"],
            "design": ["inspiration", "color tools", "templates"],
        }

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

        # Activity-context-aware suggestions
        if self.activity_monitor:
            activity_suggestions = await self._activity_context_suggestions()
            suggestions.extend(activity_suggestions)

        # Filter by cooldown and trust level
        filtered = self._filter_suggestions(suggestions)

        return filtered[:3]  # Return top 3

    def create_tasks_from_goals(self) -> List[GoalDerivedTask]:
        """
        Create background tasks from active goals.

        Requirements:
        - Trust level must be > 0.7
        - Respects task creation cooldown per goal
        - Task type is 'research' for learning goals, 'plan' for others
        - All tasks have origin='goal_derived'
        """
        tasks = []

        # Check trust level
        trust = getattr(self.engine, "trust", None)
        trust_level = trust.level if trust else 0.5

        if trust_level < self.min_trust_for_tasks:
            return tasks  # Not enough trust to create tasks

        if not self.goal_detector:
            return tasks

        # Get active goals
        goals = self.goal_detector.get_active_goals()

        for goal in goals:
            # Check cooldown for this goal
            if not self._should_create_task_for_goal(goal.id):
                continue

            # Determine task type based on goal category
            if goal.category == "learning":
                task_type = "research"
                description = f"Research resources and latest information about: {goal.description}"
            else:
                task_type = "plan"
                description = f"Create actionable steps for: {goal.description}"

            # Create task
            task = GoalDerivedTask(
                task_id=f"goal_{goal.id}_{int(datetime.now().timestamp())}",
                goal_id=goal.id,
                task_type=task_type,
                description=description,
                parameters={
                    "goal_description": goal.description,
                    "goal_category": goal.category,
                    "goal_confidence": goal.confidence,
                    "goal_progress": goal.progress,
                },
                origin="goal_derived",
            )

            tasks.append(task)

            # Mark as created
            self.created_task_ids[goal.id] = datetime.now()

        return tasks

    def _should_create_task_for_goal(self, goal_id: str) -> bool:
        """Check if we should create a task for this goal (respects cooldown)."""
        if goal_id not in self.created_task_ids:
            return True

        elapsed = datetime.now() - self.created_task_ids[goal_id]
        return elapsed >= self.task_creation_cooldown

    def get_task_creation_status(self) -> Dict[str, Any]:
        """Get status of task creation from goals."""
        trust = getattr(self.engine, "trust", None)
        trust_level = trust.level if trust else 0.5

        return {
            "trust_level": trust_level,
            "min_trust_required": self.min_trust_for_tasks,
            "can_create_tasks": trust_level >= self.min_trust_for_tasks,
            "goals_with_recent_tasks": len(self.created_task_ids),
            "cooldown_hours": self.task_creation_cooldown.total_seconds() / 3600,
        }

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

    async def _activity_context_suggestions(self) -> List[Dict]:
        """
        Generate suggestions based on ActivityMonitor context.

        Considers:
        - Current activity type for relevant resource suggestions
        - Activity duration for break suggestions
        - Project context for targeted help
        """
        suggestions = []

        if not self.activity_monitor:
            return suggestions

        try:
            # Get current activity context
            current_context = self.activity_monitor.get_current_context()
            current_project = self.activity_monitor.get_current_project()
            activity_summary = self.activity_monitor.get_activity_summary(hours=4)

            # Check for break suggestion based on activity duration
            break_suggestion = self._check_break_needed(activity_summary)
            if break_suggestion:
                suggestions.append(break_suggestion)

            # Suggest relevant resources based on current activity
            resource_suggestion = self._suggest_resources_for_activity(
                current_context, current_project
            )
            if resource_suggestion:
                suggestions.append(resource_suggestion)

            # Context-specific suggestions
            context_suggestion = self._suggest_for_context(
                current_context, current_project, activity_summary
            )
            if context_suggestion:
                suggestions.append(context_suggestion)

        except Exception:
            # Activity suggestions are optional
            pass

        return suggestions

    def _check_break_needed(self, activity_summary: Dict[str, Any]) -> Optional[Dict]:
        """Check if user needs a break based on continuous activity."""
        if not activity_summary:
            return None

        # Get time spent on primary activity
        time_by_context = activity_summary.get("time_by_context", {})
        if not time_by_context:
            return None

        # Find the dominant activity and its duration
        for context, minutes in time_by_context.items():
            if minutes >= self.break_suggestion_minutes:
                suggestion_id = f"break_{context}"
                if self._should_suggest(suggestion_id):
                    hours = minutes / 60
                    return {
                        "type": "activity_break",
                        "id": suggestion_id,
                        "priority": 0.75,
                        "title": "Time for a break?",
                        "action": f"You've been {context} for {hours:.1f} hours. A short break might help you stay fresh!",
                        "task_type": "wellness",
                        "activity_context": context,
                        "duration_minutes": minutes,
                    }

        return None

    def _suggest_resources_for_activity(
        self,
        context: str,
        project: Optional[str]
    ) -> Optional[Dict]:
        """Suggest relevant resources based on current activity."""
        if not context:
            return None

        # Map context to resource types
        resources = self.activity_resource_map.get(context, [])
        if not resources:
            return None

        suggestion_id = f"resource_{context}"
        if not self._should_suggest(suggestion_id):
            return None

        resource_list = ", ".join(resources[:2])
        project_context = f" for {project}" if project else ""

        return {
            "type": "activity_resource",
            "id": suggestion_id,
            "priority": 0.5,
            "title": f"Resources for {context}",
            "action": f"I can help find {resource_list}{project_context}. Would you like me to research?",
            "task_type": "research",
            "activity_context": context,
            "project": project,
            "suggested_resources": resources,
        }

    def _suggest_for_context(
        self,
        context: str,
        project: Optional[str],
        activity_summary: Dict[str, Any]
    ) -> Optional[Dict]:
        """Generate context-specific suggestions."""
        if not context:
            return None

        suggestion_id = f"context_{context}_{project or 'general'}"
        if not self._should_suggest(suggestion_id):
            return None

        # Context-specific actions
        if context == "coding" and project:
            return {
                "type": "activity_coding",
                "id": suggestion_id,
                "priority": 0.45,
                "title": f"Working on {project}",
                "action": f"I see you're coding on {project}. Want me to help with documentation or code review?",
                "task_type": "assist",
                "activity_context": context,
                "project": project,
            }
        elif context == "research":
            return {
                "type": "activity_research",
                "id": suggestion_id,
                "priority": 0.45,
                "title": "Research assistance",
                "action": "I can help organize your research findings or find related sources. Interested?",
                "task_type": "research",
                "activity_context": context,
            }
        elif context == "writing":
            return {
                "type": "activity_writing",
                "id": suggestion_id,
                "priority": 0.45,
                "title": "Writing support",
                "action": "Need help with proofreading, restructuring, or expanding your writing?",
                "task_type": "assist",
                "activity_context": context,
            }

        return None

    def get_activity_suggestion_status(self) -> Dict[str, Any]:
        """Get status of activity-based suggestions."""
        return {
            "has_activity_monitor": self.activity_monitor is not None,
            "break_threshold_minutes": self.break_suggestion_minutes,
            "resource_categories": list(self.activity_resource_map.keys()),
        }

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

        # Show some suggestions even at low trust for MVP (bootstrap value)
        # More suggestions unlocked at higher trust
        if trust_level < 0.3:
            return []  # Only block at very low trust

        # Higher trust = more suggestions
        max_suggestions = 1 if trust_level < 0.5 else (2 if trust_level < 0.7 else 3)

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
            "need_patterns_count": len(self.need_patterns),
            "prefetched_research_count": len(self.prefetched_research),
        }

    def analyze_needs_patterns(
        self,
        episodes: Optional[List[Any]] = None,
        activity_context: Optional[str] = None
    ) -> List[NeedPattern]:
        """
        Analyze conversation patterns for recurring needs.

        Looks for:
        - Topics that come up repeatedly
        - Time-of-day correlations
        - Activity-based correlations
        """
        patterns = []

        if episodes is None:
            try:
                episodes = self.engine.memory._episodic.get_recent(limit=100)
            except Exception:
                return patterns

        if not episodes:
            return patterns

        # Track topic occurrences with time and context
        topic_occurrences: Dict[str, List[Dict[str, Any]]] = {}

        for ep in episodes:
            # Extract key topics (simple keyword extraction)
            words = ep.input.lower().split()
            # Filter to meaningful words
            topics = [
                w for w in words
                if len(w) > 4
                and w not in ["about", "would", "could", "should", "there", "their", "where", "which", "these", "those"]
            ]

            hour = getattr(ep, "timestamp", datetime.now()).hour if hasattr(ep, "timestamp") else datetime.now().hour
            context = activity_context or getattr(ep, "context", "general")

            for topic in set(topics):  # Dedupe within episode
                if topic not in topic_occurrences:
                    topic_occurrences[topic] = []
                topic_occurrences[topic].append({
                    "hour": hour,
                    "context": context,
                    "timestamp": getattr(ep, "timestamp", datetime.now()),
                })

        # Convert to patterns (topics with >= min_pattern_frequency occurrences)
        for topic, occurrences in topic_occurrences.items():
            if len(occurrences) >= self.min_pattern_frequency:
                # Calculate time slots (hours where this topic commonly occurs)
                hours = [o["hour"] for o in occurrences]
                time_slots = list(set(hours))

                # Calculate associated activities
                contexts = [o["context"] for o in occurrences]
                associated_activities = list(set(contexts))

                # Confidence based on consistency
                confidence = min(len(occurrences) / 10, 1.0)  # Cap at 1.0

                pattern = NeedPattern(
                    topic=topic,
                    frequency=len(occurrences),
                    time_slots=time_slots,
                    confidence=confidence,
                    last_occurrence=max(o["timestamp"] for o in occurrences),
                    associated_activities=associated_activities,
                )

                patterns.append(pattern)

                # Store in internal dict
                self.need_patterns[topic] = pattern

        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)

        return patterns

    def predict_needs(
        self,
        current_hour: Optional[int] = None,
        current_activity: Optional[str] = None
    ) -> List[PredictedNeed]:
        """
        Predict needs based on time-of-day + activity patterns.

        Returns list of predicted needs sorted by confidence.
        """
        predictions = []

        if current_hour is None:
            current_hour = datetime.now().hour

        for topic, pattern in self.need_patterns.items():
            # Check time match
            time_match = pattern.matches_time(current_hour)

            # Check activity match
            activity_match = (
                current_activity is not None
                and current_activity in pattern.associated_activities
            )

            # Calculate prediction confidence
            base_confidence = pattern.confidence
            if time_match:
                base_confidence *= 1.2  # Boost for time match
            if activity_match:
                base_confidence *= 1.3  # Boost for activity match

            # Only predict if reasonably confident
            if base_confidence >= 0.3 and (time_match or activity_match):
                prediction = PredictedNeed(
                    topic=topic,
                    predicted_time=datetime.now(),
                    confidence=min(base_confidence, 1.0),
                    source_pattern=f"freq:{pattern.frequency}, time:{time_match}, activity:{activity_match}",
                    prefetch_query=f"information about {topic}",
                )
                predictions.append(prediction)

        # Sort by confidence
        predictions.sort(key=lambda p: p.confidence, reverse=True)

        return predictions[:5]  # Return top 5 predictions

    async def prefetch_research_for_needs(
        self,
        predictions: Optional[List[PredictedNeed]] = None,
        model: Optional[Any] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Pre-fetch research for predicted needs.

        Returns dict of topic -> research results.
        """
        if predictions is None:
            predictions = self.predict_needs()

        results = {}

        # Use provided model or engine's model
        llm = model or (self.engine.model if self.engine else None)

        for prediction in predictions:
            topic = prediction.topic

            # Skip if already prefetched recently
            if topic in self.prefetched_research:
                existing = self.prefetched_research[topic]
                if existing.get("fetched_at"):
                    age = datetime.now() - existing["fetched_at"]
                    if age < timedelta(hours=6):  # Cache for 6 hours
                        results[topic] = existing
                        continue

            # Generate research using LLM if available
            if llm:
                try:
                    prompt = f"""Provide a brief, helpful summary about: {prediction.prefetch_query}

Include:
1. Key points to know
2. Common questions answered
3. Useful tips

Keep it concise (2-3 paragraphs)."""

                    summary = await llm.generate(prompt)

                    research = {
                        "topic": topic,
                        "summary": summary,
                        "confidence": prediction.confidence,
                        "fetched_at": datetime.now(),
                        "source": "llm_prefetch",
                    }

                    results[topic] = research
                    self.prefetched_research[topic] = research
                except Exception as e:
                    results[topic] = {
                        "topic": topic,
                        "error": str(e),
                        "fetched_at": datetime.now(),
                    }
            else:
                # No LLM, just mark as needing research
                results[topic] = {
                    "topic": topic,
                    "summary": f"Research needed for: {topic}",
                    "confidence": prediction.confidence,
                    "fetched_at": datetime.now(),
                    "source": "placeholder",
                }
                self.prefetched_research[topic] = results[topic]

        return results

    def get_prefetched_research(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get prefetched research for a topic if available."""
        return self.prefetched_research.get(topic)

    def clear_prefetched_research(self, topic: Optional[str] = None) -> None:
        """Clear prefetched research cache."""
        if topic:
            self.prefetched_research.pop(topic, None)
        else:
            self.prefetched_research.clear()

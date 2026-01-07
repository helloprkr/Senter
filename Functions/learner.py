#!/usr/bin/env python3
"""
Self-Learning System for Senter
Analyzes conversations to learn user preferences and adapt behavior
"""

import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict, field
from collections import Counter

logger = logging.getLogger("senter.learner")

# Try to import Ollama for LLM-based analysis
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class Insights:
    """Extracted insights from conversation analysis"""
    response_preference: Optional[str] = None  # brief, detailed, balanced
    formality: Optional[str] = None  # casual, professional, neutral
    code_language: Optional[str] = None  # detected preferred language
    topics: list[str] = field(default_factory=list)
    interaction_type: Optional[str] = None  # research, coding, creative, conversation
    feedback_signals: list[str] = field(default_factory=list)  # positive/negative signals
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Insights":
        return cls(**data)


class SenterLearner:
    """Learns from user interactions and adapts behavior"""

    def __init__(self, senter_root: Path = None):
        self.senter_root = senter_root or Path(".")
        self.profile_path = self.senter_root / "config" / "user_profile.json"
        self.insights_path = self.senter_root / "data" / "learning_insights.json"

        # Ensure data directory exists
        (self.senter_root / "data").mkdir(parents=True, exist_ok=True)

        # Load current profile and insights
        self.profile = self._load_profile()
        self.insights_history = self._load_insights_history()

        logger.info(f"Learner initialized with {len(self.insights_history)} historical insights")

    def _load_profile(self) -> dict:
        """Load user profile"""
        if not self.profile_path.exists():
            return self._create_default_profile()

        try:
            return json.loads(self.profile_path.read_text())
        except Exception as e:
            logger.error(f"Error loading profile: {e}")
            return self._create_default_profile()

    def _create_default_profile(self) -> dict:
        """Create default user profile with learning fields"""
        return {
            "learned_preferences": {
                "response_length": "balanced",
                "code_language": None,
                "formality": "neutral",
                "topics_of_interest": [],
                "last_updated": None
            },
            "interaction_patterns": {
                "total_sessions": 0,
                "total_messages": 0,
                "common_requests": [],
                "average_session_length": 0,
                "focus_usage": {}
            }
        }

    def _save_profile(self):
        """Save user profile"""
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)

            # Merge learned preferences into existing profile
            if self.profile_path.exists():
                existing = json.loads(self.profile_path.read_text())
                existing["learned_preferences"] = self.profile.get("learned_preferences", {})
                existing["interaction_patterns"] = self.profile.get("interaction_patterns", {})
                self.profile_path.write_text(json.dumps(existing, indent=2))
            else:
                self.profile_path.write_text(json.dumps(self.profile, indent=2))

            logger.info("Saved updated user profile")
        except Exception as e:
            logger.error(f"Error saving profile: {e}")

    def _load_insights_history(self) -> list[dict]:
        """Load historical insights"""
        if not self.insights_path.exists():
            return []

        try:
            return json.loads(self.insights_path.read_text())
        except Exception as e:
            logger.warning(f"Error loading insights: {e}")
            return []

    def _save_insights_history(self):
        """Save insights history"""
        try:
            # Keep only last 100 insights
            self.insights_history = self.insights_history[-100:]
            self.insights_path.write_text(json.dumps(self.insights_history, indent=2))
        except Exception as e:
            logger.error(f"Error saving insights: {e}")

    def analyze_conversation(self, messages: list[dict]) -> Insights:
        """
        Analyze a conversation to extract learning insights.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Insights object with extracted learnings
        """
        insights = Insights(timestamp=datetime.now().isoformat())

        if not messages:
            return insights

        # Extract user messages
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        all_text = " ".join(user_messages).lower()

        # Detect response preference from feedback signals
        insights.response_preference = self._detect_response_preference(messages)

        # Detect formality
        insights.formality = self._detect_formality(user_messages)

        # Detect code language preference
        insights.code_language = self._detect_code_language(all_text)

        # Extract topics
        insights.topics = self._extract_topics(all_text)

        # Detect interaction type
        insights.interaction_type = self._detect_interaction_type(all_text)

        # Extract feedback signals
        insights.feedback_signals = self._extract_feedback_signals(messages)

        return insights

    def _detect_response_preference(self, messages: list[dict]) -> str:
        """Detect if user prefers brief or detailed responses"""
        user_text = " ".join(m["content"].lower() for m in messages if m.get("role") == "user")

        # Signals for brief responses
        brief_signals = ["tldr", "briefly", "short", "quick", "summarize", "in short", "keep it short"]
        # Signals for detailed responses
        detail_signals = ["explain", "detail", "elaborate", "expand", "more about", "tell me more", "go deeper"]

        brief_count = sum(1 for s in brief_signals if s in user_text)
        detail_count = sum(1 for s in detail_signals if s in user_text)

        if brief_count > detail_count:
            return "brief"
        elif detail_count > brief_count:
            return "detailed"
        return "balanced"

    def _detect_formality(self, user_messages: list[str]) -> str:
        """Detect user's communication formality"""
        all_text = " ".join(user_messages).lower()

        # Casual indicators
        casual_patterns = [
            r"\bhey\b", r"\bhi\b", r"\byeah\b", r"\bnope\b",
            r"\bcool\b", r"\bawesome\b", r"\bthanks\b", r"\bgonna\b",
            r"!{2,}", r"\blol\b", r"\bhaha\b"
        ]

        # Professional indicators
        professional_patterns = [
            r"\bplease\b", r"\bkindly\b", r"\bwould you\b",
            r"\bcould you\b", r"\bthank you\b", r"\bregards\b",
            r"\baccordingly\b", r"\btherefore\b"
        ]

        casual_count = sum(1 for p in casual_patterns if re.search(p, all_text))
        professional_count = sum(1 for p in professional_patterns if re.search(p, all_text))

        if casual_count > professional_count + 2:
            return "casual"
        elif professional_count > casual_count + 2:
            return "professional"
        return "neutral"

    def _detect_code_language(self, text: str) -> Optional[str]:
        """Detect preferred programming language"""
        language_patterns = {
            "python": [r"\bpython\b", r"\.py\b", r"pip install", r"def \w+\("],
            "javascript": [r"\bjavascript\b", r"\bjs\b", r"node\b", r"npm\b", r"const \w+ ="],
            "typescript": [r"\btypescript\b", r"\.ts\b", r"interface \w+", r": string\b"],
            "rust": [r"\brust\b", r"cargo\b", r"fn \w+\(", r"let mut"],
            "go": [r"\bgolang\b", r"\bgo\b.*\bcode\b", r"func \w+\("],
            "java": [r"\bjava\b", r"public class", r"System\.out"],
            "c++": [r"\bc\+\+\b", r"\bcpp\b", r"#include", r"std::"],
        }

        counts = {}
        for lang, patterns in language_patterns.items():
            counts[lang] = sum(1 for p in patterns if re.search(p, text))

        if counts:
            best_lang = max(counts, key=counts.get)
            if counts[best_lang] > 0:
                return best_lang

        return None

    def _extract_topics(self, text: str) -> list[str]:
        """Extract topics of interest from text"""
        # Common topic keywords
        topic_keywords = {
            "ai": ["ai", "artificial intelligence", "machine learning", "ml", "neural", "llm"],
            "web": ["web", "html", "css", "frontend", "backend", "api"],
            "data": ["data", "database", "sql", "analytics", "visualization"],
            "devops": ["docker", "kubernetes", "ci/cd", "deployment", "aws", "cloud"],
            "security": ["security", "encryption", "auth", "vulnerability"],
            "mobile": ["mobile", "ios", "android", "app"],
            "startup": ["startup", "business", "investor", "funding", "pitch"],
            "productivity": ["productivity", "workflow", "automation", "efficiency"],
        }

        found_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                found_topics.append(topic)

        return found_topics[:5]  # Limit to top 5

    def _detect_interaction_type(self, text: str) -> str:
        """Detect the type of interaction"""
        if any(w in text for w in ["code", "function", "debug", "error", "implement"]):
            return "coding"
        elif any(w in text for w in ["research", "find", "search", "what is", "how does"]):
            return "research"
        elif any(w in text for w in ["write", "poem", "story", "creative", "draft"]):
            return "creative"
        elif any(w in text for w in ["plan", "goal", "task", "schedule"]):
            return "planning"
        return "conversation"

    def _extract_feedback_signals(self, messages: list[dict]) -> list[str]:
        """Extract positive/negative feedback signals"""
        signals = []

        for msg in messages:
            if msg.get("role") != "user":
                continue
            text = msg["content"].lower()

            # Positive signals
            if any(w in text for w in ["thanks", "perfect", "great", "exactly", "helpful"]):
                signals.append("positive")
            # Negative signals
            if any(w in text for w in ["wrong", "not what", "incorrect", "no,", "actually"]):
                signals.append("correction")
            # Expansion requests
            if any(w in text for w in ["more", "explain", "elaborate", "expand"]):
                signals.append("wants_more_detail")
            # Brevity requests
            if any(w in text for w in ["shorter", "brief", "summarize", "tldr"]):
                signals.append("wants_less_detail")

        return signals

    def update_user_profile(self, insights: Insights):
        """Update user profile based on insights"""
        prefs = self.profile.setdefault("learned_preferences", {})
        patterns = self.profile.setdefault("interaction_patterns", {})

        # Update response preference if detected
        if insights.response_preference and insights.response_preference != "balanced":
            prefs["response_length"] = insights.response_preference
            logger.info(f"Learned: User prefers {insights.response_preference} responses")

        # Update formality if detected
        if insights.formality and insights.formality != "neutral":
            prefs["formality"] = insights.formality
            logger.info(f"Learned: User communication style is {insights.formality}")

        # Update code language if detected
        if insights.code_language:
            prefs["code_language"] = insights.code_language
            logger.info(f"Learned: User prefers {insights.code_language}")

        # Update topics of interest
        existing_topics = prefs.get("topics_of_interest", [])
        for topic in insights.topics:
            if topic not in existing_topics:
                existing_topics.append(topic)
        prefs["topics_of_interest"] = existing_topics[-10:]  # Keep last 10

        # Update interaction patterns
        patterns["total_sessions"] = patterns.get("total_sessions", 0) + 1
        patterns["last_session"] = datetime.now().isoformat()

        # Track common request types
        if insights.interaction_type:
            request_counts = Counter(patterns.get("common_requests", []))
            request_counts[insights.interaction_type] += 1
            patterns["common_requests"] = [t for t, _ in request_counts.most_common(5)]

        prefs["last_updated"] = datetime.now().isoformat()

        # Save updated profile
        self._save_profile()

        # Save insights to history
        self.insights_history.append(insights.to_dict())
        self._save_insights_history()

    def get_learned_preferences(self) -> dict:
        """Get current learned preferences"""
        return self.profile.get("learned_preferences", {})

    def get_interaction_patterns(self) -> dict:
        """Get interaction patterns"""
        return self.profile.get("interaction_patterns", {})

    def get_system_prompt_additions(self) -> str:
        """Get additions to system prompt based on learned preferences"""
        prefs = self.get_learned_preferences()

        additions = []

        response_length = prefs.get("response_length")
        if response_length == "brief":
            additions.append("User prefers concise, brief responses.")
        elif response_length == "detailed":
            additions.append("User prefers detailed, thorough explanations.")

        formality = prefs.get("formality")
        if formality == "casual":
            additions.append("Use a casual, friendly tone.")
        elif formality == "professional":
            additions.append("Use a professional, formal tone.")

        code_lang = prefs.get("code_language")
        if code_lang:
            additions.append(f"When showing code examples, prefer {code_lang}.")

        topics = prefs.get("topics_of_interest", [])
        if topics:
            additions.append(f"User is interested in: {', '.join(topics)}.")

        if additions:
            return "\n[Learned about this user:]\n" + "\n".join(f"- {a}" for a in additions)
        return ""

    def get_stats(self) -> dict:
        """Get learning statistics"""
        prefs = self.get_learned_preferences()
        patterns = self.get_interaction_patterns()

        return {
            "total_insights": len(self.insights_history),
            "total_sessions": patterns.get("total_sessions", 0),
            "known_preferences": sum(1 for v in prefs.values() if v),
            "topics_learned": len(prefs.get("topics_of_interest", [])),
            "last_updated": prefs.get("last_updated")
        }


def analyze_recent_conversations(senter_root: Path = None, last_n: int = 10) -> list[Insights]:
    """Analyze recent conversations and return insights"""
    learner = SenterLearner(senter_root)

    # Load recent conversations from memory
    try:
        from memory import ConversationMemory
        memory = ConversationMemory(senter_root)
        conversations = memory.get_recent_conversations(last_n)

        all_insights = []
        for conv in conversations:
            insights = learner.analyze_conversation(conv.messages)
            all_insights.append(insights)

        return all_insights
    except ImportError:
        logger.warning("Memory module not available for learning analysis")
        return []


def apply_learnings(insights: list[Insights], senter_root: Path = None):
    """Apply learnings from insights to user profile"""
    learner = SenterLearner(senter_root)

    for insight in insights:
        learner.update_user_profile(insight)

    logger.info(f"Applied {len(insights)} insights to user profile")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Senter Self-Learning System")
    parser.add_argument("--stats", action="store_true", help="Show learning statistics")
    parser.add_argument("--preferences", "-p", action="store_true", help="Show learned preferences")
    parser.add_argument("--patterns", action="store_true", help="Show interaction patterns")
    parser.add_argument("--analyze", help="Analyze text for insights")
    parser.add_argument("--test", action="store_true", help="Run test analysis")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    learner = SenterLearner(Path("."))

    if args.stats:
        stats = learner.get_stats()
        print("\nLearning Statistics:")
        print(f"  Total insights: {stats['total_insights']}")
        print(f"  Total sessions: {stats['total_sessions']}")
        print(f"  Known preferences: {stats['known_preferences']}")
        print(f"  Topics learned: {stats['topics_learned']}")
        if stats['last_updated']:
            print(f"  Last updated: {stats['last_updated'][:16]}")

    elif args.preferences:
        prefs = learner.get_learned_preferences()
        print("\nLearned Preferences:")
        for key, value in prefs.items():
            if value:
                print(f"  {key}: {value}")

        # Show system prompt additions
        additions = learner.get_system_prompt_additions()
        if additions:
            print("\nSystem prompt additions:")
            print(additions)

    elif args.patterns:
        patterns = learner.get_interaction_patterns()
        print("\nInteraction Patterns:")
        for key, value in patterns.items():
            print(f"  {key}: {value}")

    elif args.analyze:
        messages = [{"role": "user", "content": args.analyze}]
        insights = learner.analyze_conversation(messages)
        print(f"\nAnalysis of: \"{args.analyze[:50]}...\"")
        print(f"  Response preference: {insights.response_preference}")
        print(f"  Formality: {insights.formality}")
        print(f"  Code language: {insights.code_language}")
        print(f"  Topics: {insights.topics}")
        print(f"  Interaction type: {insights.interaction_type}")
        print(f"  Feedback signals: {insights.feedback_signals}")

    elif args.test:
        print("\nRunning learning analysis test...")

        test_conversations = [
            [
                {"role": "user", "content": "Hey, can you help me debug this Python code? It's throwing an error."},
                {"role": "assistant", "content": "Sure! Please share the code and error message."},
                {"role": "user", "content": "Thanks! Here's the code... [code]"},
                {"role": "assistant", "content": "The issue is..."},
                {"role": "user", "content": "Perfect, that fixed it! You're awesome!"}
            ],
            [
                {"role": "user", "content": "Could you please provide a detailed explanation of how neural networks work?"},
                {"role": "assistant", "content": "Certainly. Neural networks are..."},
                {"role": "user", "content": "Thank you. Could you elaborate more on backpropagation?"}
            ],
            [
                {"role": "user", "content": "tldr what is rust programming language"},
                {"role": "assistant", "content": "Rust is a systems programming language..."},
                {"role": "user", "content": "cool thx"}
            ]
        ]

        for i, conv in enumerate(test_conversations, 1):
            print(f"\nConversation {i}:")
            insights = learner.analyze_conversation(conv)
            print(f"  Response pref: {insights.response_preference}")
            print(f"  Formality: {insights.formality}")
            print(f"  Code lang: {insights.code_language}")
            print(f"  Topics: {insights.topics}")
            print(f"  Type: {insights.interaction_type}")
            print(f"  Feedback: {insights.feedback_signals}")

            # Apply learnings
            learner.update_user_profile(insights)

        # Show final preferences
        print("\n\nFinal learned preferences:")
        prefs = learner.get_learned_preferences()
        for key, value in prefs.items():
            if value:
                print(f"  {key}: {value}")

        print("\nTest complete!")

    else:
        parser.print_help()

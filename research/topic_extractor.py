#!/usr/bin/env python3
"""
RE-001: Senter Notices What You're Curious About

Semantic topic extraction using LLM. Identifies genuine curiosity,
not just keyword mentions.

VALUE: After any conversation, Senter identifies what you're genuinely
curious about—not just keywords, but real interest signals.
"""

import json
import logging
import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger("senter.research.topic_extractor")

# Curiosity signal patterns (used for priority boosting)
CURIOSITY_SIGNALS = [
    r"\bi wonder\b",
    r"\bwhat is\b",
    r"\bhow does\b",
    r"\bhow do\b",
    r"\bwhy does\b",
    r"\bwhy do\b",
    r"\bcan you explain\b",
    r"\btell me about\b",
    r"\bi('m| am) curious\b",
    r"\bi('d| would) like to (know|learn|understand)\b",
    r"\bshould i\b",
    r"\bis it (worth|better|possible)\b",
    r"\bwhat('s| is) the (difference|best|right)\b",
]


@dataclass
class ExtractedTopic:
    """A topic extracted from conversation with metadata."""
    topic: str
    priority: float  # 0.0 to 1.0
    reason: str
    source_messages: List[str] = field(default_factory=list)
    extracted_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "priority": self.priority,
            "reason": self.reason,
            "source_messages": self.source_messages,
            "extracted_at": self.extracted_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedTopic":
        return cls(**data)


class TopicExtractor:
    """
    Extracts research-worthy topics from conversations using LLM.

    Key behaviors:
    - Identifies genuine curiosity (questions, "I wonder", etc.)
    - Assigns priority based on signal strength
    - Provides human-readable reason for extraction
    - Deduplicates against recently researched topics
    """

    # Prompt for LLM to extract topics
    EXTRACTION_PROMPT = """Analyze this conversation and identify topics the user seems genuinely curious about or interested in learning more about.

CONVERSATION:
{conversation}

INSTRUCTIONS:
1. Look for signals of genuine curiosity:
   - Direct questions ("what is", "how does", "why")
   - Expressions of wonder ("I wonder", "I'm curious")
   - Requests to learn ("tell me about", "explain")
   - Uncertainty that could benefit from research ("should I", "is it worth")

2. Ignore casual mentions without curiosity signals.

3. For each topic found, provide:
   - topic: The specific topic to research (concise, searchable)
   - priority: 0.0-1.0 based on how strong the curiosity signal is
   - reason: Why you think the user wants to know about this

Return JSON array. If no genuine curiosity detected, return empty array [].

EXAMPLE OUTPUT:
[
  {{"topic": "event sourcing patterns", "priority": 0.85, "reason": "User explicitly wondered if event sourcing would help with their audit requirements"}},
  {{"topic": "kubernetes networking", "priority": 0.6, "reason": "User asked how kubernetes handles network policies"}}
]

YOUR RESPONSE (JSON only, no markdown):"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        recently_researched_file: Optional[Path] = None,
        dedup_days: int = 7
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.dedup_days = dedup_days
        self.recently_researched_file = recently_researched_file

        # Load recently researched topics for deduplication
        self._recently_researched: Dict[str, float] = {}
        if recently_researched_file and recently_researched_file.exists():
            try:
                self._recently_researched = json.loads(recently_researched_file.read_text())
            except Exception as e:
                logger.warning(f"Could not load recently researched: {e}")

    def extract_topics(
        self,
        messages: List[Dict[str, str]],
        min_priority: float = 0.3
    ) -> List[ExtractedTopic]:
        """
        Extract research-worthy topics from conversation messages.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            min_priority: Minimum priority threshold (default 0.3)

        Returns:
            List of ExtractedTopic objects, sorted by priority descending
        """
        if not messages:
            return []

        # Format conversation for LLM
        conversation = self._format_conversation(messages)

        # Check for curiosity signals to boost priority
        curiosity_boost = self._detect_curiosity_signals(conversation)

        # Call LLM for extraction
        try:
            raw_topics = self._call_llm(conversation)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Fallback to pattern-based extraction
            raw_topics = self._fallback_extraction(messages)

        # Process and filter topics
        topics = []
        for raw in raw_topics:
            topic_text = raw.get("topic", "").strip()
            if not topic_text:
                continue

            # Calculate final priority with curiosity boost
            base_priority = raw.get("priority", 0.5)
            if curiosity_boost > 0:
                base_priority = min(1.0, base_priority + curiosity_boost * 0.2)

            # Skip low priority
            if base_priority < min_priority:
                continue

            # Skip recently researched
            if self._is_recently_researched(topic_text):
                logger.debug(f"Skipping recently researched: {topic_text}")
                continue

            # Find source messages
            source_msgs = self._find_source_messages(topic_text, messages)

            topics.append(ExtractedTopic(
                topic=topic_text,
                priority=round(base_priority, 2),
                reason=raw.get("reason", "Detected interest in this topic"),
                source_messages=source_msgs
            ))

        # Sort by priority
        topics.sort(key=lambda t: t.priority, reverse=True)

        return topics

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for LLM prompt."""
        lines = []
        for msg in messages[-20:]:  # Last 20 messages max
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _detect_curiosity_signals(self, text: str) -> int:
        """Count curiosity signals in text."""
        text_lower = text.lower()
        count = 0
        for pattern in CURIOSITY_SIGNALS:
            if re.search(pattern, text_lower):
                count += 1
        return count

    def _call_llm(self, conversation: str) -> List[Dict[str, Any]]:
        """Call Ollama for topic extraction."""
        import requests

        prompt = self.EXTRACTION_PROMPT.format(conversation=conversation)

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for consistency
                    "num_predict": 1000
                }
            },
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        text = result.get("response", "").strip()

        # Parse JSON from response
        try:
            # Try to extract JSON array
            if text.startswith("["):
                return json.loads(text)
            # Try to find JSON in response
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {text}")
            return []

    def _fallback_extraction(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Fallback pattern-based extraction when LLM fails."""
        topics = []
        text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        text_lower = text.lower()

        # Look for "I wonder about X" patterns
        wonder_match = re.search(r"i wonder (?:if |about |whether )(.+?)(?:\.|$)", text_lower)
        if wonder_match:
            topics.append({
                "topic": wonder_match.group(1).strip()[:50],
                "priority": 0.8,
                "reason": "User expressed wonder about this topic"
            })

        # Look for "what is X" patterns
        what_match = re.search(r"what (?:is|are) (.+?)(?:\?|$)", text_lower)
        if what_match:
            topics.append({
                "topic": what_match.group(1).strip()[:50],
                "priority": 0.7,
                "reason": "User asked what this is"
            })

        # Look for "how does X work" patterns
        how_match = re.search(r"how (?:does|do) (.+?) work", text_lower)
        if how_match:
            topics.append({
                "topic": how_match.group(1).strip()[:50],
                "priority": 0.75,
                "reason": "User asked how this works"
            })

        return topics

    def _is_recently_researched(self, topic: str) -> bool:
        """Check if topic was recently researched."""
        topic_lower = topic.lower()
        cutoff = time.time() - (self.dedup_days * 24 * 3600)

        for researched_topic, timestamp in self._recently_researched.items():
            if timestamp < cutoff:
                continue
            # Fuzzy match: check if topics are similar
            if (topic_lower in researched_topic.lower() or
                researched_topic.lower() in topic_lower or
                self._word_overlap(topic_lower, researched_topic.lower()) > 0.5):
                return True
        return False

    def _word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap ratio between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        overlap = len(words1 & words2)
        return overlap / min(len(words1), len(words2))

    def _find_source_messages(
        self,
        topic: str,
        messages: List[Dict[str, str]]
    ) -> List[str]:
        """Find messages that likely triggered this topic."""
        sources = []
        topic_words = set(topic.lower().split())

        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            content_words = set(content.lower().split())

            # Check for word overlap
            if topic_words & content_words:
                sources.append(content[:200])  # Truncate long messages

        return sources[:3]  # Max 3 source messages

    def mark_as_researched(self, topic: str):
        """Mark a topic as researched to prevent duplicates."""
        self._recently_researched[topic.lower()] = time.time()

        if self.recently_researched_file:
            try:
                self.recently_researched_file.parent.mkdir(parents=True, exist_ok=True)
                self.recently_researched_file.write_text(
                    json.dumps(self._recently_researched, indent=2)
                )
            except Exception as e:
                logger.warning(f"Could not save recently researched: {e}")

    def get_recently_researched(self) -> List[str]:
        """Get list of recently researched topics."""
        cutoff = time.time() - (self.dedup_days * 24 * 3600)
        return [
            topic for topic, ts in self._recently_researched.items()
            if ts >= cutoff
        ]


# CLI for testing
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Test topic extraction")
    parser.add_argument("--conversation", "-c", help="Conversation text to analyze")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    extractor = TopicExtractor()

    if args.interactive:
        print("\n=== Topic Extractor Interactive Mode ===")
        print("Enter conversation messages (empty line to finish):")

        messages = []
        while True:
            line = input("USER: ")
            if not line:
                break
            messages.append({"role": "user", "content": line})

            assistant = input("ASSISTANT (or empty): ")
            if assistant:
                messages.append({"role": "assistant", "content": assistant})

        if messages:
            print("\nExtracting topics...")
            topics = extractor.extract_topics(messages)

            if topics:
                print(f"\nFound {len(topics)} topic(s):\n")
                for t in topics:
                    print(f"  📌 {t.topic}")
                    print(f"     Priority: {t.priority}")
                    print(f"     Reason: {t.reason}")
                    print()
            else:
                print("\nNo research-worthy topics detected.")

    elif args.conversation:
        messages = [{"role": "user", "content": args.conversation}]
        topics = extractor.extract_topics(messages)

        print(f"\nTopics: {json.dumps([t.to_dict() for t in topics], indent=2)}")

    else:
        # Demo with sample conversation
        demo_messages = [
            {"role": "user", "content": "I'm working on a data pipeline project."},
            {"role": "assistant", "content": "That sounds interesting! What kind of data are you processing?"},
            {"role": "user", "content": "Mostly event data. I wonder if event sourcing would help with the audit requirements we have."},
            {"role": "assistant", "content": "Event sourcing could definitely help with audit trails. Would you like me to explain how it works?"},
            {"role": "user", "content": "Yeah, and I'm also curious about how it compares to traditional CRUD approaches."}
        ]

        print("\n=== Demo Conversation ===")
        for msg in demo_messages:
            print(f"{msg['role'].upper()}: {msg['content']}")

        print("\n=== Extracting Topics ===")
        topics = extractor.extract_topics(demo_messages)

        if topics:
            for t in topics:
                print(f"\n📌 Topic: {t.topic}")
                print(f"   Priority: {t.priority}")
                print(f"   Reason: {t.reason}")
        else:
            print("\nNo topics extracted (is Ollama running?)")

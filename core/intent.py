"""
Intent Parser - Understands what the user wants.

Not keyword matching - actual semantic understanding of intent.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from models.base import ModelInterface
    from coupling.human_model import HumanCognitiveState


@dataclass
class Intent:
    """Parsed user intent."""

    primary: str  # Main intent (what they want)
    entities: List[str] = field(default_factory=list)
    context: str = ""
    tone: Literal["positive", "negative", "neutral", "frustrated", "excited"] = "neutral"
    urgency: Literal["low", "medium", "high"] = "medium"
    capabilities_needed: List[str] = field(default_factory=list)
    raw_input: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.primary,
            "entities": self.entities,
            "context": self.context,
            "tone": self.tone,
            "urgency": self.urgency,
            "capabilities": self.capabilities_needed,
        }


class IntentParser:
    """
    Parses user input to extract intent.

    Uses the LLM for semantic understanding rather than
    simple keyword matching.
    """

    # Capability trigger mappings
    CAPABILITY_TRIGGERS = {
        "web_search": ["current", "latest", "news", "weather", "price", "today", "search"],
        "remember": ["remember", "don't forget", "note that", "keep in mind", "save"],
        "recall": ["what did", "when did", "remind me", "do you remember", "what was"],
    }

    def __init__(self, model: Optional["ModelInterface"] = None):
        self.model = model

    async def parse(
        self,
        input_text: str,
        cognitive_state: Optional["HumanCognitiveState"] = None,
    ) -> Intent:
        """
        Parse intent from user input.

        Uses LLM for deep understanding when available,
        falls back to heuristics otherwise.
        """
        if self.model:
            return await self._parse_with_model(input_text, cognitive_state)
        else:
            return self._parse_heuristic(input_text, cognitive_state)

    async def _parse_with_model(
        self,
        input_text: str,
        cognitive_state: Optional["HumanCognitiveState"] = None,
    ) -> Intent:
        """Use LLM for semantic intent parsing."""
        cognitive_info = ""
        if cognitive_state:
            cognitive_info = f"\nUser cognitive state: {cognitive_state.mode} mode, frustration={cognitive_state.frustration:.2f}"

        prompt = f"""Analyze this user input and extract:
1. Primary intent (what they want)
2. Entities mentioned
3. Implied context
4. Emotional tone
5. Urgency level

{cognitive_info}

Input: {input_text}

Respond in YAML format only:
intent: <main intent in 3-5 words>
entities: [<list of key entities>]
context: <implied context>
tone: <positive|negative|neutral|frustrated|excited>
urgency: <low|medium|high>"""

        try:
            response = await self.model.generate(prompt, max_tokens=200)
            parsed = yaml.safe_load(response)

            if not isinstance(parsed, dict):
                raise ValueError("Invalid YAML response")

            return Intent(
                primary=parsed.get("intent", input_text[:50]),
                entities=parsed.get("entities", []),
                context=parsed.get("context", ""),
                tone=parsed.get("tone", "neutral"),
                urgency=parsed.get("urgency", "medium"),
                capabilities_needed=self._detect_capabilities(input_text),
                raw_input=input_text,
            )
        except Exception:
            # Fall back to heuristic parsing
            return self._parse_heuristic(input_text, cognitive_state)

    def _parse_heuristic(
        self,
        input_text: str,
        cognitive_state: Optional["HumanCognitiveState"] = None,
    ) -> Intent:
        """Heuristic-based intent parsing (fallback)."""
        input_lower = input_text.lower()

        # Detect tone - check frustrated BEFORE excited since "!" can appear in both
        tone: Literal["positive", "negative", "neutral", "frustrated", "excited"] = "neutral"
        frustrated_words = ["frustrated", "annoying", "broken", "ugh", "error", "bug", "debug", "fix", "wrong", "fail", "crash"]
        excited_words = ["excited", "amazing", "wow", "can't wait"]

        if any(w in input_lower for w in ["thanks", "great", "awesome", "love"]):
            tone = "positive"
        elif any(w in input_lower for w in frustrated_words):
            tone = "frustrated"
        elif any(w in input_lower for w in excited_words) or (input_text.count("!") >= 2):
            tone = "excited"

        # Detect urgency
        urgency: Literal["low", "medium", "high"] = "medium"
        if any(w in input_lower for w in ["asap", "urgent", "immediately", "now"]):
            urgency = "high"
        elif any(w in input_lower for w in ["no rush", "whenever", "eventually"]):
            urgency = "low"

        # Use cognitive state if available
        if cognitive_state:
            if cognitive_state.frustration > 0.5:
                tone = "frustrated"
            if cognitive_state.time_pressure in ("high", "urgent"):
                urgency = "high"

        return Intent(
            primary=input_text[:50],  # First 50 chars as intent summary
            entities=self._extract_entities(input_text),
            context="",
            tone=tone,
            urgency=urgency,
            capabilities_needed=self._detect_capabilities(input_text),
            raw_input=input_text,
        )

    def _detect_capabilities(self, input_text: str) -> List[str]:
        """Detect which capabilities are needed."""
        input_lower = input_text.lower()
        needed = []

        for capability, triggers in self.CAPABILITY_TRIGGERS.items():
            if any(trigger in input_lower for trigger in triggers):
                needed.append(capability)

        return needed

    def _extract_entities(self, input_text: str) -> List[str]:
        """Extract potential entities from text (simple heuristic)."""
        # Simple: extract capitalized words that aren't at start of sentence
        words = input_text.split()
        entities = []

        for i, word in enumerate(words):
            # Skip first word of sentences
            if i > 0 and word[0].isupper() and word.isalpha():
                entities.append(word)

        return entities[:5]  # Limit to 5 entities

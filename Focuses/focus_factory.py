#!/usr/bin/env python3
"""
Focus Factory - Dynamic Focus Creation
Creates new Focus directories and SENTER.md files with user's model

Enhanced for:
- FA-001: Dynamic focus creation from conversation
- FA-002: Focus merging for overlapping topics
- FA-003: Focus evolution tracking
"""

import os
import sys
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import numpy as np

logger = logging.getLogger('senter.focus_factory')

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Focuses"))
from senter_md_parser import SenterMdParser


# ========== FA-001: Topic Frequency Detection ==========

@dataclass
class TopicMention:
    """Single topic mention in conversation"""
    topic: str
    timestamp: float
    query_text: str
    confidence: float = 0.5


@dataclass
class TopicFrequencyResult:
    """Result of topic frequency analysis"""
    topic: str
    mention_count: int
    first_seen: float
    last_seen: float
    confidence: float
    sample_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TopicFrequencyTracker:
    """
    Track topic mentions to detect when new focus should be created (FA-001)

    Triggers focus creation when:
    - Topic mentioned 5+ times in 24 hours
    - No existing focus matches the topic
    """

    MIN_MENTIONS = 5
    TIME_WINDOW_HOURS = 24

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.mentions_file = self.senter_root / "data" / "learning" / "topic_mentions.json"
        self.mentions_file.parent.mkdir(parents=True, exist_ok=True)
        self._mentions: Dict[str, List[TopicMention]] = {}
        self._load_mentions()

    def _load_mentions(self):
        """Load topic mentions from disk"""
        if self.mentions_file.exists():
            try:
                data = json.loads(self.mentions_file.read_text())
                for topic, mentions in data.items():
                    self._mentions[topic] = [
                        TopicMention(**m) for m in mentions
                    ]
            except Exception as e:
                logger.warning(f"Could not load mentions: {e}")
                self._mentions = {}

    def _save_mentions(self):
        """Save topic mentions to disk"""
        data = {}
        for topic, mentions in self._mentions.items():
            data[topic] = [asdict(m) for m in mentions]
        self.mentions_file.write_text(json.dumps(data, indent=2))

    def record_mention(self, topic: str, query_text: str, confidence: float = 0.5):
        """Record a topic mention from conversation"""
        mention = TopicMention(
            topic=topic.lower().strip(),
            timestamp=time.time(),
            query_text=query_text[:200],  # Truncate
            confidence=confidence
        )

        if mention.topic not in self._mentions:
            self._mentions[mention.topic] = []

        self._mentions[mention.topic].append(mention)
        self._cleanup_old_mentions()
        self._save_mentions()

    def _cleanup_old_mentions(self):
        """Remove mentions older than time window"""
        cutoff = time.time() - (self.TIME_WINDOW_HOURS * 3600)
        for topic in list(self._mentions.keys()):
            self._mentions[topic] = [
                m for m in self._mentions[topic]
                if m.timestamp > cutoff
            ]
            if not self._mentions[topic]:
                del self._mentions[topic]

    def get_frequent_topics(self, min_mentions: int = None) -> List[TopicFrequencyResult]:
        """Get topics that exceed mention threshold"""
        min_mentions = min_mentions or self.MIN_MENTIONS
        self._cleanup_old_mentions()

        results = []
        for topic, mentions in self._mentions.items():
            if len(mentions) >= min_mentions:
                avg_confidence = sum(m.confidence for m in mentions) / len(mentions)
                results.append(TopicFrequencyResult(
                    topic=topic,
                    mention_count=len(mentions),
                    first_seen=min(m.timestamp for m in mentions),
                    last_seen=max(m.timestamp for m in mentions),
                    confidence=avg_confidence,
                    sample_queries=[m.query_text for m in mentions[:3]]
                ))

        return sorted(results, key=lambda x: x.mention_count, reverse=True)

    def should_create_focus(self, topic: str) -> bool:
        """Check if topic has enough mentions to create focus"""
        topic = topic.lower().strip()
        self._cleanup_old_mentions()

        mentions = self._mentions.get(topic, [])
        return len(mentions) >= self.MIN_MENTIONS

    def clear_topic(self, topic: str):
        """Clear mentions for a topic (after focus created)"""
        topic = topic.lower().strip()
        if topic in self._mentions:
            del self._mentions[topic]
            self._save_mentions()


# ========== FA-002: Focus Merging ==========

@dataclass
class FocusSimilarityResult:
    """Result of focus similarity comparison"""
    focus_a: str
    focus_b: str
    similarity: float
    shared_topics: List[str] = field(default_factory=list)
    merge_recommended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FocusMerger:
    """
    Detect and merge overlapping focuses (FA-002)

    Detects when two focuses have:
    - Embedding similarity > 0.9
    - Shared system prompt concepts
    """

    SIMILARITY_THRESHOLD = 0.9

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.parser = SenterMdParser(senter_root)
        self._embedding_available = self._check_embedding_support()

    def _check_embedding_support(self) -> bool:
        """Check if embedding support is available"""
        try:
            sys.path.insert(0, str(self.senter_root / "Functions"))
            from embedding_utils import get_default_embedding_model
            return get_default_embedding_model() is not None
        except Exception:
            return False

    def compute_similarity(self, focus_a: str, focus_b: str) -> FocusSimilarityResult:
        """Compute similarity between two focuses"""
        config_a = self.parser.load_focus_config(focus_a)
        config_b = self.parser.load_focus_config(focus_b)

        prompt_a = config_a.get("system_prompt", "")
        prompt_b = config_b.get("system_prompt", "")

        # Try embedding similarity first
        if self._embedding_available:
            similarity = self._embedding_similarity(prompt_a, prompt_b)
        else:
            similarity = self._keyword_similarity(prompt_a, prompt_b)

        # Find shared topics
        shared = self._find_shared_topics(prompt_a, prompt_b)

        return FocusSimilarityResult(
            focus_a=focus_a,
            focus_b=focus_b,
            similarity=round(similarity, 3),
            shared_topics=shared,
            merge_recommended=similarity >= self.SIMILARITY_THRESHOLD
        )

    def _embedding_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate cosine similarity using embeddings"""
        try:
            sys.path.insert(0, str(self.senter_root / "Functions"))
            from embedding_utils import create_embeddings, get_default_embedding_model

            model = get_default_embedding_model()
            embeddings = create_embeddings([text_a, text_b], model)

            # Cosine similarity
            norm_a = np.linalg.norm(embeddings[0])
            norm_b = np.linalg.norm(embeddings[1])

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return float(np.dot(embeddings[0], embeddings[1]) / (norm_a * norm_b))
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}")
            return self._keyword_similarity(text_a, text_b)

    def _keyword_similarity(self, text_a: str, text_b: str) -> float:
        """Fallback: keyword-based Jaccard similarity"""
        import re

        words_a = set(re.findall(r'\b[a-z]+\b', text_a.lower()))
        words_b = set(re.findall(r'\b[a-z]+\b', text_b.lower()))

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for', 'with',
                      'you', 'your', 'this', 'that', 'it', 'as', 'by', 'from'}
        words_a -= stop_words
        words_b -= stop_words

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def _find_shared_topics(self, prompt_a: str, prompt_b: str) -> List[str]:
        """Find shared topic keywords between prompts"""
        import re

        # Extract potential topics (capitalized words, noun phrases)
        topic_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'

        topics_a = set(re.findall(topic_pattern, prompt_a))
        topics_b = set(re.findall(topic_pattern, prompt_b))

        # Also check for common domain words
        domain_words = {'code', 'coding', 'programming', 'research', 'writing',
                        'creative', 'technology', 'science', 'data', 'analysis'}

        words_a = set(prompt_a.lower().split())
        words_b = set(prompt_b.lower().split())

        shared_domain = (words_a & words_b & domain_words)
        shared_topics = topics_a & topics_b

        return list(shared_topics) + list(shared_domain)

    def find_merge_candidates(self) -> List[FocusSimilarityResult]:
        """Find all focus pairs that should be merged"""
        focuses = self.parser.list_all_focuses()

        # Filter out internal focuses
        focuses = [f for f in focuses if not f.startswith('internal')]

        candidates = []
        seen_pairs = set()

        for i, focus_a in enumerate(focuses):
            for focus_b in focuses[i+1:]:
                pair_key = tuple(sorted([focus_a, focus_b]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                result = self.compute_similarity(focus_a, focus_b)
                if result.merge_recommended:
                    candidates.append(result)

        return sorted(candidates, key=lambda x: x.similarity, reverse=True)

    def merge_focuses(self, focus_a: str, focus_b: str, new_name: str) -> Path:
        """
        Merge two focuses into a new combined focus

        Requires user approval before calling this method.
        """
        config_a = self.parser.load_focus_config(focus_a)
        config_b = self.parser.load_focus_config(focus_b)

        # Combine system prompts
        prompt_a = config_a.get("system_prompt", "")
        prompt_b = config_b.get("system_prompt", "")

        combined_prompt = self._combine_prompts(prompt_a, prompt_b, new_name)

        # Merge context
        context_a = config_a.get("context", {}).get("content", "")
        context_b = config_b.get("context", {}).get("content", "")
        combined_context = f"{context_a}\n\n---\n\n{context_b}"

        # Create merged focus
        factory = FocusFactory(self.senter_root)
        focus_dir = factory.create_focus(new_name, combined_context)

        # Update system prompt
        senter_file = focus_dir / "SENTER.md"
        content = senter_file.read_text()

        # Replace system prompt in the file
        import re
        content = re.sub(
            r'system_prompt: \|.*?(?=\n\w)',
            f'system_prompt: |\n{combined_prompt}\n\n',
            content,
            flags=re.DOTALL
        )
        senter_file.write_text(content)

        # Log merge
        self._log_merge(focus_a, focus_b, new_name)

        return focus_dir

    def _combine_prompts(self, prompt_a: str, prompt_b: str, new_name: str) -> str:
        """Combine two system prompts intelligently"""
        return f"""  You are Senter's agent for the {new_name} Focus.

  This focus combines knowledge from multiple related domains.

  Original domain knowledge:
  {prompt_a.strip()}

  Additional domain knowledge:
  {prompt_b.strip()}

  Use all available context to provide comprehensive assistance."""

    def _log_merge(self, focus_a: str, focus_b: str, new_name: str):
        """Log merge operation for audit trail"""
        log_file = self.senter_root / "data" / "focuses" / "merge_history.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        history = []
        if log_file.exists():
            try:
                history = json.loads(log_file.read_text())
            except:
                pass

        history.append({
            "timestamp": datetime.now().isoformat(),
            "focus_a": focus_a,
            "focus_b": focus_b,
            "merged_to": new_name
        })

        log_file.write_text(json.dumps(history, indent=2))


# ========== FA-003: Focus Evolution Tracking ==========

@dataclass
class FocusEvolutionEntry:
    """Single evolution entry for a focus"""
    timestamp: float
    change_type: str  # 'created', 'context_updated', 'goals_updated', 'wiki_updated'
    section_changed: str
    old_hash: str  # Hash of previous content
    new_hash: str  # Hash of new content
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FocusEvolutionTracker:
    """
    Track focus evolution over time (FA-003)

    Records changes to:
    - Context sections
    - Goals
    - Wiki content
    - System prompts
    """

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.parser = SenterMdParser(senter_root)

    def _get_evolution_file(self, focus_name: str) -> Path:
        """Get evolution file path for focus"""
        focus_dir = self.senter_root / "Focuses" / focus_name
        return focus_dir / ".evolution.json"

    def _content_hash(self, content: str) -> str:
        """Generate hash of content for change detection"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def load_history(self, focus_name: str) -> List[FocusEvolutionEntry]:
        """Load evolution history for a focus"""
        evolution_file = self._get_evolution_file(focus_name)

        if not evolution_file.exists():
            return []

        try:
            data = json.loads(evolution_file.read_text())
            return [FocusEvolutionEntry(**entry) for entry in data]
        except Exception as e:
            logger.warning(f"Could not load evolution history: {e}")
            return []

    def _save_history(self, focus_name: str, history: List[FocusEvolutionEntry]):
        """Save evolution history"""
        evolution_file = self._get_evolution_file(focus_name)
        evolution_file.parent.mkdir(parents=True, exist_ok=True)

        data = [entry.to_dict() for entry in history]
        evolution_file.write_text(json.dumps(data, indent=2))

    def record_creation(self, focus_name: str):
        """Record focus creation event"""
        entry = FocusEvolutionEntry(
            timestamp=time.time(),
            change_type="created",
            section_changed="all",
            old_hash="",
            new_hash=self._get_focus_hash(focus_name),
            summary=f"Focus '{focus_name}' created"
        )

        self._save_history(focus_name, [entry])

    def record_change(self, focus_name: str, section: str, old_content: str,
                      new_content: str, summary: str = ""):
        """Record a change to a focus section"""
        old_hash = self._content_hash(old_content)
        new_hash = self._content_hash(new_content)

        # Don't record if no actual change
        if old_hash == new_hash:
            return

        change_type = self._infer_change_type(section)

        entry = FocusEvolutionEntry(
            timestamp=time.time(),
            change_type=change_type,
            section_changed=section,
            old_hash=old_hash,
            new_hash=new_hash,
            summary=summary or f"Updated {section}"
        )

        history = self.load_history(focus_name)
        history.append(entry)

        # Keep last 100 entries
        if len(history) > 100:
            history = history[-100:]

        self._save_history(focus_name, history)

    def _infer_change_type(self, section: str) -> str:
        """Infer change type from section name"""
        section_lower = section.lower()
        if 'goal' in section_lower:
            return 'goals_updated'
        elif 'context' in section_lower:
            return 'context_updated'
        elif 'wiki' in section_lower:
            return 'wiki_updated'
        elif 'prompt' in section_lower:
            return 'prompt_updated'
        else:
            return 'section_updated'

    def _get_focus_hash(self, focus_name: str) -> str:
        """Get hash of entire focus content"""
        focus_dir = self.senter_root / "Focuses" / focus_name
        senter_file = focus_dir / "SENTER.md"

        if senter_file.exists():
            return self._content_hash(senter_file.read_text())
        return ""

    def get_summary(self, focus_name: str, days: int = 30) -> Dict[str, Any]:
        """Get evolution summary for a focus"""
        history = self.load_history(focus_name)

        cutoff = time.time() - (days * 86400)
        recent = [e for e in history if e.timestamp > cutoff]

        # Count changes by type
        change_counts = Counter(e.change_type for e in recent)

        # Get timeline
        timeline = []
        for entry in recent[-10:]:  # Last 10 changes
            timeline.append({
                "timestamp": datetime.fromtimestamp(entry.timestamp).isoformat(),
                "type": entry.change_type,
                "section": entry.section_changed,
                "summary": entry.summary
            })

        return {
            "focus_name": focus_name,
            "total_changes": len(recent),
            "days_analyzed": days,
            "change_breakdown": dict(change_counts),
            "recent_timeline": timeline,
            "first_change": datetime.fromtimestamp(history[0].timestamp).isoformat() if history else None,
            "last_change": datetime.fromtimestamp(history[-1].timestamp).isoformat() if history else None
        }

    def get_version_at(self, focus_name: str, timestamp: float) -> Optional[str]:
        """Get hash of focus at a specific timestamp"""
        history = self.load_history(focus_name)

        # Find latest entry before timestamp
        relevant = [e for e in history if e.timestamp <= timestamp]
        if relevant:
            return relevant[-1].new_hash
        return None


class FocusFactory:
    """Factory for creating new Focuses dynamically"""

    CONVERSATIONAL_KEYWORDS = [
        "news",
        "research",
        "coding",
        "creative",
        "learning",
        "bitcoin",
        "ai",
        "technology",
        "writing",
        "blog",
        "crypto",
        "finance",
        "science",
        "history",
        "philosophy",
    ]

    FUNCTIONAL_KEYWORDS = [
        "lights",
        "calendar",
        "todo",
        "reminder",
        "email",
        "control",
        "switch",
        "toggle",
        "schedule",
        "alarm",
        "automation",
        "smart",
        "home",
        "device",
    ]

    def __init__(self, senter_root: Path):
        self.senter_root = senter_root
        self.focuses_dir = senter_root / "Focuses"
        self.config_dir = senter_root / "config"
        self.parser = SenterMdParser(senter_root)

    def create_focus(self, focus_name: str, initial_context: str = "",
                     track_evolution: bool = True) -> Path:
        """
        Create new Focus with user's default model

        Args:
            focus_name: Name for new Focus
            initial_context: First user query/prompt
            track_evolution: Whether to record creation in evolution history

        Returns:
            Path to created Focus directory
        """
        # Sanitize focus name
        safe_name = self._sanitize_focus_name(focus_name)

        # Create directory
        focus_dir = self.focuses_dir / safe_name
        focus_dir.mkdir(parents=True, exist_ok=True)

        # Get user's default model
        user_model = self._get_user_default_model()

        # Generate SENTER.md with user's model (NOT hardcoded!)
        senter_md = self._generate_senter_md(safe_name, user_model, initial_context)
        (focus_dir / "SENTER.md").write_text(senter_md, encoding="utf-8")

        # Create wiki.md for conversational Focuses
        if self._is_conversational(safe_name):
            wiki_content = self._generate_wiki_content(safe_name, initial_context)
            (focus_dir / "wiki.md").write_text(wiki_content, encoding="utf-8")
            logger.info(f"Created conversational Focus: {safe_name}")
        else:
            logger.info(f"Created functional Focus: {safe_name}")

        # Track evolution (FA-003)
        if track_evolution:
            tracker = FocusEvolutionTracker(self.senter_root)
            tracker.record_creation(safe_name)

        return focus_dir

    def _sanitize_focus_name(self, name: str) -> str:
        """Sanitize focus name for directory creation"""
        # Replace spaces and special chars with underscores
        safe = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        # Remove leading/trailing underscores
        safe = safe.strip("_")
        return safe or "unnamed_focus"

    def _is_conversational(self, focus_name: str) -> bool:
        """
        Determine if Focus is conversational (has wiki) or functional (no wiki)

        Heuristics based on keywords and context
        """
        focus_lower = focus_name.lower()

        # Check conversational keywords
        if any(kw in focus_lower for kw in self.CONVERSATIONAL_KEYWORDS):
            return True

        # Check functional keywords
        if any(kw in focus_lower for kw in self.FUNCTIONAL_KEYWORDS):
            return False

        # Default to conversational
        return True

    def _get_user_default_model(self) -> Dict[str, Any]:
        """
        Get user's default model config from profile

        Priority:
        1. user_profile.json (explicit user config)
        2. senter_config.json recommended models
        3. Default to empty dict (user must configure later)
        """
        # Check user profile
        user_profile = self.config_dir / "user_profile.json"
        if user_profile.exists():
            with open(user_profile, "r") as f:
                profile = json.load(f)
                if "central_model" in profile:
                    return profile["central_model"]

        # Check global config for recommended models
        senter_config = self.config_dir / "senter_config.json"
        if senter_config.exists():
            with open(senter_config, "r") as f:
                config = json.load(f)
                # Don't hardcode Hermes/Qwen - just use what's available
                if "recommended_models" in config:
                    # Return first available recommended model (if any)
                    rec_models = config["recommended_models"]
                    if rec_models:
                        first_key = list(rec_models.keys())[0]
                        return rec_models[first_key]

        # Return empty - user must configure
        return {}

    def _generate_senter_md(
        self, focus_name: str, model_config: Dict[str, Any], initial_context: str
    ) -> str:
        """
        Generate SENTER.md with user's model config

        Uses mixed YAML + Markdown format
        """
        # Model YAML section
        model_yaml = yaml.dump(
            {"model": model_config}, default_flow_style=False, sort_keys=False
        )

        timestamp = datetime.now().isoformat()

        return f"""---
manifest_version: "1.0"
focus:
  name: "{focus_name}"
  id: "ajson://senter/focuses/{focus_name.lower().replace(" ", "_")}"
  type: "conversational"
  created: "{timestamp}"

{model_yaml}
settings:
  max_tokens: {model_config.get("max_tokens", 512)}
  temperature: {model_config.get("temperature", 0.7)}

system_prompt: |
  You are Senter's agent for the {focus_name} Focus.
  Your purpose is to assist the user with anything related to {focus_name}.
  Use the provided context and wiki to give helpful, accurate responses.
  If you don't have enough context, ask clarifying questions.

ui_config:
  show_wiki: {"true" if self._is_conversational(focus_name) else "false"}
  widgets: []

context:
  type: "wiki"
  content: |
{initial_context}
---

## Detected Goals
*None yet*

## Explorative Follow-Up Questions
*None yet*

## Wiki Content
# {focus_name}

{initial_context}
"""

    def _generate_wiki_content(self, focus_name: str, initial_context: str) -> str:
        """Generate initial wiki.md content"""
        return f"""# {focus_name}

## Overview
{focus_name} is a Focus topic in Senter's knowledge base.

## Initial Context
{initial_context}

## Notes
*This wiki will update as Senter learns more about this topic through conversations and research.*
"""

    def create_internal_focus(self, internal_name: str, internal_type: str) -> Path:
        """
        Create internal Focus for Senter's own operation

        Args:
            internal_name: Name of internal Focus (e.g., Focus_Reviewer)
            internal_type: Type of internal operation (review, plan, code, profile)

        Returns:
            Path to created internal Focus directory
        """
        # Create internal directory
        internal_dir = self.focuses_dir / "internal" / internal_name
        internal_dir.mkdir(parents=True, exist_ok=True)

        # Generate appropriate SENTER.md based on type
        senter_md = self._generate_internal_senter_md(internal_name, internal_type)
        (internal_dir / "SENTER.md").write_text(senter_md, encoding="utf-8")

        print(f"   🔧 Created internal Focus: internal/{internal_name}")
        return internal_dir

    def _generate_internal_senter_md(
        self, internal_name: str, internal_type: str
    ) -> str:
        """Generate SENTER.md for internal Focuses"""
        timestamp = datetime.now().isoformat()

        # System prompts based on type
        system_prompts = {
            "review": f"You are the Focus_Reviewer agent. Your job is to review Focuses and determine if they need updates, merging, or splitting. Be thorough but conservative - only suggest changes when clearly beneficial.",
            "merge": f"You are the Focus_Merger agent. Your job is to combine multiple Focuses that should be merged together based on overlapping content. Preserve important information from both Focuses in the merged version.",
            "split": f"You are the Focus_Splitter agent. Your job is to identify when a Focus has grown too large or diverse, and suggest how to split it into more focused sub-Focuses.",
            "plan": f"You are the Planner_Agent. Your job is to break down user goals into actionable steps. Each step should be specific, achievable, and clearly related to achieving the overall goal.",
            "code": f"You are the Coder_Agent. Your job is to write and fix code for Senter's functions. When you receive an error report, analyze it and produce a fix.",
            "profile": f"You are the User_Profiler agent. Your job is to analyze user interactions using psychology-based approaches to detect: long-term goals, sense of humor, personality traits, communication style. Generate explorative follow-up questions to validate detected goals.",
        }

        system_prompt = system_prompts.get(
            internal_type,
            f"You are the {internal_name} agent for Senter's internal operations.",
        )

        return f"""---
manifest_version: "1.0"
focus:
  name: "{internal_name}"
  id: "ajson://senter/focuses/internal/{internal_name.lower()}"
  type: "internal"
  created: "{timestamp}"

model:
  type: null  # Uses user's default model

settings:
  max_tokens: 512
  temperature: 0.7

system_prompt: |
{system_prompt}

context:
  type: "internal_instructions"
  content: |
    Instructions for this internal agent are provided in the system prompt above.
    This Focus manages internal Senter operations.
---
"""

    def focus_exists(self, focus_name: str) -> bool:
        """Check if a focus already exists"""
        safe_name = self._sanitize_focus_name(focus_name)
        focus_dir = self.focuses_dir / safe_name
        return focus_dir.exists() and (focus_dir / "SENTER.md").exists()


# ========== Dynamic Focus Creator (FA-001 Integration) ==========

@dataclass
class FocusCreationResult:
    """Result of dynamic focus creation"""
    success: bool
    focus_name: str
    focus_path: Optional[Path] = None
    message: str = ""
    notification_sent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['focus_path'] = str(d['focus_path']) if d['focus_path'] else None
        return d


class DynamicFocusCreator:
    """
    Orchestrates dynamic focus creation from conversation (FA-001)

    Integrates:
    - TopicFrequencyTracker: Detects topics discussed 5+ times in 24h
    - FocusFactory: Creates new focuses with user's model
    - FocusEvolutionTracker: Records focus creation
    - Notification: Alerts user about new focus
    """

    def __init__(self, senter_root: Path, message_bus=None):
        self.senter_root = Path(senter_root)
        self.tracker = TopicFrequencyTracker(senter_root)
        self.factory = FocusFactory(senter_root)
        self.evolution_tracker = FocusEvolutionTracker(senter_root)
        self.message_bus = message_bus
        self.parser = SenterMdParser(senter_root)

    def process_query(self, query_text: str, detected_topic: str = None,
                      confidence: float = 0.5) -> Optional[FocusCreationResult]:
        """
        Process a user query and potentially create a new focus

        Called after each user query to track topics and trigger focus creation.

        Args:
            query_text: The user's query text
            detected_topic: Topic extracted from query (if any)
            confidence: Confidence in topic detection

        Returns:
            FocusCreationResult if a new focus was created, None otherwise
        """
        if not detected_topic:
            detected_topic = self._extract_topic(query_text)

        if not detected_topic:
            return None

        # Record the mention
        self.tracker.record_mention(detected_topic, query_text, confidence)

        # Check if we should create a focus
        if self.tracker.should_create_focus(detected_topic):
            # Check if focus already exists
            if self._focus_exists_for_topic(detected_topic):
                return None

            # Create the focus
            return self.create_focus_for_topic(detected_topic)

        return None

    def _extract_topic(self, query_text: str) -> Optional[str]:
        """Extract dominant topic from query text"""
        import re

        # Simple extraction: look for capitalized phrases or domain keywords
        query_lower = query_text.lower()

        # Domain keywords
        domain_keywords = [
            'python', 'javascript', 'typescript', 'rust', 'go', 'java',  # Programming
            'machine learning', 'ai', 'deep learning', 'neural networks',  # AI
            'docker', 'kubernetes', 'aws', 'cloud',  # DevOps
            'react', 'vue', 'angular', 'frontend', 'backend',  # Web
            'database', 'sql', 'mongodb', 'postgres',  # Data
            'crypto', 'bitcoin', 'ethereum', 'blockchain',  # Crypto
            'finance', 'investing', 'trading',  # Finance
        ]

        for keyword in domain_keywords:
            if keyword in query_lower:
                return keyword.title().replace(' ', '_')

        # Look for capitalized phrases
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query_text)
        if caps:
            return caps[0].replace(' ', '_')

        return None

    def _focus_exists_for_topic(self, topic: str) -> bool:
        """Check if a focus already exists for this topic"""
        # Check exact match
        if self.factory.focus_exists(topic):
            return True

        # Check similarity with existing focuses
        existing = self.parser.list_all_focuses()
        topic_lower = topic.lower()

        for focus in existing:
            focus_lower = focus.lower()
            # Check if topic is contained in focus name or vice versa
            if topic_lower in focus_lower or focus_lower in topic_lower:
                return True

        return False

    def create_focus_for_topic(self, topic: str) -> FocusCreationResult:
        """Create a new focus for a frequently discussed topic"""
        try:
            # Get sample queries for initial context
            frequent = self.tracker.get_frequent_topics()
            topic_data = next((t for t in frequent if t.topic == topic.lower()), None)

            initial_context = ""
            if topic_data and topic_data.sample_queries:
                initial_context = "Initial user interests:\n" + "\n".join(
                    f"- {q}" for q in topic_data.sample_queries
                )

            # Create the focus
            focus_path = self.factory.create_focus(topic, initial_context)

            # Clear the topic from tracker (focus created)
            self.tracker.clear_topic(topic)

            # Send notification
            notification_sent = self._notify_user(topic, focus_path)

            logger.info(f"Created dynamic focus: {topic}")

            return FocusCreationResult(
                success=True,
                focus_name=topic,
                focus_path=focus_path,
                message=f"Created new focus '{topic}' based on your frequent discussions",
                notification_sent=notification_sent
            )

        except Exception as e:
            logger.error(f"Failed to create focus for topic {topic}: {e}")
            return FocusCreationResult(
                success=False,
                focus_name=topic,
                message=f"Failed to create focus: {str(e)}"
            )

    def _notify_user(self, topic: str, focus_path: Path) -> bool:
        """Notify user about new focus creation"""
        try:
            if self.message_bus:
                # Send notification via message bus
                from daemon.message_bus import Message, MessageType
                msg = Message(
                    type=MessageType.NOTIFICATION,
                    source="focus_factory",
                    payload={
                        "event": "focus_created",
                        "title": "New Focus Created",
                        "message": f"I noticed you've been discussing {topic} frequently. "
                                   f"I created a new Focus for it to better assist you!",
                        "focus_name": topic,
                        "focus_path": str(focus_path)
                    }
                )
                self.message_bus.publish(msg)
                return True
        except Exception as e:
            logger.debug(f"Could not send notification: {e}")

        # Also log to notification file for CLI to pick up
        try:
            notif_file = self.senter_root / "data" / "notifications" / "pending.json"
            notif_file.parent.mkdir(parents=True, exist_ok=True)

            notifications = []
            if notif_file.exists():
                try:
                    notifications = json.loads(notif_file.read_text())
                except:
                    pass

            notifications.append({
                "timestamp": datetime.now().isoformat(),
                "type": "focus_created",
                "title": "New Focus Created",
                "message": f"Created focus '{topic}' based on your frequent discussions",
                "data": {"focus_name": topic}
            })

            notif_file.write_text(json.dumps(notifications, indent=2))
            return True

        except Exception as e:
            logger.debug(f"Could not save notification: {e}")

        return False

    def get_pending_notifications(self) -> List[Dict]:
        """Get pending notifications for user"""
        notif_file = self.senter_root / "data" / "notifications" / "pending.json"

        if not notif_file.exists():
            return []

        try:
            return json.loads(notif_file.read_text())
        except:
            return []

    def clear_notifications(self):
        """Clear pending notifications"""
        notif_file = self.senter_root / "data" / "notifications" / "pending.json"
        if notif_file.exists():
            notif_file.unlink()

    def get_topic_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked topics"""
        frequent = self.tracker.get_frequent_topics(min_mentions=1)

        return {
            "tracked_topics": len(frequent),
            "topics_near_threshold": len([t for t in frequent
                                         if t.mention_count >= TopicFrequencyTracker.MIN_MENTIONS - 2]),
            "topics_ready_for_focus": len([t for t in frequent
                                          if t.mention_count >= TopicFrequencyTracker.MIN_MENTIONS]),
            "top_topics": [{"topic": t.topic, "mentions": t.mention_count}
                          for t in frequent[:5]]
        }

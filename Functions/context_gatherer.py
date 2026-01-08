#!/usr/bin/env python3
"""
Context Gatherer for Senter (IA-002)

Continuously updates focus SENTER.md sections with incremental context.
Uses append/modify strategy instead of full replacement.
"""

import json
import logging
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib

logger = logging.getLogger("senter.context_gatherer")


class UpdateMode(Enum):
    """How to update a section"""
    APPEND = "append"           # Add to end of section
    PREPEND = "prepend"         # Add to beginning of section
    MERGE = "merge"             # Intelligently merge with existing
    REPLACE_SECTION = "replace" # Replace entire section


@dataclass
class ContextUpdate:
    """A single context update to apply"""
    section_name: str
    content: str
    update_mode: UpdateMode = UpdateMode.APPEND
    source: str = "conversation"  # conversation, research, user_input
    timestamp: float = 0.0
    confidence: float = 0.8

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["update_mode"] = self.update_mode.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ContextUpdate":
        data = data.copy()
        data["update_mode"] = UpdateMode(data["update_mode"])
        return cls(**data)


@dataclass
class UpdateResult:
    """Result of applying a context update"""
    success: bool
    focus_name: str
    section_name: str
    old_size: int = 0
    new_size: int = 0
    change_type: str = ""  # appended, prepended, merged, replaced
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContextGatherer:
    """
    Continuously gathers and updates focus context (IA-002)

    Features:
    - Incremental updates (append/modify, not replace)
    - Multiple update modes: append, prepend, merge, replace
    - Change tracking for evolution history
    - Deduplication of similar content
    """

    # Standard sections in SENTER.md
    STANDARD_SECTIONS = [
        "Detected Goals",
        "Explorative Follow-Up Questions",
        "Wiki Content",
        "Recent Insights",
        "Key Entities",
        "Related Topics"
    ]

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.focuses_dir = self.senter_root / "Focuses"
        self.update_log = self.senter_root / "data" / "context_updates.json"
        self.update_log.parent.mkdir(parents=True, exist_ok=True)

        # Import evolution tracker if available
        try:
            import sys
            sys.path.insert(0, str(self.senter_root / "Focuses"))
            from focus_factory import FocusEvolutionTracker
            self.evolution_tracker = FocusEvolutionTracker(senter_root)
        except ImportError:
            self.evolution_tracker = None

    def update_focus_context(self, focus_name: str, update: ContextUpdate) -> UpdateResult:
        """
        Apply a context update to a focus

        Args:
            focus_name: Name of the focus to update
            update: The ContextUpdate to apply

        Returns:
            UpdateResult with status and details
        """
        focus_dir = self.focuses_dir / focus_name
        senter_file = focus_dir / "SENTER.md"

        if not senter_file.exists():
            return UpdateResult(
                success=False,
                focus_name=focus_name,
                section_name=update.section_name,
                error=f"Focus '{focus_name}' not found"
            )

        try:
            content = senter_file.read_text(encoding="utf-8")
            old_size = len(content)

            # Parse and find section
            old_section = self._extract_section(content, update.section_name)

            # Apply update based on mode
            if update.update_mode == UpdateMode.APPEND:
                new_content = self._append_to_section(content, update)
                change_type = "appended"
            elif update.update_mode == UpdateMode.PREPEND:
                new_content = self._prepend_to_section(content, update)
                change_type = "prepended"
            elif update.update_mode == UpdateMode.MERGE:
                new_content = self._merge_section(content, update)
                change_type = "merged"
            else:  # REPLACE_SECTION
                new_content = self._replace_section(content, update)
                change_type = "replaced"

            # Write updated content
            senter_file.write_text(new_content, encoding="utf-8")

            # Track evolution if available
            if self.evolution_tracker:
                new_section = self._extract_section(new_content, update.section_name)
                self.evolution_tracker.record_change(
                    focus_name,
                    update.section_name,
                    old_section,
                    new_section,
                    f"{change_type} content from {update.source}"
                )

            # Log the update
            self._log_update(focus_name, update, change_type)

            return UpdateResult(
                success=True,
                focus_name=focus_name,
                section_name=update.section_name,
                old_size=old_size,
                new_size=len(new_content),
                change_type=change_type
            )

        except Exception as e:
            logger.error(f"Failed to update {focus_name}: {e}")
            return UpdateResult(
                success=False,
                focus_name=focus_name,
                section_name=update.section_name,
                error=str(e)
            )

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a section from SENTER.md content"""
        # Pattern for markdown section
        pattern = rf"##\s*{re.escape(section_name)}\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _append_to_section(self, content: str, update: ContextUpdate) -> str:
        """Append content to a section"""
        section_pattern = rf"(##\s*{re.escape(update.section_name)}\s*\n)(.*?)(\n##|\Z)"

        def replacer(match):
            header = match.group(1)
            existing = match.group(2).rstrip()
            ending = match.group(3)

            # Don't add duplicate content
            if self._is_duplicate(existing, update.content):
                return match.group(0)

            timestamp = datetime.fromtimestamp(update.timestamp).strftime("%Y-%m-%d")
            new_entry = f"\n\n[{timestamp}] {update.content}"

            return f"{header}{existing}{new_entry}\n{ending}"

        if re.search(section_pattern, content, re.DOTALL | re.IGNORECASE):
            return re.sub(section_pattern, replacer, content, flags=re.DOTALL | re.IGNORECASE)
        else:
            # Section doesn't exist, create it
            return self._add_new_section(content, update)

    def _prepend_to_section(self, content: str, update: ContextUpdate) -> str:
        """Prepend content to a section"""
        section_pattern = rf"(##\s*{re.escape(update.section_name)}\s*\n)(.*?)(\n##|\Z)"

        def replacer(match):
            header = match.group(1)
            existing = match.group(2).strip()
            ending = match.group(3)

            if self._is_duplicate(existing, update.content):
                return match.group(0)

            timestamp = datetime.fromtimestamp(update.timestamp).strftime("%Y-%m-%d")
            new_entry = f"[{timestamp}] {update.content}\n\n"

            return f"{header}{new_entry}{existing}\n{ending}"

        if re.search(section_pattern, content, re.DOTALL | re.IGNORECASE):
            return re.sub(section_pattern, replacer, content, flags=re.DOTALL | re.IGNORECASE)
        else:
            return self._add_new_section(content, update)

    def _merge_section(self, content: str, update: ContextUpdate) -> str:
        """Intelligently merge content into section (deduplicates)"""
        section_pattern = rf"(##\s*{re.escape(update.section_name)}\s*\n)(.*?)(\n##|\Z)"
        match = re.search(section_pattern, content, re.DOTALL | re.IGNORECASE)

        if not match:
            return self._add_new_section(content, update)

        existing = match.group(2).strip()

        # If new content is similar to existing, skip
        if self._is_duplicate(existing, update.content):
            return content

        # Merge by appending unique content
        merged = existing + "\n\n" + update.content

        return content[:match.start(2)] + merged + "\n" + content[match.end(2):]

    def _replace_section(self, content: str, update: ContextUpdate) -> str:
        """Replace entire section content"""
        section_pattern = rf"(##\s*{re.escape(update.section_name)}\s*\n)(.*?)(\n##|\Z)"

        def replacer(match):
            header = match.group(1)
            ending = match.group(3)
            return f"{header}{update.content}\n{ending}"

        if re.search(section_pattern, content, re.DOTALL | re.IGNORECASE):
            return re.sub(section_pattern, replacer, content, flags=re.DOTALL | re.IGNORECASE)
        else:
            return self._add_new_section(content, update)

    def _add_new_section(self, content: str, update: ContextUpdate) -> str:
        """Add a new section to the end of the markdown content"""
        # Find the end of YAML frontmatter
        yaml_end = content.find("---\n", 4)  # Find second ---
        if yaml_end == -1:
            yaml_end = len(content)
        else:
            yaml_end += 4

        new_section = f"\n## {update.section_name}\n{update.content}\n"

        return content[:yaml_end] + new_section + content[yaml_end:]

    def _is_duplicate(self, existing: str, new_content: str) -> bool:
        """Check if new content is duplicate of existing"""
        # Normalize for comparison
        existing_norm = existing.lower().strip()
        new_norm = new_content.lower().strip()

        # Exact match
        if new_norm in existing_norm:
            return True

        # Hash-based similarity (for longer content)
        if len(new_norm) > 50:
            existing_hash = hashlib.md5(existing_norm.encode()).hexdigest()[:8]
            new_hash = hashlib.md5(new_norm.encode()).hexdigest()[:8]
            # Very rough check - same first 8 chars of hash
            if existing_hash == new_hash:
                return True

        return False

    def _log_update(self, focus_name: str, update: ContextUpdate, change_type: str):
        """Log the update for audit trail"""
        log = []
        if self.update_log.exists():
            try:
                log = json.loads(self.update_log.read_text())
            except:
                pass

        log.append({
            "timestamp": datetime.now().isoformat(),
            "focus_name": focus_name,
            "section": update.section_name,
            "change_type": change_type,
            "source": update.source,
            "content_preview": update.content[:100]
        })

        # Keep last 200 entries
        log = log[-200:]
        self.update_log.write_text(json.dumps(log, indent=2))

    # ========== Batch Operations ==========

    def batch_update(self, focus_name: str, updates: List[ContextUpdate]) -> List[UpdateResult]:
        """Apply multiple updates to a focus"""
        results = []
        for update in updates:
            result = self.update_focus_context(focus_name, update)
            results.append(result)
        return results

    def gather_from_conversation(self, focus_name: str, messages: List[Dict],
                                 extract_goals: bool = True,
                                 extract_entities: bool = True) -> List[UpdateResult]:
        """
        Gather context from a conversation and update focus

        Args:
            focus_name: Focus to update
            messages: List of conversation messages
            extract_goals: Whether to extract and add detected goals
            extract_entities: Whether to extract key entities

        Returns:
            List of UpdateResults
        """
        updates = []
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        all_text = " ".join(user_messages)

        # Extract goals if requested
        if extract_goals:
            goals = self._extract_goals_from_text(all_text)
            if goals:
                updates.append(ContextUpdate(
                    section_name="Detected Goals",
                    content="\n".join(f"- {g}" for g in goals),
                    update_mode=UpdateMode.APPEND,
                    source="conversation"
                ))

        # Extract key entities if requested
        if extract_entities:
            entities = self._extract_key_entities(all_text)
            if entities:
                updates.append(ContextUpdate(
                    section_name="Key Entities",
                    content=", ".join(entities),
                    update_mode=UpdateMode.MERGE,
                    source="conversation"
                ))

        # Extract follow-up questions
        questions = self._extract_follow_up_questions(messages)
        if questions:
            updates.append(ContextUpdate(
                section_name="Explorative Follow-Up Questions",
                content="\n".join(f"- {q}" for q in questions),
                update_mode=UpdateMode.APPEND,
                source="conversation"
            ))

        return self.batch_update(focus_name, updates) if updates else []

    def _extract_goals_from_text(self, text: str) -> List[str]:
        """Extract goal-like statements from text"""
        goals = []
        patterns = [
            r"i (?:need|want|have) to ([\w\s]+?)(?:\.|,|$)",
            r"i'm (?:going to|planning to) ([\w\s]+?)(?:\.|,|$)",
            r"my goal is to ([\w\s]+?)(?:\.|,|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                goal = match.strip().capitalize()
                if len(goal) > 10 and goal not in goals:
                    goals.append(goal)

        return goals[:5]

    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key named entities from text"""
        entities = []

        # Capitalized phrases (simple NER)
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        entities.extend(caps[:10])

        # Domain-specific keywords
        keywords = [
            'python', 'javascript', 'react', 'ai', 'machine learning',
            'database', 'api', 'docker', 'kubernetes'
        ]
        for kw in keywords:
            if kw.lower() in text.lower():
                entities.append(kw.title())

        return list(set(entities))[:10]

    def _extract_follow_up_questions(self, messages: List[Dict]) -> List[str]:
        """Extract potential follow-up questions from assistant responses"""
        questions = []

        for msg in messages:
            if msg.get("role") == "assistant":
                # Find questions in assistant responses
                found = re.findall(r'([^.!]*\?)', msg["content"])
                questions.extend(q.strip() for q in found if len(q) > 10)

        return questions[:5]

    # ========== Wiki Management ==========

    def update_wiki(self, focus_name: str, new_content: str,
                    mode: UpdateMode = UpdateMode.APPEND) -> bool:
        """
        Update wiki.md for a focus

        Args:
            focus_name: Focus name
            new_content: Content to add/update
            mode: How to update (APPEND, PREPEND, or REPLACE)

        Returns:
            True if successful
        """
        wiki_file = self.focuses_dir / focus_name / "wiki.md"

        if not wiki_file.parent.exists():
            return False

        try:
            existing = wiki_file.read_text() if wiki_file.exists() else ""

            if self._is_duplicate(existing, new_content):
                return True  # No change needed

            if mode == UpdateMode.APPEND:
                updated = existing + "\n\n" + new_content
            elif mode == UpdateMode.PREPEND:
                updated = new_content + "\n\n" + existing
            else:
                updated = new_content

            wiki_file.write_text(updated)

            # Track evolution
            if self.evolution_tracker:
                self.evolution_tracker.record_change(
                    focus_name,
                    "wiki",
                    existing,
                    updated,
                    f"Wiki {mode.value}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to update wiki for {focus_name}: {e}")
            return False

    def get_update_history(self, focus_name: str = None, limit: int = 50) -> List[Dict]:
        """Get update history, optionally filtered by focus"""
        if not self.update_log.exists():
            return []

        try:
            log = json.loads(self.update_log.read_text())

            if focus_name:
                log = [e for e in log if e.get("focus_name") == focus_name]

            return log[-limit:]

        except Exception:
            return []

#!/usr/bin/env python3
"""
Focus Factory - Dynamic Focus Creation
Creates new Focus directories and SENTER.md files with user's model
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Focuses"))
from senter_md_parser import SenterMdParser


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

    def create_focus(self, focus_name: str, initial_context: str = "") -> Path:
        """
        Create new Focus with user's default model

        Args:
            focus_name: Name for new Focus
            initial_context: First user query/prompt

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
            print(f"   📚 Created conversational Focus: {safe_name}")
        else:
            print(f"   🔧 Created functional Focus: {safe_name}")

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
            "review": "You are the Focus_Reviewer agent. Your job is to review Focuses and determine if they need updates, merging, or splitting. Be thorough but conservative - only suggest changes when clearly beneficial.",
            "merge": "You are the Focus_Merger agent. Your job is to combine multiple Focuses that should be merged together based on overlapping content. Preserve important information from both Focuses in the merged version.",
            "split": "You are the Focus_Splitter agent. Your job is to identify when a Focus has grown too large or diverse, and suggest how to split it into more focused sub-Focuses.",
            "plan": "You are the Planner_Agent. Your job is to break down user goals into actionable steps. Each step should be specific, achievable, and clearly related to achieving the overall goal.",
            "code": "You are the Coder_Agent. Your job is to write and fix code for Senter's functions. When you receive an error report, analyze it and produce a fix.",
            "profile": "You are the User_Profiler agent. Your job is to analyze user interactions using psychology-based approaches to detect: long-term goals, sense of humor, personality traits, communication style. Generate explorative follow-up questions to validate detected goals.",
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

#!/usr/bin/env python3
"""
SENTER.md Parser - Parse YAML frontmatter + Markdown sections
Model-agnostic configuration loader for Focus system
"""

import os
import sys
import re
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List


# Use os.path.join for cross-platform path handling
def join_path(*parts):
    """Join path components safely"""
    return os.path.join(*parts)


class SenterMdParser:
    """Parse SENTER.md files with YAML frontmatter + Markdown sections"""

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root) if not isinstance(senter_root, Path) else senter_root
        self.focuses_dir = self.senter_root / "Focuses"
        self.config_dir = self.senter_root / "config"
        self.cache = {}  # Simple cache for parsed configs

    def load_focus_config(self, focus_name: str) -> Dict[str, Any]:
        """
        Load complete Focus configuration from SENTER.md

        Priority chain:
        1. Focus SENTER.md explicit model config
        2. user_profile.json central_model
        3. senter_config.json default_models

        Args:
            focus_name: Name of Focus (directory name)

        Returns:
            Complete configuration dict
        """
        if focus_name in self.cache:
            return self.cache[focus_name]

        focus_dir = self.focuses_dir / focus_name
        senter_file = focus_dir / "SENTER.md"

        if not senter_file.exists():
            return {}

        with open(senter_file, "r", encoding="utf-8") as f:
            content = f.read()

        parsed = self._parse_yaml_frontmatter(content)
        parsed["focus_name"] = focus_name

        # Resolve model config with priority chain
        if "model" not in parsed or not parsed["model"]:
            parsed["model"] = self._resolve_model_config()

        # Resolve model type
        parsed["model"] = self._expand_model_config(parsed["model"])

        self.cache[focus_name] = parsed
        return parsed

    def _parse_yaml_frontmatter(self, content: str) -> Dict[str, Any]:
        """
        Parse YAML frontmatter from SENTER.md

        Format:
        ---
        # YAML content
        ---

        # Markdown sections
        ...
        """
        split_match = re.search(
            r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL | re.MULTILINE
        )

        if not split_match:
            return {"raw_markdown": content}

        yaml_content = split_match.group(1)
        markdown_content = split_match.group(2)

        try:
            yaml_data = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError as e:
            print(f"   ⚠️  YAML parsing error: {e}")
            yaml_data = {}

        yaml_data["raw_markdown"] = markdown_content

        return yaml_data

    def _resolve_model_config(self) -> Dict[str, Any]:
        """
        Resolve model config with priority chain
        """
        # 1. Check user_profile.json
        user_profile = self.config_dir / "user_profile.json"
        if user_profile.exists():
            with open(user_profile, "r") as f:
                profile = json.load(f)
                if "central_model" in profile:
                    return profile["central_model"]

        # 2. Check senter_config.json
        senter_config = self.config_dir / "senter_config.json"
        if senter_config.exists():
            with open(senter_config, "r") as f:
                config = json.load(f)
                # Check if there's a default_models section
                if "default_models" in config:
                    return config["default_models"].get("central_model", {})

        # 3. Return empty dict (user must configure)
        return {}

    def _expand_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand model config based on type

        Adds appropriate defaults for GGUF, OpenAI, or vLLM
        """
        if not model_config or not isinstance(model_config, dict):
            # No model config specified
            return {"type": "gguf", "is_vlm": False}

        model_type = model_config.get("type", "gguf")

        if model_type == "gguf":
            return {
                "type": "gguf",
                "path": model_config.get("path"),
                "n_gpu_layers": model_config.get("gguf_settings", {}).get(
                    "n_gpu_layers", -1
                ),
                "n_ctx": model_config.get("context_window", 8192),
                "verbose": model_config.get("gguf_settings", {}).get("verbose", False),
                "is_vlm": model_config.get("is_vlm", False),
                "max_tokens": model_config.get("max_tokens", 512),
                "temperature": model_config.get("temperature", 0.7),
            }
        elif model_type == "openai":
            return {
                "type": "openai",
                "endpoint": model_config.get("endpoint"),
                "model_name": model_config.get("model_name"),
                "api_key": model_config.get("api_key"),
                "is_vlm": model_config.get("is_vlm", False),
                "max_tokens": model_config.get("max_tokens", 512),
                "temperature": model_config.get("temperature", 0.7),
            }
        elif model_type == "vllm":
            return {
                "type": "vllm",
                "vllm_endpoint": model_config.get("vllm_endpoint"),
                "is_vlm": model_config.get("is_vlm", False),
                "max_tokens": model_config.get("max_tokens", 512),
                "temperature": model_config.get("temperature", 0.7),
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_system_prompt(self, focus_name: str) -> str:
        """Get system prompt from Focus SENTER.md"""
        config = self.load_focus_config(focus_name)
        return config.get("system_prompt", "")

    def get_focus_type(self, focus_name: str) -> str:
        """Get Focus type (conversational or functional)"""
        config = self.load_focus_config(focus_name)
        return config.get("focus", {}).get("type", "conversational")

    def has_wiki(self, focus_name: str) -> bool:
        """Check if Focus should display wiki"""
        config = self.load_focus_config(focus_name)
        ui_config = config.get("ui_config", {})
        return ui_config.get("show_wiki", False)

    def get_wiki_content(self, focus_name: str) -> str:
        """Get wiki.md content for Focus"""
        focus_dir = self.focuses_dir / focus_name
        wiki_file = focus_dir / "wiki.md"

        if wiki_file.exists():
            return wiki_file.read_text(encoding="utf-8")

        return ""

    def update_markdown_section(
        self, focus_name: str, section: str, content: str
    ) -> bool:
        """
        Update a markdown section in SENTER.md

        Args:
            focus_name: Focus directory name
            section: Section header (e.g., "## Detected Goals")
            content: New content for section

        Returns:
            True if successful
        """
        focus_dir = self.focuses_dir / focus_name
        senter_file = focus_dir / "SENTER.md"

        if not senter_file.exists():
            return False

        with open(senter_file, "r", encoding="utf-8") as f:
            full_content = f.read()

        parsed = self._parse_yaml_frontmatter(full_content)
        markdown = parsed["raw_markdown"]

        # Update or append section
        section_pattern = rf"^## {re.escape(section)}\s*.*?$"

        if re.search(section_pattern, markdown, re.MULTILINE):
            # Replace existing section
            updated = re.sub(
                section_pattern,
                f"## {section}\n\n{content}",
                markdown,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            # Append new section
            updated = markdown + f"\n\n## {section}\n\n{content}\n"

        # Rebuild SENTER.md
        split_match = re.search(r"^(---.*?---)\n", full_content, re.DOTALL)

        if split_match:
            new_content = split_match.group(1) + updated
            with open(senter_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True

        return False

    def update_wiki(self, focus_name: str, new_content: str) -> bool:
        """Update wiki.md for a Focus"""
        focus_dir = self.focuses_dir / focus_name
        wiki_file = focus_dir / "wiki.md"

        if not wiki_file.exists():
            # Create wiki.md
            focus_dir.mkdir(exist_ok=True)
            wiki_file.write_text(new_content, encoding="utf-8")
            return True

        # Append to existing wiki
        existing = wiki_file.read_text(encoding="utf-8")
        wiki_file.write_text(existing + "\n\n" + new_content, encoding="utf-8")
        return True

    def list_all_focuses(self) -> List[str]:
        """List all available Focus directories"""
        focuses = []
        focuses_path = Path(self.focuses_dir)
        if not focuses_path.exists():
            return focuses
        for item in focuses_path.iterdir():
            if item.is_dir() and (item / "SENTER.md").exists():
                focuses.append(item.name)
        return focuses

    def get_focus_context(self, focus_name: str) -> str:
        """Get context section from SENTER.md"""
        config = self.load_focus_config(focus_name)
        context_config = config.get("context", {})
        context_type = context_config.get("type", "internal_instructions")
        return context_config.get("content", "")

    def get_focus_functions(self, focus_name: str) -> List[Dict[str, Any]]:
        """Get functions defined in Focus SENTER.md"""
        config = self.load_focus_config(focus_name)
        return config.get("functions", [])

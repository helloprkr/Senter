#!/usr/bin/env python3
"""
Agent Registry - DEPRECATED

This module has been superseded by the SENTER.md Focus system.
Agent configuration now lives in Focuses/*/SENTER.md files.

Kept for backwards compatibility but not actively used.
"""

# This file is intentionally minimal as the functionality
# has been moved to the Focus system (senter_md_parser.py)


class AgentRegistry:
    """Deprecated agent registry - use SenterMdParser instead"""

    def __init__(
        self,
        agents_dir: str = "../../Agents",
        functions_dir: str = "../../Functions",
    ):
        self.agents_dir = agents_dir
        self.functions_dir = functions_dir
        print("Warning: AgentRegistry is deprecated. Use SenterMdParser for Focus discovery.")

    def list_agents(self):
        """Deprecated - use SenterMdParser.list_all_focuses()"""
        return []

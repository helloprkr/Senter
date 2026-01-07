#!/usr/bin/env python3
"""
Senter Selection Utilities
Modular functions for embed filtering + LLM selection
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from qwen25_omni_agent import QwenOmniAgent

class SenterSelector:
    """Modular embed + LLM selection system"""

    def __init__(self):
        self.omni_agent = None
        self.embed_model = None
        self._init_components()

    def _init_components(self):
        """Initialize required components"""
        if self.omni_agent is None:
            self.omni_agent = QwenOmniAgent()

        # TODO: Initialize nomic embed model
        # self.embed_model = load_nomic_embed_model()

    def select_from_options(self,
                          query: str,
                          options: List[str],
                          max_final_options: int = 4,
                          allow_new: bool = False,
                          context: str = "") -> Tuple[str, str]:
        """
        Modular selection function: embed filtering + LLM final selection

        Args:
            query: User query or item to match
            options: List of available options
            max_final_options: Maximum options to present to LLM (default 4)
            allow_new: Whether to allow creating new option if none fit
            context: Additional context for selection

        Returns:
            Tuple of (selected_option, reasoning)
        """
        if len(options) <= max_final_options:
            # Direct LLM selection
            return self._llm_select(query, options, allow_new, context)
        else:
            # Embed filtering to max_final_options, then LLM selection
            filtered_options = self._embed_filter(query, options, max_final_options)
            return self._llm_select(query, filtered_options, allow_new, context)

    def _embed_filter(self, query: str, options: List[str], max_options: int) -> List[str]:
        """
        Use nomic embed to filter options down to top N most similar
        """
        # TODO: Implement actual nomic embed similarity
        # For now, return first max_options
        return options[:max_options]

    def _llm_select(self, query: str, options: List[str], allow_new: bool, context: str) -> Tuple[str, str]:
        """
        Use LLM to make final selection from filtered options
        """
        new_option_text = "\n- **CREATE NEW**: Create a new option if none of the above fit well" if allow_new else ""

        selection_prompt = f"""
        Analyze this query and select the best matching option from the provided list.

        Query: "{query}"
        {"Context: " + context if context else ""}

        Available Options:
        {chr(10).join(f"- **{opt}**: {opt}" for opt in options)}{new_option_text}

        Selection Criteria:
        1. Semantic similarity to the query
        2. Contextual relevance
        3. Specificity of match
        4. Practical utility

        Respond with JSON:
        {{
            "selected_option": "exact_option_name_or_CREATE_NEW",
            "reasoning": "brief explanation of why this option was selected",
            "confidence": "high/medium/low"
        }}
        """

        response = self.omni_agent.generate_response([{
            "role": "user",
            "content": [{"type": "text", "text": selection_prompt}]
        }])

        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                selection_data = json.loads(json_match.group())
                selected = selection_data.get("selected_option", options[0])
                reasoning = selection_data.get("reasoning", "LLM selection")

                # Handle CREATE NEW case
                if selected == "CREATE_NEW" and allow_new:
                    return "CREATE_NEW", reasoning

                return selected, reasoning
        except Exception as e:
            print(f"Selection parsing error: {e}")

        # Fallback to first option
        return options[0], "Fallback selection"

def select_topic_and_agent(query: str,
                          available_topics: List[str],
                          available_agents: List[str],
                          context: str = "") -> Tuple[str, str, str]:
    """
    Specialized function for topic and agent selection

    Returns: (topic, agent, reasoning)
    """
    selector = SenterSelector()

    # First select topic
    topic, topic_reasoning = selector.select_from_options(
        query=f"Determine the most relevant topic for: {query}",
        options=available_topics,
        allow_new=True,
        context=context
    )

    # Then select agent
    agent, agent_reasoning = selector.select_from_options(
        query=f"Select the best agent for this {topic} topic query: {query}",
        options=available_agents,
        allow_new=False,
        context=f"Topic: {topic}"
    )

    combined_reasoning = f"Topic: {topic_reasoning} | Agent: {agent_reasoning}"

    return topic, agent, combined_reasoning

def select_relevant_content(query: str,
                           content_options: List[Dict[str, Any]],
                           max_options: int = 4) -> List[Dict[str, Any]]:
    """
    Select most relevant content items using embed + LLM approach

    Args:
        query: Search query
        content_options: List of content dicts with 'title', 'content', 'metadata'
        max_options: Maximum items to return

    Returns:
        Filtered list of most relevant content
    """
    selector = SenterSelector()

    # Create option strings from content
    option_strings = []
    content_map = {}

    for i, content in enumerate(content_options):
        option_str = f"{content.get('title', f'Content {i}')} - {content.get('content', '')[:200]}..."
        option_strings.append(option_str)
        content_map[option_str] = content

    # Select top options
    selected_option, _ = selector.select_from_options(
        query=f"Find content most relevant to: {query}",
        options=option_strings,
        max_final_options=max_options,
        allow_new=False
    )

    # Return corresponding content items
    if selected_option in content_map:
        return [content_map[selected_option]]
    else:
        # Return top N by default
        return content_options[:max_options]

# Convenience functions for common use cases
def select_agent_for_task(task_description: str, available_agents: List[str]) -> Tuple[str, str]:
    """Select best agent for a task"""
    selector = SenterSelector()
    return selector.select_from_options(
        query=f"Select the best agent for this task: {task_description}",
        options=available_agents,
        allow_new=False
    )

def categorize_content(content: str, categories: List[str]) -> Tuple[str, str]:
    """Categorize content into predefined categories"""
    selector = SenterSelector()
    return selector.select_from_options(
        query=f"Categorize this content: {content[:500]}...",
        options=categories,
        allow_new=True  # Allow new category if needed
    )
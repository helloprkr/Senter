#!/usr/bin/env python3
"""
Focus Review Chain - Background Focus Review System
Uses omniagent instances to review and update all Focuses
"""

import sys
import json
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from senter_md_parser import SenterMdParser
from omniagent import SenterOmniAgent


class FocusReviewChain:
    """
    Chain of omniagent instances for reviewing Focuses

    Each review is a separate omniagent with its own SENTER.md
    """

    def __init__(self, senter_root: Path):
        self.senter_root = senter_root
        self.parser = SenterMdParser(senter_root)

        # Load internal review agents
        self.focus_reviewer = self._load_internal_agent("Focus_Reviewer")
        self.focus_merger = self._load_internal_agent("Focus_Merger")
        self.focus_splitter = self._load_internal_agent("Focus_Splitter")

        self.review_queue = queue.Queue()
        self.running = False
        self.worker_thread = None

    def _load_internal_agent(self, agent_name: str) -> Optional[SenterOmniAgent]:
        """
        Load internal agent Focus as omniagent instance
        """
        try:
            internal_dir = self.senter_root / "Focuses" / "internal" / agent_name
            if not internal_dir.exists():
                return None

            # Load internal agent config
            config = self.parser.load_focus_config(f"internal/{agent_name}")

            # Internal agents use same omniagent infrastructure
            # with their own system prompts from SENTER.md
            omni = SenterOmniAgent(
                model_config=config.get("model", {}),
                omni_config=self._get_infrastructure_config(),
                embed_config=self._get_embed_config()
            )

            return omni
        except Exception as e:
            print(f"   ⚠️  Failed to load internal agent {agent_name}: {e}")
            return None

    def _get_infrastructure_config(self) -> dict:
        """Get infrastructure models config (Omni 3B + embedding)"""
        config_file = self.senter_root / "config" / "senter_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("infrastructure_models", {})
        return {}

    def _get_embed_config(self) -> dict:
        """Get embedding model config"""
        config_file = self.senter_root / "config" / "senter_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("infrastructure_models", {}).get("embedding_model", {})
        return {}

    def start_review_worker(self):
        """Start background review worker thread"""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._review_loop,
            daemon=True,
            name="focus-review-worker"
        )
        self.worker_thread.start()
        print("   ✅ Focus Review Chain started")

    def stop_review_worker(self):
        """Stop background review worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        print("   🛑 Focus Review Chain stopped")

    def _review_loop(self):
        """Main review loop"""
        while self.running:
            try:
                # Check for queued reviews
                if not self.review_queue.empty():
                    task = self.review_queue.get(timeout=1)
                    self._process_review_task(task)
                else:
                    # Idle - periodic check
                    pass
            except queue.Empty:
                pass
            except Exception as e:
                print(f"   ⚠️  Review error: {e}")

    def queue_review(self, focus_name: str, new_data: Dict[str, Any] = None):
        """
        Queue a Focus for review

        Args:
            focus_name: Focus to review
            new_data: New data (web results, chat, function outputs)
        """
        self.review_queue.put({
            "focus_name": focus_name,
            "new_data": new_data or {},
            "timestamp": datetime.now().isoformat()
        })

    def _process_review_task(self, task: Dict[str, Any]):
        """
        Process a review task

        Uses Focus_Reviewer omniagent to determine action needed
        """
        focus_name = task["focus_name"]
        new_data = task["new_data"]

        if not self.focus_reviewer:
            print(f"   ⚠️  Focus_Reviewer not available, skipping review of {focus_name}")
            return

        # Build review prompt
        review_prompt = f"""
        Review Focus: {focus_name}

        New Data Received:
        - Web results: {new_data.get('web', 'N/A')}
        - Chat interactions: {new_data.get('chat', 'N/A')}
        - Function outputs: {new_data.get('function', 'N/A')}

        Analyze this Focus and determine what action is needed:
        1. **UPDATE**: Add new information to Focus context
        2. **MERGE**: Combine with another Focus that has significant overlap
        3. **SPLIT**: Focus has grown too diverse, split into sub-Focuses
        4. **NONE**: No action needed

        Respond with JSON format:
        {{
          "action": "update|merge|split|none",
          "reasoning": "brief explanation of why this action is needed",
          "target_focus": "target Focus name if merge/split",
          "confidence": 0.0-1.0
        }}
        """

        # Use Focus_Reviewer omniagent
        try:
            response = self.focus_reviewer.process_text(review_prompt, max_tokens=512)

            # Parse response
            action_data = self._parse_review_response(response)

            if action_data:
                self._execute_review_action(action_data)

                # Log review
                self._log_review(focus_name, action_data, new_data)

        except Exception as e:
            print(f"   ⚠️  Review processing failed for {focus_name}: {e}")

    def _parse_review_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse review response JSON"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"   ⚠️  Failed to parse review response: {e}")
            return None

    def _execute_review_action(self, action_data: Dict[str, Any]):
        """Execute the review action (update, merge, split)"""
        action = action_data.get("action", "none").lower()
        focus_name = action_data.get("focus_name", "")

        if action == "update":
            self._update_focus(focus_name, action_data.get("content", ""))

        elif action == "merge":
            target_focus = action_data.get("target_focus", "")
            if target_focus:
                self._merge_focuses(focus_name, target_focus)

        elif action == "split":
            self._split_focus(focus_name, action_data.get("sub_focuses", []))

    def _update_focus(self, focus_name: str, content: str):
        """Update Focus SENTER.md with new content"""
        if not content:
            return

        # Update markdown section
        success = self.parser.update_markdown_section(
            focus_name,
            "Detected Goals",
            content
        )

        if success:
            print(f"   ✅ Updated Focus: {focus_name}")

    def _merge_focuses(self, focus1: str, focus2: str):
        """Merge two Focuses using Focus_Merger agent"""
        if not self.focus_merger:
            print("   ⚠️  Focus_Merger not available, skipping merge")
            return

        # Load both Focus configs
        config1 = self.parser.load_focus_config(focus1)
        config2 = self.parser.load_focus_config(focus2)

        # Build merge prompt
        merge_prompt = f"""
        Merge these two Focuses:

        Focus 1: {focus1}
        System: {config1.get('system_prompt', 'N/A')}

        Focus 2: {focus2}
        System: {config2.get('system_prompt', 'N/A')}

        Goal: Create a merged Focus that preserves important information from both Focuses.

        Respond with JSON:
        {{
          "merged_name": "name for merged Focus",
          "merged_system_prompt": "combined system prompt",
          "key_content": "summary of what to preserve from both"
        }}
        """

        try:
            response = self.focus_merger.process_text(merge_prompt, max_tokens=512)
            merge_data = self._parse_merge_response(response)

            if merge_data:
                self._create_merged_focus(merge_data)

        except Exception as e:
            print(f"   ⚠️  Merge failed: {e}")

    def _parse_merge_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse merge response JSON"""
        try:
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"   ⚠️  Failed to parse merge response: {e}")
            return None

    def _create_merged_focus(self, merge_data: Dict[str, Any]):
        """Create new merged Focus"""
        merged_name = merge_data.get("merged_name", f"merged_{datetime.now().timestamp()}")

        # Use FocusFactory to create new Focus
        from focus_factory import FocusFactory
        factory = FocusFactory(self.senter_root)

        # Create merged Focus
        factory.create_focus(
            merged_name,
            initial_context=f"Merged from two Focuses.\n\nKey content: {merge_data.get('key_content', '')}"
        )

        print(f"   ✅ Created merged Focus: {merged_name}")

    def _split_focus(self, focus_name: str, sub_focuses: List[str]):
        """Split Focus into sub-Focuses"""
        if not self.focus_splitter:
            print("   ⚠️  Focus_Splitter not available, skipping split")
            return

        # Load original Focus config
        config = self.parser.load_focus_config(focus_name)

        # Build split prompt
        split_prompt = f"""
        Split this Focus into more focused sub-Focuses:

        Focus: {focus_name}
        System: {config.get('system_prompt', 'N/A')}

        Identify 2-4 sub-Focuses that would better organize the content.
        Each sub-focus should be specific and focused.

        Respond with JSON:
        {{
          "sub_focuses": [
            {{"name": "sub_focus_1", "content_description": "what goes here"}},
            {{"name": "sub_focus_2", "content_description": "what goes here"}}
          ]
        }}
        """

        try:
            response = self.focus_splitter.process_text(split_prompt, max_tokens=512)
            split_data = self._parse_split_response(response)

            if split_data:
                self._create_sub_focuses(split_data)

        except Exception as e:
            print(f"   ⚠️  Split failed: {e}")

    def _parse_split_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse split response JSON"""
        try:
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"   ⚠️  Failed to parse split response: {e}")
            return None

    def _create_sub_focuses(self, split_data: Dict[str, Any]):
        """Create sub-Focuses from split data"""
        sub_focuses = split_data.get("sub_focuses", [])

        from focus_factory import FocusFactory
        factory = FocusFactory(self.senter_root)

        for sub_focus in sub_focuses:
            name = sub_focus.get("name", f"sub_focus_{datetime.now().timestamp()}")
            content = sub_focus.get("content_description", "")

            factory.create_focus(name, initial_context=content)

        print(f"   ✅ Created {len(sub_focuses)} sub-Focuses")

    def _log_review(self, focus_name: str, action_data: Dict[str, Any], new_data: Dict[str, Any]):
        """Log review activity"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "focus_name": focus_name,
            "action": action_data.get("action", "none"),
            "reasoning": action_data.get("reasoning", ""),
            "new_data_type": list(new_data.keys()) if new_data else []
        }

        # Could log to file in future
        # For now, print to console
        print(f"   📝 Review logged: {log_entry}")


def main():
    """Test Focus Review Chain"""
    senter_root = Path(__file__).parent.parent
    review_chain = FocusReviewChain(senter_root)

    print("🔄 Starting Focus Review Chain test...")

    review_chain.start_review_worker()

    # Test review
    review_chain.queue_review("general", {
        "web": "Some web search results",
        "chat": "User said something",
        "function": "Function output"
    })

    # Wait for processing
    import time
    time.sleep(5)

    review_chain.stop_review_worker()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
OmniAgent Chain - Async orchestrator for omniagent instances
Main entry point for Senter - everything is omniagent with SENTER.md configs
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Import Senter utilities
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(1, str(Path(__file__).parent))

from Functions.omniagent_async import OmniAgentAsync
from Focuses.senter_md_parser import SenterMdParser


class OmniAgentChain:
    """Async chain of omniagent instances for Senter"""

    def __init__(self, senter_root: Optional[Path] = None):
        """
        Initialize OmniAgentChain

        Args:
            senter_root: Path to Senter directory
        """
        self.senter_root = senter_root or Path(__file__).parent.parent
        self.agents: Dict[str, OmniAgentAsync] = {}
        self.parser = SenterMdParser(self.senter_root)
        self.config_dir = self.senter_root / "config"
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = 50

    async def initialize(self):
        """Load all Focus agents on startup"""
        print("🔄 Initializing OmniAgent Chain...")

        focus_names = self.parser.list_all_focuses()

        # Load all agents in parallel
        tasks = [self._load_agent(focus) for focus in focus_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        print(f"✅ Loaded {len(self.agents)} Focus agents:")
        for name in sorted(self.agents.keys()):
            print(f"   - {name}")

    async def _load_agent(self, focus_name: str):
        """Load single agent"""
        try:
            self.agents[focus_name] = OmniAgentAsync(
                senter_root=self.senter_root, focus_name=focus_name
            )
        except Exception as e:
            print(f"   ⚠️  Failed to load {focus_name}: {e}")

    async def process_query(
        self,
        query: str,
        context: str = "",
        focus_hint: Optional[str] = None,
    ) -> str:
        """
        Main query processing chain

        Args:
            query: User query
            context: Additional context from previous interactions
            focus_hint: Optional Focus hint (if user specified /focus)

        Returns:
            Response from chat agent
        """
        print(f"\n📤 Processing query...")
        print(f"   Query: {query[:100]}...")

        # Add to conversation history
        self.conversation_history.append(
            {
                "role": "user",
                "content": query,
                "focus": focus_hint,
                "timestamp": str(Path(__file__).stat().st_mtime),
            }
        )

        # Step 1: Router selects target Focus (async)
        target_focus = await self._route_query(query, context, focus_hint)
        print(f"   🎯 Selected Focus: {target_focus}")

        # Step 2: Parallel context gathering (async)
        if target_focus != "general":
            await self._gather_context(target_focus, query)

        # Step 3: Parallel goal extraction (async)
        await self._extract_goals(target_focus, query)

        # Step 4: Get system prompt and context for target Focus
        system_prompt = self.parser.get_system_prompt(target_focus)
        focus_context = self.parser.get_focus_context(target_focus)

        # Step 5: Build final prompt for Chat agent
        final_prompt = self._build_final_prompt(
            query, system_prompt, focus_context, context
        )

        # Step 6: Target Focus processes with full context
        if target_focus in self.agents:
            response = await self.agents[target_focus].process_text(final_prompt)
        else:
            print(f"   ⚠️  Focus {target_focus} not loaded, using general")
            response = await self.agents["general"].process_text(final_prompt)

        # Add response to history
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": response,
                "focus": target_focus,
                "timestamp": str(Path(__file__).stat().st_mtime),
            }
        )

        # Trim history
        self.conversation_history = self.conversation_history[-self.max_history :]

        return response

    async def _route_query(
        self, query: str, context: str, focus_hint: Optional[str]
    ) -> str:
        """Router selects target Focus"""
        if focus_hint and focus_hint in self.agents:
            return focus_hint

        router_agent = self.agents.get("Router")
        if not router_agent:
            return "general"

        router_prompt = f"""
Available Focuses: {list(self.agents.keys())}

Query: {query}
Context: {context}

Your job: Select the best matching Focus for this query.
Output JSON: {{"focus": "focus_name", "reasoning": "brief explanation"}}
"""

        try:
            router_result = await router_agent.process_text(router_prompt)
            target_focus = self._parse_router_output(router_result)
            return target_focus
        except Exception as e:
            print(f"   ⚠️  Router failed: {e}")
            return "general"

    def _parse_router_output(self, output: str) -> str:
        """Parse router JSON output"""
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("focus", "general")
            except:
                pass

        # Fallback: simple keyword matching
        output_lower = output.lower()
        for focus in self.agents.keys():
            if focus.lower() in output_lower:
                return focus

        return "general"

    async def _gather_context(self, focus_name: str, query: str):
        """Gather and update context for a Focus"""
        context_gatherer = self.agents.get("Context_Gatherer")
        if not context_gatherer:
            return

        context_prompt = f"""
Gather context for Focus: {focus_name}
Query: {query}
Update SENTER.md with conversation summary.
"""

        try:
            result = await context_gatherer.process_text(context_prompt)
            print(f"   📚 Context gathered: {result[:100]}...")
        except Exception as e:
            print(f"   ⚠️  Context gathering failed: {e}")

    async def _extract_goals(self, focus_name: str, query: str):
        """Extract and update goals for a Focus"""
        goal_detector = self.agents.get("Goal_Detector")
        if not goal_detector:
            return

        goal_prompt = f"""
Query: {query}
Focus: {focus_name}
Extract goals specific to this Focus.
No cap on number of goals - detect all mentioned.
"""

        try:
            result = await goal_detector.process_text(goal_prompt)
            self._update_focus_goals(focus_name, result)
            print(f"   🎯 Goals extracted: {result[:100]}...")
        except Exception as e:
            print(f"   ⚠️  Goal extraction failed: {e}")

    def _update_focus_goals(self, focus_name: str, goals_json: str):
        """Update SENTER.md with new goals"""
        try:
            goals_data = json.loads(goals_json)
            goals = goals_data.get("goals", [])

            if goals:
                # Update SENTER.md "Detected Goals" section
                goals_text = "\n".join([f"- {g.get('text', '')}" for g in goals])
                self.parser.update_markdown_section(
                    focus_name, "Goals & Objectives", goals_text
                )
        except Exception as e:
            print(f"   ⚠️  Failed to update goals: {e}")

    def _build_final_prompt(
        self,
        query: str,
        system_prompt: str,
        focus_context: str,
        user_context: str,
    ) -> str:
        """Build final prompt for Chat agent"""
        parts = [f"Query: {query}"]

        if system_prompt:
            parts.append(f"System: {system_prompt}")

        if focus_context:
            parts.append(f"Focus Context:\n{focus_context}")

        if user_context:
            parts.append(f"Context: {user_context}")

        return "\n\n".join(parts)

    async def discover_tools(self):
        """Background cycle for tool discovery"""
        print("\n🔍 Running tool discovery cycle...")

        tool_discovery = self.agents.get("Tool_Discovery")
        if not tool_discovery:
            print("   ⚠️  Tool_Discovery agent not found")
            return

        try:
            result = await tool_discovery.process_text(
                "Discover all tools in Functions/ directory"
            )
            print(f"   {result}")
        except Exception as e:
            print(f"   ❌ Tool discovery failed: {e}")

    async def profile_cycle(self):
        """Background cycle for profiling all Focuses"""
        print("\n🧠 Running profiling cycle...")

        profiler_tasks = []
        for focus_name, agent in self.agents.items():
            # Skip internal agents
            if focus_name in [
                "Router",
                "Goal_Detector",
                "Tool_Discovery",
                "Context_Gatherer",
                "Planner",
                "Profiler",
                "Chat",
            ]:
                continue

            profiler_prompt = f"""
Profile user interactions for Focus: {focus_name}
Update User Preferences and Patterns Observed.
"""

            profiler_tasks.append(agent.process_text(profiler_prompt))

        # Run all profilers in parallel
        if profiler_tasks:
            results = await asyncio.gather(*profiler_tasks, return_exceptions=True)
            print(f"   ✅ Profiled {len(results)} Focuses")

    async def run_background_tasks(self):
        """Run all background cycles (discovery + profiling)"""
        print("\n🔄 Running background tasks...")

        await asyncio.gather(
            self.discover_tools(), self.profile_cycle(), return_exceptions=True
        )

        print("✅ Background tasks completed")

    def list_focuses(self) -> List[str]:
        """List all available Focuses"""
        return list(self.agents.keys())

    def list_internal_focuses(self) -> List[str]:
        """List internal Focus agents"""
        internal_focuses = [
            "Router",
            "Goal_Detector",
            "Tool_Discovery",
            "Context_Gatherer",
            "Planner",
            "Profiler",
            "Chat",
        ]
        return [f for f in internal_focuses if f in self.agents]

    def list_user_focuses(self) -> List[str]:
        """List user-facing Focus agents"""
        internal = self.list_internal_focuses()
        return [f for f in self.agents.keys() if f not in internal]

    async def close(self):
        """Cleanup all agents"""
        print("\n🛑 Closing OmniAgent Chain...")

        tasks = [agent.close() for agent in self.agents.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

        print("✅ All agents closed")


async def main_async():
    """Async main function"""
    print("=" * 60)
    print("🚀 SENTER - Async OmniAgent Chain")
    print("=" * 60)

    # Initialize chain
    chain = OmniAgentChain()
    await chain.initialize()

    # Run background discovery
    await chain.run_background_tasks()

    # Interactive loop
    print("\n✅ Senter is ready!")
    print("\nCommands:")
    print("  /list       - List all Focuses")
    print("  /focus <name> - Set Focus")
    print("  /goals      - Show current goals for Focus")
    print("  /exit       - Exit\n")

    current_focus = None

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break

            # Handle commands
            if user_input.startswith("/"):
                if user_input == "/list":
                    print("\n📁 Available Focuses:")
                    for focus in chain.list_user_focuses():
                        print(f"   - {focus}")
                elif user_input.startswith("/focus "):
                    current_focus = user_input.split(" ", 1)[1]
                    print(f"\n🎯 Focus set to: {current_focus}")
                elif user_input == "/goals" and current_focus:
                    print(f"\n🎯 Goals for {current_focus}:")
                    senter_file = (
                        chain.senter_root / "Focuses" / current_focus / "SENTER.md"
                    )
                    if senter_file.exists():
                        with open(senter_file) as f:
                            content = f.read()
                            if "Goals & Objectives" in content:
                                start = content.find("Goals & Objectives")
                                end = content.find("\n\n", start)
                                if end != -1:
                                    print(content[start:end])
                else:
                    print("⚠️  No Focus set. Use /focus <name> first")
                continue

            # Regular query
            response = await chain.process_query(user_input, focus_hint=current_focus)
            print(f"\nSenter: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

    # Cleanup
    await chain.close()


def main():
    """Sync entry point"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

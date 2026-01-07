#!/usr/bin/env python3
"""
Senter CLI - Universal AI Personal Assistant
Polished version with clean startup, help system, and feature indicators
"""

import asyncio
import sys
import os
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Suppress warnings before imports
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("senter.cli")

# Setup path
senter_root = Path(__file__).parent.parent
sys.path.insert(0, str(senter_root))
sys.path.insert(1, str(senter_root / "Functions"))
sys.path.insert(2, str(senter_root / "Focuses"))

# ============================================================
# HELP SYSTEM
# ============================================================

HELP_MAIN = """
📖 SENTER HELP

COMMANDS:
  /list              Show available focuses
  /focus <name>      Switch to a focus
  /exit              Save and exit

FEATURES:
  /route <query>     Test semantic routing
  /autoroute         Toggle auto-routing
  /goals             List active goals
  /memory            Show memory statistics
  /recall <query>    Search past conversations
  /learn             Show learned preferences

Type /help <topic> for detailed help on: goals, memory, learn, route
"""

HELP_TOPICS = {
    "goals": """
🎯 GOAL TRACKING

Senter automatically extracts goals from your conversations.

Examples that create goals:
  • "I need to finish my report by Friday"
  • "I want to learn Python this month"
  • "My goal is to exercise more"

Commands:
  /goals           List all active goals

Goals are saved between sessions and surface when relevant.
""",
    "memory": """
🧠 CONVERSATION MEMORY

Senter remembers your past conversations.

Commands:
  /memory          Show statistics
  /recall <query>  Search past conversations

Memory is used automatically to provide context in responses.
Ask "what do you remember about X?" to test recall.
""",
    "learn": """
📚 SELF-LEARNING

Senter learns your preferences from conversations:
  • Response style (brief vs detailed)
  • Preferred coding languages
  • Topics of interest
  • Communication formality

Commands:
  /learn           Show learned preferences

Preferences are applied automatically to improve responses.
""",
    "route": """
🧠 SEMANTIC ROUTING

Queries are automatically routed to the best Focus based on meaning.

Commands:
  /route <query>   Test where a query would route
  /autoroute       Toggle auto-routing on/off

Focuses: general, coding, research, creative, user_personal
""",
}


def show_help(topic: str = ""):
    """Show help for a topic"""
    topic = topic.strip().lower()
    if topic and topic in HELP_TOPICS:
        print(HELP_TOPICS[topic])
    else:
        print(HELP_MAIN)


# ============================================================
# PREFLIGHT CHECKS
# ============================================================

def check_ollama() -> bool:
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def check_model(model_name: str) -> bool:
    """Check if a model is available in Ollama"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(m.get("name", "").startswith(model_name) for m in models)
    except:
        pass
    return False


def preflight_checks(quiet: bool = False) -> bool:
    """Run preflight checks, return True if OK to proceed"""
    if not check_ollama():
        print("❌ Ollama is not running!")
        print("   Start it with: ollama serve")
        return False

    if not check_model("llama3.2"):
        print("❌ llama3.2 model not found!")
        print("   Install with: ollama pull llama3.2")
        return False

    if not quiet:
        if not check_model("nomic-embed-text"):
            print("  ⚠️  nomic-embed-text not found - install for better routing")

    return True


# ============================================================
# IMPORTS (after path setup)
# ============================================================

# Redirect stdout temporarily to suppress omniagent startup noise
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


# Try imports
try:
    from Functions.embedding_router import EmbeddingRouter
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

try:
    from Functions.memory import ConversationMemory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

try:
    from Functions.goal_tracker import GoalTracker
    GOALS_AVAILABLE = True
except ImportError:
    GOALS_AVAILABLE = False

try:
    from Functions.learner import SenterLearner
    LEARNER_AVAILABLE = True
except ImportError:
    LEARNER_AVAILABLE = False

try:
    from Functions.parallel_inference import parallel_respond, needs_research
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False


# ============================================================
# SESSION TRACKER
# ============================================================

class SessionTracker:
    """Track session activity for summaries"""
    def __init__(self):
        self.topics = []
        self.goals_updated = []
        self.preferences_learned = []
        self.message_count = 0
        self.start_time = datetime.now()

    def add_topic(self, query: str):
        """Extract and track topic from query"""
        # Simple topic extraction - first few words
        words = query.split()[:4]
        topic = " ".join(words)
        if topic and topic not in self.topics:
            self.topics.append(topic)

    def generate_summary(self) -> str:
        """Generate session summary"""
        lines = ["\n📝 Session Summary:"]

        if self.topics:
            lines.append(f"  • Discussed: {', '.join(self.topics[:3])}")

        if self.goals_updated:
            for goal in self.goals_updated[:2]:
                lines.append(f"  • Goal: {goal}")

        if self.preferences_learned:
            lines.append(f"  • Learned: {self.preferences_learned[0]}")

        lines.append(f"  • Messages: {self.message_count}")

        return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

async def main_async(quiet=False, verbose=False):
    """Main async function"""

    # Preflight checks
    if not preflight_checks(quiet):
        return

    # Clean startup banner
    print("\n" + "=" * 50)
    print("  🚀 SENTER - AI Personal Assistant")
    print("=" * 50)

    # Initialize parser
    from Focuses.senter_md_parser import SenterMdParser
    parser = SenterMdParser(senter_root)

    # Initialize features (quietly)
    router = None
    auto_route = False
    memory = None
    use_memory = False
    goal_tracker = None
    learner = None
    goal_stats = {}
    learner_stats = {}
    stats = {}

    print("\n  Loading features...", end="", flush=True)

    # Router
    if ROUTER_AVAILABLE:
        try:
            router = EmbeddingRouter(senter_root)
            auto_route = True
        except:
            pass

    # Memory
    if MEMORY_AVAILABLE:
        try:
            memory = ConversationMemory(senter_root)
            use_memory = True
            stats = memory.get_stats()
        except:
            pass

    # Goals
    if GOALS_AVAILABLE:
        try:
            goal_tracker = GoalTracker(senter_root)
            goal_stats = goal_tracker.get_stats()
        except:
            pass

    # Learner
    if LEARNER_AVAILABLE:
        try:
            learner = SenterLearner(senter_root)
            learner_stats = learner.get_stats()
        except:
            pass

    print(" done!")

    # Load LLM (suppress verbose output)
    print("  Loading LLM...", end="", flush=True)
    try:
        if not verbose:
            with SuppressOutput():
                from Functions.omniagent import SenterOmniAgent
                omniagent = SenterOmniAgent(verbose=False)
        else:
            from Functions.omniagent import SenterOmniAgent
            omniagent = SenterOmniAgent(verbose=True)
        print(" done!")
    except Exception as e:
        print(f" failed: {e}")
        return

    # Show focuses
    available_focuses = parser.list_all_focuses()
    print(f"\n  Focuses: {', '.join(available_focuses)}")

    # Show feature status
    features = []
    if router:
        features.append("routing")
    if memory:
        features.append(f"memory({stats.get('total_conversations', 0)})")
    if goal_tracker:
        features.append(f"goals({goal_stats.get('active', 0)})")
    if learner:
        features.append("learning")
    if PARALLEL_AVAILABLE:
        features.append("parallel")

    if features:
        print(f"  Features: {', '.join(features)}")

    # Proactive goal surfacing
    if goal_tracker:
        active_goals = goal_tracker.get_active_goals()
        if active_goals:
            print(f"\n  📋 Active goals ({len(active_goals)}):")
            for goal in active_goals[:3]:
                print(f"     • {goal.description[:50]}")

    print(f"\n  Type /help for commands\n")

    # Interactive loop
    current_focus = "general"
    session_messages = []
    session = SessionTracker()
    use_parallel = PARALLEL_AVAILABLE

    while True:
        try:
            user_input = input(f"[{current_focus}] You: ").strip()

            if not user_input:
                continue

            # Exit handling
            if user_input.lower() in ["/exit", "quit", "q", "/quit"]:
                # Learn from session
                if learner and session_messages:
                    try:
                        insights = learner.analyze_conversation(session_messages)
                        learner.update_user_profile(insights)
                        if insights.topics:
                            session.preferences_learned.append(f"topics: {', '.join(insights.topics)}")
                    except:
                        pass

                # Save conversation
                if use_memory and memory and session_messages:
                    conv_id = memory.save_conversation(session_messages, focus=current_focus)

                # Show summary
                if session.message_count > 0:
                    print(session.generate_summary())

                print("\n👋 See you next time!")
                break

            # Help command
            if user_input.lower().startswith("/help"):
                topic = user_input[5:].strip()
                show_help(topic)
                continue

            # Command handling
            if user_input.startswith("/"):
                cmd = user_input.strip().lower()

                if cmd == "/list":
                    print(f"\n📁 Focuses: {', '.join(available_focuses)}")
                    continue

                elif cmd.startswith("/focus ") and len(cmd.split()) > 1:
                    new_focus = cmd.split()[1]
                    if new_focus in available_focuses:
                        current_focus = new_focus
                        auto_route = False
                        print(f"\n🎯 Switched to: {new_focus}")
                    else:
                        print(f"\n⚠️ Unknown focus: {new_focus}")
                    continue

                elif cmd == "/autoroute":
                    if router:
                        auto_route = not auto_route
                        print(f"\n🧠 Auto-routing: {'ON' if auto_route else 'OFF'}")
                    else:
                        print("\n⚠️ Router not available")
                    continue

                elif cmd.startswith("/route "):
                    if router:
                        test_query = user_input[7:].strip()
                        if test_query:
                            focus, score, all_scores = router.route_query(test_query)
                            print(f"\n🧠 \"{test_query[:30]}...\" → {focus} ({score:.2f})")
                    else:
                        print("\n⚠️ Router not available")
                    continue

                elif cmd == "/memory":
                    if memory:
                        s = memory.get_stats()
                        print(f"\n💾 Memory: {s['total_conversations']} conversations, {s['total_chunks']} chunks")
                    else:
                        print("\n⚠️ Memory not available")
                    continue

                elif cmd.startswith("/recall "):
                    if memory:
                        query = user_input[8:].strip()
                        results = memory.search_memory(query, limit=3)
                        if results:
                            print(f"\n💾 Found {len(results)} memories:")
                            for r in results:
                                content = r.content[:80] + "..." if len(r.content) > 80 else r.content
                                print(f"   • {content}")
                        else:
                            print("\n   No relevant memories found")
                    else:
                        print("\n⚠️ Memory not available")
                    continue

                elif cmd == "/goals":
                    if goal_tracker:
                        active = goal_tracker.get_active_goals()
                        if active:
                            print(f"\n🎯 Active goals ({len(active)}):")
                            for g in active:
                                print(f"   • {g.description}")
                        else:
                            print("\n🎯 No active goals")
                    else:
                        print("\n⚠️ Goals not available")
                    continue

                elif cmd == "/learn":
                    if learner:
                        prefs = learner.get_learned_preferences()
                        print("\n📚 Learned preferences:")
                        if prefs.get("response_length"):
                            print(f"   • Style: {prefs['response_length']}")
                        if prefs.get("code_language"):
                            print(f"   • Language: {prefs['code_language']}")
                        if prefs.get("topics_of_interest"):
                            print(f"   • Interests: {', '.join(prefs['topics_of_interest'][:3])}")
                        if not any([prefs.get("response_length"), prefs.get("code_language"), prefs.get("topics_of_interest")]):
                            print("   • (none yet - keep chatting!)")
                    else:
                        print("\n⚠️ Learning not available")
                    continue

                else:
                    print(f"\n⚠️ Unknown command: {cmd}")
                    print("   Type /help for available commands")
                    continue

            # Process query
            session.add_topic(user_input)
            session.message_count += 1

            # Auto-route
            if auto_route and router:
                routed_focus, score, _ = router.route_query(user_input)
                if routed_focus != current_focus and score >= 0.4:
                    current_focus = routed_focus
                    print(f"  [→ {current_focus}]")

            # Get context
            indicators = []

            # Memory context
            memory_context = ""
            if use_memory and memory:
                results = memory.search_memory(user_input, limit=2)
                if results:
                    memory_context = memory.get_context_for_query(user_input, max_context=2)
                    indicators.append(f"🧠 {len(results)} memories")

            # Goals context
            goals_context = ""
            if goal_tracker:
                relevant = goal_tracker.get_relevant_goals(user_input, limit=2)
                if relevant:
                    goals_context = goal_tracker.get_goals_context(user_input)
                    indicators.append(f"🎯 {len(relevant)} goals")

            # Show indicators
            if indicators and not quiet:
                print(f"  [{' | '.join(indicators)}]")

            # Build query with context
            context_parts = []
            if memory_context:
                context_parts.append(memory_context)
            if goals_context:
                context_parts.append(goals_context)

            query_with_context = user_input
            if context_parts:
                query_with_context = "\n\n".join(context_parts) + f"\n\n[Query]: {user_input}"

            # Track message
            session_messages.append({"role": "user", "content": user_input})

            # Generate response
            try:
                if use_parallel and PARALLEL_AVAILABLE and needs_research(user_input):
                    print("  [🔍 researching...]")
                    result = parallel_respond(query_with_context, senter_root)
                    response = result.enhanced_response
                else:
                    response = omniagent.process_text(query_with_context)

                print(f"\nSenter: {response}\n")
                session_messages.append({"role": "assistant", "content": response})

                # Extract goals
                if goal_tracker:
                    try:
                        goals = goal_tracker.extract_goals_from_message(user_input, response)
                        for g in goals:
                            goal_tracker.save_goal(g)
                            session.goals_updated.append(g.description[:30])
                            if not quiet:
                                print(f"  [🎯 Goal detected: {g.description[:40]}...]")
                    except:
                        pass

            except Exception as e:
                print(f"\n❌ Error: {e}\n")

        except KeyboardInterrupt:
            # Save on Ctrl+C
            if learner and session_messages:
                try:
                    insights = learner.analyze_conversation(session_messages)
                    learner.update_user_profile(insights)
                except:
                    pass

            if use_memory and memory and session_messages:
                memory.save_conversation(session_messages, focus=current_focus)

            print("\n\n👋 See you next time!")
            break

        except EOFError:
            break


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description="Senter CLI")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all debug output")
    args = parser.parse_args()

    asyncio.run(main_async(quiet=args.quiet, verbose=args.verbose))


if __name__ == "__main__":
    main()

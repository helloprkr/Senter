#!/usr/bin/env python3
"""
Senter 3.0 - Your 24/7 AI Assistant

Usage:
  python senter.py              # Interactive CLI (standalone)
  python senter.py --daemon     # Start as background daemon
  python senter.py --connect    # Connect to running daemon
  python senter.py --voice      # Enable voice input
  python senter.py --status     # Show daemon status
  python senter.py --version    # Show version
"""

import argparse
import asyncio
import sys
from pathlib import Path

from core.engine import Senter


async def run_cli(senter: Senter, voice_enabled: bool = False) -> None:
    """Run the interactive CLI."""
    interface_config = senter.genome.interface
    cli_config = interface_config.get("cli", {})
    show_ai_state = cli_config.get("show_ai_state", True)

    # Initialize optional components
    voice = None
    goal_detector = None
    proactive = None

    # Try to load goal detector and proactive engine
    try:
        from intelligence.goals import GoalDetector
        from intelligence.proactive import ProactiveSuggestionEngine

        goal_detector = GoalDetector(senter.memory)
        proactive = ProactiveSuggestionEngine(senter, goal_detector)
    except Exception as e:
        print(f"Note: Intelligence modules not available: {e}")

    # Try to load voice interface
    if voice_enabled:
        try:
            from interface.voice import VoiceInterface

            voice = VoiceInterface()
            voice.load()
            if voice.is_available():
                print("Voice input enabled. Press Enter to record (5s)")
        except Exception as e:
            print(f"Note: Voice interface not available: {e}")
            voice = None

    # Show proactive suggestions on startup
    if proactive:
        try:
            suggestions = await proactive.generate_suggestions()
            if suggestions:
                print("\nSuggestions for you:")
                for s in suggestions:
                    print(f"  - {s['title']}")
                    proactive.mark_suggested(s["id"])
                print()
        except Exception:
            pass

    while True:
        try:
            mode = senter.coupling.get_mode_name()
            prompt = cli_config.get("prompt", "[{mode}] You: ").format(mode=mode)

            # Check for voice input
            if voice and voice.is_available():
                user_input = input(prompt).strip()
                if user_input == "":
                    # Empty input = record voice
                    print("Recording...")
                    user_input = await voice.record_and_transcribe(5.0)
                    if user_input:
                        print(f"You said: {user_input}")
                    else:
                        continue
            else:
                user_input = input(prompt).strip()

            if user_input.lower() in ("quit", "exit", "q"):
                await senter.shutdown()
                break

            if not user_input:
                continue

            # Handle special commands
            if user_input.startswith("/"):
                await handle_command(senter, user_input, goal_detector, proactive)
                continue

            # Regular interaction
            response = await senter.interact(user_input)

            # Show AI state if configured
            if show_ai_state:
                print(
                    f"\n[AI State: Focus={response.ai_state.focus}, "
                    f"Mode={response.ai_state.mode}, "
                    f"Trust={response.ai_state.trust_level:.2f}]"
                )

            print(f"\nSenter: {response.text}\n")

            # Analyze for goals
            if goal_detector:
                try:
                    new_goals = goal_detector.analyze_interaction(user_input, response.text)
                    for goal in new_goals:
                        if goal.confidence > 0.7:
                            print(f"[Detected goal: {goal.description}]")
                except Exception:
                    pass

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            await senter.shutdown()
            break
        except EOFError:
            await senter.shutdown()
            break
        except Exception as e:
            print(f"\nError: {e}\n")


async def handle_command(senter: Senter, command: str, goal_detector=None, proactive=None) -> None:
    """Handle slash commands."""
    parts = command.split()
    cmd = parts[0].lower()

    if cmd == "/status":
        status = senter.get_status()
        print("\n--- System Status ---")
        print(f"Version: {status['version']}")
        print(f"Memory: {status['memory_stats']}")
        print(f"Trust: {status['trust']['level']:.2f} ({status['trust']['trend']})")
        print(f"Fitness: {status['fitness']['average']:.2f} ({status['fitness']['trend']})")

        # Show evolution status
        evolution_stats = senter.mutations.get_stats()
        print(f"Evolution: {evolution_stats['total_proposed']} mutations, {evolution_stats['total_applied']} applied")
        if evolution_stats.get("active_experiment"):
            print(f"  Active experiment: {evolution_stats['experiment_progress']}")
        print()

    elif cmd == "/memory":
        stats = senter.memory.get_stats()
        print("\n--- Memory Stats ---")
        for key, value in stats.items():
            print(f"{key}: {value}")
        print()

    elif cmd == "/trust":
        trust = senter.trust.to_dict()
        print("\n--- Trust Level ---")
        print(f"Level: {trust['level']:.2f}")
        print(f"Trend: {trust['trend']}")
        print(f"Can suggest confidently: {trust['can_suggest_confidently']}")
        print(f"Can be proactive: {trust['can_be_proactive']}")
        print()

    elif cmd == "/capabilities":
        caps = senter.capabilities.get_available_names()
        print("\n--- Available Capabilities ---")
        for cap in caps:
            print(f"  - {cap}")
        print()

    elif cmd == "/goals":
        if goal_detector:
            print("\n--- Your Goals ---")
            goals = goal_detector.get_active_goals()
            if goals:
                for g in goals[:10]:
                    progress_bar = "=" * int(g.progress * 10) + "-" * (10 - int(g.progress * 10))
                    print(f"  [{progress_bar}] {g.description}")
                    print(f"      Confidence: {g.confidence:.0%}, Category: {g.category}")
            else:
                print("  No goals detected yet.")
            print()
        else:
            print("Goal tracking not available")

    elif cmd == "/evolution":
        stats = senter.mutations.get_stats()
        summary = senter.mutations.get_evolution_summary()
        print("\n--- Evolution Status ---")
        print(f"Total mutations: {summary['total']}")
        print(f"Successful: {summary['successful']}")
        print(f"Rolled back: {summary['rolled_back']}")
        if summary.get("avg_fitness_improvement"):
            print(f"Avg improvement: {summary['avg_fitness_improvement']:.3f}")
        if summary.get("recent_mutations"):
            print("\nRecent mutations:")
            for m in summary["recent_mutations"]:
                status = "kept" if m["success"] else "rolled back"
                print(f"  - {m['type']}: {m['reason'][:50]}... ({status})")
        print()

    elif cmd == "/suggest":
        if proactive:
            suggestions = await proactive.generate_suggestions()
            if suggestions:
                print("\n--- Suggestions ---")
                for s in suggestions:
                    print(f"  {s['title']}")
                    print(f"    {s['action']}")
                    proactive.mark_suggested(s["id"])
                print()
            else:
                print("No suggestions right now (trust may be too low)")
        else:
            print("Proactive suggestions not available")

    elif cmd == "/help":
        print("\n--- Commands ---")
        print("/status       - Show system status")
        print("/memory       - Show memory statistics")
        print("/trust        - Show trust level")
        print("/capabilities - Show available capabilities")
        print("/goals        - Show detected goals")
        print("/evolution    - Show evolution/mutation history")
        print("/suggest      - Get proactive suggestions")
        print("/help         - Show this help")
        print("quit/exit     - Exit Senter")
        print()

    else:
        print(f"Unknown command: {cmd}")
        print("Type /help for available commands")


async def run_daemon(genome_path: Path) -> None:
    """Run as background daemon."""
    from daemon.senter_daemon import SenterDaemon

    daemon = SenterDaemon(genome_path)
    await daemon.start()


async def connect_daemon() -> None:
    """Connect to running daemon."""
    from daemon.senter_client import main as client_main

    await client_main()


async def run_multimodal(genome_path: Path) -> None:
    """Run with multimodal (voice + gaze) input."""
    try:
        from interface.multimodal import MultimodalInterface

        # Initialize Senter
        senter = Senter(genome_path)
        await senter.initialize()

        print(f"Senter {senter.genome.version} (Multimodal Mode)")

        # Initialize multimodal interface
        multimodal = MultimodalInterface()
        if not multimodal.load():
            print("Multimodal interface not available, falling back to CLI")
            await run_cli(senter, voice_enabled=True)
            return

        print("Look at camera to activate voice input...")
        print("Press Ctrl+C to exit")

        # Handle voice input
        async def on_voice_input(text: str):
            if text.lower() in ("quit", "exit"):
                multimodal.stop()
                return

            response = await senter.interact(text)
            print(f"\nYou: {text}")
            print(f"Senter: {response.text}\n")

        await multimodal.start(on_voice_input)

    except ImportError as e:
        print(f"Multimodal not available: {e}")
        print("Install with: pip install openai-whisper sounddevice mediapipe opencv-python")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        await senter.shutdown()


async def show_status(genome_path: Path) -> None:
    """Show daemon status."""
    socket_path = genome_path.parent / "data" / "senter.sock"

    if socket_path.exists():
        # Try to connect to daemon
        try:
            from daemon.senter_client import SenterClient

            client = SenterClient(socket_path)
            await client.connect()
            status = await client.status()
            await client.disconnect()

            print("Senter Daemon Status")
            print(f"  Running: {status['running']}")
            print(f"  Pending tasks: {status['pending_tasks']}")
            print(f"  Current task: {status['current_task'] or 'None'}")
            print(f"  Completed tasks: {status['completed_tasks']}")
            print(f"  Trust level: {status['trust_level']:.2f}")
            print(f"  Memory episodes: {status['memory_episodes']}")
        except Exception as e:
            print(f"Could not connect to daemon: {e}")
    else:
        print("Daemon not running")


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    genome_path = Path(args.genome)

    if not genome_path.exists():
        print(f"Error: genome.yaml not found at {genome_path}")
        return 1

    # Validate configuration
    from core.config_validator import validate_config
    if not validate_config(genome_path):
        print("Fix configuration errors before starting.")
        return 1

    if args.daemon:
        await run_daemon(genome_path)
        return 0

    if args.connect:
        await connect_daemon()
        return 0

    if args.status:
        await show_status(genome_path)
        return 0

    if args.voice:
        await run_multimodal(genome_path)
        return 0

    # Default: standalone CLI mode
    try:
        senter = Senter(genome_path)
        await senter.initialize()

        print(f"Senter {senter.genome.version} initialized")
        print(f"Trust level: {senter.trust.level:.2f}")
        print(f"Memory episodes: {len(senter.memory.episodic)}")
        print("Type '/help' for commands, 'quit' to exit\n")

        await run_cli(senter)

    except Exception as e:
        print(f"Error initializing Senter: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Senter 3.0 - Your 24/7 AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python senter.py              # Interactive CLI
  python senter.py --daemon     # Start background daemon
  python senter.py --connect    # Connect to daemon
  python senter.py --voice      # Voice + gaze mode
  python senter.py --status     # Show daemon status
        """,
    )
    parser.add_argument(
        "--genome", "-g",
        type=str,
        default="genome.yaml",
        help="Path to genome.yaml configuration",
    )
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Start as background daemon",
    )
    parser.add_argument(
        "--connect", "-c",
        action="store_true",
        help="Connect to running daemon",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice + gaze input",
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show daemon status",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Rich TUI interface",
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    if args.version:
        print("Senter 3.0.0")
        return 0

    # TUI runs its own event loop, handle before asyncio.run()
    if args.tui:
        genome_path = Path(args.genome)
        if not genome_path.exists():
            print(f"Error: genome.yaml not found at {genome_path}")
            return 1
        from core.config_validator import validate_config
        if not validate_config(genome_path):
            print("Fix configuration errors before starting.")
            return 1
        from interface.tui import run_tui
        run_tui(genome_path)
        return 0

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

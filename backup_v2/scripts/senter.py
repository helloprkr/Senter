#!/usr/bin/env python3
"""
Senter CLI - Simple Universal AI Personal Assistant
Uses direct omniagent with Focus system
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Setup path
senter_root = Path(__file__).parent.parent
sys.path.insert(0, str(senter_root))
sys.path.insert(1, str(senter_root / "Functions"))
sys.path.insert(2, str(senter_root / "Focuses"))

from Functions.omniagent import SenterOmniAgent
from Focuses.senter_md_parser import SenterMdParser


async def main_async(quiet=False):
    """Async main function"""
    print("=" * 60)
    print("🚀 SENTER v2.0 - Universal AI Personal Assistant")
    print("=" * 60)

    # Initialize parser
    parser = SenterMdParser(senter_root)

    # List available Focuses
    print("\n📁 Available Focuses:")
    available_focuses = parser.list_all_focuses()
    for focus in available_focuses:
        print(f"   - {focus}")

    # Simple: use general focus
    focus_name = "general"

    # Initialize omniagent with general focus config
    if not quiet:
        print(f"\n🔄 Loading model for Focus: {focus_name}...")

    # Get general focus config
    general_config = parser.load_focus_config("general")

    # Create omniagent with general config (pass quiet mode)
    omniagent = SenterOmniAgent(verbose=not quiet)

    if not quiet:
        print("✅ Senter initialized!")
        print("\n💬 Commands:")
        print("  /list         - List all Focuses")
        print("  /focus <name> - Set Focus")
        print("  /exit         - Exit\n")

    # Interactive loop
    current_focus = "general"

    while True:
        try:
            user_input = input(f"[{current_focus}] You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.strip().lower()

                if cmd == "/list":
                    print("\n📁 Available Focuses:")
                    for focus in available_focuses:
                        print(f"   - {focus}")
                elif cmd.startswith("/focus ") and len(cmd.split()) > 1:
                    new_focus = cmd.split()[1]
                    if new_focus in available_focuses:
                        current_focus = new_focus
                        print(f"\n🎯 Focus set to: {new_focus}")
                    else:
                        print(f"\n⚠️ Unknown focus: {new_focus}")
                else:
                    print(f"\n⚠️  Unknown command: {user_input}")
                    continue

            # Regular query
            print(f"\n📤 Processing: {user_input[:100]}...")

            try:
                response = omniagent.process_text(user_input)
                print(f"\n✅ Senter: {response[:100]}...")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

        except KeyboardInterrupt:
            if not quiet:
                print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            sys.exit(1)


def main():
    """Sync entry point"""
    parser = argparse.ArgumentParser(description="Senter CLI")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress debug output"
    )
    args = parser.parse_args()

    asyncio.run(main_async(quiet=args.quiet))


if __name__ == "__main__":
    main()

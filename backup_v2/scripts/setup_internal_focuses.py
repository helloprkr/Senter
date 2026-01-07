#!/usr/bin/env python3
"""
Setup Internal Focuses - Create all internal Senter Focuses
Creates: Focus_Reviewer, Focus_Merger, Focus_Splitter, Planner_Agent,
         Coder_Agent, User_Profiler, Diagnostic_Agent, Chat_Agent
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Focuses.focus_factory import FocusFactory


def setup_internal_focuses():
    """Create all internal Focuses for Senter operations"""
    senter_root = Path(__file__).parent.parent.parent
    factory = FocusFactory(senter_root)

    print("🔧 Creating Internal Focuses for Senter...")
    print("=" * 60)

    internal_focuses = [
        ("Focus_Reviewer", "review"),
        ("Focus_Merger", "merge"),
        ("Focus_Splitter", "split"),
        ("Planner_Agent", "plan"),
        ("Coder_Agent", "code"),
        ("User_Profiler", "profile"),
        ("Diagnostic_Agent", "diagnostic"),
        ("Chat_Agent", "chat"),
    ]

    created = []
    failed = []

    for focus_name, focus_type in internal_focuses:
        try:
            print(f"\n📝 Creating internal Focus: {focus_name}")
            factory.create_internal_focus(focus_name, focus_type)
            created.append(focus_name)
            print(f"   ✅ Created: {focus_name}")
        except Exception as e:
            failed.append((focus_name, str(e)))
            print(f"   ❌ Failed to create {focus_name}: {e}")

    print("\n" + "=" * 60)
    print(f"\n✅ Setup Complete!")
    print(f"   Created: {len(created)}/{len(internal_focuses)} internal Focuses")

    if failed:
        print(f"\n⚠️  Failed to create {len(failed)} Focuses:")
        for name, error in failed:
            print(f"      - {name}: {error}")

    print("\n💡 Next Steps:")
    print("   1. Configure user's central model in config/user_profile.json")
    print("   2. Run Senter app: python3 scripts/senter_app.py")
    print("   3. Internal Focuses will automatically be loaded and used")


if __name__ == "__main__":
    setup_internal_focuses()

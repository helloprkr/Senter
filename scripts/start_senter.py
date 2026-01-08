#!/usr/bin/env python3
"""
INT-001: The Full Experience Works

Single entry point to start Senter with all features.

VALUE: User runs one command → Senter lives in their menubar,
researches in the background, notifies when done.
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.pipeline import ResearchPipeline
from research.research_store import ResearchStore
from research.feedback import FeedbackAnalyzer

try:
    from ui.menubar_app import SenterMenubar, RUMPS_AVAILABLE
except ImportError:
    RUMPS_AVAILABLE = False
    SenterMenubar = None

try:
    from ui.research_panel import ResearchPanel, PYQT_AVAILABLE
except ImportError:
    PYQT_AVAILABLE = False
    ResearchPanel = None


def check_dependencies() -> bool:
    """Check if all dependencies are available."""
    issues = []

    # Check rumps
    if not RUMPS_AVAILABLE:
        issues.append("rumps not installed (pip install rumps)")

    # Check PyQt
    if not PYQT_AVAILABLE:
        issues.append("PyQt6 not installed (pip install PyQt6)")

    # Check Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code != 200:
            issues.append("Ollama not responding at localhost:11434")
    except Exception:
        issues.append("Ollama not running - start with: ollama serve")

    # Check web search
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            issues.append("ddgs not installed (pip install ddgs)")

    if issues:
        print("\n⚠️  Missing dependencies:\n")
        for issue in issues:
            print(f"  • {issue}")
        print("\nInstall with:")
        print("  pip install rumps PyQt6 ddgs httpx readability-lxml beautifulsoup4 lxml")
        print("\nStart Ollama with:")
        print("  ollama serve")
        return False

    return True


def show_status():
    """Show current Senter status."""
    store = ResearchStore()
    analyzer = FeedbackAnalyzer()

    stats = store.get_stats()
    feedback_stats = analyzer.get_stats()

    print("\n=== Senter Status ===")
    print(f"\nResearch:")
    print(f"  Total: {stats['total']}")
    print(f"  Unviewed: {stats['unviewed']}")
    print(f"  Avg confidence: {stats['avg_confidence']}")

    print(f"\nFeedback:")
    print(f"  Rated: {feedback_stats.total_ratings}")
    print(f"  Avg rating: {feedback_stats.average_rating}/5")

    # Recent research
    recent = store.get_recent(3)
    if recent:
        print(f"\nRecent Research:")
        for r in recent:
            status = "🆕" if not r.viewed else "👁"
            print(f"  {status} {r.topic}")

    # Suggestions
    suggestions = analyzer.get_improvement_suggestions()
    print(f"\nSuggestions:")
    for s in suggestions:
        print(f"  • {s}")


def run_demo():
    """Run a demo of the full pipeline."""
    print("\n=== Senter Demo ===\n")

    # Demo conversation
    messages = [
        {"role": "user", "content": "I'm building a web application with React."},
        {"role": "assistant", "content": "Nice! What kind of features are you adding?"},
        {"role": "user", "content": "I wonder if I should use Redux or Context API for state management."},
        {"role": "assistant", "content": "Both have their use cases. Redux is great for complex state."},
        {"role": "user", "content": "Yeah, I'm curious about when each one is appropriate."}
    ]

    print("Simulated conversation:")
    for msg in messages:
        print(f"  {msg['role'].upper()}: {msg['content']}")

    print("\n--- Running Research Pipeline ---\n")

    pipeline = ResearchPipeline(max_sources=3)
    results = pipeline.process_conversation(messages, max_topics=1)

    if results:
        for result in results:
            print(f"\n✓ Topic: {result.topic}")
            print(f"  Sources: {result.sources_found}")
            print(f"  Confidence: {result.confidence}")
            print(f"  Time: {result.total_time_ms}ms")

            if result.success:
                print(f"\n  Summary:")
                for line in result.summary.split("\n")[:3]:
                    print(f"    {line}")

                if result.key_insights:
                    print(f"\n  Key Insights:")
                    for insight in result.key_insights[:2]:
                        print(f"    • {insight[:80]}...")

                print(f"\n  Stored as ID: {result.research_id}")
    else:
        print("No research-worthy topics detected.")

    print("\n=== Demo Complete ===\n")


def run_menubar():
    """Run the menubar application."""
    if not RUMPS_AVAILABLE:
        print("rumps not available - cannot run menubar app")
        sys.exit(1)

    print("Starting Senter menubar app...")
    print("Look for ◇ in your menubar!")

    def on_show_panel():
        """Show research panel when clicked."""
        if PYQT_AVAILABLE:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            panel = ResearchPanel()
            panel.show()
            app.exec()

    menubar = SenterMenubar(on_show_panel=on_show_panel)
    menubar.run()


def run_panel():
    """Run just the research panel."""
    if not PYQT_AVAILABLE:
        print("PyQt6 not available - cannot run panel")
        sys.exit(1)

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    panel = ResearchPanel()
    panel.show()

    sys.exit(app.exec())


def run_research(topic: str):
    """Run research on a specific topic."""
    print(f"\nResearching: {topic}\n")

    pipeline = ResearchPipeline(max_sources=5)
    result = pipeline.research_topic(topic)

    if result.success:
        print(f"✓ Research complete!")
        print(f"  Sources: {result.sources_found}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Time: {result.total_time_ms}ms")
        print(f"\nSummary:\n{result.summary}")

        if result.key_insights:
            print("\nKey Insights:")
            for insight in result.key_insights:
                print(f"  • {insight}")

        print(f"\nStored as ID: {result.research_id}")
    else:
        print(f"✗ Research failed: {result.error}")


def run_all_tests():
    """Run all Senter tests."""
    import subprocess

    test_dir = Path(__file__).parent.parent / "tests"
    test_files = list(test_dir.glob("test_*.py"))

    print(f"\n=== Running {len(test_files)} Test Files ===\n")

    total_passed = 0
    total_failed = 0

    for test_file in sorted(test_files):
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            cwd=str(test_file.parent.parent)
        )

        # Parse results from output
        output = result.stdout
        if "Results:" in output:
            line = [l for l in output.split("\n") if "Results:" in l][0]
            # Extract numbers
            import re
            match = re.search(r"(\d+) passed, (\d+) failed", line)
            if match:
                passed = int(match.group(1))
                failed = int(match.group(2))
                total_passed += passed
                total_failed += failed

                status = "✓" if failed == 0 else "✗"
                print(f"{status} {test_file.name}: {passed} passed, {failed} failed")

    print(f"\n{'='*60}")
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print(f"{'='*60}")

    return total_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Senter - AI Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_senter.py                  # Start menubar app
  python start_senter.py --demo           # Run demo
  python start_senter.py --panel          # Show research panel
  python start_senter.py --research "AI"  # Research a topic
  python start_senter.py --status         # Show status
  python start_senter.py --test           # Run all tests
        """
    )

    parser.add_argument("--demo", "-d", action="store_true",
                        help="Run demo of research pipeline")
    parser.add_argument("--status", "-s", action="store_true",
                        help="Show current status")
    parser.add_argument("--panel", "-p", action="store_true",
                        help="Open research panel")
    parser.add_argument("--research", "-r", type=str,
                        help="Research a specific topic")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Run all tests")
    parser.add_argument("--check", "-c", action="store_true",
                        help="Check dependencies")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    if args.check:
        if check_dependencies():
            print("\n✓ All dependencies available!")
        sys.exit(0)

    if args.test:
        success = run_all_tests()
        sys.exit(0 if success else 1)

    if args.status:
        show_status()
        sys.exit(0)

    if args.demo:
        run_demo()
        sys.exit(0)

    if args.panel:
        run_panel()
        sys.exit(0)

    if args.research:
        run_research(args.research)
        sys.exit(0)

    # Default: run menubar
    if not check_dependencies():
        sys.exit(1)

    run_menubar()


if __name__ == "__main__":
    main()

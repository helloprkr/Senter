#!/usr/bin/env python3
"""
Senter WebView Application

PyWebView-based UI with glassmorphism design.
Bridges web frontend with Python research backend.
"""

import logging
import sys
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger("senter.ui.webview")

try:
    import webview
    WEBVIEW_AVAILABLE = True
except ImportError:
    WEBVIEW_AVAILABLE = False
    logger.warning("pywebview not installed - run: pip install pywebview")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from research.pipeline import ResearchPipeline
    from research.research_store import ResearchStore
except ImportError as e:
    logger.error(f"Could not import research modules: {e}")
    ResearchPipeline = None
    ResearchStore = None


class SenterAPI:
    """
    Python API exposed to JavaScript via pywebview.

    Handles research requests and data retrieval.
    """

    def __init__(self):
        self._pipeline: Optional[ResearchPipeline] = None
        self._store: Optional[ResearchStore] = None
        self._init_backend()

    def _init_backend(self):
        """Initialize research backend."""
        try:
            self._store = ResearchStore()
            self._pipeline = ResearchPipeline(max_sources=5)
            logger.info("Backend initialized")
        except Exception as e:
            logger.error(f"Backend init failed: {e}")

    def research(self, query: str) -> Dict[str, Any]:
        """
        Execute research on a query.

        Called from JavaScript when user submits a question.
        """
        logger.info(f"Research request: {query}")

        if not self._pipeline:
            return {"success": False, "error": "Backend not available"}

        try:
            result = self._pipeline.research_topic(query)

            return {
                "success": result.success,
                "topic": result.topic,
                "summary": result.summary,
                "key_insights": result.key_insights,
                "confidence": result.confidence,
                "sources_found": result.sources_found,
                "research_id": result.research_id,
                "error": result.error,
                "timestamp": self._format_time()
            }
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {"success": False, "error": str(e)}

    def get_research(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent research results."""
        if not self._store:
            return []

        try:
            results = self._store.get_recent(limit)
            return [
                {
                    "research_id": r.id,
                    "topic": r.topic,
                    "summary": r.summary,
                    "key_insights": r.key_insights,
                    "confidence": r.confidence,
                    "sources_found": len(r.sources),
                    "rating": r.feedback_rating,
                    "timestamp": r.created_at.strftime("%I:%M %p")
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Get research failed: {e}")
            return []

    def rate_research(self, research_id: int, rating: int) -> bool:
        """Save rating for a research result."""
        if not self._store:
            return False

        try:
            self._store.set_feedback(research_id, rating)
            logger.info(f"Rated research {research_id}: {rating} stars")
            return True
        except Exception as e:
            logger.error(f"Rating failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get research statistics."""
        if not self._store:
            return {}

        try:
            return self._store.get_stats()
        except Exception as e:
            logger.error(f"Stats failed: {e}")
            return {}

    def _format_time(self) -> str:
        """Format current time."""
        from datetime import datetime
        return datetime.now().strftime("%I:%M:%S %p")


class SenterWebView:
    """
    Main WebView application.

    Creates a window with the web-based UI.
    """

    def __init__(
        self,
        title: str = "Senter",
        width: int = 500,
        height: int = 700,
        transparent: bool = True
    ):
        if not WEBVIEW_AVAILABLE:
            raise RuntimeError("pywebview not available")

        self.title = title
        self.width = width
        self.height = height
        self.transparent = transparent
        self.api = SenterAPI()
        self._window: Optional[webview.Window] = None

    def run(self):
        """Run the webview application."""
        # Get path to web files
        web_dir = Path(__file__).parent / "web"
        index_path = web_dir / "index.html"

        if not index_path.exists():
            raise FileNotFoundError(f"Web UI not found: {index_path}")

        logger.info(f"Starting Senter WebView from {index_path}")

        # Create window
        self._window = webview.create_window(
            self.title,
            str(index_path),
            width=self.width,
            height=self.height,
            js_api=self.api,
            background_color='#0a0a0a',  # Dark background
            frameless=False,  # Keep frame for now
            easy_drag=True,
            text_select=True
        )

        # Start webview
        webview.start(debug=False)

    def show(self):
        """Show the window (if hidden)."""
        if self._window:
            self._window.show()

    def hide(self):
        """Hide the window."""
        if self._window:
            self._window.hide()

    def destroy(self):
        """Destroy the window."""
        if self._window:
            self._window.destroy()


def run_webview():
    """Run the Senter webview application."""
    app = SenterWebView()
    app.run()


# Headless version for testing
class HeadlessWebView:
    """Headless version for testing."""

    def __init__(self, **kwargs):
        self.api = SenterAPI()

    def run(self):
        pass

    def show(self):
        pass

    def hide(self):
        pass


# CLI
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Senter WebView App")
    parser.add_argument("--test", action="store_true", help="Test mode")
    parser.add_argument("--api-test", action="store_true", help="Test API only")

    args = parser.parse_args()

    if args.test:
        print("=== WebView Test ===")
        if WEBVIEW_AVAILABLE:
            print("✓ pywebview is available")
            api = SenterAPI()
            print(f"✓ API initialized")
            research = api.get_research(5)
            print(f"✓ Found {len(research)} research items")
            print("✓ WebView ready")
        else:
            print("✗ pywebview not available")

    elif args.api_test:
        print("=== API Test ===")
        api = SenterAPI()

        # Test get_research
        research = api.get_research()
        print(f"Research items: {len(research)}")

        # Test research (will take time)
        print("\nResearching 'Python decorators'...")
        result = api.research("Python decorators")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Summary: {result['summary'][:100]}...")

    else:
        if not WEBVIEW_AVAILABLE:
            print("pywebview not available - install with: pip install pywebview")
            sys.exit(1)

        print("Starting Senter...")
        run_webview()

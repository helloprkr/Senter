#!/usr/bin/env python3
"""
UI-001: Menubar Shows Senter is Alive

macOS menubar app showing Senter status with unread count.

VALUE: User glances at menubar → sees Senter has 3 new findings.
One click → panel slides down with summaries.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger("senter.ui.menubar")

try:
    import rumps
    RUMPS_AVAILABLE = True
except ImportError:
    RUMPS_AVAILABLE = False
    logger.warning("rumps not installed - run: pip install rumps")

# Import research components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from research.research_store import ResearchStore
    from research.pipeline import ResearchPipeline
except ImportError:
    ResearchStore = None
    ResearchPipeline = None


class SenterMenubar:
    """
    macOS menubar application for Senter.

    Features:
    - Shows unread research count in menubar
    - Click to view pending research
    - Status indicator (idle, researching, new findings)
    """

    # Icons for different states
    ICONS = {
        "idle": "◇",      # Diamond outline - idle
        "active": "◆",    # Diamond filled - researching
        "new": "●",       # Circle - new findings
    }

    def __init__(
        self,
        on_show_panel: Optional[Callable] = None,
        on_research_clicked: Optional[Callable[[int], None]] = None
    ):
        """
        Initialize menubar app.

        Args:
            on_show_panel: Callback when user wants to see research panel
            on_research_clicked: Callback when specific research is clicked
        """
        if not RUMPS_AVAILABLE:
            raise RuntimeError("rumps not available - cannot create menubar app")

        self.on_show_panel = on_show_panel
        self.on_research_clicked = on_research_clicked

        # State
        self._state = "idle"
        self._unread_count = 0
        self._store: Optional[ResearchStore] = None
        self._app: Optional[rumps.App] = None

        # Try to connect to store
        try:
            self._store = ResearchStore()
        except Exception as e:
            logger.warning(f"Could not connect to research store: {e}")

    def _create_app(self) -> rumps.App:
        """Create the rumps application."""
        app = rumps.App(
            "Senter",
            title=self._get_title(),
            quit_button=None  # We'll add our own
        )

        # Add menu items
        app.menu = [
            rumps.MenuItem("View Research", callback=self._on_view_research),
            rumps.MenuItem("Research Now...", callback=self._on_research_now),
            None,  # Separator
            rumps.MenuItem("Status: Idle"),
            None,  # Separator
            rumps.MenuItem("Quit Senter", callback=self._on_quit)
        ]

        return app

    def _get_title(self) -> str:
        """Get menubar title based on state."""
        icon = self.ICONS.get(self._state, self.ICONS["idle"])

        if self._unread_count > 0:
            return f"{icon} {self._unread_count}"
        return icon

    def _update_title(self):
        """Update the menubar title."""
        if self._app:
            self._app.title = self._get_title()

    def _on_view_research(self, sender):
        """Handle View Research menu click."""
        logger.info("View Research clicked")
        if self.on_show_panel:
            self.on_show_panel()
        else:
            # Open webview panel
            self._open_webview_panel()

    def _on_research_now(self, sender):
        """Handle Research Now menu click."""
        logger.info("Research Now clicked")
        # Show input dialog for topic
        response = rumps.Window(
            message="Enter a topic to research:",
            title="Senter Research",
            default_text="",
            ok="Research",
            cancel="Cancel",
            dimensions=(300, 24)
        ).run()

        if response.clicked and response.text.strip():
            topic = response.text.strip()
            self._start_research(topic)

    def _on_quit(self, sender):
        """Handle quit."""
        logger.info("Quitting Senter")
        rumps.quit_application()

    def _open_webview_panel(self):
        """Open the webview research panel."""
        try:
            from .webview_app import SenterWebView
            import subprocess
            import sys

            # Launch webview in separate process
            webview_script = Path(__file__).parent / "webview_app.py"
            subprocess.Popen([sys.executable, str(webview_script)])
            logger.info("Launched webview panel")

        except Exception as e:
            logger.error(f"Could not open webview: {e}")
            # Fallback to alert
            self._show_research_alert()

    def _show_research_alert(self):
        """Show alert with pending research."""
        if not self._store:
            rumps.alert("Senter", "Research store not available")
            return

        pending = self._store.get_unviewed(limit=5)
        if not pending:
            rumps.alert("Senter", "No pending research")
            return

        # Format pending research
        message = "Pending Research:\n\n"
        for r in pending:
            message += f"• {r.topic}\n"
            message += f"  {r.summary[:100]}...\n\n"

        rumps.alert("Senter Research", message)

        # Mark as viewed
        for r in pending:
            self._store.mark_viewed(r.id)
        self._refresh_count()

    def _start_research(self, topic: str):
        """Start research in background."""
        self.set_state("active")
        self._update_status(f"Researching: {topic}")

        def do_research():
            try:
                pipeline = ResearchPipeline(max_sources=3)
                result = pipeline.research_topic(topic)

                if result.success:
                    self.set_state("new")
                    self._refresh_count()
                    rumps.notification(
                        title="Senter",
                        subtitle=f"Research Complete: {topic}",
                        message=result.summary[:100] + "..."
                    )
                else:
                    self.set_state("idle")
                    rumps.notification(
                        title="Senter",
                        subtitle="Research Failed",
                        message=result.error or "Unknown error"
                    )
            except Exception as e:
                logger.error(f"Research failed: {e}")
                self.set_state("idle")
            finally:
                self._update_status("Idle")

        thread = threading.Thread(target=do_research, daemon=True)
        thread.start()

    def _update_status(self, status: str):
        """Update status menu item."""
        if self._app and len(self._app.menu) > 3:
            self._app.menu["Status: Idle"].title = f"Status: {status}"

    def _refresh_count(self):
        """Refresh unread count from store."""
        if self._store:
            try:
                unviewed = self._store.get_unviewed()
                self._unread_count = len(unviewed)
                if self._unread_count > 0:
                    self._state = "new"
                elif self._state == "new":
                    self._state = "idle"
                self._update_title()
            except Exception as e:
                logger.warning(f"Could not refresh count: {e}")

    def set_state(self, state: str):
        """Set menubar state (idle, active, new)."""
        if state in self.ICONS:
            self._state = state
            self._update_title()

    def run(self):
        """Run the menubar app (blocking)."""
        self._app = self._create_app()
        self._refresh_count()

        # Start periodic refresh
        def refresh_loop():
            while True:
                time.sleep(60)  # Refresh every minute
                self._refresh_count()

        refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        refresh_thread.start()

        logger.info("Starting Senter menubar app")
        self._app.run()


# Headless mode for testing without GUI
class HeadlessMenubar:
    """Headless version for testing without macOS GUI."""

    def __init__(self, **kwargs):
        self.state = "idle"
        self.unread_count = 0
        self._store = None
        try:
            self._store = ResearchStore()
        except:
            pass

    def set_state(self, state: str):
        self.state = state

    def refresh_count(self):
        if self._store:
            self.unread_count = len(self._store.get_unviewed())
        return self.unread_count

    def run(self):
        pass  # No-op for headless


# CLI for testing
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Senter Menubar App")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--test", action="store_true", help="Test mode - don't run app")

    args = parser.parse_args()

    if args.test:
        print("=== Menubar Test ===")
        if RUMPS_AVAILABLE:
            print("✓ rumps is available")
            menubar = HeadlessMenubar()
            print(f"✓ State: {menubar.state}")
            print(f"✓ Unread count: {menubar.refresh_count()}")
            print("✓ Menubar ready (would show in macOS menubar)")
        else:
            print("✗ rumps not available - install with: pip install rumps")

    elif args.headless:
        print("Running in headless mode...")
        menubar = HeadlessMenubar()
        print(f"State: {menubar.state}")
        print(f"Unread: {menubar.refresh_count()}")

    else:
        if not RUMPS_AVAILABLE:
            print("rumps not available - install with: pip install rumps")
            sys.exit(1)

        print("Starting Senter menubar...")
        menubar = SenterMenubar()
        menubar.run()

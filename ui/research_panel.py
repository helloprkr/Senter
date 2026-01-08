#!/usr/bin/env python3
"""
UI-002: Research Panel That's Actually Nice to Read

Glassmorphism-styled panel for displaying research summaries.

VALUE: User opens panel → sees beautiful, scannable summaries.
Not a wall of text—structured insights they can act on.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List, Callable

logger = logging.getLogger("senter.ui.panel")

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QScrollArea, QFrame, QProgressBar,
        QSizePolicy, QSpacerItem
    )
    from PyQt6.QtCore import Qt, QSize, pyqtSignal
    from PyQt6.QtGui import QFont, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logger.warning("PyQt6 not installed - run: pip install PyQt6")

# Import styles
try:
    from .styles import FULL_STYLESHEET, COLORS, get_confidence_color
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from styles import FULL_STYLESHEET, COLORS, get_confidence_color

# Import research store
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from research.research_store import ResearchStore, StoredResearch
except ImportError:
    ResearchStore = None
    StoredResearch = None


class ResearchCard(QFrame):
    """A card displaying a single research result."""

    feedback_given = pyqtSignal(int, int)  # research_id, rating

    def __init__(self, research: "StoredResearch", parent=None):
        super().__init__(parent)
        self.research = research
        self.setObjectName("researchCard")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header: Topic + Confidence
        header = QHBoxLayout()

        title = QLabel(self.research.topic)
        title.setObjectName("title")
        title.setWordWrap(True)
        header.addWidget(title, 1)

        # Confidence badge
        conf_text = f"{int(self.research.confidence * 100)}%"
        conf_label = QLabel(conf_text)
        conf_label.setStyleSheet(f"""
            background-color: {get_confidence_color(self.research.confidence)};
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        """)
        header.addWidget(conf_label)

        layout.addLayout(header)

        # Summary
        summary = QLabel(self.research.summary)
        summary.setObjectName("body")
        summary.setWordWrap(True)
        summary.setMaximumHeight(100)
        layout.addWidget(summary)

        # Key insights
        if self.research.key_insights:
            insights_label = QLabel("Key Insights")
            insights_label.setObjectName("subtitle")
            layout.addWidget(insights_label)

            for insight in self.research.key_insights[:3]:
                bullet = QLabel(f"• {insight}")
                bullet.setObjectName("insight")
                bullet.setWordWrap(True)
                layout.addWidget(bullet)

        # Footer: Sources + Rating
        footer = QHBoxLayout()

        # Sources count
        source_count = len(self.research.sources)
        sources_label = QLabel(f"📚 {source_count} sources")
        sources_label.setObjectName("muted")
        footer.addWidget(sources_label)

        # Time
        time_str = self.research.created_at.strftime("%b %d, %H:%M")
        time_label = QLabel(f"🕐 {time_str}")
        time_label.setObjectName("muted")
        footer.addWidget(time_label)

        footer.addStretch()

        # Rating stars
        self._add_rating(footer)

        layout.addLayout(footer)

    def _add_rating(self, layout: QHBoxLayout):
        """Add star rating buttons."""
        current_rating = self.research.feedback_rating or 0

        for i in range(1, 6):
            star = QPushButton("★" if i <= current_rating else "☆")
            star.setObjectName("starFilled" if i <= current_rating else "star")
            star.setFixedSize(28, 28)
            star.clicked.connect(lambda checked, r=i: self._on_rate(r))
            layout.addWidget(star)

    def _on_rate(self, rating: int):
        """Handle rating click."""
        self.feedback_given.emit(self.research.id, rating)


class ResearchPanel(QMainWindow):
    """
    Main research panel window.

    Displays pending and recent research in a scrollable list.
    """

    def __init__(
        self,
        on_feedback: Optional[Callable[[int, int], None]] = None,
        parent=None
    ):
        super().__init__(parent)
        self.on_feedback = on_feedback
        self._store: Optional[ResearchStore] = None

        try:
            self._store = ResearchStore()
        except Exception as e:
            logger.warning(f"Could not connect to store: {e}")

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Senter Research")
        self.setMinimumSize(500, 600)
        self.resize(550, 700)

        # Apply stylesheet
        self.setStyleSheet(FULL_STYLESHEET)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QHBoxLayout()
        title = QLabel("Research")
        title.setStyleSheet(f"""
            color: {COLORS['text']};
            font-size: 24px;
            font-weight: bold;
        """)
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Scroll area for cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.cards_widget = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_widget)
        self.cards_layout.setSpacing(16)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(self.cards_widget)
        layout.addWidget(scroll)

        # Load initial data
        self.refresh()

    def refresh(self):
        """Refresh research list."""
        # Clear existing cards
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._store:
            self._add_placeholder("Research store not available")
            return

        # Get research
        researches = self._store.get_recent(limit=10)

        if not researches:
            self._add_placeholder("No research yet. Start a conversation!")
            return

        # Add cards
        for research in researches:
            card = ResearchCard(research)
            card.feedback_given.connect(self._on_feedback)
            self.cards_layout.addWidget(card)

            # Mark as viewed
            if not research.viewed:
                self._store.mark_viewed(research.id)

        # Add spacer at bottom
        self.cards_layout.addStretch()

    def _add_placeholder(self, message: str):
        """Add placeholder message."""
        label = QLabel(message)
        label.setObjectName("muted")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cards_layout.addWidget(label)

    def _on_feedback(self, research_id: int, rating: int):
        """Handle feedback from card."""
        logger.info(f"Feedback: research={research_id}, rating={rating}")
        if self._store:
            self._store.set_feedback(research_id, rating)
        if self.on_feedback:
            self.on_feedback(research_id, rating)
        self.refresh()


# Headless version for testing
class HeadlessPanel:
    """Headless panel for testing without GUI."""

    def __init__(self, **kwargs):
        self._store = None
        try:
            self._store = ResearchStore()
        except:
            pass

    def get_research_count(self) -> int:
        if self._store:
            return len(self._store.get_recent(limit=100))
        return 0

    def refresh(self):
        pass

    def show(self):
        pass


# CLI for testing
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Senter Research Panel")
    parser.add_argument("--headless", action="store_true", help="Headless mode")
    parser.add_argument("--test", action="store_true", help="Test mode")

    args = parser.parse_args()

    if args.test:
        print("=== Research Panel Test ===")
        if PYQT_AVAILABLE:
            print("✓ PyQt6 is available")
            panel = HeadlessPanel()
            print(f"✓ Research count: {panel.get_research_count()}")
            print("✓ Panel ready (would show GUI window)")
        else:
            print("✗ PyQt6 not available - install with: pip install PyQt6")

    elif args.headless:
        panel = HeadlessPanel()
        print(f"Research count: {panel.get_research_count()}")

    else:
        if not PYQT_AVAILABLE:
            print("PyQt6 not available - install with: pip install PyQt6")
            sys.exit(1)

        app = QApplication(sys.argv)

        # Set dark palette
        app.setStyle("Fusion")

        panel = ResearchPanel()
        panel.show()

        sys.exit(app.exec())

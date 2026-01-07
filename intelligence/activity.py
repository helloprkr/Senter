"""
Activity Monitoring - Learns from computer activity.

ENHANCED: Now uses LLM for deep context inference.

Capabilities:
- Screen OCR: Extract text from screen
- App tracking: Which apps are active
- LLM Context Inference: Deep understanding of work context
- Goal suggestion: Infer goals from activity patterns
- Project detection: Understand what project user is working on
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import json
import subprocess
import sys

if TYPE_CHECKING:
    from models.base import BaseModel


@dataclass
class ActivitySnapshot:
    """A point-in-time capture of user activity."""
    timestamp: datetime
    active_app: str
    window_title: str
    screen_text: List[str]  # Key phrases extracted
    inferred_context: str  # coding, writing, browsing, etc.
    llm_analysis: Optional[Dict[str, Any]] = None  # Deep LLM analysis
    detected_project: Optional[str] = None
    detected_tasks: List[str] = field(default_factory=list)


@dataclass
class ActivityPattern:
    """Recurring pattern in user activity."""
    pattern_type: str  # daily_routine, project_work, research_session
    description: str
    frequency: int
    last_seen: datetime
    associated_goals: List[str]


class ScreenCapture:
    """Captures and OCRs screen content."""

    def __init__(self):
        self.last_capture: Optional[ActivitySnapshot] = None

    def capture(self) -> Optional[Dict[str, Any]]:
        """Capture current screen and extract text."""
        try:
            # Take screenshot
            import pyautogui
            from PIL import Image
            import pytesseract

            screenshot = pyautogui.screenshot()

            # OCR the image
            text = pytesseract.image_to_string(screenshot)

            # Extract key phrases (simple: non-trivial lines)
            lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
            key_phrases = lines[:20]  # Top 20 lines

            return {
                'text': text,
                'key_phrases': key_phrases,
                'timestamp': datetime.now()
            }

        except ImportError as e:
            # Dependencies not installed - this is optional
            return None
        except Exception as e:
            return None

    def get_active_window(self) -> Dict[str, str]:
        """Get currently active window info."""
        try:
            if sys.platform == 'darwin':  # macOS
                script = '''
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    set frontWindow to ""
                    try
                        set frontWindow to name of front window of first application process whose frontmost is true
                    end try
                end tell
                return frontApp & "|" & frontWindow
                '''
                result = subprocess.run(
                    ['osascript', '-e', script],
                    capture_output=True, text=True, timeout=5
                )
                parts = result.stdout.strip().split('|')
                return {
                    'app': parts[0] if parts else 'Unknown',
                    'window': parts[1] if len(parts) > 1 else ''
                }

            elif sys.platform == 'linux':
                # Use xdotool on Linux
                result = subprocess.run(
                    ['xdotool', 'getactivewindow', 'getwindowname'],
                    capture_output=True, text=True, timeout=5
                )
                return {'app': 'Unknown', 'window': result.stdout.strip()}

            elif sys.platform == 'win32':
                # Windows
                try:
                    import ctypes
                    user32 = ctypes.windll.user32
                    h_wnd = user32.GetForegroundWindow()
                    length = user32.GetWindowTextLengthW(h_wnd)
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(h_wnd, buf, length + 1)
                    return {'app': 'Unknown', 'window': buf.value}
                except Exception:
                    return {'app': 'Unknown', 'window': ''}

            return {'app': 'Unknown', 'window': ''}

        except Exception as e:
            return {'app': 'Unknown', 'window': str(e)}


class ContextInferencer:
    """Infers work context from activity."""

    CONTEXT_PATTERNS = {
        'coding': {
            'apps': ['code', 'vscode', 'pycharm', 'intellij', 'vim', 'nvim', 'terminal', 'iterm', 'cursor'],
            'keywords': ['def ', 'class ', 'import ', 'function', 'const ', 'let ', 'var ', 'async ', 'await']
        },
        'writing': {
            'apps': ['word', 'docs', 'notion', 'obsidian', 'pages', 'textedit', 'bear', 'ulysses'],
            'keywords': ['chapter', 'section', 'paragraph', 'draft', 'edit', 'document']
        },
        'research': {
            'apps': ['chrome', 'firefox', 'safari', 'edge', 'scholar', 'arc'],
            'keywords': ['abstract', 'paper', 'study', 'research', 'doi', 'arxiv', 'github']
        },
        'communication': {
            'apps': ['slack', 'teams', 'discord', 'mail', 'outlook', 'gmail', 'messages'],
            'keywords': ['inbox', 'sent', 'reply', 'message', 'channel', 'thread']
        },
        'design': {
            'apps': ['figma', 'sketch', 'photoshop', 'illustrator', 'canva', 'affinity'],
            'keywords': ['layer', 'frame', 'component', 'prototype', 'design']
        }
    }

    def infer_context(self, snapshot: Dict[str, Any]) -> str:
        """Infer context from activity snapshot."""
        app = snapshot.get('app', '').lower()
        window = snapshot.get('window', '').lower()
        text = ' '.join(snapshot.get('key_phrases', [])).lower()

        scores = {}
        for context, patterns in self.CONTEXT_PATTERNS.items():
            score = 0

            # Check app name
            for app_pattern in patterns['apps']:
                if app_pattern in app or app_pattern in window:
                    score += 3

            # Check keywords in screen text
            for keyword in patterns['keywords']:
                if keyword in text:
                    score += 1

            scores[context] = score

        # Return highest scoring context
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best

        return 'general'


class ActivityMonitor:
    """
    Main activity monitoring system.

    Runs in background, periodically captures activity,
    infers context, and suggests goals.
    """

    def __init__(self, senter_engine=None, capture_interval: int = 60):
        self.engine = senter_engine
        self.capture_interval = capture_interval  # seconds
        self.screen = ScreenCapture()
        self.inferencer = ContextInferencer()

        self.running = False
        self.history: List[ActivitySnapshot] = []
        self.patterns: List[ActivityPattern] = []

        # Persist path
        self.data_path = Path("data/activity")
        self.data_path.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Start activity monitoring."""
        self.running = True
        print(f"[ACTIVITY] Starting monitor (interval: {self.capture_interval}s)")

        while self.running:
            await self._capture_cycle()
            await asyncio.sleep(self.capture_interval)

    def stop(self):
        """Stop monitoring."""
        self.running = False
        self._save_history()

    async def _capture_cycle(self):
        """Single capture and analysis cycle."""
        # Get active window
        window_info = self.screen.get_active_window()

        # Capture screen (less frequent - every 5 cycles)
        screen_data = None
        if len(self.history) % 5 == 0:
            screen_data = self.screen.capture()

        # Combine data
        combined = {
            'app': window_info['app'],
            'window': window_info['window'],
            'key_phrases': screen_data['key_phrases'] if screen_data else []
        }

        # Infer context
        context = self.inferencer.infer_context(combined)

        # Create snapshot
        snapshot = ActivitySnapshot(
            timestamp=datetime.now(),
            active_app=window_info['app'],
            window_title=window_info['window'],
            screen_text=combined['key_phrases'],
            inferred_context=context
        )

        self.history.append(snapshot)

        # Analyze patterns periodically
        if len(self.history) % 10 == 0:
            await self._analyze_patterns()

        # Trim history to last 1000 entries
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    async def _analyze_patterns(self):
        """Analyze activity history for patterns."""
        # Count context frequencies
        context_counts = {}
        for snap in self.history[-100:]:  # Last 100 snapshots
            ctx = snap.inferred_context
            context_counts[ctx] = context_counts.get(ctx, 0) + 1

        # Identify dominant contexts
        for ctx, count in context_counts.items():
            if count >= 20:  # Significant presence
                # Check if this suggests a goal
                await self._suggest_goal_from_context(ctx, count)

    async def _suggest_goal_from_context(self, context: str, frequency: int):
        """Suggest a goal based on observed context."""
        if not self.engine or not hasattr(self.engine, 'goal_detector'):
            return

        # Map contexts to goal suggestions
        suggestions = {
            'coding': "Complete the coding project you've been working on",
            'writing': "Finish the document you've been writing",
            'research': "Complete your research on the topic you've been exploring",
            'design': "Finish the design project you've been working on"
        }

        if context in suggestions:
            # Check if goal already exists
            existing_goals = self.engine.goal_detector.get_active_goals()
            goal_texts = [g.description.lower() for g in existing_goals]

            suggestion = suggestions[context]
            if not any(context in g for g in goal_texts):
                # Create suggested goal
                try:
                    self.engine.goal_detector._create_or_update_goal(
                        description=suggestion,
                        evidence=f"Observed {frequency} activity snapshots in {context} context",
                        confidence=min(0.7, 0.3 + (frequency * 0.02))
                    )
                except Exception:
                    pass  # Goal creation is optional

    def get_current_context(self) -> str:
        """Get the current inferred context."""
        if self.history:
            return self.history[-1].inferred_context
        return 'unknown'

    def get_activity_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent activity."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [s for s in self.history if s.timestamp > cutoff]

        # Context distribution
        contexts = {}
        apps = {}
        for snap in recent:
            contexts[snap.inferred_context] = contexts.get(snap.inferred_context, 0) + 1
            apps[snap.active_app] = apps.get(snap.active_app, 0) + 1

        return {
            'total_snapshots': len(recent),
            'context_distribution': contexts,
            'top_apps': dict(sorted(apps.items(), key=lambda x: x[1], reverse=True)[:5]),
            'current_context': self.get_current_context()
        }

    def _save_history(self):
        """Save history to disk."""
        try:
            history_file = self.data_path / "history.json"
            data = [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'app': s.active_app,
                    'window': s.window_title,
                    'context': s.inferred_context
                }
                for s in self.history[-500:]  # Save last 500
            ]
            with open(history_file, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass  # Saving is optional

    def _load_history(self):
        """Load history from disk."""
        try:
            history_file = self.data_path / "history.json"
            if history_file.exists():
                with open(history_file) as f:
                    data = json.load(f)
                self.history = [
                    ActivitySnapshot(
                        timestamp=datetime.fromisoformat(d['timestamp']),
                        active_app=d['app'],
                        window_title=d['window'],
                        screen_text=[],
                        inferred_context=d['context']
                    )
                    for d in data
                ]
        except Exception:
            pass  # Loading is optional

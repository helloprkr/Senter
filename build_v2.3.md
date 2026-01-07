Here's exactly what to tell Claude Code:

---

```
Read this complete implementation plan and execute in order. Do not skip steps. Show output at each stage.

═══════════════════════════════════════════════════════════════════════════════
SENTER 3.0 FINALIZATION - COMPLETE IMPLEMENTATION SEQUENCE
═══════════════════════════════════════════════════════════════════════════════

PHASE 0: ULTIMATE DAEMON VALIDATION
───────────────────────────────────────────────────────────────────────────────

Run this validation first. Do not proceed until it passes.

Step 1: Start daemon in background
```bash
cd "/Users/maximvs/Desktop/__2026_/Senter 3.0 ⏅"
python senter.py --daemon &
sleep 5  # Wait for startup
```

Step 2: Connect and have a conversation
```bash
python senter.py --connect << 'EOF'
Hello, I'm working on an AI project called Senter
I want to make it the best personal AI assistant
My goal is to have it work autonomously while I'm away
What features would make this most useful?
/goals
/status
/task research "autonomous AI assistants best practices"
quit
EOF
```

Step 3: Wait and reconnect
```bash
sleep 90  # Wait for background task
python senter.py --connect << 'EOF'
/status
What did you research while I was away?
/evolution
quit
EOF
```

Step 4: Verify results
- Did "While you were away" show the research results?
- Did /goals show detected goals?
- Did /evolution show the system is tracking fitness?

If any step fails, fix before proceeding.

═══════════════════════════════════════════════════════════════════════════════
PHASE A: VOICE & GAZE TESTING
═══════════════════════════════════════════════════════════════════════════════

A1. Verify hardware
```python
# Test microphone
import sounddevice as sd
print("Audio devices:", sd.query_devices())
print("Default input:", sd.query_devices(kind='input'))

# Test camera
import cv2
cap = cv2.VideoCapture(0)
print("Camera available:", cap.isOpened())
if cap.isOpened():
    ret, frame = cap.read()
    print("Frame captured:", ret, frame.shape if ret else None)
cap.release()
```

A2. Test Whisper voice recognition
```python
import asyncio
from interface.voice import VoiceInterface

async def test_voice():
    v = VoiceInterface(model_size="base")
    v.load()
    
    heard_text = []
    def on_text(text):
        print(f"✓ Heard: {text}")
        heard_text.append(text)
    
    print("Speak something in the next 10 seconds...")
    
    # Listen for 10 seconds
    import time
    task = asyncio.create_task(v.start_listening(on_text))
    await asyncio.sleep(10)
    v.stop_listening()
    
    print(f"Total utterances: {len(heard_text)}")
    return len(heard_text) > 0

asyncio.run(test_voice())
```

A3. Test gaze detection
```python
import asyncio
from interface.gaze import GazeDetector

async def test_gaze():
    g = GazeDetector()
    g.load()
    
    events = []
    def on_start():
        print("👀 GAZE DETECTED - Looking at camera")
        events.append('start')
    
    def on_end():
        print("👀 GAZE ENDED - Looked away")
        events.append('end')
    
    print("Look at camera, then look away. Testing for 15 seconds...")
    
    task = asyncio.create_task(g.start(on_start, on_end))
    await asyncio.sleep(15)
    g.stop()
    
    print(f"Events detected: {events}")
    return len(events) >= 2

asyncio.run(test_gaze())
```

A4. Test full multimodal integration
```python
# With daemon running:
python senter.py --voice

# Instructions:
# 1. Look at camera for 1 second
# 2. Say "Hello Senter, what can you help me with?"
# 3. Look away to stop listening
# 4. Verify response is generated
```

A5. Fix any issues found, then confirm:
```bash
echo "Voice + Gaze Status:"
python -c "
from interface.voice import VoiceInterface
from interface.gaze import GazeDetector
v = VoiceInterface(); v.load(); print('✓ Voice OK')
g = GazeDetector(); g.load(); print('✓ Gaze OK')
"
```

═══════════════════════════════════════════════════════════════════════════════
PHASE B: TUI INTERFACE
═══════════════════════════════════════════════════════════════════════════════

B1. Create rich TUI with real-time panels

Create file: interface/tui.py

```python
"""
Senter TUI - Rich terminal interface with real-time system state.

Panels:
┌─────────────────────────────────────────────────────────────────────┐
│ SENTER 3.0                                    Trust: 0.72 | DIALOGUE │
├─────────────────────────────────────┬───────────────────────────────┤
│ Chat                                │ AI State                      │
│                                     │ ─────────────────────────────│
│ You: Hello                          │ Mode: DIALOGUE                │
│ Senter: Hi! How can I help?         │ Focus: greeting               │
│                                     │ Frustration: 0.0              │
│ You: I'm working on my thesis       │ Energy: 0.8                   │
│ Senter: I can help with that...     │                               │
│                                     ├───────────────────────────────┤
│                                     │ Goals                         │
│                                     │ ─────────────────────────────│
│                                     │ • Learn ML (0.85)             │
│                                     │ • Finish thesis (0.72)        │
├─────────────────────────────────────┼───────────────────────────────┤
│ Background Tasks                    │ Evolution                     │
│ ─────────────────────────────────── │ ─────────────────────────────│
│ ✓ research: AI safety (done)        │ Mutations: 3 (2 kept)         │
│ ◐ summarize: daily (running)        │ Fitness: 0.67 ↑               │
│ ○ research: ML papers (pending)     │ Last: threshold_change ✓      │
└─────────────────────────────────────┴───────────────────────────────┘
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Input, RichLog
from textual.reactive import reactive
from textual import events
import asyncio
from pathlib import Path
from datetime import datetime


class ChatPanel(RichLog):
    """Scrollable chat history."""
    
    def add_message(self, role: str, text: str):
        if role == "user":
            self.write(f"[bold cyan]You:[/] {text}")
        else:
            self.write(f"[bold green]Senter:[/] {text}")


class AIStatePanel(Static):
    """Real-time AI state display."""
    
    mode = reactive("DIALOGUE")
    focus = reactive("none")
    trust = reactive(0.5)
    frustration = reactive(0.0)
    energy = reactive(0.8)
    
    def render(self) -> str:
        return f"""[bold]AI State[/]
────────────────────
Mode: [cyan]{self.mode}[/]
Focus: {self.focus}
Trust: [{"green" if self.trust > 0.7 else "yellow" if self.trust > 0.5 else "red"}]{self.trust:.2f}[/]
Frustration: {self.frustration:.2f}
Energy: {self.energy:.2f}
"""


class GoalsPanel(Static):
    """Detected goals display."""
    
    goals = reactive([])
    
    def render(self) -> str:
        lines = ["[bold]Goals[/]", "────────────────────"]
        if self.goals:
            for g in self.goals[:5]:
                lines.append(f"• {g['desc'][:25]} ({g['conf']:.0%})")
        else:
            lines.append("[dim]No goals detected yet[/]")
        return "\n".join(lines)


class TasksPanel(Static):
    """Background tasks display."""
    
    pending = reactive([])
    current = reactive(None)
    completed = reactive([])
    
    def render(self) -> str:
        lines = ["[bold]Background Tasks[/]", "────────────────────"]
        
        for t in self.completed[-3:]:
            lines.append(f"[green]✓[/] {t['type']}: {t['desc'][:20]}")
        
        if self.current:
            lines.append(f"[yellow]◐[/] {self.current['type']}: {self.current['desc'][:20]}")
        
        for t in self.pending[:3]:
            lines.append(f"[dim]○[/] {t['type']}: {t['desc'][:20]}")
        
        if not (self.completed or self.current or self.pending):
            lines.append("[dim]No tasks[/]")
        
        return "\n".join(lines)


class EvolutionPanel(Static):
    """Evolution status display."""
    
    total = reactive(0)
    successful = reactive(0)
    fitness = reactive(0.5)
    trend = reactive("→")
    last_mutation = reactive(None)
    
    def render(self) -> str:
        trend_color = "green" if self.trend == "↑" else "red" if self.trend == "↓" else "white"
        return f"""[bold]Evolution[/]
────────────────────
Mutations: {self.total} ({self.successful} kept)
Fitness: [{trend_color}]{self.fitness:.2f} {self.trend}[/]
Last: {self.last_mutation or 'none'}
"""


class SenterTUI(App):
    """Main TUI application."""
    
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
    }
    
    #chat-container {
        row-span: 2;
        border: solid green;
    }
    
    #chat-log {
        height: 1fr;
    }
    
    #chat-input {
        dock: bottom;
        height: 3;
    }
    
    #right-top {
        layout: vertical;
        border: solid cyan;
    }
    
    #right-bottom {
        layout: horizontal;
        border: solid yellow;
    }
    
    .panel {
        height: auto;
        padding: 1;
    }
    """
    
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+t", "add_task", "Add Task"),
    ]
    
    def __init__(self, genome_path: Path):
        super().__init__()
        self.genome_path = genome_path
        self.engine = None
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container(id="chat-container"):
            yield ChatPanel(id="chat-log", highlight=True, markup=True)
            yield Input(placeholder="Type message or /command...", id="chat-input")
        
        with Vertical(id="right-top"):
            yield AIStatePanel(id="ai-state", classes="panel")
            yield GoalsPanel(id="goals", classes="panel")
        
        with Horizontal(id="right-bottom"):
            yield TasksPanel(id="tasks", classes="panel")
            yield EvolutionPanel(id="evolution", classes="panel")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Initialize engine on mount."""
        from core.engine import Senter
        self.engine = Senter(self.genome_path)
        
        # Start update loop
        self.set_interval(2.0, self.update_panels)
        
        self.query_one("#chat-log", ChatPanel).add_message(
            "system", "Senter TUI ready. Type a message or use /commands."
        )
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input."""
        text = event.value.strip()
        if not text:
            return
        
        event.input.value = ""
        chat = self.query_one("#chat-log", ChatPanel)
        
        # Handle commands
        if text.startswith("/"):
            await self.handle_command(text, chat)
            return
        
        # Regular message
        chat.add_message("user", text)
        
        # Get response
        response = await self.engine.interact(text)
        chat.add_message("assistant", response.text)
        
        # Update AI state panel
        ai_state = self.query_one("#ai-state", AIStatePanel)
        ai_state.mode = response.ai_state.mode
        ai_state.focus = response.ai_state.focus or "none"
        ai_state.trust = response.ai_state.trust_level
    
    async def handle_command(self, cmd: str, chat: ChatPanel):
        """Handle slash commands."""
        parts = cmd.split()
        command = parts[0].lower()
        
        if command == "/help":
            chat.add_message("system", """Commands:
/status - System status
/goals - Active goals  
/task <type> <desc> - Add background task
/evolution - Evolution status
/quit - Exit""")
        
        elif command == "/goals":
            goals = self.engine.goal_detector.get_active_goals()
            if goals:
                for g in goals[:5]:
                    chat.add_message("system", f"Goal: {g.description} ({g.confidence:.0%})")
            else:
                chat.add_message("system", "No goals detected yet")
        
        elif command == "/task" and len(parts) >= 3:
            task_type = parts[1]
            desc = " ".join(parts[2:])
            chat.add_message("system", f"Task added: {task_type} - {desc}")
            # TODO: Connect to daemon task queue
        
        elif command == "/evolution":
            summary = self.engine.mutations.get_evolution_summary()
            chat.add_message("system", 
                f"Mutations: {summary['total']} ({summary['successful']} successful)")
        
        elif command == "/quit":
            await self.engine.shutdown()
            self.exit()
    
    async def update_panels(self) -> None:
        """Periodically update all panels."""
        if not self.engine:
            return
        
        # Update goals
        goals_panel = self.query_one("#goals", GoalsPanel)
        goals = self.engine.goal_detector.get_active_goals()
        goals_panel.goals = [{"desc": g.description, "conf": g.confidence} for g in goals]
        
        # Update evolution
        evo_panel = self.query_one("#evolution", EvolutionPanel)
        summary = self.engine.mutations.get_evolution_summary()
        evo_panel.total = summary.get("total", 0)
        evo_panel.successful = summary.get("successful", 0)


def run_tui(genome_path: Path = Path("genome.yaml")):
    """Run the TUI application."""
    app = SenterTUI(genome_path)
    app.run()


if __name__ == "__main__":
    run_tui()
```

B2. Add TUI command to senter.py

Add to argument parser:
```python
parser.add_argument('--tui', action='store_true', help='Rich TUI interface')
```

Add to main():
```python
elif args.tui:
    from interface.tui import run_tui
    run_tui(args.genome)
```

B3. Test TUI
```bash
python senter.py --tui
```

Verify:
- Chat panel accepts input and shows responses
- AI State updates after each message
- Goals panel shows detected goals
- Evolution panel shows mutation count

═══════════════════════════════════════════════════════════════════════════════
PHASE C: ACTIVITY MONITORING
═══════════════════════════════════════════════════════════════════════════════

C1. Create activity monitoring module

Create file: intelligence/activity.py

```python
"""
Activity Monitoring - Learns from computer activity.

Capabilities:
- Screen OCR: Extract text from screen
- App tracking: Which apps are active
- Context inference: What is the user working on?
- Goal suggestion: Infer goals from activity patterns
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class ActivitySnapshot:
    """A point-in-time capture of user activity."""
    timestamp: datetime
    active_app: str
    window_title: str
    screen_text: List[str]  # Key phrases extracted
    inferred_context: str  # coding, writing, browsing, etc.


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
            print(f"[ACTIVITY] Missing dependency: {e}")
            print("[ACTIVITY] Install: pip install pyautogui pytesseract pillow")
            return None
        except Exception as e:
            print(f"[ACTIVITY] Capture error: {e}")
            return None
    
    def get_active_window(self) -> Dict[str, str]:
        """Get currently active window info."""
        try:
            import subprocess
            import sys
            
            if sys.platform == 'darwin':  # macOS
                script = '''
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    set frontWindow to name of front window of first application process whose frontmost is true
                end tell
                return frontApp & "|" & frontWindow
                '''
                result = subprocess.run(
                    ['osascript', '-e', script],
                    capture_output=True, text=True
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
                    capture_output=True, text=True
                )
                return {'app': 'Unknown', 'window': result.stdout.strip()}
            
            else:  # Windows
                import ctypes
                user32 = ctypes.windll.user32
                h_wnd = user32.GetForegroundWindow()
                length = user32.GetWindowTextLengthW(h_wnd)
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(h_wnd, buf, length + 1)
                return {'app': 'Unknown', 'window': buf.value}
                
        except Exception as e:
            return {'app': 'Unknown', 'window': str(e)}


class ContextInferencer:
    """Infers work context from activity."""
    
    CONTEXT_PATTERNS = {
        'coding': {
            'apps': ['code', 'vscode', 'pycharm', 'intellij', 'vim', 'nvim', 'terminal', 'iterm'],
            'keywords': ['def ', 'class ', 'import ', 'function', 'const ', 'let ', 'var ']
        },
        'writing': {
            'apps': ['word', 'docs', 'notion', 'obsidian', 'pages', 'textedit'],
            'keywords': ['chapter', 'section', 'paragraph', 'draft', 'edit']
        },
        'research': {
            'apps': ['chrome', 'firefox', 'safari', 'edge', 'scholar'],
            'keywords': ['abstract', 'paper', 'study', 'research', 'doi', 'arxiv']
        },
        'communication': {
            'apps': ['slack', 'teams', 'discord', 'mail', 'outlook', 'gmail'],
            'keywords': ['inbox', 'sent', 'reply', 'message', 'channel']
        },
        'design': {
            'apps': ['figma', 'sketch', 'photoshop', 'illustrator', 'canva'],
            'keywords': ['layer', 'frame', 'component', 'prototype']
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
    
    def __init__(self, senter_engine, capture_interval: int = 60):
        self.engine = senter_engine
        self.capture_interval = capture_interval  # seconds
        self.screen = ScreenCapture()
        self.inferencer = ContextInferencer()
        
        self.running = False
        self.history: List[ActivitySnapshot] = []
        self.patterns: List[ActivityPattern] = []
        
        # Persist path
        self.data_path = Path("data/activity")
        self.data_path.mkdir(exist_ok=True)
    
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
                self.engine.goal_detector._create_or_update_goal(
                    description=suggestion,
                    evidence=f"Observed {frequency} activity snapshots in {context} context",
                    confidence=min(0.7, 0.3 + (frequency * 0.02))
                )
    
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
```

C2. Integrate with engine

Add to core/engine.py _init_intelligence_layer():
```python
from intelligence.activity import ActivityMonitor
self.activity = ActivityMonitor(self, capture_interval=60)
```

Add to start_background_tasks():
```python
asyncio.create_task(self.activity.start())
```

Add to shutdown():
```python
self.activity.stop()
```

C3. Add activity command to TUI/CLI
```python
elif command == '/activity':
    summary = self.engine.activity.get_activity_summary(hours=24)
    chat.add_message("system", f"Current context: {summary['current_context']}")
    chat.add_message("system", f"Top apps: {summary['top_apps']}")
```

C4. Install dependencies
```bash
pip install pyautogui pytesseract pillow
# macOS may need: brew install tesseract
```

C5. Test activity monitoring
```bash
python -c "
import asyncio
from intelligence.activity import ActivityMonitor, ScreenCapture

# Test screen capture
sc = ScreenCapture()
window = sc.get_active_window()
print(f'Active window: {window}')

# Test context inference
from intelligence.activity import ContextInferencer
ci = ContextInferencer()
context = ci.infer_context({'app': 'vscode', 'window': 'main.py', 'key_phrases': ['def test', 'import asyncio']})
print(f'Inferred context: {context}')
"
```

═══════════════════════════════════════════════════════════════════════════════
PHASE D: COMPREHENSIVE TEST SUITE
═══════════════════════════════════════════════════════════════════════════════

D1. Create vision validation tests

Create file: tests/test_vision.py

```python
"""
Vision Validation Tests

Tests that the complete Senter vision is achieved:
- 24/7 operation
- Learns from conversations
- Proactive intelligence
- Memory continuity
- Activity awareness
"""

import pytest
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import sys


class TestDaemonOperation:
    """Test 24/7 daemon operation."""
    
    @pytest.fixture
    def daemon_process(self):
        """Start daemon for testing."""
        proc = subprocess.Popen(
            [sys.executable, 'senter.py', '--daemon'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)  # Wait for startup
        yield proc
        proc.terminate()
        proc.wait()
    
    def test_daemon_starts(self, daemon_process):
        """Daemon process starts successfully."""
        assert daemon_process.poll() is None  # Still running
        
        # Check socket exists
        socket_path = Path("data/senter.sock")
        assert socket_path.exists()
    
    @pytest.mark.asyncio
    async def test_client_connects(self, daemon_process):
        """Client can connect to daemon."""
        from daemon.senter_client import SenterClient
        
        client = SenterClient()
        await client.connect()
        
        status = await client.status()
        assert status['status'] == 'ok'
        assert status['running'] == True
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_background_tasks_execute(self, daemon_process):
        """Background tasks execute while client disconnected."""
        from daemon.senter_client import SenterClient
        
        client = SenterClient()
        await client.connect()
        
        # Add task
        result = await client.add_task('summarize', 'Test summary task')
        task_id = result['task_id']
        
        await client.disconnect()
        
        # Wait for execution
        await asyncio.sleep(30)
        
        # Reconnect and check
        await client.connect()
        completed = await client.completed_tasks()
        
        task_ids = [t['id'] for t in completed['tasks']]
        assert task_id in task_ids
        
        await client.disconnect()


class TestLearningFromConversations:
    """Test that system learns from conversations."""
    
    @pytest.fixture
    def engine(self):
        from core.engine import Senter
        engine = Senter(Path('genome.yaml'))
        yield engine
        asyncio.get_event_loop().run_until_complete(engine.shutdown())
    
    @pytest.mark.asyncio
    async def test_goal_detection(self, engine):
        """Goals are detected from conversation patterns."""
        # Have conversations that imply goals
        await engine.interact("I want to learn machine learning")
        await engine.interact("I need to finish my thesis by March")
        await engine.interact("I'm trying to get better at Python")
        
        goals = engine.goal_detector.get_active_goals()
        
        assert len(goals) >= 2
        goal_texts = ' '.join(g.description.lower() for g in goals)
        assert 'learn' in goal_texts or 'machine learning' in goal_texts
    
    @pytest.mark.asyncio
    async def test_pattern_learning(self, engine):
        """System learns user patterns."""
        # Repeated similar queries
        for _ in range(5):
            await engine.interact("How do I use async/await in Python?")
        
        # Check procedural memory
        patterns = engine.memory.procedural.get_patterns('response')
        assert len(patterns) > 0
    
    @pytest.mark.asyncio
    async def test_preference_learning(self, engine):
        """System learns user preferences."""
        await engine.interact("I prefer concise responses")
        await engine.interact("Remember I like code examples")
        
        facts = engine.memory.semantic.search("prefer")
        assert len(facts) > 0


class TestProactiveIntelligence:
    """Test proactive suggestion system."""
    
    @pytest.fixture
    def engine(self):
        from core.engine import Senter
        engine = Senter(Path('genome.yaml'))
        yield engine
        asyncio.get_event_loop().run_until_complete(engine.shutdown())
    
    @pytest.mark.asyncio
    async def test_suggestions_at_high_trust(self, engine):
        """Proactive suggestions appear at high trust."""
        # Raise trust
        engine.trust.level = 0.85
        
        # Get suggestions
        suggestions = await engine.proactive.generate_suggestions()
        
        assert len(suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_no_suggestions_at_low_trust(self, engine):
        """No proactive suggestions at low trust."""
        engine.trust.level = 0.3
        
        suggestions = await engine.proactive.generate_suggestions()
        
        assert len(suggestions) == 0
    
    @pytest.mark.asyncio
    async def test_goal_based_suggestions(self, engine):
        """Suggestions relate to detected goals."""
        # Create a goal
        await engine.interact("I want to learn Spanish")
        
        engine.trust.level = 0.85
        suggestions = await engine.proactive.generate_suggestions()
        
        # At least one suggestion should relate to the goal
        suggestion_texts = ' '.join(s.get('title', '') + s.get('action', '') for s in suggestions)
        # May or may not contain "Spanish" depending on timing


class TestMemoryContinuity:
    """Test memory persistence across sessions."""
    
    @pytest.mark.asyncio
    async def test_semantic_memory_persists(self):
        """Semantic facts persist across restart."""
        from core.engine import Senter
        
        # Session 1: Store fact
        engine1 = Senter(Path('genome.yaml'))
        await engine1.interact("Remember my favorite color is blue")
        await engine1.shutdown()
        
        # Session 2: Recall
        engine2 = Senter(Path('genome.yaml'))
        facts = engine2.memory.semantic.search("favorite color")
        await engine2.shutdown()
        
        assert len(facts) > 0
        assert any('blue' in f['content'].lower() for f in facts)
    
    @pytest.mark.asyncio
    async def test_trust_persists(self):
        """Trust level persists across restart."""
        from core.engine import Senter
        
        # Session 1: Build trust
        engine1 = Senter(Path('genome.yaml'))
        for _ in range(10):
            await engine1.interact("Thanks, that's helpful!")
        trust1 = engine1.trust.level
        await engine1.shutdown()
        
        # Session 2: Check trust
        engine2 = Senter(Path('genome.yaml'))
        trust2 = engine2.trust.level
        await engine2.shutdown()
        
        assert trust2 >= trust1 - 0.05  # Allow small variance
    
    @pytest.mark.asyncio
    async def test_evolution_persists(self):
        """Evolution history persists."""
        from core.engine import Senter
        
        # Session 1: Trigger evolution
        engine1 = Senter(Path('genome.yaml'))
        for _ in range(20):
            await engine1.interact("This is frustrating!")
        mutations1 = engine1.mutations.get_evolution_summary()['total']
        await engine1.shutdown()
        
        # Session 2: Check history
        engine2 = Senter(Path('genome.yaml'))
        mutations2 = engine2.mutations.get_evolution_summary()['total']
        await engine2.shutdown()
        
        assert mutations2 >= mutations1


class TestEvolutionSystem:
    """Test that evolution actually improves the system."""
    
    @pytest.mark.asyncio
    async def test_genome_modification(self):
        """Evolution modifies genome.yaml."""
        import shutil
        
        # Backup original
        original = Path('genome.yaml')
        backup = Path('genome.yaml.test_backup')
        shutil.copy(original, backup)
        
        try:
            from core.engine import Senter
            engine = Senter(original)
            
            # Force low fitness interactions
            for _ in range(30):
                await engine.interact("This doesn't work! So frustrating!")
            
            await engine.shutdown()
            
            # Check if genome changed
            original_content = backup.read_text()
            new_content = original.read_text()
            
            # Should have changed (or at least backup exists)
            backup_dir = Path('data/genome_backups')
            assert backup_dir.exists()
            
        finally:
            # Restore original
            shutil.copy(backup, original)
            backup.unlink()
    
    @pytest.mark.asyncio
    async def test_fitness_improves(self):
        """Fitness trend is positive over time."""
        from core.engine import Senter
        
        engine = Senter(Path('genome.yaml'))
        
        fitness_scores = []
        for i in range(20):
            response = await engine.interact(f"Help me with task {i}")
            # Record fitness would be internal
            fitness_scores.append(engine.fitness.history[-1] if engine.fitness.history else 0.5)
        
        await engine.shutdown()
        
        # Check trend (simple: end > start)
        if len(fitness_scores) >= 10:
            early_avg = sum(fitness_scores[:5]) / 5
            late_avg = sum(fitness_scores[-5:]) / 5
            # Fitness should not significantly decrease
            assert late_avg >= early_avg - 0.1


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

D2. Create test runner script

Create file: scripts/validate_vision.py

```python
#!/usr/bin/env python3
"""
Vision Validation Script

Runs all validation tests and produces a report.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def main():
    print("=" * 60)
    print("SENTER 3.0 VISION VALIDATION")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Run pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", 
         "tests/test_vision.py", 
         "-v", "--tb=short",
         "--junit-xml=data/validation_report.xml"],
        cwd=Path(__file__).parent.parent
    )
    
    print("\n" + "=" * 60)
    if result.returncode == 0:
        print("✓ ALL VISION TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
```

D3. Run the tests
```bash
python scripts/validate_vision.py
```

═══════════════════════════════════════════════════════════════════════════════
PHASE E: PERFORMANCE & POLISH
═══════════════════════════════════════════════════════════════════════════════

E1. Add structured logging

Create file: utils/logging.py

```python
"""
Structured logging for Senter.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logs."""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(log_dir: Path = Path("data/logs"), level: str = "INFO"):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    
    # Console handler (human readable)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    ))
    root.addHandler(console)
    
    # File handler (JSON, rotating)
    file_handler = RotatingFileHandler(
        log_dir / "senter.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)
    
    # Separate error log
    error_handler = RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=5_000_000,
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root.addHandler(error_handler)
    
    return root


# Usage in engine.py:
# from utils.logging import setup_logging
# setup_logging(level="INFO")
# logger = logging.getLogger(__name__)
```

E2. Add error handling wrapper

Add to core/engine.py:

```python
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def handle_errors(func):
    """Decorator for graceful error handling."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ConnectionError as e:
            logger.error(f"Connection error: {e}", extra={'func': func.__name__})
            return Response(text="I'm having trouble connecting. Please check if the model is running.", ai_state=AIState())
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}")
            return Response(text="Something went wrong. Please try again.", ai_state=AIState())
    return wrapper

# Apply to interact():
@handle_errors
async def interact(self, input_text: str) -> Response:
    ...
```

E3. Add configuration validation

Create file: core/config_validator.py

```python
"""
Genome configuration validator.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import yaml


class ConfigValidator:
    """Validates genome.yaml configuration."""
    
    REQUIRED_SECTIONS = [
        'models',
        'memory',
        'coupling',
        'evolution'
    ]
    
    REQUIRED_MODEL_FIELDS = ['type']
    
    def __init__(self, genome_path: Path):
        self.genome_path = genome_path
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate configuration.
        
        Returns: (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Check file exists
        if not self.genome_path.exists():
            self.errors.append(f"Genome file not found: {self.genome_path}")
            return False, self.errors, self.warnings
        
        # Parse YAML
        try:
            with open(self.genome_path) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML: {e}")
            return False, self.errors, self.warnings
        
        # Check required sections
        for section in self.REQUIRED_SECTIONS:
            if section not in config:
                self.errors.append(f"Missing required section: {section}")
        
        # Validate models
        if 'models' in config:
            self._validate_models(config['models'])
        
        # Validate coupling
        if 'coupling' in config:
            self._validate_coupling(config['coupling'])
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_models(self, models: Dict):
        """Validate models configuration."""
        if 'primary' not in models:
            self.errors.append("Missing models.primary configuration")
            return
        
        primary = models['primary']
        if 'type' not in primary:
            self.errors.append("Missing models.primary.type")
        
        model_type = primary.get('type', '')
        if model_type == 'ollama':
            if 'model' not in primary:
                self.warnings.append("models.primary.model not set, will use default")
        elif model_type == 'gguf':
            if 'path' not in primary:
                self.errors.append("models.primary.path required for GGUF")
        elif model_type == 'openai':
            if 'model' not in primary:
                self.warnings.append("models.primary.model not set, will use gpt-4")
    
    def _validate_coupling(self, coupling: Dict):
        """Validate coupling configuration."""
        if 'trust' in coupling:
            trust = coupling['trust']
            initial = trust.get('initial', 0.5)
            if not 0 <= initial <= 1:
                self.errors.append(f"coupling.trust.initial must be 0-1, got {initial}")


def validate_config(genome_path: Path) -> bool:
    """Validate and report config issues."""
    validator = ConfigValidator(genome_path)
    is_valid, errors, warnings = validator.validate()
    
    if warnings:
        for w in warnings:
            print(f"⚠️  Warning: {w}")
    
    if errors:
        for e in errors:
            print(f"❌ Error: {e}")
        return False
    
    print("✓ Configuration valid")
    return True
```

E4. Add startup validation to senter.py

```python
# At start of main():
from core.config_validator import validate_config

if not validate_config(args.genome):
    print("Fix configuration errors before starting.")
    sys.exit(1)
```

E5. Performance profiling

Add to core/engine.py:

```python
import time
from contextlib import contextmanager

@contextmanager
def profile_section(name: str):
    """Profile a code section."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if elapsed > 0.5:  # Log slow operations
        logger.warning(f"Slow operation: {name} took {elapsed:.2f}s")

# Usage in interact():
async def interact(self, input_text: str) -> Response:
    with profile_section("total_interaction"):
        with profile_section("cognitive_state"):
            cognitive_state = self.human_model.infer_state(input_text)
        
        with profile_section("intent_parsing"):
            intent = await self._understand(input_text, cognitive_state)
        
        with profile_section("memory_retrieval"):
            memory_context = self.memory.retrieve(input_text)
        
        with profile_section("response_generation"):
            response_text = await self._compose(intent, context, mode, cognitive_state)
        
        # ... rest of method
```

E6. Final verification
```bash
# Run all tests
pytest tests/ -v

# Run vision validation
python scripts/validate_vision.py

# Check logs directory created
ls -la data/logs/

# Validate config
python -c "from core.config_validator import validate_config; validate_config(Path('genome.yaml'))"
```

═══════════════════════════════════════════════════════════════════════════════
FINAL VERIFICATION
═══════════════════════════════════════════════════════════════════════════════

After completing all phases, run this final check:

```bash
python -c "
import asyncio
from pathlib import Path
from datetime import datetime

async def final_verification():
    print('=' * 60)
    print('SENTER 3.0 FINAL VERIFICATION')
    print(f'Time: {datetime.now()}')
    print('=' * 60)
    
    checks = []
    
    # 1. Core engine
    print('\\n[1] Core Engine...')
    try:
        from core.engine import Senter
        s = Senter(Path('genome.yaml'))
        r = await s.interact('Hello')
        checks.append(('Core Engine', True, len(r.text) > 0))
        await s.shutdown()
    except Exception as e:
        checks.append(('Core Engine', False, str(e)))
    
    # 2. Goal Detection
    print('[2] Goal Detection...')
    try:
        from core.engine import Senter
        s = Senter(Path('genome.yaml'))
        await s.interact('I want to learn Spanish')
        goals = s.goal_detector.get_active_goals()
        checks.append(('Goal Detection', True, len(goals) > 0))
        await s.shutdown()
    except Exception as e:
        checks.append(('Goal Detection', False, str(e)))
    
    # 3. Proactive Suggestions
    print('[3] Proactive Suggestions...')
    try:
        from core.engine import Senter
        s = Senter(Path('genome.yaml'))
        s.trust.level = 0.85
        suggestions = await s.proactive.generate_suggestions()
        checks.append(('Proactive', True, 'Ready'))
        await s.shutdown()
    except Exception as e:
        checks.append(('Proactive', False, str(e)))
    
    # 4. Evolution
    print('[4] Evolution System...')
    try:
        from core.engine import Senter
        s = Senter(Path('genome.yaml'))
        summary = s.mutations.get_evolution_summary()
        checks.append(('Evolution', True, f'{summary[\"total\"]} mutations'))
        await s.shutdown()
    except Exception as e:
        checks.append(('Evolution', False, str(e)))
    
    # 5. Voice Interface
    print('[5] Voice Interface...')
    try:
        from interface.voice import VoiceInterface
        v = VoiceInterface()
        v.load()
        checks.append(('Voice', True, 'Whisper loaded'))
    except Exception as e:
        checks.append(('Voice', False, str(e)))
    
    # 6. Gaze Interface
    print('[6] Gaze Interface...')
    try:
        from interface.gaze import GazeDetector
        g = GazeDetector()
        g.load()
        checks.append(('Gaze', True, 'MediaPipe loaded'))
    except Exception as e:
        checks.append(('Gaze', False, str(e)))
    
    # 7. Activity Monitor
    print('[7] Activity Monitor...')
    try:
        from intelligence.activity import ActivityMonitor, ScreenCapture
        sc = ScreenCapture()
        window = sc.get_active_window()
        checks.append(('Activity', True, f'Window: {window[\"app\"]}'))
    except Exception as e:
        checks.append(('Activity', False, str(e)))
    
    # 8. TUI
    print('[8] TUI Interface...')
    try:
        from interface.tui import SenterTUI
        checks.append(('TUI', True, 'Module loads'))
    except Exception as e:
        checks.append(('TUI', False, str(e)))
    
    # Results
    print('\\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    
    passed = 0
    for name, success, detail in checks:
        icon = '✓' if success else '✗'
        print(f'{icon} {name}: {detail}')
        if success:
            passed += 1
    
    print(f'\\nPassed: {passed}/{len(checks)}')
    
    if passed == len(checks):
        print('\\n🎉 ALL SYSTEMS OPERATIONAL - SENTER 3.0 COMPLETE')
    else:
        print('\\n⚠️  Some systems need attention')

asyncio.run(final_verification())
"
```

═══════════════════════════════════════════════════════════════════════════════

Execute each phase in order. Show me the output of each step.
Do not proceed to the next phase until the current one passes.
```

---

This single comprehensive prompt will guide Claude Code through:

1. **Phase 0**: Ultimate daemon validation (proves 24/7 works)
2. **Phase A**: Voice & gaze testing (multimodal input)
3. **Phase B**: TUI interface (beautiful real-time display)
4. **Phase C**: Activity monitoring (learns from computer use)
5. **Phase D**: Test suite (ensures it all works together)
6. **Phase E**: Polish (logging, error handling, validation)

Each phase builds on the previous, and the final verification confirms everything is operational.
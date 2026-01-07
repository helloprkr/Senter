"""
Senter TUI - Rich terminal interface with real-time system state.

Panels:
+---------------------------------------------------------------------+
| SENTER 3.0                                    Trust: 0.72 | DIALOGUE |
+-------------------------------------+-------------------------------+
| Chat                                | AI State                      |
|                                     | -----------------------------||
| You: Hello                          | Mode: DIALOGUE                |
| Senter: Hi! How can I help?         | Focus: greeting               |
|                                     | Frustration: 0.0              |
| You: I'm working on my thesis       | Energy: 0.8                   |
| Senter: I can help with that...     |                               |
|                                     +-------------------------------+
|                                     | Goals                         |
|                                     | -----------------------------||
|                                     | - Learn ML (0.85)             |
|                                     | - Finish thesis (0.72)        |
+-------------------------------------+-------------------------------+
| Background Tasks                    | Evolution                     |
| ----------------------------------- | -----------------------------||
| [OK] research: AI safety (done)     | Mutations: 3 (2 kept)         |
| [..] summarize: daily (running)     | Fitness: 0.67 ^               |
| [ ] research: ML papers (pending)   | Last: threshold_change OK     |
+-------------------------------------+-------------------------------+
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Input, RichLog
from textual.reactive import reactive
from pathlib import Path


class ChatPanel(RichLog):
    """Scrollable chat history."""

    def add_message(self, role: str, text: str):
        if role == "user":
            self.write(f"[bold cyan]You:[/] {text}")
        elif role == "system":
            self.write(f"[dim]{text}[/]")
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
        trust_color = "green" if self.trust > 0.7 else "yellow" if self.trust > 0.5 else "red"
        return f"""[bold]AI State[/]
--------------------
Mode: [cyan]{self.mode}[/]
Focus: {self.focus}
Trust: [{trust_color}]{self.trust:.2f}[/]
Frustration: {self.frustration:.2f}
Energy: {self.energy:.2f}
"""


class GoalsPanel(Static):
    """Detected goals display."""

    goals = reactive([])

    def render(self) -> str:
        lines = ["[bold]Goals[/]", "--------------------"]
        goals_list = self.goals if isinstance(self.goals, list) else []
        if goals_list:
            for g in goals_list[:5]:
                desc = g.get("desc", "")[:25]
                conf = g.get("conf", 0)
                lines.append(f"- {desc} ({conf:.0%})")
        else:
            lines.append("[dim]No goals detected yet[/]")
        return "\n".join(lines)


class TasksPanel(Static):
    """Background tasks display."""

    pending = reactive([])
    current = reactive(None)
    completed = reactive([])

    def render(self) -> str:
        lines = ["[bold]Background Tasks[/]", "--------------------"]

        completed_list = self.completed if isinstance(self.completed, list) else []
        pending_list = self.pending if isinstance(self.pending, list) else []

        for t in completed_list[-3:]:
            lines.append(f"[green][OK][/] {t.get('type', '')}: {t.get('desc', '')[:20]}")

        if self.current:
            lines.append(f"[yellow][..][/] {self.current.get('type', '')}: {self.current.get('desc', '')[:20]}")

        for t in pending_list[:3]:
            lines.append(f"[dim][ ][/] {t.get('type', '')}: {t.get('desc', '')[:20]}")

        if not (completed_list or self.current or pending_list):
            lines.append("[dim]No tasks[/]")

        return "\n".join(lines)


class EvolutionPanel(Static):
    """Evolution status display."""

    total = reactive(0)
    successful = reactive(0)
    fitness = reactive(0.5)
    trend = reactive("->")
    last_mutation = reactive(None)

    def render(self) -> str:
        trend_color = "green" if self.trend == "^" else "red" if self.trend == "v" else "white"
        last = self.last_mutation or "none"
        return f"""[bold]Evolution[/]
--------------------
Mutations: {self.total} ({self.successful} kept)
Fitness: [{trend_color}]{self.fitness:.2f} {self.trend}[/]
Last: {last}
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
            yield ChatPanel(id="chat-log", highlight=True, markup=True, wrap=True)
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
        await self.engine.initialize()

        # Start update loop
        self.set_interval(2.0, self.update_panels)

        self.query_one("#chat-log", ChatPanel).add_message(
            "system", "Senter TUI ready. Type a message or use /commands."
        )
        self.query_one("#chat-log", ChatPanel).add_message(
            "system", "Commands: /status /goals /evolution /help /quit"
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
        try:
            response = await self.engine.interact(text)
            chat.add_message("assistant", response.text)

            # Update AI state panel
            ai_state = self.query_one("#ai-state", AIStatePanel)
            ai_state.mode = response.ai_state.mode
            ai_state.focus = response.ai_state.focus or "none"
            ai_state.trust = response.ai_state.trust_level
        except Exception as e:
            chat.add_message("system", f"Error: {e}")

    async def handle_command(self, cmd: str, chat: ChatPanel):
        """Handle slash commands."""
        parts = cmd.split()
        command = parts[0].lower()

        if command == "/help":
            chat.add_message(
                "system",
                """Commands:
/status - System status
/goals - Active goals
/task <type> <desc> - Add background task
/evolution - Evolution status
/quit - Exit""",
            )

        elif command == "/status":
            status = self.engine.get_status()
            chat.add_message("system", f"Version: {status['version']}")
            chat.add_message("system", f"Trust: {status['trust']['level']:.2f} ({status['trust']['trend']})")
            chat.add_message("system", f"Fitness: {status['fitness']['average']:.2f}")

        elif command == "/goals":
            if self.engine.goal_detector:
                goals = self.engine.goal_detector.get_active_goals()
                if goals:
                    for g in goals[:5]:
                        chat.add_message("system", f"Goal: {g.description} ({g.confidence:.0%})")
                else:
                    chat.add_message("system", "No goals detected yet")
            else:
                chat.add_message("system", "Goal tracking not available")

        elif command == "/task" and len(parts) >= 3:
            task_type = parts[1]
            desc = " ".join(parts[2:])
            chat.add_message("system", f"Task added: {task_type} - {desc}")
            # TODO: Connect to daemon task queue

        elif command == "/evolution":
            summary = self.engine.mutations.get_evolution_summary()
            chat.add_message(
                "system", f"Mutations: {summary['total']} ({summary['successful']} successful)"
            )
            if summary.get("recent_mutations"):
                for m in summary["recent_mutations"][:3]:
                    status_str = "kept" if m.get("success") else "rolled back"
                    chat.add_message("system", f"  - {m.get('type')}: {status_str}")

        elif command == "/quit":
            await self.engine.shutdown()
            self.exit()

        else:
            chat.add_message("system", f"Unknown command: {command}")
            chat.add_message("system", "Type /help for available commands")

    async def update_panels(self) -> None:
        """Periodically update all panels."""
        if not self.engine:
            return

        try:
            # Update goals
            if self.engine.goal_detector:
                goals_panel = self.query_one("#goals", GoalsPanel)
                goals = self.engine.goal_detector.get_active_goals()
                goals_panel.goals = [{"desc": g.description, "conf": g.confidence} for g in goals]

            # Update evolution
            evo_panel = self.query_one("#evolution", EvolutionPanel)
            summary = self.engine.mutations.get_evolution_summary()
            evo_panel.total = summary.get("total", 0)
            evo_panel.successful = summary.get("successful", 0)

            # Update fitness from tracker
            fitness_data = self.engine.fitness.to_dict()
            evo_panel.fitness = fitness_data.get("average", 0.5)
            evo_panel.trend = "^" if fitness_data.get("trend") == "improving" else "v" if fitness_data.get("trend") == "declining" else "->"
        except Exception:
            pass  # Ignore update errors


def run_tui(genome_path: Path = Path("genome.yaml")):
    """Run the TUI application."""
    app = SenterTUI(genome_path)
    app.run()


if __name__ == "__main__":
    run_tui()

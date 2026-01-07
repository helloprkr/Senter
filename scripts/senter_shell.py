#!/usr/bin/env python3
"""
Senter Interactive Shell

A rich interactive shell for communicating with the Senter daemon.
Supports commands, queries, and real-time status updates.
"""

import os
import sys
import time
import json
import readline
from pathlib import Path

# Setup path
SENTER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SENTER_ROOT))

from daemon.ipc_client import IPCClient

# Try to import rich for better output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class TermColors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        return f"{color}{text}{cls.RESET}"


class SimpleCLI:
    """Simple CLI without rich library"""

    def __init__(self):
        self.client = IPCClient()
        self.colors = TermColors()

    def print_header(self):
        """Print welcome header"""
        print()
        print(self.colors.colorize("=" * 50, TermColors.CYAN))
        print(self.colors.colorize("  SENTER INTERACTIVE SHELL", TermColors.BOLD + TermColors.CYAN))
        print(self.colors.colorize("=" * 50, TermColors.CYAN))
        print()
        print("Type queries or commands. Use /help for help.")
        print()

    def print_status(self):
        """Print daemon status"""
        result = self.client.status()

        if "error" in result and result.get("error"):
            print(self.colors.colorize(f"Error: {result['error']}", TermColors.RED))
            return

        # Header
        pid = result.get("pid", "?")
        uptime = result.get("uptime", 0)
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)

        print()
        print(self.colors.colorize(f"SENTER DAEMON (PID: {pid})", TermColors.BOLD + TermColors.GREEN))
        print(f"Uptime: {hours}h {mins}m")

        # Health
        overall = result.get("overall_health", False)
        health_color = TermColors.GREEN if overall else TermColors.YELLOW
        print(f"Health: {self.colors.colorize('OK' if overall else 'DEGRADED', health_color)}")

        # Components
        components = result.get("components", {})
        if components:
            print()
            print(self.colors.colorize("Components:", TermColors.BOLD))
            print("-" * 40)

            for name, info in components.items():
                alive = info.get("is_alive", False)
                pid_str = info.get("pid", "-")
                restarts = info.get("restart_count", 0)

                status_char = self.colors.colorize("✓", TermColors.GREEN) if alive else \
                              self.colors.colorize("✗", TermColors.RED)
                restart_str = f" [{restarts} restarts]" if restarts > 0 else ""

                print(f"  {status_char} {name:<20} PID: {pid_str}{restart_str}")

        print()

    def print_response(self, result: dict):
        """Print query response"""
        if "error" in result:
            print(self.colors.colorize(f"Error: {result['error']}", TermColors.RED))
            return

        response = result.get("response", "No response")
        latency = result.get("latency", 0)
        worker = result.get("worker", "unknown")

        print()
        print(response)
        print()
        print(self.colors.colorize(f"[{worker}, {latency:.2f}s]", TermColors.DIM))

    def print_goals(self):
        """Print active goals"""
        result = self.client.goals()

        if "error" in result:
            print(self.colors.colorize(f"Error: {result['error']}", TermColors.RED))
            return

        goals = result.get("goals", [])
        if not goals:
            print(self.colors.colorize("No active goals", TermColors.YELLOW))
            return

        print()
        print(self.colors.colorize("Active Goals:", TermColors.BOLD))
        print("-" * 50)

        for i, goal in enumerate(goals, 1):
            status = goal.get("status", "unknown")
            desc = goal.get("description", "No description")[:50]
            progress = goal.get("progress", 0)

            status_color = {
                "active": TermColors.GREEN,
                "pending": TermColors.YELLOW,
                "blocked": TermColors.RED
            }.get(status, TermColors.WHITE)

            print(f"{i}. [{self.colors.colorize(status, status_color)}] {desc}")
            if progress > 0:
                print(f"   Progress: {progress}%")

        print()

    def print_progress(self, hours: int = 24):
        """Print progress report"""
        result = self.client.progress(hours=hours)

        if "error" in result:
            print(self.colors.colorize(f"Error: {result['error']}", TermColors.RED))
            return

        summary = result.get("summary", "No progress data")
        print()
        print(self.colors.colorize(f"Progress Report (last {hours}h):", TermColors.BOLD))
        print("-" * 40)
        print(summary)
        print()

    def print_help(self):
        """Print help message"""
        help_text = """
Available Commands:
-------------------
  /status     Show daemon and component status
  /goals      List active goals
  /progress   Show activity digest
  /health     Quick health check
  /help       Show this help
  /quit       Exit shell

Tips:
-----
  • Just type to send a query
  • Goals are extracted automatically from queries
  • Use arrow keys for history
"""
        print(self.colors.colorize(help_text, TermColors.CYAN))


class RichCLI:
    """Rich CLI with colors and tables"""

    def __init__(self):
        self.client = IPCClient()
        self.console = Console()

    def print_header(self):
        """Print welcome header"""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan]SENTER INTERACTIVE SHELL[/bold cyan]\n\n"
            "Type queries or commands.\n"
            "Use [bold]/help[/bold] for available commands.",
            title="Welcome",
            border_style="cyan"
        ))
        self.console.print()

    def print_status(self):
        """Print daemon status with rich formatting"""
        result = self.client.status()

        if "error" in result and result.get("error"):
            self.console.print(f"[red]Error: {result['error']}[/red]")
            return

        # Header
        pid = result.get("pid", "?")
        uptime = result.get("uptime", 0)
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)

        self.console.print(Panel.fit(
            f"[bold green]SENTER DAEMON[/bold green]",
            subtitle=f"PID: {pid} | Uptime: {hours}h {mins}m"
        ))

        # Health
        overall = result.get("overall_health", False)
        health_style = "green" if overall else "yellow"
        self.console.print(f"Overall Health: [{health_style}]{'OK' if overall else 'DEGRADED'}[/{health_style}]")

        # Components table
        components = result.get("components", {})
        if components:
            table = Table(title="Components", box=box.ROUNDED)
            table.add_column("Component", style="cyan")
            table.add_column("Status", justify="center")
            table.add_column("PID", justify="right")
            table.add_column("Restarts", justify="right")

            for name, info in components.items():
                alive = info.get("is_alive", False)
                pid_str = str(info.get("pid", "-"))
                restarts = str(info.get("restart_count", 0))

                status = "[green]✓ Running[/green]" if alive else "[red]✗ Stopped[/red]"

                table.add_row(name, status, pid_str, restarts)

            self.console.print(table)

    def print_response(self, result: dict):
        """Print query response with rich formatting"""
        if "error" in result:
            self.console.print(f"[red]Error: {result['error']}[/red]")
            return

        response = result.get("response", "No response")
        latency = result.get("latency", 0)
        worker = result.get("worker", "unknown")

        self.console.print()
        self.console.print(Panel(response, border_style="green"))
        self.console.print(f"[dim]Worker: {worker} | Latency: {latency:.2f}s[/dim]")

    def print_goals(self):
        """Print active goals with rich formatting"""
        result = self.client.goals()

        if "error" in result:
            self.console.print(f"[red]Error: {result['error']}[/red]")
            return

        goals = result.get("goals", [])
        if not goals:
            self.console.print("[yellow]No active goals[/yellow]")
            return

        table = Table(title="Active Goals", box=box.ROUNDED)
        table.add_column("#", style="dim", width=3)
        table.add_column("Goal")
        table.add_column("Status")
        table.add_column("Progress", justify="right")

        for i, goal in enumerate(goals, 1):
            status = goal.get("status", "unknown")
            desc = goal.get("description", "No description")[:50]
            progress = goal.get("progress", 0)

            status_colors = {"active": "green", "pending": "yellow", "blocked": "red"}
            color = status_colors.get(status, "white")

            table.add_row(
                str(i),
                desc,
                f"[{color}]{status}[/{color}]",
                f"{progress}%"
            )

        self.console.print(table)

    def print_progress(self, hours: int = 24):
        """Print progress report"""
        result = self.client.progress(hours=hours)

        if "error" in result:
            self.console.print(f"[red]Error: {result['error']}[/red]")
            return

        summary = result.get("summary", "No progress data")
        self.console.print(Panel(summary, title=f"Activity Digest (last {hours}h)"))

    def print_help(self):
        """Print help message"""
        help_text = """
[bold]Available Commands:[/bold]

  [cyan]/status[/cyan]     Show daemon and component status
  [cyan]/goals[/cyan]      List active goals
  [cyan]/progress[/cyan]   Show activity digest
  [cyan]/health[/cyan]     Quick health check
  [cyan]/help[/cyan]       Show this help
  [cyan]/quit[/cyan]       Exit shell

[bold]Tips:[/bold]
  • Just type to send a query
  • Goals are extracted automatically from queries
  • Use arrow keys for history
"""
        self.console.print(Panel(help_text, title="Help"))


def run_shell():
    """Run the interactive shell"""
    # Choose CLI based on rich availability
    if RICH_AVAILABLE:
        cli = RichCLI()
    else:
        cli = SimpleCLI()

    client = cli.client

    # Check daemon
    if not client.is_daemon_running():
        if RICH_AVAILABLE:
            Console().print("[red]Daemon is not running. Start it first:[/red]")
            Console().print("  python3 scripts/senter_ctl.py start")
        else:
            print("Daemon is not running. Start it first:")
            print("  python3 scripts/senter_ctl.py start")
        return

    # Setup history
    history_file = Path.home() / ".senter_history"
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass

    # Print header
    cli.print_header()

    # Command handlers
    commands = {
        "/status": lambda: cli.print_status(),
        "/goals": lambda: cli.print_goals(),
        "/progress": lambda: cli.print_progress(),
        "/health": lambda: print("Healthy!" if client.health().get("healthy") else "Not healthy"),
        "/help": lambda: cli.print_help(),
        "/quit": lambda: sys.exit(0),
        "/exit": lambda: sys.exit(0),
    }

    # Main loop
    while True:
        try:
            # Get current status for prompt
            status = client.status()
            health = "+" if status.get("overall_health") else "-"

            # Get input
            user_input = input(f"[{health}] > ").strip()

            if not user_input:
                continue

            # Check for command
            if user_input.startswith("/"):
                cmd = user_input.split()[0].lower()
                if cmd in commands:
                    commands[cmd]()
                else:
                    print(f"Unknown command: {cmd}. Use /help for help.")
            else:
                # Send as query
                if RICH_AVAILABLE:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=Console(),
                        transient=True
                    ) as progress:
                        progress.add_task("Processing...", total=None)
                        result = client.query(user_input)
                else:
                    print("Processing...")
                    result = client.query(user_input)

                cli.print_response(result)

        except KeyboardInterrupt:
            print("\nUse /quit to exit")
        except EOFError:
            break

    # Save history
    try:
        readline.write_history_file(history_file)
    except:
        pass

    print("\nGoodbye!")


if __name__ == "__main__":
    run_shell()

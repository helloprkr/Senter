#!/usr/bin/env python3
"""
Senter Daemon Control Script

Commands:
    senter_ctl.py start    - Start the daemon
    senter_ctl.py stop     - Stop the daemon
    senter_ctl.py status   - Show daemon status
    senter_ctl.py restart  - Restart the daemon
    senter_ctl.py logs     - View daemon logs
    senter_ctl.py config   - Edit configuration
"""

import os
import sys
import signal
import time
import json
import subprocess
from pathlib import Path
from typing import Dict

# Setup path
senter_root = Path(__file__).parent.parent
sys.path.insert(0, str(senter_root))

# Files
PID_FILE = senter_root / "data" / "senter.pid"
LOG_FILE = senter_root / "data" / "daemon.log"
CONFIG_FILE = senter_root / "config" / "daemon_config.json"


def get_pid() -> int:
    """Get daemon PID if running"""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)
        return pid
    except (ProcessLookupError, ValueError, OSError):
        return None


def is_running() -> bool:
    """Check if daemon is running"""
    return get_pid() is not None


def start_daemon(foreground: bool = False):
    """Start the daemon"""
    if is_running():
        print("Senter daemon is already running")
        return

    print("Starting Senter daemon...")

    if foreground:
        # Run in foreground
        from daemon.senter_daemon import SenterDaemon
        daemon = SenterDaemon()
        daemon.start(foreground=True)
    else:
        # Run in background
        cmd = [sys.executable, "-m", "daemon.senter_daemon", "start"]
        process = subprocess.Popen(
            cmd,
            cwd=str(senter_root),
            stdout=open(senter_root / "data" / "daemon_stdout.log", "a"),
            stderr=open(senter_root / "data" / "daemon_stderr.log", "a"),
            start_new_session=True
        )

        # Wait a moment and check
        time.sleep(2)
        if is_running():
            print(f"Senter daemon started (PID: {get_pid()})")
        else:
            print("Failed to start daemon. Check logs.")


def stop_daemon():
    """Stop the daemon"""
    pid = get_pid()
    if not pid:
        print("Senter daemon is not running")
        return

    print(f"Stopping Senter daemon (PID: {pid})...")

    try:
        os.kill(pid, signal.SIGTERM)

        # Wait for process to stop
        for _ in range(10):
            time.sleep(0.5)
            if not is_running():
                print("Senter daemon stopped")
                return

        # Force kill if still running
        os.kill(pid, signal.SIGKILL)
        print("Senter daemon force killed")

    except ProcessLookupError:
        print("Senter daemon already stopped")
        if PID_FILE.exists():
            PID_FILE.unlink()


def show_status():
    """Show daemon status"""
    pid = get_pid()

    if not pid:
        print("Senter daemon is not running")
        return

    print(f"Senter daemon is running (PID: {pid})")

    # Try to get component status via IPC
    try:
        from daemon.ipc_client import IPCClient
        client = IPCClient()
        result = client.status()

        if "error" in result:
            print(f"\nCould not get status: {result['error']}")
            return

        # Show uptime
        uptime = result.get("uptime", 0)
        if uptime > 0:
            hours = int(uptime // 3600)
            mins = int((uptime % 3600) // 60)
            print(f"Uptime: {hours}h {mins}m")

        # Show health
        overall = result.get("overall_health", False)
        print(f"\nOverall health: {'OK' if overall else 'DEGRADED'}")

        # Show components
        components = result.get("components", {})
        if components:
            print("\nComponents:")
            for name, info in components.items():
                alive = "✓" if info.get("is_alive") else "✗"
                pid_str = f" (PID: {info.get('pid')})" if info.get('pid') else ""
                restarts = info.get("restart_count", 0)
                restart_str = f" [{restarts} restarts]" if restarts > 0 else ""
                print(f"  {alive} {name}{pid_str}{restart_str}")

        # Show queue sizes
        queue_sizes = result.get("queue_sizes", {})
        if queue_sizes and any(v > 0 for v in queue_sizes.values()):
            print("\nQueue sizes:")
            for name, size in queue_sizes.items():
                if size > 0:
                    print(f"  {name}: {size}")

    except Exception as e:
        print(f"\nCould not get detailed status: {e}")


def view_logs(lines: int = 50, follow: bool = False):
    """View daemon logs"""
    if not LOG_FILE.exists():
        print("No logs found")
        return

    if follow:
        # Use tail -f
        subprocess.run(["tail", "-f", str(LOG_FILE)])
    else:
        # Show last N lines
        subprocess.run(["tail", f"-{lines}", str(LOG_FILE)])


def edit_config():
    """Open config in editor"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not CONFIG_FILE.exists():
        # Create default config
        default = {
            "components": {
                "model_workers": {"enabled": True},
                "audio_pipeline": {"enabled": False},
                "gaze_detection": {"enabled": False},
                "task_engine": {"enabled": True},
                "scheduler": {"enabled": True},
                "reporter": {"enabled": True},
                "learning": {"enabled": True}
            }
        }
        CONFIG_FILE.write_text(json.dumps(default, indent=2))

    # Open in editor
    editor = os.environ.get("EDITOR", "nano")
    subprocess.run([editor, str(CONFIG_FILE)])


def show_progress(hours: int = 24):
    """Show progress report"""
    try:
        sys.path.insert(0, str(senter_root / "reporter"))
        from reporter.progress_reporter import ActivityLog, DigestGenerator

        log_dir = senter_root / "data" / "progress" / "activity"
        activity_log = ActivityLog(log_dir)
        digest_gen = DigestGenerator(activity_log)

        since = time.time() - (hours * 3600)
        summary = activity_log.get_summary(since=since)

        print(f"\n📊 Progress Report (last {hours}h)")
        print("=" * 40)
        print(f"Total activities: {summary['total_activities']}")

        if summary["by_type"]:
            print("\nBreakdown:")
            for activity_type, count in sorted(summary["by_type"].items(),
                                               key=lambda x: x[1], reverse=True):
                print(f"  • {activity_type}: {count}")

        if summary["recent_goals"]:
            print("\nRecent completions:")
            for goal in summary["recent_goals"]:
                desc = goal.get("description", "Task")[:60]
                print(f"  ✓ {desc}")

    except Exception as e:
        print(f"Error getting progress: {e}")


def create_goal(description: str):
    """Create a new goal"""
    if not is_running():
        print("Senter daemon is not running. Start it first.")
        return

    try:
        from daemon.ipc_client import IPCClient
        client = IPCClient()

        # Use query to create goal (the model can interpret intent)
        result = client.query(f"Create a goal: {description}")

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Goal created: {description}")
            if result.get("response"):
                print(f"\n{result['response']}")

    except Exception as e:
        print(f"Error creating goal: {e}")


def send_query(text: str):
    """Send a query to the daemon"""
    if not is_running():
        print("Senter daemon is not running. Start it first.")
        return

    try:
        from daemon.ipc_client import IPCClient
        client = IPCClient()

        print("Sending query...")
        result = client.query(text)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            response = result.get("response", "No response")
            latency = result.get("latency", 0)
            worker = result.get("worker", "unknown")

            print(f"\n{response}")
            print(f"\n[Worker: {worker}, Latency: {latency:.2f}s]")

    except Exception as e:
        print(f"Error sending query: {e}")


def show_report(hours: int = 24):
    """Show activity report - what did Senter accomplish (US-006)"""
    if not is_running():
        print("Senter daemon is not running. Start it first.")
        return

    try:
        from daemon.ipc_client import IPCClient
        from datetime import datetime

        client = IPCClient()
        report = client.activity_report(hours=hours)

        if "error" in report:
            print(f"Error: {report['error']}")
            return

        # Header
        print(f"\n{'='*60}")
        print(f"  SENTER ACTIVITY REPORT - Last {hours} hours")
        print(f"{'='*60}")
        print(f"  Period: {report.get('since', 'N/A')[:16]} to {report.get('until', 'N/A')[:16]}")
        print()

        # Time Stats Summary
        stats = report.get("time_stats", {})
        tasks_count = stats.get("tasks_completed_count", 0)
        research_count = stats.get("research_completed_count", 0)
        total_activities = stats.get("total_activities", 0)
        time_spent = stats.get("total_task_time_seconds", 0)

        print(f"  SUMMARY:")
        print(f"    Tasks completed:     {tasks_count}")
        print(f"    Research completed:  {research_count}")
        print(f"    Total activities:    {total_activities}")
        print(f"    Processing time:     {time_spent:.1f}s")
        print()

        # Completed Tasks
        tasks = report.get("tasks_completed", [])
        if tasks:
            print(f"  COMPLETED TASKS ({len(tasks)}):")
            for task in tasks[:10]:  # Show max 10
                desc = task.get("description", "Unknown")[:50]
                worker = task.get("worker", "?")
                latency = task.get("latency_ms", 0) / 1000
                print(f"    - {desc}")
                print(f"      Worker: {worker}, Time: {latency:.1f}s")
            if len(tasks) > 10:
                print(f"    ... and {len(tasks) - 10} more")
            print()

        # Research Done
        research = report.get("research_done", [])
        if research:
            print(f"  RESEARCH COMPLETED ({len(research)}):")
            for r in research[:5]:  # Show max 5
                topic = r.get("topic", "unknown")
                desc = r.get("description", "")[:50]
                print(f"    - [{topic}] {desc}")
            if len(research) > 5:
                print(f"    ... and {len(research) - 5} more")
            print()

        # Activity Breakdown
        activity_summary = report.get("activity_summary", {})
        by_type = activity_summary.get("by_type", {})
        if by_type:
            print(f"  ACTIVITY BREAKDOWN:")
            for activity_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                print(f"    {activity_type}: {count}")
            print()

        print(f"{'='*60}")

    except Exception as e:
        print(f"Error getting report: {e}")
        import traceback
        traceback.print_exc()


def show_events(hours: int = 24, event_type: str = None, limit: int = 50):
    """Show user interaction events (US-008)"""
    if not is_running():
        print("Senter daemon is not running. Start it first.")
        return

    try:
        from daemon.ipc_client import IPCClient
        from datetime import datetime

        client = IPCClient()
        result = client.get_events(hours=hours, event_type=event_type, limit=limit)

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        events = result.get("events", [])
        counts = result.get("event_counts", {})
        total = result.get("total_events", 0)

        # Header
        print(f"\n{'='*60}")
        print(f"  USER INTERACTION EVENTS - Last {hours} hours")
        print(f"{'='*60}")

        # Summary
        print(f"\n  EVENT COUNTS:")
        for etype, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {etype}: {count}")
        print(f"\n  Total events in database: {total}")

        # Events list
        if events:
            print(f"\n  RECENT EVENTS ({len(events)} shown):")
            print(f"  {'-'*56}")
            for event in events[:20]:  # Show max 20
                etype = event.get("event_type", "?")
                ts = event.get("datetime", "")[:16]
                ctx = event.get("context", {})
                meta = event.get("metadata", {})

                if etype == "query":
                    query = ctx.get("query", "")[:40]
                    topic = meta.get("topic", "?")
                    print(f"  [{ts}] QUERY ({topic}): {query}...")
                elif etype == "response":
                    latency = meta.get("latency_ms", 0)
                    topic = meta.get("topic", "?")
                    print(f"  [{ts}] RESPONSE ({topic}): {latency}ms")
                else:
                    print(f"  [{ts}] {etype.upper()}")

            if len(events) > 20:
                print(f"\n  ... and {len(events) - 20} more")
        else:
            print(f"\n  No events found in the last {hours} hours")

        print(f"\n{'='*60}")

    except Exception as e:
        print(f"Error getting events: {e}")
        import traceback
        traceback.print_exc()


def show_goals():
    """Show active goals"""
    if not is_running():
        print("Senter daemon is not running. Start it first.")
        return

    try:
        from daemon.ipc_client import IPCClient
        client = IPCClient()

        result = client.goals()

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        goals = result.get("goals", [])
        if not goals:
            print("No active goals")
            return

        print("\nActive Goals:")
        print("-" * 50)
        for i, goal in enumerate(goals, 1):
            status = goal.get("status", "unknown")
            desc = goal.get("description", "No description")[:60]
            progress = goal.get("progress", 0)
            print(f"{i}. [{status}] {desc}")
            if progress > 0:
                print(f"   Progress: {progress}%")

    except Exception as e:
        print(f"Error getting goals: {e}")


def show_mcp_servers():
    """Show configured MCP servers (MCP-004)"""
    try:
        from mcp.mcp_client import MCPClient
        client = MCPClient(senter_root)

        print(f"\n{'='*60}")
        print(f"  MCP SERVER CONFIGURATION")
        print(f"{'='*60}")

        configs = client.server_configs
        if not configs:
            print("\n  No MCP servers configured.")
            print(f"  Edit {client.config_path} to add servers.")
            print(f"\n{'='*60}")
            return

        print(f"\n  Configured Servers ({len(configs)}):")
        print(f"  {'-'*56}")

        for name, config in configs.items():
            status = "ENABLED" if config.enabled else "disabled"
            transport = config.transport.upper()

            print(f"\n  [{status}] {name}")
            print(f"    Transport: {transport}")

            if config.transport == "stdio":
                cmd = f"{config.command} {' '.join(config.args)}"
                print(f"    Command:   {cmd[:50]}{'...' if len(cmd) > 50 else ''}")
            elif config.transport == "http":
                print(f"    URL:       {config.url}")

            if config.env:
                env_keys = list(config.env.keys())
                print(f"    Env vars:  {', '.join(env_keys)}")

        print(f"\n  {'-'*56}")
        print(f"  Config file: {client.config_path}")
        print(f"\n{'='*60}")

    except Exception as e:
        print(f"Error loading MCP config: {e}")
        import traceback
        traceback.print_exc()


def list_mcp_tools():
    """List tools from connected MCP servers (MCP-004)"""
    try:
        from mcp.mcp_client import MCPClient
        client = MCPClient(senter_root)

        print(f"\n{'='*60}")
        print(f"  MCP TOOLS")
        print(f"{'='*60}")

        # Try to connect to enabled servers
        enabled_servers = [n for n, c in client.server_configs.items() if c.enabled]

        if not enabled_servers:
            print("\n  No MCP servers enabled.")
            print("  Enable servers in config/mcp_servers.json")
            print(f"\n{'='*60}")
            return

        print(f"\n  Connecting to {len(enabled_servers)} enabled server(s)...")

        client.connect_all()

        tools = client.list_tools()

        if not tools:
            print("\n  No tools discovered from connected servers.")
        else:
            print(f"\n  Discovered Tools ({len(tools)}):")
            print(f"  {'-'*56}")

            # Group by server
            by_server: Dict[str, list] = {}
            for tool in tools:
                by_server.setdefault(tool.server_name, []).append(tool)

            for server, server_tools in by_server.items():
                print(f"\n  [{server}]")
                for tool in server_tools:
                    print(f"    - {tool.name}")
                    if tool.description:
                        desc = tool.description[:50]
                        print(f"      {desc}{'...' if len(tool.description) > 50 else ''}")

        client.disconnect_all()
        print(f"\n{'='*60}")

    except Exception as e:
        print(f"Error listing MCP tools: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Senter Daemon Control")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # start
    start_parser = subparsers.add_parser("start", help="Start the daemon")
    start_parser.add_argument("-f", "--foreground", action="store_true",
                              help="Run in foreground")

    # stop
    subparsers.add_parser("stop", help="Stop the daemon")

    # status
    subparsers.add_parser("status", help="Show daemon status")

    # restart
    subparsers.add_parser("restart", help="Restart the daemon")

    # logs
    logs_parser = subparsers.add_parser("logs", help="View logs")
    logs_parser.add_argument("-n", "--lines", type=int, default=50,
                            help="Number of lines to show")
    logs_parser.add_argument("-f", "--follow", action="store_true",
                            help="Follow log output")

    # config
    subparsers.add_parser("config", help="Edit configuration")

    # progress
    progress_parser = subparsers.add_parser("progress", help="Show progress report")
    progress_parser.add_argument("-H", "--hours", type=int, default=24,
                                help="Hours to look back")

    # report (US-006)
    report_parser = subparsers.add_parser("report", help="Show activity report - what did Senter do")
    report_parser.add_argument("-H", "--hours", type=int, default=24,
                              help="Hours to look back (default: 24)")

    # events (US-008)
    events_parser = subparsers.add_parser("events", help="Show user interaction events")
    events_parser.add_argument("-H", "--hours", type=int, default=24,
                              help="Hours to look back (default: 24)")
    events_parser.add_argument("-t", "--type", type=str, dest="event_type",
                              help="Filter by event type (query, response)")
    events_parser.add_argument("-n", "--limit", type=int, default=50,
                              help="Maximum events to show (default: 50)")

    # goal
    goal_parser = subparsers.add_parser("goal", help="Create a new goal")
    goal_parser.add_argument("description", nargs="+", help="Goal description")

    # goals (list)
    subparsers.add_parser("goals", help="List active goals")

    # query
    query_parser = subparsers.add_parser("query", help="Send a query to the model")
    query_parser.add_argument("text", nargs="+", help="Query text")

    # shell
    subparsers.add_parser("shell", help="Start interactive shell")

    # mcp (MCP-004)
    mcp_parser = subparsers.add_parser("mcp", help="MCP server management")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP commands")
    mcp_subparsers.add_parser("list", help="List configured MCP servers")
    mcp_subparsers.add_parser("tools", help="List tools from connected MCP servers")

    args = parser.parse_args()

    if args.command == "start":
        start_daemon(foreground=args.foreground)
    elif args.command == "stop":
        stop_daemon()
    elif args.command == "status":
        show_status()
    elif args.command == "restart":
        stop_daemon()
        time.sleep(1)
        start_daemon()
    elif args.command == "logs":
        view_logs(lines=args.lines, follow=args.follow)
    elif args.command == "config":
        edit_config()
    elif args.command == "progress":
        show_progress(hours=args.hours)
    elif args.command == "report":
        show_report(hours=args.hours)
    elif args.command == "events":
        show_events(hours=args.hours, event_type=args.event_type, limit=args.limit)
    elif args.command == "goal":
        create_goal(" ".join(args.description))
    elif args.command == "goals":
        show_goals()
    elif args.command == "query":
        send_query(" ".join(args.text))
    elif args.command == "shell":
        from scripts.senter_shell import run_shell
        run_shell()
    elif args.command == "mcp":
        if args.mcp_command == "list":
            show_mcp_servers()
        elif args.mcp_command == "tools":
            list_mcp_tools()
        else:
            mcp_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

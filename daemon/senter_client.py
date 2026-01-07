"""
CLI client that connects to the Senter daemon.

Usage:
    python -m daemon.senter_client           # Interactive mode
    python -m daemon.senter_client --status  # Show status
    python -m daemon.senter_client --shutdown # Stop daemon
"""

from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict


class SenterClient:
    """Client that connects to running daemon."""

    def __init__(self, socket_path: Optional[Path] = None):
        self.socket_path = socket_path or Path("data/senter.sock")
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> None:
        """Connect to daemon."""
        if not self.socket_path.exists():
            raise RuntimeError(
                f"Daemon not running. Socket not found: {self.socket_path}\n"
                "Start with: python -m daemon.senter_daemon"
            )

        self.reader, self.writer = await asyncio.open_unix_connection(
            path=str(self.socket_path)
        )

    async def disconnect(self) -> None:
        """Disconnect from daemon."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()

    async def send(self, request: Dict) -> Dict:
        """Send request and get response."""
        if not self.writer:
            raise RuntimeError("Not connected")

        self.writer.write(json.dumps(request).encode() + b"\n")
        await self.writer.drain()

        data = await self.reader.readline()
        return json.loads(data.decode())

    async def interact(self, input_text: str) -> Dict:
        """Send interaction to daemon."""
        return await self.send({"action": "interact", "input": input_text})

    async def add_task(
        self, task_type: str, description: str, priority: str = "NORMAL", **params
    ) -> Dict:
        """Add background task."""
        return await self.send(
            {
                "action": "add_task",
                "task_type": task_type,
                "description": description,
                "priority": priority,
                "parameters": params,
            }
        )

    async def status(self) -> Dict:
        """Get daemon status."""
        return await self.send({"action": "status"})

    async def completed_tasks(self) -> Dict:
        """Get completed background tasks."""
        return await self.send({"action": "completed_tasks"})

    async def list_tasks(self) -> Dict:
        """Get pending tasks."""
        return await self.send({"action": "list_tasks"})

    async def shutdown(self) -> Dict:
        """Request daemon shutdown."""
        return await self.send({"action": "shutdown"})


async def interactive_mode(client: SenterClient) -> None:
    """Run interactive chat mode."""
    # Show what daemon did while away
    completed = await client.completed_tasks()
    if completed["tasks"]:
        print("\nWhile you were away, I completed:")
        for task in completed["tasks"]:
            print(f"  - {task['desc']}")
            if task["result"]:
                result_preview = task["result"][:100]
                if len(task["result"]) > 100:
                    result_preview += "..."
                print(f"    Result: {result_preview}")
        print()

    # Show status
    status = await client.status()
    print(f"Pending tasks: {status['pending_tasks']}")
    if status["current_task"]:
        print(f"Currently working on: {status['current_task']}")
    print()

    # Interactive loop
    print("Senter 3.0 (Connected to daemon)")
    print("Commands:")
    print("  /task <type> <description> - Add background task (research, remind, summarize)")
    print("  /tasks                     - List pending tasks")
    print("  /status                    - Show daemon status")
    print("  /completed                 - Show completed tasks")
    print("  quit                       - Exit (daemon keeps running)")
    print()

    while True:
        try:
            user_input = input("[daemon] You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Disconnecting (daemon still running)...")
                break

            if user_input.startswith("/task "):
                # Add background task: /task research Latest AI news
                parts = user_input[6:].split(" ", 1)
                if len(parts) == 2:
                    task_type, desc = parts
                    result = await client.add_task(task_type, desc)
                    if result["status"] == "ok":
                        print(f"  Task added: {result.get('task_id')}")
                    else:
                        print(f"  Error: {result.get('message')}")
                else:
                    print("  Usage: /task <type> <description>")
                    print("  Types: research, remind, summarize, organize")
                continue

            if user_input == "/tasks":
                tasks = await client.list_tasks()
                if tasks["tasks"]:
                    print("  Pending tasks:")
                    for t in tasks["tasks"]:
                        print(f"    - [{t['type']}] {t['desc']}")
                else:
                    print("  No pending tasks")
                continue

            if user_input == "/status":
                status = await client.status()
                print(f"  Running: {status['running']}")
                print(f"  Pending tasks: {status['pending_tasks']}")
                print(f"  Current task: {status['current_task'] or 'None'}")
                print(f"  Completed: {status['completed_tasks']}")
                print(f"  Trust level: {status['trust_level']:.2f}")
                print(f"  Memory episodes: {status['memory_episodes']}")
                continue

            if user_input == "/completed":
                completed = await client.completed_tasks()
                if completed["tasks"]:
                    print("  Recently completed:")
                    for t in completed["tasks"]:
                        print(f"    - [{t['type']}] {t['desc']}")
                        if t["result"]:
                            print(f"      Result: {t['result'][:80]}...")
                else:
                    print("  No completed tasks")
                continue

            if user_input == "/shutdown":
                result = await client.shutdown()
                print(f"  {result.get('message')}")
                break

            # Regular interaction
            response = await client.interact(user_input)

            if response["status"] == "ok":
                ai_state = response["ai_state"]
                print(
                    f"\n[AI State: Mode={ai_state['mode']}, Trust={ai_state['trust']:.2f}]"
                )
                print(f"Senter: {response['response']}\n")
            else:
                print(f"Error: {response.get('message')}")

        except KeyboardInterrupt:
            print("\nDisconnecting...")
            break
        except EOFError:
            break


async def main():
    args = sys.argv[1:]

    # Find socket path
    socket_path = Path("data/senter.sock")
    if not socket_path.exists():
        # Try in current directory
        alt_path = Path.cwd() / "data" / "senter.sock"
        if alt_path.exists():
            socket_path = alt_path

    client = SenterClient(socket_path)

    try:
        await client.connect()

        if "--status" in args:
            status = await client.status()
            print(f"Running: {status['running']}")
            print(f"Pending tasks: {status['pending_tasks']}")
            print(f"Current task: {status['current_task'] or 'None'}")
            print(f"Completed: {status['completed_tasks']}")
            print(f"Trust: {status['trust_level']:.2f}")
        elif "--shutdown" in args:
            result = await client.shutdown()
            print(result.get("message"))
        elif "--completed" in args:
            completed = await client.completed_tasks()
            for t in completed["tasks"]:
                print(f"[{t['type']}] {t['desc']}")
                if t["result"]:
                    print(f"  {t['result']}")
        else:
            await interactive_mode(client)

    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

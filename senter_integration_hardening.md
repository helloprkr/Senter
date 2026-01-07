# SENTER: Integration & Hardening Prompts

**Current State**: All 8 components implemented, tests passing  
**Goal**: Make it bulletproof and actually usable  
**Focus**: Integration, error handling, real-world testing

---

## THE GAP NOW

| Have | Need |
|------|------|
| Components exist individually | Components work together seamlessly |
| Happy path works | Edge cases handled |
| Daemon starts | Daemon recovers from failures |
| Tests pass in isolation | End-to-end scenarios work |
| CLI works | CLI shows what daemon is doing |

---

## PROMPT SET 1: END-TO-END INTEGRATION

### Prompt 1A: Full Flow Integration Test

```markdown
# Task: Create End-to-End Integration Test Suite

## Goal
Test complete flows that exercise multiple components together:
- User query → Router → Worker → Response → Learning → Memory
- Goal creation → Task Engine → Executor → Reporter
- Scheduled job → Task Engine → Worker → Progress

## Create: `tests/test_e2e.py`

```python
"""
End-to-end integration tests.

These tests start the full daemon and verify complete flows.
"""

import time
import json
import subprocess
import sys
from pathlib import Path
from multiprocessing import Process, Queue

# Test scenarios to implement:

def test_query_to_response_flow():
    """
    1. Start daemon in test mode
    2. Send query via IPC
    3. Verify: routing happened, worker processed, response returned
    4. Verify: learning DB logged the interaction
    5. Verify: memory saved the conversation
    """
    pass

def test_goal_execution_flow():
    """
    1. Create a goal: "Research Python decorators"
    2. Verify: Task Engine creates plan
    3. Verify: Tasks execute in order
    4. Verify: Reporter logs activities
    5. Verify: Results saved to correct focus
    """
    pass

def test_scheduled_job_flow():
    """
    1. Create scheduled job (run in 5 seconds)
    2. Wait for trigger
    3. Verify: Task Engine received job
    4. Verify: Execution happened
    5. Verify: Reporter notified
    """
    pass

def test_attention_to_voice_flow():
    """
    1. Simulate attention_gained event
    2. Simulate voice input
    3. Verify: Audio pipeline activated
    4. Verify: Transcription sent to worker
    5. Verify: Response generated
    """
    pass

def test_failure_recovery():
    """
    1. Start daemon
    2. Kill a worker process
    3. Verify: Health monitor detects
    4. Verify: Worker restarted
    5. Verify: System continues functioning
    """
    pass

def test_graceful_shutdown():
    """
    1. Start daemon with pending tasks
    2. Send shutdown signal
    3. Verify: In-progress tasks complete or save state
    4. Verify: All processes terminate cleanly
    5. Verify: State persisted for restart
    """
    pass
```

## Run with:
```bash
python3 -m pytest tests/test_e2e.py -v --timeout=60
```

## Acceptance Criteria
- [ ] All 6 scenario tests pass
- [ ] Tests complete in under 2 minutes total
- [ ] No zombie processes after tests
- [ ] Clean state between tests
```

---

### Prompt 1B: IPC Communication Layer

```markdown
# Task: Implement Robust IPC for CLI ↔ Daemon Communication

## Problem
CLI needs to talk to running daemon. Current implementation may be incomplete.

## Requirements
1. Unix socket for local communication
2. Request/response pattern with correlation IDs
3. Timeout handling
4. Reconnection logic

## Create: `daemon/ipc_server.py`

```python
"""
IPC Server for daemon communication.

Provides:
- Unix socket server
- HTTP fallback (optional)
- Request/response matching
- Client authentication (future)
"""

import json
import socket
import threading
import logging
from pathlib import Path
from typing import Optional
from multiprocessing import Queue, Event

logger = logging.getLogger('senter.ipc')

class IPCServer:
    def __init__(
        self,
        socket_path: str,
        message_bus: Queue,
        response_queues: dict,
        shutdown_event: Event
    ):
        self.socket_path = Path(socket_path)
        self.message_bus = message_bus
        self.response_queues = response_queues
        self.shutdown_event = shutdown_event
        
        self.pending_requests: dict[str, Queue] = {}
        self.server_socket = None
    
    def run(self):
        """Main server loop"""
        self._cleanup_socket()
        self._create_socket()
        
        logger.info(f"IPC server listening on {self.socket_path}")
        
        while not self.shutdown_event.is_set():
            try:
                self.server_socket.settimeout(1.0)
                try:
                    conn, addr = self.server_socket.accept()
                    threading.Thread(
                        target=self._handle_client,
                        args=(conn,),
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue
            except Exception as e:
                logger.error(f"IPC error: {e}")
        
        self._cleanup_socket()
    
    def _handle_client(self, conn: socket.socket):
        """Handle a client connection"""
        try:
            data = conn.recv(65536)
            if not data:
                return
            
            request = json.loads(data.decode())
            response = self._process_request(request)
            
            conn.sendall(json.dumps(response).encode())
        except Exception as e:
            logger.error(f"Client error: {e}")
            conn.sendall(json.dumps({"error": str(e)}).encode())
        finally:
            conn.close()
    
    def _process_request(self, request: dict) -> dict:
        """Process an IPC request"""
        cmd = request.get("command")
        
        if cmd == "query":
            return self._handle_query(request)
        elif cmd == "status":
            return self._handle_status()
        elif cmd == "goals":
            return self._handle_goals()
        elif cmd == "progress":
            return self._handle_progress()
        elif cmd == "schedule":
            return self._handle_schedule(request)
        else:
            return {"error": f"Unknown command: {cmd}"}
    
    # Implement handlers...
```

## Create: `daemon/ipc_client.py`

```python
"""
IPC Client for CLI tools.
"""

import json
import socket
from pathlib import Path

class IPCClient:
    def __init__(self, socket_path: str = "/tmp/senter.sock"):
        self.socket_path = Path(socket_path)
    
    def send(self, command: str, **kwargs) -> dict:
        """Send command to daemon and get response"""
        if not self.socket_path.exists():
            return {"error": "Daemon not running"}
        
        request = {"command": command, **kwargs}
        
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(self.socket_path))
            sock.settimeout(30.0)
            
            sock.sendall(json.dumps(request).encode())
            
            response_data = sock.recv(65536)
            return json.loads(response_data.decode())
        except socket.timeout:
            return {"error": "Request timed out"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            sock.close()
    
    def query(self, text: str) -> dict:
        return self.send("query", text=text)
    
    def status(self) -> dict:
        return self.send("status")
    
    def goals(self) -> dict:
        return self.send("goals")
    
    def progress(self, hours: int = 24) -> dict:
        return self.send("progress", hours=hours)
```

## Update `scripts/senter_ctl.py` to use IPC client

## Acceptance Criteria
- [ ] CLI can send queries to running daemon
- [ ] Status command shows all component health
- [ ] Goals command lists active goals from task engine
- [ ] Progress command shows reporter digest
- [ ] Handles daemon not running gracefully
```

---

## PROMPT SET 2: ERROR HANDLING & RECOVERY

### Prompt 2A: Comprehensive Error Handling

```markdown
# Task: Add Robust Error Handling Throughout

## Problem
Components may fail in various ways. Need graceful handling.

## Error Categories
1. **Startup errors**: Missing config, port in use, missing model
2. **Runtime errors**: API timeout, model OOM, disk full
3. **Communication errors**: Queue full, socket disconnect
4. **External errors**: Ollama down, network issues

## Implementation

### 1. Create error types: `daemon/errors.py`
```python
class SenterError(Exception):
    """Base Senter error"""
    pass

class ConfigurationError(SenterError):
    """Configuration is invalid or missing"""
    pass

class ModelError(SenterError):
    """Model loading or inference error"""
    pass

class CommunicationError(SenterError):
    """IPC or message bus error"""
    pass

class ComponentError(SenterError):
    """Component failed to start or crashed"""
    def __init__(self, component: str, message: str):
        self.component = component
        super().__init__(f"{component}: {message}")
```

### 2. Add retry logic to workers: `workers/model_worker.py`
```python
def _process_with_retry(self, msg: dict, max_retries: int = 3):
    """Process request with retry logic"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return self._process_request(msg)
        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"Attempt {attempt + 1} failed: {e}, waiting {wait_time}s")
            time.sleep(wait_time)
    
    # All retries failed
    self._send_error_response(msg, str(last_error))
```

### 3. Add circuit breaker for external services
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
            else:
                raise CircuitOpenError("Circuit is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
```

### 4. Add to all components:
- Try/except around main loops
- Logging with context
- Graceful degradation where possible
- State preservation on error

## Acceptance Criteria
- [ ] No unhandled exceptions crash daemon
- [ ] Errors logged with full context
- [ ] Retries for transient failures
- [ ] Circuit breaker prevents cascade failures
- [ ] User sees helpful error messages
```

---

### Prompt 2B: State Persistence & Recovery

```markdown
# Task: Implement State Persistence for Crash Recovery

## Problem
If daemon crashes, all in-progress work is lost.

## Requirements
1. Persist state periodically
2. Recover state on restart
3. Resume interrupted tasks
4. Handle partial state gracefully

## Implementation

### Create: `daemon/state_manager.py`
```python
"""
State persistence for crash recovery.
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Any

logger = logging.getLogger('senter.state')

class StateManager:
    def __init__(self, state_dir: str = "data/state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.state_dir / "daemon_state.json"
        self.checkpoint_interval = 30  # seconds
        self.last_checkpoint = 0
    
    def save_component_state(self, component: str, state: dict):
        """Save state for a specific component"""
        component_file = self.state_dir / f"{component}.json"
        
        data = {
            "component": component,
            "state": state,
            "saved_at": time.time()
        }
        
        # Write atomically
        temp_file = component_file.with_suffix('.tmp')
        temp_file.write_text(json.dumps(data, indent=2, default=str))
        temp_file.rename(component_file)
        
        logger.debug(f"Saved state for {component}")
    
    def load_component_state(self, component: str) -> dict:
        """Load state for a specific component"""
        component_file = self.state_dir / f"{component}.json"
        
        if not component_file.exists():
            return {}
        
        try:
            data = json.loads(component_file.read_text())
            age = time.time() - data.get("saved_at", 0)
            
            if age > 3600:  # State older than 1 hour
                logger.warning(f"State for {component} is {age/3600:.1f} hours old")
            
            return data.get("state", {})
        except Exception as e:
            logger.error(f"Failed to load state for {component}: {e}")
            return {}
    
    def checkpoint(self, components: dict[str, Any]):
        """Checkpoint all component states"""
        now = time.time()
        
        if now - self.last_checkpoint < self.checkpoint_interval:
            return
        
        for name, component in components.items():
            if hasattr(component, 'get_state'):
                self.save_component_state(name, component.get_state())
        
        self.last_checkpoint = now
        logger.info("Checkpoint completed")
    
    def clear_state(self, component: str = None):
        """Clear state (on clean shutdown)"""
        if component:
            state_file = self.state_dir / f"{component}.json"
            if state_file.exists():
                state_file.unlink()
        else:
            for f in self.state_dir.glob("*.json"):
                f.unlink()
```

### Add to each component:
```python
def get_state(self) -> dict:
    """Return current state for persistence"""
    return {
        "pending_tasks": [t.to_dict() for t in self.pending_tasks],
        "in_progress": [t.to_dict() for t in self.in_progress],
        # ... component-specific state
    }

def restore_state(self, state: dict):
    """Restore state from persistence"""
    for task_data in state.get("pending_tasks", []):
        task = Task.from_dict(task_data)
        self.pending_tasks.append(task)
    # ... restore other state
```

### On daemon start:
```python
def _restore_previous_state(self):
    """Restore state from previous run"""
    for component_name, component in self.components.items():
        if hasattr(component, 'restore_state'):
            state = self.state_manager.load_component_state(component_name)
            if state:
                logger.info(f"Restoring state for {component_name}")
                component.restore_state(state)
```

## Acceptance Criteria
- [ ] State checkpointed every 30 seconds
- [ ] Restart resumes in-progress tasks
- [ ] Completed tasks not re-executed
- [ ] Corrupted state handled gracefully
- [ ] Clean shutdown clears state
```

---

## PROMPT SET 3: REAL-WORLD TESTING

### Prompt 3A: Manual Test Scenarios

```markdown
# Task: Create Manual Test Playbook

## Purpose
Verify real-world usage scenarios work correctly.

## Create: `docs/TEST_PLAYBOOK.md`

# Senter Test Playbook

## Scenario 1: Basic Query Flow
**Steps:**
1. Start daemon: `python3 scripts/senter_ctl.py start`
2. Check status: `python3 scripts/senter_ctl.py status`
3. Send query: `python3 scripts/senter_ctl.py query "What is Python?"`
4. Verify response received
5. Check learning: `python3 scripts/senter_ctl.py learn`

**Expected:**
- [ ] Daemon starts without errors
- [ ] Status shows all components healthy
- [ ] Query returns LLM response
- [ ] Learning DB records the interaction

---

## Scenario 2: Goal Tracking
**Steps:**
1. Create goal via query: "I need to finish my presentation by Friday"
2. Check goals: `python3 scripts/senter_ctl.py goals`
3. Wait for task execution
4. Check progress: `python3 scripts/senter_ctl.py progress`

**Expected:**
- [ ] Goal extracted and saved
- [ ] Tasks created for goal
- [ ] Progress shows task activity

---

## Scenario 3: Scheduled Job
**Steps:**
1. Check scheduler: `python3 scripts/senter_ctl.py schedule list`
2. Note next background_research time
3. Wait for trigger (or manually trigger)
4. Check progress for research activity

**Expected:**
- [ ] Scheduled jobs listed
- [ ] Job triggers at correct time
- [ ] Research results saved

---

## Scenario 4: Daemon Recovery
**Steps:**
1. Start daemon
2. Create a goal (in-progress task)
3. Kill daemon: `kill -9 $(cat data/senter.pid)`
4. Restart daemon
5. Check if task resumes

**Expected:**
- [ ] Daemon restarts cleanly
- [ ] Previous state restored
- [ ] In-progress task continues

---

## Scenario 5: Long-Running Session
**Steps:**
1. Start daemon
2. Leave running for 4+ hours
3. Interact periodically
4. Check memory usage
5. Check activity log

**Expected:**
- [ ] No memory leaks
- [ ] Logs don't grow unbounded
- [ ] All interactions logged
- [ ] Scheduled jobs fired

---

## Scenario 6: Voice Interaction (if enabled)
**Steps:**
1. Enable audio in config
2. Start daemon
3. Look at camera
4. Speak: "Hello Senter"
5. Verify response

**Expected:**
- [ ] Attention detected
- [ ] Speech transcribed
- [ ] Response spoken
```

---

### Prompt 3B: Stress Testing

```markdown
# Task: Create Stress Tests

## Purpose
Verify system handles high load and edge cases.

## Create: `tests/test_stress.py`

```python
"""
Stress tests for Senter daemon.
"""

import time
import threading
import concurrent.futures
from daemon.ipc_client import IPCClient

def test_concurrent_queries():
    """Send 50 queries simultaneously"""
    client = IPCClient()
    
    def send_query(i):
        return client.query(f"Test query {i}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(send_query, i) for i in range(50)]
        results = [f.result() for f in futures]
    
    # Verify all got responses
    errors = [r for r in results if "error" in r]
    assert len(errors) < 5, f"Too many errors: {len(errors)}"

def test_rapid_fire_queries():
    """Send 100 queries as fast as possible"""
    client = IPCClient()
    
    start = time.time()
    for i in range(100):
        client.query(f"Rapid query {i}")
    elapsed = time.time() - start
    
    # Should complete in reasonable time
    assert elapsed < 120, f"Took too long: {elapsed}s"

def test_large_query():
    """Send a very large query"""
    client = IPCClient()
    
    large_text = "Lorem ipsum " * 10000  # ~120KB
    result = client.query(large_text)
    
    # Should handle gracefully (reject or process)
    assert "error" not in result or "too large" in result.get("error", "")

def test_message_bus_saturation():
    """Flood the message bus"""
    # Would need daemon access to test directly
    pass

def test_memory_under_load():
    """Monitor memory during sustained load"""
    import psutil
    
    client = IPCClient()
    initial_memory = psutil.Process().memory_info().rss
    
    for _ in range(1000):
        client.query("Memory test query")
    
    final_memory = psutil.Process().memory_info().rss
    growth = (final_memory - initial_memory) / 1024 / 1024
    
    # Memory shouldn't grow more than 100MB
    assert growth < 100, f"Memory grew by {growth:.1f}MB"
```

## Acceptance Criteria
- [ ] Handles 50 concurrent queries
- [ ] No crashes under load
- [ ] Memory stable over time
- [ ] Graceful degradation when overloaded
```

---

## PROMPT SET 4: USER EXPERIENCE

### Prompt 4A: Rich CLI Output

```markdown
# Task: Improve CLI Output and Feedback

## Problem
Current CLI output is minimal. Users need better visibility.

## Requirements
1. Colored output for status
2. Progress indicators for long operations
3. Formatted tables for data
4. Interactive mode improvements

## Update: `scripts/senter_ctl.py`

```python
"""
Enhanced Senter control CLI.
"""

import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from daemon.ipc_client import IPCClient

console = Console()

def cmd_status():
    """Show daemon status with rich formatting"""
    client = IPCClient()
    result = client.status()
    
    if "error" in result:
        console.print(f"[red]✗ {result['error']}[/red]")
        return
    
    # Header
    console.print(Panel.fit(
        "[bold green]SENTER DAEMON[/bold green]",
        subtitle=f"PID: {result.get('pid', 'unknown')}"
    ))
    
    # Components table
    table = Table(title="Components")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("PID")
    table.add_column("Uptime")
    
    for name, info in result.get("components", {}).items():
        status = "✓ Running" if info.get("alive") else "✗ Stopped"
        style = "green" if info.get("alive") else "red"
        table.add_row(
            name,
            f"[{style}]{status}[/{style}]",
            str(info.get("pid", "-")),
            format_uptime(info.get("uptime", 0))
        )
    
    console.print(table)

def cmd_query(text: str):
    """Send query with progress indicator"""
    client = IPCClient()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=None)
        result = client.query(text)
        progress.update(task, completed=True)
    
    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
    else:
        console.print(Panel(result.get("response", "No response")))
        
        # Show metadata
        if result.get("focus"):
            console.print(f"[dim]Focus: {result['focus']}[/dim]")
        if result.get("latency"):
            console.print(f"[dim]Latency: {result['latency']:.2f}s[/dim]")

def cmd_goals():
    """Show goals with formatting"""
    client = IPCClient()
    result = client.goals()
    
    if not result.get("goals"):
        console.print("[yellow]No active goals[/yellow]")
        return
    
    table = Table(title="Active Goals")
    table.add_column("#", style="dim")
    table.add_column("Goal")
    table.add_column("Status")
    table.add_column("Progress")
    table.add_column("Deadline")
    
    for i, goal in enumerate(result["goals"], 1):
        status_colors = {"active": "green", "pending": "yellow", "blocked": "red"}
        color = status_colors.get(goal.get("status"), "white")
        
        table.add_row(
            str(i),
            goal.get("description", "")[:50],
            f"[{color}]{goal.get('status', 'unknown')}[/{color}]",
            f"{goal.get('progress', 0)}%",
            goal.get("deadline", "-")
        )
    
    console.print(table)

def cmd_progress():
    """Show progress digest"""
    client = IPCClient()
    result = client.progress()
    
    if "summary" in result:
        console.print(Panel(result["summary"], title="Activity Digest"))
    else:
        console.print("[yellow]No recent activity[/yellow]")
```

## Acceptance Criteria
- [ ] Status shows colored health indicators
- [ ] Query shows spinner while processing
- [ ] Goals displayed in formatted table
- [ ] Progress shows readable digest
```

---

### Prompt 4B: Interactive Mode

```markdown
# Task: Add Interactive Shell Mode

## Purpose
Allow continuous interaction without restarting CLI.

## Add to: `scripts/senter_ctl.py`

```python
def cmd_shell():
    """Interactive Senter shell"""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    
    console.print(Panel.fit(
        "[bold]Senter Interactive Shell[/bold]\n"
        "Type queries or commands. Use /help for available commands.",
        title="Welcome"
    ))
    
    client = IPCClient()
    session = PromptSession(
        history=FileHistory('.senter_history'),
        auto_suggest=AutoSuggestFromHistory(),
    )
    
    commands = {
        "/status": lambda: cmd_status(),
        "/goals": lambda: cmd_goals(),
        "/progress": lambda: cmd_progress(),
        "/focus": lambda args: cmd_focus(args),
        "/help": lambda: show_help(),
        "/exit": lambda: sys.exit(0),
        "/quit": lambda: sys.exit(0),
    }
    
    while True:
        try:
            # Get current focus for prompt
            status = client.status()
            focus = status.get("current_focus", "general")
            
            user_input = session.prompt(f"[{focus}] > ")
            
            if not user_input.strip():
                continue
            
            # Check for command
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd in commands:
                    commands[cmd](args) if args else commands[cmd]()
                else:
                    console.print(f"[red]Unknown command: {cmd}[/red]")
            else:
                # Regular query
                cmd_query(user_input)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Use /exit to quit[/yellow]")
        except EOFError:
            break
    
    console.print("[green]Goodbye![/green]")

def show_help():
    """Show interactive help"""
    help_text = """
[bold]Available Commands:[/bold]

  /status     Show daemon and component status
  /goals      List active goals
  /progress   Show activity digest
  /focus      Show or change current focus
  /help       Show this help
  /exit       Exit shell

[bold]Tips:[/bold]
  • Just type to send a query
  • Goals are extracted automatically from your queries
  • Use arrow keys for history
"""
    console.print(Panel(help_text, title="Help"))
```

## Acceptance Criteria
- [ ] Shell starts with welcome message
- [ ] Commands recognized with /prefix
- [ ] Regular text sent as query
- [ ] History persisted between sessions
- [ ] Ctrl+C doesn't exit (shows message)
- [ ] /exit or /quit cleanly exits
```

---

## PROMPT SET 5: DOCUMENTATION

### Prompt 5A: Generate API Documentation

```markdown
# Task: Create Comprehensive Documentation

## Create the following docs:

### 1. `docs/DAEMON.md` - Daemon Operations
```markdown
# Senter Daemon

## Starting the Daemon

### Foreground (for debugging)
```bash
python3 -m daemon.senter_daemon start
```

### Background (production)
```bash
python3 scripts/senter_ctl.py start
```

### As System Service (macOS)
Create `~/Library/LaunchAgents/com.senter.daemon.plist`:
...

## Monitoring

### Status Check
```bash
python3 scripts/senter_ctl.py status
```

### Logs
```bash
tail -f data/daemon.log
```

## Configuration

See `config/daemon_config.json` for all options.

## Troubleshooting

### Daemon won't start
1. Check if already running: `cat data/senter.pid`
2. Check port conflicts
3. Verify Ollama is running

### Component keeps crashing
1. Check component log
2. Verify dependencies installed
3. Check config for component
```

### 2. `docs/ARCHITECTURE.md` - System Architecture
Detailed explanation of:
- Component responsibilities
- Message flow
- Data flow
- State management

### 3. `docs/DEVELOPMENT.md` - Developer Guide
How to:
- Add new components
- Add new tools
- Extend the scheduler
- Add learning signals

## Acceptance Criteria
- [ ] DAEMON.md covers all operations
- [ ] ARCHITECTURE.md explains system design
- [ ] DEVELOPMENT.md enables contributions
```

---

## EXECUTION ORDER

### Immediate (Do Now)
1. **Prompt 1B**: IPC Communication — CLI can't talk to daemon without this
2. **Prompt 2A**: Error Handling — Prevents crashes in production

### This Week
3. **Prompt 1A**: E2E Integration Tests — Verify everything works together
4. **Prompt 2B**: State Persistence — Survive crashes
5. **Prompt 4A**: Rich CLI — Users can see what's happening

### Next Week
6. **Prompt 3A**: Manual Test Playbook — Real-world validation
7. **Prompt 4B**: Interactive Shell — Better UX
8. **Prompt 3B**: Stress Testing — Production readiness
9. **Prompt 5A**: Documentation — Others can use/extend

---

## VALIDATION CHECKLIST

After completing all prompts:

### Functionality
- [ ] Can start/stop daemon cleanly
- [ ] Can send queries and get responses
- [ ] Goals are tracked and executed
- [ ] Scheduled jobs fire on time
- [ ] Learning DB accumulates data
- [ ] Progress reports are generated

### Reliability
- [ ] Survives component crashes
- [ ] Recovers state after restart
- [ ] Handles high load gracefully
- [ ] No memory leaks over time

### Usability
- [ ] CLI provides clear feedback
- [ ] Errors are understandable
- [ ] Help is available
- [ ] Documentation is complete

---

## THE PERFECT SESSION (After Hardening)

```bash
$ python3 scripts/senter_ctl.py shell

┌─────────────────────────────────────┐
│     Senter Interactive Shell        │
│                                     │
│ Type queries or commands.           │
│ Use /help for available commands.   │
└─────────────────────────────────────┘

[general] > /status

╭──────────────────────────────────╮
│       SENTER DAEMON              │
│       PID: 12345                 │
╰──────────────────────────────────╯

┌─────────────┬───────────┬───────┬─────────┐
│ Component   │ Status    │ PID   │ Uptime  │
├─────────────┼───────────┼───────┼─────────┤
│ model_primary   │ ✓ Running │ 12346 │ 2h 15m  │
│ model_research  │ ✓ Running │ 12347 │ 2h 15m  │
│ task_engine     │ ✓ Running │ 12348 │ 2h 15m  │
│ scheduler       │ ✓ Running │ 12349 │ 2h 15m  │
│ reporter        │ ✓ Running │ 12350 │ 2h 15m  │
│ learning_db     │ ✓ Running │ 12351 │ 2h 15m  │
└─────────────┴───────────┴───────┴─────────┘

[general] > I need to research AI trends for my investor deck

⠋ Processing...

╭──────────────────────────────────────────╮
│ I'll research AI trends for you. I've   │
│ created a goal to track this:           │
│                                         │
│ Goal: "Research AI trends for investor  │
│ deck"                                   │
│                                         │
│ I'm starting background research now.   │
│ Check /progress for updates.            │
╰──────────────────────────────────────────╯

Focus: research | Latency: 1.23s

[general] > /goals

┌───┬────────────────────────────┬────────┬──────────┬──────────┐
│ # │ Goal                       │ Status │ Progress │ Deadline │
├───┼────────────────────────────┼────────┼──────────┼──────────┤
│ 1 │ Research AI trends for ... │ active │ 25%      │ -        │
└───┴────────────────────────────┴────────┴──────────┴──────────┘

[general] > /exit

Goodbye! 👋
```

**That's a hardened, production-ready Senter.**

---

**End of Document**

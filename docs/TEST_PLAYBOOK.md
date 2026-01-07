# Senter Test Playbook

Manual test scenarios for validating Senter daemon functionality.

---

## Prerequisites

1. Ensure Ollama is running: `ollama serve`
2. Ensure you have a model: `ollama pull llama3.2`
3. Start the daemon: `python3 scripts/senter_ctl.py start`

---

## Scenario 1: Basic Query Flow

**Purpose:** Verify the query → model → response pipeline works.

**Steps:**
1. Start daemon:
   ```bash
   python3 scripts/senter_ctl.py start
   ```
2. Check status:
   ```bash
   python3 scripts/senter_ctl.py status
   ```
3. Send query:
   ```bash
   python3 scripts/senter_ctl.py query "What is Python?"
   ```
4. Verify response received

**Expected Results:**
- [ ] Daemon starts without errors
- [ ] Status shows all components healthy (green ✓)
- [ ] Query returns LLM response within 60 seconds
- [ ] Response is relevant to the question

---

## Scenario 2: Interactive Shell

**Purpose:** Verify interactive shell works correctly.

**Steps:**
1. Start shell:
   ```bash
   python3 scripts/senter_ctl.py shell
   ```
2. Try commands:
   ```
   /status
   /health
   /help
   ```
3. Send a query:
   ```
   What is 2+2?
   ```
4. Exit:
   ```
   /quit
   ```

**Expected Results:**
- [ ] Shell shows welcome message
- [ ] /status shows component table
- [ ] Query returns response
- [ ] /quit exits cleanly

---

## Scenario 3: Concurrent Queries

**Purpose:** Verify system handles multiple simultaneous queries.

**Steps:**
1. Open 3 terminal windows
2. In each, run:
   ```bash
   python3 scripts/senter_ctl.py query "Tell me a joke"
   ```
3. Run all at the same time
4. Check all get responses

**Expected Results:**
- [ ] All 3 queries get responses
- [ ] No queries timeout
- [ ] Daemon remains stable

---

## Scenario 4: Daemon Recovery

**Purpose:** Verify daemon survives crashes and restarts.

**Steps:**
1. Start daemon:
   ```bash
   python3 scripts/senter_ctl.py start
   ```
2. Note the PID:
   ```bash
   cat data/senter.pid
   ```
3. Kill daemon forcefully:
   ```bash
   kill -9 $(cat data/senter.pid)
   ```
4. Restart daemon:
   ```bash
   python3 scripts/senter_ctl.py start
   ```
5. Verify working:
   ```bash
   python3 scripts/senter_ctl.py query "Hello"
   ```

**Expected Results:**
- [ ] Daemon restarts cleanly
- [ ] No zombie processes left
- [ ] Queries work after restart

---

## Scenario 5: Graceful Shutdown

**Purpose:** Verify daemon stops cleanly.

**Steps:**
1. Start daemon:
   ```bash
   python3 scripts/senter_ctl.py start
   ```
2. Stop daemon:
   ```bash
   python3 scripts/senter_ctl.py stop
   ```
3. Check no processes remain:
   ```bash
   ps aux | grep senter_daemon
   ```
4. Check files cleaned:
   ```bash
   ls data/senter.pid  # Should not exist
   ls /tmp/senter.sock  # Should not exist
   ```

**Expected Results:**
- [ ] Daemon stops within 10 seconds
- [ ] No zombie processes
- [ ] PID file removed
- [ ] Socket file removed

---

## Scenario 6: Long-Running Session

**Purpose:** Verify stability over extended operation.

**Steps:**
1. Start daemon
2. Leave running for 4+ hours
3. Periodically run:
   ```bash
   python3 scripts/senter_ctl.py status
   python3 scripts/senter_ctl.py query "Test"
   ```
4. Check memory usage:
   ```bash
   ps aux | grep senter
   ```

**Expected Results:**
- [ ] All interactions logged
- [ ] Memory stable (no leaks)
- [ ] Logs don't grow unbounded
- [ ] All queries succeed

---

## Scenario 7: Status Under Load

**Purpose:** Verify status endpoint during heavy load.

**Steps:**
1. In one terminal, run rapid queries:
   ```bash
   for i in {1..20}; do
     python3 scripts/senter_ctl.py query "Test $i" &
   done
   ```
2. In another terminal, check status:
   ```bash
   python3 scripts/senter_ctl.py status
   ```

**Expected Results:**
- [ ] Status responds quickly
- [ ] Shows correct component status
- [ ] Doesn't hang or timeout

---

## Scenario 8: IPC Resilience

**Purpose:** Verify IPC handles errors gracefully.

**Steps:**
1. Start daemon
2. Try to send malformed requests:
   ```python
   import socket
   sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
   sock.connect('/tmp/senter.sock')
   sock.send(b'not json')
   print(sock.recv(1024))
   sock.close()
   ```
3. Verify daemon still works:
   ```bash
   python3 scripts/senter_ctl.py query "Still working?"
   ```

**Expected Results:**
- [ ] Daemon doesn't crash
- [ ] Returns error response for bad input
- [ ] Continues to work normally

---

## Automated Tests

Run the full test suites:

```bash
# E2E tests
python3 tests/test_e2e.py

# Stress tests
python3 tests/test_stress.py

# Integration tests
python3 -m pytest tests/ -v
```

---

## Troubleshooting

### Daemon won't start
1. Check if already running: `cat data/senter.pid`
2. Check logs: `tail -50 data/daemon.log`
3. Verify Ollama running: `curl http://localhost:11434/api/tags`

### Queries timeout
1. Check model worker status: `python3 scripts/senter_ctl.py status`
2. Check Ollama responding: `ollama run llama3.2 "test"`
3. Check queue sizes in status output

### Component keeps crashing
1. Check component log
2. Verify dependencies installed
3. Check restart count in status

---

## Test Results Log

| Date | Scenario | Pass/Fail | Notes |
|------|----------|-----------|-------|
|      | 1. Basic Query |     |       |
|      | 2. Shell |     |       |
|      | 3. Concurrent |     |       |
|      | 4. Recovery |     |       |
|      | 5. Shutdown |     |       |
|      | 6. Long-Running |     |       |
|      | 7. Load |     |       |
|      | 8. IPC |     |       |

# Senter Demo Guide

**Status**: Demo Ready (as of January 7, 2026)
**Backend**: Ollama with llama3.2

---

## Quick Start (60-Second Demo)

```bash
# 1. Start CLI
python3 scripts/senter.py --quiet

# 2. List focuses
/list

# 3. Switch focus
/focus coding

# 4. Ask a question
What is a Python decorator?

# 5. Exit cleanly
/exit
```

---

## What TO Demonstrate

| Feature | Command/Action | Expected Result |
|---------|---------------|-----------------|
| CLI startup | `python3 scripts/senter.py` | Clean prompt with focus list |
| Focus list | `/list` | Shows 5 focuses |
| Focus switch | `/focus research` | Confirms switch |
| Basic query | "What is Python?" | LLM responds via Ollama |
| Clean exit | `/exit` | Goodbye message |

---

## What Works

- CLI interface with focus management
- Ollama integration via OpenAI-compatible API
- SENTER.md parsing and focus loading
- 5 user focuses (general, coding, research, creative, user_personal)
- 7 internal agent prompt templates
- Web search function (limited by DuckDuckGo API)

---

## What NOT to Claim

| Claim | Reality |
|-------|---------|
| "7 working agents" | 7 SENTER.md prompt templates (not Python code) |
| "Self-learning" | Stub only - prints log message, no actual learning |
| "Semantic routing" | Uses LLM for routing, not embeddings |
| "Parallel inference" | Background threads exist, not dual simultaneous inference |
| "Always learning" | No persistent learning implemented |

---

## Honest Talking Points

**What Senter IS:**
- A local-first AI assistant framework
- Uses Ollama for LLM inference (privacy-first)
- Configurable focus areas for different tasks
- Extensible via SENTER.md configurations
- CLI interface (TUI also available)

**What Senter WILL BE (roadmap):**
- Self-learning from conversations (evolution logic stub exists)
- Semantic routing via embeddings (nomic-embed available in Ollama)
- Voice input/output (TTS code exists but not configured)
- Autonomous background tasks

---

## Troubleshooting

### "Model not found"
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags
```

### "Connection refused"
```bash
# Start Ollama
ollama serve
```

### Slow responses
- Try smaller model: Edit config to use `llama3.2:1b`
- Or run: `ollama pull llama3.2:1b`

---

## Demo Verification

Run the verification script before demoing:
```bash
python3 demo_test.py
```

Expected: 8/8 tests pass, "DEMO READY" message.

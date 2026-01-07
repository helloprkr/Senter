# Senter Quick Start Guide

## Installation

1. **Clone or copy Senter**:
```bash
cd /path/to/ai-toolbox/Senter
```

2. **Run setup wizard**:
```bash
python3 scripts/setup_senter.py
```

The setup wizard will:
- Download infrastructure models (Qwen2.5-Omni-3B + Nomic Embed)
- Configure your central model
  - Option A: Download recommended (Hermes 3B or Qwen VL 8B)
  - Option B: Use existing local model
  - Option C: Use OpenAI-compatible API
- Verify all components work

3. **Launch Senter**:
```bash
# Option 1: CLI
python3 scripts/senter.py "Your message here"

# Option 2: TUI
python3 scripts/senter_app.py

# Option 3: Use alias
senter
```

## First Run

When you first run Senter:

1. **Models load on demand** (lazy loading prevents memory issues)
2. **Default Focuses are available**:
   - general
   - coding
   - creative
   - research
   - user_personal

3. **Dynamic Focus creation** enabled:
   - Ask about any topic → Senter creates new Focus
   - Focuses evolve with your interactions
   - No fixed categories!

## Examples

```bash
# Chat about AI
senter "Explain machine learning in simple terms"

# Ask a coding question
senter "How do I create a Python class?"

# Research something
senter "What are the latest developments in quantum computing?"

# Create a specific Focus
python3 scripts/senter.py --create-focus "Quantum Computing" \
  --focus-description "I want to learn about quantum computing concepts"
```

## Troubleshooting

### Startup Issues

If Senter crashes on startup:

```bash
# Check GPU memory
nvidia-smi

# Check system memory
free -h

# Run setup again
python3 scripts/setup_senter.py
```

### Model Not Found

```bash
# Verify model paths
cat config/senter_config.json
cat config/user_profile.json

# Check if models exist
ls -la Models/
```

## Next Steps

1. Explore default Focuses
2. Try creating new Focuses
3. Check SENTER.md files to see what Senter learned
4. Customize config in `config/user_profile.json`

---

Need more help? See [README.md](README.md) or [SENTER_DOCUMENTATION.md](SENTER_DOCUMENTATION.md)

#!/bin/bash
# Senter 3.0 - Quick run script

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install pyyaml numpy httpx rich aiosqlite
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Warning: Ollama doesn't appear to be running."
    echo "Start Ollama first, or configure genome.yaml for OpenAI."
fi

# Run Senter
python senter.py "$@"

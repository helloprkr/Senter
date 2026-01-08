#!/bin/bash

# Senter 3.0 MVP Startup Script
# Starts both the Python backend and the Electron/Next.js frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "  Senter 3.0 MVP - Starting Up"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install Python dependencies if needed
echo -e "${BLUE}[1/4] Checking Python dependencies...${NC}"
pip install -q aiohttp aiohttp_cors > /dev/null 2>&1

# Start the Senter backend HTTP server
echo -e "${BLUE}[2/4] Starting Senter backend server on port 8420...${NC}"
python3 -c "
import asyncio
from pathlib import Path
from daemon.http_server import SenterHTTPServer

async def main():
    # Use genome.yaml from project root
    genome_path = Path('genome.yaml')
    if not genome_path.exists():
        print(f'Error: genome.yaml not found at {genome_path.absolute()}')
        return

    server = SenterHTTPServer(genome_path)
    await server.start()

    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()

asyncio.run(main())
" &
BACKEND_PID=$!

# Give the backend a moment to start
sleep 2

# Check if backend started successfully
if ps -p $BACKEND_PID > /dev/null 2>&1; then
    echo -e "${GREEN}Backend started successfully (PID: $BACKEND_PID)${NC}"
else
    echo "Error: Backend failed to start"
    exit 1
fi

# Start the frontend
echo -e "${BLUE}[3/4] Starting frontend...${NC}"
cd ui

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install --legacy-peer-deps
fi

echo -e "${BLUE}[4/4] Launching Electron app...${NC}"
npm run electron:dev &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}======================================"
echo "  Senter 3.0 MVP is running!"
echo "======================================"
echo ""
echo "  Backend:  http://127.0.0.1:8420/api"
echo "  Frontend: http://localhost:3000"
echo ""
echo "  Press Ctrl+C to stop both services"
echo "======================================${NC}"

# Wait for both processes
cleanup() {
    echo ""
    echo "Shutting down Senter MVP..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

wait $FRONTEND_PID

#!/bin/bash
#
# Senter Service Installer (DI-001)
# Installs launchd service for macOS auto-start
#

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Service identifiers
SERVICE_NAME="com.senter.daemon"
PLIST_FILE="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"

# Get script directory (where plist template is)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENTER_ROOT="$(dirname "$SCRIPT_DIR")"

# Find Python
if command -v python3 &> /dev/null; then
    PYTHON_PATH=$(which python3)
elif [ -f "$SENTER_ROOT/venv/bin/python3" ]; then
    PYTHON_PATH="$SENTER_ROOT/venv/bin/python3"
else
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Senter Service Installer${NC}"
echo "========================"
echo
echo "Configuration:"
echo "  SENTER_ROOT: $SENTER_ROOT"
echo "  Python: $PYTHON_PATH"
echo "  Plist: $PLIST_FILE"
echo

# Check if service is already installed
if [ -f "$PLIST_FILE" ]; then
    echo -e "${YELLOW}Service already installed. Unloading existing service...${NC}"
    launchctl unload "$PLIST_FILE" 2>/dev/null || true
fi

# Create LaunchAgents directory if needed
mkdir -p "$HOME/Library/LaunchAgents"

# Create data directory for logs
mkdir -p "$SENTER_ROOT/data"

# Generate plist from template
echo "Generating service file..."
TEMPLATE="$SCRIPT_DIR/com.senter.daemon.plist"

if [ ! -f "$TEMPLATE" ]; then
    echo -e "${RED}Error: Template not found: $TEMPLATE${NC}"
    exit 1
fi

# Replace placeholders
sed -e "s|__PYTHON_PATH__|$PYTHON_PATH|g" \
    -e "s|__SENTER_ROOT__|$SENTER_ROOT|g" \
    "$TEMPLATE" > "$PLIST_FILE"

# Set permissions
chmod 644 "$PLIST_FILE"

# Load the service
echo "Loading service..."
launchctl load "$PLIST_FILE"

# Check status
sleep 1
if launchctl list | grep -q "$SERVICE_NAME"; then
    echo
    echo -e "${GREEN}Success! Senter service installed and running.${NC}"
    echo
    echo "Management commands:"
    echo "  Status:  launchctl list | grep senter"
    echo "  Stop:    launchctl stop $SERVICE_NAME"
    echo "  Start:   launchctl start $SERVICE_NAME"
    echo "  Restart: launchctl kickstart -k gui/\$(id -u)/$SERVICE_NAME"
    echo "  Logs:    tail -f $SENTER_ROOT/data/launchd_stdout.log"
    echo
    echo "To uninstall: ./scripts/uninstall-service.sh"
else
    echo -e "${RED}Warning: Service loaded but may not be running.${NC}"
    echo "Check: launchctl print gui/\$(id -u)/$SERVICE_NAME"
fi

#!/bin/bash
#
# Senter Service Uninstaller (DI-001)
# Removes launchd service for Senter
#

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Service identifiers
SERVICE_NAME="com.senter.daemon"
PLIST_FILE="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"

echo -e "${YELLOW}Senter Service Uninstaller${NC}"
echo "=========================="
echo

# Check if service exists
if [ ! -f "$PLIST_FILE" ]; then
    echo "Service not installed at: $PLIST_FILE"
    exit 0
fi

# Stop the service
echo "Stopping service..."
launchctl stop "$SERVICE_NAME" 2>/dev/null || true

# Unload the service
echo "Unloading service..."
launchctl unload "$PLIST_FILE" 2>/dev/null || true

# Remove the plist file
echo "Removing service file..."
rm -f "$PLIST_FILE"

echo
echo -e "${GREEN}Senter service uninstalled.${NC}"
echo
echo "Note: Log files remain in data/launchd_*.log"
echo "To reinstall: ./scripts/install-service.sh"

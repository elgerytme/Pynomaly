#!/bin/bash
# Bash script to install Git hooks for Pynomaly project
# Cross-platform alternative to `make git-hooks`

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔗 Installing Git hooks...${NC}"

# Set Git hooks path
echo -e "${YELLOW}Setting Git hooks path to scripts/git/hooks/${NC}"
git config core.hooksPath scripts/git/hooks

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Git hooks installed successfully!${NC}"
    echo -e "${CYAN}Hooks available:${NC}"
    echo -e "${CYAN}  - pre-commit  -> branch naming lint + partial linting${NC}"
    echo -e "${CYAN}  - pre-push    -> run unit tests${NC}"
    echo -e "${CYAN}  - post-checkout -> remind to restart long-running services${NC}"
    echo -e "${GREEN}🎯 Cross-platform installation complete!${NC}"
else
    echo -e "${RED}❌ Failed to install Git hooks${NC}"
    exit 1
fi

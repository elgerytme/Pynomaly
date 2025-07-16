#!/bin/bash
#
# Git Hooks Installation Script
# =============================
# Installs repository organization git hooks
#

set -e

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Repository root
REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"
SOURCE_DIR="$REPO_ROOT/scripts/git-hooks"

echo -e "${BLUE}🔧 Installing repository organization git hooks...${NC}"

# Check if we're in a git repository
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo -e "${RED}❌ Not in a git repository${NC}"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Install hooks
HOOKS=("pre-commit" "post-commit" "pre-push")

for hook in "${HOOKS[@]}"; do
    source_file="$SOURCE_DIR/$hook"
    target_file="$HOOKS_DIR/$hook"
    
    if [ ! -f "$source_file" ]; then
        echo -e "${RED}❌ Source hook not found: $source_file${NC}"
        continue
    fi
    
    # Backup existing hook if it exists
    if [ -f "$target_file" ]; then
        backup_file="$target_file.backup.$(date +%Y%m%d_%H%M%S)"
        echo -e "${YELLOW}📋 Backing up existing $hook hook to $backup_file${NC}"
        cp "$target_file" "$backup_file"
    fi
    
    # Copy and make executable
    cp "$source_file" "$target_file"
    chmod +x "$target_file"
    
    echo -e "${GREEN}✅ Installed $hook hook${NC}"
done

# Create a script to enable/disable hooks
cat > "$HOOKS_DIR/toggle-organization-hooks.sh" << 'EOF'
#!/bin/bash
# Toggle repository organization hooks

HOOKS_DIR="$(dirname "$0")"

case "$1" in
    enable)
        for hook in pre-commit post-commit pre-push; do
            if [ -f "$HOOKS_DIR/$hook.disabled" ]; then
                mv "$HOOKS_DIR/$hook.disabled" "$HOOKS_DIR/$hook"
                echo "Enabled $hook hook"
            fi
        done
        ;;
    disable)
        for hook in pre-commit post-commit pre-push; do
            if [ -f "$HOOKS_DIR/$hook" ]; then
                mv "$HOOKS_DIR/$hook" "$HOOKS_DIR/$hook.disabled"
                echo "Disabled $hook hook"
            fi
        done
        ;;
    status)
        for hook in pre-commit post-commit pre-push; do
            if [ -f "$HOOKS_DIR/$hook" ]; then
                echo "$hook: enabled"
            elif [ -f "$HOOKS_DIR/$hook.disabled" ]; then
                echo "$hook: disabled"
            else
                echo "$hook: not installed"
            fi
        done
        ;;
    *)
        echo "Usage: $0 {enable|disable|status}"
        echo "  enable  - Enable all organization hooks"
        echo "  disable - Disable all organization hooks"
        echo "  status  - Show status of hooks"
        exit 1
        ;;
esac
EOF

chmod +x "$HOOKS_DIR/toggle-organization-hooks.sh"

echo ""
echo -e "${GREEN}✅ Git hooks installation complete!${NC}"
echo ""
echo -e "${BLUE}Installed hooks:${NC}"
for hook in "${HOOKS[@]}"; do
    echo "  - $hook"
done
echo ""
echo -e "${BLUE}Utility commands:${NC}"
echo "  - Toggle hooks: .git/hooks/toggle-organization-hooks.sh {enable|disable|status}"
echo "  - Test validation: scripts/validation/validate_organization.py"
echo "  - Auto-organize: scripts/cleanup/auto_organize.py"
echo ""
echo -e "${YELLOW}Note: Hooks will run automatically on git operations${NC}"
echo -e "${YELLOW}Use --no-verify to skip hooks when needed${NC}"
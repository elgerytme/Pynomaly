#!/bin/bash
#
# Install Changelog Management Automation
#
# This script sets up the changelog update automation including:
# - Pre-commit hooks
# - Git aliases for common changelog operations
# - Validation scripts
#

set -e

echo "🔧 Installing Pynomaly Changelog Management Automation"
echo "=" * 60

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "❌ Not in a git repository!"
    echo "   Please run this script from within the Pynomaly git repository"
    exit 1
fi

# Check if required scripts exist
REQUIRED_SCRIPTS=(
    "scripts/check_changelog_update.py"
    "scripts/update_changelog_helper.py"
    "scripts/pre_commit_changelog_hook.sh"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$PROJECT_ROOT/$script" ]; then
        echo "❌ Required script not found: $script"
        exit 1
    fi
done

echo "✅ All required scripts found"

# Install pre-commit hook
echo ""
echo "📎 Installing pre-commit hook..."

if [ ! -d "$HOOKS_DIR" ]; then
    echo "❌ Git hooks directory not found: $HOOKS_DIR"
    exit 1
fi

# Backup existing pre-commit hook if it exists
if [ -f "$HOOKS_DIR/pre-commit" ]; then
    echo "⚠️  Existing pre-commit hook found, backing up..."
    cp "$HOOKS_DIR/pre-commit" "$HOOKS_DIR/pre-commit.backup.$(date +%Y%m%d-%H%M%S)"
fi

# Install our pre-commit hook
cp "$PROJECT_ROOT/scripts/pre_commit_changelog_hook.sh" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Pre-commit hook installed"

# Set up git aliases
echo ""
echo "🔗 Setting up git aliases..."

# Alias for updating changelog
git config alias.changelog-update '!python3 scripts/update_changelog_helper.py'

# Alias for checking if changelog update is needed
git config alias.changelog-check '!python3 scripts/check_changelog_update.py'

# Alias for committing with changelog check bypass
git config alias.commit-bypass 'commit --no-verify'

# Alias for viewing recent changelog entries
git config alias.changelog-recent '!head -50 CHANGELOG.md'

echo "✅ Git aliases configured:"
echo "   git changelog-update    - Interactive changelog update helper"
echo "   git changelog-check     - Check if changelog update is needed"
echo "   git changelog-recent    - View recent changelog entries"
echo "   git commit-bypass       - Commit without pre-commit hooks (use sparingly)"

# Create convenience scripts directory
SCRIPTS_BIN="$PROJECT_ROOT/.scripts"
mkdir -p "$SCRIPTS_BIN"

# Create convenience wrapper scripts
cat > "$SCRIPTS_BIN/changelog" << 'EOF'
#!/bin/bash
# Convenience script for changelog operations

case "$1" in
    "update"|"u")
        python3 scripts/update_changelog_helper.py
        ;;
    "check"|"c")
        python3 scripts/check_changelog_update.py
        ;;
    "recent"|"r")
        head -50 CHANGELOG.md
        ;;
    "help"|"h"|"")
        echo "Pynomaly Changelog Helper"
        echo ""
        echo "Usage: ./scripts/changelog <command>"
        echo ""
        echo "Commands:"
        echo "  update, u    - Interactive changelog update helper"
        echo "  check, c     - Check if changelog update is needed"
        echo "  recent, r    - View recent changelog entries"
        echo "  help, h      - Show this help message"
        echo ""
        echo "Git aliases are also available:"
        echo "  git changelog-update"
        echo "  git changelog-check"
        echo "  git changelog-recent"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './scripts/changelog help' for usage information"
        exit 1
        ;;
esac
EOF

chmod +x "$SCRIPTS_BIN/changelog"

echo ""
echo "📋 Created convenience script: .scripts/changelog"

# Set up shell completions (optional)
COMPLETION_DIR="$PROJECT_ROOT/.completions"
mkdir -p "$COMPLETION_DIR"

cat > "$COMPLETION_DIR/changelog_completion.bash" << 'EOF'
# Bash completion for changelog script
_changelog_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="update check recent help"

    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}

complete -F _changelog_completion changelog
EOF

echo "✅ Bash completion created (optional): .completions/changelog_completion.bash"

# Validate installation
echo ""
echo "🔍 Validating installation..."

# Test pre-commit hook
if [ -x "$HOOKS_DIR/pre-commit" ]; then
    echo "✅ Pre-commit hook is executable"
else
    echo "❌ Pre-commit hook is not executable"
    exit 1
fi

# Test git aliases
if git config --get alias.changelog-update > /dev/null; then
    echo "✅ Git aliases configured"
else
    echo "❌ Git aliases not configured"
    exit 1
fi

# Test Python scripts
if python3 -c "import sys; sys.exit(0)" 2>/dev/null; then
    echo "✅ Python 3 available"
else
    echo "❌ Python 3 not available"
    exit 1
fi

echo ""
echo "🎉 Changelog management automation installed successfully!"
echo ""
echo "📋 Summary of installed components:"
echo "   ✅ Pre-commit hook: Checks changelog updates before commits"
echo "   ✅ Git aliases: Convenient changelog commands"
echo "   ✅ Helper scripts: Interactive changelog management"
echo "   ✅ CI integration: GitHub Actions for PR validation"
echo ""
echo "🚀 Usage examples:"
echo "   git changelog-update                    # Interactive changelog helper"
echo "   git changelog-check                     # Check if update needed"
echo "   .scripts/changelog update               # Alternative helper access"
echo "   git commit -m 'feat: new feature'      # Triggers changelog check"
echo "   git commit-bypass -m 'docs: typo fix'  # Bypass check (use sparingly)"
echo ""
echo "📖 For more information, see CLAUDE.md > Changelog Management Rules"
echo ""
echo "⚠️  Important reminders:"
echo "   • Update CHANGELOG.md for all significant changes"
echo "   • Follow semantic versioning (MAJOR.MINOR.PATCH)"
echo "   • Include both CHANGELOG.md and TODO.md in commits"
echo "   • Use the interactive helper for proper formatting"
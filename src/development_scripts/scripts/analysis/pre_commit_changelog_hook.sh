#!/bin/bash
#
# Pre-commit hook to check if CHANGELOG.md needs to be updated
#
# This hook runs the changelog update checker and reminds developers
# to update the changelog when making significant changes.
#
# To install this hook:
# 1. Copy this file to .git/hooks/pre-commit
# 2. Make it executable: chmod +x .git/hooks/pre-commit
#

echo "🔍 Checking if CHANGELOG.md update is required..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository"
    exit 1
fi

# Get the project root directory
PROJECT_ROOT=$(git rev-parse --show-toplevel)

# Check if the changelog checker script exists
CHECKER_SCRIPT="$PROJECT_ROOT/scripts/check_changelog_update.py"
if [ ! -f "$CHECKER_SCRIPT" ]; then
    echo "⚠️  Changelog checker script not found at $CHECKER_SCRIPT"
    echo "   Skipping changelog check..."
    exit 0
fi

# Run the changelog checker
python3 "$CHECKER_SCRIPT"
CHECKER_EXIT_CODE=$?

# Handle the result
if [ $CHECKER_EXIT_CODE -eq 0 ]; then
    echo "✅ Changelog check passed"
    exit 0
elif [ $CHECKER_EXIT_CODE -eq 1 ]; then
    echo ""
    echo "💡 You can use the interactive helper to create a changelog entry:"
    echo "   python3 scripts/update_changelog_helper.py"
    echo ""
    echo "❌ Commit blocked - CHANGELOG.md update required"
    echo ""
    echo "To bypass this check (not recommended):"
    echo "   git commit --no-verify"
    echo ""
    exit 1
else
    echo "⚠️  Changelog checker encountered an error"
    echo "   Proceeding with commit..."
    exit 0
fi

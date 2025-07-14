#!/bin/bash

# Install Pynomaly isolation Git hooks
# Sets up automation for the isolation system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GIT_HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "🔧 Installing Pynomaly isolation Git hooks..."

# Check if we're in a Git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "❌ Error: Not in a Git repository"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$GIT_HOOKS_DIR"

# Install pre-commit hook
echo "📝 Installing pre-commit hook..."
cp "$PROJECT_ROOT/.project-rules/hooks/pre-commit-isolation" "$GIT_HOOKS_DIR/pre-commit"
chmod +x "$GIT_HOOKS_DIR/pre-commit"

# Create post-commit hook to clean up old isolations
cat > "$GIT_HOOKS_DIR/post-commit" << 'EOF'
#!/bin/bash

# Post-commit hook for Pynomaly isolation cleanup
# Optionally cleans up old isolation environments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Only run cleanup if isolation is enabled
if grep -q "enabled: true" "$PROJECT_ROOT/.project-rules/isolation-config.yaml" 2>/dev/null; then
    # Run auto-cleanup in background to not slow down commits
    (
        sleep 5  # Brief delay to not interfere with commit
        "$PROJECT_ROOT/.project-rules/scripts/isolate.sh" auto-cleanup 7 >/dev/null 2>&1 || true
    ) &
fi
EOF
chmod +x "$GIT_HOOKS_DIR/post-commit"

# Create prepare-commit-msg hook to add isolation context
cat > "$GIT_HOOKS_DIR/prepare-commit-msg" << 'EOF'
#!/bin/bash

# Prepare commit message hook for isolation context
# Adds isolation information to commit messages

COMMIT_MSG_FILE="$1"
COMMIT_SOURCE="$2"
SHA1="$3"

# Only add context for regular commits (not merges, etc.)
if [ "$COMMIT_SOURCE" = "" ] || [ "$COMMIT_SOURCE" = "message" ]; then
    if [ "${ISOLATION_MODE:-}" = "true" ]; then
        # We're in an isolation environment
        ISOLATION_ID="${ISOLATION_ID:-unknown}"

        # Add isolation context to commit message
        echo "" >> "$COMMIT_MSG_FILE"
        echo "# Isolation Context:" >> "$COMMIT_MSG_FILE"
        echo "# - Environment: ${PYNOMALY_ENV:-unknown}" >> "$COMMIT_MSG_FILE"
        echo "# - Isolation ID: $ISOLATION_ID" >> "$COMMIT_MSG_FILE"
        echo "# - Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "$COMMIT_MSG_FILE"
    fi
fi
EOF
chmod +x "$GIT_HOOKS_DIR/prepare-commit-msg"

# Create pre-push hook to validate isolation state
cat > "$GIT_HOOKS_DIR/pre-push" << 'EOF'
#!/bin/bash

# Pre-push hook for isolation validation
# Ensures isolation requirements are met before pushing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/.project-rules/isolation-config.yaml"

# Check if isolation is enabled
if ! grep -q "enabled: true" "$CONFIG_FILE" 2>/dev/null; then
    # Isolation disabled, allow push
    exit 0
fi

# Get branch being pushed
BRANCH=$(git branch --show-current)

# Check if we're pushing from main without proper process
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
    if grep -q "prevent_direct_main_commits: true" "$CONFIG_FILE" 2>/dev/null; then
        echo "❌ Direct pushes to main branch are not allowed"
        echo "Please use feature branches and isolation environment"
        exit 1
    fi
fi

# If we're in isolation, warn about pushing isolated work
if [ "${ISOLATION_MODE:-}" = "true" ]; then
    echo "⚠️  You are pushing from an isolation environment"
    echo "Isolation ID: ${ISOLATION_ID:-unknown}"
    echo ""
    read -p "Are you sure you want to push isolated work? (y/N): " confirm
    case $confirm in
        [Yy]*)
            echo "Proceeding with push from isolation..."
            ;;
        *)
            echo "Push cancelled"
            exit 1
            ;;
    esac
fi
EOF
chmod +x "$GIT_HOOKS_DIR/pre-push"

echo "✅ Git hooks installed successfully!"
echo ""
echo "📋 Installed hooks:"
echo "  - pre-commit: Triggers isolation for qualifying changes"
echo "  - post-commit: Cleans up old isolation environments"
echo "  - prepare-commit-msg: Adds isolation context to commits"
echo "  - pre-push: Validates isolation state before pushes"
echo ""
echo "🔧 To uninstall hooks, remove files from .git/hooks/"
echo "💡 To test the setup, try making a change to a Python file and committing"

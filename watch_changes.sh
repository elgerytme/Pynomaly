#!/bin/bash

echo "🔍 Starting file watcher for Pynomaly project..."
echo "📁 Watching: /mnt/c/Users/andre/Pynomaly"
echo "⏹️  Press Ctrl+C to stop watching"
echo ""

# Function to check git status and suggest commits
check_changes() {
    echo "📝 Changes detected at $(date)"
    
    # Check if there are any changes
    if ! git diff --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
        echo "🔄 Git status:"
        git status --porcelain
        echo ""
        echo "💡 Suggestion: Run 'git add <files>' and 'git commit -m \"<message>\"' to commit logical units"
        echo "🤖 Or ask me to organize and commit these changes for you!"
        echo ""
    else
        echo "✅ No uncommitted changes found"
        echo ""
    fi
}

# Watch for file changes (excluding .git directory and common temp files)
inotifywait -m -r -e modify,create,delete,move \
    --exclude '\.git/|\.pyc$|__pycache__/|\.tmp$|\.swp$|\.DS_Store' \
    /mnt/c/Users/andre/Pynomaly \
    --format '%T %w %f %e' --timefmt '%H:%M:%S' | \
while read TIME DIR FILE EVENT; do
    echo "📄 $TIME: $FILE ($EVENT) in $DIR"
    check_changes
done

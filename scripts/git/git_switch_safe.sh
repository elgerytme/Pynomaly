#!/bin/bash
# Switch to a Git branch safely, ensuring no uncommitted changes

NAME=$1

if [ -z "$NAME" ]; then
    echo "Error: Branch name is required."
    echo "Usage: ./git_switch_safe.sh <branch-name>"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Error: You have uncommitted changes."
    echo "Please commit or stash your changes before switching branches."
    exit 1
fi

git switch $NAME
if [ $? -ne 0 ]; then
    echo "Failed to switch to branch '$NAME'"
    exit 1
fi

echo "Switched to branch '$NAME' successfully!"

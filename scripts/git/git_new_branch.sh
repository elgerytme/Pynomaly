#!/bin/bash
# Create a new Git branch with validation

TYPE=$1
NAME=$2

# Validate input
if [ -z "$TYPE" ] || [ -z "$NAME" ]; then
    echo "Error: Both TYPE and NAME are required."
    echo "Usage: ./git_new_branch.sh <type> <name>"
    echo "Valid types: feature, bugfix, hotfix, release, chore, docs"
    exit 1
fi

VALID_TYPES=("feature" "bugfix" "hotfix" "release" "chore" "docs")
if [[ ! " ${VALID_TYPES[@]} " =~ " ${TYPE} " ]]; then
    echo "Error: Invalid branch type '$TYPE'."
    echo "Valid types: ${VALID_TYPES[*]}"
    exit 1
fi

if ! [[ "$NAME" =~ ^[a-z0-9-]+$ ]]; then
    echo "Error: Invalid branch name '$NAME'."
    echo "Branch names must contain only lowercase letters, numbers, and hyphens."
    exit 1
fi

BRANCH_NAME="$TYPE/$NAME"

git show-ref --verify --quiet refs/heads/$BRANCH_NAME
if [ $? -eq 0 ]; then
    echo "Error: Branch '$BRANCH_NAME' already exists."
    exit 1
fi

echo "Creating and switching to new branch '$BRANCH_NAME'..."
git checkout -b $BRANCH_NAME


#!/bin/bash

# Git History Cleanup Script
# This script helps maintain a clean git history by providing utilities
# for managing commits and tags

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi
}

# Function to create a backup tag
create_backup_tag() {
    local tag_name="backup-$(date +%Y%m%d-%H%M%S)"
    print_status "Creating backup tag: $tag_name"
    git tag -a "$tag_name" -m "Backup created on $(date)"
    echo "$tag_name"
}

# Function to remove auto-sync commits (creates new branch)
remove_auto_sync_commits() {
    local branch_name="clean-history-$(date +%Y%m%d-%H%M%S)"
    print_status "Creating clean history branch: $branch_name"
    
    # Create backup first
    local backup_tag=$(create_backup_tag)
    
    # Create new branch
    git checkout -b "$branch_name"
    
    # Use git rebase to remove auto-sync commits interactively
    print_status "Use 'git rebase -i' to remove auto-sync commits"
    print_status "Look for commits with message: '🤖 Auto-sync GitHub issues to TODO.md'"
    print_status "Delete those lines in the interactive rebase"
    
    # Get the commit hash from a reasonable point in history
    local base_commit=$(git log --oneline --grep="feat:" | head -20 | tail -1 | cut -d' ' -f1)
    
    if [ -n "$base_commit" ]; then
        print_status "Starting interactive rebase from commit: $base_commit"
        git rebase -i "$base_commit"
    else
        print_warning "Could not find a good base commit for rebase"
    fi
    
    print_status "Clean history branch created: $branch_name"
    print_status "Backup tag created: $backup_tag"
}

# Function to create logical version tags
create_version_tags() {
    print_status "Creating logical version tags..."
    
    # Get commits that represent major milestones
    local pypi_commit=$(git log --oneline --grep="Complete Issue #136" -1 | cut -d' ' -f1)
    local mlops_commit=$(git log --oneline --grep="Complete MLOps Phase 4.*Issue #139" -1 | cut -d' ' -f1)
    local transform_commit=$(git log --oneline --grep="Complete comprehensive data transformation" -1 | cut -d' ' -f1)
    local current_commit=$(git rev-parse HEAD)
    
    # Create tags if commits exist and tags don't already exist
    if [ -n "$pypi_commit" ] && ! git tag -l | grep -q "v1.1.0"; then
        git tag -a "v1.1.0" "$pypi_commit" -m "v1.1.0: PyPI Package Release Preparation Complete"
        print_status "Created tag v1.1.0 for PyPI package release"
    fi
    
    if [ -n "$mlops_commit" ] && ! git tag -l | grep -q "v1.2.0"; then
        git tag -a "v1.2.0" "$mlops_commit" -m "v1.2.0: MLOps Phase 4 - Pipeline Orchestration System Complete"
        print_status "Created tag v1.2.0 for MLOps Phase 4"
    fi
    
    if [ -n "$transform_commit" ] && ! git tag -l | grep -q "v1.3.0"; then
        git tag -a "v1.3.0" "$transform_commit" -m "v1.3.0: Comprehensive Data Transformation Package Integration"
        print_status "Created tag v1.3.0 for data transformation"
    fi
    
    if ! git tag -l | grep -q "v1.4.0"; then
        git tag -a "v1.4.0" "$current_commit" -m "v1.4.0: Current development state with clean history"
        print_status "Created tag v1.4.0 for current state"
    fi
}

# Function to show commit statistics
show_commit_stats() {
    print_status "Commit Statistics:"
    echo "Total commits: $(git rev-list --count HEAD)"
    echo "Auto-sync commits: $(git log --oneline --grep='🤖 Auto-sync' | wc -l)"
    echo "Feature commits: $(git log --oneline --grep='feat:' | wc -l)"
    echo "Fix commits: $(git log --oneline --grep='fix:' | wc -l)"
    echo "Recent commits (last 10):"
    git log --oneline -10
}

# Function to configure git for better commit messages
configure_git_commit_template() {
    print_status "Configuring git commit message template..."
    
    # Set commit message template
    git config commit.template .gitmessage
    
    # Configure git to use the template
    print_status "Git commit template configured"
    print_status "Use 'git commit' (without -m) to use the template"
}

# Main script logic
main() {
    check_git_repo
    
    case "${1:-help}" in
        "stats")
            show_commit_stats
            ;;
        "backup")
            create_backup_tag
            ;;
        "clean")
            remove_auto_sync_commits
            ;;
        "tags")
            create_version_tags
            ;;
        "configure")
            configure_git_commit_template
            ;;
        "help"|*)
            echo "Git History Cleanup Script"
            echo "Usage: $0 [command]"
            echo
            echo "Commands:"
            echo "  stats     - Show commit statistics"
            echo "  backup    - Create backup tag"
            echo "  clean     - Remove auto-sync commits (creates new branch)"
            echo "  tags      - Create logical version tags"
            echo "  configure - Configure git commit message template"
            echo "  help      - Show this help message"
            ;;
    esac
}

main "$@"
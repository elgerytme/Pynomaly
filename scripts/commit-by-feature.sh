#!/bin/bash

# Commit by Feature Script
# This script helps organize commits by feature/issue for cleaner history

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi
}

# Function to show current staged changes
show_staged_changes() {
    print_info "Currently staged changes:"
    git diff --cached --name-only
}

# Function to commit with feature-based message
commit_feature() {
    local issue_number="$1"
    local feature_description="$2"
    local commit_type="${3:-feat}"
    
    if [ -z "$issue_number" ] || [ -z "$feature_description" ]; then
        print_error "Usage: $0 feature <issue_number> <feature_description> [commit_type]"
        exit 1
    fi
    
    # Check if there are staged changes
    if ! git diff --cached --quiet; then
        local commit_message="${commit_type}: ${feature_description} (Issue #${issue_number})"
        
        print_status "Committing with message: $commit_message"
        git commit -m "$commit_message"
        
        # Suggest tagging if this is a major feature
        if [ "$commit_type" = "feat" ]; then
            print_info "Consider creating a version tag for this feature:"
            print_info "git tag -a v1.X.Y -m 'Version 1.X.Y: $feature_description'"
        fi
    else
        print_warning "No staged changes to commit"
    fi
}

# Function to commit a fix
commit_fix() {
    local issue_number="$1"
    local fix_description="$2"
    
    if [ -z "$issue_number" ] || [ -z "$fix_description" ]; then
        print_error "Usage: $0 fix <issue_number> <fix_description>"
        exit 1
    fi
    
    commit_feature "$issue_number" "$fix_description" "fix"
}

# Function to commit documentation
commit_docs() {
    local description="$1"
    
    if [ -z "$description" ]; then
        print_error "Usage: $0 docs <description>"
        exit 1
    fi
    
    if ! git diff --cached --quiet; then
        local commit_message="docs: ${description}"
        
        print_status "Committing documentation with message: $commit_message"
        git commit -m "$commit_message"
    else
        print_warning "No staged changes to commit"
    fi
}

# Function to commit refactoring
commit_refactor() {
    local description="$1"
    
    if [ -z "$description" ]; then
        print_error "Usage: $0 refactor <description>"
        exit 1
    fi
    
    if ! git diff --cached --quiet; then
        local commit_message="refactor: ${description}"
        
        print_status "Committing refactoring with message: $commit_message"
        git commit -m "$commit_message"
    else
        print_warning "No staged changes to commit"
    fi
}

# Function to show commit guidelines
show_guidelines() {
    echo "Git Commit Guidelines for Clean History:"
    echo
    echo "1. Commit Types:"
    echo "   feat:     A new feature"
    echo "   fix:      A bug fix"
    echo "   docs:     Documentation only changes"
    echo "   style:    Changes that don't affect meaning (formatting, etc.)"
    echo "   refactor: Code change that neither fixes a bug nor adds a feature"
    echo "   perf:     Code change that improves performance"
    echo "   test:     Adding missing tests or correcting existing tests"
    echo "   chore:    Changes to build process or auxiliary tools"
    echo
    echo "2. Commit Message Format:"
    echo "   <type>: <description> (Issue #<number>)"
    echo
    echo "3. Examples:"
    echo "   feat: Add anomaly detection algorithm (Issue #123)"
    echo "   fix: Resolve memory leak in data processing (Issue #456)"
    echo "   docs: Update API documentation for new endpoints"
    echo "   refactor: Simplify data transformation pipeline"
    echo
    echo "4. Best Practices:"
    echo "   - Use imperative mood ('Add feature' not 'Added feature')"
    echo "   - Keep subject line under 50 characters"
    echo "   - Reference issue numbers when applicable"
    echo "   - Group related changes into single commits"
    echo "   - Don't commit broken or incomplete features"
}

# Function to create a version tag
create_version_tag() {
    local version="$1"
    local description="$2"
    
    if [ -z "$version" ] || [ -z "$description" ]; then
        print_error "Usage: $0 tag <version> <description>"
        exit 1
    fi
    
    # Check if tag already exists
    if git tag -l | grep -q "^$version$"; then
        print_error "Tag $version already exists"
        exit 1
    fi
    
    # Create annotated tag
    git tag -a "$version" -m "$version: $description"
    print_status "Created tag $version: $description"
    
    # Show tags
    print_info "Recent tags:"
    git tag -l | sort -V | tail -5
}

# Main script logic
main() {
    check_git_repo
    
    case "${1:-help}" in
        "feature")
            shift
            commit_feature "$@"
            ;;
        "fix")
            shift
            commit_fix "$@"
            ;;
        "docs")
            shift
            commit_docs "$@"
            ;;
        "refactor")
            shift
            commit_refactor "$@"
            ;;
        "tag")
            shift
            create_version_tag "$@"
            ;;
        "staged")
            show_staged_changes
            ;;
        "guidelines")
            show_guidelines
            ;;
        "help"|*)
            echo "Commit by Feature Script"
            echo "Usage: $0 [command] [arguments]"
            echo
            echo "Commands:"
            echo "  feature <issue_number> <description>  - Commit a new feature"
            echo "  fix <issue_number> <description>      - Commit a bug fix"
            echo "  docs <description>                    - Commit documentation changes"
            echo "  refactor <description>                - Commit refactoring changes"
            echo "  tag <version> <description>           - Create a version tag"
            echo "  staged                                - Show staged changes"
            echo "  guidelines                            - Show commit guidelines"
            echo "  help                                  - Show this help message"
            echo
            echo "Examples:"
            echo "  $0 feature 123 'Add new anomaly detection algorithm'"
            echo "  $0 fix 456 'Resolve memory leak in data processing'"
            echo "  $0 docs 'Update API documentation'"
            echo "  $0 tag v1.5.0 'Major feature release with new algorithms'"
            ;;
    esac
}

main "$@"
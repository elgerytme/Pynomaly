#!/usr/bin/env python3
"""
Manual GitHub Issues Sync Script

This script can be run manually to sync GitHub issues to TODO.md
without requiring GitHub Actions environment.

Usage:
    # Set your GitHub token
    export GITHUB_TOKEN="your_token_here"

    # Run the sync
    python manual_sync.py
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sync_github_issues_to_todo import GitHubIssuesSync  # noqa: E402


def main():
    """Manual sync execution."""
    # Set repository if not already set
    if not os.getenv("GITHUB_REPOSITORY"):
        os.environ["GITHUB_REPOSITORY"] = "anthropics/pynomaly"  # Update as needed

    # Check for token
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ GITHUB_TOKEN environment variable required")
        print("💡 Set it with: export GITHUB_TOKEN='your_token_here'")
        sys.exit(1)

    print("🚀 Starting manual GitHub Issues sync...")

    # Change to project root directory
    os.chdir(project_root)

    # Run sync
    syncer = GitHubIssuesSync()
    success = syncer.run()

    if success:
        print("✅ Manual sync completed successfully!")
        print("💡 Check TODO.md for updated Issues section")
    else:
        print("❌ Manual sync failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

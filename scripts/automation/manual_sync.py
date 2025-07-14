#!/usr/bin/env python3
"""
Manual GitHub Issues to TODO.md Sync

This script provides a manual way to sync GitHub issues to TODO.md
with additional options for testing and debugging.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sync_github_issues_to_todo import SyncManager, get_github_token

def main():
    parser = argparse.ArgumentParser(description="Manually sync GitHub issues to TODO.md")
    parser.add_argument("--repo-owner", default="elgerytme", help="GitHub repository owner")
    parser.add_argument("--repo-name", default="Pynomaly", help="GitHub repository name")
    parser.add_argument("--todo-file", help="Path to TODO.md file")
    parser.add_argument("--token", help="GitHub token (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced without writing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Set up paths
    if args.todo_file:
        todo_file = Path(args.todo_file)
    else:
        todo_file = Path(__file__).parent.parent.parent / "TODO.md"
    
    # Get token
    token = args.token or get_github_token()
    
    # Create sync manager
    sync_manager = SyncManager(args.repo_owner, args.repo_name, todo_file, token)
    
    if args.dry_run:
        print("🔍 Dry run mode - fetching issues without updating TODO.md")
        issues = sync_manager.github_api.get_issues()
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  #{issue.number}: {issue.title} [{issue.priority}] [{issue.status}]")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        return
    
    # Perform sync
    print(f"🔄 Syncing GitHub issues from {args.repo_owner}/{args.repo_name} to {todo_file}")
    success = sync_manager.sync_issues_to_todo()
    
    if success:
        print("✅ Sync completed successfully!")
        print(f"📄 TODO.md updated at: {todo_file}")
    else:
        print("❌ Sync failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
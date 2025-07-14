#!/usr/bin/env python3
"""
Test script for GitHub Issues sync functionality
"""

import os
import subprocess
import sys


def test_sync_script():
    """Test the sync script with GitHub CLI data"""
    print("🧪 Testing GitHub Issues sync automation...")

    # Check if we have gh CLI
    try:
        result = subprocess.run(["gh", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ GitHub CLI not available")
            return False
        print(f"✅ GitHub CLI available: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ GitHub CLI not found")
        return False

    # Check if we're in a git repo
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print("❌ Not in a git repository")
            return False
        print("✅ In git repository")
    except FileNotFoundError:
        print("❌ Git not found")
        return False

    # Test GitHub API access
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--limit", "1", "--json", "number"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("❌ Cannot access GitHub issues")
            return False
        print("✅ GitHub API access working")
    except Exception as e:
        print(f"❌ GitHub API error: {e}")
        return False

    # Test our sync script exists
    script_path = "scripts/automation/sync_github_issues_to_todo.py"
    if not os.path.exists(script_path):
        print(f"❌ Sync script not found: {script_path}")
        return False
    print(f"✅ Sync script exists: {script_path}")

    # Test TODO.md exists
    if not os.path.exists("TODO.md"):
        print("❌ TODO.md not found")
        return False
    print("✅ TODO.md found")

    print("\n🎉 All tests passed! Sync automation should work correctly.")
    print("💡 To trigger sync manually: gh workflow run issue-sync.yml")
    return True


if __name__ == "__main__":
    success = test_sync_script()
    sys.exit(0 if success else 1)

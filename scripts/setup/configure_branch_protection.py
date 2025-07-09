#!/usr/bin/env python3
"""
Configure GitHub branch protection settings via API.

This script sets up branch protection rules for the main and develop branches
to enforce quality gates and require PR reviews.

Usage:
    python scripts/setup/configure_branch_protection.py --token YOUR_GITHUB_TOKEN
    
Or set the GITHUB_TOKEN environment variable:
    export GITHUB_TOKEN=your_token_here
    python scripts/setup/configure_branch_protection.py
"""

import argparse
import json
import os
import requests
import sys
from typing import Dict, Any


def get_branch_protection_config() -> Dict[str, Any]:
    """Get branch protection configuration for main and develop branches."""
    
    # Required status checks for quality gates
    required_contexts = [
        # Enhanced CI Quality Gates - ALL MUST PASS
        "Quality Gate: Linting & Formatting",
        "Quality Gate: Unit Tests", 
        "Quality Gate: Documentation",
        "Quality Gate: Security Scan",
        "Quality Gate: Build & Package",
        "Quality Gate: Integration Tests",
        "📊 Quality Gate Summary",
        
        # Original CI checks (for backwards compatibility)
        "Code Quality & Linting",
        "Build & Package", 
        "Test Suite",
        "API & CLI Testing",
        "Security & Dependencies",
        "CI Summary",
        
        # Quality Gates workflow checks
        "Test Quality Gate",
        "Security Quality Gate", 
        "Quality Gate Summary",
    ]
    
    # Main branch protection (strict)
    main_protection = {
        "required_status_checks": {
            "strict": True,
            "contexts": required_contexts
        },
        "enforce_admins": True,
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": True,
            "require_last_push_approval": True
        },
        "restrictions": None,  # No push restrictions
        "required_linear_history": True,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": False,
        "required_conversation_resolution": True,
        "lock_branch": False,
        "fork_syncing": True
    }
    
    # Develop branch protection (more relaxed)
    develop_protection = {
        "required_status_checks": {
            "strict": True,
            "contexts": [
                "Quality Gate: Linting & Formatting",
                "Quality Gate: Unit Tests",
                "Quality Gate: Documentation", 
                "Quality Gate: Security Scan",
                "Quality Gate: Build & Package",
                "📊 Quality Gate Summary",
            ]
        },
        "enforce_admins": False,
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": False,
            "require_last_push_approval": False
        },
        "restrictions": None,
        "required_linear_history": False,
        "allow_force_pushes": True,
        "allow_deletions": False,
        "required_conversation_resolution": True,
        "lock_branch": False,
        "fork_syncing": True
    }
    
    return {
        "main": main_protection,
        "develop": develop_protection
    }


def configure_branch_protection(
    owner: str,
    repo: str,
    branch: str,
    protection_config: Dict[str, Any],
    token: str
) -> bool:
    """Configure branch protection for a specific branch."""
    
    url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}/protection"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    
    print(f"🔒 Configuring branch protection for '{branch}' branch...")
    
    try:
        response = requests.put(
            url,
            headers=headers,
            data=json.dumps(protection_config),
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ Successfully configured branch protection for '{branch}'")
            return True
        elif response.status_code == 403:
            print(f"❌ Permission denied. Make sure your token has 'repo' permissions.")
            print(f"Response: {response.text}")
            return False
        else:
            print(f"❌ Failed to configure branch protection for '{branch}'")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error configuring branch protection for '{branch}': {e}")
        return False


def get_repo_info() -> tuple[str, str]:
    """Extract owner and repo name from git remote URL."""
    
    try:
        import subprocess
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True
        )
        
        remote_url = result.stdout.strip()
        
        # Handle both SSH and HTTPS URLs
        if remote_url.startswith("git@github.com:"):
            # SSH: git@github.com:owner/repo.git
            parts = remote_url.replace("git@github.com:", "").replace(".git", "").split("/")
        elif remote_url.startswith("https://github.com/"):
            # HTTPS: https://github.com/owner/repo.git
            parts = remote_url.replace("https://github.com/", "").replace(".git", "").split("/")
        else:
            raise ValueError(f"Unsupported remote URL format: {remote_url}")
            
        if len(parts) != 2:
            raise ValueError(f"Invalid repository URL: {remote_url}")
            
        return parts[0], parts[1]
        
    except subprocess.CalledProcessError:
        print("❌ Error: Could not get git remote URL. Make sure you're in a git repository.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error parsing repository information: {e}")
        sys.exit(1)


def main():
    """Main function to configure branch protection."""
    
    parser = argparse.ArgumentParser(
        description="Configure GitHub branch protection settings"
    )
    parser.add_argument(
        "--token",
        help="GitHub personal access token with repo permissions"
    )
    parser.add_argument(
        "--owner",
        help="Repository owner (auto-detected if not provided)"
    )
    parser.add_argument(
        "--repo", 
        help="Repository name (auto-detected if not provided)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be configured without making changes"
    )
    
    args = parser.parse_args()
    
    # Get GitHub token
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ Error: GitHub token required. Set GITHUB_TOKEN environment variable or use --token argument.")
        print("Create a token at: https://github.com/settings/tokens")
        print("Required permissions: repo")
        sys.exit(1)
    
    # Get repository information
    if args.owner and args.repo:
        owner, repo = args.owner, args.repo
    else:
        owner, repo = get_repo_info()
        
    print(f"📁 Repository: {owner}/{repo}")
    
    # Get branch protection configurations
    branch_configs = get_branch_protection_config()
    
    if args.dry_run:
        print("🧪 DRY RUN - No changes will be made\\n")
        for branch, config in branch_configs.items():
            print(f"Branch: {branch}")
            print(f"Config: {json.dumps(config, indent=2)}")
            print()
        return
    
    # Configure branch protection for each branch
    success = True
    for branch, config in branch_configs.items():
        if not configure_branch_protection(owner, repo, branch, config, token):
            success = False
    
    if success:
        print("\\n🎉 Branch protection configuration completed successfully!")
        print("\\n📋 Summary:")
        print("- Main branch: Strict protection with required reviews and quality gates")
        print("- Develop branch: Relaxed protection with quality gates")
        print("\\n💡 All PRs must now pass quality gates before merging!")
    else:
        print("\\n❌ Some branch protection configurations failed.")
        print("Please check the errors above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

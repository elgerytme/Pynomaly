#!/usr/bin/env python3
"""
Git Branch Isolation Manager

Automates branch creation, validation, and management to ensure isolation
between users, agents, and development workstreams.
"""

import os
import re
import subprocess
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BranchInfo:
    """Branch information structure."""
    name: str
    type: str
    scope: str
    description: str
    created: datetime
    last_commit: datetime
    author: str
    is_merged: bool


class GitBranchIsolationManager:
    """Manages git branch isolation and automation."""
    
    VALID_TYPES = ['feature', 'bugfix', 'hotfix', 'experiment', 'agent', 'user']
    MAX_BRANCHES_PER_SCOPE = 5
    MAX_EXPERIMENT_BRANCHES = 3
    BRANCH_RETENTION_DAYS = 30
    MERGED_BRANCH_CLEANUP_DAYS = 7
    
    def __init__(self, repo_root: str = None):
        """Initialize the branch isolation manager."""
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.config_dir = self.repo_root / "scripts" / "config" / "git"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "branch_isolation_config.json"
        self.load_config()
    
    def load_config(self):
        """Load branch isolation configuration."""
        default_config = {
            "protected_branches": ["main", "develop", "staging"],
            "isolation_scopes": {
                "user": {"max_branches": 5, "retention_days": 30},
                "agent": {"max_branches": 3, "retention_days": 14},
                "pkg": {"max_branches": 8, "retention_days": 45},
                "experiment": {"max_branches": 3, "retention_days": 60}
            },
            "branch_patterns": {
                "feature": r"^feature/(user|agent|pkg)-[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$",
                "bugfix": r"^bugfix/(user|agent|pkg)-[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$",
                "hotfix": r"^hotfix/(user|agent|pkg)-[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$",
                "experiment": r"^experiment/(user|agent|pkg)-[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$"
            },
            "auto_cleanup_enabled": True,
            "notification_enabled": True
        }
        
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = {**default_config, **json.load(f)}
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save branch isolation configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def run_git_command(self, args: List[str]) -> Tuple[int, str, str]:
        """Run a git command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                ['git'] + args,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return 1, "", "Git command timed out"
        except Exception as e:
            return 1, "", str(e)
    
    def validate_branch_name(self, branch_name: str) -> Tuple[bool, str, Dict[str, str]]:
        """
        Validate branch name against isolation rules.
        
        Returns:
            (is_valid, error_message, parsed_info)
        """
        # Parse branch name
        parts = branch_name.split('/')
        if len(parts) < 3:
            return False, "Branch name must follow pattern: <type>/<scope>/<description>", {}
        
        branch_type = parts[0]
        scope_part = parts[1]
        description = '/'.join(parts[2:])  # Handle nested descriptions
        
        # Validate type
        if branch_type not in self.VALID_TYPES:
            return False, f"Invalid branch type '{branch_type}'. Must be one of: {', '.join(self.VALID_TYPES)}", {}
        
        # Validate scope format
        scope_pattern = r"^(user|agent|pkg)-[a-zA-Z0-9_-]+$"
        if not re.match(scope_pattern, scope_part):
            return False, f"Invalid scope format '{scope_part}'. Must match: (user|agent|pkg)-<identifier>", {}
        
        scope_type, scope_id = scope_part.split('-', 1)
        
        # Validate against configured patterns
        if branch_type in self.config["branch_patterns"]:
            pattern = self.config["branch_patterns"][branch_type]
            if not re.match(pattern, branch_name):
                return False, f"Branch name doesn't match required pattern for {branch_type}", {}
        
        # Validate description
        if not re.match(r"^[a-zA-Z0-9_/-]+$", description):
            return False, "Description can only contain alphanumeric characters, underscores, hyphens, and slashes", {}
        
        return True, "", {
            "type": branch_type,
            "scope_type": scope_type,
            "scope_id": scope_id,
            "description": description
        }
    
    def get_all_branches(self) -> List[BranchInfo]:
        """Get information about all branches."""
        exit_code, stdout, stderr = self.run_git_command(['branch', '-a', '--format=%(refname:short)|%(committerdate:iso)|%(authorname)'])
        
        if exit_code != 0:
            print(f"Error getting branches: {stderr}")
            return []
        
        branches = []
        for line in stdout.split('\n'):
            if not line.strip():
                continue
                
            parts = line.split('|')
            if len(parts) < 3:
                continue
                
            branch_name = parts[0].strip()
            if branch_name.startswith('origin/'):
                continue  # Skip remote tracking branches
                
            try:
                commit_date = datetime.fromisoformat(parts[1].replace(' +', '+'))
            except ValueError:
                commit_date = datetime.now()
                
            author = parts[2].strip()
            
            # Check if merged
            exit_code, _, _ = self.run_git_command(['merge-base', '--is-ancestor', branch_name, 'main'])
            is_merged = exit_code == 0 and branch_name != 'main'
            
            # Parse branch info
            is_valid, _, parsed = self.validate_branch_name(branch_name)
            
            branches.append(BranchInfo(
                name=branch_name,
                type=parsed.get("type", "unknown"),
                scope=parsed.get("scope_type", "unknown"),
                description=parsed.get("description", ""),
                created=commit_date,  # Approximation
                last_commit=commit_date,
                author=author,
                is_merged=is_merged
            ))
        
        return branches
    
    def check_branch_limits(self, scope_type: str, scope_id: str, branch_type: str) -> Tuple[bool, str]:
        """Check if creating a new branch would violate limits."""
        branches = self.get_all_branches()
        
        # Count branches for this scope
        scope_branches = [
            b for b in branches
            if b.scope == scope_type and scope_id in b.name and not b.is_merged
        ]
        
        # Get limits from config
        if scope_type in self.config["isolation_scopes"]:
            max_branches = self.config["isolation_scopes"][scope_type]["max_branches"]
        else:
            max_branches = self.MAX_BRANCHES_PER_SCOPE
        
        if len(scope_branches) >= max_branches:
            return False, f"Branch limit exceeded for {scope_type}-{scope_id}: {len(scope_branches)}/{max_branches}"
        
        # Special check for experiment branches
        if branch_type == "experiment":
            experiment_branches = [b for b in branches if b.type == "experiment" and not b.is_merged]
            if len(experiment_branches) >= self.MAX_EXPERIMENT_BRANCHES:
                return False, f"Global experiment branch limit exceeded: {len(experiment_branches)}/{self.MAX_EXPERIMENT_BRANCHES}"
        
        return True, ""
    
    def create_isolated_branch(self, branch_name: str, from_branch: str = "main") -> bool:
        """Create a new isolated branch with validation."""
        # Validate branch name
        is_valid, error_msg, parsed = self.validate_branch_name(branch_name)
        if not is_valid:
            print(f"❌ Invalid branch name: {error_msg}")
            return False
        
        # Check branch limits
        limits_ok, limit_error = self.check_branch_limits(
            parsed["scope_type"], 
            parsed["scope_id"], 
            parsed["type"]
        )
        if not limits_ok:
            print(f"❌ {limit_error}")
            return False
        
        # Check if branch already exists
        exit_code, _, _ = self.run_git_command(['show-ref', '--verify', '--quiet', f'refs/heads/{branch_name}'])
        if exit_code == 0:
            print(f"❌ Branch '{branch_name}' already exists")
            return False
        
        # Create the branch
        exit_code, stdout, stderr = self.run_git_command(['checkout', '-b', branch_name, from_branch])
        if exit_code != 0:
            print(f"❌ Failed to create branch: {stderr}")
            return False
        
        print(f"✅ Created isolated branch: {branch_name}")
        
        # Set up branch-specific configuration
        self.setup_branch_environment(branch_name, parsed)
        
        return True
    
    def setup_branch_environment(self, branch_name: str, parsed_info: Dict[str, str]):
        """Set up isolated environment for the branch."""
        scope_type = parsed_info["scope_type"]
        scope_id = parsed_info["scope_id"]
        
        # Create branch-specific environment variables
        env_file = self.config_dir / f"branch_env_{scope_type}_{scope_id}.env"
        with open(env_file, 'w') as f:
            f.write(f"# Environment for {branch_name}\n")
            f.write(f"BRANCH_ISOLATION_SCOPE={scope_type}\n")
            f.write(f"BRANCH_ISOLATION_ID={scope_id}\n")
            f.write(f"BRANCH_ISOLATION_NAME={branch_name}\n")
            f.write(f"BRANCH_ISOLATION_CREATED={datetime.now().isoformat()}\n")
        
        print(f"📁 Created environment file: {env_file}")
    
    def cleanup_stale_branches(self, dry_run: bool = False) -> List[str]:
        """Clean up stale and merged branches."""
        if not self.config.get("auto_cleanup_enabled", True):
            print("🔒 Auto cleanup is disabled")
            return []
        
        branches = self.get_all_branches()
        cleanup_candidates = []
        
        now = datetime.now()
        
        for branch in branches:
            # Skip protected branches
            if branch.name in self.config["protected_branches"]:
                continue
            
            # Skip current branch
            exit_code, current_branch, _ = self.run_git_command(['branch', '--show-current'])
            if exit_code == 0 and branch.name == current_branch:
                continue
            
            should_cleanup = False
            reason = ""
            
            # Check if merged and old enough
            if branch.is_merged:
                days_since_merge = (now - branch.last_commit).days
                if days_since_merge >= self.MERGED_BRANCH_CLEANUP_DAYS:
                    should_cleanup = True
                    reason = f"merged {days_since_merge} days ago"
            
            # Check if stale
            else:
                # Get retention days for scope
                retention_days = self.BRANCH_RETENTION_DAYS
                if branch.scope in self.config["isolation_scopes"]:
                    retention_days = self.config["isolation_scopes"][branch.scope]["retention_days"]
                
                days_since_commit = (now - branch.last_commit).days
                if days_since_commit >= retention_days:
                    should_cleanup = True
                    reason = f"stale for {days_since_commit} days (limit: {retention_days})"
            
            if should_cleanup:
                cleanup_candidates.append((branch.name, reason))
        
        if not cleanup_candidates:
            print("✅ No branches need cleanup")
            return []
        
        print(f"🧹 Found {len(cleanup_candidates)} branches for cleanup:")
        
        cleaned_branches = []
        for branch_name, reason in cleanup_candidates:
            print(f"  - {branch_name} ({reason})")
            
            if not dry_run:
                exit_code, _, stderr = self.run_git_command(['branch', '-d', branch_name])
                if exit_code == 0:
                    cleaned_branches.append(branch_name)
                    print(f"    ✅ Deleted")
                else:
                    # Try force delete if normal delete fails
                    exit_code, _, stderr2 = self.run_git_command(['branch', '-D', branch_name])
                    if exit_code == 0:
                        cleaned_branches.append(branch_name)
                        print(f"    ✅ Force deleted")
                    else:
                        print(f"    ❌ Failed to delete: {stderr2}")
        
        if dry_run:
            print("🔍 Dry run - no branches were actually deleted")
        
        return cleaned_branches
    
    def validate_current_branch(self) -> Tuple[bool, str]:
        """Validate the current branch against isolation rules."""
        exit_code, current_branch, stderr = self.run_git_command(['branch', '--show-current'])
        
        if exit_code != 0:
            return False, f"Cannot determine current branch: {stderr}"
        
        if not current_branch:
            return False, "Not on any branch (detached HEAD)"
        
        # Skip validation for protected branches
        if current_branch in self.config["protected_branches"]:
            return True, f"On protected branch: {current_branch}"
        
        is_valid, error_msg, _ = self.validate_branch_name(current_branch)
        if not is_valid:
            return False, f"Current branch '{current_branch}' violates isolation rules: {error_msg}"
        
        return True, f"Current branch '{current_branch}' follows isolation rules"
    
    def list_isolated_branches(self, scope_filter: str = None):
        """List all branches with isolation information."""
        branches = self.get_all_branches()
        
        print("🌿 Branch Isolation Status:")
        print("=" * 80)
        
        for branch in branches:
            # Apply scope filter
            if scope_filter and scope_filter not in branch.name:
                continue
            
            status_icon = "✅" if branch.name not in self.config["protected_branches"] else "🔒"
            if branch.is_merged:
                status_icon = "🔀"
            
            age_days = (datetime.now() - branch.last_commit).days
            
            print(f"{status_icon} {branch.name}")
            print(f"    Type: {branch.type} | Scope: {branch.scope} | Age: {age_days} days")
            print(f"    Author: {branch.author} | Last commit: {branch.last_commit.strftime('%Y-%m-%d')}")
            
            if branch.is_merged:
                print(f"    Status: MERGED")
            elif age_days > 30:
                print(f"    Status: STALE (>{age_days} days)")
            else:
                print(f"    Status: ACTIVE")
            print()


def main():
    """Main entry point for the branch isolation manager."""
    parser = argparse.ArgumentParser(description="Git Branch Isolation Manager")
    parser.add_argument("--repo-root", help="Repository root directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create branch command
    create_parser = subparsers.add_parser("create", help="Create an isolated branch")
    create_parser.add_argument("branch_name", help="Name of the branch to create")
    create_parser.add_argument("--from", dest="from_branch", default="main", help="Base branch to create from")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate branch name or current branch")
    validate_parser.add_argument("branch_name", nargs="?", help="Branch name to validate (current branch if not specified)")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up stale branches")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned up without doing it")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List branches with isolation info")
    list_parser.add_argument("--scope", help="Filter by scope (user, agent, pkg)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    manager = GitBranchIsolationManager(args.repo_root)
    
    try:
        if args.command == "create":
            success = manager.create_isolated_branch(args.branch_name, args.from_branch)
            return 0 if success else 1
        
        elif args.command == "validate":
            if args.branch_name:
                is_valid, message, _ = manager.validate_branch_name(args.branch_name)
            else:
                is_valid, message = manager.validate_current_branch()
            
            print(message)
            return 0 if is_valid else 1
        
        elif args.command == "cleanup":
            cleaned = manager.cleanup_stale_branches(args.dry_run)
            if cleaned:
                print(f"\n✅ Cleaned up {len(cleaned)} branches")
            return 0
        
        elif args.command == "list":
            manager.list_isolated_branches(args.scope)
            return 0
    
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Buck2 Git Integration for Pynomaly
Advanced Git integration for change detection and commit-based testing.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Import our change detector
from buck2_change_detector import Buck2ChangeDetector, ChangeAnalysis
from buck2_incremental_test import Buck2IncrementalTestRunner, TestRunSummary

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CommitInfo:
    """Information about a Git commit."""

    hash: str
    author: str
    date: str
    message: str
    files_changed: List[str]


@dataclass
class BranchInfo:
    """Information about a Git branch."""

    name: str
    base_commit: str
    head_commit: str
    commits: List[CommitInfo]
    total_files_changed: Set[str]


class Buck2GitIntegration:
    """Enhanced Git integration for Buck2 incremental testing."""

    def __init__(self, repo_root: Path = None):
        self.repo_root = repo_root or Path.cwd()
        self.change_detector = Buck2ChangeDetector(repo_root)
        self.test_runner = Buck2IncrementalTestRunner(repo_root)

    def get_current_branch(self) -> str:
        """Get the current Git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get current branch: {e}")
            return "unknown"

    def get_main_branch(self) -> str:
        """Get the main branch name (main or master)."""
        try:
            # Check if main exists
            subprocess.run(
                ["git", "rev-parse", "--verify", "origin/main"],
                capture_output=True,
                check=True,
                cwd=self.repo_root,
            )
            return "main"
        except subprocess.CalledProcessError:
            try:
                # Check if master exists
                subprocess.run(
                    ["git", "rev-parse", "--verify", "origin/master"],
                    capture_output=True,
                    check=True,
                    cwd=self.repo_root,
                )
                return "master"
            except subprocess.CalledProcessError:
                logger.warning("Neither main nor master branch found, using 'main'")
                return "main"

    def get_commit_info(self, commit_hash: str) -> CommitInfo:
        """Get detailed information about a commit."""
        try:
            # Get commit details
            result = subprocess.run(
                ["git", "show", "--format=%H%n%an%n%ad%n%s", "--no-patch", commit_hash],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )

            lines = result.stdout.strip().split("\n")
            hash_val = lines[0]
            author = lines[1]
            date = lines[2]
            message = lines[3] if len(lines) > 3 else ""

            # Get changed files
            files_result = subprocess.run(
                [
                    "git",
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    commit_hash,
                ],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )

            files_changed = [
                f.strip() for f in files_result.stdout.split("\n") if f.strip()
            ]

            return CommitInfo(
                hash=hash_val,
                author=author,
                date=date,
                message=message,
                files_changed=files_changed,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get commit info for {commit_hash}: {e}")
            return CommitInfo(
                hash=commit_hash,
                author="unknown",
                date="unknown",
                message="unknown",
                files_changed=[],
            )

    def get_branch_info(
        self, branch: str = None, base_branch: str = None
    ) -> BranchInfo:
        """Get comprehensive information about a branch."""
        if branch is None:
            branch = self.get_current_branch()
        if base_branch is None:
            base_branch = self.get_main_branch()

        try:
            # Get base commit (merge base)
            base_result = subprocess.run(
                ["git", "merge-base", f"origin/{base_branch}", branch],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )
            base_commit = base_result.stdout.strip()

            # Get head commit
            head_result = subprocess.run(
                ["git", "rev-parse", branch],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )
            head_commit = head_result.stdout.strip()

            # Get commits in the branch
            commits_result = subprocess.run(
                ["git", "rev-list", f"{base_commit}..{head_commit}"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )

            commit_hashes = [
                c.strip() for c in commits_result.stdout.split("\n") if c.strip()
            ]
            commits = [self.get_commit_info(commit) for commit in commit_hashes]

            # Get all files changed in the branch
            files_result = subprocess.run(
                ["git", "diff", "--name-only", f"{base_commit}..{head_commit}"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )

            total_files_changed = set(
                f.strip() for f in files_result.stdout.split("\n") if f.strip()
            )

            return BranchInfo(
                name=branch,
                base_commit=base_commit,
                head_commit=head_commit,
                commits=commits,
                total_files_changed=total_files_changed,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get branch info for {branch}: {e}")
            return BranchInfo(
                name=branch,
                base_commit="unknown",
                head_commit="unknown",
                commits=[],
                total_files_changed=set(),
            )

    def test_commit_range(
        self, base_commit: str, target_commit: str, **kwargs
    ) -> TestRunSummary:
        """Test a specific commit range."""
        logger.info(f"Testing commit range {base_commit}..{target_commit}")
        return self.test_runner.run_incremental_tests(
            base_commit, target_commit, **kwargs
        )

    def test_current_branch(self, base_branch: str = None, **kwargs) -> TestRunSummary:
        """Test the current branch against base branch."""
        branch_info = self.get_branch_info(base_branch=base_branch)
        logger.info(
            f"Testing branch '{branch_info.name}' against '{base_branch or self.get_main_branch()}'"
        )
        return self.test_runner.run_incremental_tests(
            branch_info.base_commit, branch_info.head_commit, **kwargs
        )

    def test_per_commit(
        self, base_branch: str = None, **kwargs
    ) -> Dict[str, TestRunSummary]:
        """Test each commit individually in the current branch."""
        branch_info = self.get_branch_info(base_branch=base_branch)
        results = {}

        if not branch_info.commits:
            logger.info("No commits to test in current branch")
            return results

        logger.info(f"Testing {len(branch_info.commits)} commits individually")

        # Test each commit
        prev_commit = branch_info.base_commit
        for commit in reversed(branch_info.commits):  # Test in chronological order
            logger.info(f"Testing commit {commit.hash[:8]}: {commit.message}")

            try:
                summary = self.test_runner.run_incremental_tests(
                    prev_commit, commit.hash, **kwargs
                )
                results[commit.hash] = summary

                # If this commit has failures, log them
                if summary.failed_targets > 0:
                    logger.warning(
                        f"Commit {commit.hash[:8]} has {summary.failed_targets} failing targets"
                    )

                prev_commit = commit.hash

            except Exception as e:
                logger.error(f"Failed to test commit {commit.hash[:8]}: {e}")

        return results

    def find_breaking_commit(self, base_branch: str = None, **kwargs) -> Optional[str]:
        """Find the first commit that introduced test failures using binary search."""
        branch_info = self.get_branch_info(base_branch=base_branch)

        if not branch_info.commits:
            logger.info("No commits to search in current branch")
            return None

        logger.info(
            f"Binary searching for breaking commit in {len(branch_info.commits)} commits"
        )

        # Test the full range first
        full_summary = self.test_runner.run_incremental_tests(
            branch_info.base_commit, branch_info.head_commit, **kwargs
        )

        if full_summary.failed_targets == 0:
            logger.info("No test failures found in branch")
            return None

        # Binary search for the breaking commit
        commits = list(reversed(branch_info.commits))  # Chronological order
        left, right = 0, len(commits) - 1
        breaking_commit = None

        while left <= right:
            mid = (left + right) // 2
            test_commit = commits[mid].hash

            logger.info(f"Testing commit {mid + 1}/{len(commits)}: {test_commit[:8]}")

            try:
                summary = self.test_runner.run_incremental_tests(
                    branch_info.base_commit, test_commit, **kwargs
                )

                if summary.failed_targets > 0:
                    # Failure found, search in earlier commits
                    breaking_commit = test_commit
                    right = mid - 1
                else:
                    # No failure, search in later commits
                    left = mid + 1

            except Exception as e:
                logger.error(f"Failed to test commit {test_commit[:8]}: {e}")
                left = mid + 1

        if breaking_commit:
            commit_info = self.get_commit_info(breaking_commit)
            logger.info(
                f"Breaking commit found: {breaking_commit[:8]} - {commit_info.message}"
            )

        return breaking_commit

    def setup_git_hooks(self):
        """Set up Git hooks for automatic testing."""
        hooks_dir = self.repo_root / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)

        # Pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
# Buck2 pre-commit hook - test staged changes
set -e

echo "Running Buck2 incremental tests on staged changes..."

# Stash unstaged changes
git stash push -k -u -m "pre-commit-$(date +%s)"

# Run tests on staged changes
python scripts/buck2_git_integration.py test-staged --fail-fast

# Pop stash if it exists
STASH_NAME="pre-commit-$(date +%s)"
if git stash list | grep -q "$STASH_NAME"; then
    git stash pop
fi
"""

        with open(pre_commit_hook, "w") as f:
            f.write(pre_commit_content)
        pre_commit_hook.chmod(0o755)

        # Pre-push hook
        pre_push_hook = hooks_dir / "pre-push"
        pre_push_content = """#!/bin/bash
# Buck2 pre-push hook - test current branch
set -e

echo "Running Buck2 incremental tests on current branch..."
python scripts/buck2_git_integration.py test-branch --fail-fast
"""

        with open(pre_push_hook, "w") as f:
            f.write(pre_push_content)
        pre_push_hook.chmod(0o755)

        logger.info("Git hooks installed successfully")

    def test_staged_changes(self, **kwargs) -> Optional[TestRunSummary]:
        """Test only staged changes."""
        try:
            # Get staged files
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )

            staged_files = [f.strip() for f in result.stdout.split("\n") if f.strip()]

            if not staged_files:
                logger.info("No staged changes to test")
                return None

            logger.info(f"Testing {len(staged_files)} staged files")

            # Create a temporary commit to test against
            temp_commit = subprocess.run(
                ["git", "stash", "create"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            ).stdout.strip()

            if temp_commit:
                return self.test_runner.run_incremental_tests(
                    "HEAD", temp_commit, **kwargs
                )
            else:
                # Fallback to testing current changes
                return self.test_runner.run_incremental_tests(
                    "HEAD~1", "HEAD", **kwargs
                )

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to test staged changes: {e}")
            return None


def main():
    """Main entry point for Buck2 Git integration."""
    parser = argparse.ArgumentParser(description="Buck2 Git Integration")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test branch command
    branch_parser = subparsers.add_parser("test-branch", help="Test current branch")
    branch_parser.add_argument("--base", help="Base branch name")
    branch_parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure"
    )
    branch_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be run"
    )

    # Test staged command
    staged_parser = subparsers.add_parser("test-staged", help="Test staged changes")
    staged_parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure"
    )
    staged_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be run"
    )

    # Test per-commit command
    commit_parser = subparsers.add_parser(
        "test-commits", help="Test each commit individually"
    )
    commit_parser.add_argument("--base", help="Base branch name")
    commit_parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure"
    )

    # Find breaking commit command
    bisect_parser = subparsers.add_parser("find-breaking", help="Find breaking commit")
    bisect_parser.add_argument("--base", help="Base branch name")

    # Setup hooks command
    hooks_parser = subparsers.add_parser("setup-hooks", help="Setup Git hooks")

    # Branch info command
    info_parser = subparsers.add_parser("branch-info", help="Show branch information")
    info_parser.add_argument("--base", help="Base branch name")

    # Global options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return

    # Initialize Git integration
    git_integration = Buck2GitIntegration()

    try:
        if args.command == "test-branch":
            summary = git_integration.test_current_branch(
                base_branch=args.base, fail_fast=args.fail_fast, dry_run=args.dry_run
            )
            git_integration.test_runner.print_summary(summary)
            sys.exit(1 if summary.failed_targets > 0 else 0)

        elif args.command == "test-staged":
            summary = git_integration.test_staged_changes(
                fail_fast=args.fail_fast, dry_run=args.dry_run
            )
            if summary:
                git_integration.test_runner.print_summary(summary)
                sys.exit(1 if summary.failed_targets > 0 else 0)
            else:
                print("No staged changes to test")

        elif args.command == "test-commits":
            results = git_integration.test_per_commit(
                base_branch=args.base, fail_fast=args.fail_fast
            )

            total_failures = sum(s.failed_targets for s in results.values())
            print(f"\nTested {len(results)} commits, {total_failures} total failures")

            for commit_hash, summary in results.items():
                commit_info = git_integration.get_commit_info(commit_hash)
                status = "✓" if summary.failed_targets == 0 else "✗"
                print(f"{status} {commit_hash[:8]} - {commit_info.message[:60]}")

            sys.exit(1 if total_failures > 0 else 0)

        elif args.command == "find-breaking":
            breaking_commit = git_integration.find_breaking_commit(
                base_branch=args.base
            )
            if breaking_commit:
                print(f"Breaking commit: {breaking_commit}")
                sys.exit(1)
            else:
                print("No breaking commit found")
                sys.exit(0)

        elif args.command == "setup-hooks":
            git_integration.setup_git_hooks()
            print("Git hooks installed successfully")

        elif args.command == "branch-info":
            branch_info = git_integration.get_branch_info(base_branch=args.base)
            print(f"Branch: {branch_info.name}")
            print(f"Base commit: {branch_info.base_commit}")
            print(f"Head commit: {branch_info.head_commit}")
            print(f"Commits: {len(branch_info.commits)}")
            print(f"Files changed: {len(branch_info.total_files_changed)}")

            if branch_info.commits:
                print("\nCommits:")
                for commit in branch_info.commits[:10]:
                    print(f"  {commit.hash[:8]} - {commit.message[:60]}")
                if len(branch_info.commits) > 10:
                    print(f"  ... and {len(branch_info.commits) - 10} more")

    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

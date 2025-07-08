#!/usr/bin/env python3
"""
Buck2 Comprehensive Workflow Script for Pynomaly
Main entry point for all Buck2 incremental testing operations.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
import argparse
import logging

# Import our tools
from buck2_change_detector import Buck2ChangeDetector, ChangeAnalysis
from buck2_incremental_test import Buck2IncrementalTestRunner, TestRunSummary
from buck2_git_integration import Buck2GitIntegration, CommitInfo, BranchInfo
from buck2_impact_analyzer import Buck2ImpactAnalyzer, ImpactAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WorkflowConfig:
    """Configuration for Buck2 workflow."""
    strategy: str = "auto"  # auto, minimal, standard, comprehensive, full
    fail_fast: bool = False
    parallel_jobs: int = 4
    timeout: int = 300
    dry_run: bool = False
    save_results: bool = True
    base_commit: str = "HEAD~1"
    target_commit: str = "HEAD"

class Buck2Workflow:
    """Comprehensive workflow orchestrator for Buck2 incremental testing."""

    def __init__(self, config: WorkflowConfig = None):
        self.config = config or WorkflowConfig()
        self.repo_root = Path.cwd()

        # Initialize components
        self.change_detector = Buck2ChangeDetector(self.repo_root)
        self.test_runner = Buck2IncrementalTestRunner(self.repo_root, self.config.dry_run)
        self.git_integration = Buck2GitIntegration(self.repo_root)
        self.impact_analyzer = Buck2ImpactAnalyzer(self.repo_root)

        # Configure test runner
        self.test_runner.parallel_jobs = self.config.parallel_jobs
        self.test_runner.test_timeout = self.config.timeout

    def run_standard_workflow(self) -> Dict:
        """Run the standard Buck2 incremental testing workflow."""
        logger.info("Starting Buck2 standard workflow")
        start_time = time.time()

        results = {
            "workflow_type": "standard",
            "config": asdict(self.config),
            "start_time": start_time,
        }

        try:
            # Step 1: Analyze changes
            logger.info("Step 1: Analyzing changes...")
            change_analysis = self.change_detector.analyze_changes(
                self.config.base_commit,
                self.config.target_commit
            )
            results["change_analysis"] = asdict(change_analysis)

            if not change_analysis.changed_files:
                logger.info("No changes detected - skipping tests")
                results["status"] = "no_changes"
                results["duration"] = time.time() - start_time
                return results

            # Step 2: Determine strategy
            if self.config.strategy == "auto":
                logger.info("Step 2: Analyzing impact and determining strategy...")
                impact_result = self.impact_analyzer.analyze_impact(
                    self.config.base_commit,
                    self.config.target_commit
                )
                strategy = impact_result.test_strategy.priority
                results["impact_analysis"] = asdict(impact_result)
            else:
                strategy = self.config.strategy
                results["impact_analysis"] = {"strategy_override": strategy}

            logger.info(f"Using test strategy: {strategy}")

            # Step 3: Run tests based on strategy
            logger.info("Step 3: Running incremental tests...")
            test_summary = self._run_tests_by_strategy(strategy, change_analysis)
            results["test_summary"] = asdict(test_summary)

            # Step 4: Save results if requested
            if self.config.save_results:
                self._save_workflow_results(results)

            results["status"] = "completed"
            results["duration"] = time.time() - start_time

            return results

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["duration"] = time.time() - start_time
            return results

    def run_commit_validation_workflow(self) -> Dict:
        """Run workflow to validate each commit in a branch."""
        logger.info("Starting Buck2 commit validation workflow")
        start_time = time.time()

        results = {
            "workflow_type": "commit_validation",
            "config": asdict(self.config),
            "start_time": start_time,
            "commit_results": {}
        }

        try:
            # Get branch information
            branch_info = self.git_integration.get_branch_info()
            results["branch_info"] = asdict(branch_info)

            if not branch_info.commits:
                logger.info("No commits to validate")
                results["status"] = "no_commits"
                results["duration"] = time.time() - start_time
                return results

            # Test each commit
            logger.info(f"Validating {len(branch_info.commits)} commits")

            prev_commit = branch_info.base_commit
            for i, commit in enumerate(reversed(branch_info.commits)):
                logger.info(f"Validating commit {i+1}/{len(branch_info.commits)}: {commit.hash[:8]}")

                try:
                    # Analyze this specific commit
                    commit_analysis = self.change_detector.analyze_changes(prev_commit, commit.hash)

                    if commit_analysis.changed_files:
                        # Run incremental tests for this commit
                        test_summary = self.test_runner.run_incremental_tests(
                            prev_commit,
                            commit.hash,
                            fail_fast=self.config.fail_fast
                        )

                        results["commit_results"][commit.hash] = {
                            "change_analysis": asdict(commit_analysis),
                            "test_summary": asdict(test_summary),
                            "status": "passed" if test_summary.failed_targets == 0 else "failed"
                        }
                    else:
                        results["commit_results"][commit.hash] = {
                            "status": "no_changes"
                        }

                    prev_commit = commit.hash

                except Exception as e:
                    logger.error(f"Failed to validate commit {commit.hash[:8]}: {e}")
                    results["commit_results"][commit.hash] = {
                        "status": "error",
                        "error": str(e)
                    }

            # Summary
            total_commits = len(results["commit_results"])
            failed_commits = sum(1 for r in results["commit_results"].values()
                               if r.get("status") == "failed")

            results["summary"] = {
                "total_commits": total_commits,
                "failed_commits": failed_commits,
                "success_rate": (total_commits - failed_commits) / total_commits if total_commits > 0 else 1.0
            }

            results["status"] = "completed"
            results["duration"] = time.time() - start_time

            return results

        except Exception as e:
            logger.error(f"Commit validation workflow failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["duration"] = time.time() - start_time
            return results

    def run_bisect_workflow(self) -> Dict:
        """Run workflow to find the first failing commit."""
        logger.info("Starting Buck2 bisect workflow")
        start_time = time.time()

        results = {
            "workflow_type": "bisect",
            "config": asdict(self.config),
            "start_time": start_time,
        }

        try:
            breaking_commit = self.git_integration.find_breaking_commit(
                fail_fast=self.config.fail_fast
            )

            if breaking_commit:
                commit_info = self.git_integration.get_commit_info(breaking_commit)
                results["breaking_commit"] = {
                    "hash": breaking_commit,
                    "info": asdict(commit_info)
                }
                results["status"] = "breaking_commit_found"
            else:
                results["status"] = "no_breaking_commit"

            results["duration"] = time.time() - start_time
            return results

        except Exception as e:
            logger.error(f"Bisect workflow failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["duration"] = time.time() - start_time
            return results

    def _run_tests_by_strategy(self, strategy: str, change_analysis: ChangeAnalysis) -> TestRunSummary:
        """Run tests based on the specified strategy."""
        if strategy == "minimal":
            # Only run tests for directly changed files
            return self.test_runner.run_incremental_tests(
                self.config.base_commit,
                self.config.target_commit,
                include_build=False,
                fail_fast=True
            )

        elif strategy == "standard":
            # Run incremental tests as determined by change analysis
            return self.test_runner.run_incremental_tests(
                self.config.base_commit,
                self.config.target_commit,
                include_build=True,
                fail_fast=self.config.fail_fast
            )

        elif strategy == "comprehensive":
            # Run incremental tests plus additional safety tests
            summary = self.test_runner.run_incremental_tests(
                self.config.base_commit,
                self.config.target_commit,
                include_build=True,
                fail_fast=self.config.fail_fast
            )

            # Add integration tests if not already included
            if "test-integration" not in summary.change_analysis.test_targets:
                logger.info("Running additional integration tests")
                # This would run additional tests - simplified for now

            return summary

        elif strategy == "full":
            # Run all tests
            logger.info("Running full test suite")
            # This would run all tests - for now, run comprehensive
            return self.test_runner.run_incremental_tests(
                self.config.base_commit,
                self.config.target_commit,
                include_build=True,
                fail_fast=False
            )

        else:
            # Default to standard
            return self.test_runner.run_incremental_tests(
                self.config.base_commit,
                self.config.target_commit,
                include_build=True,
                fail_fast=self.config.fail_fast
            )

    def _save_workflow_results(self, results: Dict) -> Path:
        """Save workflow results to file."""
        timestamp = int(time.time())
        output_file = self.repo_root / f"buck2_workflow_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Workflow results saved to {output_file}")
        return output_file

    def print_workflow_summary(self, results: Dict):
        """Print a summary of workflow results."""
        print(f"\n=== Buck2 Workflow Summary ===")
        print(f"Type: {results['workflow_type']}")
        print(f"Status: {results['status']}")
        print(f"Duration: {results['duration']:.2f}s")

        if results.get("change_analysis"):
            ca = results["change_analysis"]
            print(f"Files changed: {len(ca['changed_files'])}")
            print(f"Targets affected: {len(ca['affected_targets'])}")

        if results.get("test_summary"):
            ts = results["test_summary"]
            print(f"Tests run: {ts['total_targets']}")
            print(f"Successful: {ts['successful_targets']}")
            print(f"Failed: {ts['failed_targets']}")

        if results.get("impact_analysis"):
            ia = results["impact_analysis"]
            if "risk_assessment" in ia:
                print(f"Risk level: {ia['risk_assessment']['level']}")
                print(f"Strategy used: {ia['test_strategy']['priority']}")

        if results.get("summary"):  # For commit validation
            s = results["summary"]
            print(f"Commits validated: {s['total_commits']}")
            print(f"Failed commits: {s['failed_commits']}")
            print(f"Success rate: {s['success_rate']:.1%}")

        if results.get("breaking_commit"):  # For bisect
            bc = results["breaking_commit"]
            print(f"Breaking commit: {bc['hash'][:8]}")
            print(f"Message: {bc['info']['message']}")

def main():
    """Main entry point for Buck2 workflow."""
    parser = argparse.ArgumentParser(description="Buck2 Comprehensive Workflow")

    # Workflow type
    subparsers = parser.add_subparsers(dest="workflow", help="Workflow type")

    # Standard workflow
    standard_parser = subparsers.add_parser("standard", help="Standard incremental testing")
    standard_parser.add_argument("--strategy", choices=["auto", "minimal", "standard", "comprehensive", "full"],
                                default="auto", help="Test strategy")
    standard_parser.add_argument("--base", default="HEAD~1", help="Base commit")
    standard_parser.add_argument("--target", default="HEAD", help="Target commit")

    # Commit validation workflow
    validate_parser = subparsers.add_parser("validate-commits", help="Validate each commit")
    validate_parser.add_argument("--base-branch", help="Base branch name")

    # Bisect workflow
    bisect_parser = subparsers.add_parser("bisect", help="Find breaking commit")
    bisect_parser.add_argument("--base-branch", help="Base branch name")

    # Branch workflow
    branch_parser = subparsers.add_parser("branch", help="Test current branch")
    branch_parser.add_argument("--strategy", choices=["auto", "minimal", "standard", "comprehensive", "full"],
                              default="auto", help="Test strategy")

    # Global options
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run")
    parser.add_argument("--jobs", "-j", type=int, default=4, help="Parallel jobs")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per target")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.workflow:
        parser.print_help()
        return

    # Configure workflow
    config = WorkflowConfig(
        fail_fast=args.fail_fast,
        parallel_jobs=args.jobs,
        timeout=args.timeout,
        dry_run=args.dry_run,
        save_results=not args.no_save
    )

    if hasattr(args, 'strategy'):
        config.strategy = args.strategy
    if hasattr(args, 'base'):
        config.base_commit = args.base
    if hasattr(args, 'target'):
        config.target_commit = args.target

    # Initialize workflow
    workflow = Buck2Workflow(config)

    try:
        # Run appropriate workflow
        if args.workflow == "standard":
            results = workflow.run_standard_workflow()
        elif args.workflow == "validate-commits":
            results = workflow.run_commit_validation_workflow()
        elif args.workflow == "bisect":
            results = workflow.run_bisect_workflow()
        elif args.workflow == "branch":
            # Test current branch
            branch_info = workflow.git_integration.get_branch_info()
            config.base_commit = branch_info.base_commit
            config.target_commit = branch_info.head_commit
            workflow.config = config
            results = workflow.run_standard_workflow()
        else:
            logger.error(f"Unknown workflow: {args.workflow}")
            sys.exit(1)

        # Print summary
        workflow.print_workflow_summary(results)

        # Exit with appropriate code
        if results["status"] == "failed":
            sys.exit(1)
        elif results.get("test_summary", {}).get("failed_targets", 0) > 0:
            sys.exit(1)
        elif results.get("summary", {}).get("failed_commits", 0) > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

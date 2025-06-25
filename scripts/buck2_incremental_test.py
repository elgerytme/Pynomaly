#!/usr/bin/env python3
"""
Buck2 Incremental Test Runner for Pynomaly
Runs only tests affected by changes for faster feedback loops.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our change detector
from buck2_change_detector import Buck2ChangeDetector, ChangeAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Result of running a Buck2 test target."""
    target: str
    success: bool
    duration: float
    stdout: str
    stderr: str
    command: str

@dataclass
class TestRunSummary:
    """Summary of an incremental test run."""
    total_targets: int
    successful_targets: int
    failed_targets: int
    skipped_targets: int
    total_duration: float
    results: List[TestResult]
    change_analysis: ChangeAnalysis
    run_metadata: Dict

class Buck2IncrementalTestRunner:
    """Runs Buck2 tests incrementally based on change analysis."""
    
    def __init__(self, repo_root: Path = None, dry_run: bool = False):
        self.repo_root = repo_root or Path.cwd()
        self.dry_run = dry_run
        self.change_detector = Buck2ChangeDetector(repo_root)
        
        # Test execution configuration
        self.test_timeout = 300  # 5 minutes per test target
        self.parallel_jobs = min(4, os.cpu_count() or 1)
        
    def check_buck2_available(self) -> bool:
        """Check if Buck2 is available in the system."""
        try:
            result = subprocess.run(["buck2", "--version"], capture_output=True, text=True, check=True)
            logger.info(f"Buck2 version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Buck2 is not available. Please install Buck2 first.")
            return False
    
    def run_buck2_target(self, target: str, target_type: str = "test") -> TestResult:
        """Run a single Buck2 target and capture results."""
        start_time = time.time()
        
        if target_type == "test":
            command = ["buck2", "test", f":{target}", "--verbose"]
        else:
            command = ["buck2", "build", f":{target}"]
        
        command_str = " ".join(command)
        logger.info(f"Running: {command_str}")
        
        if self.dry_run:
            logger.info(f"DRY RUN: Would execute {command_str}")
            return TestResult(
                target=target,
                success=True,
                duration=0.0,
                stdout=f"DRY RUN: {command_str}",
                stderr="",
                command=command_str
            )
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
                cwd=self.repo_root
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            if success:
                logger.info(f"✓ {target} passed ({duration:.2f}s)")
            else:
                logger.error(f"✗ {target} failed ({duration:.2f}s)")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr[:500]}...")
            
            return TestResult(
                target=target,
                success=success,
                duration=duration,
                stdout=result.stdout,
                stderr=result.stderr,
                command=command_str
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"✗ {target} timed out after {duration:.2f}s")
            return TestResult(
                target=target,
                success=False,
                duration=duration,
                stdout="",
                stderr=f"Test timed out after {self.test_timeout}s",
                command=command_str
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"✗ {target} failed with exception: {e}")
            return TestResult(
                target=target,
                success=False,
                duration=duration,
                stdout="",
                stderr=str(e),
                command=command_str
            )
    
    def run_targets_parallel(self, targets: Set[str], target_type: str = "test") -> List[TestResult]:
        """Run multiple Buck2 targets in parallel."""
        if not targets:
            return []
        
        logger.info(f"Running {len(targets)} {target_type} targets with {self.parallel_jobs} parallel jobs")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
            # Submit all tasks
            future_to_target = {
                executor.submit(self.run_buck2_target, target, target_type): target
                for target in targets
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_target):
                target = future_to_target[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Exception running {target}: {e}")
                    results.append(TestResult(
                        target=target,
                        success=False,
                        duration=0.0,
                        stdout="",
                        stderr=str(e),
                        command=f"buck2 {target_type} :{target}"
                    ))
        
        return results
    
    def run_incremental_tests(self, 
                            base_commit: str = "HEAD~1", 
                            target_commit: str = "HEAD",
                            include_build: bool = True,
                            fail_fast: bool = False) -> TestRunSummary:
        """Run incremental tests based on changes."""
        start_time = time.time()
        
        # Check Buck2 availability
        if not self.dry_run and not self.check_buck2_available():
            raise RuntimeError("Buck2 is not available")
        
        # Analyze changes
        logger.info("Analyzing changes...")
        analysis = self.change_detector.analyze_changes(base_commit, target_commit)
        
        if not analysis.test_targets and not analysis.build_targets:
            logger.info("No targets affected by changes")
            return TestRunSummary(
                total_targets=0,
                successful_targets=0,
                failed_targets=0,
                skipped_targets=0,
                total_duration=0.0,
                results=[],
                change_analysis=analysis,
                run_metadata={"status": "no_targets", "dry_run": self.dry_run}
            )
        
        # Run tests
        all_results = []
        
        # Run test targets first
        if analysis.test_targets:
            logger.info(f"Running {len(analysis.test_targets)} test targets...")
            test_results = self.run_targets_parallel(analysis.test_targets, "test")
            all_results.extend(test_results)
            
            # Check for failures if fail_fast is enabled
            if fail_fast:
                failed_tests = [r for r in test_results if not r.success]
                if failed_tests:
                    logger.error(f"Stopping due to {len(failed_tests)} test failures (fail_fast=True)")
                    include_build = False
        
        # Run build targets if requested and no failures (or fail_fast disabled)
        if include_build and analysis.build_targets:
            logger.info(f"Running {len(analysis.build_targets)} build targets...")
            build_results = self.run_targets_parallel(analysis.build_targets, "build")
            all_results.extend(build_results)
        
        # Calculate summary
        total_duration = time.time() - start_time
        successful_targets = sum(1 for r in all_results if r.success)
        failed_targets = sum(1 for r in all_results if not r.success)
        total_expected = len(analysis.test_targets) + (len(analysis.build_targets) if include_build else 0)
        skipped_targets = total_expected - len(all_results)
        
        run_metadata = {
            "base_commit": base_commit,
            "target_commit": target_commit,
            "include_build": include_build,
            "fail_fast": fail_fast,
            "dry_run": self.dry_run,
            "parallel_jobs": self.parallel_jobs,
            "test_timeout": self.test_timeout,
            "run_timestamp": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True).stdout.strip(),
        }
        
        return TestRunSummary(
            total_targets=len(all_results),
            successful_targets=successful_targets,
            failed_targets=failed_targets,
            skipped_targets=skipped_targets,
            total_duration=total_duration,
            results=all_results,
            change_analysis=analysis,
            run_metadata=run_metadata
        )
    
    def save_test_results(self, summary: TestRunSummary, output_file: Path = None) -> Path:
        """Save test results to JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = self.repo_root / f"buck2_test_results_{timestamp}.json"
        
        # Prepare data for JSON serialization
        summary_dict = asdict(summary)
        
        # Convert sets in change_analysis to lists
        summary_dict["change_analysis"]["affected_targets"] = list(summary.change_analysis.affected_targets)
        summary_dict["change_analysis"]["test_targets"] = list(summary.change_analysis.test_targets)
        summary_dict["change_analysis"]["build_targets"] = list(summary.change_analysis.build_targets)
        
        with open(output_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        logger.info(f"Test results saved to {output_file}")
        return output_file
    
    def print_summary(self, summary: TestRunSummary):
        """Print a human-readable test summary."""
        print(f"\n=== Buck2 Incremental Test Results ===")
        print(f"Commit range: {summary.change_analysis.commit_range}")
        print(f"Total duration: {summary.total_duration:.2f}s")
        print(f"Total targets: {summary.total_targets}")
        print(f"Successful: {summary.successful_targets}")
        print(f"Failed: {summary.failed_targets}")
        print(f"Skipped: {summary.skipped_targets}")
        
        if summary.run_metadata.get("dry_run"):
            print("(DRY RUN - no actual tests executed)")
        
        # Show changed files
        if summary.change_analysis.changed_files:
            print(f"\nChanged files ({len(summary.change_analysis.changed_files)}):")
            for file in summary.change_analysis.changed_files[:5]:
                print(f"  - {file}")
            if len(summary.change_analysis.changed_files) > 5:
                print(f"  ... and {len(summary.change_analysis.changed_files) - 5} more")
        
        # Show failed targets
        failed_results = [r for r in summary.results if not r.success]
        if failed_results:
            print(f"\nFailed targets:")
            for result in failed_results:
                print(f"  ✗ {result.target} ({result.duration:.2f}s)")
                if result.stderr:
                    error_lines = result.stderr.split('\n')[:3]
                    for line in error_lines:
                        if line.strip():
                            print(f"    {line.strip()}")
        
        # Show successful targets
        successful_results = [r for r in summary.results if r.success]
        if successful_results and len(successful_results) <= 10:
            print(f"\nSuccessful targets:")
            for result in successful_results:
                print(f"  ✓ {result.target} ({result.duration:.2f}s)")
        elif successful_results:
            print(f"\nSuccessful targets: {len(successful_results)} targets passed")

def main():
    """Main entry point for the incremental test runner."""
    parser = argparse.ArgumentParser(description="Buck2 Incremental Test Runner")
    parser.add_argument("--base", default="HEAD~1", help="Base commit for comparison")
    parser.add_argument("--target", default="HEAD", help="Target commit for comparison")
    parser.add_argument("--no-build", action="store_true", help="Skip build targets")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument("--jobs", "-j", type=int, help="Number of parallel jobs")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per target in seconds")
    parser.add_argument("--output", type=Path, help="Output file for test results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize test runner
    runner = Buck2IncrementalTestRunner(dry_run=args.dry_run)
    
    if args.jobs:
        runner.parallel_jobs = args.jobs
    if args.timeout:
        runner.test_timeout = args.timeout
    
    try:
        # Run incremental tests
        summary = runner.run_incremental_tests(
            base_commit=args.base,
            target_commit=args.target,
            include_build=not args.no_build,
            fail_fast=args.fail_fast
        )
        
        # Print summary
        runner.print_summary(summary)
        
        # Save results if requested
        if args.output:
            runner.save_test_results(summary, args.output)
        
        # Exit with appropriate code
        if summary.failed_targets > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Test run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
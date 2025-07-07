#!/usr/bin/env python3
"""
Pytest-based Incremental Test Runner for Pynomaly
Fallback implementation that uses pytest instead of Buck2 while maintaining incremental testing logic.
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

# Import our existing change detection logic
from buck2_change_detector import Buck2ChangeDetector, ChangeAnalysis
from buck2_impact_analyzer import Buck2ImpactAnalyzer, ImpactAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PytestResult:
    """Result of running pytest."""
    command: str
    success: bool
    duration: float
    stdout: str
    stderr: str
    tests_run: int
    tests_passed: int
    tests_failed: int

@dataclass
class IncrementalPytestSummary:
    """Summary of incremental pytest run."""
    total_commands: int
    successful_commands: int
    failed_commands: int
    total_duration: float
    results: List[PytestResult]
    change_analysis: ChangeAnalysis
    run_metadata: Dict

class PytestIncrementalRunner:
    """Runs pytest incrementally based on Buck2 change analysis."""
    
    def __init__(self, repo_root: Path = None, dry_run: bool = False):
        self.repo_root = repo_root or Path.cwd()
        self.dry_run = dry_run
        self.change_detector = Buck2ChangeDetector(repo_root)
        self.impact_analyzer = Buck2ImpactAnalyzer(repo_root)
        
        # Map Buck2 targets to pytest commands
        self.target_to_pytest = {
            "test-domain": "pytest tests/domain/ -v",
            "test-application": "pytest tests/application/ -v", 
            "test-infrastructure": "pytest tests/infrastructure/ -v",
            "test-presentation": "pytest tests/presentation/ -v",
            "test-integration": "pytest tests/integration/ tests/e2e/ -v",
            "benchmarks": "pytest tests/benchmarks/ tests/performance/ -v",
            "property-tests": "pytest tests/property/ -v",
            "mutation-tests": "pytest tests/mutation/ -v",
            "security-tests": "pytest tests/security/ -v",
        }
        
        # Default pytest options
        self.pytest_base_options = [
            "--tb=short",  # Short traceback format
            "--strict-markers",  # Strict marker validation
            "--disable-warnings",  # Disable warnings for cleaner output
        ]
    
    def run_pytest_command(self, command: str) -> PytestResult:
        """Run a single pytest command and capture results."""
        start_time = time.time()
        
        if self.dry_run:
            logger.info(f"DRY RUN: Would execute {command}")
            return PytestResult(
                command=command,
                success=True,
                duration=0.0,
                stdout=f"DRY RUN: {command}",
                stderr="",
                tests_run=0,
                tests_passed=0,
                tests_failed=0
            )
        
        logger.info(f"Running: {command}")
        
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.repo_root
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # Parse pytest output for test counts
            tests_run, tests_passed, tests_failed = self._parse_pytest_output(result.stdout)
            
            if success:
                logger.info(f"✓ pytest completed ({duration:.2f}s, {tests_passed}/{tests_run} passed)")
            else:
                logger.error(f"✗ pytest failed ({duration:.2f}s, {tests_failed}/{tests_run} failed)")
            
            return PytestResult(
                command=command,
                success=success,
                duration=duration,
                stdout=result.stdout,
                stderr=result.stderr,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"✗ pytest timed out after {duration:.2f}s")
            return PytestResult(
                command=command,
                success=False,
                duration=duration,
                stdout="",
                stderr="Test execution timed out",
                tests_run=0,
                tests_passed=0,
                tests_failed=0
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"✗ pytest failed with exception: {e}")
            return PytestResult(
                command=command,
                success=False,
                duration=duration,
                stdout="",
                stderr=str(e),
                tests_run=0,
                tests_passed=0,
                tests_failed=0
            )
    
    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int]:
        """Parse pytest output to extract test counts."""
        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        
        # Look for pytest summary lines like "5 passed, 2 failed"
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line or 'failed' in line or 'error' in line:
                # Extract numbers from lines like "5 passed, 2 failed in 1.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        count = int(part)
                        if i + 1 < len(parts):
                            next_word = parts[i + 1]
                            if next_word == 'passed':
                                tests_passed += count
                            elif next_word in ['failed', 'error']:
                                tests_failed += count
        
        tests_run = tests_passed + tests_failed
        return tests_run, tests_passed, tests_failed
    
    def get_pytest_commands_for_targets(self, targets: Set[str]) -> List[str]:
        """Convert Buck2 targets to pytest commands."""
        commands = []
        
        for target in targets:
            if target in self.target_to_pytest:
                command = self.target_to_pytest[target]
                
                # Add base options
                base_cmd = command.split()[0]  # 'pytest'
                test_paths = command.split()[1:]  # test paths and options
                
                full_command = [base_cmd] + self.pytest_base_options + test_paths
                commands.append(" ".join(full_command))
            else:
                logger.warning(f"No pytest mapping for target: {target}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_commands = []
        for cmd in commands:
            if cmd not in seen:
                seen.add(cmd)
                unique_commands.append(cmd)
        
        return unique_commands
    
    def run_incremental_tests(self, 
                            base_commit: str = "HEAD~1", 
                            target_commit: str = "HEAD",
                            fail_fast: bool = False) -> IncrementalPytestSummary:
        """Run incremental tests using pytest based on changes."""
        start_time = time.time()
        
        # Analyze changes
        logger.info("Analyzing changes...")
        change_analysis = self.change_detector.analyze_changes(base_commit, target_commit)
        
        if not change_analysis.test_targets:
            logger.info("No test targets affected by changes")
            return IncrementalPytestSummary(
                total_commands=0,
                successful_commands=0,
                failed_commands=0,
                total_duration=0.0,
                results=[],
                change_analysis=change_analysis,
                run_metadata={"status": "no_tests", "dry_run": self.dry_run}
            )
        
        # Convert targets to pytest commands
        pytest_commands = self.get_pytest_commands_for_targets(change_analysis.test_targets)
        
        if not pytest_commands:
            logger.warning("No pytest commands generated from targets")
            return IncrementalPytestSummary(
                total_commands=0,
                successful_commands=0,
                failed_commands=0,
                total_duration=0.0,
                results=[],
                change_analysis=change_analysis,
                run_metadata={"status": "no_commands", "dry_run": self.dry_run}
            )
        
        logger.info(f"Running {len(pytest_commands)} pytest command(s)")
        
        # Run pytest commands
        results = []
        for command in pytest_commands:
            result = self.run_pytest_command(command)
            results.append(result)
            
            # Stop on first failure if fail_fast is enabled
            if fail_fast and not result.success:
                logger.error("Stopping due to failure (fail_fast=True)")
                break
        
        # Calculate summary
        total_duration = time.time() - start_time
        successful_commands = sum(1 for r in results if r.success)
        failed_commands = sum(1 for r in results if not r.success)
        
        run_metadata = {
            "base_commit": base_commit,
            "target_commit": target_commit,
            "fail_fast": fail_fast,
            "dry_run": self.dry_run,
            "pytest_version": self._get_pytest_version(),
            "run_timestamp": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True).stdout.strip(),
        }
        
        return IncrementalPytestSummary(
            total_commands=len(results),
            successful_commands=successful_commands,
            failed_commands=failed_commands,
            total_duration=total_duration,
            results=results,
            change_analysis=change_analysis,
            run_metadata=run_metadata
        )
    
    def _get_pytest_version(self) -> str:
        """Get pytest version."""
        try:
            result = subprocess.run(
                ["pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def save_results(self, summary: IncrementalPytestSummary, output_file: Path = None) -> Path:
        """Save test results to JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = self.repo_root / f"pytest_incremental_results_{timestamp}.json"
        
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
    
    def print_summary(self, summary: IncrementalPytestSummary):
        """Print human-readable test summary."""
        print(f"\n=== Pytest Incremental Test Results ===")
        print(f"Commit range: {summary.change_analysis.commit_range}")
        print(f"Total duration: {summary.total_duration:.2f}s")
        print(f"Commands run: {summary.total_commands}")
        print(f"Successful: {summary.successful_commands}")
        print(f"Failed: {summary.failed_commands}")
        
        if summary.run_metadata.get("dry_run"):
            print("(DRY RUN - no actual tests executed)")
        
        # Show changed files
        if summary.change_analysis.changed_files:
            print(f"\nChanged files ({len(summary.change_analysis.changed_files)}):")
            for file in summary.change_analysis.changed_files[:5]:
                print(f"  - {file}")
            if len(summary.change_analysis.changed_files) > 5:
                print(f"  ... and {len(summary.change_analysis.changed_files) - 5} more")
        
        # Show test results
        if summary.results:
            print(f"\nTest Results:")
            total_tests_run = sum(r.tests_run for r in summary.results)
            total_tests_passed = sum(r.tests_passed for r in summary.results)
            total_tests_failed = sum(r.tests_failed for r in summary.results)
            
            if total_tests_run > 0:
                print(f"  Total tests: {total_tests_run}")
                print(f"  Passed: {total_tests_passed}")
                print(f"  Failed: {total_tests_failed}")
            
            # Show individual command results
            for result in summary.results:
                status = "✓" if result.success else "✗"
                duration_str = f"({result.duration:.2f}s)"
                test_info = f"{result.tests_passed}/{result.tests_run} passed" if result.tests_run > 0 else "no tests"
                print(f"  {status} {result.command} {duration_str} - {test_info}")
        
        # Show failed commands
        failed_results = [r for r in summary.results if not r.success]
        if failed_results:
            print(f"\nFailed Commands:")
            for result in failed_results:
                print(f"  ✗ {result.command}")
                if result.stderr:
                    error_lines = result.stderr.split('\n')[:3]
                    for line in error_lines:
                        if line.strip():
                            print(f"    {line.strip()}")

def main():
    """Main entry point for the pytest incremental runner."""
    parser = argparse.ArgumentParser(description="Pytest Incremental Test Runner")
    parser.add_argument("--base", default="HEAD~1", help="Base commit for comparison")
    parser.add_argument("--target", default="HEAD", help="Target commit for comparison")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument("--output", type=Path, help="Output file for test results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize test runner
    runner = PytestIncrementalRunner(dry_run=args.dry_run)
    
    try:
        # Run incremental tests
        summary = runner.run_incremental_tests(
            base_commit=args.base,
            target_commit=args.target,
            fail_fast=args.fail_fast
        )
        
        # Print summary
        runner.print_summary(summary)
        
        # Save results if requested
        if args.output:
            runner.save_results(summary, args.output)
        
        # Exit with appropriate code
        if summary.failed_commands > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Test run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
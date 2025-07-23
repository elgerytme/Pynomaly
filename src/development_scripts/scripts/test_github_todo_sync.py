#!/usr/bin/env python3
"""
Comprehensive GitHub-Todo Sync Test Suite

This script thoroughly tests the GitHub-Todo synchronization system to ensure
it functions correctly in live environments.

Usage:
    python3 test_github_todo_sync.py --run-all
    python3 test_github_todo_sync.py --test-sync-only
    python3 test_github_todo_sync.py --cleanup-test-issues
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class TestResult:
    """Represents a test result."""
    test_name: str
    success: bool
    message: str
    duration: float
    details: Dict[str, Any] = None


class GitHubTodoSyncTester:
    """Comprehensive test suite for GitHub-Todo sync functionality."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results: List[TestResult] = []
        self.test_issues_created: List[int] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose or important."""
        if self.verbose or level in ["ERROR", "WARNING", "SUCCESS"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, cmd: List[str], capture_output: bool = True) -> tuple[bool, str, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(cmd, capture_output=capture_output, text=True, check=True)
            return True, result.stdout.strip(), result.stderr.strip()
        except subprocess.CalledProcessError as e:
            return False, e.stdout if e.stdout else "", e.stderr if e.stderr else str(e)
    
    def test_github_cli_availability(self) -> TestResult:
        """Test if GitHub CLI is available and authenticated."""
        start_time = time.time()
        
        # Test gh command availability
        success, stdout, stderr = self.run_command(['gh', '--version'])
        if not success:
            return TestResult(
                test_name="GitHub CLI Availability",
                success=False,
                message="GitHub CLI not available",
                duration=time.time() - start_time,
                details={"error": stderr}
            )
        
        # Test authentication
        success, stdout, stderr = self.run_command(['gh', 'auth', 'status'])
        if not success:
            return TestResult(
                test_name="GitHub CLI Availability",
                success=False,
                message="GitHub CLI not authenticated",
                duration=time.time() - start_time,
                details={"error": stderr}
            )
        
        return TestResult(
            test_name="GitHub CLI Availability",
            success=True,
            message="GitHub CLI available and authenticated",
            duration=time.time() - start_time,
            details={"version": stdout.split('\\n')[0]}
        )
    
    def test_sync_script_execution(self) -> TestResult:
        """Test that the sync script can execute successfully."""
        start_time = time.time()
        
        success, stdout, stderr = self.run_command([
            'python3', '.github/scripts/sync-todos-with-issues.py', '--dry-run', '--verbose'
        ])
        
        if not success:
            return TestResult(
                test_name="Sync Script Execution",
                success=False,
                message="Sync script failed to execute",
                duration=time.time() - start_time,
                details={"error": stderr, "stdout": stdout}
            )
        
        # Check if sync produced valid output
        lines = stderr.split('\\n') if stderr else []
        sync_lines = [line for line in lines if 'INFO: Fetched' in line or 'INFO: Created todo' in line]
        
        return TestResult(
            test_name="Sync Script Execution",
            success=True,
            message=f"Sync script executed successfully, processed {len(sync_lines)} items",
            duration=time.time() - start_time,
            details={"sync_items": len(sync_lines), "output_lines": len(lines)}
        )
    
    def test_issue_creation_and_sync(self) -> TestResult:
        """Test creating an issue and verifying it appears in sync."""
        start_time = time.time()
        
        # Create test issue
        test_title = f"SYNC-TEST: Test Issue {int(time.time())}"
        success, stdout, stderr = self.run_command([
            'gh', 'issue', 'create',
            '--title', test_title,
            '--body', 'Test issue for GitHub-Todo sync validation',
            '--label', 'P1-High'
        ])
        
        if not success:
            return TestResult(
                test_name="Issue Creation and Sync",
                success=False,
                message="Failed to create test issue",
                duration=time.time() - start_time,
                details={"error": stderr}
            )
        
        # Extract issue number from URL
        issue_url = stdout.strip()
        issue_number = int(issue_url.split('/')[-1])
        self.test_issues_created.append(issue_number)
        
        self.log(f"Created test issue #{issue_number}")
        
        # Wait a moment for GitHub to process
        time.sleep(2)
        
        # Run sync and check if issue appears
        success, stdout, stderr = self.run_command([
            'python3', '.github/scripts/sync-todos-with-issues.py', '--dry-run', '--verbose'
        ])
        
        if not success:
            return TestResult(
                test_name="Issue Creation and Sync",
                success=False,
                message="Sync failed after issue creation",
                duration=time.time() - start_time,
                details={"error": stderr, "issue_number": issue_number}
            )
        
        # Check if our test issue appears in sync output
        if f"#{issue_number}" in stderr:
            return TestResult(
                test_name="Issue Creation and Sync",
                success=True,
                message=f"Test issue #{issue_number} successfully appeared in sync",
                duration=time.time() - start_time,
                details={"issue_number": issue_number, "title": test_title}
            )
        else:
            return TestResult(
                test_name="Issue Creation and Sync",
                success=False,
                message=f"Test issue #{issue_number} did not appear in sync",
                duration=time.time() - start_time,
                details={"issue_number": issue_number, "sync_output": stderr}
            )
    
    def test_issue_status_changes(self) -> TestResult:
        """Test that issue status changes are reflected in sync."""
        start_time = time.time()
        
        # Create test issue
        test_title = f"SYNC-STATUS-TEST: Status Test {int(time.time())}"
        success, stdout, stderr = self.run_command([
            'gh', 'issue', 'create',
            '--title', test_title,
            '--body', 'Test issue for status change sync validation',
            '--label', 'P1-High'
        ])
        
        if not success:
            return TestResult(
                test_name="Issue Status Changes",
                success=False,
                message="Failed to create test issue for status testing",
                duration=time.time() - start_time,
                details={"error": stderr}
            )
        
        issue_number = int(stdout.strip().split('/')[-1])
        self.test_issues_created.append(issue_number)
        
        # Test 1: Initial sync (should be pending)
        time.sleep(2)
        success, sync_out, sync_err = self.run_command([
            'python3', '.github/scripts/sync-todos-with-issues.py', '--dry-run', '--verbose'
        ])
        
        if not success or f"#{issue_number}" not in sync_err:
            return TestResult(
                test_name="Issue Status Changes",
                success=False,
                message="Initial sync failed",
                duration=time.time() - start_time,
                details={"error": "Issue not found in initial sync"}
            )
        
        initial_status = "pending" if "(pending, high)" in sync_err else "unknown"
        
        # Test 2: Add in-progress label
        success, _, _ = self.run_command([
            'gh', 'issue', 'edit', str(issue_number), '--add-label', 'in-progress'
        ])
        
        if not success:
            return TestResult(
                test_name="Issue Status Changes",
                success=False,
                message="Failed to add in-progress label",
                duration=time.time() - start_time,
                details={"issue_number": issue_number}
            )
        
        time.sleep(2)
        success, sync_out, sync_err = self.run_command([
            'python3', '.github/scripts/sync-todos-with-issues.py', '--dry-run', '--verbose'
        ])
        
        progress_status = "in_progress" if "(in_progress, high)" in sync_err else "unknown"
        
        # Test 3: Close issue
        success, _, _ = self.run_command([
            'gh', 'issue', 'close', str(issue_number), '--comment', 'Closing for sync test'
        ])
        
        if not success:
            return TestResult(
                test_name="Issue Status Changes",
                success=False,
                message="Failed to close issue",
                duration=time.time() - start_time,
                details={"issue_number": issue_number}
            )
        
        time.sleep(2)
        success, sync_out, sync_err = self.run_command([
            'python3', '.github/scripts/sync-todos-with-issues.py', '--dry-run', '--verbose'
        ])
        
        # Closed issue should not appear in sync
        closed_absent = f"#{issue_number}" not in sync_err
        
        status_test_passed = (
            initial_status == "pending" and
            progress_status == "in_progress" and
            closed_absent
        )
        
        return TestResult(
            test_name="Issue Status Changes",
            success=status_test_passed,
            message=f"Status transitions: pending→{progress_status}→removed: {status_test_passed}",
            duration=time.time() - start_time,
            details={
                "issue_number": issue_number,
                "initial_status": initial_status,
                "progress_status": progress_status,
                "closed_absent": closed_absent
            }
        )
    
    def test_priority_filtering(self) -> TestResult:
        """Test that priority filtering works correctly."""
        start_time = time.time()
        
        # Create issues with different priorities
        test_issues = []
        priorities = [("P1-High", "high"), ("P2-Medium", "medium"), ("P3-Low", "low")]
        
        for priority_label, expected_priority in priorities:
            title = f"SYNC-PRIORITY-TEST: {priority_label} Test {int(time.time())}"
            success, stdout, stderr = self.run_command([
                'gh', 'issue', 'create',
                '--title', title,
                '--body', f'Test issue for {priority_label} priority filtering',
                '--label', priority_label
            ])
            
            if not success:
                return TestResult(
                    test_name="Priority Filtering",
                    success=False,
                    message=f"Failed to create {priority_label} test issue",
                    duration=time.time() - start_time,
                    details={"error": stderr}
                )
            
            issue_number = int(stdout.strip().split('/')[-1])
            test_issues.append((issue_number, priority_label, expected_priority))
            self.test_issues_created.append(issue_number)
        
        # Wait for GitHub to process
        time.sleep(3)
        
        # Run sync and analyze results
        success, sync_out, sync_err = self.run_command([
            'python3', '.github/scripts/sync-todos-with-issues.py', '--dry-run', '--verbose'
        ])
        
        if not success:
            return TestResult(
                test_name="Priority Filtering",
                success=False,
                message="Sync failed during priority testing",
                duration=time.time() - start_time,
                details={"error": sync_err}
            )
        
        # Check which issues appear in sync
        results = {}
        for issue_number, priority_label, expected_priority in test_issues:
            appears_in_sync = f"#{issue_number}" in sync_err
            results[priority_label] = {
                "issue_number": issue_number,
                "appears_in_sync": appears_in_sync,
                "expected_to_appear": priority_label in ["P1-High", "P2-Medium"],
                "correct": appears_in_sync == (priority_label in ["P1-High", "P2-Medium"])
            }
        
        all_correct = all(result["correct"] for result in results.values())
        
        return TestResult(
            test_name="Priority Filtering",
            success=all_correct,
            message=f"Priority filtering: {sum(r['correct'] for r in results.values())}/3 correct",
            duration=time.time() - start_time,
            details=results
        )
    
    def test_workflow_trigger(self) -> TestResult:
        """Test that GitHub Actions workflow can be triggered."""
        start_time = time.time()
        
        # Get current workflow runs
        success, stdout, stderr = self.run_command([
            'gh', 'run', 'list', '--workflow=sync-todos.yml', '--limit', '1', '--json', 'status,conclusion,createdAt'
        ])
        
        if not success:
            return TestResult(
                test_name="Workflow Trigger",
                success=False,
                message="Failed to check existing workflow runs",
                duration=time.time() - start_time,
                details={"error": stderr}
            )
        
        try:
            runs_before = json.loads(stdout) if stdout.strip() else []
            before_count = len(runs_before)
        except json.JSONDecodeError:
            before_count = 0
        
        # Trigger workflow
        success, stdout, stderr = self.run_command([
            'gh', 'workflow', 'run', 'sync-todos.yml', '--field', 'force_sync=true'
        ])
        
        if not success:
            return TestResult(
                test_name="Workflow Trigger",
                success=False,
                message="Failed to trigger workflow",
                duration=time.time() - start_time,
                details={"error": stderr}
            )
        
        # Wait a moment and check if new run appeared
        time.sleep(5)
        success, stdout, stderr = self.run_command([
            'gh', 'run', 'list', '--workflow=sync-todos.yml', '--limit', '3', '--json', 'status,conclusion,createdAt'
        ])
        
        if not success:
            return TestResult(
                test_name="Workflow Trigger",
                success=False,
                message="Failed to check workflow runs after trigger",
                duration=time.time() - start_time,
                details={"error": stderr}
            )
        
        try:
            runs_after = json.loads(stdout) if stdout.strip() else []
            after_count = len(runs_after)
            new_runs = after_count > before_count
        except json.JSONDecodeError:
            new_runs = False
        
        return TestResult(
            test_name="Workflow Trigger",
            success=new_runs,
            message=f"Workflow trigger: {after_count - before_count} new runs",
            duration=time.time() - start_time,
            details={"runs_before": before_count, "runs_after": after_count}
        )
    
    def cleanup_test_issues(self) -> TestResult:
        """Clean up test issues created during testing."""
        start_time = time.time()
        cleaned = 0
        errors = []
        
        for issue_number in self.test_issues_created:
            self.log(f"Cleaning up test issue #{issue_number}")
            success, stdout, stderr = self.run_command([
                'gh', 'issue', 'close', str(issue_number), 
                '--comment', 'Auto-closed after GitHub-Todo sync testing'
            ])
            
            if success:
                cleaned += 1
            else:
                errors.append(f"#{issue_number}: {stderr}")
        
        return TestResult(
            test_name="Test Cleanup",
            success=len(errors) == 0,
            message=f"Cleaned up {cleaned}/{len(self.test_issues_created)} test issues",
            duration=time.time() - start_time,
            details={"cleaned": cleaned, "total": len(self.test_issues_created), "errors": errors}
        )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        self.log("Starting comprehensive GitHub-Todo sync testing", "SUCCESS")
        
        # Define test sequence
        tests = [
            self.test_github_cli_availability,
            self.test_sync_script_execution,
            self.test_issue_creation_and_sync,
            self.test_issue_status_changes,
            self.test_priority_filtering,
            self.test_workflow_trigger,
        ]
        
        # Run tests
        for test_func in tests:
            self.log(f"Running {test_func.__name__}")
            result = test_func()
            self.test_results.append(result)
            
            status = "PASS" if result.success else "FAIL"
            self.log(f"{status}: {result.message}")
            
            if not result.success and self.verbose:
                self.log(f"Details: {result.details}", "ERROR")
        
        # Cleanup
        if self.test_issues_created:
            cleanup_result = self.cleanup_test_issues()
            self.test_results.append(cleanup_result)
        
        # Generate summary
        passed = sum(1 for r in self.test_results if r.success)
        total = len(self.test_results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": success_rate,
            "total_duration": sum(r.duration for r in self.test_results),
            "test_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "message": r.message,
                    "duration": round(r.duration, 2),
                    "details": r.details
                }
                for r in self.test_results
            ]
        }
        
        return summary
    
    def generate_test_report(self, summary: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report = f"""# GitHub-Todo Sync Test Report

**Test Run**: {summary['timestamp']}  
**Success Rate**: {summary['success_rate']:.1f}% ({summary['passed']}/{summary['total_tests']} tests passed)  
**Total Duration**: {summary['total_duration']:.2f} seconds

## 🎯 Test Summary

"""
        
        for result in summary['test_results']:
            status_icon = "✅" if result['success'] else "❌"
            report += f"### {status_icon} {result['test_name']}\n"
            report += f"**Status**: {'PASS' if result['success'] else 'FAIL'}  \n"
            report += f"**Duration**: {result['duration']}s  \n"
            report += f"**Message**: {result['message']}  \n"
            
            if result['details']:
                report += f"**Details**: {json.dumps(result['details'], indent=2)}  \n"
            
            report += "\\n"
        
        # Overall assessment
        if summary['success_rate'] >= 90:
            assessment = "🎉 **EXCELLENT**: GitHub-Todo sync is working perfectly!"
        elif summary['success_rate'] >= 80:
            assessment = "✅ **GOOD**: GitHub-Todo sync is working well with minor issues."
        elif summary['success_rate'] >= 60:
            assessment = "⚠️ **NEEDS ATTENTION**: GitHub-Todo sync has some issues that need fixing."
        else:
            assessment = "❌ **CRITICAL**: GitHub-Todo sync has major issues and needs immediate attention."
        
        report += f"""## 🏆 Overall Assessment

{assessment}

## 📊 Live Environment Validation

The GitHub-Todo sync automation has been tested in a live environment with the following results:

- **GitHub CLI Integration**: {'✅ Working' if any(r['test_name'] == 'GitHub CLI Availability' and r['success'] for r in summary['test_results']) else '❌ Failed'}
- **Sync Script Execution**: {'✅ Working' if any(r['test_name'] == 'Sync Script Execution' and r['success'] for r in summary['test_results']) else '❌ Failed'}
- **Issue Creation Sync**: {'✅ Working' if any(r['test_name'] == 'Issue Creation and Sync' and r['success'] for r in summary['test_results']) else '❌ Failed'}
- **Status Change Sync**: {'✅ Working' if any(r['test_name'] == 'Issue Status Changes' and r['success'] for r in summary['test_results']) else '❌ Failed'}
- **Priority Filtering**: {'✅ Working' if any(r['test_name'] == 'Priority Filtering' and r['success'] for r in summary['test_results']) else '❌ Failed'}
- **Workflow Automation**: {'✅ Working' if any(r['test_name'] == 'Workflow Trigger' and r['success'] for r in summary['test_results']) else '❌ Failed'}

## 🚀 Next Steps

Based on the test results:

1. **If all tests passed**: The GitHub-Todo sync is production-ready
2. **If some tests failed**: Review the failed tests and fix the underlying issues
3. **Monitor in production**: Set up monitoring for ongoing sync health
4. **Document usage**: Create user guides for the sync system

---

*Generated by GitHub-Todo Sync Test Suite*  
*Test completed at {summary['timestamp']}*
"""
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Test GitHub-Todo sync functionality')
    parser.add_argument('--run-all', action='store_true', 
                       help='Run all comprehensive tests')
    parser.add_argument('--test-sync-only', action='store_true', 
                       help='Run only sync functionality tests')
    parser.add_argument('--cleanup-test-issues', action='store_true', 
                       help='Clean up any existing test issues')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--output', type=str, 
                       help='Output file for test report')
    
    args = parser.parse_args()
    
    if not any([args.run_all, args.test_sync_only, args.cleanup_test_issues]):
        parser.print_help()
        return
    
    tester = GitHubTodoSyncTester(verbose=args.verbose)
    
    if args.cleanup_test_issues:
        # Find and clean up existing test issues
        success, stdout, stderr = tester.run_command([
            'gh', 'issue', 'list', '--search', 'SYNC-TEST in:title', '--json', 'number'
        ])
        
        if success and stdout.strip():
            try:
                issues = json.loads(stdout)
                tester.test_issues_created = [issue['number'] for issue in issues]
                cleanup_result = tester.cleanup_test_issues()
                print(f"Cleanup: {cleanup_result.message}")
            except json.JSONDecodeError:
                print("No test issues found to clean up")
        return
    
    if args.test_sync_only:
        # Run only core sync tests
        results = []
        results.append(tester.test_github_cli_availability())
        results.append(tester.test_sync_script_execution())
        tester.test_results = results
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "success_rate": (sum(1 for r in results if r.success) / len(results)) * 100,
            "total_duration": sum(r.duration for r in results),
            "test_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "message": r.message,
                    "duration": round(r.duration, 2),
                    "details": r.details
                }
                for r in results
            ]
        }
    else:
        # Run all tests
        summary = tester.run_all_tests()
    
    # Generate and output report
    report = tester.generate_test_report(summary)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Test report saved to {args.output}")
    else:
        print(report)
    
    # Exit with appropriate code
    if summary['success_rate'] < 100:
        sys.exit(1)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Comprehensive Script Testing using Multi-Environment Tester

This script tests all specified Pynomaly scripts across multiple environments
and logs issues to be added to README.md as tasks to be fixed.

Usage:
    python scripts/test_all_scripts.py
    python scripts/test_all_scripts.py --quick  # Only test current environment
    python scripts/test_all_scripts.py --report test_results.md
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Scripts to test
SCRIPTS_TO_TEST = [
    "setup.bat",
    "test_setup.py",
    "scripts/cli.py",
    "scripts/run_api.py",
    "scripts/run_app.py",
    "scripts/run_cli.py",
    "scripts/run_monorepo.py",
    "scripts/run_web_app.py",
    "scripts/run_web_ui.py",
    "scripts/setup_simple.py",
    "scripts/setup_standalone.py",
    "scripts/setup_windows.ps1",
]

# Script arguments for testing
SCRIPT_ARGS = {
    "setup.bat": [],
    "test_setup.py": [],
    "scripts/cli.py": ["--help"],
    "scripts/run_api.py": ["--help"],
    "scripts/run_app.py": ["--help"],
    "scripts/run_cli.py": ["--help"],
    "scripts/run_monorepo.py": ["--help"],
    "scripts/run_web_app.py": ["--help"],
    "scripts/run_web_ui.py": ["--help"],
    "scripts/setup_simple.py": ["--help"],
    "scripts/setup_standalone.py": ["--help"],  # Test with help option
    "scripts/setup_windows.ps1": [],  # No help option
}


class ScriptTester:
    """Manages comprehensive script testing using multi-environment tester."""

    def __init__(self, quick_mode: bool = False, verbose: bool = False):
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.tester_path = self.project_root / "scripts" / "multi_environment_tester.py"
        self.issues_found = []

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        if level == "ERROR":
            print(f"[{timestamp}] ❌ {message}")
        elif level == "WARNING":
            print(f"[{timestamp}] ⚠️  {message}")
        elif level == "SUCCESS":
            print(f"[{timestamp}] ✅ {message}")
        elif level == "DEBUG" and self.verbose:
            print(f"[{timestamp}] 🔍 {message}")
        else:
            print(f"[{timestamp}] ℹ️  {message}")

    def run_tester(self, script_path: str, args: list[str] = None) -> dict:
        """Run multi-environment tester on a single script."""
        args = args or []

        # Build command
        cmd = [sys.executable, str(self.tester_path), script_path]

        # Add script arguments
        if args:
            cmd.extend(args)

        # Add testing options
        if self.quick_mode:
            cmd.append("--current-only")

        if self.verbose:
            cmd.append("--verbose")

        self.log(f"Testing {script_path} with args: {args}", "DEBUG")

        try:
            # Run the tester with a reasonable timeout
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes max per script
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out after 300 seconds",
                "command": " ".join(cmd),
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -2,
                "stdout": "",
                "stderr": f"Test execution failed: {e}",
                "command": " ".join(cmd),
            }

    def analyze_test_result(self, script_path: str, result: dict) -> dict:
        """Analyze test result and extract issues."""
        analysis = {
            "script": script_path,
            "status": "PASS" if result["success"] else "FAIL",
            "issues": [],
            "warnings": [],
            "notes": [],
        }

        if not result["success"]:
            # Parse output to identify specific issues
            stderr = result.get("stderr", "")
            stdout = result.get("stdout", "")
            combined_output = stdout + stderr

            # Common issue patterns
            if "ModuleNotFoundError" in combined_output:
                missing_modules = []
                for line in combined_output.split("\n"):
                    if "ModuleNotFoundError" in line and "No module named" in line:
                        # Extract module name
                        parts = line.split("'")
                        if len(parts) >= 2:
                            missing_modules.append(parts[1])

                if missing_modules:
                    analysis["issues"].append(
                        {
                            "type": "missing_dependency",
                            "description": f"Missing modules: {', '.join(set(missing_modules))}",
                            "severity": "high",
                        }
                    )

            if "ImportError" in combined_output:
                analysis["issues"].append(
                    {
                        "type": "import_error",
                        "description": "Import error - check dependencies and paths",
                        "severity": "high",
                    }
                )

            if "SyntaxError" in combined_output:
                analysis["issues"].append(
                    {
                        "type": "syntax_error",
                        "description": "Python syntax error in script",
                        "severity": "critical",
                    }
                )

            if "FileNotFoundError" in combined_output:
                analysis["issues"].append(
                    {
                        "type": "file_not_found",
                        "description": "Required files or paths not found",
                        "severity": "medium",
                    }
                )

            if "PermissionError" in combined_output:
                analysis["issues"].append(
                    {
                        "type": "permission_error",
                        "description": "File or directory permission issues",
                        "severity": "medium",
                    }
                )

            if "timed out" in stderr:
                analysis["issues"].append(
                    {
                        "type": "timeout",
                        "description": "Script execution timed out",
                        "severity": "medium",
                    }
                )

            if "Cannot run .bat files on non-Windows" in combined_output:
                analysis["warnings"].append(
                    {
                        "type": "platform_specific",
                        "description": "Windows-specific script tested on non-Windows system",
                    }
                )

            if "Cannot run .ps1 files on non-Windows" in combined_output:
                analysis["warnings"].append(
                    {
                        "type": "platform_specific",
                        "description": "PowerShell script tested on non-Windows system",
                    }
                )

            # If no specific issues found, add generic failure
            if not analysis["issues"] and not analysis["warnings"]:
                analysis["issues"].append(
                    {
                        "type": "generic_failure",
                        "description": f"Script failed with return code {result['returncode']}",
                        "severity": "medium",
                    }
                )

        # Check for warnings even in successful runs
        if result["success"]:
            stdout = result.get("stdout", "")

            if "Warning" in stdout or "WARNING" in stdout:
                analysis["warnings"].append(
                    {
                        "type": "runtime_warning",
                        "description": "Script completed with warnings",
                    }
                )

            if "deprecated" in stdout.lower():
                analysis["warnings"].append(
                    {
                        "type": "deprecation",
                        "description": "Uses deprecated functionality",
                    }
                )

        return analysis

    def test_all_scripts(self) -> tuple[dict[str, dict], list[dict]]:
        """Test all scripts and return results."""
        self.log("Starting comprehensive script testing", "INFO")
        self.log(
            f"Testing mode: {'Quick (current environment only)' if self.quick_mode else 'Full (all environments)'}",
            "INFO",
        )
        self.log(f"Scripts to test: {len(SCRIPTS_TO_TEST)}", "INFO")

        results = {}
        all_issues = []

        for i, script in enumerate(SCRIPTS_TO_TEST, 1):
            self.log(f"[{i}/{len(SCRIPTS_TO_TEST)}] Testing {script}", "INFO")

            # Get script-specific arguments
            args = SCRIPT_ARGS.get(script, [])

            # Run the test
            test_result = self.run_tester(script, args)

            # Analyze results
            analysis = self.analyze_test_result(script, test_result)

            # Store results
            results[script] = {"test_result": test_result, "analysis": analysis}

            # Collect issues
            if analysis["issues"]:
                all_issues.extend(
                    [{**issue, "script": script} for issue in analysis["issues"]]
                )

            # Log result
            if analysis["status"] == "PASS":
                warnings_msg = (
                    f" (with {len(analysis['warnings'])} warnings)"
                    if analysis["warnings"]
                    else ""
                )
                self.log(f"✅ {script}: PASS{warnings_msg}", "SUCCESS")
            else:
                issues_count = len(analysis["issues"])
                self.log(f"❌ {script}: FAIL ({issues_count} issues)", "ERROR")

        self.log("Testing completed", "INFO")
        return results, all_issues

    def generate_summary_report(
        self, results: dict[str, dict], issues: list[dict]
    ) -> str:
        """Generate a summary report of test results."""
        total_scripts = len(results)
        passed_scripts = sum(
            1 for r in results.values() if r["analysis"]["status"] == "PASS"
        )
        failed_scripts = total_scripts - passed_scripts

        report = []
        report.append("# Script Testing Summary Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary statistics
        report.append("## Summary")
        report.append(f"- **Total Scripts**: {total_scripts}")
        report.append(f"- **Passed**: {passed_scripts}")
        report.append(f"- **Failed**: {failed_scripts}")
        report.append(
            f"- **Success Rate**: {passed_scripts / total_scripts * 100:.1f}%"
        )
        report.append(
            f"- **Testing Mode**: {'Quick (current environment only)' if self.quick_mode else 'Full (all environments)'}"
        )
        report.append("")

        # Issues summary
        if issues:
            report.append("## Issues Found")
            report.append("")

            # Group issues by type
            issue_types = {}
            for issue in issues:
                issue_type = issue["type"]
                if issue_type not in issue_types:
                    issue_types[issue_type] = []
                issue_types[issue_type].append(issue)

            for issue_type, type_issues in issue_types.items():
                report.append(f"### {issue_type.replace('_', ' ').title()}")
                for issue in type_issues:
                    report.append(
                        f"- **{issue['script']}**: {issue['description']} (Severity: {issue['severity']})"
                    )
                report.append("")
        else:
            report.append("## ✅ No Issues Found")
            report.append("All scripts passed testing successfully!")
            report.append("")

        # Detailed results
        report.append("## Detailed Results")
        report.append("")

        for script, result in results.items():
            analysis = result["analysis"]
            status_emoji = "✅" if analysis["status"] == "PASS" else "❌"

            report.append(f"### {script} {status_emoji}")
            report.append(f"**Status**: {analysis['status']}")

            if analysis["issues"]:
                report.append("**Issues**:")
                for issue in analysis["issues"]:
                    report.append(
                        f"- {issue['description']} (Severity: {issue['severity']})"
                    )

            if analysis["warnings"]:
                report.append("**Warnings**:")
                for warning in analysis["warnings"]:
                    report.append(f"- {warning['description']}")

            if analysis["notes"]:
                report.append("**Notes**:")
                for note in analysis["notes"]:
                    report.append(f"- {note}")

            report.append("")

        return "\n".join(report)

    def generate_readme_tasks(self, issues: list[dict]) -> list[str]:
        """Generate tasks to be added to README.md based on issues found."""
        if not issues:
            return []

        tasks = []
        tasks.append("## Issues Found in Script Testing")
        tasks.append("")
        tasks.append(
            "The following issues were identified during comprehensive script testing and need to be addressed:"
        )
        tasks.append("")

        # Group issues by severity
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        high_issues = [i for i in issues if i.get("severity") == "high"]
        medium_issues = [i for i in issues if i.get("severity") == "medium"]

        if critical_issues:
            tasks.append("### 🚨 Critical Issues (Fix Immediately)")
            for issue in critical_issues:
                tasks.append(f"- [ ] **{issue['script']}**: {issue['description']}")
            tasks.append("")

        if high_issues:
            tasks.append("### ⚠️ High Priority Issues")
            for issue in high_issues:
                tasks.append(f"- [ ] **{issue['script']}**: {issue['description']}")
            tasks.append("")

        if medium_issues:
            tasks.append("### 📋 Medium Priority Issues")
            for issue in medium_issues:
                tasks.append(f"- [ ] **{issue['script']}**: {issue['description']}")
            tasks.append("")

        tasks.append("### Testing Commands")
        tasks.append("```bash")
        tasks.append("# Re-run comprehensive testing after fixes")
        tasks.append("python scripts/test_all_scripts.py")
        tasks.append("")
        tasks.append("# Quick testing (current environment only)")
        tasks.append("python scripts/test_all_scripts.py --quick")
        tasks.append("")
        tasks.append("# Test specific script")
        tasks.append("python scripts/multi_environment_tester.py <script_path>")
        tasks.append("```")
        tasks.append("")

        return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Script Testing using Multi-Environment Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--quick", action="store_true", help="Only test current environment (faster)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument("--report", help="Save report to specified file")

    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Update README.md with issues as tasks",
    )

    args = parser.parse_args()

    # Create tester
    tester = ScriptTester(quick_mode=args.quick, verbose=args.verbose)

    # Check if multi-environment tester exists
    if not tester.tester_path.exists():
        tester.log(
            f"Multi-environment tester not found at: {tester.tester_path}", "ERROR"
        )
        tester.log("Please ensure scripts/multi_environment_tester.py exists", "ERROR")
        sys.exit(1)

    # Run tests
    results, issues = tester.test_all_scripts()

    # Generate report
    report = tester.generate_summary_report(results, issues)

    # Save report if requested
    if args.report:
        with open(args.report, "w") as f:
            f.write(report)
        tester.log(f"Report saved to: {args.report}", "SUCCESS")
    else:
        # Print report to console
        print("\n" + "=" * 80)
        print("COMPREHENSIVE SCRIPT TESTING REPORT")
        print("=" * 80)
        print(report)

    # Update README.md if requested
    if args.update_readme and issues:
        tester.log("Updating README.md with issues as tasks", "INFO")

        # Generate tasks
        readme_tasks = tester.generate_readme_tasks(issues)

        # Read current README
        readme_path = tester.project_root / "README.md"
        if readme_path.exists():
            current_readme = readme_path.read_text()

            # Append tasks section
            updated_readme = current_readme + "\n\n" + "\n".join(readme_tasks)

            # Write back
            readme_path.write_text(updated_readme)
            tester.log(
                f"README.md updated with {len(issues)} issues as tasks", "SUCCESS"
            )
        else:
            tester.log("README.md not found", "ERROR")

    # Exit with appropriate code
    if issues:
        critical_or_high = [
            i for i in issues if i.get("severity") in ["critical", "high"]
        ]
        if critical_or_high:
            tester.log(
                f"Testing completed with {len(critical_or_high)} critical/high priority issues",
                "ERROR",
            )
            sys.exit(1)
        else:
            tester.log(
                f"Testing completed with {len(issues)} medium priority issues",
                "WARNING",
            )
            sys.exit(0)
    else:
        tester.log("All scripts passed testing successfully!", "SUCCESS")
        sys.exit(0)


if __name__ == "__main__":
    main()

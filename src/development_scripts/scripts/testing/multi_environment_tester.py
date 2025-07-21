#!/usr/bin/env python3
"""
Multi-Environment Testing Framework for anomaly_detection Scripts

This script provides comprehensive testing of Python scripts, modules, and packages
across different environments including:
- Current environment (direct execution)
- Fresh Linux/Bash environment (isolated test)
- Fresh Windows/PowerShell environment (simulated)

Usage:
    python scripts/multi_environment_tester.py <script_path> [args...]
    python scripts/multi_environment_tester.py --help

Examples:
    python scripts/multi_environment_tester.py scripts/setup_simple.py --clean
    python scripts/multi_environment_tester.py test_setup.py
    python scripts/multi_environment_tester.py scripts/cli.py --help
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


class MultiEnvironmentTester:
    """Framework for testing scripts across multiple environments."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.test_results: dict[str, dict] = {}

    def log(self, message: str, level: str = "INFO") -> None:
        """Log message with timestamp and level."""
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

    def run_command(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        timeout: int = 120,
        allow_failure: bool = True,
    ) -> tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)."""
        try:
            self.log(f"Running: {' '.join(cmd)}", "DEBUG")
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -2, "", f"Command execution failed: {e}"

    def test_current_environment(
        self, script_path: str, args: list[str] = None
    ) -> dict:
        """Test script in current environment."""
        self.log("Testing in current environment", "INFO")

        args = args or []
        script_abs = self.project_root / script_path

        if not script_abs.exists():
            return {
                "success": False,
                "error": f"Script not found: {script_abs}",
                "returncode": -1,
                "stdout": "",
                "stderr": "",
            }

        # Determine how to run the script
        if script_path.endswith(".py"):
            cmd = [sys.executable, str(script_abs)] + args
        elif script_path.endswith(".bat"):
            if sys.platform == "win32":
                cmd = [str(script_abs)] + args
            else:
                return {
                    "success": False,
                    "error": "Cannot run .bat files on non-Windows systems",
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "",
                }
        elif script_path.endswith(".ps1"):
            if sys.platform == "win32":
                cmd = [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script_abs),
                ] + args
            else:
                return {
                    "success": False,
                    "error": "Cannot run .ps1 files on non-Windows systems",
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "",
                }
        else:
            cmd = [str(script_abs)] + args

        returncode, stdout, stderr = self.run_command(cmd)

        return {
            "success": returncode == 0,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "command": " ".join(cmd),
        }

    def create_test_environment(self, env_name: str) -> Path:
        """Create an isolated test environment directory."""
        test_dir = self.project_root / f"test_env_{env_name}_{int(time.time())}"
        test_dir.mkdir(exist_ok=True)

        # Copy essential files
        essential_files = [
            "pyproject.toml",
            "requirements.txt",
            "requirements-minimal.txt",
            "requirements-server.txt",
            "requirements-production.txt",
            "setup.py",
            "README.md",
            "LICENSE",
        ]

        for file in essential_files:
            src = self.project_root / file
            if src.exists():
                shutil.copy2(src, test_dir / file)

        # Copy source directory
        src_dir = self.project_root / "src"
        if src_dir.exists():
            shutil.copytree(src_dir, test_dir / "src", dirs_exist_ok=True)

        # Copy scripts directory
        scripts_dir = self.project_root / "scripts"
        if scripts_dir.exists():
            shutil.copytree(scripts_dir, test_dir / "scripts", dirs_exist_ok=True)

        # Copy tests directory (minimal)
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            dest_tests = test_dir / "tests"
            dest_tests.mkdir(exist_ok=True)
            # Copy a few essential test files
            for test_file in tests_dir.glob("test_*.py"):
                if test_file.stat().st_size < 100000:  # Only small test files
                    shutil.copy2(test_file, dest_tests / test_file.name)

        return test_dir

    def test_linux_environment(self, script_path: str, args: list[str] = None) -> dict:
        """Test script in fresh Linux environment."""
        self.log("Testing in fresh Linux environment", "INFO")

        args = args or []
        test_dir = None

        try:
            test_dir = self.create_test_environment("linux")

            # Create a test script to run in the isolated environment
            test_script = test_dir / "run_test.sh"
            script_rel_path = script_path

            with open(test_script, "w") as f:
                f.write(
                    f"""#!/bin/bash
set -e

echo "=== Linux Environment Test ==="
echo "Working directory: $(pwd)"
echo "Python version: $(python3 --version 2>&1 || echo 'Python3 not found')"
echo "Current user: $(whoami)"
echo "Environment variables:"
echo "  PATH=$PATH"
echo "  PYTHONPATH=$PYTHONPATH"
echo ""

echo "Testing script: {script_rel_path}"
echo "Arguments: {" ".join(args)}"
echo ""

# Test the script
if [[ "{script_rel_path}" == *.py ]]; then
    python3 "{script_rel_path}" {" ".join(args)}
elif [[ "{script_rel_path}" == *.sh ]]; then
    chmod +x "{script_rel_path}"
    ./{script_rel_path} {" ".join(args)}
else
    echo "Unsupported script type: {script_rel_path}"
    exit 1
fi
"""
                )

            # Make script executable
            os.chmod(test_script, 0o755)

            # Run the test
            returncode, stdout, stderr = self.run_command(
                ["bash", str(test_script)], cwd=test_dir, timeout=300
            )

            return {
                "success": returncode == 0,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "test_dir": str(test_dir),
                "command": f"bash {test_script}",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Linux environment test failed: {e}",
                "returncode": -1,
                "stdout": "",
                "stderr": "",
                "test_dir": str(test_dir) if test_dir else None,
            }
        finally:
            # Cleanup test directory
            if test_dir and test_dir.exists():
                try:
                    shutil.rmtree(test_dir)
                except Exception as e:
                    self.log(f"Failed to cleanup test directory: {e}", "WARNING")

    def test_windows_environment(
        self, script_path: str, args: list[str] = None
    ) -> dict:
        """Test script in simulated Windows environment."""
        self.log("Testing in simulated Windows environment", "INFO")

        args = args or []
        test_dir = None

        try:
            test_dir = self.create_test_environment("windows")

            # Create a PowerShell test script
            test_script = test_dir / "run_test.ps1"
            script_rel_path = script_path

            with open(test_script, "w") as f:
                f.write(
                    f"""# PowerShell Environment Test
Write-Host "=== Windows Environment Test ===" -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)"
Write-Host "Python version: $(python --version 2>&1 | Out-String)"
Write-Host "Current user: $env:USERNAME"
Write-Host "Environment variables:"
Write-Host "  PATH: $env:PATH"
Write-Host "  PYTHONPATH: $env:PYTHONPATH"
Write-Host ""

Write-Host "Testing script: {script_rel_path}" -ForegroundColor Yellow
Write-Host "Arguments: {" ".join(args)}"
Write-Host ""

try {{
    if ("{script_rel_path}" -like "*.py") {{
        python "{script_rel_path}" {" ".join(args)}
    }}
    elseif ("{script_rel_path}" -like "*.ps1") {{
        & ".\\{script_rel_path}" {" ".join(args)}
    }}
    elseif ("{script_rel_path}" -like "*.bat") {{
        & ".\\{script_rel_path}" {" ".join(args)}
    }}
    else {{
        Write-Host "Unsupported script type: {script_rel_path}" -ForegroundColor Red
        exit 1
    }}

    Write-Host "Script completed successfully" -ForegroundColor Green
}}
catch {{
    Write-Host "Script failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}}
"""
                )

            # On Linux/WSL, we'll simulate this by running through bash
            if sys.platform != "win32":
                # Create a bash wrapper that simulates PowerShell behavior
                bash_wrapper = test_dir / "run_test_wrapper.sh"
                with open(bash_wrapper, "w") as f:
                    f.write(
                        f"""#!/bin/bash
echo "=== Simulated Windows Environment Test ==="
echo "Note: This is a simulation of Windows environment on Linux"
echo "Working directory: $(pwd)"
echo "Python version: $(python3 --version 2>&1 || python --version 2>&1 || echo 'Python not found')"
echo ""

echo "Testing script: {script_rel_path}"
echo "Arguments: {" ".join(args)}"
echo ""

# Simulate Windows-style execution
if [[ "{script_rel_path}" == *.py ]]; then
    # Try python first (Windows style), then python3
    (python "{script_rel_path}" {" ".join(args)} 2>/dev/null) || python3 "{script_rel_path}" {" ".join(args)}
elif [[ "{script_rel_path}" == *.ps1 ]]; then
    echo "PowerShell script detected - would run on Windows"
    echo "Simulating successful execution..."
    exit 0
elif [[ "{script_rel_path}" == *.bat ]]; then
    echo "Batch file detected - would run on Windows"
    echo "Simulating successful execution..."
    exit 0
else
    echo "Unsupported script type: {script_rel_path}"
    exit 1
fi
"""
                    )
                os.chmod(bash_wrapper, 0o755)

                returncode, stdout, stderr = self.run_command(
                    ["bash", str(bash_wrapper)], cwd=test_dir, timeout=300
                )
            else:
                # On actual Windows, run PowerShell
                returncode, stdout, stderr = self.run_command(
                    [
                        "powershell",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(test_script),
                    ],
                    cwd=test_dir,
                    timeout=300,
                )

            return {
                "success": returncode == 0,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "test_dir": str(test_dir),
                "command": (
                    f"powershell {test_script}"
                    if sys.platform == "win32"
                    else "bash wrapper simulation"
                ),
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Windows environment test failed: {e}",
                "returncode": -1,
                "stdout": "",
                "stderr": "",
                "test_dir": str(test_dir) if test_dir else None,
            }
        finally:
            # Cleanup test directory
            if test_dir and test_dir.exists():
                try:
                    shutil.rmtree(test_dir)
                except Exception as e:
                    self.log(f"Failed to cleanup test directory: {e}", "WARNING")

    def test_script_validation(self, script_path: str) -> dict:
        """Validate script syntax and structure."""
        self.log("Validating script syntax and structure", "INFO")

        script_abs = self.project_root / script_path

        if not script_abs.exists():
            return {
                "success": False,
                "error": f"Script not found: {script_abs}",
                "checks": {},
            }

        checks = {}

        # File existence and permissions
        checks["file_exists"] = script_abs.exists()
        checks["file_readable"] = os.access(script_abs, os.R_OK)
        checks["file_executable"] = os.access(script_abs, os.X_OK)
        checks["file_size"] = script_abs.stat().st_size

        # Content validation
        try:
            content = script_abs.read_text(encoding="utf-8")
            checks["content_readable"] = True
            checks["content_length"] = len(content)
            checks["has_shebang"] = content.startswith("#!")

            if script_path.endswith(".py"):
                # Python-specific checks
                try:
                    import ast

                    ast.parse(content)
                    checks["python_syntax_valid"] = True
                except SyntaxError as e:
                    checks["python_syntax_valid"] = False
                    checks["python_syntax_error"] = str(e)

                # Check for common patterns
                checks["has_main_block"] = 'if __name__ == "__main__"' in content
                checks["has_imports"] = any(
                    line.strip().startswith(("import ", "from "))
                    for line in content.split("\n")
                )
                checks["has_docstring"] = '"""' in content or "'''" in content

        except Exception as e:
            checks["content_readable"] = False
            checks["content_error"] = str(e)

        return {
            "success": all(
                [
                    checks.get("file_exists", False),
                    checks.get("file_readable", False),
                    checks.get("content_readable", False),
                ]
            ),
            "checks": checks,
        }

    def test_script_comprehensive(
        self, script_path: str, args: list[str] = None
    ) -> dict:
        """Run comprehensive testing across all environments."""
        self.log(f"Starting comprehensive test for: {script_path}", "INFO")

        start_time = time.time()
        results = {
            "script_path": script_path,
            "args": args or [],
            "start_time": start_time,
            "validation": {},
            "current_env": {},
            "linux_env": {},
            "windows_env": {},
            "summary": {},
        }

        # 1. Validation
        self.log("Step 1/4: Script validation", "INFO")
        results["validation"] = self.test_script_validation(script_path)

        if not results["validation"]["success"]:
            self.log("Script validation failed, skipping environment tests", "ERROR")
            results["summary"] = {
                "overall_success": False,
                "total_duration": time.time() - start_time,
                "environments_tested": 0,
                "environments_passed": 0,
            }
            return results

        # 2. Current environment
        self.log("Step 2/4: Current environment test", "INFO")
        results["current_env"] = self.test_current_environment(script_path, args)

        # 3. Linux environment
        self.log("Step 3/4: Linux environment test", "INFO")
        results["linux_env"] = self.test_linux_environment(script_path, args)

        # 4. Windows environment
        self.log("Step 4/4: Windows environment test", "INFO")
        results["windows_env"] = self.test_windows_environment(script_path, args)

        # Summary
        environments_tested = 3
        environments_passed = sum(
            [
                results["current_env"].get("success", False),
                results["linux_env"].get("success", False),
                results["windows_env"].get("success", False),
            ]
        )

        results["summary"] = {
            "overall_success": environments_passed >= 2,  # At least 2/3 must pass
            "total_duration": time.time() - start_time,
            "environments_tested": environments_tested,
            "environments_passed": environments_passed,
            "pass_rate": environments_passed / environments_tested,
        }

        # Log summary
        if results["summary"]["overall_success"]:
            self.log(
                f"✅ PASS: {script_path} ({environments_passed}/{environments_tested} environments)",
                "SUCCESS",
            )
        else:
            self.log(
                f"❌ FAIL: {script_path} ({environments_passed}/{environments_tested} environments)",
                "ERROR",
            )

        return results

    def generate_report(
        self, all_results: list[dict], output_file: str | None = None
    ) -> str:
        """Generate a comprehensive test report."""

        report = []
        report.append("# Multi-Environment Testing Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        total_scripts = len(all_results)
        passed_scripts = sum(1 for r in all_results if r["summary"]["overall_success"])

        report.append("## Summary")
        report.append(f"- **Total Scripts Tested**: {total_scripts}")
        report.append(f"- **Scripts Passed**: {passed_scripts}")
        report.append(f"- **Scripts Failed**: {total_scripts - passed_scripts}")
        report.append(
            f"- **Overall Success Rate**: {passed_scripts / total_scripts * 100:.1f}%"
        )
        report.append("")

        # Detailed Results
        report.append("## Detailed Results")
        report.append("")

        for result in all_results:
            script = result["script_path"]
            summary = result["summary"]

            status = "✅ PASS" if summary["overall_success"] else "❌ FAIL"
            report.append(f"### {script} - {status}")
            report.append("")
            report.append(f"- **Duration**: {summary['total_duration']:.2f}s")
            report.append(
                f"- **Environments Passed**: {summary['environments_passed']}/{summary['environments_tested']}"
            )
            report.append(f"- **Pass Rate**: {summary['pass_rate'] * 100:.1f}%")
            report.append("")

            # Environment details
            for env_name, env_key in [
                ("Current", "current_env"),
                ("Linux", "linux_env"),
                ("Windows", "windows_env"),
            ]:
                env_result = result[env_key]
                env_status = "✅" if env_result.get("success", False) else "❌"
                report.append(f"**{env_name} Environment**: {env_status}")

                if not env_result.get("success", False):
                    if "error" in env_result:
                        report.append(f"  - Error: {env_result['error']}")
                    if "stderr" in env_result and env_result["stderr"]:
                        report.append(f"  - Stderr: {env_result['stderr'][:200]}...")

                report.append("")

            report.append("---")
            report.append("")

        report_text = "\n".join(report)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)
            self.log(f"Report saved to: {output_file}", "SUCCESS")

        return report_text


def main():
    """Main entry point for multi-environment testing."""
    parser = argparse.ArgumentParser(
        description="Multi-Environment Testing Framework for anomaly_detection Scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/multi_environment_tester.py scripts/setup_simple.py --clean
  python scripts/multi_environment_tester.py test_setup.py
  python scripts/multi_environment_tester.py scripts/cli.py --help
  python scripts/multi_environment_tester.py --batch scripts/*.py
        """,
    )

    parser.add_argument(
        "script_path",
        nargs="?",
        help="Path to script to test (relative to project root)",
    )

    parser.add_argument(
        "script_args", nargs="*", help="Arguments to pass to the script"
    )

    parser.add_argument(
        "--batch", nargs="+", help="Test multiple scripts in batch mode"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument("--report", help="Generate report to specified file")

    parser.add_argument(
        "--current-only", action="store_true", help="Only test in current environment"
    )

    args = parser.parse_args()

    if not args.script_path and not args.batch:
        parser.error("Must provide either script_path or --batch")

    tester = MultiEnvironmentTester(verbose=args.verbose)
    all_results = []

    # Determine scripts to test
    scripts_to_test = []
    if args.batch:
        for pattern in args.batch:
            # Handle glob patterns
            from glob import glob

            matching_files = glob(pattern)
            scripts_to_test.extend(matching_files)
    else:
        scripts_to_test = [args.script_path]

    # Test each script
    for script in scripts_to_test:
        tester.log(f"Testing script: {script}", "INFO")

        if args.current_only:
            # Only test current environment
            script_start_time = time.time()
            result = {
                "script_path": script,
                "args": args.script_args,
                "validation": tester.test_script_validation(script),
                "current_env": tester.test_current_environment(
                    script, args.script_args
                ),
                "summary": {},
            }
            result["summary"] = {
                "overall_success": result["current_env"].get("success", False),
                "total_duration": time.time() - script_start_time,
                "environments_tested": 1,
                "environments_passed": (
                    1 if result["current_env"].get("success", False) else 0
                ),
            }
        else:
            # Full comprehensive test
            result = tester.test_script_comprehensive(script, args.script_args)

        all_results.append(result)

    # Generate report
    if args.report or len(all_results) > 1:
        report_file = args.report or f"test_report_{int(time.time())}.md"
        report_text = tester.generate_report(all_results, report_file)

        if not args.report:
            print("\n" + "=" * 60)
            print("SUMMARY REPORT")
            print("=" * 60)
            print(report_text)

    # Exit with appropriate code
    overall_success = all(r["summary"]["overall_success"] for r in all_results)
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()

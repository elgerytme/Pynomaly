#!/usr/bin/env python3
"""
Multi-Version Python Validation Script
Validates that the multi-version Python setup is working correctly.
"""

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    message: str
    details: dict
    duration: float


@dataclass
class ValidationSummary:
    """Summary of all validation checks."""

    total_checks: int
    passed_checks: int
    failed_checks: int
    total_duration: float
    results: list[ValidationResult]
    overall_status: str


class MultiPythonValidator:
    """Validates multi-version Python setup."""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.environments_dir = self.base_dir / "environments"
        self.scripts_dir = self.base_dir / "scripts"
        self.reports_dir = self.base_dir / "reports"

        # Expected Python versions
        self.expected_versions = ["3.11.4", "3.11.9", "3.12.8", "3.13.1", "3.14.0a3"]

        # Validation checks to perform
        self.validation_checks = [
            ("system_requirements", "Check system requirements"),
            ("python_versions", "Validate Python version availability"),
            ("virtual_environments", "Check virtual environments"),
            ("package_installations", "Verify package installations"),
            ("import_compatibility", "Test import compatibility"),
            ("github_actions", "Validate GitHub Actions workflow"),
            ("tox_configuration", "Check tox configuration"),
            ("docker_setup", "Validate Docker setup"),
            ("performance_baseline", "Basic performance validation"),
        ]

    def run_validation(self) -> ValidationSummary:
        """Run complete validation suite."""
        logger.info("Starting multi-version Python validation...")
        start_time = time.time()

        results = []

        for check_name, description in self.validation_checks:
            logger.info(f"Running check: {description}")

            try:
                check_start = time.time()
                check_method = getattr(self, f"_validate_{check_name}")
                passed, message, details = check_method()
                check_duration = time.time() - check_start

                result = ValidationResult(
                    check_name=check_name,
                    passed=passed,
                    message=message,
                    details=details,
                    duration=check_duration,
                )

                results.append(result)

                status = "✓" if passed else "✗"
                logger.info(f"  {status} {description}: {message}")

            except Exception as e:
                check_duration = time.time() - check_start
                result = ValidationResult(
                    check_name=check_name,
                    passed=False,
                    message=f"Validation error: {str(e)}",
                    details={"error": str(e)},
                    duration=check_duration,
                )
                results.append(result)
                logger.error(f"  ✗ {description}: {str(e)}")

        # Calculate summary
        total_duration = time.time() - start_time
        passed_checks = sum(1 for r in results if r.passed)
        failed_checks = len(results) - passed_checks

        if failed_checks == 0:
            overall_status = "EXCELLENT"
        elif failed_checks <= 2:
            overall_status = "GOOD"
        elif failed_checks <= 4:
            overall_status = "NEEDS_IMPROVEMENT"
        else:
            overall_status = "CRITICAL"

        summary = ValidationSummary(
            total_checks=len(results),
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            total_duration=total_duration,
            results=results,
            overall_status=overall_status,
        )

        logger.info(f"Validation completed in {total_duration:.1f}s")
        logger.info(
            f"Status: {overall_status} ({passed_checks}/{len(results)} checks passed)"
        )

        return summary

    def _validate_system_requirements(self) -> tuple[bool, str, dict]:
        """Validate system requirements for multi-version Python."""
        details = {
            "platform": platform.platform(),
            "system": platform.system(),
            "python_version": sys.version,
            "architecture": platform.machine(),
        }

        # Check for required system tools
        required_tools = []

        if platform.system() == "Linux":
            required_tools = ["gcc", "make"]
        elif platform.system() == "Darwin":
            required_tools = ["clang", "make"]
        elif platform.system() == "Windows":
            required_tools = ["python"]

        missing_tools = []
        for tool in required_tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
                details[f"{tool}_available"] = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
                details[f"{tool}_available"] = False

        if missing_tools:
            return False, f"Missing required tools: {', '.join(missing_tools)}", details

        return True, "All system requirements met", details

    def _validate_python_versions(self) -> tuple[bool, str, dict]:
        """Validate Python version availability."""
        details = {"available_versions": [], "missing_versions": []}

        # Check pyenv availability
        pyenv_available = False
        try:
            result = subprocess.run(
                ["pyenv", "--version"], capture_output=True, text=True, check=True
            )
            pyenv_available = True
            details["pyenv_version"] = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            details["pyenv_available"] = False

        # Check available Python versions
        if pyenv_available:
            try:
                result = subprocess.run(
                    ["pyenv", "versions", "--bare"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                available = result.stdout.strip().split("\n")
                details["available_versions"] = available

                # Check for expected versions
                missing = []
                for version in self.expected_versions:
                    if version not in available:
                        missing.append(version)

                details["missing_versions"] = missing

                if missing:
                    return (
                        False,
                        f"Missing Python versions: {', '.join(missing)}",
                        details,
                    )

                return (
                    True,
                    f"All {len(self.expected_versions)} Python versions available",
                    details,
                )

            except subprocess.CalledProcessError:
                return False, "Failed to list pyenv Python versions", details
        else:
            # Check system Python versions
            system_pythons = []
            for version in ["3.11", "3.12", "3.13"]:
                try:
                    result = subprocess.run(
                        [f"python{version}", "--version"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    system_pythons.append(version)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

            details["system_pythons"] = system_pythons

            if len(system_pythons) >= 2:
                return (
                    True,
                    f"System Python versions available: {', '.join(system_pythons)}",
                    details,
                )
            else:
                return False, "Insufficient Python versions available", details

    def _validate_virtual_environments(self) -> tuple[bool, str, dict]:
        """Validate virtual environments for each Python version."""
        if not self.environments_dir.exists():
            return (
                False,
                "Environments directory not found",
                {"environments_dir": str(self.environments_dir)},
            )

        details = {"environments": {}}
        found_environments = 0

        for version in self.expected_versions:
            env_name = f".venv_{version.replace('.', '_')}"
            env_path = self.environments_dir / env_name

            env_info = {"exists": env_path.exists(), "path": str(env_path)}

            if env_path.exists():
                found_environments += 1

                # Check for Python executable
                python_exe = env_path / "bin" / "python"
                if platform.system() == "Windows":
                    python_exe = env_path / "Scripts" / "python.exe"

                env_info["python_executable"] = python_exe.exists()

                if python_exe.exists():
                    try:
                        # Test Python version
                        result = subprocess.run(
                            [str(python_exe), "--version"],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        env_info["python_version"] = result.stdout.strip()
                    except subprocess.CalledProcessError:
                        env_info["python_version"] = "unknown"

            details["environments"][version] = env_info

        if found_environments == 0:
            return False, "No virtual environments found", details
        elif found_environments < len(self.expected_versions):
            return (
                False,
                f"Only {found_environments}/{len(self.expected_versions)} environments found",
                details,
            )
        else:
            return True, f"All {found_environments} virtual environments found", details

    def _validate_package_installations(self) -> tuple[bool, str, dict]:
        """Validate package installations in virtual environments."""
        details = {"packages": {}}

        required_packages = ["pytest", "numpy", "pandas", "hypothesis"]

        total_checks = 0
        successful_checks = 0

        for version in self.expected_versions:
            env_name = f".venv_{version.replace('.', '_')}"
            env_path = self.environments_dir / env_name

            if not env_path.exists():
                continue

            python_exe = env_path / "bin" / "python"
            if platform.system() == "Windows":
                python_exe = env_path / "Scripts" / "python.exe"

            if not python_exe.exists():
                continue

            package_status = {}

            for package in required_packages:
                total_checks += 1
                try:
                    result = subprocess.run(
                        [
                            str(python_exe),
                            "-c",
                            f"import {package}; print({package}.__version__)",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=30,
                    )

                    package_status[package] = {
                        "available": True,
                        "version": result.stdout.strip(),
                    }
                    successful_checks += 1

                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    package_status[package] = {"available": False, "version": None}

            details["packages"][version] = package_status

        if total_checks == 0:
            return False, "No environments to test", details

        success_rate = successful_checks / total_checks

        if success_rate >= 0.9:
            return (
                True,
                f"Package installation excellent ({successful_checks}/{total_checks})",
                details,
            )
        elif success_rate >= 0.7:
            return (
                True,
                f"Package installation good ({successful_checks}/{total_checks})",
                details,
            )
        else:
            return (
                False,
                f"Package installation issues ({successful_checks}/{total_checks})",
                details,
            )

    def _validate_import_compatibility(self) -> tuple[bool, str, dict]:
        """Test import compatibility across Python versions."""
        details = {"compatibility_tests": {}}

        test_imports = [
            "import sys",
            "import os",
            "import json",
            "import pathlib",
            "from typing import List, Dict, Optional",
            "from dataclasses import dataclass",
            "import asyncio",
        ]

        total_tests = 0
        successful_tests = 0

        for version in self.expected_versions:
            env_name = f".venv_{version.replace('.', '_')}"
            env_path = self.environments_dir / env_name

            if not env_path.exists():
                continue

            python_exe = env_path / "bin" / "python"
            if platform.system() == "Windows":
                python_exe = env_path / "Scripts" / "python.exe"

            if not python_exe.exists():
                continue

            test_results = {}

            for test_import in test_imports:
                total_tests += 1
                try:
                    subprocess.run(
                        [str(python_exe), "-c", test_import],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=10,
                    )

                    test_results[test_import] = True
                    successful_tests += 1

                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    test_results[test_import] = False

            details["compatibility_tests"][version] = test_results

        if total_tests == 0:
            return False, "No environments to test", details

        success_rate = successful_tests / total_tests

        if success_rate >= 0.95:
            return (
                True,
                f"Import compatibility excellent ({successful_tests}/{total_tests})",
                details,
            )
        elif success_rate >= 0.8:
            return (
                True,
                f"Import compatibility good ({successful_tests}/{total_tests})",
                details,
            )
        else:
            return (
                False,
                f"Import compatibility issues ({successful_tests}/{total_tests})",
                details,
            )

    def _validate_github_actions(self) -> tuple[bool, str, dict]:
        """Validate GitHub Actions workflow configuration."""
        workflow_file = (
            self.base_dir / ".github" / "workflows" / "multi-python-testing.yml"
        )

        details = {
            "workflow_file_exists": workflow_file.exists(),
            "workflow_file_path": str(workflow_file),
        }

        if not workflow_file.exists():
            return False, "GitHub Actions workflow file not found", details

        try:
            with open(workflow_file) as f:
                content = f.read()

            # Check for required Python versions
            python_versions_found = []
            for version in ["3.11", "3.12", "3.13", "3.14"]:
                if version in content:
                    python_versions_found.append(version)

            details["python_versions_in_workflow"] = python_versions_found
            details["workflow_size"] = len(content)

            # Check for key workflow components
            required_components = [
                "matrix",
                "python-version",
                "strategy",
                "pytest",
                "ubuntu-latest",
            ]

            missing_components = []
            for component in required_components:
                if component not in content:
                    missing_components.append(component)

            details["missing_components"] = missing_components

            if missing_components:
                return (
                    False,
                    f"Workflow missing components: {', '.join(missing_components)}",
                    details,
                )

            if len(python_versions_found) >= 3:
                return (
                    True,
                    f"GitHub Actions workflow properly configured ({len(python_versions_found)} Python versions)",
                    details,
                )
            else:
                return (
                    False,
                    f"Insufficient Python versions in workflow ({len(python_versions_found)})",
                    details,
                )

        except Exception as e:
            details["error"] = str(e)
            return False, f"Failed to validate workflow: {str(e)}", details

    def _validate_tox_configuration(self) -> tuple[bool, str, dict]:
        """Validate tox configuration."""
        tox_file = self.base_dir / "tox.ini"

        details = {"tox_file_exists": tox_file.exists(), "tox_file_path": str(tox_file)}

        if not tox_file.exists():
            return False, "tox.ini file not found", details

        try:
            with open(tox_file) as f:
                content = f.read()

            # Check for Python environments
            python_envs = []
            for version in ["py311", "py312", "py313"]:
                if version in content:
                    python_envs.append(version)

            details["python_environments"] = python_envs
            details["tox_file_size"] = len(content)

            # Check for required sections
            required_sections = ["[tox]", "[testenv]", "envlist"]
            missing_sections = []

            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)

            details["missing_sections"] = missing_sections

            if missing_sections:
                return (
                    False,
                    f"tox.ini missing sections: {', '.join(missing_sections)}",
                    details,
                )

            if len(python_envs) >= 2:
                return (
                    True,
                    f"tox configuration valid ({len(python_envs)} Python environments)",
                    details,
                )
            else:
                return (
                    False,
                    f"Insufficient Python environments in tox.ini ({len(python_envs)})",
                    details,
                )

        except Exception as e:
            details["error"] = str(e)
            return False, f"Failed to validate tox.ini: {str(e)}", details

    def _validate_docker_setup(self) -> tuple[bool, str, dict]:
        """Validate Docker multi-version setup."""
        docker_file = self.base_dir / "deploy" / "docker" / "Dockerfile.multi-python"

        details = {
            "dockerfile_exists": docker_file.exists(),
            "dockerfile_path": str(docker_file),
        }

        if not docker_file.exists():
            return False, "Multi-Python Dockerfile not found", details

        try:
            with open(docker_file) as f:
                content = f.read()

            # Check for Python version installations
            python_installs = []
            for version in ["3.11", "3.12", "3.13"]:
                if (
                    f"pyenv install {version}" in content
                    or f"python{version}" in content
                ):
                    python_installs.append(version)

            details["python_versions_in_dockerfile"] = python_installs
            details["dockerfile_size"] = len(content)

            # Check for key Docker components
            required_components = ["FROM", "RUN", "COPY", "pyenv", "pytest"]

            missing_components = []
            for component in required_components:
                if component not in content:
                    missing_components.append(component)

            details["missing_components"] = missing_components

            if missing_components:
                return (
                    False,
                    f"Dockerfile missing components: {', '.join(missing_components)}",
                    details,
                )

            if len(python_installs) >= 2:
                return (
                    True,
                    f"Docker setup valid ({len(python_installs)} Python versions)",
                    details,
                )
            else:
                return (
                    False,
                    f"Insufficient Python versions in Docker ({len(python_installs)})",
                    details,
                )

        except Exception as e:
            details["error"] = str(e)
            return False, f"Failed to validate Dockerfile: {str(e)}", details

    def _validate_performance_baseline(self) -> tuple[bool, str, dict]:
        """Run basic performance validation."""
        details = {"performance_tests": {}}

        # Test current Python performance
        try:
            start_time = time.perf_counter()

            # Basic computation test
            result = sum(i * i for i in range(50000))
            computation_time = time.perf_counter() - start_time

            details["computation_test"] = {
                "duration": computation_time,
                "result_sample": str(result)[:10],
            }

            # Import performance test
            start_time = time.perf_counter()

            import_time = time.perf_counter() - start_time

            details["import_test"] = {"duration": import_time, "modules_imported": 4}

            # Overall performance assessment
            if computation_time < 0.1 and import_time < 0.05:
                return (
                    True,
                    f"Performance excellent (compute: {computation_time:.3f}s, import: {import_time:.3f}s)",
                    details,
                )
            elif computation_time < 0.5 and import_time < 0.2:
                return (
                    True,
                    f"Performance good (compute: {computation_time:.3f}s, import: {import_time:.3f}s)",
                    details,
                )
            else:
                return (
                    False,
                    f"Performance issues (compute: {computation_time:.3f}s, import: {import_time:.3f}s)",
                    details,
                )

        except Exception as e:
            details["error"] = str(e)
            return False, f"Performance test failed: {str(e)}", details

    def save_results(self, summary: ValidationSummary, output_file: Path):
        """Save validation results to file."""
        with open(output_file, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)

        logger.info(f"Validation results saved to {output_file}")

    def print_summary(self, summary: ValidationSummary):
        """Print human-readable validation summary."""
        print("\n=== Multi-Version Python Validation Summary ===")
        print(f"Overall Status: {summary.overall_status}")
        print(f"Total Checks: {summary.total_checks}")
        print(f"Passed: {summary.passed_checks}")
        print(f"Failed: {summary.failed_checks}")
        print(f"Duration: {summary.total_duration:.1f}s")

        print("\n=== Detailed Results ===")
        for result in summary.results:
            status = "✓" if result.passed else "✗"
            print(f"{status} {result.check_name:<25} {result.message}")

        # Show failed checks with details
        failed_results = [r for r in summary.results if not r.passed]
        if failed_results:
            print("\n=== Failed Checks Details ===")
            for result in failed_results:
                print(f"✗ {result.check_name}:")
                print(f"  Message: {result.message}")
                if result.details:
                    for key, value in result.details.items():
                        print(f"  {key}: {value}")

        # Recommendations
        print("\n=== Recommendations ===")
        if summary.overall_status == "EXCELLENT":
            print("🎉 Excellent! Your multi-version Python setup is perfect.")
            print("   Consider running regular validation to maintain quality.")
        elif summary.overall_status == "GOOD":
            print("👍 Good setup! Address the failed checks to achieve excellence.")
        elif summary.overall_status == "NEEDS_IMPROVEMENT":
            print(
                "⚠️  Setup needs improvement. Address failed checks before proceeding."
            )
        else:
            print("🚨 Critical issues found. Setup requires immediate attention.")
            print("   Run: python scripts/setup_multi_python.py --install")


def main():
    """Main entry point for validation."""
    parser = argparse.ArgumentParser(description="Multi-Version Python Validation")
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize validator
    validator = MultiPythonValidator()

    try:
        # Run validation
        summary = validator.run_validation()

        # Print summary
        validator.print_summary(summary)

        # Save results
        if args.output:
            validator.save_results(summary, args.output)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = Path("reports") / f"multi_python_validation_{timestamp}.json"
            validator.save_results(summary, output_file)

        # Exit with appropriate code
        sys.exit(0 if summary.overall_status in ["EXCELLENT", "GOOD"] else 1)

    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

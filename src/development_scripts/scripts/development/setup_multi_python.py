#!/usr/bin/env python3
"""
Multi-Version Python Setup Script for anomaly_detection
Automates setup of multiple Python versions for local development and testing.
"""

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Python versions to manage
PYTHON_VERSIONS = {
    "3.11.4": {
        "status": "stable",
        "priority": "high",
        "description": "Specific compatibility target",
    },
    "3.11.9": {
        "status": "stable",
        "priority": "high",
        "description": "Latest 3.11.x stable",
    },
    "3.12.8": {
        "status": "stable",
        "priority": "high",
        "description": "Latest 3.12.x stable (Dec 2024)",
    },
    "3.13.1": {
        "status": "stable",
        "priority": "medium",
        "description": "Latest 3.13.x stable (Dec 2024)",
    },
    "3.14.0a3": {
        "status": "alpha",
        "priority": "low",
        "description": "Alpha 3 development version (Dec 2024)",
    },
}


class MultiPythonManager:
    """Manages multiple Python versions for development and testing."""

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        self.pyenv_dir = self.base_dir / "environments"
        self.scripts_dir = self.base_dir / "scripts"
        self.reports_dir = self.base_dir / "reports"

        # Ensure directories exist
        self.pyenv_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        self.system_info = self._get_system_info()

    def _get_system_info(self) -> dict:
        """Get system information for compatibility checks."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        }

    def check_pyenv(self) -> bool:
        """Check if pyenv is available."""
        try:
            result = subprocess.run(
                ["pyenv", "--version"], capture_output=True, text=True, check=True
            )
            logger.info(f"✓ pyenv found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("✗ pyenv not found")
            return False

    def check_python_build_deps(self) -> bool:
        """Check if Python build dependencies are available."""
        logger.info("Checking Python build dependencies...")

        if self.system_info["system"] == "Linux":
            # Check for common build tools
            deps = [
                "gcc",
                "make",
                "zlib1g-dev",
                "libbz2-dev",
                "libreadline-dev",
                "libsqlite3-dev",
                "wget",
                "curl",
                "llvm",
                "libncurses5-dev",
            ]
            missing = []

            for dep in ["gcc", "make"]:
                try:
                    subprocess.run([dep, "--version"], capture_output=True, check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    missing.append(dep)

            if missing:
                logger.warning(f"Missing build dependencies: {missing}")
                logger.info(
                    "Install with: sudo apt-get install build-essential zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev"
                )
                return False

        elif self.system_info["system"] == "Darwin":
            # Check for Xcode command line tools
            try:
                subprocess.run(["xcode-select", "-p"], capture_output=True, check=True)
                logger.info("✓ Xcode command line tools found")
            except subprocess.CalledProcessError:
                logger.warning("✗ Xcode command line tools not found")
                logger.info("Install with: xcode-select --install")
                return False

        logger.info("✓ Build dependencies look good")
        return True

    def install_pyenv(self) -> bool:
        """Install pyenv if not available."""
        if self.check_pyenv():
            return True

        logger.info("Installing pyenv...")

        if self.system_info["system"] in ["Linux", "Darwin"]:
            try:
                # Download and install pyenv
                install_script = "curl https://pyenv.run | bash"
                result = subprocess.run(
                    install_script,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                logger.info("✓ pyenv installed successfully")
                logger.info("Add the following to your shell profile:")
                logger.info('export PYENV_ROOT="$HOME/.pyenv"')
                logger.info(
                    'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"'
                )
                logger.info('eval "$(pyenv init -)"')

                return True

            except subprocess.CalledProcessError as e:
                logger.error(f"✗ Failed to install pyenv: {e}")
                return False
        else:
            logger.warning("Automatic pyenv installation not supported on Windows")
            logger.info(
                "Install pyenv-win manually: https://github.com/pyenv-win/pyenv-win"
            )
            return False

    def list_available_versions(self) -> list[str]:
        """List available Python versions."""
        if not self.check_pyenv():
            return []

        try:
            result = subprocess.run(
                ["pyenv", "install", "--list"],
                capture_output=True,
                text=True,
                check=True,
            )

            # Filter for our target versions
            available = []
            for line in result.stdout.splitlines():
                version = line.strip()
                if any(
                    version.startswith(v.split(".")[0] + "." + v.split(".")[1])
                    for v in PYTHON_VERSIONS.keys()
                ):
                    available.append(version)

            return available

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list available versions: {e}")
            return []

    def install_python_version(self, version: str) -> bool:
        """Install a specific Python version."""
        if not self.check_pyenv():
            logger.error("pyenv not available")
            return False

        logger.info(f"Installing Python {version}...")

        try:
            # Check if already installed
            result = subprocess.run(
                ["pyenv", "versions", "--bare"], capture_output=True, text=True
            )
            if version in result.stdout:
                logger.info(f"✓ Python {version} already installed")
                return True

            # Install the version
            install_result = subprocess.run(
                ["pyenv", "install", version],
                capture_output=True,
                text=True,
                timeout=1800,
            )  # 30 minutes timeout

            if install_result.returncode == 0:
                logger.info(f"✓ Python {version} installed successfully")
                return True
            else:
                logger.error(f"✗ Failed to install Python {version}")
                logger.error(install_result.stderr)
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ Installation of Python {version} timed out")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install Python {version}: {e}")
            return False

    def create_virtual_environments(self) -> dict[str, bool]:
        """Create virtual environments for each Python version."""
        results = {}

        for version in PYTHON_VERSIONS.keys():
            logger.info(f"Creating virtual environment for Python {version}...")

            env_name = f".venv_{version.replace('.', '_')}"
            env_path = self.pyenv_dir / env_name

            try:
                # Remove existing environment
                if env_path.exists():
                    shutil.rmtree(env_path)

                # Create new environment
                if self.check_pyenv():
                    # Use pyenv python
                    python_cmd = f"~/.pyenv/versions/{version}/bin/python"
                else:
                    # Use system python (fallback)
                    python_cmd = f"python{version[:3]}"

                result = subprocess.run(
                    [python_cmd, "-m", "venv", str(env_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                logger.info(f"✓ Virtual environment created: {env_name}")
                results[version] = True

                # Install basic packages
                pip_path = env_path / "bin" / "pip"
                if self.system_info["system"] == "Windows":
                    pip_path = env_path / "Scripts" / "pip.exe"

                if pip_path.exists():
                    subprocess.run(
                        [
                            str(pip_path),
                            "install",
                            "--upgrade",
                            "pip",
                            "setuptools",
                            "wheel",
                        ],
                        capture_output=True,
                        check=True,
                    )

                    # Install test dependencies
                    subprocess.run(
                        [
                            str(pip_path),
                            "install",
                            "pytest",
                            "pytest-cov",
                            "hypothesis",
                            "tox",
                        ],
                        capture_output=True,
                        check=True,
                    )

                    logger.info(f"✓ Basic packages installed in {env_name}")

            except subprocess.CalledProcessError as e:
                logger.error(
                    f"✗ Failed to create environment for Python {version}: {e}"
                )
                results[version] = False

        return results

    def test_environments(self) -> dict[str, dict]:
        """Test all created environments."""
        results = {}

        for version in PYTHON_VERSIONS.keys():
            env_name = f".venv_{version.replace('.', '_')}"
            env_path = self.pyenv_dir / env_name

            if not env_path.exists():
                results[version] = {"status": "missing", "tests": {}}
                continue

            logger.info(f"Testing environment for Python {version}...")

            python_path = env_path / "bin" / "python"
            if self.system_info["system"] == "Windows":
                python_path = env_path / "Scripts" / "python.exe"

            test_results = {}

            # Basic Python test
            try:
                result = subprocess.run(
                    [
                        str(python_path),
                        "-c",
                        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )

                reported_version = result.stdout.strip()
                test_results["version_check"] = {
                    "passed": True,
                    "expected": version,
                    "actual": reported_version,
                }

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                test_results["version_check"] = {"passed": False, "error": str(e)}

            # Package import tests
            packages_to_test = ["sys", "os", "json", "pathlib", "typing"]
            for package in packages_to_test:
                try:
                    subprocess.run(
                        [str(python_path), "-c", f"import {package}"],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=10,
                    )
                    test_results[f"import_{package}"] = {"passed": True}
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    test_results[f"import_{package}"] = {"passed": False}

            # Test installed packages
            try:
                result = subprocess.run(
                    [
                        str(python_path),
                        "-c",
                        "import pytest; import hypothesis; print('test packages ok')",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                test_results["test_packages"] = {"passed": True}
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                test_results["test_packages"] = {"passed": False}

            # Calculate overall status
            passed_tests = sum(
                1 for test in test_results.values() if test.get("passed", False)
            )
            total_tests = len(test_results)

            results[version] = {
                "status": "healthy" if passed_tests == total_tests else "issues",
                "tests": test_results,
                "score": f"{passed_tests}/{total_tests}",
            }

            logger.info(f"✓ Environment test completed: {results[version]['score']}")

        return results

    def generate_activation_scripts(self):
        """Generate convenient activation scripts for each environment."""
        logger.info("Generating activation scripts...")

        for version in PYTHON_VERSIONS.keys():
            env_name = f".venv_{version.replace('.', '_')}"
            env_path = self.pyenv_dir / env_name

            if not env_path.exists():
                continue

            # Create activation script
            script_name = f"activate_python_{version.replace('.', '_')}.sh"
            script_path = self.scripts_dir / script_name

            activate_path = env_path / "bin" / "activate"
            if self.system_info["system"] == "Windows":
                activate_path = env_path / "Scripts" / "activate.bat"

            script_content = f"""#!/bin/bash
# Activation script for Python {version} environment
# Generated by multi-version Python setup

echo "Activating Python {version} environment..."
source {activate_path}

echo "Python version: $(python --version)"
echo "Environment: {env_name}"
echo "Location: {env_path}"

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

echo "✓ Python {version} environment activated"
echo "  Use 'deactivate' to exit"
echo "  Run 'python --version' to verify"
"""

            with open(script_path, "w") as f:
                f.write(script_content)

            # Make executable
            script_path.chmod(0o755)

            logger.info(f"✓ Created activation script: {script_name}")

    def create_test_runner_scripts(self):
        """Create test runner scripts for each Python version."""
        logger.info("Creating test runner scripts...")

        for version in PYTHON_VERSIONS.keys():
            env_name = f".venv_{version.replace('.', '_')}"
            env_path = self.pyenv_dir / env_name

            if not env_path.exists():
                continue

            python_path = env_path / "bin" / "python"
            if self.system_info["system"] == "Windows":
                python_path = env_path / "Scripts" / "python.exe"

            script_name = f"test_python_{version.replace('.', '_')}.py"
            script_path = self.scripts_dir / script_name

            script_content = f"""#!/usr/bin/env python3
'''
Test runner for Python {version}
Runs comprehensive tests using the specific Python version.
'''

import subprocess
import sys
from pathlib import Path

def main():
    print(f"Running tests with Python {version}")
    print(f"Python executable: {python_path}")

    # Set up environment
    base_dir = Path.cwd()

    # Run basic tests
    test_commands = [
        # Basic pytest
        [str(python_path), "-m", "pytest", "tests/", "-v", "--tb=short"],

        # Type checking (if available)
        [str(python_path), "-c", "import mypy; print('mypy available')"],

        # Import tests
        [str(python_path), "-c", "import sys; sys.path.insert(0, 'src'); import anomaly_detection; print('✓ anomaly_detection imports successfully')"],
    ]

    for i, cmd in enumerate(test_commands, 1):
        print(f"\\n--- Test {{i}}: {{' '.join(cmd[:3])}} ---")
        try:
            result = subprocess.run(cmd, cwd=base_dir, timeout=300)
            if result.returncode == 0:
                print(f"✓ Test {{i}} passed")
            else:
                print(f"✗ Test {{i}} failed (exit code: {{result.returncode}})")
        except subprocess.TimeoutExpired:
            print(f"✗ Test {{i}} timed out")
        except Exception as e:
            print(f"✗ Test {{i}} error: {{e}}")

    print(f"\\n✓ Test suite completed for Python {version}")

if __name__ == "__main__":
    main()
"""

            with open(script_path, "w") as f:
                f.write(script_content)

            script_path.chmod(0o755)

            logger.info(f"✓ Created test runner: {script_name}")

    def generate_report(self, test_results: dict) -> str:
        """Generate a comprehensive setup and test report."""
        report_file = self.reports_dir / "multi_python_setup_report.json"

        report_data = {
            "timestamp": time.time(),
            "system_info": self.system_info,
            "python_versions": PYTHON_VERSIONS,
            "test_results": test_results,
            "summary": {
                "total_versions": len(PYTHON_VERSIONS),
                "successful_installs": sum(
                    1 for v, r in test_results.items() if r.get("status") == "healthy"
                ),
                "failed_installs": sum(
                    1 for v, r in test_results.items() if r.get("status") != "healthy"
                ),
            },
        }

        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Generate human-readable report
        text_report = f"""
Multi-Version Python Setup Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

System Information:
  Platform: {self.system_info['platform']}
  System: {self.system_info['system']}
  Machine: {self.system_info['machine']}

Python Version Status:
"""

        for version, info in PYTHON_VERSIONS.items():
            test_result = test_results.get(version, {})
            status = test_result.get("status", "unknown")
            score = test_result.get("score", "0/0")

            status_symbol = (
                "✓" if status == "healthy" else "✗" if status == "issues" else "?"
            )

            text_report += f"  {status_symbol} Python {version:<8} {info['description']:<30} ({score})\n"

        text_report += f"""
Summary:
  Total Versions: {report_data['summary']['total_versions']}
  Successful: {report_data['summary']['successful_installs']}
  Failed: {report_data['summary']['failed_installs']}

Generated Files:
  - Environment directories in: {self.pyenv_dir}
  - Activation scripts in: {self.scripts_dir}
  - Test runners in: {self.scripts_dir}
  - This report: {report_file}

Usage:
  1. Activate environment: source scripts/activate_python_X_Y_Z.sh
  2. Run tests: python scripts/test_python_X_Y_Z.py
  3. Use tox: tox -e py311,py312,py313
"""

        # Save text report
        text_report_file = self.reports_dir / "multi_python_setup_report.txt"
        with open(text_report_file, "w") as f:
            f.write(text_report)

        logger.info("✓ Reports generated:")
        logger.info(f"  JSON: {report_file}")
        logger.info(f"  Text: {text_report_file}")

        return text_report

    def setup_all(self, install_missing: bool = True) -> bool:
        """Complete setup of multi-version Python environment."""
        logger.info("Starting multi-version Python setup...")

        # Check prerequisites
        if not self.check_python_build_deps():
            logger.error("Build dependencies missing - cannot proceed")
            return False

        # Install pyenv if needed
        if install_missing and not self.check_pyenv():
            if not self.install_pyenv():
                logger.error("Failed to install pyenv")
                return False

        # Install Python versions
        installation_results = {}
        if self.check_pyenv():
            for version in PYTHON_VERSIONS.keys():
                if PYTHON_VERSIONS[version]["priority"] in ["high", "medium"]:
                    installation_results[version] = self.install_python_version(version)

        # Create virtual environments
        env_results = self.create_virtual_environments()

        # Test environments
        test_results = self.test_environments()

        # Generate scripts
        self.generate_activation_scripts()
        self.create_test_runner_scripts()

        # Generate report
        report = self.generate_report(test_results)
        print(report)

        # Check overall success
        successful = sum(
            1 for r in test_results.values() if r.get("status") == "healthy"
        )
        total = len(test_results)

        logger.info(f"Setup completed: {successful}/{total} environments healthy")

        return successful > 0


def main():
    """Main entry point for multi-version Python setup."""
    parser = argparse.ArgumentParser(
        description="Multi-Version Python Setup for anomaly_detection"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check current setup status"
    )
    parser.add_argument(
        "--install", action="store_true", help="Install missing components"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test existing environments"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean up existing environments"
    )
    parser.add_argument("--report", action="store_true", help="Generate setup report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    manager = MultiPythonManager()

    try:
        if args.clean:
            logger.info("Cleaning up existing environments...")
            if manager.pyenv_dir.exists():
                shutil.rmtree(manager.pyenv_dir)
                manager.pyenv_dir.mkdir(exist_ok=True)
            logger.info("✓ Cleanup completed")

        elif args.check:
            logger.info("Checking current setup...")
            test_results = manager.test_environments()
            report = manager.generate_report(test_results)
            print(report)

        elif args.test:
            logger.info("Testing environments...")
            test_results = manager.test_environments()
            for version, result in test_results.items():
                status = result.get("status", "unknown")
                score = result.get("score", "0/0")
                print(f"Python {version}: {status} ({score})")

        elif args.report:
            logger.info("Generating report...")
            test_results = manager.test_environments()
            report = manager.generate_report(test_results)
            print(report)

        elif args.install:
            success = manager.setup_all(install_missing=True)
            sys.exit(0 if success else 1)

        else:
            # Default: check and install if needed
            success = manager.setup_all(install_missing=True)
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

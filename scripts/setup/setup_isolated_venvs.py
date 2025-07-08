#!/usr/bin/env python3
"""
Isolated Python Virtual Environment Setup Script for Pynomaly
Creates isolated Python 3.11/3.12 virtual environments with specific extras.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Target Python versions
PYTHON_VERSIONS = ["3.11", "3.12"]

# Available extras and their descriptions
EXTRAS = {
    "test": "Testing dependencies (pytest, coverage, etc.)",
    "server": "API server dependencies (FastAPI, uvicorn, etc.)",
    "cli": "Command line interface dependencies (typer, rich, etc.)",
    "all": "All dependencies (complete package)"
}

class IsolatedVenvManager:
    """Manages isolated Python virtual environments for development."""

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        self.venv_dir = self.base_dir / "environments"
        self.scripts_dir = self.base_dir / "scripts"
        self.reports_dir = self.base_dir / "reports"
        
        # Ensure directories exist
        self.venv_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # System information
        self.system_info = {
            "platform": sys.platform,
            "python_version": sys.version,
            "python_executable": sys.executable,
        }

    def find_python_executable(self, version: str) -> Optional[str]:
        """Find Python executable for the specified version."""
        possible_names = [
            f"python{version}",
            f"python{version}.exe",
            f"py -{version}",  # Windows py launcher
            "python",
            "python3"
        ]
        
        for name in possible_names:
            try:
                if name.startswith("py -"):
                    # Windows py launcher
                    result = subprocess.run(
                        name.split(), 
                        capture_output=True, 
                        text=True, 
                        check=True
                    )
                    if version in result.stdout:
                        return name
                else:
                    # Regular executable
                    result = subprocess.run(
                        [name, "--version"], 
                        capture_output=True, 
                        text=True, 
                        check=True
                    )
                    if version in result.stdout:
                        return name
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
                
        return None

    def create_venv(self, python_version: str, clean: bool = False) -> bool:
        """Create a virtual environment for the specified Python version."""
        venv_name = f"venv-py{python_version}"
        venv_path = self.venv_dir / venv_name
        
        if clean and venv_path.exists():
            logger.info(f"Removing existing environment: {venv_name}")
            shutil.rmtree(venv_path)
        
        if venv_path.exists():
            logger.info(f"Environment already exists: {venv_name}")
            return True
        
        # Find Python executable
        python_exe = self.find_python_executable(python_version)
        if not python_exe:
            logger.error(f"Python {python_version} not found")
            return False
        
        logger.info(f"Creating virtual environment: {venv_name}")
        logger.info(f"Using Python executable: {python_exe}")
        
        try:
            # Create virtual environment
            if python_exe.startswith("py -"):
                # Windows py launcher
                cmd = python_exe.split() + ["-m", "venv", str(venv_path)]
            else:
                cmd = [python_exe, "-m", "venv", str(venv_path)]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✓ Virtual environment created: {venv_name}")
            
            # Upgrade pip
            pip_exe = self._get_pip_executable(venv_path)
            if pip_exe:
                subprocess.run([
                    pip_exe, "install", "--upgrade", "pip", "setuptools", "wheel"
                ], check=True, capture_output=True)
                logger.info(f"✓ Upgraded pip in {venv_name}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to create environment {venv_name}: {e}")
            return False

    def _get_pip_executable(self, venv_path: Path) -> Optional[str]:
        """Get pip executable path for the virtual environment."""
        if sys.platform == "win32":
            pip_exe = venv_path / "Scripts" / "pip.exe"
            if not pip_exe.exists():
                pip_exe = venv_path / "Scripts" / "pip"
        else:
            pip_exe = venv_path / "bin" / "pip"
        
        return str(pip_exe) if pip_exe.exists() else None

    def _get_python_executable(self, venv_path: Path) -> Optional[str]:
        """Get Python executable path for the virtual environment."""
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
            if not python_exe.exists():
                python_exe = venv_path / "Scripts" / "python"
        else:
            python_exe = venv_path / "bin" / "python"
        
        return str(python_exe) if python_exe.exists() else None

    def install_extras(self, python_version: str, extras: List[str]) -> bool:
        """Install specified extras in the virtual environment."""
        venv_name = f"venv-py{python_version}"
        venv_path = self.venv_dir / venv_name
        
        if not venv_path.exists():
            logger.error(f"Virtual environment not found: {venv_name}")
            return False
        
        pip_exe = self._get_pip_executable(venv_path)
        if not pip_exe:
            logger.error(f"pip not found in {venv_name}")
            return False
        
        success = True
        for extra in extras:
            if extra not in EXTRAS:
                logger.warning(f"Unknown extra: {extra}")
                continue
            
            logger.info(f"Installing extra '{extra}' in {venv_name}")
            
            try:
                # Install the package with the specific extra
                cmd = [pip_exe, "install", "-e", f".[{extra}]"]
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"✓ Installed extra '{extra}' in {venv_name}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ Failed to install extra '{extra}' in {venv_name}: {e}")
                success = False
        
        return success

    def test_environment(self, python_version: str) -> Dict:
        """Test the virtual environment functionality."""
        venv_name = f"venv-py{python_version}"
        venv_path = self.venv_dir / venv_name
        
        if not venv_path.exists():
            return {"status": "missing", "tests": {}}
        
        python_exe = self._get_python_executable(venv_path)
        if not python_exe:
            return {"status": "broken", "tests": {}}
        
        logger.info(f"Testing environment: {venv_name}")
        
        test_results = {}
        
        # Test 1: Python version check
        try:
            result = subprocess.run([
                python_exe, "-c", 
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            ], capture_output=True, text=True, check=True, timeout=30)
            
            reported_version = result.stdout.strip()
            test_results["version_check"] = {
                "passed": python_version in reported_version,
                "expected": python_version,
                "actual": reported_version
            }
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            test_results["version_check"] = {"passed": False, "error": str(e)}
        
        # Test 2: Basic imports
        basic_imports = ["sys", "os", "pathlib", "json", "typing"]
        for pkg in basic_imports:
            try:
                subprocess.run([
                    python_exe, "-c", f"import {pkg}"
                ], capture_output=True, text=True, check=True, timeout=10)
                test_results[f"import_{pkg}"] = {"passed": True}
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                test_results[f"import_{pkg}"] = {"passed": False}
        
        # Test 3: Core package import
        try:
            subprocess.run([
                python_exe, "-c", 
                "import sys; sys.path.insert(0, 'src'); import pynomaly; print('✓ pynomaly imported')"
            ], capture_output=True, text=True, check=True, timeout=30)
            test_results["pynomaly_import"] = {"passed": True}
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            test_results["pynomaly_import"] = {"passed": False}
        
        # Test 4: Check installed packages
        try:
            result = subprocess.run([
                python_exe, "-m", "pip", "list", "--format=json"
            ], capture_output=True, text=True, check=True, timeout=30)
            
            packages = json.loads(result.stdout)
            installed_packages = [pkg["name"].lower() for pkg in packages]
            
            test_results["installed_packages"] = {
                "passed": True,
                "count": len(installed_packages),
                "has_pytest": "pytest" in installed_packages,
                "has_fastapi": "fastapi" in installed_packages,
                "has_typer": "typer" in installed_packages,
                "has_pynomaly": "pynomaly" in installed_packages
            }
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            test_results["installed_packages"] = {"passed": False}
        
        # Calculate overall status
        passed_tests = sum(1 for test in test_results.values() if test.get("passed", False))
        total_tests = len(test_results)
        
        return {
            "status": "healthy" if passed_tests == total_tests else "issues",
            "tests": test_results,
            "score": f"{passed_tests}/{total_tests}"
        }

    def create_activation_scripts(self, python_version: str) -> bool:
        """Create activation scripts for the virtual environment."""
        venv_name = f"venv-py{python_version}"
        venv_path = self.venv_dir / venv_name
        
        if not venv_path.exists():
            logger.error(f"Virtual environment not found: {venv_name}")
            return False
        
        # Create activation script
        if sys.platform == "win32":
            script_name = f"activate-py{python_version}.bat"
            activate_path = venv_path / "Scripts" / "activate.bat"
            script_content = f"""@echo off
REM Activation script for Python {python_version} environment
echo Activating Python {python_version} environment...
call "{activate_path}"
echo.
echo Python version: 
python --version
echo Environment: {venv_name}
echo Location: {venv_path}
echo.
echo Environment activated successfully!
echo Use 'deactivate' to exit this environment.
"""
        else:
            script_name = f"activate-py{python_version}.sh"
            activate_path = venv_path / "bin" / "activate"
            script_content = f"""#!/bin/bash
# Activation script for Python {python_version} environment
echo "Activating Python {python_version} environment..."
source "{activate_path}"

echo
echo "Python version: $(python --version)"
echo "Environment: {venv_name}"
echo "Location: {venv_path}"
echo

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

echo "✓ Python {python_version} environment activated"
echo "  Use 'deactivate' to exit this environment"
echo "  Run 'python --version' to verify"
"""
        
        script_path = self.scripts_dir / script_name
        with open(script_path, "w") as f:
            f.write(script_content)
        
        if not sys.platform == "win32":
            script_path.chmod(0o755)
        
        logger.info(f"✓ Created activation script: {script_name}")
        return True

    def generate_report(self, test_results: Dict) -> str:
        """Generate a comprehensive setup report."""
        report_file = self.reports_dir / "isolated_venv_setup_report.json"
        
        report_data = {
            "timestamp": time.time(),
            "system_info": self.system_info,
            "python_versions": PYTHON_VERSIONS,
            "extras": EXTRAS,
            "test_results": test_results,
            "summary": {
                "total_environments": len(PYTHON_VERSIONS),
                "healthy_environments": sum(
                    1 for v, r in test_results.items() if r.get("status") == "healthy"
                ),
                "environments_with_issues": sum(
                    1 for v, r in test_results.items() if r.get("status") == "issues"
                ),
                "missing_environments": sum(
                    1 for v, r in test_results.items() if r.get("status") == "missing"
                )
            }
        }
        
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate human-readable report
        text_report = f"""
Isolated Python Virtual Environment Setup Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

System Information:
  Platform: {self.system_info['platform']}
  Python Version: {self.system_info['python_version'].split()[0]}

Environment Status:
"""
        
        for version in PYTHON_VERSIONS:
            test_result = test_results.get(version, {})
            status = test_result.get("status", "unknown")
            score = test_result.get("score", "0/0")
            
            status_symbol = {
                "healthy": "✓",
                "issues": "⚠",
                "missing": "✗",
                "unknown": "?"
            }.get(status, "?")
            
            text_report += f"  {status_symbol} Python {version:<4} {status:<8} ({score})\n"
        
        text_report += f"""
Summary:
  Total Environments: {report_data['summary']['total_environments']}
  Healthy: {report_data['summary']['healthy_environments']}
  With Issues: {report_data['summary']['environments_with_issues']}
  Missing: {report_data['summary']['missing_environments']}

Available Extras:
"""
        
        for extra, description in EXTRAS.items():
            text_report += f"  {extra:<8} {description}\n"
        
        text_report += f"""
Generated Files:
  - Environment directories: {self.venv_dir}
  - Activation scripts: {self.scripts_dir}
  - This report: {report_file}

Usage Examples:
  # Activate Python 3.11 environment
  source scripts/activate-py3.11.sh  # Linux/macOS
  scripts\\activate-py3.11.bat        # Windows
  
  # Install with specific extras
  python {__file__} --version 3.11 --extras test,cli
  
  # Create all environments with all extras
  python {__file__} --all-versions --extras all
"""
        
        # Save text report
        text_report_file = self.reports_dir / "isolated_venv_setup_report.txt"
        with open(text_report_file, "w") as f:
            f.write(text_report)
        
        logger.info(f"✓ Reports generated:")
        logger.info(f"  JSON: {report_file}")
        logger.info(f"  Text: {text_report_file}")
        
        return text_report

    def setup_environments(self, versions: List[str], extras: List[str], clean: bool = False) -> Dict:
        """Set up isolated virtual environments with specified extras."""
        results = {}
        
        for version in versions:
            logger.info(f"Setting up Python {version} environment...")
            
            # Create virtual environment
            if self.create_venv(version, clean=clean):
                # Install extras
                if self.install_extras(version, extras):
                    # Create activation scripts
                    self.create_activation_scripts(version)
                    logger.info(f"✓ Python {version} environment setup complete")
                else:
                    logger.error(f"✗ Failed to install extras for Python {version}")
            else:
                logger.error(f"✗ Failed to create Python {version} environment")
        
        # Test all environments
        for version in versions:
            results[version] = self.test_environment(version)
        
        return results


def main():
    """Main entry point for isolated virtual environment setup."""
    parser = argparse.ArgumentParser(
        description="Create isolated Python virtual environments with specified extras"
    )
    parser.add_argument(
        "--version", 
        choices=PYTHON_VERSIONS, 
        help="Python version to set up"
    )
    parser.add_argument(
        "--all-versions", 
        action="store_true", 
        help="Set up all supported Python versions"
    )
    parser.add_argument(
        "--extras", 
        default="test", 
        help="Comma-separated list of extras to install (default: test)"
    )
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="Remove existing environments before creating new ones"
    )
    parser.add_argument(
        "--test-only", 
        action="store_true", 
        help="Only test existing environments"
    )
    parser.add_argument(
        "--report", 
        action="store_true", 
        help="Generate setup report"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine versions to work with
    if args.all_versions:
        versions = PYTHON_VERSIONS
    elif args.version:
        versions = [args.version]
    else:
        versions = PYTHON_VERSIONS
    
    # Parse extras
    extras = [e.strip() for e in args.extras.split(",") if e.strip()]
    
    manager = IsolatedVenvManager()
    
    try:
        if args.test_only:
            # Only test existing environments
            results = {}
            for version in versions:
                results[version] = manager.test_environment(version)
        else:
            # Set up environments
            results = manager.setup_environments(versions, extras, clean=args.clean)
        
        # Generate report
        report = manager.generate_report(results)
        
        if args.report or args.test_only:
            print(report)
        
        # Check overall success
        healthy_count = sum(1 for r in results.values() if r.get("status") == "healthy")
        total_count = len(results)
        
        if healthy_count == total_count:
            logger.info(f"✓ All environments healthy ({healthy_count}/{total_count})")
            sys.exit(0)
        else:
            logger.warning(f"⚠ Some environments have issues ({healthy_count}/{total_count} healthy)")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

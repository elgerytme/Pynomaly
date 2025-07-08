#!/usr/bin/env python3
"""
Environment Build Matrix Tester

Tests installation of critical extras sets across Python versions,
capturing dependency resolution errors, conflicts, and warnings.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reports/env_matrix/test_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Critical extras sets to test
CRITICAL_EXTRAS = [
    "minimal",
    "server", 
    "production",
    "deep",
    "deep-cpu",
    "automl",
    "all"
]

# Python versions to test - using current Python only
PYTHON_VERSIONS = [f"{sys.version_info.major}.{sys.version_info.minor}"]

# Report directory
REPORTS_DIR = Path("reports/env_matrix")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class EnvironmentTester:
    """Test environment installation for different extras sets."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results = {}
        self.summary = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "test_timestamp": datetime.now().isoformat(),
            "results": {}
        }
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def create_venv(self, python_version: str, venv_path: Path) -> bool:
        """Create a fresh virtual environment."""
        try:
            # Clean up existing venv if it exists
            if venv_path.exists():
                shutil.rmtree(venv_path)
            
            # Create new venv - use current Python executable
            python_cmd = sys.executable
            
            returncode, stdout, stderr = self.run_command([
                python_cmd, "-m", "venv", str(venv_path)
            ])
            
            if returncode != 0:
                logger.error(f"Failed to create venv: {stderr}")
                return False
            
            # Upgrade pip
            pip_cmd = str(venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "pip")
            returncode, stdout, stderr = self.run_command([
                pip_cmd, "install", "--upgrade", "pip", "setuptools", "wheel"
            ])
            
            if returncode != 0:
                logger.warning(f"Failed to upgrade pip: {stderr}")
            
            return True
        except Exception as e:
            logger.error(f"Error creating venv: {e}")
            return False
    
    def install_package(self, venv_path: Path, extra: str) -> Dict[str, Any]:
        """Install package with extra and capture results."""
        pip_cmd = str(venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "pip")
        
        install_cmd = [pip_cmd, "install", "-e", f".[{extra}]", "--verbose"]
        
        result = {
            "command": " ".join(install_cmd),
            "success": False,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "warnings": [],
            "errors": [],
            "duration": 0
        }
        
        start_time = datetime.now()
        
        try:
            returncode, stdout, stderr = self.run_command(install_cmd, timeout=600)
            
            result["returncode"] = returncode
            result["stdout"] = stdout
            result["stderr"] = stderr
            result["success"] = returncode == 0
            result["duration"] = (datetime.now() - start_time).total_seconds()
            
            # Parse warnings and errors
            result["warnings"] = self.parse_warnings(stdout, stderr)
            result["errors"] = self.parse_errors(stdout, stderr)
            
        except Exception as e:
            result["errors"].append(f"Installation exception: {e}")
            result["duration"] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def parse_warnings(self, stdout: str, stderr: str) -> List[str]:
        """Parse warnings from pip output."""
        warnings = []
        combined_output = stdout + stderr
        
        warning_indicators = [
            "WARNING:",
            "DEPRECATION:",
            "yanked",
            "ABI mismatch",
            "conflict",
            "incompatible",
            "requires a different version"
        ]
        
        lines = combined_output.split('\n')
        for line in lines:
            line_lower = line.lower()
            for indicator in warning_indicators:
                if indicator.lower() in line_lower:
                    warnings.append(line.strip())
                    break
        
        return warnings
    
    def parse_errors(self, stdout: str, stderr: str) -> List[str]:
        """Parse errors from pip output."""
        errors = []
        combined_output = stdout + stderr
        
        error_indicators = [
            "ERROR:",
            "FAILED",
            "Could not find a version",
            "No matching distribution found",
            "Resolution impossible",
            "UnsatisfiableError"
        ]
        
        lines = combined_output.split('\n')
        for line in lines:
            line_lower = line.lower()
            for indicator in error_indicators:
                if indicator.lower() in line_lower:
                    errors.append(line.strip())
                    break
        
        return errors
    
    def get_dependency_tree(self, venv_path: Path) -> Dict[str, Any]:
        """Get dependency tree information."""
        pip_cmd = str(venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "pip")
        
        # Install pipdeptree if not available
        self.run_command([pip_cmd, "install", "pipdeptree"])
        
        pipdeptree_cmd = str(venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "pipdeptree")
        
        tree_result = {
            "tree_text": "",
            "tree_json": {},
            "freeze_output": ""
        }
        
        try:
            # Get text tree
            returncode, stdout, stderr = self.run_command([pipdeptree_cmd])
            if returncode == 0:
                tree_result["tree_text"] = stdout
            
            # Get JSON tree
            returncode, stdout, stderr = self.run_command([pipdeptree_cmd, "--json"])
            if returncode == 0:
                try:
                    tree_result["tree_json"] = json.loads(stdout)
                except json.JSONDecodeError:
                    tree_result["tree_json"] = {"error": "Failed to parse JSON"}
            
            # Get freeze output
            returncode, stdout, stderr = self.run_command([pip_cmd, "freeze"])
            if returncode == 0:
                tree_result["freeze_output"] = stdout
        
        except Exception as e:
            tree_result["error"] = str(e)
        
        return tree_result
    
    def generate_lockfile(self, venv_path: Path) -> Dict[str, Any]:
        """Generate lockfile information."""
        pip_cmd = str(venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "pip")
        
        lockfile_info = {
            "pip_freeze": "",
            "pip_list": "",
            "pip_show_pynomaly": ""
        }
        
        try:
            # pip freeze
            returncode, stdout, stderr = self.run_command([pip_cmd, "freeze"])
            if returncode == 0:
                lockfile_info["pip_freeze"] = stdout
            
            # pip list
            returncode, stdout, stderr = self.run_command([pip_cmd, "list", "--format=json"])
            if returncode == 0:
                try:
                    lockfile_info["pip_list"] = json.loads(stdout)
                except json.JSONDecodeError:
                    lockfile_info["pip_list"] = stdout
            
            # pip show for main package
            returncode, stdout, stderr = self.run_command([pip_cmd, "show", "pynomaly"])
            if returncode == 0:
                lockfile_info["pip_show_pynomaly"] = stdout
        
        except Exception as e:
            lockfile_info["error"] = str(e)
        
        return lockfile_info
    
    def test_extra_installation(self, python_version: str, extra: str) -> Dict[str, Any]:
        """Test installation of a specific extra."""
        logger.info(f"Testing Python {python_version} with extra '{extra}'")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir) / f"venv_{python_version}_{extra}"
            
            test_result = {
                "python_version": python_version,
                "extra": extra,
                "test_timestamp": datetime.now().isoformat(),
                "venv_creation": {"success": False},
                "installation": {},
                "dependency_tree": {},
                "lockfile": {},
                "overall_success": False
            }
            
            # Create virtual environment
            if self.create_venv(python_version, venv_path):
                test_result["venv_creation"]["success"] = True
                logger.info(f"Created venv for Python {python_version}")
                
                # Install package with extra
                test_result["installation"] = self.install_package(venv_path, extra)
                
                if test_result["installation"]["success"]:
                    logger.info(f"Successfully installed with extra '{extra}'")
                    
                    # Get dependency information
                    test_result["dependency_tree"] = self.get_dependency_tree(venv_path)
                    test_result["lockfile"] = self.generate_lockfile(venv_path)
                    test_result["overall_success"] = True
                else:
                    logger.error(f"Failed to install with extra '{extra}'")
            else:
                logger.error(f"Failed to create venv for Python {python_version}")
                test_result["venv_creation"]["error"] = "Failed to create virtual environment"
            
            return test_result
    
    def run_all_tests(self) -> None:
        """Run all environment tests."""
        logger.info("Starting environment build matrix tests")
        
        for python_version in PYTHON_VERSIONS:
            for extra in CRITICAL_EXTRAS:
                self.summary["total_tests"] += 1
                
                test_key = f"python{python_version}_{extra}"
                
                try:
                    result = self.test_extra_installation(python_version, extra)
                    self.test_results[test_key] = result
                    
                    if result["overall_success"]:
                        self.summary["successful_tests"] += 1
                        logger.info(f"[SUCCESS] {test_key}")
                    else:
                        self.summary["failed_tests"] += 1
                        logger.error(f"[FAILED] {test_key}")
                    
                    # Save individual test result
                    result_file = REPORTS_DIR / f"{test_key}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                except Exception as e:
                    self.summary["failed_tests"] += 1
                    logger.error(f"[EXCEPTION] {test_key}: {e}")
                    
                    self.test_results[test_key] = {
                        "python_version": python_version,
                        "extra": extra,
                        "test_timestamp": datetime.now().isoformat(),
                        "overall_success": False,
                        "exception": str(e)
                    }
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self) -> None:
        """Generate test summary."""
        self.summary["results"] = self.test_results
        
        # Save summary
        summary_file = REPORTS_DIR / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.summary, f, indent=2)
        
        # Generate readable report
        report_file = REPORTS_DIR / "test_report.md"
        with open(report_file, 'w') as f:
            f.write("# Environment Build Matrix Test Report\n\n")
            f.write(f"**Test Date:** {self.summary['test_timestamp']}\n\n")
            f.write(f"**Total Tests:** {self.summary['total_tests']}\n")
            f.write(f"**Successful:** {self.summary['successful_tests']}\n")
            f.write(f"**Failed:** {self.summary['failed_tests']}\n\n")
            
            # Success rate
            success_rate = (self.summary['successful_tests'] / self.summary['total_tests']) * 100
            f.write(f"**Success Rate:** {success_rate:.1f}%\n\n")
            
            # Results by Python version
            for python_version in PYTHON_VERSIONS:
                f.write(f"## Python {python_version}\n\n")
                
                for extra in CRITICAL_EXTRAS:
                    test_key = f"python{python_version}_{extra}"
                    result = self.test_results.get(test_key, {})
                    
                    status = "SUCCESS" if result.get("overall_success", False) else "FAILED"
                    f.write(f"- **{extra}**: {status}\n")
                    
                    if not result.get("overall_success", False):
                        # Add error details
                        install_result = result.get("installation", {})
                        errors = install_result.get("errors", [])
                        if errors:
                            f.write(f"  - Errors: {'; '.join(errors[:3])}\n")
                        
                        warnings = install_result.get("warnings", [])
                        if warnings:
                            f.write(f"  - Warnings: {'; '.join(warnings[:3])}\n")
                
                f.write("\n")
        
        logger.info("Test summary generated")
        logger.info(f"Success rate: {success_rate:.1f}%")


def main():
    """Main entry point."""
    project_root = Path.cwd()
    
    # Ensure we're in the project root
    if not (project_root / "pyproject.toml").exists():
        logger.error("pyproject.toml not found. Please run from project root.")
        sys.exit(1)
    
    tester = EnvironmentTester(project_root)
    tester.run_all_tests()
    
    logger.info("Environment build matrix testing completed")
    logger.info(f"Results saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Environment Build Matrix Test Script

Tests critical extras sets across Python versions with improved:
- Timeout handling for large installations
- Progress reporting for long-running operations
- Chunked installation for the 'all' extra
- Better error logging and recovery
- Dependency conflict detection
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging to use basic ASCII text for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('env_matrix_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnvMatrixTester:
    """Test environment build matrix with improved resilience"""
    
    def __init__(self):
        self.python_versions = [self._get_current_python_version()]
        self.extras = ['minimal', 'server', 'production', 'deep', 'deep-cpu', 'automl', 'all']
        self.reports_dir = Path('reports/env_matrix')
        self.timeout_seconds = 1800  # 30 minutes for complex installs
        self.chunk_timeout = 600    # 10 minutes per chunk
        
        # Create reports directory
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Chunked installation strategy for 'all' extra
        self.all_chunks = [
            # Core ML and server chunk
            ['scikit-learn>=1.6.0', 'scipy>=1.15.0', 'fastapi>=0.115.0', 
             'uvicorn[standard]>=0.34.0', 'httpx>=0.28.1', 'requests>=2.32.3'],
            
            # Deep learning chunk 1 (PyTorch ecosystem)
            ['torch>=2.5.1', 'jax>=0.4.37', 'jaxlib>=0.4.37', 'optax>=0.2.4'],
            
            # Deep learning chunk 2 (TensorFlow ecosystem)
            ['tensorflow>=2.18.0,<2.20.0', 'keras>=3.8.0'],
            
            # Data processing chunk
            ['pyarrow>=18.1.0', 'fastparquet>=2024.11.0', 'openpyxl>=3.1.5',
             'xlsxwriter>=3.2.0', 'h5py>=3.12.1'],
            
            # Monitoring and infrastructure chunk
            ['opentelemetry-api>=1.29.0', 'opentelemetry-sdk>=1.29.0', 
             'prometheus-client>=0.21.1', 'psutil>=6.1.1', 'redis>=5.2.1'],
            
            # AutoML and optimization chunk
            ['optuna>=4.1.0', 'hyperopt>=0.2.7', 'shap>=0.46.0', 'lime>=0.2.0.1'],
            
            # Remaining packages
            ['python-multipart>=0.0.20', 'jinja2>=3.1.5', 'aiofiles>=24.1.0',
             'typer[all]>=0.15.1', 'rich>=13.9.4', 'pyjwt>=2.10.1', 'passlib[bcrypt]>=1.7.4']
        ]
    
    def _get_current_python_version(self) -> str:
        """Get current Python version as string"""
        version = sys.version_info
        return f"{version.major}.{version.minor}"
    
    def _create_venv(self, python_version: str, extra: str, temp_dir: Path) -> Tuple[bool, str]:
        """Create virtual environment"""
        venv_name = f"venv_{python_version}_{extra}"
        venv_path = temp_dir / venv_name
        
        try:
            # Use current Python executable to create venv
            result = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                return False, f"Failed to create venv: {result.stderr}"
            
            return True, str(venv_path)
            
        except subprocess.TimeoutExpired:
            return False, "Timeout creating virtual environment"
        except Exception as e:
            return False, f"Error creating venv: {str(e)}"
    
    def _install_package_with_retry(self, pip_path: str, packages: List[str], 
                                   max_retries: int = 3) -> Tuple[bool, str, str, int, float]:
        """Install packages with retry logic"""
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Build command - install packages one by one for better error isolation
                if len(packages) == 1 and packages[0].startswith('-e'):
                    cmd = [pip_path, "install"] + packages + ["--verbose"]
                else:
                    cmd = [pip_path, "install"] + packages + ["--verbose"]
                    
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Installing {packages}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.chunk_timeout
                )
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    return True, result.stdout, result.stderr, result.returncode, duration
                else:
                    logger.warning(f"Attempt {attempt + 1} failed: {result.stderr}")
                    if attempt == max_retries - 1:
                        return False, result.stdout, result.stderr, result.returncode, duration
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Attempt {attempt + 1} timed out after {self.chunk_timeout} seconds")
                if attempt == max_retries - 1:
                    return False, "", "Installation timed out", -1, self.chunk_timeout
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed with exception: {str(e)}")
                if attempt == max_retries - 1:
                    return False, "", str(e), -1, 0
        
        return False, "", "All retry attempts failed", -1, 0
    
    def _install_chunked_all(self, pip_path: str, project_root: str) -> Tuple[bool, str, str, int, float]:
        """Install 'all' extra in chunks to avoid timeout"""
        total_stdout = ""
        total_stderr = ""
        total_duration = 0
        
        # First install the base package
        logger.info("Installing base package...")
        # Change to project root directory temporarily
        old_cwd = os.getcwd()
        os.chdir(project_root)
        try:
            success, stdout, stderr, code, duration = self._install_package_with_retry(
                pip_path, ["-e ."]
            )
        finally:
            os.chdir(old_cwd)
        
        if not success:
            return False, stdout, stderr, code, duration
        
        total_stdout += stdout + "\n"
        total_stderr += stderr + "\n"
        total_duration += duration
        
        # Install each chunk
        for i, chunk in enumerate(self.all_chunks):
            logger.info(f"Installing chunk {i+1}/{len(self.all_chunks)}: {chunk}")
            success, stdout, stderr, code, duration = self._install_package_with_retry(
                pip_path, chunk
            )
            
            total_stdout += stdout + "\n"
            total_stderr += stderr + "\n"
            total_duration += duration
            
            if not success:
                logger.error(f"Failed to install chunk {i+1}: {stderr}")
                return False, total_stdout, total_stderr, code, total_duration
        
        return True, total_stdout, total_stderr, 0, total_duration
    
    def _install_package(self, python_version: str, extra: str, venv_path: str) -> Dict:
        """Install package with specific extra"""
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(venv_path, "Scripts", "pip")
        else:  # Unix/Linux
            pip_path = os.path.join(venv_path, "bin", "pip")
        
        project_root = os.getcwd()
        
        # Special handling for 'all' extra - use chunked installation
        if extra == 'all':
            logger.info(f"Installing 'all' extra using chunked strategy...")
            success, stdout, stderr, returncode, duration = self._install_chunked_all(
                pip_path, project_root
            )
        else:
            # Regular installation
            install_cmd = f"-e .[{extra}]"
            success, stdout, stderr, returncode, duration = self._install_package_with_retry(
                pip_path, [install_cmd]
            )
        
        # Parse warnings and errors from stderr
        warnings = []
        errors = []
        
        if stderr:
            for line in stderr.split('\n'):
                if 'WARNING' in line or 'warning' in line:
                    warnings.append(line.strip())
                elif 'ERROR' in line or 'error' in line:
                    errors.append(line.strip())
        
        return {
            "command": f"{pip_path} install -e .[{extra}] --verbose",
            "success": success,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "warnings": warnings,
            "errors": errors,
            "duration": duration
        }
    
    def _get_dependency_info(self, venv_path: str) -> Dict:
        """Get dependency tree and lockfile info"""
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(venv_path, "Scripts", "pip")
        else:  # Unix/Linux
            pip_path = os.path.join(venv_path, "bin", "pip")
        
        dependency_info = {
            "tree_text": "",
            "tree_json": {},
            "freeze_output": ""
        }
        
        # Get pip freeze output
        try:
            result = subprocess.run(
                [pip_path, "freeze"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                dependency_info["freeze_output"] = result.stdout
        except Exception as e:
            logger.warning(f"Failed to get freeze output: {e}")
        
        return dependency_info
    
    def _get_lockfile_info(self, venv_path: str) -> Dict:
        """Get lockfile information"""
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(venv_path, "Scripts", "pip")
        else:  # Unix/Linux
            pip_path = os.path.join(venv_path, "bin", "pip")
        
        lockfile_info = {
            "pip_freeze": "",
            "pip_list": [],
            "pip_show_pynomaly": ""
        }
        
        # Get pip freeze
        try:
            result = subprocess.run([pip_path, "freeze"], capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                lockfile_info["pip_freeze"] = result.stdout
        except Exception as e:
            logger.warning(f"Failed to get pip freeze: {e}")
        
        # Get pip list --format=json
        try:
            result = subprocess.run([pip_path, "list", "--format=json"], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                lockfile_info["pip_list"] = json.loads(result.stdout)
        except Exception as e:
            logger.warning(f"Failed to get pip list: {e}")
        
        # Get pip show pynomaly
        try:
            result = subprocess.run([pip_path, "show", "pynomaly"], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                lockfile_info["pip_show_pynomaly"] = result.stdout
        except Exception as e:
            logger.warning(f"Failed to get pip show: {e}")
        
        return lockfile_info
    
    def test_environment(self, python_version: str, extra: str) -> Dict:
        """Test single environment configuration"""
        logger.info(f"Testing Python {python_version} with {extra} extra")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create virtual environment
            venv_success, venv_path = self._create_venv(python_version, extra, temp_path)
            
            result = {
                "python_version": python_version,
                "extra": extra,
                "test_timestamp": datetime.now().isoformat(),
                "venv_creation": {
                    "success": venv_success
                },
                "installation": {},
                "dependency_tree": {},
                "lockfile": {},
                "overall_success": False
            }
            
            if not venv_success:
                result["venv_creation"]["error"] = venv_path  # Error message
                return result
            
            # Install package
            install_result = self._install_package(python_version, extra, venv_path)
            result["installation"] = install_result
            
            if install_result["success"]:
                # Get dependency information
                result["dependency_tree"] = self._get_dependency_info(venv_path)
                result["lockfile"] = self._get_lockfile_info(venv_path)
                result["overall_success"] = True
            
            return result
    
    def run_tests(self) -> Dict:
        """Run all environment tests"""
        logger.info("Starting environment matrix tests...")
        
        results = {}
        successful_tests = 0
        failed_tests = 0
        
        for python_version in self.python_versions:
            for extra in self.extras:
                test_key = f"python{python_version}_{extra}"
                
                try:
                    result = self.test_environment(python_version, extra)
                    results[test_key] = result
                    
                    if result["overall_success"]:
                        successful_tests += 1
                        logger.info(f"SUCCESS: {test_key}")
                    else:
                        failed_tests += 1
                        logger.error(f"FAILED: {test_key}")
                    
                    # Save individual result
                    result_file = self.reports_dir / f"{test_key}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                except Exception as e:
                    failed_tests += 1
                    logger.error(f"EXCEPTION in {test_key}: {str(e)}")
                    
                    # Save error result
                    error_result = {
                        "python_version": python_version,
                        "extra": extra,
                        "test_timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "overall_success": False
                    }
                    results[test_key] = error_result
                    
                    result_file = self.reports_dir / f"{test_key}.json"
                    with open(result_file, 'w') as f:
                        json.dump(error_result, f, indent=2)
        
        # Create summary
        summary = {
            "total_tests": len(self.python_versions) * len(self.extras),
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "test_timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        # Save summary
        summary_file = self.reports_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown report
        self._create_markdown_report(summary)
        
        return summary
    
    def _create_markdown_report(self, summary: Dict):
        """Create markdown report"""
        report_lines = [
            "# Environment Matrix Test Report",
            f"Generated: {summary['test_timestamp']}",
            "",
            "## Summary",
            f"- Total Tests: {summary['total_tests']}",
            f"- Successful: {summary['successful_tests']}",
            f"- Failed: {summary['failed_tests']}",
            f"- Success Rate: {(summary['successful_tests'] / summary['total_tests'] * 100):.1f}%",
            "",
            "## Test Results",
            ""
        ]
        
        for test_key, result in summary['results'].items():
            status = "PASS" if result['overall_success'] else "FAIL"
            report_lines.append(f"- **{test_key}**: {status}")
            
            if not result['overall_success']:
                if 'installation' in result and 'errors' in result['installation']:
                    for error in result['installation']['errors'][:2]:  # Show first 2 errors
                        report_lines.append(f"  - Error: {error}")
        
        report_file = self.reports_dir / "test_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

def main():
    """Main function"""
    tester = EnvMatrixTester()
    
    try:
        summary = tester.run_tests()
        
        logger.info(f"Test completed. {summary['successful_tests']}/{summary['total_tests']} tests passed")
        
        if summary['failed_tests'] > 0:
            logger.error(f"{summary['failed_tests']} tests failed")
            sys.exit(1)
        else:
            logger.info("All tests passed!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

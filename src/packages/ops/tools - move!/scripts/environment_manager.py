#!/usr/bin/env python3
"""
Environment Management Script for Multi-Environment Testing - Issue #214

This script provides utilities for setting up, validating, and managing
test environments across different platforms and configurations.
"""

import json
import os
import platform
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import logging

import yaml


class EnvironmentManager:
    """Manages test environments for multi-platform testing."""
    
    def __init__(self, config_file: Path = None):
        self.config_file = config_file or Path("tests/test_matrix_enhanced.yml")
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _load_config(self) -> Dict[str, Any]:
        """Load test matrix configuration."""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_file}")
            return {}
    
    def get_current_platform(self) -> str:
        """Get current platform identifier."""
        system = platform.system().lower()
        if "linux" in system:
            return "ubuntu-latest"
        elif "windows" in system:
            return "windows-latest"
        elif "darwin" in system:
            return "macos-latest"
        else:
            return "ubuntu-latest"  # Default fallback
    
    def create_virtual_environment(self, env_name: str, python_version: str) -> Path:
        """Create a virtual environment for testing."""
        env_dir = Path(f"test_environments/{env_name}")
        env_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating virtual environment: {env_name}")
        
        try:
            # Create virtual environment
            venv.create(env_dir, with_pip=True, clear=True)
            
            # Get paths
            if platform.system() == "Windows":
                python_executable = env_dir / "Scripts" / "python.exe"
                pip_executable = env_dir / "Scripts" / "pip.exe"
            else:
                python_executable = env_dir / "bin" / "python"
                pip_executable = env_dir / "bin" / "pip"
            
            # Upgrade pip
            subprocess.run([
                str(python_executable), "-m", "pip", "install", "--upgrade", "pip"
            ], check=True)
            
            self.logger.info(f"✅ Virtual environment created: {env_dir}")
            return env_dir
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create virtual environment: {e}")
            raise
    
    def install_dependencies(self, env_dir: Path, dependency_group: str) -> bool:
        """Install dependencies for a specific group."""
        self.logger.info(f"Installing dependencies for group: {dependency_group}")
        
        # Get executable paths
        if platform.system() == "Windows":
            python_executable = env_dir / "Scripts" / "python.exe"
            pip_executable = env_dir / "Scripts" / "pip.exe"
        else:
            python_executable = env_dir / "bin" / "python"
            pip_executable = env_dir / "bin" / "pip"
        
        try:
            # Get dependency configuration
            dep_config = self.config.get("dependency_groups", {}).get(dependency_group, {})
            
            if "install_command" in dep_config:
                # Use custom install command
                install_cmd = dep_config["install_command"].split()
                subprocess.run([str(python_executable)] + install_cmd[1:], check=True)
            else:
                # Install base dependencies first
                base_group = dep_config.get("base")
                if base_group and base_group != dependency_group:
                    self.install_dependencies(env_dir, base_group)
                
                # Install main packages
                packages = dep_config.get("packages", [])
                if packages:
                    install_cmd = [str(pip_executable), "install"] + packages
                    subprocess.run(install_cmd, check=True)
                
                # Install additional packages
                additional_packages = dep_config.get("additional_packages", [])
                if additional_packages:
                    install_cmd = [str(pip_executable), "install"] + additional_packages
                    subprocess.run(install_cmd, check=True)
                
                # Install main package in development mode
                subprocess.run([
                    str(pip_executable), "install", "-e", "."
                ], check=True)
            
            self.logger.info(f"✅ Dependencies installed for {dependency_group}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def validate_environment(self, env_dir: Path, dependency_group: str) -> Dict[str, Any]:
        """Validate environment setup and functionality."""
        self.logger.info(f"Validating environment: {env_dir.name}")
        
        # Get executable paths
        if platform.system() == "Windows":
            python_executable = env_dir / "Scripts" / "python.exe"
        else:
            python_executable = env_dir / "bin" / "python"
        
        validation_results = {
            "environment": env_dir.name,
            "dependency_group": dependency_group,
            "platform": platform.system(),
            "python_version": None,
            "core_imports": {},
            "dependency_validation": {},
            "functional_tests": {},
        }
        
        try:
            # Check Python version
            result = subprocess.run([
                str(python_executable), "-c", 
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            ], capture_output=True, text=True, check=True)
            validation_results["python_version"] = result.stdout.strip()
            
            # Test core imports
            core_imports = [
                "monorepo",
                "numpy", 
                "pandas",
                "pydantic",
                "pyod",
            ]
            
            for module in core_imports:
                try:
                    subprocess.run([
                        str(python_executable), "-c", f"import {module}"
                    ], check=True, capture_output=True)
                    validation_results["core_imports"][module] = True
                except subprocess.CalledProcessError:
                    validation_results["core_imports"][module] = False
            
            # Test group-specific dependencies
            dep_config = self.config.get("dependency_groups", {}).get(dependency_group, {})
            group_deps = dep_config.get("additional_packages", [])
            
            for package in group_deps:
                # Extract package name (remove version specifiers)
                package_name = package.split(">=")[0].split("==")[0].split("[")[0]
                try:
                    subprocess.run([
                        str(python_executable), "-c", f"import {package_name}"
                    ], check=True, capture_output=True)
                    validation_results["dependency_validation"][package_name] = True
                except subprocess.CalledProcessError:
                    validation_results["dependency_validation"][package_name] = False
            
            # Run functional tests
            validation_results["functional_tests"]["basic_import"] = self._test_basic_functionality(python_executable)
            validation_results["functional_tests"]["cli_help"] = self._test_cli_functionality(python_executable)
            
            self.logger.info(f"✅ Environment validation completed")
            
        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            validation_results["error"] = str(e)
        
        return validation_results
    
    def _test_basic_functionality(self, python_executable: Path) -> bool:
        """Test basic Pynomaly functionality."""
        try:
            test_script = """
import monorepo
from monorepo.domain.entities import Dataset, Detector
from monorepo.application.services import DetectionService
print("✅ Basic functionality test passed")
"""
            subprocess.run([
                str(python_executable), "-c", test_script
            ], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _test_cli_functionality(self, python_executable: Path) -> bool:
        """Test CLI functionality."""
        try:
            subprocess.run([
                str(python_executable), "-m", "monorepo.presentation.cli.app", "--help"
            ], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def setup_environment_matrix(self, scope: str = "quick") -> List[Dict[str, Any]]:
        """Setup complete environment matrix for testing."""
        self.logger.info(f"Setting up environment matrix with scope: {scope}")
        
        # Get scope configuration
        scope_config = self.config.get("testing_scopes", {}).get(scope, {})
        if not scope_config:
            self.logger.error(f"Unknown testing scope: {scope}")
            return []
        
        current_platform = self.get_current_platform()
        if current_platform not in scope_config.get("os_matrix", []):
            self.logger.warning(f"Current platform {current_platform} not in scope matrix")
            return []
        
        environments = []
        
        for python_version in scope_config.get("python_versions", []):
            for dep_group in scope_config.get("dependency_groups", []):
                env_name = f"{current_platform}-py{python_version}-{dep_group}"
                
                self.logger.info(f"Setting up environment: {env_name}")
                
                try:
                    # Create environment
                    env_dir = self.create_virtual_environment(env_name, python_version)
                    
                    # Install dependencies
                    install_success = self.install_dependencies(env_dir, dep_group)
                    
                    if install_success:
                        # Validate environment
                        validation_results = self.validate_environment(env_dir, dep_group)
                        
                        environment_info = {
                            "name": env_name,
                            "path": str(env_dir),
                            "python_version": python_version,
                            "dependency_group": dep_group,
                            "platform": current_platform,
                            "test_categories": scope_config.get("test_categories", []),
                            "validation": validation_results,
                            "setup_success": True,
                        }
                    else:
                        environment_info = {
                            "name": env_name,
                            "setup_success": False,
                            "error": "Dependency installation failed",
                        }
                    
                    environments.append(environment_info)
                    
                except Exception as e:
                    self.logger.error(f"Failed to setup environment {env_name}: {e}")
                    environments.append({
                        "name": env_name,
                        "setup_success": False,
                        "error": str(e),
                    })
        
        # Save environment matrix
        self._save_environment_matrix(environments, scope)
        
        return environments
    
    def _save_environment_matrix(self, environments: List[Dict[str, Any]], scope: str) -> None:
        """Save environment matrix configuration."""
        output_dir = Path("test_environments")
        output_dir.mkdir(exist_ok=True)
        
        matrix_file = output_dir / f"environment_matrix_{scope}.json"
        
        matrix_data = {
            "meta": {
                "scope": scope,
                "generated_at": str(pd.Timestamp.now()),
                "platform": self.get_current_platform(),
                "total_environments": len(environments),
                "successful_environments": len([e for e in environments if e.get("setup_success", False)]),
            },
            "environments": environments,
        }
        
        with open(matrix_file, 'w') as f:
            json.dump(matrix_data, f, indent=2)
        
        self.logger.info(f"Environment matrix saved to: {matrix_file}")
    
    def run_tests_in_environment(self, env_name: str, test_categories: List[str]) -> Dict[str, Any]:
        """Run specific test categories in an environment."""
        env_dir = Path(f"test_environments/{env_name}")
        
        if not env_dir.exists():
            raise ValueError(f"Environment not found: {env_name}")
        
        # Get executable paths
        if platform.system() == "Windows":
            python_executable = env_dir / "Scripts" / "python.exe"
        else:
            python_executable = env_dir / "bin" / "python"
        
        results = {
            "environment": env_name,
            "test_results": {},
            "overall_status": "success",
        }
        
        for category in test_categories:
            self.logger.info(f"Running {category} tests in {env_name}")
            
            # Get test command
            test_config = self.config.get("test_categories", {}).get(category, {})
            test_command = test_config.get("command", f"pytest tests/{category}/")
            
            try:
                # Set environment variables
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path.cwd() / "src")
                
                # Run test
                result = subprocess.run([
                    str(python_executable), "-m"
                ] + test_command.split()[1:], 
                    capture_output=True, 
                    text=True, 
                    timeout=test_config.get("timeout", 300),
                    env=env
                )
                
                test_status = "passed" if result.returncode == 0 else "failed"
                
                results["test_results"][category] = {
                    "status": test_status,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
                    "stderr": result.stderr[-1000:] if result.stderr else "",  # Last 1000 chars
                }
                
                if test_status == "failed":
                    results["overall_status"] = "failed"
                
            except subprocess.TimeoutExpired:
                results["test_results"][category] = {
                    "status": "timeout",
                    "error": f"Test timed out after {test_config.get('timeout', 300)} seconds",
                }
                results["overall_status"] = "failed"
                
            except Exception as e:
                results["test_results"][category] = {
                    "status": "error",
                    "error": str(e),
                }
                results["overall_status"] = "failed"
        
        return results
    
    def cleanup_environments(self, scope: str = None) -> None:
        """Cleanup test environments."""
        env_base_dir = Path("test_environments")
        
        if not env_base_dir.exists():
            self.logger.info("No test environments to cleanup")
            return
        
        environments_cleaned = 0
        
        for env_dir in env_base_dir.iterdir():
            if env_dir.is_dir() and env_dir.name.startswith(self.get_current_platform()):
                if scope is None or scope in env_dir.name:
                    try:
                        import shutil
                        shutil.rmtree(env_dir)
                        environments_cleaned += 1
                        self.logger.info(f"Cleaned up environment: {env_dir.name}")
                    except Exception as e:
                        self.logger.error(f"Failed to cleanup {env_dir.name}: {e}")
        
        self.logger.info(f"Cleaned up {environments_cleaned} environments")
    
    def generate_environment_report(self, scope: str) -> None:
        """Generate comprehensive environment report."""
        matrix_file = Path(f"test_environments/environment_matrix_{scope}.json")
        
        if not matrix_file.exists():
            self.logger.error(f"Environment matrix file not found: {matrix_file}")
            return
        
        with open(matrix_file, 'r') as f:
            matrix_data = json.load(f)
        
        # Generate report
        report_file = Path(f"test_environments/environment_report_{scope}.md")
        
        with open(report_file, 'w') as f:
            f.write(f"# Environment Setup Report - {scope.upper()}\n\n")
            f.write(f"**Generated:** {matrix_data['meta']['generated_at']}\n")
            f.write(f"**Platform:** {matrix_data['meta']['platform']}\n")
            f.write(f"**Total Environments:** {matrix_data['meta']['total_environments']}\n")
            f.write(f"**Successful:** {matrix_data['meta']['successful_environments']}\n\n")
            
            # Environment details
            f.write("## Environment Details\n\n")
            
            for env in matrix_data["environments"]:
                status_icon = "✅" if env.get("setup_success", False) else "❌"
                f.write(f"### {status_icon} {env['name']}\n\n")
                
                if env.get("setup_success", False):
                    validation = env.get("validation", {})
                    f.write(f"- **Python Version:** {validation.get('python_version', 'Unknown')}\n")
                    f.write(f"- **Dependency Group:** {env.get('dependency_group', 'Unknown')}\n")
                    
                    # Core imports
                    core_imports = validation.get("core_imports", {})
                    working_imports = [k for k, v in core_imports.items() if v]
                    failed_imports = [k for k, v in core_imports.items() if not v]
                    
                    f.write(f"- **Working Imports:** {', '.join(working_imports) if working_imports else 'None'}\n")
                    if failed_imports:
                        f.write(f"- **Failed Imports:** {', '.join(failed_imports)}\n")
                    
                    # Functional tests
                    functional = validation.get("functional_tests", {})
                    f.write(f"- **Basic Functionality:** {'✅' if functional.get('basic_import', False) else '❌'}\n")
                    f.write(f"- **CLI Functionality:** {'✅' if functional.get('cli_help', False) else '❌'}\n")
                else:
                    f.write(f"- **Error:** {env.get('error', 'Unknown error')}\n")
                
                f.write("\n")
        
        self.logger.info(f"Environment report generated: {report_file}")


def main():
    """Main entry point for environment manager."""
    parser = argparse.ArgumentParser(
        description="Environment Manager for Multi-Environment Testing (Issue #214)"
    )
    parser.add_argument(
        "action",
        choices=["setup", "validate", "test", "cleanup", "report"],
        help="Action to perform"
    )
    parser.add_argument(
        "--scope",
        choices=["quick", "standard", "comprehensive", "stress"],
        default="quick",
        help="Testing scope (default: quick)"
    )
    parser.add_argument(
        "--environment",
        help="Specific environment name (for test action)"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Test categories to run (for test action)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to test matrix configuration file"
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = EnvironmentManager(config_file=args.config)
    
    try:
        if args.action == "setup":
            environments = manager.setup_environment_matrix(args.scope)
            print(f"✅ Setup completed: {len(environments)} environments")
            
        elif args.action == "validate":
            matrix_file = Path(f"test_environments/environment_matrix_{args.scope}.json")
            if matrix_file.exists():
                with open(matrix_file, 'r') as f:
                    matrix_data = json.load(f)
                successful = matrix_data['meta']['successful_environments']
                total = matrix_data['meta']['total_environments']
                print(f"✅ Validation: {successful}/{total} environments ready")
            else:
                print(f"❌ No environment matrix found for scope: {args.scope}")
                
        elif args.action == "test":
            if not args.environment or not args.categories:
                print("❌ Environment name and test categories required for test action")
                sys.exit(1)
            
            results = manager.run_tests_in_environment(args.environment, args.categories)
            status = results["overall_status"]
            print(f"{'✅' if status == 'success' else '❌'} Tests completed: {status}")
            
        elif args.action == "cleanup":
            manager.cleanup_environments(args.scope)
            print("✅ Cleanup completed")
            
        elif args.action == "report":
            manager.generate_environment_report(args.scope)
            print("✅ Report generated")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Add required import for timestamp
    try:
        import pandas as pd
    except ImportError:
        import datetime
        # Fallback if pandas not available
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    return datetime.datetime.now().isoformat()
    
    main()
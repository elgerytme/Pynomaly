#!/usr/bin/env python3
"""
Automated Development Environment Setup Script

This script automates the setup of a complete development environment for Pynomaly.
It handles virtual environment creation, dependency installation, and development tools.
"""

import argparse
import os
import platform
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Optional


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class DevSetup:
    """Development environment setup automation."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.environments_dir = self.project_root / "environments"
        self.venv_path = self.environments_dir / ".venv"
        self.system = platform.system().lower()
        self.python_cmd = self._find_python()
        
    def _find_python(self) -> str:
        """Find the best Python executable."""
        candidates = ["python3.12", "python3.11", "python3", "python"]
        
        for cmd in candidates:
            if shutil.which(cmd):
                try:
                    result = subprocess.run([cmd, "--version"], 
                                         capture_output=True, text=True)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        if "3.11" in version or "3.12" in version or "3.13" in version:
                            return cmd
                except subprocess.SubprocessError:
                    continue
        
        raise RuntimeError("Python 3.11+ not found. Please install Python 3.11 or higher.")
    
    def _run_command(self, cmd: List[str], description: str, 
                    cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command with nice output."""
        print(f"{Colors.BLUE}→{Colors.END} {description}")
        print(f"  {Colors.CYAN}${Colors.END} {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=cwd or self.project_root,
                check=check,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    print(f"  {line}")
            
            print(f"{Colors.GREEN}✓{Colors.END} {description} completed\n")
            return result
            
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}✗{Colors.END} {description} failed")
            if e.stderr:
                print(f"Error: {e.stderr}")
            if not check:
                return e
            raise
    
    def _get_activation_script(self) -> str:
        """Get the appropriate venv activation script."""
        if self.system == "windows":
            return str(self.venv_path / "Scripts" / "activate.bat")
        else:
            return f"source {self.venv_path}/bin/activate"
    
    def _get_pip_path(self) -> Path:
        """Get the pip executable path in the virtual environment."""
        if self.system == "windows":
            return self.venv_path / "Scripts" / "pip"
        else:
            return self.venv_path / "bin" / "pip"
    
    def _get_python_path(self) -> Path:
        """Get the Python executable path in the virtual environment."""
        if self.system == "windows":
            return self.venv_path / "Scripts" / "python"
        else:
            return self.venv_path / "bin" / "python"
    
    def check_prerequisites(self) -> bool:
        """Check if prerequisites are installed."""
        print(f"{Colors.BOLD}{Colors.WHITE}🔍 Checking Prerequisites{Colors.END}\n")
        
        success = True
        
        # Check Python version
        try:
            result = subprocess.run([self.python_cmd, "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"{Colors.GREEN}✓{Colors.END} {version} found")
            else:
                print(f"{Colors.RED}✗{Colors.END} Python not accessible")
                success = False
        except FileNotFoundError:
            print(f"{Colors.RED}✗{Colors.END} Python not found")
            success = False
        
        # Check Git
        try:
            result = subprocess.run(["git", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"{Colors.GREEN}✓{Colors.END} {version} found")
            else:
                print(f"{Colors.RED}✗{Colors.END} Git not accessible")
                success = False
        except FileNotFoundError:
            print(f"{Colors.RED}✗{Colors.END} Git not found. Please install Git.")
            success = False
        
        # Check if we're in the right directory
        if not (self.project_root / "pyproject.toml").exists():
            print(f"{Colors.RED}✗{Colors.END} pyproject.toml not found. Are you in the Pynomaly root directory?")
            success = False
        else:
            print(f"{Colors.GREEN}✓{Colors.END} Project root directory confirmed")
        
        print()
        return success
    
    def create_virtual_environment(self) -> None:
        """Create virtual environment."""
        print(f"{Colors.BOLD}{Colors.WHITE}🏗️ Creating Virtual Environment{Colors.END}\n")
        
        # Create environments directory
        self.environments_dir.mkdir(exist_ok=True)
        
        # Remove existing venv if it exists
        if self.venv_path.exists():
            print(f"{Colors.YELLOW}⚠{Colors.END} Removing existing virtual environment")
            shutil.rmtree(self.venv_path)
        
        # Create new virtual environment
        self._run_command(
            [self.python_cmd, "-m", "venv", str(self.venv_path)],
            "Creating virtual environment"
        )
        
        # Upgrade pip
        pip_path = str(self._get_pip_path())
        self._run_command(
            [pip_path, "install", "--upgrade", "pip", "setuptools", "wheel"],
            "Upgrading pip and setuptools"
        )
    
    def install_dependencies(self, include_dev: bool = True, include_optional: bool = True) -> None:
        """Install project dependencies."""
        print(f"{Colors.BOLD}{Colors.WHITE}📦 Installing Dependencies{Colors.END}\n")
        
        pip_path = str(self._get_pip_path())
        
        # Install core package
        install_cmd = [pip_path, "install", "-e", "."]
        if include_dev:
            install_cmd.append("[dev,test,lint]")
        
        self._run_command(install_cmd, "Installing core package with development dependencies")
        
        # Install optional dependencies
        if include_optional:
            optional_packages = [
                # ML frameworks
                "torch>=2.0.0",
                "torchvision",
                "tensorflow>=2.13.0",
                # Data science
                "jupyter",
                "notebook",
                "ipykernel",
                # Visualization
                "matplotlib",
                "seaborn",
                "plotly",
                # Development tools
                "pre-commit",
                "coverage[toml]",
                "pytest-xdist",
                "pytest-benchmark",
            ]
            
            for package in optional_packages:
                try:
                    self._run_command(
                        [pip_path, "install", package],
                        f"Installing {package}",
                        check=False
                    )
                except subprocess.CalledProcessError:
                    print(f"{Colors.YELLOW}⚠{Colors.END} Optional package {package} failed to install")
    
    def setup_development_tools(self) -> None:
        """Setup development tools and configurations."""
        print(f"{Colors.BOLD}{Colors.WHITE}🔧 Setting Up Development Tools{Colors.END}\n")
        
        python_path = str(self._get_python_path())
        
        # Install pre-commit hooks
        if (self.project_root / ".pre-commit-config.yaml").exists():
            self._run_command(
                [python_path, "-m", "pre_commit", "install"],
                "Installing pre-commit hooks"
            )
        
        # Create .env file for development
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_content = f"""# Pynomaly Development Environment
PYNOMALY_ENVIRONMENT=development
PYNOMALY_LOG_LEVEL=DEBUG
PYNOMALY_CACHE_ENABLED=true
PYTHONPATH={self.project_root}/src
"""
            env_file.write_text(env_content)
            print(f"{Colors.GREEN}✓{Colors.END} Created .env file for development")
        
        # Setup VS Code configuration (if .vscode directory doesn't exist)
        vscode_dir = self.project_root / ".vscode"
        if not vscode_dir.exists():
            vscode_dir.mkdir()
            
            # Python interpreter settings
            settings = {
                "python.defaultInterpreterPath": str(self._get_python_path()),
                "python.terminal.activateEnvironment": True,
                "python.linting.enabled": True,
                "python.linting.ruffEnabled": True,
                "python.formatting.provider": "ruff",
                "python.testing.pytestEnabled": True,
                "python.testing.pytestArgs": ["tests/"],
                "files.exclude": {
                    "**/__pycache__": True,
                    "**/.mypy_cache": True,
                    "**/.ruff_cache": True,
                    "**/node_modules": True
                }
            }
            
            import json
            settings_file = vscode_dir / "settings.json"
            settings_file.write_text(json.dumps(settings, indent=2))
            print(f"{Colors.GREEN}✓{Colors.END} Created VS Code settings")
    
    def validate_installation(self) -> bool:
        """Validate the installation."""
        print(f"{Colors.BOLD}{Colors.WHITE}✅ Validating Installation{Colors.END}\n")
        
        python_path = str(self._get_python_path())
        success = True
        
        # Test Python import
        try:
            self._run_command(
                [python_path, "-c", "import monorepo; print(f'✓ Pynomaly v{monorepo.__version__} imported successfully')"],
                "Testing Pynomaly import"
            )
        except subprocess.CalledProcessError:
            success = False
        
        # Test basic functionality
        try:
            test_code = """
import numpy as np
import pandas as pd
from monorepo.domain.entities import Dataset
from monorepo.domain.value_objects import ContaminationRate
print('✓ Core domain objects imported successfully')
"""
            self._run_command(
                [python_path, "-c", test_code],
                "Testing core functionality"
            )
        except subprocess.CalledProcessError:
            success = False
        
        # Run a quick test
        try:
            self._run_command(
                [python_path, "-m", "pytest", "tests/unit/domain/", "-v", "--tb=short", "-x"],
                "Running basic domain tests"
            )
        except subprocess.CalledProcessError:
            print(f"{Colors.YELLOW}⚠{Colors.END} Some tests failed, but core installation seems OK")
        
        return success
    
    def print_next_steps(self) -> None:
        """Print next steps for the developer."""
        activation_cmd = self._get_activation_script()
        
        print(f"{Colors.BOLD}{Colors.GREEN}🎉 Development Environment Setup Complete!{Colors.END}\n")
        
        print(f"{Colors.BOLD}📋 Next Steps:{Colors.END}")
        print(f"  1. Activate your environment:")
        print(f"     {Colors.CYAN}{activation_cmd}{Colors.END}")
        print()
        print(f"  2. Verify installation:")
        print(f"     {Colors.CYAN}python -c \"import monorepo; print('Success!')\"{Colors.END}")
        print()
        print(f"  3. Run tests:")
        print(f"     {Colors.CYAN}pytest tests/unit/domain/ -v{Colors.END}")
        print()
        print(f"  4. Start development server:")
        print(f"     {Colors.CYAN}uvicorn monorepo.presentation.api.app:app --reload{Colors.END}")
        print()
        print(f"  5. Access documentation:")
        print(f"     • API docs: {Colors.CYAN}http://localhost:8000/docs{Colors.END}")
        print(f"     • Developer guide: {Colors.CYAN}docs/developer-guides/DEVELOPER_ONBOARDING.md{Colors.END}")
        print()
        
        print(f"{Colors.BOLD}🛠️ Development Commands:{Colors.END}")
        print(f"  • Format code: {Colors.CYAN}ruff format src/ tests/{Colors.END}")
        print(f"  • Lint code: {Colors.CYAN}ruff check src/ tests/{Colors.END}")
        print(f"  • Type check: {Colors.CYAN}mypy src/pynomaly/{Colors.END}")
        print(f"  • Run all tests: {Colors.CYAN}pytest tests/{Colors.END}")
        print(f"  • Security scan: {Colors.CYAN}bandit -r src/pynomaly/{Colors.END}")
        print()
        
        print(f"{Colors.BOLD}📚 Resources:{Colors.END}")
        print(f"  • Developer onboarding: {Colors.CYAN}docs/developer-guides/DEVELOPER_ONBOARDING.md{Colors.END}")
        print(f"  • Architecture guide: {Colors.CYAN}docs/architecture/README.md{Colors.END}")
        print(f"  • Contributing guide: {Colors.CYAN}docs/developer-guides/contributing/CONTRIBUTING.md{Colors.END}")
        print()
        
        print(f"{Colors.GREEN}Happy coding! 🚀{Colors.END}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated Pynomaly development environment setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup/setup_development.py                    # Full setup
  python scripts/setup/setup_development.py --minimal          # Minimal setup
  python scripts/setup/setup_development.py --check-only       # Check prerequisites only
        """
    )
    
    parser.add_argument(
        "--minimal", 
        action="store_true",
        help="Minimal setup without optional dependencies"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true", 
        help="Only check prerequisites, don't install anything"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    try:
        setup = DevSetup(args.project_root)
        
        # Always check prerequisites
        if not setup.check_prerequisites():
            print(f"{Colors.RED}❌ Prerequisites check failed. Please install missing requirements.{Colors.END}")
            sys.exit(1)
        
        if args.check_only:
            print(f"{Colors.GREEN}✅ Prerequisites check passed!{Colors.END}")
            return
        
        # Run setup
        setup.create_virtual_environment()
        setup.install_dependencies(
            include_dev=True,
            include_optional=not args.minimal
        )
        setup.setup_development_tools()
        
        # Validate installation
        if setup.validate_installation():
            setup.print_next_steps()
        else:
            print(f"{Colors.RED}❌ Installation validation failed.{Colors.END}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠ Setup interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}❌ Setup failed: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
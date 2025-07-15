#!/usr/bin/env python3
"""
Pynomaly Development Environment Setup

Automated setup script for Pynomaly development environment.
Handles workspace initialization, dependency installation, and tool configuration.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Optional
import json


class DevEnvironmentSetup:
    """Manages development environment setup for Pynomaly workspace."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.workspace_config = self._load_workspace_config()
        self.platform = platform.system().lower()
        
    def _load_workspace_config(self) -> dict:
        """Load workspace configuration."""
        config_path = self.root_path / "workspace.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed."""
        print("🔍 Checking prerequisites...")
        
        required_tools = {
            "python": "python --version",
            "git": "git --version", 
            "hatch": "hatch --version"
        }
        
        optional_tools = {
            "docker": "docker --version",
            "node": "node --version",
            "buck2": "buck2 --version"
        }
        
        all_good = True
        
        # Check required tools
        for tool, command in required_tools.items():
            if self._check_command(command):
                print(f"  ✅ {tool} is available")
            else:
                print(f"  ❌ {tool} is missing (required)")
                all_good = False
        
        # Check optional tools
        for tool, command in optional_tools.items():
            if self._check_command(command):
                print(f"  ✅ {tool} is available")
            else:
                print(f"  ⚠️  {tool} is missing (optional)")
        
        # Check Python version
        if self._check_python_version():
            print("  ✅ Python version is compatible")
        else:
            print("  ❌ Python version is too old (requires 3.11+)")
            all_good = False
        
        return all_good
    
    def _check_command(self, command: str) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run(command.split(), capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        try:
            version_info = sys.version_info
            return version_info >= (3, 11)
        except Exception:
            return False
    
    def setup_virtual_environment(self) -> bool:
        """Set up Python virtual environment."""
        print("🐍 Setting up Python virtual environment...")
        
        venv_path = self.root_path / "environments" / ".venv"
        
        if venv_path.exists():
            print("  ℹ️  Virtual environment already exists")
            return True
        
        try:
            # Create environments directory
            venv_path.parent.mkdir(exist_ok=True)
            
            # Create virtual environment
            subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], check=True)
            
            # Upgrade pip
            pip_cmd = self._get_pip_command(venv_path)
            subprocess.run([
                str(pip_cmd), "install", "--upgrade", "pip", "setuptools", "wheel"
            ], check=True)
            
            print(f"  ✅ Virtual environment created at {venv_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to create virtual environment: {e}")
            return False
    
    def _get_pip_command(self, venv_path: Path) -> Path:
        """Get the pip command for the virtual environment."""
        if self.platform == "windows":
            return venv_path / "Scripts" / "pip"
        else:
            return venv_path / "bin" / "pip"
    
    def _get_python_command(self, venv_path: Path) -> Path:
        """Get the python command for the virtual environment."""
        if self.platform == "windows":
            return venv_path / "Scripts" / "python"
        else:
            return venv_path / "bin" / "python"
    
    def install_dependencies(self) -> bool:
        """Install all workspace dependencies."""
        print("📦 Installing workspace dependencies...")
        
        venv_path = self.root_path / "environments" / ".venv"
        if not venv_path.exists():
            print("  ❌ Virtual environment not found. Run setup first.")
            return False
        
        pip_cmd = self._get_pip_command(venv_path)
        
        # Install development dependencies
        dev_deps = [
            "hatch",
            "pytest", "pytest-cov", "pytest-xdist", "pytest-mock",
            "black", "isort", "ruff", "mypy",
            "pre-commit",
            "build", "twine"
        ]
        
        try:
            subprocess.run([
                str(pip_cmd), "install", "--upgrade"
            ] + dev_deps, check=True)
            
            print("  ✅ Development dependencies installed")
            
            # Install main package in development mode
            subprocess.run([
                str(pip_cmd), "install", "-e", "."
            ], cwd=str(self.root_path), check=True)
            
            print("  ✅ Main package installed in development mode")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install dependencies: {e}")
            return False
    
    def setup_pre_commit(self) -> bool:
        """Set up pre-commit hooks."""
        print("🪝 Setting up pre-commit hooks...")
        
        venv_path = self.root_path / "environments" / ".venv"
        if not venv_path.exists():
            print("  ❌ Virtual environment not found")
            return False
        
        python_cmd = self._get_python_command(venv_path)
        
        try:
            # Install pre-commit hooks
            subprocess.run([
                str(python_cmd), "-m", "pre_commit", "install"
            ], cwd=str(self.root_path), check=True)
            
            print("  ✅ Pre-commit hooks installed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to setup pre-commit: {e}")
            return False
    
    def setup_ide_configuration(self) -> bool:
        """Set up IDE configuration files."""
        print("⚙️  Setting up IDE configuration...")
        
        # VS Code settings
        vscode_dir = self.root_path / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        # Create settings.json
        settings = {
            "python.defaultInterpreterPath": "./environments/.venv/bin/python",
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["tests/"],
            "python.formatting.provider": "black",
            "python.linting.enabled": True,
            "python.linting.ruffEnabled": True,
            "editor.formatOnSave": True,
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                "**/node_modules": True,
                "**/dist": True,
                "**/build": True
            }
        }
        
        with open(vscode_dir / "settings.json", "w") as f:
            json.dump(settings, f, indent=2)
        
        # Create launch.json for debugging
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Current File",
                    "type": "python",
                    "request": "launch",
                    "program": "${file}",
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}"
                },
                {
                    "name": "Python: Pytest",
                    "type": "python",
                    "request": "launch",
                    "module": "pytest",
                    "args": ["${workspaceFolder}/tests"],
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}"
                }
            ]
        }
        
        with open(vscode_dir / "launch.json", "w") as f:
            json.dump(launch_config, f, indent=2)
        
        print("  ✅ IDE configuration created")
        return True
    
    def validate_setup(self) -> bool:
        """Validate that the development environment is working."""
        print("✅ Validating development environment...")
        
        venv_path = self.root_path / "environments" / ".venv"
        python_cmd = self._get_python_command(venv_path)
        
        checks = [
            {
                "name": "Import main package",
                "command": [str(python_cmd), "-c", "import pynomaly; print('✅ Import successful')"]
            },
            {
                "name": "Run basic tests",
                "command": [str(python_cmd), "-m", "pytest", "--version"]
            },
            {
                "name": "Check formatting tools",
                "command": [str(python_cmd), "-m", "black", "--version"]
            },
            {
                "name": "Check linting tools", 
                "command": [str(python_cmd), "-m", "ruff", "--version"]
            }
        ]
        
        all_passed = True
        for check in checks:
            try:
                result = subprocess.run(
                    check["command"], 
                    capture_output=True, 
                    text=True,
                    cwd=str(self.root_path)
                )
                if result.returncode == 0:
                    print(f"  ✅ {check['name']}")
                else:
                    print(f"  ❌ {check['name']}: {result.stderr.strip()}")
                    all_passed = False
            except Exception as e:
                print(f"  ❌ {check['name']}: {e}")
                all_passed = False
        
        return all_passed
    
    def print_next_steps(self):
        """Print next steps for the developer."""
        venv_path = self.root_path / "environments" / ".venv"
        
        print("\n🎉 Development environment setup complete!")
        print("\n📋 Next steps:")
        print("1. Activate the virtual environment:")
        
        if self.platform == "windows":
            print(f"   .\\environments\\.venv\\Scripts\\activate")
        else:
            print(f"   source environments/.venv/bin/activate")
        
        print("\n2. Run workspace commands:")
        print("   python scripts/workspace.py list")
        print("   python scripts/workspace.py test all")
        print("   python scripts/workspace.py build all")
        
        print("\n3. Start development:")
        print("   code .  # Open VS Code")
        print("   pytest tests/  # Run tests")
        print("   hatch run format  # Format code")
        
        print(f"\n📁 Virtual environment: {venv_path}")
        print("📖 Documentation: docs/developer-guides/")
        print("🐛 Issues: https://github.com/elgerytme/Pynomaly/issues")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pynomaly Development Environment Setup")
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                       help="Project root directory")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip prerequisite checks")
    parser.add_argument("--minimal", action="store_true",
                       help="Minimal setup (no IDE config, pre-commit)")
    
    args = parser.parse_args()
    
    print("🚀 Pynomaly Development Environment Setup")
    print("=" * 50)
    
    setup = DevEnvironmentSetup(args.root)
    
    # Check prerequisites
    if not args.skip_checks:
        if not setup.check_prerequisites():
            print("\n❌ Prerequisites not met. Please install missing tools.")
            return 1
    
    # Setup steps
    steps = [
        ("Virtual Environment", setup.setup_virtual_environment),
        ("Dependencies", setup.install_dependencies),
    ]
    
    if not args.minimal:
        steps.extend([
            ("Pre-commit Hooks", setup.setup_pre_commit),
            ("IDE Configuration", setup.setup_ide_configuration),
        ])
    
    steps.append(("Validation", setup.validate_setup))
    
    # Execute setup steps
    for step_name, step_func in steps:
        print(f"\n{'=' * 20} {step_name} {'=' * 20}")
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            return 1
    
    # Print success message and next steps
    setup.print_next_steps()
    return 0


if __name__ == "__main__":
    sys.exit(main())
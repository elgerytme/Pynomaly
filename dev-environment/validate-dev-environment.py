#!/usr/bin/env python3
"""
Development Environment Validator
Validates that the development environment is properly configured.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util


class DevEnvironmentValidator:
    """Validates development environment setup."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        
    def run_command(self, cmd: List[str]) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            return result.returncode == 0, result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, ""
    
    def check_python_environment(self) -> None:
        """Check Python installation and version."""
        print("🐍 Checking Python environment...")
        
        # Check Python version
        success, output = self.run_command(["python", "--version"])
        if not success:
            self.errors.append("Python not found in PATH")
            return
        
        # Parse version
        try:
            version_str = output.split()[1]
            major, minor = map(int, version_str.split('.')[:2])
            
            if major != 3 or minor < 11:
                self.errors.append(f"Python {version_str} found, but 3.11+ required")
            else:
                self.info.append(f"✅ Python {version_str}")
        except (IndexError, ValueError):
            self.errors.append(f"Could not parse Python version: {output}")
        
        # Check pip
        success, output = self.run_command(["pip", "--version"])
        if not success:
            self.errors.append("pip not found")
        else:
            self.info.append("✅ pip available")
    
    def check_package_managers(self) -> None:
        """Check package managers."""
        print("📦 Checking package managers...")
        
        # Check Poetry
        success, output = self.run_command(["poetry", "--version"])
        if success:
            self.info.append(f"✅ Poetry: {output}")
        else:
            self.warnings.append("Poetry not installed (optional but recommended)")
        
        # Check pyenv
        success, output = self.run_command(["pyenv", "--version"])
        if success:
            self.info.append(f"✅ pyenv: {output}")
        else:
            self.warnings.append("pyenv not installed (optional but recommended)")
    
    def check_git_setup(self) -> None:
        """Check Git configuration."""
        print("📝 Checking Git setup...")
        
        # Check Git
        success, output = self.run_command(["git", "--version"])
        if not success:
            self.errors.append("Git not found")
            return
        
        self.info.append(f"✅ Git: {output}")
        
        # Check Git config
        success, name = self.run_command(["git", "config", "user.name"])
        success2, email = self.run_command(["git", "config", "user.email"])
        
        if not success or not success2:
            self.warnings.append("Git user.name or user.email not configured")
        else:
            self.info.append(f"✅ Git user: {name} <{email}>")
    
    def check_docker_setup(self) -> None:
        """Check Docker installation."""
        print("🐳 Checking Docker setup...")
        
        # Check Docker
        success, output = self.run_command(["docker", "--version"])
        if not success:
            self.warnings.append("Docker not installed")
            return
        
        self.info.append(f"✅ Docker: {output}")
        
        # Check Docker Compose
        success, output = self.run_command(["docker", "compose", "version"])
        if success:
            self.info.append(f"✅ Docker Compose: {output}")
        else:
            # Try legacy docker-compose
            success, output = self.run_command(["docker-compose", "--version"])
            if success:
                self.info.append(f"✅ Docker Compose (legacy): {output}")
            else:
                self.warnings.append("Docker Compose not found")
        
        # Test Docker daemon
        success, _ = self.run_command(["docker", "info"])
        if not success:
            self.warnings.append("Docker daemon not running or not accessible")
    
    def check_development_tools(self) -> None:
        """Check development tools."""
        print("🔧 Checking development tools...")
        
        tools = {
            "black": "Python code formatter",
            "ruff": "Python linter",
            "mypy": "Python type checker",
            "pytest": "Python testing framework",
            "pre-commit": "Pre-commit hooks",
            "bandit": "Python security scanner",
            "safety": "Python security scanner"
        }
        
        for tool, description in tools.items():
            success, output = self.run_command([tool, "--version"])
            if success:
                self.info.append(f"✅ {tool}: {description}")
            else:
                if tool in ["pre-commit", "bandit", "safety"]:
                    self.errors.append(f"{tool} not installed ({description})")
                else:
                    self.warnings.append(f"{tool} not installed ({description})")
    
    def check_build_tools(self) -> None:
        """Check build tools."""
        print("🏗️ Checking build tools...")
        
        # Check Buck2
        success, output = self.run_command(["buck2", "--version"])
        if success:
            self.info.append(f"✅ Buck2: {output}")
        else:
            self.warnings.append("Buck2 not installed (build system)")
        
        # Check make
        success, output = self.run_command(["make", "--version"])
        if success:
            self.info.append("✅ make available")
        
        # Check Node.js (for some tools)
        success, output = self.run_command(["node", "--version"])
        if success:
            self.info.append(f"✅ Node.js: {output}")
        else:
            self.warnings.append("Node.js not installed (needed for some tools)")
    
    def check_security_tools(self) -> None:
        """Check security tools."""
        print("🔒 Checking security tools...")
        
        security_tools = {
            "hadolint": "Dockerfile linter",
            "trivy": "Container security scanner",
            "semgrep": "Static analysis tool",
            "detect-secrets": "Secret detection"
        }
        
        for tool, description in security_tools.items():
            success, _ = self.run_command([tool, "--version"])
            if success:
                self.info.append(f"✅ {tool}: {description}")
            else:
                self.warnings.append(f"{tool} not installed ({description})")
    
    def check_ide_configuration(self) -> None:
        """Check IDE configurations."""
        print("💻 Checking IDE configuration...")
        
        # Check VS Code configuration
        vscode_dir = self.repo_root / ".vscode"
        if vscode_dir.exists():
            config_files = [
                "settings.json",
                "tasks.json", 
                "launch.json",
                "extensions.json"
            ]
            
            missing_configs = []
            for config_file in config_files:
                if not (vscode_dir / config_file).exists():
                    missing_configs.append(config_file)
            
            if missing_configs:
                self.warnings.append(f"VS Code configs missing: {', '.join(missing_configs)}")
            else:
                self.info.append("✅ VS Code configuration complete")
        else:
            self.warnings.append("VS Code configuration directory not found")
        
        # Check EditorConfig
        if (self.repo_root / ".editorconfig").exists():
            self.info.append("✅ EditorConfig present")
        else:
            self.warnings.append("EditorConfig not found")
    
    def check_pre_commit_hooks(self) -> None:
        """Check pre-commit hooks setup."""
        print("🪝 Checking pre-commit hooks...")
        
        # Check pre-commit config
        if not (self.repo_root / ".pre-commit-config.yaml").exists():
            self.errors.append("Pre-commit configuration not found")
            return
        
        # Check if hooks are installed
        git_hooks_dir = self.repo_root / ".git" / "hooks"
        if (git_hooks_dir / "pre-commit").exists():
            self.info.append("✅ Pre-commit hooks installed")
        else:
            self.warnings.append("Pre-commit hooks not installed (run: pre-commit install)")
        
        # Test pre-commit
        success, output = self.run_command([
            "pre-commit", "run", "--all-files", "--show-diff-on-failure"
        ])
        if success:
            self.info.append("✅ Pre-commit hooks pass")
        else:
            self.warnings.append("Pre-commit hooks have issues (run: pre-commit run --all-files)")
    
    def check_project_structure(self) -> None:
        """Check project structure."""
        print("📁 Checking project structure...")
        
        required_dirs = [
            "src",
            "tests", 
            "docs",
            ".github",
            "scripts"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            if not (self.repo_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            self.warnings.append(f"Project directories missing: {', '.join(missing_dirs)}")
        else:
            self.info.append("✅ Project structure looks good")
        
        # Check key files
        key_files = [
            "README.md",
            "requirements-prod.txt",
            ".gitignore",
            "pytest.ini"
        ]
        
        missing_files = []
        for file_name in key_files:
            if not (self.repo_root / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            self.warnings.append(f"Key files missing: {', '.join(missing_files)}")
    
    def check_python_packages(self) -> None:
        """Check important Python packages are available."""
        print("📚 Checking Python packages...")
        
        important_packages = [
            "pytest",
            "black", 
            "ruff",
            "mypy",
            "bandit",
            "safety",
            "pre-commit"
        ]
        
        missing_packages = []
        for package in important_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.errors.append(f"Required Python packages missing: {', '.join(missing_packages)}")
        else:
            self.info.append("✅ All required Python packages available")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        total_checks = len(self.info) + len(self.warnings) + len(self.errors)
        passed_checks = len(self.info)
        
        score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        return {
            "score": round(score, 1),
            "total_checks": total_checks,
            "passed": len(self.info),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "status": "excellent" if score >= 90 else "good" if score >= 70 else "needs_improvement",
            "details": {
                "info": self.info,
                "warnings": self.warnings,
                "errors": self.errors
            }
        }
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating development environment...\n")
        
        # Run all checks
        self.check_python_environment()
        self.check_package_managers()
        self.check_git_setup()
        self.check_docker_setup()
        self.check_development_tools()
        self.check_build_tools()
        self.check_security_tools()
        self.check_ide_configuration()
        self.check_pre_commit_hooks()
        self.check_project_structure()
        self.check_python_packages()
        
        # Generate and display report
        report = self.generate_report()
        
        print(f"\n📊 Validation Report")
        print(f"{'='*50}")
        print(f"Overall Score: {report['score']}%")
        print(f"Status: {report['status'].replace('_', ' ').title()}")
        print(f"Checks Passed: {report['passed']}/{report['total_checks']}")
        print(f"Warnings: {report['warnings']}")
        print(f"Errors: {report['errors']}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.info:
            print(f"\n✅ PASSED CHECKS ({len(self.info)}):")
            for info in self.info:
                print(f"   • {info}")
        
        # Recommendations
        if report['status'] == 'needs_improvement':
            print("\n💡 RECOMMENDATIONS:")
            print("   • Run the setup script: ./dev-environment/setup-dev-environment.sh")
            print("   • Install missing tools manually")
            print("   • Check the documentation for setup instructions")
        
        return len(self.errors) == 0


def main():
    """Main function."""
    validator = DevEnvironmentValidator()
    success = validator.run_validation()
    
    if success:
        print(f"\n🎉 Development environment validation passed!")
        print("Your development environment is ready for development.")
    else:
        print(f"\n🔧 Development environment needs attention.")
        print("Please address the errors above before proceeding.")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
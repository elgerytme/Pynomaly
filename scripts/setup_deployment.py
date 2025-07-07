#!/usr/bin/env python3
"""
Setup script for Pynomaly deployment infrastructure.

This script sets up the complete deployment infrastructure:
1. GitHub repository secrets and environments
2. PyPI/TestPyPI API tokens configuration
3. Pre-commit hooks installation
4. CI/CD pipeline validation
5. Release workflow testing

Usage:
    python scripts/setup_deployment.py --check-all
    python scripts/setup_deployment.py --setup-precommit
    python scripts/setup_deployment.py --validate-ci
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class DeploymentSetup:
    """Manages deployment infrastructure setup for Pynomaly."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.github_dir = root_dir / ".github"
        self.workflows_dir = self.github_dir / "workflows"
        
    def run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        print(f"🔧 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root_dir)
        
        if check and result.returncode != 0:
            print(f"❌ Command failed: {' '.join(cmd)}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return result
        
        return result
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required tools are installed."""
        print("🔍 Checking dependencies...")
        
        tools = {
            "git": ["git", "--version"],
            "python": [sys.executable, "--version"],
            "hatch": ["hatch", "--version"],
            "pre-commit": ["pre-commit", "--version"],
            "gh": ["gh", "--version"],  # GitHub CLI (optional)
        }
        
        results = {}
        for tool, cmd in tools.items():
            result = self.run_command(cmd, check=False)
            available = result.returncode == 0
            results[tool] = available
            
            if available:
                version = result.stdout.strip().split('\n')[0]
                print(f"   ✅ {tool}: {version}")
            else:
                print(f"   ❌ {tool}: Not found")
        
        return results
    
    def setup_precommit_hooks(self) -> None:
        """Install and configure pre-commit hooks."""
        print("🪝 Setting up pre-commit hooks...")
        
        # Check if pre-commit is installed
        result = self.run_command(["pre-commit", "--version"], check=False)
        if result.returncode != 0:
            print("📦 Installing pre-commit...")
            self.run_command(["pip", "install", "pre-commit"])
        
        # Install hooks
        print("🔧 Installing pre-commit hooks...")
        result = self.run_command(["pre-commit", "install"], check=False)
        if result.returncode == 0:
            print("✅ Pre-commit hooks installed successfully")
        else:
            print("⚠️  Pre-commit installation had issues")
        
        # Install commit-msg hook
        self.run_command(["pre-commit", "install", "--hook-type", "commit-msg"], check=False)
        self.run_command(["pre-commit", "install", "--hook-type", "pre-push"], check=False)
        
        # Test hooks
        print("🧪 Testing pre-commit hooks...")
        result = self.run_command(["pre-commit", "run", "--all-files"], check=False)
        if result.returncode == 0:
            print("✅ All pre-commit hooks passed")
        else:
            print("⚠️  Some pre-commit hooks failed (this is normal for initial setup)")
            print("   Run 'pre-commit run --all-files' to see details")
    
    def validate_github_workflows(self) -> None:
        """Validate GitHub Actions workflows."""
        print("🔍 Validating GitHub workflows...")
        
        required_workflows = [
            "ci.yml",
            "cd.yml", 
            "release-pypi.yml"
        ]
        
        for workflow in required_workflows:
            workflow_path = self.workflows_dir / workflow
            if workflow_path.exists():
                print(f"   ✅ {workflow}: Found")
                
                # Basic YAML validation
                try:
                    import yaml
                    with open(workflow_path) as f:
                        yaml.safe_load(f)
                    print(f"   ✅ {workflow}: Valid YAML")
                except ImportError:
                    print(f"   ⚠️  {workflow}: Could not validate YAML (PyYAML not installed)")
                except yaml.YAMLError as e:
                    print(f"   ❌ {workflow}: Invalid YAML - {e}")
            else:
                print(f"   ❌ {workflow}: Missing")
    
    def check_package_configuration(self) -> None:
        """Check package configuration for PyPI readiness."""
        print("📦 Checking package configuration...")
        
        # Check pyproject.toml
        pyproject_path = self.root_dir / "pyproject.toml"
        if not pyproject_path.exists():
            print("   ❌ pyproject.toml: Missing")
            return
        
        print("   ✅ pyproject.toml: Found")
        
        # Check Hatch configuration
        result = self.run_command(["hatch", "project", "metadata", "name"], check=False)
        if result.returncode == 0:
            name = result.stdout.strip()
            print(f"   ✅ Package name: {name}")
        else:
            print("   ❌ Package name: Could not retrieve")
        
        # Check version
        result = self.run_command(["hatch", "version"], check=False)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   ✅ Package version: {version}")
        else:
            print("   ❌ Package version: Could not retrieve")
        
        # Check required files
        required_files = ["README.md", "LICENSE", "CHANGELOG.md"]
        for file in required_files:
            file_path = self.root_dir / file
            if file_path.exists():
                print(f"   ✅ {file}: Found")
            else:
                print(f"   ❌ {file}: Missing")
    
    def test_package_build(self) -> None:
        """Test package building."""
        print("🏗️  Testing package build...")
        
        # Clean previous builds
        result = self.run_command(["hatch", "clean"], check=False)
        
        # Build package
        result = self.run_command(["hatch", "build", "--clean"], check=False)
        if result.returncode == 0:
            print("   ✅ Package builds successfully")
            
            # Check artifacts
            dist_dir = self.root_dir / "dist"
            if dist_dir.exists():
                artifacts = list(dist_dir.glob("*"))
                print(f"   📄 Build artifacts: {len(artifacts)} files")
                for artifact in artifacts[:3]:  # Show first 3
                    print(f"      - {artifact.name}")
            else:
                print("   ⚠️  No dist/ directory found")
        else:
            print("   ❌ Package build failed")
            print(f"      Error: {result.stderr}")
    
    def check_github_setup(self) -> None:
        """Check GitHub repository setup."""
        print("🐙 Checking GitHub setup...")
        
        # Check if we're in a git repository
        result = self.run_command(["git", "status"], check=False)
        if result.returncode != 0:
            print("   ❌ Not a git repository")
            return
        
        print("   ✅ Git repository")
        
        # Check remote origin
        result = self.run_command(["git", "remote", "get-url", "origin"], check=False)
        if result.returncode == 0:
            origin = result.stdout.strip()
            print(f"   ✅ Remote origin: {origin}")
        else:
            print("   ❌ No remote origin configured")
        
        # Check if GitHub CLI is available
        result = self.run_command(["gh", "auth", "status"], check=False)
        if result.returncode == 0:
            print("   ✅ GitHub CLI authenticated")
        else:
            print("   ⚠️  GitHub CLI not authenticated")
            print("      Run 'gh auth login' to authenticate")
    
    def show_secrets_setup_guide(self) -> None:
        """Show guide for setting up GitHub secrets."""
        print("\n🔐 GitHub Secrets Setup Guide")
        print("=" * 50)
        
        secrets_guide = """
To complete the deployment setup, configure these GitHub repository secrets:

1. PyPI API Tokens:
   Repository Settings → Secrets and variables → Actions → Repository secrets
   
   Required secrets:
   - PYPI_API_TOKEN: Your PyPI API token
     • Generate at: https://pypi.org/manage/account/token/
     • Scope: "Entire account" or specific to pynomaly project
   
   - TEST_PYPI_API_TOKEN: Your TestPyPI API token
     • Generate at: https://test.pypi.org/manage/account/token/
     • Scope: "Entire account" or specific to pynomaly project

2. Optional secrets (for enhanced features):
   - SLACK_WEBHOOK: For deployment notifications
   - CODECOV_TOKEN: For coverage reporting

3. GitHub Environments (for deployment protection):
   Repository Settings → Environments
   
   Create environments:
   - pypi: Production PyPI deployment
     • Add protection rules (require reviews, etc.)
     • Add PYPI_API_TOKEN as environment secret
   
   - testpypi: TestPyPI deployment
     • Add TEST_PYPI_API_TOKEN as environment secret

4. Repository Settings:
   - Actions → General → Workflow permissions:
     ✓ Read and write permissions
     ✓ Allow GitHub Actions to create and approve pull requests

Commands to run after secrets setup:
  # Test the release workflow (dry run)
  python scripts/release.py --version 0.1.0-dev --environment testpypi --dry-run
  
  # Actual TestPyPI release
  python scripts/release.py --auto-bump patch --environment testpypi
  
  # Production PyPI release
  python scripts/release.py --version 0.1.0 --environment pypi
"""
        print(secrets_guide)
    
    def run_comprehensive_check(self) -> None:
        """Run comprehensive deployment readiness check."""
        print("🚀 Comprehensive Deployment Readiness Check")
        print("=" * 50)
        
        # Check dependencies
        deps = self.check_dependencies()
        
        # Check package configuration
        self.check_package_configuration()
        
        # Test package build
        self.test_package_build()
        
        # Validate workflows
        self.validate_github_workflows()
        
        # Check GitHub setup
        self.check_github_setup()
        
        # Summary
        print("\n📊 Summary")
        print("-" * 20)
        
        essential_tools = ["git", "python", "hatch"]
        missing_essential = [tool for tool in essential_tools if not deps.get(tool, False)]
        
        if missing_essential:
            print(f"❌ Missing essential tools: {', '.join(missing_essential)}")
            print("   Install these tools before proceeding")
        else:
            print("✅ All essential tools available")
        
        if not deps.get("pre-commit", False):
            print("⚠️  Pre-commit not installed (run --setup-precommit)")
        
        if not deps.get("gh", False):
            print("⚠️  GitHub CLI not available (optional but recommended)")
        
        print("\n📚 Next Steps:")
        print("1. Run: python scripts/setup_deployment.py --setup-precommit")
        print("2. Configure GitHub secrets (see --secrets-guide)")
        print("3. Test release workflow with TestPyPI")
        print("4. Deploy to production PyPI")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pynomaly deployment setup")
    
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Run comprehensive deployment readiness check"
    )
    
    parser.add_argument(
        "--setup-precommit",
        action="store_true",
        help="Install and configure pre-commit hooks"
    )
    
    parser.add_argument(
        "--validate-ci",
        action="store_true",
        help="Validate CI/CD workflows"
    )
    
    parser.add_argument(
        "--test-build",
        action="store_true",
        help="Test package building"
    )
    
    parser.add_argument(
        "--secrets-guide",
        action="store_true",
        help="Show GitHub secrets setup guide"
    )
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    if not (root_dir / "pyproject.toml").exists():
        print("❌ pyproject.toml not found. Run from project root.")
        sys.exit(1)
    
    setup = DeploymentSetup(root_dir)
    
    if args.check_all:
        setup.run_comprehensive_check()
    elif args.setup_precommit:
        setup.setup_precommit_hooks()
    elif args.validate_ci:
        setup.validate_github_workflows()
    elif args.test_build:
        setup.test_package_build()
    elif args.secrets_guide:
        setup.show_secrets_setup_guide()
    else:
        print("Please specify an action. Use --help for options.")
        parser.print_help()


if __name__ == "__main__":
    main()
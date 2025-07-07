#!/usr/bin/env python3
"""
Automated release script for Pynomaly PyPI deployment.

This script automates the complete release process:
1. Version validation and bump
2. Changelog updates  
3. Quality checks and tests
4. Package building
5. PyPI deployment coordination
6. Git tagging and release creation

Usage:
    python scripts/release.py --version 0.1.0 --environment testpypi
    python scripts/release.py --version 0.1.0 --environment pypi
    python scripts/release.py --auto-bump patch --environment testpypi
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


class ReleaseManager:
    """Manages the complete release process for Pynomaly."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.pyproject_path = root_dir / "pyproject.toml"
        self.changelog_path = root_dir / "CHANGELOG.md"
        self.readme_path = root_dir / "README.md"
        
    def run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        print(f"🔧 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root_dir)
        
        if check and result.returncode != 0:
            print(f"❌ Command failed: {' '.join(cmd)}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            sys.exit(1)
        
        return result
    
    def get_current_version(self) -> str:
        """Get current version from Hatch."""
        result = self.run_command(["hatch", "version"])
        return result.stdout.strip()
    
    def validate_version_format(self, version: str) -> bool:
        """Validate semantic version format."""
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:[-.]?(alpha|beta|rc)\.?(\d+)?)?$"
        return bool(re.match(pattern, version))
    
    def bump_version(self, bump_type: str) -> str:
        """Bump version automatically."""
        current = self.get_current_version()
        print(f"📌 Current version: {current}")
        
        # Parse current version
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", current)
        if not match:
            raise ValueError(f"Invalid current version format: {current}")
        
        major, minor, patch = map(int, match.groups())
        
        if bump_type == "major":
            new_version = f"{major + 1}.0.0"
        elif bump_type == "minor":
            new_version = f"{major}.{minor + 1}.0"
        elif bump_type == "patch":
            new_version = f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
        
        print(f"📈 Bumping version: {current} → {new_version}")
        return new_version
    
    def check_pypi_availability(self, version: str, environment: str = "pypi") -> bool:
        """Check if version is available on PyPI."""
        base_url = "https://pypi.org" if environment == "pypi" else "https://test.pypi.org"
        url = f"{base_url}/pypi/pynomaly/{version}/json"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"⚠️  Version {version} already exists on {environment}")
                return False
            return True
        except requests.RequestException:
            print(f"🔍 Cannot check {environment} availability (assuming available)")
            return True
    
    def update_changelog(self, version: str) -> None:
        """Update CHANGELOG.md with new version."""
        if not self.changelog_path.exists():
            print(f"📝 Creating CHANGELOG.md")
            changelog_content = f"""# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [{version}] - {datetime.now().strftime('%Y-%m-%d')}

### Added
- Initial release of Pynomaly
- Comprehensive anomaly detection capabilities
- Clean architecture implementation
- PyPI package deployment

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A
"""
        else:
            # Read existing changelog
            with open(self.changelog_path, 'r') as f:
                content = f.read()
            
            # Check if version already exists
            if f"## [{version}]" in content:
                print(f"✅ Version {version} already in changelog")
                return
            
            # Add new version entry
            today = datetime.now().strftime('%Y-%m-%d')
            new_entry = f"\n## [{version}] - {today}\n\n### Added\n- Release {version}\n\n### Changed\n- Performance improvements and bug fixes\n\n"
            
            # Insert after [Unreleased] section
            if "## [Unreleased]" in content:
                content = content.replace("## [Unreleased]", f"## [Unreleased]{new_entry}")
            else:
                # Insert at the beginning of the changelog
                lines = content.split('\n')
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.startswith('## ['):
                        insert_index = i
                        break
                
                lines.insert(insert_index, new_entry.strip())
                content = '\n'.join(lines)
            
            changelog_content = content
        
        with open(self.changelog_path, 'w') as f:
            f.write(changelog_content)
        
        print(f"📝 Updated CHANGELOG.md for version {version}")
    
    def run_quality_checks(self) -> None:
        """Run comprehensive quality checks."""
        print("🔍 Running quality checks...")
        
        # Code style and linting
        print("  📝 Code style checks...")
        self.run_command(["hatch", "env", "run", "lint:style"])
        
        # Type checking
        print("  🔎 Type checking...")
        self.run_command(["hatch", "env", "run", "lint:typing"])
        
        # Core tests
        print("  🧪 Core tests...")
        self.run_command(["hatch", "env", "run", "test:run", "tests/domain/", "tests/application/", "-v"])
        
        # Integration tests (critical ones)
        print("  🔗 Integration tests...")
        self.run_command([
            "hatch", "env", "run", "test:run", 
            "tests/infrastructure/test_optimization_service.py", 
            "-v"
        ])
        
        # Security scan
        print("  🔒 Security scan...")
        result = self.run_command(["bandit", "-r", "src/", "-f", "txt"], check=False)
        if result.returncode != 0:
            print("⚠️  Security scan found issues (non-blocking)")
        
        print("✅ Quality checks completed")
    
    def build_package(self) -> None:
        """Build the package with Hatch."""
        print("📦 Building package...")
        
        # Clean previous builds
        self.run_command(["hatch", "clean"])
        
        # Build package
        self.run_command(["hatch", "build", "--clean"])
        
        # Verify build artifacts
        dist_dir = self.root_dir / "dist"
        if not dist_dir.exists():
            raise RuntimeError("Build failed: dist/ directory not created")
        
        artifacts = list(dist_dir.glob("*"))
        if not artifacts:
            raise RuntimeError("Build failed: no artifacts in dist/")
        
        print(f"✅ Package built successfully:")
        for artifact in artifacts:
            print(f"   📄 {artifact.name}")
    
    def test_package_installation(self) -> None:
        """Test package installation."""
        print("🧪 Testing package installation...")
        
        # Create temporary virtual environment
        import tempfile
        import venv
        
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir = Path(temp_dir) / "test_env"
            venv.create(venv_dir, with_pip=True)
            
            # Get paths
            if sys.platform == "win32":
                pip_path = venv_dir / "Scripts" / "pip.exe"
                python_path = venv_dir / "Scripts" / "python.exe"
            else:
                pip_path = venv_dir / "bin" / "pip"
                python_path = venv_dir / "bin" / "python"
            
            # Install wheel
            wheel_files = list((self.root_dir / "dist").glob("*.whl"))
            if not wheel_files:
                raise RuntimeError("No wheel file found")
            
            wheel_file = wheel_files[0]
            self.run_command([str(pip_path), "install", str(wheel_file)])
            
            # Test basic import
            test_script = """
import pynomaly
from pynomaly.domain.entities import Dataset, Anomaly
from pynomaly.domain.value_objects import AnomalyScore
print(f"✅ Package test successful: {pynomaly.__version__}")
"""
            
            result = self.run_command([str(python_path), "-c", test_script])
            print("✅ Package installation test passed")
    
    def create_git_tag(self, version: str) -> None:
        """Create and push git tag."""
        tag_name = f"v{version}"
        
        print(f"🏷️  Creating git tag: {tag_name}")
        
        # Check if tag already exists
        result = self.run_command(["git", "tag", "-l", tag_name], check=False)
        if result.stdout.strip():
            print(f"⚠️  Tag {tag_name} already exists")
            return
        
        # Create tag
        self.run_command(["git", "add", "."])
        self.run_command(["git", "commit", "-m", f"chore: prepare release {version}"])
        self.run_command(["git", "tag", "-a", tag_name, "-m", f"Release {version}"])
        
        # Push tag
        self.run_command(["git", "push", "origin", tag_name])
        
        print(f"✅ Git tag {tag_name} created and pushed")
    
    def trigger_github_workflow(self, version: str, environment: str) -> None:
        """Trigger GitHub Actions workflow for PyPI deployment."""
        print(f"🚀 Triggering GitHub Actions workflow for {environment}...")
        
        # Use GitHub CLI if available
        gh_result = self.run_command(["which", "gh"], check=False)
        if gh_result.returncode == 0:
            self.run_command([
                "gh", "workflow", "run", "release-pypi.yml",
                "-f", f"version={version}",
                "-f", f"environment={environment}"
            ])
            print(f"✅ GitHub Actions workflow triggered")
        else:
            print("⚠️  GitHub CLI not found. Please manually trigger the release workflow:")
            print(f"   Go to: https://github.com/pynomaly/pynomaly/actions/workflows/release-pypi.yml")
            print(f"   Click 'Run workflow' and set:")
            print(f"   - Version: {version}")
            print(f"   - Environment: {environment}")
    
    def release(self, version: Optional[str], auto_bump: Optional[str], environment: str, skip_tests: bool = False) -> None:
        """Execute the complete release process."""
        print("🚀 Starting Pynomaly release process...")
        print(f"   Environment: {environment}")
        
        # Determine version
        if auto_bump:
            version = self.bump_version(auto_bump)
        elif version:
            if not self.validate_version_format(version):
                raise ValueError(f"Invalid version format: {version}")
        else:
            raise ValueError("Either --version or --auto-bump must be specified")
        
        print(f"📌 Target version: {version}")
        
        # Check PyPI availability
        if not self.check_pypi_availability(version, environment):
            response = input(f"Version {version} exists on {environment}. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Release cancelled")
                return
        
        try:
            # Update changelog
            self.update_changelog(version)
            
            # Run quality checks
            if not skip_tests:
                self.run_quality_checks()
            else:
                print("⚠️  Skipping quality checks (--skip-tests)")
            
            # Build package
            self.build_package()
            
            # Test package installation
            if not skip_tests:
                self.test_package_installation()
            
            # Create git tag
            self.create_git_tag(version)
            
            # Trigger deployment workflow
            self.trigger_github_workflow(version, environment)
            
            print(f"🎉 Release process completed successfully!")
            print(f"   Version: {version}")
            print(f"   Environment: {environment}")
            print(f"   Next steps:")
            print(f"   1. Monitor GitHub Actions workflow")
            print(f"   2. Verify package on {environment}")
            print(f"   3. Test installation: pip install pynomaly=={version}")
            
        except Exception as e:
            print(f"❌ Release failed: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated Pynomaly release script")
    
    version_group = parser.add_mutually_exclusive_group(required=True)
    version_group.add_argument(
        "--version", 
        type=str, 
        help="Specific version to release (e.g., 0.1.0)"
    )
    version_group.add_argument(
        "--auto-bump", 
        choices=["major", "minor", "patch"],
        help="Automatically bump version"
    )
    
    parser.add_argument(
        "--environment",
        choices=["testpypi", "pypi"],
        default="testpypi",
        help="Target environment (default: testpypi)"
    )
    
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip quality checks and tests (use with caution)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    if not (root_dir / "pyproject.toml").exists():
        print("❌ pyproject.toml not found. Run from project root.")
        sys.exit(1)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print(f"   Version: {args.version or f'auto-bump {args.auto_bump}'}")
        print(f"   Environment: {args.environment}")
        print(f"   Skip tests: {args.skip_tests}")
        return
    
    # Create release manager and execute
    release_manager = ReleaseManager(root_dir)
    release_manager.release(
        version=args.version,
        auto_bump=args.auto_bump,
        environment=args.environment,
        skip_tests=args.skip_tests
    )


if __name__ == "__main__":
    main()
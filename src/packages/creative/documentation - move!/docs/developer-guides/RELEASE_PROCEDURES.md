# Pynomaly Release Procedures

This document outlines the comprehensive release procedures for Pynomaly, ensuring consistent, reliable, and high-quality releases. Our release process follows semantic versioning and includes automated testing, quality gates, and deployment procedures.

## 📋 Table of Contents

- [Release Strategy](#release-strategy)
- [Semantic Versioning](#semantic-versioning)
- [Release Types](#release-types)
- [Pre-Release Checklist](#pre-release-checklist)
- [Release Process](#release-process)
- [Hotfix Process](#hotfix-process)
- [Post-Release Activities](#post-release-activities)
- [Rollback Procedures](#rollback-procedures)
- [Automation Scripts](#automation-scripts)

## 🎯 Release Strategy

### Release Philosophy
- **Quality First**: No release without passing all quality gates
- **Predictable Cadence**: Regular, scheduled releases
- **Semantic Versioning**: Clear versioning that communicates change impact
- **Automated Testing**: Comprehensive test coverage before release
- **Staged Deployment**: Gradual rollout with monitoring

### Release Schedule
- **Major Releases** (X.0.0): Every 6-12 months
- **Minor Releases** (X.Y.0): Every 4-6 weeks
- **Patch Releases** (X.Y.Z): As needed for critical fixes
- **Pre-releases** (X.Y.Z-alpha/beta/rc): Weekly during development

## 📝 Semantic Versioning

We follow [Semantic Versioning 2.0.0](https://semver.org/):

### Version Format: `MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]`

#### MAJOR (X.0.0)
- Breaking changes to public API
- Removal of deprecated features
- Major architectural changes
- Database schema changes requiring migration

**Examples:**
- Removal of deprecated `detect_anomalies()` function
- Change in core data structures
- New authentication requirements

#### MINOR (0.X.0)
- New features that are backward compatible
- New API endpoints
- New algorithm support
- Performance improvements

**Examples:**
- New algorithm adapter (PyGOD integration)
- New API endpoints for model management
- Enhanced web interface features

#### PATCH (0.0.X)
- Bug fixes that are backward compatible
- Security patches
- Documentation updates
- Performance optimizations without API changes

**Examples:**
- Fix memory leak in data processing
- Correct algorithm parameter validation
- Update dependencies for security

#### Pre-release Identifiers
- **alpha**: Early development, unstable
- **beta**: Feature complete, testing phase
- **rc** (release candidate): Final testing before release

## 🚀 Release Types

### 1. Development Release (Alpha)
```bash
# Example: 1.2.0-alpha.1
git tag v1.2.0-alpha.1
```

**Characteristics:**
- Early development features
- May have known issues
- API may change
- Internal testing only

### 2. Beta Release
```bash
# Example: 1.2.0-beta.1
git tag v1.2.0-beta.1
```

**Characteristics:**
- Feature complete
- API stable
- Limited external testing
- Documentation updated

### 3. Release Candidate (RC)
```bash
# Example: 1.2.0-rc.1
git tag v1.2.0-rc.1
```

**Characteristics:**
- Production ready
- Final testing
- Full documentation
- Ready for wider testing

### 4. Stable Release
```bash
# Example: 1.2.0
git tag v1.2.0
```

**Characteristics:**
- Production ready
- All tests passing
- Complete documentation
- Security reviewed

## ✅ Pre-Release Checklist

### Code Quality
- [ ] All tests passing (unit, integration, E2E)
- [ ] Code coverage ≥85% overall, ≥95% for domain layer
- [ ] No critical security vulnerabilities
- [ ] Code review completed for all changes
- [ ] Performance regression tests passed

### Documentation
- [ ] CHANGELOG.md updated with all changes
- [ ] API documentation updated
- [ ] User guide reflects new features
- [ ] Migration guide created (for breaking changes)
- [ ] Release notes drafted

### Version Management
- [ ] Version number updated in `pyproject.toml`
- [ ] Version number updated in `__init__.py`
- [ ] Version compatibility matrix updated
- [ ] Dependency versions finalized

### Testing
- [ ] Full test suite execution completed
- [ ] Performance benchmarks validated
- [ ] Security scan completed
- [ ] Manual testing of critical paths
- [ ] Compatibility testing across Python versions

### Dependencies
- [ ] Dependencies updated to latest stable versions
- [ ] Security advisories reviewed
- [ ] License compatibility verified
- [ ] Dependency lock file updated

## 🔄 Release Process

### Phase 1: Preparation

#### 1. Create Release Branch
```bash
# Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.2.0

# Update version in pyproject.toml
vim pyproject.toml  # Update version = "1.2.0"

# Update version in package
vim src/pynomaly/__init__.py  # Update __version__ = "1.2.0"
```

#### 2. Update Documentation
```bash
# Update CHANGELOG.md
vim CHANGELOG.md

# Generate API documentation
python scripts/docs/generate_api_docs.py

# Update version in documentation
find docs/ -name "*.md" -exec sed -i 's/v1\.1\.0/v1.2.0/g' {} +
```

#### 3. Run Quality Gates
```bash
# Run complete test suite
python scripts/quality_gates.py

# Security scan
bandit -r src/pynomaly/ -f json -o security-report.json

# Performance benchmarks
pytest tests/performance/ --benchmark-only --benchmark-json=benchmark-results.json
```

### Phase 2: Testing

#### 4. Automated Testing
```bash
# Run full test suite with coverage
pytest tests/ --cov=src/pynomaly --cov-report=html --cov-report=term

# Run integration tests
pytest tests/integration/ -v --tb=short

# Run end-to-end tests
pytest tests/e2e/ -v --tb=short --timeout=300
```

#### 5. Build and Package Testing
```bash
# Build package
python -m build

# Test installation in clean environment
python -m venv test-env
source test-env/bin/activate
pip install dist/pynomaly-1.2.0-py3-none-any.whl

# Verify installation
python -c "import pynomaly; print(pynomaly.__version__)"
```

#### 6. Manual Testing
```bash
# Test CLI commands
pynomaly --version
pynomaly detector list
pynomaly dataset create --help

# Test API endpoints
curl -X GET http://localhost:8000/api/v1/health
curl -X GET http://localhost:8000/api/v1/detectors

# Test web interface
# Navigate to http://localhost:8000/app and test key workflows
```

### Phase 3: Release

#### 7. Finalize Release Branch
```bash
# Commit version updates
git add .
git commit -m "chore: bump version to 1.2.0"

# Create pull request to main
gh pr create --title "Release v1.2.0" --body "$(cat CHANGELOG.md | head -20)"
```

#### 8. Merge and Tag
```bash
# After PR approval, merge to main
gh pr merge --merge

# Switch to main and tag
git checkout main
git pull origin main
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0
```

#### 9. Deploy to PyPI
```bash
# Build distribution
python -m build

# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ pynomaly==1.2.0

# Upload to production PyPI
python -m twine upload dist/*
```

### Phase 4: Publication

#### 10. Create GitHub Release
```bash
# Create GitHub release with notes
gh release create v1.2.0 \
  --title "Pynomaly v1.2.0" \
  --notes-file RELEASE_NOTES.md \
  --draft=false \
  --prerelease=false \
  dist/*
```

#### 11. Update Documentation Sites
```bash
# Deploy documentation
mkdocs gh-deploy --config-file mkdocs.yml

# Update Docker images
docker build -t pynomaly:1.2.0 -t pynomaly:latest .
docker push pynomaly:1.2.0
docker push pynomaly:latest
```

## 🚨 Hotfix Process

For critical issues requiring immediate release:

### 1. Create Hotfix Branch
```bash
# Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/v1.2.1

# Make minimal fix
# Edit necessary files

# Update version (patch increment)
vim pyproject.toml  # version = "1.2.1"
vim src/pynomaly/__init__.py  # __version__ = "1.2.1"
```

### 2. Test and Release
```bash
# Run critical tests only
pytest tests/unit/critical/ tests/integration/critical/

# Quick quality check
ruff check src/ tests/
mypy src/pynomaly/

# Commit and tag
git add .
git commit -m "fix: critical issue description (fixes #123)"
git tag -a v1.2.1 -m "Hotfix version 1.2.1"

# Push and deploy
git push origin hotfix/v1.2.1
git push origin v1.2.1

# Create release
gh release create v1.2.1 --title "Pynomaly v1.2.1 (Hotfix)" --notes "Critical fix for..."
```

### 3. Merge Back
```bash
# Merge back to main and develop
git checkout main
git merge hotfix/v1.2.1
git push origin main

git checkout develop
git merge hotfix/v1.2.1
git push origin develop

# Delete hotfix branch
git branch -d hotfix/v1.2.1
git push origin --delete hotfix/v1.2.1
```

## 📊 Post-Release Activities

### 1. Monitor Release
```bash
# Monitor PyPI downloads
python scripts/monitoring/check_pypi_stats.py

# Monitor error rates
python scripts/monitoring/check_error_rates.py

# Check GitHub issues for bug reports
gh issue list --label "bug" --state "open"
```

### 2. Update Documentation
```bash
# Update installation instructions
# Update compatibility matrix
# Update migration guides (if needed)
```

### 3. Communication
```bash
# Send release announcement
# Update social media
# Notify users of breaking changes
# Update community forums
```

### 4. Prepare for Next Release
```bash
# Create next milestone
gh milestone create "v1.3.0" --due-date "2024-03-01"

# Update project board
# Plan next features
# Update roadmap
```

## 🔄 Rollback Procedures

### Emergency Rollback

#### 1. Remove from PyPI (if possible)
```bash
# Contact PyPI support for package removal
# Note: Usually only possible within first few hours
```

#### 2. Revert Git Tag
```bash
# Delete problematic tag
git tag -d v1.2.0
git push origin --delete v1.2.0

# Create new tag on previous stable commit
git tag -a v1.1.1 -m "Rollback to stable version"
git push origin v1.1.1
```

#### 3. Create Rollback Release
```bash
# Create immediate rollback release
gh release create v1.2.1 \
  --title "Pynomaly v1.2.1 (Rollback)" \
  --notes "Rollback from v1.2.0 due to critical issue. Use v1.1.x instead."
```

#### 4. Communication
```bash
# Immediate communication
echo "URGENT: Issue with v1.2.0. Please use v1.1.x. Working on fix." | \
  gh issue create --title "URGENT: v1.2.0 Issue" --body-file -
```

## 🤖 Automation Scripts

### Release Automation Script
```python
#!/usr/bin/env python3
"""
Automated release script for Pynomaly.

Usage:
    python scripts/release/release.py --version 1.2.0 --type minor
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path
from typing import List


class ReleaseManager:
    """Manages the release process."""
    
    def __init__(self, version: str, release_type: str):
        self.version = version
        self.release_type = release_type
        self.project_root = Path(__file__).parent.parent.parent
        
    def validate_version(self) -> bool:
        """Validate version format."""
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z]+\.\d+)?$'
        return bool(re.match(pattern, self.version))
    
    def run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run command and handle errors."""
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.project_root, check=check)
        return result
    
    def update_version_files(self) -> None:
        """Update version in project files."""
        # Update pyproject.toml
        pyproject_file = self.project_root / "pyproject.toml"
        content = pyproject_file.read_text()
        content = re.sub(
            r'version = "[^"]*"',
            f'version = "{self.version}"',
            content
        )
        pyproject_file.write_text(content)
        
        # Update __init__.py
        init_file = self.project_root / "src" / "pynomaly" / "__init__.py"
        content = init_file.read_text()
        content = re.sub(
            r'__version__ = "[^"]*"',
            f'__version__ = "{self.version}"',
            content
        )
        init_file.write_text(content)
        
        print(f"Updated version to {self.version}")
    
    def run_tests(self) -> None:
        """Run comprehensive test suite."""
        print("Running test suite...")
        
        # Run quality gates
        self.run_command(["python", "scripts/quality_gates.py"])
        
        # Run full test suite
        self.run_command([
            "pytest", "tests/", 
            "--cov=src/pynomaly", 
            "--cov-fail-under=85"
        ])
        
        print("All tests passed!")
    
    def build_package(self) -> None:
        """Build the package."""
        print("Building package...")
        
        # Clean previous builds
        build_dir = self.project_root / "build"
        dist_dir = self.project_root / "dist"
        
        if build_dir.exists():
            self.run_command(["rm", "-rf", str(build_dir)])
        if dist_dir.exists():
            self.run_command(["rm", "-rf", str(dist_dir)])
        
        # Build package
        self.run_command(["python", "-m", "build"])
        
        print("Package built successfully!")
    
    def create_git_tag(self) -> None:
        """Create and push git tag."""
        tag_name = f"v{self.version}"
        
        # Create tag
        self.run_command([
            "git", "tag", "-a", tag_name,
            "-m", f"Release version {self.version}"
        ])
        
        # Push tag
        self.run_command(["git", "push", "origin", tag_name])
        
        print(f"Created and pushed tag {tag_name}")
    
    def publish_to_pypi(self, test_only: bool = False) -> None:
        """Publish package to PyPI."""
        if test_only:
            print("Publishing to TestPyPI...")
            self.run_command([
                "python", "-m", "twine", "upload",
                "--repository", "testpypi",
                "dist/*"
            ])
        else:
            print("Publishing to PyPI...")
            self.run_command([
                "python", "-m", "twine", "upload",
                "dist/*"
            ])
        
        print("Package published successfully!")
    
    def create_github_release(self) -> None:
        """Create GitHub release."""
        tag_name = f"v{self.version}"
        title = f"Pynomaly {tag_name}"
        
        # Create release
        self.run_command([
            "gh", "release", "create", tag_name,
            "--title", title,
            "--generate-notes",
            "dist/*"
        ])
        
        print(f"Created GitHub release {tag_name}")
    
    def release(self, dry_run: bool = False, test_only: bool = False) -> None:
        """Execute the complete release process."""
        if not self.validate_version():
            raise ValueError(f"Invalid version format: {self.version}")
        
        print(f"Starting release process for version {self.version}")
        
        if dry_run:
            print("DRY RUN MODE - No changes will be made")
            return
        
        try:
            # Update version files
            self.update_version_files()
            
            # Run tests
            self.run_tests()
            
            # Build package
            self.build_package()
            
            # Commit version changes
            self.run_command(["git", "add", "."])
            self.run_command([
                "git", "commit", "-m", 
                f"chore: bump version to {self.version}"
            ])
            
            # Create tag
            self.create_git_tag()
            
            # Publish to PyPI
            self.publish_to_pypi(test_only=test_only)
            
            # Create GitHub release
            self.create_github_release()
            
            print(f"Release {self.version} completed successfully! 🎉")
            
        except subprocess.CalledProcessError as e:
            print(f"Release failed: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Release Pynomaly")
    parser.add_argument("--version", required=True, help="Version to release")
    parser.add_argument(
        "--type", 
        choices=["major", "minor", "patch", "prerelease"],
        help="Release type"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Run without making changes"
    )
    parser.add_argument(
        "--test-only",
        action="store_true", 
        help="Publish to TestPyPI only"
    )
    
    args = parser.parse_args()
    
    release_manager = ReleaseManager(args.version, args.type)
    release_manager.release(dry_run=args.dry_run, test_only=args.test_only)


if __name__ == "__main__":
    main()
```

### Version Bump Script
```python
#!/usr/bin/env python3
"""
Version bumping utility for Pynomaly.

Usage:
    python scripts/release/bump_version.py patch
    python scripts/release/bump_version.py minor --prerelease alpha
"""

import argparse
import re
from pathlib import Path


class VersionBumper:
    """Handles version bumping logic."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
    
    def get_current_version(self) -> str:
        """Get current version from pyproject.toml."""
        pyproject_file = self.project_root / "pyproject.toml"
        content = pyproject_file.read_text()
        
        match = re.search(r'version = "([^"]*)"', content)
        if not match:
            raise ValueError("Could not find version in pyproject.toml")
        
        return match.group(1)
    
    def parse_version(self, version: str) -> dict:
        """Parse version string into components."""
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z]+)\.(\d+))?$'
        match = re.match(pattern, version)
        
        if not match:
            raise ValueError(f"Invalid version format: {version}")
        
        return {
            'major': int(match.group(1)),
            'minor': int(match.group(2)),
            'patch': int(match.group(3)),
            'prerelease': match.group(4),
            'prerelease_num': int(match.group(5)) if match.group(5) else 0
        }
    
    def bump_version(self, bump_type: str, prerelease: str = None) -> str:
        """Bump version according to type."""
        current = self.get_current_version()
        parts = self.parse_version(current)
        
        if bump_type == "major":
            parts['major'] += 1
            parts['minor'] = 0
            parts['patch'] = 0
            parts['prerelease'] = None
            parts['prerelease_num'] = 0
        elif bump_type == "minor":
            parts['minor'] += 1
            parts['patch'] = 0
            parts['prerelease'] = None
            parts['prerelease_num'] = 0
        elif bump_type == "patch":
            parts['patch'] += 1
            parts['prerelease'] = None
            parts['prerelease_num'] = 0
        elif bump_type == "prerelease":
            if parts['prerelease'] == prerelease:
                parts['prerelease_num'] += 1
            else:
                parts['prerelease'] = prerelease
                parts['prerelease_num'] = 1
        
        # Build new version string
        new_version = f"{parts['major']}.{parts['minor']}.{parts['patch']}"
        if parts['prerelease']:
            new_version += f"-{parts['prerelease']}.{parts['prerelease_num']}"
        
        return new_version


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Bump Pynomaly version")
    parser.add_argument(
        "type",
        choices=["major", "minor", "patch", "prerelease"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--prerelease",
        choices=["alpha", "beta", "rc"],
        help="Prerelease identifier"
    )
    
    args = parser.parse_args()
    
    if args.type == "prerelease" and not args.prerelease:
        parser.error("--prerelease is required when bumping prerelease version")
    
    bumper = VersionBumper()
    current = bumper.get_current_version()
    new_version = bumper.bump_version(args.type, args.prerelease)
    
    print(f"Current version: {current}")
    print(f"New version: {new_version}")


if __name__ == "__main__":
    main()
```

## 📋 Release Checklist Template

```markdown
# Release Checklist: v1.2.0

## Pre-Release
- [ ] Version updated in pyproject.toml
- [ ] Version updated in __init__.py
- [ ] CHANGELOG.md updated
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Security scan completed
- [ ] Performance benchmarks validated

## Release
- [ ] Release branch created
- [ ] Quality gates passed
- [ ] Package built successfully
- [ ] TestPyPI upload successful
- [ ] Manual testing completed
- [ ] Tag created and pushed
- [ ] PyPI upload successful
- [ ] GitHub release created

## Post-Release
- [ ] Documentation deployed
- [ ] Docker images updated
- [ ] Release announcement sent
- [ ] Monitoring dashboards checked
- [ ] Next milestone created
- [ ] Rollback plan documented

## Sign-off
- [ ] Tech Lead approval: ___________
- [ ] QA approval: ___________
- [ ] Product Owner approval: ___________
```

---

## 📝 Summary

These release procedures ensure:

1. **Quality Assurance**: Comprehensive testing and validation
2. **Consistency**: Standardized process for all releases
3. **Traceability**: Clear documentation and version history
4. **Automation**: Reduced manual effort and human error
5. **Safety**: Rollback procedures for emergency situations

Remember to:
- Follow semantic versioning strictly
- Test thoroughly before releasing
- Communicate changes clearly to users
- Monitor releases post-deployment
- Maintain detailed release notes

For questions about the release process, contact the release team or check the [troubleshooting guide](./TROUBLESHOOTING.md).

---

*Last updated: 2025-01-14*
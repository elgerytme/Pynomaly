#!/usr/bin/env python3
"""
Validate Single Package Development Rule

This tool enforces that only one package is actively developed at a time,
with rare exceptions for breaking changes that require coordinated updates.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Set, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of single package development validation."""
    is_valid: bool
    packages: Set[str]
    changed_files: List[str]
    error_message: Optional[str] = None
    breaking_change_justified: bool = False

class SinglePackageDevelopmentValidator:
    """Validator for single package development rule."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.packages_dir = self.repo_root / "src" / "packages"
        
    def get_changed_files(self, base_ref: str = "HEAD~1") -> List[str]:
        """Get list of changed files from git."""
        try:
            # Get changed files compared to base reference
            result = subprocess.run(
                ["git", "diff", "--name-only", base_ref, "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            
            if result.returncode != 0:
                # Fallback to staged files if commit comparison fails
                result = subprocess.run(
                    ["git", "diff", "--name-only", "--cached"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_root
                )
            
            return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            
        except Exception as e:
            print(f"Warning: Could not get changed files from git: {e}")
            return []
    
    def extract_packages_from_files(self, files: List[str]) -> Set[str]:
        """Extract package names from file paths."""
        packages = set()
        
        for file_path in files:
            if file_path.startswith('src/packages/'):
                # Extract package path (first 3 directory levels)
                parts = file_path.split('/')
                if len(parts) >= 3:
                    package = '/'.join(parts[:3])  # src/packages/package_name
                    packages.add(package)
        
        return packages
    
    def check_breaking_change_justification(self) -> bool:
        """Check if breaking change justification exists."""
        justification_files = [
            "BREAKING_CHANGE_JUSTIFICATION.md",
            "docs/BREAKING_CHANGE_JUSTIFICATION.md",
            ".github/BREAKING_CHANGE_JUSTIFICATION.md"
        ]
        
        for file_path in justification_files:
            if (self.repo_root / file_path).exists():
                return True
        
        return False
    
    def validate_commit_message(self) -> bool:
        """Check if commit message indicates breaking change."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            
            commit_message = result.stdout.lower()
            breaking_indicators = [
                "breaking change",
                "breaking:",
                "feat!:",
                "fix!:",
                "BREAKING CHANGE:"
            ]
            
            return any(indicator in commit_message for indicator in breaking_indicators)
            
        except Exception:
            return False
    
    def validate_single_package_development(self, base_ref: str = "HEAD~1") -> ValidationResult:
        """Validate that changes are confined to a single package."""
        changed_files = self.get_changed_files(base_ref)
        
        if not changed_files:
            return ValidationResult(
                is_valid=True,
                packages=set(),
                changed_files=[],
                error_message="No files changed"
            )
        
        packages = self.extract_packages_from_files(changed_files)
        
        # Allow changes to root-level files (documentation, configuration)
        non_package_files = [f for f in changed_files if not f.startswith('src/packages/')]
        
        if len(packages) <= 1:
            return ValidationResult(
                is_valid=True,
                packages=packages,
                changed_files=changed_files
            )
        
        # Multiple packages changed - check for breaking change justification
        breaking_change_justified = (
            self.check_breaking_change_justification() or
            self.validate_commit_message()
        )
        
        if breaking_change_justified:
            return ValidationResult(
                is_valid=True,
                packages=packages,
                changed_files=changed_files,
                breaking_change_justified=True
            )
        
        # Multiple packages without justification - violation
        package_list = ", ".join(sorted(packages))
        error_message = f"""
ERROR: Changes span multiple packages: {package_list}

The Single Package Development Rule requires that changes be confined to one package at a time.

To fix this:
1. Split your changes into separate commits, one per package
2. Or, if this is a legitimate breaking change, create a BREAKING_CHANGE_JUSTIFICATION.md file

Changed files:
{chr(10).join(f'  - {f}' for f in changed_files if f.startswith('src/packages/'))}

For breaking changes, include:
- Justification for the coordinated update
- Impact assessment of affected packages
- Rollback plan
- Testing strategy for all affected packages
"""
        
        return ValidationResult(
            is_valid=False,
            packages=packages,
            changed_files=changed_files,
            error_message=error_message
        )
    
    def generate_report(self, result: ValidationResult) -> str:
        """Generate a detailed validation report."""
        report = []
        report.append("=== Single Package Development Validation Report ===")
        report.append(f"Status: {'✅ VALID' if result.is_valid else '❌ INVALID'}")
        report.append(f"Packages affected: {len(result.packages)}")
        
        if result.packages:
            report.append("Package list:")
            for package in sorted(result.packages):
                report.append(f"  - {package}")
        
        report.append(f"Files changed: {len(result.changed_files)}")
        
        if result.breaking_change_justified:
            report.append("⚠️  Breaking change justification provided")
        
        if result.error_message:
            report.append("\n" + result.error_message)
        
        return "\n".join(report)

def main():
    """Main entry point for validation."""
    validator = SinglePackageDevelopmentValidator()
    
    # Get base reference from command line or use default
    base_ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD~1"
    
    # Validate single package development
    result = validator.validate_single_package_development(base_ref)
    
    # Generate and print report
    report = validator.generate_report(result)
    print(report)
    
    # Exit with appropriate code
    sys.exit(0 if result.is_valid else 1)

if __name__ == "__main__":
    main()
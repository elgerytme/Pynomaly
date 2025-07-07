#!/usr/bin/env python3
"""
Root Directory Organization Validator

This script enforces the PROJECT_ORGANIZATION_PLAN.md rules for root directory
organization, preventing prohibited files from being committed to the root.
"""

import fnmatch
import json
import os
import sys
from pathlib import Path
from typing import List, Set, Tuple


class RootDirectoryValidator:
    """Validates root directory organization compliance."""

    # Essential files allowed in root directory
    ALLOWED_ROOT_FILES = {
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "TODO.md",
        "CLAUDE.md",
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "package-lock.json",
        "Makefile",
        "Pynomaly.code-workspace",
        "CONTRIBUTING.md",
    }

    # Essential directories allowed in root
    ALLOWED_ROOT_DIRECTORIES = {
        "src",
        "tests",
        "docs",
        "scripts",
        "config",
        "deploy",
        "examples",
        "environments",
        "reports",
        "storage",
        "templates",
        "tools",
        "toolchains",
    }

    # File patterns that are PROHIBITED in root
    PROHIBITED_PATTERNS = {
        "test_*.py",
        "*_test.py",
        "conftest.py",
        "setup_*.py",
        "install_*.py",
        "fix_*.py",
        "update_*.py",
        "deploy_*.py",
        "*_GUIDE.md",
        "*_MANUAL.md",
        "*_REPORT.md",
        "*_SUMMARY.md",
        "TESTING_*.md",
        "IMPLEMENTATION_*.md",
        "config_*.py",
        "settings_*.py",
        "*.ini",
        "*.config.*",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "*.egg-info",
        "*.backup",
        "*.bak",
        "*.tmp",
        "*.temp",
        "=*",
        "*.0",
        "buck-out",
    }

    # Directories that should be gitignored if present
    SHOULD_BE_GITIGNORED = {
        "node_modules",
        "__pycache__",
        "buck-out",
        "dist",
        "build",
        ".venv",
        "venv",
        "env",
        ".pytest_cache",
        ".mypy_cache",
        "*.egg-info",
    }

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.violations: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate root directory organization.

        Returns:
            Tuple of (is_valid, violations, warnings)
        """
        self.violations.clear()
        self.warnings.clear()

        # Check files in root directory
        self._check_root_files()

        # Check for prohibited patterns
        self._check_prohibited_patterns()

        # Check gitignore compliance
        self._check_gitignore_compliance()

        # Check directory count
        self._check_directory_limits()

        is_valid = len(self.violations) == 0
        return is_valid, self.violations.copy(), self.warnings.copy()

    def _check_root_files(self):
        """Check files directly in root directory."""
        for item in self.project_root.iterdir():
            if item.name.startswith("."):
                continue  # Skip hidden files

            if item.is_file():
                if item.name not in self.ALLOWED_ROOT_FILES:
                    self.violations.append(f"Prohibited file in root: {item.name}")
            elif item.is_dir():
                if item.name not in self.ALLOWED_ROOT_DIRECTORIES:
                    # Check if it should be gitignored
                    if any(
                        fnmatch.fnmatch(item.name, pattern)
                        for pattern in self.SHOULD_BE_GITIGNORED
                    ):
                        self.warnings.append(
                            f"Directory should be gitignored: {item.name}"
                        )
                    else:
                        self.violations.append(
                            f"Unexpected directory in root: {item.name}"
                        )

    def _check_prohibited_patterns(self):
        """Check for files matching prohibited patterns."""
        for item in self.project_root.iterdir():
            if item.name.startswith("."):
                continue

            for pattern in self.PROHIBITED_PATTERNS:
                if fnmatch.fnmatch(item.name, pattern):
                    self.violations.append(
                        f"File matches prohibited pattern '{pattern}': {item.name}"
                    )
                    break

    def _check_gitignore_compliance(self):
        """Check that build artifacts are properly gitignored."""
        gitignore_path = self.project_root / ".gitignore"

        if not gitignore_path.exists():
            self.warnings.append("No .gitignore file found")
            return

        try:
            gitignore_content = gitignore_path.read_text(encoding="utf-8")

            # Check for essential gitignore patterns
            required_patterns = ["node_modules/", "buck-out/", "dist/", "__pycache__/"]

            for pattern in required_patterns:
                if pattern not in gitignore_content:
                    self.warnings.append(f"Missing gitignore pattern: {pattern}")

        except Exception as e:
            self.warnings.append(f"Error reading .gitignore: {e}")

    def _check_directory_limits(self):
        """Check if root directory exceeds recommended limits."""
        items = list(self.project_root.iterdir())
        visible_items = [item for item in items if not item.name.startswith(".")]

        if len(visible_items) > 20:
            self.warnings.append(
                f"Root directory has {len(visible_items)} items (recommended: ≤20)"
            )

        files = [item for item in visible_items if item.is_file()]
        if len(files) > 12:
            self.violations.append(
                f"Root directory has {len(files)} files (maximum: 12)"
            )

    def get_relocation_suggestions(self) -> List[str]:
        """Provide suggestions for relocating problematic files."""
        suggestions = []

        for item in self.project_root.iterdir():
            if item.name.startswith(".") or item.is_dir():
                continue

            if item.name not in self.ALLOWED_ROOT_FILES:
                suggestion = self._get_relocation_suggestion(item.name)
                if suggestion:
                    suggestions.append(f"{item.name} → {suggestion}")

        return suggestions

    def _get_relocation_suggestion(self, filename: str) -> str:
        """Get relocation suggestion for a specific file."""
        # Test files
        if any(
            fnmatch.fnmatch(filename, pattern) for pattern in ["test_*.py", "*_test.py"]
        ):
            return "tests/"

        # Documentation files
        if any(
            fnmatch.fnmatch(filename, pattern)
            for pattern in ["*_GUIDE.md", "*_MANUAL.md", "*_REPORT.md"]
        ):
            return "docs/project/"

        # Configuration files
        if any(
            fnmatch.fnmatch(filename, pattern)
            for pattern in ["*.ini", "*.config.*", "config_*.py"]
        ):
            return "config/"

        # Scripts
        if any(
            fnmatch.fnmatch(filename, pattern)
            for pattern in ["setup_*.py", "fix_*.py", "install_*.py"]
        ):
            return "scripts/"

        # Build artifacts
        if any(
            fnmatch.fnmatch(filename, pattern)
            for pattern in ["*.egg-info", "dist", "build"]
        ):
            return "reports/builds/"

        return "appropriate directory"


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate root directory organization compliance"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to project root directory",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )
    parser.add_argument(
        "--fix-suggestions", action="store_true", help="Show relocation suggestions"
    )

    args = parser.parse_args()

    validator = RootDirectoryValidator(args.project_root)
    is_valid, violations, warnings = validator.validate()

    if args.json:
        result = {"valid": is_valid, "violations": violations, "warnings": warnings}

        if args.fix_suggestions:
            result["suggestions"] = validator.get_relocation_suggestions()

        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        print("🔍 Root Directory Organization Validation")
        print("=" * 50)

        if is_valid:
            print("✅ Root directory organization is COMPLIANT")
        else:
            print("❌ Root directory organization has VIOLATIONS")

        if violations:
            print(f"\n❌ Violations ({len(violations)}):")
            for violation in violations:
                print(f"  • {violation}")

        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"  • {warning}")

        if args.fix_suggestions:
            suggestions = validator.get_relocation_suggestions()
            if suggestions:
                print(f"\n💡 Relocation Suggestions:")
                for suggestion in suggestions:
                    print(f"  • {suggestion}")

    # Exit with error code if there are violations
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

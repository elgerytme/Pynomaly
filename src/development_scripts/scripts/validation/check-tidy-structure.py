#!/usr/bin/env python3
"""
Project Tidy Structure Check

This script enforces the tidiness rules defined in docs/developer-guides/project_tidy_rules.md
and blocks commits/PRs that violate the project structure standards.
"""

import re
import sys
from pathlib import Path


class TidyStructureChecker:
    """Check project structure against tidiness rules."""

    def __init__(self, root_path: Path = None):
        """Initialize the checker with the project root path."""
        self.root_path = root_path or Path.cwd()
        self.violations = []
        self.warnings = []

        # Forbidden directories at root level
        self.forbidden_root_dirs = {
            "build",
            "dist",
            "output",
            "tmp",
            "temp",
            "cache",
            "backup",
            "old",
            "legacy",
            "vendor",
            "lib",
            "bin",
            "data",
            "samples",
            "misc",
        }

        # Allowed directories at root level
        self.allowed_root_dirs = {
            "src",
            "tests",
            "docs",
            "examples",
            "scripts",
            "config",
            "environments",
            "deploy",
            "tools",
            "artifacts",
            "reports",
            "storage",
            "baseline_outputs",
            "test_reports",
            ".github",
            ".venv",
            ".env",
            ".storybook",
            ".hypothesis",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".tox",
            ".claude",
            "node_modules",
            "htmlcov",
            "templates",
            "stories",
            "toolchains",
        }

        # File naming patterns
        self.python_module_pattern = re.compile(r"^[a-z_][a-z0-9_]*\.py$")
        self.test_file_pattern = re.compile(r"^test_[a-z0-9_]+\.py$")
        self.markdown_pattern = re.compile(r"^[a-z0-9_-]+\.md$")
        self.config_file_pattern = re.compile(
            r"^[a-z0-9_.-]+\.(yaml|yml|toml|json|cfg|ini)$"
        )

        # Special case patterns
        self.readme_pattern = re.compile(r"^README\.md$")
        self.api_doc_pattern = re.compile(r"^API_[A-Z0-9_]+\.md$")
        self.adr_pattern = re.compile(r"^ADR-\d{3}-.+\.md$")

    def check_structure(self) -> bool:
        """Run all structure checks. Returns True if all checks pass."""
        print("🔍 Running project tidy structure check...")

        # Check forbidden directories
        self._check_forbidden_directories()

        # Check file naming conventions
        self._check_file_naming()

        # Check source code organization
        self._check_source_organization()

        # Check test organization
        self._check_test_organization()

        # Check documentation organization
        self._check_docs_organization()

        # Check archive organization
        self._check_archive_organization()

        # Report results
        self._report_results()

        return len(self.violations) == 0

    def _check_forbidden_directories(self):
        """Check for forbidden directories at root level."""
        for item in self.root_path.iterdir():
            if item.is_dir() and item.name in self.forbidden_root_dirs:
                self.violations.append(
                    f"Forbidden directory found: {item.name}/ "
                    f"(use alternatives specified in project_tidy_rules.md)"
                )

    def _check_file_naming(self):
        """Check file naming conventions across the project."""
        # Check Python files
        for py_file in self.root_path.rglob("*.py"):
            if self._should_skip_path(py_file):
                continue

            filename = py_file.name

            # Skip special files
            if filename in ["__init__.py", "conftest.py", "setup.py"]:
                continue

            # Check test files
            if py_file.parts and "test" in py_file.parts:
                if not (
                    self.test_file_pattern.match(filename)
                    or filename.startswith("test_")
                    or filename.endswith("_test.py")
                ):
                    self.violations.append(
                        f"Test file naming violation: {py_file.relative_to(self.root_path)} "
                        f"(should be test_*.py or *_test.py)"
                    )
            else:
                # Check regular Python modules
                if not self.python_module_pattern.match(filename):
                    self.violations.append(
                        f"Python module naming violation: {py_file.relative_to(self.root_path)} "
                        f"(should be snake_case.py)"
                    )

        # Check markdown files
        for md_file in self.root_path.rglob("*.md"):
            if self._should_skip_path(md_file):
                continue

            filename = md_file.name

            # Skip special markdown files
            if (
                self.readme_pattern.match(filename)
                or self.api_doc_pattern.match(filename)
                or self.adr_pattern.match(filename)
            ):
                continue

            # Check regular markdown files
            if not self.markdown_pattern.match(filename.lower()):
                self.violations.append(
                    f"Markdown file naming violation: {md_file.relative_to(self.root_path)} "
                    f"(should be kebab-case.md)"
                )

    def _check_source_organization(self):
        """Check source code directory organization."""
        src_path = self.root_path / "src" / "anomaly_detection"

        if not src_path.exists():
            self.violations.append("Source directory src/anomaly_detection/ not found")
            return

        # Check for expected domain structure
        expected_dirs = {"domain", "application", "infrastructure", "presentation"}
        found_dirs = {d.name for d in src_path.iterdir() if d.is_dir()}

        missing_dirs = expected_dirs - found_dirs
        if missing_dirs:
            self.warnings.append(
                f"Missing expected source directories: {', '.join(missing_dirs)}"
            )

        # Check for test files in source
        for test_file in src_path.rglob("test_*.py"):
            self.violations.append(
                f"Test file in source directory: {test_file.relative_to(self.root_path)} "
                f"(should be in tests/ directory)"
            )

    def _check_test_organization(self):
        """Check test directory organization."""
        tests_path = self.root_path / "tests"

        if not tests_path.exists():
            self.warnings.append("Tests directory not found")
            return

        # Check for non-test files in tests directory
        for py_file in tests_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            if not (
                py_file.name.startswith("test_")
                or py_file.name.endswith("_test.py")
                or py_file.name == "conftest.py"
            ):
                self.violations.append(
                    f"Non-test file in tests directory: {py_file.relative_to(self.root_path)} "
                    f"(should be test_*.py or *_test.py)"
                )

    def _check_docs_organization(self):
        """Check documentation directory organization."""
        docs_path = self.root_path / "docs"

        if not docs_path.exists():
            self.warnings.append("Documentation directory not found")
            return

        # Check for archive organization
        archive_path = docs_path / "archive"
        if archive_path.exists():
            self._check_archive_structure(archive_path)

    def _check_archive_organization(self):
        """Check archive directory organization."""
        archive_path = self.root_path / "docs" / "archive"

        if not archive_path.exists():
            return

        # Check for proper archive structure
        expected_archive_dirs = {
            "historical-project-docs",
            "legacy-algorithm-docs",
            "experimental",
            "deprecated-features",
            "old-configs",
        }

        found_dirs = {d.name for d in archive_path.iterdir() if d.is_dir()}

        # Check for loose files in archive root
        loose_files = [f for f in archive_path.iterdir() if f.is_file()]
        if loose_files:
            self.violations.append(
                f"Loose files in archive root: {[f.name for f in loose_files]} "
                f"(should be organized in subdirectories)"
            )

    def _check_archive_structure(self, archive_path: Path):
        """Check the structure of archive directories."""
        # Check for proper organization
        items = list(archive_path.iterdir())

        # All items should be directories
        files = [item for item in items if item.is_file()]
        if files:
            self.violations.append(
                f"Files in archive root should be in subdirectories: {[f.name for f in files]}"
            )

        # Check directory naming
        for item in items:
            if item.is_dir():
                name = item.name
                # Should follow naming convention
                if not (name.islower() or "-" in name or "_" in name):
                    self.violations.append(
                        f"Archive directory naming violation: {name} "
                        f"(should be lowercase with hyphens or underscores)"
                    )

    def _should_skip_path(self, path: Path) -> bool:
        """Check if a path should be skipped during validation."""
        skip_patterns = [
            ".venv",
            ".env",
            "venv",
            "env",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".hypothesis",
            "node_modules",
            "htmlcov",
            "artifacts",
            "reports",
            "storage",
            "baseline_outputs",
            "test_reports",
            ".claude",
        ]

        path_str = str(path)
        return any(pattern in path_str for pattern in skip_patterns)

    def _report_results(self):
        """Report the results of the structure check."""
        print("\n📊 Structure Check Results:")
        print(f"  Violations: {len(self.violations)}")
        print(f"  Warnings: {len(self.warnings)}")

        if self.violations:
            print("\n❌ VIOLATIONS (must be fixed):")
            for i, violation in enumerate(self.violations, 1):
                print(f"  {i}. {violation}")

        if self.warnings:
            print("\n⚠️  WARNINGS (recommendations):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if not self.violations and not self.warnings:
            print("\n✅ All structure checks passed!")
        elif not self.violations:
            print(
                f"\n✅ No violations found, but {len(self.warnings)} warnings to address."
            )
        else:
            print(
                f"\n❌ {len(self.violations)} violations must be fixed before proceeding."
            )
            print("\nRemediation steps:")
            print("1. Review docs/developer-guides/project_tidy_rules.md")
            print("2. Fix violations listed above")
            print("3. Run this check again to verify fixes")
            print("4. Consider using automated cleanup scripts in scripts/cleanup/")


def main():
    """Main function for command-line usage."""
    checker = TidyStructureChecker()

    try:
        success = checker.check_structure()

        if success:
            print("\n🎉 Project structure is tidy and compliant!")
            sys.exit(0)
        else:
            print("\n🚫 Project structure has violations that must be fixed.")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Error during structure check: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

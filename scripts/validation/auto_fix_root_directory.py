#!/usr/bin/env python3
"""
Automatic Root Directory Organization Fixer

This script automatically relocates files that violate the root directory
organization rules to their appropriate directories.
"""

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List


class RootDirectoryFixer:
    """Automatically fixes root directory organization violations."""

    def __init__(self, project_root: Path = None, dry_run: bool = True):
        self.project_root = project_root or Path.cwd()
        self.dry_run = dry_run
        self.moves_made: List[str] = []
        self.deletions_made: List[str] = []

    def fix_violations(self) -> bool:
        """
        Automatically fix root directory violations.

        Returns:
            True if all fixes were successful, False otherwise
        """
        success = True

        try:
            # Create necessary directories
            self._ensure_directories_exist()

            # Fix files based on patterns
            success &= self._fix_test_files()
            success &= self._fix_documentation_files()
            success &= self._fix_configuration_files()
            success &= self._fix_script_files()
            success &= self._fix_build_artifacts()
            success &= self._delete_version_artifacts()

            # Report results
            self._report_results()

        except Exception as e:
            print(f"❌ Error during auto-fix: {e}", file=sys.stderr)
            success = False

        return success

    def _ensure_directories_exist(self):
        """Create necessary target directories."""
        directories = [
            "docs/project",
            "docs/testing",
            "config",
            "config/web",
            "config/environments",
            "scripts/maintenance",
            "scripts/testing",
            "scripts/setup",
            "tests/scripts",
            "reports/builds",
            "storage",
        ]

        for directory in directories:
            target_dir = self.project_root / directory
            if not self.dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(f"[DRY RUN] Would create directory: {directory}")

    def _fix_test_files(self) -> bool:
        """Move test files to tests/ directory."""
        patterns = ["test_*.py", "*_test.py"]
        target_base = "tests"

        return self._move_files_by_pattern(patterns, target_base, "test files")

    def _fix_documentation_files(self) -> bool:
        """Move documentation files to docs/ subdirectories."""
        moves = {
            "*_REPORT.md": "docs/testing",
            "*_SUMMARY.md": "docs/project",
            "*_GUIDE.md": "docs/project",
            "*_MANUAL.md": "docs/project",
            "TESTING_*.md": "docs/testing",
            "IMPLEMENTATION_*.md": "docs/project",
            "*_PLAN.md": "docs/project/plans",
        }

        success = True
        for pattern, target in moves.items():
            success &= self._move_files_by_pattern(
                [pattern], target, f"documentation files ({pattern})"
            )

        return success

    def _fix_configuration_files(self) -> bool:
        """Move configuration files to config/ directory."""
        moves = {
            "*.ini": "config",
            "*.config.*": "config",
            "config_*.py": "config",
            "settings_*.py": "config",
            "tox.ini": "config",
            "pytest*.ini": "config",
            "lighthouse*.js": "config/web",
            "tailwind.config.js": "config/web",
            "playwright.config.*": "config",
        }

        success = True
        for pattern, target in moves.items():
            success &= self._move_files_by_pattern(
                [pattern], target, f"config files ({pattern})"
            )

        return success

    def _fix_script_files(self) -> bool:
        """Move script files to scripts/ subdirectories."""
        moves = {
            "setup_*.py": "scripts/setup",
            "install_*.py": "scripts/setup",
            "fix_*.py": "scripts/maintenance",
            "update_*.py": "scripts/maintenance",
            "deploy_*.py": "scripts/deploy",
            "*.sh": "scripts/testing",
            "*.ps1": "scripts/testing",
            "*.bat": "scripts/setup",
        }

        success = True
        for pattern, target in moves.items():
            success &= self._move_files_by_pattern(
                [pattern], target, f"script files ({pattern})"
            )

        return success

    def _fix_build_artifacts(self) -> bool:
        """Move build artifacts to reports/builds/ or delete."""
        artifacts = ["dist", "build", "*.egg-info"]
        target = "reports/builds"

        return self._move_files_by_pattern(artifacts, target, "build artifacts")

    def _delete_version_artifacts(self) -> bool:
        """Delete version artifact files."""
        patterns = ["=*", "*.0", "2.0", "*.backup", "*.bak", "*.tmp"]

        success = True
        for item in self.project_root.iterdir():
            if item.name.startswith("."):
                continue

            for pattern in patterns:
                if fnmatch.fnmatch(item.name, pattern):
                    if self.dry_run:
                        print(f"[DRY RUN] Would delete: {item.name}")
                    else:
                        try:
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
                            self.deletions_made.append(str(item.name))
                            print(f"🗑️  Deleted: {item.name}")
                        except Exception as e:
                            print(f"❌ Failed to delete {item.name}: {e}")
                            success = False
                    break

        return success

    def _move_files_by_pattern(
        self, patterns: List[str], target_dir: str, description: str
    ) -> bool:
        """Move files matching patterns to target directory."""
        success = True
        moved_count = 0

        for item in self.project_root.iterdir():
            if item.name.startswith("."):
                continue

            for pattern in patterns:
                if fnmatch.fnmatch(item.name, pattern):
                    target_path = self.project_root / target_dir / item.name

                    if self.dry_run:
                        print(f"[DRY RUN] Would move: {item.name} → {target_dir}/")
                    else:
                        try:
                            # Ensure target directory exists
                            target_path.parent.mkdir(parents=True, exist_ok=True)

                            # Move the file/directory
                            shutil.move(str(item), str(target_path))

                            self.moves_made.append(f"{item.name} → {target_dir}/")
                            moved_count += 1
                            print(f"📁 Moved: {item.name} → {target_dir}/")

                        except Exception as e:
                            print(f"❌ Failed to move {item.name}: {e}")
                            success = False
                    break

        if moved_count > 0:
            print(f"✅ Moved {moved_count} {description}")

        return success

    def _report_results(self):
        """Report the results of the auto-fix operation."""
        if self.dry_run:
            print("\n🔍 DRY RUN COMPLETE - No changes were made")
            print("Run with --apply to actually make changes")
        else:
            print(f"\n✅ AUTO-FIX COMPLETE")
            print(f"Files moved: {len(self.moves_made)}")
            print(f"Files deleted: {len(self.deletions_made)}")

            if self.moves_made:
                print("\nFiles moved:")
                for move in self.moves_made:
                    print(f"  • {move}")

            if self.deletions_made:
                print("\nFiles deleted:")
                for deletion in self.deletions_made:
                    print(f"  • {deletion}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automatically fix root directory organization violations"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to project root directory",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (default is dry-run)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    print("🛠️  Root Directory Auto-Fix Tool")
    print("=" * 40)

    if not args.apply:
        print("🔍 RUNNING IN DRY-RUN MODE")
        print("Use --apply to actually make changes")
        print()

    fixer = RootDirectoryFixer(project_root=args.project_root, dry_run=not args.apply)

    success = fixer.fix_violations()

    if success:
        print("\n✅ Auto-fix completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Auto-fix completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()

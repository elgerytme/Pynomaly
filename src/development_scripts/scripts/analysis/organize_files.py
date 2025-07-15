#!/usr/bin/env python3
"""Automatically organize files according to project standards."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

# Import our analysis tools
try:
    from analyze_project_structure import (
        categorize_stray_directory,
        categorize_stray_file,
        get_recommended_location,
        load_config,
    )
except ImportError:
    # Fallback implementations if import fails
    def load_config(config_path=None):
        """Fallback config loader."""
        return None

    def categorize_stray_file(file_path, config=None):
        """Fallback categorization function."""
        name = file_path.name.lower()
        if any(x in name for x in ["test_", "testing", "_test"]):
            return "testing"
        elif name.endswith((".py", ".sh", ".ps1")) and any(
            name.startswith(x) for x in ["fix_", "setup_", "run_"]
        ):
            return "scripts"
        elif any(x in name for x in ["report", "summary", "analysis"]):
            return "reports"
        elif name.startswith("=") or name.replace(".", "").isdigit():
            return "version_artifacts"
        elif any(x in name for x in ["temp", "tmp", "backup"]):
            return "temporary"
        return "miscellaneous"

    def categorize_stray_directory(dir_path, config=None):
        """Fallback directory categorization."""
        name = dir_path.name.lower()
        if any(x in name for x in ["test_", "testing", "_test"]):
            return "testing"
        elif any(x in name for x in ["temp", "tmp", "env", "venv"]):
            return "temporary"
        return "miscellaneous"

    def get_recommended_location(item_name, category):
        """Fallback location recommendation."""
        location_map = {
            "testing": "tests/",
            "scripts": "scripts/",
            "reports": "reports/",
            "temporary": "DELETE",
            "version_artifacts": "DELETE",
            "artifacts_for_deletion": "DELETE",
            "miscellaneous": "REVIEW",
        }
        return location_map.get(category, "REVIEW")


class FileOrganizer:
    """Automated file organization system."""

    def __init__(
        self, project_root: Path = None, dry_run: bool = True, config_path: str = None
    ):
        self.project_root = project_root or Path.cwd()
        self.dry_run = dry_run
        self.operations = []
        self.errors = []
        self.config = load_config(config_path)

    def analyze_repository(self) -> dict:
        """Analyze the current repository structure."""
        print("🔍 Analyzing repository structure...")

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "stray_files": [],
            "stray_directories": [],
            "operations_planned": [],
            "errors": [],
        }

        # Scan root directory for stray items
        for item in self.project_root.iterdir():
            if item.name.startswith(".") and item.name not in {
                ".gitignore",
                ".gitattributes",
                ".pre-commit-config.yaml",
            }:
                continue

            if item.is_file():
                if not self._is_allowed_root_file(item.name):
                    category = categorize_stray_file(item, self.config)
                    target = get_recommended_location(item.name, category)

                    stray_info = {
                        "path": str(item.relative_to(self.project_root)),
                        "name": item.name,
                        "category": category,
                        "target": target,
                        "size": item.stat().st_size,
                    }
                    analysis["stray_files"].append(stray_info)

            elif item.is_dir():
                if not self._is_allowed_root_directory(item.name):
                    category = categorize_stray_directory(item, self.config)
                    target = get_recommended_location(item.name, category)

                    stray_info = {
                        "path": str(item.relative_to(self.project_root)),
                        "name": item.name,
                        "category": category,
                        "target": target,
                        "item_count": len(list(item.iterdir())) if item.exists() else 0,
                    }
                    analysis["stray_directories"].append(stray_info)

        return analysis

    def plan_organization(self, analysis: dict) -> list[dict]:
        """Plan file organization operations."""
        print("📋 Planning organization operations...")

        operations = []

        # Plan file operations
        for file_info in analysis["stray_files"]:
            operation = self._plan_file_operation(file_info)
            if operation:
                operations.append(operation)

        # Plan directory operations
        for dir_info in analysis["stray_directories"]:
            operation = self._plan_directory_operation(dir_info)
            if operation:
                operations.append(operation)

        # Sort operations by priority (deletions first, then moves)
        operations.sort(
            key=lambda x: (0 if x["action"] == "delete" else 1, x["source"])
        )

        return operations

    def execute_operations(self, operations: list[dict]) -> dict:
        """Execute the planned operations."""
        print(
            f"🔄 {'Simulating' if self.dry_run else 'Executing'} {len(operations)} operations..."
        )

        results = {"executed": [], "skipped": [], "errors": [], "summary": {}}

        for operation in operations:
            try:
                if self._execute_operation(operation):
                    results["executed"].append(operation)
                else:
                    results["skipped"].append(operation)
            except Exception as e:
                error_info = {"operation": operation, "error": str(e)}
                results["errors"].append(error_info)
                print(
                    f"❌ Error executing {operation['action']} on {operation['source']}: {e}"
                )

        # Generate summary
        actions = [op["action"] for op in results["executed"]]
        results["summary"] = {
            "total_operations": len(operations),
            "executed": len(results["executed"]),
            "skipped": len(results["skipped"]),
            "errors": len(results["errors"]),
            "moves": actions.count("move"),
            "deletions": actions.count("delete"),
            "creations": actions.count("create_directory"),
        }

        return results

    def _is_allowed_root_file(self, filename: str) -> bool:
        """Check if file is allowed in root directory."""
        allowed_files = {
            "README.md",
            "LICENSE",
            "CHANGELOG.md",
            "TODO.md",
            "CLAUDE.md",
            "CONTRIBUTING.md",
            "MANIFEST.in",
            "Makefile",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "requirements-minimal.txt",
            "requirements-server.txt",
            "requirements-production.txt",
            "requirements-test.txt",
            "package.json",
            "package-lock.json",
            "Pynomaly.code-workspace",
            ".gitignore",
            ".gitattributes",
            ".pre-commit-config.yaml",
            ".pyno-org.yaml",
        }

        # Update with configuration if available
        if self.config and "allowed_root_files" in self.config:
            allowed_files.update(self.config["allowed_root_files"])

        return filename in allowed_files

    def _is_allowed_root_directory(self, dirname: str) -> bool:
        """Check if directory is allowed in root."""
        allowed_dirs = {
            "src",
            "tests",
            "docs",
            "examples",
            "scripts",
            "deploy",
            "config",
            "reports",
            "storage",
            "templates",
            "analytics",
            "screenshots",
            "environments",
            "baseline_outputs",
            "test_reports",
            ".github",
            ".git",
            "node_modules",
        }

        # Update with configuration if available
        if self.config and "allowed_root_directories" in self.config:
            allowed_dirs.update(self.config["allowed_root_directories"])

        return dirname in allowed_dirs

    def _plan_file_operation(self, file_info: dict) -> dict | None:
        """Plan operation for a stray file."""
        self.project_root / file_info["path"]
        target = file_info["target"]

        if target == "DELETE":
            return {
                "action": "delete",
                "type": "file",
                "source": file_info["path"],
                "target": None,
                "reason": f"Delete {file_info['category']} file",
                "category": file_info["category"],
            }

        elif target.endswith("/"):
            target_dir = self.project_root / target.rstrip("/")
            target_path = target_dir / file_info["name"]

            return {
                "action": "move",
                "type": "file",
                "source": file_info["path"],
                "target": str(target_path.relative_to(self.project_root)),
                "reason": f"Move {file_info['category']} file to appropriate directory",
                "category": file_info["category"],
                "requires_directory": str(target_dir.relative_to(self.project_root)),
            }

        return None

    def _plan_directory_operation(self, dir_info: dict) -> dict | None:
        """Plan operation for a stray directory."""
        target = dir_info["target"]

        if target == "DELETE":
            return {
                "action": "delete",
                "type": "directory",
                "source": dir_info["path"],
                "target": None,
                "reason": f"Delete {dir_info['category']} directory",
                "category": dir_info["category"],
            }

        elif target.endswith("/") and dir_info["category"] == "testing":
            # Special handling for test directories - merge with tests/
            return {
                "action": "merge",
                "type": "directory",
                "source": dir_info["path"],
                "target": "tests/",
                "reason": f"Merge {dir_info['category']} directory contents",
                "category": dir_info["category"],
            }

        return None

    def _execute_operation(self, operation: dict) -> bool:
        """Execute a single operation."""
        action = operation["action"]
        source_path = self.project_root / operation["source"]

        if self.dry_run:
            print(
                f"  📋 [DRY RUN] {action.upper()}: {operation['source']} -> {operation.get('target', 'DELETED')}"
            )
            return True

        if action == "delete":
            return self._delete_item(source_path, operation)
        elif action == "move":
            return self._move_item(source_path, operation)
        elif action == "merge":
            return self._merge_directory(source_path, operation)
        elif action == "create_directory":
            return self._create_directory(source_path, operation)

        return False

    def _delete_item(self, source_path: Path, operation: dict) -> bool:
        """Delete a file or directory."""
        if not source_path.exists():
            return False

        print(f"  🗑️  DELETE: {operation['source']}")

        if source_path.is_file():
            source_path.unlink()
        else:
            shutil.rmtree(source_path)

        return True

    def _move_item(self, source_path: Path, operation: dict) -> bool:
        """Move a file or directory."""
        target_path = self.project_root / operation["target"]

        # Create target directory if needed
        if "requires_directory" in operation:
            target_dir = self.project_root / operation["requires_directory"]
            target_dir.mkdir(parents=True, exist_ok=True)

        print(f"  📁 MOVE: {operation['source']} -> {operation['target']}")

        # Check if target already exists
        if target_path.exists():
            print(f"    ⚠️  Target exists, renaming to {target_path.name}.backup")
            backup_path = target_path.with_suffix(target_path.suffix + ".backup")
            target_path.rename(backup_path)

        shutil.move(str(source_path), str(target_path))
        return True

    def _merge_directory(self, source_path: Path, operation: dict) -> bool:
        """Merge directory contents."""
        target_dir = self.project_root / operation["target"]
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"  🔀 MERGE: {operation['source']} -> {operation['target']}")

        # Move all contents
        for item in source_path.iterdir():
            target_item = target_dir / item.name
            if target_item.exists():
                backup_name = f"{item.name}.backup.{int(datetime.now().timestamp())}"
                target_item = target_dir / backup_name

            shutil.move(str(item), str(target_item))

        # Remove empty source directory
        source_path.rmdir()
        return True

    def _create_directory(self, path: Path, operation: dict) -> bool:
        """Create a directory."""
        print(f"  📁 CREATE: {operation['source']}")
        path.mkdir(parents=True, exist_ok=True)
        return True


def print_summary(results: dict, dry_run: bool):
    """Print operation summary."""
    summary = results["summary"]

    print(f"\n📊 Organization Summary ({'DRY RUN' if dry_run else 'EXECUTED'})")
    print("=" * 50)
    print(f"Total operations: {summary['total_operations']}")
    print(f"Executed: {summary['executed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Errors: {summary['errors']}")
    print()
    print(f"Moves: {summary['moves']}")
    print(f"Deletions: {summary['deletions']}")
    print(f"Directory creations: {summary['creations']}")

    if results["errors"]:
        print("\n❌ Errors encountered:")
        for error in results["errors"]:
            print(f"  • {error['operation']['source']}: {error['error']}")


def main():
    """Main organization function - now calls the unified CLI."""
    print("[!] DEPRECATED: This script has been replaced by 'pyno-org organize'")
    print("    Falling back to the unified CLI...\n")

    try:
        # Map old arguments to new CLI
        import sys

        new_args = [
            sys.executable,
            str(Path(__file__).parent.parent / "pyno_org.py"),
            "organize",
        ]

        # Parse original arguments to map to new CLI
        parser = argparse.ArgumentParser(description="Organize Pynomaly project files")
        parser.add_argument(
            "--execute",
            action="store_true",
            help="Execute operations (default is dry-run)",
        )
        parser.add_argument(
            "--force", action="store_true", help="Force execution without confirmation"
        )
        parser.add_argument("--output", type=str, help="Save report to file")

        args = parser.parse_args()

        # Map arguments to new CLI
        if args.execute:
            new_args.append("--fix")
        else:
            new_args.append("--dry")

        if args.force:
            new_args.append("--force")

        if args.output:
            new_args.extend(["--output", args.output])

        # Call the unified CLI
        result = subprocess.run(new_args, capture_output=False, text=True)
        sys.exit(result.returncode)

    except Exception as e:
        print(f"[-] Error calling unified CLI: {e}")
        print("    Falling back to original implementation...\n")

        # Fallback to original implementation
        parser = argparse.ArgumentParser(description="Organize Pynomaly project files")
        parser.add_argument(
            "--execute",
            action="store_true",
            help="Execute operations (default is dry-run)",
        )
        parser.add_argument(
            "--force", action="store_true", help="Force execution without confirmation"
        )
        parser.add_argument("--output", type=str, help="Save report to file")

        args = parser.parse_args()

        # Initialize organizer
        organizer = FileOrganizer(dry_run=not args.execute)

        print("[*] Pynomaly File Organization Tool")
        print("=" * 50)

        # Analyze current state
        analysis = organizer.analyze_repository()

        if not analysis["stray_files"] and not analysis["stray_directories"]:
            print("[+] Repository is already well-organized!")
            return

        print(
            f"Found {len(analysis['stray_files'])} stray files and {len(analysis['stray_directories'])} stray directories"
        )

        # Plan operations
        operations = organizer.plan_organization(analysis)

        if not operations:
            print("[!] No operations needed")
            return

        print(f"\nPlanned {len(operations)} operations:")
        for op in operations:
            print(
                f"  • {op['action'].upper()}: {op['source']} -> {op.get('target', 'DELETED')}"
            )

        # Confirm execution if not dry run
        if args.execute and not args.force:
            response = input(f"\n[!] Execute {len(operations)} operations? [y/N]: ")
            if response.lower() != "y":
                print("[-] Operation cancelled")
                return

        # Execute operations
        results = organizer.execute_operations(operations)

        # Print summary
        print_summary(results, not args.execute)

        # Save report if requested
        if args.output:
            report = {
                "analysis": analysis,
                "operations": operations,
                "results": results,
            }
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n[>] Report saved to: {args.output}")


if __name__ == "__main__":
    main()

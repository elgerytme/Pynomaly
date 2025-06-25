#!/usr/bin/env python3
"""Detect and report stray files in the repository."""

import json
import re
import sys
from pathlib import Path


def detect_stray_files(filenames: list[str] = None) -> tuple[list[dict], list[str]]:
    """
    Detect stray files that are in incorrect locations.

    Args:
        filenames: List of files to check. If None, checks all files.

    Returns:
        (stray_files, suggestions)
    """

    if filenames is None:
        # Check all files in repository
        project_root = Path.cwd()
        all_files = []
        for file_path in project_root.rglob("*"):
            if file_path.is_file() and not any(
                part.startswith(".") for part in file_path.parts[1:]
            ):
                all_files.append(str(file_path.relative_to(project_root)))
        filenames = all_files

    stray_files = []
    suggestions = []

    for filename in filenames:
        file_path = Path(filename)

        # Skip hidden files and directories
        if any(part.startswith(".") for part in file_path.parts):
            continue

        # Check if file is in wrong location
        current_dir = str(file_path.parent) if file_path.parent != Path(".") else "root"
        expected_dir = get_expected_directory(file_path.name, current_dir)

        if expected_dir and expected_dir != current_dir:
            stray_files.append(
                {
                    "file": filename,
                    "current_location": current_dir,
                    "expected_location": expected_dir,
                    "category": categorize_file(file_path.name),
                }
            )

            if expected_dir == "DELETE":
                suggestions.append(f"DELETE {filename} (temporary/artifact file)")
            else:
                suggestions.append(
                    f"MOVE {filename} from {current_dir} to {expected_dir}"
                )

    return stray_files, suggestions


def categorize_file(filename: str) -> str:
    """Categorize a file based on its name and extension."""
    name_lower = filename.lower()

    # Testing files
    if any(x in name_lower for x in ["test_", "_test", "testing"]):
        return "testing"

    # Documentation files
    if filename.endswith(".md"):
        if any(
            x in filename.upper()
            for x in ["_REPORT", "_SUMMARY", "_GUIDE", "_ANALYSIS", "_PLAN"]
        ):
            return "documentation"
        elif any(
            x in filename.upper()
            for x in ["TESTING_", "DEPLOYMENT_", "IMPLEMENTATION_"]
        ):
            return "documentation"

    # Script files
    if filename.endswith((".py", ".sh", ".ps1", ".bat")):
        if any(
            filename.startswith(x)
            for x in ["fix_", "setup_", "install_", "run_", "deploy_", "build_"]
        ):
            return "scripts"

    # Report files
    if any(
        x in name_lower for x in ["report", "summary", "analysis"]
    ) and filename.endswith((".json", ".html", ".xml")):
        return "reports"

    # Temporary files
    if any(x in name_lower for x in ["temp", "tmp", "backup", "scratch", "debug"]):
        return "temporary"

    # Version artifacts
    if filename.startswith("=") or re.match(r"^\d+\.\d+(\.\d+)?.*$", filename):
        return "version_artifacts"

    # Configuration files
    if (
        filename.endswith((".json", ".yaml", ".yml", ".ini", ".toml", ".cfg"))
        and "config" in name_lower
    ):
        return "configuration"

    return "miscellaneous"


def get_expected_directory(filename: str, current_dir: str) -> str:
    """Get the expected directory for a file."""
    category = categorize_file(filename)

    # Files that should be deleted
    if category in ["temporary", "version_artifacts"]:
        return "DELETE"

    # Files that should be in specific directories
    expected_dirs = {
        "testing": "tests",
        "scripts": "scripts",
        "documentation": "docs",
        "reports": "reports",
        "configuration": "config",
    }

    expected = expected_dirs.get(category)
    if expected and current_dir != expected:
        return expected

    # Special rules for root directory
    if current_dir == "root":
        # Only certain files are allowed in root
        allowed_root_files = {
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
        }

        if filename not in allowed_root_files:
            if category == "testing":
                return "tests"
            elif category == "scripts":
                return "scripts"
            elif category == "documentation":
                return "docs"
            elif category == "reports":
                return "reports"
            elif category == "configuration":
                return "config"
            else:
                return "appropriate_directory"

    return None  # File is in correct location


def print_detection_results(stray_files: list[dict], suggestions: list[str]):
    """Print stray file detection results."""
    if not stray_files:
        print("✅ No stray files detected")
        return

    print(f"⚠️  Detected {len(stray_files)} stray files")

    # Group by category
    by_category = {}
    for file_info in stray_files:
        category = file_info["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(file_info)

    print("\n📊 Stray files by category:")
    for category, files in by_category.items():
        print(f"  {category}: {len(files)} files")
        for file_info in files[:3]:  # Show first 3 files
            print(
                f"    • {file_info['file']} ({file_info['current_location']} → {file_info['expected_location']})"
            )
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")

    print("\n💡 Recommended actions:")
    for suggestion in suggestions[:10]:  # Show first 10 suggestions
        print(f"  • {suggestion}")
    if len(suggestions) > 10:
        print(f"  ... and {len(suggestions) - 10} more actions")


def main():
    """Main detection function."""
    # Get files from command line arguments or check all files
    filenames = sys.argv[1:] if len(sys.argv) > 1 else None

    print("🔍 Detecting stray files...")

    stray_files, suggestions = detect_stray_files(filenames)

    print_detection_results(stray_files, suggestions)

    # Save detection report
    report = {
        "stray_files": stray_files,
        "suggestions": suggestions,
        "total_stray": len(stray_files),
        "categories": list(set(f["category"] for f in stray_files)),
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    with open(reports_dir / "stray_files_detection.json", "w") as f:
        json.dump(report, f, indent=2)

    # Exit with warning if stray files found
    if stray_files:
        sys.exit(1)  # Non-zero exit for pre-commit hook

    sys.exit(0)


if __name__ == "__main__":
    main()

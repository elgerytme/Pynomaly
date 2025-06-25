#!/usr/bin/env python3
"""
Changelog Update Checker

Validates that CHANGELOG.md has been updated when significant changes are made.
This script can be used as a pre-commit hook or CI check.
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_diff_files():
    """Get list of files changed in current commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []


def get_git_diff_stats():
    """Get git diff statistics."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--numstat"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []


def is_significant_change(changed_files, diff_stats):
    """Determine if changes are significant enough to require changelog update."""

    # Files that always require changelog updates
    critical_paths = [
        "src/",
        "pynomaly/",
        "examples/",
        "docs/",
        "scripts/",
        "docker/",
        "tests/",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "Dockerfile",
    ]

    # Files that don't require changelog updates
    ignore_paths = [
        ".gitignore",
        ".github/",
        "TODO.md",
        "CLAUDE.md",
        ".pytest_cache/",
        "__pycache__/",
        ".venv/",
        "node_modules/",
        ".coverage",
        "htmlcov/",
        ".mypy_cache/",
    ]

    significant_files = []
    total_lines_changed = 0

    for file_path in changed_files:
        # Skip empty or non-existent files
        if not file_path:
            continue

        # Check if file should be ignored
        if any(file_path.startswith(ignore) for ignore in ignore_paths):
            continue

        # Check if file is in critical paths
        if any(file_path.startswith(critical) for critical in critical_paths):
            significant_files.append(file_path)

    # Calculate total lines changed
    for stat in diff_stats:
        if not stat:
            continue
        parts = stat.split("\t")
        if len(parts) >= 2:
            try:
                added = int(parts[0]) if parts[0] != "-" else 0
                deleted = int(parts[1]) if parts[1] != "-" else 0
                total_lines_changed += added + deleted
            except ValueError:
                continue

    # Determine significance
    is_significant = (
        len(significant_files) > 0  # Any critical files changed
        or total_lines_changed > 20  # More than 20 lines changed total
    )

    return is_significant, significant_files, total_lines_changed


def check_changelog_updated(changed_files):
    """Check if CHANGELOG.md was updated."""
    return "CHANGELOG.md" in changed_files


def get_latest_changelog_date():
    """Get the date of the latest changelog entry."""
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        return None

    with open(changelog_path) as f:
        content = f.read()

    # Look for date pattern [Version] - YYYY-MM-DD
    date_pattern = r"\[([\d.]+)\]\s*-\s*(\d{4}-\d{2}-\d{2})"
    matches = re.findall(date_pattern, content)

    if matches:
        # Return the first (latest) date found
        return matches[0][1]

    return None


def suggest_changelog_entry(significant_files, total_lines_changed):
    """Suggest a changelog entry based on changed files."""
    suggestions = []

    # Analyze file types
    file_categories = {
        "src/": "Core functionality",
        "examples/": "Examples and demonstrations",
        "docs/": "Documentation",
        "tests/": "Testing infrastructure",
        "scripts/": "Utility scripts and tools",
        "docker/": "Docker and deployment",
        "pyproject.toml": "Dependencies and configuration",
        "requirements.txt": "Dependencies",
        "README.md": "Project documentation",
    }

    categories_found = set()
    for file_path in significant_files:
        for pattern, category in file_categories.items():
            if file_path.startswith(pattern) or pattern in file_path:
                categories_found.add(category)

    today = datetime.now().strftime("%Y-%m-%d")

    print("\n📝 Suggested CHANGELOG.md entry template:")
    print("```markdown")
    print(f"## [X.Y.Z] - {today}")
    print("")

    if "Core functionality" in categories_found:
        print("### Added")
        print("- [Describe new features or capabilities]")
        print("")
        print("### Changed")
        print("- [Describe changes in existing functionality]")
        print("")
        print("### Fixed")
        print("- [Describe bug fixes or issue resolutions]")
        print("")

    if "Documentation" in categories_found:
        print("### Documentation")
        print("- [Describe documentation updates]")
        print("")

    if "Testing infrastructure" in categories_found:
        print("### Testing")
        print("- [Describe test improvements or additions]")
        print("")

    if "Docker and deployment" in categories_found:
        print("### Infrastructure")
        print("- [Describe deployment or infrastructure changes]")
        print("")

    print("```")
    print("")
    print(
        f"📊 Change summary: {len(significant_files)} files, {total_lines_changed} lines changed"
    )
    print(f"📁 File categories: {', '.join(sorted(categories_found))}")


def main():
    """Main changelog update checker."""
    print("🔍 Checking if CHANGELOG.md update is required...")

    # Get changed files
    changed_files = get_git_diff_files()
    diff_stats = get_git_diff_stats()

    if not changed_files:
        print("✅ No files changed in commit")
        return 0

    # Check if changes are significant
    is_significant, significant_files, total_lines_changed = is_significant_change(
        changed_files, diff_stats
    )

    if not is_significant:
        print("✅ Changes don't require CHANGELOG.md update")
        print(f"   Changed files: {', '.join(changed_files)}")
        return 0

    # Check if changelog was updated
    changelog_updated = check_changelog_updated(changed_files)

    if changelog_updated:
        print("✅ CHANGELOG.md has been updated")
        latest_date = get_latest_changelog_date()
        if latest_date:
            print(f"   Latest entry: {latest_date}")
        return 0

    # Changelog update required but missing
    print("❌ CHANGELOG.md update required but missing!")
    print(f"   Significant files changed: {len(significant_files)}")
    print(f"   Total lines changed: {total_lines_changed}")
    print(f"   Changed files: {', '.join(significant_files[:5])}")
    if len(significant_files) > 5:
        print(f"   ... and {len(significant_files) - 5} more")

    suggest_changelog_entry(significant_files, total_lines_changed)

    print("\n💡 To fix this:")
    print("   1. Update CHANGELOG.md with details about your changes")
    print("   2. Follow the format in CLAUDE.md > Changelog Management Rules")
    print("   3. Add the changelog update to your commit")

    # Return exit code 1 to fail the check
    return 1


if __name__ == "__main__":
    sys.exit(main())

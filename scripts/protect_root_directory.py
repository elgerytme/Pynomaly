#!/usr/bin/env python3
"""Protect root directory from stray files during commits."""

import re
import sys
from pathlib import Path

# Patterns for files that should NOT be in root directory
PROHIBITED_PATTERNS = [
    # Testing files
    r"^test_.*\.(py|sh|ps1)$",
    r"^.*_test\.(py|sh|ps1)$",
    r"^testing_.*",
    r"^.*_testing_.*",
    r"^execute_.*_test.*",
    # Script files
    r"^fix_.*\.(py|sh|ps1)$",
    r"^setup_.*\.(py|sh|ps1|bat)$",
    r"^install_.*\.(py|sh|ps1)$",
    r"^run_.*\.(py|sh|ps1)$",
    r"^deploy_.*\.(py|sh|ps1)$",
    # Documentation that belongs elsewhere
    r".*_REPORT\.md$",
    r".*_SUMMARY\.md$",
    r".*_GUIDE\.md$",
    r".*_ANALYSIS.*\.md$",
    r".*_PLAN\.md$",
    r"^TESTING_.*\.md$",
    r"^DEPLOYMENT_.*\.md$",
    r"^IMPLEMENTATION_.*\.md$",
    # Temporary and backup files
    r".*\.(backup|bak|tmp|temp)$",
    r"^temp_.*",
    r"^tmp_.*",
    r"^scratch_.*",
    r"^debug_.*",
    r"^backup_.*",
    # Version artifacts
    r"^=.*",
    r"^\d+\.\d+(\.\d+)?.*$",
    # Environment files
    r"^\.env_.*",
    r"^env_.*",
    r"^venv_.*",
    r"^\.venv_.*",
]

# Define correct locations for different file types
LOCATION_MAP = {
    "testing": "tests/",
    "scripts": "scripts/",
    "documentation": "docs/",
    "reports": "reports/",
    "temporary": "DELETE (temporary files should not be committed)",
    "version_artifacts": "DELETE (version artifacts should not be committed)",
    "environment": "DELETE (environment files should not be committed)",
}


def categorize_file(filename: str) -> str:
    """Categorize a file to determine its proper location."""
    name_lower = filename.lower()

    # Check for testing files
    if any(pattern in name_lower for pattern in ["test_", "_test", "testing"]):
        return "testing"

    # Check for script files
    if filename.endswith((".py", ".sh", ".ps1", ".bat")) and any(
        filename.startswith(prefix)
        for prefix in ["fix_", "setup_", "install_", "run_", "deploy_"]
    ):
        return "scripts"

    # Check for documentation
    if filename.endswith(".md") and any(
        suffix in filename.upper()
        for suffix in [
            "_REPORT",
            "_SUMMARY",
            "_GUIDE",
            "_ANALYSIS",
            "_PLAN",
            "TESTING_",
            "DEPLOYMENT_",
            "IMPLEMENTATION_",
        ]
    ):
        return "documentation"

    # Check for temporary files
    if any(
        pattern in name_lower
        for pattern in ["temp", "tmp", "backup", "scratch", "debug"]
    ) or filename.endswith((".backup", ".bak", ".tmp", ".temp")):
        return "temporary"

    # Check for version artifacts
    if filename.startswith("=") or re.match(r"^\d+\.\d+(\.\d+)?.*$", filename):
        return "version_artifacts"

    # Check for environment files
    if any(pattern in name_lower for pattern in ["env_", "venv_", ".env_", ".venv_"]):
        return "environment"

    return "unknown"


def check_files(filenames: list) -> bool:
    """Check if any files should not be in root directory."""
    violations = []

    for filename in filenames:
        # Skip if file is not in root directory
        if "/" in filename:
            continue

        # Check against prohibited patterns
        for pattern in PROHIBITED_PATTERNS:
            if re.match(pattern, filename, re.IGNORECASE):
                category = categorize_file(filename)
                correct_location = LOCATION_MAP.get(category, "appropriate directory")
                violations.append(
                    {
                        "file": filename,
                        "category": category,
                        "location": correct_location,
                    }
                )
                break

    if violations:
        print("❌ Root directory protection: Prohibited files detected")
        print("\n🚨 Files that should not be in root directory:")

        for violation in violations:
            print(f"  • {violation['file']} → {violation['location']}")

        print("\n💡 Actions required:")
        print("  1. Move files to their appropriate directories")
        print("  2. Delete temporary/artifact files")
        print("  3. Update .gitignore to prevent future violations")
        print("\n📚 See: docs/development/FILE_ORGANIZATION_STANDARDS.md")

        return False

    return True


def main():
    """Main protection function."""
    # Get files from git staging area or command line arguments
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        # Get staged files from git
        import subprocess

        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                check=True,
            )
            filenames = (
                result.stdout.strip().split("\n") if result.stdout.strip() else []
            )
        except subprocess.CalledProcessError:
            # If not in git repo, check all files in current directory
            filenames = [f.name for f in Path(".").iterdir() if f.is_file()]

    # Filter to only root directory files
    root_files = [f for f in filenames if "/" not in f and f != ""]

    if not root_files:
        return  # No root files to check

    print(f"🛡️  Checking {len(root_files)} root directory files...")

    if check_files(root_files):
        print("✅ Root directory protection: All files are properly located")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

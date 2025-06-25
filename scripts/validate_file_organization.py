#!/usr/bin/env python3
"""Validate file organization against project standards."""

import json
import sys
from pathlib import Path

# Define allowed files in root directory
ALLOWED_ROOT_FILES = {
    # Essential project files
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "TODO.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "MANIFEST.in",
    "Makefile",
    # Python package configuration
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    # Requirements files
    "requirements.txt",
    "requirements-minimal.txt",
    "requirements-server.txt",
    "requirements-production.txt",
    "requirements-test.txt",
    # Node.js/Frontend
    "package.json",
    "package-lock.json",
    # IDE/Editor
    "Pynomaly.code-workspace",
    # Git and CI/CD
    ".gitignore",
    ".gitattributes",
    ".pre-commit-config.yaml",
    # Hidden files that are acceptable
    ".github",
    ".git",
}

# Define allowed directories in root
ALLOWED_ROOT_DIRS = {
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
    ".github",
    ".git",
    "node_modules",
}

# File patterns that should be moved to specific directories
RELOCATION_RULES = {
    "tests/": [
        r"^test_.*\.(py|sh|ps1)$",
        r"^.*_test\.(py|sh|ps1)$",
        r"^testing_.*\.(py|sh|md)$",
        r"^.*_testing_.*\.(py|sh|md)$",
        r"^execute_.*_test.*\.(py|sh)$",
        r".*TESTING.*\.md$",
        r".*TEST.*\.md$",
    ],
    "scripts/": [
        r"^fix_.*\.(py|sh|ps1)$",
        r"^setup_.*\.(py|sh|ps1|bat)$",
        r"^install_.*\.(py|sh|ps1)$",
        r"^run_.*\.(py|sh|ps1)$",
        r"^deploy_.*\.(py|sh|ps1)$",
        r"^build_.*\.(py|sh|ps1)$",
    ],
    "docs/": [
        r".*_REPORT\.md$",
        r".*_SUMMARY\.md$",
        r".*_GUIDE\.md$",
        r".*_ANALYSIS.*\.md$",
        r".*_PLAN\.md$",
        r"^DEPLOYMENT_.*\.md$",
        r"^IMPLEMENTATION_.*\.md$",
    ],
    "reports/": [
        r".*_report\.(json|html|xml)$",
        r".*_results\.(json|html|xml)$",
        r".*_analysis\.(json|html|xml)$",
    ],
    "DELETE": [
        r"^=.*",  # Version artifacts
        r"^\d+\.\d+(\.\d+)?.*",  # Version numbers
        r".*\.(backup|bak|tmp|temp)$",  # Backup/temp files
        r"^temp_.*",
        r"^tmp_.*",
        r"^scratch_.*",
        r"^debug_.*",
    ],
}


def validate_file_organization() -> tuple[bool, list[str], list[str]]:
    """
    Validate project file organization.

    Returns:
        (is_valid, violations, suggestions)
    """
    project_root = Path.cwd()
    violations = []
    suggestions = []
    is_valid = True

    # Check root directory contents
    for item in project_root.iterdir():
        if item.name.startswith(".") and item.name not in {
            ".gitignore",
            ".gitattributes",
            ".pre-commit-config.yaml",
            ".github",
            ".git",
        }:
            continue

        if item.is_file():
            if item.name not in ALLOWED_ROOT_FILES:
                is_valid = False
                violation = f"Stray file in root: {item.name}"
                violations.append(violation)

                # Find appropriate location
                suggested_location = get_suggested_location(item.name)
                suggestions.append(f"Move {item.name} to {suggested_location}")

        elif item.is_dir():
            if item.name not in ALLOWED_ROOT_DIRS:
                is_valid = False
                violation = f"Stray directory in root: {item.name}/"
                violations.append(violation)

                # Suggest relocation or deletion
                if any(
                    pattern in item.name.lower()
                    for pattern in ["test", "temp", "tmp", "env", "venv"]
                ):
                    if "test" in item.name.lower():
                        suggestions.append(
                            f"Move {item.name}/ contents to tests/ or delete if obsolete"
                        )
                    else:
                        suggestions.append(
                            f"DELETE {item.name}/ (temporary/environment directory)"
                        )
                else:
                    suggestions.append(
                        f"Review {item.name}/ and move to appropriate location"
                    )

    return is_valid, violations, suggestions


def get_suggested_location(filename: str) -> str:
    """Get suggested location for a file based on patterns."""
    import re

    for target_dir, patterns in RELOCATION_RULES.items():
        for pattern in patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return target_dir

    # Default suggestions based on file extension
    ext = Path(filename).suffix.lower()
    if ext == ".md":
        return "docs/"
    elif ext in [".py", ".sh", ".ps1", ".bat"]:
        return "scripts/"
    elif ext in [".json", ".yaml", ".yml"]:
        return "config/"

    return "REVIEW (manual classification needed)"


def print_results(is_valid: bool, violations: list[str], suggestions: list[str]):
    """Print validation results."""
    if is_valid:
        print("✅ File organization validation PASSED")
        return

    print("❌ File organization validation FAILED")
    print("\n🚨 Violations found:")
    for violation in violations:
        print(f"  • {violation}")

    if suggestions:
        print("\n💡 Suggested actions:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")

    print("\n📚 For more information, see:")
    print("  docs/development/FILE_ORGANIZATION_STANDARDS.md")


def main():
    """Main validation function."""
    print("🔍 Validating file organization...")

    is_valid, violations, suggestions = validate_file_organization()

    print_results(is_valid, violations, suggestions)

    # Save validation report
    report = {
        "is_valid": is_valid,
        "violations": violations,
        "suggestions": suggestions,
        "timestamp": str(Path.cwd()),
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    with open(reports_dir / "file_organization_validation.json", "w") as f:
        json.dump(report, f, indent=2)

    # Exit with error code if validation failed
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

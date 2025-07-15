#!/usr/bin/env python3
"""
Environment Organization Validation Script

This script validates that:
1. All Python environments are in the environments/ directory
2. All environments use dot-prefix naming convention
3. No environment directories exist in the project root
4. Code quality tools are configured to ignore environments
5. Git is properly configured to exclude environments
"""

import re
import sys
from pathlib import Path
from typing import Any


def validate_environment_organization() -> dict[str, Any]:
    """Validate environment organization and return results."""
    results = {"success": True, "errors": [], "warnings": [], "info": []}

    project_root = Path(__file__).parent.parent
    environments_dir = project_root / "environments"

    # Check 1: environments/ directory exists
    if not environments_dir.exists():
        results["errors"].append("environments/ directory does not exist")
        results["success"] = False
        return results

    results["info"].append("✓ environments/ directory exists")

    # Check 2: environments/README.md exists
    readme_path = environments_dir / "README.md"
    if not readme_path.exists():
        results["errors"].append("environments/README.md does not exist")
        results["success"] = False
    else:
        results["info"].append("✓ environments/README.md exists")

    # Check 3: All subdirectories in environments/ use dot-prefix naming
    env_dirs = [d for d in environments_dir.iterdir() if d.is_dir()]
    invalid_names = []
    valid_names = []

    for env_dir in env_dirs:
        if not env_dir.name.startswith("."):
            invalid_names.append(env_dir.name)
        else:
            valid_names.append(env_dir.name)

    if invalid_names:
        results["errors"].append(
            f"Environment directories without dot-prefix: {invalid_names}"
        )
        results["success"] = False

    if valid_names:
        results["info"].append(f"✓ Valid environment names: {valid_names}")

    # Check 4: No environment directories in project root
    root_env_patterns = [
        r"\.?venv.*",
        r"\.?env.*",
        r"test_env.*",
        r".*_env",
        r".*_venv",
    ]

    root_env_dirs = []
    for item in project_root.iterdir():
        if item.is_dir() and item.name != "environments":
            for pattern in root_env_patterns:
                if re.match(pattern, item.name, re.IGNORECASE):
                    root_env_dirs.append(item.name)
                    break

    if root_env_dirs:
        results["warnings"].append(
            f"Environment directories found in root (should be moved): {root_env_dirs}"
        )
    else:
        results["info"].append("✓ No environment directories in project root")

    # Check 5: .gitignore properly configured
    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        gitignore_content = gitignore_path.read_text()

        # Check for environments/ exclusion
        if (
            "environments/" in gitignore_content
            and "!environments/README.md" in gitignore_content
        ):
            results["info"].append(
                "✓ .gitignore properly excludes environments/ directory"
            )
        else:
            results["errors"].append(
                ".gitignore does not properly exclude environments/ directory"
            )
            results["success"] = False
    else:
        results["warnings"].append(".gitignore file not found")

    # Check 6: pyproject.toml tool configurations
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        pyproject_content = pyproject_path.read_text()

        tools_to_check = ["ruff", "black", "isort", "mypy", "bandit"]
        configured_tools = []
        missing_tools = []

        for tool in tools_to_check:
            if (
                f"[tool.{tool}]" in pyproject_content
                and "environments" in pyproject_content
            ):
                configured_tools.append(tool)
            else:
                missing_tools.append(tool)

        if configured_tools:
            results["info"].append(
                f"✓ Tools configured to ignore environments: {configured_tools}"
            )

        if missing_tools:
            results["warnings"].append(
                f"Tools may not be configured to ignore environments: {missing_tools}"
            )
    else:
        results["warnings"].append("pyproject.toml file not found")

    # Check 7: CLAUDE.md contains environment rules
    claude_md_path = project_root / "CLAUDE.md"
    if claude_md_path.exists():
        claude_content = claude_md_path.read_text()

        if (
            "Environment Management Rules" in claude_content
            and "environments/" in claude_content
        ):
            results["info"].append("✓ CLAUDE.md contains environment management rules")
        else:
            results["warnings"].append(
                "CLAUDE.md may not contain complete environment management rules"
            )
    else:
        results["warnings"].append("CLAUDE.md file not found")

    return results


def print_results(results: dict[str, Any]) -> None:
    """Print validation results in a formatted manner."""
    print("🔍 Environment Organization Validation")
    print("=" * 50)

    if results["info"]:
        print("\n✅ Success:")
        for item in results["info"]:
            print(f"   {item}")

    if results["warnings"]:
        print("\n⚠️  Warnings:")
        for item in results["warnings"]:
            print(f"   {item}")

    if results["errors"]:
        print("\n❌ Errors:")
        for item in results["errors"]:
            print(f"   {item}")

    print("\n" + "=" * 50)
    if results["success"]:
        print("🎉 Environment organization validation PASSED!")
        return True
    else:
        print("💥 Environment organization validation FAILED!")
        return False


def main() -> int:
    """Main validation function."""
    try:
        results = validate_environment_organization()
        success = print_results(results)
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

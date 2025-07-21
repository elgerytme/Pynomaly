#!/usr/bin/env python3
"""Analyze project structure and identify stray files."""

import fnmatch
import json
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: str | None = None) -> dict | None:
    """Load configuration from .pyno-org.yaml file."""
    if config_path is None:
        config_path = ".pyno-org.yaml"

    config_file = Path(config_path)
    if not config_file.exists():
        return None

    if yaml is None:
        print("Warning: PyYAML not installed. Using fallback configuration.")
        return None

    try:
        with open(config_file) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return None


def matches_pattern(path: str, patterns: list[str]) -> bool:
    """Check if a path matches any of the given patterns."""
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
            path.lower(), pattern.lower()
        ):
            return True
    return False


def should_delete_item(
    item_path: Path, delete_patterns: list[str], allowlist: list[str]
) -> bool:
    """Determine if an item should be deleted based on patterns and allowlist."""
    relative_path = str(item_path.relative_to(Path.cwd()))

    # For directories, also check with trailing slash
    if item_path.is_dir():
        relative_path_with_slash = relative_path + "/"
    else:
        relative_path_with_slash = relative_path

    # Check if item is in allowlist
    if matches_pattern(relative_path, allowlist) or matches_pattern(
        relative_path_with_slash, allowlist
    ):
        return False

    # Check if item matches delete patterns
    return matches_pattern(relative_path, delete_patterns) or matches_pattern(
        relative_path_with_slash, delete_patterns
    )


def analyze_project_structure(config_path: str | None = None) -> dict:
    """Analyze the project structure and categorize files."""

    project_root = Path.cwd()

    # Load configuration
    config = load_config(config_path)

    # Define what should be in the root directory
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
        # Git
        ".gitignore",
        ".gitattributes",
        # CI/CD
        ".pre-commit-config.yaml",
        # Configuration
        ".pyno-org.yaml",
    }

    # Update allowed files and directories if specified in config
    if config:
        config_allowed_files = config.get("allowed_root_files", [])
        if config_allowed_files:
            ALLOWED_ROOT_FILES.update(config_allowed_files)

    # Define delete and move patterns
    delete_patterns = config.get("delete_patterns", []) if config else []
    allowlist = config.get("allowlist", []) if config else []

    # Define allowed root directories
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
        "environments",
        "baseline_outputs",
        "test_reports",
        ".github",
        ".git",
        "node_modules",
    }

    # Update allowed directories if specified in config
    if config:
        config_allowed_dirs = config.get("allowed_root_directories", [])
        if config_allowed_dirs:
            ALLOWED_ROOT_DIRS.update(config_allowed_dirs)

    # Scan root directory
    root_items = list(project_root.iterdir())

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "total_root_items": len(root_items),
        "stray_files": [],
        "stray_directories": [],
        "allowed_files": [],
        "allowed_directories": [],
        "problematic_items": [],
        "recommendations": [],
    }

    for item in root_items:
        item_name = item.name

        # Skip hidden files/directories for now
        if item_name.startswith(".") and item_name not in {
            ".gitignore",
            ".gitattributes",
            ".github",
            ".git",
        }:
            continue

        if item.is_file():
            if item_name in ALLOWED_ROOT_FILES:
                analysis["allowed_files"].append(
                    {
                        "name": item_name,
                        "size": item.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            item.stat().st_mtime
                        ).isoformat(),
                    }
                )
            else:
                # Categorize stray files
                category = categorize_stray_file(item, config)
                analysis["stray_files"].append(
                    {
                        "name": item_name,
                        "category": category,
                        "size": item.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            item.stat().st_mtime
                        ).isoformat(),
                        "recommended_location": get_recommended_location(
                            item_name, category
                        ),
                    }
                )

        elif item.is_dir():
            if item_name in ALLOWED_ROOT_DIRS:
                analysis["allowed_directories"].append(
                    {
                        "name": item_name,
                        "item_count": len(list(item.iterdir())) if item.exists() else 0,
                    }
                )
            else:
                # Categorize stray directories
                category = categorize_stray_directory(item, config)
                analysis["stray_directories"].append(
                    {
                        "name": item_name,
                        "category": category,
                        "item_count": len(list(item.iterdir())) if item.exists() else 0,
                        "recommended_location": get_recommended_location(
                            item_name, category
                        ),
                    }
                )

    # Generate recommendations
    analysis["recommendations"] = generate_recommendations(analysis)

    return analysis


def categorize_stray_file(file_path: Path, config: dict | None = None) -> str:
    """Categorize a stray file based on its name, extension, and configuration."""
    name = file_path.name.lower()
    relative_path = str(file_path.relative_to(Path.cwd()))

    # Check if it should be deleted based on configuration
    if config:
        delete_patterns = config.get("delete_patterns", [])
        allowlist = config.get("allowlist", [])

        if should_delete_item(file_path, delete_patterns, allowlist):
            return "artifacts_for_deletion"

    # Check configuration move patterns
    if config and "move_patterns" in config:
        move_patterns = config["move_patterns"]
        for category, pattern_info in move_patterns.items():
            patterns = pattern_info.get("patterns", [])
            excludes = pattern_info.get("excludes", [])

            # Check if file matches any pattern
            if matches_pattern(relative_path, patterns):
                # Check if file should be excluded
                if not matches_pattern(relative_path, excludes):
                    return category

    # Fallback to original categorization logic
    # Testing files
    if any(x in name for x in ["test_", "testing", "_test"]):
        return "testing"

    # Documentation
    if name.endswith((".md", ".rst", ".txt")) and any(
        x in name for x in ["readme", "doc", "guide", "manual"]
    ):
        return "documentation"

    # Reports
    if any(x in name for x in ["report", "summary", "analysis"]):
        return "reports"

    # Scripts
    if name.endswith((".py", ".sh", ".ps1", ".bat")) and not name.startswith("test_"):
        return "scripts"

    # Temporary/cache files
    if any(x in name for x in ["temp", "tmp", "cache", "backup"]):
        return "temporary"

    # Build artifacts
    if any(x in name for x in ["build", "dist", ".egg-info"]):
        return "build_artifacts"

    # Configuration
    if (
        name.endswith((".json", ".yaml", ".yml", ".ini", ".toml", ".cfg"))
        and "config" in name
    ):
        return "configuration"

    # Version numbers (like "2.0", "=0.2.0.1")
    if name.replace(".", "").replace("=", "").isdigit() or name.startswith("="):
        return "version_artifacts"

    # PowerShell/Windows specific
    if name.endswith(".ps1"):
        return "scripts"

    return "miscellaneous"


def categorize_stray_directory(dir_path: Path, config: dict | None = None) -> str:
    """Categorize a stray directory based on its name, contents, and configuration."""
    name = dir_path.name.lower()
    relative_path = str(dir_path.relative_to(Path.cwd()))

    # Check if it should be deleted based on configuration
    if config:
        delete_patterns = config.get("delete_patterns", [])
        allowlist = config.get("allowlist", [])

        if should_delete_item(dir_path, delete_patterns, allowlist):
            return "artifacts_for_deletion"

    # Check configuration move patterns
    if config and "move_patterns" in config:
        move_patterns = config["move_patterns"]
        for category, pattern_info in move_patterns.items():
            patterns = pattern_info.get("patterns", [])
            excludes = pattern_info.get("excludes", [])

            # Check if directory matches any pattern
            if matches_pattern(relative_path, patterns):
                # Check if directory should be excluded
                if not matches_pattern(relative_path, excludes):
                    return category

    # Fallback to original categorization logic
    # Testing directories
    if any(x in name for x in ["test_", "testing", "_test"]):
        return "testing"

    # Temporary/environment directories
    if any(x in name for x in ["temp", "tmp", "env", "venv", ".venv"]):
        return "temporary"

    # Specific testing environments
    if "test_env" in name or "test_venv" in name:
        return "testing"

    # Virtual environments
    if name in ["venv", ".venv", "env", ".env"]:
        return "environment"

    # Package installation artifacts
    if name == "node_modules" or ".egg-info" in name:
        return "build_artifacts"

    # Storage/runtime
    if any(x in name for x in ["storage", "data", "logs"]):
        return "runtime_data"

    return "miscellaneous"


def get_recommended_location(item_name: str, category: str) -> str:
    """Get recommended location for a stray item."""

    location_map = {
        "testing": "tests/",
        "documentation": "docs/",
        "reports": "reports/",
        "scripts": "scripts/",
        "temporary": "DELETE (temporary files)",
        "build_artifacts": "DELETE (build artifacts)",
        "version_artifacts": "DELETE (version artifacts)",
        "configuration": "config/",
        "environment": "DELETE (virtual environments)",
        "runtime_data": "storage/",
        "artifacts_for_deletion": "DELETE (artifacts)",
        "miscellaneous": "REVIEW (manual classification needed)",
    }

    return location_map.get(category, "REVIEW")


def generate_recommendations(analysis: dict) -> list[str]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []

    # Count items by category
    file_categories = {}
    dir_categories = {}

    for file_info in analysis["stray_files"]:
        cat = file_info["category"]
        file_categories[cat] = file_categories.get(cat, 0) + 1

    for dir_info in analysis["stray_directories"]:
        cat = dir_info["category"]
        dir_categories[cat] = dir_categories.get(cat, 0) + 1

    # Generate specific recommendations
    if file_categories.get("testing", 0) > 0:
        recommendations.append(
            f"Move {file_categories['testing']} testing files to tests/ directory"
        )

    if file_categories.get("scripts", 0) > 0:
        recommendations.append(
            f"Move {file_categories['scripts']} script files to scripts/ directory"
        )

    if file_categories.get("reports", 0) > 0:
        recommendations.append(
            f"Move {file_categories['reports']} report files to reports/ directory"
        )

    if file_categories.get("temporary", 0) > 0:
        recommendations.append(f"DELETE {file_categories['temporary']} temporary files")

    if file_categories.get("build_artifacts", 0) > 0:
        recommendations.append(
            f"DELETE {file_categories['build_artifacts']} build artifact files"
        )

    if file_categories.get("version_artifacts", 0) > 0:
        recommendations.append(
            f"DELETE {file_categories['version_artifacts']} version artifact files"
        )

    if file_categories.get("artifacts_for_deletion", 0) > 0:
        recommendations.append(
            f"DELETE {file_categories['artifacts_for_deletion']} artifact files (configured for deletion)"
        )

    if dir_categories.get("testing", 0) > 0:
        recommendations.append(
            f"Move/consolidate {dir_categories['testing']} testing directories"
        )

    if dir_categories.get("temporary", 0) > 0:
        recommendations.append(
            f"DELETE {dir_categories['temporary']} temporary directories"
        )

    if dir_categories.get("environment", 0) > 0:
        recommendations.append(
            f"DELETE {dir_categories['environment']} virtual environment directories"
        )

    if dir_categories.get("artifacts_for_deletion", 0) > 0:
        recommendations.append(
            f"DELETE {dir_categories['artifacts_for_deletion']} artifact directories (configured for deletion)"
        )

    # General recommendations
    total_stray = len(analysis["stray_files"]) + len(analysis["stray_directories"])
    if total_stray > 10:
        recommendations.append(
            "Consider implementing automated file organization rules"
        )
        recommendations.append("Add comprehensive .gitignore patterns")
        recommendations.append(
            "Create pre-commit hooks for file organization validation"
        )

    return recommendations


def generate_full_tree_report(project_root: Path) -> dict:
    """Generate a full tree report of all files and directories."""
    full_tree = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "tree": [],
    }

    def scan_directory(directory: Path, relative_path: str = "") -> list:
        items = []
        try:
            for item in sorted(directory.iterdir()):
                # Skip .git directory contents to avoid huge output
                if item.name == ".git":
                    items.append(
                        {
                            "name": item.name,
                            "type": "directory",
                            "path": str(item.relative_to(project_root)),
                            "size": None,
                            "modified": None,
                            "note": "Git repository (contents not scanned)",
                        }
                    )
                    continue

                try:
                    # Safely check file type
                    is_dir = item.is_dir()
                    is_file = item.is_file()
                except (OSError, PermissionError):
                    # Handle broken symlinks, permission issues, etc.
                    items.append(
                        {
                            "name": item.name,
                            "type": "unknown",
                            "path": str(item.relative_to(project_root)),
                            "size": None,
                            "modified": None,
                            "note": "Access denied or broken symlink",
                        }
                    )
                    continue

                # Safely get file stats
                size = None
                modified = None
                try:
                    if is_file:
                        size = item.stat().st_size
                    if item.exists():
                        modified = datetime.fromtimestamp(
                            item.stat().st_mtime
                        ).isoformat()
                except (OSError, PermissionError):
                    pass

                item_info = {
                    "name": item.name,
                    "type": "directory" if is_dir else "file",
                    "path": str(item.relative_to(project_root)),
                    "size": size,
                    "modified": modified,
                }

                if is_dir and item.name != "__pycache__":
                    try:
                        item_info["children"] = scan_directory(item)
                    except (OSError, PermissionError):
                        item_info["note"] = "Directory access denied"
                        item_info["children"] = []

                items.append(item_info)
        except PermissionError:
            pass

        return items

    full_tree["tree"] = scan_directory(project_root)
    return full_tree


def generate_violations_report(analysis: dict) -> dict:
    """Generate a report of only violating/offending paths."""
    violations = {
        "timestamp": datetime.now().isoformat(),
        "project_root": analysis["project_root"],
        "total_violations": len(analysis["stray_files"])
        + len(analysis["stray_directories"]),
        "violations": [],
    }

    # Add stray files as violations
    for file_info in analysis["stray_files"]:
        violations["violations"].append(
            {
                "path": file_info["name"],
                "type": "file",
                "violation_type": "stray_file",
                "category": file_info["category"],
                "rule_violated": f"File should not be in root directory - belongs in {file_info['category']} category",
                "recommended_location": file_info["recommended_location"],
                "size": file_info["size"],
                "modified": file_info["modified"],
            }
        )

    # Add stray directories as violations
    for dir_info in analysis["stray_directories"]:
        violations["violations"].append(
            {
                "path": dir_info["name"],
                "type": "directory",
                "violation_type": "stray_directory",
                "category": dir_info["category"],
                "rule_violated": f"Directory should not be in root directory - belongs in {dir_info['category']} category",
                "recommended_location": dir_info["recommended_location"],
                "item_count": dir_info["item_count"],
            }
        )

    return violations


def main():
    """Main function to run the analysis."""
    print("🔍 Analyzing project structure...")

    project_root = Path.cwd()
    analysis = analyze_project_structure()

    # Create reports directory structure
    reports_dir = Path("reports/structure")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save full tree report
    print("📊 Generating full tree report...")
    full_tree = generate_full_tree_report(project_root)
    current_layout_file = reports_dir / "current_layout.json"
    with open(current_layout_file, "w") as f:
        json.dump(full_tree, f, indent=2)

    # Generate and save violations report
    print("⚠️  Generating violations report...")
    violations = generate_violations_report(analysis)
    violations_file = reports_dir / "violations.json"
    with open(violations_file, "w") as f:
        json.dump(violations, f, indent=2)

    # Save detailed analysis to file (keep original functionality)
    output_file = Path("reports/project_structure_analysis.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    print("\n📊 Project Structure Analysis Summary")
    print("=" * 50)
    print(f"Total root items: {analysis['total_root_items']}")
    print(f"Allowed files: {len(analysis['allowed_files'])}")
    print(f"Allowed directories: {len(analysis['allowed_directories'])}")
    print(f"Stray files: {len(analysis['stray_files'])}")
    print(f"Stray directories: {len(analysis['stray_directories'])}")

    if analysis["stray_files"]:
        print("\n🗂️ Stray Files by Category:")
        categories = {}
        for file_info in analysis["stray_files"]:
            cat = file_info["category"]
            categories[cat] = categories.get(cat, 0) + 1

        for category, count in sorted(categories.items()):
            print(f"  {category}: {count} files")

    if analysis["stray_directories"]:
        print("\n📁 Stray Directories by Category:")
        categories = {}
        for dir_info in analysis["stray_directories"]:
            cat = dir_info["category"]
            categories[cat] = categories.get(cat, 0) + 1

        for category, count in sorted(categories.items()):
            print(f"  {category}: {count} directories")

    if analysis["recommendations"]:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"  {i}. {rec}")

    print("\n📄 Reports generated:")
    print(f"  - Full tree layout: {current_layout_file}")
    print(f"  - Violations report: {violations_file}")
    print(f"  - Detailed analysis: {output_file}")
    print(
        f"\n🎯 Found {violations['total_violations']} violations that need attention."
    )

    return analysis


if __name__ == "__main__":
    main()

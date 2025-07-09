"""
Configuration and constants for structure enforcer.
"""

import re
from pathlib import Path

# Define allowed files in root directory
ALLOWED_ROOT_FILES: set[str] = {
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
ALLOWED_ROOT_DIRS: set[str] = {
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
    "environments",  # Added for current project structure
}

# File patterns that should be moved to specific directories
RELOCATION_RULES: dict[str, list[str]] = {
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

# Define the expected Clean Architecture structure
EXPECTED_STRUCTURE: dict[str, dict] = {
    "domain": {
        "required": True,
        "description": "Pure business logic layer",
        "subdirs": ["entities", "value_objects", "services", "exceptions"],
        "dependencies": [],  # No external dependencies allowed
    },
    "application": {
        "required": True,
        "description": "Application logic layer",
        "subdirs": ["use_cases", "services", "dto"],
        "dependencies": ["domain"],  # Can only depend on domain
    },
    "infrastructure": {
        "required": True,
        "description": "External integrations layer",
        "subdirs": ["adapters", "persistence", "config", "monitoring"],
        "dependencies": ["domain", "application"],  # Can depend on inner layers
    },
    "presentation": {
        "required": True,
        "description": "User interface layer",
        "subdirs": ["api", "cli", "sdk", "web"],
        "dependencies": ["domain", "application", "infrastructure"],
    },
    "shared": {
        "required": False,
        "description": "Shared utilities and protocols",
        "subdirs": ["protocols", "utils"],
        "dependencies": [],  # Utility layer
    },
}

# Define naming conventions
NAMING_CONVENTIONS: dict[str, str] = {
    "modules": r"^[a-z][a-z0-9_]*[a-z0-9]$",
    "classes": r"^[A-Z][a-zA-Z0-9]*$",
    "functions": r"^[a-z][a-z0-9_]*[a-z0-9]$",
    "constants": r"^[A-Z][A-Z0-9_]*[A-Z0-9]$",
    "private": r"^_[a-z][a-z0-9_]*[a-z0-9]$",
}

# Allowed imports for domain layer (to maintain purity)
ALLOWED_DOMAIN_IMPORTS: set[str] = {
    # Python standard library
    "typing",
    "abc",
    "dataclasses",
    "enum",
    "functools",
    "itertools",
    "collections",
    "datetime",
    "uuid",
    "math",
    "re",
    "json",
    "pathlib",
    "logging",
    "time",
    "os",
    "sys",
    "warnings",
    "weakref",
    "copy",
    "decimal",
    "fractions",
    "operator",
    "random",
    "string",
    "textwrap",
    "unicodedata",
    # Special cases
    "__future__",  # Future annotations are allowed
}

# Layer dependency order for validation
LAYER_ORDER: list[str] = ["domain", "application", "infrastructure", "presentation"]

# Help URLs for different violation types
HELP_URLS: dict[str, str] = {
    "file_organization": "https://github.com/pynomaly/pynomaly/blob/main/docs/development/FILE_ORGANIZATION_STANDARDS.md",
    "architecture": "https://github.com/pynomaly/pynomaly/blob/main/docs/development/ARCHITECTURE_GUIDELINES.md",
    "naming": "https://github.com/pynomaly/pynomaly/blob/main/docs/development/NAMING_CONVENTIONS.md",
}


def get_suggested_location(filename: str) -> str:
    """Get suggested location for a file based on patterns."""
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


def is_python_file(path: Path) -> bool:
    """Check if a file is a Python file."""
    return path.suffix.lower() == ".py"


def is_test_file(path: Path) -> bool:
    """Check if a file is a test file."""
    name = path.name.lower()
    return name.startswith("test_") or name.endswith("_test.py") or "test" in path.parts


def is_script_file(path: Path) -> bool:
    """Check if a file is a script file."""
    return path.suffix.lower() in [".py", ".sh", ".ps1", ".bat"]


def is_config_file(path: Path) -> bool:
    """Check if a file is a configuration file."""
    return path.suffix.lower() in [".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"]


def is_docs_file(path: Path) -> bool:
    """Check if a file is a documentation file."""
    return path.suffix.lower() in [".md", ".rst", ".txt"]


def get_file_type_info(path: Path) -> dict[str, bool]:
    """Get file type information."""
    return {
        "is_python": is_python_file(path),
        "is_test": is_test_file(path),
        "is_script": is_script_file(path),
        "is_config": is_config_file(path),
        "is_docs": is_docs_file(path),
    }

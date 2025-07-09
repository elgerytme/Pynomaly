"""
Violations detector implementation.
"""

import ast
import re

from .config import (
    ALLOWED_DOMAIN_IMPORTS,
    ALLOWED_ROOT_DIRS,
    ALLOWED_ROOT_FILES,
    EXPECTED_STRUCTURE,
    LAYER_ORDER,
    NAMING_CONVENTIONS,
)
from .models import Model, Severity, Violation, ViolationType


class ViolationDetector:
    """Detects structure violations in a repository model."""

    def __init__(self, model: Model):
        self.model = model
        self.violations: list[Violation] = []

    def detect(self) -> list[Violation]:
        """Detect all violations in the model."""
        self.violations = []

        # Check file organization violations
        self._check_file_organization()

        # Check architecture violations
        self._check_architecture()

        # Check naming conventions
        self._check_naming_conventions()

        # Check dependency violations
        self._check_dependency_violations()

        # Check domain purity
        self._check_domain_purity()

        return self.violations

    def _check_file_organization(self):
        """Check file organization violations."""
        root_dir = self.model.root_directory

        # Check stray files in root
        for file_node in root_dir.files:
            if file_node.name.startswith(".") and file_node.name not in {
                ".gitignore",
                ".gitattributes",
                ".pre-commit-config.yaml",
            }:
                continue

            if file_node.name not in ALLOWED_ROOT_FILES:
                self.violations.append(
                    Violation(
                        type=ViolationType.STRAY_FILE,
                        severity=Severity.ERROR,
                        message=f"Stray file in root: {file_node.name}",
                        file_path=file_node.path,
                    )
                )

        # Check stray directories in root
        for dir_node in root_dir.subdirectories:
            if dir_node.name.startswith(".") and dir_node.name not in {
                ".git",
                ".github",
            }:
                continue

            if dir_node.name not in ALLOWED_ROOT_DIRS:
                self.violations.append(
                    Violation(
                        type=ViolationType.STRAY_DIRECTORY,
                        severity=Severity.ERROR,
                        message=f"Stray directory in root: {dir_node.name}/",
                        directory_path=dir_node.path,
                    )
                )

    def _check_architecture(self):
        """Check Clean Architecture violations."""
        # Check for required layers
        for layer_name, layer_config in EXPECTED_STRUCTURE.items():
            if layer_config["required"] and layer_name not in self.model.layers:
                self.violations.append(
                    Violation(
                        type=ViolationType.MISSING_LAYER,
                        severity=Severity.ERROR,
                        message=f"Missing required layer: {layer_name}",
                        directory_path=self.model.root_path
                        / "src"
                        / "pynomaly"
                        / layer_name,
                    )
                )

        # Check for missing __init__.py files in layers
        for layer_name, layer_node in self.model.layers.items():
            if not layer_node.has_init_file():
                self.violations.append(
                    Violation(
                        type=ViolationType.MISSING_INIT,
                        severity=Severity.ERROR,
                        message=f"Missing __init__.py in layer: {layer_name}",
                        directory_path=layer_node.path,
                    )
                )

            # Check subdirectories
            layer_config = EXPECTED_STRUCTURE.get(layer_name, {})
            for subdir in layer_node.subdirectories:
                if not subdir.has_init_file():
                    self.violations.append(
                        Violation(
                            type=ViolationType.MISSING_INIT,
                            severity=Severity.ERROR,
                            message=f"Missing __init__.py in: {layer_name}/{subdir.name}",
                            directory_path=subdir.path,
                        )
                    )

                # Check for empty directories
                if len(subdir.files) == 0 and len(subdir.subdirectories) == 0:
                    self.violations.append(
                        Violation(
                            type=ViolationType.EMPTY_DIRECTORY,
                            severity=Severity.WARNING,
                            message=f"Empty directory: {layer_name}/{subdir.name}",
                            directory_path=subdir.path,
                        )
                    )

    def _check_naming_conventions(self):
        """Check naming convention violations."""
        self._check_naming_recursive(self.model.root_directory)

    def _check_naming_recursive(self, dir_node):
        """Recursively check naming conventions."""
        for file_node in dir_node.files:
            if file_node.is_python and file_node.name != "__init__.py":
                # Check module name
                module_name = file_node.path.stem
                if not re.match(NAMING_CONVENTIONS["modules"], module_name):
                    self.violations.append(
                        Violation(
                            type=ViolationType.NAMING_CONVENTION,
                            severity=Severity.WARNING,
                            message=f"Module name violates convention: {module_name}",
                            file_path=file_node.path,
                        )
                    )

                # Check class and function names within the file
                self._check_python_file_naming(file_node)

        # Recursively check subdirectories
        for subdir in dir_node.subdirectories:
            self._check_naming_recursive(subdir)

    def _check_python_file_naming(self, file_node):
        """Check naming conventions within a Python file."""
        if not file_node.is_python:
            return

        try:
            with open(file_node.path, encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not re.match(NAMING_CONVENTIONS["classes"], node.name):
                        self.violations.append(
                            Violation(
                                type=ViolationType.NAMING_CONVENTION,
                                severity=Severity.WARNING,
                                message=f"Class name violates convention: {node.name}",
                                file_path=file_node.path,
                                line_number=node.lineno,
                            )
                        )

                elif isinstance(node, ast.FunctionDef):
                    if node.name.startswith("_"):
                        if not re.match(NAMING_CONVENTIONS["private"], node.name):
                            self.violations.append(
                                Violation(
                                    type=ViolationType.NAMING_CONVENTION,
                                    severity=Severity.WARNING,
                                    message=f"Private function name violates convention: {node.name}",
                                    file_path=file_node.path,
                                    line_number=node.lineno,
                                )
                            )
                    else:
                        if not re.match(NAMING_CONVENTIONS["functions"], node.name):
                            self.violations.append(
                                Violation(
                                    type=ViolationType.NAMING_CONVENTION,
                                    severity=Severity.WARNING,
                                    message=f"Function name violates convention: {node.name}",
                                    file_path=file_node.path,
                                    line_number=node.lineno,
                                )
                            )

        except (SyntaxError, UnicodeDecodeError, OSError):
            # Skip files that can't be parsed
            pass

    def _check_dependency_violations(self):
        """Check dependency direction violations."""
        for i, layer in enumerate(LAYER_ORDER):
            if layer not in self.model.dependencies:
                continue

            allowed_dependencies = set(LAYER_ORDER[: i + 1])  # Current and inner layers
            allowed_dependencies.add("shared")  # Shared is always allowed

            layer_deps = self.model.dependencies[layer]
            for dep in layer_deps:
                if dep not in allowed_dependencies:
                    self.violations.append(
                        Violation(
                            type=ViolationType.INVALID_DEPENDENCY,
                            severity=Severity.ERROR,
                            message=f"Invalid dependency direction: {layer} layer depends on {dep} layer",
                            directory_path=(
                                self.model.layers.get(layer, {}).path
                                if layer in self.model.layers
                                else None
                            ),
                        )
                    )

    def _check_domain_purity(self):
        """Check domain layer purity violations."""
        if "domain" not in self.model.layers:
            return

        domain_layer = self.model.layers["domain"]

        # Check all Python files in domain layer
        for file_node in self._get_python_files_recursive(domain_layer):
            self._check_domain_file_purity(file_node)

    def _get_python_files_recursive(self, dir_node):
        """Get all Python files recursively from a directory node."""
        files = []

        # Add Python files from this directory
        files.extend(dir_node.get_python_files())

        # Recursively add from subdirectories
        for subdir in dir_node.subdirectories:
            files.extend(self._get_python_files_recursive(subdir))

        return files

    def _check_domain_file_purity(self, file_node):
        """Check domain purity for a single file."""
        if not file_node.is_python:
            return

        try:
            with open(file_node.path, encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module not in ALLOWED_DOMAIN_IMPORTS:
                            self.violations.append(
                                Violation(
                                    type=ViolationType.DOMAIN_PURITY,
                                    severity=Severity.ERROR,
                                    message=f"Domain layer imports external dependency: {module}",
                                    file_path=file_node.path,
                                    line_number=node.lineno,
                                )
                            )

                elif isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split(".")[0]
                    if (
                        module not in ALLOWED_DOMAIN_IMPORTS
                        and not node.module.startswith("pynomaly.domain")
                        and not node.module.startswith("pynomaly.shared")
                    ):
                        self.violations.append(
                            Violation(
                                type=ViolationType.DOMAIN_PURITY,
                                severity=Severity.ERROR,
                                message=f"Domain layer imports external dependency: {node.module}",
                                file_path=file_node.path,
                                line_number=node.lineno,
                            )
                        )

        except (SyntaxError, UnicodeDecodeError, OSError):
            # Skip files that can't be parsed
            pass


def detect_violations(model: Model) -> list[Violation]:
    """
    Detect structure violations from the model.

    Args:
        model: Repository structure model.

    Returns:
        List[Violation]: List of detected violations.
    """
    detector = ViolationDetector(model)
    return detector.detect()

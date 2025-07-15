"""Structure checker for documentation validation."""

from pathlib import Path

from ..core.config import ValidationConfig
from ..core.exceptions import ValidationError


class StructureChecker:
    """Checks documentation structure and organization."""

    def __init__(self, config: ValidationConfig):
        """Initialize the structure checker.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.errors: list[ValidationError] = []

    def check_structure(self, files: list[Path]) -> list[ValidationError]:
        """Check structure of documentation files.

        Args:
            files: List of documentation files to check

        Returns:
            List of validation errors found
        """
        self.errors = []

        for file_path in files:
            if file_path.suffix not in [".md", ".rst", ".txt"]:
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check heading structure
                self._check_heading_structure(file_path, content)

                # Check file organization
                self._check_file_organization(file_path, content)

            except Exception as e:
                self.errors.append(
                    ValidationError(
                        f"Error reading file {file_path}: {e}",
                        file_path,
                        severity="error",
                    )
                )

        return self.errors

    def _check_heading_structure(self, file_path: Path, content: str) -> None:
        """Check heading hierarchy and structure."""
        lines = content.split("\n")
        heading_levels = []

        for i, line in enumerate(lines):
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                heading_levels.append((level, i + 1))

                # Check for proper heading hierarchy
                if len(heading_levels) > 1:
                    prev_level = heading_levels[-2][0]
                    if level > prev_level + 1:
                        self.errors.append(
                            ValidationError(
                                f"Heading level skipped: h{prev_level} to h{level}",
                                file_path,
                                line_number=i + 1,
                                severity="warning",
                            )
                        )

    def _check_file_organization(self, file_path: Path, content: str) -> None:
        """Check file organization and structure."""
        # Check for required sections in README files
        if file_path.name.lower() == "readme.md":
            self._check_readme_structure(file_path, content)

        # Check for proper front matter in markdown files
        if file_path.suffix == ".md":
            self._check_front_matter(file_path, content)

    def _check_readme_structure(self, file_path: Path, content: str) -> None:
        """Check README file structure."""
        required_sections = ["installation", "usage", "features"]
        content_lower = content.lower()

        for section in required_sections:
            if section not in content_lower:
                self.errors.append(
                    ValidationError(
                        f"README missing recommended section: {section}",
                        file_path,
                        severity="info",
                    )
                )

    def _check_front_matter(self, file_path: Path, content: str) -> None:
        """Check for front matter in markdown files."""
        # Check if file starts with YAML front matter
        if content.startswith("---"):
            end_index = content.find("---", 3)
            if end_index == -1:
                self.errors.append(
                    ValidationError(
                        "Malformed YAML front matter", file_path, severity="warning"
                    )
                )

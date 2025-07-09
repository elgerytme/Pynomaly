"""Consistency checker for documentation validation."""

from pathlib import Path

from ..core.config import ValidationConfig
from ..core.exceptions import ValidationError


class ConsistencyChecker:
    """Checks consistency across documentation files."""

    def __init__(self, config: ValidationConfig):
        """Initialize the consistency checker.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.errors: list[ValidationError] = []

    def check_consistency(self, files: list[Path]) -> list[ValidationError]:
        """Check consistency across documentation files.

        Args:
            files: List of documentation files to check

        Returns:
            List of validation errors found
        """
        self.errors = []

        # Check for consistent formatting
        self._check_formatting_consistency(files)

        # Check for consistent terminology
        self._check_terminology_consistency(files)

        # Check for consistent structure
        self._check_structure_consistency(files)

        return self.errors

    def _check_formatting_consistency(self, files: list[Path]) -> None:
        """Check for consistent formatting across files."""
        # Basic formatting consistency checks
        for file_path in files:
            if file_path.suffix not in [".md", ".rst", ".txt"]:
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for consistent heading styles
                self._check_heading_consistency(file_path, content)

            except Exception as e:
                self.errors.append(
                    ValidationError(
                        f"Error reading file {file_path}: {e}",
                        file_path,
                        severity="error",
                    )
                )

    def _check_terminology_consistency(self, files: list[Path]) -> None:
        """Check for consistent terminology usage."""
        # Placeholder for terminology consistency checks
        pass

    def _check_structure_consistency(self, files: list[Path]) -> None:
        """Check for consistent structure across files."""
        # Placeholder for structure consistency checks
        pass

    def _check_heading_consistency(self, file_path: Path, content: str) -> None:
        """Check heading consistency in a file."""
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if line.startswith("#"):
                # Check for proper heading format
                if not line.startswith("# ") and len(line) > 1:
                    self.errors.append(
                        ValidationError(
                            f"Heading should have space after #: {line}",
                            file_path,
                            line_number=i + 1,
                            severity="warning",
                        )
                    )

"""Content checker for documentation validation."""

from pathlib import Path

from ..core.config import ValidationConfig
from ..core.exceptions import ValidationError


class ContentChecker:
    """Checks content quality and accuracy in documentation."""

    def __init__(self, config: ValidationConfig):
        """Initialize the content checker.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.errors: list[ValidationError] = []

    def check_content(self, files: list[Path]) -> list[ValidationError]:
        """Check content quality across documentation files.

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

                # Check content quality
                self._check_content_quality(file_path, content)

                # Check for common issues
                self._check_common_issues(file_path, content)

            except Exception as e:
                self.errors.append(
                    ValidationError(
                        f"Error reading file {file_path}: {e}",
                        file_path,
                        severity="error",
                    )
                )

        return self.errors

    def _check_content_quality(self, file_path: Path, content: str) -> None:
        """Check content quality indicators."""
        lines = content.split("\n")

        # Check for empty sections
        for i, line in enumerate(lines):
            if line.startswith("#") and i + 1 < len(lines):
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                if not next_line.strip():
                    # Check if next non-empty line is another heading
                    j = i + 2
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and lines[j].startswith("#"):
                        self.errors.append(
                            ValidationError(
                                f"Empty section found: {line}",
                                file_path,
                                line_number=i + 1,
                                severity="warning",
                            )
                        )

    def _check_common_issues(self, file_path: Path, content: str) -> None:
        """Check for common content issues."""
        lines = content.split("\n")

        for i, line in enumerate(lines):
            # Check for TODO/FIXME comments
            if "TODO" in line or "FIXME" in line:
                self.errors.append(
                    ValidationError(
                        f"TODO/FIXME found: {line.strip()}",
                        file_path,
                        line_number=i + 1,
                        severity="info",
                    )
                )

            # Check for placeholder text
            if "[placeholder]" in line.lower() or "lorem ipsum" in line.lower():
                self.errors.append(
                    ValidationError(
                        f"Placeholder text found: {line.strip()}",
                        file_path,
                        line_number=i + 1,
                        severity="warning",
                    )
                )

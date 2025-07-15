"""Link checker for documentation validation."""

import re
import urllib.parse
from pathlib import Path

from ..core.config import ValidationConfig
from ..core.exceptions import ValidationError


class LinkChecker:
    """Checks links in documentation files."""

    def __init__(self, config: ValidationConfig):
        """Initialize the link checker.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.errors: list[ValidationError] = []

    def check_links(self, files: list[Path]) -> list[ValidationError]:
        """Check links in documentation files.

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

                # Check markdown links
                self._check_markdown_links(file_path, content)

                # Check internal references
                self._check_internal_references(file_path, content, files)

            except Exception as e:
                self.errors.append(
                    ValidationError(
                        f"Error reading file {file_path}: {e}",
                        file_path,
                        severity="error",
                    )
                )

        return self.errors

    def _check_markdown_links(self, file_path: Path, content: str) -> None:
        """Check markdown links in content."""
        # Find all markdown links [text](url)
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

        for match in re.finditer(link_pattern, content):
            text = match.group(1)
            url = match.group(2)

            # Check for empty links
            if not url.strip():
                self.errors.append(
                    ValidationError(
                        f"Empty link found: [{text}]()", file_path, severity="warning"
                    )
                )

            # Check for malformed URLs
            if url.startswith("http") and not self._is_valid_url(url):
                self.errors.append(
                    ValidationError(
                        f"Malformed URL: {url}", file_path, severity="warning"
                    )
                )

    def _check_internal_references(
        self, file_path: Path, content: str, all_files: list[Path]
    ) -> None:
        """Check internal file references."""
        # Find relative file references
        ref_pattern = r"\[([^\]]+)\]\(([^)]+\.md)\)"

        for match in re.finditer(ref_pattern, content):
            text = match.group(1)
            ref_path = match.group(2)

            # Resolve relative path
            if not ref_path.startswith("/"):
                full_path = file_path.parent / ref_path
                full_path = full_path.resolve()

                # Check if referenced file exists
                if not full_path.exists():
                    self.errors.append(
                        ValidationError(
                            f"Broken internal reference: {ref_path}",
                            file_path,
                            severity="error",
                        )
                    )

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

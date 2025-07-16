"""Semantic version value object for model versioning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SemanticVersion:
    """Semantic versioning for models following semver.org specification.

    Provides version numbering in the format MAJOR.MINOR.PATCH where:
    - MAJOR: Incompatible API changes or breaking changes
    - MINOR: New functionality in a backwards compatible manner
    - PATCH: Backwards compatible bug fixes

    Attributes:
        major: Major version number for breaking changes
        minor: Minor version number for new features
        patch: Patch version number for bug fixes
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        """Validate version numbers."""
        if not isinstance(self.major, int) or self.major < 0:
            raise ValueError(
                f"Major version must be non-negative integer, got {self.major}"
            )

        if not isinstance(self.minor, int) or self.minor < 0:
            raise ValueError(
                f"Minor version must be non-negative integer, got {self.minor}"
            )

        if not isinstance(self.patch, int) or self.patch < 0:
            raise ValueError(
                f"Patch version must be non-negative integer, got {self.patch}"
            )

    @property
    def version_string(self) -> str:
        """Get version as string in format MAJOR.MINOR.PATCH."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_str: str) -> SemanticVersion:
        """Create SemanticVersion from string.

        Args:
            version_str: Version string in format "MAJOR.MINOR.PATCH"

        Returns:
            SemanticVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        if not isinstance(version_str, str):
            raise ValueError(f"Version string must be str, got {type(version_str)}")

        # Remove 'v' prefix if present
        version_str = version_str.lstrip("v")

        # Match semantic version pattern
        pattern = r"^(\d+)\.(\d+)\.(\d+)$"
        match = re.match(pattern, version_str)

        if not match:
            raise ValueError(
                f"Invalid version string '{version_str}'. "
                f"Expected format: MAJOR.MINOR.PATCH (e.g., '1.2.3')"
            )

        major, minor, patch = map(int, match.groups())
        return cls(major=major, minor=minor, patch=patch)

    @classmethod
    def initial(cls) -> SemanticVersion:
        """Create initial version 0.1.0."""
        return cls(major=0, minor=1, patch=0)

    @classmethod
    def stable_initial(cls) -> SemanticVersion:
        """Create initial stable version 1.0.0."""
        return cls(major=1, minor=0, patch=0)

    def increment_major(self) -> SemanticVersion:
        """Increment major version and reset minor and patch to 0."""
        return SemanticVersion(major=self.major + 1, minor=0, patch=0)

    def increment_minor(self) -> SemanticVersion:
        """Increment minor version and reset patch to 0."""
        return SemanticVersion(major=self.major, minor=self.minor + 1, patch=0)

    def increment_patch(self) -> SemanticVersion:
        """Increment patch version."""
        return SemanticVersion(major=self.major, minor=self.minor, patch=self.patch + 1)

    def is_compatible_with(self, other: SemanticVersion) -> bool:
        """Check if this version can be used as a drop-in replacement for another.

        Args:
            other: Version to check compatibility with

        Returns:
            True if this version can replace other without breaking compatibility
        """
        if not isinstance(other, SemanticVersion):
            raise TypeError(f"Can only compare with SemanticVersion, got {type(other)}")

        # Different major versions are not compatible
        if self.major != other.major:
            return False

        # Same major version - check if this version can be safely downgraded to other
        if self.minor == other.minor:
            # Same minor version - can downgrade if this patch is newer or equal
            return self.patch >= other.patch
        else:
            # Different minor versions are not compatible
            return False

    def is_newer_than(self, other: SemanticVersion) -> bool:
        """Check if this version is newer than another.

        Args:
            other: Version to compare with

        Returns:
            True if this version is newer
        """
        if not isinstance(other, SemanticVersion):
            raise TypeError(f"Can only compare with SemanticVersion, got {type(other)}")

        if self.major > other.major:
            return True
        elif self.major == other.major:
            if self.minor > other.minor:
                return True
            elif self.minor == other.minor:
                return self.patch > other.patch

        return False

    def is_prerelease(self) -> bool:
        """Check if this is a pre-release version (major version 0)."""
        return self.major == 0

    def is_stable(self) -> bool:
        """Check if this is a stable version (major version >= 1)."""
        return self.major >= 1

    def distance_from(self, other: SemanticVersion) -> int:
        """Calculate version distance for comparison.

        Args:
            other: Version to calculate distance from

        Returns:
            Positive integer representing version distance
        """
        if not isinstance(other, SemanticVersion):
            raise TypeError(f"Can only compare with SemanticVersion, got {type(other)}")

        # Weight different version components
        major_weight = 1000000
        minor_weight = 1000
        patch_weight = 1

        self_score = (
            self.major * major_weight
            + self.minor * minor_weight
            + self.patch * patch_weight
        )

        other_score = (
            other.major * major_weight
            + other.minor * minor_weight
            + other.patch * patch_weight
        )

        return abs(self_score - other_score)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "version_string": self.version_string,
            "is_prerelease": self.is_prerelease(),
            "is_stable": self.is_stable(),
        }

    def __str__(self) -> str:
        """String representation."""
        return self.version_string

    def __lt__(self, other: SemanticVersion) -> bool:
        """Less than comparison."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return other.is_newer_than(self)

    def __le__(self, other: SemanticVersion) -> bool:
        """Less than or equal comparison."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self == other or other.is_newer_than(self)

    def __gt__(self, other: SemanticVersion) -> bool:
        """Greater than comparison."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self.is_newer_than(other)

    def __ge__(self, other: SemanticVersion) -> bool:
        """Greater than or equal comparison."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self == other or self.is_newer_than(other)

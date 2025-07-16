"""
Version Number Value Object

Represents semantic version numbers for domain entities with increment operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VersionNumber:
    """
    Value object representing a semantic version number.
    
    Supports semantic versioning format (MAJOR.MINOR.PATCH) with validation
    and increment operations.
    """
    
    value: str
    
    def __post_init__(self) -> None:
        """Validate the version number format."""
        if not self._is_valid_version(self.value):
            raise ValueError(f"Invalid version number format: {self.value}")
    
    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Check if version follows semantic versioning pattern."""
        pattern = r'^(\d+)\.(\d+)\.(\d+)$'
        return bool(re.match(pattern, version))
    
    @property
    def major(self) -> int:
        """Get major version number."""
        return int(self.value.split('.')[0])
    
    @property
    def minor(self) -> int:
        """Get minor version number."""
        return int(self.value.split('.')[1])
    
    @property
    def patch(self) -> int:
        """Get patch version number."""
        return int(self.value.split('.')[2])
    
    def increment(self, version_type: str = "patch") -> VersionNumber:
        """
        Create a new version with incremented number.
        
        Args:
            version_type: One of "major", "minor", or "patch"
            
        Returns:
            New VersionNumber with incremented version
        """
        major, minor, patch = self.major, self.minor, self.patch
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        elif version_type == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid version type: {version_type}")
        
        return VersionNumber(f"{major}.{minor}.{patch}")
    
    def compare(self, other: VersionNumber) -> int:
        """
        Compare with another version number.
        
        Returns:
            -1 if self < other, 0 if equal, 1 if self > other
        """
        if (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch):
            return -1
        elif (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch):
            return 1
        else:
            return 0
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if not isinstance(other, VersionNumber):
            return False
        return self.value == other.value
    
    def __lt__(self, other: VersionNumber) -> bool:
        """Less than comparison."""
        return self.compare(other) < 0
    
    def __le__(self, other: VersionNumber) -> bool:
        """Less than or equal comparison."""
        return self.compare(other) <= 0
    
    def __gt__(self, other: VersionNumber) -> bool:
        """Greater than comparison."""
        return self.compare(other) > 0
    
    def __ge__(self, other: VersionNumber) -> bool:
        """Greater than or equal comparison."""
        return self.compare(other) >= 0
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)
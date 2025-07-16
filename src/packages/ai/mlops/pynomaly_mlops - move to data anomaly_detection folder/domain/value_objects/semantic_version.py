"""Semantic Version Value Object

Immutable value object representing semantic versioning (SemVer).
"""

import re
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class SemanticVersion:
    """Semantic version value object following SemVer specification.
    
    Represents version in format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
    """
    
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    # SemVer regex pattern
    _SEMVER_PATTERN = re.compile(
        r"^(?P<major>0|[1-9]\d*)\."
        r"(?P<minor>0|[1-9]\d*)\."
        r"(?P<patch>0|[1-9]\d*)"
        r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))"
        r"?(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.major < 0:
            raise ValueError("Major version cannot be negative")
        if self.minor < 0:
            raise ValueError("Minor version cannot be negative")
        if self.patch < 0:
            raise ValueError("Patch version cannot be negative")
        
        if self.prerelease is not None and not self.prerelease:
            raise ValueError("Prerelease cannot be empty string")
        
        if self.build is not None and not self.build:
            raise ValueError("Build cannot be empty string")
    
    @classmethod
    def from_string(cls, version_string: str) -> "SemanticVersion":
        """Create SemanticVersion from string representation.
        
        Args:
            version_string: Version string in SemVer format
            
        Returns:
            SemanticVersion instance
            
        Raises:
            ValueError: If version string is invalid
        """
        match = cls._SEMVER_PATTERN.match(version_string)
        if not match:
            raise ValueError(f"Invalid semantic version: {version_string}")
        
        groups = match.groupdict()
        return cls(
            major=int(groups["major"]),
            minor=int(groups["minor"]),
            patch=int(groups["patch"]),
            prerelease=groups.get("prerelease"),
            build=groups.get("build"),
        )
    
    def __str__(self) -> str:
        """String representation of semantic version."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.prerelease:
            version += f"-{self.prerelease}"
        
        if self.build:
            version += f"+{self.build}"
        
        return version
    
    def __repr__(self) -> str:
        """Developer representation of semantic version."""
        return f"SemanticVersion('{str(self)}')"
    
    def __lt__(self, other: "SemanticVersion") -> bool:
        """Less than comparison."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        
        # Compare core version numbers
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        # Handle prerelease comparison
        if self.prerelease is None and other.prerelease is not None:
            return False  # Release version is higher than prerelease
        if self.prerelease is not None and other.prerelease is None:
            return True   # Prerelease is lower than release
        if self.prerelease is not None and other.prerelease is not None:
            return self._compare_prerelease(self.prerelease, other.prerelease) < 0
        
        return False  # Versions are equal
    
    def __le__(self, other: "SemanticVersion") -> bool:
        """Less than or equal comparison."""
        return self < other or self == other
    
    def __gt__(self, other: "SemanticVersion") -> bool:
        """Greater than comparison."""
        return not self <= other
    
    def __ge__(self, other: "SemanticVersion") -> bool:
        """Greater than or equal comparison."""
        return not self < other
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison (build metadata is ignored)."""
        if not isinstance(other, SemanticVersion):
            return False
        
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.prerelease == other.prerelease
        )
    
    def __hash__(self) -> int:
        """Hash function (build metadata is ignored)."""
        return hash((self.major, self.minor, self.patch, self.prerelease))
    
    def _compare_prerelease(self, pre1: str, pre2: str) -> int:
        """Compare prerelease versions.
        
        Returns:
            -1 if pre1 < pre2, 0 if equal, 1 if pre1 > pre2
        """
        parts1 = pre1.split(".")
        parts2 = pre2.split(".")
        
        # Compare each part
        for i in range(max(len(parts1), len(parts2))):
            part1 = parts1[i] if i < len(parts1) else None
            part2 = parts2[i] if i < len(parts2) else None
            
            if part1 is None:
                return -1  # Shorter prerelease is less
            if part2 is None:
                return 1   # Longer prerelease is greater
            
            # Try to compare as integers first
            try:
                num1 = int(part1)
                try:
                    num2 = int(part2)
                    if num1 != num2:
                        return -1 if num1 < num2 else 1
                except ValueError:
                    return -1  # Numeric < alphabetic
            except ValueError:
                try:
                    int(part2)
                    return 1   # Alphabetic > numeric
                except ValueError:
                    # Both are alphabetic
                    if part1 != part2:
                        return -1 if part1 < part2 else 1
        
        return 0  # Equal
    
    def bump_major(self) -> "SemanticVersion":
        """Create new version with incremented major version."""
        return SemanticVersion(
            major=self.major + 1,
            minor=0,
            patch=0,
            build=self.build
        )
    
    def bump_minor(self) -> "SemanticVersion":
        """Create new version with incremented minor version."""
        return SemanticVersion(
            major=self.major,
            minor=self.minor + 1,
            patch=0,
            build=self.build
        )
    
    def bump_patch(self) -> "SemanticVersion":
        """Create new version with incremented patch version."""
        return SemanticVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch + 1,
            build=self.build
        )
    
    def with_prerelease(self, prerelease: str) -> "SemanticVersion":
        """Create new version with specified prerelease."""
        return SemanticVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            prerelease=prerelease,
            build=self.build
        )
    
    def with_build(self, build: str) -> "SemanticVersion":
        """Create new version with specified build metadata."""
        return SemanticVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            prerelease=self.prerelease,
            build=build
        )
    
    def without_prerelease(self) -> "SemanticVersion":
        """Create new version without prerelease."""
        return SemanticVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            build=self.build
        )
    
    def without_build(self) -> "SemanticVersion":
        """Create new version without build metadata."""
        return SemanticVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            prerelease=self.prerelease
        )
    
    @property
    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version."""
        return self.prerelease is not None
    
    @property
    def is_stable(self) -> bool:
        """Check if this is a stable release version."""
        return self.prerelease is None
    
    @property
    def core_version(self) -> str:
        """Get core version string without prerelease/build."""
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """Check if this version is compatible with another (same major version).
        
        Args:
            other: Version to check compatibility with
            
        Returns:
            True if compatible (same major version)
        """
        return self.major == other.major
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "prerelease": self.prerelease,
            "build": self.build,
            "version_string": str(self),
            "is_prerelease": self.is_prerelease,
            "is_stable": self.is_stable,
            "core_version": self.core_version,
        }
"""Schema versioning utilities.

This module provides utilities for managing schema versions, including version
comparison, migration, and compatibility checks.

Classes:
    SchemaVersion: Represents a schema version using semantic versioning.

Functions:
    compare_versions: Compare two schema versions.
    migrate_schema: Migrate one schema version to another.
    is_compatible_version: Check if two schema versions are compatible.
"""

from __future__ import annotations

from typing import Union
import re

class SchemaVersion:
    """Represents a schema version using semantic versioning."""

    MAJOR: int
    MINOR: int
    PATCH: int

    def __init__(self, version_str: str):
        """Initialize the schema version from a string."""
        pattern = r"^(\d+)\.(\d+)\.(\d+)$"
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")

        self.MAJOR, self.MINOR, self.PATCH = map(int, match.groups())

    def __lt__(self, other: SchemaVersion) -> bool:
        return (self.MAJOR, self.MINOR, self.PATCH) < (other.MAJOR, other.MINOR, other.PATCH)

    def __le__(self, other: SchemaVersion) -> bool:
        return (self.MAJOR, self.MINOR, self.PATCH) <= (other.MAJOR, other.MINOR, other.PATCH)

    def __eq__(self, other: SchemaVersion) -> bool:
        return (self.MAJOR, self.MINOR, self.PATCH) == (other.MAJOR, other.MINOR, other.PATCH)

    def __str__(self) -> str:
        return f"{self.MAJOR}.{self.MINOR}.{self.PATCH}"

    def bump_major(self) -> SchemaVersion:
        return SchemaVersion(f"{self.MAJOR + 1}.0.0")

    def bump_minor(self) -> SchemaVersion:
        return SchemaVersion(f"{self.MAJOR}.{self.MINOR + 1}.0")

    def bump_patch(self) -> SchemaVersion:
        return SchemaVersion(f"{self.MAJOR}.{self.MINOR}.{self.PATCH + 1}")


def compare_versions(version_a: Union[str, SchemaVersion], version_b: Union[str, SchemaVersion]) -> int:
    """Compare two schema versions.

    Returns:
        -1 if version_a < version_b
         0 if version_a == version_b
         1 if version_a > version_b
    """
    if isinstance(version_a, str):
        version_a = SchemaVersion(version_a)
    if isinstance(version_b, str):
        version_b = SchemaVersion(version_b)

    if version_a == version_b:
        return 0
    elif version_a < version_b:
        return -1
    else:
        return 1


def migrate_schema(old_version: str, new_version: str) -> None:
    """Migrate schema from old_version to new_version.

    Raises:
        NotImplementedError: Migration logic must be implemented by schema authors.
    """
    # Placeholders for actual migration logic
    raise NotImplementedError(f"Migration from {old_version} to {new_version} not implemented.")


def get_schema_version(schema_class: type) -> str:
    """Get the schema version from a schema class.
    
    Args:
        schema_class: The schema class to get version from
        
    Returns:
        Schema version string
    """
    # Look for version in class attributes or metadata
    if hasattr(schema_class, '__version__'):
        return schema_class.__version__
    elif hasattr(schema_class, 'Config') and hasattr(schema_class.Config, 'version'):
        return schema_class.Config.version
    else:
        # Default to 1.0.0 if no version specified
        return "1.0.0"


def is_compatible_version(version_a: str, version_b: str) -> bool:
    """Check if two schema versions are compatible.

    Compatibility is defined as having the same MAJOR version.
    """
    schema_version_a = SchemaVersion(version_a)
    schema_version_b = SchemaVersion(version_b)
    
    return schema_version_a.MAJOR == schema_version_b.MAJOR


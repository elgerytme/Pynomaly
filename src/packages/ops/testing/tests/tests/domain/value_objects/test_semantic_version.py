"""
Comprehensive tests for SemanticVersion value object.

This module tests the SemanticVersion value object to ensure proper validation,
behavior, and immutability across all use cases.
"""

import pytest

from pynomaly.domain.value_objects import SemanticVersion


class TestSemanticVersionCreation:
    """Test SemanticVersion creation and validation."""

    def test_basic_creation(self):
        """Test basic semantic version creation."""
        version = SemanticVersion(major=1, minor=2, patch=3)

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_zero_versions(self):
        """Test creation with zero values."""
        version = SemanticVersion(major=0, minor=0, patch=0)

        assert version.major == 0
        assert version.minor == 0
        assert version.patch == 0

    def test_large_version_numbers(self):
        """Test creation with large version numbers."""
        version = SemanticVersion(major=999, minor=999, patch=999)

        assert version.major == 999
        assert version.minor == 999
        assert version.patch == 999


class TestSemanticVersionValidation:
    """Test validation of SemanticVersion parameters."""

    def test_negative_major_version(self):
        """Test that negative major version raises error."""
        with pytest.raises(
            ValueError, match="Major version must be non-negative integer"
        ):
            SemanticVersion(major=-1, minor=0, patch=0)

    def test_negative_minor_version(self):
        """Test that negative minor version raises error."""
        with pytest.raises(
            ValueError, match="Minor version must be non-negative integer"
        ):
            SemanticVersion(major=0, minor=-1, patch=0)

    def test_negative_patch_version(self):
        """Test that negative patch version raises error."""
        with pytest.raises(
            ValueError, match="Patch version must be non-negative integer"
        ):
            SemanticVersion(major=0, minor=0, patch=-1)

    def test_non_integer_major_version(self):
        """Test that non-integer major version raises error."""
        invalid_values = [1.5, "1", None, [1], {1: 1}]

        for value in invalid_values:
            with pytest.raises(
                ValueError, match="Major version must be non-negative integer"
            ):
                SemanticVersion(major=value, minor=0, patch=0)  # type: ignore

    def test_non_integer_minor_version(self):
        """Test that non-integer minor version raises error."""
        invalid_values = [1.5, "1", None, [1], {1: 1}]

        for value in invalid_values:
            with pytest.raises(
                ValueError, match="Minor version must be non-negative integer"
            ):
                SemanticVersion(major=0, minor=value, patch=0)  # type: ignore

    def test_non_integer_patch_version(self):
        """Test that non-integer patch version raises error."""
        invalid_values = [1.5, "1", None, [1], {1: 1}]

        for value in invalid_values:
            with pytest.raises(
                ValueError, match="Patch version must be non-negative integer"
            ):
                SemanticVersion(major=0, minor=0, patch=value)  # type: ignore


class TestSemanticVersionProperties:
    """Test SemanticVersion properties and methods."""

    def test_version_string_property(self):
        """Test version_string property."""
        test_cases = [
            (1, 2, 3, "1.2.3"),
            (0, 0, 1, "0.0.1"),
            (10, 20, 30, "10.20.30"),
            (999, 0, 1, "999.0.1"),
        ]

        for major, minor, patch, expected in test_cases:
            version = SemanticVersion(major=major, minor=minor, patch=patch)
            assert version.version_string == expected

    def test_is_prerelease(self):
        """Test is_prerelease method."""
        prerelease_versions = [
            SemanticVersion(0, 1, 0),
            SemanticVersion(0, 0, 1),
            SemanticVersion(0, 9, 9),
        ]

        for version in prerelease_versions:
            assert version.is_prerelease() is True

        stable_versions = [
            SemanticVersion(1, 0, 0),
            SemanticVersion(1, 2, 3),
            SemanticVersion(2, 0, 0),
        ]

        for version in stable_versions:
            assert version.is_prerelease() is False

    def test_is_stable(self):
        """Test is_stable method."""
        stable_versions = [
            SemanticVersion(1, 0, 0),
            SemanticVersion(1, 2, 3),
            SemanticVersion(2, 0, 0),
            SemanticVersion(10, 5, 2),
        ]

        for version in stable_versions:
            assert version.is_stable() is True

        prerelease_versions = [
            SemanticVersion(0, 1, 0),
            SemanticVersion(0, 0, 1),
            SemanticVersion(0, 9, 9),
        ]

        for version in prerelease_versions:
            assert version.is_stable() is False


class TestSemanticVersionFromString:
    """Test SemanticVersion.from_string method."""

    def test_valid_version_strings(self):
        """Test valid version strings."""
        test_cases = [
            ("1.2.3", 1, 2, 3),
            ("0.0.1", 0, 0, 1),
            ("10.20.30", 10, 20, 30),
            ("999.0.1", 999, 0, 1),
        ]

        for version_str, major, minor, patch in test_cases:
            version = SemanticVersion.from_string(version_str)
            assert version.major == major
            assert version.minor == minor
            assert version.patch == patch

    def test_version_string_with_v_prefix(self):
        """Test version strings with 'v' prefix."""
        test_cases = [
            ("v1.2.3", 1, 2, 3),
            ("v0.0.1", 0, 0, 1),
            ("v10.20.30", 10, 20, 30),
        ]

        for version_str, major, minor, patch in test_cases:
            version = SemanticVersion.from_string(version_str)
            assert version.major == major
            assert version.minor == minor
            assert version.patch == patch

    def test_invalid_version_strings(self):
        """Test invalid version strings."""
        invalid_strings = [
            "1.2",  # Missing patch
            "1.2.3.4",  # Too many components
            "1.2.x",  # Non-numeric patch
            "a.b.c",  # All non-numeric
            "1.2.3-alpha",  # Pre-release suffix
            "1.2.3+build",  # Build metadata
            "",  # Empty string
            "v",  # Just prefix
            "1",  # Single number
            "1.2.3.4.5",  # Too many components
        ]

        for invalid_str in invalid_strings:
            with pytest.raises(ValueError, match="Invalid version string"):
                SemanticVersion.from_string(invalid_str)

    def test_non_string_input(self):
        """Test non-string input to from_string."""
        invalid_inputs = [123, None, [], {}, 1.23]

        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError, match="Version string must be str"):
                SemanticVersion.from_string(invalid_input)  # type: ignore


class TestSemanticVersionClassMethods:
    """Test SemanticVersion class methods."""

    def test_initial_method(self):
        """Test initial class method."""
        version = SemanticVersion.initial()

        assert version.major == 0
        assert version.minor == 1
        assert version.patch == 0
        assert version.is_prerelease() is True

    def test_stable_initial_method(self):
        """Test stable_initial class method."""
        version = SemanticVersion.stable_initial()

        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.is_stable() is True


class TestSemanticVersionIncrement:
    """Test SemanticVersion increment methods."""

    def test_increment_patch(self):
        """Test increment_patch method."""
        version = SemanticVersion(1, 2, 3)
        new_version = version.increment_patch()

        assert new_version.major == 1
        assert new_version.minor == 2
        assert new_version.patch == 4

        # Original version unchanged
        assert version.patch == 3

    def test_increment_minor(self):
        """Test increment_minor method."""
        version = SemanticVersion(1, 2, 3)
        new_version = version.increment_minor()

        assert new_version.major == 1
        assert new_version.minor == 3
        assert new_version.patch == 0  # Reset to 0

        # Original version unchanged
        assert version.minor == 2
        assert version.patch == 3

    def test_increment_major(self):
        """Test increment_major method."""
        version = SemanticVersion(1, 2, 3)
        new_version = version.increment_major()

        assert new_version.major == 2
        assert new_version.minor == 0  # Reset to 0
        assert new_version.patch == 0  # Reset to 0

        # Original version unchanged
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_increment_from_zero(self):
        """Test incrementing from zero versions."""
        version = SemanticVersion(0, 0, 0)

        # Increment patch
        patch_incremented = version.increment_patch()
        assert patch_incremented == SemanticVersion(0, 0, 1)

        # Increment minor
        minor_incremented = version.increment_minor()
        assert minor_incremented == SemanticVersion(0, 1, 0)

        # Increment major
        major_incremented = version.increment_major()
        assert major_incremented == SemanticVersion(1, 0, 0)


class TestSemanticVersionComparison:
    """Test SemanticVersion comparison operations."""

    def test_equality(self):
        """Test equality comparison."""
        version1 = SemanticVersion(1, 2, 3)
        version2 = SemanticVersion(1, 2, 3)
        version3 = SemanticVersion(1, 2, 4)

        assert version1 == version2
        assert version1 != version3

    def test_less_than(self):
        """Test less than comparison."""
        test_cases = [
            (SemanticVersion(1, 0, 0), SemanticVersion(2, 0, 0)),  # Major difference
            (SemanticVersion(1, 1, 0), SemanticVersion(1, 2, 0)),  # Minor difference
            (SemanticVersion(1, 1, 1), SemanticVersion(1, 1, 2)),  # Patch difference
            (
                SemanticVersion(0, 9, 9),
                SemanticVersion(1, 0, 0),
            ),  # Major trumps minor/patch
        ]

        for smaller, larger in test_cases:
            assert smaller < larger
            assert not (larger < smaller)

    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        version1 = SemanticVersion(1, 2, 3)
        version2 = SemanticVersion(1, 2, 3)
        version3 = SemanticVersion(1, 2, 4)

        assert version1 <= version2  # Equal
        assert version1 <= version3  # Less than
        assert not (version3 <= version1)

    def test_greater_than(self):
        """Test greater than comparison."""
        test_cases = [
            (SemanticVersion(2, 0, 0), SemanticVersion(1, 0, 0)),  # Major difference
            (SemanticVersion(1, 2, 0), SemanticVersion(1, 1, 0)),  # Minor difference
            (SemanticVersion(1, 1, 2), SemanticVersion(1, 1, 1)),  # Patch difference
            (
                SemanticVersion(1, 0, 0),
                SemanticVersion(0, 9, 9),
            ),  # Major trumps minor/patch
        ]

        for larger, smaller in test_cases:
            assert larger > smaller
            assert not (smaller > larger)

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        version1 = SemanticVersion(1, 2, 3)
        version2 = SemanticVersion(1, 2, 3)
        version3 = SemanticVersion(1, 2, 2)

        assert version1 >= version2  # Equal
        assert version1 >= version3  # Greater than
        assert not (version3 >= version1)

    def test_is_newer_than(self):
        """Test is_newer_than method."""
        newer_version = SemanticVersion(1, 2, 4)
        older_version = SemanticVersion(1, 2, 3)

        assert newer_version.is_newer_than(older_version)
        assert not older_version.is_newer_than(newer_version)
        assert not newer_version.is_newer_than(newer_version)  # Not newer than itself

    def test_comparison_with_non_semantic_version(self):
        """Test comparison with non-SemanticVersion objects."""
        version = SemanticVersion(1, 2, 3)

        # Should return NotImplemented for non-SemanticVersion comparisons
        assert (version < "1.2.4") is NotImplemented
        assert (version > 123) is NotImplemented

    def test_is_newer_than_with_invalid_type(self):
        """Test is_newer_than with invalid type."""
        version = SemanticVersion(1, 2, 3)

        with pytest.raises(TypeError):
            version.is_newer_than("1.2.4")  # type: ignore


class TestSemanticVersionCompatibility:
    """Test SemanticVersion compatibility checking."""

    def test_is_compatible_with_same_major_minor(self):
        """Test compatibility within same major.minor version."""
        base_version = SemanticVersion(1, 2, 0)
        patch_versions = [
            SemanticVersion(1, 2, 1),
            SemanticVersion(1, 2, 2),
            SemanticVersion(1, 2, 10),
        ]

        for patch_version in patch_versions:
            assert patch_version.is_compatible_with(base_version)

        # Base version is not compatible with newer patches
        assert not base_version.is_compatible_with(SemanticVersion(1, 2, 1))

    def test_is_compatible_with_different_minor(self):
        """Test compatibility across different minor versions."""
        version1 = SemanticVersion(1, 2, 0)
        version2 = SemanticVersion(1, 3, 0)

        # Different minor versions are not compatible
        assert not version1.is_compatible_with(version2)
        assert not version2.is_compatible_with(version1)

    def test_is_compatible_with_different_major(self):
        """Test compatibility across different major versions."""
        version1 = SemanticVersion(1, 0, 0)
        version2 = SemanticVersion(2, 0, 0)

        # Different major versions are not compatible
        assert not version1.is_compatible_with(version2)
        assert not version2.is_compatible_with(version1)

    def test_is_compatible_with_same_version(self):
        """Test compatibility with exact same version."""
        version = SemanticVersion(1, 2, 3)
        same_version = SemanticVersion(1, 2, 3)

        assert version.is_compatible_with(same_version)
        assert same_version.is_compatible_with(version)

    def test_is_compatible_with_invalid_type(self):
        """Test is_compatible_with with invalid type."""
        version = SemanticVersion(1, 2, 3)

        with pytest.raises(TypeError):
            version.is_compatible_with("1.2.3")  # type: ignore


class TestSemanticVersionDistance:
    """Test SemanticVersion distance calculation."""

    def test_distance_from_same_version(self):
        """Test distance from same version."""
        version = SemanticVersion(1, 2, 3)
        same_version = SemanticVersion(1, 2, 3)

        assert version.distance_from(same_version) == 0

    def test_distance_from_patch_difference(self):
        """Test distance with patch differences."""
        version1 = SemanticVersion(1, 2, 3)
        version2 = SemanticVersion(1, 2, 5)

        distance = version1.distance_from(version2)
        assert distance == 2  # 5 - 3 = 2

    def test_distance_from_minor_difference(self):
        """Test distance with minor differences."""
        version1 = SemanticVersion(1, 2, 0)
        version2 = SemanticVersion(1, 4, 0)

        distance = version1.distance_from(version2)
        assert distance == 2000  # 2 * 1000 (minor weight)

    def test_distance_from_major_difference(self):
        """Test distance with major differences."""
        version1 = SemanticVersion(1, 0, 0)
        version2 = SemanticVersion(3, 0, 0)

        distance = version1.distance_from(version2)
        assert distance == 2000000  # 2 * 1000000 (major weight)

    def test_distance_calculation_symmetry(self):
        """Test that distance calculation is symmetric."""
        version1 = SemanticVersion(1, 2, 3)
        version2 = SemanticVersion(2, 1, 5)

        distance1 = version1.distance_from(version2)
        distance2 = version2.distance_from(version1)

        assert distance1 == distance2

    def test_distance_from_invalid_type(self):
        """Test distance_from with invalid type."""
        version = SemanticVersion(1, 2, 3)

        with pytest.raises(TypeError):
            version.distance_from("1.2.3")  # type: ignore


class TestSemanticVersionSerialization:
    """Test SemanticVersion serialization methods."""

    def test_to_dict(self):
        """Test to_dict method."""
        version = SemanticVersion(1, 2, 3)
        result = version.to_dict()

        expected = {
            "major": 1,
            "minor": 2,
            "patch": 3,
            "version_string": "1.2.3",
            "is_prerelease": False,
            "is_stable": True,
        }

        assert result == expected

    def test_to_dict_prerelease(self):
        """Test to_dict with prerelease version."""
        version = SemanticVersion(0, 1, 0)
        result = version.to_dict()

        assert result["is_prerelease"] is True
        assert result["is_stable"] is False

    def test_str_representation(self):
        """Test string representation."""
        version = SemanticVersion(1, 2, 3)

        assert str(version) == "1.2.3"

    def test_repr_representation(self):
        """Test repr representation."""
        version = SemanticVersion(1, 2, 3)
        repr_str = repr(version)

        assert "SemanticVersion" in repr_str
        assert "major=1" in repr_str
        assert "minor=2" in repr_str
        assert "patch=3" in repr_str


class TestSemanticVersionImmutability:
    """Test SemanticVersion immutability."""

    def test_frozen_dataclass(self):
        """Test that SemanticVersion is frozen."""
        version = SemanticVersion(1, 2, 3)

        with pytest.raises(AttributeError):
            version.major = 2

        with pytest.raises(AttributeError):
            version.minor = 3

        with pytest.raises(AttributeError):
            version.patch = 4

    def test_hash_consistency(self):
        """Test that equal versions have equal hashes."""
        version1 = SemanticVersion(1, 2, 3)
        version2 = SemanticVersion(1, 2, 3)
        version3 = SemanticVersion(1, 2, 4)

        assert hash(version1) == hash(version2)
        assert hash(version1) != hash(version3)

    def test_use_in_sets(self):
        """Test using semantic versions in sets."""
        version1 = SemanticVersion(1, 2, 3)
        version2 = SemanticVersion(1, 2, 4)
        version3 = SemanticVersion(1, 2, 3)  # Same as version1

        version_set = {version1, version2, version3}

        assert len(version_set) == 2  # version1 and version3 are the same
        assert version1 in version_set
        assert version2 in version_set

    def test_use_as_dict_keys(self):
        """Test using semantic versions as dictionary keys."""
        version1 = SemanticVersion(1, 2, 3)
        version2 = SemanticVersion(1, 2, 4)

        version_dict = {version1: "stable", version2: "latest"}

        assert len(version_dict) == 2
        assert version_dict[version1] == "stable"
        assert version_dict[version2] == "latest"


class TestSemanticVersionEdgeCases:
    """Test SemanticVersion edge cases and boundary conditions."""

    def test_max_integer_values(self):
        """Test with maximum reasonable integer values."""
        # Test with large but reasonable version numbers
        version = SemanticVersion(major=999999, minor=999999, patch=999999)

        assert version.version_string == "999999.999999.999999"
        assert version.is_stable()

    def test_version_string_roundtrip(self):
        """Test version string serialization roundtrip."""
        original = SemanticVersion(1, 2, 3)
        version_str = original.version_string
        reconstructed = SemanticVersion.from_string(version_str)

        assert original == reconstructed

    def test_from_string_roundtrip_with_prefix(self):
        """Test from_string roundtrip with v prefix."""
        version_str = "v1.2.3"
        version = SemanticVersion.from_string(version_str)

        # Should parse correctly without prefix in output
        assert str(version) == "1.2.3"
        assert version == SemanticVersion(1, 2, 3)

    def test_sorting_versions(self):
        """Test sorting semantic versions."""
        versions = [
            SemanticVersion(2, 0, 0),
            SemanticVersion(1, 2, 0),
            SemanticVersion(1, 1, 5),
            SemanticVersion(1, 2, 1),
            SemanticVersion(0, 1, 0),
        ]

        sorted_versions = sorted(versions)

        expected_order = [
            SemanticVersion(0, 1, 0),
            SemanticVersion(1, 1, 5),
            SemanticVersion(1, 2, 0),
            SemanticVersion(1, 2, 1),
            SemanticVersion(2, 0, 0),
        ]

        assert sorted_versions == expected_order

    def test_complex_version_operations(self):
        """Test complex combinations of version operations."""
        initial = SemanticVersion.initial()  # 0.1.0

        # Increment to get version progression
        v0_1_1 = initial.increment_patch()  # 0.1.1
        v0_2_0 = v0_1_1.increment_minor()  # 0.2.0
        v1_0_0 = v0_2_0.increment_major()  # 1.0.0

        # Test progression
        assert initial < v0_1_1 < v0_2_0 < v1_0_0

        # Test compatibility
        assert v0_1_1.is_compatible_with(initial)
        assert not v0_2_0.is_compatible_with(initial)  # Different minor
        assert not v1_0_0.is_compatible_with(initial)  # Different major

        # Test prerelease/stable
        assert initial.is_prerelease()
        assert v0_1_1.is_prerelease()
        assert v0_2_0.is_prerelease()
        assert v1_0_0.is_stable()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

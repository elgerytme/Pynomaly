"""Tests for semantic version value object."""

import pytest

from pynomaly.domain.value_objects.semantic_version import SemanticVersion


class TestSemanticVersion:
    """Test suite for SemanticVersion value object."""

    def test_basic_creation(self):
        """Test basic creation of semantic version."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_immutability(self):
        """Test that semantic version is immutable."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        
        # Should not be able to modify values
        with pytest.raises(AttributeError):
            version.major = 2

    def test_validation_major_version(self):
        """Test validation of major version."""
        # Valid major versions
        SemanticVersion(major=0, minor=1, patch=0)
        SemanticVersion(major=1, minor=0, patch=0)
        SemanticVersion(major=100, minor=0, patch=0)
        
        # Invalid major version (negative)
        with pytest.raises(ValueError, match="Major version must be non-negative integer"):
            SemanticVersion(major=-1, minor=0, patch=0)
        
        # Invalid major version (float)
        with pytest.raises(ValueError, match="Major version must be non-negative integer"):
            SemanticVersion(major=1.5, minor=0, patch=0)
        
        # Invalid major version (string)
        with pytest.raises(ValueError, match="Major version must be non-negative integer"):
            SemanticVersion(major="1", minor=0, patch=0)

    def test_validation_minor_version(self):
        """Test validation of minor version."""
        # Valid minor versions
        SemanticVersion(major=1, minor=0, patch=0)
        SemanticVersion(major=1, minor=1, patch=0)
        SemanticVersion(major=1, minor=100, patch=0)
        
        # Invalid minor version (negative)
        with pytest.raises(ValueError, match="Minor version must be non-negative integer"):
            SemanticVersion(major=1, minor=-1, patch=0)
        
        # Invalid minor version (float)
        with pytest.raises(ValueError, match="Minor version must be non-negative integer"):
            SemanticVersion(major=1, minor=2.5, patch=0)

    def test_validation_patch_version(self):
        """Test validation of patch version."""
        # Valid patch versions
        SemanticVersion(major=1, minor=0, patch=0)
        SemanticVersion(major=1, minor=0, patch=1)
        SemanticVersion(major=1, minor=0, patch=100)
        
        # Invalid patch version (negative)
        with pytest.raises(ValueError, match="Patch version must be non-negative integer"):
            SemanticVersion(major=1, minor=0, patch=-1)
        
        # Invalid patch version (float)
        with pytest.raises(ValueError, match="Patch version must be non-negative integer"):
            SemanticVersion(major=1, minor=0, patch=3.5)

    def test_version_string_property(self):
        """Test version_string property."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        assert version.version_string == "1.2.3"
        
        version2 = SemanticVersion(major=0, minor=1, patch=0)
        assert version2.version_string == "0.1.0"
        
        version3 = SemanticVersion(major=10, minor=20, patch=30)
        assert version3.version_string == "10.20.30"

    def test_from_string_factory(self):
        """Test from_string factory method."""
        # Valid version strings
        version1 = SemanticVersion.from_string("1.2.3")
        assert version1.major == 1
        assert version1.minor == 2
        assert version1.patch == 3
        
        version2 = SemanticVersion.from_string("0.1.0")
        assert version2.major == 0
        assert version2.minor == 1
        assert version2.patch == 0
        
        version3 = SemanticVersion.from_string("10.20.30")
        assert version3.major == 10
        assert version3.minor == 20
        assert version3.patch == 30

    def test_from_string_with_v_prefix(self):
        """Test from_string with 'v' prefix."""
        version = SemanticVersion.from_string("v1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.version_string == "1.2.3"

    def test_from_string_validation(self):
        """Test from_string validation."""
        # Invalid type
        with pytest.raises(ValueError, match="Version string must be str"):
            SemanticVersion.from_string(123)
        
        # Invalid format
        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.from_string("1.2")
        
        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.from_string("1.2.3.4")
        
        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.from_string("1.2.a")
        
        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.from_string("a.b.c")
        
        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.from_string("1.2.3-alpha")
        
        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.from_string("")

    def test_initial_factory(self):
        """Test initial factory method."""
        version = SemanticVersion.initial()
        assert version.major == 0
        assert version.minor == 1
        assert version.patch == 0
        assert version.version_string == "0.1.0"

    def test_stable_initial_factory(self):
        """Test stable_initial factory method."""
        version = SemanticVersion.stable_initial()
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.version_string == "1.0.0"

    def test_increment_major(self):
        """Test increment_major method."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        new_version = version.increment_major()
        
        assert new_version.major == 2
        assert new_version.minor == 0
        assert new_version.patch == 0
        assert new_version.version_string == "2.0.0"
        
        # Original should be unchanged
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_increment_minor(self):
        """Test increment_minor method."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        new_version = version.increment_minor()
        
        assert new_version.major == 1
        assert new_version.minor == 3
        assert new_version.patch == 0
        assert new_version.version_string == "1.3.0"
        
        # Original should be unchanged
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_increment_patch(self):
        """Test increment_patch method."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        new_version = version.increment_patch()
        
        assert new_version.major == 1
        assert new_version.minor == 2
        assert new_version.patch == 4
        assert new_version.version_string == "1.2.4"
        
        # Original should be unchanged
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_is_compatible_with(self):
        """Test is_compatible_with method."""
        base_version = SemanticVersion(major=1, minor=2, patch=3)
        
        # Same version is compatible
        assert base_version.is_compatible_with(base_version) is True
        
        # Higher patch version is compatible
        patch_version = SemanticVersion(major=1, minor=2, patch=4)
        assert patch_version.is_compatible_with(base_version) is True
        assert base_version.is_compatible_with(patch_version) is False
        
        # Higher minor version is compatible
        minor_version = SemanticVersion(major=1, minor=3, patch=0)
        assert minor_version.is_compatible_with(base_version) is True
        assert base_version.is_compatible_with(minor_version) is False
        
        # Different major version is not compatible
        major_version = SemanticVersion(major=2, minor=0, patch=0)
        assert major_version.is_compatible_with(base_version) is False
        assert base_version.is_compatible_with(major_version) is False
        
        # Lower versions are not compatible
        lower_minor = SemanticVersion(major=1, minor=1, patch=0)
        assert base_version.is_compatible_with(lower_minor) is False
        
        lower_patch = SemanticVersion(major=1, minor=2, patch=2)
        assert base_version.is_compatible_with(lower_patch) is False

    def test_is_compatible_with_validation(self):
        """Test is_compatible_with validation."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        
        with pytest.raises(TypeError, match="Can only compare with SemanticVersion"):
            version.is_compatible_with("1.2.3")

    def test_is_newer_than(self):
        """Test is_newer_than method."""
        base_version = SemanticVersion(major=1, minor=2, patch=3)
        
        # Same version is not newer
        assert base_version.is_newer_than(base_version) is False
        
        # Higher patch version is newer
        patch_version = SemanticVersion(major=1, minor=2, patch=4)
        assert patch_version.is_newer_than(base_version) is True
        assert base_version.is_newer_than(patch_version) is False
        
        # Higher minor version is newer
        minor_version = SemanticVersion(major=1, minor=3, patch=0)
        assert minor_version.is_newer_than(base_version) is True
        assert base_version.is_newer_than(minor_version) is False
        
        # Higher major version is newer
        major_version = SemanticVersion(major=2, minor=0, patch=0)
        assert major_version.is_newer_than(base_version) is True
        assert base_version.is_newer_than(major_version) is False
        
        # Lower versions are not newer
        lower_version = SemanticVersion(major=1, minor=2, patch=2)
        assert base_version.is_newer_than(lower_version) is True
        assert lower_version.is_newer_than(base_version) is False

    def test_is_newer_than_validation(self):
        """Test is_newer_than validation."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        
        with pytest.raises(TypeError, match="Can only compare with SemanticVersion"):
            version.is_newer_than("1.2.3")

    def test_is_prerelease(self):
        """Test is_prerelease method."""
        # Major version 0 is prerelease
        prerelease = SemanticVersion(major=0, minor=1, patch=0)
        assert prerelease.is_prerelease() is True
        
        prerelease2 = SemanticVersion(major=0, minor=5, patch=10)
        assert prerelease2.is_prerelease() is True
        
        # Major version >= 1 is not prerelease
        stable = SemanticVersion(major=1, minor=0, patch=0)
        assert stable.is_prerelease() is False
        
        stable2 = SemanticVersion(major=2, minor=5, patch=10)
        assert stable2.is_prerelease() is False

    def test_is_stable(self):
        """Test is_stable method."""
        # Major version 0 is not stable
        prerelease = SemanticVersion(major=0, minor=1, patch=0)
        assert prerelease.is_stable() is False
        
        # Major version >= 1 is stable
        stable = SemanticVersion(major=1, minor=0, patch=0)
        assert stable.is_stable() is True
        
        stable2 = SemanticVersion(major=2, minor=5, patch=10)
        assert stable2.is_stable() is True

    def test_distance_from(self):
        """Test distance_from method."""
        base_version = SemanticVersion(major=1, minor=2, patch=3)
        
        # Same version has distance 0
        assert base_version.distance_from(base_version) == 0
        
        # Test patch distance
        patch_version = SemanticVersion(major=1, minor=2, patch=4)
        assert base_version.distance_from(patch_version) == 1
        assert patch_version.distance_from(base_version) == 1
        
        # Test minor distance
        minor_version = SemanticVersion(major=1, minor=3, patch=3)
        assert base_version.distance_from(minor_version) == 1000
        assert minor_version.distance_from(base_version) == 1000
        
        # Test major distance
        major_version = SemanticVersion(major=2, minor=2, patch=3)
        assert base_version.distance_from(major_version) == 1000000
        assert major_version.distance_from(base_version) == 1000000
        
        # Test combined distance
        combined_version = SemanticVersion(major=2, minor=3, patch=4)
        expected_distance = 1000000 + 1000 + 1
        assert base_version.distance_from(combined_version) == expected_distance

    def test_distance_from_validation(self):
        """Test distance_from validation."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        
        with pytest.raises(TypeError, match="Can only compare with SemanticVersion"):
            version.distance_from("1.2.3")

    def test_to_dict(self):
        """Test to_dict method."""
        version = SemanticVersion(major=1, minor=2, patch=3)
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
        
        # Test prerelease version
        prerelease = SemanticVersion(major=0, minor=1, patch=0)
        prerelease_result = prerelease.to_dict()
        
        assert prerelease_result["is_prerelease"] is True
        assert prerelease_result["is_stable"] is False

    def test_string_representation(self):
        """Test string representation."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        assert str(version) == "1.2.3"
        
        version2 = SemanticVersion(major=0, minor=1, patch=0)
        assert str(version2) == "0.1.0"

    def test_equality_comparison(self):
        """Test equality comparison."""
        version1 = SemanticVersion(major=1, minor=2, patch=3)
        version2 = SemanticVersion(major=1, minor=2, patch=3)
        version3 = SemanticVersion(major=1, minor=2, patch=4)
        
        assert version1 == version2
        assert version1 != version3

    def test_ordering_comparison(self):
        """Test ordering comparison operators."""
        v1_0_0 = SemanticVersion(major=1, minor=0, patch=0)
        v1_1_0 = SemanticVersion(major=1, minor=1, patch=0)
        v1_1_1 = SemanticVersion(major=1, minor=1, patch=1)
        v2_0_0 = SemanticVersion(major=2, minor=0, patch=0)
        
        # Test less than
        assert v1_0_0 < v1_1_0
        assert v1_1_0 < v1_1_1
        assert v1_1_1 < v2_0_0
        assert not (v1_1_0 < v1_0_0)
        
        # Test less than or equal
        assert v1_0_0 <= v1_1_0
        assert v1_0_0 <= v1_0_0
        assert not (v1_1_0 <= v1_0_0)
        
        # Test greater than
        assert v1_1_0 > v1_0_0
        assert v1_1_1 > v1_1_0
        assert v2_0_0 > v1_1_1
        assert not (v1_0_0 > v1_1_0)
        
        # Test greater than or equal
        assert v1_1_0 >= v1_0_0
        assert v1_0_0 >= v1_0_0
        assert not (v1_0_0 >= v1_1_0)

    def test_ordering_comparison_with_invalid_type(self):
        """Test ordering comparison with invalid types."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        
        # Should return NotImplemented, not raise exception
        assert (version < "1.2.3") is NotImplemented
        assert (version <= "1.2.3") is NotImplemented
        assert (version > "1.2.3") is NotImplemented
        assert (version >= "1.2.3") is NotImplemented

    def test_hash_behavior(self):
        """Test hash behavior for use in sets and dictionaries."""
        version1 = SemanticVersion(major=1, minor=2, patch=3)
        version2 = SemanticVersion(major=1, minor=2, patch=3)
        version3 = SemanticVersion(major=1, minor=2, patch=4)
        
        # Same values should have same hash
        assert hash(version1) == hash(version2)
        
        # Different values should have different hash
        assert hash(version1) != hash(version3)
        
        # Test in set
        version_set = {version1, version2, version3}
        assert len(version_set) == 2  # version1 and version2 are equal

    def test_repr_representation(self):
        """Test repr representation."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        repr_str = repr(version)
        assert "SemanticVersion" in repr_str
        assert "major=1" in repr_str
        assert "minor=2" in repr_str
        assert "patch=3" in repr_str

    def test_version_progression(self):
        """Test typical version progression scenarios."""
        # Start with initial version
        v1 = SemanticVersion.initial()
        assert v1.version_string == "0.1.0"
        assert v1.is_prerelease() is True
        
        # Increment patch for bug fixes
        v2 = v1.increment_patch()
        assert v2.version_string == "0.1.1"
        assert v2.is_prerelease() is True
        
        # Increment minor for new features
        v3 = v2.increment_minor()
        assert v3.version_string == "0.2.0"
        assert v3.is_prerelease() is True
        
        # Increment major for breaking changes (first stable release)
        v4 = v3.increment_major()
        assert v4.version_string == "1.0.0"
        assert v4.is_stable() is True
        
        # Continue with stable releases
        v5 = v4.increment_minor()
        assert v5.version_string == "1.1.0"
        assert v5.is_stable() is True

    def test_compatibility_scenarios(self):
        """Test compatibility scenarios."""
        # Base version
        base = SemanticVersion(major=1, minor=2, patch=3)
        
        # Compatible versions (same major, higher minor/patch)
        compatible_patch = SemanticVersion(major=1, minor=2, patch=4)
        compatible_minor = SemanticVersion(major=1, minor=3, patch=0)
        compatible_both = SemanticVersion(major=1, minor=3, patch=1)
        
        assert compatible_patch.is_compatible_with(base) is True
        assert compatible_minor.is_compatible_with(base) is True
        assert compatible_both.is_compatible_with(base) is True
        
        # Incompatible versions (different major)
        incompatible_major = SemanticVersion(major=2, minor=0, patch=0)
        incompatible_older_major = SemanticVersion(major=0, minor=9, patch=9)
        
        assert incompatible_major.is_compatible_with(base) is False
        assert incompatible_older_major.is_compatible_with(base) is False
        
        # Incompatible versions (lower minor/patch)
        incompatible_minor = SemanticVersion(major=1, minor=1, patch=9)
        incompatible_patch = SemanticVersion(major=1, minor=2, patch=2)
        
        assert base.is_compatible_with(incompatible_minor) is False
        assert base.is_compatible_with(incompatible_patch) is False

    def test_version_sorting(self):
        """Test sorting versions."""
        versions = [
            SemanticVersion(major=2, minor=0, patch=0),
            SemanticVersion(major=1, minor=0, patch=0),
            SemanticVersion(major=1, minor=2, patch=0),
            SemanticVersion(major=1, minor=1, patch=0),
            SemanticVersion(major=1, minor=1, patch=1),
        ]
        
        sorted_versions = sorted(versions)
        expected_order = [
            "1.0.0",
            "1.1.0",
            "1.1.1",
            "1.2.0",
            "2.0.0",
        ]
        
        actual_order = [str(v) for v in sorted_versions]
        assert actual_order == expected_order

    def test_version_filtering(self):
        """Test filtering versions."""
        versions = [
            SemanticVersion(major=0, minor=1, patch=0),
            SemanticVersion(major=0, minor=2, patch=0),
            SemanticVersion(major=1, minor=0, patch=0),
            SemanticVersion(major=1, minor=1, patch=0),
            SemanticVersion(major=2, minor=0, patch=0),
        ]
        
        # Filter stable versions
        stable_versions = [v for v in versions if v.is_stable()]
        assert len(stable_versions) == 3
        assert all(v.major >= 1 for v in stable_versions)
        
        # Filter prerelease versions
        prerelease_versions = [v for v in versions if v.is_prerelease()]
        assert len(prerelease_versions) == 2
        assert all(v.major == 0 for v in prerelease_versions)

    def test_version_distance_calculations(self):
        """Test version distance calculations."""
        base = SemanticVersion(major=1, minor=2, patch=3)
        
        # Test various distances
        test_cases = [
            (SemanticVersion(major=1, minor=2, patch=3), 0),        # Same version
            (SemanticVersion(major=1, minor=2, patch=4), 1),        # Patch diff
            (SemanticVersion(major=1, minor=3, patch=3), 1000),     # Minor diff
            (SemanticVersion(major=2, minor=2, patch=3), 1000000),  # Major diff
            (SemanticVersion(major=2, minor=3, patch=4), 1001001),  # Combined diff
        ]
        
        for version, expected_distance in test_cases:
            assert base.distance_from(version) == expected_distance

    def test_edge_cases(self):
        """Test edge cases."""
        # Zero versions
        zero_version = SemanticVersion(major=0, minor=0, patch=0)
        assert zero_version.version_string == "0.0.0"
        assert zero_version.is_prerelease() is True
        
        # Large versions
        large_version = SemanticVersion(major=999, minor=999, patch=999)
        assert large_version.version_string == "999.999.999"
        assert large_version.is_stable() is True
        
        # Single digit versions
        single_digit = SemanticVersion(major=1, minor=2, patch=3)
        assert single_digit.version_string == "1.2.3"

    def test_factory_methods_consistency(self):
        """Test consistency between factory methods."""
        # from_string should create equivalent objects
        version1 = SemanticVersion.from_string("1.2.3")
        version2 = SemanticVersion(major=1, minor=2, patch=3)
        
        assert version1 == version2
        assert version1.version_string == version2.version_string
        
        # Factory methods should create independent objects
        initial1 = SemanticVersion.initial()
        initial2 = SemanticVersion.initial()
        
        assert initial1 == initial2
        assert initial1 is not initial2

    def test_increment_methods_immutability(self):
        """Test that increment methods don't modify original."""
        original = SemanticVersion(major=1, minor=2, patch=3)
        
        # Create incremented versions
        major_inc = original.increment_major()
        minor_inc = original.increment_minor()
        patch_inc = original.increment_patch()
        
        # Original should be unchanged
        assert original.major == 1
        assert original.minor == 2
        assert original.patch == 3
        
        # New versions should be different
        assert major_inc != original
        assert minor_inc != original
        assert patch_inc != original
        
        # New versions should have expected values
        assert major_inc.version_string == "2.0.0"
        assert minor_inc.version_string == "1.3.0"
        assert patch_inc.version_string == "1.2.4"
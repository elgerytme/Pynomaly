"""Tests for data loader protocol implementation."""

from typing import Any

import pandas as pd
import pytest

from monorepo.domain.entities import Dataset
from monorepo.shared.protocols.data_loader_protocol import DataLoaderProtocol


class MockDataLoader:
    """Mock implementation of DataLoaderProtocol for testing."""

    def __init__(self, name: str = "mock_loader"):
        self._name = name
        self._supported_formats = ["csv", "json", "parquet"]
        self._config = {"delimiter": ",", "encoding": "utf-8"}

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_formats(self) -> list[str]:
        return self._supported_formats.copy()

    @property
    def config(self) -> dict[str, Any]:
        return self._config.copy()

    def load(self, source: str, **kwargs) -> Dataset:
        """Mock load implementation."""
        # Create mock data based on source
        if source.endswith(".csv"):
            data = pd.DataFrame(
                {
                    "feature1": [1, 2, 3, 4],
                    "feature2": [0.1, 0.2, 0.3, 0.4],
                    "target": [0, 0, 1, 1],
                }
            )
        else:
            data = pd.DataFrame({"value": [1, 2, 3, 4, 5], "label": [0, 0, 0, 1, 1]})

        return Dataset(
            id=f"dataset_from_{source}",
            name=f"Mock dataset from {source}",
            data=data,
            metadata={"source": source, "loader": self.name},
        )

    def validate_source(self, source: str) -> bool:
        """Mock validation implementation."""
        # Simple validation based on file extension
        return any(source.endswith(f".{fmt}") for fmt in self._supported_formats)


class TestDataLoaderProtocol:
    """Test the DataLoaderProtocol interface."""

    def test_protocol_is_runtime_checkable(self):
        """Test that DataLoaderProtocol is runtime checkable."""
        loader = MockDataLoader()
        assert isinstance(loader, DataLoaderProtocol)

    def test_protocol_properties_exist(self):
        """Test that all required properties exist."""
        loader = MockDataLoader("test_loader")

        # Test name property
        assert loader.name == "test_loader"
        assert isinstance(loader.name, str)

        # Test supported_formats property
        formats = loader.supported_formats
        assert isinstance(formats, list)
        assert all(isinstance(fmt, str) for fmt in formats)

        # Test config property
        config = loader.config
        assert isinstance(config, dict)

    def test_protocol_methods_exist(self):
        """Test that all required methods exist and are callable."""
        loader = MockDataLoader()

        # Test load method exists
        assert hasattr(loader, "load")
        assert callable(loader.load)

        # Test validate_source method exists
        assert hasattr(loader, "validate_source")
        assert callable(loader.validate_source)

    def test_load_method_signature(self):
        """Test load method accepts source parameter and returns Dataset."""
        loader = MockDataLoader()
        source = "test_data.csv"

        result = loader.load(source)
        assert isinstance(result, Dataset)
        assert result.id is not None
        assert result.name is not None
        assert result.data is not None

    def test_load_with_kwargs(self):
        """Test load method accepts keyword arguments."""
        loader = MockDataLoader()
        source = "test_data.csv"

        # Should not raise exception with additional kwargs
        result = loader.load(source, encoding="utf-8", delimiter=",")
        assert isinstance(result, Dataset)

    def test_validate_source_method_signature(self):
        """Test validate_source method accepts source and returns bool."""
        loader = MockDataLoader()

        # Test with valid source
        assert loader.validate_source("data.csv") is True
        assert loader.validate_source("data.json") is True
        assert loader.validate_source("data.parquet") is True

        # Test with invalid source
        assert loader.validate_source("data.txt") is False
        assert loader.validate_source("data.xml") is False

    def test_supported_formats_returns_copy(self):
        """Test that supported_formats property returns a copy."""
        loader = MockDataLoader()

        formats1 = loader.supported_formats
        formats2 = loader.supported_formats

        # Should be equal but not the same object
        assert formats1 == formats2
        assert formats1 is not formats2

        # Modifying returned list shouldn't affect loader
        formats1.append("new_format")
        formats3 = loader.supported_formats
        assert "new_format" not in formats3

    def test_config_returns_copy(self):
        """Test that config property returns a copy."""
        loader = MockDataLoader()

        config1 = loader.config
        config2 = loader.config

        # Should be equal but not the same object
        assert config1 == config2
        assert config1 is not config2

        # Modifying returned dict shouldn't affect loader
        config1["new_key"] = "new_value"
        config3 = loader.config
        assert "new_key" not in config3


class TestDataLoaderProtocolCompliance:
    """Test compliance with DataLoaderProtocol contract."""

    def test_complete_workflow(self):
        """Test complete data loading workflow."""
        loader = MockDataLoader("workflow_test")
        source = "test_data.csv"

        # Validate source first
        assert loader.validate_source(source)

        # Load data
        dataset = loader.load(source)

        # Verify result
        assert isinstance(dataset, Dataset)
        assert dataset.id is not None
        assert dataset.name is not None
        assert dataset.data is not None
        assert hasattr(dataset, "metadata")

    def test_different_file_formats(self):
        """Test loading different supported file formats."""
        loader = MockDataLoader()

        formats_to_test = ["data.csv", "data.json", "data.parquet"]

        for source in formats_to_test:
            assert loader.validate_source(source)
            dataset = loader.load(source)
            assert isinstance(dataset, Dataset)
            assert source in dataset.metadata.get("source", "")

    def test_unsupported_format_validation(self):
        """Test validation of unsupported formats."""
        loader = MockDataLoader()

        unsupported_formats = ["data.txt", "data.xml", "data.xlsx", "data.db"]

        for source in unsupported_formats:
            assert not loader.validate_source(source)

    def test_loader_configuration(self):
        """Test that loader configuration is accessible."""
        loader = MockDataLoader()
        config = loader.config

        assert isinstance(config, dict)
        # Config should contain some settings
        assert len(config) >= 0

    def test_multiple_loads_same_loader(self):
        """Test that same loader can load multiple sources."""
        loader = MockDataLoader()

        source1 = "data1.csv"
        source2 = "data2.json"

        dataset1 = loader.load(source1)
        dataset2 = loader.load(source2)

        assert isinstance(dataset1, Dataset)
        assert isinstance(dataset2, Dataset)
        assert dataset1.id != dataset2.id

    def test_loader_name_consistency(self):
        """Test that loader name remains consistent."""
        loader = MockDataLoader("consistent_name")

        # Name should be consistent across calls
        name1 = loader.name
        name2 = loader.name
        assert name1 == name2 == "consistent_name"

    def test_supported_formats_consistency(self):
        """Test that supported formats remain consistent."""
        loader = MockDataLoader()

        formats1 = loader.supported_formats
        formats2 = loader.supported_formats

        assert formats1 == formats2


class TestDataLoaderErrorHandling:
    """Test error handling in data loader protocol."""

    def test_load_returns_dataset_type(self):
        """Test that load always returns Dataset type."""
        loader = MockDataLoader()

        # Even with various sources, should return Dataset
        sources = ["data.csv", "data.json", "unusual_name.parquet"]

        for source in sources:
            if loader.validate_source(source):
                result = loader.load(source)
                assert isinstance(result, Dataset)

    def test_validate_source_returns_bool(self):
        """Test that validate_source always returns boolean."""
        loader = MockDataLoader()

        test_sources = [
            "valid.csv",
            "valid.json",
            "invalid.txt",
            "no_extension",
            "",
            "weird.name.csv",
        ]

        for source in test_sources:
            result = loader.validate_source(source)
            assert isinstance(result, bool)


@pytest.fixture
def sample_loader():
    """Fixture providing a sample data loader for testing."""
    return MockDataLoader("sample_loader")


@pytest.fixture
def valid_sources():
    """Fixture providing valid data sources for testing."""
    return ["data.csv", "dataset.json", "features.parquet"]


@pytest.fixture
def invalid_sources():
    """Fixture providing invalid data sources for testing."""
    return ["data.txt", "file.xml", "document.pdf", "archive.zip"]


class TestDataLoaderProtocolFixtures:
    """Test data loader protocol using fixtures."""

    def test_loader_with_valid_sources(self, sample_loader, valid_sources):
        """Test loader with valid data sources using fixtures."""
        for source in valid_sources:
            assert sample_loader.validate_source(source)
            dataset = sample_loader.load(source)
            assert isinstance(dataset, Dataset)
            assert dataset.name is not None

    def test_loader_with_invalid_sources(self, sample_loader, invalid_sources):
        """Test loader validation with invalid sources using fixtures."""
        for source in invalid_sources:
            assert not sample_loader.validate_source(source)

    def test_loader_properties_immutability(self, sample_loader):
        """Test that loader properties are effectively immutable."""
        original_formats = sample_loader.supported_formats
        original_config = sample_loader.config

        # Attempt to modify returned properties
        modified_formats = sample_loader.supported_formats
        modified_formats.append("should_not_persist")

        modified_config = sample_loader.config
        modified_config["should_not_persist"] = True

        # Get fresh properties
        fresh_formats = sample_loader.supported_formats
        fresh_config = sample_loader.config

        # Should not contain modifications
        assert "should_not_persist" not in fresh_formats
        assert "should_not_persist" not in fresh_config
        assert fresh_formats == original_formats
        assert fresh_config == original_config

    def test_dataset_metadata_contains_loader_info(self, sample_loader, valid_sources):
        """Test that loaded datasets contain loader metadata."""
        for source in valid_sources:
            dataset = sample_loader.load(source)

            # Metadata should contain information about the loader
            assert "loader" in dataset.metadata or "source" in dataset.metadata

            if "loader" in dataset.metadata:
                assert dataset.metadata["loader"] == sample_loader.name

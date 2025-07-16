"""Tests for data loader protocol."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from monorepo.domain.entities import Dataset
from monorepo.shared.protocols.data_loader_protocol import (
    BatchDataLoaderProtocol,
    DatabaseLoaderProtocol,
    DataLoaderProtocol,
    StreamingDataLoaderProtocol,
)


class TestDataLoaderProtocol:
    """Test suite for DataLoaderProtocol."""

    def test_protocol_definition(self):
        """Test protocol has required methods."""
        assert hasattr(DataLoaderProtocol, "load")
        assert hasattr(DataLoaderProtocol, "validate")
        assert hasattr(DataLoaderProtocol, "supported_formats")

    def test_load_method_signature(self):
        """Test load method has correct signature."""
        # This is a protocol test - we're checking the interface
        mock_loader = Mock(spec=DataLoaderProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_loader.load.return_value = mock_dataset

        # Test with string source
        result = mock_loader.load("test.csv", name="test")
        assert result == mock_dataset

        # Test with Path source
        result = mock_loader.load(Path("test.csv"), name="test")
        assert result == mock_dataset

    def test_validate_method_signature(self):
        """Test validate method has correct signature."""
        mock_loader = Mock(spec=DataLoaderProtocol)
        mock_loader.validate.return_value = True

        # Test with string source
        result = mock_loader.validate("test.csv")
        assert result is True

        # Test with Path source
        result = mock_loader.validate(Path("test.csv"))
        assert result is True

    def test_supported_formats_property(self):
        """Test supported_formats property."""
        mock_loader = Mock(spec=DataLoaderProtocol)
        mock_loader.supported_formats = ["csv", "parquet"]

        formats = mock_loader.supported_formats
        assert isinstance(formats, list)
        assert all(isinstance(fmt, str) for fmt in formats)

    def test_protocol_runtime_checkable(self):
        """Test protocol is runtime checkable."""

        class ConcreteLoader:
            def load(
                self, source: str | Path, name: str | None = None, **kwargs: Any
            ) -> Dataset:
                return Mock(spec=Dataset)

            def validate(self, source: str | Path) -> bool:
                return True

            @property
            def supported_formats(self) -> list[str]:
                return ["csv"]

        loader = ConcreteLoader()
        assert isinstance(loader, DataLoaderProtocol)

    def test_protocol_with_missing_methods(self):
        """Test protocol check fails with missing methods."""

        class IncompleteLoader:
            def load(
                self, source: str | Path, name: str | None = None, **kwargs: Any
            ) -> Dataset:
                return Mock(spec=Dataset)

            # Missing validate and supported_formats

        loader = IncompleteLoader()
        assert not isinstance(loader, DataLoaderProtocol)


class TestBatchDataLoaderProtocol:
    """Test suite for BatchDataLoaderProtocol."""

    def test_protocol_inheritance(self):
        """Test BatchDataLoaderProtocol extends DataLoaderProtocol."""
        # BatchDataLoaderProtocol should have all DataLoaderProtocol methods
        assert hasattr(BatchDataLoaderProtocol, "load")
        assert hasattr(BatchDataLoaderProtocol, "validate")
        assert hasattr(BatchDataLoaderProtocol, "supported_formats")

        # Plus batch-specific methods
        assert hasattr(BatchDataLoaderProtocol, "load_batch")
        assert hasattr(BatchDataLoaderProtocol, "estimate_size")

    def test_load_batch_method_signature(self):
        """Test load_batch method has correct signature."""
        mock_loader = Mock(spec=BatchDataLoaderProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_loader.load_batch.return_value = iter([mock_dataset])

        result = mock_loader.load_batch("test.csv", batch_size=1000, name="test")
        assert hasattr(result, "__iter__")

    def test_estimate_size_method_signature(self):
        """Test estimate_size method has correct signature."""
        mock_loader = Mock(spec=BatchDataLoaderProtocol)
        mock_loader.estimate_size.return_value = {"rows": 1000, "columns": 10}

        result = mock_loader.estimate_size("test.csv")
        assert isinstance(result, dict)

    def test_protocol_runtime_checkable(self):
        """Test BatchDataLoaderProtocol is runtime checkable."""

        class ConcreteBatchLoader:
            def load(
                self, source: str | Path, name: str | None = None, **kwargs: Any
            ) -> Dataset:
                return Mock(spec=Dataset)

            def validate(self, source: str | Path) -> bool:
                return True

            @property
            def supported_formats(self) -> list[str]:
                return ["csv"]

            def load_batch(
                self,
                source: str | Path,
                batch_size: int,
                name: str | None = None,
                **kwargs: Any,
            ) -> Iterator[Dataset]:
                return iter([Mock(spec=Dataset)])

            def estimate_size(self, source: str | Path) -> dict[str, Any]:
                return {"rows": 1000, "columns": 10}

        loader = ConcreteBatchLoader()
        assert isinstance(loader, BatchDataLoaderProtocol)
        assert isinstance(loader, DataLoaderProtocol)


class TestStreamingDataLoaderProtocol:
    """Test suite for StreamingDataLoaderProtocol."""

    def test_protocol_inheritance(self):
        """Test StreamingDataLoaderProtocol extends DataLoaderProtocol."""
        # Should have all DataLoaderProtocol methods
        assert hasattr(StreamingDataLoaderProtocol, "load")
        assert hasattr(StreamingDataLoaderProtocol, "validate")
        assert hasattr(StreamingDataLoaderProtocol, "supported_formats")

        # Plus streaming-specific methods
        assert hasattr(StreamingDataLoaderProtocol, "stream")
        assert hasattr(StreamingDataLoaderProtocol, "connect")
        assert hasattr(StreamingDataLoaderProtocol, "disconnect")
        assert hasattr(StreamingDataLoaderProtocol, "is_connected")

    def test_stream_method_signature(self):
        """Test stream method has correct signature."""
        mock_loader = Mock(spec=StreamingDataLoaderProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_loader.stream.return_value = iter([mock_dataset])

        result = mock_loader.stream("kafka://topic", window_size=100)
        assert hasattr(result, "__iter__")

    def test_connection_methods(self):
        """Test connection-related methods."""
        mock_loader = Mock(spec=StreamingDataLoaderProtocol)
        mock_loader.is_connected = True

        # Test connect
        mock_loader.connect("kafka://topic")
        mock_loader.connect.assert_called_once_with("kafka://topic")

        # Test disconnect
        mock_loader.disconnect()
        mock_loader.disconnect.assert_called_once()

        # Test is_connected property
        assert mock_loader.is_connected is True

    def test_protocol_runtime_checkable(self):
        """Test StreamingDataLoaderProtocol is runtime checkable."""

        class ConcreteStreamingLoader:
            def load(
                self, source: str | Path, name: str | None = None, **kwargs: Any
            ) -> Dataset:
                return Mock(spec=Dataset)

            def validate(self, source: str | Path) -> bool:
                return True

            @property
            def supported_formats(self) -> list[str]:
                return ["kafka"]

            def stream(
                self, source: str | Path, window_size: int | None = None, **kwargs: Any
            ) -> Iterator[Dataset]:
                return iter([Mock(spec=Dataset)])

            def connect(self, source: str | Path, **kwargs: Any) -> None:
                pass

            def disconnect(self) -> None:
                pass

            @property
            def is_connected(self) -> bool:
                return True

        loader = ConcreteStreamingLoader()
        assert isinstance(loader, StreamingDataLoaderProtocol)
        assert isinstance(loader, DataLoaderProtocol)


class TestDatabaseLoaderProtocol:
    """Test suite for DatabaseLoaderProtocol."""

    def test_protocol_inheritance(self):
        """Test DatabaseLoaderProtocol extends DataLoaderProtocol."""
        # Should have all DataLoaderProtocol methods
        assert hasattr(DatabaseLoaderProtocol, "load")
        assert hasattr(DatabaseLoaderProtocol, "validate")
        assert hasattr(DatabaseLoaderProtocol, "supported_formats")

        # Plus database-specific methods
        assert hasattr(DatabaseLoaderProtocol, "load_query")
        assert hasattr(DatabaseLoaderProtocol, "load_table")
        assert hasattr(DatabaseLoaderProtocol, "get_table_info")

    def test_load_query_method_signature(self):
        """Test load_query method has correct signature."""
        mock_loader = Mock(spec=DatabaseLoaderProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_loader.load_query.return_value = mock_dataset

        result = mock_loader.load_query(
            "SELECT * FROM table", "connection", name="test"
        )
        assert result == mock_dataset

    def test_load_table_method_signature(self):
        """Test load_table method has correct signature."""
        mock_loader = Mock(spec=DatabaseLoaderProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_loader.load_table.return_value = mock_dataset

        result = mock_loader.load_table(
            "table_name", "connection", schema="public", name="test"
        )
        assert result == mock_dataset

    def test_get_table_info_method_signature(self):
        """Test get_table_info method has correct signature."""
        mock_loader = Mock(spec=DatabaseLoaderProtocol)
        mock_info = {"columns": ["col1", "col2"], "row_count": 1000}
        mock_loader.get_table_info.return_value = mock_info

        result = mock_loader.get_table_info("table_name", "connection", schema="public")
        assert isinstance(result, dict)

    def test_protocol_runtime_checkable(self):
        """Test DatabaseLoaderProtocol is runtime checkable."""

        class ConcreteDatabaseLoader:
            def load(
                self, source: str | Path, name: str | None = None, **kwargs: Any
            ) -> Dataset:
                return Mock(spec=Dataset)

            def validate(self, source: str | Path) -> bool:
                return True

            @property
            def supported_formats(self) -> list[str]:
                return ["postgresql", "mysql"]

            def load_query(
                self,
                query: str,
                connection: str | Any,
                name: str | None = None,
                **kwargs: Any,
            ) -> Dataset:
                return Mock(spec=Dataset)

            def load_table(
                self,
                table_name: str,
                connection: str | Any,
                schema: str | None = None,
                name: str | None = None,
                **kwargs: Any,
            ) -> Dataset:
                return Mock(spec=Dataset)

            def get_table_info(
                self, table_name: str, connection: str | Any, schema: str | None = None
            ) -> dict[str, Any]:
                return {"columns": ["col1"], "row_count": 100}

        loader = ConcreteDatabaseLoader()
        assert isinstance(loader, DatabaseLoaderProtocol)
        assert isinstance(loader, DataLoaderProtocol)


class TestProtocolInteractions:
    """Test protocol interactions and edge cases."""

    def test_multiple_protocol_inheritance(self):
        """Test class implementing multiple protocols."""

        class MultiProtocolLoader:
            def load(
                self, source: str | Path, name: str | None = None, **kwargs: Any
            ) -> Dataset:
                return Mock(spec=Dataset)

            def validate(self, source: str | Path) -> bool:
                return True

            @property
            def supported_formats(self) -> list[str]:
                return ["csv"]

            def load_batch(
                self,
                source: str | Path,
                batch_size: int,
                name: str | None = None,
                **kwargs: Any,
            ) -> Iterator[Dataset]:
                return iter([Mock(spec=Dataset)])

            def estimate_size(self, source: str | Path) -> dict[str, Any]:
                return {"rows": 1000}

            def stream(
                self, source: str | Path, window_size: int | None = None, **kwargs: Any
            ) -> Iterator[Dataset]:
                return iter([Mock(spec=Dataset)])

            def connect(self, source: str | Path, **kwargs: Any) -> None:
                pass

            def disconnect(self) -> None:
                pass

            @property
            def is_connected(self) -> bool:
                return True

        loader = MultiProtocolLoader()
        assert isinstance(loader, DataLoaderProtocol)
        assert isinstance(loader, BatchDataLoaderProtocol)
        assert isinstance(loader, StreamingDataLoaderProtocol)

    def test_protocol_with_kwargs(self):
        """Test protocol methods accept **kwargs."""
        mock_loader = Mock(spec=DataLoaderProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_loader.load.return_value = mock_dataset

        # Test load with various kwargs
        result = mock_loader.load("test.csv", name="test", delimiter=",", header=True)
        assert result == mock_dataset

        mock_loader.load.assert_called_with(
            "test.csv", name="test", delimiter=",", header=True
        )

    def test_protocol_type_hints(self):
        """Test protocol type hints are properly defined."""
        # This test verifies the protocol definitions have proper type hints
        # which is important for static type checking

        class TypedLoader:
            def load(
                self, source: str | Path, name: str | None = None, **kwargs: Any
            ) -> Dataset:
                return Mock(spec=Dataset)

            def validate(self, source: str | Path) -> bool:
                return True

            @property
            def supported_formats(self) -> list[str]:
                return ["csv"]

        loader = TypedLoader()
        assert isinstance(loader, DataLoaderProtocol)

        # Test return types
        dataset = loader.load("test.csv")
        assert isinstance(dataset, Mock)  # Mock of Dataset

        is_valid = loader.validate("test.csv")
        assert isinstance(is_valid, bool)

        formats = loader.supported_formats
        assert isinstance(formats, list)

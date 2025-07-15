"""Data loader protocol for infrastructure adapters."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pynomaly.domain.entities import Dataset


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """Protocol defining the interface for data loader implementations.

    This protocol must be implemented by all infrastructure adapters
    that load data from various sources (CSV, Parquet, databases, etc.).
    """

    def load(
        self, source: str | Path, name: str | None = None, **kwargs: Any
    ) -> Dataset:
        """Load data from a source into a Dataset.

        Args:
            source: Path or connection string to data source
            name: Optional name for the dataset
            **kwargs: Additional loader-specific arguments

        Returns:
            Loaded dataset
        """
        ...

    def validate(self, source: str | Path) -> bool:
        """Validate if the source can be loaded by this loader.

        Args:
            source: Path or connection string to validate

        Returns:
            True if source is valid for this loader
        """
        ...

    @property
    def supported_formats(self) -> list[str]:
        """Get list of supported file formats/source types."""
        ...


@runtime_checkable
class BatchDataLoaderProtocol(DataLoaderProtocol, Protocol):
    """Protocol for loaders that support batch/chunked loading."""

    def load_batch(
        self,
        source: str | Path,
        batch_size: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> Iterator[Dataset]:
        """Load data in batches.

        Args:
            source: Path or connection string to data source
            batch_size: Number of rows per batch
            name: Optional name prefix for datasets
            **kwargs: Additional loader-specific arguments

        Yields:
            Dataset batches
        """
        ...

    def estimate_size(self, source: str | Path) -> dict[str, Any]:
        """Estimate the size of the data source.

        Args:
            source: Path or connection string

        Returns:
            Dictionary with size information (rows, columns, memory, etc.)
        """
        ...


@runtime_checkable
class StreamingDataLoaderProtocol(DataLoaderProtocol, Protocol):
    """Protocol for loaders that support streaming data."""

    def stream(
        self, source: str | Path, window_size: int | None = None, **kwargs: Any
    ) -> Iterator[Dataset]:
        """Stream data from source.

        Args:
            source: Path or connection string to data source
            window_size: Optional sliding window size
            **kwargs: Additional loader-specific arguments

        Yields:
            Dataset windows/chunks
        """
        ...

    def connect(self, source: str | Path, **kwargs: Any) -> None:
        """Establish connection to streaming source.

        Args:
            source: Connection string or path
            **kwargs: Connection parameters
        """
        ...

    def disconnect(self) -> None:
        """Close connection to streaming source."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected to streaming source."""
        ...


@runtime_checkable
class DatabaseLoaderProtocol(DataLoaderProtocol, Protocol):
    """Protocol for database loaders."""

    def load_query(
        self,
        query: str,
        connection: str | Any,
        name: str | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """Load data using SQL query.

        Args:
            query: SQL query to execute
            connection: Database connection string or object
            name: Optional name for the dataset
            **kwargs: Additional query parameters

        Returns:
            Query result as dataset
        """
        ...

    def load_table(
        self,
        table_name: str,
        connection: str | Any,
        schema: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """Load entire table as dataset.

        Args:
            table_name: Name of the table
            connection: Database connection string or object
            schema: Optional database schema
            name: Optional name for the dataset
            **kwargs: Additional parameters

        Returns:
            Table data as dataset
        """
        ...

    def get_table_info(
        self, table_name: str, connection: str | Any, schema: str | None = None
    ) -> dict[str, Any]:
        """Get information about a database table.

        Args:
            table_name: Name of the table
            connection: Database connection
            schema: Optional database schema

        Returns:
            Table metadata (columns, types, row count, etc.)
        """
        ...

"""Data partitioning for distributed processing."""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .distributed_config import PartitionStrategy, get_distributed_config_manager

logger = logging.getLogger(__name__)


class PartitionType(str, Enum):
    """Types of data partitions."""

    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    INFERENCE = "inference"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class PartitionMetadata:
    """Metadata for data partitions."""

    # Basic information
    partition_id: str
    partition_type: PartitionType
    strategy: PartitionStrategy

    # Data characteristics
    total_samples: int
    feature_count: int
    data_size_bytes: int

    # Partition boundaries
    start_index: int
    end_index: int

    # Statistical information
    class_distribution: dict[str, int] = field(default_factory=dict)
    feature_statistics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Quality metrics
    balance_score: float = 0.0
    diversity_score: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Processing hints
    preferred_worker_capabilities: list[str] = field(default_factory=list)
    estimated_processing_time: float = 0.0
    memory_requirement_mb: float = 0.0


class DataPartition(BaseModel):
    """Represents a partition of data for distributed processing."""

    # Core information
    partition_id: str = Field(..., description="Unique partition identifier")
    data: Any = Field(..., description="Partition data (DataFrame, array, etc.)")
    metadata: PartitionMetadata = Field(..., description="Partition metadata")

    # Context information
    parent_dataset_id: str | None = Field(
        default=None, description="Parent dataset identifier"
    )
    chunk_index: int = Field(default=0, description="Index within chunk sequence")
    total_chunks: int = Field(default=1, description="Total number of chunks")

    # Processing state
    is_processed: bool = Field(
        default=False, description="Whether partition has been processed"
    )
    assigned_worker: str | None = Field(
        default=None, description="Assigned worker ID"
    )
    processing_started_at: datetime | None = Field(
        default=None, description="Processing start time"
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def size_mb(self) -> float:
        """Get partition size in MB."""
        return self.metadata.data_size_bytes / (1024 * 1024)

    @property
    def is_balanced(self) -> bool:
        """Check if partition is well-balanced."""
        return self.metadata.balance_score > 0.7

    def get_summary(self) -> dict[str, Any]:
        """Get partition summary information."""
        return {
            "partition_id": self.partition_id,
            "samples": self.metadata.total_samples,
            "features": self.metadata.feature_count,
            "size_mb": self.size_mb,
            "balance_score": self.metadata.balance_score,
            "diversity_score": self.metadata.diversity_score,
            "is_processed": self.is_processed,
            "assigned_worker": self.assigned_worker,
        }


class PartitioningStrategy(ABC):
    """Abstract base class for partitioning strategies."""

    @abstractmethod
    def partition(
        self, data: Any, num_partitions: int, **kwargs
    ) -> list[DataPartition]:
        """Partition data into chunks.

        Args:
            data: Data to partition
            num_partitions: Number of partitions to create
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of data partitions
        """
        pass

    @abstractmethod
    def estimate_partition_characteristics(
        self, data: Any, num_partitions: int
    ) -> dict[str, Any]:
        """Estimate characteristics of partitions without actually creating them.

        Args:
            data: Data to analyze
            num_partitions: Number of partitions

        Returns:
            Estimated characteristics
        """
        pass


class RoundRobinStrategy(PartitioningStrategy):
    """Round-robin partitioning strategy."""

    def partition(
        self, data: Any, num_partitions: int, **kwargs
    ) -> list[DataPartition]:
        """Partition data using round-robin assignment."""
        if isinstance(data, pd.DataFrame):
            return self._partition_dataframe(data, num_partitions, **kwargs)
        elif isinstance(data, np.ndarray):
            return self._partition_array(data, num_partitions, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _partition_dataframe(
        self, df: pd.DataFrame, num_partitions: int, **kwargs
    ) -> list[DataPartition]:
        """Partition pandas DataFrame."""
        partitions = []
        rows_per_partition = len(df) // num_partitions

        for i in range(num_partitions):
            start_idx = i * rows_per_partition
            if i == num_partitions - 1:
                # Last partition gets remaining rows
                end_idx = len(df)
            else:
                end_idx = (i + 1) * rows_per_partition

            partition_data = df.iloc[start_idx:end_idx].copy()

            # Create metadata
            metadata = PartitionMetadata(
                partition_id=f"round_robin_{i}",
                partition_type=PartitionType.BATCH,
                strategy=PartitionStrategy.ROUND_ROBIN,
                total_samples=len(partition_data),
                feature_count=len(partition_data.columns),
                data_size_bytes=partition_data.memory_usage(deep=True).sum(),
                start_index=start_idx,
                end_index=end_idx - 1,
            )

            # Calculate statistics
            self._calculate_dataframe_statistics(partition_data, metadata)

            # Create partition
            partition = DataPartition(
                partition_id=metadata.partition_id,
                data=partition_data,
                metadata=metadata,
                chunk_index=i,
                total_chunks=num_partitions,
            )

            partitions.append(partition)

        return partitions

    def _partition_array(
        self, arr: np.ndarray, num_partitions: int, **kwargs
    ) -> list[DataPartition]:
        """Partition numpy array."""
        partitions = []
        rows_per_partition = len(arr) // num_partitions

        for i in range(num_partitions):
            start_idx = i * rows_per_partition
            if i == num_partitions - 1:
                end_idx = len(arr)
            else:
                end_idx = (i + 1) * rows_per_partition

            partition_data = arr[start_idx:end_idx].copy()

            # Create metadata
            metadata = PartitionMetadata(
                partition_id=f"round_robin_{i}",
                partition_type=PartitionType.BATCH,
                strategy=PartitionStrategy.ROUND_ROBIN,
                total_samples=len(partition_data),
                feature_count=partition_data.shape[1] if partition_data.ndim > 1 else 1,
                data_size_bytes=partition_data.nbytes,
                start_index=start_idx,
                end_index=end_idx - 1,
            )

            # Calculate statistics
            self._calculate_array_statistics(partition_data, metadata)

            # Create partition
            partition = DataPartition(
                partition_id=metadata.partition_id,
                data=partition_data,
                metadata=metadata,
                chunk_index=i,
                total_chunks=num_partitions,
            )

            partitions.append(partition)

        return partitions

    def estimate_partition_characteristics(
        self, data: Any, num_partitions: int
    ) -> dict[str, Any]:
        """Estimate round-robin partition characteristics."""
        if isinstance(data, pd.DataFrame):
            total_samples = len(data)
            len(data.columns)
            data_size = data.memory_usage(deep=True).sum()
        elif isinstance(data, np.ndarray):
            total_samples = len(data)
            data.shape[1] if data.ndim > 1 else 1
            data_size = data.nbytes
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        avg_samples_per_partition = total_samples // num_partitions
        avg_size_per_partition = data_size // num_partitions

        return {
            "strategy": "round_robin",
            "total_samples": total_samples,
            "avg_samples_per_partition": avg_samples_per_partition,
            "avg_size_mb_per_partition": avg_size_per_partition / (1024 * 1024),
            "expected_balance_score": 0.9,  # Round-robin is naturally balanced
            "estimated_processing_time": avg_samples_per_partition
            * 0.001,  # 1ms per sample estimate
        }

    def _calculate_dataframe_statistics(
        self, df: pd.DataFrame, metadata: PartitionMetadata
    ) -> None:
        """Calculate statistics for DataFrame partition."""
        try:
            # Basic feature statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                metadata.feature_statistics[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }

            # Class distribution (if target column exists)
            if "target" in df.columns or "label" in df.columns:
                target_col = "target" if "target" in df.columns else "label"
                metadata.class_distribution = df[target_col].value_counts().to_dict()

                # Calculate balance score
                if metadata.class_distribution:
                    class_counts = list(metadata.class_distribution.values())
                    min_count = min(class_counts)
                    max_count = max(class_counts)
                    metadata.balance_score = (
                        min_count / max_count if max_count > 0 else 0.0
                    )

            # Diversity score (simplified)
            metadata.diversity_score = min(1.0, len(df.columns) / 100.0)

        except Exception as e:
            logger.warning(f"Could not calculate statistics: {e}")

    def _calculate_array_statistics(
        self, arr: np.ndarray, metadata: PartitionMetadata
    ) -> None:
        """Calculate statistics for array partition."""
        try:
            if arr.ndim > 1:
                for i in range(arr.shape[1]):
                    col_data = arr[:, i]
                    metadata.feature_statistics[f"feature_{i}"] = {
                        "mean": float(np.mean(col_data)),
                        "std": float(np.std(col_data)),
                        "min": float(np.min(col_data)),
                        "max": float(np.max(col_data)),
                    }

            # Simple balance and diversity scores
            metadata.balance_score = 0.9  # Assume balanced for arrays
            metadata.diversity_score = (
                min(1.0, arr.shape[1] / 100.0) if arr.ndim > 1 else 0.5
            )

        except Exception as e:
            logger.warning(f"Could not calculate array statistics: {e}")


class HashBasedStrategy(PartitioningStrategy):
    """Hash-based partitioning strategy."""

    def __init__(self, hash_column: str | None = None):
        """Initialize hash-based strategy.

        Args:
            hash_column: Column to use for hashing (default: index)
        """
        self.hash_column = hash_column

    def partition(
        self, data: Any, num_partitions: int, **kwargs
    ) -> list[DataPartition]:
        """Partition data using hash-based assignment."""
        if isinstance(data, pd.DataFrame):
            return self._partition_dataframe(data, num_partitions, **kwargs)
        else:
            raise ValueError("Hash-based partitioning only supports DataFrames")

    def _partition_dataframe(
        self, df: pd.DataFrame, num_partitions: int, **kwargs
    ) -> list[DataPartition]:
        """Partition DataFrame using hash-based strategy."""
        partitions = [[] for _ in range(num_partitions)]

        # Determine hash column
        hash_col = self.hash_column
        if hash_col is None or hash_col not in df.columns:
            # Use index as hash source
            hash_values = df.index.astype(str)
        else:
            hash_values = df[hash_col].astype(str)

        # Assign rows to partitions based on hash
        for idx, hash_val in enumerate(hash_values):
            hash_int = int(hashlib.md5(hash_val.encode()).hexdigest(), 16)
            partition_idx = hash_int % num_partitions
            partitions[partition_idx].append(idx)

        # Create partition objects
        result_partitions = []
        for i, row_indices in enumerate(partitions):
            if not row_indices:
                # Empty partition
                partition_data = df.iloc[
                    0:0
                ].copy()  # Empty DataFrame with same structure
            else:
                partition_data = df.iloc[row_indices].copy()

            # Create metadata
            metadata = PartitionMetadata(
                partition_id=f"hash_based_{i}",
                partition_type=PartitionType.BATCH,
                strategy=PartitionStrategy.HASH_BASED,
                total_samples=len(partition_data),
                feature_count=len(partition_data.columns),
                data_size_bytes=partition_data.memory_usage(deep=True).sum(),
                start_index=min(row_indices) if row_indices else 0,
                end_index=max(row_indices) if row_indices else 0,
            )

            # Calculate statistics
            self._calculate_dataframe_statistics(partition_data, metadata)

            # Create partition
            partition = DataPartition(
                partition_id=metadata.partition_id,
                data=partition_data,
                metadata=metadata,
                chunk_index=i,
                total_chunks=num_partitions,
            )

            result_partitions.append(partition)

        return result_partitions

    def estimate_partition_characteristics(
        self, data: Any, num_partitions: int
    ) -> dict[str, Any]:
        """Estimate hash-based partition characteristics."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Hash-based partitioning only supports DataFrames")

        total_samples = len(data)
        len(data.columns)
        data_size = data.memory_usage(deep=True).sum()

        # Hash-based partitioning can be uneven
        avg_samples_per_partition = total_samples // num_partitions

        return {
            "strategy": "hash_based",
            "total_samples": total_samples,
            "avg_samples_per_partition": avg_samples_per_partition,
            "avg_size_mb_per_partition": (data_size // num_partitions) / (1024 * 1024),
            "expected_balance_score": 0.7,  # Hash-based can be less balanced
            "estimated_processing_time": avg_samples_per_partition * 0.001,
        }

    def _calculate_dataframe_statistics(
        self, df: pd.DataFrame, metadata: PartitionMetadata
    ) -> None:
        """Calculate statistics for DataFrame partition."""
        # Reuse implementation from RoundRobinStrategy
        round_robin = RoundRobinStrategy()
        round_robin._calculate_dataframe_statistics(df, metadata)


class SizeBasedStrategy(PartitioningStrategy):
    """Size-based partitioning strategy."""

    def __init__(self, target_size_mb: float = 100.0):
        """Initialize size-based strategy.

        Args:
            target_size_mb: Target size per partition in MB
        """
        self.target_size_mb = target_size_mb

    def partition(
        self, data: Any, num_partitions: int, **kwargs
    ) -> list[DataPartition]:
        """Partition data based on target size."""
        target_size_bytes = self.target_size_mb * 1024 * 1024

        if isinstance(data, pd.DataFrame):
            return self._partition_dataframe_by_size(data, target_size_bytes, **kwargs)
        elif isinstance(data, np.ndarray):
            return self._partition_array_by_size(data, target_size_bytes, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _partition_dataframe_by_size(
        self, df: pd.DataFrame, target_size_bytes: int, **kwargs
    ) -> list[DataPartition]:
        """Partition DataFrame by target size."""
        partitions = []
        current_start = 0
        partition_index = 0

        while current_start < len(df):
            # Estimate rows for target size
            sample_size = min(1000, len(df) - current_start)
            sample_df = df.iloc[current_start : current_start + sample_size]
            bytes_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)

            target_rows = max(1, int(target_size_bytes / bytes_per_row))
            current_end = min(current_start + target_rows, len(df))

            # Create partition
            partition_data = df.iloc[current_start:current_end].copy()

            # Create metadata
            metadata = PartitionMetadata(
                partition_id=f"size_based_{partition_index}",
                partition_type=PartitionType.BATCH,
                strategy=PartitionStrategy.SIZE_BASED,
                total_samples=len(partition_data),
                feature_count=len(partition_data.columns),
                data_size_bytes=partition_data.memory_usage(deep=True).sum(),
                start_index=current_start,
                end_index=current_end - 1,
            )

            # Calculate statistics
            self._calculate_dataframe_statistics(partition_data, metadata)

            # Create partition
            partition = DataPartition(
                partition_id=metadata.partition_id,
                data=partition_data,
                metadata=metadata,
                chunk_index=partition_index,
                total_chunks=-1,  # Unknown total chunks
            )

            partitions.append(partition)

            current_start = current_end
            partition_index += 1

        # Update total chunks
        for partition in partitions:
            partition.total_chunks = len(partitions)

        return partitions

    def _partition_array_by_size(
        self, arr: np.ndarray, target_size_bytes: int, **kwargs
    ) -> list[DataPartition]:
        """Partition array by target size."""
        partitions = []
        bytes_per_row = arr.dtype.itemsize * (arr.shape[1] if arr.ndim > 1 else 1)
        target_rows = max(1, int(target_size_bytes / bytes_per_row))

        current_start = 0
        partition_index = 0

        while current_start < len(arr):
            current_end = min(current_start + target_rows, len(arr))
            partition_data = arr[current_start:current_end].copy()

            # Create metadata
            metadata = PartitionMetadata(
                partition_id=f"size_based_{partition_index}",
                partition_type=PartitionType.BATCH,
                strategy=PartitionStrategy.SIZE_BASED,
                total_samples=len(partition_data),
                feature_count=partition_data.shape[1] if partition_data.ndim > 1 else 1,
                data_size_bytes=partition_data.nbytes,
                start_index=current_start,
                end_index=current_end - 1,
            )

            # Calculate statistics
            self._calculate_array_statistics(partition_data, metadata)

            # Create partition
            partition = DataPartition(
                partition_id=metadata.partition_id,
                data=partition_data,
                metadata=metadata,
                chunk_index=partition_index,
                total_chunks=-1,
            )

            partitions.append(partition)

            current_start = current_end
            partition_index += 1

        # Update total chunks
        for partition in partitions:
            partition.total_chunks = len(partitions)

        return partitions

    def estimate_partition_characteristics(
        self, data: Any, num_partitions: int
    ) -> dict[str, Any]:
        """Estimate size-based partition characteristics."""
        if isinstance(data, pd.DataFrame):
            total_size = data.memory_usage(deep=True).sum()
            total_samples = len(data)
        elif isinstance(data, np.ndarray):
            total_size = data.nbytes
            total_samples = len(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        target_size_bytes = self.target_size_mb * 1024 * 1024
        estimated_partitions = max(1, int(total_size / target_size_bytes))
        avg_samples_per_partition = total_samples // estimated_partitions

        return {
            "strategy": "size_based",
            "total_samples": total_samples,
            "total_size_mb": total_size / (1024 * 1024),
            "estimated_partitions": estimated_partitions,
            "target_size_mb": self.target_size_mb,
            "avg_samples_per_partition": avg_samples_per_partition,
            "expected_balance_score": 0.95,  # Size-based is very balanced
            "estimated_processing_time": avg_samples_per_partition * 0.001,
        }

    def _calculate_dataframe_statistics(
        self, df: pd.DataFrame, metadata: PartitionMetadata
    ) -> None:
        """Calculate statistics for DataFrame partition."""
        round_robin = RoundRobinStrategy()
        round_robin._calculate_dataframe_statistics(df, metadata)

    def _calculate_array_statistics(
        self, arr: np.ndarray, metadata: PartitionMetadata
    ) -> None:
        """Calculate statistics for array partition."""
        round_robin = RoundRobinStrategy()
        round_robin._calculate_array_statistics(arr, metadata)


class DataPartitioner:
    """Main data partitioning interface."""

    def __init__(self, strategy: PartitionStrategy | None = None):
        """Initialize data partitioner.

        Args:
            strategy: Partitioning strategy to use
        """
        self.strategy = (
            strategy
            or get_distributed_config_manager()
            .get_effective_config()
            .partition_strategy
        )
        self._strategies = self._initialize_strategies()

    def _initialize_strategies(self) -> dict[PartitionStrategy, PartitioningStrategy]:
        """Initialize available partitioning strategies."""
        return {
            PartitionStrategy.ROUND_ROBIN: RoundRobinStrategy(),
            PartitionStrategy.HASH_BASED: HashBasedStrategy(),
            PartitionStrategy.SIZE_BASED: SizeBasedStrategy(),
            PartitionStrategy.ADAPTIVE: RoundRobinStrategy(),  # Default to round-robin for adaptive
        }

    def partition_data(
        self,
        data: Any,
        num_partitions: int | None = None,
        strategy: PartitionStrategy | None = None,
        **kwargs,
    ) -> list[DataPartition]:
        """Partition data into chunks for distributed processing.

        Args:
            data: Data to partition
            num_partitions: Number of partitions (auto-determined if None)
            strategy: Partitioning strategy (uses default if None)
            **kwargs: Strategy-specific parameters

        Returns:
            List of data partitions
        """
        # Determine strategy
        used_strategy = strategy or self.strategy
        partitioning_strategy = self._strategies.get(used_strategy)

        if not partitioning_strategy:
            raise ValueError(f"Unknown partitioning strategy: {used_strategy}")

        # Determine number of partitions
        if num_partitions is None:
            num_partitions = self._auto_determine_partitions(data)

        logger.info(
            f"Partitioning data using {used_strategy} strategy into {num_partitions} partitions"
        )

        # Create partitions
        partitions = partitioning_strategy.partition(data, num_partitions, **kwargs)

        # Log partition statistics
        total_samples = sum(p.metadata.total_samples for p in partitions)
        total_size_mb = sum(p.size_mb for p in partitions)
        avg_balance = sum(p.metadata.balance_score for p in partitions) / len(
            partitions
        )

        logger.info(
            f"Created {len(partitions)} partitions: "
            f"total_samples={total_samples}, "
            f"total_size_mb={total_size_mb:.2f}, "
            f"avg_balance={avg_balance:.3f}"
        )

        return partitions

    def estimate_partitioning(
        self,
        data: Any,
        num_partitions: int | None = None,
        strategy: PartitionStrategy | None = None,
    ) -> dict[str, Any]:
        """Estimate partitioning characteristics without creating actual partitions.

        Args:
            data: Data to analyze
            num_partitions: Number of partitions
            strategy: Partitioning strategy

        Returns:
            Estimated characteristics
        """
        used_strategy = strategy or self.strategy
        partitioning_strategy = self._strategies.get(used_strategy)

        if not partitioning_strategy:
            raise ValueError(f"Unknown partitioning strategy: {used_strategy}")

        if num_partitions is None:
            num_partitions = self._auto_determine_partitions(data)

        return partitioning_strategy.estimate_partition_characteristics(
            data, num_partitions
        )

    def _auto_determine_partitions(self, data: Any) -> int:
        """Automatically determine optimal number of partitions.

        Args:
            data: Data to partition

        Returns:
            Optimal number of partitions
        """
        # Get data characteristics
        if isinstance(data, pd.DataFrame):
            total_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
            total_samples = len(data)
        elif isinstance(data, np.ndarray):
            total_size_mb = data.nbytes / (1024 * 1024)
            total_samples = len(data)
        else:
            # Default for unknown types
            return 4

        # Target partition size (configurable)
        target_partition_size_mb = 100.0  # 100MB per partition
        min_samples_per_partition = 1000
        max_partitions = 32

        # Calculate based on size
        size_based_partitions = max(1, int(total_size_mb / target_partition_size_mb))

        # Calculate based on samples
        sample_based_partitions = max(1, int(total_samples / min_samples_per_partition))

        # Use the smaller of the two, capped at max_partitions
        optimal_partitions = min(
            max_partitions, min(size_based_partitions, sample_based_partitions)
        )

        logger.debug(
            f"Auto-determined {optimal_partitions} partitions for data: "
            f"size={total_size_mb:.2f}MB, samples={total_samples}"
        )

        return optimal_partitions

    def get_partition_iterator(
        self, partitions: list[DataPartition]
    ) -> Iterator[DataPartition]:
        """Get iterator over partitions for processing.

        Args:
            partitions: List of partitions

        Yields:
            Data partitions in processing order
        """
        # Sort by chunk index for ordered processing
        sorted_partitions = sorted(partitions, key=lambda p: p.chunk_index)

        yield from sorted_partitions

    def merge_partition_results(
        self, partition_results: list[tuple[DataPartition, Any]]
    ) -> Any:
        """Merge results from multiple partitions.

        Args:
            partition_results: List of (partition, result) tuples

        Returns:
            Merged result
        """
        if not partition_results:
            return None

        # Sort by chunk index
        sorted_results = sorted(partition_results, key=lambda x: x[0].chunk_index)

        # Extract just the results
        results = [result for _, result in sorted_results]

        # Simple concatenation for now (can be enhanced for specific result types)
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif isinstance(results[0], np.ndarray):
            return np.concatenate(results, axis=0)
        elif isinstance(results[0], list):
            merged = []
            for result in results:
                merged.extend(result)
            return merged
        elif isinstance(results[0], dict):
            # Merge dictionaries (assuming similar structure)
            merged = {}
            for result in results:
                for key, value in result.items():
                    if key not in merged:
                        merged[key] = []
                    if isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        merged[key].append(value)
            return merged
        else:
            # Return list of results if no specific merge strategy
            return results

    def get_statistics(self, partitions: list[DataPartition]) -> dict[str, Any]:
        """Get statistics for a set of partitions.

        Args:
            partitions: List of partitions

        Returns:
            Partition statistics
        """
        if not partitions:
            return {}

        total_samples = sum(p.metadata.total_samples for p in partitions)
        total_size_mb = sum(p.size_mb for p in partitions)
        avg_balance = sum(p.metadata.balance_score for p in partitions) / len(
            partitions
        )
        avg_diversity = sum(p.metadata.diversity_score for p in partitions) / len(
            partitions
        )

        size_distribution = [p.size_mb for p in partitions]
        sample_distribution = [p.metadata.total_samples for p in partitions]

        return {
            "total_partitions": len(partitions),
            "total_samples": total_samples,
            "total_size_mb": total_size_mb,
            "avg_samples_per_partition": total_samples / len(partitions),
            "avg_size_mb_per_partition": total_size_mb / len(partitions),
            "avg_balance_score": avg_balance,
            "avg_diversity_score": avg_diversity,
            "size_distribution": {
                "min": min(size_distribution),
                "max": max(size_distribution),
                "std": float(np.std(size_distribution)),
            },
            "sample_distribution": {
                "min": min(sample_distribution),
                "max": max(sample_distribution),
                "std": float(np.std(sample_distribution)),
            },
        }

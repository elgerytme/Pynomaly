"""Memory optimization service for efficient anomaly processing.

This service provides high-level memory optimization capabilities for
anomaly processing workflows, integrating with the feature flag system.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ...domain.entities import Dataset, DetectionResult
from ...infrastructure.config.feature_flags import require_feature
from ...infrastructure.data_processing import (
    LargeDatasetAnalyzer,
    MemoryOptimizedDataLoader,
    StreamingDataProcessor,
    get_memory_usage,
)


class MemoryOptimizationService:
    """Service for memory-optimized anomaly processing operations."""

    def __init__(
        self,
        chunk_size: int = 10000,
        memory_limit_mb: int = 500,
        enable_streaming: bool = True,
    ):
        """Initialize memory optimization service.

        Args:
            chunk_size: Size of data chunks for streaming processing
            memory_limit_mb: Memory limit for processing operations
            enable_streaming: Whether to enable streaming for large datasets
        """
        self.streaming_processor = StreamingDataProcessor(
            chunk_size=chunk_size, memory_limit_mb=memory_limit_mb
        )
        self.data_loader = MemoryOptimizedDataLoader(self.streaming_processor)
        self.data_collection_analyzer = LargeDatasetAnalyzer(self.streaming_processor)
        self.enable_streaming = enable_streaming

        # Memory optimization statistics
        self.optimization_stats = {
            "datasets_optimized": 0,
            "memory_saved_mb": 0.0,
            "streaming_operations": 0,
        }

    @require_feature("memory_efficiency")
    def optimize_dataset_for_detection(
        self,
        data_collection: DataCollection | str | pd.DataFrame,
        target_memory_mb: int | None = None,
    ) -> tuple[DataCollection, dict[str, Any]]:
        """Optimize data_collection for memory-efficient anomaly processing.

        Args:
            data_collection: Input data_collection or path to data
            target_memory_mb: Target memory usage limit

        Returns:
            Tuple of (optimized_data_collection, optimization_info)
        """
        memory_before = get_memory_usage()
        optimization_info = {
            "original_memory_mb": memory_before,
            "optimization_applied": [],
            "streaming_used": False,
            "memory_reduction_percent": 0.0,
        }

        # Load data_collection efficiently
        if not isinstance(data_collection, DataCollection):
            data_collection = self.data_loader.load_data_collection_efficiently(
                data_collection, target_memory_mb
            )
            optimization_info["optimization_applied"].append("efficient_loading")

        # Check if streaming is needed
        if hasattr(data_collection.data, "nbytes"):
            data_collection_size_mb = data_collection.data.nbytes / 1024 / 1024
        else:
            # Estimate size for other data types
            data_collection_size_mb = (
                len(data_collection.data) * 8 / 1024 / 1024
            )  # Assume 8 bytes per element
        if self.enable_streaming and data_collection_size_mb > (target_memory_mb or 100):
            optimization_info["streaming_used"] = True
            optimization_info["optimization_applied"].append("streaming_enabled")

        # Apply memory optimizations to data
        if isinstance(data_collection.data, np.ndarray):
            optimized_data = self._optimize_numpy_array(data_collection.data)
            if optimized_data.nbytes < data_collection.data.nbytes:
                data_collection.data = optimized_data
                optimization_info["optimization_applied"].append("dtype_optimization")

        memory_after = get_memory_usage()
        memory_saved = memory_before - memory_after

        if memory_saved > 0:
            optimization_info["memory_reduction_percent"] = (
                memory_saved / memory_before
            ) * 100
            self.optimization_stats["memory_saved_mb"] += memory_saved

        optimization_info["final_memory_mb"] = memory_after
        self.optimization_stats["datasets_optimized"] += 1

        return data_collection, optimization_info

    @require_feature("memory_efficiency")
    def stream_anomaly_detection(
        self,
        data_source: str | pd.DataFrame | np.ndarray,
        detector_func: callable,
        aggregation_strategy: str = "concatenate",
    ) -> DetectionResult:
        """Perform anomaly processing on large datasets using streaming.

        Args:
            data_source: Large data_collection source
            detector_func: Function that performs processing on chunks
            aggregation_strategy: How to combine chunk results

        Returns:
            Aggregated processing result
        """
        self.optimization_stats["streaming_operations"] += 1

        chunk_results = []
        total_samples = 0

        # Process data in chunks
        for chunk_data_collection in self.streaming_processor.process_large_data_collection(
            data_source
        ):
            try:
                chunk_result = detector_func(chunk_data_collection)
                chunk_results.append(chunk_result)
                total_samples += len(chunk_data_collection.data)

            except Exception as e:
                warnings.warn(f"Failed to process chunk: {e}", stacklevel=2)
                continue

        # Aggregate results
        if not chunk_results:
            raise RuntimeError("No chunks were successfully processed")

        return self._aggregate_processing_results(
            chunk_results, aggregation_strategy, total_samples
        )

    @require_feature("memory_efficiency")
    def analyze_large_dataset_characteristics(
        self, data_source: str | pd.DataFrame | np.ndarray
    ) -> dict[str, Any]:
        """Analyze characteristics of large datasets efficiently.

        Args:
            data_source: Large data_collection to analyze

        Returns:
            DataCollection characteristics and recommendations
        """
        # Get basic statistics
        stats = self.data_collection_analyzer.analyze_data_collection_statistics(data_source)

        # Detect potential anomaly candidates
        candidates = self.data_collection_analyzer.detect_anomaly_candidates(data_source)

        # Generate recommendations
        recommendations = self._generate_memory_recommendations(stats)

        return {
            "statistics": stats,
            "anomaly_candidates": candidates,
            "memory_recommendations": recommendations,
            "processing_feasible": stats["memory_estimate_mb"] < 2000,  # 2GB limit
        }

    @require_feature("memory_efficiency")
    def recommend_optimal_configuration(
        self, data_collection_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Recommend optimal configuration for memory-efficient processing.

        Args:
            data_collection_info: Information about the data_collection

        Returns:
            Recommended configuration parameters
        """
        stats = data_collection_info.get("statistics", {})
        memory_estimate = stats.get("memory_estimate_mb", 0)
        total_rows = stats.get("total_rows", 0)

        config = {
            "chunk_size": 10000,
            "memory_limit_mb": 500,
            "enable_streaming": False,
            "recommended_algorithm": "isolation_forest",  # Memory efficient default
            "batch_size": 1000,
            "optimization_level": "standard",
        }

        # Adjust based on data_collection size
        if memory_estimate > 1000:  # > 1GB
            config.update(
                {
                    "chunk_size": 5000,
                    "enable_streaming": True,
                    "optimization_level": "aggressive",
                    "recommended_algorithm": "local_outlier_factor",  # More memory efficient
                }
            )
        elif memory_estimate > 2000:  # > 2GB
            config.update(
                {
                    "chunk_size": 2000,
                    "memory_limit_mb": 1000,
                    "batch_size": 500,
                    "optimization_level": "maximum",
                }
            )

        # Adjust based on row count
        if total_rows > 1000000:  # > 1M rows
            config["chunk_size"] = min(config["chunk_size"], 5000)
            config["enable_streaming"] = True

        return config

    def get_optimization_statistics(self) -> dict[str, Any]:
        """Get memory optimization statistics."""
        return {
            **self.optimization_stats,
            "current_memory_usage_mb": get_memory_usage(),
            "streaming_processor_chunks": self.streaming_processor._processed_chunks,
            "total_memory_used_mb": self.streaming_processor._total_memory_used,
        }

    def _optimize_numpy_array(self, array: np.ndarray) -> np.ndarray:
        """Optimize numpy array memory usage."""
        if array.dtype == np.float64:
            # Check if we can downcast to float32
            if np.allclose(array, array.astype(np.float32), rtol=1e-6):
                return array.astype(np.float32)
        elif array.dtype in [np.int64, np.int32]:
            # Try to downcast integers
            if array.min() >= 0:
                # Unsigned integers
                if array.max() < 256:
                    return array.astype(np.uint8)
                elif array.max() < 65536:
                    return array.astype(np.uint16)
            else:
                # Signed integers
                if array.min() >= -128 and array.max() <= 127:
                    return array.astype(np.int8)
                elif array.min() >= -32768 and array.max() <= 32767:
                    return array.astype(np.int16)

        return array

    def _aggregate_detection_results(
        self, chunk_results: list[DetectionResult], strategy: str, total_samples: int
    ) -> DetectionResult:
        """Aggregate processing results from multiple chunks."""
        if strategy == "concatenate":
            # Concatenate all scores and labels
            all_scores = np.concatenate([result.scores for result in chunk_results])
            all_labels = np.concatenate([result.labels for result in chunk_results])

            return DetectionResult(
                scores=all_scores,
                labels=all_labels,
                threshold=chunk_results[0].threshold,  # Use first chunk's threshold
                metadata={
                    "aggregation_strategy": strategy,
                    "chunk_count": len(chunk_results),
                    "total_samples": total_samples,
                },
            )

        elif strategy == "average_scores":
            # Average scores across chunks (assuming aligned samples)
            avg_scores = np.mean([result.scores for result in chunk_results], axis=0)
            # Use majority vote for labels
            all_labels = np.array([result.labels for result in chunk_results])
            majority_labels = np.round(np.mean(all_labels, axis=0)).astype(int)

            return DetectionResult(
                scores=avg_scores,
                labels=majority_labels,
                threshold=np.mean([result.threshold for result in chunk_results]),
                metadata={
                    "aggregation_strategy": strategy,
                    "chunk_count": len(chunk_results),
                },
            )

        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def _generate_memory_recommendations(self, stats: dict[str, Any]) -> list[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        memory_estimate = stats.get("memory_estimate_mb", 0)
        total_columns = stats.get("total_columns", 0)

        if memory_estimate > 500:
            recommendations.append("Enable streaming processing for large data_collection")

        if memory_estimate > 1000:
            recommendations.append("Consider reducing chunk size to 5000 or less")
            recommendations.append("Use aggressive memory optimization")

        if total_columns > 100:
            recommendations.append(
                "Consider feature selection to reduce dimensionality"
            )

        if memory_estimate > 2000:
            recommendations.append("DataCollection may require distributed processing")
            recommendations.append("Consider sampling for initial analysis")

        return recommendations


class MemoryProfiler:
    """Profiler for memory usage during anomaly processing."""

    def __init__(self):
        """Initialize memory profiler."""
        self.profile_data = []
        self.baseline_memory = get_memory_usage()

    def start_profiling(self, operation_name: str) -> None:
        """Start profiling a memory operation."""
        self.profile_data.append(
            {
                "operation": operation_name,
                "start_memory": get_memory_usage(),
                "start_time": pd.Timestamp.now(),
            }
        )

    def end_profiling(self, operation_name: str) -> dict[str, Any]:
        """End profiling and return results."""
        end_memory = get_memory_usage()
        end_time = pd.Timestamp.now()

        # Find matching start record
        start_record = None
        for record in reversed(self.profile_data):
            if record["operation"] == operation_name and "end_memory" not in record:
                start_record = record
                break

        if start_record is None:
            raise ValueError(
                f"No matching start record for operation: {operation_name}"
            )

        # Update record with end data
        start_record.update(
            {
                "end_memory": end_memory,
                "end_time": end_time,
                "memory_delta": end_memory - start_record["start_memory"],
                "duration_seconds": (
                    end_time - start_record["start_time"]
                ).total_seconds(),
            }
        )

        return start_record.copy()

    def get_profile_summary(self) -> dict[str, Any]:
        """Get summary of memory profiling results."""
        completed_profiles = [p for p in self.profile_data if "end_memory" in p]

        if not completed_profiles:
            return {"message": "No completed profiles available"}

        total_memory_delta = sum(p["memory_delta"] for p in completed_profiles)
        max_memory_usage = max(p["end_memory"] for p in completed_profiles)

        return {
            "total_operations": len(completed_profiles),
            "total_memory_delta_mb": total_memory_delta,
            "max_memory_usage_mb": max_memory_usage,
            "baseline_memory_mb": self.baseline_memory,
            "operations": completed_profiles,
        }

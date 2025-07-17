"""Memory optimization utilities for anomaly detection."""

from __future__ import annotations

from typing import Any, Dict, Optional, Generator
import numpy as np
import numpy.typing as npt
import gc


class MemoryOptimizer:
    """Memory optimization utilities for large-scale anomaly detection."""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics (simplified)."""
        # Simplified memory tracking without psutil dependency
        return {
            "rss_mb": 0.0,  # Would need psutil for actual values
            "vms_mb": 0.0,
            "percent": 0.0,
            "available_mb": 1000.0  # Placeholder
        }

    @staticmethod
    def optimize_array_dtype(data: npt.NDArray) -> npt.NDArray:
        """Optimize array data type to reduce memory usage."""
        if data.dtype == np.float64:
            # Check if we can safely convert to float32
            if np.all(np.isfinite(data)) and np.max(np.abs(data)) < 1e30:
                return data.astype(np.float32)
        return data

    @staticmethod
    def create_memory_efficient_batches(
        data: npt.NDArray[np.floating],
        batch_size: int,
        overlap: int = 0
    ) -> Generator[npt.NDArray[np.floating], None, None]:
        """Create memory-efficient batches that don't load all data at once."""
        start = 0
        while start < len(data):
            end = min(start + batch_size, len(data))
            yield data[start:end].copy()  # Copy to avoid reference issues
            start = end - overlap
            if start >= len(data):
                break

    @staticmethod
    def cleanup_memory() -> Dict[str, Any]:
        """Force garbage collection and return memory stats."""
        before = MemoryOptimizer.get_memory_usage()
        gc.collect()
        after = MemoryOptimizer.get_memory_usage()
        
        return {
            "before_mb": before["rss_mb"],
            "after_mb": after["rss_mb"],
            "freed_mb": before["rss_mb"] - after["rss_mb"]
        }
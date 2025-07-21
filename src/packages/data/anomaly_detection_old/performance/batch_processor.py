"""Optimized batch processing for large datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterator
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

from simplified_services.core_detection_service import CoreDetectionService, DetectionResult


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 1000
    parallel: bool = True
    max_workers: Optional[int] = None
    use_processes: bool = False  # Use processes instead of threads
    memory_efficient: bool = True
    overlap_ratio: float = 0.0  # Overlap between batches (0.0 to 0.5)


class BatchProcessor:
    """Optimized batch processor for large-scale anomaly detection.
    
    This class addresses the performance issues identified in Phase 1 by:
    - Implementing efficient batch processing strategies
    - Supporting parallel processing with threads/processes
    - Memory optimization for large datasets
    - Intelligent batch size selection
    - Result aggregation and consistency checking
    """

    def __init__(self):
        """Initialize batch processor."""
        self.detection_service = CoreDetectionService()
        self._processing_stats: List[Dict[str, Any]] = []

    def process_large_dataset(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest",
        contamination: float = 0.1,
        config: Optional[BatchConfig] = None,
        **kwargs: Any
    ) -> DetectionResult:
        """Process large dataset efficiently using batch processing.
        
        Args:
            data: Input data array
            algorithm: Algorithm to use
            contamination: Expected contamination rate
            config: Batch processing configuration
            **kwargs: Additional algorithm parameters
            
        Returns:
            Combined DetectionResult for entire dataset
        """
        if config is None:
            config = self._auto_configure_batching(data)
        
        print(f"📦 Batch: Processing {len(data)} samples in batches of {config.batch_size}")
        start_time = time.time()
        
        # Create batches
        batches = self._create_batches(data, config)
        
        # Process batches
        if config.parallel and len(batches) > 1:
            batch_results = self._process_batches_parallel(
                batches, algorithm, contamination, config, **kwargs
            )
        else:
            batch_results = self._process_batches_sequential(
                batches, algorithm, contamination, **kwargs
            )
        
        # Combine results
        combined_result = self._combine_batch_results(
            batch_results, algorithm, contamination, data.shape
        )
        
        combined_result.execution_time = time.time() - start_time
        
        # Track performance
        self._processing_stats.append({
            "timestamp": time.time(),
            "data_shape": data.shape,
            "batch_size": config.batch_size,
            "n_batches": len(batches),
            "parallel": config.parallel,
            "execution_time": combined_result.execution_time,
            "throughput": len(data) / combined_result.execution_time
        })
        
        print(f"✅ Batch: Completed in {combined_result.execution_time:.3f}s "
              f"({len(data) / combined_result.execution_time:.0f} samples/sec)")
        
        return combined_result

    def _auto_configure_batching(self, data: npt.NDArray[np.floating]) -> BatchConfig:
        """Automatically configure batch processing based on data characteristics."""
        n_samples, n_features = data.shape
        
        # Estimate memory usage (rough approximation)
        estimated_memory_mb = (n_samples * n_features * 8) / (1024 * 1024)  # 8 bytes per float64
        
        # Determine optimal batch size
        if n_samples <= 1000:
            batch_size = n_samples  # Process all at once
            parallel = False
        elif n_samples <= 10000:
            batch_size = 2000
            parallel = True
        elif estimated_memory_mb > 1000:  # Large memory usage
            batch_size = 1000
            parallel = True
        else:
            batch_size = 5000
            parallel = True
        
        # Use processes for CPU-intensive algorithms with large datasets
        use_processes = n_samples > 50000 and n_features > 10
        
        max_workers = min(mp.cpu_count(), 4) if parallel else 1
        
        return BatchConfig(
            batch_size=batch_size,
            parallel=parallel,
            max_workers=max_workers,
            use_processes=use_processes,
            memory_efficient=estimated_memory_mb > 500
        )

    def _create_batches(
        self,
        data: npt.NDArray[np.floating],
        config: BatchConfig
    ) -> List[npt.NDArray[np.floating]]:
        """Create batches from data with optional overlap."""
        batches = []
        step_size = int(config.batch_size * (1 - config.overlap_ratio))
        
        for i in range(0, len(data), step_size):
            end_idx = min(i + config.batch_size, len(data))
            batch = data[i:end_idx]
            
            if len(batch) > 0:  # Ensure non-empty batch
                batches.append(batch)
                
            if end_idx >= len(data):
                break
        
        return batches

    def _process_batches_parallel(
        self,
        batches: List[npt.NDArray[np.floating]],
        algorithm: str,
        contamination: float,
        config: BatchConfig,
        **kwargs: Any
    ) -> List[DetectionResult]:
        """Process batches in parallel using threads or processes."""
        results = []
        
        executor_class = ProcessPoolExecutor if config.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=config.max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(
                    self._process_single_batch,
                    batch, algorithm, contamination, i, **kwargs
                ): i
                for i, batch in enumerate(batches)
            }
            
            # Collect results in order
            batch_results = [None] * len(batches)
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    result = future.result()
                    batch_results[batch_idx] = result
                    print(f"   ✓ Batch {batch_idx + 1}/{len(batches)}: "
                          f"{result.n_anomalies} anomalies")
                except Exception as e:
                    print(f"   ✗ Batch {batch_idx + 1} failed: {e}")
                    # Create dummy result to maintain order
                    batch_results[batch_idx] = self._create_dummy_result(
                        len(batches[batch_idx]), algorithm
                    )
            
            # Filter out None results
            results = [r for r in batch_results if r is not None]
        
        return results

    def _process_batches_sequential(
        self,
        batches: List[npt.NDArray[np.floating]],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> List[DetectionResult]:
        """Process batches sequentially."""
        results = []
        
        for i, batch in enumerate(batches):
            try:
                result = self._process_single_batch(
                    batch, algorithm, contamination, i, **kwargs
                )
                results.append(result)
                print(f"   ✓ Batch {i + 1}/{len(batches)}: {result.n_anomalies} anomalies")
            except Exception as e:
                print(f"   ✗ Batch {i + 1} failed: {e}")
                # Create dummy result
                dummy_result = self._create_dummy_result(len(batch), algorithm)
                results.append(dummy_result)
        
        return results

    def _process_single_batch(
        self,
        batch: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        batch_idx: int,
        **kwargs: Any
    ) -> DetectionResult:
        """Process a single batch."""
        result = self.detection_service.detect_anomalies(
            batch, algorithm=algorithm, contamination=contamination, **kwargs
        )
        result.metadata["batch_index"] = batch_idx
        return result

    def _create_dummy_result(self, n_samples: int, algorithm: str) -> DetectionResult:
        """Create dummy result for failed batch."""
        return DetectionResult(
            predictions=np.zeros(n_samples, dtype=int),
            scores=None,
            algorithm=algorithm,
            n_samples=n_samples,
            n_anomalies=0,
            execution_time=0.0,
            metadata={"failed_batch": True}
        )

    def _combine_batch_results(
        self,
        batch_results: List[DetectionResult],
        algorithm: str,
        contamination: float,
        data_shape: tuple
    ) -> DetectionResult:
        """Combine results from multiple batches."""
        if not batch_results:
            return DetectionResult(
                predictions=np.array([], dtype=int),
                algorithm=algorithm,
                contamination=contamination
            )
        
        # Concatenate predictions
        all_predictions = np.concatenate([result.predictions for result in batch_results])
        
        # Concatenate scores if available
        all_scores = None
        if all(result.scores is not None for result in batch_results):
            all_scores = np.concatenate([result.scores for result in batch_results])
        
        # Calculate combined statistics
        total_execution_time = sum(result.execution_time for result in batch_results)
        total_anomalies = sum(result.n_anomalies for result in batch_results)
        
        # Create combined result
        combined_result = DetectionResult(
            predictions=all_predictions,
            scores=all_scores,
            algorithm=f"batch_{algorithm}",
            contamination=contamination,
            n_samples=len(all_predictions),
            n_anomalies=total_anomalies,
            execution_time=total_execution_time,
            metadata={
                "batch_processing": True,
                "n_batches": len(batch_results),
                "individual_times": [r.execution_time for r in batch_results],
                "failed_batches": sum(1 for r in batch_results if r.metadata.get("failed_batch", False)),
                "data_shape": data_shape
            }
        )
        
        return combined_result

    def benchmark_batch_sizes(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest",
        batch_sizes: Optional[List[int]] = None,
        contamination: float = 0.1
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark different batch sizes to find optimal configuration.
        
        Args:
            data: Input data for benchmarking
            algorithm: Algorithm to test
            batch_sizes: List of batch sizes to test
            contamination: Contamination rate
            
        Returns:
            Dictionary mapping batch size to performance metrics
        """
        if batch_sizes is None:
            batch_sizes = [500, 1000, 2000, 5000, 10000]
        
        print(f"📊 Benchmarking batch sizes for {len(data)} samples...")
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size >= len(data):
                continue
                
            print(f"   Testing batch size: {batch_size}")
            
            config = BatchConfig(
                batch_size=batch_size,
                parallel=True,
                max_workers=2  # Limited for benchmarking
            )
            
            try:
                start_time = time.time()
                result = self.process_large_dataset(
                    data, algorithm, contamination, config
                )
                execution_time = time.time() - start_time
                
                results[batch_size] = {
                    "execution_time": execution_time,
                    "throughput": len(data) / execution_time,
                    "n_anomalies": result.n_anomalies,
                    "n_batches": result.metadata["n_batches"],
                    "memory_efficiency": batch_size,  # Smaller = more memory efficient
                }
                
            except Exception as e:
                print(f"   ✗ Batch size {batch_size} failed: {e}")
                results[batch_size] = {"error": str(e)}
        
        # Find optimal batch size
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            best_batch_size = min(valid_results.keys(), 
                                key=lambda k: valid_results[k]["execution_time"])
            print(f"✅ Optimal batch size: {best_batch_size}")
        
        return results

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get batch processing performance statistics."""
        if not self._processing_stats:
            return {"total_processed": 0}
        
        execution_times = [s["execution_time"] for s in self._processing_stats]
        throughputs = [s["throughput"] for s in self._processing_stats]
        batch_sizes = [s["batch_size"] for s in self._processing_stats]
        
        return {
            "total_processed": len(self._processing_stats),
            "average_execution_time": np.mean(execution_times),
            "average_throughput": np.mean(throughputs),
            "best_throughput": np.max(throughputs),
            "average_batch_size": np.mean(batch_sizes),
            "total_samples_processed": sum(s["data_shape"][0] for s in self._processing_stats),
            "parallel_usage": sum(1 for s in self._processing_stats if s["parallel"]) / len(self._processing_stats)
        }

    def optimize_for_dataset(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest"
    ) -> BatchConfig:
        """Optimize batch configuration for specific dataset.
        
        Args:
            data: Input data to optimize for
            algorithm: Algorithm to optimize for
            
        Returns:
            Optimized BatchConfig
        """
        print(f"🎯 Optimizing batch configuration for {data.shape}")
        
        # Test with sample of data if very large
        test_data = data[:min(5000, len(data))]
        
        # Benchmark different configurations
        benchmark_results = self.benchmark_batch_sizes(test_data, algorithm)
        
        if not benchmark_results:
            return self._auto_configure_batching(data)
        
        # Find best performing configuration
        valid_results = {k: v for k, v in benchmark_results.items() if "error" not in v}
        
        if valid_results:
            # Choose based on best throughput with reasonable batch size
            best_batch_size = max(valid_results.keys(),
                                key=lambda k: valid_results[k]["throughput"] * 
                                             (1 - abs(k - 2000) / 10000))  # Prefer ~2000 batch size
        else:
            best_batch_size = 2000
        
        # Create optimized configuration
        optimized_config = self._auto_configure_batching(data)
        optimized_config.batch_size = best_batch_size
        
        print(f"✅ Optimized: batch_size={best_batch_size}, parallel={optimized_config.parallel}")
        
        return optimized_config
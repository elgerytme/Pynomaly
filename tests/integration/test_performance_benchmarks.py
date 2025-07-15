"""Performance benchmark integration tests for anomaly detection platform."""

from __future__ import annotations

import asyncio
import gc
import time

import numpy as np
import psutil
import pytest


@pytest.mark.performance
@pytest.mark.asyncio
class TestPerformanceBenchmarks:
    """Test performance benchmarks and resource utilization."""

    async def test_detection_latency_benchmark(
        self,
        detection_service,
        sample_datasets: list,
        performance_benchmarks: dict,
    ):
        """Benchmark detection latency across different algorithms."""
        dataset = sample_datasets[1]  # Use anomalous dataset

        algorithms = ["isolation_forest", "one_class_svm", "lof"]
        latency_results = {}

        for algorithm in algorithms:
            from pynomaly.domain.value_objects import (
                AlgorithmType,
                DetectorConfig,
                ModelType,
            )

            config = DetectorConfig(
                algorithm_type=getattr(AlgorithmType, algorithm.upper()),
                model_type=ModelType.UNSUPERVISED,
                parameters={
                    "contamination": 0.1,
                    "random_state": 42,
                },
            )

            from pynomaly.domain.models import Detector
            detector = Detector(name=f"{algorithm}_latency_benchmark", config=config)

            # Train detector
            training_start = time.perf_counter()
            await detection_service.train_detector(detector, dataset)
            training_time = time.perf_counter() - training_start

            # Benchmark detection latency
            detection_times = []
            batch_size = 100
            num_batches = 10

            for batch_idx in range(num_batches):
                start_idx = (batch_idx * batch_size) % len(dataset.data)
                end_idx = start_idx + batch_size
                batch_data = dataset.data[start_idx:end_idx]

                detection_start = time.perf_counter()
                result = await detection_service.detect(detector, batch_data)
                detection_time = time.perf_counter() - detection_start
                detection_times.append(detection_time * 1000)  # Convert to ms

            # Calculate statistics
            avg_latency = np.mean(detection_times)
            p95_latency = np.percentile(detection_times, 95)
            p99_latency = np.percentile(detection_times, 99)

            latency_results[algorithm] = {
                "training_time_ms": training_time * 1000,
                "avg_detection_latency_ms": avg_latency,
                "p95_detection_latency_ms": p95_latency,
                "p99_detection_latency_ms": p99_latency,
                "samples_per_second": (batch_size * 1000) / avg_latency,
            }

            # Validate against benchmarks
            assert avg_latency <= performance_benchmarks["max_detection_latency_ms"], \
                f"{algorithm} average latency {avg_latency:.2f}ms exceeds benchmark"

        # Print results for analysis
        print("\nDetection Latency Benchmark Results:")
        for algorithm, results in latency_results.items():
            print(f"{algorithm}:")
            print(f"  Training time: {results['training_time_ms']:.2f}ms")
            print(f"  Average latency: {results['avg_detection_latency_ms']:.2f}ms")
            print(f"  P95 latency: {results['p95_detection_latency_ms']:.2f}ms")
            print(f"  P99 latency: {results['p99_detection_latency_ms']:.2f}ms")
            print(f"  Throughput: {results['samples_per_second']:.1f} samples/sec")

    async def test_memory_usage_benchmark(
        self,
        detection_service,
        sample_datasets: list,
        performance_benchmarks: dict,
    ):
        """Benchmark memory usage during training and detection."""
        dataset = sample_datasets[1]  # Use anomalous dataset

        # Measure baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024

        algorithms = ["isolation_forest", "one_class_svm", "lof"]
        memory_results = {}

        for algorithm in algorithms:
            from pynomaly.domain.value_objects import (
                AlgorithmType,
                DetectorConfig,
                ModelType,
            )

            config = DetectorConfig(
                algorithm_type=getattr(AlgorithmType, algorithm.upper()),
                model_type=ModelType.UNSUPERVISED,
                parameters={
                    "contamination": 0.1,
                    "random_state": 42,
                },
            )

            from pynomaly.domain.models import Detector
            detector = Detector(name=f"{algorithm}_memory_benchmark", config=config)

            # Measure memory during training
            gc.collect()
            pre_training_memory = psutil.Process().memory_info().rss / 1024 / 1024

            await detection_service.train_detector(detector, dataset)

            gc.collect()
            post_training_memory = psutil.Process().memory_info().rss / 1024 / 1024

            training_memory_delta = post_training_memory - pre_training_memory

            # Measure memory during detection
            pre_detection_memory = psutil.Process().memory_info().rss / 1024 / 1024

            await detection_service.detect(detector, dataset.data)

            gc.collect()
            post_detection_memory = psutil.Process().memory_info().rss / 1024 / 1024

            detection_memory_delta = post_detection_memory - pre_detection_memory

            memory_results[algorithm] = {
                "training_memory_delta_mb": training_memory_delta,
                "detection_memory_delta_mb": detection_memory_delta,
                "total_memory_mb": post_detection_memory - baseline_memory,
            }

            # Validate against benchmark
            assert memory_results[algorithm]["total_memory_mb"] <= \
                performance_benchmarks["memory_usage_mb"], \
                f"{algorithm} memory usage exceeds benchmark"

        # Print results for analysis
        print("\nMemory Usage Benchmark Results:")
        for algorithm, results in memory_results.items():
            print(f"{algorithm}:")
            print(f"  Training memory delta: {results['training_memory_delta_mb']:.2f}MB")
            print(f"  Detection memory delta: {results['detection_memory_delta_mb']:.2f}MB")
            print(f"  Total memory usage: {results['total_memory_mb']:.2f}MB")

    async def test_throughput_benchmark(
        self,
        streaming_service,
        sample_datasets: list,
        performance_benchmarks: dict,
    ):
        """Benchmark streaming throughput."""
        dataset = sample_datasets[1]  # Use anomalous dataset

        # Start streaming service
        await streaming_service.start()

        try:
            # Generate continuous data stream
            data_stream = []
            for i in range(1000):
                sample = dataset.data[i % len(dataset.data)]
                data_stream.append({
                    "data": sample.tolist(),
                    "sample_id": f"benchmark_sample_{i}",
                    "timestamp": time.time(),
                })

            # Benchmark throughput
            start_time = time.time()

            # Process all samples
            tasks = []
            for sample in data_stream:
                task = streaming_service.process_sample(
                    sample["data"], sample["sample_id"]
                )
                tasks.append(task)

            # Wait for all processing to complete
            results = await asyncio.gather(*tasks)

            end_time = time.time()

            # Calculate throughput
            total_time = end_time - start_time
            throughput = len(data_stream) / total_time

            # Validate against benchmark
            assert throughput >= performance_benchmarks["throughput_samples_per_second"], \
                f"Throughput {throughput:.2f} samples/sec below benchmark"

            # Validate all samples were processed
            assert len(results) == len(data_stream), "Not all samples were processed"

            # Calculate additional metrics
            processing_times = []
            for i, result in enumerate(results):
                if hasattr(result, 'processing_time_ms'):
                    processing_times.append(result.processing_time_ms)

            if processing_times:
                avg_processing_time = np.mean(processing_times)
                p95_processing_time = np.percentile(processing_times, 95)
                p99_processing_time = np.percentile(processing_times, 99)

                print("\nStreaming Throughput Benchmark Results:")
                print(f"  Throughput: {throughput:.2f} samples/sec")
                print(f"  Average processing time: {avg_processing_time:.2f}ms")
                print(f"  P95 processing time: {p95_processing_time:.2f}ms")
                print(f"  P99 processing time: {p99_processing_time:.2f}ms")

        finally:
            await streaming_service.stop()

    async def test_resource_cleanup_benchmark(
        self,
        detection_service,
        test_cache,
        sample_datasets: list,
    ):
        """Benchmark resource cleanup and memory management."""
        dataset = sample_datasets[1]

        # Measure baseline
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Create and train multiple detectors
        detectors = []
        for i in range(10):
            from pynomaly.domain.value_objects import (
                AlgorithmType,
                DetectorConfig,
                ModelType,
            )

            config = DetectorConfig(
                algorithm_type=AlgorithmType.ISOLATION_FOREST,
                model_type=ModelType.UNSUPERVISED,
                parameters={
                    "contamination": 0.1,
                    "random_state": 42 + i,
                },
            )

            from pynomaly.domain.models import Detector
            detector = Detector(name=f"cleanup_detector_{i}", config=config)
            await detection_service.train_detector(detector, dataset)
            detectors.append(detector)

        # Perform detection operations
        for detector in detectors:
            await detection_service.detect(detector, dataset.data)

        # Measure peak memory
        gc.collect()
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = peak_memory - baseline_memory

        # Clear cache
        await test_cache.clear()

        # Delete detectors
        detectors.clear()

        # Force garbage collection
        for _ in range(3):
            gc.collect()

        # Measure memory after cleanup
        post_cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_recovered = peak_memory - post_cleanup_memory
        recovery_rate = memory_recovered / memory_growth if memory_growth > 0 else 1.0

        # Validate memory recovery
        assert recovery_rate >= 0.8, f"Memory recovery rate {recovery_rate:.2f} too low"

        # Memory should return close to baseline
        memory_leak = post_cleanup_memory - baseline_memory
        assert memory_leak <= 50, f"Potential memory leak detected: {memory_leak:.2f}MB"

        print("\nResource Cleanup Benchmark Results:")
        print(f"  Baseline memory: {baseline_memory:.2f}MB")
        print(f"  Peak memory: {peak_memory:.2f}MB")
        print(f"  Memory growth: {memory_growth:.2f}MB")
        print(f"  Post-cleanup memory: {post_cleanup_memory:.2f}MB")
        print(f"  Memory recovered: {memory_recovered:.2f}MB")
        print(f"  Recovery rate: {recovery_rate:.2f}")
        print(f"  Potential leak: {memory_leak:.2f}MB")

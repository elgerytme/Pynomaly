"""Memory usage and leak testing for anomaly detection system."""

import pytest
import gc
import sys
import time
import numpy as np
import threading
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from unittest.mock import Mock, patch

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.worker import AnomalyDetectionWorker, JobType, JobPriority
from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_mb: float
    vms_mb: float
    objects_count: int
    gc_counts: Tuple[int, int, int]
    description: str


class MemoryProfiler:
    """Memory profiling and leak detection utilities."""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_snapshot: Optional[MemorySnapshot] = None
    
    def take_snapshot(self, description: str = "") -> MemorySnapshot:
        """Take a memory usage snapshot."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            rss_mb = memory_info.rss / 1024 / 1024
            vms_mb = memory_info.vms / 1024 / 1024
        except ImportError:
            # Fallback without psutil
            rss_mb = 0.0
            vms_mb = 0.0
        
        # Count objects
        objects_count = len(gc.get_objects())
        
        # GC stats
        gc_counts = gc.get_count()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            objects_count=objects_count,
            gc_counts=gc_counts,
            description=description
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def set_baseline(self, description: str = "baseline"):
        """Set baseline memory snapshot."""
        self.baseline_snapshot = self.take_snapshot(description)
    
    def get_memory_growth(self) -> float:
        """Get memory growth since baseline in MB."""
        if not self.baseline_snapshot or not self.snapshots:
            return 0.0
        
        current = self.snapshots[-1]
        return current.rss_mb - self.baseline_snapshot.rss_mb
    
    def get_object_growth(self) -> int:
        """Get object count growth since baseline."""
        if not self.baseline_snapshot or not self.snapshots:
            return 0
        
        current = self.snapshots[-1]
        return current.objects_count - self.baseline_snapshot.objects_count
    
    def detect_memory_leak(self, threshold_mb: float = 10.0) -> bool:
        """Detect potential memory leak."""
        if len(self.snapshots) < 3:
            return False
        
        # Check if memory consistently grows
        recent_snapshots = self.snapshots[-3:]
        memory_values = [s.rss_mb for s in recent_snapshots]
        
        # Simple trend detection
        is_increasing = all(memory_values[i] <= memory_values[i+1] 
                          for i in range(len(memory_values)-1))
        
        growth = memory_values[-1] - memory_values[0]
        return is_increasing and growth > threshold_mb
    
    def print_memory_report(self):
        """Print memory usage report."""
        if not self.snapshots:
            print("No memory snapshots available")
            return
        
        print("\nMemory Usage Report:")
        print("=" * 60)
        
        for i, snapshot in enumerate(self.snapshots):
            growth_indicator = ""
            if self.baseline_snapshot and snapshot != self.baseline_snapshot:
                growth = snapshot.rss_mb - self.baseline_snapshot.rss_mb
                growth_indicator = f" (+{growth:.1f}MB)" if growth > 0 else f" ({growth:.1f}MB)"
            
            print(f"{i+1:2d}. {snapshot.description:<20} "
                  f"RSS: {snapshot.rss_mb:6.1f}MB{growth_indicator:<12} "
                  f"Objects: {snapshot.objects_count:>6}")
        
        if self.baseline_snapshot:
            print(f"\nTotal Memory Growth: {self.get_memory_growth():.1f}MB")
            print(f"Total Object Growth: {self.get_object_growth()}")
            
            if self.detect_memory_leak():
                print("⚠️  Potential memory leak detected!")


class ObjectTracker:
    """Track object creation and deletion."""
    
    def __init__(self):
        self.object_counts = defaultdict(int)
        self.initial_counts = {}
    
    def start_tracking(self):
        """Start tracking object counts."""
        self.initial_counts = self._get_current_object_counts()
    
    def get_object_growth(self) -> Dict[str, int]:
        """Get object count growth by type."""
        current_counts = self._get_current_object_counts()
        growth = {}
        
        for obj_type, count in current_counts.items():
            initial = self.initial_counts.get(obj_type, 0)
            if count > initial:
                growth[obj_type] = count - initial
        
        return growth
    
    def _get_current_object_counts(self) -> Dict[str, int]:
        """Get current object counts by type."""
        counts = defaultdict(int)
        
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            counts[obj_type] += 1
        
        return dict(counts)
    
    def print_object_report(self):
        """Print object growth report."""
        growth = self.get_object_growth()
        
        if not growth:
            print("No object growth detected")
            return
        
        print("\nObject Growth Report:")
        print("=" * 40)
        
        # Sort by growth amount
        sorted_growth = sorted(growth.items(), key=lambda x: x[1], reverse=True)
        
        for obj_type, count in sorted_growth[:10]:  # Top 10
            print(f"{obj_type:<20}: +{count:>6}")


@pytest.mark.performance
@pytest.mark.memory
class TestDetectionServiceMemory:
    """Memory tests for detection service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
        self.profiler = MemoryProfiler()
        self.object_tracker = ObjectTracker()
        
        # Generate test data
        np.random.seed(42)
        self.small_data = np.random.normal(0, 1, (100, 5)).tolist()
        self.medium_data = np.random.normal(0, 1, (1000, 10)).tolist()
        self.large_data = np.random.normal(0, 1, (5000, 20)).tolist()
    
    def test_single_detection_memory_usage(self):
        """Test memory usage for single detection."""
        self.profiler.set_baseline("before_detection")
        self.object_tracker.start_tracking()
        
        # Run detection
        result = self.detection_service.detect_anomalies(
            self.medium_data,
            algorithm="isolation_forest",
            contamination=0.1
        )
        
        self.profiler.take_snapshot("after_detection")
        
        # Force garbage collection
        gc.collect()
        self.profiler.take_snapshot("after_gc")
        
        # Memory growth should be reasonable
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth < 50.0, f"Memory growth too high: {memory_growth}MB"
        
        # Object growth should be minimal after GC
        object_growth = self.profiler.get_object_growth()
        assert object_growth < 1000, f"Too many objects retained: {object_growth}"
        
        self.profiler.print_memory_report()
        self.object_tracker.print_object_report()
    
    def test_repeated_detection_memory_leak(self):
        """Test for memory leaks in repeated detections."""
        self.profiler.set_baseline("initial")
        
        # Run multiple detections
        for i in range(10):
            result = self.detection_service.detect_anomalies(
                self.small_data,
                algorithm="isolation_forest",
                contamination=0.1
            )
            
            if i % 3 == 0:  # Sample snapshots
                self.profiler.take_snapshot(f"iteration_{i+1}")
            
            # Clear reference to result
            del result
        
        # Force garbage collection
        gc.collect()
        self.profiler.take_snapshot("after_all_gc")
        
        # Check for memory leak
        assert not self.profiler.detect_memory_leak(threshold_mb=5.0), "Memory leak detected"
        
        # Memory growth should be minimal
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth < 20.0, f"Excessive memory growth: {memory_growth}MB"
        
        self.profiler.print_memory_report()
    
    def test_large_dataset_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        self.profiler.set_baseline("before_large_dataset")
        
        # Process large dataset
        result = self.detection_service.detect_anomalies(
            self.large_data,
            algorithm="isolation_forest",
            contamination=0.1
        )
        
        self.profiler.take_snapshot("after_large_dataset")
        
        # Clear result
        del result
        gc.collect()
        self.profiler.take_snapshot("after_cleanup")
        
        # Memory usage should scale reasonably with data size
        memory_growth = self.profiler.get_memory_growth()
        data_size_mb = len(self.large_data) * len(self.large_data[0]) * 8 / 1024 / 1024  # Rough estimate
        
        # Memory growth should not be excessive compared to data size
        memory_ratio = memory_growth / data_size_mb if data_size_mb > 0 else 0
        assert memory_ratio < 10.0, f"Memory usage ratio too high: {memory_ratio}x"
        
        self.profiler.print_memory_report()
    
    def test_algorithm_memory_comparison(self):
        """Compare memory usage across different algorithms."""
        algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
        memory_usage = {}
        
        for algorithm in algorithms:
            self.profiler.set_baseline(f"before_{algorithm}")
            
            result = self.detection_service.detect_anomalies(
                self.medium_data,
                algorithm=algorithm,
                contamination=0.1
            )
            
            self.profiler.take_snapshot(f"after_{algorithm}")
            memory_usage[algorithm] = self.profiler.get_memory_growth()
            
            # Cleanup
            del result
            gc.collect()
        
        print(f"\nAlgorithm Memory Usage Comparison:")
        for algorithm, usage in memory_usage.items():
            print(f"  {algorithm:<20}: {usage:>6.1f}MB")
        
        # All algorithms should have reasonable memory usage
        for algorithm, usage in memory_usage.items():
            assert usage < 100.0, f"{algorithm} uses too much memory: {usage}MB"


@pytest.mark.performance
@pytest.mark.memory
class TestEnsembleServiceMemory:
    """Memory tests for ensemble service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ensemble_service = EnsembleService()
        self.profiler = MemoryProfiler()
        
        np.random.seed(42)
        self.test_data = np.random.normal(0, 1, (500, 10)).tolist()
    
    def test_ensemble_memory_usage(self):
        """Test ensemble memory usage."""
        self.profiler.set_baseline("before_ensemble")
        
        result = self.ensemble_service.detect_with_ensemble(
            self.test_data,
            algorithms=["isolation_forest", "one_class_svm", "local_outlier_factor"],
            ensemble_method="majority",
            contamination=0.1
        )
        
        self.profiler.take_snapshot("after_ensemble")
        
        # Cleanup
        del result
        gc.collect()
        self.profiler.take_snapshot("after_cleanup")
        
        # Ensemble should use more memory than individual algorithms
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth > 0, "No memory usage detected"
        assert memory_growth < 200.0, f"Ensemble uses too much memory: {memory_growth}MB"
        
        self.profiler.print_memory_report()
    
    def test_ensemble_size_scaling(self):
        """Test memory scaling with ensemble size."""
        ensemble_sizes = [2, 3, 4]
        memory_usage = {}
        
        algorithms_pool = ["isolation_forest", "one_class_svm", "local_outlier_factor", "elliptic_envelope"]
        
        for size in ensemble_sizes:
            self.profiler.set_baseline(f"before_ensemble_{size}")
            
            algorithms = algorithms_pool[:size]
            result = self.ensemble_service.detect_with_ensemble(
                self.test_data,
                algorithms=algorithms,
                ensemble_method="majority",
                contamination=0.1
            )
            
            self.profiler.take_snapshot(f"after_ensemble_{size}")
            memory_usage[size] = self.profiler.get_memory_growth()
            
            del result
            gc.collect()
        
        print(f"\nEnsemble Size Memory Scaling:")
        for size, usage in memory_usage.items():
            print(f"  {size} algorithms: {usage:>6.1f}MB")
        
        # Memory should scale somewhat with ensemble size
        assert memory_usage[4] > memory_usage[2], "Memory usage should increase with ensemble size"
        
        # But not excessively
        scaling_ratio = memory_usage[4] / memory_usage[2] if memory_usage[2] > 0 else 1
        assert scaling_ratio < 3.0, f"Memory scaling too poor: {scaling_ratio}x"


@pytest.mark.performance
@pytest.mark.memory
class TestStreamingServiceMemory:
    """Memory tests for streaming service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.streaming_service = StreamingService()
        self.profiler = MemoryProfiler()
        
        # Generate streaming data
        np.random.seed(42)
        self.batch_size = 100
        self.n_features = 8
        self.streaming_batches = []
        
        for i in range(50):
            batch = np.random.normal(0, 1, (self.batch_size, self.n_features)).tolist()
            self.streaming_batches.append(batch)
    
    def test_streaming_batch_memory_stability(self):
        """Test memory stability during streaming processing."""
        self.profiler.set_baseline("streaming_start")
        
        # Process multiple batches
        for i, batch in enumerate(self.streaming_batches[:20]):
            result = self.streaming_service.process_streaming_batch(
                batch,
                algorithm="isolation_forest",
                buffer_size=300
            )
            
            if i % 5 == 0:
                self.profiler.take_snapshot(f"batch_{i}")
                
            # Clear result
            del result
        
        # Final cleanup
        gc.collect()
        self.profiler.take_snapshot("streaming_end")
        
        # Memory should remain stable
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth < 30.0, f"Memory growth too high: {memory_growth}MB"
        
        # Check for memory leak
        assert not self.profiler.detect_memory_leak(threshold_mb=5.0), "Memory leak in streaming"
        
        self.profiler.print_memory_report()
    
    def test_streaming_buffer_memory_management(self):
        """Test memory management with different buffer sizes."""
        buffer_sizes = [100, 500, 1000]
        memory_usage = {}
        
        for buffer_size in buffer_sizes:
            self.profiler.set_baseline(f"buffer_{buffer_size}_start")
            
            # Process several batches with this buffer size
            for batch in self.streaming_batches[:10]:
                result = self.streaming_service.process_streaming_batch(
                    batch,
                    algorithm="isolation_forest",
                    buffer_size=buffer_size
                )
                del result
            
            self.profiler.take_snapshot(f"buffer_{buffer_size}_end")
            memory_usage[buffer_size] = self.profiler.get_memory_growth()
            
            gc.collect()
        
        print(f"\nBuffer Size Memory Usage:")
        for buffer_size, usage in memory_usage.items():
            print(f"  Buffer {buffer_size:>4}: {usage:>6.1f}MB")
        
        # Larger buffers should use more memory
        assert memory_usage[1000] >= memory_usage[100], "Buffer size not affecting memory usage"
    
    def test_streaming_long_running_memory(self):
        """Test memory usage in long-running streaming scenario."""
        self.profiler.set_baseline("long_streaming_start")
        memory_readings = []
        
        # Simulate long-running streaming
        for i, batch in enumerate(self.streaming_batches):
            result = self.streaming_service.process_streaming_batch(
                batch,
                algorithm="isolation_forest",
                buffer_size=200
            )
            
            # Record memory periodically
            if i % 5 == 0:
                snapshot = self.profiler.take_snapshot(f"long_batch_{i}")
                memory_readings.append(snapshot.rss_mb)
            
            del result
            
            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # Final snapshot
        self.profiler.take_snapshot("long_streaming_end")
        
        # Check memory trend
        if len(memory_readings) >= 3:
            # Simple linear regression to detect trend
            x = list(range(len(memory_readings)))
            y = memory_readings
            
            # Calculate trend slope
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] * x[i] for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            print(f"\nLong-running Streaming Memory Trend: {slope:.3f} MB/batch")
            
            # Slope should be close to zero (stable memory)
            assert abs(slope) < 0.1, f"Memory trend too steep: {slope:.3f} MB/batch"
        
        self.profiler.print_memory_report()


@pytest.mark.performance
@pytest.mark.memory
class TestWorkerMemory:
    """Memory tests for worker system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.worker = AnomalyDetectionWorker(max_workers=2)
        self.profiler = MemoryProfiler()
        
        np.random.seed(42)
        self.test_data = np.random.normal(0, 1, (50, 5)).tolist()
    
    def test_worker_job_queue_memory(self):
        """Test worker job queue memory usage."""
        self.profiler.set_baseline("empty_queue")
        
        # Add many jobs
        job_ids = []
        for i in range(200):
            job_id = self.worker.add_job(
                JobType.DETECTION,
                {
                    "algorithm": "isolation_forest",
                    "data": self.test_data,
                    "contamination": 0.1
                },
                JobPriority.NORMAL
            )
            job_ids.append(job_id)
            
            if i % 50 == 0:
                self.profiler.take_snapshot(f"jobs_{i}")
        
        self.profiler.take_snapshot("all_jobs_queued")
        
        # Clear all jobs
        self.worker.job_queue.clear()
        gc.collect()
        self.profiler.take_snapshot("queue_cleared")
        
        # Memory should be released after clearing queue
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth < 20.0, f"Queue memory not properly released: {memory_growth}MB"
        
        self.profiler.print_memory_report()
    
    def test_worker_job_processing_memory(self):
        """Test memory usage during job processing."""
        # Mock the detection service to avoid actual processing
        with patch.object(self.worker, 'detection_service') as mock_detection:
            mock_detection.detect_anomalies = Mock(return_value={
                "anomalies": [0, 1, 0],
                "scores": [0.1, 0.8, 0.2],
                "algorithm": "isolation_forest"
            })
            
            self.profiler.set_baseline("before_processing")
            
            # Add and simulate processing jobs
            for i in range(20):
                job_id = self.worker.add_job(
                    JobType.DETECTION,
                    {
                        "algorithm": "isolation_forest", 
                        "data": self.test_data,
                        "contamination": 0.1
                    },
                    JobPriority.NORMAL
                )
                
                # Simulate job completion
                job = self.worker.job_queue.get_next_job()
                if job:
                    # Add to completed jobs list
                    completed_job = Mock()
                    completed_job.job_id = job_id
                    completed_job.result = {"anomalies": [0, 1, 0]}
                    self.worker.completed_jobs.append(completed_job)
                
                if i % 5 == 0:
                    self.profiler.take_snapshot(f"processed_{i}")
            
            self.profiler.take_snapshot("all_processed")
            
            # Clear completed jobs
            self.worker.clear_completed_jobs()
            gc.collect()
            self.profiler.take_snapshot("cleaned_up")
            
            # Memory should be stable
            memory_growth = self.profiler.get_memory_growth()
            assert memory_growth < 25.0, f"Processing memory growth too high: {memory_growth}MB"
            
            self.profiler.print_memory_report()
    
    def test_worker_concurrent_memory_safety(self):
        """Test memory safety during concurrent operations."""
        self.profiler.set_baseline("concurrent_start")
        
        def add_jobs():
            for i in range(50):
                self.worker.add_job(
                    JobType.DETECTION,
                    {"algorithm": "isolation_forest", "data": self.test_data},
                    JobPriority.NORMAL
                )
        
        def process_jobs():
            for _ in range(25):
                job = self.worker.job_queue.get_next_job()
                if job:
                    # Simulate processing
                    time.sleep(0.001)
        
        # Run concurrent operations
        threads = []
        threads.append(threading.Thread(target=add_jobs))
        threads.append(threading.Thread(target=add_jobs))
        threads.append(threading.Thread(target=process_jobs))
        threads.append(threading.Thread(target=process_jobs))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        self.profiler.take_snapshot("concurrent_end")
        
        # Clean up
        self.worker.job_queue.clear()
        gc.collect()
        self.profiler.take_snapshot("concurrent_cleanup")
        
        # Memory should be reasonable
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth < 30.0, f"Concurrent memory usage too high: {memory_growth}MB"
        
        self.profiler.print_memory_report()


@pytest.mark.performance
@pytest.mark.memory
class TestModelRepositoryMemory:
    """Memory tests for model repository."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository()
        self.profiler = MemoryProfiler()
        
        # Create mock models
        self.mock_models = []
        for i in range(10):
            model = Mock()
            model.model_id = f"test_model_{i}"
            model.name = f"Test Model {i}"
            model.algorithm = "isolation_forest"
            # Simulate model data
            model.data = np.random.normal(0, 1, (100, 10)).tolist()
            self.mock_models.append(model)
    
    def test_model_storage_memory(self):
        """Test memory usage for model storage operations."""
        self.profiler.set_baseline("before_storage")
        
        # Save multiple models
        saved_model_ids = []
        for i, model in enumerate(self.mock_models):
            with patch('builtins.open', mock_open()):
                with patch('pickle.dump'):
                    with patch('json.dump'):
                        model_id = self.repository.save(model)
                        saved_model_ids.append(model_id)
            
            if i % 3 == 0:
                self.profiler.take_snapshot(f"saved_{i+1}_models")
        
        self.profiler.take_snapshot("all_models_saved")
        
        # Clear references
        del saved_model_ids
        gc.collect()
        self.profiler.take_snapshot("after_cleanup")
        
        # Memory growth should be reasonable
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth < 50.0, f"Model storage memory too high: {memory_growth}MB"
        
        self.profiler.print_memory_report()
    
    def test_model_loading_memory(self):
        """Test memory usage for model loading operations."""
        # Mock the repository to have models available
        with patch.object(self.repository, 'list_models') as mock_list:
            mock_list.return_value = [
                {"model_id": f"model_{i}", "name": f"Model {i}"} 
                for i in range(5)
            ]
            
            with patch.object(self.repository, 'load') as mock_load:
                mock_load.return_value = self.mock_models[0]
                
                self.profiler.set_baseline("before_loading")
                
                # Load models multiple times
                loaded_models = []
                for i in range(10):
                    model = self.repository.load(f"model_{i % 5}")
                    loaded_models.append(model)
                    
                    if i % 3 == 0:
                        self.profiler.take_snapshot(f"loaded_{i+1}_times")
                
                self.profiler.take_snapshot("all_loads_complete")
                
                # Clear loaded models
                del loaded_models
                gc.collect()
                self.profiler.take_snapshot("loads_cleared")
                
                # Memory should be properly managed
                memory_growth = self.profiler.get_memory_growth()
                assert memory_growth < 30.0, f"Model loading memory too high: {memory_growth}MB"
                
                self.profiler.print_memory_report()


@pytest.mark.performance
@pytest.mark.memory
class TestMemoryLeakDetection:
    """Comprehensive memory leak detection tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
        self.profiler = MemoryProfiler()
        self.object_tracker = ObjectTracker()
        
        np.random.seed(42)
        self.test_data = np.random.normal(0, 1, (200, 8)).tolist()
    
    def test_cyclic_reference_detection(self):
        """Test detection of cyclic references."""
        self.profiler.set_baseline("before_cycles")
        self.object_tracker.start_tracking()
        
        # Create objects with potential cyclic references
        objects_with_cycles = []
        
        for i in range(100):
            # Create a simple cycle
            obj1 = {"id": i, "ref": None}
            obj2 = {"id": i + 100, "ref": obj1}
            obj1["ref"] = obj2
            objects_with_cycles.append(obj1)
            
            if i % 25 == 0:
                self.profiler.take_snapshot(f"cycle_{i}")
        
        self.profiler.take_snapshot("cycles_created")
        
        # Clear references
        del objects_with_cycles
        
        # Force garbage collection
        gc.collect()
        self.profiler.take_snapshot("after_gc")
        
        # Objects should be cleaned up
        object_growth = self.profiler.get_object_growth()
        assert object_growth < 500, f"Cyclic references not cleaned up: {object_growth} objects"
        
        self.profiler.print_memory_report()
        self.object_tracker.print_object_report()
    
    def test_repeated_operations_leak_detection(self):
        """Test for leaks in repeated operations."""
        self.profiler.set_baseline("repeated_ops_start")
        
        # Perform the same operation many times
        for i in range(50):
            # Detection operation
            result = self.detection_service.detect_anomalies(
                self.test_data,
                algorithm="isolation_forest",
                contamination=0.1
            )
            
            # Clear result immediately
            del result
            
            # Periodic snapshots and GC
            if i % 10 == 0:
                gc.collect()
                self.profiler.take_snapshot(f"repeat_{i}")
        
        # Final cleanup
        gc.collect()
        self.profiler.take_snapshot("repeated_ops_end")
        
        # Check for memory leak
        assert not self.profiler.detect_memory_leak(threshold_mb=3.0), "Memory leak in repeated operations"
        
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth < 15.0, f"Memory growth too high: {memory_growth}MB"
        
        self.profiler.print_memory_report()
    
    def test_exception_handling_memory_safety(self):
        """Test memory safety when exceptions occur."""
        self.profiler.set_baseline("before_exceptions")
        
        # Cause exceptions and ensure memory is cleaned up
        for i in range(20):
            try:
                # This should cause an exception
                result = self.detection_service.detect_anomalies(
                    [],  # Empty data
                    algorithm="invalid_algorithm",
                    contamination=0.1
                )
            except Exception:
                # Expected exception
                pass
            
            if i % 5 == 0:
                gc.collect()
                self.profiler.take_snapshot(f"exception_{i}")
        
        # Final cleanup
        gc.collect()
        self.profiler.take_snapshot("exceptions_end")
        
        # Memory should remain stable even with exceptions
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth < 10.0, f"Exception handling memory leak: {memory_growth}MB"
        
        self.profiler.print_memory_report()
    
    def test_thread_local_memory_isolation(self):
        """Test memory isolation between threads."""
        self.profiler.set_baseline("thread_isolation_start")
        
        def thread_work():
            """Work function for each thread."""
            # Each thread does some detection work
            local_data = np.random.normal(0, 1, (100, 5)).tolist()
            
            for _ in range(5):
                result = self.detection_service.detect_anomalies(
                    local_data,
                    algorithm="isolation_forest",
                    contamination=0.1
                )
                del result
            
            del local_data
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_work)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check memory after thread completion
        gc.collect()
        self.profiler.take_snapshot("threads_completed")
        
        # Memory should be cleaned up
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth < 25.0, f"Thread memory not isolated: {memory_growth}MB"
        
        self.profiler.print_memory_report()


@pytest.mark.performance
@pytest.mark.memory
class TestMemoryOptimization:
    """Memory optimization and efficiency tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
        self.profiler = MemoryProfiler()
    
    def test_memory_efficient_data_processing(self):
        """Test memory efficiency of data processing."""
        # Test different data formats
        data_formats = {
            "list_of_lists": [[float(j) for j in range(10)] for _ in range(1000)],
            "numpy_array": np.random.normal(0, 1, (1000, 10)),
            "nested_structure": [{"features": [float(j) for j in range(10)]} for _ in range(1000)]
        }
        
        memory_usage = {}
        
        for format_name, data in data_formats.items():
            self.profiler.set_baseline(f"before_{format_name}")
            
            # Convert to list format for detection service
            if format_name == "numpy_array":
                processed_data = data.tolist()
            elif format_name == "nested_structure":
                processed_data = [item["features"] for item in data]
            else:
                processed_data = data
            
            result = self.detection_service.detect_anomalies(
                processed_data,
                algorithm="isolation_forest",
                contamination=0.1
            )
            
            self.profiler.take_snapshot(f"after_{format_name}")
            memory_usage[format_name] = self.profiler.get_memory_growth()
            
            del result
            del processed_data
            gc.collect()
        
        print(f"\nData Format Memory Efficiency:")
        for format_name, usage in memory_usage.items():
            print(f"  {format_name:<20}: {usage:>6.1f}MB")
        
        # All formats should be reasonably efficient
        for format_name, usage in memory_usage.items():
            assert usage < 80.0, f"{format_name} too memory intensive: {usage}MB"
    
    def test_memory_pooling_simulation(self):
        """Simulate memory pooling effects."""
        self.profiler.set_baseline("pooling_start")
        
        # Create and reuse data structures
        data_pool = []
        
        # Phase 1: Create data structures
        for i in range(20):
            data = np.random.normal(0, 1, (100, 8)).tolist()
            data_pool.append(data)
            
            if i % 5 == 0:
                self.profiler.take_snapshot(f"pool_create_{i}")
        
        self.profiler.take_snapshot("pool_created")
        
        # Phase 2: Reuse data structures
        for i in range(30):
            # Reuse existing data
            data = data_pool[i % len(data_pool)]
            
            result = self.detection_service.detect_anomalies(
                data,
                algorithm="isolation_forest",
                contamination=0.1
            )
            
            del result
            
            if i % 10 == 0:
                self.profiler.take_snapshot(f"pool_reuse_{i}")
        
        self.profiler.take_snapshot("pool_reused")
        
        # Cleanup pool
        del data_pool
        gc.collect()
        self.profiler.take_snapshot("pool_cleanup")
        
        # Memory reuse should be efficient
        memory_growth = self.profiler.get_memory_growth()
        assert memory_growth < 40.0, f"Memory pooling not effective: {memory_growth}MB"
        
        self.profiler.print_memory_report()


if __name__ == "__main__":
    # Run specific memory tests
    pytest.main([
        __file__ + "::TestDetectionServiceMemory::test_single_detection_memory_usage",
        "-v", "-s", "--tb=short"
    ])
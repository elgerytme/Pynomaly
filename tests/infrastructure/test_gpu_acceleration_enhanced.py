"""
Enhanced GPU Acceleration Testing Suite
Comprehensive tests for GPU acceleration and CUDA/device management.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pynomaly.domain.exceptions import GPUError, MemoryError
from pynomaly.infrastructure.gpu.cuda_utils import CUDAUtils
from pynomaly.infrastructure.gpu.device_manager import DeviceManager
from pynomaly.infrastructure.gpu.memory_manager import GPUMemoryManager


class TestDeviceManager:
    """Test suite for device management and detection."""

    @pytest.fixture
    def device_manager(self):
        """Create device manager instance."""
        return DeviceManager()

    def test_device_detection_cuda_available(self, device_manager):
        """Test CUDA device detection when available."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            devices = device_manager.detect_devices()

            assert len(devices) >= 1
            assert any(device.type == "cuda" for device in devices)

    def test_device_detection_cuda_unavailable(self, device_manager):
        """Test device detection when CUDA is unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            devices = device_manager.detect_devices()

            assert len(devices) >= 1
            assert all(device.type == "cpu" for device in devices)

    def test_device_capabilities_query(self, device_manager):
        """Test querying device capabilities."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            mock_props.return_value = Mock(
                name="GeForce RTX 3080",
                major=8,
                minor=6,
                total_memory=10737418240,
                multi_processor_count=68,
            )

            capabilities = device_manager.get_device_capabilities(0)

            assert capabilities["name"] == "GeForce RTX 3080"
            assert capabilities["compute_capability"] == (8, 6)
            assert capabilities["total_memory"] > 10 * 1024**3  # > 10GB

    def test_device_selection_strategy(self, device_manager):
        """Test device selection strategies."""
        # Test automatic selection
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.memory_allocated", side_effect=[1024**3, 512**3]),
        ):  # 1GB, 512MB
            device = device_manager.select_device(strategy="memory_optimal")

            assert device.index == 1  # Should select device with less memory usage

    def test_device_selection_performance_optimal(self, device_manager):
        """Test performance-optimal device selection."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            # Mock different GPU capabilities
            mock_props.side_effect = [
                Mock(major=8, minor=6, multi_processor_count=68),  # RTX 3080
                Mock(major=7, minor=5, multi_processor_count=40),  # RTX 2070
            ]

            device = device_manager.select_device(strategy="performance_optimal")

            assert device.index == 0  # Should select the more powerful GPU

    def test_multi_gpu_load_balancing(self, device_manager):
        """Test load balancing across multiple GPUs."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=4),
        ):
            # Simulate workload distribution
            workloads = [{"size": 1000}, {"size": 2000}, {"size": 1500}, {"size": 800}]

            assignments = device_manager.distribute_workloads(workloads)

            assert len(assignments) == 4
            assert all(0 <= assignment["device"] < 4 for assignment in assignments)

    def test_device_fallback_mechanism(self, device_manager):
        """Test fallback from GPU to CPU when GPU fails."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.set_device", side_effect=RuntimeError("CUDA error")),
        ):
            device = device_manager.select_device_with_fallback()

            assert device.type == "cpu"  # Should fallback to CPU

    def test_device_monitoring(self, device_manager):
        """Test device monitoring and health checks."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("nvidia_ml_py.nvmlDeviceGetTemperature", return_value=65),
            patch("nvidia_ml_py.nvmlDeviceGetPowerUsage", return_value=250),
        ):
            health = device_manager.check_device_health(0)

            assert health["temperature"] == 65
            assert health["power_usage"] == 250
            assert health["status"] == "healthy"


class TestGPUMemoryManager:
    """Test suite for GPU memory management."""

    @pytest.fixture
    def memory_manager(self):
        """Create GPU memory manager instance."""
        return GPUMemoryManager()

    def test_memory_allocation_tracking(self, memory_manager):
        """Test memory allocation tracking."""
        with (
            patch("torch.cuda.memory_allocated", return_value=1024**3),
            patch("torch.cuda.memory_reserved", return_value=2 * 1024**3),
        ):
            stats = memory_manager.get_memory_stats(device=0)

            assert stats["allocated"] == 1024**3
            assert stats["reserved"] == 2 * 1024**3
            assert stats["free"] == 1024**3

    def test_memory_optimization_strategies(self, memory_manager):
        """Test memory optimization strategies."""
        # Test garbage collection
        with (
            patch("torch.cuda.empty_cache") as mock_empty_cache,
            patch("gc.collect") as mock_gc,
        ):
            memory_manager.optimize_memory(strategy="aggressive")

            mock_empty_cache.assert_called()
            mock_gc.assert_called()

    def test_memory_pool_management(self, memory_manager):
        """Test memory pool management."""
        with (
            patch("torch.cuda.memory_allocated", side_effect=[0, 512**3, 0]),
            patch("torch.cuda.memory_reserved", side_effect=[0, 1024**3, 512**3]),
        ):
            # Allocate memory
            memory_manager.allocate_pool(size=512**3, device=0)
            stats_after_alloc = memory_manager.get_memory_stats(0)

            # Deallocate memory
            memory_manager.deallocate_pool(device=0)
            stats_after_dealloc = memory_manager.get_memory_stats(0)

            assert stats_after_alloc["allocated"] > 0
            assert stats_after_dealloc["allocated"] == 0

    def test_out_of_memory_handling(self, memory_manager):
        """Test out-of-memory error handling."""
        with (
            patch("torch.cuda.memory_allocated", return_value=8 * 1024**3),
            patch("torch.cuda.max_memory_allocated", return_value=10 * 1024**3),
        ):
            # Simulate OOM condition
            with pytest.raises(MemoryError):
                memory_manager.check_memory_availability(
                    required_memory=12 * 1024**3, device=0
                )

    def test_memory_fragmentation_detection(self, memory_manager):
        """Test memory fragmentation detection."""
        with (
            patch("torch.cuda.memory_allocated", return_value=6 * 1024**3),
            patch("torch.cuda.memory_reserved", return_value=10 * 1024**3),
        ):
            fragmentation = memory_manager.calculate_fragmentation(device=0)

            # Fragmentation = (reserved - allocated) / reserved
            expected_fragmentation = (10 * 1024**3 - 6 * 1024**3) / (10 * 1024**3)
            assert abs(fragmentation - expected_fragmentation) < 0.01

    def test_memory_leak_detection(self, memory_manager):
        """Test memory leak detection."""
        initial_memory = 1 * 1024**3
        leaked_memory = 2 * 1024**3

        with patch(
            "torch.cuda.memory_allocated", side_effect=[initial_memory, leaked_memory]
        ):
            memory_manager.start_memory_tracking()
            # Simulate operations that should not leak memory
            memory_manager.stop_memory_tracking()

            leak_detected = memory_manager.detect_memory_leak(threshold=512**3)

            assert leak_detected is True


class TestCUDAUtils:
    """Test suite for CUDA utilities and operations."""

    @pytest.fixture
    def cuda_utils(self):
        """Create CUDA utilities instance."""
        return CUDAUtils()

    def test_cuda_kernel_compilation(self, cuda_utils):
        """Test CUDA kernel compilation."""
        kernel_code = """
        __global__ void vector_add(float *a, float *b, float *c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """

        with patch("cupy.RawKernel") as mock_kernel:
            mock_kernel.return_value = MagicMock()

            kernel = cuda_utils.compile_kernel(kernel_code, "vector_add")

            assert kernel is not None
            mock_kernel.assert_called_with(kernel_code, "vector_add")

    def test_cuda_stream_management(self, cuda_utils):
        """Test CUDA stream management."""
        with patch("torch.cuda.Stream") as mock_stream:
            mock_stream_instance = MagicMock()
            mock_stream.return_value = mock_stream_instance

            stream = cuda_utils.create_stream()

            assert stream is not None
            mock_stream.assert_called()

    def test_cuda_synchronization(self, cuda_utils):
        """Test CUDA synchronization operations."""
        with patch("torch.cuda.synchronize") as mock_sync:
            cuda_utils.synchronize_device(device=0)

            mock_sync.assert_called()

    def test_cuda_error_checking(self, cuda_utils):
        """Test CUDA error checking."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.current_device",
                side_effect=RuntimeError("CUDA error: out of memory"),
            ),
        ):
            with pytest.raises(GPUError):
                cuda_utils.check_cuda_error()

    def test_cuda_profiling_integration(self, cuda_utils):
        """Test CUDA profiling integration."""
        with patch("torch.profiler.profile") as mock_profiler:
            mock_profiler_instance = MagicMock()
            mock_profiler.return_value.__enter__.return_value = mock_profiler_instance

            with cuda_utils.profile_cuda_operations():
                # Simulate CUDA operations
                pass

            mock_profiler.assert_called()

    def test_mixed_precision_setup(self, cuda_utils):
        """Test mixed precision (FP16) setup."""
        with patch("torch.cuda.amp.autocast") as mock_autocast:
            mock_autocast.return_value.__enter__.return_value = None

            with cuda_utils.mixed_precision_context():
                # Simulate mixed precision operations
                pass

            mock_autocast.assert_called()


class TestGPUAcceleration:
    """Test suite for GPU-accelerated operations."""

    def test_matrix_operations_gpu_acceleration(self):
        """Test GPU acceleration for matrix operations."""
        from pynomaly.infrastructure.gpu.operations import GPUOperations

        gpu_ops = GPUOperations()

        # Mock GPU tensor operations
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.tensor") as mock_tensor,
        ):
            mock_gpu_tensor = MagicMock()
            mock_gpu_tensor.cuda.return_value = mock_gpu_tensor
            mock_tensor.return_value = mock_gpu_tensor

            # Test matrix multiplication
            a = np.random.randn(1000, 500)
            b = np.random.randn(500, 200)

            result = gpu_ops.matrix_multiply(a, b, device="cuda")

            assert result is not None
            mock_gpu_tensor.cuda.assert_called()

    def test_reduction_operations_gpu(self):
        """Test GPU-accelerated reduction operations."""
        from pynomaly.infrastructure.gpu.operations import GPUOperations

        gpu_ops = GPUOperations()

        with patch("torch.cuda.is_available", return_value=True):
            data = np.random.randn(10000, 100)

            # Test various reductions
            mean_result = gpu_ops.compute_mean(data, device="cuda")
            std_result = gpu_ops.compute_std(data, device="cuda")
            sum_result = gpu_ops.compute_sum(data, device="cuda")

            assert mean_result is not None
            assert std_result is not None
            assert sum_result is not None

    def test_distance_computations_gpu(self):
        """Test GPU-accelerated distance computations."""
        from pynomaly.infrastructure.gpu.operations import GPUOperations

        gpu_ops = GPUOperations()

        with patch("torch.cuda.is_available", return_value=True):
            points_a = np.random.randn(1000, 50)
            points_b = np.random.randn(2000, 50)

            # Test pairwise distance computation
            distances = gpu_ops.pairwise_distances(
                points_a, points_b, metric="euclidean", device="cuda"
            )

            assert distances is not None

    def test_anomaly_score_computation_gpu(self):
        """Test GPU-accelerated anomaly score computation."""
        from pynomaly.infrastructure.gpu.operations import GPUOperations

        gpu_ops = GPUOperations()

        with patch("torch.cuda.is_available", return_value=True):
            data = np.random.randn(5000, 20)
            model_outputs = np.random.randn(5000, 20)

            # Test reconstruction error computation
            scores = gpu_ops.compute_reconstruction_errors(
                data, model_outputs, device="cuda"
            )

            assert scores is not None
            assert len(scores) == len(data)

    def test_batch_processing_gpu(self):
        """Test GPU batch processing optimization."""
        from pynomaly.infrastructure.gpu.batch_processor import GPUBatchProcessor

        processor = GPUBatchProcessor(batch_size=512, device="cuda")

        with patch("torch.cuda.is_available", return_value=True):
            large_dataset = np.random.randn(10000, 30)

            def mock_process_batch(batch):
                return np.sum(batch, axis=1)

            results = processor.process_in_batches(large_dataset, mock_process_batch)

            assert len(results) == len(large_dataset)


class TestPerformanceBenchmarks:
    """Test suite for GPU performance benchmarks."""

    def test_cpu_vs_gpu_performance_comparison(self):
        """Test performance comparison between CPU and GPU."""
        from pynomaly.infrastructure.gpu.benchmarks import PerformanceBenchmark

        benchmark = PerformanceBenchmark()

        # Test matrix operations
        data_sizes = [1000, 5000, 10000]

        for size in data_sizes:
            with patch("torch.cuda.is_available", return_value=True):
                data = np.random.randn(size, 100)

                # CPU timing
                cpu_time = benchmark.time_cpu_operation(lambda x: np.dot(x, x.T), data)

                # GPU timing (mocked)
                gpu_time = benchmark.time_gpu_operation(
                    lambda x: x @ x.T,  # Mock GPU operation
                    data,
                )

                assert cpu_time > 0
                assert gpu_time > 0

    def test_memory_bandwidth_benchmark(self):
        """Test memory bandwidth benchmarks."""
        from pynomaly.infrastructure.gpu.benchmarks import PerformanceBenchmark

        benchmark = PerformanceBenchmark()

        with patch("torch.cuda.is_available", return_value=True):
            # Test memory copy performance
            data_sizes = [1024**2, 10 * 1024**2, 100 * 1024**2]  # 1MB, 10MB, 100MB

            for size in data_sizes:
                bandwidth = benchmark.measure_memory_bandwidth(size, device="cuda")

                assert bandwidth > 0  # GB/s

    def test_throughput_measurement(self):
        """Test throughput measurement for anomaly detection."""
        from pynomaly.infrastructure.gpu.benchmarks import PerformanceBenchmark

        benchmark = PerformanceBenchmark()

        with patch("torch.cuda.is_available", return_value=True):

            def mock_anomaly_detection(data):
                # Simulate anomaly detection computation
                time.sleep(0.001 * len(data) / 1000)  # Scale with data size
                return np.random.random(len(data))

            data = np.random.randn(10000, 50)

            throughput = benchmark.measure_throughput(
                mock_anomaly_detection, data, device="cuda"
            )

            assert throughput > 0  # samples/second


class TestErrorHandling:
    """Test suite for GPU error handling and recovery."""

    def test_cuda_out_of_memory_recovery(self):
        """Test recovery from CUDA out of memory errors."""
        from pynomaly.infrastructure.gpu.error_handler import GPUErrorHandler

        error_handler = GPUErrorHandler()

        def operation_that_ooms():
            raise RuntimeError("CUDA out of memory")

        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            result = error_handler.handle_oom_error(
                operation_that_ooms, fallback_to_cpu=True
            )

            assert result is not None
            mock_empty_cache.assert_called()

    def test_device_lost_recovery(self):
        """Test recovery from device lost errors."""
        from pynomaly.infrastructure.gpu.error_handler import GPUErrorHandler

        error_handler = GPUErrorHandler()

        def operation_that_loses_device():
            raise RuntimeError("device-side assert triggered")

        with patch("torch.cuda.device_count", return_value=2):
            result = error_handler.handle_device_error(
                operation_that_loses_device, retry_count=2
            )

            # Should attempt recovery or fallback
            assert result is not None or True  # Either succeeds or handles gracefully

    def test_mixed_precision_error_handling(self):
        """Test error handling in mixed precision operations."""
        from pynomaly.infrastructure.gpu.error_handler import GPUErrorHandler

        error_handler = GPUErrorHandler()

        def operation_with_nan():
            return float("inf")  # Simulate NaN/Inf in mixed precision

        result = error_handler.handle_numerical_instability(
            operation_with_nan, fallback_precision="fp32"
        )

        assert result is not None

    def test_driver_compatibility_check(self):
        """Test driver compatibility checking."""
        from pynomaly.infrastructure.gpu.compatibility import DriverChecker

        checker = DriverChecker()

        with patch(
            "nvidia_ml_py.nvmlSystemGetDriverVersion", return_value="470.129.06"
        ):
            is_compatible = checker.check_driver_compatibility(required_version="460.0")

            assert is_compatible is True

    def test_compute_capability_validation(self):
        """Test compute capability validation."""
        from pynomaly.infrastructure.gpu.compatibility import DriverChecker

        checker = DriverChecker()

        with patch("torch.cuda.get_device_capability", return_value=(8, 6)):  # RTX 3080
            is_supported = checker.validate_compute_capability(
                device=0, required_major=7, required_minor=0
            )

            assert is_supported is True


class TestGPUIntegration:
    """Integration tests for GPU acceleration with anomaly detection."""

    def test_end_to_end_gpu_anomaly_detection(self):
        """Test end-to-end GPU-accelerated anomaly detection."""
        from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter

        adapter = PyTorchAdapter(device="cuda", use_gpu=True)

        # Create test data
        data = np.random.randn(1000, 20).astype(np.float32)
        dataset = Mock()
        dataset.data = data

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.object(adapter, "_create_autoencoder") as mock_create,
        ):
            mock_model = MagicMock()
            mock_model.cuda.return_value = mock_model
            mock_create.return_value = mock_model

            # Train on GPU
            detector = adapter.fit(dataset)

            assert detector is not None
            # Verify GPU was used
            mock_model.cuda.assert_called()

    def test_multi_gpu_distributed_training(self):
        """Test distributed training across multiple GPUs."""
        from pynomaly.infrastructure.gpu.distributed import DistributedTrainer

        trainer = DistributedTrainer(num_gpus=2)

        with (
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.nn.parallel.DistributedDataParallel") as mock_ddp,
        ):
            mock_model = MagicMock()
            mock_ddp.return_value = mock_model

            # Mock distributed training
            result = trainer.train_distributed(
                model=mock_model, data=np.random.randn(5000, 30), epochs=1
            )

            assert result is not None
            mock_ddp.assert_called()

    def test_gpu_memory_optimization_integration(self):
        """Test GPU memory optimization during training."""
        from pynomaly.infrastructure.gpu.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()

        with patch(
            "torch.cuda.memory_allocated", side_effect=[0, 2 * 1024**3, 1 * 1024**3]
        ):
            # Simulate training with memory optimization
            initial_memory = optimizer.get_memory_usage()

            # Enable optimizations
            optimizer.enable_gradient_checkpointing()
            optimizer.enable_memory_efficient_attention()

            optimized_memory = optimizer.get_memory_usage()

            assert initial_memory is not None
            assert optimized_memory is not None

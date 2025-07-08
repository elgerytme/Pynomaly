"""Tests for performance configuration v2 functionality."""

import json
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

import pytest

from pynomaly.application.services.performance_benchmarking_service import (
    BenchmarkConfig,
    PerformanceBenchmarkingService,
    PerformanceMetrics,
)


class TestPerformanceConfigV2:
    """Test cases for performance configuration v2."""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config_data = {
                'version': 2,
                'metadata': {
                    'schema_version': '2.0',
                    'python_version': '3.11+',
                    'os': 'cross-platform',
                    'architecture': 'x86_64'
                },
                'performance_thresholds': {
                    'execution_time': {
                        'max_execution_time_seconds': 250.0,
                        'performance_degradation_threshold_percent': 15.0
                    },
                    'memory_usage': {
                        'max_memory_usage_mb': 3072.0,
                        'max_peak_memory_mb': 1536.0
                    },
                    'throughput': {
                        'min_throughput_samples_per_second': 25.0
                    },
                    'quality_metrics': {
                        'min_accuracy_score': 0.75,
                        'min_f1_score': 0.70
                    }
                },
                'algorithm_thresholds': {
                    'fast_algorithms': {
                        'algorithms': ['TestFastAlgorithm']
                    },
                    'heavy_algorithms': {
                        'algorithms': ['TestHeavyAlgorithm']
                    }
                }
            }
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        yield temp_path
        temp_path.unlink()

    @pytest.fixture
    def temp_v1_config_file(self):
        """Create temporary v1 config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'performance_metrics': {
                    'cli_startup_time': 900.0,
                    'container_init_time': 350.0,
                    'basic_workflow_time': 500.0,
                    'baseline_memory_mb': 90.0,
                    'peak_memory_mb': 150.0,
                    'end_memory_mb': 95.0,
                    'domain_entities_import_time': 140.0,
                    'application_services_import_time': 300.0
                }
            }
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        yield temp_path
        temp_path.unlink()

    def test_load_v2_config(self, temp_config_file, tmp_path):
        """Test loading v2 configuration."""
        with patch('pynomaly.application.services.performance_benchmarking_service.Path') as mock_path:
            def mock_path_func(path_str):
                if path_str == "performance_config.yml":
                    return temp_config_file
                return Path(path_str)
            
            mock_path.side_effect = mock_path_func
            
            service = PerformanceBenchmarkingService(storage_path=tmp_path)
            
            assert service.is_v2_config()
            assert service.get_threshold('performance_thresholds.execution_time.max_execution_time_seconds') == 250.0
            assert service.get_threshold('performance_thresholds.memory_usage.max_memory_usage_mb') == 3072.0

    def test_load_v1_config_with_migration(self, temp_v1_config_file, tmp_path):
        """Test loading v1 configuration with migration."""
        with patch('pynomaly.application.services.performance_benchmarking_service.Path') as mock_path:
            # Mock the first path (v2 config) to not exist
            mock_path.side_effect = lambda x: temp_v1_config_file if 'performance_baseline.json' in str(x) else Path(x)
            
            def mock_exists(self):
                return 'performance_baseline.json' in str(self)
            
            Path.exists = mock_exists
            
            service = PerformanceBenchmarkingService(storage_path=tmp_path)
            
            # Should migrate v1 to v2 structure
            assert service.performance_config.get('version') == 2
            assert 'performance_thresholds' in service.performance_config

    def test_get_threshold_with_fallback(self, temp_config_file, tmp_path):
        """Test get_threshold with fallback values."""
        with patch('pynomaly.application.services.performance_benchmarking_service.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: str(temp_config_file)
            mock_path.return_value.open = temp_config_file.open
            
            service = PerformanceBenchmarkingService(storage_path=tmp_path)
            
            # Test existing threshold
            assert service.get_threshold('performance_thresholds.execution_time.max_execution_time_seconds', 300.0) == 250.0
            
            # Test non-existing threshold with fallback
            assert service.get_threshold('non.existing.threshold', 999.0) == 999.0

    def test_benchmark_config_apply_performance_config(self, temp_config_file, tmp_path):
        """Test BenchmarkConfig applying performance configuration."""
        with patch('pynomaly.application.services.performance_benchmarking_service.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: str(temp_config_file)
            mock_path.return_value.open = temp_config_file.open
            
            service = PerformanceBenchmarkingService(storage_path=tmp_path)
            config = BenchmarkConfig()
            
            # Apply configuration
            config.apply_performance_config(service)
            
            # Check that thresholds were updated
            assert config.max_execution_time_seconds == 250.0
            assert config.max_memory_usage_mb == 3072.0
            assert config.min_throughput_samples_per_second == 25.0

    def test_validate_performance_metrics(self, temp_config_file, tmp_path):
        """Test performance metrics validation."""
        with patch('pynomaly.application.services.performance_benchmarking_service.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: str(temp_config_file)
            mock_path.return_value.open = temp_config_file.open
            
            service = PerformanceBenchmarkingService(storage_path=tmp_path)
            
            # Test metrics that pass validation
            good_metrics = PerformanceMetrics(
                execution_time_seconds=200.0,  # < 250.0
                peak_memory_mb=1000.0,  # < 1536.0
                training_throughput=30.0,  # > 25.0
                accuracy_score=0.80,  # > 0.75
                f1_score=0.75,  # > 0.70
                cpu_usage_percent=50.0  # < 80.0 (default)
            )
            
            validation_results = service.validate_performance_metrics(good_metrics)
            
            assert validation_results['execution_time_ok'] is True
            assert validation_results['memory_usage_ok'] is True
            assert validation_results['throughput_ok'] is True
            assert validation_results['accuracy_ok'] is True
            assert validation_results['f1_score_ok'] is True
            
            # Test metrics that fail validation
            bad_metrics = PerformanceMetrics(
                execution_time_seconds=300.0,  # > 250.0
                peak_memory_mb=2000.0,  # > 1536.0
                training_throughput=20.0,  # < 25.0
                accuracy_score=0.65,  # < 0.75
                f1_score=0.60,  # < 0.70
                cpu_usage_percent=90.0  # > 80.0 (default)
            )
            
            validation_results = service.validate_performance_metrics(bad_metrics)
            
            assert validation_results['execution_time_ok'] is False
            assert validation_results['memory_usage_ok'] is False
            assert validation_results['throughput_ok'] is False
            assert validation_results['accuracy_ok'] is False
            assert validation_results['f1_score_ok'] is False

    def test_check_performance_regression(self, temp_config_file, tmp_path):
        """Test performance regression checking."""
        with patch('pynomaly.application.services.performance_benchmarking_service.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: str(temp_config_file)
            mock_path.return_value.open = temp_config_file.open
            
            service = PerformanceBenchmarkingService(storage_path=tmp_path)
            
            baseline_metrics = PerformanceMetrics(
                execution_time_seconds=100.0,
                peak_memory_mb=1000.0,
                accuracy_score=0.85
            )
            
            # Test regression (20% slower, threshold is 15%)
            current_metrics = PerformanceMetrics(
                execution_time_seconds=120.0,  # 20% increase
                peak_memory_mb=1000.0,
                accuracy_score=0.85
            )
            
            regression_results = service.check_performance_regression(current_metrics, baseline_metrics)
            
            assert regression_results['execution_time_regression'] is True
            assert regression_results['execution_time_change_percent'] == 20.0

    def test_get_algorithm_specific_thresholds(self, temp_config_file, tmp_path):
        """Test algorithm-specific threshold retrieval."""
        with patch('pynomaly.application.services.performance_benchmarking_service.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: str(temp_config_file)
            mock_path.return_value.open = temp_config_file.open
            
            service = PerformanceBenchmarkingService(storage_path=tmp_path)
            
            # Test fast algorithm
            fast_thresholds = service.get_algorithm_specific_thresholds('TestFastAlgorithm')
            assert 'max_execution_time' in fast_thresholds
            assert 'max_memory_usage' in fast_thresholds
            
            # Test heavy algorithm  
            heavy_thresholds = service.get_algorithm_specific_thresholds('TestHeavyAlgorithm')
            assert 'max_execution_time' in heavy_thresholds
            assert 'max_memory_usage' in heavy_thresholds
            
            # Test unknown algorithm (should get defaults)
            default_thresholds = service.get_algorithm_specific_thresholds('UnknownAlgorithm')
            assert 'max_execution_time' in default_thresholds
            assert 'max_memory_usage' in default_thresholds

    def test_migration_v1_to_v2(self, temp_v1_config_file, tmp_path):
        """Test migration from v1 to v2 configuration."""
        with patch('pynomaly.application.services.performance_benchmarking_service.Path') as mock_path:
            mock_path.side_effect = lambda x: temp_v1_config_file if 'performance_baseline.json' in str(x) else Path(x)
            
            def mock_exists(self):
                return 'performance_baseline.json' in str(self)
            
            Path.exists = mock_exists
            
            service = PerformanceBenchmarkingService(storage_path=tmp_path)
            
            # Check that migration occurred
            assert service.performance_config.get('version') == 2
            assert 'performance_thresholds' in service.performance_config
            assert 'import_time_thresholds' in service.performance_config
            
            # Check specific migrated values
            assert service.performance_config['performance_thresholds']['execution_time']['cli_startup_time_ms'] == 900.0
            assert service.performance_config['performance_thresholds']['memory_usage']['baseline_memory_mb'] == 90.0
            assert service.performance_config['import_time_thresholds']['domain_entities_import_time'] == 140.0

    @pytest.mark.asyncio
    async def test_create_benchmark_suite_applies_config(self, temp_config_file, tmp_path):
        """Test that create_benchmark_suite applies performance configuration."""
        with patch('pynomaly.application.services.performance_benchmarking_service.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: str(temp_config_file)
            mock_path.return_value.open = temp_config_file.open
            
            service = PerformanceBenchmarkingService(storage_path=tmp_path)
            config = BenchmarkConfig()
            
            # Initial values should be defaults
            assert config.max_execution_time_seconds == 300.0
            
            # Create benchmark suite (should apply config)
            suite_id = await service.create_benchmark_suite(
                suite_name="Test Suite",
                description="Test Description",
                config=config
            )
            
            # Config should now have values from configuration file
            assert config.max_execution_time_seconds == 250.0
            assert config.max_memory_usage_mb == 3072.0
            assert config.min_throughput_samples_per_second == 25.0
            
            # Verify suite was created with updated config
            suite = service.benchmark_results[suite_id]
            assert suite.config.max_execution_time_seconds == 250.0

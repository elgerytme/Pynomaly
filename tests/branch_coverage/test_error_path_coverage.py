"""
Branch Coverage Enhancement - Error Path Testing
Comprehensive tests targeting error handling paths, exception branches, and failure scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, side_effect
from datetime import datetime, timezone
import sys
import os
import tempfile
import threading
import time
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.domain.exceptions import (
    DetectorNotFittedError, FittingError, InvalidAlgorithmError,
    ValidationError, ConfigurationError, AdapterError
)


class TestFileOperationErrorPaths:
    """Test file operation error handling branches."""

    def test_file_loading_error_paths(self):
        """Test file loading error scenarios."""
        from pynomaly.infrastructure.data.loaders import FileLoader
        
        loader = FileLoader()
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            loader.load_csv('/nonexistent/path/file.csv')
        
        # Test permission denied
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            f.write(b"col1,col2\n1,2\n3,4")
        
        try:
            # Change permissions to read-only for owner only
            os.chmod(temp_path, 0o000)
            
            with pytest.raises(PermissionError):
                loader.load_csv(temp_path)
        finally:
            # Restore permissions and cleanup
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)
        
        # Test corrupted CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2,3\n4,5")  # Inconsistent column count
            corrupted_csv_path = f.name
        
        try:
            with pytest.raises((pd.errors.ParserError, pd.errors.DtypeWarning)):
                loader.load_csv(corrupted_csv_path, strict_parsing=True)
        finally:
            os.unlink(corrupted_csv_path)
        
        # Test empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")  # Empty file
            empty_file_path = f.name
        
        try:
            with pytest.raises(pd.errors.EmptyDataError):
                loader.load_csv(empty_file_path)
        finally:
            os.unlink(empty_file_path)
        
        # Test unsupported file format
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_file("data.xyz")

    def test_database_connection_error_paths(self):
        """Test database connection error scenarios."""
        from pynomaly.infrastructure.data.database import DatabaseLoader
        
        # Test connection timeout
        with patch('sqlalchemy.create_engine') as mock_engine:
            mock_engine.side_effect = TimeoutError("Connection timeout")
            
            loader = DatabaseLoader()
            with pytest.raises(TimeoutError):
                loader.connect("postgresql://user:pass@unreachable:5432/db")
        
        # Test invalid connection string
        with patch('sqlalchemy.create_engine') as mock_engine:
            mock_engine.side_effect = ValueError("Invalid connection string")
            
            loader = DatabaseLoader()
            with pytest.raises(ValueError):
                loader.connect("invalid://connection/string")
        
        # Test authentication failure
        with patch('sqlalchemy.create_engine') as mock_engine:
            mock_engine.side_effect = Exception("Authentication failed")
            
            loader = DatabaseLoader()
            with pytest.raises(Exception, match="Authentication failed"):
                loader.connect("postgresql://wrong:credentials@host:5432/db")
        
        # Test query execution error
        with patch('sqlalchemy.create_engine') as mock_engine:
            mock_connection = Mock()
            mock_connection.execute.side_effect = Exception("Table 'nonexistent' doesn't exist")
            mock_engine.return_value.connect.return_value = mock_connection
            
            loader = DatabaseLoader()
            loader.engine = mock_engine.return_value
            
            with pytest.raises(Exception, match="doesn't exist"):
                loader.load_query("SELECT * FROM nonexistent")

    def test_serialization_error_paths(self):
        """Test serialization/deserialization error scenarios."""
        from pynomaly.infrastructure.persistence.serializer import ModelSerializer
        
        serializer = ModelSerializer()
        
        # Test serialization of non-serializable object
        class NonSerializable:
            def __getstate__(self):
                raise Exception("Cannot serialize this object")
        
        non_serializable = NonSerializable()
        with pytest.raises(Exception, match="Cannot serialize"):
            serializer.serialize(non_serializable)
        
        # Test deserialization of corrupted data
        corrupted_data = b"corrupted binary data that's not a pickle"
        with pytest.raises(Exception):
            serializer.deserialize(corrupted_data)
        
        # Test deserialization with security risk
        malicious_pickle = b"malicious pickle data"
        with patch('pickle.loads') as mock_loads:
            mock_loads.side_effect = Exception("Security risk detected")
            
            with pytest.raises(Exception, match="Security risk"):
                serializer.deserialize(malicious_pickle)

    def test_memory_error_paths(self):
        """Test memory-related error scenarios."""
        from pynomaly.infrastructure.data.processors import DataProcessor
        
        processor = DataProcessor()
        
        # Test processing extremely large dataset
        with patch('pandas.DataFrame') as mock_df:
            mock_df.side_effect = MemoryError("Not enough memory")
            
            with pytest.raises(MemoryError):
                processor.create_large_dataset(rows=10**9, columns=10**6)
        
        # Test memory allocation failure during computation
        with patch('numpy.zeros') as mock_zeros:
            mock_zeros.side_effect = MemoryError("Cannot allocate memory")
            
            with pytest.raises(MemoryError):
                processor.allocate_computation_matrix(size=(10**6, 10**6))


class TestConcurrencyErrorPaths:
    """Test concurrency and threading error scenarios."""

    def test_thread_safety_error_paths(self):
        """Test thread safety error scenarios."""
        from pynomaly.infrastructure.adapters.base_adapter import BaseAdapter
        
        adapter = BaseAdapter()
        errors = []
        
        def concurrent_operation(operation_id):
            try:
                # Simulate concurrent access to shared resource
                adapter._shared_resource = operation_id
                time.sleep(0.01)  # Small delay to increase race condition chance
                
                if adapter._shared_resource != operation_id:
                    errors.append(f"Race condition detected in operation {operation_id}")
            except Exception as e:
                errors.append(f"Exception in operation {operation_id}: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check for race conditions
        if len(errors) > 0:
            pytest.fail(f"Concurrency errors detected: {errors}")

    def test_deadlock_prevention(self):
        """Test deadlock prevention mechanisms."""
        from pynomaly.infrastructure.concurrency.locks import DeadlockDetector
        
        detector = DeadlockDetector()
        
        # Simulate potential deadlock scenario
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        deadlock_detected = []
        
        def thread1():
            try:
                with detector.acquire_with_timeout(lock1, timeout=1.0):
                    time.sleep(0.1)
                    with detector.acquire_with_timeout(lock2, timeout=1.0):
                        pass
            except TimeoutError:
                deadlock_detected.append("Thread1 timeout")
        
        def thread2():
            try:
                with detector.acquire_with_timeout(lock2, timeout=1.0):
                    time.sleep(0.1)
                    with detector.acquire_with_timeout(lock1, timeout=1.0):
                        pass
            except TimeoutError:
                deadlock_detected.append("Thread2 timeout")
        
        t1 = threading.Thread(target=thread1)
        t2 = threading.Thread(target=thread2)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # At least one thread should timeout to prevent deadlock
        assert len(deadlock_detected) >= 1


class TestResourceExhaustionPaths:
    """Test resource exhaustion error scenarios."""

    def test_file_descriptor_exhaustion(self):
        """Test file descriptor exhaustion handling."""
        from pynomaly.infrastructure.resources.file_manager import FileManager
        
        manager = FileManager()
        open_files = []
        
        try:
            # Try to open many files to exhaust descriptors
            for i in range(1000):
                try:
                    f = manager.open_temp_file()
                    open_files.append(f)
                except OSError as e:
                    if "Too many open files" in str(e):
                        # Expected behavior - resource exhaustion handled
                        break
                    else:
                        raise
        finally:
            # Cleanup
            for f in open_files:
                try:
                    f.close()
                except:
                    pass

    def test_disk_space_exhaustion(self):
        """Test disk space exhaustion handling."""
        from pynomaly.infrastructure.storage.manager import StorageManager
        
        manager = StorageManager()
        
        # Mock disk space check
        with patch('shutil.disk_usage') as mock_disk_usage:
            # Simulate full disk
            mock_disk_usage.return_value = (1000, 0, 0)  # total, used, free
            
            with pytest.raises(OSError, match="No space left"):
                manager.write_large_file("/tmp/large_file.dat", size_mb=100)

    def test_network_timeout_paths(self):
        """Test network timeout error scenarios."""
        from pynomaly.infrastructure.network.client import NetworkClient
        
        client = NetworkClient()
        
        # Test connection timeout
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectTimeout("Connection timeout")
            
            with pytest.raises(requests.exceptions.ConnectTimeout):
                client.fetch_data("http://unreachable-server.com/data")
        
        # Test read timeout
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ReadTimeout("Read timeout")
            
            with pytest.raises(requests.exceptions.ReadTimeout):
                client.fetch_data("http://slow-server.com/data")
        
        # Test retry mechanism
        with patch('requests.get') as mock_get:
            mock_get.side_effect = [
                requests.exceptions.ConnectTimeout("Timeout 1"),
                requests.exceptions.ConnectTimeout("Timeout 2"),
                Mock(status_code=200, json=lambda: {"data": "success"})
            ]
            
            # Should succeed after retries
            result = client.fetch_data_with_retry("http://flaky-server.com/data", max_retries=3)
            assert result["data"] == "success"


class TestValidationErrorPaths:
    """Test validation error scenarios and edge cases."""

    def test_data_type_validation_errors(self):
        """Test data type validation error paths."""
        from pynomaly.infrastructure.validation.type_validator import TypeValidator
        
        validator = TypeValidator()
        
        # Test invalid numeric data
        invalid_numeric_data = pd.DataFrame({
            'feature': ['not_a_number', 'also_not_number', 'definitely_not']
        })
        
        with pytest.raises(ValidationError, match="non-numeric"):
            validator.validate_numeric_dataframe(invalid_numeric_data)
        
        # Test mixed valid/invalid data
        mixed_data = pd.DataFrame({
            'valid_numeric': [1, 2, 3],
            'invalid_numeric': ['a', 'b', 'c'],
            'valid_float': [1.1, 2.2, 3.3]
        })
        
        validation_result = validator.validate_mixed_dataframe(mixed_data)
        assert not validation_result.is_valid
        assert 'invalid_numeric' in str(validation_result.errors)
        
        # Test edge case: DataFrame with no columns
        empty_columns_df = pd.DataFrame()
        with pytest.raises(ValidationError, match="no columns"):
            validator.validate_dataframe_structure(empty_columns_df)
        
        # Test edge case: DataFrame with no rows
        empty_rows_df = pd.DataFrame(columns=['col1', 'col2'])
        with pytest.raises(ValidationError, match="no rows"):
            validator.validate_dataframe_structure(empty_rows_df)

    def test_parameter_validation_edge_cases(self):
        """Test parameter validation edge cases."""
        from pynomaly.infrastructure.validation.parameter_validator import ParameterValidator
        
        validator = ParameterValidator()
        
        # Test parameter type coercion failure
        with pytest.raises(ValidationError, match="cannot be coerced"):
            validator.validate_and_coerce("not_a_number", int)
        
        # Test nested parameter validation
        nested_params = {
            'algorithm': {
                'name': 'IsolationForest',
                'params': {
                    'n_estimators': 'invalid_number'  # Should be int
                }
            }
        }
        
        with pytest.raises(ValidationError, match="nested parameter"):
            validator.validate_nested_parameters(nested_params)
        
        # Test circular reference in parameters
        circular_params = {}
        circular_params['self_ref'] = circular_params
        
        with pytest.raises(ValidationError, match="circular reference"):
            validator.validate_parameter_structure(circular_params)
        
        # Test parameter with function/lambda (security risk)
        dangerous_params = {
            'callback': lambda x: os.system('rm -rf /')
        }
        
        with pytest.raises(ValidationError, match="functions not allowed"):
            validator.validate_parameter_security(dangerous_params)

    def test_schema_validation_errors(self):
        """Test schema validation error paths."""
        from pynomaly.infrastructure.validation.schema_validator import SchemaValidator
        
        validator = SchemaValidator()
        
        # Test schema mismatch
        expected_schema = {
            'required_fields': ['field1', 'field2'],
            'field_types': {
                'field1': str,
                'field2': int,
                'field3': float
            }
        }
        
        invalid_data = {
            'field1': 'valid_string',
            'field2': 'invalid_int',  # Should be int
            'missing_field3': 1.0     # Wrong field name
        }
        
        with pytest.raises(ValidationError, match="schema validation failed"):
            validator.validate_against_schema(invalid_data, expected_schema)
        
        # Test version compatibility
        old_version_data = {
            'version': '1.0.0',
            'data': {'old_format': True}
        }
        
        current_version_schema = {
            'min_version': '2.0.0',
            'required_fields': ['new_format']
        }
        
        with pytest.raises(ValidationError, match="version incompatible"):
            validator.validate_version_compatibility(old_version_data, current_version_schema)


class TestIntegrationErrorPaths:
    """Test integration error scenarios between components."""

    def test_adapter_integration_errors(self):
        """Test adapter integration error scenarios."""
        from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
        
        # Test adapter with missing dependencies
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("PyOD not installed")
            
            with pytest.raises(InvalidAlgorithmError, match="PyOD not installed"):
                adapter = PyODAdapter(algorithm_name="IsolationForest")
        
        # Test adapter with incompatible version
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.__version__ = "0.5.0"  # Old version
            mock_import.return_value = mock_module
            
            adapter = PyODAdapter(algorithm_name="IsolationForest")
            
            if hasattr(adapter, '_check_version_compatibility'):
                with pytest.raises(ValueError, match="incompatible version"):
                    adapter._check_version_compatibility(min_version="1.0.0")
        
        # Test adapter with corrupted model state
        adapter = PyODAdapter(algorithm_name="IsolationForest")
        
        # Mock a corrupted fitted model
        corrupted_model = Mock()
        corrupted_model.predict.side_effect = Exception("Model corrupted")
        adapter._model = corrupted_model
        adapter.is_fitted = True
        
        sample_data = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        sample_dataset = Mock()
        sample_dataset.features = sample_data
        sample_dataset.get_numeric_features.return_value = ['feature']
        
        with pytest.raises(Exception, match="Model corrupted"):
            adapter.detect(sample_dataset)

    def test_pipeline_integration_errors(self):
        """Test pipeline integration error scenarios."""
        from pynomaly.application.pipelines.detection_pipeline import DetectionPipeline
        
        pipeline = DetectionPipeline()
        
        # Test pipeline with incompatible components
        incompatible_preprocessor = Mock()
        incompatible_preprocessor.output_format = "format_a"
        
        incompatible_detector = Mock()
        incompatible_detector.required_input_format = "format_b"
        
        with pytest.raises(ValueError, match="incompatible formats"):
            pipeline.add_step(incompatible_preprocessor)
            pipeline.add_step(incompatible_detector)
            pipeline.validate_pipeline()
        
        # Test pipeline execution with step failure
        failing_step = Mock()
        failing_step.execute.side_effect = Exception("Step execution failed")
        
        pipeline = DetectionPipeline()
        pipeline.add_step(failing_step)
        
        with pytest.raises(Exception, match="Step execution failed"):
            pipeline.execute({"input": "data"})
        
        # Test pipeline with circular dependencies
        step_a = Mock()
        step_a.dependencies = ["step_b"]
        step_a.name = "step_a"
        
        step_b = Mock()
        step_b.dependencies = ["step_a"]
        step_b.name = "step_b"
        
        pipeline = DetectionPipeline()
        pipeline.add_step(step_a)
        pipeline.add_step(step_b)
        
        with pytest.raises(ValueError, match="circular dependency"):
            pipeline.resolve_dependencies()

    def test_configuration_integration_errors(self):
        """Test configuration integration error scenarios."""
        from pynomaly.infrastructure.config.manager import ConfigManager
        
        manager = ConfigManager()
        
        # Test conflicting configuration sources
        config_source_1 = {
            'algorithm': 'IsolationForest',
            'contamination': 0.1
        }
        
        config_source_2 = {
            'algorithm': 'LOF',  # Conflicting algorithm
            'contamination': 0.2  # Conflicting contamination
        }
        
        with pytest.raises(ConfigurationError, match="conflicting configurations"):
            manager.merge_configs([config_source_1, config_source_2], strict=True)
        
        # Test invalid configuration cross-references
        config_with_refs = {
            'database': {
                'host': '${nonexistent_var}',
                'port': 5432
            }
        }
        
        with pytest.raises(ConfigurationError, match="unresolved reference"):
            manager.resolve_config_references(config_with_refs)
        
        # Test configuration validation with missing environment
        env_dependent_config = {
            'mode': 'production',
            'debug': False,
            'database_url': '${DATABASE_URL}'  # Missing env var
        }
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="missing environment variable"):
                manager.validate_environment_config(env_dependent_config)


class TestPerformanceErrorPaths:
    """Test performance-related error scenarios."""

    def test_timeout_error_paths(self):
        """Test timeout error scenarios."""
        from pynomaly.infrastructure.performance.timeout_manager import TimeoutManager
        
        manager = TimeoutManager()
        
        # Test operation timeout
        def long_running_operation():
            time.sleep(10)  # Simulates long operation
            return "completed"
        
        with pytest.raises(TimeoutError):
            manager.execute_with_timeout(long_running_operation, timeout_seconds=1)
        
        # Test nested timeout
        def nested_operation():
            def inner_operation():
                time.sleep(5)
                return "inner_completed"
            
            return manager.execute_with_timeout(inner_operation, timeout_seconds=3)
        
        with pytest.raises(TimeoutError):
            manager.execute_with_timeout(nested_operation, timeout_seconds=2)

    def test_resource_limit_errors(self):
        """Test resource limit error scenarios."""
        from pynomaly.infrastructure.performance.resource_monitor import ResourceMonitor
        
        monitor = ResourceMonitor()
        
        # Test memory limit exceeded
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 95  # 95% memory usage
            
            with pytest.raises(ResourceError, match="memory limit exceeded"):
                monitor.check_memory_limit(max_memory_percent=90)
        
        # Test CPU limit exceeded
        with patch('psutil.cpu_percent') as mock_cpu:
            mock_cpu.return_value = 98  # 98% CPU usage
            
            with pytest.raises(ResourceError, match="CPU limit exceeded"):
                monitor.check_cpu_limit(max_cpu_percent=95)
        
        # Test disk space limit
        with patch('shutil.disk_usage') as mock_disk:
            mock_disk.return_value = (1000, 950, 50)  # total, used, free (95% used)
            
            with pytest.raises(ResourceError, match="disk space limit"):
                monitor.check_disk_space_limit("/tmp", max_usage_percent=90)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
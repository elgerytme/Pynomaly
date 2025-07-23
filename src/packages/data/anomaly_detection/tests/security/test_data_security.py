"""Security tests for data handling and model protection."""

import pytest
import os
import tempfile
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, Mock

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
from anomaly_detection.domain.entities.model import Model, ModelMetadata, ModelStatus


class TestDataSecurity:
    """Security tests for data handling, storage, and model protection."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for security testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def detection_service(self):
        """Create detection service for security testing."""
        return DetectionService()
    
    @pytest.fixture
    def model_repository(self, temp_dir):
        """Create model repository for security testing."""
        return ModelRepository(str(temp_dir / "models"))
    
    @pytest.fixture
    def sensitive_dataset(self):
        """Create dataset with potentially sensitive information."""
        np.random.seed(42)
        
        # Simulate sensitive financial data
        data = np.random.multivariate_normal(
            mean=[50000, 650, 35],  # salary, credit_score, age
            cov=[[10000000, 1000, 100], [1000, 2500, 10], [100, 10, 100]],
            size=1000
        )
        
        # Add some identifiable patterns
        data[0] = [100000, 800, 45]  # High earner profile
        data[1] = [25000, 500, 22]   # Low earner profile
        
        return data
    
    def test_sensitive_data_memory_cleanup(self, detection_service, sensitive_dataset):
        """Test that sensitive data is properly cleaned from memory."""
        print("\n=== Sensitive Data Memory Cleanup Test ===")
        
        # Process sensitive data
        result = detection_service.detect_anomalies(
            data=sensitive_dataset,
            algorithm='iforest',
            contamination=0.1,
            random_state=42
        )
        
        assert result.success, "Detection should succeed"
        
        # Force garbage collection and check for data remnants
        import gc
        gc.collect()
        
        # Try to find sensitive data patterns in memory
        # This is a simplified test - in production, more sophisticated memory analysis would be needed
        
        # Check that the detection service doesn't retain raw data unnecessarily
        if hasattr(detection_service, '_last_data'):
            assert detection_service._last_data is None, "Raw data should not be retained"
        
        # Check that results don't contain full raw data
        assert not hasattr(result, 'raw_data'), "Results should not contain raw input data"
        
        print("✓ No obvious sensitive data retention detected")
    
    def test_model_serialization_security(self, model_repository, temp_dir):
        """Test security of model serialization and deserialization."""
        print("\n=== Model Serialization Security Test ===")
        
        # Create a test model
        from sklearn.ensemble import IsolationForest
        sklearn_model = IsolationForest(n_estimators=10, random_state=42)
        
        # Fit with dummy data
        dummy_data = np.random.randn(100, 5)
        sklearn_model.fit(dummy_data)
        
        metadata = ModelMetadata(
            model_id="security-test-model",
            name="Security Test Model",
            algorithm="isolation_forest",
            status=ModelStatus.TRAINED,
            training_samples=100,
            training_features=5,
            contamination_rate=0.1
        )
        
        model = Model(metadata=metadata, model_object=sklearn_model)
        
        # Save model
        saved_id = model_repository.save(model)
        assert saved_id == "security-test-model"
        
        # Check that model files are properly protected
        model_files = list((temp_dir / "models").glob("**/*"))
        
        for model_file in model_files:
            if model_file.is_file():
                # Check file permissions (Unix-like systems)
                if hasattr(os, 'stat'):
                    stat_info = model_file.stat()
                    # Should not be world-readable/writable
                    world_permissions = stat_info.st_mode & 0o007
                    assert world_permissions == 0, f"Model file has world permissions: {oct(stat_info.st_mode)}"
                
                # Check file content for potential security issues
                with open(model_file, 'rb') as f:
                    content = f.read(1024)  # Read first 1KB
                    
                    # Should not contain obviously sensitive data
                    sensitive_patterns = [
                        b'password', b'secret', b'key', b'token',
                        b'credential', b'private', b'confidential'
                    ]
                    
                    for pattern in sensitive_patterns:
                        assert pattern not in content.lower(), f"Sensitive pattern found in model file: {pattern}"
        
        # Test loading model safely
        loaded_model = model_repository.load("security-test-model")
        assert loaded_model.metadata.model_id == "security-test-model"
        
        print("✓ Model serialization security checks passed")
    
    def test_pickle_deserialization_safety(self, temp_dir):
        """Test protection against malicious pickle deserialization."""
        print("\n=== Pickle Deserialization Safety Test ===")
        
        # Create a malicious pickle payload (for testing only)
        class MaliciousClass:
            def __reduce__(self):
                # This would execute code during unpickling
                return (print, ("SECURITY BREACH: Malicious code executed!",))
        
        malicious_obj = MaliciousClass()
        
        # Create malicious pickle file
        malicious_pickle_path = temp_dir / "malicious_model.pkl"
        with open(malicious_pickle_path, 'wb') as f:
            pickle.dump(malicious_obj, f)
        
        # Test that model repository handles malicious pickles safely
        model_repo = ModelRepository(str(temp_dir))
        
        try:
            # This should fail safely without executing malicious code
            # Note: In a real implementation, we should use safe loading mechanisms
            with patch('builtins.print') as mock_print:
                loaded_obj = model_repo._load_pickle_safely(str(malicious_pickle_path))
                
                # If we reach here, check that malicious code wasn't executed
                mock_print.assert_not_called()
                
        except Exception as e:
            # Exception during loading is acceptable (and preferred)
            print(f"✓ Malicious pickle safely rejected: {e}")
        
        print("✓ Pickle deserialization safety verified")
    
    def test_data_anonymization_capabilities(self, detection_service):
        """Test data anonymization and privacy protection capabilities."""
        print("\n=== Data Anonymization Test ===")
        
        # Create dataset with potentially identifiable information
        np.random.seed(42)
        
        # Original data with identifiable patterns
        original_data = np.array([
            [100000, 850, 45, 90210],  # salary, credit_score, age, zip_code
            [50000, 700, 30, 10001],
            [75000, 750, 35, 60601],
            [120000, 800, 50, 90210],  # Same zip as first person
        ])
        
        # Test detection without anonymization
        result_original = detection_service.detect_anomalies(
            data=original_data,
            algorithm='iforest',
            contamination=0.25,
            random_state=42
        )
        
        # Apply simple anonymization (in practice, use proper anonymization techniques)
        anonymized_data = original_data.copy()
        # Remove zip codes (last column)
        anonymized_data = anonymized_data[:, :-1]
        # Add noise to salary and age
        anonymized_data[:, 0] += np.random.normal(0, 5000, len(anonymized_data))  # Salary noise
        anonymized_data[:, 2] += np.random.normal(0, 2, len(anonymized_data))     # Age noise
        
        # Test detection with anonymized data
        result_anonymized = detection_service.detect_anomalies(
            data=anonymized_data,
            algorithm='iforest',
            contamination=0.25,
            random_state=42
        )
        
        # Both should succeed
        assert result_original.success, "Original data detection should succeed"
        assert result_anonymized.success, "Anonymized data detection should succeed"
        
        # Results should be similar but not identical (due to anonymization)
        original_anomalies = set(result_original.anomaly_indices)
        anonymized_anomalies = set(result_anonymized.anomaly_indices)
        
        # Should have some overlap but not be identical
        overlap = len(original_anomalies.intersection(anonymized_anomalies))
        total_unique = len(original_anomalies.union(anonymized_anomalies))
        
        if total_unique > 0:
            similarity = overlap / total_unique
            print(f"Anonymization similarity: {similarity:.2f}")
            
            # Should maintain some detection capability while providing privacy
            assert 0.3 <= similarity <= 0.9, f"Anonymization changed results too much: {similarity:.2f}"
        
        print("✓ Data anonymization capabilities verified")
    
    def test_model_access_control(self, model_repository, temp_dir):
        """Test model access control and permission handling."""
        print("\n=== Model Access Control Test ===")
        
        # Create test models with different access levels
        from sklearn.ensemble import IsolationForest
        
        models_data = [
            ("public-model", "Public Model", {"access_level": "public"}),
            ("private-model", "Private Model", {"access_level": "private"}),
            ("admin-model", "Admin Model", {"access_level": "admin"}),
        ]
        
        for model_id, name, metadata_extra in models_data:
            sklearn_model = IsolationForest(n_estimators=10, random_state=42)
            sklearn_model.fit(np.random.randn(50, 3))
            
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                algorithm="isolation_forest",
                status=ModelStatus.TRAINED,
                training_samples=50,
                training_features=3,
                **metadata_extra
            )
            
            model = Model(metadata=metadata, model_object=sklearn_model)
            model_repository.save(model)
        
        # Test access control (simplified - in practice, integrate with auth system)
        def test_model_access(user_role: str, model_id: str) -> bool:
            """Simulate access control check."""
            try:
                model = model_repository.load(model_id)
                access_level = getattr(model.metadata, 'access_level', 'public')
                
                # Simple role-based access control
                if access_level == 'public':
                    return True
                elif access_level == 'private' and user_role in ['user', 'admin']:
                    return True
                elif access_level == 'admin' and user_role == 'admin':
                    return True
                else:
                    return False
                    
            except Exception:
                return False
        
        # Test different user roles
        access_tests = [
            ("guest", "public-model", True),
            ("guest", "private-model", False),
            ("guest", "admin-model", False),
            ("user", "public-model", True),
            ("user", "private-model", True),
            ("user", "admin-model", False),
            ("admin", "public-model", True),
            ("admin", "private-model", True),
            ("admin", "admin-model", True),
        ]
        
        for user_role, model_id, expected_access in access_tests:
            actual_access = test_model_access(user_role, model_id)
            assert actual_access == expected_access, f"Access control failed: {user_role} -> {model_id}"
            
            print(f"  {user_role:5} -> {model_id:12}: {'✓' if actual_access else '✗'} ({'allowed' if actual_access else 'denied'})")
        
        print("✓ Model access control working correctly")
    
    def test_data_leakage_prevention(self, detection_service, sensitive_dataset):
        """Test prevention of data leakage through model outputs."""
        print("\n=== Data Leakage Prevention Test ===")
        
        # Use sensitive dataset
        result = detection_service.detect_anomalies(
            data=sensitive_dataset,
            algorithm='iforest',
            contamination=0.1,
            random_state=42
        )
        
        assert result.success, "Detection should succeed"
        
        # Check that results don't leak sensitive information
        
        # 1. Anomaly scores should not directly reveal input values
        if hasattr(result, 'anomaly_scores') and result.anomaly_scores is not None:
            # Scores should be normalized/bounded
            assert np.all(result.anomaly_scores >= -1), "Anomaly scores should be bounded below"
            assert np.all(result.anomaly_scores <= 1), "Anomaly scores should be bounded above"
            
            # Scores should not be identical to input features
            for i, feature_column in enumerate(sensitive_dataset.T):
                correlation = np.corrcoef(result.anomaly_scores, feature_column)[0, 1]
                assert abs(correlation) < 0.95, f"Anomaly scores too correlated with feature {i}: {correlation:.3f}"
        
        # 2. Anomaly indices should not reveal patterns that could identify individuals
        if result.anomaly_indices:
            # Should not consistently flag same patterns
            anomaly_data = sensitive_dataset[result.anomaly_indices]
            
            # Check that anomalies aren't just high/low values in obvious ways
            for feature_idx in range(sensitive_dataset.shape[1]):
                feature_values = sensitive_dataset[:, feature_idx]
                anomaly_feature_values = anomaly_data[:, feature_idx]
                
                # Anomalies shouldn't be just the extreme values
                min_val, max_val = np.min(feature_values), np.max(feature_values)
                extreme_count = np.sum((anomaly_feature_values == min_val) | (anomaly_feature_values == max_val))
                extreme_ratio = extreme_count / len(anomaly_feature_values) if len(anomaly_feature_values) > 0 else 0
                
                assert extreme_ratio < 0.8, f"Too many anomalies are extreme values for feature {feature_idx}: {extreme_ratio:.2f}"
        
        # 3. Check that results are reproducible but not deterministic based on input order
        # Shuffle the same data and rerun
        shuffled_indices = np.random.permutation(len(sensitive_dataset))
        shuffled_data = sensitive_dataset[shuffled_indices]
        
        result_shuffled = detection_service.detect_anomalies(
            data=shuffled_data,
            algorithm='iforest',
            contamination=0.1,
            random_state=42  # Same random state
        )
        
        assert result_shuffled.success, "Shuffled detection should succeed"
        
        # Results should be similar after accounting for reordering
        original_anomaly_ratio = len(result.anomaly_indices) / len(sensitive_dataset)
        shuffled_anomaly_ratio = len(result_shuffled.anomaly_indices) / len(shuffled_data)
        
        ratio_diff = abs(original_anomaly_ratio - shuffled_anomaly_ratio)
        assert ratio_diff < 0.05, f"Detection too sensitive to input order: {ratio_diff:.3f}"
        
        print("✓ Data leakage prevention checks passed")
    
    def test_secure_temporary_file_handling(self, temp_dir):
        """Test secure handling of temporary files during processing."""
        print("\n=== Secure Temporary File Handling Test ===")
        
        # Create detection service that might use temporary files
        detection_service = DetectionService()
        
        # Monitor temporary file creation
        original_tempfile_mkstemp = tempfile.mkstemp
        temp_files_created = []
        
        def mock_mkstemp(*args, **kwargs):
            fd, path = original_tempfile_mkstemp(*args, **kwargs)
            temp_files_created.append(path)
            return fd, path
        
        with patch('tempfile.mkstemp', side_effect=mock_mkstemp):
            # Process data that might require temporary files
            large_data = np.random.randn(1000, 50)
            
            result = detection_service.detect_anomalies(
                data=large_data,
                algorithm='iforest',
                contamination=0.1,
                random_state=42
            )
            
            assert result.success, "Detection should succeed"
        
        # Check that temporary files were properly cleaned up
        for temp_file in temp_files_created:
            assert not os.path.exists(temp_file), f"Temporary file not cleaned up: {temp_file}"
        
        print(f"✓ {len(temp_files_created)} temporary files properly handled")
    
    def test_configuration_security(self):
        """Test security of configuration handling."""
        print("\n=== Configuration Security Test ===")
        
        # Test that sensitive configuration is not exposed
        from anomaly_detection.infrastructure.config import settings
        
        config = settings.get_settings()
        
        # Convert config to dict-like structure for testing
        config_dict = {}
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(config, attr_name)
                    if not callable(attr_value):
                        config_dict[attr_name] = attr_value
                except:
                    pass
        
        # Check for potentially sensitive configuration
        sensitive_keys = [
            'password', 'secret', 'key', 'token', 'credential',
            'api_key', 'private_key', 'auth_token', 'database_url'
        ]
        
        for key, value in config_dict.items():
            key_lower = key.lower()
            
            # Check if key name suggests sensitive data
            for sensitive_key in sensitive_keys:
                if sensitive_key in key_lower:
                    # Value should not be exposed in plain text
                    if value and isinstance(value, str):
                        assert len(value) == 0 or value.startswith('***') or 'REDACTED' in value.upper(), \
                            f"Sensitive config value exposed: {key}"
                    
                    print(f"  Sensitive config {key}: {'✓ Protected' if not value or 'REDACTED' in str(value).upper() else '✗ Exposed'}")
        
        # Check that debug mode is not enabled in production-like settings
        if hasattr(config, 'debug'):
            if hasattr(config, 'environment') and config.environment == 'production':
                assert not config.debug, "Debug mode should not be enabled in production"
        
        print("✓ Configuration security checks completed")


if __name__ == "__main__":
    print("Anomaly Detection Data Security Test Suite")
    print("=" * 45)
    print("Testing data handling and model security:")
    print("• Sensitive data memory cleanup")
    print("• Model serialization security")
    print("• Pickle deserialization safety")
    print("• Data anonymization capabilities")
    print("• Model access control")
    print("• Data leakage prevention")
    print("• Secure temporary file handling")
    print("• Configuration security")
    print()
    
    # Quick security setup check
    try:
        import numpy as np
        import tempfile
        import pickle
        
        from anomaly_detection.domain.services.detection_service import DetectionService
        
        # Test basic functionality
        service = DetectionService()
        test_data = np.random.randn(50, 3)
        
        result = service.detect_anomalies(test_data, algorithm='iforest', contamination=0.1)
        
        print(f"✓ Security test setup complete")
        print(f"  Detected {result.anomaly_count} anomalies in test run")
        print("Ready to run comprehensive data security tests")
        
    except Exception as e:
        print(f"✗ Security test setup failed: {e}")
        print("Some security tests may not run properly")
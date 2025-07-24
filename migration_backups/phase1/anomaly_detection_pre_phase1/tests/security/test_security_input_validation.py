"""Security tests for input validation and data sanitization."""

import pytest
import json
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.api.v1.detection import detect_anomalies, train_model
from anomaly_detection.api.v1.models import load_model, save_model
from anomaly_detection.cli_new.commands.models import train_command, predict_command
from anomaly_detection.web.api.htmx import detection_endpoint, upload_data


@pytest.mark.security
class TestInputValidation:
    """Test input validation across all service layers."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
        self.ensemble_service = EnsembleService()
        self.streaming_service = StreamingService()
    
    def test_malicious_data_injection(self):
        """Test protection against malicious data injection."""
        malicious_payloads = [
            # SQL injection attempts
            "'; DROP TABLE models; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users",
            
            # Script injection
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "${7*7}",
            "{{7*7}}",
            
            # Command injection
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& whoami",
            
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            
            # Format string attacks
            "%s%s%s%s",
            "%x%x%x%x",
        ]
        
        for payload in malicious_payloads:
            with pytest.raises((ValueError, TypeError, AttributeError)):
                # Test detection service
                self.detection_service.detect_anomalies(
                    [[payload]], 
                    algorithm="isolation_forest"
                )
            
            with pytest.raises((ValueError, TypeError, AttributeError)):
                # Test with algorithm parameter
                self.detection_service.detect_anomalies(
                    [[1, 2, 3]], 
                    algorithm=payload
                )
    
    def test_oversized_input_protection(self):
        """Test protection against oversized inputs."""
        # Test extremely large datasets
        with pytest.raises((ValueError, MemoryError, OverflowError)):
            huge_data = [[1.0] * 10000 for _ in range(10000)]  # 100M elements
            self.detection_service.detect_anomalies(
                huge_data,
                algorithm="isolation_forest"
            )
        
        # Test deeply nested structures
        nested_data = {"level": 1}
        for i in range(1000):  # Create deeply nested dict
            nested_data = {"level": i + 2, "data": nested_data}
        
        with pytest.raises((ValueError, RecursionError)):
            self.detection_service.detect_anomalies(
                [nested_data],
                algorithm="isolation_forest"
            )
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        invalid_inputs = [
            None,
            {},
            set([1, 2, 3]),
            lambda x: x,
            open(__file__),
            complex(1, 2),
            object(),
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, TypeError)):
                self.detection_service.detect_anomalies(
                    invalid_input,
                    algorithm="isolation_forest"
                )
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and infinity values."""
        problematic_data = [
            [float('nan'), 1, 2],
            [float('inf'), 1, 2],
            [float('-inf'), 1, 2],
            [1e308, 1, 2],  # Very large number
            [-1e308, 1, 2],  # Very small number
        ]
        
        for data in problematic_data:
            with pytest.raises((ValueError, OverflowError)):
                self.detection_service.detect_anomalies(
                    [data],
                    algorithm="isolation_forest"
                )
    
    def test_algorithm_parameter_validation(self):
        """Test algorithm parameter validation."""
        invalid_algorithms = [
            "",
            None,
            123,
            [],
            {},
            "nonexistent_algorithm",
            "isolation_forest'; DROP TABLE models; --",
            "../../../etc/passwd",
        ]
        
        valid_data = [[1, 2, 3], [4, 5, 6]]
        
        for algorithm in invalid_algorithms:
            with pytest.raises((ValueError, KeyError, AttributeError)):
                self.detection_service.detect_anomalies(
                    valid_data,
                    algorithm=algorithm
                )
    
    def test_contamination_parameter_validation(self):
        """Test contamination parameter validation."""
        invalid_contaminations = [
            -0.1,  # Negative
            1.1,   # Greater than 1
            "0.1", # String
            None,
            [],
            {},
            float('inf'),
            float('nan'),
        ]
        
        valid_data = [[1, 2, 3], [4, 5, 6]]
        
        for contamination in invalid_contaminations:
            with pytest.raises((ValueError, TypeError)):
                self.detection_service.detect_anomalies(
                    valid_data,
                    algorithm="isolation_forest",
                    contamination=contamination
                )


@pytest.mark.security
class TestAPIInputValidation:
    """Test input validation at API layer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_request = Mock()
        self.mock_request.json = Mock()
    
    @patch('anomaly_detection.api.v1.detection.DetectionService')
    def test_api_malicious_json_payload(self, mock_service):
        """Test API protection against malicious JSON payloads."""
        malicious_payloads = [
            '{"data": "\\u0000\\u0001\\u0002"}',  # Null bytes
            '{"data": "' + 'A' * 1000000 + '"}',  # Oversized string
            '{"__proto__": {"isAdmin": true}}',  # Prototype pollution
            '{"constructor": {"prototype": {"isAdmin": true}}}',
        ]
        
        for payload in malicious_payloads:
            self.mock_request.json.return_value = json.loads(payload)
            
            with pytest.raises((ValueError, TypeError, json.JSONDecodeError)):
                detect_anomalies(self.mock_request)
    
    @patch('anomaly_detection.api.v1.detection.DetectionService')
    def test_api_missing_required_fields(self, mock_service):
        """Test API handling of missing required fields."""
        invalid_payloads = [
            {},  # Empty payload
            {"algorithm": "isolation_forest"},  # Missing data
            {"data": [[1, 2, 3]]},  # Missing algorithm
            {"data": None, "algorithm": "isolation_forest"},  # Null data
        ]
        
        for payload in invalid_payloads:
            self.mock_request.json.return_value = payload
            
            with pytest.raises((ValueError, KeyError)):
                detect_anomalies(self.mock_request)
    
    @patch('anomaly_detection.api.v1.detection.DetectionService')
    def test_api_data_size_limits(self, mock_service):
        """Test API data size limits."""
        # Test oversized data array
        oversized_data = [[1.0] * 1000 for _ in range(1000)]  # 1M elements
        
        self.mock_request.json.return_value = {
            "data": oversized_data,
            "algorithm": "isolation_forest"
        }
        
        with pytest.raises((ValueError, MemoryError)):
            detect_anomalies(self.mock_request)
    
    @patch('anomaly_detection.api.v1.models.ModelRepository')
    def test_model_name_validation(self, mock_repo):
        """Test model name validation for path traversal."""
        malicious_names = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "model; rm -rf /",
            "model`whoami`",
            "model$(whoami)",
            "model|cat /etc/passwd",
        ]
        
        for name in malicious_names:
            self.mock_request.json.return_value = {"model_name": name}
            
            with pytest.raises((ValueError, FileNotFoundError)):
                load_model(self.mock_request)


@pytest.mark.security
class TestCLIInputValidation:
    """Test CLI input validation."""
    
    def test_cli_file_path_validation(self):
        """Test CLI file path validation against path traversal."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "data.csv; rm -rf /",
            "data.csv`whoami`",
            "data.csv$(whoami)",
            "data.csv|cat /etc/passwd",
        ]
        
        for path in malicious_paths:
            with pytest.raises((ValueError, FileNotFoundError, PermissionError)):
                # Mock the CLI command execution
                with patch('builtins.open', side_effect=FileNotFoundError):
                    train_command(
                        data_file=path,
                        algorithm="isolation_forest",
                        output_path="model.pkl"
                    )
    
    def test_cli_parameter_injection(self):
        """Test CLI parameter injection protection."""
        injection_attempts = [
            "isolation_forest; rm -rf /",
            "isolation_forest && whoami",
            "isolation_forest | cat /etc/passwd",
            "isolation_forest`id`",
            "isolation_forest$(id)",
        ]
        
        for algorithm in injection_attempts:
            with pytest.raises((ValueError, KeyError)):
                train_command(
                    data_file="test.csv",
                    algorithm=algorithm,
                    output_path="model.pkl"
                )


@pytest.mark.security
class TestWebInputValidation:
    """Test web interface input validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_request = Mock()
        self.mock_request.form = {}
        self.mock_request.files = {}
    
    def test_file_upload_validation(self):
        """Test file upload validation."""
        # Test malicious filenames
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam", 
            "file.csv; rm -rf /",
            "file.exe",
            "file.bat",
            "file.sh",
            "file.php",
            ".htaccess",
        ]
        
        for filename in malicious_filenames:
            mock_file = Mock()
            mock_file.filename = filename
            mock_file.read.return_value = b"1,2,3\n4,5,6"
            
            self.mock_request.files = {"file": mock_file}
            
            with pytest.raises((ValueError, SecurityError)):
                upload_data(self.mock_request)
    
    def test_form_data_validation(self):
        """Test form data validation."""
        malicious_form_data = [
            {"algorithm": "<script>alert('xss')</script>"},
            {"contamination": "'; DROP TABLE models; --"},
            {"model_name": "../../../etc/passwd"},
            {"data": "javascript:alert('xss')"},
        ]
        
        for form_data in malicious_form_data:
            self.mock_request.form = form_data
            
            with pytest.raises((ValueError, SecurityError)):
                detection_endpoint(self.mock_request)
    
    def test_content_type_validation(self):
        """Test content type validation for uploaded files."""
        invalid_content_types = [
            "application/x-executable",
            "application/x-msdownload",
            "application/x-sh",
            "text/x-php",
            "text/html",
            "application/javascript",
        ]
        
        for content_type in invalid_content_types:
            mock_file = Mock()
            mock_file.filename = "data.csv"
            mock_file.content_type = content_type
            mock_file.read.return_value = b"malicious content"
            
            self.mock_request.files = {"file": mock_file}
            
            with pytest.raises((ValueError, SecurityError)):
                upload_data(self.mock_request)


@pytest.mark.security
class TestDataSanitization:
    """Test data sanitization and encoding."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
    
    def test_unicode_normalization(self):
        """Test Unicode normalization and validation."""
        # Test various Unicode attacks
        unicode_attacks = [
            "\u0000",  # Null byte
            "\u200B",  # Zero-width space
            "\u200C",  # Zero-width non-joiner
            "\u200D",  # Zero-width joiner
            "\u2028",  # Line separator
            "\u2029",  # Paragraph separator
            "\uFEFF",  # Byte order mark
            "test\u0008\u0008\u0008hack",  # Backspace characters
        ]
        
        for attack in unicode_attacks:
            with pytest.raises((ValueError, UnicodeError)):
                self.detection_service.detect_anomalies(
                    [[attack]],
                    algorithm="isolation_forest"
                )
    
    def test_encoding_validation(self):
        """Test various text encoding attacks."""
        encoding_attacks = [
            b"\x00\x01\x02\x03",  # Binary data
            "café".encode('utf-8').decode('latin-1'),  # Encoding confusion
            "test\x00hidden",  # Embedded nulls
            "\x7F\x80\x81",  # Control characters
        ]
        
        for attack in encoding_attacks:
            with pytest.raises((ValueError, UnicodeError, TypeError)):
                self.detection_service.detect_anomalies(
                    [[attack]],
                    algorithm="isolation_forest"
                )
    
    def test_html_entity_sanitization(self):
        """Test HTML entity sanitization."""
        html_entities = [
            "&lt;script&gt;alert('xss')&lt;/script&gt;",
            "&#60;script&#62;alert('xss')&#60;/script&#62;",
            "&amp;lt;script&amp;gt;",
            "&#x3C;script&#x3E;",
        ]
        
        for entity in html_entities:
            # Should not raise security exceptions, but should be sanitized
            try:
                result = self.detection_service.detect_anomalies(
                    [[entity]],
                    algorithm="isolation_forest"
                )
                # If it doesn't raise an exception, ensure the output is sanitized
                assert "<script>" not in str(result)
                assert "alert(" not in str(result)
            except (ValueError, TypeError):
                # This is also acceptable - rejecting the input entirely
                pass


@pytest.mark.security
class TestStreamingInputValidation:
    """Test streaming service input validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.streaming_service = StreamingService()
    
    def test_streaming_batch_size_limits(self):
        """Test streaming batch size limits."""
        # Test oversized batch
        oversized_batch = [[1.0] * 100 for _ in range(10000)]  # 1M elements
        
        with pytest.raises((ValueError, MemoryError)):
            self.streaming_service.process_streaming_batch(
                oversized_batch,
                algorithm="isolation_forest"
            )
    
    def test_streaming_buffer_size_validation(self):
        """Test streaming buffer size validation."""
        invalid_buffer_sizes = [
            -1,
            0,
            1.5,
            "100",
            None,
            float('inf'),
            float('nan'),
            1000000,  # Too large
        ]
        
        valid_batch = [[1, 2, 3], [4, 5, 6]]
        
        for buffer_size in invalid_buffer_sizes:
            with pytest.raises((ValueError, TypeError)):
                self.streaming_service.process_streaming_batch(
                    valid_batch,
                    algorithm="isolation_forest",
                    buffer_size=buffer_size
                )
    
    def test_concept_drift_threshold_validation(self):
        """Test concept drift threshold validation."""
        invalid_thresholds = [
            -0.1,
            1.1,
            "0.05",
            None,
            [],
            {},
            float('inf'),
            float('nan'),
        ]
        
        valid_batch = [[1, 2, 3], [4, 5, 6]]
        
        for threshold in invalid_thresholds:
            with pytest.raises((ValueError, TypeError)):
                self.streaming_service.detect_concept_drift(
                    valid_batch,
                    threshold=threshold
                )


@pytest.mark.security
class TestEnsembleInputValidation:
    """Test ensemble service input validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ensemble_service = EnsembleService()
    
    def test_algorithm_list_validation(self):
        """Test ensemble algorithm list validation."""
        invalid_algorithm_lists = [
            [],  # Empty list
            None,
            "isolation_forest",  # String instead of list
            123,
            [None],
            [""],
            ["nonexistent_algorithm"],
            ["isolation_forest", "nonexistent"],
            ["isolation_forest; DROP TABLE models"],
        ]
        
        valid_data = [[1, 2, 3], [4, 5, 6]]
        
        for algorithms in invalid_algorithm_lists:
            with pytest.raises((ValueError, TypeError, KeyError)):
                self.ensemble_service.detect_with_ensemble(
                    valid_data,
                    algorithms=algorithms,
                    ensemble_method="majority"
                )
    
    def test_ensemble_method_validation(self):
        """Test ensemble method validation."""
        invalid_methods = [
            "",
            None,
            123,
            [],
            {},
            "nonexistent_method",
            "majority; DROP TABLE models",
            "../../../etc/passwd",
        ]
        
        valid_data = [[1, 2, 3], [4, 5, 6]]
        valid_algorithms = ["isolation_forest", "one_class_svm"]
        
        for method in invalid_methods:
            with pytest.raises((ValueError, KeyError)):
                self.ensemble_service.detect_with_ensemble(
                    valid_data,
                    algorithms=valid_algorithms,
                    ensemble_method=method
                )


if __name__ == "__main__":
    # Run specific security tests
    pytest.main([
        __file__ + "::TestInputValidation::test_malicious_data_injection",
        "-v", "-s", "--tb=short"
    ])
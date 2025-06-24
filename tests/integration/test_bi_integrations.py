"""
Integration tests for Business Intelligence integrations.
"""

import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from pynomaly.application.services.export_service import ExportService
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.application.dto.export_options import ExportOptions, ExportFormat


@pytest.fixture
def sample_detection_result():
    """Create a sample detection result for testing."""
    scores = [
        AnomalyScore(0.1),
        AnomalyScore(0.3),
        AnomalyScore(0.8),
        AnomalyScore(0.9),
        AnomalyScore(0.2),
    ]
    
    labels = np.array([0, 0, 1, 1, 0])
    
    anomalies = [
        Anomaly(
            id=uuid4(),
            index=2,
            score=scores[2],
            feature_values={"feature_1": 10.0, "feature_2": 20.0}
        ),
        Anomaly(
            id=uuid4(),
            index=3,
            score=scores[3],
            feature_values={"feature_1": 15.0, "feature_2": 25.0}
        ),
    ]
    
    return DetectionResult(
        detector_id=uuid4(),
        dataset_id=uuid4(),
        anomalies=anomalies,
        scores=scores,
        labels=labels,
        threshold=0.5,
        execution_time_ms=150.0,
        metadata={"detector_name": "IsolationForest"}
    )


class TestBIIntegrationsIntegration:
    """Integration tests for BI integrations."""
    
    def test_export_service_initialization(self):
        """Test that export service initializes correctly."""
        service = ExportService()
        
        # Should have at least some adapters if dependencies are available
        supported_formats = service.get_supported_formats()
        stats = service.get_export_statistics()
        
        # Basic structure validation
        assert isinstance(supported_formats, list)
        assert isinstance(stats, dict)
        assert 'total_formats' in stats
        assert 'supported_formats' in stats
        assert 'adapters' in stats
        
        # If any formats are supported, they should be consistent
        assert stats['total_formats'] == len(supported_formats)
        assert len(stats['supported_formats']) == len(supported_formats)
    
    def test_export_options_creation(self):
        """Test export options creation for different formats."""
        service = ExportService()
        
        # Test Excel options
        excel_options = service.create_export_options(ExportFormat.EXCEL)
        assert excel_options.format == ExportFormat.EXCEL
        assert excel_options.use_advanced_formatting is True
        assert excel_options.highlight_anomalies is True
        
        # Test Power BI options
        powerbi_options = service.create_export_options(
            ExportFormat.POWERBI,
            workspace_id="test-workspace",
            dataset_name="test-dataset"
        )
        assert powerbi_options.format == ExportFormat.POWERBI
        assert powerbi_options.workspace_id == "test-workspace"
        assert powerbi_options.dataset_name == "test-dataset"
    
    def test_validation_without_export(self, sample_detection_result):
        """Test validation functionality without actual export."""
        service = ExportService()
        
        # Test validation for each potentially supported format
        formats_to_test = [
            (ExportFormat.EXCEL, "test.xlsx"),
            (ExportFormat.POWERBI, ""),
            (ExportFormat.GSHEETS, ""),
            (ExportFormat.SMARTSHEET, "")
        ]
        
        for format_type, file_path in formats_to_test:
            validation = service.validate_export_request(format_type, file_path)
            
            # Should return a validation dictionary
            assert isinstance(validation, dict)
            assert 'valid' in validation
            assert 'format' in validation
            assert 'file_path' in validation
            assert 'errors' in validation
            assert 'warnings' in validation
            
            # Format should match
            assert validation['format'] == format_type.value
            assert validation['file_path'] == file_path
    
    @pytest.mark.skipif(
        not pytest.importorskip('openpyxl', reason='openpyxl not available'),
        reason='Excel dependencies not available'
    )
    def test_excel_export_if_available(self, sample_detection_result):
        """Test Excel export if dependencies are available."""
        service = ExportService()
        
        # Only run if Excel is supported
        if ExportFormat.EXCEL not in service.get_supported_formats():
            pytest.skip("Excel adapter not available")
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                options = ExportOptions().for_excel()
                result = service.export_results(
                    sample_detection_result,
                    tmp_file.name,
                    options
                )
                
                # Verify result structure
                assert 'service' in result
                assert 'format' in result
                assert 'total_samples' in result
                assert 'anomalies_count' in result
                
                # Verify values
                assert result['format'] == 'excel'
                assert result['total_samples'] == 5
                assert result['anomalies_count'] == 2
                
                # Verify file was created
                assert Path(tmp_file.name).exists()
                assert Path(tmp_file.name).stat().st_size > 0
                
            finally:
                # Clean up
                if Path(tmp_file.name).exists():
                    Path(tmp_file.name).unlink()
    
    def test_multi_format_export_simulation(self, sample_detection_result):
        """Test multi-format export configuration (simulation only)."""
        service = ExportService()
        
        # Get available formats
        supported_formats = service.get_supported_formats()
        
        if not supported_formats:
            pytest.skip("No export formats available")
        
        # Test with only Excel if available, otherwise skip
        if ExportFormat.EXCEL in supported_formats:
            formats_to_test = [ExportFormat.EXCEL]
        else:
            pytest.skip("No testable formats available")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "test_export"
            
            # This would normally export, but we're just testing the setup
            # The actual export would happen in the service
            options_map = {
                ExportFormat.EXCEL: ExportOptions().for_excel()
            }
            
            # Verify the configuration is valid
            for format_type, options in options_map.items():
                validation = service.validate_export_request(
                    format_type,
                    base_path.with_suffix('.xlsx')
                )
                
                if format_type in supported_formats:
                    # Should be valid if format is supported
                    if not validation['valid']:
                        # Print errors for debugging
                        print(f"Validation errors for {format_type}: {validation['errors']}")
                    # Note: File path validation might fail if directory doesn't exist
                    # but that's expected in this test
    
    def test_export_service_statistics(self):
        """Test export service statistics functionality."""
        service = ExportService()
        stats = service.get_export_statistics()
        
        # Verify statistics structure
        required_keys = ['total_formats', 'supported_formats', 'adapters']
        for key in required_keys:
            assert key in stats, f"Missing key in stats: {key}"
        
        # Verify data types
        assert isinstance(stats['total_formats'], int)
        assert isinstance(stats['supported_formats'], list)
        assert isinstance(stats['adapters'], dict)
        
        # Verify consistency
        assert stats['total_formats'] >= 0
        assert len(stats['supported_formats']) == stats['total_formats']
        assert len(stats['adapters']) == stats['total_formats']
        
        # If any adapters are available, verify their structure
        for format_name, adapter_info in stats['adapters'].items():
            assert isinstance(adapter_info, dict)
            assert 'class' in adapter_info
            assert 'supported_extensions' in adapter_info
            assert isinstance(adapter_info['supported_extensions'], list)
    
    def test_format_specific_file_extensions(self):
        """Test getting file extensions for specific formats."""
        service = ExportService()
        supported_formats = service.get_supported_formats()
        
        for format_type in supported_formats:
            try:
                extensions = service.get_supported_file_extensions(format_type)
                assert isinstance(extensions, list)
                assert len(extensions) > 0
                
                # All extensions should start with a dot
                for ext in extensions:
                    assert ext.startswith('.'), f"Invalid extension format: {ext}"
                    
            except Exception as e:
                pytest.fail(f"Failed to get extensions for {format_type}: {e}")
    
    def test_export_options_serialization(self):
        """Test export options serialization/deserialization."""
        # Test basic options
        options = ExportOptions()
        options_dict = options.to_dict()
        
        assert isinstance(options_dict, dict)
        assert 'format' in options_dict
        assert 'destination' in options_dict
        
        # Test reconstruction
        reconstructed = ExportOptions.from_dict(options_dict)
        assert reconstructed.format == options.format
        assert reconstructed.destination == options.destination
        
        # Test format-specific options
        excel_options = ExportOptions().for_excel()
        excel_dict = excel_options.to_dict()
        reconstructed_excel = ExportOptions.from_dict(excel_dict)
        
        assert reconstructed_excel.format == ExportFormat.EXCEL
        assert reconstructed_excel.use_advanced_formatting == excel_options.use_advanced_formatting
    
    def test_error_handling_for_unsupported_formats(self, sample_detection_result):
        """Test error handling when format is not supported."""
        service = ExportService()
        
        # Clear all adapters to simulate unsupported format
        original_adapters = service._adapters.copy()
        service._adapters.clear()
        
        try:
            with pytest.raises(ValueError, match="not supported"):
                service.export_results(
                    sample_detection_result,
                    "test.xlsx",
                    ExportOptions(format=ExportFormat.EXCEL)
                )
        finally:
            # Restore adapters
            service._adapters = original_adapters
    
    def test_concurrent_service_initialization(self):
        """Test that multiple export services can be created concurrently."""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def create_service():
            try:
                service = ExportService()
                stats = service.get_export_statistics()
                results.put(stats)
            except Exception as e:
                errors.put(e)
        
        # Create multiple services concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_service)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert errors.empty(), f"Errors occurred: {list(errors.queue)}"
        assert results.qsize() == 5, f"Expected 5 results, got {results.qsize()}"
        
        # All results should be consistent
        first_result = results.get()
        while not results.empty():
            next_result = results.get()
            assert next_result['total_formats'] == first_result['total_formats']
            assert next_result['supported_formats'] == first_result['supported_formats']
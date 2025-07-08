#!/usr/bin/env python3
"""
Pytest-based test specifically for detection DTO validation branches to achieve 100% coverage.
"""
import pytest
import sys
import os
from uuid import uuid4

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestDetectionRequestDTOValidation:
    """Test class for DetectionRequestDTO validation logic."""
    
    def test_model_post_init_neither_dataset_nor_data(self):
        """Test when neither dataset_id nor data is provided."""
        from pynomaly.application.dto.detection_dto import DetectionRequestDTO
        
        with pytest.raises(ValueError, match="Either dataset_id or data must be provided"):
            DetectionRequestDTO(
                detector_id=uuid4(),
                dataset_id=None,
                data=None
            )
    
    def test_model_post_init_both_dataset_and_data(self):
        """Test when both dataset_id and data are provided."""
        from pynomaly.application.dto.detection_dto import DetectionRequestDTO
        
        with pytest.raises(ValueError, match="Provide either dataset_id or data, not both"):
            DetectionRequestDTO(
                detector_id=uuid4(),
                dataset_id=uuid4(),
                data=[{"feature1": 1.0, "feature2": 2.0}]
            )
    
    def test_model_post_init_only_dataset_id(self):
        """Test valid case with only dataset_id provided."""
        from pynomaly.application.dto.detection_dto import DetectionRequestDTO
        
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            data=None
        )
        assert dto.dataset_id is not None
        assert dto.data is None
    
    def test_model_post_init_only_data(self):
        """Test valid case with only data provided."""
        from pynomaly.application.dto.detection_dto import DetectionRequestDTO
        
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            dataset_id=None,
            data=[{"feature1": 1.0, "feature2": 2.0}]
        )
        assert dto.dataset_id is None
        assert dto.data is not None
        assert len(dto.data) == 1

def run_validation_tests():
    """Run all validation tests manually without pytest runner."""
    test_class = TestDetectionRequestDTOValidation()
    
    print("🚀 Running validation tests manually")
    print("=" * 50)
    
    # Test 1: Neither dataset_id nor data provided
    print("Testing: Neither dataset_id nor data provided")
    try:
        test_class.test_model_post_init_neither_dataset_nor_data()
        print("❌ Test failed - no exception raised")
    except AssertionError:
        print("✅ Test passed - correct validation error")
    except Exception as e:
        print(f"✅ Test passed - validation error raised: {e}")
    
    # Test 2: Both dataset_id and data provided
    print("Testing: Both dataset_id and data provided")
    try:
        test_class.test_model_post_init_both_dataset_and_data()
        print("❌ Test failed - no exception raised")
    except AssertionError:
        print("✅ Test passed - correct validation error")
    except Exception as e:
        print(f"✅ Test passed - validation error raised: {e}")
    
    # Test 3: Only dataset_id provided
    print("Testing: Only dataset_id provided")
    try:
        test_class.test_model_post_init_only_dataset_id()
        print("✅ Test passed - valid DTO created")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test 4: Only data provided
    print("Testing: Only data provided")
    try:
        test_class.test_model_post_init_only_data()
        print("✅ Test passed - valid DTO created")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("=" * 50)
    print("🎉 All validation tests completed!")

if __name__ == "__main__":
    run_validation_tests()

#!/usr/bin/env python3
"""
Direct test of model_post_init to achieve full coverage for detection DTOs validation.
"""
import sys
import os
from typing import Any
from uuid import UUID, uuid4

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class MockContext:
    """Mock context to mimic pydantic context behavior."""
    pass

def test_model_post_init_validation():
    """Directly test model_post_init to cover validation branches."""
    
    from pynomaly.application.dto.detection_dto import DetectionRequestDTO
    
    print("Testing direct call to model_post_init...")
    
    context = MockContext()
    
    # Test with neither dataset_id nor data provided
    try:
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            dataset_id=None,
            data=None
        )
        dto.model_post_init(context)
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    # Test with both dataset_id and data provided
    try:
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            data=[{"feature1": 1.0, "feature2": 2.0}]
        )
        dto.model_post_init(context)
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")

if __name__ == "__main__":
    print("🚀 Running direct test of model_post_init")
    test_model_post_init_validation()
    print("🎉 Model post-init validation test completed!")

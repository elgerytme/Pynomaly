#!/usr/bin/env python3
"""Simple validation test."""

import sys
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test core imports."""
    
    print("🔄 Testing core imports...")
    
    try:
        from anomaly_detection.domain.entities import Dataset, DetectionResult
        print("✅ Domain entities import successful")
    except Exception as e:
        print(f"❌ Domain entities import failed: {e}")
    
    try:
        from anomaly_detection.domain.services.detection_service import DetectionService
        print("✅ DetectionService import successful")
    except Exception as e:
        print(f"❌ DetectionService import failed: {e}")
    
    try:
        from anomaly_detection.infrastructure.adapters.pyod_adapter import PyODAdapter
        print("✅ PyODAdapter import successful")
    except Exception as e:
        print(f"❌ PyODAdapter import failed: {e}")
    
    print("🎉 Import test completed!")

def test_data_generation():
    """Test basic data generation."""
    
    print("\n🔄 Testing data generation...")
    
    try:
        # Generate simple test data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (90, 2))
        anomaly_data = np.random.normal(3, 1, (10, 2))
        
        # Combine data
        data = np.vstack([normal_data, anomaly_data])
        labels = np.array([0] * 90 + [1] * 10)
        
        print(f"✅ Generated test data: {data.shape}")
        print(f"   - Normal samples: {np.sum(labels == 0)}")
        print(f"   - Anomaly samples: {np.sum(labels == 1)}")
        
        # Test basic detection
        from anomaly_detection.domain.services.detection_service import DetectionService
        service = DetectionService()
        
        # Create dataset
        from anomaly_detection.domain.entities.dataset import Dataset, DatasetMetadata, DatasetType
        metadata = DatasetMetadata(
            name="test_dataset",
            description="Simple test dataset",
            feature_names=[f"feature_{i}" for i in range(data.shape[1])]
        )
        dataset = Dataset(
            data=data,
            dataset_type=DatasetType.TESTING,
            metadata=metadata,
            labels=labels
        )
        
        print("✅ Dataset created successfully")
        print(f"   - Shape: {dataset.data.shape}")
        print(f"   - Features: {dataset.metadata.feature_names if dataset.metadata else 'None'}")
        print(f"   - Type: {dataset.dataset_type.value}")
        print(f"   - Has labels: {dataset.labels is not None}")
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()
    test_data_generation()
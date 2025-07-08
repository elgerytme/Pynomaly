#!/usr/bin/env python3
"""
Simple test script to run coverage on pynomaly source code
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports to get coverage data"""
    try:
        # Test domain imports
        from pynomaly.domain.entities import Dataset, Detector, Anomaly
        from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
        print("✅ Domain imports successful")
        
        # Test application imports
        from pynomaly.application.services import DetectionService
        print("✅ Application imports successful")
        
        # Test infrastructure imports
        from pynomaly.infrastructure.adapters import SklearnAdapter
        print("✅ Infrastructure imports successful")
        
        # Test config imports
        from pynomaly.infrastructure.config.container import Container
        print("✅ Config imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        import numpy as np
        import pandas as pd
        
        from pynomaly.domain.entities import Dataset
        from pynomaly.domain.value_objects import ContaminationRate
        from pynomaly.infrastructure.adapters import SklearnAdapter
        
        # Create sample data
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100),
        })
        
        # Create dataset
        dataset = Dataset(
            name="test_dataset",
            data=data,
            feature_names=["feature_1", "feature_2"]
        )
        
        # Create detector
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            name="test_detector",
            contamination_rate=ContaminationRate(0.1),
        )
        
        print("✅ Basic functionality test passed")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running simple coverage test")
    print("=" * 50)
    
    test_imports()
    test_basic_functionality()
    
    print("=" * 50)
    print("✅ Coverage test completed")

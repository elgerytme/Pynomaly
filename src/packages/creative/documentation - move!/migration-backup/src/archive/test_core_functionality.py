#!/usr/bin/env python3
"""
Test core functionality of Pynomaly package
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

def test_basic_imports():
    """Test that core modules can be imported"""
    print("Testing basic imports...")
    
    # Test core imports
    import monorepo
    from monorepo.domain.entities import Dataset
    from monorepo.domain.value_objects import ContaminationRate
    from monorepo.infrastructure.adapters.sklearn_adapter import SklearnAdapter
    
    print("✓ All basic imports successful")
    
def test_dataset_creation():
    """Test Dataset creation"""
    print("Testing Dataset creation...")
    
    from monorepo.domain.entities import Dataset
    
    # Create sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10]
    })
    
    # Create dataset
    dataset = Dataset(name="Test Dataset", data=data)
    
    assert dataset.name == "Test Dataset"
    assert len(dataset.data) == 5
    assert dataset.data.shape == (5, 2)
    
    print("✓ Dataset creation successful")

def test_contamination_rate():
    """Test ContaminationRate value object"""
    print("Testing ContaminationRate...")
    
    from monorepo.domain.value_objects import ContaminationRate
    
    # Create contamination rate
    rate = ContaminationRate(0.1)
    
    assert rate.value == 0.1
    assert 0.0 <= rate.value <= 1.0
    
    print("✓ ContaminationRate creation successful")

def test_sklearn_adapter():
    """Test SklearnAdapter"""
    print("Testing SklearnAdapter...")
    
    from monorepo.infrastructure.adapters.sklearn_adapter import SklearnAdapter
    from monorepo.domain.entities import Dataset
    from monorepo.domain.value_objects import ContaminationRate
    
    # Create sample data with some outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 2))
    outliers = np.random.uniform(-4, 4, (10, 2))
    data = np.vstack([normal_data, outliers])
    
    df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    dataset = Dataset(name="Test Data", data=df)
    
    # Create adapter
    adapter = SklearnAdapter(
        algorithm_name="IsolationForest",
        name="Test Detector",
        contamination_rate=ContaminationRate(0.1),
        random_state=42
    )
    
    # Test fit
    adapter.fit(dataset)
    
    # Test predict
    result = adapter.detect(dataset)
    
    assert result is not None
    assert hasattr(result, 'anomalies')
    assert hasattr(result, 'scores')
    assert len(result.scores) == len(df)
    
    print("✓ SklearnAdapter functionality successful")

def test_pyod_integration():
    """Test PyOD integration"""
    print("Testing PyOD integration...")
    
    from monorepo.infrastructure.adapters.pyod_adapter import PyODAdapter
    from monorepo.domain.entities import Dataset
    from monorepo.domain.value_objects import ContaminationRate
    
    # Create sample data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 2))
    outliers = np.random.uniform(-4, 4, (10, 2))
    data = np.vstack([normal_data, outliers])
    
    df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    dataset = Dataset(name="Test Data", data=df)
    
    # Create adapter
    adapter = PyODAdapter(
        algorithm_name="IsolationForest",
        name="PyOD Test Detector",
        contamination_rate=ContaminationRate(0.1),
        random_state=42
    )
    
    # Test fit
    adapter.fit(dataset)
    
    # Test predict
    result = adapter.detect(dataset)
    
    assert result is not None
    assert hasattr(result, 'anomalies')
    assert hasattr(result, 'scores')
    assert len(result.scores) == len(df)
    
    print("✓ PyOD integration successful")

def main():
    """Run all tests"""
    print("🔍 Testing Pynomaly Core Functionality\n")
    
    try:
        test_basic_imports()
        test_dataset_creation()
        test_contamination_rate()
        test_sklearn_adapter()
        test_pyod_integration()
        
        print("\n✅ All core functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
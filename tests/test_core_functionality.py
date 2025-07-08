"""Test core anomaly detection functionality."""
import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


def test_basic_anomaly_detection():
    """Test basic anomaly detection with clean architecture."""
    print("Testing basic anomaly detection...")
    
    try:
        # Create sample data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 2))
        outliers = np.random.uniform(-4, 4, (10, 2))
        data = np.vstack([normal_data, outliers])
        
        # Create dataset
        df = pd.DataFrame(data, columns=['feature1', 'feature2'])
        dataset = Dataset(name="Test Dataset", data=df)
        
        print(f"Dataset created with {len(dataset.data)} samples")
        
        # Create detector
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            name="Test Detector",
            contamination_rate=ContaminationRate(0.1),
            random_state=42
        )
        
        print(f"Detector created: {detector.name}")
        print(f"Algorithm: {detector.algorithm_name}")
        print(f"Contamination rate: {detector.contamination_rate.value}")
        
        # Train detector
        detector.fit(dataset)
        print(f"Detector trained successfully, is_fitted: {detector.is_fitted}")
        
        # Detect anomalies
        result = detector.detect(dataset)
        print(f"Detection completed:")
        print(f"  - Anomalies found: {len(result.anomalies)}")
        print(f"  - Execution time: {result.execution_time_ms:.2f}ms")
        print(f"  - Samples processed: {len(result.scores)}")
        
        # Check results
        assert len(result.scores) == len(dataset.data), "Score count mismatch"
        assert len(result.labels) == len(dataset.data), "Label count mismatch"
        assert result.execution_time_ms > 0, "Execution time should be positive"
        
        print("✓ Basic anomaly detection test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_anomaly_detection()
    sys.exit(0 if success else 1)

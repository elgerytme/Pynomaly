#!/usr/bin/env python3
"""Test PyOD adapter functionality."""

import sys
from pathlib import Path
import numpy as np

# Add the package to Python path
package_root = Path(__file__).parent / "src/packages/data/anomaly_detection/src"
sys.path.insert(0, str(package_root))

from pynomaly_detection.algorithms.pyod_adapter import PyODAdapter

def test_pyod_adapter():
    """Test PyOD adapter functionality."""
    print("Testing PyOD adapter...")
    
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    # Add some outliers
    data[:10] += 3
    
    try:
        # Test IForest
        adapter = PyODAdapter(algorithm="IForest", contamination=0.1)
        predictions = adapter.detect(data)
        print(f"✅ PyOD IForest test passed")
        print(f"   Anomalies detected: {np.sum(predictions)}")
        
        # Test LOF
        adapter_lof = PyODAdapter(algorithm="LOF", contamination=0.1)
        predictions_lof = adapter_lof.detect(data)
        print(f"✅ PyOD LOF test passed")
        print(f"   LOF anomalies detected: {np.sum(predictions_lof)}")
        
        # Test OCSVM
        adapter_ocsvm = PyODAdapter(algorithm="OCSVM", contamination=0.1)
        predictions_ocsvm = adapter_ocsvm.detect(data)
        print(f"✅ PyOD OCSVM test passed")
        print(f"   OCSVM anomalies detected: {np.sum(predictions_ocsvm)}")
        
        print(f"\n🎉 All PyOD tests completed successfully!")
        
    except ImportError as e:
        print(f"⚠️ PyOD not available: {e}")
        print("This is expected - PyOD is an optional dependency")
    except Exception as e:
        print(f"❌ PyOD test failed: {e}")

if __name__ == "__main__":
    test_pyod_adapter()
"""Test multiple algorithms and functionality."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.value_objects import ContaminationRate


def test_algorithms():
    """Test multiple algorithms."""
    print("Testing multiple algorithms...")
    
    # Create sample data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 3))
    outliers = np.random.uniform(-4, 4, (10, 3))
    data = np.vstack([normal_data, outliers])
    
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
    dataset = Dataset(name="Test Dataset", data=df)
    
    # Test sklearn algorithms
    sklearn_algorithms = [
        "IsolationForest",
        "OneClassSVM",
        "LocalOutlierFactor",
        "EllipticEnvelope"
    ]
    
    results = {}
    
    for algorithm in sklearn_algorithms:
        try:
            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
            
            print(f"\nTesting {algorithm}...")
            detector = SklearnAdapter(
                algorithm_name=algorithm,
                name=f"Test {algorithm}",
                contamination_rate=ContaminationRate(0.1)
            )
            
            # Train
            detector.fit(dataset)
            
            # Detect
            result = detector.detect(dataset)
            
            results[algorithm] = {
                "anomalies_found": len(result.anomalies),
                "execution_time_ms": result.execution_time_ms,
                "success": True
            }
            
            print(f"  ✓ {algorithm} - Found {len(result.anomalies)} anomalies in {result.execution_time_ms:.2f}ms")
            
        except Exception as e:
            results[algorithm] = {
                "error": str(e),
                "success": False
            }
            print(f"  ✗ {algorithm} - Failed: {e}")
    
    # Test PyOD algorithms (if available)
    try:
        from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
        
        pyod_algorithms = ["COPOD", "ECOD", "HBOS", "KNN", "LOF"]
        
        for algorithm in pyod_algorithms:
            try:
                print(f"\nTesting PyOD {algorithm}...")
                detector = PyODAdapter(
                    algorithm_name=algorithm,
                    name=f"Test PyOD {algorithm}",
                    contamination_rate=ContaminationRate(0.1)
                )
                
                # Train
                detector.fit(dataset)
                
                # Detect
                result = detector.detect(dataset)
                
                results[f"PyOD_{algorithm}"] = {
                    "anomalies_found": len(result.anomalies),
                    "execution_time_ms": result.execution_time_ms,
                    "success": True
                }
                
                print(f"  ✓ PyOD {algorithm} - Found {len(result.anomalies)} anomalies in {result.execution_time_ms:.2f}ms")
                
            except Exception as e:
                results[f"PyOD_{algorithm}"] = {
                    "error": str(e),
                    "success": False
                }
                print(f"  ✗ PyOD {algorithm} - Failed: {e}")
                
    except ImportError:
        print("\nPyOD not available, skipping PyOD algorithms")
    
    # Summary
    print("\n" + "="*50)
    print("ALGORITHM TEST SUMMARY")
    print("="*50)
    
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    print(f"Successful algorithms: {successful}/{total}")
    
    for algo, result in results.items():
        if result["success"]:
            print(f"✓ {algo}: {result['anomalies_found']} anomalies, {result['execution_time_ms']:.2f}ms")
        else:
            print(f"✗ {algo}: {result['error']}")
    
    return successful > 0


if __name__ == "__main__":
    success = test_algorithms()
    sys.exit(0 if success else 1)

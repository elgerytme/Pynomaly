import sys
sys.path.append('src')

import pandas as pd
import numpy as np

# Test basic imports
try:
    from pynomaly.domain.entities import Dataset
    print("✅ Successfully imported Dataset")
except ImportError as e:
    print(f"❌ Failed to import Dataset: {e}")
    sys.exit(1)

try:
    from pynomaly.domain.value_objects import ContaminationRate
    print("✅ Successfully imported ContaminationRate")
except ImportError as e:
    print(f"❌ Failed to import ContaminationRate: {e}")
    sys.exit(1)

try:
    from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
    print("✅ Successfully imported SklearnAdapter")
except ImportError as e:
    print(f"❌ Failed to import SklearnAdapter: {e}")
    sys.exit(1)

# Test basic functionality
def test_basic_example():
    print("\n🧪 Testing basic anomaly detection...")
    
    # Create sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 2))
    outliers = np.random.uniform(-4, 4, (10, 2))
    data = np.vstack([normal_data, outliers])
    
    # Create dataset
    df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    dataset = Dataset(name="Sample Data", data=df)
    print(f"✅ Created dataset with {len(df)} samples")
    
    # Create detector using Pynomaly's clean architecture
    detector = SklearnAdapter(
        algorithm_name="IsolationForest",
        name="Basic Detector",
        contamination_rate=ContaminationRate(0.1),
        random_state=42,
        n_estimators=100
    )
    print("✅ Created IsolationForest detector")
    
    # Train detector
    detector.fit(dataset)
    print("✅ Trained detector")
    
    # Detect anomalies
    result = detector.detect(dataset)
    print("✅ Detected anomalies")
    
    # Results
    anomaly_count = len(result.anomalies)
    scores = [score.value for score in result.scores]
    print(f"✅ Detected {anomaly_count} anomalies out of {len(data)} samples")
    print(f"✅ Anomaly scores range: {min(scores):.3f} to {max(scores):.3f}")
    print(f"✅ Detection completed in {result.execution_time_ms:.2f}ms")
    
    return result.labels, scores

if __name__ == "__main__":
    try:
        predictions, scores = test_basic_example()
        print("\n🎉 Example completed successfully!")
        print(f"📊 Found {sum(predictions)} anomalies out of {len(predictions)} samples")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

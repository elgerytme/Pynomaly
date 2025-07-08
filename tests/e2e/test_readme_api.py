#!/usr/bin/env python3
"""Test the Python API example from README.md"""

import sys

sys.path.insert(0, "/mnt/c/Users/andre/Pynomaly/src")

try:
    import numpy as np
    import pandas as pd

    from pynomaly.domain.entities import Dataset
    from pynomaly.domain.value_objects import ContaminationRate
    from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

    # Basic usage example with Pynomaly's SklearnAdapter
    def basic_example():
        print("Testing basic anomaly detection example...")

        # Create sample data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 2))
        outliers = np.random.uniform(-4, 4, (10, 2))
        data = np.vstack([normal_data, outliers])

        # Create dataset
        df = pd.DataFrame(data, columns=["feature1", "feature2"])
        dataset = Dataset(name="Sample Data", data=df)

        # Create detector using Pynomaly's clean architecture
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            name="Basic Detector",
            contamination_rate=ContaminationRate(0.1),
            random_state=42,
            n_estimators=100,
        )

        # Train detector
        detector.fit(dataset)

        # Detect anomalies
        result = detector.detect(dataset)

        # Results
        anomaly_count = len(result.anomalies)
        scores = [score.value for score in result.scores]
        print(f"Detected {anomaly_count} anomalies out of {len(data)} samples")
        print(f"Anomaly scores range: {min(scores):.3f} to {max(scores):.3f}")
        print(f"Detection completed in {result.execution_time_ms:.2f}ms")

        return result.labels, scores

    # Run example
    if __name__ == "__main__":
        predictions, scores = basic_example()
        print("Example completed successfully!")
        print("✅ Python API example works correctly!")

except Exception as e:
    print(f"❌ Error testing Python API example: {e}")
    import traceback

    traceback.print_exc()

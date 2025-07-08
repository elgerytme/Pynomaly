#!/usr/bin/env python3
"""Test script to demonstrate Pynomaly functionality."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    import pandas as pd
    print("✓ NumPy and Pandas available")
    
    # Import core Pynomaly components
    from pynomaly.domain.entities import Dataset
    from pynomaly.domain.value_objects import ContaminationRate
    from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
    
    print("✓ Core Pynomaly modules imported successfully")
    
    # Create sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 2))
    outliers = np.random.uniform(-4, 4, (10, 2))
    data = np.vstack([normal_data, outliers])
    
    # Create dataset
    df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    dataset = Dataset(name="Sample Data", data=df)
    
    print("✓ Dataset created successfully")
    
    # Create detector using Pynomaly's clean architecture
    detector = SklearnAdapter(
        algorithm_name="IsolationForest",
        name="Basic Detector",
        contamination_rate=ContaminationRate(0.1),
        random_state=42,
        n_estimators=100
    )
    
    print("✓ Detector created successfully")
    
    # Train detector
    detector.fit(dataset)
    print("✓ Detector trained successfully")
    
    # Detect anomalies
    result = detector.detect(dataset)
    print("✓ Anomaly detection completed")
    
    # Results
    anomaly_count = len(result.anomalies)
    scores = [score.value for score in result.scores]
    
    print(f"\n🎯 Results:")
    print(f"   Detected {anomaly_count} anomalies out of {len(data)} samples")
    print(f"   Anomaly scores range: {min(scores):.3f} to {max(scores):.3f}")
    print(f"   Detection completed in {result.execution_time_ms:.2f}ms")
    
    print("\n✅ Pynomaly test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

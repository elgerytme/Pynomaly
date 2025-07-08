#!/usr/bin/env python3
"""
Demo test file to showcase the Hatch testing infrastructure
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
def test_basic_import():
    """Unit test - basic import functionality"""
    import sys

    assert sys.version_info >= (3, 11)


@pytest.mark.unit
def test_numpy_pandas():
    """Unit test - numpy and pandas functionality"""
    # Test numpy
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0

    # Test pandas
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    assert len(df) == 3
    assert df["A"].sum() == 6


@pytest.mark.integration
def test_data_pipeline():
    """Integration test - basic data pipeline"""
    # Create sample data
    data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.normal(0, 1, 100),
        }
    )

    # Add some anomalies
    data.iloc[-5:] = 5  # Last 5 points are anomalies

    # Basic anomaly detection simulation
    z_scores = np.abs((data - data.mean()) / data.std())
    anomalies = (z_scores > 2).any(axis=1)

    # Should detect the anomalies we added
    assert anomalies.sum() >= 5


@pytest.mark.slow
def test_performance_simulation():
    """Slow test - performance simulation"""
    import time

    # Simulate slow computation
    start_time = time.time()

    # Generate large dataset
    large_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 10000),
            "feature_2": np.random.normal(0, 1, 10000),
        }
    )

    # Simulate processing
    result = large_data.describe()

    end_time = time.time()
    processing_time = end_time - start_time

    # Should complete within reasonable time
    assert processing_time < 5.0  # 5 seconds
    assert len(result) > 0


@pytest.mark.unit
def test_coverage_demo():
    """Unit test to demonstrate coverage"""

    def simple_function(x):
        if x > 0:
            return x * 2
        elif x < 0:
            return x * -1
        else:
            return 0

    # Test all branches
    assert simple_function(5) == 10
    assert simple_function(-3) == 3
    assert simple_function(0) == 0


@pytest.mark.integration
@pytest.mark.slow
def test_integration_slow():
    """Integration test marked as slow"""
    import time

    # Create multiple datasets
    datasets = []
    for i in range(3):
        data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 1000),
                "feature_2": np.random.normal(0, 1, 1000),
            }
        )
        datasets.append(data)

    # Simulate slow processing
    time.sleep(0.1)  # Small delay

    # Combine datasets
    combined = pd.concat(datasets, ignore_index=True)

    assert len(combined) == 3000
    assert combined.shape[1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

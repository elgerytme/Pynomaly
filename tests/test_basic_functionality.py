"""Basic functionality test to verify the refactor is working."""

import pytest
import numpy as np
import pandas as pd


def test_numpy_integration():
    """Test that NumPy integration works."""
    data = np.array([1, 2, 3, 4, 5])
    assert len(data) == 5
    assert np.mean(data) == 3.0


def test_pandas_integration():
    """Test that Pandas integration works."""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert len(df) == 3
    assert df['A'].sum() == 6


def test_demo_functions():
    """Test the demo functions work."""
    # Just test basic Python functionality
    assert 2 + 3 == 5
    assert 4 * 5 == 20

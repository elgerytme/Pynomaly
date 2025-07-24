#!/usr/bin/env python3
"""Tests for time series detection service."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

from anomaly_detection.domain.services.time_series_detection_service import TimeSeriesDetectionService
from anomaly_detection.domain.entities.detection_result import DetectionResult


class TestTimeSeriesDetectionService:
    """Test time series detection service functionality."""
    
    @pytest.fixture
    def ts_service(self):
        """Create time series detection service."""
        return TimeSeriesDetectionService()
    
    @pytest.fixture
    def sample_univariate_data(self):
        """Generate sample univariate time series data."""
        np.random.seed(42)
        # Create trend + seasonal + noise + anomalies
        t = np.arange(100)
        trend = 0.1 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 20)
        noise = np.random.normal(0, 0.5, 100)
        normal_data = trend + seasonal + noise
        
        # Add some anomalies
        anomaly_indices = [25, 50, 75]
        normal_data[anomaly_indices] += np.random.choice([-5, 5], size=len(anomaly_indices))
        
        return normal_data
    
    @pytest.fixture
    def sample_multivariate_data(self):
        """Generate sample multivariate time series data."""
        np.random.seed(42)
        n_samples, n_features = 80, 3
        
        # Generate correlated time series
        t = np.arange(n_samples)
        
        # Base patterns
        pattern1 = np.sin(2 * np.pi * t / 15)
        pattern2 = np.cos(2 * np.pi * t / 12)
        pattern3 = 0.5 * np.sin(2 * np.pi * t / 8)
        
        data = np.column_stack([
            pattern1 + np.random.normal(0, 0.2, n_samples),
            pattern2 + np.random.normal(0, 0.2, n_samples),
            pattern3 + np.random.normal(0, 0.2, n_samples)
        ])
        
        # Add anomalies
        data[20, :] += [3, -3, 2]
        data[40, :] += [-2, 4, -3]
        data[60, :] += [4, 2, -4]
        
        return data
    
    @pytest.fixture
    def sample_timestamps(self):
        """Generate sample timestamps."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        return [start_time + timedelta(hours=i) for i in range(100)]
    
    def test_statistical_detection_univariate(self, ts_service, sample_univariate_data):
        """Test statistical detection on univariate data."""
        result = ts_service.detect_anomalies(
            data=sample_univariate_data,
            algorithm="statistical",
            contamination=0.1,
            window_size=10
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.algorithm == "statistical"
        assert result.total_samples == len(sample_univariate_data)
        assert result.anomaly_count > 0
        assert result.anomaly_count <= len(sample_univariate_data) * 0.15  # Reasonable bounds
        assert len(result.predictions) == len(sample_univariate_data)
        assert len(result.confidence_scores) == len(sample_univariate_data)
        assert result.metadata["is_time_series"] is True
        
        print(f"Statistical detection: {result.anomaly_count} anomalies detected")
    
    def test_statistical_detection_multivariate(self, ts_service, sample_multivariate_data):
        """Test statistical detection on multivariate data."""
        result = ts_service.detect_anomalies(
            data=sample_multivariate_data,
            algorithm="statistical",
            contamination=0.1,
            window_size=8
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.algorithm == "statistical"
        assert result.total_samples == len(sample_multivariate_data)
        assert result.anomaly_count > 0
        assert len(result.predictions) == len(sample_multivariate_data)
        assert result.metadata["data_shape"] == sample_multivariate_data.shape
        
        print(f"Multivariate statistical detection: {result.anomaly_count} anomalies detected")
    
    def test_isolation_forest_ts(self, ts_service, sample_univariate_data):
        """Test Isolation Forest with time series features."""
        result = ts_service.detect_anomalies(
            data=sample_univariate_data,
            algorithm="isolation_forest_ts",
            contamination=0.1,
            window_size=5,
            n_estimators=50
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.algorithm == "isolation_forest_ts"
        assert result.total_samples == len(sample_univariate_data)
        assert result.anomaly_count > 0
        assert len(result.predictions) == len(sample_univariate_data)
        
        print(f"Isolation Forest TS: {result.anomaly_count} anomalies detected")
    
    def test_lstm_autoencoder_fallback(self, ts_service, sample_univariate_data):
        """Test LSTM autoencoder (should fallback to statistical if TensorFlow not available)."""
        result = ts_service.detect_anomalies(
            data=sample_univariate_data,
            algorithm="lstm_autoencoder",
            contamination=0.1,
            sequence_length=5,
            epochs=10  # Small for testing
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        # Algorithm might be lstm_autoencoder or statistical (fallback)
        assert result.algorithm in ["lstm_autoencoder", "statistical"]
        assert result.total_samples == len(sample_univariate_data)
        assert len(result.predictions) == len(sample_univariate_data)
        
        print(f"LSTM autoencoder: {result.anomaly_count} anomalies detected (algorithm: {result.algorithm})")
    
    def test_prophet_fallback(self, ts_service, sample_univariate_data, sample_timestamps):
        """Test Prophet detection (should fallback to statistical if Prophet not available)."""
        result = ts_service.detect_anomalies(
            data=sample_univariate_data,
            algorithm="prophet",
            timestamps=sample_timestamps[:len(sample_univariate_data)],
            contamination=0.1
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        # Algorithm might be prophet or statistical (fallback)
        assert result.algorithm in ["prophet", "statistical"]
        assert result.total_samples == len(sample_univariate_data)
        assert len(result.predictions) == len(sample_univariate_data)
        
        print(f"Prophet: {result.anomaly_count} anomalies detected (algorithm: {result.algorithm})")
    
    def test_pandas_series_input(self, ts_service):
        """Test with pandas Series input."""
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        values = np.sin(np.arange(50) * 0.3) + np.random.normal(0, 0.1, 50)
        values[10] += 3  # Add anomaly
        values[25] -= 2  # Add another anomaly
        
        series = pd.Series(values, index=dates)
        
        result = ts_service.detect_anomalies(
            data=series,
            algorithm="statistical",
            timestamps=dates,
            contamination=0.1
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.total_samples == len(values)
        assert result.metadata["time_range"]["start"] is not None
        assert result.metadata["time_range"]["end"] is not None
        
        print(f"Pandas Series: {result.anomaly_count} anomalies detected")
    
    def test_pandas_dataframe_input(self, ts_service):
        """Test with pandas DataFrame input."""
        dates = pd.date_range('2024-01-01', periods=40, freq='h')
        df = pd.DataFrame({
            'sensor1': np.sin(np.arange(40) * 0.2) + np.random.normal(0, 0.1, 40),
            'sensor2': np.cos(np.arange(40) * 0.3) + np.random.normal(0, 0.1, 40),
            'sensor3': np.arange(40) * 0.1 + np.random.normal(0, 0.05, 40)
        }, index=dates)
        
        # Add anomalies
        df.iloc[15, :] += [2, -2, 1]
        df.iloc[30, :] += [-1.5, 3, -2]
        
        result = ts_service.detect_anomalies(
            data=df,
            algorithm="statistical",
            timestamps=dates,
            contamination=0.1
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.total_samples == len(df)
        assert result.metadata["data_shape"] == df.shape
        
        print(f"Pandas DataFrame: {result.anomaly_count} anomalies detected")
    
    def test_algorithm_info(self, ts_service):
        """Test algorithm information retrieval."""
        algorithms = ts_service.list_available_algorithms()
        assert "statistical" in algorithms
        assert "lstm_autoencoder" in algorithms
        assert "prophet" in algorithms
        assert "isolation_forest_ts" in algorithms
        
        # Test individual algorithm info
        statistical_info = ts_service.get_algorithm_info("statistical")
        assert statistical_info["type"] == "time_series"
        assert statistical_info["supports_multivariate"] is True
        assert statistical_info["supports_online"] is True
        
        lstm_info = ts_service.get_algorithm_info("lstm_autoencoder")
        assert "tensorflow" in lstm_info["requires"]
        assert lstm_info["supports_multivariate"] is True
        
        prophet_info = ts_service.get_algorithm_info("prophet")
        assert "prophet" in prophet_info["requires"]
        assert prophet_info["supports_multivariate"] is False
    
    def test_error_handling(self, ts_service):
        """Test error handling for invalid inputs."""
        # Empty data
        with pytest.raises(Exception):
            ts_service.detect_anomalies(data=np.array([]), algorithm="statistical")
        
        # Too few samples
        with pytest.raises(Exception):
            ts_service.detect_anomalies(data=np.array([1, 2]), algorithm="statistical")
        
        # Invalid algorithm
        with pytest.raises(Exception):
            ts_service.detect_anomalies(
                data=np.random.randn(50), 
                algorithm="invalid_algorithm"
            )
        
        # Invalid contamination
        with pytest.raises(Exception):
            ts_service.detect_anomalies(
                data=np.random.randn(50), 
                algorithm="statistical",
                contamination=1.5
            )
    
    def test_timestamp_generation(self, ts_service, sample_univariate_data):
        """Test automatic timestamp generation."""
        result = ts_service.detect_anomalies(
            data=sample_univariate_data,
            algorithm="statistical",
            contamination=0.1
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.metadata["timestamps"] is not None
        assert len(result.metadata["timestamps"]) == len(sample_univariate_data)
        assert result.metadata["time_range"]["start"] is not None
        assert result.metadata["time_range"]["end"] is not None
        
        print(f"Auto-generated timestamps: {result.metadata['time_range']}")


if __name__ == "__main__":
    print("Time Series Detection Service Test")
    print("=" * 40)
    
    # Quick smoke test
    service = TimeSeriesDetectionService()
    
    # Generate test data
    np.random.seed(42)
    test_data = np.sin(np.arange(30) * 0.5) + np.random.normal(0, 0.2, 30)
    test_data[10] += 2  # Add anomaly
    
    try:
        result = service.detect_anomalies(
            data=test_data,
            algorithm="statistical",
            contamination=0.1
        )
        
        print(f"✓ Quick test passed: {result.anomaly_count} anomalies detected")
        print(f"  Algorithm: {result.algorithm}")
        print(f"  Time range: {result.metadata['time_range']['frequency']}")
        print("Ready to run comprehensive time series tests")
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        print("Time series tests may not run properly")
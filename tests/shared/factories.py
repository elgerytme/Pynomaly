"""Unified test factories for creating mock objects and test data."""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest


class MockFactory:
    """Unified factory for creating mock objects used across all tests."""
    
    def __init__(self):
        self._mocks: List[Any] = []
    
    def create_mock(self, spec: Optional[type] = None, **kwargs) -> MagicMock:
        """Create a MagicMock with optional spec."""
        mock = MagicMock(spec=spec, **kwargs)
        self._mocks.append(mock)
        return mock
    
    def create_async_mock(self, spec: Optional[type] = None, **kwargs) -> AsyncMock:
        """Create an AsyncMock with optional spec."""
        mock = AsyncMock(spec=spec, **kwargs)
        self._mocks.append(mock)
        return mock
    
    def create_detector_mock(self, anomaly_scores: Optional[List[float]] = None) -> MagicMock:
        """Create a mock anomaly detector."""
        if anomaly_scores is None:
            anomaly_scores = [0.1, 0.8, 0.3, 0.9, 0.2]
        
        mock = self.create_mock()
        mock.fit.return_value = None
        mock.predict.return_value = anomaly_scores
        mock.decision_function.return_value = anomaly_scores
        mock.score_samples.return_value = anomaly_scores
        return mock
    
    def create_model_mock(self, model_type: str = "isolation_forest") -> MagicMock:
        """Create a mock ML model."""
        mock = self.create_mock()
        mock.model_type = model_type
        mock.fit.return_value = None
        mock.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        return mock
    
    def create_database_mock(self) -> MagicMock:
        """Create a mock database connection."""
        mock = self.create_mock()
        mock.execute.return_value = None
        mock.fetchall.return_value = []
        mock.fetchone.return_value = None
        mock.commit.return_value = None
        mock.rollback.return_value = None
        mock.close.return_value = None
        return mock
    
    def create_api_client_mock(self) -> MagicMock:
        """Create a mock API client."""
        mock = self.create_mock()
        mock.get.return_value.status_code = 200
        mock.get.return_value.json.return_value = {"status": "success"}
        mock.post.return_value.status_code = 201
        mock.post.return_value.json.return_value = {"id": "test-id"}
        mock.put.return_value.status_code = 200
        mock.delete.return_value.status_code = 204
        return mock
    
    def create_storage_mock(self) -> MagicMock:
        """Create a mock storage service."""
        mock = self.create_mock()
        mock.save.return_value = "test-file-id"
        mock.load.return_value = b"test-data"
        mock.delete.return_value = True
        mock.exists.return_value = True
        mock.list.return_value = ["file1.csv", "file2.json"]
        return mock
    
    def create_logger_mock(self) -> MagicMock:
        """Create a mock logger."""
        mock = self.create_mock()
        mock.debug.return_value = None
        mock.info.return_value = None
        mock.warning.return_value = None
        mock.error.return_value = None
        mock.critical.return_value = None
        return mock
    
    def reset_all_mocks(self) -> None:
        """Reset all created mocks."""
        for mock in self._mocks:
            if hasattr(mock, 'reset_mock'):
                mock.reset_mock()
    
    def clear_mocks(self) -> None:
        """Clear all mock references."""
        self._mocks.clear()


class DataFactory:
    """Unified factory for generating test data."""
    
    @staticmethod
    def create_sample_dataframe(
        rows: int = 100,
        columns: int = 5,
        with_anomalies: bool = True,
        anomaly_ratio: float = 0.1,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        np.random.seed(random_state)
        
        # Generate normal data
        data = np.random.randn(rows, columns)
        
        # Add anomalies if requested
        if with_anomalies:
            n_anomalies = int(rows * anomaly_ratio)
            anomaly_indices = np.random.choice(rows, n_anomalies, replace=False)
            
            # Make anomalies by multiplying by a large factor
            data[anomaly_indices] *= np.random.uniform(3, 5, size=(n_anomalies, columns))
        
        # Create DataFrame with meaningful column names
        columns_names = [f"feature_{i}" for i in range(columns)]
        df = pd.DataFrame(data, columns=columns_names)
        
        # Add timestamp column
        start_date = datetime.now() - timedelta(days=rows)
        df['timestamp'] = pd.date_range(start=start_date, periods=rows, freq='H')
        
        return df
    
    @staticmethod
    def create_time_series_data(
        length: int = 1000,
        trend: bool = True,
        seasonality: bool = True,
        noise_level: float = 0.1,
        anomaly_ratio: float = 0.05,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Create time series data with optional trend, seasonality, and anomalies."""
        np.random.seed(random_state)
        
        # Generate time index
        timestamps = pd.date_range(start='2023-01-01', periods=length, freq='H')
        
        # Base signal
        values = np.zeros(length)
        
        # Add trend
        if trend:
            values += np.linspace(0, 10, length)
        
        # Add seasonality
        if seasonality:
            values += 5 * np.sin(2 * np.pi * np.arange(length) / 24)  # Daily pattern
            values += 2 * np.sin(2 * np.pi * np.arange(length) / (24 * 7))  # Weekly pattern
        
        # Add noise
        values += np.random.normal(0, noise_level, length)
        
        # Add anomalies
        if anomaly_ratio > 0:
            n_anomalies = int(length * anomaly_ratio)
            anomaly_indices = np.random.choice(length, n_anomalies, replace=False)
            anomaly_magnitudes = np.random.uniform(5, 15, n_anomalies)
            anomaly_signs = np.random.choice([-1, 1], n_anomalies)
            values[anomaly_indices] += anomaly_magnitudes * anomaly_signs
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
    
    @staticmethod
    def create_multivariate_data(
        samples: int = 500,
        features: int = 10,
        correlation_strength: float = 0.3,
        with_categorical: bool = True,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Create multivariate data with correlations and categorical features."""
        np.random.seed(random_state)
        
        # Create correlation matrix
        correlations = np.random.uniform(-correlation_strength, correlation_strength, (features, features))
        correlations = (correlations + correlations.T) / 2  # Make symmetric
        np.fill_diagonal(correlations, 1.0)
        
        # Generate correlated data
        data = np.random.multivariate_normal(
            mean=np.zeros(features),
            cov=correlations,
            size=samples
        )
        
        # Create feature names
        feature_names = [f"numeric_feature_{i}" for i in range(features)]
        df = pd.DataFrame(data, columns=feature_names)
        
        # Add categorical features
        if with_categorical:
            categories = ['category_A', 'category_B', 'category_C', 'category_D']
            df['categorical_feature'] = np.random.choice(categories, samples)
            
            # Binary feature
            df['binary_feature'] = np.random.choice([0, 1], samples)
        
        return df
    
    @staticmethod
    def create_anomaly_detection_dataset(
        normal_samples: int = 1000,
        anomaly_samples: int = 50,
        features: int = 5,
        random_state: int = 42
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Create a labeled anomaly detection dataset."""
        np.random.seed(random_state)
        
        # Normal samples (mean=0, std=1)
        normal_data = np.random.randn(normal_samples, features)
        
        # Anomalous samples (different distribution)
        anomaly_data = np.random.randn(anomaly_samples, features) * 3 + 5
        
        # Combine data
        all_data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([np.zeros(normal_samples), np.ones(anomaly_samples)])
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(features)]
        df = pd.DataFrame(all_data, columns=feature_names)
        
        return df, labels
    
    @staticmethod
    def create_configuration_dict(
        config_type: str = "default",
        **overrides
    ) -> Dict[str, Any]:
        """Create test configuration dictionaries."""
        base_configs = {
            "default": {
                "model": {
                    "type": "isolation_forest",
                    "contamination": 0.1,
                    "random_state": 42
                },
                "data": {
                    "features": ["feature_0", "feature_1", "feature_2"],
                    "target": "anomaly"
                },
                "preprocessing": {
                    "scale": True,
                    "normalize": False
                }
            },
            "advanced": {
                "model": {
                    "type": "one_class_svm",
                    "nu": 0.05,
                    "kernel": "rbf",
                    "gamma": "scale"
                },
                "data": {
                    "features": "all",
                    "target": "anomaly"
                },
                "preprocessing": {
                    "scale": True,
                    "normalize": True,
                    "pca_components": 0.95
                }
            },
            "minimal": {
                "model": {
                    "type": "local_outlier_factor"
                }
            }
        }
        
        config = base_configs.get(config_type, base_configs["default"]).copy()
        
        # Apply overrides
        for key, value in overrides.items():
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        
        return config
    
    @staticmethod
    def create_random_string(length: int = 10) -> str:
        """Create random string for testing."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def create_file_content(content_type: str = "csv", rows: int = 100) -> str:
        """Create file content for testing file operations."""
        if content_type == "csv":
            df = DataFactory.create_sample_dataframe(rows=rows, columns=3)
            return df.to_csv(index=False)
        elif content_type == "json":
            data = DataFactory.create_configuration_dict()
            import json
            return json.dumps(data, indent=2)
        elif content_type == "text":
            return "\n".join([f"Line {i}" for i in range(rows)])
        else:
            raise ValueError(f"Unsupported content type: {content_type}")


@pytest.fixture(scope="function")
def mock_factory() -> MockFactory:
    """Provide a MockFactory instance for tests."""
    factory = MockFactory()
    yield factory
    factory.reset_all_mocks()
    factory.clear_mocks()


@pytest.fixture(scope="function") 
def data_factory() -> DataFactory:
    """Provide a DataFactory instance for tests."""
    return DataFactory()


@pytest.fixture(scope="function")
def sample_dataframe(data_factory) -> pd.DataFrame:
    """Provide a sample DataFrame for testing."""
    return data_factory.create_sample_dataframe()


@pytest.fixture(scope="function")
def sample_time_series(data_factory) -> pd.DataFrame:
    """Provide sample time series data for testing."""
    return data_factory.create_time_series_data()


@pytest.fixture(scope="function")
def anomaly_dataset(data_factory) -> tuple[pd.DataFrame, np.ndarray]:
    """Provide labeled anomaly detection dataset."""
    return data_factory.create_anomaly_detection_dataset()
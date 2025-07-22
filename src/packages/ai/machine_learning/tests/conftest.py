"""
Pytest configuration for machine learning package testing.
Provides fixtures for algorithm validation, benchmark datasets, and performance testing.
"""

import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


# =========================================================================
# TEST DATA FIXTURES - Algorithm Validation Datasets
# =========================================================================

@pytest.fixture
def benchmark_anomaly_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Create benchmark anomaly detection dataset with known ground truth."""
    np.random.seed(42)
    
    # Normal data (majority class)
    n_normal = 900
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(10), 
        cov=np.eye(10), 
        size=n_normal
    )
    
    # Anomalous data (minority class)
    n_anomalies = 100
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(10) * 3,  # Shifted mean
        cov=np.eye(10) * 2,    # Different covariance
        size=n_anomalies
    )
    
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.ones(n_normal), -np.ones(n_anomalies)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


@pytest.fixture
def high_dimensional_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Create high-dimensional dataset for testing feature selection and dimensionality reduction."""
    np.random.seed(42)
    
    # High-dimensional normal data
    n_samples = 500
    n_features = 100
    X = np.random.randn(n_samples, n_features)
    
    # Add some signal in first 10 features
    X[:, :10] += np.random.randn(10) * 2
    
    # Create anomalies in last 250 samples
    X[-50:, :10] += 5  # Strong signal in first 10 features
    
    y = np.ones(n_samples)
    y[-50:] = -1  # Mark last 50 as anomalies
    
    return X, y


@pytest.fixture
def time_series_anomaly_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Create time series dataset with temporal anomalies."""
    np.random.seed(42)
    
    # Generate normal time series with trend and seasonality
    n_points = 1000
    time = np.arange(n_points)
    
    # Trend + seasonal + noise
    trend = 0.01 * time
    seasonal = 2 * np.sin(2 * np.pi * time / 50)  # Period of 50
    noise = np.random.normal(0, 0.5, n_points)
    
    ts = trend + seasonal + noise
    
    # Inject anomalies
    anomaly_indices = [200, 300, 500, 700, 850]
    for idx in anomaly_indices:
        ts[idx] += np.random.choice([-8, 8])  # Point anomalies
    
    # Add contextual anomaly (different seasonal pattern)
    ts[400:450] = 5 * np.sin(2 * np.pi * np.arange(50) / 10)
    
    # Create labels
    y = np.ones(n_points)
    for idx in anomaly_indices:
        y[idx] = -1
    y[400:450] = -1
    
    return ts.reshape(-1, 1), y


@pytest.fixture
def mixed_data_types_dataset() -> pd.DataFrame:
    """Create dataset with mixed data types for comprehensive testing."""
    np.random.seed(42)
    
    n_samples = 500
    
    data = {
        # Numerical features
        'numerical_1': np.random.randn(n_samples),
        'numerical_2': np.random.exponential(2, n_samples),
        'numerical_3': np.random.gamma(2, 2, n_samples),
        
        # Categorical features
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        
        # Binary feature
        'binary': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        
        # Text-like categorical with high cardinality
        'high_cardinality': [f'item_{i}' for i in np.random.randint(0, 100, n_samples)],
        
        # Missing values
        'with_missing': np.random.randn(n_samples)
    }
    
    # Introduce missing values
    missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    data['with_missing'][missing_idx] = np.nan
    
    df = pd.DataFrame(data)
    
    # Add target for supervised learning scenarios
    df['target'] = (
        0.5 * df['numerical_1'] + 
        0.3 * df['numerical_2'] + 
        0.2 * (df['categorical_1'] == 'A').astype(int) +
        np.random.randn(n_samples) * 0.1
    )
    
    return df


# =========================================================================
# ML ALGORITHM FIXTURES
# =========================================================================

@pytest.fixture
def automl_test_config() -> Dict[str, Any]:
    """Configuration for AutoML testing."""
    return {
        'max_time_minutes': 2,  # Short time for testing
        'max_models': 5,
        'validation_split': 0.2,
        'cross_validation_folds': 3,
        'algorithms': ['random_forest', 'gradient_boosting', 'svm'],
        'hyperparameter_optimization': True,
        'feature_engineering': True,
        'preprocessing_steps': ['scaling', 'encoding', 'feature_selection']
    }


@pytest.fixture
def explainable_ai_config() -> Dict[str, Any]:
    """Configuration for explainable AI testing."""
    return {
        'explanation_methods': ['shap', 'lime', 'permutation'],
        'num_samples': 100,
        'feature_importance_threshold': 0.01,
        'generate_plots': False,  # Disable for testing
        'explanation_depth': 'global'
    }


@pytest.fixture
def ensemble_config() -> Dict[str, Any]:
    """Configuration for ensemble method testing."""
    return {
        'base_models': ['isolation_forest', 'one_class_svm', 'local_outlier_factor'],
        'combination_method': 'average',
        'bootstrap_samples': True,
        'n_estimators': 10,  # Small for testing
        'voting_threshold': 0.5
    }


# =========================================================================
# PERFORMANCE TESTING FIXTURES
# =========================================================================

@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return self.end_time - self.start_time
    
    return Timer()


@pytest.fixture
def large_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Large dataset for performance testing."""
    np.random.seed(42)
    
    n_samples = 10000
    n_features = 50
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=30,
        n_redundant=10,
        n_clusters_per_class=2,
        flip_y=0.1,
        random_state=42
    )
    
    return X, y


@pytest.fixture
def memory_profiler():
    """Memory profiling fixture."""
    try:
        import psutil
        import os
        
        class MemoryProfiler:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.start_memory = None
                
            def start(self):
                self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                
            def get_current_usage(self) -> float:
                return self.process.memory_info().rss / 1024 / 1024  # MB
                
            def get_memory_increase(self) -> float:
                if self.start_memory is None:
                    return 0.0
                return self.get_current_usage() - self.start_memory
        
        return MemoryProfiler()
    except ImportError:
        # Mock profiler if psutil not available
        class MockMemoryProfiler:
            def start(self): pass
            def get_current_usage(self) -> float: return 0.0
            def get_memory_increase(self) -> float: return 0.0
        
        return MockMemoryProfiler()


# =========================================================================
# VALIDATION FIXTURES
# =========================================================================

@pytest.fixture
def accuracy_thresholds() -> Dict[str, float]:
    """Expected accuracy thresholds for different algorithms."""
    return {
        'isolation_forest': 0.80,
        'one_class_svm': 0.75,
        'local_outlier_factor': 0.70,
        'autoencoder': 0.78,
        'ensemble': 0.85,
        'automl': 0.82
    }


@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Performance thresholds for different operations."""
    return {
        'training_time_seconds': 30.0,
        'prediction_time_per_sample_ms': 1.0,
        'memory_increase_mb': 200.0,
        'model_size_mb': 50.0
    }


# =========================================================================
# MOCK FIXTURES FOR EXTERNAL DEPENDENCIES
# =========================================================================

@pytest.fixture
def mock_model_registry():
    """Mock model registry for testing."""
    mock = Mock()
    mock.save_model.return_value = "model_id_123"
    mock.load_model.return_value = Mock()
    mock.get_model_metadata.return_value = {
        'accuracy': 0.85,
        'created_at': '2024-01-01T00:00:00Z',
        'version': '1.0.0'
    }
    return mock


@pytest.fixture
def mock_feature_store():
    """Mock feature store for testing."""
    mock = Mock()
    mock.get_features.return_value = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100)
    })
    mock.save_features.return_value = True
    return mock


# =========================================================================
# PYTEST CONFIGURATION
# =========================================================================

def pytest_configure(config):
    """Configure pytest with custom markers for ML testing."""
    
    markers = [
        "algorithm_validation: Tests that validate algorithm accuracy",
        "performance: Performance and benchmark tests",
        "integration: Integration tests across ML components",
        "automl: AutoML-specific tests",
        "explainable: Explainable AI tests",
        "ensemble: Ensemble method tests",
        "slow: Tests that take more than 5 seconds",
        "requires_gpu: Tests requiring GPU acceleration",
        "requires_large_memory: Tests requiring >2GB memory",
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_runtest_setup(item):
    """Setup for each test item."""
    # Skip GPU tests if no GPU available
    if item.get_closest_marker("requires_gpu"):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available for GPU testing")
    
    # Skip large memory tests in constrained environments
    if item.get_closest_marker("requires_large_memory"):
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
            if available_memory < 4:
                pytest.skip("Insufficient memory for large memory tests")
        except ImportError:
            pass  # Assume sufficient memory if can't check


# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================

def calculate_anomaly_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive anomaly detection metrics."""
    try:
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, 
            roc_auc_score, confusion_matrix
        )
        
        # Handle different anomaly labeling conventions
        if set(np.unique(y_true)) == {-1, 1}:
            # Convert to 0/1 for sklearn metrics
            y_true_binary = (y_true == -1).astype(int)
            y_pred_binary = (y_pred == -1).astype(int)
        else:
            y_true_binary = y_true
            y_pred_binary = y_pred
            
        metrics = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        }
        
        # ROC AUC if possible
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_binary, y_pred_binary)
        except ValueError:
            metrics['roc_auc'] = 0.0
            
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        metrics.update({
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        })
        
        return metrics
        
    except ImportError:
        # Fallback if sklearn not available
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0,
            'accuracy': 0.0
        }
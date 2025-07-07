"""Test data generation and management system for consistent testing."""

import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification, make_moons

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate


class TestDataGenerator:
    """Generate standardized test data for different testing scenarios."""

    def __init__(self, random_state: int = 42):
        """Initialize test data generator.

        Args:
            random_state: Random seed for reproducible data generation
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_simple_dataset(
        self, n_samples: int = 1000, n_features: int = 10, contamination: float = 0.1
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate a simple dataset with known anomalies.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            contamination: Fraction of samples that are anomalies

        Returns:
            Tuple of (DataFrame, labels) where labels indicate anomalies
        """
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal

        # Generate normal data
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features), cov=np.eye(n_features), size=n_normal
        )

        # Generate anomalous data
        anomaly_data = np.random.multivariate_normal(
            mean=np.full(n_features, 3.0),  # Shifted mean
            cov=np.eye(n_features) * 2.0,  # Different covariance
            size=n_anomalies,
        )

        # Combine and shuffle
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        # Create DataFrame
        columns = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)

        return df, y

    def generate_clustered_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 5,
        n_clusters: int = 3,
        contamination: float = 0.1,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate dataset with clustered normal data and scattered anomalies.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_clusters: Number of clusters for normal data
            contamination: Fraction of samples that are anomalies

        Returns:
            Tuple of (DataFrame, labels)
        """
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal

        # Generate clustered normal data
        normal_data, _ = make_blobs(
            n_samples=n_normal,
            centers=n_clusters,
            n_features=n_features,
            cluster_std=1.0,
            random_state=self.random_state,
        )

        # Generate scattered anomalies
        data_min = normal_data.min(axis=0) - 3
        data_max = normal_data.max(axis=0) + 3

        anomaly_data = np.random.uniform(
            low=data_min, high=data_max, size=(n_anomalies, n_features)
        )

        # Combine and shuffle
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        columns = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)

        return df, y

    def generate_time_series_dataset(
        self,
        n_timestamps: int = 1000,
        n_features: int = 5,
        anomaly_periods: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate time series data with temporal anomalies.

        Args:
            n_timestamps: Number of time points
            n_features: Number of features
            anomaly_periods: List of (start, end) tuples for anomaly periods

        Returns:
            Tuple of (DataFrame with timestamp, labels)
        """
        if anomaly_periods is None:
            anomaly_periods = [(200, 250), (700, 750)]

        # Generate base time series
        timestamps = pd.date_range(start="2023-01-01", periods=n_timestamps, freq="1H")

        # Generate normal seasonal pattern
        t = np.arange(n_timestamps)
        data = np.zeros((n_timestamps, n_features))

        for i in range(n_features):
            # Base trend
            trend = 0.001 * t * (i + 1)

            # Seasonal patterns
            daily_pattern = np.sin(2 * np.pi * t / 24) * (i + 1)
            weekly_pattern = np.sin(2 * np.pi * t / (24 * 7)) * 0.5 * (i + 1)

            # Noise
            noise = np.random.normal(0, 0.1, n_timestamps)

            data[:, i] = trend + daily_pattern + weekly_pattern + noise

        # Add anomalies
        labels = np.zeros(n_timestamps)
        for start, end in anomaly_periods:
            # Add spikes or drops
            for i in range(n_features):
                if np.random.random() > 0.5:
                    # Spike
                    data[start:end, i] += np.random.uniform(2, 5) * (i + 1)
                else:
                    # Drop
                    data[start:end, i] -= np.random.uniform(1, 3) * (i + 1)

            labels[start:end] = 1

        # Create DataFrame
        columns = [f"sensor_{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=columns, index=timestamps)
        df["timestamp"] = timestamps

        return df, labels

    def generate_mixed_type_dataset(
        self, n_samples: int = 1000, contamination: float = 0.1
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate dataset with mixed data types (numeric, categorical, text).

        Args:
            n_samples: Number of samples
            contamination: Fraction of anomalous samples

        Returns:
            Tuple of (DataFrame, labels)
        """
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal

        # Numeric features
        numeric_normal = np.random.normal(0, 1, (n_normal, 3))
        numeric_anomalies = np.random.normal(5, 2, (n_anomalies, 3))
        numeric_data = np.vstack([numeric_normal, numeric_anomalies])

        # Categorical features
        normal_categories = np.random.choice(
            ["A", "B", "C"], n_normal, p=[0.5, 0.3, 0.2]
        )
        anomaly_categories = np.random.choice(
            ["A", "B", "C", "D"], n_anomalies, p=[0.1, 0.1, 0.2, 0.6]
        )
        categorical_data = np.hstack([normal_categories, anomaly_categories])

        # Boolean features
        boolean_normal = np.random.choice([True, False], n_normal, p=[0.7, 0.3])
        boolean_anomalies = np.random.choice([True, False], n_anomalies, p=[0.2, 0.8])
        boolean_data = np.hstack([boolean_normal, boolean_anomalies])

        # Text-like features (simulated as length)
        text_normal = np.random.normal(50, 10, n_normal).astype(int)
        text_anomalies = np.random.normal(200, 50, n_anomalies).astype(int)
        text_data = np.hstack([text_normal, text_anomalies])

        # Combine and shuffle
        labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
        indices = np.random.permutation(n_samples)

        df = pd.DataFrame(
            {
                "numeric_1": numeric_data[indices, 0],
                "numeric_2": numeric_data[indices, 1],
                "numeric_3": numeric_data[indices, 2],
                "category": categorical_data[indices],
                "boolean_flag": boolean_data[indices],
                "text_length": text_data[indices],
                "id": range(n_samples),
            }
        )

        labels = labels[indices]

        return df, labels

    def generate_high_dimensional_dataset(
        self, n_samples: int = 500, n_features: int = 100, contamination: float = 0.1
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate high-dimensional dataset for testing curse of dimensionality.

        Args:
            n_samples: Number of samples
            n_features: Number of features (high)
            contamination: Fraction of anomalous samples

        Returns:
            Tuple of (DataFrame, labels)
        """
        df, labels = self.generate_simple_dataset(n_samples, n_features, contamination)

        # Add noise features that don't contribute to anomaly detection
        n_noise_features = n_features // 4
        noise_data = np.random.normal(0, 0.1, (n_samples, n_noise_features))

        for i in range(n_noise_features):
            df[f"noise_{i}"] = noise_data[:, i]

        return df, labels


class TestDataManager:
    """Manage and cache test datasets for consistent testing."""

    def __init__(self, cache_dir: str = "test_data_cache"):
        """Initialize test data manager.

        Args:
            cache_dir: Directory to store cached test data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.generator = TestDataGenerator()

    def get_dataset(
        self, dataset_type: str, use_cache: bool = True, **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Get a dataset of specified type, using cache if available.

        Args:
            dataset_type: Type of dataset ('simple', 'clustered', 'timeseries', etc.)
            use_cache: Whether to use cached data if available
            **kwargs: Parameters for dataset generation

        Returns:
            Tuple of (DataFrame, labels)
        """
        # Create cache key from parameters
        cache_key = f"{dataset_type}_{hash(str(sorted(kwargs.items())))}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Try to load from cache
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                # Cache corrupted, regenerate
                pass

        # Generate new dataset
        if dataset_type == "simple":
            data = self.generator.generate_simple_dataset(**kwargs)
        elif dataset_type == "clustered":
            data = self.generator.generate_clustered_dataset(**kwargs)
        elif dataset_type == "timeseries":
            data = self.generator.generate_time_series_dataset(**kwargs)
        elif dataset_type == "mixed_types":
            data = self.generator.generate_mixed_type_dataset(**kwargs)
        elif dataset_type == "high_dimensional":
            data = self.generator.generate_high_dimensional_dataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Cache the result
        if use_cache:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(data, f)
            except Exception:
                # Caching failed, continue without cache
                pass

        return data

    def create_domain_entities(
        self, dataset_type: str, **kwargs
    ) -> Tuple[Dataset, List[Anomaly], DetectionResult]:
        """Create domain entities from test data.

        Args:
            dataset_type: Type of dataset to generate
            **kwargs: Parameters for dataset generation

        Returns:
            Tuple of (Dataset, list of Anomalies, DetectionResult)
        """
        df, labels = self.get_dataset(dataset_type, **kwargs)

        # Create Dataset entity
        dataset = Dataset(
            name=f"test_{dataset_type}_dataset", data=df, target_column=None
        )

        # Create Anomaly entities for anomalous samples
        anomalies = []
        anomaly_indices = np.where(labels == 1)[0]

        for idx in anomaly_indices:
            score = AnomalyScore(
                value=np.random.uniform(0.7, 1.0),  # High score for known anomalies
                method="test_generator",
            )

            data_point = df.iloc[idx].to_dict()

            anomaly = Anomaly(
                score=score, data_point=data_point, detector_name="test_detector"
            )
            anomalies.append(anomaly)

        # Create DetectionResult
        scores = []
        for i in range(len(df)):
            if labels[i] == 1:
                score_value = np.random.uniform(0.7, 1.0)
            else:
                score_value = np.random.uniform(0.0, 0.3)

            scores.append(AnomalyScore(value=score_value, method="test_generator"))

        detection_result = DetectionResult(
            detector_id="test_detector_123",
            dataset_id=dataset.name,
            anomalies=anomalies,
            scores=scores,
            labels=labels.tolist(),
            threshold=0.5,
        )

        return dataset, anomalies, detection_result

    def create_test_detector(
        self, algorithm_name: str = "TestAlgorithm", contamination: float = 0.1
    ) -> Detector:
        """Create a test detector entity.

        Args:
            algorithm_name: Name of the algorithm
            contamination: Contamination rate

        Returns:
            Detector entity
        """
        contamination_rate = ContaminationRate(contamination)

        detector = Detector(
            name=f"test_{algorithm_name.lower()}_detector",
            algorithm_name=algorithm_name,
            contamination_rate=contamination_rate,
            parameters={"n_estimators": 100, "random_state": 42, "test_mode": True},
            metadata={"created_for": "testing", "environment": "test"},
        )

        return detector

    def clear_cache(self):
        """Clear all cached test data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached datasets.

        Returns:
            Dictionary with cache information
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "num_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "files": [f.name for f in cache_files],
        }


class TestScenarioFactory:
    """Factory for creating complete test scenarios."""

    def __init__(self):
        """Initialize test scenario factory."""
        self.data_manager = TestDataManager()

    def create_basic_anomaly_detection_scenario(self) -> Dict[str, Any]:
        """Create a basic anomaly detection test scenario.

        Returns:
            Dictionary containing all test components
        """
        dataset, anomalies, detection_result = self.data_manager.create_domain_entities(
            "simple", n_samples=500, n_features=8, contamination=0.1
        )

        detector = self.data_manager.create_test_detector(
            algorithm_name="IsolationForest", contamination=0.1
        )

        return {
            "scenario_name": "basic_anomaly_detection",
            "dataset": dataset,
            "detector": detector,
            "anomalies": anomalies,
            "detection_result": detection_result,
            "expected_anomaly_count": len(anomalies),
            "test_assertions": {
                "min_anomalies": 40,  # At least 40 anomalies in 500 samples with 10% contamination
                "max_anomalies": 60,  # At most 60 anomalies
                "min_precision": 0.7,  # At least 70% precision expected
                "dataset_not_empty": True,
                "all_scores_valid": True,
            },
        }

    def create_high_dimensional_scenario(self) -> Dict[str, Any]:
        """Create high-dimensional data test scenario."""
        dataset, anomalies, detection_result = self.data_manager.create_domain_entities(
            "high_dimensional", n_samples=300, n_features=100, contamination=0.15
        )

        detector = self.data_manager.create_test_detector(
            algorithm_name="IsolationForest", contamination=0.15
        )

        return {
            "scenario_name": "high_dimensional",
            "dataset": dataset,
            "detector": detector,
            "anomalies": anomalies,
            "detection_result": detection_result,
            "expected_anomaly_count": len(anomalies),
            "test_assertions": {
                "min_features": 100,
                "algorithm_scalability": True,
                "reasonable_performance": True,
            },
        }

    def create_time_series_scenario(self) -> Dict[str, Any]:
        """Create time series anomaly detection scenario."""
        dataset, anomalies, detection_result = self.data_manager.create_domain_entities(
            "timeseries",
            n_timestamps=1000,
            n_features=5,
            anomaly_periods=[(200, 250), (700, 750)],
        )

        detector = self.data_manager.create_test_detector(
            algorithm_name="TimeSeriesAD", contamination=0.1
        )

        return {
            "scenario_name": "time_series",
            "dataset": dataset,
            "detector": detector,
            "anomalies": anomalies,
            "detection_result": detection_result,
            "expected_anomaly_count": len(anomalies),
            "test_assertions": {
                "has_temporal_structure": True,
                "anomaly_periods_detected": True,
                "temporal_consistency": True,
            },
        }

    def create_mixed_data_types_scenario(self) -> Dict[str, Any]:
        """Create scenario with mixed data types."""
        dataset, anomalies, detection_result = self.data_manager.create_domain_entities(
            "mixed_types", n_samples=800, contamination=0.12
        )

        detector = self.data_manager.create_test_detector(
            algorithm_name="MixedTypeAD", contamination=0.12
        )

        return {
            "scenario_name": "mixed_data_types",
            "dataset": dataset,
            "detector": detector,
            "anomalies": anomalies,
            "detection_result": detection_result,
            "expected_anomaly_count": len(anomalies),
            "test_assertions": {
                "handles_numeric_features": True,
                "handles_categorical_features": True,
                "handles_boolean_features": True,
                "data_preprocessing_needed": True,
            },
        }


# Pytest fixtures for easy use in tests
def pytest_test_data_manager():
    """Pytest fixture for test data manager."""
    return TestDataManager()


def pytest_test_scenario_factory():
    """Pytest fixture for test scenario factory."""
    return TestScenarioFactory()


# Export common test datasets as constants
SMALL_DATASET_PARAMS = {"n_samples": 100, "n_features": 5, "contamination": 0.1}

MEDIUM_DATASET_PARAMS = {"n_samples": 1000, "n_features": 15, "contamination": 0.1}

LARGE_DATASET_PARAMS = {"n_samples": 10000, "n_features": 30, "contamination": 0.05}

HIGH_DIM_DATASET_PARAMS = {"n_samples": 500, "n_features": 200, "contamination": 0.1}

"""Mock data registries and lightweight data generation for fast testing."""

from __future__ import annotations

import unittest.mock
from contextlib import contextmanager
from typing import Dict, List, Any
from uuid import uuid4, UUID

import pandas as pd
import numpy as np

try:
    from pynomaly.domain.entities import Dataset, Detector
    from pynomaly.domain.value_objects import ContaminationRate
except ImportError:
    # Handle case where domain entities are not available
    Dataset = None
    Detector = None
    ContaminationRate = None


def create_lightweight_dataset(
    name: str = "Test Dataset", 
    rows: int = 100, 
    features: int = 5,
    anomaly_rate: float = 0.1,
    **kwargs
) -> Any:
    """Create a lightweight dataset for testing."""
    if Dataset is None:
        # Return mock if entities not available
        mock_dataset = unittest.mock.MagicMock()
        mock_dataset.name = name
        mock_dataset.id = uuid4()
        mock_dataset.data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, rows)
            for i in range(features)
        })
        return mock_dataset
    
    # Generate deterministic data for consistency
    np.random.seed(42)
    
    # Create feature data
    data = {}
    for i in range(features):
        if i == 0:
            # Make first feature slightly more discriminative
            normal_data = np.random.normal(0, 1, int(rows * (1 - anomaly_rate)))
            anomaly_data = np.random.normal(3, 1, int(rows * anomaly_rate))
            data[f'feature_{i}'] = np.concatenate([normal_data, anomaly_data])
        else:
            data[f'feature_{i}'] = np.random.normal(0, 1, rows)
    
    # Add target column if requested
    if kwargs.get('include_target', True):
        target = np.zeros(rows)
        target[int(rows * (1 - anomaly_rate)):] = 1
        data['target'] = target
    
    df = pd.DataFrame(data)
    
    return Dataset(
        id=kwargs.get('id', uuid4()),
        name=name,
        data=df,
        target_column=kwargs.get('target_column', 'target' if 'target' in data else None),
        feature_names=[f'feature_{i}' for i in range(features)],
        metadata={
            'test': True,
            'rows': rows,
            'features': features,
            'anomaly_rate': anomaly_rate,
            **kwargs.get('metadata', {})
        }
    )


def create_lightweight_detector(
    name: str = "Test Detector",
    algorithm: str = "IsolationForest", 
    **kwargs
) -> Any:
    """Create a lightweight detector for testing."""
    if Detector is None or ContaminationRate is None:
        # Return mock if entities not available
        mock_detector = unittest.mock.MagicMock()
        mock_detector.name = name
        mock_detector.algorithm_name = algorithm
        mock_detector.id = uuid4()
        mock_detector.parameters = kwargs.get('parameters', {'contamination': 0.1})
        mock_detector.is_fitted = kwargs.get('is_fitted', False)
        return mock_detector
    
    default_params = {
        'IsolationForest': {'contamination': 0.1, 'random_state': 42},
        'LOF': {'contamination': 0.1, 'n_neighbors': 5},
        'OneClassSVM': {'nu': 0.1, 'kernel': 'rbf'},
        'ABOD': {'contamination': 0.1},
        'CBLOF': {'contamination': 0.1, 'n_clusters': 8},
        'HBOS': {'contamination': 0.1, 'n_bins': 10},
        'KNN': {'contamination': 0.1, 'n_neighbors': 5},
        'PCA': {'contamination': 0.1}
    }
    
    parameters = kwargs.get('parameters', default_params.get(algorithm, {'contamination': 0.1}))
    contamination = parameters.get('contamination', 0.1)
    
    return Detector(
        id=kwargs.get('id', uuid4()),
        name=name,
        algorithm_name=algorithm,
        parameters=parameters,
        contamination_rate=ContaminationRate(contamination),
        is_fitted=kwargs.get('is_fitted', False),
        metadata={
            'test': True,
            'fast_mode': True,
            **kwargs.get('metadata', {})
        }
    )


class MockDatasetRegistry:
    """Lightweight in-memory dataset registry for testing."""
    
    def __init__(self):
        """Initialize with predefined test datasets."""
        self.datasets = {
            'small': create_lightweight_dataset(
                name='Small Test Dataset',
                rows=100,
                features=3
            ),
            'medium': create_lightweight_dataset(
                name='Medium Test Dataset',
                rows=1000,
                features=5
            ),
            'large': create_lightweight_dataset(
                name='Large Test Dataset',
                rows=5000,  # Still lightweight for tests
                features=10
            ),
            'high_dim': create_lightweight_dataset(
                name='High Dimensional Dataset',
                rows=500,
                features=50
            ),
            'sparse': create_lightweight_dataset(
                name='Sparse Dataset',
                rows=200,
                features=3,
                anomaly_rate=0.01  # Very few anomalies
            ),
            'noisy': create_lightweight_dataset(
                name='Noisy Dataset',
                rows=500,
                features=5,
                anomaly_rate=0.3  # Many anomalies
            )
        }
    
    def get_dataset(self, name: str) -> Any:
        """Get dataset by name."""
        return self.datasets.get(name)
    
    def get_cached_dataset(self, name: str) -> Any:
        """Get cached dataset by name."""
        return self.datasets.get(name)
    
    def create_dataset(self, name: str, **kwargs) -> Any:
        """Create a new dataset."""
        return create_lightweight_dataset(name=name, **kwargs)
    
    def list_datasets(self) -> List[str]:
        """List available dataset names."""
        return list(self.datasets.keys())
    
    def add_dataset(self, name: str, dataset: Any) -> None:
        """Add a dataset to the registry."""
        self.datasets[name] = dataset
    
    def clear(self) -> None:
        """Clear all datasets."""
        self.datasets.clear()
    
    @contextmanager
    def patch_repository(self):
        """Patch dataset repository with mock registry."""
        mock_repo = unittest.mock.MagicMock()
        mock_repo.find_all.return_value = list(self.datasets.values())
        mock_repo.find_by_id.side_effect = lambda id: next(
            (d for d in self.datasets.values() if d.id == id), None
        )
        mock_repo.find_by_name.side_effect = lambda name: self.datasets.get(name)
        
        with unittest.mock.patch('pynomaly.infrastructure.repositories.dataset_repository.DatasetRepository', return_value=mock_repo):
            yield mock_repo


class MockDetectorRegistry:
    """Lightweight in-memory detector registry for testing."""
    
    def __init__(self):
        """Initialize with predefined test detectors."""
        self.detectors = {
            'isolation_forest': create_lightweight_detector(
                name='Test Isolation Forest',
                algorithm='IsolationForest'
            ),
            'lof': create_lightweight_detector(
                name='Test LOF',
                algorithm='LOF'
            ),
            'one_class_svm': create_lightweight_detector(
                name='Test One-Class SVM',
                algorithm='OneClassSVM'
            ),
            'abod': create_lightweight_detector(
                name='Test ABOD',
                algorithm='ABOD'
            ),
            'cblof': create_lightweight_detector(
                name='Test CBLOF',
                algorithm='CBLOF'
            ),
            'hbos': create_lightweight_detector(
                name='Test HBOS',
                algorithm='HBOS'
            ),
            'knn': create_lightweight_detector(
                name='Test KNN',
                algorithm='KNN'
            ),
            'pca': create_lightweight_detector(
                name='Test PCA',
                algorithm='PCA'
            )
        }
        
        # Mark some as fitted for testing
        self.detectors['isolation_forest'].is_fitted = True
        self.detectors['lof'].is_fitted = True
    
    def get_detector(self, name: str) -> Any:
        """Get detector by name."""
        return self.detectors.get(name)
    
    def get_cached_detector(self, name: str) -> Any:
        """Get cached detector by name."""
        return self.detectors.get(name)
    
    def create_detector(self, algorithm: str, **kwargs) -> Any:
        """Create a new detector."""
        return create_lightweight_detector(algorithm=algorithm, **kwargs)
    
    def list_detectors(self) -> List[str]:
        """List available detector names."""
        return list(self.detectors.keys())
    
    def add_detector(self, name: str, detector: Any) -> None:
        """Add a detector to the registry."""
        self.detectors[name] = detector
    
    def clear(self) -> None:
        """Clear all detectors."""
        self.detectors.clear()
    
    @contextmanager
    def patch_repository(self):
        """Patch detector repository with mock registry."""
        mock_repo = unittest.mock.MagicMock()
        mock_repo.find_all.return_value = list(self.detectors.values())
        mock_repo.find_by_id.side_effect = lambda id: next(
            (d for d in self.detectors.values() if d.id == id), None
        )
        mock_repo.find_by_name.side_effect = lambda name: self.detectors.get(name)
        
        with unittest.mock.patch('pynomaly.infrastructure.repositories.detector_repository.DetectorRepository', return_value=mock_repo):
            yield mock_repo


class MockResultRegistry:
    """Mock detection results for testing."""
    
    @staticmethod
    def create_mock_detection_result(
        detector_id: UUID = None,
        dataset_id: UUID = None,
        n_samples: int = 100,
        anomaly_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Create mock detection result."""
        np.random.seed(42)
        
        n_anomalies = int(n_samples * anomaly_rate)
        
        # Generate scores (higher for anomalies)
        normal_scores = np.random.beta(2, 5, n_samples - n_anomalies)  # Lower scores
        anomaly_scores = np.random.beta(5, 2, n_anomalies)  # Higher scores
        scores = np.concatenate([normal_scores, anomaly_scores])
        
        # Generate predictions based on threshold
        threshold = np.percentile(scores, (1 - anomaly_rate) * 100)
        predictions = (scores > threshold).astype(int)
        
        return {
            'detector_id': detector_id or uuid4(),
            'dataset_id': dataset_id or uuid4(),
            'scores': scores.tolist(),
            'predictions': predictions.tolist(),
            'threshold': threshold,
            'n_samples': n_samples,
            'n_anomalies': int(predictions.sum()),
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.90,
                'f1_score': 0.85,
                'auc': 0.88
            },
            'execution_time': 0.001,  # 1ms for fast tests
            'metadata': {
                'test': True,
                'mock_result': True
            }
        }


# Context manager for comprehensive data mocking
@contextmanager
def mock_all_data_operations():
    """Comprehensive context manager for all data operation mocks."""
    dataset_registry = MockDatasetRegistry()
    detector_registry = MockDetectorRegistry()
    
    with dataset_registry.patch_repository() as dataset_repo, \
         detector_registry.patch_repository() as detector_repo:
        yield {
            'dataset_registry': dataset_registry,
            'detector_registry': detector_registry,
            'dataset_repo': dataset_repo,
            'detector_repo': detector_repo
        }

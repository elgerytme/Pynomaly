"""
Pytest configuration for MLOps package testing.
Provides fixtures for model management, pipeline orchestration, and experiment tracking testing.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import time
from uuid import uuid4, UUID
from datetime import datetime, timedelta
from unittest.mock import Mock


@pytest.fixture
def sample_model_data() -> Dict[str, Any]:
    """Sample model data for testing."""
    return {
        'id': uuid4(),
        'name': 'test_ml_model',
        'description': 'Test machine learning model',
        'model_type': 'classification',
        'algorithm_family': 'isolation_forest',
        'created_by': 'test_user',
        'team': 'ml_team',
        'use_cases': ['fraud_detection', 'system_monitoring'],
        'data_requirements': {
            'features': 10,
            'min_samples': 1000,
            'data_types': ['numerical', 'categorical']
        }
    }


@pytest.fixture
def sample_performance_metrics() -> Dict[str, float]:
    """Sample performance metrics for testing."""
    return {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1_score': 0.85,
        'auc_roc': 0.90,
        'training_time': 45.2,
        'inference_time': 0.05,
        'memory_usage': 256.7
    }


@pytest.fixture
def training_dataset() -> pd.DataFrame:
    """Generate training dataset for MLOps testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate normal data
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=int(n_samples * 0.9)
    )
    
    # Generate anomalous data
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,
        cov=np.eye(n_features) * 2,
        size=int(n_samples * 0.1)
    )
    
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.ones(len(normal_data)), -np.ones(len(anomaly_data))])
    
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y
    df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='h')
    
    return df


@pytest.fixture
def experiment_config() -> Dict[str, Any]:
    """Standard experiment configuration."""
    return {
        'experiment_name': 'ml_model_v1',
        'algorithm': 'isolation_forest',
        'hyperparameters': {
            'n_estimators': 100,
            'contamination': 0.1,
            'random_state': 42,
            'max_features': 1.0
        },
        'cross_validation': {
            'cv_folds': 5,
            'stratified': True
        },
        'evaluation_metrics': [
            'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'
        ],
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001
        }
    }


@pytest.fixture
def pipeline_config() -> Dict[str, Any]:
    """Standard ML pipeline configuration."""
    return {
        'pipeline_name': 'ml_training_pipeline',
        'stages': [
            {
                'name': 'data_ingestion',
                'type': 'data_loader',
                'config': {'batch_size': 1000}
            },
            {
                'name': 'data_preprocessing',
                'type': 'preprocessor',
                'config': {
                    'scaling': 'standard',
                    'handle_missing': 'median'
                }
            },
            {
                'name': 'feature_engineering',
                'type': 'feature_transformer',
                'config': {'method': 'pca', 'n_components': 0.95}
            },
            {
                'name': 'model_training',
                'type': 'trainer',
                'config': {'algorithm': 'isolation_forest'}
            },
            {
                'name': 'model_evaluation',
                'type': 'evaluator',
                'config': {'metrics': ['precision', 'recall', 'f1']}
            }
        ],
        'data_validation': {
            'schema_validation': True,
            'data_drift_detection': True,
            'quality_checks': True
        },
        'model_validation': {
            'performance_threshold': 0.8,
            'bias_detection': True,
            'explainability_check': True
        }
    }


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
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
def mock_model_registry():
    """Mock model registry for testing."""
    class MockModelRegistry:
        def __init__(self):
            self.models = {}
            self.versions = {}
            
        def register_model(self, model_name: str, model_data: Dict[str, Any]) -> UUID:
            model_id = uuid4()
            self.models[model_id] = {
                'id': model_id,
                'name': model_name,
                'created_at': datetime.utcnow(),
                **model_data
            }
            return model_id
            
        def register_version(self, model_id: UUID, version_data: Dict[str, Any]) -> UUID:
            version_id = uuid4()
            self.versions[version_id] = {
                'id': version_id,
                'model_id': model_id,
                'created_at': datetime.utcnow(),
                **version_data
            }
            return version_id
            
        def get_model(self, model_id: UUID) -> Dict[str, Any]:
            return self.models.get(model_id)
            
        def get_version(self, version_id: UUID) -> Dict[str, Any]:
            return self.versions.get(version_id)
            
        def list_models(self) -> List[Dict[str, Any]]:
            return list(self.models.values())
            
        def list_versions(self, model_id: UUID) -> List[Dict[str, Any]]:
            return [v for v in self.versions.values() if v['model_id'] == model_id]
    
    return MockModelRegistry()


@pytest.fixture
def mock_experiment_tracker():
    """Mock experiment tracker for testing."""
    class MockExperimentTracker:
        def __init__(self):
            self.experiments = {}
            self.runs = {}
            
        def create_experiment(self, experiment_name: str, config: Dict[str, Any]) -> UUID:
            experiment_id = uuid4()
            self.experiments[experiment_id] = {
                'id': experiment_id,
                'name': experiment_name,
                'config': config,
                'created_at': datetime.utcnow(),
                'runs': []
            }
            return experiment_id
            
        def start_run(self, experiment_id: UUID, run_config: Dict[str, Any]) -> UUID:
            run_id = uuid4()
            self.runs[run_id] = {
                'id': run_id,
                'experiment_id': experiment_id,
                'config': run_config,
                'metrics': {},
                'artifacts': {},
                'status': 'running',
                'started_at': datetime.utcnow()
            }
            
            if experiment_id in self.experiments:
                self.experiments[experiment_id]['runs'].append(run_id)
                
            return run_id
            
        def log_metrics(self, run_id: UUID, metrics: Dict[str, float]):
            if run_id in self.runs:
                self.runs[run_id]['metrics'].update(metrics)
                
        def log_artifact(self, run_id: UUID, artifact_name: str, artifact_path: str):
            if run_id in self.runs:
                self.runs[run_id]['artifacts'][artifact_name] = artifact_path
                
        def finish_run(self, run_id: UUID, status: str = 'completed'):
            if run_id in self.runs:
                self.runs[run_id]['status'] = status
                self.runs[run_id]['finished_at'] = datetime.utcnow()
                
        def get_experiment(self, experiment_id: UUID) -> Dict[str, Any]:
            return self.experiments.get(experiment_id)
            
        def get_run(self, run_id: UUID) -> Dict[str, Any]:
            return self.runs.get(run_id)
    
    return MockExperimentTracker()


@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Large dataset for performance testing."""
    np.random.seed(42)
    n_rows = 50000
    n_features = 20
    
    data = np.random.randn(n_rows, n_features)
    columns = [f'feature_{i}' for i in range(n_features)]
    
    df = pd.DataFrame(data, columns=columns)
    df['target'] = np.random.choice([0, 1], n_rows, p=[0.9, 0.1])
    df['timestamp'] = pd.date_range('2024-01-01', periods=n_rows, freq='min')
    
    return df


def pytest_configure(config):
    """Configure pytest markers for MLOps testing."""
    markers = [
        "mlops: MLOps functionality tests",
        "model_management: Model lifecycle management tests",
        "experiment_tracking: Experiment tracking tests",
        "pipeline_orchestration: ML pipeline tests",
        "performance: MLOps performance tests",
        "integration: MLOps integration tests"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)

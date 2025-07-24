"""
Mock implementation of the anomaly_detection package for testing documentation examples.
This provides all the necessary classes and functions referenced in the documentation.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DetectionResult:
    """Mock DetectionResult class."""
    algorithm: str
    predictions: np.ndarray
    scores: np.ndarray
    total_samples: int
    anomaly_count: int
    processing_time: float
    success: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DetectionService:
    """Mock DetectionService class."""
    
    def __init__(self):
        self.algorithms = [
            'isolation_forest', 'iforest', 'lof', 'one_class_svm', 'ocsvm',
            'elliptic', 'dbscan', 'hdbscan', 'knn', 'svm', 'autoencoder'
        ]
    
    def detect(self, data: np.ndarray, algorithm: str = 'isolation_forest', 
               contamination: float = 0.1, **kwargs) -> DetectionResult:
        """Mock detect method."""
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        
        n_samples = len(data) if data.ndim == 1 else data.shape[0]
        n_anomalies = max(1, int(n_samples * contamination))
        
        # Generate mock predictions (-1 for anomaly, 1 for normal)
        predictions = np.ones(n_samples)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        predictions[anomaly_indices] = -1
        
        # Generate mock scores (higher = more anomalous)
        scores = np.random.random(n_samples)
        scores[anomaly_indices] *= 2  # Make anomalies have higher scores
        
        return DetectionResult(
            algorithm=algorithm,
            predictions=predictions,
            scores=scores,
            total_samples=n_samples,
            anomaly_count=n_anomalies,
            processing_time=np.random.uniform(0.1, 2.0),
            success=True,
            metadata={'contamination': contamination, **kwargs}
        )
    
    def detect_anomalies(self, data: np.ndarray, algorithm: str = 'isolation_forest', 
                        contamination: float = 0.1, **kwargs) -> DetectionResult:
        """Alternative method name for compatibility."""
        return self.detect(data, algorithm, contamination, **kwargs)


class EnsembleService:
    """Mock EnsembleService class."""
    
    def __init__(self):
        self.methods = ['voting', 'averaging', 'stacking', 'weighted']
    
    def detect(self, data: np.ndarray, algorithms: List[str] = None, 
               method: str = 'voting', contamination: float = 0.1, **kwargs) -> DetectionResult:
        """Mock ensemble detect method."""
        if algorithms is None:
            algorithms = ['isolation_forest', 'lof', 'one_class_svm']
        
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        
        n_samples = len(data) if data.ndim == 1 else data.shape[0]
        n_anomalies = max(1, int(n_samples * contamination))
        
        # Generate mock predictions
        predictions = np.ones(n_samples)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        predictions[anomaly_indices] = -1
        
        # Generate mock scores
        scores = np.random.random(n_samples)
        scores[anomaly_indices] *= 2
        
        return DetectionResult(
            algorithm=f"ensemble_{method}",
            predictions=predictions,
            scores=scores,
            total_samples=n_samples,
            anomaly_count=n_anomalies,
            processing_time=np.random.uniform(0.5, 3.0),
            success=True,
            metadata={
                'algorithms': algorithms,
                'method': method,
                'contamination': contamination,
                **kwargs
            }
        )


class AnomalyPlotter:
    """Mock visualization class."""
    
    def __init__(self):
        pass
    
    def plot_anomaly_scores(self, data: np.ndarray, scores: np.ndarray, 
                           predictions: np.ndarray, title: str = "Anomaly Detection Results"):
        """Mock plotting method."""
        print(f"📊 Plotting: {title}")
        print(f"Data shape: {data.shape}")
        print(f"Anomalies detected: {np.sum(predictions == -1)}")
        print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        return None
    
    def plot_detection_results(self, result: DetectionResult, data: np.ndarray):
        """Mock result plotting method."""
        return self.plot_anomaly_scores(data, result.scores, result.predictions, 
                                       f"{result.algorithm} Results")


class StandardScaler:
    """Mock StandardScaler class."""
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.fitted = False
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Mock fit_transform method."""
        if isinstance(X, (list, tuple)):
            X = np.array(X)
        
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.fitted = True
        
        # Return standardized data
        return (X - self.mean_) / (self.scale_ + 1e-8)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Mock transform method."""
        if not self.fitted:
            raise ValueError("Must fit scaler first")
        return (X - self.mean_) / (self.scale_ + 1e-8)


class RobustScaler:
    """Mock RobustScaler class."""
    
    def __init__(self):
        self.median_ = None
        self.scale_ = None
        self.fitted = False
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Mock fit_transform method."""
        if isinstance(X, (list, tuple)):
            X = np.array(X)
        
        self.median_ = np.median(X, axis=0)
        self.scale_ = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        self.fitted = True
        
        return (X - self.median_) / (self.scale_ + 1e-8)


# Mock streaming classes
class StreamingService:
    """Mock streaming service."""
    
    def __init__(self):
        self.running = False
    
    def start_monitoring(self, source: str, **kwargs):
        """Mock start monitoring."""
        print(f"🔄 Started monitoring stream: {source}")
        self.running = True
    
    def stop_monitoring(self):
        """Mock stop monitoring."""
        print("⏹️  Stopped monitoring")
        self.running = False


# Mock model management classes
class ModelManager:
    """Mock model manager."""
    
    def __init__(self):
        self.models = {}
    
    def save_model(self, model, name: str, version: str = "1.0"):
        """Mock save model."""
        self.models[f"{name}:{version}"] = model
        print(f"💾 Saved model: {name}:{version}")
    
    def load_model(self, name: str, version: str = "1.0"):
        """Mock load model."""
        key = f"{name}:{version}"
        if key in self.models:
            print(f"📂 Loaded model: {name}:{version}")
            return self.models[key]
        else:
            # Return a mock service
            print(f"🔄 Creating new model: {name}:{version}")
            return DetectionService()


# Mock performance classes
@dataclass
class PerformanceMetrics:
    """Mock performance metrics."""
    total_detections: int = 0
    total_anomalies: int = 0
    average_detection_time: float = 0.0
    success_rate: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0


@dataclass
class AlgorithmStats:
    """Mock algorithm statistics."""
    algorithm: str
    detections_count: int = 0
    anomalies_found: int = 0
    average_score: float = 0.0
    average_time: float = 0.0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None


@dataclass
class DataQualityMetrics:
    """Mock data quality metrics."""
    total_samples: int = 0
    missing_values: int = 0
    duplicate_samples: int = 0
    outliers_count: int = 0
    data_drift_events: int = 0
    quality_score: float = 0.0


class AnalyticsService:
    """Mock analytics service."""
    
    def __init__(self):
        pass
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Mock performance metrics."""
        return PerformanceMetrics(
            total_detections=150,
            total_anomalies=23,
            average_detection_time=1.45,
            success_rate=96.7,
            throughput=10.2,
            error_rate=3.3
        )
    
    def get_dashboard_stats(self) -> dict:
        """Mock dashboard stats."""
        return {
            'total_detections': 150,
            'total_anomalies': 23,
            'active_algorithms': 3,
            'average_detection_time': 1.45,
            'system_status': 'healthy',
            'success_rate': 96.7
        }


# Package-level imports simulation
def get_version():
    """Mock version function."""
    return "2.1.0"


# Create mock modules for imports
class MockModule:
    """Mock module to handle dynamic imports."""
    
    def __init__(self, **items):
        for name, item in items.items():
            setattr(self, name, item)


# Mock preprocessing module
preprocessing = MockModule(
    StandardScaler=StandardScaler,
    RobustScaler=RobustScaler
)

# Mock visualization module  
visualization = MockModule(
    AnomalyPlotter=AnomalyPlotter
)

# Mock domain services
domain_services = MockModule(
    DetectionService=DetectionService,
    EnsembleService=EnsembleService,
    AnalyticsService=AnalyticsService
)

# Mock streaming
streaming = MockModule(
    StreamingService=StreamingService
)

# Mock model management
model_management = MockModule(
    ModelManager=ModelManager
)


# Function to generate realistic sample data
def generate_sample_data(n_samples: int = 1000, n_features: int = 2, 
                        anomaly_rate: float = 0.1, random_state: int = 42) -> np.ndarray:
    """Generate sample data with embedded anomalies."""
    np.random.seed(random_state)
    
    n_normal = int(n_samples * (1 - anomaly_rate))
    n_anomalies = n_samples - n_normal
    
    # Normal data
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal
    )
    
    # Anomalous data
    anomaly_data = np.random.multivariate_normal(
        mean=np.full(n_features, 3.0),
        cov=np.eye(n_features) * 0.5,
        size=n_anomalies
    )
    
    # Combine and shuffle
    data = np.vstack([normal_data, anomaly_data])
    indices = np.random.permutation(len(data))
    
    return data[indices]
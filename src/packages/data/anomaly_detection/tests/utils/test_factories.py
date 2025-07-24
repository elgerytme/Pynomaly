"""Test factories and utilities for generating test data and objects."""

import numpy as np
from datetime import datetime, timedelta

# Mock factory-boy if not available
try:
    import factory
except ImportError:
    # Simple mock for factory functionality
    class FactoryMeta(type):
        def __new__(cls, name, bases, attrs):
            return super().__new__(cls, name, bases, attrs)
    
    class Factory(metaclass=FactoryMeta):
        class Meta:
            model = None
        
        @classmethod
        def create(cls, **kwargs):
            if hasattr(cls.Meta, 'model') and cls.Meta.model:
                # Create a simple instance with default values
                return cls.Meta.model()
            return None
        
        @staticmethod
        def Faker(provider, **kwargs):
            # Simple faker replacement
            if provider == 'random_element':
                elements = kwargs.get('elements', [])
                return np.random.choice(elements) if elements else None
            elif provider == 'pyfloat':
                min_val = kwargs.get('min_value', 0.0)
                max_val = kwargs.get('max_value', 1.0)
                return np.random.uniform(min_val, max_val)
            elif provider == 'pyint':
                min_val = kwargs.get('min_value', 0)
                max_val = kwargs.get('max_value', 100)
                return np.random.randint(min_val, max_val + 1)
            elif provider == 'date_time_this_year':
                return datetime.now()
            elif provider == 'sentence':
                return "Test description"
            elif provider == 'word':
                return "test_dataset"
            return None
        
        @staticmethod
        def LazyFunction(func):
            return func()
        
        @staticmethod
        def LazyAttribute(func):
            return func
        
        @staticmethod
        def SubFactory(factory_class):
            return factory_class.create()
    
    factory = type('MockFactory', (), {
        'Factory': Factory,
        'Faker': Factory.Faker,
        'LazyFunction': Factory.LazyFunction,
        'LazyAttribute': Factory.LazyAttribute,
        'SubFactory': Factory.SubFactory
    })
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import uuid

from anomaly_detection.domain.entities.model import Model, ModelMetadata, ModelStatus
from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.domain.entities.anomaly import Anomaly
from anomaly_detection.domain.entities.dataset import Dataset


@dataclass
class TestDataSpecification:
    """Specification for generating test datasets."""
    n_samples: int = 1000
    n_features: int = 5
    n_anomalies: int = 50
    anomaly_factor: float = 3.0
    noise_level: float = 0.1
    random_seed: int = 42
    include_missing: bool = False
    missing_ratio: float = 0.05


class TestDataGenerator:
    """Comprehensive test data generator for various scenarios."""
    
    @staticmethod
    def create_synthetic_dataset(spec: TestDataSpecification) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic dataset with known anomalies.
        
        Args:
            spec: Data generation specification
            
        Returns:
            Tuple of (data, labels) where labels are 1=normal, -1=anomaly
        """
        np.random.seed(spec.random_seed)
        
        # Generate normal data
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(spec.n_features),
            cov=np.eye(spec.n_features),
            size=spec.n_samples - spec.n_anomalies
        )
        
        # Generate anomalous data
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(spec.n_features) * spec.anomaly_factor,
            cov=np.eye(spec.n_features) * 0.5,
            size=spec.n_anomalies
        )
        
        # Combine data
        all_data = np.vstack([normal_data, anomaly_data])
        labels = np.concatenate([
            np.ones(spec.n_samples - spec.n_anomalies),
            -np.ones(spec.n_anomalies)
        ])
        
        # Add noise
        if spec.noise_level > 0:
            noise = np.random.normal(0, spec.noise_level, all_data.shape)
            all_data += noise
        
        # Add missing values
        if spec.include_missing:
            n_missing = int(all_data.size * spec.missing_ratio)
            missing_indices = np.random.choice(all_data.size, n_missing, replace=False)
            flat_data = all_data.flatten()
            flat_data[missing_indices] = np.nan
            all_data = flat_data.reshape(all_data.shape)
        
        # Shuffle data
        indices = np.random.permutation(len(all_data))
        return all_data[indices], labels[indices]
    
    @staticmethod
    def create_financial_dataset(n_transactions: int = 1000, fraud_rate: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Create realistic financial transaction dataset.
        
        Args:
            n_transactions: Total number of transactions
            fraud_rate: Proportion of fraudulent transactions
            
        Returns:
            Tuple of (transactions, labels)
        """
        np.random.seed(42)
        n_fraud = int(n_transactions * fraud_rate)
        n_normal = n_transactions - n_fraud
        
        # Normal transactions: amount, merchant_category, time_of_day, location_risk, user_history
        normal_transactions = np.column_stack([
            np.random.lognormal(3, 1, n_normal),  # amount (mean ~$20)
            np.random.randint(1, 10, n_normal),   # merchant category
            np.random.normal(14, 4, n_normal),    # time of day (peak at 2 PM)
            np.random.beta(2, 8, n_normal),       # location risk (mostly low)
            np.random.gamma(2, 5, n_normal)       # user transaction history
        ])
        
        # Fraudulent transactions: higher amounts, unusual patterns
        fraud_transactions = np.column_stack([
            np.random.lognormal(5, 1.5, n_fraud),  # higher amounts
            np.random.randint(1, 10, n_fraud),      # merchant category
            np.random.normal(3, 2, n_fraud),        # unusual times (3 AM)
            np.random.beta(8, 2, n_fraud),          # high location risk
            np.random.gamma(1, 2, n_fraud)          # suspicious user history
        ])
        
        all_transactions = np.vstack([normal_transactions, fraud_transactions])
        labels = np.concatenate([np.ones(n_normal), -np.ones(n_fraud)])
        
        # Shuffle
        indices = np.random.permutation(len(all_transactions))
        return all_transactions[indices], labels[indices]
    
    @staticmethod
    def create_network_dataset(n_packets: int = 1000, intrusion_rate: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
        """Create network traffic dataset with intrusions.
        
        Args:
            n_packets: Total number of network packets
            intrusion_rate: Proportion of intrusion attempts
            
        Returns:
            Tuple of (packets, labels)
        """
        np.random.seed(123)
        n_intrusion = int(n_packets * intrusion_rate)
        n_normal = n_packets - n_intrusion
        
        # Normal traffic: packet_size, duration, src_port, dst_port, protocol
        normal_traffic = np.column_stack([
            np.random.normal(1500, 500, n_normal),    # packet size
            np.random.exponential(0.1, n_normal),     # duration
            np.random.randint(1024, 65535, n_normal), # source port
            np.random.choice([80, 443, 22, 21], n_normal), # common destination ports
            np.random.choice([6, 17], n_normal)       # TCP/UDP
        ])
        
        # Intrusion traffic: unusual patterns
        intrusion_traffic = np.column_stack([
            np.random.normal(8000, 2000, n_intrusion),  # large packets
            np.random.exponential(2.0, n_intrusion),    # longer duration
            np.random.randint(1, 1024, n_intrusion),    # privileged ports
            np.random.randint(8000, 9999, n_intrusion), # unusual ports
            np.random.choice([1, 47], n_intrusion)      # unusual protocols
        ])
        
        all_traffic = np.vstack([normal_traffic, intrusion_traffic])
        labels = np.concatenate([np.ones(n_normal), -np.ones(n_intrusion)])
        
        # Shuffle
        indices = np.random.permutation(len(all_traffic))
        return all_traffic[indices], labels[indices]
    
    @staticmethod
    def create_iot_sensor_dataset(n_readings: int = 1000, fault_rate: float = 0.08) -> Tuple[np.ndarray, np.ndarray]:
        """Create IoT sensor dataset with sensor faults.
        
        Args:
            n_readings: Total number of sensor readings
            fault_rate: Proportion of faulty readings
            
        Returns:
            Tuple of (readings, labels)
        """
        np.random.seed(456)
        n_faults = int(n_readings * fault_rate)
        n_normal = n_readings - n_faults
        
        # Normal sensor readings: temperature, humidity, pressure, light, vibration
        normal_readings = np.column_stack([
            np.random.normal(20, 2, n_normal),      # temperature (°C)
            np.random.normal(50, 10, n_normal),     # humidity (%)
            np.random.normal(1013, 5, n_normal),    # pressure (hPa)
            np.random.normal(300, 100, n_normal),   # light (lux)
            np.random.normal(10, 3, n_normal)       # vibration
        ])
        
        # Faulty sensor readings
        fault_readings = np.column_stack([
            np.random.choice([0, 50], n_faults),              # temperature sensor failure or overheating
            np.random.choice([0, 100], n_faults),             # humidity sensor failure or saturation
            np.random.normal(900, 50, n_faults),              # pressure sensor drift
            np.random.choice([0, 1000], n_faults),            # light sensor failure
            np.random.normal(100, 20, n_faults)               # excessive vibration
        ])
        
        all_readings = np.vstack([normal_readings, fault_readings])
        labels = np.concatenate([np.ones(n_normal), -np.ones(n_faults)])
        
        # Shuffle
        indices = np.random.permutation(len(all_readings))
        return all_readings[indices], labels[indices]
    
    @staticmethod
    def create_streaming_generator(spec: TestDataSpecification, batch_size: int = 1):
        """Create a streaming data generator.
        
        Args:
            spec: Data generation specification
            batch_size: Number of samples per batch
            
        Yields:
            Batches of streaming data
        """
        data, labels = TestDataGenerator.create_synthetic_dataset(spec)
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            yield batch_data, batch_labels


class ModelMetadataFactory(factory.Factory):
    """Factory for creating ModelMetadata instances."""
    
    class Meta:
        model = ModelMetadata
    
    algorithm = factory.Faker('random_element', elements=['iforest', 'lof', 'ocsvm'])
    contamination = factory.Faker('pyfloat', min_value=0.01, max_value=0.3)
    training_samples = factory.Faker('pyint', min_value=100, max_value=10000)
    feature_count = factory.Faker('pyint', min_value=2, max_value=50)
    performance_metrics = factory.LazyFunction(lambda: {
        'precision': np.random.uniform(0.5, 0.95),
        'recall': np.random.uniform(0.5, 0.95),
        'f1_score': np.random.uniform(0.5, 0.95)
    })
    created_at = factory.Faker('date_time_this_year')
    updated_at = factory.LazyAttribute(lambda obj: obj.created_at + timedelta(hours=1))
    version = factory.Faker('random_element', elements=['1.0', '1.1', '2.0'])
    status = factory.Faker('random_element', elements=list(ModelStatus))
    tags = factory.LazyFunction(lambda: ['test', 'synthetic'])
    description = factory.Faker('sentence')


class ModelFactory(factory.Factory):
    """Factory for creating Model instances."""
    
    class Meta:
        model = Model
    
    id = factory.LazyFunction(lambda: f"model_{uuid.uuid4().hex[:8]}")
    metadata = factory.SubFactory(ModelMetadataFactory)
    model_data = factory.LazyFunction(lambda: {'fitted': True, 'parameters': {}})


class DetectionResultFactory(factory.Factory):
    """Factory for creating DetectionResult instances."""
    
    class Meta:
        model = DetectionResult
    
    predictions = factory.LazyFunction(lambda: np.random.choice([-1, 1], size=100))
    confidence_scores = factory.LazyFunction(lambda: np.random.uniform(0, 1, size=100))
    algorithm = factory.Faker('random_element', elements=['iforest', 'lof', 'ocsvm'])
    metadata = factory.LazyFunction(lambda: {
        'contamination': 0.1,
        'processing_time_ms': np.random.uniform(10, 1000)
    })


class AnomalyFactory(factory.Factory):
    """Factory for creating Anomaly instances."""
    
    class Meta:
        model = Anomaly
    
    index = factory.Faker('pyint', min_value=0, max_value=9999)
    score = factory.Faker('pyfloat', min_value=0.0, max_value=1.0)
    feature_vector = factory.LazyFunction(lambda: np.random.rand(5).tolist())
    timestamp = factory.Faker('date_time_this_year')
    metadata = factory.LazyFunction(lambda: {
        'algorithm': 'iforest',
        'confidence': np.random.uniform(0.7, 1.0)
    })


class DatasetFactory(factory.Factory):
    """Factory for creating Dataset instances."""
    
    class Meta:
        model = Dataset
    
    name = factory.Faker('word')
    data = factory.LazyFunction(lambda: np.random.rand(100, 5))
    feature_names = factory.LazyFunction(lambda: [f'feature_{i}' for i in range(5)])
    metadata = factory.LazyFunction(lambda: {
        'source': 'synthetic',
        'created_at': datetime.utcnow().isoformat()
    })


class TestScenarioBuilder:
    """Builder for creating complex test scenarios."""
    
    def __init__(self):
        self.scenarios = {}
    
    def create_fraud_detection_scenario(self) -> Dict[str, Any]:
        """Create a fraud detection test scenario."""
        data, labels = TestDataGenerator.create_financial_dataset(
            n_transactions=1000, 
            fraud_rate=0.05
        )
        
        return {
            'name': 'fraud_detection',
            'data': data,
            'labels': labels,
            'expected_anomaly_rate': 0.05,
            'recommended_algorithms': ['iforest', 'lof'],
            'contamination': 0.05,
            'performance_thresholds': {
                'min_precision': 0.3,
                'min_recall': 0.3,
                'max_processing_time_ms': 5000
            }
        }
    
    def create_network_security_scenario(self) -> Dict[str, Any]:
        """Create a network security test scenario."""
        data, labels = TestDataGenerator.create_network_dataset(
            n_packets=800, 
            intrusion_rate=0.03
        )
        
        return {
            'name': 'network_security',
            'data': data,
            'labels': labels,
            'expected_anomaly_rate': 0.03,
            'recommended_algorithms': ['iforest', 'ocsvm'],
            'contamination': 0.03,
            'performance_thresholds': {
                'min_precision': 0.2,
                'min_recall': 0.2,
                'max_processing_time_ms': 3000
            }
        }
    
    def create_iot_monitoring_scenario(self) -> Dict[str, Any]:
        """Create an IoT monitoring test scenario."""
        data, labels = TestDataGenerator.create_iot_sensor_dataset(
            n_readings=1200, 
            fault_rate=0.08
        )
        
        return {
            'name': 'iot_monitoring',
            'data': data,
            'labels': labels,
            'expected_anomaly_rate': 0.08,
            'recommended_algorithms': ['lof', 'ocsvm'],
            'contamination': 0.08,
            'performance_thresholds': {
                'min_precision': 0.4,
                'min_recall': 0.4,
                'max_processing_time_ms': 2000
            }
        }
    
    def create_streaming_scenario(self, n_batches: int = 50) -> Dict[str, Any]:
        """Create a streaming data test scenario."""
        spec = TestDataSpecification(
            n_samples=n_batches * 10,
            n_features=4,
            n_anomalies=n_batches,
            anomaly_factor=2.5
        )
        
        return {
            'name': 'streaming_detection',
            'data_generator': TestDataGenerator.create_streaming_generator(spec, batch_size=10),
            'total_batches': n_batches,
            'batch_size': 10,
            'expected_anomaly_rate': 0.1,
            'recommended_algorithms': ['iforest'],
            'contamination': 0.1,
            'performance_thresholds': {
                'max_processing_time_per_batch_ms': 100,
                'max_memory_usage_mb': 500
            }
        }
    
    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        """Get all available test scenarios."""
        return [
            self.create_fraud_detection_scenario(),
            self.create_network_security_scenario(),
            self.create_iot_monitoring_scenario(),
            self.create_streaming_scenario()
        ]


class TestDataValidator:
    """Validator for test data quality and consistency."""
    
    @staticmethod
    def validate_dataset(data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Validate dataset quality.
        
        Args:
            data: Feature data
            labels: Target labels
            
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Basic shape validation
        if data.shape[0] != len(labels):
            results['is_valid'] = False
            results['issues'].append('Data and labels have different lengths')
        
        # Check for missing values
        if np.isnan(data).any():
            missing_count = np.isnan(data).sum()
            results['issues'].append(f'Dataset contains {missing_count} missing values')
        
        # Check for infinite values
        if np.isinf(data).any():
            inf_count = np.isinf(data).sum()
            results['is_valid'] = False
            results['issues'].append(f'Dataset contains {inf_count} infinite values')
        
        # Check label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))
        
        if len(unique_labels) < 2:
            results['is_valid'] = False
            results['issues'].append('Dataset contains only one class')
        
        # Calculate statistics
        results['statistics'] = {
            'n_samples': len(data),
            'n_features': data.shape[1] if data.ndim > 1 else 1,
            'label_distribution': label_distribution,
            'anomaly_rate': counts[unique_labels == -1][0] / len(labels) if -1 in unique_labels else 0,
            'feature_means': np.mean(data, axis=0).tolist(),
            'feature_stds': np.std(data, axis=0).tolist()
        }
        
        return results
    
    @staticmethod
    def validate_detection_result(result: DetectionResult, expected_samples: int) -> Dict[str, Any]:
        """Validate detection result quality.
        
        Args:
            result: Detection result to validate
            expected_samples: Expected number of samples
            
        Returns:
            Validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Basic validation
        if not result.success:
            validation['is_valid'] = False
            validation['issues'].append('Detection was not successful')
        
        if result.total_samples != expected_samples:
            validation['issues'].append(f'Expected {expected_samples} samples, got {result.total_samples}')
        
        if len(result.predictions) != expected_samples:
            validation['is_valid'] = False
            validation['issues'].append('Predictions length does not match expected samples')
        
        # Check prediction values
        unique_predictions = np.unique(result.predictions)
        if not all(pred in [-1, 1] for pred in unique_predictions):
            validation['is_valid'] = False
            validation['issues'].append('Predictions contain invalid values (should be -1 or 1)')
        
        # Check confidence scores if present
        if result.confidence_scores is not None:
            if len(result.confidence_scores) != expected_samples:
                validation['issues'].append('Confidence scores length does not match expected samples')
            
            if not all(0 <= score <= 1 for score in result.confidence_scores):
                validation['issues'].append('Confidence scores contain values outside [0, 1] range')
        
        # Calculate statistics
        validation['statistics'] = {
            'anomaly_count': result.anomaly_count,
            'anomaly_rate': result.anomaly_rate,
            'has_confidence_scores': result.confidence_scores is not None,
            'algorithm': result.algorithm,
            'prediction_distribution': dict(zip(*np.unique(result.predictions, return_counts=True)))
        }
        
        return validation


# Convenience functions for common test data
def create_simple_dataset(n_samples: int = 100, n_features: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple synthetic dataset for basic testing."""
    spec = TestDataSpecification(
        n_samples=n_samples,
        n_features=n_features,
        n_anomalies=int(n_samples * 0.1)
    )
    return TestDataGenerator.create_synthetic_dataset(spec)


def create_large_dataset(n_samples: int = 10000, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Create a large synthetic dataset for performance testing."""
    spec = TestDataSpecification(
        n_samples=n_samples,
        n_features=n_features,
        n_anomalies=int(n_samples * 0.05),
        random_seed=789
    )
    return TestDataGenerator.create_synthetic_dataset(spec)


def create_high_dimensional_dataset(n_samples: int = 500, n_features: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Create a high-dimensional dataset for algorithm testing."""
    spec = TestDataSpecification(
        n_samples=n_samples,
        n_features=n_features,
        n_anomalies=int(n_samples * 0.08),
        random_seed=321
    )
    return TestDataGenerator.create_synthetic_dataset(spec)
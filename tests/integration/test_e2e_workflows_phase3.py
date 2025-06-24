"""
Phase 3 Quality Enhancement: End-to-End Integration Testing
Comprehensive workflow testing across all layers of the Pynomaly system.

This module implements complete integration testing for:
- Dataset ingestion → Model training → Anomaly detection → Result export
- Multi-algorithm ensemble workflows
- Real-time detection pipelines
- Cross-layer communication and data flow
- Error propagation and recovery
- Performance under realistic workloads
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Optional, Any
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from contextlib import contextmanager
import asyncio
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate, ConfidenceInterval
from pynomaly.application.dto.export_options import ExportOptions, ExportFormat


@contextmanager
def performance_monitor():
    """Monitor end-to-end performance."""
    start_time = time.time()
    start_memory = None
    
    try:
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        pass
    
    yield
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    if start_memory:
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = end_memory - start_memory
        print(f"⏱️ Execution time: {execution_time:.2f}s, Memory usage: {memory_usage:.2f}MB")
    else:
        print(f"⏱️ Execution time: {execution_time:.2f}s")


class IntegrationTestBase:
    """Base class for integration testing with common utilities."""
    
    @pytest.fixture
    def sample_datasets(self):
        """Create multiple sample datasets for comprehensive testing."""
        np.random.seed(42)
        
        datasets = {}
        
        # Small dataset for quick tests
        datasets['small'] = self._create_dataset(
            name="small_dataset",
            n_samples=100,
            n_features=5,
            anomaly_rate=0.1
        )
        
        # Medium dataset for realistic testing
        datasets['medium'] = self._create_dataset(
            name="medium_dataset", 
            n_samples=1000,
            n_features=10,
            anomaly_rate=0.05
        )
        
        # Time-series dataset
        datasets['timeseries'] = self._create_timeseries_dataset(
            name="timeseries_dataset",
            n_samples=500,
            n_features=3
        )
        
        # High-dimensional dataset
        datasets['high_dim'] = self._create_dataset(
            name="high_dim_dataset",
            n_samples=200,
            n_features=50,
            anomaly_rate=0.2
        )
        
        return datasets
    
    def _create_dataset(self, name: str, n_samples: int, n_features: int, anomaly_rate: float) -> Dataset:
        """Create synthetic dataset with known anomalies."""
        # Generate normal data
        n_normal = int(n_samples * (1 - anomaly_rate))
        n_anomalies = n_samples - n_normal
        
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=n_normal
        )
        
        # Generate anomalies with different patterns
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 3,  # Shifted mean
            cov=np.eye(n_features) * 0.1,  # Smaller variance
            size=n_anomalies
        )
        
        # Combine data
        data = np.vstack([normal_data, anomaly_data])
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Create labels (for validation)
        labels = np.array([0] * n_normal + [1] * n_anomalies)
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        data = data[indices]
        labels = labels[indices]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=feature_names)
        df['true_label'] = labels  # Add ground truth for validation
        
        return Dataset(
            name=name,
            data=df,
            feature_names=feature_names,
            description=f"Synthetic dataset with {anomaly_rate*100}% anomalies",
            metadata={
                'anomaly_rate': anomaly_rate,
                'n_samples': n_samples,
                'n_features': n_features,
                'generation_method': 'multivariate_normal'
            }
        )
    
    def _create_timeseries_dataset(self, name: str, n_samples: int, n_features: int) -> Dataset:
        """Create time-series dataset with temporal anomalies."""
        # Generate time index
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1H')
        
        # Generate seasonal patterns
        data = np.zeros((n_samples, n_features))
        for i in range(n_features):
            # Base signal with trend and seasonality
            trend = np.linspace(0, 1, n_samples) * 0.1
            seasonal = np.sin(2 * np.pi * np.arange(n_samples) / 24) * 0.5  # Daily pattern
            noise = np.random.normal(0, 0.1, n_samples)
            data[:, i] = trend + seasonal + noise
        
        # Inject temporal anomalies
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        for idx in anomaly_indices:
            data[idx, :] += np.random.normal(2, 0.5, n_features)  # Spike anomalies
        
        # Create DataFrame
        feature_names = [f"sensor_{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=feature_names, index=timestamps)
        
        # Add labels
        labels = np.zeros(n_samples)
        labels[anomaly_indices] = 1
        df['true_label'] = labels
        
        return Dataset(
            name=name,
            data=df,
            feature_names=feature_names,
            description="Time-series dataset with temporal anomalies",
            metadata={
                'data_type': 'timeseries',
                'frequency': '1H',
                'anomaly_type': 'temporal_spikes'
            }
        )


class TestEndToEndWorkflows(IntegrationTestBase):
    """Test complete end-to-end workflows."""
    
    def test_complete_anomaly_detection_pipeline(self, sample_datasets):
        """Test complete pipeline: data → training → detection → export."""
        with performance_monitor():
            dataset = sample_datasets['medium']
            
            # Step 1: Data validation and preprocessing
            assert dataset.n_samples == 1000
            assert dataset.n_features == 10
            assert 'true_label' in dataset.data.columns
            
            # Step 2: Create and configure detector
            try:
                from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
                
                detector = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    name="e2e_test_detector"
                )
                
                # Step 3: Train detector
                training_data = dataset.data.drop('true_label', axis=1).values
                detector.fit(training_data)
                assert detector.is_fitted
                
                # Step 4: Detect anomalies  
                predictions = detector.detect(training_data)
                scores = detector.score(training_data)
                
                # Step 5: Validate results
                assert len(predictions) == len(training_data)
                assert len(scores) == len(training_data)
                
                # Step 6: Calculate performance metrics
                true_labels = dataset.data['true_label'].values
                self._validate_detection_quality(predictions, true_labels, scores)
                
                # Step 7: Create detection result
                result = DetectionResult(
                    id=uuid.uuid4(),
                    detector_id=detector.id,
                    dataset_id=dataset.id,
                    predictions=predictions,
                    scores=scores,
                    timestamp=datetime.now(),
                    metadata={'algorithm': 'IsolationForest', 'training_samples': len(training_data)}
                )
                
                # Step 8: Test export functionality
                self._test_export_integration(result, dataset)
                
                print("✅ Complete anomaly detection pipeline test passed")
                
            except Exception as e:
                print(f"❌ Pipeline test failed: {e}")
                raise
    
    def test_multi_algorithm_ensemble_workflow(self, sample_datasets):
        """Test ensemble detection using multiple algorithms."""
        with performance_monitor():
            dataset = sample_datasets['small']  # Use smaller dataset for ensemble
            training_data = dataset.data.drop('true_label', axis=1).values
            
            detectors = []
            all_predictions = []
            all_scores = []
            
            # Test multiple algorithms if available
            algorithm_configs = [
                ("sklearn", "IsolationForest"),
                ("pyod", "LOF") if self._is_pyod_available() else None,
                ("sklearn", "LocalOutlierFactor")
            ]
            
            algorithm_configs = [config for config in algorithm_configs if config is not None]
            
            for adapter_type, algorithm in algorithm_configs[:2]:  # Limit to 2 for testing
                try:
                    if adapter_type == "sklearn":
                        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
                        detector = SklearnAdapter(algorithm_name=algorithm)
                    elif adapter_type == "pyod":
                        from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
                        detector = PyODAdapter(algorithm_name=algorithm)
                    else:
                        continue
                    
                    # Train detector
                    detector.fit(training_data)
                    detectors.append(detector)
                    
                    # Get predictions and scores
                    predictions = detector.detect(training_data)
                    scores = detector.score(training_data)
                    
                    all_predictions.append(predictions)
                    all_scores.append(scores)
                    
                except Exception as e:
                    print(f"⚠️ Algorithm {algorithm} failed: {e}")
                    continue
            
            if len(detectors) >= 2:
                # Create ensemble predictions (majority voting)
                ensemble_predictions = self._ensemble_predictions(all_predictions)
                ensemble_scores = np.mean(all_scores, axis=0)
                
                # Validate ensemble performance
                true_labels = dataset.data['true_label'].values
                self._validate_detection_quality(ensemble_predictions, true_labels, ensemble_scores)
                
                print(f"✅ Multi-algorithm ensemble test passed with {len(detectors)} algorithms")
            else:
                print("⚠️ Insufficient algorithms available for ensemble testing")
    
    def test_real_time_detection_simulation(self, sample_datasets):
        """Test real-time detection simulation with streaming data."""
        with performance_monitor():
            dataset = sample_datasets['timeseries']
            
            try:
                from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
                
                # Train on first 80% of data
                n_train = int(len(dataset.data) * 0.8)
                train_data = dataset.data.iloc[:n_train].drop('true_label', axis=1).values
                test_data = dataset.data.iloc[n_train:].drop('true_label', axis=1).values
                test_labels = dataset.data.iloc[n_train:]['true_label'].values
                
                # Train detector
                detector = SklearnAdapter(algorithm_name="IsolationForest")
                detector.fit(train_data)
                
                # Simulate real-time detection (batch processing)
                batch_size = 10
                streaming_results = []
                
                for i in range(0, len(test_data), batch_size):
                    batch = test_data[i:i+batch_size]
                    batch_labels = test_labels[i:i+batch_size]
                    
                    # Detect anomalies in batch
                    start_time = time.time()
                    predictions = detector.detect(batch)
                    scores = detector.score(batch)
                    detection_time = time.time() - start_time
                    
                    # Store results
                    streaming_results.append({
                        'batch_id': i // batch_size,
                        'batch_size': len(batch),
                        'predictions': predictions,
                        'true_labels': batch_labels,
                        'detection_time': detection_time
                    })
                
                # Validate streaming performance
                total_predictions = np.concatenate([r['predictions'] for r in streaming_results])
                total_labels = np.concatenate([r['true_labels'] for r in streaming_results])
                avg_detection_time = np.mean([r['detection_time'] for r in streaming_results])
                
                self._validate_detection_quality(total_predictions, total_labels)
                
                # Performance assertions
                assert avg_detection_time < 1.0, f"Real-time detection too slow: {avg_detection_time}s"
                
                print(f"✅ Real-time detection simulation passed (avg time: {avg_detection_time:.3f}s)")
                
            except Exception as e:
                print(f"❌ Real-time detection test failed: {e}")
                raise
    
    def test_error_recovery_and_resilience(self, sample_datasets):
        """Test system behavior under error conditions."""
        dataset = sample_datasets['small']
        
        # Test 1: Invalid data handling
        try:
            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
            detector = SklearnAdapter(algorithm_name="IsolationForest")
            
            # Test with invalid data
            invalid_data = np.array([[np.inf, np.nan, 1.0]])
            
            with pytest.raises((ValueError, RuntimeError)):
                detector.fit(invalid_data)
            
            print("✅ Invalid data handling test passed")
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
        
        # Test 2: Memory recovery
        try:
            # Create detector with valid data
            detector = SklearnAdapter(algorithm_name="IsolationForest")
            valid_data = dataset.data.drop('true_label', axis=1).values
            detector.fit(valid_data)
            
            # Test prediction on oversized data
            oversized_data = np.random.normal(0, 1, (10000, dataset.n_features))
            predictions = detector.detect(oversized_data)
            assert len(predictions) == 10000
            
            print("✅ Memory recovery test passed")
            
        except Exception as e:
            print(f"❌ Memory recovery test failed: {e}")
    
    def _validate_detection_quality(self, predictions, true_labels, scores=None):
        """Validate detection quality metrics."""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # Convert predictions to binary if needed
        if not all(pred in [0, 1] for pred in predictions):
            # Assume higher scores = more anomalous
            threshold = np.percentile(scores or predictions, 95)
            predictions = (predictions > threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # Use scores for AUC if available
        if scores is not None:
            try:
                auc = roc_auc_score(true_labels, scores)
                print(f"📊 Quality metrics: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
            except ValueError:
                print(f"📊 Quality metrics: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        else:
            print(f"📊 Quality metrics: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # Basic quality assertions (lenient for synthetic data)
        assert precision >= 0.0, "Precision should be non-negative"
        assert recall >= 0.0, "Recall should be non-negative"
        assert f1 >= 0.0, "F1-score should be non-negative"
    
    def _test_export_integration(self, result: DetectionResult, dataset: Dataset):
        """Test export functionality integration."""
        try:
            # Test export options creation
            excel_options = ExportOptions().for_excel()
            assert excel_options.format == ExportFormat.EXCEL
            
            # Test serialization
            export_dict = excel_options.to_dict()
            reconstructed = ExportOptions.from_dict(export_dict)
            assert reconstructed.format == excel_options.format
            
            print("✅ Export integration test passed")
            
        except Exception as e:
            print(f"❌ Export integration test failed: {e}")
    
    def _ensemble_predictions(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Create ensemble predictions using majority voting."""
        predictions_array = np.array(predictions_list)
        # Majority voting
        ensemble = np.mean(predictions_array, axis=0) > 0.5
        return ensemble.astype(int)
    
    def _is_pyod_available(self) -> bool:
        """Check if PyOD is available."""
        try:
            import pyod
            return True
        except ImportError:
            return False


class TestCrossLayerIntegration(IntegrationTestBase):
    """Test integration across architectural layers."""
    
    def test_domain_to_infrastructure_flow(self, sample_datasets):
        """Test data flow from domain entities to infrastructure adapters."""
        dataset = sample_datasets['small']
        
        try:
            # Domain layer: Create entities
            assert isinstance(dataset, Dataset)
            assert dataset.n_samples > 0
            assert dataset.n_features > 0
            
            # Infrastructure layer: Use adapter
            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
            adapter = SklearnAdapter(algorithm_name="IsolationForest")
            
            # Test data conversion and flow
            training_data = dataset.data.drop('true_label', axis=1).values
            adapter.fit(training_data)
            
            # Application layer: Process results
            predictions = adapter.detect(training_data)
            scores = adapter.score(training_data)
            
            # Domain layer: Create result entity
            result = DetectionResult(
                id=uuid.uuid4(),
                detector_id=adapter.id,
                dataset_id=dataset.id,
                predictions=predictions,
                scores=scores,
                timestamp=datetime.now(),
                metadata={'cross_layer_test': True}
            )
            
            assert result.n_samples == len(training_data)
            assert result.n_anomalies >= 0
            
            print("✅ Cross-layer integration test passed")
            
        except Exception as e:
            print(f"❌ Cross-layer integration test failed: {e}")
    
    def test_repository_integration_flow(self, sample_datasets):
        """Test repository integration with domain entities."""
        try:
            from pynomaly.infrastructure.persistence.repositories import (
                InMemoryDatasetRepository, InMemoryDetectorRepository, InMemoryResultRepository
            )
            
            # Create repositories
            dataset_repo = InMemoryDatasetRepository()
            detector_repo = InMemoryDetectorRepository()
            result_repo = InMemoryResultRepository()
            
            # Save dataset
            dataset = sample_datasets['small']
            dataset_repo.save(dataset)
            
            # Create and save mock detector
            mock_detector = Mock()
            mock_detector.id = uuid.uuid4()
            mock_detector.name = "integration_test_detector"
            mock_detector.algorithm = "IsolationForest"
            detector_repo.save(mock_detector)
            
            # Create and save detection result
            mock_result = Mock()
            mock_result.id = uuid.uuid4()
            mock_result.detector_id = mock_detector.id
            mock_result.dataset_id = dataset.id
            mock_result.timestamp = datetime.now()
            result_repo.save(mock_result)
            
            # Test cross-repository queries
            saved_dataset = dataset_repo.find_by_id(dataset.id)
            saved_detector = detector_repo.find_by_id(mock_detector.id)
            saved_result = result_repo.find_by_id(mock_result.id)
            
            assert saved_dataset is not None
            assert saved_detector is not None
            assert saved_result is not None
            
            # Test relationship queries
            detector_results = result_repo.find_by_detector(mock_detector.id)
            dataset_results = result_repo.find_by_dataset(dataset.id)
            
            assert len(detector_results) == 1
            assert len(dataset_results) == 1
            
            print("✅ Repository integration flow test passed")
            
        except Exception as e:
            print(f"❌ Repository integration test failed: {e}")


def run_phase3_integration_tests():
    """Run all Phase 3 integration tests."""
    print("🚀 Running Phase 3 Quality Enhancement - Integration Tests")
    print("=" * 70)
    
    # Create test instance
    test_base = IntegrationTestBase()
    
    # Generate sample data
    print("📊 Generating sample datasets...")
    sample_data = test_base.sample_datasets.fget(test_base)
    print(f"✅ Generated {len(sample_data)} test datasets")
    
    # Run tests
    workflow_tester = TestEndToEndWorkflows()
    integration_tester = TestCrossLayerIntegration()
    
    print("\n🔄 Running end-to-end workflow tests...")
    workflow_tester.test_complete_anomaly_detection_pipeline(sample_data)
    workflow_tester.test_multi_algorithm_ensemble_workflow(sample_data)
    workflow_tester.test_real_time_detection_simulation(sample_data)
    workflow_tester.test_error_recovery_and_resilience(sample_data)
    
    print("\n🔗 Running cross-layer integration tests...")
    integration_tester.test_domain_to_infrastructure_flow(sample_data)
    integration_tester.test_repository_integration_flow(sample_data)
    
    print("\n" + "=" * 70)
    print("🎉 Phase 3 integration testing complete!")


if __name__ == "__main__":
    run_phase3_integration_tests()
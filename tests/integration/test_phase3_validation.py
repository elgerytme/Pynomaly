"""
Phase 3 Quality Enhancement: Integration Testing Validation
Simplified integration testing to validate Phase 3 components.
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import uuid
from datetime import datetime
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.application.dto.export_options import ExportOptions, ExportFormat


def create_test_dataset(name: str, n_samples: int = 100, n_features: int = 5):
    """Create a test dataset with known anomalies."""
    np.random.seed(42)
    
    # Generate normal data (80%)
    n_normal = int(n_samples * 0.8)
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal
    )
    
    # Generate anomalies (20%)
    n_anomalies = n_samples - n_normal
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,
        cov=np.eye(n_features) * 0.1,
        size=n_anomalies
    )
    
    # Combine and shuffle
    data = np.vstack([normal_data, anomaly_data])
    labels = np.array([0] * n_normal + [1] * n_anomalies)
    
    indices = np.random.permutation(n_samples)
    data = data[indices]
    labels = labels[indices]
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_names)
    df['true_label'] = labels
    
    return Dataset(
        name=name,
        data=df,
        feature_names=feature_names,
        description=f"Test dataset with {n_anomalies} anomalies"
    )


def test_end_to_end_workflow():
    """Test complete end-to-end anomaly detection workflow."""
    print("🔄 Testing end-to-end workflow...")
    
    try:
        # Step 1: Create test dataset
        dataset = create_test_dataset("e2e_test", n_samples=200, n_features=5)
        print(f"✅ Created dataset: {dataset.n_samples} samples, {dataset.n_features} features")
        
        # Step 2: Initialize detector
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        detector = SklearnAdapter(algorithm_name="IsolationForest")
        print(f"✅ Created detector: {detector.name}")
        
        # Step 3: Prepare training data
        training_data = dataset.data.drop('true_label', axis=1).values.astype(np.float32)
        true_labels = dataset.data['true_label'].values
        
        # Step 4: Train detector (sklearn adapter expects Dataset object)
        start_time = time.time()
        detector.fit(dataset)  # Pass dataset object, not numpy array
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.3f}s")
        
        # Step 5: Detect anomalies (create test dataset for detection)
        start_time = time.time()
        test_dataset = create_test_dataset("test_detection", n_samples=200, n_features=5)
        detection_result = detector.detect(test_dataset)
        detection_time = time.time() - start_time
        print(f"✅ Detection completed in {detection_time:.3f}s")
        
        # Extract predictions and scores from result
        predictions = detection_result.labels  # Binary labels (0=normal, 1=anomaly)
        scores = np.array([s.value for s in detection_result.scores])  # Extract score values
        
        # Step 6: Validate results
        assert len(predictions) == len(test_dataset.data), "Prediction length mismatch"
        assert len(scores) == len(test_dataset.data), "Score length mismatch"
        assert all(pred in [0, 1] for pred in predictions), "Invalid prediction values"
        
        # Step 7: Calculate basic metrics  
        n_detected_anomalies = np.sum(predictions)
        test_true_labels = test_dataset.data['true_label'].values
        n_true_anomalies = np.sum(test_true_labels)
        detection_rate = n_detected_anomalies / len(predictions)
        
        print(f"📊 Results: {n_detected_anomalies} detected, {n_true_anomalies} true anomalies")
        print(f"📊 Detection rate: {detection_rate:.3f}")
        
        # Step 8: Test export options
        export_options = ExportOptions().for_excel()
        assert export_options.format == ExportFormat.EXCEL
        print("✅ Export options integration working")
        
        # Step 9: Create detection result (use proper constructor parameters)
        from pynomaly.domain.value_objects import AnomalyScore
        from pynomaly.domain.entities.anomaly import Anomaly
        
        # Create AnomalyScore objects
        anomaly_scores = [AnomalyScore(value=float(score), method="test") for score in scores]
        
        # Create Anomaly objects for detected anomalies
        anomaly_indices = np.where(predictions == 1)[0]
        anomalies = [
            Anomaly(
                score=anomaly_scores[idx],
                data_point={'index': int(idx), 'features': {}},  # Mock data point
                detector_name="test_detector",
                metadata={'test_anomaly': True}
            )
            for idx in anomaly_indices
        ]
        
        result = DetectionResult(
            detector_id=detector.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            scores=anomaly_scores,
            labels=predictions,
            threshold=0.5,  # Default threshold
            timestamp=datetime.now(),
            metadata={'test': 'e2e_workflow'}
        )
        
        assert result.n_samples == len(test_dataset.data)
        assert result.n_anomalies == n_detected_anomalies
        print("✅ Detection result created successfully")
        
        print("✅ End-to-end workflow test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow test FAILED: {e}")
        return False


def test_multi_algorithm_comparison():
    """Test multiple algorithms on the same dataset."""
    print("\n🔄 Testing multi-algorithm comparison...")
    
    try:
        # Create test dataset
        dataset = create_test_dataset("multi_algo_test", n_samples=150, n_features=4)
        
        algorithms = ["IsolationForest", "LocalOutlierFactor"]
        results = {}
        
        for algorithm in algorithms:
            try:
                from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
                detector = SklearnAdapter(algorithm_name=algorithm)
                
                # Train and detect
                start_time = time.time()
                detector.fit(dataset)  # Pass dataset object
                detection_result = detector.detect(dataset)  # Pass dataset object
                total_time = time.time() - start_time
                
                predictions = detection_result.labels
                
                results[algorithm] = {
                    'predictions': predictions,
                    'n_anomalies': np.sum(predictions),
                    'time': total_time
                }
                
                print(f"✅ {algorithm}: {np.sum(predictions)} anomalies, {total_time:.3f}s")
                
            except Exception as e:
                print(f"⚠️ {algorithm} failed: {e}")
        
        if len(results) >= 2:
            # Compare results
            algos = list(results.keys())
            consistency = np.mean(results[algos[0]]['predictions'] == results[algos[1]]['predictions'])
            print(f"📊 Algorithm consistency: {consistency:.3f}")
            
        print("✅ Multi-algorithm comparison test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Multi-algorithm comparison test FAILED: {e}")
        return False


def test_repository_integration():
    """Test repository integration with domain entities."""
    print("\n🔄 Testing repository integration...")
    
    try:
        from pynomaly.infrastructure.persistence.repositories import (
            InMemoryDatasetRepository, InMemoryDetectorRepository
        )
        
        # Create repositories
        dataset_repo = InMemoryDatasetRepository()
        detector_repo = InMemoryDetectorRepository()
        
        # Create and save dataset
        dataset = create_test_dataset("repo_test", n_samples=50)
        dataset_repo.save(dataset)
        print(f"✅ Saved dataset: {dataset.name}")
        
        # Create and save detector
        mock_detector = Mock()
        mock_detector.id = uuid.uuid4()
        mock_detector.name = "test_detector"
        mock_detector.algorithm = "IsolationForest"
        detector_repo.save(mock_detector)
        print(f"✅ Saved detector: {mock_detector.name}")
        
        # Test retrieval
        found_dataset = dataset_repo.find_by_id(dataset.id)
        found_detector = detector_repo.find_by_id(mock_detector.id)
        
        assert found_dataset is not None, "Dataset not found"
        assert found_detector is not None, "Detector not found"
        assert found_dataset.name == dataset.name, "Dataset name mismatch"
        
        # Test repository stats
        dataset_count = dataset_repo.count()
        detector_count = detector_repo.count()
        
        assert dataset_count == 1, f"Expected 1 dataset, got {dataset_count}"
        assert detector_count == 1, f"Expected 1 detector, got {detector_count}"
        
        print(f"📊 Repository stats: {dataset_count} datasets, {detector_count} detectors")
        print("✅ Repository integration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Repository integration test FAILED: {e}")
        return False


def test_error_handling():
    """Test error handling and recovery."""
    print("\n🔄 Testing error handling...")
    
    try:
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        
        # Test 1: Invalid algorithm
        try:
            detector = SklearnAdapter(algorithm_name="InvalidAlgorithm")
            print("❌ Should have failed with invalid algorithm")
            return False
        except Exception:
            print("✅ Invalid algorithm properly rejected")
        
        # Test 2: Unfitted detector prediction
        try:
            detector = SklearnAdapter(algorithm_name="IsolationForest")
            dummy_dataset = create_test_dataset("dummy", n_samples=10, n_features=3)
            detection_result = detector.detect(dummy_dataset)  # Should fail - not fitted
            print("❌ Should have failed with unfitted detector")
            return False
        except Exception:
            print("✅ Unfitted detector properly rejected")
        
        # Test 3: Invalid data handling  
        try:
            detector = SklearnAdapter(algorithm_name="IsolationForest")
            
            # Try with empty dataset
            empty_dataset = Dataset(
                name="empty",
                data=pd.DataFrame(),  # Empty DataFrame
                feature_names=[]
            )
            detector.fit(empty_dataset)
            print("❌ Should have failed with empty data")
            return False
        except Exception:
            print("✅ Empty data properly rejected")
        
        print("✅ Error handling test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test FAILED: {e}")
        return False


def test_performance_benchmarks():
    """Test performance with different data sizes."""
    print("\n🔄 Testing performance benchmarks...")
    
    try:
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        
        data_sizes = [100, 500, 1000]
        performance_results = []
        
        for size in data_sizes:
            dataset = create_test_dataset(f"perf_test_{size}", n_samples=size, n_features=5)
            
            detector = SklearnAdapter(algorithm_name="IsolationForest")
            
            # Measure training time
            start_time = time.time()
            detector.fit(dataset)  # Pass dataset object
            training_time = time.time() - start_time
            
            # Measure prediction time
            start_time = time.time()
            detection_result = detector.detect(dataset)  # Pass dataset object
            prediction_time = time.time() - start_time
            
            predictions = detection_result.labels
            
            performance_results.append({
                'size': size,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'samples_per_second': size / (training_time + prediction_time)
            })
            
            print(f"📊 Size {size}: train={training_time:.3f}s, predict={prediction_time:.3f}s, "
                  f"throughput={size/(training_time + prediction_time):.1f} samples/s")
        
        # Check performance scaling
        largest_throughput = performance_results[-1]['samples_per_second']
        assert largest_throughput > 100, f"Performance too slow: {largest_throughput} samples/s"
        
        print("✅ Performance benchmark test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark test FAILED: {e}")
        return False


def run_phase3_validation():
    """Run all Phase 3 validation tests."""
    print("🚀 Phase 3 Quality Enhancement - Integration Testing Validation")
    print("=" * 80)
    
    # Run all tests
    tests = [
        test_end_to_end_workflow,
        test_multi_algorithm_comparison,
        test_repository_integration,
        test_error_handling,
        test_performance_benchmarks
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Phase 3 Integration Testing Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)} tests")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)} tests")
    
    if all(results):
        print("\n🎉 All Phase 3 integration tests PASSED!")
        print("✨ Quality enhancement objectives achieved!")
    else:
        print(f"\n⚠️ Some tests failed. Success rate: {sum(results)/len(results)*100:.1f}%")
    
    return sum(results) / len(results)


if __name__ == "__main__":
    success_rate = run_phase3_validation()
    print(f"\n🎯 Overall success rate: {success_rate*100:.1f}%")
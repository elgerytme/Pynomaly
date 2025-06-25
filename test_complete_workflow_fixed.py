#!/usr/bin/env python3
"""Test complete training and detection workflow with adapter registry."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd

def test_complete_workflow_with_adapters():
    """Test complete workflow with the new adapter registry system."""
    print("🔧 Testing Complete Workflow with Adapter Registry\n")
    
    try:
        # 1. Create and test adapter registry
        print("1. Setting up algorithm adapter registry...")
        from pynomaly.application.services.algorithm_adapter_registry import AlgorithmAdapterRegistry
        
        registry = AlgorithmAdapterRegistry()
        supported_algorithms = registry.get_supported_algorithms()
        print(f"✓ Adapter registry created with {len(supported_algorithms)} algorithms")
        print(f"   Sample algorithms: {', '.join(supported_algorithms[:5])}...")
        
        # Check if IsolationForest is supported
        if 'IsolationForest' in supported_algorithms:
            print("✓ IsolationForest algorithm supported")
        else:
            print("✗ IsolationForest not supported")
            return False
        
        # 2. Create test data
        print("\n2. Creating synthetic test data...")
        np.random.seed(42)
        
        # Normal data (80 samples)
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0],
            cov=np.eye(3),
            size=80
        )
        
        # Anomalous data (20 samples)
        anomaly_data = np.random.multivariate_normal(
            mean=[3, 3, 3],
            cov=np.eye(3) * 0.5,
            size=20
        )
        
        # Combine and shuffle
        all_data = np.vstack([normal_data, anomaly_data])
        np.random.shuffle(all_data)
        
        df = pd.DataFrame(all_data, columns=['feature_1', 'feature_2', 'feature_3'])
        
        # Create dataset entity
        from pynomaly.domain.entities import Dataset
        test_dataset = Dataset(
            name='test_workflow_data',
            data=df,
            description='Test data for complete workflow'
        )
        
        print(f"✓ Test dataset created: {test_dataset.n_samples} samples x {test_dataset.n_features} features")
        
        # 3. Create detector
        print("\n3. Creating anomaly detector...")
        from pynomaly.domain.entities import Detector
        
        detector = Detector(
            name='workflow_test_detector',
            algorithm_name='IsolationForest',
            parameters={'contamination': 0.2, 'random_state': 42}
        )
        
        print(f"✓ Detector created: {detector.name}")
        print(f"   Algorithm: {detector.algorithm_name}")
        print(f"   Parameters: {detector.parameters}")
        
        # 4. Test training with adapter registry
        print("\n4. Testing training workflow...")
        
        print("   Fitting detector with adapter registry...")
        try:
            registry.fit_detector(detector, test_dataset)
            print("   ✓ Training completed successfully!")
            
            # Update detector status (this would normally be done by the use case)
            detector.is_fitted = True
            detector.trained_at = pd.Timestamp.now()
            
        except Exception as e:
            print(f"   ✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. Test detection with adapter registry
        print("\n5. Testing detection workflow...")
        
        print("   Running detection with adapter registry...")
        try:
            # Get scores and predictions
            scores = registry.score_with_detector(detector, test_dataset)
            predictions = registry.predict_with_detector(detector, test_dataset)
            
            print(f"   ✓ Detection completed successfully!")
            print(f"     Samples processed: {len(scores)}")
            print(f"     Anomalies detected: {sum(predictions)}")
            print(f"     Anomaly rate: {sum(predictions) / len(predictions):.2%}")
            
            # Show score statistics
            score_values = [s.value for s in scores]
            print(f"     Score range: {min(score_values):.3f} to {max(score_values):.3f}")
            print(f"     Mean score: {np.mean(score_values):.3f}")
            
        except Exception as e:
            print(f"   ✗ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 6. Test use case integration
        print("\n6. Testing use case integration...")
        
        try:
            # Create repositories (in-memory for testing)
            from pynomaly.infrastructure.repositories import InMemoryDetectorRepository, InMemoryDatasetRepository
            from pynomaly.domain.services import FeatureValidator
            
            detector_repo = InMemoryDetectorRepository()
            dataset_repo = InMemoryDatasetRepository()
            feature_validator = FeatureValidator()
            
            # Save entities
            detector_repo.save(detector)
            dataset_repo.save(test_dataset)
            
            print("   ✓ Repositories and services created")
            
            # Test training use case
            from pynomaly.application.use_cases import TrainDetectorUseCase, TrainDetectorRequest
            
            train_use_case = TrainDetectorUseCase(
                detector_repository=detector_repo,
                feature_validator=feature_validator,
                adapter_registry=registry
            )
            
            # Create fresh detector for training test
            fresh_detector = Detector(
                name='fresh_test_detector',
                algorithm_name='IsolationForest',
                parameters={'contamination': 0.15, 'random_state': 123}
            )
            detector_repo.save(fresh_detector)
            
            train_request = TrainDetectorRequest(
                detector_id=fresh_detector.id,
                training_data=test_dataset,
                validate_data=True,
                save_model=True
            )
            
            print("   Testing training use case...")
            import asyncio
            train_response = asyncio.run(train_use_case.execute(train_request))
            
            print("   ✓ Training use case completed!")
            print(f"     Training time: {train_response.training_time_ms:.2f}ms")
            print(f"     Detector fitted: {train_response.trained_detector.is_fitted}")
            
            # Test detection use case
            from pynomaly.application.use_cases import DetectAnomaliesUseCase, DetectAnomaliesRequest
            
            detect_use_case = DetectAnomaliesUseCase(
                detector_repository=detector_repo,
                feature_validator=feature_validator,
                adapter_registry=registry
            )
            
            detect_request = DetectAnomaliesRequest(
                detector_id=train_response.trained_detector.id,
                dataset=test_dataset,
                validate_features=True,
                save_results=True
            )
            
            print("   Testing detection use case...")
            detect_response = asyncio.run(detect_use_case.execute(detect_request))
            
            print("   ✓ Detection use case completed!")
            result = detect_response.result
            print(f"     Anomalies found: {result.n_anomalies}")
            print(f"     Anomaly rate: {result.anomaly_rate:.2%}")
            
            exec_time = result.execution_time_ms or result.metadata.get('execution_time_ms', 0)
            print(f"     Execution time: {exec_time:.2f}ms")
            
            if detect_response.warnings:
                print("   ⚠ Warnings:")
                for warning in detect_response.warnings:
                    print(f"     - {warning}")
            
        except Exception as e:
            print(f"   ✗ Use case integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n✅ Complete workflow test successful!")
        print("\n📋 Workflow Summary:")
        print("   ✓ Adapter registry system working")
        print("   ✓ Algorithm instantiation working")
        print("   ✓ Training workflow functional")
        print("   ✓ Detection workflow functional")
        print("   ✓ Use case integration working")
        print("   ✓ End-to-end pipeline operational")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_workflow_with_adapters()
    print(f"\nResult: {'🎉 SUCCESS' if success else '💥 FAILED'}")
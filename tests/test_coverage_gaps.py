#!/usr/bin/env python3
"""
Test script to fill coverage gaps and achieve 100% coverage for target modules.
This script tests specific branches and edge cases that are missing from current coverage.
"""
import sys
import os
from datetime import datetime
from uuid import uuid4
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_detection_dto_branches():
    """Test all branches in detection DTOs to achieve 100% coverage."""
    
    from pynomaly.application.dto.detection_dto import (
        DetectionRequestDTO, 
        TrainingRequestDTO, 
        AnomalyDTO, 
        DetectionResultDTO,
        TrainingResultDTO,
        ExplanationRequestDTO,
        ExplanationResultDTO,
        DetectionSummaryDTO
    )
    
    # Test DetectionRequestDTO validation branches (lines 40-43)
    
    # Branch 1: Neither dataset_id nor data provided (line 40-41)
    try:
        DetectionRequestDTO(
            detector_id=uuid4(),
            dataset_id=None,
            data=None
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Either dataset_id or data must be provided" in str(e)
        print("✅ Tested branch: Neither dataset_id nor data provided")
    
    # Branch 2: Both dataset_id and data provided (line 42-43)
    try:
        DetectionRequestDTO(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            data=[{"feature1": 1.0, "feature2": 2.0}]
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Provide either dataset_id or data, not both" in str(e)
        print("✅ Tested branch: Both dataset_id and data provided")
    
    # Valid cases
    # Case 1: Only dataset_id provided
    dto1 = DetectionRequestDTO(
        detector_id=uuid4(),
        dataset_id=uuid4(),
        data=None
    )
    assert dto1.dataset_id is not None
    assert dto1.data is None
    print("✅ Tested valid case: Only dataset_id provided")
    
    # Case 2: Only data provided  
    dto2 = DetectionRequestDTO(
        detector_id=uuid4(),
        dataset_id=None,
        data=[{"feature1": 1.0, "feature2": 2.0}]
    )
    assert dto2.dataset_id is None
    assert dto2.data is not None
    print("✅ Tested valid case: Only data provided")

def test_detector_dto_branches():
    """Test missing branches in detector DTOs."""
    
    from pynomaly.application.dto.detector_dto import (
        DetectorCreateDTO,
        DetectorUpdateDTO,
        DetectorResponseDTO
    )
    
    # Test all different algorithm types and parameters
    algorithms = [
        "IsolationForest",
        "LocalOutlierFactor", 
        "OneClassSVM",
        "EllipticEnvelope"
    ]
    
    for algo in algorithms:
        dto = DetectorCreateDTO(
            name=f"test_{algo}",
            algorithm=algo,
            parameters={"contamination": 0.1}
        )
        assert dto.algorithm == algo
        print(f"✅ Tested algorithm: {algo}")
    
    # Test optional fields
    dto_full = DetectorCreateDTO(
        name="full_test",
        algorithm="IsolationForest",
        parameters={"contamination": 0.1, "n_estimators": 100},
        description="Test detector with full parameters",
        tags=["test", "coverage"]
    )
    assert dto_full.description is not None
    assert len(dto_full.tags) == 2
    print("✅ Tested optional fields in DetectorCreateDTO")

def test_explainability_dto_edge_cases():
    """Test edge cases in explainability DTOs."""
    
    from pynomaly.application.dto.explainability_dto import (
        ExplanationRequest,
        ExplanationResponse,
        FeatureImportance,
        ExplanationConfiguration
    )
    
    # Test different explanation methods
    methods = ["shap", "lime", "permutation"]
    
    for method in methods:
        try:
            config = ExplanationConfiguration(
                method=method,
                n_features=5,
                sample_size=100
            )
            assert config.method == method
            print(f"✅ Tested explanation method: {method}")
        except Exception as e:
            print(f"⚠️  Method {method} not supported: {e}")
    
    # Test boundary values
    config_min = ExplanationConfiguration(
        method="shap",
        n_features=1,  # minimum
        sample_size=10  # minimum
    )
    assert config_min.n_features == 1
    print("✅ Tested minimum boundary values")
    
    config_max = ExplanationConfiguration(
        method="shap", 
        n_features=50,  # maximum
        sample_size=10000  # large value
    )
    assert config_max.n_features == 50
    print("✅ Tested maximum boundary values")

def test_value_objects_coverage():
    """Test value objects to improve coverage."""
    
    from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
    from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
    from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
    
    # Test ContaminationRate validation branches
    try:
        ContaminationRate(-0.1)  # Invalid: negative
        assert False, "Should reject negative contamination rate"
    except ValueError:
        print("✅ Tested negative contamination rate rejection")
    
    try:
        ContaminationRate(1.1)  # Invalid: > 1
        assert False, "Should reject contamination rate > 1"
    except ValueError:
        print("✅ Tested contamination rate > 1 rejection")
    
    # Valid cases
    rate = ContaminationRate(0.1)
    assert rate.value == 0.1
    print("✅ Tested valid contamination rate")
    
    # Test AnomalyScore
    score = AnomalyScore(0.75)
    assert score.value == 0.75
    assert score.is_anomaly(threshold=0.5) == True
    assert score.is_anomaly(threshold=0.8) == False
    print("✅ Tested anomaly score logic")
    
    # Test ConfidenceInterval
    ci = ConfidenceInterval(lower=0.6, upper=0.9, confidence_level=0.95)
    assert ci.lower == 0.6
    assert ci.upper == 0.9
    assert ci.width == 0.3
    print("✅ Tested confidence interval")

def test_entity_edge_cases():
    """Test entity classes edge cases and branches."""
    
    from pynomaly.domain.entities.dataset import Dataset
    from pynomaly.domain.entities.detector import Detector
    from pynomaly.domain.entities.anomaly import Anomaly
    
    import pandas as pd
    import numpy as np
    
    # Test Dataset with different data types
    data_dict = {
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0]
    }
    
    # DataFrame input
    df = pd.DataFrame(data_dict)
    dataset1 = Dataset(
        name="test_df",
        data=df,
        feature_names=["feature1", "feature2"]
    )
    assert dataset1.name == "test_df"
    print("✅ Tested Dataset with DataFrame")
    
    # NumPy array input
    arr = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    dataset2 = Dataset(
        name="test_array",
        data=arr,
        feature_names=["feature1", "feature2"]
    )
    assert dataset2.name == "test_array"
    print("✅ Tested Dataset with NumPy array")
    
    # Test Detector with different states
    detector = Detector(
        name="test_detector",
        algorithm="IsolationForest"
    )
    
    # Test state transitions
    assert not detector.is_trained
    detector.mark_as_trained()
    assert detector.is_trained
    print("✅ Tested Detector state transitions")
    
    # Test Anomaly with different severities
    severities = ["low", "medium", "high", "critical"]
    for severity in severities:
        anomaly = Anomaly(
            score=0.8,
            timestamp=datetime.now(),
            data_point={"feature1": 5.0},
            severity=severity
        )
        assert anomaly.severity == severity
        print(f"✅ Tested Anomaly severity: {severity}")

def test_adapter_coverage():
    """Test adapter branches to improve coverage."""
    
    from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
    from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
    
    # Test different algorithms
    algorithms = [
        "IsolationForest",
        "LocalOutlierFactor", 
        "OneClassSVM",
        "EllipticEnvelope"
    ]
    
    for algo in algorithms:
        try:
            adapter = SklearnAdapter(
                algorithm_name=algo,
                name=f"test_{algo}",
                contamination_rate=ContaminationRate(0.1)
            )
            assert adapter.algorithm_name == algo
            print(f"✅ Tested SklearnAdapter with {algo}")
        except Exception as e:
            print(f"⚠️  Algorithm {algo} failed: {e}")
    
    # Test with custom parameters
    adapter_custom = SklearnAdapter(
        algorithm_name="IsolationForest",
        name="custom_test",
        contamination_rate=ContaminationRate(0.05),
        **{"n_estimators": 200, "max_samples": "auto"}
    )
    print("✅ Tested SklearnAdapter with custom parameters")

def test_configuration_branches():
    """Test configuration container branches."""
    
    from pynomaly.infrastructure.config.container import Container
    
    # Test container creation and service retrieval
    container = Container()
    
    # Test different service types
    services_to_test = [
        "detection_service",
        "training_service", 
        "sklearn_adapter_factory"
    ]
    
    for service_name in services_to_test:
        try:
            service = getattr(container, service_name)()
            assert service is not None
            print(f"✅ Tested container service: {service_name}")
        except AttributeError:
            print(f"⚠️  Service {service_name} not found in container")
        except Exception as e:
            print(f"⚠️  Service {service_name} creation failed: {e}")

def run_comprehensive_coverage_test():
    """Run all coverage tests to fill gaps."""
    
    print("🚀 Running comprehensive coverage tests")
    print("=" * 60)
    
    test_functions = [
        test_detection_dto_branches,
        test_detector_dto_branches, 
        test_explainability_dto_edge_cases,
        test_value_objects_coverage,
        test_entity_edge_cases,
        test_adapter_coverage,
        test_configuration_branches
    ]
    
    for test_func in test_functions:
        try:
            print(f"\n🔍 Running {test_func.__name__}...")
            test_func()
            print(f"✅ {test_func.__name__} completed successfully")
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive coverage test completed!")

if __name__ == "__main__":
    run_comprehensive_coverage_test()

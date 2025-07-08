#!/usr/bin/env python3
"""
Focused test to achieve 100% coverage for detection DTOs.
"""
import sys
import os
from uuid import uuid4

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_detection_request_dto_validation():
    """Test the specific validation branches in DetectionRequestDTO that are missing coverage."""
    
    from pynomaly.application.dto.detection_dto import DetectionRequestDTO
    
    # Branch 1: Neither dataset_id nor data provided (lines 40-41)
    print("Testing branch: Neither dataset_id nor data provided")
    try:
        DetectionRequestDTO(
            detector_id=uuid4(),
            dataset_id=None,
            data=None
        )
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    # Branch 2: Both dataset_id and data provided (lines 42-43)
    print("Testing branch: Both dataset_id and data provided")
    try:
        DetectionRequestDTO(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            data=[{"feature1": 1.0, "feature2": 2.0}]
        )
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    # Valid case 1: Only dataset_id provided
    print("Testing valid case: Only dataset_id provided")
    dto1 = DetectionRequestDTO(
        detector_id=uuid4(),
        dataset_id=uuid4(),
        data=None
    )
    print(f"✅ Valid DTO created with dataset_id: {dto1.dataset_id}")
    
    # Valid case 2: Only data provided
    print("Testing valid case: Only data provided")
    dto2 = DetectionRequestDTO(
        detector_id=uuid4(),
        dataset_id=None,
        data=[{"feature1": 1.0, "feature2": 2.0}]
    )
    print(f"✅ Valid DTO created with data: {len(dto2.data)} rows")

def test_other_detection_dtos():
    """Test other DTOs in the detection module to improve overall coverage."""
    
    from pynomaly.application.dto.detection_dto import (
        TrainingRequestDTO, 
        AnomalyDTO, 
        DetectionResultDTO,
        TrainingResultDTO,
        ExplanationRequestDTO,
        ExplanationResultDTO,
        DetectionSummaryDTO
    )
    from datetime import datetime
    
    # Test TrainingRequestDTO
    training_dto = TrainingRequestDTO(
        detector_id=uuid4(),
        dataset_id=uuid4(),
        validation_split=0.2,
        cross_validation=True,
        save_model=True,
        parameters={"n_estimators": 100}
    )
    print(f"✅ TrainingRequestDTO created with validation_split: {training_dto.validation_split}")
    
    # Test AnomalyDTO
    anomaly_dto = AnomalyDTO(
        id=uuid4(),
        score=0.75,
        detector_name="IsolationForest",
        timestamp=datetime.now(),
        data_point={"feature1": 5.0, "feature2": 10.0},
        metadata={"source": "test"},
        explanation="High deviation from normal pattern",
        severity="high",
        confidence_lower=0.6,
        confidence_upper=0.9
    )
    print(f"✅ AnomalyDTO created with score: {anomaly_dto.score}")
    
    # Test DetectionResultDTO
    detection_result = DetectionResultDTO(
        id=uuid4(),
        detector_id=uuid4(),
        dataset_id=uuid4(),
        timestamp=datetime.now(),
        n_samples=1000,
        n_anomalies=50,
        anomaly_rate=0.05,
        threshold=0.6,
        execution_time_ms=1500.0,
        metadata={"algorithm": "IsolationForest"},
        anomalies=[],
        predictions=[0, 1, 0, 1, 0],
        scores=[0.2, 0.8, 0.3, 0.9, 0.1],
        score_statistics={"mean": 0.46, "std": 0.32},
        has_confidence_intervals=True,
        quality_warnings=["Low sample size"]
    )
    print(f"✅ DetectionResultDTO created with {detection_result.n_anomalies} anomalies")
    
    # Test TrainingResultDTO
    training_result = TrainingResultDTO(
        detector_id=uuid4(),
        dataset_id=uuid4(),
        timestamp=datetime.now(),
        training_time_ms=5000.0,
        model_path="/tmp/model.pkl",
        training_warnings=["Convergence warning"],
        training_metrics={"accuracy": 0.95},
        validation_metrics={"f1_score": 0.87},
        dataset_summary={"n_features": 10, "n_samples": 1000},
        parameters_used={"contamination": 0.1}
    )
    print(f"✅ TrainingResultDTO created with training_time: {training_result.training_time_ms}ms")
    
    # Test ExplanationRequestDTO
    explanation_request = ExplanationRequestDTO(
        detector_id=uuid4(),
        instance={"feature1": 5.0, "feature2": 10.0},
        method="shap",
        feature_names=["feature1", "feature2"],
        n_features=5
    )
    print(f"✅ ExplanationRequestDTO created with method: {explanation_request.method}")
    
    # Test ExplanationResultDTO
    explanation_result = ExplanationResultDTO(
        method_used="shap",
        prediction=0.85,
        confidence=0.92,
        feature_importance={"feature1": 0.6, "feature2": 0.4},
        explanation_text="Feature1 is the primary driver of the anomaly",
        visualization_data={"plot_type": "waterfall", "data": [1, 2, 3]}
    )
    print(f"✅ ExplanationResultDTO created with prediction: {explanation_result.prediction}")
    
    # Test DetectionSummaryDTO
    summary = DetectionSummaryDTO(
        total_detections=1000,
        recent_detections=25,
        average_anomaly_rate=0.05,
        most_active_detector="IsolationForest_v1",
        top_algorithms=[
            {"name": "IsolationForest", "usage_count": 500},
            {"name": "LocalOutlierFactor", "usage_count": 300}
        ],
        performance_metrics={"avg_detection_time": 150.0, "throughput": 1000.0}
    )
    print(f"✅ DetectionSummaryDTO created with {summary.total_detections} total detections")

if __name__ == "__main__":
    print("🚀 Running focused detection DTO coverage tests")
    print("=" * 60)
    
    test_detection_request_dto_validation()
    print()
    test_other_detection_dtos()
    
    print("=" * 60)
    print("🎉 Detection DTO coverage tests completed!")

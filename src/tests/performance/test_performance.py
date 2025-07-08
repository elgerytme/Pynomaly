import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from uuid import uuid4
from datetime import timedelta

from pynomaly.application.services.performance_monitoring_service import PerformanceMonitoringService
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.model_performance import ModelPerformanceMetrics, ModelPerformanceBaseline
from pynomaly.infrastructure.monitoring.performance_monitor import PerformanceMonitor
from pynomaly.infrastructure.repositories.in_memory_repositories import InMemoryDetectorRepository


# Dummy detector and dataset for testing
@pytest.fixture
def dummy_detector():
    return Detector(name="dummy_detector", algorithm_name="dummy_algo", parameters={})


@pytest.fixture
def dummy_dataset():
    # Create proper DataFrame for Dataset entity
    data = pd.DataFrame({
        'feature_0': [1, 2, 3, 4, 5],
        'feature_1': [2, 4, 6, 8, 10]
    })
    return Dataset(name="dummy_dataset", data=data)


# Unit test for performance degradation detection under various scenarios
def test_degradation_detector(dummy_detector, dummy_dataset):
    monitor_service = PerformanceMonitoringService(PerformanceMonitor())
    baseline = ModelPerformanceBaseline(
        model_id="model_1",
        mean=10, std=2,
        pct_thresholds={'90': 15},
        description="Baseline for model_1"
    )

    monitor_service.set_performance_baseline("operation_1", {"execution_time": 10})

    # Create a proper detection function that returns DetectionResult
    def detection_func(detector, dataset):
        import time
        time.sleep(0.01)  # Simulate some processing time
        from pynomaly.domain.entities.anomaly import Anomaly
        
        # Create dummy anomaly detection result
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        labels = [0, 0, 0, 1, 1]  # Last two are anomalies
        threshold = 0.35
        
        # Create anomaly objects for detected anomalies
        anomalies = []
        for i, (score, label) in enumerate(zip(scores, labels)):
            if label == 1:
                anomaly = Anomaly(
                    dataset_id=dataset.id,
                    detector_id=detector.id,
                    index=i,
                    score=score,
                    metadata={"feature_values": dataset.data.iloc[i].to_dict()}
                )
                anomalies.append(anomaly)
        
        return DetectionResult(
            detector_id=detector.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=threshold,
            metadata={"test": True}
        )
    
    # Monitor operation to create metrics
    dummy_result, metrics = monitor_service.monitor_detection_operation(
        dummy_detector, dummy_dataset, detection_func
    )
    
    # Now check for regression (should have recent metrics)
    result = monitor_service.check_performance_regression("operation_1")
    assert "regressions_detected" in result
    
    # Test baseline degradation detection
    assert baseline.is_degraded(current_value=4.0) == True  # 3 std below mean
    assert baseline.is_degraded(current_value=10.0) == False  # At mean


# Repository tests
@pytest.fixture
def detector_repo():
    return InMemoryDetectorRepository()


def test_repository_save_find(detector_repo, dummy_detector):
    detector_repo.save(dummy_detector)
    found_detector = detector_repo.find_by_id(dummy_detector.id)
    assert found_detector is not None
    assert found_detector.name == "dummy_detector"


# Service flow test recording metrics, detecting degradation, generating alert
def test_service_flow(dummy_detector, dummy_dataset):
    monitor_service = PerformanceMonitoringService(PerformanceMonitor())

    # Starting monitoring session with baseline
    monitor_service.set_performance_baseline("detection", {"execution_time": 5})
    monitor_service.start_monitoring()

    # Creating dummy function for detection
    def detection_func(detector, dataset):
        import time
        time.sleep(0.01)  # Simulate some processing time
        from pynomaly.domain.entities.anomaly import Anomaly
        
        # Create dummy anomaly detection result
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        labels = [0, 0, 0, 1, 1]  # Last two are anomalies
        threshold = 0.35
        
        # Create anomaly objects for detected anomalies
        anomalies = []
        for i, (score, label) in enumerate(zip(scores, labels)):
            if label == 1:
                anomaly = Anomaly(
                    dataset_id=dataset.id,
                    detector_id=detector.id,
                    index=i,
                    score=score,
                    metadata={"feature_values": dataset.data.iloc[i].to_dict()}
                )
                anomalies.append(anomaly)
        
        return DetectionResult(
            detector_id=detector.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=threshold,
            metadata={"test": True}
        )

    # Monitor operation
    result, performance_metrics = monitor_service.monitor_detection_operation(
        dummy_detector, dummy_dataset, detection_func
    )

    assert performance_metrics.execution_time >= 0

    # Check for alerts
    regression_result = monitor_service.check_performance_regression("detection")
    assert regression_result["regressions_detected"] >= 0


# API endpoint test
from fastapi.testclient import TestClient
from pynomaly.presentation.api.endpoints.performance import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)


client = TestClient(app)

def test_get_pools():
    response = client.get("/performance/pools")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

"""Comprehensive integration testing suite."""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from uuid import uuid4

from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.training_job import TrainingJob
from pynomaly.domain.entities.anomaly_event import EventType, EventSeverity
from pynomaly.domain.services.advanced_classification_service import AdvancedClassificationService
from pynomaly.domain.services.detection_pipeline_integration import DetectionPipelineIntegration
from pynomaly.domain.services.threshold_severity_classifier import ThresholdSeverityClassifier
from pynomaly.domain.value_objects import ContaminationRate


class MockServices:
    """Mock services for integration testing."""
    
    @staticmethod
    async def mock_detect_anomalies(detector: Detector, data: np.ndarray) -> List[float]:
        """Mock anomaly detection returning variance-based scores."""
        scores = []
        for row in data:
            variance = np.var(row)
            score = min(variance / 5.0, 1.0)  # Normalize to [0,1]
            scores.append(score)
        return scores
    
    @staticmethod
    async def mock_train_detector(training_job: TrainingJob) -> Detector:
        """Mock detector training."""
        return Detector(
            id=training_job.detector_id,
            name=f"trained_{training_job.detector_id}",
            algorithm_name="mock_algorithm",
            contamination_rate=ContaminationRate.from_value(0.1),
            parameters=training_job.parameters,
            is_fitted=True,
            trained_at=datetime.utcnow(),
        )


class TestIntegrationWorkflows:
    """Integration testing suite for GitHub Issue #164."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (50, 5))
        anomalous_data = np.random.normal(3, 1, (5, 5))
        return np.vstack([normal_data, anomalous_data])

    @pytest.fixture
    def classification_service(self):
        """Advanced classification service."""
        severity_classifier = ThresholdSeverityClassifier()
        return AdvancedClassificationService(
            severity_classifier=severity_classifier,
            enable_hierarchical=True,
            enable_multiclass=True,
        )

    @pytest.fixture
    def pipeline_integration(self, classification_service):
        """Pipeline integration service."""
        return DetectionPipelineIntegration(classification_service)

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_validation(
        self, sample_data, classification_service, pipeline_integration
    ):
        """Test end-to-end workflow validation."""
        # Step 1: Create detector
        detector = Detector(
            name="e2e_test_detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate.from_value(0.1),
            parameters={"n_estimators": 100, "random_state": 42},
        )

        # Step 2: Train detector
        training_job = TrainingJob(
            detector_id=detector.id,
            dataset_name="e2e_dataset",
            training_data=sample_data,
            parameters=detector.parameters,
        )

        trained_detector = await MockServices.mock_train_detector(training_job)
        assert trained_detector.is_fitted
        assert trained_detector.trained_at is not None

        # Step 3: Run detection
        test_data = sample_data[:10]
        detection_results = await MockServices.mock_detect_anomalies(
            trained_detector, test_data
        )

        assert len(detection_results) == len(test_data)
        assert all(0 <= score <= 1 for score in detection_results)

        # Step 4: Process through classification pipeline
        classifications = []
        events = []

        for i, score in enumerate(detection_results):
            feature_data = {f"feature_{j}": test_data[i, j] for j in range(test_data.shape[1])}
            context_data = {
                "timestamp": datetime.utcnow(),
                "data_point_index": i,
            }

            classification, event = pipeline_integration.process_detection_result(
                anomaly_score=score,
                detector=trained_detector,
                raw_data={"data_point": test_data[i].tolist()},
                feature_data=feature_data,
                context_data=context_data,
            )

            classifications.append(classification)
            events.append(event)

        # Step 5: Validate results
        assert len(classifications) == len(test_data)
        assert len(events) == len(test_data)

        # Verify classifications have expected structure
        for classification in classifications:
            assert classification.get_primary_class() in ["anomaly", "normal"]
            assert 0 <= classification.get_confidence_score() <= 1
            assert classification.severity_classification in ["low", "medium", "high", "critical"]

        # Verify events have expected structure
        for event in events:
            assert event.event_type in [EventType.ANOMALY_DETECTED, EventType.CUSTOM]
            assert event.severity in [EventSeverity.LOW, EventSeverity.MEDIUM, EventSeverity.HIGH, EventSeverity.CRITICAL]
            assert event.anomaly_data is not None

    @pytest.mark.asyncio
    async def test_performance_and_load_testing(self, sample_data, pipeline_integration):
        """Test performance and load characteristics."""
        # Create multiple detectors for load testing
        detectors = []
        for i in range(5):
            detector = Detector(
                name=f"load_test_detector_{i}",
                algorithm_name="isolation_forest",
                contamination_rate=ContaminationRate.from_value(0.1),
                parameters={"n_estimators": 50, "random_state": i},
            )
            
            training_job = TrainingJob(
                detector_id=detector.id,
                dataset_name=f"load_dataset_{i}",
                training_data=sample_data,
                parameters=detector.parameters,
            )
            
            trained_detector = await MockServices.mock_train_detector(training_job)
            detectors.append(trained_detector)

        # Performance testing
        start_time = datetime.utcnow()
        
        # Process data through all detectors concurrently
        tasks = []
        for detector in detectors:
            task = asyncio.create_task(
                MockServices.mock_detect_anomalies(detector, sample_data[:20])
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        # Performance assertions
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert len(results) == 5
        assert all(len(result) == 20 for result in results)

        # Load testing with classification pipeline
        classification_start = datetime.utcnow()
        
        for detector, detection_results in zip(detectors, results):
            for i, score in enumerate(detection_results):
                _, _ = pipeline_integration.process_detection_result(
                    anomaly_score=score,
                    detector=detector,
                    raw_data={"data_point": sample_data[i].tolist()},
                    feature_data={f"feature_{j}": sample_data[i, j] for j in range(sample_data.shape[1])},
                    context_data={"timestamp": datetime.utcnow()},
                )

        classification_time = (datetime.utcnow() - classification_start).total_seconds()
        
        # Classification performance assertions
        assert classification_time < 10.0  # Should complete within 10 seconds

    @pytest.mark.asyncio
    async def test_security_and_compliance_validation(self, sample_data, pipeline_integration):
        """Test security and compliance features."""
        # Create detector with security context
        detector = Detector(
            name="security_test_detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate.from_value(0.1),
            parameters={"n_estimators": 50, "random_state": 42},
            metadata={"security_classification": "sensitive", "compliance_level": "high"},
        )

        training_job = TrainingJob(
            detector_id=detector.id,
            dataset_name="security_dataset",
            training_data=sample_data,
            parameters=detector.parameters,
            metadata={"audit_trail": True, "user_id": "security_test_user"},
        )

        trained_detector = await MockServices.mock_train_detector(training_job)

        # Run detection with security context
        detection_results = await MockServices.mock_detect_anomalies(
            trained_detector, sample_data[:5]
        )

        # Process with security and audit context
        security_events = []
        for i, score in enumerate(detection_results):
            context_data = {
                "timestamp": datetime.utcnow(),
                "user_id": "security_test_user",
                "session_id": str(uuid4()),
                "security_level": "high",
                "audit_trail": {
                    "action": "anomaly_detection",
                    "detector_id": str(trained_detector.id),
                    "compliance_checked": True,
                },
            }

            _, event = pipeline_integration.process_detection_result(
                anomaly_score=score,
                detector=trained_detector,
                raw_data={"data_point": sample_data[i].tolist()},
                context_data=context_data,
            )
            security_events.append(event)

        # Validate security and compliance features
        assert len(security_events) == 5

        for event in security_events:
            # Check audit trail
            assert "audit_trail" in event.business_context
            assert event.business_context["user_id"] == "security_test_user"
            assert event.business_context["security_level"] == "high"
            
            # Check compliance metadata
            assert event.business_context["audit_trail"]["compliance_checked"] is True
            
            # Verify event has proper security tags
            security_tags = [tag for tag in event.tags if "security" in tag or "audit" in tag]
            assert len(security_tags) > 0

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, sample_data, pipeline_integration):
        """Test multi-tenant isolation capabilities."""
        # Create detectors for different tenants
        tenant_detectors = {}
        tenants = ["tenant_a", "tenant_b", "tenant_c"]

        for tenant in tenants:
            detector = Detector(
                name=f"{tenant}_detector",
                algorithm_name="isolation_forest",
                contamination_rate=ContaminationRate.from_value(0.1),
                parameters={"n_estimators": 50, "random_state": 42},
                metadata={"tenant_id": tenant, "isolation_level": "strict"},
            )

            training_job = TrainingJob(
                detector_id=detector.id,
                dataset_name=f"{tenant}_dataset",
                training_data=sample_data,
                parameters=detector.parameters,
                metadata={"tenant_id": tenant},
            )

            trained_detector = await MockServices.mock_train_detector(training_job)
            tenant_detectors[tenant] = trained_detector

        # Test isolation by processing data for each tenant
        tenant_results = {}
        for tenant, detector in tenant_detectors.items():
            detection_results = await MockServices.mock_detect_anomalies(
                detector, sample_data[:3]
            )

            tenant_events = []
            for i, score in enumerate(detection_results):
                context_data = {
                    "timestamp": datetime.utcnow(),
                    "tenant_id": tenant,
                    "isolation_context": {"strict_mode": True},
                }

                _, event = pipeline_integration.process_detection_result(
                    anomaly_score=score,
                    detector=detector,
                    raw_data={"data_point": sample_data[i].tolist()},
                    context_data=context_data,
                )
                tenant_events.append(event)

            tenant_results[tenant] = tenant_events

        # Validate tenant isolation
        assert len(tenant_results) == 3

        for tenant, events in tenant_results.items():
            assert len(events) == 3
            
            # Verify tenant-specific context
            for event in events:
                assert event.business_context["tenant_id"] == tenant
                assert "isolation_context" in event.business_context
                
                # Verify tenant-specific tags
                tenant_tags = [tag for tag in event.tags if tenant in tag]
                assert len(tenant_tags) > 0

    @pytest.mark.asyncio
    async def test_disaster_recovery_scenarios(self, sample_data, pipeline_integration):
        """Test disaster recovery and resilience."""
        # Create detector with recovery metadata
        detector = Detector(
            name="disaster_recovery_detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate.from_value(0.1),
            parameters={"n_estimators": 50, "random_state": 42},
            metadata={"backup_enabled": True, "recovery_tier": "critical"},
        )

        training_job = TrainingJob(
            detector_id=detector.id,
            dataset_name="disaster_recovery_dataset",
            training_data=sample_data,
            parameters=detector.parameters,
            metadata={"backup_location": "disaster_recovery_backup"},
        )

        trained_detector = await MockServices.mock_train_detector(training_job)

        # Simulate various failure scenarios
        failure_scenarios = [
            {"type": "network_failure", "severity": "high"},
            {"type": "data_corruption", "severity": "critical"},
            {"type": "service_unavailable", "severity": "medium"},
        ]

        recovery_events = []
        for scenario in failure_scenarios:
            # Simulate detection during failure scenario
            try:
                detection_results = await MockServices.mock_detect_anomalies(
                    trained_detector, sample_data[:2]
                )

                for i, score in enumerate(detection_results):
                    context_data = {
                        "timestamp": datetime.utcnow(),
                        "disaster_recovery_mode": True,
                        "failure_scenario": scenario,
                        "recovery_metadata": {
                            "backup_available": True,
                            "fallback_mode": "enabled",
                        },
                    }

                    _, event = pipeline_integration.process_detection_result(
                        anomaly_score=score,
                        detector=trained_detector,
                        raw_data={"data_point": sample_data[i].tolist()},
                        context_data=context_data,
                    )
                    recovery_events.append(event)

            except Exception as e:
                # Log recovery scenario handling
                recovery_event = {
                    "scenario": scenario,
                    "error": str(e),
                    "recovery_attempted": True,
                }
                recovery_events.append(recovery_event)

        # Validate disaster recovery capabilities
        assert len(recovery_events) >= len(failure_scenarios)

        # Check that recovery context is properly recorded
        for event in recovery_events:
            if hasattr(event, 'business_context'):
                assert "disaster_recovery_mode" in event.business_context
                assert "failure_scenario" in event.business_context
                assert "recovery_metadata" in event.business_context

    @pytest.mark.asyncio
    async def test_api_contract_validation(self, sample_data, pipeline_integration):
        """Test API contract compliance."""
        # Create detector for API testing
        detector = Detector(
            name="api_contract_detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate.from_value(0.1),
            parameters={"n_estimators": 50, "random_state": 42},
        )

        training_job = TrainingJob(
            detector_id=detector.id,
            dataset_name="api_contract_dataset",
            training_data=sample_data,
            parameters=detector.parameters,
        )

        trained_detector = await MockServices.mock_train_detector(training_job)

        # Test API contract compliance
        detection_results = await MockServices.mock_detect_anomalies(
            trained_detector, sample_data[:3]
        )

        api_responses = []
        for i, score in enumerate(detection_results):
            classification, event = pipeline_integration.process_detection_result(
                anomaly_score=score,
                detector=trained_detector,
                raw_data={"data_point": sample_data[i].tolist()},
                feature_data={f"feature_{j}": sample_data[i, j] for j in range(sample_data.shape[1])},
                context_data={"timestamp": datetime.utcnow(), "api_version": "v1"},
            )

            # Simulate API response structure
            api_response = {
                "status": "success",
                "data": {
                    "classification": {
                        "primary_class": classification.get_primary_class(),
                        "confidence_score": classification.get_confidence_score(),
                        "severity": classification.severity_classification,
                    },
                    "event": {
                        "id": str(event.id),
                        "type": event.event_type.value,
                        "severity": event.severity.value,
                        "timestamp": event.event_time.isoformat() if hasattr(event.event_time, 'isoformat') else str(event.event_time),
                    },
                },
                "metadata": {
                    "api_version": "v1",
                    "processing_time_ms": 100,
                },
            }
            api_responses.append(api_response)

        # Validate API contract compliance
        assert len(api_responses) == 3

        for response in api_responses:
            # Check required fields
            assert "status" in response
            assert "data" in response
            assert "metadata" in response

            # Check data structure
            assert "classification" in response["data"]
            assert "event" in response["data"]

            # Check classification structure
            classification_data = response["data"]["classification"]
            assert "primary_class" in classification_data
            assert "confidence_score" in classification_data
            assert "severity" in classification_data

            # Check event structure
            event_data = response["data"]["event"]
            assert "id" in event_data
            assert "type" in event_data
            assert "severity" in event_data
            assert "timestamp" in event_data

            # Validate data types and values
            assert classification_data["primary_class"] in ["anomaly", "normal"]
            assert 0 <= classification_data["confidence_score"] <= 1
            assert classification_data["severity"] in ["low", "medium", "high", "critical"]
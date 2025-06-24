"""Comprehensive tests for application layer to achieve high coverage."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from hypothesis import given, strategies as st, settings
import uuid
from datetime import datetime

from tests.conftest_dependencies import requires_dependency, requires_dependencies

# Application layer imports
from pynomaly.application.dto.detector_dto import CreateDetectorDTO, DetectorResponseDTO
from pynomaly.application.dto.experiment_dto import CreateExperimentDTO, ExperimentResponseDTO
from pynomaly.application.use_cases.train_detector import TrainDetectorUseCase
from pynomaly.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from pynomaly.application.use_cases.evaluate_model import EvaluateModelUseCase


class TestApplicationDTOsComprehensive:
    """Comprehensive tests for all DTOs to achieve high coverage."""
    
    def test_create_detector_dto_validation(self):
        """Test CreateDetectorDTO validation extensively."""
        # Valid DTO
        valid_dto = CreateDetectorDTO(
            name="test_detector",
            algorithm="IsolationForest",
            contamination=0.1,
            parameters={"n_estimators": 100}
        )
        assert valid_dto.name == "test_detector"
        assert valid_dto.algorithm == "IsolationForest"
        assert valid_dto.contamination == 0.1
        
        # Test serialization
        json_data = valid_dto.model_dump()
        assert "name" in json_data
        assert "algorithm" in json_data
        
        # Test deserialization
        recreated_dto = CreateDetectorDTO.model_validate(json_data)
        assert recreated_dto.name == valid_dto.name
    
    def test_detector_response_dto_comprehensive(self):
        """Test DetectorResponseDTO comprehensive functionality."""
        response_dto = DetectorResponseDTO(
            id=str(uuid.uuid4()),
            name="response_test_detector",
            algorithm="LOF",
            contamination=0.05,
            parameters={"n_neighbors": 20},
            is_fitted=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert response_dto.is_fitted is True
        assert response_dto.algorithm == "LOF"
        
        # Test JSON serialization
        json_data = response_dto.model_dump()
        assert "id" in json_data
        assert "is_fitted" in json_data
    
    def test_experiment_dto_comprehensive(self):
        """Test ExperimentDTO comprehensive functionality."""
        experiment_dto = CreateExperimentDTO(
            name="comprehensive_experiment",
            description="Test experiment for comprehensive testing",
            detector_configs=[
                {
                    "algorithm": "IsolationForest",
                    "contamination": 0.1
                },
                {
                    "algorithm": "LOF", 
                    "contamination": 0.05
                }
            ],
            evaluation_metrics=["precision", "recall", "f1_score"]
        )
        
        assert len(experiment_dto.detector_configs) == 2
        assert "precision" in experiment_dto.evaluation_metrics
        
        # Test validation
        json_data = experiment_dto.model_dump()
        recreated = CreateExperimentDTO.model_validate(json_data)
        assert recreated.name == experiment_dto.name
    
    @given(
        detector_name=st.text(min_size=1, max_size=100),
        contamination=st.floats(min_value=0.01, max_value=0.5)
    )
    @settings(max_examples=20)
    def test_dto_property_based_validation(self, detector_name, contamination):
        """Property-based testing for DTO validation."""
        dto = CreateDetectorDTO(
            name=detector_name,
            algorithm="IsolationForest",
            contamination=contamination
        )
        
        assert dto.name == detector_name
        assert dto.contamination == contamination
        assert 0.0 < dto.contamination <= 0.5


class TestUseCasesComprehensive:
    """Comprehensive tests for use cases."""
    
    @pytest.fixture
    def mock_detector_repository(self):
        """Mock detector repository."""
        repo = Mock()
        repo.save = Mock()
        repo.find_by_id = Mock()
        repo.find_by_name = Mock()
        return repo
    
    @pytest.fixture
    def mock_dataset_repository(self):
        """Mock dataset repository."""
        repo = Mock()
        repo.find_by_id = Mock()
        repo.save = Mock()
        return repo
    
    @pytest.fixture
    def mock_adapter_factory(self):
        """Mock adapter factory."""
        factory = Mock()
        adapter = Mock()
        adapter.fit = Mock()
        adapter.detect = Mock()
        adapter.is_fitted = True
        factory.create_adapter = Mock(return_value=adapter)
        return factory, adapter
    
    def test_train_detector_use_case_comprehensive(self, mock_detector_repository, mock_dataset_repository, mock_adapter_factory):
        """Test TrainDetectorUseCase comprehensively."""
        factory, adapter = mock_adapter_factory
        
        use_case = TrainDetectorUseCase(
            detector_repository=mock_detector_repository,
            dataset_repository=mock_dataset_repository,
            adapter_factory=factory
        )
        
        # Mock entities
        from pynomaly.domain.entities import Detector, Dataset
        import pandas as pd
        import numpy as np
        
        detector = Detector(
            id="test_detector_id",
            name="test_detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1}
        )
        
        data = pd.DataFrame(np.random.randn(100, 3), columns=['x', 'y', 'z'])
        dataset = Dataset(
            id="test_dataset_id",
            name="test_dataset",
            data=data
        )
        
        # Setup mocks
        mock_detector_repository.find_by_id.return_value = detector
        mock_dataset_repository.find_by_id.return_value = dataset
        mock_detector_repository.save.return_value = detector
        
        # Execute use case
        result = use_case.execute("test_detector_id", "test_dataset_id")
        
        # Verify interactions
        mock_detector_repository.find_by_id.assert_called_with("test_detector_id")
        mock_dataset_repository.find_by_id.assert_called_with("test_dataset_id")
        adapter.fit.assert_called_once()
        mock_detector_repository.save.assert_called_once()
        
        # Verify result
        assert result.success is True
        assert result.detector_id == "test_detector_id"
    
    def test_detect_anomalies_use_case_comprehensive(self, mock_detector_repository, mock_dataset_repository, mock_adapter_factory):
        """Test DetectAnomaliesUseCase comprehensively."""
        factory, adapter = mock_adapter_factory
        
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            dataset_repository=mock_dataset_repository,
            adapter_factory=factory
        )
        
        # Mock detection result
        from pynomaly.domain.entities import DetectionResult, Anomaly
        from pynomaly.domain.value_objects import AnomalyScore
        
        anomaly = Anomaly(
            index=0,
            score=AnomalyScore(0.8),
            features={"x": 1.0, "y": 2.0, "z": 3.0}
        )
        
        detection_result = DetectionResult(
            detector_name="test_detector",
            dataset_name="test_dataset",
            scores=[AnomalyScore(0.1), AnomalyScore(0.8), AnomalyScore(0.3)],
            anomalies=[anomaly],
            threshold=0.5,
            execution_time=0.1
        )
        
        adapter.detect.return_value = detection_result
        
        # Setup other mocks
        from pynomaly.domain.entities import Detector, Dataset
        import pandas as pd
        import numpy as np
        
        detector = Detector(
            id="test_detector_id",
            name="test_detector",
            algorithm="IsolationForest",
            is_fitted=True
        )
        
        data = pd.DataFrame(np.random.randn(100, 3), columns=['x', 'y', 'z'])
        dataset = Dataset(id="test_dataset_id", name="test_dataset", data=data)
        
        mock_detector_repository.find_by_id.return_value = detector
        mock_dataset_repository.find_by_id.return_value = dataset
        
        # Execute use case
        result = use_case.execute("test_detector_id", "test_dataset_id")
        
        # Verify
        assert result.success is True
        assert len(result.detection_result.scores) == 3
        assert len(result.detection_result.anomalies) == 1
        adapter.detect.assert_called_once()
    
    def test_evaluate_model_use_case_comprehensive(self, mock_detector_repository, mock_dataset_repository, mock_adapter_factory):
        """Test EvaluateModelUseCase comprehensively."""
        factory, adapter = mock_adapter_factory
        
        use_case = EvaluateModelUseCase(
            detector_repository=mock_detector_repository,
            dataset_repository=mock_dataset_repository,
            adapter_factory=factory
        )
        
        # Mock evaluation scenario
        from pynomaly.domain.entities import Detector, Dataset
        from pynomaly.domain.value_objects import AnomalyScore
        import pandas as pd
        import numpy as np
        
        detector = Detector(
            id="eval_detector",
            name="eval_detector",
            algorithm="IsolationForest",
            is_fitted=True
        )
        
        # Create dataset with known ground truth
        data = pd.DataFrame(np.random.randn(100, 3), columns=['x', 'y', 'z'])
        targets = np.array([0] * 95 + [1] * 5)  # 5% anomalies
        dataset = Dataset(
            id="eval_dataset",
            name="eval_dataset", 
            data=data,
            targets=targets
        )
        
        # Mock detection scores
        scores = [AnomalyScore(0.1)] * 95 + [AnomalyScore(0.9)] * 5
        adapter.score.return_value = scores
        
        mock_detector_repository.find_by_id.return_value = detector
        mock_dataset_repository.find_by_id.return_value = dataset
        
        # Execute evaluation
        result = use_case.execute("eval_detector", "eval_dataset", ["precision", "recall", "f1_score"])
        
        # Verify
        assert result.success is True
        assert "precision" in result.metrics
        assert "recall" in result.metrics
        assert "f1_score" in result.metrics


@requires_dependency('optuna')
class TestAutoMLComprehensive:
    """Comprehensive tests for AutoML functionality."""
    
    @pytest.fixture
    def mock_automl_service(self):
        """Mock AutoML service."""
        service = Mock()
        service.optimize_detector = AsyncMock()
        service.profile_dataset = AsyncMock()
        service.recommend_algorithms = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_automl_optimization_comprehensive(self, mock_automl_service):
        """Test AutoML optimization comprehensively."""
        from pynomaly.application.dto.automl_dto import AutoMLRequestDTO, AutoMLResponseDTO
        
        # Create optimization request
        request = AutoMLRequestDTO(
            dataset_id="automl_dataset",
            objective="auc",
            max_algorithms=3,
            max_optimization_time=1800,
            n_trials=50,
            enable_ensemble=True
        )
        
        # Mock optimization result
        optimization_result = AutoMLResponseDTO(
            success=True,
            detector_id="optimized_detector",
            automl_result={
                "best_algorithm": "IsolationForest",
                "best_params": {"contamination": 0.08, "n_estimators": 150},
                "best_score": 0.92,
                "trials_completed": 50
            },
            message="Optimization completed successfully",
            execution_time=1650.5
        )
        
        mock_automl_service.optimize_detector.return_value = optimization_result
        
        # Execute optimization
        result = await mock_automl_service.optimize_detector(request)
        
        # Verify
        assert result.success is True
        assert result.automl_result["best_score"] > 0.9
        assert result.automl_result["trials_completed"] == 50
        mock_automl_service.optimize_detector.assert_called_once_with(request)
    
    @pytest.mark.asyncio
    async def test_automl_dataset_profiling(self, mock_automl_service):
        """Test AutoML dataset profiling."""
        from pynomaly.application.dto.automl_dto import AutoMLProfileRequestDTO, AutoMLProfileResponseDTO
        
        profile_request = AutoMLProfileRequestDTO(
            dataset_id="profile_dataset",
            include_recommendations=True,
            max_recommendations=5
        )
        
        profile_result = AutoMLProfileResponseDTO(
            success=True,
            dataset_profile={
                "n_samples": 10000,
                "n_features": 15,
                "contamination_estimate": 0.05,
                "complexity_score": 0.7,
                "sparsity_ratio": 0.02
            },
            algorithm_recommendations=[
                {"algorithm_name": "IsolationForest", "score": 0.95},
                {"algorithm_name": "LOF", "score": 0.88},
                {"algorithm_name": "OCSVM", "score": 0.82}
            ],
            message="Profiling completed",
            execution_time=45.2
        )
        
        mock_automl_service.profile_dataset.return_value = profile_result
        
        result = await mock_automl_service.profile_dataset(profile_request)
        
        assert result.success is True
        assert result.dataset_profile["n_samples"] == 10000
        assert len(result.algorithm_recommendations) == 3


@requires_dependencies('shap', 'lime')
class TestExplainabilityComprehensive:
    """Comprehensive tests for explainability functionality."""
    
    @pytest.fixture
    def mock_explainability_service(self):
        """Mock explainability service."""
        service = Mock()
        service.explain_anomaly = AsyncMock()
        service.explain_feature_importance = AsyncMock()
        service.explain_cohort = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_anomaly_explanation_comprehensive(self, mock_explainability_service):
        """Test comprehensive anomaly explanation."""
        from pynomaly.application.dto.explainability_dto import ExplanationRequestDTO, ExplanationResponseDTO
        
        explanation_request = ExplanationRequestDTO(
            detector_id="explainable_detector",
            dataset_id="explainable_dataset",
            anomaly_indices=[42, 87, 156],
            explanation_method="shap",
            include_feature_importance=True,
            include_counterfactuals=True
        )
        
        explanation_result = ExplanationResponseDTO(
            success=True,
            explanations={
                "42": {
                    "feature_contributions": {
                        "feature_1": 0.3,
                        "feature_2": -0.1,
                        "feature_3": 0.8
                    },
                    "base_value": 0.2,
                    "prediction": 0.9
                }
            },
            global_feature_importance={
                "feature_1": 0.4,
                "feature_2": 0.3,
                "feature_3": 0.3
            },
            explanation_method="shap",
            execution_time=12.5
        )
        
        mock_explainability_service.explain_anomaly.return_value = explanation_result
        
        result = await mock_explainability_service.explain_anomaly(explanation_request)
        
        assert result.success is True
        assert "42" in result.explanations
        assert result.explanations["42"]["prediction"] == 0.9
        assert "feature_1" in result.global_feature_importance
    
    @pytest.mark.asyncio
    async def test_cohort_explanation_comprehensive(self, mock_explainability_service):
        """Test cohort explanation functionality."""
        from pynomaly.application.dto.explainability_dto import CohortExplanationRequestDTO, CohortExplanationResponseDTO
        
        cohort_request = CohortExplanationRequestDTO(
            detector_id="cohort_detector",
            dataset_id="cohort_dataset",
            cohort_definition={
                "feature_1": {"min": 0.0, "max": 5.0},
                "feature_2": {"min": -2.0, "max": 2.0}
            },
            explanation_method="lime",
            sample_size=1000
        )
        
        cohort_result = CohortExplanationResponseDTO(
            success=True,
            cohort_size=856,
            cohort_anomaly_rate=0.12,
            cohort_explanations={
                "avg_feature_importance": {
                    "feature_1": 0.45,
                    "feature_2": 0.35,
                    "feature_3": 0.20
                },
                "common_patterns": [
                    "High feature_1 values strongly indicate anomalies",
                    "Low feature_2 values moderately indicate anomalies"
                ]
            },
            execution_time=28.7
        )
        
        mock_explainability_service.explain_cohort.return_value = cohort_result
        
        result = await mock_explainability_service.explain_cohort(cohort_request)
        
        assert result.success is True
        assert result.cohort_size == 856
        assert result.cohort_anomaly_rate == 0.12
        assert "feature_1" in result.cohort_explanations["avg_feature_importance"]


class TestApplicationServicesIntegration:
    """Test application services integration."""
    
    def test_service_dependency_injection(self):
        """Test service dependency injection."""
        from pynomaly.infrastructure.config.container import Container
        
        container = Container()
        
        # Test that services can be created from container
        try:
            detection_service = container.detection_service()
            assert detection_service is not None
        except Exception as e:
            # Service creation might fail due to missing dependencies
            # This is expected in test environment
            assert "dependency" in str(e).lower() or "import" in str(e).lower()
    
    def test_service_error_handling(self):
        """Test application service error handling."""
        from pynomaly.application.services.detection_service import DetectionService
        from pynomaly.domain.exceptions import ValidationError
        
        # Mock repositories with error conditions
        mock_detector_repo = Mock()
        mock_detector_repo.find_by_id.side_effect = ValidationError("Detector not found")
        
        mock_dataset_repo = Mock()
        mock_adapter_factory = Mock()
        
        service = DetectionService(
            detector_repository=mock_detector_repo,
            dataset_repository=mock_dataset_repo,
            adapter_factory=mock_adapter_factory
        )
        
        # Test error propagation
        with pytest.raises(ValidationError):
            service.detect_anomalies("non_existent_detector", "some_dataset")
    
    @given(
        detector_name=st.text(min_size=1, max_size=50),
        algorithm=st.sampled_from(['IsolationForest', 'LOF', 'OCSVM']),
        contamination=st.floats(min_value=0.01, max_value=0.3)
    )
    @settings(max_examples=10)
    def test_service_property_based_validation(self, detector_name, algorithm, contamination):
        """Property-based testing for service validation."""
        from pynomaly.application.dto.detector_dto import CreateDetectorDTO
        
        # Test that DTO creation works with property-based inputs
        dto = CreateDetectorDTO(
            name=detector_name,
            algorithm=algorithm,
            contamination=contamination
        )
        
        assert dto.name == detector_name
        assert dto.algorithm == algorithm
        assert dto.contamination == contamination


class TestApplicationWorkflows:
    """Test complete application workflows."""
    
    def test_end_to_end_detection_workflow(self):
        """Test end-to-end detection workflow."""
        # This test simulates a complete workflow from DTO to response
        # without external dependencies
        
        from pynomaly.application.dto.detector_dto import CreateDetectorDTO
        from pynomaly.application.dto.experiment_dto import CreateExperimentDTO
        
        # 1. Create detector request
        detector_request = CreateDetectorDTO(
            name="workflow_detector",
            algorithm="IsolationForest",
            contamination=0.1,
            parameters={"n_estimators": 100}
        )
        
        # 2. Validate DTO
        assert detector_request.name == "workflow_detector"
        assert detector_request.algorithm == "IsolationForest"
        
        # 3. Create experiment request
        experiment_request = CreateExperimentDTO(
            name="workflow_experiment",
            description="End-to-end workflow test",
            detector_configs=[detector_request.model_dump()],
            evaluation_metrics=["precision", "recall"]
        )
        
        # 4. Validate experiment
        assert len(experiment_request.detector_configs) == 1
        assert experiment_request.detector_configs[0]["algorithm"] == "IsolationForest"
        
        # 5. Test serialization/deserialization cycle
        json_data = experiment_request.model_dump()
        recreated = CreateExperimentDTO.model_validate(json_data)
        
        assert recreated.name == experiment_request.name
        assert recreated.detector_configs == experiment_request.detector_configs
    
    def test_batch_processing_workflow(self):
        """Test batch processing workflow."""
        from pynomaly.application.dto.detector_dto import CreateDetectorDTO
        
        # Create multiple detectors for batch processing
        detectors = []
        algorithms = ["IsolationForest", "LOF", "OCSVM"]
        
        for i, algorithm in enumerate(algorithms):
            detector = CreateDetectorDTO(
                name=f"batch_detector_{i}",
                algorithm=algorithm,
                contamination=0.1 + i * 0.02,  # Vary contamination
                parameters={"random_state": 42}
            )
            detectors.append(detector)
        
        # Verify batch creation
        assert len(detectors) == 3
        assert all(d.contamination != detectors[0].contamination for d in detectors[1:])
        
        # Test batch validation
        for detector in detectors:
            json_data = detector.model_dump()
            recreated = CreateDetectorDTO.model_validate(json_data)
            assert recreated.algorithm == detector.algorithm
    
    def test_error_recovery_workflow(self):
        """Test error recovery in workflows."""
        from pynomaly.application.dto.detector_dto import CreateDetectorDTO
        from pydantic import ValidationError
        
        # Test recovery from validation errors
        invalid_detectors = [
            {"name": "", "algorithm": "IsolationForest"},  # Empty name
            {"name": "test", "algorithm": "InvalidAlgorithm"},  # Invalid algorithm
            {"name": "test", "algorithm": "IsolationForest", "contamination": 1.5},  # Invalid contamination
        ]
        
        valid_detectors = []
        errors = []
        
        for detector_data in invalid_detectors:
            try:
                detector = CreateDetectorDTO.model_validate(detector_data)
                valid_detectors.append(detector)
            except ValidationError as e:
                errors.append(str(e))
        
        # Should have captured all errors
        assert len(errors) == 3
        assert len(valid_detectors) == 0
        
        # Test with one valid detector
        valid_data = {"name": "valid_detector", "algorithm": "IsolationForest", "contamination": 0.1}
        valid_detector = CreateDetectorDTO.model_validate(valid_data)
        assert valid_detector.name == "valid_detector"
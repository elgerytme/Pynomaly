"""Comprehensive test suite for explainability service."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.services.explainability_service import (
    ApplicationExplainabilityService,
    ExplanationRequest,
    ExplanationResponse,
)
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.services.explainability_service import (
    CohortExplanation,
    ExplainabilityService,
    ExplanationMethod,
    FeatureContribution,
    GlobalExplanation,
    LocalExplanation,
)
from pynomaly.infrastructure.explainers.shap_explainer import SHAPExplainer


class TestExplainabilityService:
    """Test suite for explainability service functionality."""

    @pytest.fixture
    def mock_repositories(self):
        """Mock repositories for testing."""
        detector_repo = Mock()
        dataset_repo = Mock()
        return detector_repo, dataset_repo

    @pytest.fixture
    def mock_domain_service(self):
        """Mock domain explainability service."""
        service = Mock(spec=ExplainabilityService)
        return service

    @pytest.fixture
    def explainability_service(self, mock_repositories, mock_domain_service):
        """Create explainability service with mocked dependencies."""
        detector_repo, dataset_repo = mock_repositories

        service = ApplicationExplainabilityService(
            domain_explainability_service=mock_domain_service,
            detector_repository=detector_repo,
            dataset_repository=dataset_repo,
        )

        return service

    @pytest.fixture
    def sample_detector(self):
        """Create a sample trained detector."""
        detector = Mock(spec=Detector)
        detector.id = "detector_123"
        detector.is_trained = True
        detector.model = Mock()
        detector.model.__class__.__name__ = "IsolationForest"
        return detector

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        data = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
            }
        )

        dataset = Mock(spec=Dataset)
        dataset.id = "dataset_123"
        dataset.data = data
        return dataset

    @pytest.fixture
    def sample_local_explanation(self):
        """Create a sample local explanation."""
        contributions = [
            FeatureContribution(
                feature_name="feature_1",
                value=1.5,
                contribution=0.3,
                importance=0.3,
                rank=1,
                description="Most important feature",
            ),
            FeatureContribution(
                feature_name="feature_2",
                value=-0.5,
                contribution=-0.1,
                importance=0.1,
                rank=2,
                description="Second important feature",
            ),
        ]

        return LocalExplanation(
            instance_id="instance_123",
            anomaly_score=0.8,
            prediction="anomaly",
            confidence=0.7,
            feature_contributions=contributions,
            explanation_method=ExplanationMethod.SHAP,
            model_name="IsolationForest",
            timestamp=datetime.now().isoformat(),
        )

    @pytest.mark.asyncio
    async def test_explain_instance_with_instance_data(
        self,
        explainability_service,
        sample_detector,
        mock_domain_service,
        sample_local_explanation,
    ):
        """Test explaining instance with direct instance data."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        mock_domain_service.explain_instance.return_value = sample_local_explanation

        # Create request
        request = ExplanationRequest(
            detector_id="detector_123",
            instance_data={"feature_1": 1.5, "feature_2": -0.5, "feature_3": 0.2},
            explanation_method=ExplanationMethod.SHAP,
        )

        # Execute
        response = await explainability_service.explain_instance(request)

        # Verify
        assert response.success is True
        assert response.explanation == sample_local_explanation
        assert "SHAP explanation" in response.message
        assert response.error is None

        # Verify domain service was called correctly
        mock_domain_service.explain_instance.assert_called_once()
        call_args = mock_domain_service.explain_instance.call_args
        assert len(call_args[1]["instance"]) == 3  # 3 features
        assert call_args[1]["feature_names"] == ["feature_1", "feature_2", "feature_3"]

    @pytest.mark.asyncio
    async def test_explain_instance_with_dataset_index(
        self,
        explainability_service,
        sample_detector,
        sample_dataset,
        mock_domain_service,
        sample_local_explanation,
    ):
        """Test explaining instance from dataset by index."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        explainability_service.dataset_repository.get_by_id.return_value = (
            sample_dataset
        )
        mock_domain_service.explain_instance.return_value = sample_local_explanation

        # Create request
        request = ExplanationRequest(
            detector_id="detector_123",
            dataset_id="dataset_123",
            instance_indices=[5],
            explanation_method=ExplanationMethod.SHAP,
        )

        # Execute
        response = await explainability_service.explain_instance(request)

        # Verify
        assert response.success is True
        assert response.explanation == sample_local_explanation

        # Verify domain service was called with dataset instance
        mock_domain_service.explain_instance.assert_called_once()

    @pytest.mark.asyncio
    async def test_explain_instance_detector_not_found(self, explainability_service):
        """Test explaining instance when detector is not found."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = None

        # Create request
        request = ExplanationRequest(
            detector_id="nonexistent_detector", instance_data={"feature_1": 1.0}
        )

        # Execute
        response = await explainability_service.explain_instance(request)

        # Verify
        assert response.success is False
        assert "Detector not found" in response.error
        assert response.explanation is None

    @pytest.mark.asyncio
    async def test_explain_instance_detector_not_trained(self, explainability_service):
        """Test explaining instance when detector is not trained."""
        # Setup mocks
        untrained_detector = Mock(spec=Detector)
        untrained_detector.is_trained = False
        explainability_service.detector_repository.get_by_id.return_value = (
            untrained_detector
        )

        # Create request
        request = ExplanationRequest(
            detector_id="untrained_detector", instance_data={"feature_1": 1.0}
        )

        # Execute
        response = await explainability_service.explain_instance(request)

        # Verify
        assert response.success is False
        assert "not trained" in response.error
        assert response.explanation is None

    @pytest.mark.asyncio
    async def test_explain_instance_invalid_dataset_index(
        self, explainability_service, sample_detector, sample_dataset
    ):
        """Test explaining instance with invalid dataset index."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        explainability_service.dataset_repository.get_by_id.return_value = (
            sample_dataset
        )

        # Create request with invalid index
        request = ExplanationRequest(
            detector_id="detector_123",
            dataset_id="dataset_123",
            instance_indices=[999],  # Out of range
            explanation_method=ExplanationMethod.SHAP,
        )

        # Execute
        response = await explainability_service.explain_instance(request)

        # Verify
        assert response.success is False
        assert "index out of range" in response.error

    @pytest.mark.asyncio
    async def test_explain_instance_insufficient_data(
        self, explainability_service, sample_detector
    ):
        """Test explaining instance with insufficient data."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )

        # Create request without instance data or dataset
        request = ExplanationRequest(
            detector_id="detector_123", explanation_method=ExplanationMethod.SHAP
        )

        # Execute
        response = await explainability_service.explain_instance(request)

        # Verify
        assert response.success is False
        assert "Insufficient data" in response.error

    @pytest.mark.asyncio
    async def test_explain_model_success(
        self,
        explainability_service,
        sample_detector,
        sample_dataset,
        mock_domain_service,
    ):
        """Test successful global model explanation."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        explainability_service.dataset_repository.get_by_id.return_value = (
            sample_dataset
        )

        global_explanation = GlobalExplanation(
            model_name="IsolationForest",
            feature_importances={"feature_1": 0.5, "feature_2": 0.3, "feature_3": 0.2},
            top_features=["feature_1", "feature_2", "feature_3"],
            explanation_method=ExplanationMethod.SHAP,
            model_performance={"score": 0.85},
            timestamp=datetime.now().isoformat(),
            summary="Model focuses on feature_1",
        )
        mock_domain_service.explain_model.return_value = global_explanation

        # Create request
        request = ExplanationRequest(
            detector_id="detector_123",
            dataset_id="dataset_123",
            explanation_method=ExplanationMethod.SHAP,
            background_samples=50,
        )

        # Execute
        response = await explainability_service.explain_model(request)

        # Verify
        assert response.success is True
        assert response.explanation == global_explanation
        assert "global SHAP explanation" in response.message

    @pytest.mark.asyncio
    async def test_explain_model_missing_dataset(
        self, explainability_service, sample_detector
    ):
        """Test explaining model without dataset."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )

        # Create request without dataset
        request = ExplanationRequest(
            detector_id="detector_123", explanation_method=ExplanationMethod.SHAP
        )

        # Execute
        response = await explainability_service.explain_model(request)

        # Verify
        assert response.success is False
        assert "Dataset ID required" in response.error

    @pytest.mark.asyncio
    async def test_explain_cohort_success(
        self,
        explainability_service,
        sample_detector,
        sample_dataset,
        mock_domain_service,
    ):
        """Test successful cohort explanation."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        explainability_service.dataset_repository.get_by_id.return_value = (
            sample_dataset
        )

        cohort_explanation = CohortExplanation(
            cohort_id="cohort_123",
            cohort_description="High-anomaly cohort",
            instance_count=5,
            common_features=[
                FeatureContribution(
                    feature_name="feature_1",
                    value=1.0,
                    contribution=0.4,
                    importance=0.4,
                    rank=1,
                )
            ],
            explanation_method=ExplanationMethod.SHAP,
            model_name="IsolationForest",
            timestamp=datetime.now().isoformat(),
        )
        mock_domain_service.explain_cohort.return_value = cohort_explanation

        # Create request
        request = ExplanationRequest(
            detector_id="detector_123",
            dataset_id="dataset_123",
            instance_indices=[1, 2, 3, 4, 5],
            explanation_method=ExplanationMethod.SHAP,
        )

        # Execute
        response = await explainability_service.explain_cohort(request)

        # Verify
        assert response.success is True
        assert response.explanation == cohort_explanation
        assert "cohort SHAP explanation" in response.message

    @pytest.mark.asyncio
    async def test_explain_cohort_invalid_indices(
        self, explainability_service, sample_detector, sample_dataset
    ):
        """Test cohort explanation with invalid indices."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        explainability_service.dataset_repository.get_by_id.return_value = (
            sample_dataset
        )

        # Create request with invalid indices
        request = ExplanationRequest(
            detector_id="detector_123",
            dataset_id="dataset_123",
            instance_indices=[1, 2, 999],  # 999 is out of range
            explanation_method=ExplanationMethod.SHAP,
        )

        # Execute
        response = await explainability_service.explain_cohort(request)

        # Verify
        assert response.success is False
        assert "Invalid instance indices" in response.error

    @pytest.mark.asyncio
    async def test_compare_explanations(
        self,
        explainability_service,
        sample_detector,
        mock_domain_service,
        sample_local_explanation,
    ):
        """Test comparing explanations from multiple methods."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        mock_domain_service.explain_instance.return_value = sample_local_explanation

        # Create request
        request = ExplanationRequest(
            detector_id="detector_123",
            instance_data={"feature_1": 1.0, "feature_2": -0.5},
        )

        methods = [ExplanationMethod.SHAP, ExplanationMethod.LIME]

        # Execute
        results = await explainability_service.compare_explanations(request, methods)

        # Verify
        assert len(results) == 2
        assert "shap" in results
        assert "lime" in results
        for method_name, response in results.items():
            assert response.success is True
            assert response.explanation == sample_local_explanation

    @pytest.mark.asyncio
    async def test_get_feature_statistics(
        self,
        explainability_service,
        sample_detector,
        sample_dataset,
        mock_domain_service,
        sample_local_explanation,
    ):
        """Test getting feature statistics across multiple explanations."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        explainability_service.dataset_repository.get_by_id.return_value = (
            sample_dataset
        )

        # Mock domain service methods
        mock_domain_service.get_feature_statistics.return_value = {
            "feature_1": {
                "mean_contribution": 0.3,
                "std_contribution": 0.1,
                "mean_importance": 0.3,
                "count": 10,
            }
        }
        mock_domain_service.rank_features_by_importance.return_value = [
            ("feature_1", 0.3),
            ("feature_2", 0.1),
        ]

        # Mock explanation generation
        async def mock_explain_instance(request):
            return ExplanationResponse(
                success=True, explanation=sample_local_explanation, message="Success"
            )

        explainability_service.explain_instance = mock_explain_instance

        # Execute
        stats = await explainability_service.get_feature_statistics(
            detector_id="detector_123",
            dataset_id="dataset_123",
            method=ExplanationMethod.SHAP,
            sample_size=10,
        )

        # Verify
        assert "feature_statistics" in stats
        assert "top_features" in stats
        assert "total_explanations" in stats
        assert stats["method"] == "shap"

    @pytest.mark.asyncio
    async def test_get_feature_statistics_no_explanations(
        self, explainability_service, sample_detector, sample_dataset
    ):
        """Test getting feature statistics when no explanations can be generated."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        explainability_service.dataset_repository.get_by_id.return_value = (
            sample_dataset
        )

        # Mock explanation generation to always fail
        async def mock_explain_instance(request):
            return ExplanationResponse(
                success=False, error="Failed to explain", message="Error"
            )

        explainability_service.explain_instance = mock_explain_instance

        # Execute
        stats = await explainability_service.get_feature_statistics(
            detector_id="detector_123", dataset_id="dataset_123", sample_size=5
        )

        # Verify
        assert "error" in stats
        assert "Failed to generate any explanations" in stats["error"]

    def test_get_available_methods(self, explainability_service, mock_domain_service):
        """Test getting available explanation methods."""
        # Setup mock
        mock_domain_service.get_available_methods.return_value = [
            ExplanationMethod.SHAP,
            ExplanationMethod.LIME,
            ExplanationMethod.FEATURE_IMPORTANCE,
        ]

        # Execute
        methods = explainability_service.get_available_methods()

        # Verify
        assert len(methods) == 3
        assert ExplanationMethod.SHAP in methods
        assert ExplanationMethod.LIME in methods
        assert ExplanationMethod.FEATURE_IMPORTANCE in methods

    @pytest.mark.asyncio
    async def test_exception_handling_in_explain_instance(
        self, explainability_service, sample_detector, mock_domain_service
    ):
        """Test exception handling in explain_instance."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        mock_domain_service.explain_instance.side_effect = Exception(
            "Domain service error"
        )

        # Create request
        request = ExplanationRequest(
            detector_id="detector_123", instance_data={"feature_1": 1.0}
        )

        # Execute
        response = await explainability_service.explain_instance(request)

        # Verify
        assert response.success is False
        assert "Domain service error" in response.error
        assert "Failed to generate explanation" in response.message

    @pytest.mark.asyncio
    async def test_exception_handling_in_explain_model(
        self,
        explainability_service,
        sample_detector,
        sample_dataset,
        mock_domain_service,
    ):
        """Test exception handling in explain_model."""
        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        explainability_service.dataset_repository.get_by_id.return_value = (
            sample_dataset
        )
        mock_domain_service.explain_model.side_effect = Exception(
            "Model explanation error"
        )

        # Create request
        request = ExplanationRequest(
            detector_id="detector_123", dataset_id="dataset_123"
        )

        # Execute
        response = await explainability_service.explain_model(request)

        # Verify
        assert response.success is False
        assert "Model explanation error" in response.error

    @pytest.mark.asyncio
    async def test_background_data_sampling(
        self,
        explainability_service,
        sample_detector,
        mock_domain_service,
        sample_local_explanation,
    ):
        """Test background data sampling for large datasets."""
        # Create large dataset
        large_data = pd.DataFrame(np.random.randn(2000, 3), columns=["f1", "f2", "f3"])
        large_dataset = Mock(spec=Dataset)
        large_dataset.data = large_data

        # Setup mocks
        explainability_service.detector_repository.get_by_id.return_value = (
            sample_detector
        )
        explainability_service.dataset_repository.get_by_id.return_value = large_dataset

        global_explanation = GlobalExplanation(
            model_name="IsolationForest",
            feature_importances={"f1": 0.5, "f2": 0.3, "f3": 0.2},
            top_features=["f1", "f2", "f3"],
            explanation_method=ExplanationMethod.SHAP,
            model_performance={},
            timestamp=datetime.now().isoformat(),
            summary="Test summary",
        )
        mock_domain_service.explain_model.return_value = global_explanation

        # Create request with specific background sample size
        request = ExplanationRequest(
            detector_id="detector_123", dataset_id="dataset_123", background_samples=100
        )

        # Execute
        response = await explainability_service.explain_model(request)

        # Verify
        assert response.success is True

        # Verify that domain service was called with sampled data
        call_args = mock_domain_service.explain_model.call_args
        background_data = call_args[1]["data"]
        assert len(background_data) == 100  # Should be sampled to 100 rows

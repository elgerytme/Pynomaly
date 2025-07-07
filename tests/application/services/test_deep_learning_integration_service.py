"""Tests for deep learning integration service."""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.services.deep_learning_integration_service import (
    DeepLearningIntegrationService,
    DLFrameworkInfo,
    DLModelPerformance,
    DLOptimizationConfig,
)
from pynomaly.domain.entities import Dataset


class TestDeepLearningIntegrationService:
    """Test deep learning integration service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = DeepLearningIntegrationService()

        # Create test dataset
        self.test_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 1000),
                "feature_2": np.random.normal(0, 1, 1000),
                "feature_3": np.random.normal(0, 1, 1000),
            }
        )

        self.test_dataset = Dataset(
            name="test_dataset",
            data=self.test_data,
            features=["feature_1", "feature_2", "feature_3"],
        )

    def test_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert isinstance(self.service.available_frameworks, dict)
        assert isinstance(self.service.performance_history, list)
        assert isinstance(self.service.framework_preferences, dict)

    def test_detect_available_frameworks(self):
        """Test framework detection."""
        frameworks = self.service._detect_available_frameworks()
        assert isinstance(frameworks, dict)

        # Check that detected frameworks have required structure
        for _name, framework_info in frameworks.items():
            assert isinstance(framework_info, DLFrameworkInfo)
            assert framework_info.name
            assert isinstance(framework_info.algorithms, list)
            assert framework_info.performance_tier in [
                "very_high",
                "high",
                "medium",
                "low",
            ]
            assert isinstance(framework_info.use_cases, list)
            assert isinstance(framework_info.hardware_requirements, dict)

    def test_get_available_frameworks(self):
        """Test getting available frameworks."""
        frameworks = self.service.get_available_frameworks()
        assert isinstance(frameworks, dict)

        # Should be a copy, not the original
        original_frameworks = self.service.available_frameworks
        assert frameworks is not original_frameworks
        assert frameworks == original_frameworks

    def test_get_framework_capabilities(self):
        """Test getting framework capabilities."""
        # Test with existing framework (if any)
        frameworks = self.service.get_available_frameworks()
        if frameworks:
            framework_name = list(frameworks.keys())[0]
            capabilities = self.service.get_framework_capabilities(framework_name)
            assert capabilities is not None
            assert isinstance(capabilities, DLFrameworkInfo)

        # Test with non-existing framework
        capabilities = self.service.get_framework_capabilities("nonexistent")
        assert capabilities is None

    def test_recommend_framework_with_target(self):
        """Test framework recommendation with target framework."""
        config = DLOptimizationConfig(target_framework="pytorch")

        # Mock pytorch as available
        with patch.object(
            self.service,
            "available_frameworks",
            {
                "pytorch": DLFrameworkInfo(
                    name="PyTorch",
                    available=True,
                    algorithms=["autoencoder", "vae"],
                    performance_tier="high",
                    use_cases=["research"],
                    hardware_requirements={},
                )
            },
        ):
            framework = self.service.recommend_framework(
                self.test_dataset, "autoencoder", config
            )
            assert framework == "pytorch"

    def test_recommend_framework_auto_select(self):
        """Test auto framework selection."""
        config = DLOptimizationConfig(auto_select_framework=True)

        # Mock multiple frameworks
        with patch.object(
            self.service,
            "available_frameworks",
            {
                "pytorch": DLFrameworkInfo(
                    name="PyTorch",
                    available=True,
                    algorithms=["autoencoder", "vae"],
                    performance_tier="high",
                    use_cases=["research"],
                    hardware_requirements={},
                ),
                "tensorflow": DLFrameworkInfo(
                    name="TensorFlow",
                    available=True,
                    algorithms=["autoencoder", "transformer"],
                    performance_tier="high",
                    use_cases=["production"],
                    hardware_requirements={},
                ),
            },
        ):
            framework = self.service.recommend_framework(
                self.test_dataset, "autoencoder", config
            )
            assert framework in ["pytorch", "tensorflow"]

    def test_recommend_framework_no_support(self):
        """Test framework recommendation with unsupported algorithm."""
        config = DLOptimizationConfig()

        # Mock frameworks that don't support the algorithm
        with patch.object(
            self.service,
            "available_frameworks",
            {
                "pytorch": DLFrameworkInfo(
                    name="PyTorch",
                    available=True,
                    algorithms=["vae"],  # No autoencoder
                    performance_tier="high",
                    use_cases=["research"],
                    hardware_requirements={},
                )
            },
        ):
            with pytest.raises(
                ValueError, match="No available framework supports algorithm"
            ):
                self.service.recommend_framework(
                    self.test_dataset, "autoencoder", config
                )

    def test_calculate_framework_score(self):
        """Test framework scoring calculation."""
        framework_info = DLFrameworkInfo(
            name="PyTorch",
            available=True,
            algorithms=["autoencoder", "vae"],
            performance_tier="high",
            use_cases=["research", "prototyping"],
            hardware_requirements={"gpu": "cuda_compatible"},
        )

        config = DLOptimizationConfig(performance_priority="speed", enable_gpu=True)

        score = self.service._calculate_framework_score(
            "pytorch", framework_info, self.test_dataset, "autoencoder", config
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 2.0  # Reasonable score range

    @pytest.mark.asyncio
    async def test_create_deep_learning_detector(self):
        """Test deep learning detector creation."""
        # Mock detector creation
        mock_detector = Mock()
        mock_detector.async_fit = AsyncMock()
        mock_detector.get_model_info.return_value = {"algorithm": "autoencoder"}

        with patch.object(
            self.service, "_create_framework_detector", return_value=mock_detector
        ) as mock_create:
            detector = await self.service.create_deep_learning_detector(
                dataset=self.test_dataset, algorithm="autoencoder", framework="pytorch"
            )

            assert detector == mock_detector
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_deep_learning_detector_auto_framework(self):
        """Test detector creation with auto framework selection."""
        mock_detector = Mock()

        with patch.object(
            self.service, "_create_framework_detector", return_value=mock_detector
        ):
            with patch.object(
                self.service, "recommend_framework", return_value="pytorch"
            ) as mock_recommend:
                detector = await self.service.create_deep_learning_detector(
                    dataset=self.test_dataset, algorithm="autoencoder"
                )

                assert detector == mock_detector
                mock_recommend.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_framework_detector_pytorch(self):
        """Test PyTorch detector creation."""
        with patch(
            "pynomaly.application.services.deep_learning_integration_service.PyTorchAdapter"
        ) as MockAdapter:
            mock_detector = Mock()
            MockAdapter.return_value = mock_detector

            detector = await self.service._create_framework_detector(
                "pytorch", "autoencoder", {}, DLOptimizationConfig()
            )

            assert detector == mock_detector
            MockAdapter.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_framework_detector_tensorflow(self):
        """Test TensorFlow detector creation."""
        with patch(
            "pynomaly.application.services.deep_learning_integration_service.TensorFlowAdapter"
        ) as MockAdapter:
            mock_detector = Mock()
            MockAdapter.return_value = mock_detector

            detector = await self.service._create_framework_detector(
                "tensorflow", "autoencoder", {}, DLOptimizationConfig()
            )

            assert detector == mock_detector
            MockAdapter.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_framework_detector_jax(self):
        """Test JAX detector creation."""
        with patch(
            "pynomaly.application.services.deep_learning_integration_service.JAXAdapter"
        ) as MockAdapter:
            mock_detector = Mock()
            MockAdapter.return_value = mock_detector

            detector = await self.service._create_framework_detector(
                "jax", "autoencoder", {}, DLOptimizationConfig()
            )

            assert detector == mock_detector
            MockAdapter.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_framework_detector_unknown(self):
        """Test detector creation with unknown framework."""
        with pytest.raises(ValueError, match="Unknown framework"):
            await self.service._create_framework_detector(
                "unknown", "autoencoder", {}, DLOptimizationConfig()
            )

    @pytest.mark.asyncio
    async def test_benchmark_frameworks(self):
        """Test framework benchmarking."""
        # Mock detectors
        mock_detector = Mock()
        mock_detector.async_fit = AsyncMock()
        mock_detector.async_predict = AsyncMock(return_value=np.array([0, 1, 0]))
        mock_detector.get_model_info.return_value = {"total_parameters": 1000}

        with patch.object(
            self.service, "_create_framework_detector", return_value=mock_detector
        ):
            with patch.object(
                self.service,
                "available_frameworks",
                {
                    "pytorch": DLFrameworkInfo(
                        name="PyTorch",
                        available=True,
                        algorithms=["autoencoder"],
                        performance_tier="high",
                        use_cases=["research"],
                        hardware_requirements={},
                    )
                },
            ):
                results = await self.service.benchmark_frameworks(
                    dataset=self.test_dataset,
                    algorithm="autoencoder",
                    frameworks=["pytorch"],
                )

                assert len(results) == 1
                assert isinstance(results[0], DLModelPerformance)
                assert results[0].framework == "pytorch"
                assert results[0].algorithm == "autoencoder"

    def test_get_performance_recommendations(self):
        """Test performance recommendations."""
        recommendations = self.service.get_performance_recommendations(
            self.test_dataset, {"performance_priority": "speed"}
        )

        assert isinstance(recommendations, dict)
        assert "primary_framework" in recommendations
        assert "alternative_frameworks" in recommendations
        assert "algorithm_suggestions" in recommendations
        assert "configuration_tips" in recommendations
        assert "performance_expectations" in recommendations

    def test_get_performance_recommendations_large_dataset(self):
        """Test recommendations for large dataset."""
        # Create large dataset
        large_data = pd.DataFrame(np.random.randn(200000, 50))
        large_dataset = Dataset(
            name="large_dataset",
            data=large_data,
            features=[f"feature_{i}" for i in range(50)],
        )

        recommendations = self.service.get_performance_recommendations(
            large_dataset, {}
        )

        # Should recommend handling large datasets
        tips = recommendations.get("configuration_tips", [])
        assert any("large dataset" in tip.lower() for tip in tips)

    def test_get_performance_recommendations_high_dimensional(self):
        """Test recommendations for high-dimensional dataset."""
        # Create high-dimensional dataset
        high_dim_data = pd.DataFrame(np.random.randn(1000, 2000))
        high_dim_dataset = Dataset(
            name="high_dim_dataset",
            data=high_dim_data,
            features=[f"feature_{i}" for i in range(2000)],
        )

        recommendations = self.service.get_performance_recommendations(
            high_dim_dataset, {}
        )

        # Should recommend handling high dimensions
        tips = recommendations.get("configuration_tips", [])
        assert any("high-dimensional" in tip.lower() for tip in tips)

    def test_update_framework_preferences(self):
        """Test updating framework preferences."""
        performance_results = [
            DLModelPerformance(
                framework="pytorch",
                algorithm="autoencoder",
                training_time=10.0,
                inference_time=0.001,
                memory_usage=100.0,
                accuracy_score=0.95,
                parameters_count=1000,
                dataset_size=1000,
            ),
            DLModelPerformance(
                framework="tensorflow",
                algorithm="autoencoder",
                training_time=15.0,
                inference_time=0.002,
                memory_usage=150.0,
                accuracy_score=0.93,
                parameters_count=1200,
                dataset_size=1000,
            ),
        ]

        initial_preferences = self.service.framework_preferences.copy()
        self.service.update_framework_preferences(performance_results)

        # Preferences should be updated
        assert len(self.service.framework_preferences) >= len(initial_preferences)
        assert "pytorch" in self.service.framework_preferences
        assert "tensorflow" in self.service.framework_preferences

    def test_get_integration_status(self):
        """Test getting integration status."""
        status = self.service.get_integration_status()

        assert isinstance(status, dict)
        assert "available_frameworks" in status
        assert "frameworks" in status
        assert "total_algorithms" in status
        assert "performance_history_size" in status
        assert "framework_preferences" in status
        assert "pytorch_available" in status
        assert "tensorflow_available" in status
        assert "jax_available" in status

        # Check data types
        assert isinstance(status["available_frameworks"], int)
        assert isinstance(status["frameworks"], list)
        assert isinstance(status["total_algorithms"], int)
        assert isinstance(status["performance_history_size"], int)
        assert isinstance(status["framework_preferences"], dict)
        assert isinstance(status["pytorch_available"], bool)
        assert isinstance(status["tensorflow_available"], bool)
        assert isinstance(status["jax_available"], bool)


class TestDLOptimizationConfig:
    """Test DL optimization configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = DLOptimizationConfig()

        assert config.target_framework is None
        assert config.performance_priority == "balanced"
        assert config.resource_constraints == {}
        assert config.optimization_objectives == ["accuracy", "speed"]
        assert config.auto_select_framework is True
        assert config.enable_gpu is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DLOptimizationConfig(
            target_framework="pytorch",
            performance_priority="speed",
            resource_constraints={"memory": "8GB"},
            optimization_objectives=["speed"],
            auto_select_framework=False,
            enable_gpu=False,
        )

        assert config.target_framework == "pytorch"
        assert config.performance_priority == "speed"
        assert config.resource_constraints == {"memory": "8GB"}
        assert config.optimization_objectives == ["speed"]
        assert config.auto_select_framework is False
        assert config.enable_gpu is False


class TestDLFrameworkInfo:
    """Test DL framework info model."""

    def test_framework_info_creation(self):
        """Test framework info creation."""
        info = DLFrameworkInfo(
            name="PyTorch",
            available=True,
            algorithms=["autoencoder", "vae"],
            performance_tier="high",
            use_cases=["research", "prototyping"],
            hardware_requirements={"gpu": "cuda"},
        )

        assert info.name == "PyTorch"
        assert info.available is True
        assert info.algorithms == ["autoencoder", "vae"]
        assert info.performance_tier == "high"
        assert info.use_cases == ["research", "prototyping"]
        assert info.hardware_requirements == {"gpu": "cuda"}


class TestDLModelPerformance:
    """Test DL model performance model."""

    def test_performance_creation(self):
        """Test performance model creation."""
        performance = DLModelPerformance(
            framework="pytorch",
            algorithm="autoencoder",
            training_time=10.0,
            inference_time=0.001,
            memory_usage=100.0,
            accuracy_score=0.95,
            parameters_count=1000,
            dataset_size=1000,
        )

        assert performance.framework == "pytorch"
        assert performance.algorithm == "autoencoder"
        assert performance.training_time == 10.0
        assert performance.inference_time == 0.001
        assert performance.memory_usage == 100.0
        assert performance.accuracy_score == 0.95
        assert performance.parameters_count == 1000
        assert performance.dataset_size == 1000

"""Deep learning integration service for managing multiple DL frameworks."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from pynomaly.domain.entities import Dataset
from pynomaly.shared.protocols import DetectorProtocol

# Import deep learning adapters with fallbacks
try:
    from pynomaly.infrastructure.adapters.deep_learning import PyTorchAdapter

    # Check if PyTorch is actually available by trying to instantiate
    try:
        import torch

        PYTORCH_AVAILABLE = True
    except ImportError:
        PYTORCH_AVAILABLE = False
except ImportError:
    PyTorchAdapter = None
    PYTORCH_AVAILABLE = False

try:
    from pynomaly.infrastructure.adapters.deep_learning import TensorFlowAdapter

    # Check if TensorFlow is actually available
    try:
        import tensorflow

        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
except ImportError:
    TensorFlowAdapter = None
    TENSORFLOW_AVAILABLE = False

try:
    from pynomaly.infrastructure.adapters.deep_learning import JAXAdapter

    # Check if JAX is actually available
    try:
        import jax

        JAX_AVAILABLE = True
    except ImportError:
        JAX_AVAILABLE = False
except ImportError:
    JAXAdapter = None
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)


class DLFrameworkInfo(BaseModel):
    """Information about deep learning framework availability."""

    name: str = Field(description="Framework name")
    available: bool = Field(description="Whether framework is available")
    algorithms: list[str] = Field(description="Supported algorithms")
    performance_tier: str = Field(description="Performance tier (high, medium, low)")
    use_cases: list[str] = Field(description="Recommended use cases")
    hardware_requirements: dict[str, str] = Field(description="Hardware requirements")


class DLModelPerformance(BaseModel):
    """Performance metrics for deep learning models."""

    framework: str = Field(description="Framework name")
    algorithm: str = Field(description="Algorithm name")
    training_time: float = Field(description="Training time in seconds")
    inference_time: float = Field(description="Average inference time per sample")
    memory_usage: float = Field(description="Memory usage in MB")
    accuracy_score: float = Field(description="Accuracy score")
    parameters_count: int = Field(description="Number of model parameters")
    dataset_size: int = Field(description="Training dataset size")


class DLOptimizationConfig(BaseModel):
    """Configuration for deep learning optimization."""

    target_framework: str | None = Field(None, description="Preferred framework")
    performance_priority: str = Field(
        default="balanced", description="Performance priority"
    )
    resource_constraints: dict[str, Any] = Field(
        default_factory=dict, description="Resource constraints"
    )
    optimization_objectives: list[str] = Field(
        default=["accuracy", "speed"], description="Optimization objectives"
    )
    auto_select_framework: bool = Field(
        default=True, description="Auto-select best framework"
    )
    enable_gpu: bool = Field(default=True, description="Enable GPU acceleration")


class DeepLearningIntegrationService:
    """Service for managing deep learning framework integration."""

    def __init__(self):
        """Initialize deep learning integration service."""
        self.available_frameworks = self._detect_available_frameworks()
        self.performance_history: list[DLModelPerformance] = []
        self.framework_preferences: dict[str, float] = {}

        logger.info(
            f"Initialized DeepLearningIntegrationService with {len(self.available_frameworks)} frameworks"
        )

    def _detect_available_frameworks(self) -> dict[str, DLFrameworkInfo]:
        """Detect available deep learning frameworks."""
        frameworks = {}

        # PyTorch
        if PYTORCH_AVAILABLE:
            frameworks["pytorch"] = DLFrameworkInfo(
                name="PyTorch",
                available=True,
                algorithms=["autoencoder", "vae", "lstm"],
                performance_tier="high",
                use_cases=["research", "prototyping", "flexible_architectures"],
                hardware_requirements={
                    "cpu": "recommended",
                    "gpu": "cuda_compatible",
                    "memory": "moderate",
                },
            )

        # TensorFlow
        if TENSORFLOW_AVAILABLE:
            frameworks["tensorflow"] = DLFrameworkInfo(
                name="TensorFlow",
                available=True,
                algorithms=["autoencoder", "vae", "lstm", "transformer"],
                performance_tier="high",
                use_cases=["production", "serving", "scalability"],
                hardware_requirements={
                    "cpu": "recommended",
                    "gpu": "cuda_compatible",
                    "memory": "moderate",
                },
            )

        # JAX
        if JAX_AVAILABLE:
            frameworks["jax"] = DLFrameworkInfo(
                name="JAX",
                available=True,
                algorithms=["autoencoder", "gmm", "svdd"],
                performance_tier="very_high",
                use_cases=["high_performance", "research", "numerical_computing"],
                hardware_requirements={
                    "cpu": "high_performance",
                    "gpu": "optional_but_recommended",
                    "memory": "high",
                },
            )

        return frameworks

    def get_available_frameworks(self) -> dict[str, DLFrameworkInfo]:
        """Get information about available frameworks."""
        return self.available_frameworks.copy()

    def get_framework_capabilities(self, framework: str) -> DLFrameworkInfo | None:
        """Get capabilities for specific framework."""
        return self.available_frameworks.get(framework)

    def recommend_framework(
        self,
        dataset: Dataset,
        algorithm: str,
        config: DLOptimizationConfig | None = None,
    ) -> str:
        """Recommend best framework for given requirements."""
        if not config:
            config = DLOptimizationConfig()

        # If target framework is specified and available
        if (
            config.target_framework
            and config.target_framework in self.available_frameworks
        ):
            framework_info = self.available_frameworks[config.target_framework]
            if algorithm in framework_info.algorithms:
                return config.target_framework

        # Auto-select based on criteria
        if config.auto_select_framework:
            return self._auto_select_framework(dataset, algorithm, config)

        # Default fallback
        for framework_name in ["tensorflow", "pytorch", "jax"]:
            if framework_name in self.available_frameworks:
                framework_info = self.available_frameworks[framework_name]
                if algorithm in framework_info.algorithms:
                    return framework_name

        raise ValueError(f"No available framework supports algorithm: {algorithm}")

    def _auto_select_framework(
        self, dataset: Dataset, algorithm: str, config: DLOptimizationConfig
    ) -> str:
        """Automatically select best framework based on criteria."""
        scores = {}

        for framework_name, framework_info in self.available_frameworks.items():
            if algorithm not in framework_info.algorithms:
                continue

            score = self._calculate_framework_score(
                framework_name, framework_info, dataset, algorithm, config
            )
            scores[framework_name] = score

        if not scores:
            raise ValueError(f"No framework supports algorithm: {algorithm}")

        # Return framework with highest score
        best_framework = max(scores.items(), key=lambda x: x[1])[0]
        logger.info(
            f"Auto-selected framework: {best_framework} (score: {scores[best_framework]:.3f})"
        )

        return best_framework

    def _calculate_framework_score(
        self,
        framework_name: str,
        framework_info: DLFrameworkInfo,
        dataset: Dataset,
        algorithm: str,
        config: DLOptimizationConfig,
    ) -> float:
        """Calculate framework suitability score."""
        score = 0.0

        # Performance tier scoring
        performance_scores = {"very_high": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}
        score += performance_scores.get(framework_info.performance_tier, 0.5) * 0.3

        # Dataset size considerations
        if hasattr(dataset.data, "shape"):
            n_samples, n_features = dataset.data.shape

            # Large datasets favor production frameworks
            if n_samples > 100000:
                if "production" in framework_info.use_cases:
                    score += 0.2
                if "scalability" in framework_info.use_cases:
                    score += 0.2

            # High-dimensional data
            if n_features > 1000:
                if framework_name == "jax":  # JAX handles high-dim well
                    score += 0.15

        # Algorithm-specific preferences
        if algorithm == "transformer" and framework_name == "tensorflow":
            score += 0.15  # TensorFlow has excellent transformer support
        elif algorithm == "vae" and framework_name == "pytorch":
            score += 0.15  # PyTorch popular for VAE research
        elif algorithm in ["gmm", "svdd"] and framework_name == "jax":
            score += 0.15  # JAX excellent for custom algorithms

        # Performance priority considerations
        if config.performance_priority == "speed":
            if framework_info.performance_tier in ["very_high", "high"]:
                score += 0.2
        elif config.performance_priority == "accuracy":
            # All modern frameworks are capable
            score += 0.1
        elif config.performance_priority == "memory":
            if framework_name == "jax":  # JAX generally more memory efficient
                score += 0.15

        # Historical performance
        if framework_name in self.framework_preferences:
            score += self.framework_preferences[framework_name] * 0.1

        # GPU availability
        if config.enable_gpu:
            gpu_req = framework_info.hardware_requirements.get("gpu", "not_supported")
            if gpu_req in ["cuda_compatible", "optional_but_recommended"]:
                score += 0.1

        return score

    async def create_deep_learning_detector(
        self,
        dataset: Dataset,
        algorithm: str,
        framework: str | None = None,
        model_config: dict[str, Any] | None = None,
        optimization_config: DLOptimizationConfig | None = None,
    ) -> DetectorProtocol:
        """Create deep learning detector with optimal framework."""
        try:
            # Recommend framework if not specified
            if not framework:
                framework = self.recommend_framework(
                    dataset, algorithm, optimization_config
                )

            # Validate framework availability
            if framework not in self.available_frameworks:
                raise ValueError(f"Framework not available: {framework}")

            framework_info = self.available_frameworks[framework]
            if algorithm not in framework_info.algorithms:
                raise ValueError(f"Algorithm {algorithm} not supported by {framework}")

            # Create detector based on framework
            detector = await self._create_framework_detector(
                framework, algorithm, model_config, optimization_config
            )

            logger.info(f"Created {framework} detector for {algorithm}")
            return detector

        except Exception as e:
            logger.error(f"Failed to create deep learning detector: {e}")
            raise

    async def _create_framework_detector(
        self,
        framework: str,
        algorithm: str,
        model_config: dict[str, Any] | None,
        optimization_config: DLOptimizationConfig | None,
    ) -> DetectorProtocol:
        """Create detector for specific framework."""
        config = model_config or {}
        opt_config = optimization_config or DLOptimizationConfig()

        # Add optimization settings to model config
        if opt_config.enable_gpu:
            if framework == "pytorch":
                config["device"] = "auto"
            elif framework == "tensorflow":
                config["gpu_memory_growth"] = True
            elif framework == "jax":
                config["device"] = "auto"

        # Create detector
        if framework == "pytorch":
            detector = PyTorchAdapter(
                algorithm=algorithm, model_config=config, random_state=42
            )
        elif framework == "tensorflow":
            detector = TensorFlowAdapter(
                algorithm=algorithm, model_config=config, random_state=42
            )
        elif framework == "jax":
            detector = JAXAdapter(
                algorithm=algorithm, model_config=config, random_state=42
            )
        else:
            raise ValueError(f"Unknown framework: {framework}")

        return detector

    async def benchmark_frameworks(
        self,
        dataset: Dataset,
        algorithm: str,
        frameworks: list[str] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> list[DLModelPerformance]:
        """Benchmark multiple frameworks on same task."""
        if not frameworks:
            frameworks = [
                name
                for name, info in self.available_frameworks.items()
                if algorithm in info.algorithms
            ]

        results = []

        for framework in frameworks:
            try:
                logger.info(f"Benchmarking {framework} with {algorithm}")

                # Create detector
                detector = await self._create_framework_detector(
                    framework, algorithm, model_config, None
                )

                # Measure training time
                start_time = datetime.now()
                await detector.async_fit(dataset.data.values)
                training_time = (datetime.now() - start_time).total_seconds()

                # Measure inference time
                sample_data = dataset.data.values[:100]  # Use subset for timing
                start_time = datetime.now()
                await detector.async_predict(sample_data)
                end_time = datetime.now()
                inference_time = (end_time - start_time).total_seconds() / len(
                    sample_data
                )

                # Get model info
                model_info = detector.get_model_info()

                performance = DLModelPerformance(
                    framework=framework,
                    algorithm=algorithm,
                    training_time=training_time,
                    inference_time=inference_time,
                    memory_usage=0.0,  # Would need proper memory tracking
                    accuracy_score=0.0,  # Would need labeled data
                    parameters_count=model_info.get("total_parameters", 0),
                    dataset_size=len(dataset.data),
                )

                results.append(performance)
                self.performance_history.append(performance)

                logger.info(
                    f"Completed {framework} benchmark: {training_time:.2f}s training"
                )

            except Exception as e:
                logger.error(f"Benchmark failed for {framework}: {e}")
                continue

        return results

    def get_performance_recommendations(
        self, dataset: Dataset, requirements: dict[str, Any]
    ) -> dict[str, Any]:
        """Get performance-based recommendations."""
        recommendations = {
            "primary_framework": None,
            "alternative_frameworks": [],
            "algorithm_suggestions": {},
            "configuration_tips": [],
            "performance_expectations": {},
        }

        # Analyze dataset characteristics
        if hasattr(dataset.data, "shape"):
            n_samples, n_features = dataset.data.shape

            # Large dataset recommendations
            if n_samples > 100000:
                recommendations["configuration_tips"].append(
                    "Consider batch processing for large datasets"
                )
                recommendations["primary_framework"] = (
                    "tensorflow"  # Good for large scale
                )

            # High-dimensional data
            if n_features > 1000:
                recommendations["configuration_tips"].append(
                    "Consider dimensionality reduction for high-dimensional data"
                )
                if "jax" in self.available_frameworks:
                    recommendations["primary_framework"] = "jax"

            # Small dataset
            if n_samples < 1000:
                recommendations["configuration_tips"].append(
                    "Small dataset: consider data augmentation or simpler models"
                )
                recommendations["primary_framework"] = (
                    "pytorch"  # Good for experimentation
                )

        # Algorithm-specific recommendations
        recommendations["algorithm_suggestions"] = {
            "general_purpose": ["autoencoder", "vae"],
            "time_series": ["lstm"],
            "high_performance": ["gmm", "svdd"],
            "interpretable": ["autoencoder"],
        }

        # Set default if not set
        if not recommendations["primary_framework"]:
            if "tensorflow" in self.available_frameworks:
                recommendations["primary_framework"] = "tensorflow"
            elif "pytorch" in self.available_frameworks:
                recommendations["primary_framework"] = "pytorch"
            elif "jax" in self.available_frameworks:
                recommendations["primary_framework"] = "jax"

        # Alternative frameworks
        for framework in self.available_frameworks:
            if framework != recommendations["primary_framework"]:
                recommendations["alternative_frameworks"].append(framework)

        return recommendations

    def update_framework_preferences(
        self, performance_results: list[DLModelPerformance]
    ):
        """Update framework preferences based on performance."""
        framework_scores = {}

        for result in performance_results:
            if result.framework not in framework_scores:
                framework_scores[result.framework] = []

            # Calculate composite score (lower is better for time, higher for accuracy)
            score = (1.0 / (result.training_time + 1.0)) + (
                1.0 / (result.inference_time + 1.0)
            )
            framework_scores[result.framework].append(score)

        # Update preferences with average scores
        for framework, scores in framework_scores.items():
            avg_score = np.mean(scores)
            self.framework_preferences[framework] = avg_score

        logger.info(f"Updated framework preferences: {self.framework_preferences}")

    def get_integration_status(self) -> dict[str, Any]:
        """Get deep learning integration status."""
        return {
            "available_frameworks": len(self.available_frameworks),
            "frameworks": list(self.available_frameworks.keys()),
            "total_algorithms": sum(
                len(info.algorithms) for info in self.available_frameworks.values()
            ),
            "performance_history_size": len(self.performance_history),
            "framework_preferences": self.framework_preferences,
            "pytorch_available": PYTORCH_AVAILABLE,
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "jax_available": JAX_AVAILABLE,
        }

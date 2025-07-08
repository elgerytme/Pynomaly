"""Algorithm factory for creating and managing anomaly detection algorithms."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pynomaly.domain.exceptions import InvalidAlgorithmError
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.shared.protocols import DetectorProtocol

# Import adapters
from .enhanced_pyod_adapter import EnhancedPyODAdapter
from .enhanced_sklearn_adapter import EnhancedSklearnAdapter
from .ensemble_meta_adapter import AggregationMethod, EnsembleMetaAdapter

# Optional deep learning adapters
try:
    from .deep_learning.pytorch_adapter import PyTorchAdapter
    PYTORCH_AVAILABLE = True
except ImportError:
    PyTorchAdapter = None
    PYTORCH_AVAILABLE = False


class AlgorithmLibrary(Enum):
    """Supported algorithm libraries."""

    PYOD = "pyod"
    SKLEARN = "sklearn"
    ENSEMBLE = "ensemble"
    PYTORCH = "pytorch"
    AUTO = "auto"


class AlgorithmCategory(Enum):
    """Algorithm categories for recommendation."""

    LINEAR = "linear"
    PROXIMITY = "proximity"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"
    PROBABILISTIC = "probabilistic"
    SUPPORT_VECTOR = "support_vector"
    COVARIANCE = "covariance"


@dataclass
class AlgorithmRecommendation:
    """Algorithm recommendation with rationale."""

    algorithm_name: str
    library: AlgorithmLibrary
    confidence: float
    rationale: str
    expected_performance: str
    computational_complexity: str


@dataclass
class DatasetCharacteristics:
    """Characteristics of a dataset for algorithm recommendation."""

    n_samples: int
    n_features: int
    has_categorical: bool = False
    has_missing_values: bool = False
    feature_correlation: Optional[float] = None
    contamination_estimate: Optional[float] = None
    data_distribution: str = "unknown"  # normal, skewed, multimodal, etc.
    computational_budget: str = "medium"  # low, medium, high


class AlgorithmFactory:
    """Factory for creating and managing anomaly detection algorithms."""

    def __init__(self):
        """Initialize the algorithm factory."""
        self._library_adapters = {
            AlgorithmLibrary.PYOD: EnhancedPyODAdapter,
            AlgorithmLibrary.SKLEARN: EnhancedSklearnAdapter,
            AlgorithmLibrary.ENSEMBLE: EnsembleMetaAdapter,
        }
        
        # Register PyTorch adapter if available
        if PYTORCH_AVAILABLE:
            self._library_adapters[AlgorithmLibrary.PYTORCH] = PyTorchAdapter

        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)

    def create_detector(
        self,
        algorithm_name: str,
        library: Optional[Union[AlgorithmLibrary, str]] = None,
        name: Optional[str] = None,
        contamination_rate: Optional[ContaminationRate] = None,
        **kwargs: Any,
    ) -> DetectorProtocol:
        """Create a detector instance.

        Args:
            algorithm_name: Name of the algorithm
            library: Library to use (auto-detected if None)
            name: Custom name for the detector
            contamination_rate: Expected contamination rate
            **kwargs: Algorithm-specific parameters

        Returns:
            Configured detector instance

        Raises:
            InvalidAlgorithmError: If algorithm is not supported
        """
        # Convert string library to enum
        if isinstance(library, str):
            try:
                library = AlgorithmLibrary(library.lower())
            except ValueError:
                library = AlgorithmLibrary.AUTO

        # Auto-detect library if not specified
        if library is None or library == AlgorithmLibrary.AUTO:
            library = self._detect_library(algorithm_name)

        # Get adapter class
        adapter_class = self._library_adapters.get(library)
        if adapter_class is None:
            raise InvalidAlgorithmError(
                algorithm_name=algorithm_name,
                supported_algorithms=self.list_all_algorithms(),
                details=f"Unsupported library: {library}",
            )

        # Handle ensemble creation differently
        if library == AlgorithmLibrary.ENSEMBLE:
            return self._create_ensemble_detector(
                algorithm_name, name, contamination_rate, **kwargs
            )

        # Create regular detector
        try:
            return adapter_class(
                algorithm_name=algorithm_name,
                name=name,
                contamination_rate=contamination_rate,
                **kwargs,
            )
        except Exception as e:
            raise InvalidAlgorithmError(
                algorithm_name=algorithm_name,
                supported_algorithms=self.list_algorithms_for_library(library),
                details=str(e),
            ) from e

    def create_ensemble(
        self,
        detector_configs: List[Dict[str, Any]],
        name: str = "AutoEnsemble",
        contamination_rate: Optional[ContaminationRate] = None,
        aggregation_method: Union[
            AggregationMethod, str
        ] = AggregationMethod.WEIGHTED_AVERAGE,
        **kwargs: Any,
    ) -> EnsembleMetaAdapter:
        """Create an ensemble detector from multiple algorithms.

        Args:
            detector_configs: List of detector configurations
            name: Name of the ensemble
            contamination_rate: Expected contamination rate
            aggregation_method: Method for combining predictions
            **kwargs: Additional ensemble parameters

        Returns:
            Configured ensemble detector
        """
        # Convert string aggregation method to enum
        if isinstance(aggregation_method, str):
            aggregation_method = AggregationMethod(aggregation_method.lower())

        # Create ensemble
        ensemble = EnsembleMetaAdapter(
            name=name,
            contamination_rate=contamination_rate,
            aggregation_method=aggregation_method,
            **kwargs,
        )

        # Add detectors to ensemble
        for config in detector_configs:
            detector_config = config.copy()
            weight = detector_config.pop("weight", 1.0)

            # Create individual detector
            detector = self.create_detector(**detector_config)
            ensemble.add_detector(detector, weight=weight)

        return ensemble

    def recommend_algorithms(
        self,
        dataset_characteristics: DatasetCharacteristics,
        top_k: int = 5,
        include_ensembles: bool = True,
    ) -> List[AlgorithmRecommendation]:
        """Recommend algorithms based on dataset characteristics.

        Args:
            dataset_characteristics: Characteristics of the dataset
            top_k: Number of recommendations to return
            include_ensembles: Whether to include ensemble recommendations

        Returns:
            List of algorithm recommendations
        """
        recommendations = []

        # Get basic algorithm recommendations
        pyod_recommendations = self._get_pyod_recommendations(dataset_characteristics)
        sklearn_recommendations = self._get_sklearn_recommendations(
            dataset_characteristics
        )

        recommendations.extend(pyod_recommendations)
        recommendations.extend(sklearn_recommendations)

        # Add ensemble recommendations if requested
        if include_ensembles:
            ensemble_recommendations = self._get_ensemble_recommendations(
                dataset_characteristics, pyod_recommendations + sklearn_recommendations
            )
            recommendations.extend(ensemble_recommendations)

        # Sort by confidence and return top_k
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:top_k]

    def create_auto_detector(
        self,
        dataset_characteristics: DatasetCharacteristics,
        performance_preference: str = "balanced",  # fast, balanced, accurate
        name: Optional[str] = None,
        contamination_rate: Optional[ContaminationRate] = None,
    ) -> DetectorProtocol:
        """Automatically create the best detector for given characteristics.

        Args:
            dataset_characteristics: Characteristics of the dataset
            performance_preference: Performance preference (fast, balanced, accurate)
            name: Custom name for the detector
            contamination_rate: Expected contamination rate

        Returns:
            Automatically selected and configured detector
        """
        # Get recommendations
        recommendations = self.recommend_algorithms(
            dataset_characteristics,
            top_k=10,
            include_ensembles=(performance_preference != "fast"),
        )

        # Filter based on performance preference
        if performance_preference == "fast":
            # Prefer fast algorithms
            fast_algorithms = [
                rec
                for rec in recommendations
                if "O(n log n)" in rec.computational_complexity
                or "O(n*p)" in rec.computational_complexity
            ]
            if fast_algorithms:
                recommendations = fast_algorithms
        elif performance_preference == "accurate":
            # Prefer ensemble methods and complex algorithms
            complex_algorithms = [
                rec
                for rec in recommendations
                if rec.library == AlgorithmLibrary.ENSEMBLE
                or "Neural" in rec.expected_performance
            ]
            if complex_algorithms:
                recommendations = complex_algorithms[:3] + recommendations[:2]

        # Select the top recommendation
        if not recommendations:
            # Fallback to IsolationForest
            return self.create_detector(
                algorithm_name="IsolationForest",
                library=AlgorithmLibrary.SKLEARN,
                name=name or "AutoDetector",
                contamination_rate=contamination_rate,
            )

        best_rec = recommendations[0]

        # Create detector based on recommendation
        if best_rec.library == AlgorithmLibrary.ENSEMBLE:
            # Create ensemble with top algorithms
            top_algorithms = recommendations[:3]
            detector_configs = []

            for i, rec in enumerate(top_algorithms):
                if rec.library != AlgorithmLibrary.ENSEMBLE:
                    detector_configs.append(
                        {
                            "algorithm_name": rec.algorithm_name,
                            "library": rec.library,
                            "weight": rec.confidence,
                            "contamination_rate": contamination_rate,
                        }
                    )

            if detector_configs:
                return self.create_ensemble(
                    detector_configs=detector_configs,
                    name=name or "AutoEnsemble",
                    contamination_rate=contamination_rate,
                    aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
                )

        # Create single detector
        return self.create_detector(
            algorithm_name=best_rec.algorithm_name,
            library=best_rec.library,
            name=name or f"Auto_{best_rec.algorithm_name}",
            contamination_rate=contamination_rate,
        )

    def list_all_algorithms(self) -> List[str]:
        """List all available algorithms across all libraries."""
        algorithms = []

        # PyOD algorithms
        algorithms.extend(EnhancedPyODAdapter.list_algorithms())

        # Sklearn algorithms
        algorithms.extend(
            [f"sklearn_{name}" for name in EnhancedSklearnAdapter.list_algorithms()]
        )

        # PyTorch algorithms
        if PYTORCH_AVAILABLE:
            algorithms.extend(["autoencoder", "vae", "lstm"])

        # Ensemble algorithms
        algorithms.extend(["ensemble", "auto_ensemble"])

        return algorithms

    def list_algorithms_for_library(self, library: AlgorithmLibrary) -> List[str]:
        """List algorithms for a specific library."""
        if library == AlgorithmLibrary.PYOD:
            return EnhancedPyODAdapter.list_algorithms()
        elif library == AlgorithmLibrary.SKLEARN:
            return EnhancedSklearnAdapter.list_algorithms()
        elif library == AlgorithmLibrary.PYTORCH:
            return ["autoencoder", "vae", "lstm"] if PYTORCH_AVAILABLE else []
        elif library == AlgorithmLibrary.ENSEMBLE:
            return ["ensemble", "auto_ensemble"]
        else:
            return []

    def get_algorithm_info(
        self, algorithm_name: str, library: Optional[AlgorithmLibrary] = None
    ) -> Dict[str, Any]:
        """Get detailed information about an algorithm."""
        if library is None:
            library = self._detect_library(algorithm_name)

        if library == AlgorithmLibrary.PYOD:
            metadata = EnhancedPyODAdapter.get_algorithm_metadata(algorithm_name)
            if metadata:
                return {
                    "name": algorithm_name,
                    "library": "PyOD",
                    "category": metadata.category,
                    "complexity_time": metadata.complexity_time,
                    "complexity_space": metadata.complexity_space,
                    "supports_streaming": metadata.supports_streaming,
                    "supports_multivariate": metadata.supports_multivariate,
                    "requires_gpu": metadata.requires_gpu,
                    "description": metadata.description,
                }
        elif library == AlgorithmLibrary.SKLEARN:
            info = EnhancedSklearnAdapter.get_algorithm_info(algorithm_name)
            if info:
                return {
                    "name": algorithm_name,
                    "library": "scikit-learn",
                    "category": info.category,
                    "complexity_time": info.complexity_time,
                    "complexity_space": info.complexity_space,
                    "supports_streaming": info.supports_streaming,
                    "requires_scaling": info.requires_scaling,
                    "description": info.description,
                }

        return {
            "name": algorithm_name,
            "library": str(library),
            "error": "Algorithm not found",
        }

    def _detect_library(self, algorithm_name: str) -> AlgorithmLibrary:
        """Auto-detect which library contains the algorithm."""
        # Check PyTorch first
        if PYTORCH_AVAILABLE and algorithm_name.lower() in [
            "autoencoder", "pytorch_autoencoder", "vae", "pytorch_vae", 
            "lstm", "pytorch_lstm", "pytorch"
        ]:
            return AlgorithmLibrary.PYTORCH
            
        # Check PyOD first
        if algorithm_name in EnhancedPyODAdapter.list_algorithms():
            return AlgorithmLibrary.PYOD

        # Check sklearn (remove sklearn_ prefix if present)
        sklearn_name = algorithm_name.replace("sklearn_", "")
        if sklearn_name in EnhancedSklearnAdapter.list_algorithms():
            return AlgorithmLibrary.SKLEARN

        # Check for ensemble indicators
        if (
            algorithm_name.lower() in ["ensemble", "auto_ensemble"]
            or "ensemble" in algorithm_name.lower()
        ):
            return AlgorithmLibrary.ENSEMBLE

        # Default to PyOD for unknown algorithms
        return AlgorithmLibrary.PYOD

    def _create_ensemble_detector(
        self,
        algorithm_name: str,
        name: Optional[str],
        contamination_rate: Optional[ContaminationRate],
        **kwargs: Any,
    ) -> EnsembleMetaAdapter:
        """Create an ensemble detector."""
        # Handle different ensemble types
        if algorithm_name.lower() == "auto_ensemble":
            # Create a balanced ensemble with diverse algorithms
            detector_configs = [
                {
                    "algorithm_name": "IsolationForest",
                    "library": AlgorithmLibrary.SKLEARN,
                    "weight": 1.0,
                },
                {
                    "algorithm_name": "LOF",
                    "library": AlgorithmLibrary.PYOD,
                    "weight": 1.0,
                },
                {
                    "algorithm_name": "COPOD",
                    "library": AlgorithmLibrary.PYOD,
                    "weight": 1.0,
                },
            ]
            return self.create_ensemble(
                detector_configs=detector_configs,
                name=name or "AutoEnsemble",
                contamination_rate=contamination_rate,
                **kwargs,
            )
        else:
            # Create empty ensemble
            return EnsembleMetaAdapter(
                name=name or "CustomEnsemble",
                contamination_rate=contamination_rate,
                **kwargs,
            )

    def _get_pyod_recommendations(
        self, characteristics: DatasetCharacteristics
    ) -> List[AlgorithmRecommendation]:
        """Get PyOD algorithm recommendations."""
        recommendations = []

        # Get recommendations from PyOD adapter
        pyod_algorithms = EnhancedPyODAdapter.recommend_algorithms(
            n_samples=characteristics.n_samples,
            n_features=characteristics.n_features,
            has_gpu=False,  # Conservative assumption
            prefer_fast=(characteristics.computational_budget == "low"),
        )

        for algo in pyod_algorithms:
            metadata = EnhancedPyODAdapter.get_algorithm_metadata(algo)
            if metadata:
                confidence = self._calculate_algorithm_confidence(
                    characteristics, metadata.category, metadata.complexity_time
                )

                recommendations.append(
                    AlgorithmRecommendation(
                        algorithm_name=algo,
                        library=AlgorithmLibrary.PYOD,
                        confidence=confidence,
                        rationale=f"PyOD {metadata.category} algorithm suitable for dataset size",
                        expected_performance=metadata.category,
                        computational_complexity=metadata.complexity_time,
                    )
                )

        return recommendations

    def _get_sklearn_recommendations(
        self, characteristics: DatasetCharacteristics
    ) -> List[AlgorithmRecommendation]:
        """Get sklearn algorithm recommendations."""
        recommendations = []

        # Get recommendations from sklearn adapter
        sklearn_algorithms = EnhancedSklearnAdapter.recommend_algorithms(
            n_samples=characteristics.n_samples,
            n_features=characteristics.n_features,
            prefer_interpretable=(characteristics.computational_budget != "high"),
        )

        for algo in sklearn_algorithms:
            info = EnhancedSklearnAdapter.get_algorithm_info(algo)
            if info:
                confidence = self._calculate_algorithm_confidence(
                    characteristics, info.category, info.complexity_time
                )

                recommendations.append(
                    AlgorithmRecommendation(
                        algorithm_name=algo,
                        library=AlgorithmLibrary.SKLEARN,
                        confidence=confidence,
                        rationale=f"Sklearn {info.category} algorithm with good interpretability",
                        expected_performance=info.category,
                        computational_complexity=info.complexity_time,
                    )
                )

        return recommendations

    def _get_ensemble_recommendations(
        self,
        characteristics: DatasetCharacteristics,
        base_recommendations: List[AlgorithmRecommendation],
    ) -> List[AlgorithmRecommendation]:
        """Get ensemble algorithm recommendations."""
        if (
            characteristics.computational_budget == "low"
            or len(base_recommendations) < 2
        ):
            return []

        # Recommend ensemble for medium to large datasets
        if characteristics.n_samples >= 1000:
            confidence = (
                0.85 if characteristics.computational_budget == "high" else 0.75
            )

            return [
                AlgorithmRecommendation(
                    algorithm_name="auto_ensemble",
                    library=AlgorithmLibrary.ENSEMBLE,
                    confidence=confidence,
                    rationale="Ensemble of diverse algorithms for improved robustness",
                    expected_performance="High accuracy, robust",
                    computational_complexity="O(k * base_complexity)",
                )
            ]

        return []

    def _calculate_algorithm_confidence(
        self, characteristics: DatasetCharacteristics, category: str, complexity: str
    ) -> float:
        """Calculate confidence score for an algorithm recommendation."""
        base_confidence = 0.7

        # Adjust based on dataset size and complexity matching
        if characteristics.n_samples < 1000:
            # Small datasets: prefer simple algorithms
            if "O(n²)" in complexity or "O(n³)" in complexity:
                base_confidence -= 0.2
        elif characteristics.n_samples > 10000:
            # Large datasets: prefer efficient algorithms
            if "O(n log n)" in complexity or "O(n*p)" in complexity:
                base_confidence += 0.15
            elif "O(n²)" in complexity:
                base_confidence -= 0.3

        # Adjust based on feature count
        if characteristics.n_features > 100:
            if category in ["Linear", "Proximity"]:
                base_confidence += 0.1

        # Adjust based on computational budget
        if characteristics.computational_budget == "low":
            if "O(n log n)" in complexity or "O(n*p)" in complexity:
                base_confidence += 0.1
            elif "O(n²)" in complexity:
                base_confidence -= 0.2

        return max(0.1, min(0.95, base_confidence))

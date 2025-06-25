"""AutoML optimization use case."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Import the existing DTOs and create our own response classes for now
from dataclasses import dataclass
from typing import Dict, Any, List
from pynomaly.application.services.automl_service import (
    AutoMLService,
    OptimizationObjective,
    AutoMLResult
)
from pynomaly.domain.entities import Detector, Dataset
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.domain.exceptions import DomainError, AutoMLError
# Use base repository types for now
from typing import Protocol


logger = logging.getLogger(__name__)


@dataclass
class AutoMLOptimizationRequest:
    """Request for AutoML optimization."""
    dataset_id: str
    optimization_objective: str = "auc"
    max_algorithms_to_try: int = 3
    max_optimization_time_minutes: int = 60
    enable_ensemble: bool = True
    detector_name: Optional[str] = None
    cross_validation_folds: int = 3
    random_state: int = 42


@dataclass
class AutoMLOptimizationResponse:
    """Response from AutoML optimization."""
    success: bool
    dataset_id: str
    optimized_detector_id: Optional[str] = None
    best_algorithm: Optional[str] = None
    best_parameters: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    optimization_time_seconds: Optional[float] = None
    trials_completed: Optional[int] = None
    algorithm_rankings: Optional[List[tuple]] = None
    ensemble_config: Optional[Dict[str, Any]] = None
    optimization_summary: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    recommendations: List[str] = None


@dataclass
class DatasetProfileResponse:
    """Response containing dataset profile information."""
    dataset_id: str
    dataset_name: str
    n_samples: int
    n_features: int
    feature_types: Dict[str, str]
    contamination_estimate: float
    complexity_score: float
    has_missing_values: bool
    has_temporal_structure: bool
    has_categorical_features: bool
    recommended_algorithms: List[str]
    profile_metadata: Dict[str, Any]


@dataclass
class AlgorithmRecommendationResponse:
    """Response containing algorithm recommendations."""
    dataset_id: str
    recommended_algorithms: List[str]
    algorithm_scores: Dict[str, float]
    reasoning: Dict[str, str]
    dataset_characteristics: Dict[str, Any]


class AutoMLOptimizationUseCase:
    """Use case for AutoML optimization workflow."""
    
    def __init__(
        self,
        automl_service: AutoMLService,
        detector_repository,  # Any repository with find_by_id method
        dataset_repository    # Any repository with find_by_id method
    ):
        self.automl_service = automl_service
        self.detector_repository = detector_repository
        self.dataset_repository = dataset_repository
    
    async def profile_dataset(self, dataset_id: str) -> DatasetProfileResponse:
        """
        Profile a dataset to understand its characteristics.
        
        Args:
            dataset_id: ID of the dataset to profile
            
        Returns:
            Dataset profile response with characteristics and recommendations
        """
        try:
            logger.info(f"Profiling dataset {dataset_id}")
            
            # Validate dataset exists
            dataset = self.dataset_repository.find_by_id(dataset_id)
            if not dataset:
                raise DomainError(f"Dataset {dataset_id} not found")
            
            # Create profile from dataset characteristics
            profile = self._create_dataset_profile(dataset)
            
            # Get algorithm recommendations
            recommendations = self.automl_service.recommend_algorithms(
                profile, max_algorithms=5
            )
            
            return DatasetProfileResponse(
                dataset_id=dataset_id,
                dataset_name=dataset.name,
                n_samples=profile.n_samples,
                n_features=profile.n_features,
                feature_types=profile.feature_types,
                contamination_estimate=profile.contamination_estimate,
                complexity_score=profile.complexity_score,
                has_missing_values=profile.missing_values_ratio > 0,
                has_temporal_structure=profile.has_temporal_structure,
                has_categorical_features=len(profile.categorical_features) > 0,
                recommended_algorithms=recommendations,
                profile_metadata={
                    "missing_values_ratio": profile.missing_values_ratio,
                    "sparsity_ratio": profile.sparsity_ratio,
                    "dimensionality_ratio": profile.dimensionality_ratio,
                    "dataset_size_mb": profile.dataset_size_mb
                }
            )
            
        except AutoMLError as e:
            logger.error(f"AutoML error during dataset profiling: {e}")
            raise DomainError(f"Dataset profiling failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during dataset profiling: {e}")
            raise DomainError(f"Dataset profiling failed: {str(e)}")
    
    async def get_algorithm_recommendations(
        self, 
        dataset_id: str, 
        max_algorithms: int = 5
    ) -> AlgorithmRecommendationResponse:
        """
        Get algorithm recommendations for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            max_algorithms: Maximum number of algorithms to recommend
            
        Returns:
            Algorithm recommendations with reasoning
        """
        try:
            logger.info(f"Getting algorithm recommendations for dataset {dataset_id}")
            
            # Profile dataset first
            profile = await self.automl_service.profile_dataset(dataset_id)
            
            # Get recommendations
            recommendations = self.automl_service.recommend_algorithms(
                profile, max_algorithms=max_algorithms
            )
            
            # Calculate recommendation reasoning
            algorithm_scores = {}
            reasoning = {}
            
            for algorithm in recommendations:
                config = self.automl_service.algorithm_configs[algorithm]
                score = self.automl_service._calculate_algorithm_score(config, profile)
                algorithm_scores[algorithm] = score
                
                # Generate reasoning
                reasoning[algorithm] = self._generate_recommendation_reasoning(
                    config, profile, score
                )
            
            return AlgorithmRecommendationResponse(
                dataset_id=dataset_id,
                recommended_algorithms=recommendations,
                algorithm_scores=algorithm_scores,
                reasoning=reasoning,
                dataset_characteristics={
                    "size_category": self._categorize_dataset_size(profile.n_samples),
                    "complexity_category": self._categorize_complexity(profile.complexity_score),
                    "feature_diversity": len(set(profile.feature_types.values())),
                    "recommended_contamination": profile.contamination_estimate
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting algorithm recommendations: {e}")
            raise DomainError(f"Algorithm recommendation failed: {str(e)}")
    
    async def optimize_single_algorithm(
        self,
        dataset_id: str,
        algorithm_name: str,
        optimization_objective: str = "auc",
        max_trials: Optional[int] = None,
        contamination_rate: Optional[float] = None
    ) -> AutoMLOptimizationResponse:
        """
        Optimize a single algorithm for the given dataset.
        
        Args:
            dataset_id: ID of the dataset
            algorithm_name: Name of the algorithm to optimize
            optimization_objective: Objective to optimize for
            max_trials: Maximum number of optimization trials
            contamination_rate: Fixed contamination rate (optional)
            
        Returns:
            Optimization results
        """
        try:
            logger.info(f"Optimizing {algorithm_name} for dataset {dataset_id}")
            
            # Validate inputs
            if algorithm_name not in self.automl_service.algorithm_configs:
                raise DomainError(f"Unsupported algorithm: {algorithm_name}")
            
            # Convert objective string to enum
            objective = OptimizationObjective(optimization_objective.lower())
            
            # Override trials if specified
            if max_trials:
                original_trials = self.automl_service.n_trials
                self.automl_service.n_trials = max_trials
            
            try:
                # Run optimization
                result = await self.automl_service.optimize_hyperparameters(
                    dataset_id=dataset_id,
                    algorithm=algorithm_name,
                    objective=objective
                )
                
                # Create optimized detector
                detector_id = await self.automl_service.create_optimized_detector(
                    result,
                    detector_name=f"AutoML_{algorithm_name}_{dataset_id[:8]}"
                )
                
                return AutoMLOptimizationResponse(
                    success=True,
                    dataset_id=dataset_id,
                    optimized_detector_id=detector_id,
                    best_algorithm=result.best_algorithm,
                    best_parameters=result.best_params,
                    best_score=result.best_score,
                    optimization_time_seconds=result.optimization_time,
                    trials_completed=result.trials_completed,
                    algorithm_rankings=result.algorithm_rankings,
                    ensemble_config=result.ensemble_config,
                    optimization_summary=self.automl_service.get_optimization_summary(result),
                    recommendations=self._generate_optimization_recommendations(result)
                )
                
            finally:
                # Restore original trials setting
                if max_trials:
                    self.automl_service.n_trials = original_trials
                    
        except AutoMLError as e:
            logger.error(f"AutoML optimization error: {e}")
            return AutoMLOptimizationResponse(
                success=False,
                dataset_id=dataset_id,
                error_message=str(e),
                recommendations=["Check dataset quality and algorithm compatibility"]
            )
        except Exception as e:
            logger.error(f"Unexpected error during optimization: {e}")
            return AutoMLOptimizationResponse(
                success=False,
                dataset_id=dataset_id,
                error_message=f"Optimization failed: {str(e)}",
                recommendations=["Contact support if the issue persists"]
            )
    
    async def auto_optimize(
        self,
        request: AutoMLOptimizationRequest
    ) -> AutoMLOptimizationResponse:
        """
        Perform complete AutoML optimization including algorithm selection.
        
        Args:
            request: AutoML optimization request
            
        Returns:
            Complete optimization results
        """
        try:
            logger.info(f"Starting full AutoML optimization for dataset {request.dataset_id}")
            
            # Convert objective string to enum
            objective = OptimizationObjective(request.optimization_objective.lower())
            
            # Run complete AutoML
            result = await self.automl_service.auto_select_and_optimize(
                dataset_id=request.dataset_id,
                objective=objective,
                max_algorithms=request.max_algorithms_to_try,
                enable_ensemble=request.enable_ensemble
            )
            
            # Create optimized detector
            detector_name = request.detector_name or f"AutoML_{request.dataset_id[:8]}"
            detector_id = await self.automl_service.create_optimized_detector(
                result,
                detector_name=detector_name
            )
            
            return AutoMLOptimizationResponse(
                success=True,
                dataset_id=request.dataset_id,
                optimized_detector_id=detector_id,
                best_algorithm=result.best_algorithm,
                best_parameters=result.best_params,
                best_score=result.best_score,
                optimization_time_seconds=result.optimization_time,
                trials_completed=result.trials_completed,
                algorithm_rankings=result.algorithm_rankings,
                ensemble_config=result.ensemble_config,
                optimization_summary=self.automl_service.get_optimization_summary(result),
                recommendations=self._generate_optimization_recommendations(result)
            )
            
        except AutoMLError as e:
            logger.error(f"AutoML optimization failed: {e}")
            return AutoMLOptimizationResponse(
                success=False,
                dataset_id=request.dataset_id,
                error_message=str(e),
                recommendations=["Check dataset format and quality"]
            )
        except Exception as e:
            logger.error(f"Unexpected error in AutoML: {e}")
            return AutoMLOptimizationResponse(
                success=False,
                dataset_id=request.dataset_id,
                error_message=f"AutoML failed: {str(e)}",
                recommendations=["Verify dataset accessibility and try again"]
            )
    
    def _generate_recommendation_reasoning(
        self, 
        config, 
        profile, 
        score: float
    ) -> str:
        """Generate human-readable reasoning for algorithm recommendation."""
        reasons = []
        
        # Dataset size reasoning
        if profile.n_samples < config.recommended_min_samples:
            reasons.append("may need more data for optimal performance")
        elif profile.n_samples > config.recommended_max_samples:
            reasons.append("very efficient for large datasets")
        else:
            reasons.append("well-suited for this dataset size")
        
        # Complexity reasoning
        if abs(config.complexity_score - profile.complexity_score) < 0.2:
            reasons.append("complexity matches dataset characteristics")
        elif config.complexity_score > profile.complexity_score:
            reasons.append("provides advanced capabilities for complex patterns")
        else:
            reasons.append("offers computational efficiency")
        
        # Family-specific reasoning
        if config.family.value == "isolation_based":
            reasons.append("excels at isolating anomalies in high-dimensional data")
        elif config.family.value == "density_based":
            reasons.append("effective for detecting density-based outliers")
        elif config.family.value == "neural_networks":
            reasons.append("can learn complex non-linear patterns")
        elif config.family.value == "statistical":
            reasons.append("provides interpretable statistical insights")
        
        return f"Score: {score:.2f} - " + "; ".join(reasons)
    
    def _categorize_dataset_size(self, n_samples: int) -> str:
        """Categorize dataset size."""
        if n_samples < 100:
            return "very_small"
        elif n_samples < 1000:
            return "small"
        elif n_samples < 10000:
            return "medium"
        elif n_samples < 100000:
            return "large"
        else:
            return "very_large"
    
    def _categorize_complexity(self, complexity_score: float) -> str:
        """Categorize dataset complexity."""
        if complexity_score < 0.3:
            return "low"
        elif complexity_score < 0.6:
            return "medium"
        else:
            return "high"
    
    def _generate_optimization_recommendations(self, result: AutoMLResult) -> List[str]:
        """Generate actionable recommendations based on optimization results."""
        recommendations = []
        
        # Performance-based recommendations
        if result.best_score < 0.6:
            recommendations.append("Consider collecting more representative training data")
            recommendations.append("Verify data quality and feature engineering")
        elif result.best_score < 0.8:
            recommendations.append("Try increasing optimization time for better results")
            recommendations.append("Consider feature selection or dimensionality reduction")
        else:
            recommendations.append("Excellent results achieved - ready for deployment")
        
        # Ensemble recommendations
        if result.ensemble_config:
            recommendations.append("Ensemble model created for improved robustness")
        elif len(result.algorithm_rankings) > 1:
            scores = [score for _, score in result.algorithm_rankings]
            if max(scores) - min(scores) < 0.1:
                recommendations.append("Consider ensemble methods for marginal improvement")
        
        # Optimization time recommendations
        if result.optimization_time < 60:
            recommendations.append("Quick optimization completed - consider longer runs for production")
        elif result.optimization_time > 1800:  # 30 minutes
            recommendations.append("Extensive optimization performed - results are well-tuned")
        
        # Algorithm-specific recommendations
        if result.best_algorithm == "IsolationForest":
            recommendations.append("Isolation Forest excels with numerical features and high dimensions")
        elif result.best_algorithm == "LOF":
            recommendations.append("LOF is effective for local density anomalies")
        elif result.best_algorithm in ["AutoEncoder", "VAE"]:
            recommendations.append("Neural network model may benefit from more training data")
        
        return recommendations
    
    def _create_dataset_profile(self, dataset: Dataset):
        """Create a dataset profile from a Dataset entity."""
        from pynomaly.application.services.automl_service import DatasetProfile
        
        df = dataset.features
        n_samples, n_features = df.shape
        
        # Feature types
        feature_types = {}
        categorical_features = []
        numerical_features = []
        time_series_features = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_types[col] = "numerical"
                numerical_features.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_types[col] = "datetime"
                time_series_features.append(col)
            else:
                feature_types[col] = "categorical"
                categorical_features.append(col)
        
        # Data quality metrics
        missing_values_ratio = df.isnull().sum().sum() / (n_samples * n_features)
        
        # Sparsity analysis (for numerical features)
        if numerical_features:
            numerical_df = df[numerical_features]
            zero_ratio = (numerical_df == 0).sum().sum() / (n_samples * len(numerical_features))
            sparsity_ratio = zero_ratio
        else:
            sparsity_ratio = 0.0
        
        # Dimensionality ratio
        dimensionality_ratio = n_features / n_samples
        
        # Contamination estimation using IQR method
        contamination_estimate = 0.1  # Default
        if numerical_features:
            numerical_data = df[numerical_features].select_dtypes(include=[np.number])
            if not numerical_data.empty:
                q1 = numerical_data.quantile(0.25)
                q3 = numerical_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((numerical_data < lower_bound) | (numerical_data > upper_bound)).any(axis=1)
                contamination_estimate = min(outliers.sum() / n_samples, 0.5)
        
        # Dataset size
        dataset_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Temporal structure detection
        has_temporal_structure = len(time_series_features) > 0
        
        # Graph structure detection (heuristic)
        has_graph_structure = False
        if any("edge" in col.lower() or "node" in col.lower() or "graph" in col.lower() for col in df.columns):
            has_graph_structure = True
        
        return DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            contamination_estimate=contamination_estimate,
            feature_types=feature_types,
            missing_values_ratio=missing_values_ratio,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            time_series_features=time_series_features,
            sparsity_ratio=sparsity_ratio,
            dimensionality_ratio=dimensionality_ratio,
            dataset_size_mb=dataset_size_mb,
            has_temporal_structure=has_temporal_structure,
            has_graph_structure=has_graph_structure
        )
"""Configuration Recommendation Service.

This service provides intelligent configuration recommendations based on dataset
characteristics, historical performance, and machine learning models that learn
from successful configuration patterns.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pynomaly.application.dto.configuration_dto import (
    ConfigurationLevel,
    ConfigurationRecommendationDTO,
    ConfigurationSearchRequestDTO,
    ConfigurationSource,
    DatasetCharacteristicsDTO,
    ExperimentConfigurationDTO,
)
from pynomaly.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)
from pynomaly.infrastructure.config.feature_flags import require_feature
from pynomaly.infrastructure.persistence.configuration_repository import (
    ConfigurationRepository,
)

logger = logging.getLogger(__name__)


class ConfigurationRecommendationService:
    """Service for intelligent configuration recommendations."""

    def __init__(
        self,
        configuration_service: ConfigurationCaptureService,
        repository: ConfigurationRepository,
        enable_ml_recommendations: bool = True,
        enable_similarity_recommendations: bool = True,
        enable_performance_prediction: bool = True,
    ):
        """Initialize configuration recommendation service.

        Args:
            configuration_service: Configuration capture service
            repository: Configuration repository
            enable_ml_recommendations: Enable ML-based recommendations
            enable_similarity_recommendations: Enable similarity-based recommendations
            enable_performance_prediction: Enable performance prediction
        """
        self.configuration_service = configuration_service
        self.repository = repository
        self.enable_ml_recommendations = enable_ml_recommendations
        self.enable_similarity_recommendations = enable_similarity_recommendations
        self.enable_performance_prediction = enable_performance_prediction

        # Recommendation statistics
        self.recommendation_stats = {
            "total_recommendations": 0,
            "ml_recommendations": 0,
            "similarity_recommendations": 0,
            "performance_predictions": 0,
            "successful_predictions": 0,
            "model_training_count": 0,
            "last_model_training": None,
        }

        # ML models for recommendations
        self.performance_predictor: RandomForestRegressor | None = None
        self.algorithm_selector: RandomForestClassifier | None = None
        self.feature_scaler: StandardScaler | None = None

        # Dataset characteristics cache
        self._dataset_cache: dict[str, DatasetCharacteristicsDTO] = {}
        self._configuration_cache: dict[UUID, ExperimentConfigurationDTO] = {}
        self._last_cache_update = datetime.min

        # Algorithm performance database
        self.algorithm_performance_db = defaultdict(list)

        # Recommendation rules
        self.recommendation_rules = self._initialize_recommendation_rules()

    @require_feature("advanced_automl")
    async def recommend_configurations(
        self,
        dataset_characteristics: DatasetCharacteristicsDTO,
        performance_requirements: dict[str, float] | None = None,
        use_case: str | None = None,
        difficulty_level: ConfigurationLevel = ConfigurationLevel.INTERMEDIATE,
        max_recommendations: int = 5,
    ) -> list[ConfigurationRecommendationDTO]:
        """Generate configuration recommendations for a dataset.

        Args:
            dataset_characteristics: Characteristics of the target dataset
            performance_requirements: Required performance thresholds
            use_case: Specific use case (e.g., "fraud_detection", "network_monitoring")
            difficulty_level: Desired configuration complexity
            max_recommendations: Maximum number of recommendations

        Returns:
            List of configuration recommendations with confidence scores
        """
        self.recommendation_stats["total_recommendations"] += 1

        logger.info(
            f"Generating recommendations for dataset with {dataset_characteristics.n_samples} samples"
        )

        # Ensure data is loaded
        await self._update_cache_if_needed()

        recommendations = []

        # 1. ML-based recommendations
        if self.enable_ml_recommendations and self.performance_predictor:
            self.recommendation_stats["ml_recommendations"] += 1
            ml_recommendations = await self._generate_ml_recommendations(
                dataset_characteristics,
                performance_requirements,
                max_recommendations // 2,
            )
            recommendations.extend(ml_recommendations)

        # 2. Similarity-based recommendations
        if self.enable_similarity_recommendations:
            self.recommendation_stats["similarity_recommendations"] += 1
            similarity_recommendations = (
                await self._generate_similarity_recommendations(
                    dataset_characteristics,
                    performance_requirements,
                    max_recommendations // 2,
                )
            )
            recommendations.extend(similarity_recommendations)

        # 3. Rule-based recommendations (fallback)
        if not recommendations:
            rule_recommendations = await self._generate_rule_based_recommendations(
                dataset_characteristics,
                performance_requirements,
                use_case,
                difficulty_level,
                max_recommendations,
            )
            recommendations.extend(rule_recommendations)

        # Remove duplicates and score
        unique_recommendations = self._deduplicate_and_score_recommendations(
            recommendations, dataset_characteristics, performance_requirements
        )

        # Sort by confidence and limit results
        unique_recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        final_recommendations = unique_recommendations[:max_recommendations]

        logger.info(
            f"Generated {len(final_recommendations)} configuration recommendations"
        )
        return final_recommendations

    async def predict_configuration_performance(
        self,
        configuration: ExperimentConfigurationDTO,
        dataset_characteristics: DatasetCharacteristicsDTO,
    ) -> dict[str, float]:
        """Predict performance for a configuration on a dataset.

        Args:
            configuration: Configuration to evaluate
            dataset_characteristics: Target dataset characteristics

        Returns:
            Predicted performance metrics
        """
        if not self.enable_performance_prediction or not self.performance_predictor:
            return {}

        self.recommendation_stats["performance_predictions"] += 1

        logger.debug(
            f"Predicting performance for {configuration.algorithm_config.algorithm_name}"
        )

        # Extract features
        features = self._extract_prediction_features(
            configuration, dataset_characteristics
        )

        # Predict performance
        try:
            predicted_accuracy = self.performance_predictor.predict([features])[0]

            # Estimate other metrics based on accuracy
            predicted_precision = predicted_accuracy * 0.95  # Typically slightly lower
            predicted_recall = predicted_accuracy * 1.02  # Can be slightly higher
            predicted_f1 = (
                2
                * (predicted_precision * predicted_recall)
                / (predicted_precision + predicted_recall)
            )

            # Estimate training time based on dataset size and algorithm
            predicted_training_time = self._estimate_training_time(
                configuration, dataset_characteristics
            )

            predictions = {
                "accuracy": max(0.0, min(1.0, predicted_accuracy)),
                "precision": max(0.0, min(1.0, predicted_precision)),
                "recall": max(0.0, min(1.0, predicted_recall)),
                "f1_score": max(0.0, min(1.0, predicted_f1)),
                "training_time_seconds": max(1.0, predicted_training_time),
            }

            self.recommendation_stats["successful_predictions"] += 1
            return predictions

        except Exception as e:
            logger.warning(f"Performance prediction failed: {e}")
            return {}

    async def train_recommendation_models(
        self, min_configurations: int = 20, test_size: float = 0.2
    ) -> dict[str, Any]:
        """Train ML models for recommendations.

        Args:
            min_configurations: Minimum configurations needed for training
            test_size: Fraction of data for testing

        Returns:
            Training results and model performance
        """
        if not self.enable_ml_recommendations:
            return {"error": "ML recommendations disabled"}

        await self._update_cache_if_needed()

        configurations = list(self._configuration_cache.values())

        # Filter configurations with performance results
        valid_configs = [
            config
            for config in configurations
            if config.performance_results
            and config.performance_results.accuracy is not None
        ]

        if len(valid_configs) < min_configurations:
            logger.warning(
                f"Insufficient data for training: {len(valid_configs)} < {min_configurations}"
            )
            return {
                "error": f"Need at least {min_configurations} configurations, got {len(valid_configs)}",
                "available_configurations": len(valid_configs),
            }

        logger.info(
            f"Training recommendation models with {len(valid_configs)} configurations"
        )

        # Extract features and targets
        features_list = []
        accuracies = []
        algorithms = []

        for config in valid_configs:
            if config.dataset_config and config.dataset_config.n_samples:
                # Create dataset characteristics from configuration
                dataset_chars = DatasetCharacteristicsDTO(
                    n_samples=config.dataset_config.n_samples,
                    n_features=config.dataset_config.n_features or 10,
                    feature_types=config.dataset_config.feature_types or [],
                    missing_values_ratio=0.0,
                    outlier_ratio=0.05,  # Default estimation
                    class_imbalance=None,
                    sparsity=0.0,
                    correlation_structure="unknown",
                )

                features = self._extract_prediction_features(config, dataset_chars)
                features_list.append(features)
                accuracies.append(config.performance_results.accuracy)
                algorithms.append(config.algorithm_config.algorithm_name)

        if len(features_list) < min_configurations:
            return {
                "error": f"Insufficient valid features: {len(features_list)} < {min_configurations}",
                "configurations_processed": len(valid_configs),
            }

        # Prepare data
        X = np.array(features_list)
        y_accuracy = np.array(accuracies)
        y_algorithm = np.array(algorithms)

        # Split data
        X_train, X_test, y_acc_train, y_acc_test, y_alg_train, y_alg_test = (
            train_test_split(
                X, y_accuracy, y_algorithm, test_size=test_size, random_state=42
            )
        )

        # Scale features
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # Train performance predictor
        self.performance_predictor = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.performance_predictor.fit(X_train_scaled, y_acc_train)

        # Train algorithm selector
        self.algorithm_selector = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.algorithm_selector.fit(X_train_scaled, y_alg_train)

        # Evaluate models
        y_acc_pred = self.performance_predictor.predict(X_test_scaled)
        y_alg_pred = self.algorithm_selector.predict(X_test_scaled)

        accuracy_mse = mean_squared_error(y_acc_test, y_acc_pred)
        algorithm_accuracy = accuracy_score(y_alg_test, y_alg_pred)

        # Update statistics
        self.recommendation_stats["model_training_count"] += 1
        self.recommendation_stats["last_model_training"] = datetime.now().isoformat()

        training_results = {
            "success": True,
            "configurations_used": len(valid_configs),
            "features_extracted": len(features_list),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "performance_predictor_mse": accuracy_mse,
            "performance_predictor_rmse": math.sqrt(accuracy_mse),
            "algorithm_selector_accuracy": algorithm_accuracy,
            "unique_algorithms": len(set(algorithms)),
            "feature_importance": self._get_feature_importance(),
        }

        logger.info(
            f"Model training completed: RMSE={training_results['performance_predictor_rmse']:.3f}, "
            f"Algorithm Accuracy={algorithm_accuracy:.3f}"
        )

        return training_results

    async def analyze_recommendation_patterns(
        self, time_period_days: int = 30
    ) -> dict[str, Any]:
        """Analyze recommendation patterns and effectiveness.

        Args:
            time_period_days: Time period for analysis

        Returns:
            Analysis results
        """
        await self._update_cache_if_needed()

        # Filter recent configurations
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        recent_configs = [
            config
            for config in self._configuration_cache.values()
            if config.metadata.created_at >= cutoff_date
        ]

        logger.info(f"Analyzing {len(recent_configs)} recent configurations")

        # Algorithm popularity
        algorithm_counts = Counter(
            config.algorithm_config.algorithm_name for config in recent_configs
        )

        # Performance analysis by algorithm
        algorithm_performance = defaultdict(list)
        for config in recent_configs:
            if config.performance_results and config.performance_results.accuracy:
                algorithm_performance[config.algorithm_config.algorithm_name].append(
                    config.performance_results.accuracy
                )

        # Dataset size patterns
        dataset_sizes = []
        for config in recent_configs:
            if config.dataset_config and config.dataset_config.n_samples:
                dataset_sizes.append(config.dataset_config.n_samples)

        # Source analysis
        source_counts = Counter(config.metadata.source for config in recent_configs)

        # Performance distribution
        accuracies = [
            config.performance_results.accuracy
            for config in recent_configs
            if config.performance_results and config.performance_results.accuracy
        ]

        analysis = {
            "time_period_days": time_period_days,
            "total_configurations": len(recent_configs),
            "algorithm_popularity": dict(algorithm_counts.most_common(10)),
            "algorithm_performance": {
                alg: {
                    "count": len(perfs),
                    "mean_accuracy": np.mean(perfs),
                    "std_accuracy": np.std(perfs),
                    "min_accuracy": np.min(perfs),
                    "max_accuracy": np.max(perfs),
                }
                for alg, perfs in algorithm_performance.items()
                if len(perfs) > 0
            },
            "dataset_characteristics": {
                "sample_count": len(dataset_sizes),
                "mean_samples": np.mean(dataset_sizes) if dataset_sizes else 0,
                "median_samples": np.median(dataset_sizes) if dataset_sizes else 0,
                "size_distribution": self._analyze_size_distribution(dataset_sizes),
            },
            "source_distribution": dict(source_counts),
            "performance_summary": {
                "count": len(accuracies),
                "mean_accuracy": np.mean(accuracies) if accuracies else 0,
                "median_accuracy": np.median(accuracies) if accuracies else 0,
                "std_accuracy": np.std(accuracies) if accuracies else 0,
                "high_performance_rate": (
                    len([a for a in accuracies if a > 0.85]) / len(accuracies)
                    if accuracies
                    else 0
                ),
            },
            "recommendation_trends": self._analyze_recommendation_trends(
                recent_configs
            ),
        }

        return analysis

    def get_recommendation_statistics(self) -> dict[str, Any]:
        """Get recommendation service statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "recommendation_stats": self.recommendation_stats,
            "model_status": {
                "performance_predictor_trained": self.performance_predictor is not None,
                "algorithm_selector_trained": self.algorithm_selector is not None,
                "feature_scaler_available": self.feature_scaler is not None,
            },
            "cache_info": {
                "cached_datasets": len(self._dataset_cache),
                "cached_configurations": len(self._configuration_cache),
                "last_cache_update": self._last_cache_update.isoformat(),
            },
            "feature_flags": {
                "ml_recommendations": self.enable_ml_recommendations,
                "similarity_recommendations": self.enable_similarity_recommendations,
                "performance_prediction": self.enable_performance_prediction,
            },
        }

    # Private methods

    async def _update_cache_if_needed(self) -> None:
        """Update cache if it's stale."""
        now = datetime.now()
        if (now - self._last_cache_update).total_seconds() > 600:  # 10 minutes
            await self._rebuild_cache()

    async def _rebuild_cache(self) -> None:
        """Rebuild configuration cache."""
        logger.debug("Rebuilding configuration cache")

        # Get all configurations
        search_request = ConfigurationSearchRequestDTO(limit=10000)
        configurations = await self.repository.search_configurations(search_request)

        # Update cache
        self._configuration_cache = {config.id: config for config in configurations}
        self._last_cache_update = datetime.now()

        logger.debug(
            f"Cache rebuilt with {len(self._configuration_cache)} configurations"
        )

    async def _generate_ml_recommendations(
        self,
        dataset_characteristics: DatasetCharacteristicsDTO,
        performance_requirements: dict[str, float] | None,
        max_recommendations: int,
    ) -> list[ConfigurationRecommendationDTO]:
        """Generate ML-based recommendations."""
        if not self.algorithm_selector or not self.feature_scaler:
            return []

        recommendations = []

        # Get algorithm probabilities
        dummy_config = self._create_dummy_configuration(
            "dummy", dataset_characteristics
        )
        features = self._extract_prediction_features(
            dummy_config, dataset_characteristics
        )
        features_scaled = self.feature_scaler.transform([features])

        # Get algorithm probabilities
        algorithm_probs = self.algorithm_selector.predict_proba(features_scaled)[0]
        algorithm_classes = self.algorithm_selector.classes_

        # Sort by probability
        sorted_indices = np.argsort(algorithm_probs)[::-1]

        for _i, idx in enumerate(sorted_indices[:max_recommendations]):
            algorithm = algorithm_classes[idx]
            confidence = algorithm_probs[idx]

            # Find similar configurations with this algorithm
            similar_configs = [
                config
                for config in self._configuration_cache.values()
                if config.algorithm_config.algorithm_name == algorithm
            ]

            if similar_configs:
                # Use best performing configuration as template
                best_config = max(
                    similar_configs,
                    key=lambda c: (
                        c.performance_results.accuracy
                        if c.performance_results and c.performance_results.accuracy
                        else 0
                    ),
                )

                # Predict performance
                predicted_performance = await self.predict_configuration_performance(
                    best_config, dataset_characteristics
                )

                recommendation = ConfigurationRecommendationDTO(
                    configuration_id=best_config.id,
                    algorithm_name=algorithm,
                    confidence_score=confidence,
                    predicted_performance=predicted_performance,
                    recommendation_reason=f"ML model suggests {algorithm} with {confidence:.1%} confidence",
                    use_cases=[],
                    difficulty_level=best_config.metadata.difficulty_level
                    or ConfigurationLevel.INTERMEDIATE,
                    estimated_training_time=predicted_performance.get(
                        "training_time_seconds", 60
                    ),
                    parameter_suggestions=best_config.algorithm_config.hyperparameters
                    or {},
                    preprocessing_suggestions=self._get_preprocessing_suggestions(
                        dataset_characteristics
                    ),
                    evaluation_suggestions={"cross_validation": True, "folds": 5},
                )

                recommendations.append(recommendation)

        return recommendations

    async def _generate_similarity_recommendations(
        self,
        dataset_characteristics: DatasetCharacteristicsDTO,
        performance_requirements: dict[str, float] | None,
        max_recommendations: int,
    ) -> list[ConfigurationRecommendationDTO]:
        """Generate similarity-based recommendations."""
        recommendations = []

        # Find configurations with similar dataset characteristics
        similar_configs = []

        for config in self._configuration_cache.values():
            if not config.dataset_config:
                continue

            # Calculate dataset similarity
            similarity = self._calculate_dataset_similarity(
                dataset_characteristics, config.dataset_config
            )

            if similarity > 0.7:  # Similarity threshold
                similar_configs.append((config, similarity))

        # Sort by similarity and performance
        similar_configs.sort(
            key=lambda x: (
                x[1],
                (
                    x[0].performance_results.accuracy
                    if x[0].performance_results and x[0].performance_results.accuracy
                    else 0
                ),
            ),
            reverse=True,
        )

        for config, similarity in similar_configs[:max_recommendations]:
            predicted_performance = await self.predict_configuration_performance(
                config, dataset_characteristics
            )

            recommendation = ConfigurationRecommendationDTO(
                configuration_id=config.id,
                algorithm_name=config.algorithm_config.algorithm_name,
                confidence_score=similarity
                * 0.8,  # Scale down similarity-based confidence
                predicted_performance=predicted_performance,
                recommendation_reason=f"Similar dataset characteristics (similarity: {similarity:.1%})",
                use_cases=[],
                difficulty_level=config.metadata.difficulty_level
                or ConfigurationLevel.INTERMEDIATE,
                estimated_training_time=predicted_performance.get(
                    "training_time_seconds", 60
                ),
                parameter_suggestions=config.algorithm_config.hyperparameters or {},
                preprocessing_suggestions=self._get_preprocessing_suggestions(
                    dataset_characteristics
                ),
                evaluation_suggestions={"cross_validation": True, "folds": 5},
            )

            recommendations.append(recommendation)

        return recommendations

    async def _generate_rule_based_recommendations(
        self,
        dataset_characteristics: DatasetCharacteristicsDTO,
        performance_requirements: dict[str, float] | None,
        use_case: str | None,
        difficulty_level: ConfigurationLevel,
        max_recommendations: int,
    ) -> list[ConfigurationRecommendationDTO]:
        """Generate rule-based recommendations as fallback."""
        recommendations = []

        # Apply recommendation rules
        for rule in self.recommendation_rules:
            if rule["condition"](dataset_characteristics, use_case):
                config_template = rule["configuration"]

                recommendation = ConfigurationRecommendationDTO(
                    configuration_id=None,  # Template recommendation
                    algorithm_name=config_template["algorithm"],
                    confidence_score=rule["confidence"],
                    predicted_performance=rule.get("expected_performance", {}),
                    recommendation_reason=rule["reason"],
                    use_cases=rule.get("use_cases", []),
                    difficulty_level=difficulty_level,
                    estimated_training_time=rule.get("training_time", 60),
                    parameter_suggestions=config_template.get("parameters", {}),
                    preprocessing_suggestions=self._get_preprocessing_suggestions(
                        dataset_characteristics
                    ),
                    evaluation_suggestions={"cross_validation": True, "folds": 5},
                )

                recommendations.append(recommendation)

                if len(recommendations) >= max_recommendations:
                    break

        return recommendations

    def _initialize_recommendation_rules(self) -> list[dict[str, Any]]:
        """Initialize rule-based recommendation system."""
        return [
            {
                "condition": lambda ds, uc: ds.n_samples < 1000,
                "configuration": {
                    "algorithm": "IsolationForest",
                    "parameters": {"n_estimators": 50},
                },
                "confidence": 0.7,
                "reason": "Small dataset - IsolationForest works well with limited data",
                "expected_performance": {"accuracy": 0.75},
                "training_time": 30,
            },
            {
                "condition": lambda ds, uc: ds.n_samples > 10000,
                "configuration": {
                    "algorithm": "LocalOutlierFactor",
                    "parameters": {"n_neighbors": 20},
                },
                "confidence": 0.8,
                "reason": "Large dataset - LOF scales well and provides good local density estimation",
                "expected_performance": {"accuracy": 0.82},
                "training_time": 120,
            },
            {
                "condition": lambda ds, uc: ds.n_features > 50,
                "configuration": {
                    "algorithm": "PCA",
                    "parameters": {"n_components": 0.95},
                },
                "confidence": 0.6,
                "reason": "High-dimensional data - PCA preprocessing recommended",
                "expected_performance": {"accuracy": 0.78},
                "training_time": 90,
            },
            {
                "condition": lambda ds, uc: uc == "fraud_detection",
                "configuration": {
                    "algorithm": "IsolationForest",
                    "parameters": {"contamination": 0.001},
                },
                "confidence": 0.9,
                "reason": "Fraud detection - IsolationForest with low contamination rate",
                "use_cases": ["fraud_detection", "financial_anomalies"],
                "expected_performance": {"accuracy": 0.85},
                "training_time": 60,
            },
            {
                "condition": lambda ds, uc: ds.sparsity and ds.sparsity > 0.5,
                "configuration": {
                    "algorithm": "OneClassSVM",
                    "parameters": {"nu": 0.05, "kernel": "linear"},
                },
                "confidence": 0.7,
                "reason": "Sparse data - One-Class SVM with linear kernel handles sparsity well",
                "expected_performance": {"accuracy": 0.76},
                "training_time": 150,
            },
        ]

    def _extract_prediction_features(
        self,
        configuration: ExperimentConfigurationDTO,
        dataset_characteristics: DatasetCharacteristicsDTO,
    ) -> list[float]:
        """Extract features for ML prediction."""
        features = []

        # Dataset features
        features.extend(
            [
                math.log(dataset_characteristics.n_samples + 1),
                math.log(dataset_characteristics.n_features + 1),
                dataset_characteristics.missing_values_ratio or 0.0,
                dataset_characteristics.outlier_ratio or 0.05,
                dataset_characteristics.sparsity or 0.0,
            ]
        )

        # Algorithm features (one-hot encoding for common algorithms)
        common_algorithms = [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
            "EllipticEnvelope",
        ]
        for algo in common_algorithms:
            features.append(
                1.0 if configuration.algorithm_config.algorithm_name == algo else 0.0
            )

        # Hyperparameter features
        params = configuration.algorithm_config.hyperparameters or {}
        features.extend(
            [
                params.get("contamination", 0.1),
                params.get("n_estimators", 100) / 100.0,  # Normalize
                params.get("n_neighbors", 20) / 20.0,  # Normalize
                len(params),  # Number of hyperparameters
            ]
        )

        # Preprocessing features
        preprocessing_config = configuration.preprocessing_config
        if preprocessing_config:
            features.extend(
                [
                    1.0 if preprocessing_config.apply_pca else 0.0,
                    1.0 if preprocessing_config.scaling_method else 0.0,
                    1.0 if preprocessing_config.handle_missing_values else 0.0,
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0])

        return features

    def _create_dummy_configuration(
        self, algorithm: str, dataset_characteristics: DatasetCharacteristicsDTO
    ) -> ExperimentConfigurationDTO:
        """Create dummy configuration for feature extraction."""
        from pynomaly.application.dto.configuration_dto import (
            AlgorithmConfigurationDTO,
            ConfigurationMetadataDTO,
            DatasetConfigurationDTO,
        )

        return ExperimentConfigurationDTO(
            id=UUID("00000000-0000-0000-0000-000000000000"),
            name="dummy",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name=algorithm, hyperparameters={}
            ),
            dataset_config=DatasetConfigurationDTO(
                n_samples=dataset_characteristics.n_samples,
                n_features=dataset_characteristics.n_features,
                feature_types=dataset_characteristics.feature_types,
            ),
            metadata=ConfigurationMetadataDTO(
                source=ConfigurationSource.MANUAL,
                created_at=datetime.now(),
                description="dummy",
                tags=[],
            ),
        )

    def _calculate_dataset_similarity(
        self, ds1: DatasetCharacteristicsDTO, ds2: DatasetConfigurationDTO
    ) -> float:
        """Calculate similarity between dataset characteristics."""
        similarity = 0.0

        # Sample size similarity (log scale)
        if ds2.n_samples:
            size_ratio = min(ds1.n_samples, ds2.n_samples) / max(
                ds1.n_samples, ds2.n_samples
            )
            similarity += 0.3 * size_ratio

        # Feature count similarity
        if ds2.n_features:
            feature_ratio = min(ds1.n_features, ds2.n_features) / max(
                ds1.n_features, ds2.n_features
            )
            similarity += 0.3 * feature_ratio

        # Feature type similarity
        if ds1.feature_types and ds2.feature_types:
            common_types = set(ds1.feature_types).intersection(set(ds2.feature_types))
            all_types = set(ds1.feature_types).union(set(ds2.feature_types))
            if all_types:
                similarity += 0.2 * (len(common_types) / len(all_types))

        # Missing values similarity
        mv1 = ds1.missing_values_ratio or 0.0
        mv2 = 0.0  # Default for dataset config
        similarity += 0.1 * (1.0 - abs(mv1 - mv2))

        # Sparsity similarity
        sp1 = ds1.sparsity or 0.0
        sp2 = 0.0  # Default for dataset config
        similarity += 0.1 * (1.0 - abs(sp1 - sp2))

        return similarity

    def _get_preprocessing_suggestions(
        self, dataset_characteristics: DatasetCharacteristicsDTO
    ) -> dict[str, Any]:
        """Get preprocessing suggestions based on dataset characteristics."""
        suggestions = {}

        # Scaling
        suggestions["scaling"] = "standard"

        # PCA for high-dimensional data
        if dataset_characteristics.n_features > 50:
            suggestions["pca"] = {"n_components": 0.95}

        # Missing value handling
        if (
            dataset_characteristics.missing_values_ratio
            and dataset_characteristics.missing_values_ratio > 0.01
        ):
            suggestions["missing_values"] = "impute"

        # Outlier handling for sparse data
        if dataset_characteristics.sparsity and dataset_characteristics.sparsity > 0.3:
            suggestions["outlier_handling"] = "clip"

        return suggestions

    def _estimate_training_time(
        self,
        configuration: ExperimentConfigurationDTO,
        dataset_characteristics: DatasetCharacteristicsDTO,
    ) -> float:
        """Estimate training time for configuration."""
        # Base time by algorithm
        algorithm_base_times = {
            "IsolationForest": 30,
            "LocalOutlierFactor": 60,
            "OneClassSVM": 120,
            "EllipticEnvelope": 45,
        }

        base_time = algorithm_base_times.get(
            configuration.algorithm_config.algorithm_name, 60
        )

        # Scale by dataset size
        size_factor = math.log(dataset_characteristics.n_samples + 1) / math.log(1000)
        feature_factor = math.log(dataset_characteristics.n_features + 1) / math.log(10)

        estimated_time = base_time * size_factor * feature_factor

        return max(10.0, estimated_time)  # Minimum 10 seconds

    def _get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from trained models."""
        if not self.performance_predictor:
            return {}

        feature_names = [
            "log_n_samples",
            "log_n_features",
            "missing_values_ratio",
            "outlier_ratio",
            "sparsity",
            "is_isolation_forest",
            "is_lof",
            "is_one_class_svm",
            "is_elliptic_envelope",
            "contamination",
            "n_estimators_norm",
            "n_neighbors_norm",
            "n_hyperparams",
            "apply_pca",
            "has_scaling",
            "handle_missing",
        ]

        importances = self.performance_predictor.feature_importances_

        return dict(zip(feature_names[: len(importances)], importances, strict=False))

    def _deduplicate_and_score_recommendations(
        self,
        recommendations: list[ConfigurationRecommendationDTO],
        dataset_characteristics: DatasetCharacteristicsDTO,
        performance_requirements: dict[str, float] | None,
    ) -> list[ConfigurationRecommendationDTO]:
        """Remove duplicate recommendations and adjust scores."""
        # Group by algorithm
        algorithm_groups = defaultdict(list)
        for rec in recommendations:
            algorithm_groups[rec.algorithm_name].append(rec)

        # Keep best recommendation per algorithm
        unique_recommendations = []
        for _algorithm, group in algorithm_groups.items():
            best_rec = max(group, key=lambda r: r.confidence_score)

            # Adjust confidence based on performance requirements
            if performance_requirements:
                predicted_perf = best_rec.predicted_performance
                if predicted_perf and "accuracy" in predicted_perf:
                    required_acc = performance_requirements.get("min_accuracy", 0.7)
                    predicted_acc = predicted_perf["accuracy"]

                    if predicted_acc >= required_acc:
                        best_rec.confidence_score *= 1.1  # Boost confidence
                    else:
                        best_rec.confidence_score *= 0.8  # Reduce confidence

            unique_recommendations.append(best_rec)

        return unique_recommendations

    def _analyze_size_distribution(self, sizes: list[int]) -> dict[str, int]:
        """Analyze dataset size distribution."""
        if not sizes:
            return {}

        distribution = {
            "small": len([s for s in sizes if s < 1000]),
            "medium": len([s for s in sizes if 1000 <= s < 10000]),
            "large": len([s for s in sizes if s >= 10000]),
        }

        return distribution

    def _analyze_recommendation_trends(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> dict[str, Any]:
        """Analyze recommendation trends from configurations."""
        trends = {
            "growing_algorithms": [],
            "declining_algorithms": [],
            "emerging_patterns": [],
        }

        # Algorithm usage over time
        algorithm_timeline = defaultdict(list)
        for config in configurations:
            algorithm_timeline[config.algorithm_config.algorithm_name].append(
                config.metadata.created_at
            )

        # Identify trends (simplified)
        for algorithm, timestamps in algorithm_timeline.items():
            if len(timestamps) >= 3:
                # Sort by time
                timestamps.sort()

                # Compare first half vs second half
                mid_point = len(timestamps) // 2
                first_half = timestamps[:mid_point]
                second_half = timestamps[mid_point:]

                if len(second_half) > len(first_half) * 1.5:
                    trends["growing_algorithms"].append(algorithm)
                elif len(second_half) < len(first_half) * 0.5:
                    trends["declining_algorithms"].append(algorithm)

        return trends

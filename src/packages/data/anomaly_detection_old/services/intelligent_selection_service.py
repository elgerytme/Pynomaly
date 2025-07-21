"""Intelligent algorithm selection service with learning capabilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from monorepo.application.dto.selection_dto import (
    AlgorithmBenchmarkDTO,
    AlgorithmPerformanceDTO,
    DatasetCharacteristicsDTO,
    LearningInsightsDTO,
    MetaLearningConfigDTO,
    OptimizationConstraintsDTO,
    SelectionHistoryDTO,
    SelectionRecommendationDTO,
)
# TODO: Create local Dataset entity

logger = logging.getLogger(__name__)


class IntelligentSelectionService:
    """Service for intelligent algorithm selection with learning capabilities."""

    def __init__(
        self,
        enable_meta_learning: bool = True,
        enable_performance_prediction: bool = True,
        enable_historical_learning: bool = True,
        selection_history_path: Path | None = None,
        meta_model_path: Path | None = None,
    ):
        """Initialize intelligent selection service.

        Args:
            enable_meta_learning: Enable meta-learning for algorithm selection
            enable_performance_prediction: Enable performance prediction
            enable_historical_learning: Enable learning from historical data
            selection_history_path: Path to store selection history
            meta_model_path: Path to store meta-learning models
        """
        self.enable_meta_learning = enable_meta_learning
        self.enable_performance_prediction = enable_performance_prediction
        self.enable_historical_learning = enable_historical_learning

        # Storage paths
        self.selection_history_path = selection_history_path or Path(
            "data/selection_history.json"
        )
        self.meta_model_path = meta_model_path or Path("data/meta_models")

        # Initialize components
        self.selection_history: list[SelectionHistoryDTO] = []
        self.meta_learner: RandomForestClassifier | None = None
        self.performance_predictor: RandomForestClassifier | None = None
        self.scaler = StandardScaler()

        # Algorithm registry
        self.algorithm_registry = self._initialize_algorithm_registry()

        # Load existing data (handled lazily)
        self._selection_history_loaded = False
        self._meta_models_loaded = False

    async def _ensure_data_loaded(self) -> None:
        """Ensure selection history and meta models are loaded."""
        if not self._selection_history_loaded:
            await self._load_selection_history()
            self._selection_history_loaded = True

        if not self._meta_models_loaded:
            await self._load_meta_models()
            self._meta_models_loaded = True

    async def recommend_algorithm(
        self,
        dataset: Dataset,
        constraints: OptimizationConstraintsDTO | None = None,
        config: MetaLearningConfigDTO | None = None,
    ) -> SelectionRecommendationDTO:
        """Recommend optimal algorithm for dataset.

        Args:
            dataset: Target dataset
            constraints: Optimization constraints
            config: Meta-learning configuration

        Returns:
            Algorithm selection recommendation
        """
        logger.info(f"Generating algorithm recommendation for dataset: {dataset.name}")

        # Ensure data is loaded
        await self._ensure_data_loaded()

        # Extract dataset characteristics
        characteristics = await self._extract_dataset_characteristics(dataset)

        # Get algorithm candidates
        candidates = self._filter_algorithm_candidates(characteristics, constraints)

        # Generate recommendations using different approaches
        recommendations = []

        # Meta-learning recommendation
        if self.enable_meta_learning and self.meta_learner is not None:
            meta_rec = await self._meta_learning_recommendation(
                characteristics, candidates, config
            )
            recommendations.append(meta_rec)

        # Historical similarity recommendation
        if self.enable_historical_learning:
            historical_rec = await self._historical_similarity_recommendation(
                characteristics, candidates
            )
            recommendations.append(historical_rec)

        # Rule-based recommendation
        rule_rec = await self._rule_based_recommendation(characteristics, candidates)
        recommendations.append(rule_rec)

        # Ensemble recommendation
        final_recommendation = await self._ensemble_recommendations(
            recommendations, characteristics, constraints
        )

        # Predict performance for top recommendations
        if self.enable_performance_prediction:
            final_recommendation = await self._predict_performance(
                final_recommendation, characteristics
            )

        return final_recommendation

    async def learn_from_result(
        self,
        dataset: Dataset,
        algorithm: str,
        performance: AlgorithmPerformanceDTO,
        selection_context: dict[str, Any],
    ) -> None:
        """Learn from algorithm selection result.

        Args:
            dataset: Dataset used
            algorithm: Algorithm selected
            performance: Achieved performance
            selection_context: Context of selection decision
        """
        logger.info(f"Learning from result: {algorithm} on {dataset.name}")

        # Ensure data is loaded
        await self._ensure_data_loaded()

        # Create history entry
        characteristics = await self._extract_dataset_characteristics(dataset)

        history_entry = SelectionHistoryDTO(
            dataset_characteristics=characteristics,
            selected_algorithm=algorithm,
            performance=performance,
            selection_context=selection_context,
            timestamp=datetime.now(),
            dataset_hash=self._compute_dataset_hash(dataset),
        )

        # Add to history
        self.selection_history.append(history_entry)

        # Update meta-learning models
        if self.enable_meta_learning:
            await self._update_meta_models()

        # Save updated history
        await self._save_selection_history()

    async def benchmark_algorithms(
        self,
        dataset: Dataset,
        algorithms: list[str] | None = None,
        cv_folds: int = 3,
        constraints: OptimizationConstraintsDTO | None = None,
    ) -> list[AlgorithmBenchmarkDTO]:
        """Benchmark algorithms on dataset.

        Args:
            dataset: Dataset to benchmark on
            algorithms: Algorithms to benchmark (all if None)
            cv_folds: Cross-validation folds
            constraints: Resource constraints

        Returns:
            Benchmark results for each algorithm
        """
        logger.info(f"Benchmarking algorithms on dataset: {dataset.name}")

        if algorithms is None:
            characteristics = await self._extract_dataset_characteristics(dataset)
            algorithms = self._filter_algorithm_candidates(characteristics, constraints)

        benchmarks = []

        for algorithm in algorithms:
            try:
                benchmark = await self._benchmark_single_algorithm(
                    dataset, algorithm, cv_folds, constraints
                )
                benchmarks.append(benchmark)

            except Exception as e:
                logger.warning(f"Failed to benchmark {algorithm}: {e}")
                continue

        # Sort by performance
        benchmarks.sort(key=lambda x: x.mean_score, reverse=True)

        return benchmarks

    async def get_learning_insights(self, min_samples: int = 10) -> LearningInsightsDTO:
        """Get insights from learning history.

        Args:
            min_samples: Minimum samples for reliable insights

        Returns:
            Learning insights and trends
        """
        logger.info("Generating learning insights from selection history")

        # Ensure data is loaded
        await self._ensure_data_loaded()

        if len(self.selection_history) < min_samples:
            logger.warning(
                f"Insufficient history for insights: {len(self.selection_history)} < {min_samples}"
            )

        # Algorithm performance analysis
        algorithm_stats = self._analyze_algorithm_performance()

        # Dataset type preferences
        dataset_preferences = self._analyze_dataset_preferences()

        # Performance trends
        trends = self._analyze_performance_trends()

        # Feature importance
        feature_importance = await self._analyze_feature_importance()

        return LearningInsightsDTO(
            total_selections=len(self.selection_history),
            algorithm_performance_stats=algorithm_stats,
            dataset_type_preferences=dataset_preferences,
            performance_trends=trends,
            feature_importance_insights=feature_importance,
            meta_model_accuracy=await self._get_meta_model_accuracy(),
            recommendation_confidence=self._calculate_recommendation_confidence(),
            generated_at=datetime.now(),
        )

    def get_service_info(self) -> dict[str, Any]:
        """Get service information and status.

        Returns:
            Service information dictionary
        """
        return {
            "meta_learning_enabled": self.enable_meta_learning,
            "performance_prediction_enabled": self.enable_performance_prediction,
            "historical_learning_enabled": self.enable_historical_learning,
            "selection_history_size": len(self.selection_history),
            "meta_model_trained": self.meta_learner is not None,
            "performance_predictor_trained": self.performance_predictor is not None,
            "available_algorithms": list(self.algorithm_registry.keys()),
            "algorithm_count": len(self.algorithm_registry),
            "history_path": str(self.selection_history_path),
            "model_path": str(self.meta_model_path),
        }

    # Private methods

    async def _extract_dataset_characteristics(
        self, dataset: Dataset
    ) -> DatasetCharacteristicsDTO:
        """Extract comprehensive dataset characteristics."""
        data = dataset.data
        if hasattr(data, "values"):
            X = data.values
        else:
            X = np.array(data)

        n_samples, n_features = X.shape

        # Basic statistics
        numeric_features = []
        categorical_features = []

        if hasattr(data, "dtypes"):
            for col, dtype in data.dtypes.items():
                if np.issubdtype(dtype, np.number):
                    numeric_features.append(col)
                else:
                    categorical_features.append(col)
        else:
            numeric_features = list(range(n_features))

        # Calculate characteristics
        characteristics = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_numeric_features": len(numeric_features),
            "n_categorical_features": len(categorical_features),
            "feature_density": np.count_nonzero(X) / X.size,
            "mean_feature_correlation": 0.0,
            "data_dimensionality_ratio": n_features / n_samples,
            "missing_value_ratio": 0.0,
            "outlier_ratio": 0.0,
            "class_imbalance": 0.0,
            "feature_variance_ratio": 0.0,
            "skewness_mean": 0.0,
            "kurtosis_mean": 0.0,
        }

        # Advanced statistics for numeric data
        if len(numeric_features) > 0:
            numeric_data = (
                X if len(categorical_features) == 0 else data[numeric_features].values
            )

            # Correlation analysis
            if numeric_data.shape[1] > 1:
                corr_matrix = np.corrcoef(numeric_data.T)
                characteristics["mean_feature_correlation"] = np.mean(
                    np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                )

            # Missing values
            characteristics["missing_value_ratio"] = (
                np.isnan(numeric_data).sum() / numeric_data.size
            )

            # Outlier detection (IQR method)
            Q1 = np.nanpercentile(numeric_data, 25, axis=0)
            Q3 = np.nanpercentile(numeric_data, 75, axis=0)
            IQR = Q3 - Q1
            outlier_mask = (numeric_data < (Q1 - 1.5 * IQR)) | (
                numeric_data > (Q3 + 1.5 * IQR)
            )
            characteristics["outlier_ratio"] = np.sum(outlier_mask) / numeric_data.size

            # Feature variance
            variances = np.nanvar(numeric_data, axis=0)
            characteristics["feature_variance_ratio"] = (
                np.std(variances) / np.mean(variances) if np.mean(variances) > 0 else 0
            )

            # Distribution characteristics
            from scipy import stats

            try:
                skewness_values = [
                    stats.skew(numeric_data[:, i][~np.isnan(numeric_data[:, i])])
                    for i in range(numeric_data.shape[1])
                ]
                kurtosis_values = [
                    stats.kurtosis(numeric_data[:, i][~np.isnan(numeric_data[:, i])])
                    for i in range(numeric_data.shape[1])
                ]
                characteristics["skewness_mean"] = np.mean(skewness_values)
                characteristics["kurtosis_mean"] = np.mean(kurtosis_values)
            except ImportError:
                pass  # scipy not available

        return DatasetCharacteristicsDTO(**characteristics)

    def _filter_algorithm_candidates(
        self,
        characteristics: DatasetCharacteristicsDTO,
        constraints: OptimizationConstraintsDTO | None,
    ) -> list[str]:
        """Filter algorithm candidates based on characteristics and constraints."""
        candidates = []

        for algo_name, algo_info in self.algorithm_registry.items():
            # Check data size compatibility
            if characteristics.n_samples < algo_info["min_samples"]:
                continue

            if characteristics.n_features > algo_info["max_features"]:
                continue

            # Check constraints
            if constraints:
                if (
                    constraints.max_memory_mb
                    and algo_info["memory_usage"] > constraints.max_memory_mb
                ):
                    continue

                if (
                    constraints.max_training_time_seconds
                    and algo_info["training_time"]
                    > constraints.max_training_time_seconds
                ):
                    continue

            candidates.append(algo_name)

        return candidates

    async def _meta_learning_recommendation(
        self,
        characteristics: DatasetCharacteristicsDTO,
        candidates: list[str],
        config: MetaLearningConfigDTO | None,
    ) -> dict[str, float]:
        """Generate meta-learning based recommendation."""
        if self.meta_learner is None:
            return dict.fromkeys(candidates, 0.0)

        # Convert characteristics to feature vector
        feature_vector = self._characteristics_to_features(characteristics)
        feature_vector = self.scaler.transform([feature_vector])

        # Get algorithm probabilities
        algorithm_probs = {}
        for algo in candidates:
            if algo in self.meta_learner.classes_:
                prob = self.meta_learner.predict_proba(feature_vector)[0]
                algo_idx = list(self.meta_learner.classes_).index(algo)
                algorithm_probs[algo] = prob[algo_idx]
            else:
                algorithm_probs[algo] = 0.0

        return algorithm_probs

    async def _historical_similarity_recommendation(
        self, characteristics: DatasetCharacteristicsDTO, candidates: list[str]
    ) -> dict[str, float]:
        """Generate recommendation based on historical similarity."""
        if not self.selection_history:
            return {algo: 1.0 / len(candidates) for algo in candidates}

        # Convert current characteristics to feature vector
        current_features = self._characteristics_to_features(characteristics)

        # Calculate similarity to historical datasets
        similarities = []
        performances = []
        algorithms = []

        for history in self.selection_history:
            hist_features = self._characteristics_to_features(
                history.dataset_characteristics
            )
            similarity = cosine_similarity([current_features], [hist_features])[0][0]

            similarities.append(similarity)
            performances.append(history.performance.primary_metric)
            algorithms.append(history.selected_algorithm)

        # Weight by similarity and performance
        algorithm_scores = {}
        for algo in candidates:
            scores = []
            weights = []

            for i, hist_algo in enumerate(algorithms):
                if hist_algo == algo:
                    scores.append(performances[i])
                    weights.append(similarities[i])

            if scores:
                weighted_score = np.average(scores, weights=weights)
                algorithm_scores[algo] = weighted_score
            else:
                algorithm_scores[algo] = 0.5  # Default score

        # Normalize scores
        max_score = max(algorithm_scores.values()) if algorithm_scores else 1.0
        if max_score > 0:
            algorithm_scores = {k: v / max_score for k, v in algorithm_scores.items()}

        return algorithm_scores

    async def _rule_based_recommendation(
        self, characteristics: DatasetCharacteristicsDTO, candidates: list[str]
    ) -> dict[str, float]:
        """Generate rule-based recommendation."""
        scores = {}

        for algo in candidates:
            score = 0.5  # Base score

            # Dataset size rules
            if characteristics.n_samples < 1000:
                if algo in ["isolation_forest", "local_outlier_factor"]:
                    score += 0.3
                elif algo in ["one_class_svm"]:
                    score -= 0.2
            elif characteristics.n_samples > 100000:
                if algo in ["isolation_forest", "histogram_based"]:
                    score += 0.3
                elif algo in ["local_outlier_factor"]:
                    score -= 0.3

            # Feature dimensionality rules
            if characteristics.n_features > 100:
                if algo in ["pca", "autoencoder"]:
                    score += 0.2
                elif algo in ["local_outlier_factor"]:
                    score -= 0.2

            # Data density rules
            if characteristics.feature_density < 0.1:
                if algo in ["sparse_pca", "robust_covariance"]:
                    score += 0.2

            # Outlier ratio rules
            if characteristics.outlier_ratio > 0.1:
                if algo in ["robust_covariance", "isolation_forest"]:
                    score += 0.2

            scores[algo] = max(0.0, min(1.0, score))

        return scores

    async def _ensemble_recommendations(
        self,
        recommendations: list[dict[str, float]],
        characteristics: DatasetCharacteristicsDTO,
        constraints: OptimizationConstraintsDTO | None,
    ) -> SelectionRecommendationDTO:
        """Ensemble multiple recommendations."""
        if not recommendations:
            return SelectionRecommendationDTO(
                recommended_algorithms=[],
                confidence_scores={},
                reasoning=[],
                dataset_characteristics=characteristics,
                selection_context={},
                timestamp=datetime.now(),
            )

        # Combine recommendations with equal weights
        ensemble_scores = {}
        all_algorithms = set()
        for rec in recommendations:
            all_algorithms.update(rec.keys())

        for algo in all_algorithms:
            scores = [rec.get(algo, 0.0) for rec in recommendations]
            ensemble_scores[algo] = np.mean(scores)

        # Sort algorithms by score
        sorted_algorithms = sorted(
            ensemble_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Generate reasoning
        reasoning = []
        top_algo = sorted_algorithms[0][0] if sorted_algorithms else None

        if top_algo:
            reasoning.append(
                f"Top recommendation: {top_algo} (score: {sorted_algorithms[0][1]:.3f})"
            )

            # Add dataset-specific reasoning
            if characteristics.n_samples < 1000:
                reasoning.append(
                    "Small dataset: prioritizing algorithms suitable for limited data"
                )
            elif characteristics.n_samples > 100000:
                reasoning.append("Large dataset: prioritizing scalable algorithms")

            if characteristics.n_features > 100:
                reasoning.append(
                    "High-dimensional data: considering dimensionality reduction"
                )

            if characteristics.outlier_ratio > 0.1:
                reasoning.append(
                    "High outlier ratio detected: prioritizing robust algorithms"
                )

        return SelectionRecommendationDTO(
            recommended_algorithms=[algo for algo, _ in sorted_algorithms[:5]],
            confidence_scores=dict(sorted_algorithms),
            reasoning=reasoning,
            dataset_characteristics=characteristics,
            selection_context={
                "recommendation_methods": len(recommendations),
                "total_candidates": len(all_algorithms),
                "ensemble_approach": "equal_weighted",
            },
            timestamp=datetime.now(),
        )

    async def _predict_performance(
        self,
        recommendation: SelectionRecommendationDTO,
        characteristics: DatasetCharacteristicsDTO,
    ) -> SelectionRecommendationDTO:
        """Predict performance for recommendations."""
        if self.performance_predictor is None:
            return recommendation

        feature_vector = self._characteristics_to_features(characteristics)
        feature_vector = self.scaler.transform([feature_vector])

        # Predict performance for each algorithm
        predicted_performances = {}
        for algo in recommendation.recommended_algorithms:
            try:
                # This is simplified - would need algorithm-specific models
                performance = self.performance_predictor.predict(feature_vector)[0]
                predicted_performances[algo] = max(0.0, min(1.0, performance))
            except Exception:
                predicted_performances[algo] = 0.5

        # Update context with predictions
        recommendation.selection_context["predicted_performances"] = (
            predicted_performances
        )

        return recommendation

    def _characteristics_to_features(
        self, characteristics: DatasetCharacteristicsDTO
    ) -> list[float]:
        """Convert dataset characteristics to feature vector."""
        return [
            float(characteristics.n_samples),
            float(characteristics.n_features),
            float(characteristics.n_numeric_features),
            float(characteristics.n_categorical_features),
            float(characteristics.feature_density),
            float(characteristics.mean_feature_correlation),
            float(characteristics.data_dimensionality_ratio),
            float(characteristics.missing_value_ratio),
            float(characteristics.outlier_ratio),
            float(characteristics.class_imbalance),
            float(characteristics.feature_variance_ratio),
            float(characteristics.skewness_mean),
            float(characteristics.kurtosis_mean),
        ]

    def _compute_dataset_hash(self, dataset: Dataset) -> str:
        """Compute hash for dataset identification."""
        data_str = str(dataset.data.shape) + str(list(dataset.feature_names or []))
        return hashlib.md5(data_str.encode()).hexdigest()

    async def _update_meta_models(self) -> None:
        """Update meta-learning models with new data."""
        if len(self.selection_history) < 5:
            return

        # Prepare training data
        X = []
        y = []

        for history in self.selection_history:
            features = self._characteristics_to_features(
                history.dataset_characteristics
            )
            X.append(features)
            y.append(history.selected_algorithm)

        X = np.array(X)

        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train meta-learner
        self.meta_learner = RandomForestClassifier(n_estimators=100, random_state=42)
        self.meta_learner.fit(X_scaled, y)

        # Train performance predictor
        y_performance = [h.performance.primary_metric for h in self.selection_history]
        self.performance_predictor = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.performance_predictor.fit(X_scaled, y_performance)

        # Save models
        await self._save_meta_models()

    async def _benchmark_single_algorithm(
        self,
        dataset: Dataset,
        algorithm: str,
        cv_folds: int,
        constraints: OptimizationConstraintsDTO | None,
    ) -> AlgorithmBenchmarkDTO:
        """Benchmark single algorithm."""
        # This is a simplified implementation
        # In practice, would actually train and evaluate the algorithm

        import time

        start_time = time.time()

        # Simulate training and evaluation
        await asyncio.sleep(0.1)  # Simulate computation time

        # Generate realistic but fake metrics
        base_score = np.random.uniform(0.6, 0.9)
        scores = np.random.normal(base_score, 0.05, cv_folds)
        scores = np.clip(scores, 0, 1)

        training_time = time.time() - start_time

        return AlgorithmBenchmarkDTO(
            algorithm_name=algorithm,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            cv_scores=scores.tolist(),
            training_time_seconds=training_time,
            memory_usage_mb=np.random.uniform(50, 500),
            hyperparameters={},
            additional_metrics={},
        )

    def _analyze_algorithm_performance(self) -> dict[str, dict[str, float]]:
        """Analyze algorithm performance from history."""
        performance_stats = {}

        for history in self.selection_history:
            algo = history.selected_algorithm
            performance = history.performance.primary_metric

            if algo not in performance_stats:
                performance_stats[algo] = []
            performance_stats[algo].append(performance)

        # Calculate statistics
        stats = {}
        for algo, performances in performance_stats.items():
            stats[algo] = {
                "mean": float(np.mean(performances)),
                "std": float(np.std(performances)),
                "min": float(np.min(performances)),
                "max": float(np.max(performances)),
                "count": len(performances),
            }

        return stats

    def _analyze_dataset_preferences(self) -> dict[str, list[str]]:
        """Analyze dataset type preferences."""
        preferences = {
            "small_datasets": [],
            "large_datasets": [],
            "high_dimensional": [],
            "sparse_data": [],
        }

        for history in self.selection_history:
            algo = history.selected_algorithm
            chars = history.dataset_characteristics

            if chars.n_samples < 1000:
                preferences["small_datasets"].append(algo)
            elif chars.n_samples > 100000:
                preferences["large_datasets"].append(algo)

            if chars.n_features > 100:
                preferences["high_dimensional"].append(algo)

            if chars.feature_density < 0.1:
                preferences["sparse_data"].append(algo)

        # Get most common algorithms for each category
        for category in preferences:
            algo_counts = {}
            for algo in preferences[category]:
                algo_counts[algo] = algo_counts.get(algo, 0) + 1

            preferences[category] = sorted(
                algo_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
            preferences[category] = [algo for algo, _ in preferences[category]]

        return preferences

    def _analyze_performance_trends(self) -> dict[str, list[float]]:
        """Analyze performance trends over time."""
        # Sort history by timestamp
        sorted_history = sorted(self.selection_history, key=lambda x: x.timestamp)

        # Calculate rolling averages
        window_size = min(10, len(sorted_history))
        trends = {"overall_performance": []}

        for i in range(len(sorted_history)):
            start_idx = max(0, i - window_size + 1)
            window_performances = [
                h.performance.primary_metric for h in sorted_history[start_idx : i + 1]
            ]
            trends["overall_performance"].append(float(np.mean(window_performances)))

        return trends

    async def _analyze_feature_importance(self) -> dict[str, float]:
        """Analyze feature importance for algorithm selection."""
        if self.meta_learner is None:
            return {}

        try:
            feature_names = [
                "n_samples",
                "n_features",
                "n_numeric_features",
                "n_categorical_features",
                "feature_density",
                "mean_feature_correlation",
                "data_dimensionality_ratio",
                "missing_value_ratio",
                "outlier_ratio",
                "class_imbalance",
                "feature_variance_ratio",
                "skewness_mean",
                "kurtosis_mean",
            ]

            importances = self.meta_learner.feature_importances_
            return {
                name: float(imp)
                for name, imp in zip(feature_names, importances, strict=False)
            }
        except Exception:
            return {}

    async def _get_meta_model_accuracy(self) -> float | None:
        """Get meta-model accuracy."""
        if self.meta_learner is None or len(self.selection_history) < 10:
            return None

        try:
            X = []
            y = []

            for history in self.selection_history:
                features = self._characteristics_to_features(
                    history.dataset_characteristics
                )
                X.append(features)
                y.append(history.selected_algorithm)

            X = np.array(X)
            X_scaled = self.scaler.transform(X)

            # Cross-validation accuracy
            scores = cross_val_score(self.meta_learner, X_scaled, y, cv=3)
            return float(np.mean(scores))
        except Exception:
            return None

    def _calculate_recommendation_confidence(self) -> float:
        """Calculate overall recommendation confidence."""
        factors = []

        # History size factor
        history_factor = min(1.0, len(self.selection_history) / 100)
        factors.append(history_factor)

        # Model availability factor
        model_factor = 0.5
        if self.meta_learner is not None:
            model_factor += 0.3
        if self.performance_predictor is not None:
            model_factor += 0.2
        factors.append(model_factor)

        # Recent performance factor
        if self.selection_history:
            recent_performances = [
                h.performance.primary_metric for h in self.selection_history[-10:]
            ]
            performance_factor = float(np.mean(recent_performances))
            factors.append(performance_factor)

        return float(np.mean(factors)) if factors else 0.5

    def _initialize_algorithm_registry(self) -> dict[str, dict[str, Any]]:
        """Initialize algorithm registry with metadata."""
        return {
            "isolation_forest": {
                "min_samples": 50,
                "max_features": 10000,
                "memory_usage": 200,
                "training_time": 30,
                "scalability": "high",
                "interpretability": "medium",
            },
            "local_outlier_factor": {
                "min_samples": 20,
                "max_features": 1000,
                "memory_usage": 500,
                "training_time": 60,
                "scalability": "low",
                "interpretability": "high",
            },
            "one_class_svm": {
                "min_samples": 100,
                "max_features": 5000,
                "memory_usage": 1000,
                "training_time": 120,
                "scalability": "medium",
                "interpretability": "low",
            },
            "elliptic_envelope": {
                "min_samples": 50,
                "max_features": 1000,
                "memory_usage": 300,
                "training_time": 45,
                "scalability": "medium",
                "interpretability": "medium",
            },
            "autoencoder": {
                "min_samples": 1000,
                "max_features": 100000,
                "memory_usage": 2000,
                "training_time": 300,
                "scalability": "high",
                "interpretability": "low",
            },
        }

    async def _load_selection_history(self) -> None:
        """Load selection history from file."""
        try:
            if self.selection_history_path.exists():
                with open(self.selection_history_path) as f:
                    history_data = json.load(f)

                self.selection_history = [
                    SelectionHistoryDTO(**entry) for entry in history_data
                ]

                logger.info(
                    f"Loaded {len(self.selection_history)} selection history entries"
                )
        except Exception as e:
            logger.warning(f"Failed to load selection history: {e}")
            self.selection_history = []

    async def _save_selection_history(self) -> None:
        """Save selection history to file."""
        try:
            self.selection_history_path.parent.mkdir(parents=True, exist_ok=True)

            history_data = [history.dict() for history in self.selection_history]

            with open(self.selection_history_path, "w") as f:
                json.dump(history_data, f, indent=2, default=str)

            logger.info(
                f"Saved {len(self.selection_history)} selection history entries"
            )
        except Exception as e:
            logger.error(f"Failed to save selection history: {e}")

    async def _load_meta_models(self) -> None:
        """Load meta-learning models from file."""
        try:
            if self.meta_model_path.exists():
                scaler_path = self.meta_model_path / "scaler.pkl"
                meta_learner_path = self.meta_model_path / "meta_learner.pkl"
                predictor_path = self.meta_model_path / "performance_predictor.pkl"

                if scaler_path.exists():
                    with open(scaler_path, "rb") as f:
                        self.scaler = pickle.load(f)

                if meta_learner_path.exists():
                    with open(meta_learner_path, "rb") as f:
                        self.meta_learner = pickle.load(f)

                if predictor_path.exists():
                    with open(predictor_path, "rb") as f:
                        self.performance_predictor = pickle.load(f)

                logger.info("Loaded meta-learning models")
        except Exception as e:
            logger.warning(f"Failed to load meta-learning models: {e}")

    async def _save_meta_models(self) -> None:
        """Save meta-learning models to file."""
        try:
            self.meta_model_path.mkdir(parents=True, exist_ok=True)

            # Save scaler
            with open(self.meta_model_path / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

            # Save meta-learner
            if self.meta_learner is not None:
                with open(self.meta_model_path / "meta_learner.pkl", "wb") as f:
                    pickle.dump(self.meta_learner, f)

            # Save performance predictor
            if self.performance_predictor is not None:
                with open(
                    self.meta_model_path / "performance_predictor.pkl", "wb"
                ) as f:
                    pickle.dump(self.performance_predictor, f)

            logger.info("Saved meta-learning models")
        except Exception as e:
            logger.error(f"Failed to save meta-learning models: {e}")

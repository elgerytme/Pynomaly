"""Autonomous algorithm recommendation engine."""

from __future__ import annotations

from monorepo.application.services.autonomous_detection_config import (
    AlgorithmRecommendation,
    AutonomousConfig,
    DataProfile,
)


class AutonomousAlgorithmRecommender:
    """Service for autonomous algorithm recommendation based on data characteristics."""

    async def recommend_algorithms(
        self, profile: DataProfile, config: AutonomousConfig
    ) -> list[AlgorithmRecommendation]:
        """Recommend algorithms based on data profile.

        Args:
            profile: Data profile with characteristics
            config: Configuration options

        Returns:
            List of algorithm recommendations sorted by confidence
        """
        recommendations = []

        # Algorithm selection logic based on data characteristics

        # 1. Isolation Forest - good general purpose
        iso_recommendation = self._recommend_isolation_forest(profile)
        recommendations.append(iso_recommendation)

        # 2. Local Outlier Factor - good for density-based anomalies
        if profile.numeric_features >= profile.n_features * 0.7:  # Mostly numeric
            lof_recommendation = self._recommend_lof(profile)
            recommendations.append(lof_recommendation)

        # 3. One-Class SVM - good for complex decision boundaries
        if profile.n_samples < 50000 and profile.complexity_score > 0.5:
            svm_recommendation = self._recommend_one_class_svm(profile)
            recommendations.append(svm_recommendation)

        # 4. Elliptic Envelope - good for Gaussian-distributed data
        if profile.correlation_score < 0.8 and profile.numeric_features > 2:
            ee_recommendation = self._recommend_elliptic_envelope(profile)
            recommendations.append(ee_recommendation)

        # 5. Deep learning approach for complex/large datasets
        if profile.n_samples > 10000 and profile.complexity_score > 0.6:
            ae_recommendation = self._recommend_autoencoder(profile)
            recommendations.append(ae_recommendation)

        # Sort by confidence and limit to max_algorithms
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[: config.max_algorithms]

    def _recommend_isolation_forest(
        self, profile: DataProfile
    ) -> AlgorithmRecommendation:
        """Recommend Isolation Forest algorithm.

        Args:
            profile: Data profile

        Returns:
            Algorithm recommendation
        """
        iso_confidence = 0.8
        iso_reasoning = "General purpose algorithm, works well with mixed data types"
        iso_hyperparams = {
            "n_estimators": min(200, max(50, profile.n_samples // 100)),
            "contamination": profile.recommended_contamination,
            "random_state": 42,
        }

        if profile.n_features > 20:
            iso_confidence += 0.1
            iso_reasoning += ", handles high-dimensional data well"

        return AlgorithmRecommendation(
            algorithm="IsolationForest",
            confidence=iso_confidence,
            reasoning=iso_reasoning,
            hyperparams=iso_hyperparams,
            expected_performance=0.75,
        )

    def _recommend_lof(self, profile: DataProfile) -> AlgorithmRecommendation:
        """Recommend Local Outlier Factor algorithm.

        Args:
            profile: Data profile

        Returns:
            Algorithm recommendation
        """
        lof_confidence = 0.75
        lof_reasoning = "Good for density-based anomalies in numeric data"
        lof_hyperparams = {
            "n_neighbors": min(30, max(5, profile.n_samples // 100)),
            "contamination": profile.recommended_contamination,
        }

        if profile.n_samples < 10000:
            lof_confidence += 0.1
            lof_reasoning += ", efficient for smaller datasets"

        return AlgorithmRecommendation(
            algorithm="LOF",
            confidence=lof_confidence,
            reasoning=lof_reasoning,
            hyperparams=lof_hyperparams,
            expected_performance=0.72,
        )

    def _recommend_one_class_svm(self, profile: DataProfile) -> AlgorithmRecommendation:
        """Recommend One-Class SVM algorithm.

        Args:
            profile: Data profile

        Returns:
            Algorithm recommendation
        """
        svm_confidence = 0.7
        svm_reasoning = "Handles complex decision boundaries well"
        svm_hyperparams = {
            "kernel": "rbf",
            "gamma": "scale",
            "nu": profile.recommended_contamination,
        }

        return AlgorithmRecommendation(
            algorithm="OneClassSVM",
            confidence=svm_confidence,
            reasoning=svm_reasoning,
            hyperparams=svm_hyperparams,
            expected_performance=0.68,
        )

    def _recommend_elliptic_envelope(
        self, profile: DataProfile
    ) -> AlgorithmRecommendation:
        """Recommend Elliptic Envelope algorithm.

        Args:
            profile: Data profile

        Returns:
            Algorithm recommendation
        """
        ee_confidence = 0.65
        ee_reasoning = "Good for Gaussian-distributed data with low correlation"
        ee_hyperparams = {
            "contamination": profile.recommended_contamination,
            "random_state": 42,
        }

        return AlgorithmRecommendation(
            algorithm="EllipticEnvelope",
            confidence=ee_confidence,
            reasoning=ee_reasoning,
            hyperparams=ee_hyperparams,
            expected_performance=0.65,
        )

    def _recommend_autoencoder(self, profile: DataProfile) -> AlgorithmRecommendation:
        """Recommend AutoEncoder algorithm.

        Args:
            profile: Data profile

        Returns:
            Algorithm recommendation
        """
        ae_confidence = 0.75
        ae_reasoning = "Deep learning approach for complex, large datasets"
        ae_hyperparams = {
            "hidden_sizes": [profile.n_features // 2, profile.n_features // 4],
            "epochs": 100,
            "batch_size": min(512, max(32, profile.n_samples // 100)),
            "contamination": profile.recommended_contamination,
        }

        return AlgorithmRecommendation(
            algorithm="AutoEncoder",
            confidence=ae_confidence,
            reasoning=ae_reasoning,
            hyperparams=ae_hyperparams,
            expected_performance=0.78,
        )

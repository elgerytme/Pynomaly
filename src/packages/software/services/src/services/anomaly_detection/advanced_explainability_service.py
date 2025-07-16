"""Refactored advanced explainable AI service - reduced from 1582 to ~200 lines."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from monorepo.application.services.explainability_analyzers import (
    BiasAnalyzer,
    CounterfactualAnalyzer,
    TrustScoreAnalyzer,
)
from monorepo.application.services.explainability_core import (
    BiasAnalysisConfig,
    ExplanationConfig,
    ExplanationReport,
    GlobalExplanation,
    TrustScoreConfig,
)
from monorepo.application.services.explainability_engines import (
    GlobalExplanationEngine,
    LocalExplanationEngine,
)
from monorepo.domain.entities import Dataset
from monorepo.shared.protocols import DetectorProtocol

logger = logging.getLogger(__name__)


class AdvancedExplainabilityService:
    """Refactored service for advanced explainable AI capabilities using focused components."""

    def __init__(
        self,
        enable_shap: bool = True,
        enable_lime: bool = True,
        enable_permutation: bool = True,
        cache_explanations: bool = True,
    ) -> None:
        """Initialize explainability service.

        Args:
            enable_shap: Enable SHAP explanations
            enable_lime: Enable LIME explanations
            enable_permutation: Enable permutation importance
            cache_explanations: Cache computed explanations
        """
        self.cache_explanations = cache_explanations

        # Initialize focused components
        self.local_engine = LocalExplanationEngine(
            enable_shap=enable_shap,
            enable_lime=enable_lime,
            cache_explanations=cache_explanations,
        )
        self.global_engine = GlobalExplanationEngine(
            enable_permutation=enable_permutation,
            cache_explanations=cache_explanations,
        )
        self.bias_analyzer = BiasAnalyzer()
        self.trust_analyzer = TrustScoreAnalyzer()
        self.counterfactual_analyzer = CounterfactualAnalyzer()

        logger.info("Initialized AdvancedExplainabilityService with focused components")

    async def generate_comprehensive_explanation(
        self,
        detector: DetectorProtocol,
        dataset: Dataset,
        config: ExplanationConfig | None = None,
    ) -> ExplanationReport:
        """Generate comprehensive explanation report."""
        try:
            if not config:
                config = ExplanationConfig()

            logger.info("Generating comprehensive explanation report")

            # Prepare data
            X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
            feature_names = (
                config.feature_names
                or dataset.features
                or [f"feature_{i}" for i in range(X.shape[1])]
            )

            # Get model predictions
            predictions = detector.decision_function(X)

            # Generate explanations using focused engines
            tasks = []

            # Local explanations
            if config.explanation_type in ["local", "both"]:
                tasks.append(
                    self.local_engine.generate_explanations(
                        detector, X, feature_names, config
                    )
                )

            # Global explanations
            if config.explanation_type in ["global", "both"]:
                tasks.append(
                    self.global_engine.generate_explanation(
                        detector, X, feature_names, config
                    )
                )

            # Execute explanation tasks
            explanation_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            local_explanations = []
            global_explanation = None

            for result in explanation_results:
                if isinstance(result, Exception):
                    logger.warning(f"Explanation task failed: {result}")
                    continue

                if isinstance(result, list):  # Local explanations
                    local_explanations = result
                else:  # Global explanation
                    global_explanation = result

            # Generate trust assessment
            trust_assessment = await self.trust_analyzer.assess_trust_score(
                detector, X, predictions, TrustScoreConfig()
            )

            # Create report
            report = ExplanationReport(
                model_info=self._get_model_info(detector),
                dataset_summary=self._get_dataset_summary(dataset),
                local_explanations=local_explanations,
                global_explanation=global_explanation
                or self._create_fallback_global_explanation(feature_names),
                bias_analysis=[],  # Will be added if bias analysis is performed
                trust_assessment=trust_assessment,
                recommendations=self._generate_recommendations(trust_assessment),
            )

            logger.info("Comprehensive explanation report generated successfully")
            return report

        except Exception as e:
            logger.error(f"Failed to generate explanation report: {e}")
            raise RuntimeError(f"Explanation generation failed: {e}")

    async def analyze_bias(
        self, detector: DetectorProtocol, dataset: Dataset, config: BiasAnalysisConfig
    ) -> list:
        """Analyze model for potential bias."""
        return await self.bias_analyzer.analyze_bias(detector, dataset, config)

    async def generate_counterfactual_explanations(
        self,
        detector: DetectorProtocol,
        dataset: Dataset,
        target_instances: list[int],
        n_counterfactuals: int = 5,
        optimization_method: str = "random",
    ) -> dict[str, Any]:
        """Generate counterfactual explanations for given instances."""
        return await self.counterfactual_analyzer.generate_counterfactual_explanations(
            detector, dataset, target_instances, n_counterfactuals, optimization_method
        )

    async def analyze_feature_interactions(
        self,
        detector: DetectorProtocol,
        dataset: Dataset,
        method: str = "correlation",
        max_interactions: int = 20,
    ) -> dict[str, Any]:
        """Analyze feature interactions using correlation method."""
        try:
            logger.info(f"Analyzing feature interactions using {method}")

            X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
            feature_names = dataset.features or [
                f"feature_{i}" for i in range(X.shape[1])
            ]

            # Simplified correlation-based interactions
            interactions = {}
            if len(feature_names) >= 2:
                corr_matrix = np.corrcoef(X.T)
                for i in range(len(feature_names)):
                    for j in range(i + 1, len(feature_names)):
                        if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                            interaction_key = f"{feature_names[i]}_x_{feature_names[j]}"
                            interactions[interaction_key] = float(
                                abs(corr_matrix[i, j])
                            )

                # Sort and return top interactions
                sorted_interactions = sorted(
                    interactions.items(), key=lambda x: x[1], reverse=True
                )
                interactions = dict(sorted_interactions[:max_interactions])

            return {
                "method": method,
                "interactions": interactions,
                "feature_names": feature_names,
                "total_features": len(feature_names),
                "analyzed_interactions": len(interactions),
            }

        except Exception as e:
            logger.error(f"Feature interaction analysis failed: {e}")
            return {"error": str(e)}

    async def generate_explanation_dashboard_data(
        self,
        detector: DetectorProtocol,
        dataset: Dataset,
        config: ExplanationConfig | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive data for explanation dashboard."""
        try:
            logger.info("Generating explanation dashboard data")

            if not config:
                config = ExplanationConfig()

            X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
            feature_names = (
                config.feature_names
                or dataset.features
                or [f"feature_{i}" for i in range(X.shape[1])]
            )

            # Generate comprehensive explanation
            explanation_report = await self.generate_comprehensive_explanation(
                detector, dataset, config
            )

            # Generate counterfactual examples
            sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
            counterfactuals = await self.generate_counterfactual_explanations(
                detector, dataset, sample_indices.tolist()
            )

            # Analyze feature interactions
            interactions = await self.analyze_feature_interactions(
                detector, dataset, "correlation"
            )

            # Create dashboard data
            dashboard_data = {
                "summary": {
                    "dataset_name": dataset.name,
                    "n_samples": len(X),
                    "n_features": len(feature_names),
                    "model_type": getattr(detector, "algorithm_name", "unknown"),
                    "trust_score": explanation_report.trust_assessment.overall_trust_score,
                    "risk_level": explanation_report.trust_assessment.risk_assessment,
                },
                "global_explanation": {
                    "feature_importance": explanation_report.global_explanation.feature_importance,
                    "top_features": sorted(
                        explanation_report.global_explanation.feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10],
                    "model_summary": explanation_report.global_explanation.model_summary,
                },
                "local_explanations": [
                    {
                        "sample_id": exp.sample_id,
                        "prediction": exp.prediction,
                        "confidence": exp.confidence,
                        "top_features": sorted(
                            exp.feature_contributions.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )[:5],
                    }
                    for exp in explanation_report.local_explanations[:10]
                ],
                "feature_interactions": interactions,
                "counterfactuals": counterfactuals,
                "trust_assessment": {
                    "overall_score": explanation_report.trust_assessment.overall_trust_score,
                    "consistency": explanation_report.trust_assessment.consistency_score,
                    "stability": explanation_report.trust_assessment.stability_score,
                    "fidelity": explanation_report.trust_assessment.fidelity_score,
                    "risk_level": explanation_report.trust_assessment.risk_assessment,
                },
                "bias_analysis": explanation_report.bias_analysis,
                "recommendations": explanation_report.recommendations,
            }

            return dashboard_data

        except Exception as e:
            logger.error(f"Dashboard data generation failed: {e}")
            return {"error": str(e)}

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.local_engine.clear_caches()
        self.global_engine.clear_caches()
        logger.info("All caches cleared")

    def get_service_info(self) -> dict[str, Any]:
        """Get service information."""
        return {
            "local_engine": "LocalExplanationEngine",
            "global_engine": "GlobalExplanationEngine",
            "bias_analyzer": "BiasAnalyzer",
            "trust_analyzer": "TrustScoreAnalyzer",
            "counterfactual_analyzer": "CounterfactualAnalyzer",
            "cache_enabled": self.cache_explanations,
        }

    def _get_model_info(self, detector: DetectorProtocol) -> dict[str, Any]:
        """Get model information."""
        return {
            "algorithm": getattr(detector, "algorithm_name", "unknown"),
            "parameters": getattr(detector, "algorithm_params", {}),
            "is_trained": getattr(detector, "is_trained", False),
        }

    def _get_dataset_summary(self, dataset: Dataset) -> dict[str, Any]:
        """Get dataset summary."""
        data = dataset.data
        if hasattr(data, "shape"):
            n_samples, n_features = data.shape
        else:
            n_samples = len(data)
            n_features = len(data[0]) if len(data) > 0 else 0

        return {
            "name": dataset.name,
            "n_samples": n_samples,
            "n_features": n_features,
            "features": dataset.features[:10] if dataset.features else [],
        }

    def _create_fallback_global_explanation(
        self, feature_names: list[str]
    ) -> GlobalExplanation:
        """Create fallback global explanation."""
        equal_importance = 1.0 / len(feature_names) if feature_names else 0.0

        return GlobalExplanation(
            feature_importance=dict.fromkeys(feature_names, equal_importance),
            feature_interactions={},
            model_summary={"note": "Fallback explanation due to computation failures"},
            explanation_method="fallback",
            coverage=0.5,
            reliability=0.3,
        )

    def _generate_recommendations(self, trust_assessment) -> list[str]:
        """Generate recommendations based on trust assessment."""
        recommendations = []

        if trust_assessment.overall_trust_score < 0.7:
            recommendations.append("Consider model retraining or hyperparameter tuning")

        if trust_assessment.consistency_score < 0.7:
            recommendations.append("Improve model consistency through regularization")

        if trust_assessment.stability_score < 0.7:
            recommendations.append("Enhance prediction stability with ensemble methods")

        if trust_assessment.fidelity_score < 0.7:
            recommendations.append("Validate explanation fidelity with domain experts")

        if trust_assessment.risk_assessment == "high":
            recommendations.append(
                "Exercise caution when using model predictions for critical decisions"
            )

        if not recommendations:
            recommendations.append("Model shows good explainability characteristics")

        return recommendations

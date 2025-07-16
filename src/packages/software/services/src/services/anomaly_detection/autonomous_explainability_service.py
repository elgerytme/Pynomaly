"""Autonomous explainability service for autonomous detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pynomaly.application.services.algorithm_recommendation_service import (
    AlgorithmRecommendation,
)
from pynomaly.application.services.data_profiling_service import DataProfile
from pynomaly.domain.entities import Dataset, DetectionResult


@dataclass
class AlgorithmExplanation:
    """Explanation for why an algorithm was selected or rejected."""

    algorithm: str
    selected: bool
    confidence: float
    reasoning: str
    data_characteristics: dict[str, Any]
    decision_factors: dict[str, float]
    alternatives_considered: list[str]
    performance_prediction: float
    computational_complexity: str
    memory_requirements: str
    interpretability_score: float


@dataclass
class AnomalyExplanation:
    """Explanation for detected anomalies."""

    sample_id: int
    anomaly_score: float
    contributing_features: dict[str, float]
    feature_importances: dict[str, float]
    normal_range_deviations: dict[str, float]
    similar_normal_samples: list[int]
    explanation_confidence: float
    explanation_method: str


@dataclass
class ExplanationReport:
    """Comprehensive explanation report for autonomous detection."""

    dataset_profile: DataProfile
    algorithm_explanations: list[AlgorithmExplanation]
    selected_algorithms: list[str]
    rejected_algorithms: list[str]
    ensemble_explanation: str | None
    anomaly_explanations: list[AnomalyExplanation]
    processing_explanation: str
    recommendations: list[str]
    decision_tree: dict[str, Any]


class AutonomousExplainabilityService:
    """Service responsible for generating explanations for autonomous detection."""

    def __init__(self):
        """Initialize autonomous explainability service."""
        self.logger = logging.getLogger(__name__)

    async def explain_algorithm_choices(
        self,
        profile: DataProfile,
        recommendations: list[AlgorithmRecommendation],
        selected_algorithms: list[str],
        confidence_threshold: float = 0.3,
        verbose: bool = False,
    ) -> list[AlgorithmExplanation]:
        """Explain algorithm selection choices.

        Args:
            profile: Data profile
            recommendations: List of algorithm recommendations
            selected_algorithms: List of selected algorithms
            confidence_threshold: Confidence threshold used for selection
            verbose: Enable verbose logging

        Returns:
            List of algorithm explanations
        """
        if verbose:
            self.logger.info(f"Explaining algorithm choices for {len(recommendations)} recommendations")

        explanations = []

        # Get all available algorithms for context
        all_algorithms = [rec.algorithm for rec in recommendations]

        for recommendation in recommendations:
            selected = recommendation.algorithm in selected_algorithms

            # Extract data characteristics relevant to this algorithm
            relevant_characteristics = self._extract_relevant_characteristics(recommendation.algorithm, profile)

            # Get alternatives considered
            alternatives = self._get_algorithm_alternatives(recommendation.algorithm, all_algorithms)

            explanation = AlgorithmExplanation(
                algorithm=recommendation.algorithm,
                selected=selected,
                confidence=recommendation.confidence,
                reasoning=self._generate_selection_reasoning(recommendation, profile, selected, confidence_threshold),
                data_characteristics=relevant_characteristics,
                decision_factors=recommendation.decision_factors,
                alternatives_considered=alternatives,
                performance_prediction=recommendation.expected_performance,
                computational_complexity=recommendation.computational_complexity,
                memory_requirements=recommendation.memory_requirements,
                interpretability_score=recommendation.interpretability_score,
            )

            explanations.append(explanation)

        if verbose:
            self.logger.info(f"Generated {len(explanations)} algorithm explanations")

        return explanations

    def _generate_selection_reasoning(
        self,
        recommendation: AlgorithmRecommendation,
        profile: DataProfile,
        selected: bool,
        confidence_threshold: float,
    ) -> str:
        """Generate reasoning for algorithm selection/rejection.

        Args:
            recommendation: Algorithm recommendation
            profile: Data profile
            selected: Whether algorithm was selected
            confidence_threshold: Confidence threshold

        Returns:
            Human-readable reasoning
        """
        if selected:
            return f"Selected due to high confidence ({recommendation.confidence:.3f}). {recommendation.reasoning}"
        else:
            if recommendation.confidence < confidence_threshold:
                return f"Rejected due to low confidence ({recommendation.confidence:.3f} < {confidence_threshold}). {recommendation.reasoning}"
            else:
                return f"Not selected despite adequate confidence ({recommendation.confidence:.3f}). {recommendation.reasoning}"

    def _extract_relevant_characteristics(self, algorithm: str, profile: DataProfile) -> dict[str, Any]:
        """Extract data characteristics relevant to algorithm selection.

        Args:
            algorithm: Algorithm name
            profile: Data profile

        Returns:
            Dictionary of relevant characteristics
        """
        characteristics = {
            "n_samples": profile.n_samples,
            "n_features": profile.n_features,
            "missing_values_ratio": profile.missing_values_ratio,
            "complexity_score": profile.complexity_score,
        }

        # Algorithm-specific characteristics
        if algorithm in ["LOF", "KNN"]:
            characteristics.update({
                "density_estimation": "local_density_based",
                "neighbor_sensitivity": "high",
                "curse_of_dimensionality": "high" if profile.n_features > 50 else "low",
            })

        elif algorithm in ["IsolationForest"]:
            characteristics.update({
                "tree_based": True,
                "ensemble_method": True,
                "feature_interactions": "automatic",
                "scalability": "high",
            })

        elif algorithm in ["COPOD", "HBOS"]:
            characteristics.update({
                "correlation_based": algorithm == "COPOD",
                "histogram_based": algorithm == "HBOS",
                "feature_independence": "assumed" if algorithm == "HBOS" else "not_assumed",
                "scalability": "very_high",
            })

        elif algorithm in ["OneClassSVM"]:
            characteristics.update({
                "kernel_method": True,
                "boundary_based": True,
                "non_linear_capable": True,
                "parameter_sensitive": "high",
            })

        elif algorithm in ["PCA"]:
            characteristics.update({
                "linear_method": True,
                "dimensionality_reduction": True,
                "variance_based": True,
                "interpretability": "medium",
            })

        elif algorithm in ["AutoEncoder"]:
            characteristics.update({
                "deep_learning": True,
                "non_linear_capable": True,
                "reconstruction_based": True,
                "training_intensive": True,
            })

        return characteristics

    def _get_algorithm_alternatives(self, algorithm: str, all_algorithms: list[str]) -> list[str]:
        """Get alternative algorithms that were considered.

        Args:
            algorithm: Algorithm name
            all_algorithms: List of all available algorithms

        Returns:
            List of alternative algorithms
        """
        alternatives_map = {
            "IsolationForest": ["COPOD", "HBOS", "LOF", "KNN"],
            "LOF": ["KNN", "OneClassSVM", "IsolationForest"],
            "COPOD": ["HBOS", "IsolationForest", "PCA"],
            "HBOS": ["COPOD", "IsolationForest", "PCA"],
            "OneClassSVM": ["LOF", "KNN", "AutoEncoder"],
            "PCA": ["COPOD", "HBOS", "IsolationForest"],
            "KNN": ["LOF", "OneClassSVM", "IsolationForest"],
            "AutoEncoder": ["OneClassSVM", "IsolationForest", "PCA"],
        }

        potential_alternatives = alternatives_map.get(algorithm, [])
        return [alg for alg in potential_alternatives if alg in all_algorithms]

    async def explain_anomalies(
        self,
        dataset: Dataset,
        detection_results: dict[str, DetectionResult],
        max_explanations: int = 10,
        explanation_method: str = "auto",
        verbose: bool = False,
    ) -> list[AnomalyExplanation]:
        """Explain detected anomalies.

        Args:
            dataset: Original dataset
            detection_results: Detection results by algorithm
            max_explanations: Maximum number of anomalies to explain
            explanation_method: Method for explanation ("auto", "statistical", "distance")
            verbose: Enable verbose logging

        Returns:
            List of anomaly explanations
        """
        if verbose:
            self.logger.info(f"Explaining anomalies using {explanation_method} method")

        explanations = []

        # Get the best result or ensemble result
        best_result = self._get_best_result(detection_results)
        if not best_result:
            return explanations

        # Get anomaly indices (sorted by score)
        anomaly_indices = self._get_top_anomaly_indices(best_result, max_explanations)

        # Prepare data for explanation
        df = dataset.data
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_columns:
            self.logger.warning("No numeric columns found for anomaly explanation")
            return explanations

        # Generate explanations for each anomaly
        for i, sample_idx in enumerate(anomaly_indices):
            if verbose:
                self.logger.info(f"Explaining anomaly {i+1}/{len(anomaly_indices)}")

            explanation = await self._explain_single_anomaly(
                dataset=dataset,
                sample_idx=sample_idx,
                detection_result=best_result,
                explanation_method=explanation_method,
                numeric_columns=numeric_columns,
            )

            if explanation:
                explanations.append(explanation)

        if verbose:
            self.logger.info(f"Generated {len(explanations)} anomaly explanations")

        return explanations

    def _get_best_result(self, detection_results: dict[str, DetectionResult]) -> DetectionResult | None:
        """Get the best detection result for explanation.

        Args:
            detection_results: Detection results by algorithm

        Returns:
            Best detection result or None
        """
        if not detection_results:
            return None

        # Prefer ensemble result if available
        if "ensemble" in detection_results:
            return detection_results["ensemble"]

        # Otherwise, return first result
        return next(iter(detection_results.values()))

    def _get_top_anomaly_indices(self, result: DetectionResult, max_count: int) -> list[int]:
        """Get indices of top anomalies sorted by score.

        Args:
            result: Detection result
            max_count: Maximum number of indices to return

        Returns:
            List of anomaly indices
        """
        if not result.scores:
            return []

        # Get anomaly indices with scores
        anomaly_data = []
        for i, (score, label) in enumerate(zip(result.scores, result.labels, strict=False)):
            if label == 1:  # Anomaly
                anomaly_data.append((i, score.value))

        # Sort by score (descending)
        anomaly_data.sort(key=lambda x: x[1], reverse=True)

        # Return top indices
        return [idx for idx, _ in anomaly_data[:max_count]]

    async def _explain_single_anomaly(
        self,
        dataset: Dataset,
        sample_idx: int,
        detection_result: DetectionResult,
        explanation_method: str,
        numeric_columns: list[str],
    ) -> AnomalyExplanation | None:
        """Explain a single anomaly.

        Args:
            dataset: Original dataset
            sample_idx: Index of the sample to explain
            detection_result: Detection result
            explanation_method: Method for explanation
            numeric_columns: List of numeric column names

        Returns:
            Anomaly explanation or None
        """
        try:
            df = dataset.data
            sample = df.iloc[sample_idx]

            # Get anomaly score
            anomaly_score = detection_result.scores[sample_idx].value if sample_idx < len(detection_result.scores) else 0.0

            # Calculate feature contributions
            if explanation_method == "statistical":
                contributing_features = self._calculate_statistical_contributions(df, sample, numeric_columns)
            elif explanation_method == "distance":
                contributing_features = self._calculate_distance_contributions(df, sample, numeric_columns)
            else:  # auto
                contributing_features = self._calculate_auto_contributions(df, sample, numeric_columns)

            # Calculate feature importances
            feature_importances = self._calculate_feature_importances(df, sample, numeric_columns)

            # Calculate normal range deviations
            normal_range_deviations = self._calculate_normal_range_deviations(df, sample, numeric_columns)

            # Find similar normal samples
            similar_normal_samples = self._find_similar_normal_samples(
                df, sample, detection_result, numeric_columns, sample_idx
            )

            # Calculate explanation confidence
            explanation_confidence = self._calculate_explanation_confidence(
                contributing_features, feature_importances, normal_range_deviations
            )

            return AnomalyExplanation(
                sample_id=sample_idx,
                anomaly_score=anomaly_score,
                contributing_features=contributing_features,
                feature_importances=feature_importances,
                normal_range_deviations=normal_range_deviations,
                similar_normal_samples=similar_normal_samples,
                explanation_confidence=explanation_confidence,
                explanation_method=explanation_method,
            )

        except Exception as e:
            self.logger.error(f"Failed to explain anomaly {sample_idx}: {e}")
            return None

    def _calculate_statistical_contributions(
        self, df: pd.DataFrame, sample: pd.Series, numeric_columns: list[str]
    ) -> dict[str, float]:
        """Calculate statistical contributions for each feature.

        Args:
            df: Complete dataframe
            sample: Sample to explain
            numeric_columns: List of numeric columns

        Returns:
            Dictionary of feature contributions
        """
        contributions = {}

        for col in numeric_columns:
            if col in sample.index:
                # Calculate z-score
                mean_val = df[col].mean()
                std_val = df[col].std()

                if std_val > 0:
                    z_score = abs((sample[col] - mean_val) / std_val)
                    contributions[col] = min(z_score / 3.0, 1.0)  # Normalize to 0-1
                else:
                    contributions[col] = 0.0

        return contributions

    def _calculate_distance_contributions(
        self, df: pd.DataFrame, sample: pd.Series, numeric_columns: list[str]
    ) -> dict[str, float]:
        """Calculate distance-based contributions for each feature.

        Args:
            df: Complete dataframe
            sample: Sample to explain
            numeric_columns: List of numeric columns

        Returns:
            Dictionary of feature contributions
        """
        contributions = {}

        for col in numeric_columns:
            if col in sample.index:
                # Calculate distance from median
                median_val = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                if iqr > 0:
                    distance = abs(sample[col] - median_val)
                    contributions[col] = min(distance / (1.5 * iqr), 1.0)  # Normalize by IQR
                else:
                    contributions[col] = 0.0

        return contributions

    def _calculate_auto_contributions(
        self, df: pd.DataFrame, sample: pd.Series, numeric_columns: list[str]
    ) -> dict[str, float]:
        """Calculate automatic contributions (combination of methods).

        Args:
            df: Complete dataframe
            sample: Sample to explain
            numeric_columns: List of numeric columns

        Returns:
            Dictionary of feature contributions
        """
        statistical_contrib = self._calculate_statistical_contributions(df, sample, numeric_columns)
        distance_contrib = self._calculate_distance_contributions(df, sample, numeric_columns)

        # Combine contributions
        contributions = {}
        for col in numeric_columns:
            if col in statistical_contrib and col in distance_contrib:
                contributions[col] = (statistical_contrib[col] + distance_contrib[col]) / 2.0

        return contributions

    def _calculate_feature_importances(
        self, df: pd.DataFrame, sample: pd.Series, numeric_columns: list[str]
    ) -> dict[str, float]:
        """Calculate feature importances for the sample.

        Args:
            df: Complete dataframe
            sample: Sample to explain
            numeric_columns: List of numeric columns

        Returns:
            Dictionary of feature importances
        """
        importances = {}

        for col in numeric_columns:
            if col in sample.index:
                # Simple importance based on variance
                col_var = df[col].var()
                total_var = sum(df[col].var() for col in numeric_columns if col in df.columns)

                if total_var > 0:
                    importances[col] = col_var / total_var
                else:
                    importances[col] = 1.0 / len(numeric_columns)

        return importances

    def _calculate_normal_range_deviations(
        self, df: pd.DataFrame, sample: pd.Series, numeric_columns: list[str]
    ) -> dict[str, float]:
        """Calculate deviations from normal ranges.

        Args:
            df: Complete dataframe
            sample: Sample to explain
            numeric_columns: List of numeric columns

        Returns:
            Dictionary of deviations
        """
        deviations = {}

        for col in numeric_columns:
            if col in sample.index:
                # Calculate normal range (IQR)
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                value = sample[col]

                if value < lower_bound:
                    deviations[col] = (lower_bound - value) / iqr if iqr > 0 else 0.0
                elif value > upper_bound:
                    deviations[col] = (value - upper_bound) / iqr if iqr > 0 else 0.0
                else:
                    deviations[col] = 0.0

        return deviations

    def _find_similar_normal_samples(
        self,
        df: pd.DataFrame,
        sample: pd.Series,
        detection_result: DetectionResult,
        numeric_columns: list[str],
        sample_idx: int,
        max_similar: int = 5,
    ) -> list[int]:
        """Find similar normal samples.

        Args:
            df: Complete dataframe
            sample: Sample to explain
            detection_result: Detection result
            numeric_columns: List of numeric columns
            sample_idx: Index of the sample
            max_similar: Maximum number of similar samples to return

        Returns:
            List of indices of similar normal samples
        """
        try:
            # Get normal sample indices
            normal_indices = [i for i, label in enumerate(detection_result.labels) if label == 0]

            if not normal_indices:
                return []

            # Calculate distances to normal samples
            distances = []
            sample_values = sample[numeric_columns].values

            for idx in normal_indices:
                if idx != sample_idx:
                    normal_sample = df.iloc[idx][numeric_columns].values

                    # Calculate Euclidean distance
                    distance = np.sqrt(np.sum((sample_values - normal_sample) ** 2))
                    distances.append((idx, distance))

            # Sort by distance and return top similar samples
            distances.sort(key=lambda x: x[1])
            return [idx for idx, _ in distances[:max_similar]]

        except Exception as e:
            self.logger.warning(f"Failed to find similar normal samples: {e}")
            return []

    def _calculate_explanation_confidence(
        self,
        contributing_features: dict[str, float],
        feature_importances: dict[str, float],
        normal_range_deviations: dict[str, float],
    ) -> float:
        """Calculate confidence in the explanation.

        Args:
            contributing_features: Feature contributions
            feature_importances: Feature importances
            normal_range_deviations: Normal range deviations

        Returns:
            Explanation confidence between 0 and 1
        """
        try:
            # High confidence if there are clear contributing features
            if contributing_features:
                max_contribution = max(contributing_features.values())
                avg_contribution = sum(contributing_features.values()) / len(contributing_features)

                # High confidence if there's a clear outlier feature
                if max_contribution > 0.8:
                    return 0.9
                elif avg_contribution > 0.5:
                    return 0.7
                else:
                    return 0.5

            return 0.3

        except Exception as e:
            self.logger.warning(f"Failed to calculate explanation confidence: {e}")
            return 0.3

    async def generate_explanation_report(
        self,
        dataset: Dataset,
        profile: DataProfile,
        recommendations: list[AlgorithmRecommendation],
        selected_algorithms: list[str],
        detection_results: dict[str, DetectionResult],
        pipeline_results: dict[str, Any],
        verbose: bool = False,
    ) -> ExplanationReport:
        """Generate comprehensive explanation report.

        Args:
            dataset: Original dataset
            profile: Data profile
            recommendations: Algorithm recommendations
            selected_algorithms: Selected algorithms
            detection_results: Detection results
            pipeline_results: Pipeline results
            verbose: Enable verbose logging

        Returns:
            Comprehensive explanation report
        """
        if verbose:
            self.logger.info("Generating comprehensive explanation report")

        # Generate algorithm explanations
        algorithm_explanations = await self.explain_algorithm_choices(
            profile=profile,
            recommendations=recommendations,
            selected_algorithms=selected_algorithms,
            verbose=verbose,
        )

        # Generate anomaly explanations
        anomaly_explanations = await self.explain_anomalies(
            dataset=dataset,
            detection_results=detection_results,
            max_explanations=10,
            verbose=verbose,
        )

        # Generate ensemble explanation
        ensemble_explanation = self._generate_ensemble_explanation(detection_results, pipeline_results)

        # Generate processing explanation
        processing_explanation = self._generate_processing_explanation(profile, pipeline_results)

        # Generate recommendations
        recommendations_list = self._generate_recommendations(profile, pipeline_results, algorithm_explanations)

        # Generate decision tree
        decision_tree = self._build_decision_tree(profile, recommendations, pipeline_results)

        # Get rejected algorithms
        rejected_algorithms = [
            rec.algorithm for rec in recommendations if rec.algorithm not in selected_algorithms
        ]

        report = ExplanationReport(
            dataset_profile=profile,
            algorithm_explanations=algorithm_explanations,
            selected_algorithms=selected_algorithms,
            rejected_algorithms=rejected_algorithms,
            ensemble_explanation=ensemble_explanation,
            anomaly_explanations=anomaly_explanations,
            processing_explanation=processing_explanation,
            recommendations=recommendations_list,
            decision_tree=decision_tree,
        )

        if verbose:
            self.logger.info("Explanation report generated successfully")

        return report

    def _generate_ensemble_explanation(
        self, detection_results: dict[str, DetectionResult], pipeline_results: dict[str, Any]
    ) -> str | None:
        """Generate explanation for ensemble results.

        Args:
            detection_results: Detection results
            pipeline_results: Pipeline results

        Returns:
            Ensemble explanation or None
        """
        if "ensemble" in detection_results:
            ensemble_result = detection_results["ensemble"]
            algorithms = ensemble_result.metadata.get("algorithms", [])

            return (
                f"Ensemble combines {len(algorithms)} algorithms: {', '.join(algorithms)}. "
                f"Uses simple averaging with equal weights. "
                f"Detected {sum(ensemble_result.labels)} anomalies with threshold {ensemble_result.threshold:.3f}."
            )

        return None

    def _generate_processing_explanation(
        self, profile: DataProfile, pipeline_results: dict[str, Any]
    ) -> str:
        """Generate explanation for data processing.

        Args:
            profile: Data profile
            pipeline_results: Pipeline results

        Returns:
            Processing explanation
        """
        explanation_parts = [
            f"Dataset processed: {profile.n_samples:,} samples with {profile.n_features} features.",
            f"Data quality: {profile.missing_values_ratio:.1%} missing values, complexity score {profile.complexity_score:.2f}.",
        ]

        if profile.preprocessing_applied:
            explanation_parts.append(f"Preprocessing applied with quality score {profile.quality_score:.2f}.")

        if profile.temporal_features > 0:
            explanation_parts.append(f"Temporal features detected: {profile.temporal_features}.")

        explanation_parts.append(f"Recommended contamination rate: {profile.recommended_contamination:.1%}.")

        return " ".join(explanation_parts)

    def _generate_recommendations(
        self,
        profile: DataProfile,
        pipeline_results: dict[str, Any],
        algorithm_explanations: list[AlgorithmExplanation],
    ) -> list[str]:
        """Generate recommendations based on results.

        Args:
            profile: Data profile
            pipeline_results: Pipeline results
            algorithm_explanations: Algorithm explanations

        Returns:
            List of recommendations
        """
        recommendations = []

        # Data quality recommendations
        if profile.missing_values_ratio > 0.1:
            recommendations.append("Consider preprocessing to handle missing values for better accuracy.")

        if profile.complexity_score > 0.8:
            recommendations.append("Dataset is complex; consider ensemble methods or deep learning approaches.")

        # Algorithm recommendations
        best_algorithm = pipeline_results.get("best_algorithm")
        if best_algorithm and best_algorithm != "ensemble":
            recommendations.append(f"Best performing algorithm: {best_algorithm}. Consider using it for production.")

        # Performance recommendations
        if len(pipeline_results.get("errors", {})) > 0:
            recommendations.append("Some algorithms failed; verify data preprocessing and feature engineering.")

        # Ensemble recommendations
        if len(pipeline_results.get("results", {})) > 1:
            recommendations.append("Multiple algorithms available; consider ensemble for improved robustness.")

        return recommendations

    def _build_decision_tree(
        self,
        profile: DataProfile,
        recommendations: list[AlgorithmRecommendation],
        pipeline_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Build decision tree for algorithm selection.

        Args:
            profile: Data profile
            recommendations: Algorithm recommendations
            pipeline_results: Pipeline results

        Returns:
            Decision tree structure
        """
        decision_tree = {
            "root": {
                "condition": "dataset_characteristics",
                "data": {
                    "n_samples": profile.n_samples,
                    "n_features": profile.n_features,
                    "complexity": profile.complexity_score,
                },
                "branches": [],
            }
        }

        # Add branches for each algorithm decision
        for rec in recommendations:
            branch = {
                "algorithm": rec.algorithm,
                "confidence": rec.confidence,
                "selected": rec.algorithm in pipeline_results.get("algorithms_used", []),
                "reasoning": rec.reasoning,
                "decision_factors": rec.decision_factors,
            }
            decision_tree["root"]["branches"].append(branch)

        return decision_tree

    def create_explanation_summary(self, report: ExplanationReport) -> str:
        """Create human-readable summary of explanation report.

        Args:
            report: Explanation report

        Returns:
            Human-readable summary
        """
        try:
            summary_lines = [
                "=== AUTONOMOUS DETECTION EXPLANATION ===",
                f"Dataset: {report.dataset_profile.n_samples:,} samples, {report.dataset_profile.n_features} features",
                f"Algorithms Selected: {len(report.selected_algorithms)}",
                f"Algorithms Rejected: {len(report.rejected_algorithms)}",
                f"Anomalies Explained: {len(report.anomaly_explanations)}",
                "",
                "=== ALGORITHM SELECTION ===",
            ]

            for exp in report.algorithm_explanations[:5]:  # Top 5
                status = "SELECTED" if exp.selected else "REJECTED"
                summary_lines.append(f"{exp.algorithm}: {status} (confidence: {exp.confidence:.3f})")

            if report.ensemble_explanation:
                summary_lines.append("")
                summary_lines.append("=== ENSEMBLE ===")
                summary_lines.append(report.ensemble_explanation)

            if report.anomaly_explanations:
                summary_lines.append("")
                summary_lines.append("=== TOP ANOMALIES ===")
                for exp in report.anomaly_explanations[:3]:  # Top 3
                    top_feature = max(exp.contributing_features.items(), key=lambda x: x[1])[0]
                    summary_lines.append(f"Sample {exp.sample_id}: score {exp.anomaly_score:.3f}, top feature: {top_feature}")

            if report.recommendations:
                summary_lines.append("")
                summary_lines.append("=== RECOMMENDATIONS ===")
                for rec in report.recommendations:
                    summary_lines.append(f"• {rec}")

            return "\n".join(summary_lines)

        except Exception as e:
            self.logger.warning(f"Failed to create explanation summary: {e}")
            return "Failed to generate explanation summary"

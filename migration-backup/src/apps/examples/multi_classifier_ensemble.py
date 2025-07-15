#!/usr/bin/env python3
"""
Multi-Classifier Ensemble Example
=================================

This example demonstrates advanced ensemble methods for anomaly detection,
including anomaly ranking, filtering, and consensus strategies across multiple
datasets and classifiers.
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from pynomaly.infrastructure.config import create_container


class VotingStrategy(Enum):
    """Ensemble voting strategies."""

    MAJORITY = "majority"
    WEIGHTED = "weighted"
    SOFT = "soft"
    CONSENSUS = "consensus"
    RANKED = "ranked"


class RankingMethod(Enum):
    """Anomaly ranking methods."""

    AVERAGE_SCORE = "average_score"
    MEDIAN_SCORE = "median_score"
    MAX_SCORE = "max_score"
    WEIGHTED_AVERAGE = "weighted_average"
    CONCORDANCE = "concordance"
    BORDA_COUNT = "borda_count"


@dataclass
class EnsembleResult:
    """Results from ensemble anomaly detection."""

    predictions: np.ndarray
    scores: np.ndarray
    individual_predictions: dict[str, np.ndarray]
    individual_scores: dict[str, np.ndarray]
    confidence: np.ndarray
    ranking: np.ndarray
    consensus_level: float
    voting_strategy: str
    ranking_method: str


@dataclass
class DetectorConfig:
    """Configuration for individual detector."""

    name: str
    algorithm: str
    parameters: dict[str, Any]
    weight: float = 1.0
    preprocessing: dict[str, Any] | None = None


class MultiClassifierEnsemble:
    """Advanced multi-classifier ensemble for anomaly detection."""

    def __init__(self, voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED):
        self.container = None
        self.detectors = {}
        self.detector_configs: list[DetectorConfig] = []
        self.voting_strategy = voting_strategy
        self.is_trained = False
        self.training_stats = {}

    async def initialize(self):
        """Initialize the ensemble system."""
        self.container = create_container()

    def add_detector(self, config: DetectorConfig):
        """Add a detector to the ensemble."""
        self.detector_configs.append(config)
        print(
            f"Added detector: {config.name} ({config.algorithm}) with weight {config.weight}"
        )

    async def train_ensemble(self, datasets: dict[str, pd.DataFrame]):
        """Train all detectors in the ensemble on multiple datasets."""

        print(
            f"Training ensemble with {len(self.detector_configs)} detectors on {len(datasets)} datasets"
        )

        detection_service = self.container.detection_service()
        dataset_service = self.container.dataset_service()

        training_results = {}

        for config in self.detector_configs:
            print(f"\nTraining {config.name}...")
            detector_results = {}

            # Create detector
            detector = await detection_service.create_detector(
                name=config.name,
                algorithm=config.algorithm,
                parameters=config.parameters,
            )

            # Train on each dataset
            for dataset_name, data in datasets.items():
                print(f"  Training on {dataset_name} ({len(data)} samples)...")

                # Create dataset
                dataset = await dataset_service.create_from_data(
                    data=data.to_dict("records"),
                    name=f"{dataset_name}_{config.name}",
                    description=f"Training data for {config.name}",
                )

                # Apply preprocessing if specified
                if config.preprocessing:
                    data = self._apply_preprocessing(data, config.preprocessing)

                # Train detector
                start_time = time.time()
                await detection_service.train_detector(detector.id, dataset.id)
                training_time = time.time() - start_time

                detector_results[dataset_name] = {
                    "training_time": training_time,
                    "samples": len(data),
                    "features": len(data.columns),
                }

            self.detectors[config.name] = detector
            training_results[config.name] = detector_results

        self.is_trained = True
        self.training_stats = training_results

        print("\n✅ Ensemble training completed!")
        self._print_training_summary()

    def _apply_preprocessing(
        self, data: pd.DataFrame, preprocessing: dict[str, Any]
    ) -> pd.DataFrame:
        """Apply preprocessing steps to data."""

        processed_data = data.copy()

        if preprocessing.get("standardize", False):
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = scaler.fit_transform(
                processed_data[numeric_cols]
            )

        if preprocessing.get("remove_outliers", False):
            # Remove extreme outliers using IQR method
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = processed_data[col].quantile(0.25)
                Q3 = processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed_data = processed_data[
                    (processed_data[col] >= lower_bound)
                    & (processed_data[col] <= upper_bound)
                ]

        return processed_data

    async def detect_ensemble(
        self,
        data: pd.DataFrame,
        ranking_method: RankingMethod = RankingMethod.WEIGHTED_AVERAGE,
        return_individual: bool = True,
    ) -> EnsembleResult:
        """Run ensemble anomaly detection with ranking and filtering."""

        if not self.is_trained:
            raise ValueError("Ensemble must be trained before detection")

        print(f"Running ensemble detection on {len(data)} samples...")

        detection_service = self.container.detection_service()

        # Store individual results
        individual_predictions = {}
        individual_scores = {}

        # Run detection with each detector
        for config in self.detector_configs:
            print(f"  Running {config.name}...")

            detector = self.detectors[config.name]

            # Apply same preprocessing as during training
            processed_data = data.copy()
            if config.preprocessing:
                processed_data = self._apply_preprocessing(
                    processed_data, config.preprocessing
                )

            # Run detection
            results = await detection_service.detect_batch(
                detector.id, processed_data.to_dict("records")
            )

            # Extract predictions and scores
            predictions = np.array([r.is_anomaly for r in results])
            scores = np.array([r.anomaly_score for r in results])

            individual_predictions[config.name] = predictions
            individual_scores[config.name] = scores

        # Generate ensemble results
        ensemble_predictions, ensemble_scores, confidence = self._combine_predictions(
            individual_predictions, individual_scores
        )

        # Rank anomalies
        ranking = self._rank_anomalies(
            individual_scores, ensemble_scores, ranking_method
        )

        # Calculate consensus level
        consensus_level = self._calculate_consensus(individual_predictions)

        result = EnsembleResult(
            predictions=ensemble_predictions,
            scores=ensemble_scores,
            individual_predictions=individual_predictions if return_individual else {},
            individual_scores=individual_scores if return_individual else {},
            confidence=confidence,
            ranking=ranking,
            consensus_level=consensus_level,
            voting_strategy=self.voting_strategy.value,
            ranking_method=ranking_method.value,
        )

        print("✅ Ensemble detection completed!")
        print(f"   Anomalies detected: {np.sum(ensemble_predictions)}")
        print(f"   Consensus level: {consensus_level:.3f}")
        print(f"   Average confidence: {np.mean(confidence):.3f}")

        return result

    def _combine_predictions(
        self,
        individual_predictions: dict[str, np.ndarray],
        individual_scores: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine individual predictions using voting strategy."""

        n_samples = len(next(iter(individual_predictions.values())))
        n_detectors = len(individual_predictions)

        if self.voting_strategy == VotingStrategy.MAJORITY:
            # Simple majority voting
            votes = np.stack(list(individual_predictions.values()), axis=1)
            ensemble_predictions = (np.sum(votes, axis=1) > n_detectors / 2).astype(int)

            # Average scores for ensemble score
            scores_matrix = np.stack(list(individual_scores.values()), axis=1)
            ensemble_scores = np.mean(scores_matrix, axis=1)

            # Confidence based on agreement level
            confidence = np.abs(np.mean(votes, axis=1) - 0.5) * 2

        elif self.voting_strategy == VotingStrategy.WEIGHTED:
            # Weighted voting based on detector weights
            weights = np.array(
                [
                    next(
                        config.weight
                        for config in self.detector_configs
                        if config.name == name
                    )
                    for name in individual_predictions.keys()
                ]
            )
            weights = weights / np.sum(weights)  # Normalize weights

            votes = np.stack(list(individual_predictions.values()), axis=1)
            weighted_votes = np.average(votes, weights=weights, axis=1)
            ensemble_predictions = (weighted_votes > 0.5).astype(int)

            scores_matrix = np.stack(list(individual_scores.values()), axis=1)
            ensemble_scores = np.average(scores_matrix, weights=weights, axis=1)

            confidence = np.abs(weighted_votes - 0.5) * 2

        elif self.voting_strategy == VotingStrategy.SOFT:
            # Soft voting using scores
            scores_matrix = np.stack(list(individual_scores.values()), axis=1)
            ensemble_scores = np.mean(scores_matrix, axis=1)

            # Use threshold for predictions (could be learned)
            threshold = 0.5
            ensemble_predictions = (ensemble_scores > threshold).astype(int)

            # Confidence based on score distribution
            confidence = 1 - np.std(scores_matrix, axis=1) / np.mean(
                scores_matrix, axis=1
            )
            confidence = np.clip(confidence, 0, 1)

        elif self.voting_strategy == VotingStrategy.CONSENSUS:
            # Require consensus (all detectors agree)
            votes = np.stack(list(individual_predictions.values()), axis=1)
            ensemble_predictions = np.all(votes == 1, axis=1).astype(int)

            scores_matrix = np.stack(list(individual_scores.values()), axis=1)
            ensemble_scores = np.mean(scores_matrix, axis=1)

            # High confidence only when all agree
            agreement = np.var(votes, axis=1)
            confidence = 1 - agreement

        else:  # RANKED
            # Ranking-based ensemble
            scores_matrix = np.stack(list(individual_scores.values()), axis=1)

            # Rank samples by each detector
            rankings = np.zeros_like(scores_matrix)
            for i, scores in enumerate(scores_matrix.T):
                rankings[:, i] = stats.rankdata(
                    -scores, method="ordinal"
                )  # Higher score = lower rank

            # Average ranks
            avg_ranks = np.mean(rankings, axis=1)

            # Convert ranks back to scores (lower rank = higher score)
            ensemble_scores = 1 - (avg_ranks - 1) / (n_samples - 1)

            # Use top percentile for predictions
            threshold = np.percentile(ensemble_scores, 90)  # Top 10%
            ensemble_predictions = (ensemble_scores >= threshold).astype(int)

            # Confidence based on rank stability
            rank_std = np.std(rankings, axis=1)
            confidence = 1 - (rank_std / n_detectors)
            confidence = np.clip(confidence, 0, 1)

        return ensemble_predictions, ensemble_scores, confidence

    def _rank_anomalies(
        self,
        individual_scores: dict[str, np.ndarray],
        ensemble_scores: np.ndarray,
        ranking_method: RankingMethod,
    ) -> np.ndarray:
        """Rank anomalies using specified method."""

        scores_matrix = np.stack(list(individual_scores.values()), axis=1)

        if ranking_method == RankingMethod.AVERAGE_SCORE:
            ranking_scores = np.mean(scores_matrix, axis=1)

        elif ranking_method == RankingMethod.MEDIAN_SCORE:
            ranking_scores = np.median(scores_matrix, axis=1)

        elif ranking_method == RankingMethod.MAX_SCORE:
            ranking_scores = np.max(scores_matrix, axis=1)

        elif ranking_method == RankingMethod.WEIGHTED_AVERAGE:
            weights = np.array(
                [
                    next(
                        config.weight
                        for config in self.detector_configs
                        if config.name == name
                    )
                    for name in individual_scores.keys()
                ]
            )
            weights = weights / np.sum(weights)
            ranking_scores = np.average(scores_matrix, weights=weights, axis=1)

        elif ranking_method == RankingMethod.CONCORDANCE:
            # Kendall's concordance-based ranking
            rankings = np.zeros_like(scores_matrix)
            for i, scores in enumerate(scores_matrix.T):
                rankings[:, i] = stats.rankdata(-scores, method="ordinal")

            # Average concordance with other detectors
            n_detectors = rankings.shape[1]
            concordance_scores = np.zeros(rankings.shape[0])

            for i in range(rankings.shape[0]):
                concordance = 0
                for j in range(n_detectors):
                    for k in range(j + 1, n_detectors):
                        # Kendall's tau correlation between ranks
                        tau, _ = stats.kendalltau(rankings[:, j], rankings[:, k])
                        concordance += tau

                concordance_scores[i] = concordance / (
                    n_detectors * (n_detectors - 1) / 2
                )

            ranking_scores = concordance_scores

        elif ranking_method == RankingMethod.BORDA_COUNT:
            # Borda count ranking
            rankings = np.zeros_like(scores_matrix)
            for i, scores in enumerate(scores_matrix.T):
                rankings[:, i] = stats.rankdata(-scores, method="ordinal")

            # Borda count: higher rank = more points
            n_samples = rankings.shape[0]
            borda_scores = np.sum(n_samples - rankings + 1, axis=1)
            ranking_scores = borda_scores / np.max(borda_scores)

        else:
            ranking_scores = ensemble_scores

        # Return ranking (lower index = higher anomaly score)
        return stats.rankdata(-ranking_scores, method="ordinal") - 1

    def _calculate_consensus(
        self, individual_predictions: dict[str, np.ndarray]
    ) -> float:
        """Calculate consensus level among detectors."""

        predictions_matrix = np.stack(list(individual_predictions.values()), axis=1)

        # Calculate agreement for each sample
        agreements = []
        for i in range(predictions_matrix.shape[0]):
            sample_predictions = predictions_matrix[i, :]
            # Count majority class
            unique, counts = np.unique(sample_predictions, return_counts=True)
            max_agreement = np.max(counts) / len(sample_predictions)
            agreements.append(max_agreement)

        return np.mean(agreements)

    def filter_anomalies(
        self,
        result: EnsembleResult,
        confidence_threshold: float = 0.7,
        consensus_threshold: float = 0.6,
        top_k: int | None = None,
        min_detectors_agree: int = 2,
    ) -> dict[str, np.ndarray]:
        """Filter anomalies based on various criteria."""

        n_samples = len(result.predictions)
        filters = {}

        # High confidence filter
        high_confidence_mask = result.confidence >= confidence_threshold
        filters["high_confidence"] = np.where(
            high_confidence_mask & result.predictions
        )[0]

        # Individual detector agreement filter
        agreement_counts = np.zeros(n_samples)
        for _name, predictions in result.individual_predictions.items():
            agreement_counts += predictions

        agreement_mask = agreement_counts >= min_detectors_agree
        filters["detector_agreement"] = np.where(agreement_mask & result.predictions)[0]

        # Top-K anomalies by ranking
        if top_k:
            top_k_indices = np.argsort(result.ranking)[:top_k]
            anomaly_indices = np.where(result.predictions)[0]
            top_k_anomalies = np.intersect1d(top_k_indices, anomaly_indices)
            filters["top_k"] = top_k_anomalies

        # Combined filter (intersection of all criteria)
        combined_mask = high_confidence_mask & agreement_mask
        if top_k:
            top_k_mask = np.isin(np.arange(n_samples), top_k_indices)
            combined_mask = combined_mask & top_k_mask

        filters["combined"] = np.where(combined_mask & result.predictions)[0]

        return filters

    def analyze_detector_performance(
        self, result: EnsembleResult, true_labels: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Analyze individual detector performance."""

        analysis = {
            "detector_stats": {},
            "correlation_matrix": None,
            "diversity_metrics": {},
        }

        # Individual detector statistics
        for name, predictions in result.individual_predictions.items():
            stats = {
                "anomaly_rate": np.mean(predictions),
                "score_mean": np.mean(result.individual_scores[name]),
                "score_std": np.std(result.individual_scores[name]),
                "score_range": np.ptp(result.individual_scores[name]),
            }

            if true_labels is not None:
                stats["accuracy"] = np.mean(predictions == true_labels)
                stats["precision"] = self._safe_precision(true_labels, predictions)
                stats["recall"] = self._safe_recall(true_labels, predictions)
                stats["f1_score"] = self._safe_f1(true_labels, predictions)
                if len(np.unique(true_labels)) > 1:
                    stats["auc_roc"] = roc_auc_score(
                        true_labels, result.individual_scores[name]
                    )

            analysis["detector_stats"][name] = stats

        # Correlation between detectors
        scores_matrix = np.stack(list(result.individual_scores.values()), axis=1)
        correlation_matrix = np.corrcoef(scores_matrix.T)
        analysis["correlation_matrix"] = correlation_matrix

        # Diversity metrics
        predictions_matrix = np.stack(
            list(result.individual_predictions.values()), axis=1
        )

        # Q-statistic (disagreement measure)
        q_statistics = []
        detector_names = list(result.individual_predictions.keys())

        for i in range(len(detector_names)):
            for j in range(i + 1, len(detector_names)):
                pred_i = predictions_matrix[:, i]
                pred_j = predictions_matrix[:, j]

                # Calculate Q-statistic
                n11 = np.sum((pred_i == 1) & (pred_j == 1))
                n10 = np.sum((pred_i == 1) & (pred_j == 0))
                n01 = np.sum((pred_i == 0) & (pred_j == 1))
                n00 = np.sum((pred_i == 0) & (pred_j == 0))

                if (n11 * n00 + n01 * n10) != 0:
                    q_stat = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
                    q_statistics.append(q_stat)

        analysis["diversity_metrics"]["avg_q_statistic"] = (
            np.mean(q_statistics) if q_statistics else 0
        )
        analysis["diversity_metrics"]["q_statistics"] = q_statistics

        # Disagreement measure
        disagreement = np.mean(np.var(predictions_matrix, axis=1))
        analysis["diversity_metrics"]["disagreement"] = disagreement

        return analysis

    def _safe_precision(self, y_true, y_pred):
        """Calculate precision with safe division."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def _safe_recall(self, y_true, y_pred):
        """Calculate recall with safe division."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def _safe_f1(self, y_true, y_pred):
        """Calculate F1-score with safe division."""
        precision = self._safe_precision(y_true, y_pred)
        recall = self._safe_recall(y_true, y_pred)
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

    def _print_training_summary(self):
        """Print training summary statistics."""

        print("\n📊 Training Summary:")
        print("=" * 50)

        total_time = 0
        total_samples = 0

        for detector_name, dataset_results in self.training_stats.items():
            detector_time = sum(r["training_time"] for r in dataset_results.values())
            detector_samples = sum(r["samples"] for r in dataset_results.values())

            print(f"\n{detector_name}:")
            for dataset_name, stats in dataset_results.items():
                print(
                    f"  {dataset_name}: {stats['training_time']:.2f}s, "
                    f"{stats['samples']} samples, {stats['features']} features"
                )

            print(f"  Total: {detector_time:.2f}s, {detector_samples} samples")

            total_time += detector_time
            total_samples += detector_samples

        print(f"\nOverall: {total_time:.2f}s total training time")
        print(f"Average time per sample: {(total_time / total_samples) * 1000:.2f}ms")

    def visualize_results(
        self,
        result: EnsembleResult,
        data: pd.DataFrame,
        save_path: str | None = None,
    ):
        """Create comprehensive visualization of ensemble results."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Multi-Classifier Ensemble Analysis", fontsize=16, fontweight="bold"
        )

        # 1. Score distribution comparison
        ax1 = axes[0, 0]
        score_data = []
        for name, scores in result.individual_scores.items():
            score_data.extend([(name, score) for score in scores])

        score_df = pd.DataFrame(score_data, columns=["Detector", "Score"])
        sns.boxplot(data=score_df, x="Detector", y="Score", ax=ax1)
        ax1.set_title("Score Distribution by Detector")
        ax1.tick_params(axis="x", rotation=45)

        # 2. Ensemble vs individual predictions
        ax2 = axes[0, 1]
        prediction_rates = []
        detector_names = list(result.individual_predictions.keys()) + ["Ensemble"]

        for name, predictions in result.individual_predictions.items():
            prediction_rates.append(np.mean(predictions))
        prediction_rates.append(np.mean(result.predictions))

        bars = ax2.bar(detector_names, prediction_rates)
        ax2.set_title("Anomaly Detection Rates")
        ax2.set_ylabel("Anomaly Rate")
        ax2.tick_params(axis="x", rotation=45)

        # Highlight ensemble bar
        bars[-1].set_color("red")
        bars[-1].set_alpha(0.7)

        # 3. Confidence distribution
        ax3 = axes[0, 2]
        ax3.hist(result.confidence, bins=30, alpha=0.7, color="green")
        ax3.set_title("Confidence Distribution")
        ax3.set_xlabel("Confidence")
        ax3.set_ylabel("Frequency")
        ax3.axvline(
            np.mean(result.confidence),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(result.confidence):.3f}",
        )
        ax3.legend()

        # 4. Detector correlation heatmap
        ax4 = axes[1, 0]
        scores_matrix = np.stack(list(result.individual_scores.values()), axis=1)
        correlation_matrix = np.corrcoef(scores_matrix.T)

        detector_names_short = [name[:8] for name in result.individual_scores.keys()]
        sns.heatmap(
            correlation_matrix,
            xticklabels=detector_names_short,
            yticklabels=detector_names_short,
            annot=True,
            fmt=".2f",
            ax=ax4,
            cmap="coolwarm",
            center=0,
        )
        ax4.set_title("Detector Score Correlations")

        # 5. Top anomalies ranking
        ax5 = axes[1, 1]
        top_k = min(20, np.sum(result.predictions))
        if top_k > 0:
            top_indices = np.argsort(result.ranking)[:top_k]
            top_scores = result.scores[top_indices]

            ax5.bar(range(len(top_scores)), sorted(top_scores, reverse=True))
            ax5.set_title(f"Top {top_k} Anomaly Scores")
            ax5.set_xlabel("Anomaly Rank")
            ax5.set_ylabel("Ensemble Score")
        else:
            ax5.text(
                0.5,
                0.5,
                "No anomalies detected",
                ha="center",
                va="center",
                transform=ax5.transAxes,
            )
            ax5.set_title("Top Anomaly Scores")

        # 6. Agreement analysis
        ax6 = axes[1, 2]
        if result.individual_predictions:
            predictions_matrix = np.stack(
                list(result.individual_predictions.values()), axis=1
            )
            agreement_counts = np.sum(predictions_matrix, axis=1)

            unique_agreements, counts = np.unique(agreement_counts, return_counts=True)
            ax6.bar(unique_agreements, counts)
            ax6.set_title("Detector Agreement Distribution")
            ax6.set_xlabel("Number of Detectors Agreeing")
            ax6.set_ylabel("Number of Samples")
            ax6.set_xticks(range(len(result.individual_predictions) + 1))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        plt.show()


class DatasetGenerator:
    """Generate synthetic datasets for ensemble testing."""

    @staticmethod
    def generate_fraud_dataset(
        n_samples: int = 1000, contamination: float = 0.1
    ) -> pd.DataFrame:
        """Generate synthetic fraud detection dataset."""

        np.random.seed(42)

        normal_samples = int(n_samples * (1 - contamination))
        fraud_samples = n_samples - normal_samples

        # Normal transactions
        normal_data = {
            "amount": np.random.lognormal(3, 1, normal_samples),
            "merchant_risk_score": np.random.beta(2, 5, normal_samples),
            "transaction_hour": np.random.choice(range(6, 23), normal_samples),
            "days_since_last": np.random.exponential(2, normal_samples),
            "velocity_1h": np.random.poisson(0.5, normal_samples),
            "card_age_days": np.random.exponential(365, normal_samples),
            "is_weekend": np.random.choice([0, 1], normal_samples, p=[0.7, 0.3]),
        }

        # Fraudulent transactions (different patterns)
        fraud_data = {
            "amount": np.concatenate(
                [
                    np.random.lognormal(6, 1, fraud_samples // 2),  # High amounts
                    np.random.uniform(1, 10, fraud_samples // 2),  # Small test amounts
                ]
            ),
            "merchant_risk_score": np.random.beta(5, 2, fraud_samples),  # Higher risk
            "transaction_hour": np.random.choice(
                range(0, 6), fraud_samples
            ),  # Unusual hours
            "days_since_last": np.random.exponential(0.1, fraud_samples),  # Very recent
            "velocity_1h": np.random.poisson(3, fraud_samples),  # High velocity
            "card_age_days": np.random.exponential(30, fraud_samples),  # New cards
            "is_weekend": np.random.choice(
                [0, 1], fraud_samples, p=[0.3, 0.7]
            ),  # Weekend bias
        }

        # Combine datasets
        all_data = {}
        for key in normal_data.keys():
            all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])

        # Add labels
        all_data["is_fraud"] = np.concatenate(
            [np.zeros(normal_samples), np.ones(fraud_samples)]
        )

        # Shuffle
        df = pd.DataFrame(all_data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df

    @staticmethod
    def generate_network_dataset(
        n_samples: int = 1000, contamination: float = 0.05
    ) -> pd.DataFrame:
        """Generate synthetic network intrusion dataset."""

        np.random.seed(43)

        normal_samples = int(n_samples * (1 - contamination))
        intrusion_samples = n_samples - normal_samples

        # Normal network traffic
        normal_data = {
            "packet_size": np.random.normal(1024, 256, normal_samples),
            "connection_duration": np.random.exponential(10, normal_samples),
            "bytes_sent": np.random.lognormal(8, 2, normal_samples),
            "bytes_received": np.random.lognormal(8, 2, normal_samples),
            "packets_per_second": np.random.gamma(2, 5, normal_samples),
            "unique_ports": np.random.poisson(3, normal_samples),
            "protocol_tcp": np.random.choice([0, 1], normal_samples, p=[0.3, 0.7]),
        }

        # Intrusion patterns
        intrusion_data = {
            "packet_size": np.concatenate(
                [
                    np.random.normal(
                        64, 10, intrusion_samples // 3
                    ),  # Small packets (scanning)
                    np.random.normal(
                        1500, 100, intrusion_samples // 3
                    ),  # Large packets (DDoS)
                    np.random.normal(
                        1024, 256, intrusion_samples // 3
                    ),  # Normal size (stealth)
                ]
            ),
            "connection_duration": np.concatenate(
                [
                    np.random.exponential(
                        0.1, intrusion_samples // 2
                    ),  # Very short (scanning)
                    np.random.exponential(
                        100, intrusion_samples // 2
                    ),  # Very long (persistent)
                ]
            ),
            "bytes_sent": np.random.lognormal(12, 3, intrusion_samples),  # High volume
            "bytes_received": np.random.lognormal(
                6, 1, intrusion_samples
            ),  # Low response
            "packets_per_second": np.random.gamma(
                10, 10, intrusion_samples
            ),  # High rate
            "unique_ports": np.random.poisson(20, intrusion_samples),  # Port scanning
            "protocol_tcp": np.random.choice(
                [0, 1], intrusion_samples, p=[0.8, 0.2]
            ),  # More UDP
        }

        # Combine datasets
        all_data = {}
        for key in normal_data.keys():
            all_data[key] = np.concatenate([normal_data[key], intrusion_data[key]])

        all_data["is_intrusion"] = np.concatenate(
            [np.zeros(normal_samples), np.ones(intrusion_samples)]
        )

        df = pd.DataFrame(all_data)
        df = df.sample(frac=1, random_state=43).reset_index(drop=True)

        return df

    @staticmethod
    def generate_sensor_dataset(
        n_samples: int = 1000, contamination: float = 0.08
    ) -> pd.DataFrame:
        """Generate synthetic sensor anomaly dataset."""

        np.random.seed(44)

        normal_samples = int(n_samples * (1 - contamination))
        anomaly_samples = n_samples - normal_samples

        # Normal sensor readings
        time_points = np.linspace(0, 100, normal_samples)
        normal_data = {
            "temperature": 20
            + 5 * np.sin(time_points * 0.1)
            + np.random.normal(0, 0.5, normal_samples),
            "pressure": 1013
            + 10 * np.cos(time_points * 0.05)
            + np.random.normal(0, 1, normal_samples),
            "humidity": 50
            + 20 * np.sin(time_points * 0.08)
            + np.random.normal(0, 2, normal_samples),
            "vibration": 0.1 + 0.05 * np.random.random(normal_samples),
            "power_consumption": 100
            + 20 * np.sin(time_points * 0.12)
            + np.random.normal(0, 3, normal_samples),
        }

        # Anomalous readings
        np.linspace(0, 100, anomaly_samples)
        anomaly_data = {
            "temperature": np.concatenate(
                [
                    np.random.normal(40, 5, anomaly_samples // 3),  # Overheating
                    np.random.normal(5, 2, anomaly_samples // 3),  # Too cold
                    np.random.normal(20, 10, anomaly_samples // 3),  # High variance
                ]
            ),
            "pressure": np.concatenate(
                [
                    np.random.normal(900, 20, anomaly_samples // 2),  # Low pressure
                    np.random.normal(1100, 30, anomaly_samples // 2),  # High pressure
                ]
            ),
            "humidity": np.random.choice([10, 95], anomaly_samples)
            + np.random.normal(0, 5, anomaly_samples),
            "vibration": np.random.exponential(1, anomaly_samples),  # High vibration
            "power_consumption": np.concatenate(
                [
                    np.random.normal(200, 20, anomaly_samples // 2),  # High consumption
                    np.random.normal(20, 5, anomaly_samples // 2),  # Low consumption
                ]
            ),
        }

        # Combine datasets
        all_data = {}
        for key in normal_data.keys():
            all_data[key] = np.concatenate([normal_data[key], anomaly_data[key]])

        all_data["is_anomaly"] = np.concatenate(
            [np.zeros(normal_samples), np.ones(anomaly_samples)]
        )

        df = pd.DataFrame(all_data)
        df = df.sample(frac=1, random_state=44).reset_index(drop=True)

        return df


async def main():
    """Demonstrate multi-classifier ensemble with multiple datasets."""

    print("🔬 Multi-Classifier Ensemble Anomaly Detection Demo")
    print("=" * 60)

    # Generate multiple datasets
    print("📊 Generating synthetic datasets...")
    datasets = {
        "fraud": DatasetGenerator.generate_fraud_dataset(800, 0.1),
        "network": DatasetGenerator.generate_network_dataset(800, 0.05),
        "sensor": DatasetGenerator.generate_sensor_dataset(800, 0.08),
    }

    for name, data in datasets.items():
        anomaly_col = [col for col in data.columns if "is_" in col][0]
        anomaly_rate = data[anomaly_col].mean()
        print(
            f"  {name}: {len(data)} samples, {len(data.columns) - 1} features, {anomaly_rate:.1%} anomalies"
        )

    # Initialize ensemble
    ensemble = MultiClassifierEnsemble(VotingStrategy.WEIGHTED)
    await ensemble.initialize()

    # Configure detectors with different strengths
    detector_configs = [
        DetectorConfig(
            name="IsolationForest_Balanced",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1, "n_estimators": 100, "max_samples": 256},
            weight=1.0,
            preprocessing={"standardize": True},
        ),
        DetectorConfig(
            name="LOF_Local",
            algorithm="LOF",
            parameters={"contamination": 0.1, "n_neighbors": 20},
            weight=0.8,
            preprocessing={"standardize": True, "remove_outliers": True},
        ),
        DetectorConfig(
            name="OCSVM_Robust",
            algorithm="OCSVM",
            parameters={"contamination": 0.1, "kernel": "rbf", "gamma": "scale"},
            weight=0.9,
            preprocessing={"standardize": True},
        ),
        DetectorConfig(
            name="COPOD_Fast",
            algorithm="COPOD",
            parameters={"contamination": 0.1},
            weight=1.1,
            preprocessing={"standardize": False},
        ),
    ]

    # Add detectors to ensemble
    for config in detector_configs:
        ensemble.add_detector(config)

    # Prepare training data (remove labels)
    training_datasets = {}
    true_labels = {}

    for name, data in datasets.items():
        label_col = [col for col in data.columns if "is_" in col][0]
        true_labels[name] = data[label_col].values
        training_datasets[name] = data.drop(columns=[label_col])

    # Train ensemble
    print("\n🎯 Training ensemble...")
    await ensemble.train_ensemble(training_datasets)

    # Test on each dataset
    print("\n🔍 Running ensemble detection...")

    all_results = {}
    all_analyses = {}

    for dataset_name, test_data in training_datasets.items():
        print(f"\n--- Testing on {dataset_name} dataset ---")

        # Run ensemble detection with different ranking methods
        result = await ensemble.detect_ensemble(
            test_data,
            ranking_method=RankingMethod.WEIGHTED_AVERAGE,
            return_individual=True,
        )

        all_results[dataset_name] = result

        # Analyze performance
        analysis = ensemble.analyze_detector_performance(
            result, true_labels[dataset_name]
        )
        all_analyses[dataset_name] = analysis

        # Filter anomalies
        filters = ensemble.filter_anomalies(
            result, confidence_threshold=0.7, min_detectors_agree=2, top_k=10
        )

        print(f"Results for {dataset_name}:")
        print(f"  Total anomalies detected: {np.sum(result.predictions)}")
        print(f"  Consensus level: {result.consensus_level:.3f}")
        print(f"  High confidence anomalies: {len(filters['high_confidence'])}")
        print(f"  Detector agreement anomalies: {len(filters['detector_agreement'])}")
        print(f"  Top-10 anomalies: {len(filters['top_k'])}")
        print(f"  Combined filter: {len(filters['combined'])}")

        # Print individual detector performance
        if true_labels[dataset_name] is not None:
            print("\n  Individual Detector Performance:")
            for detector_name, stats in analysis["detector_stats"].items():
                if "f1_score" in stats:
                    print(
                        f"    {detector_name}: F1={stats['f1_score']:.3f}, "
                        f"Precision={stats['precision']:.3f}, "
                        f"Recall={stats['recall']:.3f}"
                    )

        # Visualize results for first dataset
        if dataset_name == "fraud":
            print(f"\n📈 Creating visualization for {dataset_name} dataset...")
            ensemble.visualize_results(
                result, test_data, f"{dataset_name}_ensemble_analysis.png"
            )

    # Compare ensemble performance across datasets
    print("\n📊 Cross-Dataset Performance Summary:")
    print("=" * 50)

    for dataset_name, result in all_results.items():
        analysis = all_analyses[dataset_name]

        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"  Ensemble anomaly rate: {np.mean(result.predictions):.3f}")
        print(f"  Average confidence: {np.mean(result.confidence):.3f}")
        print(f"  Consensus level: {result.consensus_level:.3f}")

        # Detector diversity
        diversity = analysis["diversity_metrics"]
        print(f"  Detector diversity (disagreement): {diversity['disagreement']:.3f}")
        print(f"  Average Q-statistic: {diversity['avg_q_statistic']:.3f}")

        if true_labels[dataset_name] is not None:
            # Ensemble performance vs ground truth
            ensemble_precision = ensemble._safe_precision(
                true_labels[dataset_name], result.predictions
            )
            ensemble_recall = ensemble._safe_recall(
                true_labels[dataset_name], result.predictions
            )
            ensemble_f1 = ensemble._safe_f1(
                true_labels[dataset_name], result.predictions
            )

            print(f"  Ensemble F1-score: {ensemble_f1:.3f}")
            print(f"  Ensemble Precision: {ensemble_precision:.3f}")
            print(f"  Ensemble Recall: {ensemble_recall:.3f}")

    print("\n✅ Multi-classifier ensemble analysis completed!")
    print("\nKey insights:")
    print("- Ensemble methods provide robust anomaly detection across diverse datasets")
    print("- Different ranking methods help prioritize anomalies by confidence")
    print("- Detector diversity metrics indicate ensemble effectiveness")
    print("- Filtering strategies allow fine-tuning for specific use cases")
    print("- Cross-dataset evaluation reveals algorithm strengths and weaknesses")


if __name__ == "__main__":
    asyncio.run(main())

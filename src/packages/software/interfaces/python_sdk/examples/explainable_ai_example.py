#!/usr/bin/env python3
"""
Pynomaly Explainable AI Example
===============================

This example demonstrates explainable AI functionality for anomaly detection
using feature importance analysis and interpretable explanations.
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from pynomaly_detection.domain.entities import Dataset
from pynomaly_detection.domain.value_objects import ContaminationRate
from pynomaly_detection.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class SimpleExplainer:
    """Simple explainable AI implementation for anomaly detection."""

    def __init__(self, detector):
        self.detector = detector
        self.feature_importance = None
        self.feature_names = None

    def calculate_feature_importance(self, dataset):
        """Calculate feature importance using permutation-based method."""
        if not self.detector.is_fitted:
            raise ValueError("Detector must be fitted before calculating importance")

        self.feature_names = dataset.feature_names or [
            f"feature_{i}" for i in range(dataset.n_features)
        ]

        # Get baseline scores
        baseline_result = self.detector.detect(dataset)
        baseline_scores = np.array([score.value for score in baseline_result.scores])
        baseline_mean = np.mean(baseline_scores)

        importance_scores = []

        print("🔍 Calculating feature importance...")

        for i, feature_name in enumerate(self.feature_names):
            # Create dataset with permuted feature
            permuted_data = dataset.data.copy()
            permuted_feature = np.random.permutation(permuted_data.iloc[:, i].values)
            permuted_data.iloc[:, i] = permuted_feature

            # Create permuted dataset
            permuted_dataset = Dataset(
                name=f"{dataset.name}_permuted_{feature_name}", data=permuted_data
            )

            # Get scores with permuted feature
            try:
                permuted_result = self.detector.detect(permuted_dataset)
                permuted_scores = np.array(
                    [score.value for score in permuted_result.scores]
                )
                permuted_mean = np.mean(permuted_scores)

                # Importance = change in mean score
                importance = abs(baseline_mean - permuted_mean)
                importance_scores.append(importance)

                print(f"   {feature_name}: {importance:.4f}")

            except Exception as e:
                print(f"   {feature_name}: Error - {e}")
                importance_scores.append(0.0)

        # Normalize importance scores
        total_importance = sum(importance_scores)
        if total_importance > 0:
            self.feature_importance = [
                score / total_importance for score in importance_scores
            ]
        else:
            self.feature_importance = [1.0 / len(importance_scores)] * len(
                importance_scores
            )

        return self.feature_importance

    def explain_prediction(self, dataset, sample_idx=None):
        """Explain prediction for a specific sample or top anomalies."""
        if not self.detector.is_fitted:
            raise ValueError("Detector must be fitted before explaining predictions")

        result = self.detector.detect(dataset)

        if sample_idx is None:
            # Find top 5 anomalies
            scores = np.array([score.value for score in result.scores])
            top_indices = np.argsort(scores)[-5:][::-1]
        else:
            top_indices = [sample_idx]

        explanations = []

        for idx in top_indices:
            sample = dataset.data.iloc[idx]
            score = result.scores[idx].value

            explanation = {
                "sample_index": idx,
                "anomaly_score": score,
                "is_anomaly": result.labels[idx] == 1,
                "sample_values": sample.to_dict(),
                "feature_contributions": {},
            }

            # Calculate feature contributions (simplified)
            if self.feature_importance is not None:
                for i, (feature_name, importance) in enumerate(
                    zip(self.feature_names, self.feature_importance, strict=False)
                ):
                    value = sample.iloc[i]
                    # Contribution = importance * deviation from mean
                    mean_value = dataset.data.iloc[:, i].mean()
                    deviation = abs(value - mean_value)
                    contribution = importance * deviation * score

                    explanation["feature_contributions"][feature_name] = {
                        "value": value,
                        "mean": mean_value,
                        "deviation": deviation,
                        "importance": importance,
                        "contribution": contribution,
                    }

            explanations.append(explanation)

        return explanations

    def generate_global_explanation(self, dataset):
        """Generate global explanation for the model."""
        if self.feature_importance is None:
            self.calculate_feature_importance(dataset)

        result = self.detector.detect(dataset)
        scores = np.array([score.value for score in result.scores])

        explanation = {
            "algorithm": self.detector.algorithm_name,
            "total_samples": len(dataset.data),
            "anomalies_detected": np.sum(result.labels),
            "detection_rate": np.mean(result.labels),
            "score_statistics": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
            },
            "feature_importance": {
                name: importance
                for name, importance in zip(self.feature_names, self.feature_importance, strict=False)
            },
            "top_features": sorted(
                zip(self.feature_names, self.feature_importance, strict=False),
                key=lambda x: x[1],
                reverse=True,
            ),
        }

        return explanation


def create_explainable_dataset():
    """Create a dataset designed to show explainable patterns."""
    np.random.seed(42)

    # Normal data with clear patterns
    n_normal = 800

    # Feature 1: Normal around 0
    feature1_normal = np.random.normal(0, 1, n_normal)

    # Feature 2: Correlated with feature 1
    feature2_normal = feature1_normal * 0.7 + np.random.normal(0, 0.5, n_normal)

    # Feature 3: Independent noise
    feature3_normal = np.random.normal(0, 0.8, n_normal)

    # Feature 4: Mostly constant (low importance)
    feature4_normal = np.random.normal(1, 0.1, n_normal)

    # Anomalies with clear deviations
    n_anomalies = 100

    # Anomaly pattern: extreme values in features 1 and 2
    feature1_anom = np.random.choice([-4, 4], n_anomalies) + np.random.normal(
        0, 0.5, n_anomalies
    )
    feature2_anom = feature1_anom * 0.8 + np.random.normal(0, 0.5, n_anomalies)
    feature3_anom = np.random.normal(0, 0.8, n_anomalies)  # Similar to normal
    feature4_anom = np.random.normal(1, 0.1, n_anomalies)  # Similar to normal

    # Combine data
    feature1 = np.concatenate([feature1_normal, feature1_anom])
    feature2 = np.concatenate([feature2_normal, feature2_anom])
    feature3 = np.concatenate([feature3_normal, feature3_anom])
    feature4 = np.concatenate([feature4_normal, feature4_anom])

    # Create labels for evaluation
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

    # Shuffle data
    indices = np.random.permutation(len(feature1))

    df = pd.DataFrame(
        {
            "primary_feature": feature1[indices],
            "correlated_feature": feature2[indices],
            "noise_feature": feature3[indices],
            "constant_feature": feature4[indices],
            "true_label": labels[indices],  # For evaluation only
        }
    )

    # Return dataset without true labels for anomaly detection
    feature_data = df[
        ["primary_feature", "correlated_feature", "noise_feature", "constant_feature"]
    ]

    return (
        Dataset(
            name="Explainable Test Dataset",
            data=feature_data,
            description="Dataset designed to show clear feature importance patterns",
        ),
        df["true_label"].values,
    )


def run_explainable_ai_example():
    """Run the explainable AI example."""
    print("🔍 Pynomaly Explainable AI Example")
    print("=" * 50)

    # Create test dataset
    dataset, true_labels = create_explainable_dataset()
    print(
        f"📊 Created dataset: {len(dataset.data)} samples, {len(dataset.feature_names)} features"
    )

    # Train detector
    print("\n🤖 Training Isolation Forest detector...")
    detector = SklearnAdapter(
        algorithm_name="IsolationForest",
        contamination_rate=ContaminationRate(0.1),
        random_state=42,
        n_estimators=100,
    )

    detector.fit(dataset)
    print("✅ Detector trained successfully")

    # Create explainer
    explainer = SimpleExplainer(detector)

    # Calculate feature importance
    print("\n🧠 Analyzing Feature Importance:")
    importance_scores = explainer.calculate_feature_importance(dataset)

    print("\n📊 Feature Importance Results:")
    for name, importance in zip(dataset.feature_names, importance_scores, strict=False):
        print(
            f"   {name:20}: {importance:.3f} {'🔥' if importance > 0.3 else '🟡' if importance > 0.1 else '🔵'}"
        )

    # Generate global explanation
    print("\n🌍 Global Model Explanation:")
    global_explanation = explainer.generate_global_explanation(dataset)

    print(f"   Algorithm: {global_explanation['algorithm']}")
    print(f"   Total samples: {global_explanation['total_samples']}")
    print(f"   Anomalies detected: {global_explanation['anomalies_detected']}")
    print(f"   Detection rate: {global_explanation['detection_rate']:.1%}")

    print("\n🏆 Top Important Features:")
    for i, (feature, importance) in enumerate(
        global_explanation["top_features"][:3], 1
    ):
        print(f"   {i}. {feature}: {importance:.3f}")

    # Explain specific predictions
    print("\n🔍 Explaining Top Anomalies:")
    explanations = explainer.explain_prediction(dataset)

    for i, explanation in enumerate(explanations[:3], 1):
        print(f"\n   Anomaly #{i} (Index: {explanation['sample_index']}):")
        print(f"      Score: {explanation['anomaly_score']:.3f}")
        print(f"      Is Anomaly: {explanation['is_anomaly']}")

        # Show top contributing features
        contributions = explanation["feature_contributions"]
        sorted_contributions = sorted(
            contributions.items(), key=lambda x: x[1]["contribution"], reverse=True
        )

        print("      Top Contributing Features:")
        for feature, contrib in sorted_contributions[:2]:
            print(
                f"         {feature}: {contrib['value']:.2f} "
                f"(deviation: {contrib['deviation']:.2f}, "
                f"contribution: {contrib['contribution']:.3f})"
            )

    # Calculate accuracy if we have true labels
    result = detector.detect(dataset)
    predicted_labels = result.labels
    accuracy = np.mean(predicted_labels == true_labels)

    print("\n🎯 Model Performance:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   True anomalies: {np.sum(true_labels)}")
    print(f"   Detected anomalies: {np.sum(predicted_labels)}")

    print("\n🎉 Explainable AI example completed successfully!")
    print("\nKey Insights:")
    print("- Primary and correlated features should show high importance")
    print("- Noise and constant features should show low importance")
    print("- Top anomalies should have extreme values in important features")

    return True


def main():
    """Main function."""
    try:
        success = run_explainable_ai_example()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Explainable AI example failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

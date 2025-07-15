#!/usr/bin/env python3
"""
Custom Algorithm Integration Example
===================================

This example demonstrates how to integrate custom anomaly detection algorithms
into the Pynomaly framework using the adapter pattern.
"""

import asyncio
import os
import sys
from typing import Any

import numpy as np

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from pynomaly.domain.entities import DetectionResult, Detector
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.shared.protocols import DetectorProtocol


class CustomStatisticalDetector:
    """
    Custom anomaly detector using statistical methods.

    This is a simple example that uses z-score and IQR methods
    to detect anomalies. In practice, you might implement more
    sophisticated algorithms here.
    """

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.feature_stats = {}
        self.is_trained = False

    def fit(self, X: np.ndarray):
        """Train the detector by computing feature statistics."""
        self.feature_stats = {}

        for i in range(X.shape[1]):
            feature_data = X[:, i]

            # Z-score statistics
            mean = np.mean(feature_data)
            std = np.std(feature_data)

            # IQR statistics
            q1 = np.percentile(feature_data, 25)
            q3 = np.percentile(feature_data, 75)
            iqr = q3 - q1

            self.feature_stats[i] = {
                "mean": mean,
                "std": std,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": q1 - self.iqr_multiplier * iqr,
                "upper_bound": q3 + self.iqr_multiplier * iqr,
            }

        self.is_trained = True

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for input data."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before making predictions")

        scores = []

        for row in X:
            max_z_score = 0
            max_iqr_violation = 0

            for i, value in enumerate(row):
                stats = self.feature_stats[i]

                # Z-score
                if stats["std"] > 0:
                    z_score = abs((value - stats["mean"]) / stats["std"])
                    max_z_score = max(max_z_score, z_score)

                # IQR violation
                if value < stats["lower_bound"]:
                    iqr_violation = (stats["lower_bound"] - value) / stats["iqr"]
                    max_iqr_violation = max(max_iqr_violation, iqr_violation)
                elif value > stats["upper_bound"]:
                    iqr_violation = (value - stats["upper_bound"]) / stats["iqr"]
                    max_iqr_violation = max(max_iqr_violation, iqr_violation)

            # Combine z-score and IQR violation
            combined_score = max(max_z_score / self.z_threshold, max_iqr_violation)
            scores.append(combined_score)

        return np.array(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for anomaly, -1 for normal)."""
        scores = self.decision_function(X)
        return np.where(scores > 1.0, 1, -1)


class CustomStatisticalAdapter(DetectorProtocol):
    """
    Adapter to integrate CustomStatisticalDetector with Pynomaly framework.

    This adapter implements the DetectorProtocol interface, allowing
    the custom algorithm to work seamlessly with the Pynomaly system.
    """

    def __init__(self, **parameters):
        self.parameters = parameters
        self.detector = None
        self.is_trained = False

    async def create_detector(self, name: str, **parameters) -> Detector:
        """Create a new detector instance."""
        # Merge default parameters with provided ones
        merged_params = {"z_threshold": 3.0, "iqr_multiplier": 1.5, **parameters}

        self.detector = CustomStatisticalDetector(
            z_threshold=merged_params["z_threshold"],
            iqr_multiplier=merged_params["iqr_multiplier"],
        )

        # Create detector entity
        detector = Detector(
            name=name,
            algorithm="CustomStatistical",
            parameters=merged_params,
            metadata={
                "algorithm_type": "statistical",
                "supports_streaming": True,
                "supports_batch": True,
                "feature_requirements": "numerical",
            },
        )

        return detector

    async def train(self, detector: Detector, training_data: list[dict[str, Any]]):
        """Train the detector with training data."""
        if not self.detector:
            raise ValueError("Detector not created. Call create_detector first.")

        # Convert training data to numpy array
        X = self._prepare_data(training_data)

        # Train the underlying detector
        self.detector.fit(X)
        self.is_trained = True

    async def detect_batch(
        self, detector: Detector, data: list[dict[str, Any]]
    ) -> list[DetectionResult]:
        """Perform batch anomaly detection."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before detection")

        X = self._prepare_data(data)
        scores = self.detector.decision_function(X)
        predictions = self.detector.predict(X)

        results = []
        for i, (score, prediction) in enumerate(zip(scores, predictions, strict=False)):
            result = DetectionResult(
                detector_id=detector.id,
                data_point=data[i],
                anomaly_score=AnomalyScore(float(score)),
                is_anomaly=prediction == 1,
                explanation=self._generate_explanation(data[i], score, prediction),
                metadata={
                    "algorithm": "CustomStatistical",
                    "z_threshold": self.detector.z_threshold,
                    "iqr_multiplier": self.detector.iqr_multiplier,
                },
            )
            results.append(result)

        return results

    async def detect_single(
        self, detector: Detector, data_point: dict[str, Any]
    ) -> DetectionResult:
        """Perform single point anomaly detection."""
        results = await self.detect_batch(detector, [data_point])
        return results[0]

    def _prepare_data(self, data: list[dict[str, Any]]) -> np.ndarray:
        """Convert data to numpy array format."""
        if not data:
            raise ValueError("No data provided")

        # Get all numeric features
        features = list(data[0].keys())
        X = []

        for row in data:
            numeric_row = []
            for feature in features:
                value = row.get(feature, 0)
                # Convert to float, handle non-numeric values
                try:
                    numeric_row.append(float(value))
                except (ValueError, TypeError):
                    numeric_row.append(0.0)
            X.append(numeric_row)

        return np.array(X)

    def _generate_explanation(
        self, data_point: dict[str, Any], score: float, prediction: int
    ) -> str:
        """Generate human-readable explanation for the detection result."""
        if prediction == 1:
            if score > 2.0:
                return f"Strong statistical anomaly detected (score: {score:.2f}). Multiple features deviate significantly from normal patterns."
            else:
                return f"Statistical anomaly detected (score: {score:.2f}). Some features are outside normal ranges."
        else:
            return f"Normal data point (score: {score:.2f}). All features within expected statistical bounds."


async def demonstrate_custom_algorithm():
    """Demonstrate the custom algorithm integration."""
    print("🔧 Custom Algorithm Integration Demo")
    print("=" * 50)

    # Generate sample data
    print("📊 Generating sample data...")
    np.random.seed(42)

    # Normal data (2D for simplicity)
    normal_data = []
    for _ in range(100):
        x = np.random.normal(0, 1)
        y = np.random.normal(0, 1)
        normal_data.append({"feature_1": x, "feature_2": y})

    # Anomalous data
    anomalous_data = []
    for _ in range(10):
        x = np.random.normal(0, 1) + np.random.choice([-3, 3])  # Shift
        y = np.random.normal(0, 1) + np.random.choice([-3, 3])  # Shift
        anomalous_data.append({"feature_1": x, "feature_2": y})

    # Test data (mix of normal and anomalous)
    test_data = normal_data[-20:] + anomalous_data
    np.random.shuffle(test_data)

    print(f"   - Training samples: {len(normal_data) - 20}")
    print(f"   - Test samples: {len(test_data)} (10 anomalies, 20 normal)")

    # Create custom adapter
    print("\n🔧 Creating custom statistical detector...")
    adapter = CustomStatisticalAdapter()

    # Create detector with custom parameters
    detector = await adapter.create_detector(
        name="Custom Statistical Detector", z_threshold=2.5, iqr_multiplier=1.5
    )

    print(f"   Algorithm: {detector.algorithm}")
    print(f"   Parameters: {detector.parameters}")

    # Train the detector
    print("\n🎯 Training detector...")
    training_data = normal_data[:-20]  # Exclude last 20 for testing
    await adapter.train(detector, training_data)

    # Test detection
    print(f"\n🔍 Testing on {len(test_data)} samples...")
    results = await adapter.detect_batch(detector, test_data)

    # Analyze results
    detected_anomalies = sum(1 for r in results if r.is_anomaly)

    print("\n📊 Detection Results:")
    print(f"   Detected anomalies: {detected_anomalies}")
    print("   Expected anomalies: 10")

    # Show detailed results
    print("\n🔍 Detailed Results (first 10):")
    for i, result in enumerate(results[:10]):
        status = "🚨 ANOMALY" if result.is_anomaly else "✅ NORMAL"
        score = result.anomaly_score.value
        f1 = result.data_point["feature_1"]
        f2 = result.data_point["feature_2"]
        print(
            f"   {i + 1:2d}. {status} | Score: {score:.3f} | F1: {f1:6.2f}, F2: {f2:6.2f}"
        )

    # Show explanations for anomalies
    anomaly_results = [r for r in results if r.is_anomaly]
    if anomaly_results:
        print("\n💬 Explanations for detected anomalies:")
        for i, result in enumerate(anomaly_results[:5]):
            print(f"   {i + 1}. {result.explanation}")


class CustomEnsembleDetector:
    """
    Custom ensemble detector that combines multiple base detectors.

    This demonstrates how to create more complex custom algorithms
    that leverage existing ones.
    """

    def __init__(self, base_detectors: list[Any], voting_strategy: str = "majority"):
        self.base_detectors = base_detectors
        self.voting_strategy = voting_strategy
        self.is_trained = False

    def fit(self, X: np.ndarray):
        """Train all base detectors."""
        for detector in self.base_detectors:
            detector.fit(X)
        self.is_trained = True

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Combine scores from all base detectors."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        all_scores = []
        for detector in self.base_detectors:
            scores = detector.decision_function(X)
            all_scores.append(scores)

        # Combine scores based on strategy
        all_scores = np.array(all_scores)

        if self.voting_strategy == "average":
            return np.mean(all_scores, axis=0)
        elif self.voting_strategy == "max":
            return np.max(all_scores, axis=0)
        elif self.voting_strategy == "majority":
            # Convert to predictions first, then vote
            predictions = np.array([s > 1.0 for s in all_scores])
            votes = np.sum(predictions, axis=0)
            # Return vote proportion as score
            return votes / len(self.base_detectors) * 2  # Scale to match other scores
        else:
            return np.mean(all_scores, axis=0)  # Default to average

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using ensemble voting."""
        if self.voting_strategy == "majority":
            all_predictions = []
            for detector in self.base_detectors:
                predictions = detector.predict(X)
                all_predictions.append(predictions == 1)  # Convert to boolean

            # Majority vote
            all_predictions = np.array(all_predictions)
            votes = np.sum(all_predictions, axis=0)
            return np.where(votes > len(self.base_detectors) / 2, 1, -1)
        else:
            scores = self.decision_function(X)
            return np.where(scores > 1.0, 1, -1)


class CustomEnsembleAdapter(DetectorProtocol):
    """Adapter for custom ensemble detector."""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.detector = None
        self.is_trained = False

    async def create_detector(self, name: str, **parameters) -> Detector:
        """Create ensemble detector with multiple base detectors."""
        voting_strategy = parameters.get("voting_strategy", "majority")

        # Create base detectors
        base_detectors = [
            CustomStatisticalDetector(z_threshold=2.0, iqr_multiplier=1.5),
            CustomStatisticalDetector(z_threshold=3.0, iqr_multiplier=2.0),
            CustomStatisticalDetector(z_threshold=2.5, iqr_multiplier=1.2),
        ]

        self.detector = CustomEnsembleDetector(base_detectors, voting_strategy)

        detector = Detector(
            name=name,
            algorithm="CustomEnsemble",
            parameters={
                "voting_strategy": voting_strategy,
                "n_base_detectors": len(base_detectors),
            },
            metadata={
                "algorithm_type": "ensemble",
                "base_algorithms": ["CustomStatistical"] * len(base_detectors),
            },
        )

        return detector

    async def train(self, detector: Detector, training_data: list[dict[str, Any]]):
        """Train the ensemble detector."""
        X = self._prepare_data(training_data)
        self.detector.fit(X)
        self.is_trained = True

    async def detect_batch(
        self, detector: Detector, data: list[dict[str, Any]]
    ) -> list[DetectionResult]:
        """Batch detection with ensemble."""
        X = self._prepare_data(data)
        scores = self.detector.decision_function(X)
        predictions = self.detector.predict(X)

        results = []
        for i, (score, prediction) in enumerate(zip(scores, predictions, strict=False)):
            result = DetectionResult(
                detector_id=detector.id,
                data_point=data[i],
                anomaly_score=AnomalyScore(float(score)),
                is_anomaly=prediction == 1,
                explanation=f"Ensemble decision (score: {score:.3f}) from {len(self.detector.base_detectors)} base detectors",
                metadata={
                    "algorithm": "CustomEnsemble",
                    "voting_strategy": self.detector.voting_strategy,
                    "n_base_detectors": len(self.detector.base_detectors),
                },
            )
            results.append(result)

        return results

    async def detect_single(
        self, detector: Detector, data_point: dict[str, Any]
    ) -> DetectionResult:
        """Single point detection."""
        results = await self.detect_batch(detector, [data_point])
        return results[0]

    def _prepare_data(self, data: list[dict[str, Any]]) -> np.ndarray:
        """Convert data to numpy array."""
        features = list(data[0].keys())
        X = []
        for row in data:
            numeric_row = []
            for feature in features:
                try:
                    numeric_row.append(float(row.get(feature, 0)))
                except (ValueError, TypeError):
                    numeric_row.append(0.0)
            X.append(numeric_row)
        return np.array(X)


async def demonstrate_custom_ensemble():
    """Demonstrate custom ensemble algorithm."""
    print("\n🎭 Custom Ensemble Algorithm Demo")
    print("=" * 50)

    # Generate more complex data
    print("📊 Generating complex dataset...")
    np.random.seed(42)

    # Normal data with correlations
    normal_data = []
    for _ in range(200):
        x = np.random.normal(0, 1)
        y = 0.5 * x + np.random.normal(0, 0.5)  # Correlated
        z = np.random.uniform(-2, 2)  # Independent
        normal_data.append({"x": x, "y": y, "z": z})

    # Different types of anomalies
    anomalous_data = []
    # Type 1: Outliers in individual features
    for _ in range(5):
        x = np.random.choice([-4, 4])
        y = 0.5 * x + np.random.normal(0, 0.5)
        z = np.random.uniform(-2, 2)
        anomalous_data.append({"x": x, "y": y, "z": z})

    # Type 2: Correlation anomalies
    for _ in range(5):
        x = np.random.normal(0, 1)
        y = -2 * x + np.random.normal(0, 0.5)  # Wrong correlation
        z = np.random.uniform(-2, 2)
        anomalous_data.append({"x": x, "y": y, "z": z})

    test_data = normal_data[-50:] + anomalous_data
    np.random.shuffle(test_data)

    print(f"   - Training samples: {len(normal_data) - 50}")
    print(f"   - Test samples: {len(test_data)} (10 anomalies)")

    # Test different voting strategies
    strategies = ["majority", "average", "max"]

    for strategy in strategies:
        print(f"\n🔧 Testing ensemble with '{strategy}' voting...")

        adapter = CustomEnsembleAdapter()
        detector = await adapter.create_detector(
            name=f"Custom Ensemble ({strategy})", voting_strategy=strategy
        )

        # Train
        training_data = normal_data[:-50]
        await adapter.train(detector, training_data)

        # Test
        results = await adapter.detect_batch(detector, test_data)

        detected = sum(1 for r in results if r.is_anomaly)
        print(f"   Detected anomalies: {detected} (expected: 10)")


if __name__ == "__main__":
    print("🔧 Pynomaly Custom Algorithm Integration Examples")
    print("=" * 60)

    # Run demonstrations
    asyncio.run(demonstrate_custom_algorithm())
    asyncio.run(demonstrate_custom_ensemble())

    print("\n✅ Custom algorithm examples completed!")
    print("\nKey concepts demonstrated:")
    print("- Implementing custom anomaly detection algorithms")
    print("- Creating adapters to integrate with Pynomaly framework")
    print("- Building ensemble methods from custom base detectors")
    print("- Providing explanations and metadata for custom algorithms")
    print("\nNext steps:")
    print("- Register custom algorithms in the AlgorithmRegistry")
    print("- Add hyperparameter optimization for custom algorithms")
    print("- Implement streaming support for custom detectors")
    print("- Create unit tests for custom algorithm adapters")

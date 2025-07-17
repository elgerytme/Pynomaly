#!/usr/bin/env python3
"""
Pynomaly AutoML Example
=======================

This example demonstrates AutoML functionality using multiple algorithms
and basic hyperparameter optimization without external dependencies.
"""

"""
TODO: This file needs dependency injection refactoring.
Replace direct monorepo imports with dependency injection.
Use interfaces/shared/base_entity.py for abstractions.
"""



import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time

import numpy as np
import pandas as pd

from interfaces.domain.entities import Dataset
from interfaces.domain.value_objects import ContaminationRate
from monorepo.infrastructure.adapters.pyod_adapter import PyODAdapter
from monorepo.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class SimpleAutoML:
    """Simple AutoML implementation for anomaly detection."""

    def __init__(self):
        self.results = []
        self.best_algorithm = None
        self.best_score = -float("inf")

    def evaluate_algorithm(self, algorithm_class, algorithm_name, dataset, params=None):
        """Evaluate a single algorithm with given parameters."""
        params = params or {}

        try:
            # Create detector
            detector = algorithm_class(
                algorithm_name=algorithm_name,
                contamination_rate=ContaminationRate(0.1),
                **params,
            )

            # Train and detect
            start_time = time.time()
            detector.fit(dataset)
            result = detector.detect(dataset)
            execution_time = time.time() - start_time

            # Calculate metrics
            n_anomalies = len(result.anomalies)
            detection_rate = n_anomalies / len(dataset.data)
            avg_score = np.mean([score.value for score in result.scores])

            # Store result
            result_data = {
                "algorithm": algorithm_name,
                "params": params,
                "n_anomalies": n_anomalies,
                "detection_rate": detection_rate,
                "avg_score": avg_score,
                "execution_time": execution_time,
                "success": True,
            }

            self.results.append(result_data)

            # Update best if this is better (using avg_score as metric)
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_algorithm = {
                    "class": algorithm_class,
                    "name": algorithm_name,
                    "params": params,
                    "score": avg_score,
                }

            return result_data

        except Exception as e:
            # Store failed result
            result_data = {
                "algorithm": algorithm_name,
                "params": params,
                "error": str(e),
                "execution_time": 0,
                "success": False,
            }
            self.results.append(result_data)
            return result_data

    def run_automl(self, dataset, max_algorithms=10):
        """Run AutoML on multiple algorithms."""
        print(f"🤖 Running SimpleAutoML on {len(dataset.data)} samples...")

        # Define algorithms to test
        sklearn_algorithms = [
            ("IsolationForest", {}),
            ("LocalOutlierFactor", {}),
            ("OneClassSVM", {"kernel": "rbf"}),
            ("EllipticEnvelope", {}),
        ]

        pyod_algorithms = [
            ("LOF", {}),
            ("COPOD", {}),
            ("ECOD", {}),
            ("PCA", {}),
            ("HBOS", {}),
            ("KNN", {}),
        ]

        total_tested = 0

        # Test Sklearn algorithms
        print("\n📊 Testing Sklearn Algorithms:")
        for algo_name, params in sklearn_algorithms[: max_algorithms // 2]:
            if total_tested >= max_algorithms:
                break
            print(f"   Testing {algo_name}...", end=" ")
            result = self.evaluate_algorithm(SklearnAdapter, algo_name, dataset, params)
            if result["success"]:
                print(
                    f"✅ Score: {result['avg_score']:.3f}, Time: {result['execution_time']:.2f}s"
                )
            else:
                print(f"❌ Failed: {result['error'][:50]}...")
            total_tested += 1

        # Test PyOD algorithms
        print("\n📊 Testing PyOD Algorithms:")
        for algo_name, params in pyod_algorithms[: max_algorithms // 2]:
            if total_tested >= max_algorithms:
                break
            print(f"   Testing {algo_name}...", end=" ")
            result = self.evaluate_algorithm(PyODAdapter, algo_name, dataset, params)
            if result["success"]:
                print(
                    f"✅ Score: {result['avg_score']:.3f}, Time: {result['execution_time']:.2f}s"
                )
            else:
                print(f"❌ Failed: {result['error'][:50]}...")
            total_tested += 1

        return self.get_summary()

    def get_summary(self):
        """Get AutoML results summary."""
        successful_results = [r for r in self.results if r["success"]]

        if not successful_results:
            return {"error": "No algorithms succeeded"}

        # Sort by score
        sorted_results = sorted(
            successful_results, key=lambda x: x["avg_score"], reverse=True
        )

        summary = {
            "total_tested": len(self.results),
            "successful": len(successful_results),
            "best_algorithm": (
                self.best_algorithm["name"] if self.best_algorithm else None
            ),
            "best_score": self.best_score if self.best_algorithm else None,
            "top_3": sorted_results[:3],
            "avg_execution_time": np.mean(
                [r["execution_time"] for r in successful_results]
            ),
            "total_time": sum([r["execution_time"] for r in successful_results]),
        }

        return summary


def create_automl_test_data():
    """Create test dataset for AutoML."""
    np.random.seed(42)

    # Generate normal data (two clusters)
    cluster1 = np.random.normal([2, 2], [0.8, 0.8], (400, 2))
    cluster2 = np.random.normal([-2, -2], [0.8, 0.8], (400, 2))
    normal_data = np.vstack([cluster1, cluster2])

    # Generate anomalies (scattered)
    anomalies = np.random.uniform(-6, 6, (100, 2))

    # Combine data
    data = np.vstack([normal_data, anomalies])

    # Shuffle
    indices = np.random.permutation(len(data))
    data = data[indices]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["feature_1", "feature_2"])

    return Dataset(
        name="AutoML Test Dataset",
        data=df,
        description="Synthetic dataset with 2 clusters and scattered anomalies",
    )


def run_automl_example():
    """Run the AutoML example."""
    print("🧠 Pynomaly AutoML Example")
    print("=" * 50)

    # Create test dataset
    dataset = create_automl_test_data()
    print(
        f"📊 Created dataset: {len(dataset.data)} samples, {len(dataset.feature_names)} features"
    )

    # Run AutoML
    automl = SimpleAutoML()
    summary = automl.run_automl(dataset, max_algorithms=10)

    # Display results
    print("\n🎯 AutoML Results Summary:")
    print("-" * 40)

    if "error" in summary:
        print(f"❌ AutoML failed: {summary['error']}")
        return False

    print(f"Total algorithms tested: {summary['total_tested']}")
    print(f"Successful algorithms: {summary['successful']}")
    print(f"Best algorithm: {summary['best_algorithm']}")
    print(f"Best score: {summary['best_score']:.3f}")
    print(f"Total execution time: {summary['total_time']:.2f}s")
    print(f"Average time per algorithm: {summary['avg_execution_time']:.2f}s")

    print("\n🏆 Top 3 Algorithms:")
    for i, result in enumerate(summary["top_3"], 1):
        print(
            f"{i}. {result['algorithm']:15} | Score: {result['avg_score']:.3f} | "
            f"Anomalies: {result['n_anomalies']:2} | Time: {result['execution_time']:.2f}s"
        )

    print("\n🎉 AutoML example completed successfully!")
    print("\nNext steps:")
    print("- Try different datasets with varying characteristics")
    print("- Implement cross-validation for more robust evaluation")
    print("- Add hyperparameter optimization with grid search")
    print("- Integrate with Optuna for advanced optimization")

    return True


def main():
    """Main function."""
    try:
        success = run_automl_example()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ AutoML example failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

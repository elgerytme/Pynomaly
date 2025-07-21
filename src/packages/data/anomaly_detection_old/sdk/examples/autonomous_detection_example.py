"""Example demonstrating autonomous anomaly detection capabilities."""

"""
TODO: This file needs dependency injection refactoring.
Replace direct monorepo imports with dependency injection.
Use interfaces/shared/base_entity.py for abstractions.
"""



import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from interfaces.application.services.autonomous_service import (
    AutonomousConfig,
    AutonomousDetectionService,
)
from monorepo.infrastructure.data_loaders import (
    CSVLoader,
    ExcelLoader,
    JSONLoader,
    ParquetLoader,
)
from monorepo.infrastructure.repositories.in_memory_repositories import (
    InMemoryDetectionResultRepository,
    InMemoryDetectorRepository,
)


def create_sample_datasets():
    """Create sample datasets for demonstration."""

    # Create output directory
    output_dir = Path("autonomous_demo_data")
    output_dir.mkdir(exist_ok=True)

    # Dataset 1: Normal tabular data with some anomalies
    np.random.seed(42)
    n_samples = 1000

    # Normal data
    normal_data = np.random.multivariate_normal(
        mean=[0, 0, 0, 0], cov=np.eye(4), size=n_samples - 50
    )

    # Anomalous data
    anomaly_data = np.random.uniform(-5, 5, (50, 4))

    # Combine
    data = np.vstack([normal_data, anomaly_data])

    # Create DataFrame
    df1 = pd.DataFrame(data, columns=["feature1", "feature2", "feature3", "feature4"])
    df1["category"] = np.random.choice(["A", "B", "C"], n_samples)
    df1["timestamp"] = pd.date_range("2023-01-01", periods=n_samples, freq="1H")

    # Save as CSV
    csv_file = output_dir / "tabular_data.csv"
    df1.to_csv(csv_file, index=False)

    # Dataset 2: High-dimensional data
    high_dim_data = np.random.normal(0, 1, (500, 50))
    # Add correlated anomalies
    high_dim_data[-25:, :] = np.random.normal(3, 0.5, (25, 50))

    df2 = pd.DataFrame(high_dim_data, columns=[f"feature_{i}" for i in range(50)])

    # Save as Parquet
    parquet_file = output_dir / "high_dimensional.parquet"
    df2.to_parquet(parquet_file, index=False)

    # Dataset 3: JSON data with nested structure
    json_data = []
    for i in range(300):
        is_anomaly = i >= 280  # Last 20 are anomalies

        record = {
            "user_id": i,
            "profile": {
                "age": (
                    np.random.randint(18, 80)
                    if not is_anomaly
                    else np.random.randint(100, 200)
                ),
                "score": (
                    np.random.normal(0.5, 0.1)
                    if not is_anomaly
                    else np.random.normal(2.0, 0.5)
                ),
                "category": np.random.choice(["premium", "basic", "trial"]),
            },
            "activity": {
                "sessions": (
                    np.random.poisson(10) if not is_anomaly else np.random.poisson(100)
                ),
                "duration_minutes": (
                    np.random.exponential(30)
                    if not is_anomaly
                    else np.random.exponential(300)
                ),
                "last_login": f"2023-01-{np.random.randint(1, 31):02d}",
            },
        }
        json_data.append(record)

    # Save as JSON
    import json

    json_file = output_dir / "user_activity.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)

    # Dataset 4: Time series data
    np.random.seed(123)
    time_range = pd.date_range("2023-01-01", periods=2000, freq="1H")

    # Base signal with trend and seasonality
    trend = np.linspace(0, 2, len(time_range))
    seasonal = 0.5 * np.sin(
        2 * np.pi * np.arange(len(time_range)) / 24
    )  # Daily pattern
    noise = np.random.normal(0, 0.1, len(time_range))

    signal = trend + seasonal + noise

    # Add anomalies at specific points
    anomaly_indices = [200, 500, 800, 1200, 1600]
    for idx in anomaly_indices:
        signal[idx] = signal[idx] + np.random.uniform(2, 4)

    df4 = pd.DataFrame(
        {
            "timestamp": time_range,
            "value": signal,
            "moving_avg": pd.Series(signal).rolling(24).mean(),
            "hour": time_range.hour,
            "day_of_week": time_range.dayofweek,
        }
    )

    # Save as CSV
    timeseries_file = output_dir / "timeseries_data.csv"
    df4.to_csv(timeseries_file, index=False)

    print(f"Sample datasets created in {output_dir}/")
    return {
        "tabular": csv_file,
        "high_dimensional": parquet_file,
        "json_nested": json_file,
        "timeseries": timeseries_file,
    }


async def run_autonomous_detection_demo():
    """Demonstrate autonomous detection on various data types."""

    print("🤖 Autonomous Anomaly Detection Demo")
    print("=" * 50)

    # Create sample datasets
    datasets = create_sample_datasets()

    # Setup autonomous service
    data_loaders = {
        "csv": CSVLoader(),
        "parquet": ParquetLoader(),
        "json": JSONLoader(),
        "excel": ExcelLoader(),
    }

    autonomous_service = AutonomousDetectionService(
        detector_repository=InMemoryDetectorRepository(),
        result_repository=InMemoryDetectionResultRepository(),
        data_loaders=data_loaders,
    )

    # Test different configurations
    configs = {
        "quick": AutonomousConfig(
            max_algorithms=2, auto_tune_hyperparams=False, verbose=True
        ),
        "comprehensive": AutonomousConfig(
            max_algorithms=4,
            auto_tune_hyperparams=True,
            confidence_threshold=0.7,
            verbose=True,
        ),
        "production": AutonomousConfig(
            max_algorithms=3,
            auto_tune_hyperparams=True,
            save_results=True,
            export_results=True,
            export_format="csv",
            verbose=False,
        ),
    }

    # Run detection on each dataset
    for dataset_name, dataset_path in datasets.items():
        print(f"\n📊 Processing {dataset_name.upper()} dataset...")
        print(f"Source: {dataset_path}")

        # Use different config based on dataset
        if dataset_name == "high_dimensional":
            config = configs["comprehensive"]
        elif dataset_name == "timeseries":
            config = configs["production"]
        else:
            config = configs["quick"]

        try:
            # Run autonomous detection
            results = await autonomous_service.detect_autonomous(
                str(dataset_path), config
            )

            # Display results summary
            _display_demo_results(dataset_name, results)

        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            continue

    print("\n🎉 Autonomous detection demo completed!")
    print("\nTo run autonomous detection on your own data:")
    print("  pynomaly auto detect your_data.csv")
    print("  pynomaly auto profile your_data.json")
    print("  pynomaly auto quick your_data.parquet --output results.csv")


def _display_demo_results(dataset_name: str, results: dict):
    """Display results summary for demo."""

    auto_results = results.get("autonomous_detection_results", {})

    if not auto_results.get("success"):
        print(f"❌ Detection failed for {dataset_name}")
        return

    print(f"✅ Detection successful for {dataset_name}")

    # Data profile
    profile = auto_results.get("data_profile", {})
    print(f"   📈 Samples: {profile.get('samples', 0):,}")
    print(f"   📊 Features: {profile.get('features', 0)}")
    print(f"   🔢 Numeric: {profile.get('numeric_features', 0)}")
    print(f"   🏷️  Categorical: {profile.get('categorical_features', 0)}")
    print(f"   🧩 Complexity: {profile.get('complexity_score', 0):.2f}")

    # Best result
    best_result = auto_results.get("best_result")
    if best_result:
        summary = best_result.get("summary", {})
        print(f"   🏆 Best Algorithm: {best_result.get('algorithm', 'Unknown')}")
        print(f"   🚨 Anomalies: {summary.get('total_anomalies', 0)}")
        print(f"   📊 Rate: {summary.get('anomaly_rate', '0%')}")
        print(f"   🎯 Confidence: {summary.get('confidence', 'Unknown')}")

    # Algorithm recommendations
    recommendations = auto_results.get("algorithm_recommendations", [])
    if recommendations:
        print(f"   🧠 Algorithms tried: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:3], 1):
            confidence = rec.get("confidence", 0)
            print(f"      {i}. {rec.get('algorithm', 'Unknown')} ({confidence:.1%})")


async def demonstrate_data_profiling():
    """Demonstrate data profiling capabilities."""

    print("\n🔍 Data Profiling Demo")
    print("=" * 30)

    # Create a complex dataset for profiling
    np.random.seed(42)

    # Mixed data types
    df = pd.DataFrame(
        {
            "numeric_normal": np.random.normal(0, 1, 1000),
            "numeric_skewed": np.random.exponential(2, 1000),
            "categorical_high_card": np.random.choice(
                [f"cat_{i}" for i in range(100)], 1000
            ),
            "categorical_low_card": np.random.choice(["A", "B", "C"], 1000),
            "boolean": np.random.choice([True, False], 1000),
            "datetime": pd.date_range("2023-01-01", periods=1000, freq="1H"),
            "text": [f"text_sample_{i}" for i in range(1000)],
            "sparse": np.random.choice([0, 0, 0, 0, 1], 1000),  # Mostly zeros
            "correlated": None,  # Will set based on numeric_normal
        }
    )

    # Add correlation
    df["correlated"] = df["numeric_normal"] * 0.8 + np.random.normal(0, 0.2, 1000)

    # Add missing values
    missing_indices = np.random.choice(1000, 50, replace=False)
    df.loc[missing_indices, "numeric_normal"] = np.nan

    # Save for profiling
    profile_file = Path("autonomous_demo_data/profile_demo.csv")
    df.to_csv(profile_file, index=False)

    # Setup service
    data_loaders = {"csv": CSVLoader()}

    autonomous_service = AutonomousDetectionService(
        detector_repository=InMemoryDetectorRepository(),
        result_repository=InMemoryDetectionResultRepository(),
        data_loaders=data_loaders,
    )

    config = AutonomousConfig(verbose=True)

    # Load and profile
    dataset = await autonomous_service._auto_load_data(str(profile_file), config)
    profile = await autonomous_service._profile_data(dataset, config)
    recommendations = await autonomous_service._recommend_algorithms(profile, config)

    # Display profile
    print(f"Dataset: {dataset.name}")
    print(f"Samples: {profile.n_samples:,}")
    print(f"Features: {profile.n_features}")
    print(f"Numeric Features: {profile.numeric_features}")
    print(f"Categorical Features: {profile.categorical_features}")
    print(f"Temporal Features: {profile.temporal_features}")
    print(f"Missing Values: {profile.missing_values_ratio:.1%}")
    print(f"Correlation Score: {profile.correlation_score:.3f}")
    print(f"Sparsity Ratio: {profile.sparsity_ratio:.1%}")
    print(f"Complexity Score: {profile.complexity_score:.3f}")
    print(f"Recommended Contamination: {profile.recommended_contamination:.1%}")

    print("\n🧠 Algorithm Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.algorithm} (confidence: {rec.confidence:.1%})")
        print(f"   Reasoning: {rec.reasoning}")
        print(f"   Expected Performance: {rec.expected_performance:.1%}")
        print()


def main():
    """Main demo function."""

    print("Pynomaly Autonomous Detection Demo")
    print("This demo showcases the autonomous anomaly detection capabilities.")
    print("\nThe system will:")
    print("1. Auto-detect data formats")
    print("2. Profile datasets to understand characteristics")
    print("3. Recommend optimal algorithms")
    print("4. Auto-tune hyperparameters")
    print("5. Run detection and provide insights")
    print("\n" + "=" * 60)

    # Run demos
    asyncio.run(run_autonomous_detection_demo())
    asyncio.run(demonstrate_data_profiling())

    print("\n" + "=" * 60)
    print("Demo completed! Try the CLI commands:")
    print("\n# Quick detection on your data:")
    print("pynomaly auto quick your_data.csv")
    print("\n# Full autonomous detection:")
    print("pynomaly auto detect your_data.csv --output results.csv")
    print("\n# Profile your data:")
    print("pynomaly auto profile your_data.csv --verbose")
    print("\n# Comprehensive detection with tuning:")
    print("pynomaly auto detect your_data.csv --auto-tune --max-algorithms 5")


if __name__ == "__main__":
    main()

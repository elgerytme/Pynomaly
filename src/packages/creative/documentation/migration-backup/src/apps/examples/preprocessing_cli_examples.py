#!/usr/bin/env python3
"""
Comprehensive examples demonstrating Pynomaly's data preprocessing CLI capabilities.

This script showcases the new preprocessing commands that bridge the gap between
raw data and anomaly detection, providing production-ready data cleaning,
transformation, and pipeline management through the command line.
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def run_command(cmd: str, description: str = ""):
    """Run a CLI command and display results."""
    print(f"\n{'=' * 60}")
    print(f"📋 {description}")
    print(f"🔧 Command: {cmd}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️  Stderr: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def create_sample_datasets():
    """Create sample datasets with various data quality issues for demonstration."""
    print("🔄 Creating sample datasets with data quality issues...")

    # Create output directory
    output_dir = Path("sample_data")
    output_dir.mkdir(exist_ok=True)

    # Dataset 1: E-commerce transactions with quality issues
    np.random.seed(42)
    n_samples = 1000

    ecommerce_data = {
        "customer_id": np.random.randint(1000, 9999, n_samples),
        "transaction_amount": np.concatenate(
            [
                np.random.lognormal(3, 1, n_samples - 50),  # Normal transactions
                np.random.uniform(10000, 50000, 50),  # Outlier transactions
            ]
        ),
        "product_category": np.random.choice(
            ["Electronics", "Clothing", "Books", "Home", "Sports"], n_samples
        ),
        "payment_method": np.random.choice(
            ["Credit", "Debit", "PayPal", "Cash"], n_samples
        ),
        "customer_rating": np.random.choice(
            [1, 2, 3, 4, 5, np.nan], n_samples, p=[0.05, 0.1, 0.15, 0.3, 0.35, 0.05]
        ),
        "shipping_cost": np.random.exponential(10, n_samples),
        "processing_time_hours": np.random.gamma(2, 3, n_samples),
        "returns": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        "constant_field": np.ones(n_samples),  # Constant feature
        "mostly_zeros": np.random.choice(
            [0, 1, 2], n_samples, p=[0.95, 0.03, 0.02]
        ),  # Low variance
    }

    # Add missing values
    missing_indices = np.random.choice(
        n_samples, size=int(n_samples * 0.1), replace=False
    )
    ecommerce_data["shipping_cost"][missing_indices] = np.nan

    # Add infinite values
    ecommerce_data["processing_time_hours"][0] = np.inf
    ecommerce_data["processing_time_hours"][1] = -np.inf

    # Create DataFrame and add duplicates
    df_ecommerce = pd.DataFrame(ecommerce_data)
    df_ecommerce = pd.concat([df_ecommerce, df_ecommerce.iloc[:20]], ignore_index=True)

    # Save dataset
    ecommerce_file = output_dir / "ecommerce_transactions.csv"
    df_ecommerce.to_csv(ecommerce_file, index=False)
    print(f"✅ Created e-commerce dataset: {ecommerce_file}")

    # Dataset 2: IoT sensor data with different issues
    iot_data = {
        "sensor_id": np.random.choice(
            ["TEMP_01", "TEMP_02", "HUMID_01", "HUMID_02", "PRESS_01"], n_samples
        ),
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1H"),
        "temperature": np.random.normal(20, 5, n_samples),
        "humidity": np.random.beta(2, 3, n_samples) * 100,
        "pressure": np.random.normal(1013, 10, n_samples),
        "battery_level": np.maximum(0, 100 - np.random.exponential(2, n_samples)),
        "signal_strength": np.random.uniform(-100, -30, n_samples),
        "status_code": np.random.choice(
            ["OK", "WARNING", "ERROR", "OFFLINE"], n_samples, p=[0.8, 0.1, 0.05, 0.05]
        ),
    }

    # Add seasonal patterns and anomalies
    iot_data["temperature"] += 10 * np.sin(
        np.arange(n_samples) * 2 * np.pi / 24
    )  # Daily cycle

    # Add missing sensor readings (realistic IoT issue)
    offline_periods = np.random.choice(n_samples, size=50, replace=False)
    for col in ["temperature", "humidity", "pressure"]:
        iot_data[col][offline_periods] = np.nan

    df_iot = pd.DataFrame(iot_data)
    iot_file = output_dir / "iot_sensor_data.csv"
    df_iot.to_csv(iot_file, index=False)
    print(f"✅ Created IoT sensor dataset: {iot_file}")

    return [ecommerce_file, iot_file]


def demonstrate_data_quality_analysis():
    """Demonstrate data quality analysis with preprocessing suggestions."""
    print("\n" + "=" * 80)
    print("🔍 DATA QUALITY ANALYSIS AND PREPROCESSING SUGGESTIONS")
    print("=" * 80)

    # Load sample dataset
    print("📤 Loading e-commerce dataset...")
    success = run_command(
        "pynomaly dataset load sample_data/ecommerce_transactions.csv --name ecommerce_demo",
        "Loading e-commerce dataset for quality analysis",
    )

    if success:
        # Get dataset ID (would need to be retrieved from actual output)
        print("📊 Analyzing data quality...")
        run_command(
            "pynomaly dataset list --limit 1", "Getting dataset ID for quality analysis"
        )

        # Note: In real usage, you'd get the actual dataset ID from the list command
        dataset_id = "ecommerce_demo"  # Placeholder

        run_command(
            f"pynomaly dataset quality {dataset_id}",
            "Comprehensive data quality analysis with preprocessing recommendations",
        )


def demonstrate_data_cleaning():
    """Demonstrate various data cleaning operations."""
    print("\n" + "=" * 80)
    print("🧹 DATA CLEANING OPERATIONS")
    print("=" * 80)

    dataset_id = "ecommerce_demo"  # Would be actual ID in practice

    # 1. Preview cleaning operations
    print("👀 Previewing cleaning operations (dry run)...")
    run_command(
        f"pynomaly data clean {dataset_id} --missing fill_median --outliers clip --duplicates --dry-run",
        "Preview comprehensive data cleaning (dry run mode)",
    )

    # 2. Apply specific cleaning operations
    print("🔧 Applying missing value handling...")
    run_command(
        f"pynomaly data clean {dataset_id} --missing fill_median --save-as ecommerce_cleaned_missing",
        "Handle missing values using median fill strategy",
    )

    print("🎯 Handling outliers...")
    run_command(
        "pynomaly data clean ecommerce_cleaned_missing --outliers clip --outlier-threshold 3.0 --save-as ecommerce_cleaned_outliers",
        "Remove outliers using 3-sigma clipping",
    )

    print("🔄 Removing duplicates...")
    run_command(
        "pynomaly data clean ecommerce_cleaned_outliers --duplicates --save-as ecommerce_cleaned_final",
        "Remove duplicate rows from dataset",
    )

    # 3. Comprehensive cleaning in one command
    print("⚡ Comprehensive cleaning in single command...")
    run_command(
        f"pynomaly data clean {dataset_id} --missing fill_median --outliers clip --duplicates --zeros remove --infinite remove --save-as ecommerce_comprehensive_clean",
        "Apply all cleaning operations in one command",
    )


def demonstrate_data_transformation():
    """Demonstrate feature transformation operations."""
    print("\n" + "=" * 80)
    print("🔄 DATA TRANSFORMATION OPERATIONS")
    print("=" * 80)

    dataset_id = "ecommerce_comprehensive_clean"

    # 1. Feature scaling
    print("📏 Applying feature scaling...")
    run_command(
        f"pynomaly data transform {dataset_id} --scaling standard --save-as ecommerce_scaled",
        "Apply standard scaling to numeric features",
    )

    # 2. Categorical encoding
    print("🏷️ Encoding categorical features...")
    run_command(
        "pynomaly data transform ecommerce_scaled --encoding onehot --save-as ecommerce_encoded",
        "Apply one-hot encoding to categorical features",
    )

    # 3. Feature selection
    print("🎯 Selecting relevant features...")
    run_command(
        "pynomaly data transform ecommerce_encoded --feature-selection variance_threshold --save-as ecommerce_selected",
        "Remove low-variance features",
    )

    # 4. Comprehensive transformation
    print("⚡ Comprehensive transformation pipeline...")
    run_command(
        f"pynomaly data transform {dataset_id} --scaling minmax --encoding label --normalize-names --optimize-dtypes --save-as ecommerce_full_transform",
        "Apply comprehensive feature transformation",
    )

    # 5. Advanced feature engineering
    print("🔬 Advanced feature engineering...")
    run_command(
        "pynomaly data transform ecommerce_scaled --polynomial 2 --save-as ecommerce_poly",
        "Generate polynomial interaction features",
    )


def demonstrate_pipeline_management():
    """Demonstrate preprocessing pipeline creation and management."""
    print("\n" + "=" * 80)
    print("🔧 PREPROCESSING PIPELINE MANAGEMENT")
    print("=" * 80)

    # 1. Create pipeline configuration file
    pipeline_config = {
        "name": "ecommerce_preprocessing_pipeline",
        "steps": [
            {
                "name": "handle_missing",
                "operation": "handle_missing_values",
                "parameters": {"strategy": "fill_median"},
                "enabled": True,
                "description": "Fill missing values with median",
            },
            {
                "name": "remove_outliers",
                "operation": "handle_outliers",
                "parameters": {"strategy": "clip", "threshold": 3.0},
                "enabled": True,
                "description": "Clip outliers beyond 3 standard deviations",
            },
            {
                "name": "remove_duplicates",
                "operation": "remove_duplicates",
                "parameters": {},
                "enabled": True,
                "description": "Remove duplicate rows",
            },
            {
                "name": "scale_features",
                "operation": "scale_features",
                "parameters": {"strategy": "standard"},
                "enabled": True,
                "description": "Apply standard scaling",
            },
            {
                "name": "encode_categorical",
                "operation": "encode_categorical",
                "parameters": {"strategy": "onehot"},
                "enabled": True,
                "description": "One-hot encode categorical features",
            },
        ],
    }

    import json

    config_file = Path("ecommerce_pipeline.json")
    with open(config_file, "w") as f:
        json.dump(pipeline_config, f, indent=2)

    print(f"📝 Created pipeline configuration: {config_file}")

    # 2. Create pipeline from config
    run_command(
        f"pynomaly data pipeline create --name ecommerce_pipeline --config {config_file}",
        "Create preprocessing pipeline from configuration file",
    )

    # 3. List available pipelines
    run_command(
        "pynomaly data pipeline list", "List all available preprocessing pipelines"
    )

    # 4. Show pipeline details
    run_command(
        "pynomaly data pipeline show --name ecommerce_pipeline",
        "Show detailed pipeline configuration",
    )

    # 5. Apply pipeline to dataset
    dataset_id = "ecommerce_demo"
    run_command(
        f"pynomaly data pipeline apply --name ecommerce_pipeline --dataset {dataset_id} --dry-run",
        "Preview pipeline application (dry run)",
    )

    # 6. Save pipeline for reuse
    run_command(
        "pynomaly data pipeline save --name ecommerce_pipeline --output saved_ecommerce_pipeline.json",
        "Save pipeline configuration for future use",
    )


def demonstrate_anomaly_detection_workflow():
    """Demonstrate complete workflow from raw data to anomaly detection."""
    print("\n" + "=" * 80)
    print("🚀 COMPLETE ANOMALY DETECTION WORKFLOW")
    print("=" * 80)

    # 1. Load and analyze data quality
    print("🔍 Step 1: Load and analyze data quality...")
    run_command(
        "pynomaly dataset load sample_data/iot_sensor_data.csv --name iot_sensors",
        "Load IoT sensor dataset",
    )

    # Note: Getting actual dataset ID would require parsing command output
    dataset_id = "iot_sensors"

    run_command(
        f"pynomaly dataset quality {dataset_id}",
        "Analyze data quality and get preprocessing recommendations",
    )

    # 2. Apply preprocessing based on recommendations
    print("🧹 Step 2: Apply preprocessing...")
    run_command(
        f"pynomaly data clean {dataset_id} --missing interpolate --infinite remove --save-as iot_cleaned",
        "Clean IoT sensor data",
    )

    run_command(
        "pynomaly data transform iot_cleaned --scaling robust --save-as iot_preprocessed",
        "Apply robust scaling for IoT sensor data",
    )

    # 3. Create detector and run anomaly detection
    print("🤖 Step 3: Create detector and detect anomalies...")
    run_command(
        "pynomaly detector create --name iot_detector --algorithm IsolationForest",
        "Create Isolation Forest detector for IoT data",
    )

    run_command(
        "pynomaly detect train --detector iot_detector --dataset iot_preprocessed",
        "Train detector on preprocessed IoT data",
    )

    run_command(
        "pynomaly detect run --detector iot_detector --dataset iot_preprocessed --output iot_anomalies.csv",
        "Detect anomalies in IoT sensor data",
    )

    # 4. Export results
    print("📊 Step 4: Export results...")
    run_command(
        "pynomaly export csv iot_anomalies.csv iot_anomaly_report.csv",
        "Export anomaly detection results",
    )


def demonstrate_autonomous_mode_with_preprocessing():
    """Demonstrate how autonomous mode benefits from preprocessing capabilities."""
    print("\n" + "=" * 80)
    print("🤖 AUTONOMOUS MODE WITH ENHANCED PREPROCESSING")
    print("=" * 80)

    print("🚀 Running autonomous anomaly detection...")
    print("   (Autonomous mode now includes automatic data quality assessment)")
    print("   (and intelligent preprocessing based on data characteristics)")

    run_command(
        "pynomaly auto detect sample_data/ecommerce_transactions.csv --output autonomous_results.csv --verbose",
        "Autonomous anomaly detection with automatic preprocessing",
    )

    print("📈 Autonomous mode advantages:")
    print("   ✅ Automatic data quality assessment")
    print("   ✅ Intelligent preprocessing strategy selection")
    print("   ✅ Algorithm recommendation based on preprocessed data characteristics")
    print("   ✅ End-to-end automation with quality assurance")


def show_cli_help():
    """Show comprehensive CLI help for preprocessing commands."""
    print("\n" + "=" * 80)
    print("📚 COMPREHENSIVE CLI HELP")
    print("=" * 80)

    run_command("pynomaly data --help", "Main preprocessing command help")
    run_command("pynomaly data clean --help", "Data cleaning command help")
    run_command("pynomaly data transform --help", "Data transformation command help")
    run_command("pynomaly data pipeline --help", "Pipeline management command help")


def cleanup():
    """Clean up generated files."""
    print("\n🧹 Cleaning up generated files...")
    files_to_remove = [
        "sample_data/",
        "ecommerce_pipeline.json",
        "saved_ecommerce_pipeline.json",
        "iot_anomalies.csv",
        "iot_anomaly_report.csv",
        "autonomous_results.csv",
    ]

    for file_path in files_to_remove:
        try:
            if Path(file_path).is_dir():
                import shutil

                shutil.rmtree(file_path)
            else:
                Path(file_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: Could not remove {file_path}: {e}")


def main():
    """Run comprehensive preprocessing CLI demonstration."""
    print("🎯 PYNOMALY PREPROCESSING CLI DEMONSTRATION")
    print("=" * 80)
    print("This script demonstrates the comprehensive data preprocessing")
    print("capabilities available through Pynomaly's command-line interface.")
    print("=" * 80)

    try:
        # Check if pynomaly CLI is available
        result = subprocess.run("pynomaly --version", shell=True, capture_output=True)
        if result.returncode != 0:
            print(
                "❌ Pynomaly CLI not available. Please install and set up the CLI first."
            )
            return

        # Create sample datasets
        create_sample_datasets()

        # Run demonstrations
        demonstrate_data_quality_analysis()
        demonstrate_data_cleaning()
        demonstrate_data_transformation()
        demonstrate_pipeline_management()
        demonstrate_anomaly_detection_workflow()
        demonstrate_autonomous_mode_with_preprocessing()
        show_cli_help()

        print("\n" + "=" * 80)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("Key takeaways:")
        print("✅ Comprehensive data preprocessing through CLI")
        print("✅ Quality analysis with actionable recommendations")
        print("✅ Flexible cleaning and transformation operations")
        print("✅ Reusable preprocessing pipelines")
        print("✅ Seamless integration with anomaly detection")
        print("✅ Enhanced autonomous mode capabilities")

    except KeyboardInterrupt:
        print("\n⚠️ Demonstration interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
    finally:
        # Clean up
        if input("\n🧹 Clean up generated files? (y/N): ").lower().startswith("y"):
            cleanup()


if __name__ == "__main__":
    main()

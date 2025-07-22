#!/usr/bin/env python3
"""Basic usage example for the Data Science package."""

import asyncio
from pathlib import Path

from data_science import (
    IntegratedDataScienceService,
    FeatureValidator,
    MetricsCalculator,
    Dataset,
    DataScienceSettings
)


async def main() -> None:
    """Run the basic usage example."""
    print("Data Science Package - Basic Usage Example")
    print("=" * 50)
    
    # Initialize settings
    settings = DataScienceSettings()
    
    # Initialize services
    data_science_service = IntegratedDataScienceService()
    feature_validator = FeatureValidator()
    metrics_calculator = MetricsCalculator()
    
    # Example 1: Create and manage an experiment
    print("\n1. Creating a new experiment...")
    experiment_config = {
        "model_type": "random_forest",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10
        },
        "validation": {
            "method": "cross_validation",
            "folds": 5
        }
    }
    
    # In a real implementation, this would create an actual experiment
    print(f"✓ Created experiment with config: {experiment_config}")
    
    # Example 2: Validate features
    print("\n2. Validating dataset features...")
    
    # Create a mock dataset for demonstration
    dataset = Dataset(
        id="dataset_001",
        name="sample_dataset",
        path="/path/to/sample_data.csv",
        features=["feature1", "feature2", "feature3"],
        metadata={"size": "1000 rows", "source": "synthetic"}
    )
    
    # In a real implementation, this would perform actual validation
    validation_config = {
        "check_null_values": True,
        "check_outliers": True,
        "check_data_types": True
    }
    
    print(f"✓ Validating dataset: {dataset.name}")
    print(f"  Features: {dataset.features}")
    print(f"  Validation config: {validation_config}")
    
    # Example 3: Calculate metrics
    print("\n3. Calculating experiment metrics...")
    
    # Mock experiment results
    experiment_results = {
        "predictions": [0.1, 0.8, 0.3, 0.9, 0.2],
        "actual": [0, 1, 0, 1, 0],
        "model_performance": {
            "training_time": "2.5 minutes",
            "prediction_time": "0.1 seconds"
        }
    }
    
    # In a real implementation, this would calculate actual metrics
    metrics_config = {
        "classification_metrics": ["accuracy", "precision", "recall", "f1"],
        "regression_metrics": ["mse", "mae", "r2"]
    }
    
    print(f"✓ Calculating metrics with config: {metrics_config}")
    print(f"  Sample results: {experiment_results}")
    
    # Example 4: Monitor performance
    print("\n4. Performance monitoring...")
    
    performance_data = {
        "memory_usage": "256 MB",
        "cpu_usage": "15%",
        "processing_time": "3.2 seconds"
    }
    
    print(f"✓ Performance monitoring data: {performance_data}")
    
    print("\n" + "=" * 50)
    print("Basic usage example completed successfully!")
    print("\nNext steps:")
    print("- Explore the CLI: data-science --help")
    print("- Start the API server: data-science-server")
    print("- Check out more examples in the examples/ directory")


if __name__ == "__main__":
    asyncio.run(main())
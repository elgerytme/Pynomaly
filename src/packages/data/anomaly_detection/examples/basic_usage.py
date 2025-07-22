#!/usr/bin/env python3
"""Basic usage example for the Anomaly Detection package."""

import asyncio
import numpy as np
from typing import List

# In a real implementation, these would be actual imports
# from anomaly_detection import DetectionService, EnsembleService, SklearnAdapter


async def main() -> None:
    """Run the basic usage example."""
    print("Anomaly Detection Package - Basic Usage Example")
    print("=" * 55)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    np.random.seed(42)
    
    # Normal data
    normal_data = np.random.normal(0, 1, (900, 5))
    
    # Anomalous data (outliers)
    anomalous_data = np.random.normal(4, 0.5, (100, 5))
    
    # Combine data
    data = np.vstack([normal_data, anomalous_data])
    
    print(f"✓ Generated {len(data)} samples with {data.shape[1]} features")
    print(f"  - Normal samples: {len(normal_data)}")
    print(f"  - Anomalous samples: {len(anomalous_data)}")
    
    # Example 2: Single algorithm detection
    print("\n2. Single Algorithm Detection...")
    
    # In real implementation:
    # sklearn_adapter = SklearnAdapter()
    # detection_service = DetectionService(adapter=sklearn_adapter)
    # results = detection_service.detect(data, algorithm="isolation_forest")
    
    # Mock results for demonstration
    mock_results = {
        "algorithm": "isolation_forest",
        "anomalies": [892, 895, 901, 905, 910, 915, 920],  # Mock anomaly indices
        "scores": [0.8, 0.7, 0.9, 0.6, 0.85, 0.75, 0.88],  # Mock anomaly scores
        "total_samples": len(data),
        "contamination": 0.1
    }
    
    print(f"✓ Detection completed using {mock_results['algorithm']}")
    print(f"  Anomalies detected: {len(mock_results['anomalies'])}")
    print(f"  Expected contamination: {mock_results['contamination']}")
    print(f"  Actual anomaly rate: {len(mock_results['anomalies'])/mock_results['total_samples']:.3f}")
    
    # Example 3: Ensemble detection
    print("\n3. Ensemble Detection...")
    
    # In real implementation:
    # ensemble_service = EnsembleService()
    # algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
    # ensemble_results = ensemble_service.detect(data, algorithms=algorithms, method="voting")
    
    ensemble_config = {
        "algorithms": ["isolation_forest", "one_class_svm", "local_outlier_factor"],
        "method": "voting",
        "weights": [0.4, 0.3, 0.3]
    }
    
    mock_ensemble_results = {
        "method": "voting",
        "algorithms": ensemble_config["algorithms"],
        "anomalies": [890, 893, 899, 902, 908, 912, 918, 922],
        "scores": [0.82, 0.76, 0.88, 0.65, 0.79, 0.83, 0.71, 0.86],
        "consensus_threshold": 0.5
    }
    
    print(f"✓ Ensemble detection using {len(ensemble_config['algorithms'])} algorithms")
    print(f"  Method: {mock_ensemble_results['method']}")
    print(f"  Algorithms: {', '.join(mock_ensemble_results['algorithms'])}")
    print(f"  Anomalies detected: {len(mock_ensemble_results['anomalies'])}")
    
    # Example 4: Streaming detection simulation
    print("\n4. Streaming Detection Simulation...")
    
    stream_config = {
        "window_size": 50,
        "threshold": 0.7,
        "algorithm": "isolation_forest"
    }
    
    print(f"✓ Simulating stream with window size: {stream_config['window_size']}")
    
    # Simulate processing data in windows
    anomalies_in_stream = []
    for i in range(0, len(data), stream_config["window_size"]):
        window = data[i:i+stream_config["window_size"]]
        
        # Mock anomaly detection on window
        # In real implementation: results = streaming_service.process_window(window)
        mock_window_anomalies = [idx for idx in range(len(window)) if idx % 15 == 0]
        
        if mock_window_anomalies:
            anomalies_in_stream.extend([i + idx for idx in mock_window_anomalies])
    
    print(f"  Windows processed: {(len(data) + stream_config['window_size'] - 1) // stream_config['window_size']}")
    print(f"  Stream anomalies detected: {len(anomalies_in_stream)}")
    
    # Example 5: Explainability
    print("\n5. Explainability Analysis...")
    
    # Mock explanation for top anomalies
    top_anomalies = mock_results["anomalies"][:3]
    
    for i, anomaly_idx in enumerate(top_anomalies):
        print(f"  Anomaly #{anomaly_idx} (score: {mock_results['scores'][i]:.2f}):")
        
        # Mock feature contributions
        feature_contributions = np.random.uniform(-0.5, 0.5, 5)
        feature_contributions[np.argmax(np.abs(feature_contributions))] = 0.8  # Make one dominant
        
        for j, contribution in enumerate(feature_contributions):
            print(f"    Feature {j}: {contribution:+.3f}")
    
    # Example 6: Performance monitoring
    print("\n6. Performance Monitoring...")
    
    performance_metrics = {
        "detection_time": "0.15 seconds",
        "memory_usage": "45 MB",
        "cpu_usage": "12%",
        "throughput": f"{len(data)/0.15:.0f} samples/second"
    }
    
    print("✓ Performance metrics:")
    for metric, value in performance_metrics.items():
        print(f"  {metric}: {value}")
    
    print("\n" + "=" * 55)
    print("Basic usage example completed successfully!")
    print("\nNext steps:")
    print("- Explore the CLI: anomaly-detection --help")
    print("- Start the API server: anomaly-detection-server")
    print("- Check out advanced examples for streaming and ensemble methods")
    print("- Review the documentation for algorithm parameters and tuning")


if __name__ == "__main__":
    asyncio.run(main())
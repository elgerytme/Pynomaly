#!/usr/bin/env python3
"""Comprehensive demonstration of PyOD adapter integration."""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anomaly_detection.infrastructure.adapters.comprehensive_pyod_adapter import (
    ComprehensivePyODAdapter, 
    PYOD_AVAILABLE,
    AlgorithmCategory
)
from anomaly_detection.infrastructure.adapters.pyod_integration import (
    create_pyod_integration,
    register_recommended_algorithms
)
from anomaly_detection.domain.services.detection_service import DetectionService


def generate_sample_data(n_samples: int = 300, n_features: int = 5, contamination: float = 0.1):
    """Generate sample data with injected anomalies."""
    np.random.seed(42)
    
    # Generate normal data
    normal_samples = int(n_samples * (1 - contamination))
    anomaly_samples = n_samples - normal_samples
    
    # Normal data from multivariate normal distribution
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=normal_samples
    )
    
    # Anomalous data with different characteristics
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,  # Shifted mean
        cov=np.eye(n_features) * 2,    # Different covariance
        size=anomaly_samples
    )
    
    # Combine data
    data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([
        np.ones(normal_samples),      # Normal = 1
        -np.ones(anomaly_samples)     # Anomaly = -1
    ])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    return data[indices], labels[indices]


def demonstrate_algorithm_discovery():
    """Demonstrate algorithm discovery and information retrieval."""
    print("=" * 60)
    print("ALGORITHM DISCOVERY AND INFORMATION")
    print("=" * 60)
    
    if not PYOD_AVAILABLE:
        print("⚠️  PyOD not fully available - showing limited functionality")
        print(f"Available algorithms: {ComprehensivePyODAdapter.list_available_algorithms()}")
        return
    
    # List all available algorithms
    print("📋 Available PyOD Algorithms:")
    algorithms = ComprehensivePyODAdapter.list_available_algorithms()
    
    for name, info in list(algorithms.items())[:10]:  # Show first 10
        print(f"  • {info['display_name']} ({name})")
        print(f"    Category: {info['category']}")
        print(f"    Complexity: {info['computational_complexity']}")
        print(f"    Memory: {info['memory_usage']}")
        print()
    
    print(f"Total algorithms available: {len(algorithms)}")
    print()
    
    # Show algorithms by category
    print("📊 Algorithms by Category:")
    for category in AlgorithmCategory:
        algos = ComprehensivePyODAdapter.get_algorithms_by_category(category)
        if algos:
            print(f"  {category.value}: {', '.join(algos[:5])}")  # Show first 5
    print()


def demonstrate_algorithm_recommendations():
    """Demonstrate algorithm recommendation system."""
    print("=" * 60)
    print("ALGORITHM RECOMMENDATIONS")
    print("=" * 60)
    
    if not PYOD_AVAILABLE:
        print("⚠️  PyOD not available - skipping recommendations")
        return
    
    # Test different scenarios
    scenarios = [
        {
            "name": "Small Dataset - High Interpretability",
            "data_size": "small",
            "complexity_preference": "low", 
            "interpretability_required": True
        },
        {
            "name": "Large Dataset - High Performance",
            "data_size": "large",
            "complexity_preference": "medium",
            "interpretability_required": False
        },
        {
            "name": "Medium Dataset - Balanced",
            "data_size": "medium",
            "complexity_preference": "medium",
            "interpretability_required": False
        }
    ]
    
    for scenario in scenarios:
        print(f"🎯 Scenario: {scenario['name']}")
        recommendations = ComprehensivePyODAdapter.get_recommended_algorithms(**scenario)
        print(f"   Recommended: {', '.join(recommendations[:5])}")  # Show first 5
        print()


def demonstrate_algorithm_suitability():
    """Demonstrate algorithm suitability evaluation."""
    print("=" * 60)
    print("ALGORITHM SUITABILITY EVALUATION")
    print("=" * 60)
    
    if not PYOD_AVAILABLE:
        print("⚠️  PyOD not available - skipping suitability evaluation")
        return
    
    # Test suitability for different data scenarios
    test_cases = [
        {"algorithm": "iforest", "data_shape": (1000, 10), "streaming": False},
        {"algorithm": "abod", "data_shape": (100000, 1000), "streaming": True},  # Should have warnings
        {"algorithm": "lof", "data_shape": (5000, 50), "streaming": False}
    ]
    
    for case in test_cases:
        try:
            adapter = ComprehensivePyODAdapter(algorithm=case["algorithm"])
            evaluation = adapter.evaluate_algorithm_suitability(
                data_shape=case["data_shape"],
                streaming=case["streaming"]
            )
            
            print(f"🔍 Algorithm: {case['algorithm']}")
            print(f"   Data Shape: {case['data_shape']}")
            print(f"   Suitability Score: {evaluation['suitability_score']}/100")
            
            if evaluation["warnings"]:
                print("   ⚠️  Warnings:")
                for warning in evaluation["warnings"]:
                    print(f"     - {warning}")
            
            if evaluation["recommendations"]:
                print("   💡 Recommendations:")
                for rec in evaluation["recommendations"]:
                    print(f"     - {rec}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to evaluate {case['algorithm']}: {e}")
            print()


def demonstrate_detection_with_pyod():
    """Demonstrate actual anomaly detection using PyOD algorithms."""
    print("=" * 60)
    print("ANOMALY DETECTION WITH PYOD")
    print("=" * 60)
    
    # Generate sample data
    print("📊 Generating sample data...")
    data, true_labels = generate_sample_data(n_samples=200, contamination=0.1)
    print(f"   Data shape: {data.shape}")
    print(f"   True anomalies: {np.sum(true_labels == -1)}")
    print()
    
    if not PYOD_AVAILABLE:
        print("⚠️  PyOD not available - using basic detection service")
        
        # Use basic detection service
        service = DetectionService()
        result = service.detect_anomalies(data, algorithm="iforest", contamination=0.1)
        
        print(f"🔍 Basic Detection Results:")
        print(f"   Algorithm: {result.algorithm}")
        print(f"   Predictions: {result.predictions[:10]}...")  # Show first 10
        print(f"   Detected anomalies: {result.anomaly_count}")
        print(f"   Anomaly rate: {result.anomaly_rate:.3f}")
        return
    
    # Test multiple PyOD algorithms
    algorithms_to_test = ["iforest", "lof", "copod", "ecod"]
    results = {}
    
    for algorithm in algorithms_to_test:
        try:
            print(f"🔍 Testing {algorithm}...")
            
            # Create and fit adapter
            adapter = ComprehensivePyODAdapter(algorithm=algorithm, contamination=0.1)
            adapter.fit(data)
            
            # Make predictions
            predictions = adapter.predict(data)
            scores = adapter.decision_function(data)
            probabilities = adapter.predict_proba(data)
            
            # Convert predictions to match true labels format (PyOD uses 0/1, we need 1/-1)
            converted_predictions = np.where(predictions == 0, 1, -1)
            
            # Calculate basic accuracy
            accuracy = np.mean(converted_predictions == true_labels)
            
            results[algorithm] = {
                "predictions": converted_predictions,
                "scores": scores,
                "probabilities": probabilities,
                "accuracy": accuracy,
                "detected_anomalies": np.sum(predictions == 1),
                "anomaly_rate": np.mean(predictions == 1)
            }
            
            print(f"   ✅ Detected anomalies: {results[algorithm]['detected_anomalies']}")
            print(f"   📊 Anomaly rate: {results[algorithm]['anomaly_rate']:.3f}")
            print(f"   🎯 Accuracy: {accuracy:.3f}")
            print()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print()
    
    # Summary comparison
    if results:
        print("📋 Algorithm Comparison Summary:")
        print(f"{'Algorithm':<12} {'Detected':<10} {'Rate':<8} {'Accuracy':<10}")
        print("-" * 42)
        
        for algo, result in results.items():
            print(f"{algo:<12} {result['detected_anomalies']:<10} "
                  f"{result['anomaly_rate']:<8.3f} {result['accuracy']:<10.3f}")
        print()


def demonstrate_integration_with_detection_service():
    """Demonstrate integration with the main detection service."""
    print("=" * 60)
    print("INTEGRATION WITH DETECTION SERVICE")
    print("=" * 60)
    
    # Create detection service
    detection_service = DetectionService()
    
    # Create PyOD integration
    integration = create_pyod_integration(detection_service)
    
    print(f"🔧 PyOD Integration Status:")
    print(f"   PyOD Available: {PYOD_AVAILABLE}")
    print(f"   Registered Algorithms: {len(integration.get_registered_algorithms())}")
    print()
    
    if not PYOD_AVAILABLE:
        print("⚠️  Limited functionality due to PyOD not being fully available")
        return
    
    # Generate sample data for recommendations
    sample_data, _ = generate_sample_data(n_samples=1000, n_features=8)
    
    # Get and register recommended algorithms
    print("🎯 Getting algorithm recommendations...")
    try:
        registered = register_recommended_algorithms(
            integration=integration,
            data_shape=sample_data.shape,
            max_algorithms=3,
            performance_preference="balanced",
            interpretability_required=False
        )
        
        print(f"   Registered algorithms: {', '.join(registered)}")
        print()
        
        # Test detection with registered algorithms
        print("🔍 Testing detection with registered algorithms...")
        
        for algo_name in registered[:2]:  # Test first 2
            try:
                result = detection_service.detect_anomalies(
                    sample_data, 
                    algorithm=algo_name, 
                    contamination=0.1
                )
                
                print(f"   {algo_name}:")
                print(f"     Detected anomalies: {result.anomaly_count}")
                print(f"     Anomaly rate: {result.anomaly_rate:.3f}")
                print()
                
            except Exception as e:
                print(f"   ❌ {algo_name} failed: {e}")
                print()
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print()


def demonstrate_model_persistence():
    """Demonstrate model saving and loading."""
    print("=" * 60)
    print("MODEL PERSISTENCE")
    print("=" * 60)
    
    if not PYOD_AVAILABLE:
        print("⚠️  PyOD not available - skipping persistence demo")
        return
    
    # Generate sample data
    data, _ = generate_sample_data(n_samples=100, n_features=3)
    
    try:
        # Create and fit model
        print("💾 Training and saving model...")
        adapter = ComprehensivePyODAdapter(algorithm="iforest", n_estimators=50)
        adapter.fit(data)
        
        # Get initial predictions
        initial_predictions = adapter.predict(data[:10])
        
        # Save model
        model_path = "/tmp/pyod_model_demo.pkl"
        adapter.save_model(model_path)
        print(f"   ✅ Model saved to {model_path}")
        
        # Load model
        print("📂 Loading model...")
        loaded_adapter = ComprehensivePyODAdapter.load_model(model_path)
        
        # Test loaded model
        loaded_predictions = loaded_adapter.predict(data[:10])
        
        # Verify consistency
        predictions_match = np.array_equal(initial_predictions, loaded_predictions)
        print(f"   ✅ Model loaded successfully")
        print(f"   🔍 Predictions consistent: {predictions_match}")
        
        if predictions_match:
            print("   💚 Persistence test PASSED")
        else:
            print("   ❌ Persistence test FAILED - predictions don't match")
        
        print()
        
        # Clean up
        os.remove(model_path)
        print("🧹 Cleaned up temporary files")
        
    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
    
    print()


def main():
    """Main demonstration function."""
    print("🚀 Comprehensive PyOD Adapter Demonstration")
    print(f"PyOD Available: {PYOD_AVAILABLE}")
    print()
    
    # Run all demonstrations
    demonstrate_algorithm_discovery()
    demonstrate_algorithm_recommendations()
    demonstrate_algorithm_suitability()
    demonstrate_detection_with_pyod()
    demonstrate_integration_with_detection_service()
    demonstrate_model_persistence()
    
    print("=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("• Comprehensive algorithm registry (40+ algorithms)")
    print("• Intelligent algorithm recommendations")
    print("• Algorithm suitability evaluation")
    print("• Seamless integration with detection services")
    print("• Model persistence and loading")
    print("• Comprehensive error handling and logging")
    print()
    
    if not PYOD_AVAILABLE:
        print("💡 To unlock full functionality:")
        print("   pip install pyod combo")
        print()


if __name__ == "__main__":
    main()
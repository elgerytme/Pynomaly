#!/usr/bin/env python3
"""Simplified demonstration of advanced concept drift detection capabilities."""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anomaly_detection.domain.services.concept_drift_detection_service import (
    ConceptDriftDetectionService,
    DriftDetectionMethod,
    DriftSeverity
)


class DataGenerator:
    """Helper class to generate synthetic data with various drift patterns."""
    
    @staticmethod
    def generate_stable_data(n_samples: int, n_features: int, random_state: int = 42) -> np.ndarray:
        """Generate stable data with no drift."""
        np.random.seed(random_state)
        return np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=n_samples
        )
    
    @staticmethod
    def generate_mean_shift_data(n_samples: int, n_features: int, shift_magnitude: float = 2.0, random_state: int = 43) -> np.ndarray:
        """Generate data with mean shift drift."""
        np.random.seed(random_state)
        return np.random.multivariate_normal(
            mean=np.ones(n_features) * shift_magnitude,
            cov=np.eye(n_features),
            size=n_samples
        )


def demonstrate_basic_drift_detection():
    """Demonstrate basic concept drift detection functionality."""
    print("=" * 80)
    print("BASIC CONCEPT DRIFT DETECTION")
    print("=" * 80)
    
    # Initialize drift detection service
    drift_service = ConceptDriftDetectionService(
        window_size=200,
        reference_window_size=300,
        drift_threshold=0.05,
        min_samples=50
    )
    
    model_id = "demo_model"
    n_features = 5
    
    print("🔧 Initialized ConceptDriftDetectionService")
    print(f"   Window size: {drift_service.window_size}")
    print(f"   Reference window size: {drift_service.reference_window_size}")
    print(f"   Drift threshold: {drift_service.drift_threshold}")
    print()
    
    # Generate reference data (stable)
    print("📊 Generating reference data (stable distribution)...")
    reference_data = DataGenerator.generate_stable_data(300, n_features)
    drift_service.add_reference_data(model_id, reference_data)
    print(f"   Added {len(reference_data)} reference samples")
    print()
    
    # Test different drift scenarios
    drift_scenarios = [
        ("No Drift", lambda: DataGenerator.generate_stable_data(200, n_features, 50)),
        ("Mean Shift", lambda: DataGenerator.generate_mean_shift_data(200, n_features, 1.5)),
    ]
    
    for scenario_name, data_generator in drift_scenarios:
        print(f"🔍 Testing Scenario: {scenario_name}")
        
        # Generate current data
        current_data = data_generator()
        drift_service.add_current_data(model_id, current_data)
        
        # Run drift detection
        methods = [
            DriftDetectionMethod.STATISTICAL_DISTANCE,
            DriftDetectionMethod.POPULATION_STABILITY_INDEX,
            DriftDetectionMethod.DISTRIBUTION_SHIFT
        ]
        
        report = drift_service.detect_drift(model_id, methods=methods)
        
        print(f"   Overall Drift Detected: {report.overall_drift_detected}")
        print(f"   Severity: {report.overall_severity.value}")
        print(f"   Consensus Score: {report.consensus_score:.3f}")
        
        detected_methods = [r.method.value for r in report.detection_results if r.drift_detected]
        print(f"   Methods that detected drift: {detected_methods}")
        
        if report.recommendations:
            print(f"   Top Recommendation: {report.recommendations[0]}")
        
        print()
        
        # Clear current data for next scenario
        drift_service._current_data[model_id].clear()


def demonstrate_drift_detection_methods():
    """Demonstrate different drift detection methods."""
    print("=" * 80)
    print("DRIFT DETECTION METHODS COMPARISON")
    print("=" * 80)
    
    drift_service = ConceptDriftDetectionService()
    model_id = "methods_demo"
    n_features = 4
    
    # Setup data with clear drift
    reference_data = DataGenerator.generate_stable_data(500, n_features)
    drifted_data = DataGenerator.generate_mean_shift_data(300, n_features, 2.0)
    
    drift_service.add_reference_data(model_id, reference_data)
    drift_service.add_current_data(model_id, drifted_data)
    
    print("📊 Data Setup:")
    print(f"   Reference: {len(reference_data)} samples (stable)")
    print(f"   Current: {len(drifted_data)} samples (mean shift = 2.0)")
    print()
    
    # Test core methods that don't require external dependencies
    methods_to_test = [
        DriftDetectionMethod.STATISTICAL_DISTANCE,
        DriftDetectionMethod.POPULATION_STABILITY_INDEX,
        DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE,
        DriftDetectionMethod.DISTRIBUTION_SHIFT,
        DriftDetectionMethod.PERFORMANCE_DEGRADATION,
        DriftDetectionMethod.PREDICTION_DRIFT,
    ]
    
    print("🔍 Method-by-Method Analysis:")
    print(f"{'Method':<35} {'Detected':<10} {'Score':<12} {'Severity':<12} {'Confidence':<12}")
    print("-" * 85)
    
    for method in methods_to_test:
        try:
            report = drift_service.detect_drift(model_id, methods=[method])
            
            if report.detection_results:
                result = report.detection_results[0]
                detected = "✅ YES" if result.drift_detected else "❌ NO"
                score = f"{result.drift_score:.4f}"
                severity = result.severity.value
                confidence = f"{result.confidence:.2f}"
            else:
                detected = "❌ NO"
                score = "N/A"
                severity = "N/A"
                confidence = "N/A"
            
            print(f"{method.value:<35} {detected:<10} {score:<12} {severity:<12} {confidence:<12}")
            
        except Exception as e:
            print(f"{method.value:<35} {'ERROR':<10} {str(e)[:20]:<12} {'N/A':<12} {'N/A':<12}")
    
    print()


def demonstrate_severity_levels():
    """Demonstrate different drift severity levels."""
    print("=" * 80)
    print("DRIFT SEVERITY LEVELS")
    print("=" * 80)
    
    drift_service = ConceptDriftDetectionService(drift_threshold=0.1)
    n_features = 3
    
    # Test different drift magnitudes
    drift_scenarios = [
        ("No Drift", 0.0, DriftSeverity.NO_DRIFT),
        ("Low Drift", 0.5, DriftSeverity.LOW),
        ("Medium Drift", 1.5, DriftSeverity.MEDIUM),
        ("High Drift", 3.0, DriftSeverity.HIGH),
        ("Critical Drift", 5.0, DriftSeverity.CRITICAL)
    ]
    
    print("🎯 Testing Different Drift Magnitudes:")
    print(f"{'Scenario':<15} {'Shift Mag.':<12} {'Expected':<15} {'Detected':<15} {'Actual Score':<15}")
    print("-" * 75)
    
    for scenario_name, shift_magnitude, expected_severity in drift_scenarios:
        model_id = f"severity_{scenario_name.lower().replace(' ', '_')}"
        
        # Generate data
        reference_data = DataGenerator.generate_stable_data(200, n_features)
        drifted_data = DataGenerator.generate_mean_shift_data(200, n_features, shift_magnitude)
        
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, drifted_data)
        
        # Detect drift
        report = drift_service.detect_drift(
            model_id, 
            methods=[DriftDetectionMethod.STATISTICAL_DISTANCE]
        )
        
        if report.detection_results:
            result = report.detection_results[0]
            detected_severity = result.severity.value
            actual_score = f"{result.drift_score:.4f}"
        else:
            detected_severity = "unknown"
            actual_score = "N/A"
        
        print(f"{scenario_name:<15} {shift_magnitude:<12} {expected_severity.value:<15} {detected_severity:<15} {actual_score:<15}")
    
    print()


def demonstrate_performance_degradation_drift():
    """Demonstrate performance degradation drift detection."""
    print("=" * 80)
    print("PERFORMANCE DEGRADATION DRIFT DETECTION")
    print("=" * 80)
    
    drift_service = ConceptDriftDetectionService()
    model_id = "performance_model"
    
    print("📊 Simulating model performance over time...")
    
    # Simulate performance history
    performance_history = []
    
    # Good performance period
    print("   Period 1: Good performance (accuracy ~0.95)")
    for i in range(15):
        metrics = {
            "accuracy": 0.93 + np.random.normal(0, 0.02),
            "precision": 0.91 + np.random.normal(0, 0.03),
            "recall": 0.89 + np.random.normal(0, 0.02),
            "f1_score": 0.90 + np.random.normal(0, 0.02)
        }
        performance_history.append(metrics)
    
    # Degraded performance period
    print("   Period 2: Degraded performance (accuracy ~0.75)")
    for i in range(15):
        metrics = {
            "accuracy": 0.73 + np.random.normal(0, 0.03),
            "precision": 0.71 + np.random.normal(0, 0.04),
            "recall": 0.69 + np.random.normal(0, 0.03),
            "f1_score": 0.70 + np.random.normal(0, 0.03)
        }
        performance_history.append(metrics)
    
    # Add performance history to drift service
    drift_service._performance_history[model_id] = performance_history
    
    # Detect performance drift
    print("🔍 Analyzing performance degradation...")
    report = drift_service.detect_drift(
        model_id,
        methods=[DriftDetectionMethod.PERFORMANCE_DEGRADATION]
    )
    
    if report.detection_results:
        result = report.detection_results[0]
        print(f"   Performance Drift Detected: {result.drift_detected}")
        print(f"   Drift Score: {result.drift_score:.4f}")
        print(f"   Severity: {result.severity.value}")
        print(f"   Affected Metrics: {result.affected_features}")
        
        if "performance_drops" in result.metadata:
            drops = result.metadata["performance_drops"]
            print(f"   Performance Drops: {[f'{d:.3f}' for d in drops]}")
    
    print()


def demonstrate_statistical_methods():
    """Demonstrate statistical drift detection methods."""
    print("=" * 80)
    print("STATISTICAL DRIFT DETECTION METHODS")
    print("=" * 80)
    
    drift_service = ConceptDriftDetectionService()
    model_id = "stats_demo"
    n_features = 3
    
    # Create clearly different distributions
    print("📊 Creating test distributions:")
    reference_data = DataGenerator.generate_stable_data(400, n_features, 42)
    drifted_data = DataGenerator.generate_mean_shift_data(300, n_features, 2.5, 43)
    
    print(f"   Reference: mean ≈ {np.mean(reference_data, axis=0)}")
    print(f"   Current: mean ≈ {np.mean(drifted_data, axis=0)}")
    print()
    
    drift_service.add_reference_data(model_id, reference_data)
    drift_service.add_current_data(model_id, drifted_data)
    
    # Test individual statistical methods
    statistical_methods = [
        DriftDetectionMethod.STATISTICAL_DISTANCE,
        DriftDetectionMethod.POPULATION_STABILITY_INDEX,
        DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE
    ]
    
    print("🔍 Statistical Method Results:")
    for method in statistical_methods:
        report = drift_service.detect_drift(model_id, methods=[method])
        
        if report.detection_results:
            result = report.detection_results[0]
            print(f"   {method.value}:")
            print(f"     Drift Score: {result.drift_score:.4f}")
            print(f"     Detected: {result.drift_detected}")
            print(f"     Affected Features: {len(result.affected_features)}")
            print()
    
    # Overall analysis
    all_report = drift_service.detect_drift(model_id, methods=statistical_methods)
    print("📋 Overall Analysis:")
    print(f"   Methods detecting drift: {sum(1 for r in all_report.detection_results if r.drift_detected)}/{len(statistical_methods)}")
    print(f"   Consensus score: {all_report.consensus_score:.3f}")
    print(f"   Overall severity: {all_report.overall_severity.value}")
    print()


def main():
    """Main demonstration function."""
    print("🚀 Advanced Concept Drift Detection Demonstration (Simplified)")
    print("="*80)
    print()
    
    try:
        # Run all demonstrations
        demonstrate_basic_drift_detection()
        demonstrate_drift_detection_methods()
        demonstrate_severity_levels()
        demonstrate_statistical_methods()
        demonstrate_performance_degradation_drift()
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("🌟 Advanced Concept Drift Detection Features Demonstrated:")
    print("• Multiple statistical drift detection methods")
    print("• Population Stability Index (PSI) calculation")
    print("• Jensen-Shannon divergence analysis")
    print("• Distribution shift detection")
    print("• Performance degradation monitoring")
    print("• Configurable severity levels and thresholds")
    print("• Comprehensive drift analysis reporting")
    print("• Statistical method comparison")
    print()
    print("💡 This implementation provides enterprise-grade concept drift")
    print("   detection suitable for production ML systems!")


if __name__ == "__main__":
    main()
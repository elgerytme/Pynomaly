#!/usr/bin/env python3
"""Comprehensive demonstration of advanced concept drift detection capabilities."""

import sys
import os
import numpy as np
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anomaly_detection.domain.services.concept_drift_detection_service import (
    ConceptDriftDetectionService,
    DriftDetectionMethod,
    DriftSeverity
)
from anomaly_detection.infrastructure.monitoring.drift_monitoring_integration import (
    DriftMonitoringIntegration,
    DriftMonitoringConfig,
    start_drift_monitoring
)
from anomaly_detection.domain.entities.detection_result import DetectionResult


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
    
    @staticmethod
    def generate_variance_shift_data(n_samples: int, n_features: int, variance_multiplier: float = 2.0, random_state: int = 44) -> np.ndarray:
        """Generate data with variance shift drift."""
        np.random.seed(random_state)
        return np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features) * variance_multiplier,
            size=n_samples
        )
    
    @staticmethod
    def generate_gradual_drift_data(n_samples: int, n_features: int, drift_rate: float = 0.01, random_state: int = 45) -> np.ndarray:
        """Generate data with gradual drift over time."""
        np.random.seed(random_state)
        data = []
        
        for i in range(n_samples):
            # Gradually shifting mean
            drift_magnitude = i * drift_rate
            mean = np.ones(n_features) * drift_magnitude
            sample = np.random.multivariate_normal(mean, np.eye(n_features))
            data.append(sample)
        
        return np.array(data)
    
    @staticmethod
    def generate_sudden_drift_data(n_samples: int, n_features: int, drift_point: float = 0.5, random_state: int = 46) -> np.ndarray:
        """Generate data with sudden drift at a specific point."""
        np.random.seed(random_state)
        split_point = int(n_samples * drift_point)
        
        # First part: stable data
        stable_part = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=split_point
        )
        
        # Second part: drifted data
        drifted_part = np.random.multivariate_normal(
            mean=np.ones(n_features) * 3,
            cov=np.eye(n_features) * 1.5,
            size=n_samples - split_point
        )
        
        return np.vstack([stable_part, drifted_part])
    
    @staticmethod
    def generate_cyclical_drift_data(n_samples: int, n_features: int, cycle_length: int = 100, random_state: int = 47) -> np.ndarray:
        """Generate data with cyclical drift patterns."""
        np.random.seed(random_state)
        data = []
        
        for i in range(n_samples):
            # Cyclical drift using sine wave
            cycle_position = (i % cycle_length) / cycle_length
            drift_magnitude = np.sin(2 * np.pi * cycle_position) * 2
            mean = np.ones(n_features) * drift_magnitude
            sample = np.random.multivariate_normal(mean, np.eye(n_features))
            data.append(sample)
        
        return np.array(data)


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
        ("Variance Shift", lambda: DataGenerator.generate_variance_shift_data(200, n_features, 2.5)),
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
    
    # Test each method individually
    all_methods = [
        DriftDetectionMethod.STATISTICAL_DISTANCE,
        DriftDetectionMethod.POPULATION_STABILITY_INDEX,
        DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE,
        DriftDetectionMethod.DISTRIBUTION_SHIFT,
        DriftDetectionMethod.PERFORMANCE_DEGRADATION,
        DriftDetectionMethod.PREDICTION_DRIFT,
        DriftDetectionMethod.FEATURE_IMPORTANCE_DRIFT
    ]
    
    print("🔍 Method-by-Method Analysis:")
    print(f"{'Method':<35} {'Detected':<10} {'Score':<12} {'Severity':<12} {'Confidence':<12}")
    print("-" * 85)
    
    for method in all_methods:
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


async def demonstrate_monitoring_integration():
    """Demonstrate automated drift monitoring integration."""
    print("=" * 80)
    print("AUTOMATED DRIFT MONITORING INTEGRATION")
    print("=" * 80)
    
    # Configure monitoring
    config = DriftMonitoringConfig(
        enabled=True,
        check_interval_minutes=1,  # Short interval for demo
        auto_analysis_enabled=True,
        alert_on_drift=True,
        min_samples_before_analysis=50,
        methods=[
            DriftDetectionMethod.STATISTICAL_DISTANCE,
            DriftDetectionMethod.DISTRIBUTION_SHIFT
        ]
    )
    
    print("🔧 Configuring drift monitoring:")
    print(f"   Check interval: {config.check_interval_minutes} minutes")
    print(f"   Auto analysis: {config.auto_analysis_enabled}")
    print(f"   Alert on drift: {config.alert_on_drift}")
    print(f"   Min samples: {config.min_samples_before_analysis}")
    print(f"   Methods: {[m.value for m in config.methods]}")
    print()
    
    # Initialize monitoring
    monitoring = DriftMonitoringIntegration(config)
    
    # Add drift callback
    drift_alerts = []
    
    def drift_callback(model_id: str, report):
        drift_alerts.append((model_id, report.overall_drift_detected, report.overall_severity))
        print(f"🚨 Drift Callback Triggered:")
        print(f"   Model: {model_id}")
        print(f"   Drift Detected: {report.overall_drift_detected}")
        print(f"   Severity: {report.overall_severity.value}")
        print()
    
    monitoring.add_drift_callback(drift_callback)
    
    # Start monitoring (but don't wait for full loop)
    print("🚀 Starting drift monitoring...")
    await monitoring.start_monitoring()
    
    # Simulate data collection and drift
    model_id = "monitored_model"
    n_features = 4
    
    print("📊 Simulating data collection...")
    
    # Phase 1: Stable reference data
    print("   Phase 1: Adding stable reference data...")
    for batch in range(3):
        stable_data = DataGenerator.generate_stable_data(20, n_features, 100 + batch)
        predictions = DetectionResult(
            predictions=np.random.choice([-1, 1], size=20, p=[0.1, 0.9]),
            algorithm="test_model"
        )
        
        monitoring.record_prediction_data(
            model_id=model_id,
            input_data=stable_data,
            prediction_result=predictions,
            is_reference=True
        )
    
    # Phase 2: Current stable data
    print("   Phase 2: Adding current stable data...")
    for batch in range(2):
        stable_data = DataGenerator.generate_stable_data(15, n_features, 200 + batch)
        predictions = DetectionResult(
            predictions=np.random.choice([-1, 1], size=15, p=[0.1, 0.9]),
            algorithm="test_model"
        )
        
        monitoring.record_prediction_data(
            model_id=model_id,
            input_data=stable_data,
            prediction_result=predictions,
            is_reference=False
        )
    
    # Manual analysis (stable data)
    print("🔍 Analyzing drift (stable data)...")
    report1 = await monitoring.analyze_drift(model_id, force_analysis=True)
    if report1:
        print(f"   Drift detected: {report1.overall_drift_detected}")
        print(f"   Severity: {report1.overall_severity.value}")
    
    # Phase 3: Drifted data
    print("   Phase 3: Adding drifted data...")
    for batch in range(3):
        drifted_data = DataGenerator.generate_mean_shift_data(15, n_features, 2.0, 300 + batch)
        predictions = DetectionResult(
            predictions=np.random.choice([-1, 1], size=15, p=[0.3, 0.7]),  # Different prediction pattern
            algorithm="test_model"
        )
        
        monitoring.record_prediction_data(
            model_id=model_id,
            input_data=drifted_data,
            prediction_result=predictions,
            is_reference=False
        )
    
    # Manual analysis (drifted data)
    print("🔍 Analyzing drift (after adding drifted data)...")
    report2 = await monitoring.analyze_drift(model_id, force_analysis=True)
    if report2:
        print(f"   Drift detected: {report2.overall_drift_detected}")
        print(f"   Severity: {report2.overall_severity.value}")
        print(f"   Consensus score: {report2.consensus_score:.3f}")
        if report2.recommendations:
            print(f"   Recommendation: {report2.recommendations[0]}")
    
    # Get monitoring status
    print("📈 Monitoring Status:")
    status = monitoring.get_model_drift_status(model_id)
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print()
    
    # Stop monitoring
    await monitoring.stop_monitoring()
    print("✅ Stopped drift monitoring")
    
    # Summary
    print("📋 Summary:")
    print(f"   Drift alerts triggered: {len(drift_alerts)}")
    for model_id, detected, severity in drift_alerts:
        print(f"     - {model_id}: {detected} ({severity.value})")
    
    print()


def demonstrate_gradual_vs_sudden_drift():
    """Demonstrate detection of gradual vs sudden drift patterns."""
    print("=" * 80)
    print("GRADUAL VS SUDDEN DRIFT DETECTION")
    print("=" * 80)
    
    drift_service = ConceptDriftDetectionService()
    n_features = 3
    n_samples = 400
    
    # Scenario 1: Gradual drift
    print("🔍 Scenario 1: Gradual Drift Detection")
    model_id_gradual = "gradual_drift_model"
    
    reference_data = DataGenerator.generate_stable_data(300, n_features)
    gradual_data = DataGenerator.generate_gradual_drift_data(n_samples, n_features, drift_rate=0.005)
    
    drift_service.add_reference_data(model_id_gradual, reference_data)
    drift_service.add_current_data(model_id_gradual, gradual_data)
    
    gradual_report = drift_service.detect_drift(
        model_id_gradual,
        methods=[DriftDetectionMethod.STATISTICAL_DISTANCE, DriftDetectionMethod.DISTRIBUTION_SHIFT]
    )
    
    print(f"   Drift Detected: {gradual_report.overall_drift_detected}")
    print(f"   Severity: {gradual_report.overall_severity.value}")
    print(f"   Consensus Score: {gradual_report.consensus_score:.3f}")
    print()
    
    # Scenario 2: Sudden drift
    print("🔍 Scenario 2: Sudden Drift Detection")
    model_id_sudden = "sudden_drift_model"
    
    sudden_data = DataGenerator.generate_sudden_drift_data(n_samples, n_features, drift_point=0.7)
    
    drift_service.add_reference_data(model_id_sudden, reference_data)
    drift_service.add_current_data(model_id_sudden, sudden_data)
    
    sudden_report = drift_service.detect_drift(
        model_id_sudden,
        methods=[DriftDetectionMethod.STATISTICAL_DISTANCE, DriftDetectionMethod.DISTRIBUTION_SHIFT]
    )
    
    print(f"   Drift Detected: {sudden_report.overall_drift_detected}")
    print(f"   Severity: {sudden_report.overall_severity.value}")
    print(f"   Consensus Score: {sudden_report.consensus_score:.3f}")
    print()
    
    # Comparison
    print("📊 Comparison:")
    print(f"   Gradual Drift - Severity: {gradual_report.overall_severity.value}, Score: {gradual_report.consensus_score:.3f}")
    print(f"   Sudden Drift - Severity: {sudden_report.overall_severity.value}, Score: {sudden_report.consensus_score:.3f}")
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
    for metrics in performance_history:
        drift_service._performance_history[model_id] = drift_service._performance_history.get(model_id, [])
        drift_service._performance_history[model_id].append(metrics)
    
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


async def main():
    """Main demonstration function."""
    print("🚀 Advanced Concept Drift Detection Demonstration")
    print("="*80)
    print()
    
    try:
        # Run all demonstrations
        demonstrate_basic_drift_detection()
        await asyncio.sleep(0.1)  # Brief pause between demos
        
        demonstrate_drift_detection_methods()
        await asyncio.sleep(0.1)
        
        demonstrate_severity_levels()
        await asyncio.sleep(0.1)
        
        await demonstrate_monitoring_integration()
        await asyncio.sleep(0.1)
        
        demonstrate_gradual_vs_sudden_drift()
        await asyncio.sleep(0.1)
        
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
    print("• Automated monitoring and alerting")
    print("• Configurable severity levels and thresholds")
    print("• Gradual vs sudden drift pattern recognition")
    print("• Integration with monitoring infrastructure")
    print("• Real-time drift callbacks and notifications")
    print()
    print("💡 This implementation provides enterprise-grade concept drift")
    print("   detection suitable for production ML systems!")


if __name__ == "__main__":
    asyncio.run(main())
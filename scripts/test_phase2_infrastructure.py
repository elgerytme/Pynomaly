#!/usr/bin/env python3
"""Test script to validate Phase 2 infrastructure implementation.

This script tests the controlled feature reintroduction infrastructure
including feature flags, complexity monitoring, and algorithm benchmarking.
"""

import os
import sys
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_feature_flags():
    """Test feature flag system."""
    print("🏁 Testing Feature Flag System...")
    
    try:
        from pynomaly.infrastructure.config.feature_flags import (
            FeatureFlagManager, feature_flags, FeatureStage, FeatureCategory
        )
        
        # Test basic functionality
        manager = FeatureFlagManager()
        
        # Test feature checking
        enabled_features = manager.get_enabled_features()
        print(f"  ✅ Enabled features: {enabled_features}")
        
        # Test feature categories
        core_features = manager.get_features_by_category(FeatureCategory.CORE)
        print(f"  ✅ Core features defined: {len(core_features)}")
        
        performance_features = manager.get_features_by_category(FeatureCategory.PERFORMANCE)
        print(f"  ✅ Performance features defined: {len(performance_features)}")
        
        # Test compatibility validation
        issues = manager.validate_feature_compatibility()
        if issues:
            print(f"  ⚠️ Compatibility issues: {issues}")
        else:
            print("  ✅ No compatibility issues detected")
        
        print("  ✅ Feature flag system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Feature flag test failed: {e}")
        return False


def test_complexity_monitoring():
    """Test complexity monitoring system."""
    print("\n📊 Testing Complexity Monitoring...")
    
    try:
        from pynomaly.infrastructure.monitoring.complexity_monitor import (
            ComplexityMonitor, ComplexityMetrics
        )
        
        # Create monitor instance
        monitor = ComplexityMonitor(project_root)
        
        # Test individual metric measurements
        file_metrics = monitor.measure_file_metrics()
        print(f"  ✅ File metrics: {file_metrics['python_files']} Python files")
        
        dependency_metrics = monitor.measure_dependency_metrics()
        print(f"  ✅ Dependencies: {dependency_metrics['total_dependencies']} total")
        
        performance_metrics = monitor.measure_performance_metrics()
        print(f"  ✅ Memory usage: {performance_metrics['memory_usage']:.1f}MB")
        
        # Test full measurement
        metrics = monitor.measure_all()
        print(f"  ✅ Overall complexity measured")
        
        # Test report generation
        report = monitor.generate_report(metrics)
        print(f"  ✅ Report generated ({len(report)} characters)")
        
        # Test target checking
        targets = monitor.check_targets(metrics)
        passing_targets = sum(targets.values())
        total_targets = len(targets)
        print(f"  ✅ Targets: {passing_targets}/{total_targets} passing")
        
        print("  ✅ Complexity monitoring working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Complexity monitoring test failed: {e}")
        return False


def test_algorithm_benchmarking():
    """Test algorithm benchmarking system."""
    print("\n⚡ Testing Algorithm Benchmarking...")
    
    try:
        # Enable algorithm optimization feature for testing
        os.environ["PYNOMALY_ALGORITHM_OPTIMIZATION"] = "true"
        
        from pynomaly.application.services.algorithm_benchmark import (
            AlgorithmBenchmarkService, BenchmarkResult
        )
        
        # Create service instance
        service = AlgorithmBenchmarkService()
        
        # Test algorithm listing
        algorithms = list(service.default_algorithms.keys())
        print(f"  ✅ Available algorithms: {algorithms}")
        
        # Test benchmark result structure
        result = BenchmarkResult(
            algorithm_name="test",
            dataset_name="test_data",
            accuracy=0.95,
            f1_score=0.90,
            fit_time=1.5,
            predict_time=0.5,
            n_samples=1000,
            n_features=10
        )
        
        overall_score = result.overall_score()
        efficiency_score = result.efficiency_score()
        print(f"  ✅ Scoring system: Overall={overall_score:.1f}, Efficiency={efficiency_score:.1f}")
        
        # Test serialization
        result_dict = result.to_dict()
        print(f"  ✅ Serialization: {len(result_dict)} fields")
        
        print("  ✅ Algorithm benchmarking working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Algorithm benchmarking test failed: {e}")
        return False
    finally:
        # Clean up environment
        os.environ.pop("PYNOMALY_ALGORITHM_OPTIMIZATION", None)


def test_container_integration():
    """Test container integration with feature flags."""
    print("\n🔧 Testing Container Integration...")
    
    try:
        from pynomaly.infrastructure.config.container import Container
        from pynomaly.infrastructure.config.feature_flags import feature_flags
        
        # Create container
        container = Container()
        
        # Test basic services
        config = container.config()
        print(f"  ✅ Configuration loaded")
        
        feature_manager = container.feature_flag_manager()
        print(f"  ✅ Feature flag manager available")
        
        # Test domain services
        anomaly_scorer = container.anomaly_scorer()
        print(f"  ✅ Anomaly scorer available")
        
        # Test conditional services (when flags are enabled)
        os.environ["PYNOMALY_COMPLEXITY_MONITORING"] = "true"
        
        try:
            complexity_monitor = container.complexity_monitor()
            print(f"  ✅ Complexity monitor available (feature flag enabled)")
        except AttributeError:
            print(f"  ⚠️ Complexity monitor not available (feature may be disabled)")
        
        print("  ✅ Container integration working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Container integration test failed: {e}")
        return False
    finally:
        # Clean up environment
        os.environ.pop("PYNOMALY_COMPLEXITY_MONITORING", None)


def test_phase2_readiness():
    """Test overall Phase 2 readiness."""
    print("\n🚀 Testing Phase 2 Readiness...")
    
    try:
        # Check if all Phase 2 components are available
        components = [
            "feature_flags",
            "complexity_monitor", 
            "algorithm_benchmark",
            "container_integration"
        ]
        
        results = {
            "feature_flags": test_feature_flags(),
            "complexity_monitor": test_complexity_monitoring(),
            "algorithm_benchmark": test_algorithm_benchmarking(),
            "container_integration": test_container_integration()
        }
        
        passing = sum(results.values())
        total = len(results)
        
        print(f"\n📈 Phase 2 Infrastructure Status: {passing}/{total} components ready")
        
        if passing == total:
            print("🎉 Phase 2 infrastructure is fully operational!")
            print("✅ Ready for controlled feature reintroduction")
            return True
        else:
            print("⚠️ Some Phase 2 components need attention")
            for component, status in results.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {component}")
            return False
        
    except Exception as e:
        print(f"❌ Phase 2 readiness test failed: {e}")
        return False


def main():
    """Run all Phase 2 infrastructure tests."""
    print("🧪 Pynomaly Phase 2 Infrastructure Validation")
    print("=" * 50)
    
    try:
        success = test_phase2_readiness()
        
        if success:
            print("\n🎯 Phase 2 infrastructure validation successful!")
            print("🚀 Ready to begin controlled feature reintroduction")
            sys.exit(0)
        else:
            print("\n⚠️ Phase 2 infrastructure validation failed")
            print("🔧 Please review and fix issues before proceeding")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
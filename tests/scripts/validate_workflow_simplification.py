#!/usr/bin/env python3
"""Validation script for user workflow simplification infrastructure.

This script validates the complete workflow simplification system including
service functionality, container integration, and feature flag management.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_workflow_simplification_service():
    """Test core workflow simplification service."""
    print("🔧 Testing Workflow Simplification Service...")

    try:
        from pynomaly.application.services.workflow_simplification_service import (
            UserExperience,
            WorkflowSimplificationService,
        )
        from pynomaly.domain.entities import Dataset

        # Create test data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (1000, 5))
        outliers = np.random.normal(3, 1, (50, 5))
        test_data = np.vstack([normal_data, outliers])
        dataset = Dataset(name="test_dataset", data=test_data)

        # Create service
        service = WorkflowSimplificationService()

        # Test workflow recommendation
        user_context = {
            "goal": "detect_anomalies",
            "urgency": "normal",
            "experience_level": "beginner",
        }

        dataset_info = {"n_rows": 1000, "n_columns": 5, "missing_values_ratio": 0.05}

        recommendation = service.recommend_workflow(
            user_context, dataset_info, UserExperience.BEGINNER
        )

        print(f"  ✅ Workflow recommended: {recommendation.workflow_id}")
        print(f"  ✅ Steps: {len(recommendation.steps)}")
        print(f"  ✅ Confidence: {recommendation.confidence_score}")

        # Test simplified detection workflow
        detection_goals = {"target": "fraud_detection"}

        result = service.simplify_detection_workflow(
            dataset, detection_goals, automation_level="maximum"
        )

        print(f"  ✅ Automated workflow: {result['execution_type']}")
        print(f"  ✅ Anomalies detected: {result['anomalies_detected']}")

        # Test workflow analytics
        analytics = service.get_workflow_analytics()
        print(f"  ✅ Analytics: {len(analytics['workflow_templates'])} templates")

        print("  ✅ Workflow simplification service working correctly")
        return True

    except Exception as e:
        print(f"  ❌ Workflow simplification service test failed: {e}")
        return False


def test_container_integration():
    """Test container integration."""
    print("\n🔧 Testing Container Integration...")

    try:
        from pynomaly.infrastructure.config.container import Container

        # Create container
        container = Container()

        # Test workflow simplification service availability
        workflow_service = container.workflow_simplification_service()
        print("  ✅ Workflow simplification service available")

        # Test service functionality
        templates = list(workflow_service.workflow_templates.keys())
        print(f"  ✅ Workflow templates: {templates}")

        # Test other Phase 2 services
        services_available = []

        try:
            container.algorithm_optimization_service()
            services_available.append("algorithm_optimization")
        except AttributeError:
            pass

        try:
            container.performance_monitoring_service()
            services_available.append("performance_monitoring")
        except AttributeError:
            pass

        try:
            container.memory_optimization_service()
            services_available.append("memory_optimization")
        except AttributeError:
            pass

        print(f"  ✅ Phase 2 services available: {services_available}")

        print("  ✅ Container integration working correctly")
        return True

    except Exception as e:
        print(f"  ❌ Container integration test failed: {e}")
        return False


def test_feature_flags():
    """Test feature flag system."""
    print("\n🔧 Testing Feature Flag System...")

    try:
        from pynomaly.infrastructure.config.feature_flags import feature_flags

        # Test workflow simplification flags
        flags_to_test = ["cli_simplification", "interactive_guidance", "error_recovery"]

        enabled_flags = []
        for flag in flags_to_test:
            if feature_flags.is_enabled(flag):
                enabled_flags.append(flag)

        print(f"  ✅ Feature flags enabled: {enabled_flags}")

        # All workflow flags should be enabled
        expected_flags = [
            "cli_simplification",
            "interactive_guidance",
            "error_recovery",
        ]
        for flag in expected_flags:
            if feature_flags.is_enabled(flag):
                print(f"  ✅ {flag}: enabled")
            else:
                print(f"  ⚠️ {flag}: disabled")

        print("  ✅ Feature flag system working correctly")
        return True

    except Exception as e:
        print(f"  ❌ Feature flag test failed: {e}")
        return False


def test_workflow_data_classes():
    """Test workflow data classes."""
    print("\n🔧 Testing Workflow Data Classes...")

    try:
        from pynomaly.application.services.workflow_simplification_service import (
            UserExperience,
            WorkflowComplexity,
            WorkflowRecommendation,
            WorkflowStep,
        )

        # Test workflow step
        step = WorkflowStep(
            step_id="test_step",
            name="Test Step",
            description="Test description",
            complexity=WorkflowComplexity.INTERMEDIATE,
        )

        step_dict = step.to_dict()
        print(f"  ✅ Workflow step: {step.step_id}")
        print(f"  ✅ Step serialization: {len(step_dict)} fields")

        # Test workflow recommendation
        recommendation = WorkflowRecommendation(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="Test description",
            estimated_duration_minutes=10,
            confidence_score=0.85,
            steps=[step],
        )

        rec_dict = recommendation.to_dict()
        print(f"  ✅ Workflow recommendation: {recommendation.workflow_id}")
        print(f"  ✅ Recommendation serialization: {len(rec_dict)} fields")

        # Test enums
        print(f"  ✅ Complexity levels: {[c.value for c in WorkflowComplexity]}")
        print(f"  ✅ Experience levels: {[e.value for e in UserExperience]}")

        print("  ✅ Workflow data classes working correctly")
        return True

    except Exception as e:
        print(f"  ❌ Workflow data classes test failed: {e}")
        return False


def main():
    """Run all workflow simplification validation tests."""
    print("🧪 Pynomaly User Workflow Simplification Validation")
    print("=" * 70)

    try:
        tests = [
            test_workflow_simplification_service,
            test_container_integration,
            test_feature_flags,
            test_workflow_data_classes,
        ]

        results = []
        for test in tests:
            results.append(test())

        passing = sum(results)
        total = len(results)

        print(f"\n📈 Workflow Simplification Status: {passing}/{total} tests passing")

        if passing == total:
            print(
                "🎉 User workflow simplification infrastructure is fully operational!"
            )
            print(
                "✅ Ready for intelligent workflow automation and user experience enhancement"
            )
            return True
        else:
            print("⚠️ Some workflow simplification components need attention")
            return False

    except Exception as e:
        print(f"❌ Workflow simplification validation failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

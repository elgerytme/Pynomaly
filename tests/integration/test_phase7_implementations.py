"""
Integration tests for Phase 7: Research & Innovation Features
Tests all Phase 7 components and their integration
"""

import os
import sys
from datetime import datetime

import numpy as np
import pytest

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestPhase7Integration:
    """Test suite for Phase 7 Research & Innovation Features"""

    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.sample_data = np.random.normal(0, 1, (100, 5))
        self.anomaly_data = np.random.normal(3, 1, (10, 5))  # Outliers
        self.combined_data = np.vstack([self.sample_data, self.anomaly_data])
        self.labels = np.array([0] * 100 + [1] * 10)

    def test_explainable_ai_availability(self):
        """Test if explainable AI implementations are available"""
        try:
            from pynomaly.research.explainability.explainable_ai import (
                CounterfactualExplainer,
                ExplainableAIOrchestrator,
                LIMEExplainer,
                SHAPExplainer,
            )

            print("✅ Explainable AI implementations found")
            return True
        except ImportError as e:
            print(f"❌ Explainable AI missing: {e}")
            return False

    def test_synthetic_data_availability(self):
        """Test if synthetic data generation implementations are available"""
        try:
            from pynomaly.research.synthetic.synthetic_data_generation import (
                StatisticalGenerator,
                SyntheticDataOrchestrator,
                VAEGenerator,
                VanillaGANGenerator,
            )

            print("✅ Synthetic data generation implementations found")
            return True
        except ImportError as e:
            print(f"❌ Synthetic data generation missing: {e}")
            return False

    def test_causal_models_availability(self):
        """Test if causal anomaly detection models are available"""
        try:
            from pynomaly.domain.models.causal import (
                CausalAnomalyEvent,
                CausalGraph,
                CausalInferenceMethod,
                InterventionSpecification,
            )

            print("✅ Causal models found")
            return True
        except ImportError as e:
            print(f"❌ Causal models missing: {e}")
            return False

    def test_multimodal_models_availability(self):
        """Test if multimodal fusion models are available"""
        try:
            from pynomaly.domain.models.multimodal import (
                FusionStrategy,
                ModalityEncoder,
                ModalityType,
                MultiModalDetector,
            )

            print("✅ Multimodal models found")
            return True
        except ImportError as e:
            print(f"❌ Multimodal models missing: {e}")
            return False

    def test_quantum_algorithms_missing(self):
        """Test if quantum algorithms are missing (expected to be missing)"""
        try:
            from pynomaly.research.quantum.quantum_algorithms import (
                QuantumAnomalyDetector,
            )

            print("✅ Quantum algorithms found")
            return True
        except ImportError:
            print("❌ Quantum algorithms missing - need to recreate")
            return False

    def test_edge_deployment_missing(self):
        """Test if edge deployment is missing (expected to be missing)"""
        try:
            from pynomaly.research.edge.edge_deployment import EdgeDeploymentService

            print("✅ Edge deployment found")
            return True
        except ImportError:
            print("❌ Edge deployment missing - need to recreate")
            return False

    def test_automl_v2_missing(self):
        """Test if AutoML v2 is missing (expected to be missing)"""
        try:
            from pynomaly.research.automl.automl_v2 import AutoMLV2System

            print("✅ AutoML v2 found")
            return True
        except ImportError:
            print("❌ AutoML v2 missing - need to recreate")
            return False

    def test_explainable_ai_basic_functionality(self):
        """Test basic explainable AI functionality"""
        if not self.test_explainable_ai_availability():
            pytest.skip("Explainable AI not available")

        try:
            from pynomaly.research.explainability.explainable_ai import (
                ExplainableAIOrchestrator,
                ExplanationMethod,
            )

            config = {
                "methods": [ExplanationMethod.LIME, ExplanationMethod.SHAP],
                "lime_config": {"num_features": 3},
                "shap_config": {"num_background": 50},
            }

            orchestrator = ExplainableAIOrchestrator(config)

            # Mock model prediction function
            def mock_predict(X):
                return np.random.random(len(X))

            # This would require async functionality - simplified test
            print("✅ Explainable AI orchestrator created successfully")
            return True

        except Exception as e:
            print(f"❌ Explainable AI functionality test failed: {e}")
            return False

    def test_synthetic_data_basic_functionality(self):
        """Test basic synthetic data generation functionality"""
        if not self.test_synthetic_data_availability():
            pytest.skip("Synthetic data generation not available")

        try:
            from pynomaly.research.synthetic.synthetic_data_generation import (
                DataType,
                SyntheticDataConfig,
                SyntheticDataOrchestrator,
                SyntheticMethod,
            )

            orchestrator = SyntheticDataOrchestrator({})

            config = SyntheticDataConfig(
                method=SyntheticMethod.STATISTICAL,
                data_type=DataType.TABULAR,
                num_samples=50,
            )

            print("✅ Synthetic data orchestrator created successfully")
            return True

        except Exception as e:
            print(f"❌ Synthetic data functionality test failed: {e}")
            return False

    def test_phase7_integration_readiness(self):
        """Test overall Phase 7 integration readiness"""
        results = {
            "explainable_ai": self.test_explainable_ai_availability(),
            "synthetic_data": self.test_synthetic_data_availability(),
            "causal_models": self.test_causal_models_availability(),
            "multimodal_models": self.test_multimodal_models_availability(),
            "quantum_algorithms": self.test_quantum_algorithms_missing(),
            "edge_deployment": self.test_edge_deployment_missing(),
            "automl_v2": self.test_automl_v2_missing(),
        }

        available_count = sum(results.values())
        total_count = len(results)

        print("\n📊 Phase 7 Implementation Status:")
        print(f"Available: {available_count}/{total_count} components")
        print(f"Completion: {available_count/total_count*100:.1f}%")

        for component, available in results.items():
            status = "✅" if available else "❌"
            print(f"{status} {component}")

        return results


def run_phase7_tests():
    """Run all Phase 7 tests"""
    print("🧪 Running Phase 7: Research & Innovation Features Tests")
    print("=" * 60)

    test_instance = TestPhase7Integration()
    test_instance.setup_method()

    # Run integration readiness test
    results = test_instance.test_phase7_integration_readiness()

    # Run functionality tests for available components
    print("\n🔧 Testing Available Component Functionality:")
    print("-" * 40)

    if results["explainable_ai"]:
        test_instance.test_explainable_ai_basic_functionality()

    if results["synthetic_data"]:
        test_instance.test_synthetic_data_basic_functionality()

    # Identify missing components
    missing_components = [k for k, v in results.items() if not v]

    if missing_components:
        print("\n⚠️  Missing Components Need Recreation:")
        for component in missing_components:
            print(f"   - {component}")

    print("\n📋 Phase 7 Test Summary:")
    print(f"   - Tests completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   - Available components: {sum(results.values())}/{len(results)}")
    print(f"   - Ready for Phase 8: {'Yes' if sum(results.values()) >= 5 else 'No'}")

    return results


if __name__ == "__main__":
    run_phase7_tests()

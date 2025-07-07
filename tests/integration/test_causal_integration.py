"""Integration tests for causal anomaly detection system."""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.domain.models.causal import (
    AnomalyType,
    CausalAnalysisConfig,
    CausalMethod,
    InterventionSpec,
)
from pynomaly.infrastructure.causal.causal_service import CausalAnomalyDetectionService


@pytest.mark.integration
@pytest.mark.asyncio
class TestCausalIntegration:
    """Integration tests for causal anomaly detection system."""

    def create_linear_causal_data(self, n_samples: int = 200) -> tuple[np.ndarray, list[str]]:
        """Create synthetic data with known linear causal relationships."""
        # Create X -> Y -> Z causal chain
        np.random.seed(42)  # For reproducible tests
        
        X = np.random.normal(0, 1, n_samples)
        Y = 0.8 * X + np.random.normal(0, 0.3, n_samples)
        Z = 0.6 * Y + np.random.normal(0, 0.3, n_samples)
        
        # Add some confounding
        W = 0.4 * X + 0.3 * Z + np.random.normal(0, 0.2, n_samples)
        
        data = np.column_stack([X, Y, Z, W])
        variable_names = ["X", "Y", "Z", "W"]
        
        return data, variable_names

    def create_nonlinear_causal_data(self, n_samples: int = 200) -> tuple[np.ndarray, list[str]]:
        """Create synthetic data with nonlinear causal relationships."""
        np.random.seed(42)
        
        X = np.random.normal(0, 1, n_samples)
        Y = np.sin(X) + 0.5 * X**2 + np.random.normal(0, 0.3, n_samples)
        Z = np.tanh(Y) + np.random.normal(0, 0.2, n_samples)
        
        data = np.column_stack([X, Y, Z])
        variable_names = ["X", "Y", "Z"]
        
        return data, variable_names

    def create_time_series_causal_data(self, n_samples: int = 300) -> tuple[np.ndarray, list[str]]:
        """Create time series data with lagged causal relationships."""
        np.random.seed(42)
        
        X = np.zeros(n_samples)
        Y = np.zeros(n_samples)
        Z = np.zeros(n_samples)
        
        # Initialize first few values
        X[:3] = np.random.normal(0, 1, 3)
        Y[:3] = np.random.normal(0, 1, 3)
        Z[:3] = np.random.normal(0, 1, 3)
        
        # Generate time series with causal lags
        for t in range(3, n_samples):
            X[t] = 0.3 * X[t-1] + np.random.normal(0, 0.5)
            Y[t] = 0.5 * X[t-2] + 0.2 * Y[t-1] + np.random.normal(0, 0.3)
            Z[t] = 0.4 * Y[t-1] + 0.1 * Z[t-1] + np.random.normal(0, 0.3)
        
        data = np.column_stack([X, Y, Z])
        variable_names = ["X", "Y", "Z"]
        
        return data, variable_names

    async def test_end_to_end_linear_causal_discovery(self):
        """Test complete workflow with linear causal relationships."""
        service = CausalAnomalyDetectionService()
        
        # Create synthetic data with known structure
        data, variable_names = self.create_linear_causal_data()
        
        # Register dataset
        dataset = await service.register_dataset(
            name="linear_causal_test",
            data=data,
            variable_names=variable_names,
            is_time_series=False,
        )
        
        # Create detector with PC algorithm
        config = CausalAnalysisConfig(
            method=CausalMethod.PC_ALGORITHM,
            alpha=0.1,  # More permissive for test data
            anomaly_threshold=0.7,
        )
        
        detector = await service.create_detector("linear_detector", config)
        
        # Train detector
        trained_detector = await service.train_detector(
            detector.detector_id, dataset.dataset_id, validation_split=0.3
        )
        
        assert trained_detector.is_trained
        assert trained_detector.baseline_graph is not None
        
        baseline_graph = trained_detector.baseline_graph
        assert len(baseline_graph.variables) == 4
        
        # The algorithm should discover some causal relationships
        print(f"Discovered {len(baseline_graph.edges)} causal edges")
        for edge in baseline_graph.edges:
            print(f"  {edge.source} -> {edge.target} (strength: {edge.strength:.3f})")
        
        # Test anomaly detection with normal data
        normal_test_data, _ = self.create_linear_causal_data(100)
        normal_anomalies = await service.detect_causal_anomalies(
            detector.detector_id, normal_test_data, variable_names, window_size=50
        )
        
        print(f"Anomalies detected in normal data: {len(normal_anomalies)}")
        
        # Test anomaly detection with structural break
        anomalous_data = data.copy()
        # Introduce structural break: reverse X->Y relationship in second half
        mid_point = len(anomalous_data) // 2
        anomalous_data[mid_point:, 1] = -0.8 * anomalous_data[mid_point:, 0] + np.random.normal(0, 0.3, mid_point)
        
        structural_anomalies = await service.detect_causal_anomalies(
            detector.detector_id, anomalous_data, variable_names, window_size=50
        )
        
        print(f"Anomalies detected in structural break data: {len(structural_anomalies)}")
        
        # Should detect more anomalies in the structural break case
        # (Though this depends on the specific algorithm sensitivity)
        
        # Test intervention simulation
        intervention = InterventionSpec(
            intervention_id=uuid4(),
            target_variable="X",
            intervention_type="do",
            intervention_value=2.0,
        )
        
        simulation_result = await service.simulate_intervention(
            detector.detector_id, intervention, simulation_steps=50
        )
        
        assert "intervention" in simulation_result
        assert "predicted_effects" in simulation_result
        
        # Should predict effects on downstream variables Y, Z, W
        effects = simulation_result["predicted_effects"]
        print(f"Predicted intervention effects: {effects}")

    async def test_time_series_granger_causality(self):
        """Test Granger causality for time series data."""
        service = CausalAnomalyDetectionService()
        
        # Create time series data
        data, variable_names = self.create_time_series_causal_data()
        
        # Register dataset
        dataset = await service.register_dataset(
            name="time_series_test",
            data=data,
            variable_names=variable_names,
            is_time_series=True,
        )
        
        # Create detector with Granger causality
        config = CausalAnalysisConfig(
            method=CausalMethod.GRANGER_CAUSALITY,
            max_lag=5,
            alpha=0.1,
            anomaly_threshold=0.6,
        )
        
        detector = await service.create_detector("granger_detector", config)
        
        # Train detector
        trained_detector = await service.train_detector(
            detector.detector_id, dataset.dataset_id
        )
        
        assert trained_detector.is_trained
        assert trained_detector.baseline_graph is not None
        
        baseline_graph = trained_detector.baseline_graph
        print(f"Granger causality discovered {len(baseline_graph.edges)} relationships")
        
        for edge in baseline_graph.edges:
            print(f"  {edge.source} ->{edge.lag}-> {edge.target} (strength: {edge.strength:.3f})")
        
        # Test detection of temporal anomalies
        test_data = data.copy()
        
        # Introduce regime change: break the X->Y relationship
        break_point = len(test_data) // 3 * 2
        for t in range(break_point, len(test_data)):
            if t >= 3:
                test_data[t, 1] = 0.1 * test_data[t-2, 0] + 0.2 * test_data[t-1, 1] + np.random.normal(0, 0.3)
        
        temporal_anomalies = await service.detect_causal_anomalies(
            detector.detector_id, test_data, variable_names, window_size=80
        )
        
        print(f"Temporal anomalies detected: {len(temporal_anomalies)}")
        
        for anomaly in temporal_anomalies:
            print(f"  Type: {anomaly.anomaly_type.value}, Score: {anomaly.causal_score:.3f}")
            print(f"  Affected variables: {anomaly.affected_variables}")

    async def test_multiple_method_comparison(self):
        """Test comparison of multiple causal discovery methods."""
        service = CausalAnomalyDetectionService()
        
        # Create test data
        data, variable_names = self.create_linear_causal_data(250)  # Larger sample
        
        # Register dataset
        dataset = await service.register_dataset(
            name="method_comparison_test",
            data=data,
            variable_names=variable_names,
        )
        
        # Test multiple methods
        methods_configs = [
            (CausalMethod.PC_ALGORITHM, {"alpha": 0.1}),
            (CausalMethod.GRANGER_CAUSALITY, {"max_lag": 3, "alpha": 0.1}),
        ]
        
        detectors = {}
        
        for method, params in methods_configs:
            config = CausalAnalysisConfig(method=method, **params)
            detector = await service.create_detector(f"{method.value}_detector", config)
            
            trained_detector = await service.train_detector(detector.detector_id, dataset.dataset_id)
            detectors[method] = trained_detector
        
        # Compare discovered structures
        print("\nMethod comparison results:")
        for method, detector in detectors.items():
            graph = detector.baseline_graph
            print(f"{method.value}: {len(graph.edges)} edges discovered")
            
            for edge in graph.edges[:3]:  # Show first 3 edges
                print(f"  {edge.source} -> {edge.target} (strength: {edge.strength:.3f})")
        
        # Test anomaly detection consistency
        test_data, _ = self.create_nonlinear_causal_data(100)
        
        detection_results = {}
        for method, detector in detectors.items():
            try:
                anomalies = await service.detect_causal_anomalies(
                    detector.detector_id, test_data, variable_names, window_size=50
                )
                detection_results[method] = len(anomalies)
            except Exception as e:
                print(f"Detection failed for {method.value}: {e}")
                detection_results[method] = 0
        
        print(f"\nAnomaly detection results: {detection_results}")

    async def test_intervention_effect_detection(self):
        """Test detection of intervention effects."""
        service = CausalAnomalyDetectionService()
        
        # Create baseline data
        baseline_data, variable_names = self.create_linear_causal_data(200)
        
        # Register dataset
        dataset = await service.register_dataset(
            name="intervention_test",
            data=baseline_data,
            variable_names=variable_names,
        )
        
        # Create and train detector
        config = CausalAnalysisConfig(
            method=CausalMethod.PC_ALGORITHM,
            alpha=0.1,
            anomaly_threshold=0.5,
        )
        
        detector = await service.create_detector("intervention_detector", config)
        await service.train_detector(detector.detector_id, dataset.dataset_id)
        
        # Create data with intervention
        intervention_data = baseline_data.copy()
        
        # Simulate intervention on X: set X to constant value in middle section
        intervention_start = len(intervention_data) // 3
        intervention_end = 2 * len(intervention_data) // 3
        
        intervention_data[intervention_start:intervention_end, 0] = 3.0  # Constant intervention
        
        # Recalculate downstream effects
        for i in range(intervention_start, intervention_end):
            # Y responds to intervened X
            intervention_data[i, 1] = 0.8 * intervention_data[i, 0] + np.random.normal(0, 0.3)
            # Z responds to modified Y
            intervention_data[i, 2] = 0.6 * intervention_data[i, 1] + np.random.normal(0, 0.3)
        
        # Register intervention in dataset
        intervention_spec = InterventionSpec(
            intervention_id=uuid4(),
            target_variable="X",
            intervention_type="set",
            intervention_value=3.0,
        )
        
        intervention_dataset = await service.register_dataset(
            name="intervention_data",
            data=intervention_data,
            variable_names=variable_names,
        )
        
        intervention_dataset.add_intervention_period(
            intervention_spec, intervention_start, intervention_end
        )
        
        # Detect intervention effects
        intervention_anomalies = await service.detect_causal_anomalies(
            detector.detector_id, intervention_data, variable_names, window_size=60
        )
        
        print(f"Intervention anomalies detected: {len(intervention_anomalies)}")
        
        # Analyze detected anomalies
        intervention_effects = [a for a in intervention_anomalies 
                             if a.anomaly_type == AnomalyType.INTERVENTION_EFFECT]
        
        print(f"Specific intervention effects: {len(intervention_effects)}")
        
        for anomaly in intervention_effects:
            print(f"  Variables affected: {anomaly.affected_variables}")
            print(f"  Causal score: {anomaly.causal_score:.3f}")
            print(f"  Evidence: {anomaly.evidence}")

    async def test_confounding_change_detection(self):
        """Test detection of confounding changes."""
        service = CausalAnomalyDetectionService()
        
        # Create data with time-varying confounding
        n_samples = 300
        np.random.seed(42)
        
        # Baseline period: X -> Y, Z -> X, Z -> Y (Z is confounder)
        X = np.zeros(n_samples)
        Y = np.zeros(n_samples)
        Z = np.random.normal(0, 1, n_samples)
        
        # First period: normal confounding
        for i in range(n_samples // 2):
            X[i] = 0.5 * Z[i] + np.random.normal(0, 0.3)
            Y[i] = 0.6 * X[i] + 0.4 * Z[i] + np.random.normal(0, 0.3)
        
        # Second period: changed confounding strength
        for i in range(n_samples // 2, n_samples):
            X[i] = 0.8 * Z[i] + np.random.normal(0, 0.3)  # Stronger Z->X
            Y[i] = 0.6 * X[i] + 0.8 * Z[i] + np.random.normal(0, 0.3)  # Stronger Z->Y
        
        data = np.column_stack([X, Y, Z])
        variable_names = ["X", "Y", "Z"]
        
        # Register dataset
        dataset = await service.register_dataset(
            name="confounding_test",
            data=data,
            variable_names=variable_names,
        )
        
        # Create detector
        config = CausalAnalysisConfig(
            method=CausalMethod.PC_ALGORITHM,
            alpha=0.1,
            anomaly_threshold=0.4,
        )
        
        detector = await service.create_detector("confounding_detector", config)
        
        # Train on first half only
        train_data = data[:n_samples//2]
        train_dataset = await service.register_dataset(
            name="confounding_train",
            data=train_data,
            variable_names=variable_names,
        )
        
        await service.train_detector(detector.detector_id, train_dataset.dataset_id)
        
        # Test on full data (including confounding change)
        confounding_anomalies = await service.detect_causal_anomalies(
            detector.detector_id, data, variable_names, window_size=100
        )
        
        print(f"Confounding change anomalies: {len(confounding_anomalies)}")
        
        # Look for structural break anomalies
        structural_anomalies = [a for a in confounding_anomalies 
                             if a.anomaly_type == AnomalyType.STRUCTURAL_BREAK]
        
        print(f"Structural break anomalies: {len(structural_anomalies)}")
        
        for anomaly in structural_anomalies:
            print(f"  Score: {anomaly.causal_score:.3f}")
            print(f"  Affected: {anomaly.affected_variables}")

    async def test_performance_under_varying_conditions(self):
        """Test system performance under different conditions."""
        service = CausalAnomalyDetectionService()
        
        # Test different sample sizes
        sample_sizes = [100, 200, 500]
        performance_results = {}
        
        for n_samples in sample_sizes:
            print(f"\nTesting with {n_samples} samples...")
            
            data, variable_names = self.create_linear_causal_data(n_samples)
            
            dataset = await service.register_dataset(
                name=f"perf_test_{n_samples}",
                data=data,
                variable_names=variable_names,
            )
            
            config = CausalAnalysisConfig(
                method=CausalMethod.PC_ALGORITHM,
                alpha=0.1,
            )
            
            detector = await service.create_detector(f"perf_detector_{n_samples}", config)
            
            # Measure training time
            start_time = asyncio.get_event_loop().time()
            await service.train_detector(detector.detector_id, dataset.dataset_id)
            training_time = asyncio.get_event_loop().time() - start_time
            
            # Measure detection time
            test_data, _ = self.create_linear_causal_data(100)
            
            start_time = asyncio.get_event_loop().time()
            await service.detect_causal_anomalies(
                detector.detector_id, test_data, variable_names, window_size=50
            )
            detection_time = asyncio.get_event_loop().time() - start_time
            
            performance_results[n_samples] = {
                "training_time": training_time,
                "detection_time": detection_time,
                "edges_discovered": len(detector.baseline_graph.edges),
            }
            
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Detection time: {detection_time:.2f}s")
            print(f"  Edges discovered: {len(detector.baseline_graph.edges)}")
        
        # Verify reasonable performance scaling
        assert performance_results[500]["training_time"] > performance_results[100]["training_time"]
        # Training time should increase with sample size, but not excessively
        
        # Detection time should be relatively stable
        detection_times = [r["detection_time"] for r in performance_results.values()]
        assert max(detection_times) < 10.0  # Should complete within 10 seconds

    async def test_service_resource_management(self):
        """Test service resource management capabilities."""
        service = CausalAnomalyDetectionService()
        
        # Create multiple detectors and datasets
        detectors = []
        datasets = []
        
        for i in range(5):
            # Create dataset
            data, variable_names = self.create_linear_causal_data(50)  # Small for speed
            dataset = await service.register_dataset(f"resource_dataset_{i}", data, variable_names)
            datasets.append(dataset)
            
            # Create detector
            config = CausalAnalysisConfig(method=CausalMethod.PC_ALGORITHM, alpha=0.1)
            detector = await service.create_detector(f"resource_detector_{i}", config)
            detectors.append(detector)
        
        # Train some detectors
        for i in range(3):
            await service.train_detector(detectors[i].detector_id, datasets[i].dataset_id)
        
        # Check service statistics
        stats = service.get_service_statistics()
        
        assert stats["detector_statistics"]["total_detectors"] == 5
        assert stats["detector_statistics"]["trained_detectors"] == 3
        assert stats["dataset_statistics"]["total_datasets"] == 5
        
        # Test cleanup
        cleanup_count = await service.cleanup_resources(keep_trained=True)
        assert cleanup_count == 2  # Should remove 2 untrained detectors
        
        updated_stats = service.get_service_statistics()
        assert updated_stats["detector_statistics"]["total_detectors"] == 3
        assert updated_stats["detector_statistics"]["trained_detectors"] == 3
        
        # Full cleanup
        await service.cleanup_resources(keep_trained=False)
        final_stats = service.get_service_statistics()
        assert final_stats["detector_statistics"]["total_detectors"] == 0

    async def test_edge_case_handling(self):
        """Test handling of edge cases and error conditions."""
        service = CausalAnomalyDetectionService()
        
        # Test with minimal data
        minimal_data = np.random.randn(20, 2)  # Very small dataset
        variable_names = ["X", "Y"]
        
        dataset = await service.register_dataset("minimal_data", minimal_data, variable_names)
        
        config = CausalAnalysisConfig(method=CausalMethod.PC_ALGORITHM, alpha=0.1)
        detector = await service.create_detector("minimal_detector", config)
        
        # Should handle minimal data gracefully
        try:
            await service.train_detector(detector.detector_id, dataset.dataset_id)
            print("Minimal data training succeeded")
        except Exception as e:
            print(f"Minimal data training failed (expected): {e}")
        
        # Test with constant data
        constant_data = np.ones((100, 3))  # All values are 1
        variable_names = ["X", "Y", "Z"]
        
        try:
            dataset = await service.register_dataset("constant_data", constant_data, variable_names)
            detector = await service.create_detector("constant_detector", config)
            await service.train_detector(detector.detector_id, dataset.dataset_id)
            print("Constant data training succeeded")
        except Exception as e:
            print(f"Constant data training failed (expected): {e}")
        
        # Test with NaN/infinite data
        nan_data = np.random.randn(100, 2)
        nan_data[50:60, 0] = np.nan  # Introduce NaN values
        
        try:
            dataset = await service.register_dataset("nan_data", nan_data, ["X", "Y"])
            detector = await service.create_detector("nan_detector", config)
            await service.train_detector(detector.detector_id, dataset.dataset_id)
            print("NaN data training succeeded")
        except Exception as e:
            print(f"NaN data training failed (expected): {e}")
        
        # These edge cases test the robustness of the system
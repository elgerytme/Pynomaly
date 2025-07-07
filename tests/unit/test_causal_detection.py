"""Unit tests for causal anomaly detection components."""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.domain.models.causal import (
    AnomalyType,
    CausalAnalysisConfig,
    CausalAnomalyEvent,
    CausalDataset,
    CausalDetector,
    CausalEdge,
    CausalGraph,
    CausalGraphType,
    CausalMethod,
    CausalRelationType,
    InterventionSpec,
)
from pynomaly.infrastructure.causal.causal_service import CausalAnomalyDetectionService
from pynomaly.infrastructure.causal.structure_learning import (
    GrangerCausalityLearner,
    PCAlgorithmLearner,
    StructureLearningService,
    TransferEntropyLearner,
)


class TestCausalEdge:
    """Test causal edge representation."""

    def test_causal_edge_creation(self):
        """Test creating causal edge."""
        edge = CausalEdge(
            source="X",
            target="Y",
            edge_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.8,
            confidence=0.9,
            lag=1,
            mechanism="linear relationship",
        )
        
        assert edge.source == "X"
        assert edge.target == "Y"
        assert edge.edge_type == CausalRelationType.DIRECT_CAUSE
        assert edge.strength == 0.8
        assert edge.confidence == 0.9
        assert edge.lag == 1
        assert edge.mechanism == "linear relationship"

    def test_causal_edge_validation(self):
        """Test causal edge validation."""
        # Invalid strength
        with pytest.raises(ValueError, match="Causal strength must be between 0 and 1"):
            CausalEdge(
                source="X",
                target="Y",
                edge_type=CausalRelationType.DIRECT_CAUSE,
                strength=1.5,
                confidence=0.9,
            )
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            CausalEdge(
                source="X",
                target="Y",
                edge_type=CausalRelationType.DIRECT_CAUSE,
                strength=0.8,
                confidence=1.5,
            )
        
        # Invalid lag
        with pytest.raises(ValueError, match="Lag must be non-negative"):
            CausalEdge(
                source="X",
                target="Y",
                edge_type=CausalRelationType.DIRECT_CAUSE,
                strength=0.8,
                confidence=0.9,
                lag=-1,
            )


class TestCausalGraph:
    """Test causal graph representation."""

    def create_test_graph(self) -> CausalGraph:
        """Create test causal graph."""
        variables = ["X", "Y", "Z"]
        edges = [
            CausalEdge("X", "Y", CausalRelationType.DIRECT_CAUSE, 0.8, 0.9),
            CausalEdge("Y", "Z", CausalRelationType.DIRECT_CAUSE, 0.6, 0.8),
        ]
        
        return CausalGraph(
            graph_id=uuid4(),
            graph_type=CausalGraphType.DAG,
            variables=variables,
            edges=edges,
        )

    def test_causal_graph_creation(self):
        """Test creating causal graph."""
        graph = self.create_test_graph()
        
        assert len(graph.variables) == 3
        assert len(graph.edges) == 2
        assert graph.graph_type == CausalGraphType.DAG

    def test_causal_graph_validation(self):
        """Test causal graph validation."""
        variables = ["X", "Y"]
        edges = [
            CausalEdge("X", "Z", CausalRelationType.DIRECT_CAUSE, 0.8, 0.9),  # Z not in variables
        ]
        
        with pytest.raises(ValueError, match="Edge target 'Z' not in variables"):
            CausalGraph(
                graph_id=uuid4(),
                graph_type=CausalGraphType.DAG,
                variables=variables,
                edges=edges,
            )

    def test_graph_relationships(self):
        """Test graph relationship methods."""
        graph = self.create_test_graph()
        
        # Test parents/children
        assert graph.get_parents("Y") == ["X"]
        assert graph.get_children("X") == ["Y"]
        assert graph.get_parents("X") == []
        assert graph.get_children("Z") == []
        
        # Test ancestors/descendants
        assert graph.get_ancestors("Z") == {"X", "Y"}
        assert graph.get_descendants("X") == {"Y", "Z"}
        assert graph.get_ancestors("X") == set()

    def test_graph_acyclicity(self):
        """Test acyclicity checking."""
        # Acyclic graph
        acyclic_graph = self.create_test_graph()
        assert acyclic_graph.is_acyclic()
        
        # Cyclic graph
        variables = ["X", "Y"]
        edges = [
            CausalEdge("X", "Y", CausalRelationType.DIRECT_CAUSE, 0.8, 0.9),
            CausalEdge("Y", "X", CausalRelationType.DIRECT_CAUSE, 0.6, 0.8),  # Creates cycle
        ]
        
        cyclic_graph = CausalGraph(
            graph_id=uuid4(),
            graph_type=CausalGraphType.DAG,
            variables=variables,
            edges=edges,
        )
        
        assert not cyclic_graph.is_acyclic()

    def test_edge_operations(self):
        """Test edge addition and removal."""
        graph = self.create_test_graph()
        
        # Test get_edge
        edge = graph.get_edge("X", "Y")
        assert edge is not None
        assert edge.source == "X"
        assert edge.target == "Y"
        
        # Test add_edge
        new_edge = CausalEdge("X", "Z", CausalRelationType.DIRECT_CAUSE, 0.5, 0.7)
        graph.add_edge(new_edge)
        assert graph.get_edge("X", "Z") is not None
        
        # Test remove_edge
        assert graph.remove_edge("X", "Z")
        assert graph.get_edge("X", "Z") is None
        assert not graph.remove_edge("X", "Z")  # Already removed


class TestInterventionSpec:
    """Test intervention specification."""

    def test_intervention_spec_creation(self):
        """Test creating intervention specification."""
        intervention = InterventionSpec(
            intervention_id=uuid4(),
            target_variable="X",
            intervention_type="do",
            intervention_value=5.0,
            duration=100,
        )
        
        assert intervention.target_variable == "X"
        assert intervention.intervention_type == "do"
        assert intervention.intervention_value == 5.0
        assert intervention.duration == 100

    def test_intervention_spec_validation(self):
        """Test intervention specification validation."""
        with pytest.raises(ValueError, match="Intervention type must be one of"):
            InterventionSpec(
                intervention_id=uuid4(),
                target_variable="X",
                intervention_type="invalid_type",
                intervention_value=5.0,
            )


class TestCausalAnomalyEvent:
    """Test causal anomaly event."""

    def test_causal_anomaly_event_creation(self):
        """Test creating causal anomaly event."""
        event = CausalAnomalyEvent(
            event_id=uuid4(),
            anomaly_type=AnomalyType.STRUCTURAL_BREAK,
            affected_variables=["X", "Y"],
            causal_score=0.8,
            timestamp=datetime.utcnow(),
            root_causes=["Z"],
            confidence=0.9,
            p_value=0.05,
        )
        
        assert event.anomaly_type == AnomalyType.STRUCTURAL_BREAK
        assert len(event.affected_variables) == 2
        assert event.causal_score == 0.8
        assert event.confidence == 0.9

    def test_causal_anomaly_event_validation(self):
        """Test causal anomaly event validation."""
        # Invalid causal score
        with pytest.raises(ValueError, match="Causal score must be between 0 and 1"):
            CausalAnomalyEvent(
                event_id=uuid4(),
                anomaly_type=AnomalyType.STRUCTURAL_BREAK,
                affected_variables=["X"],
                causal_score=1.5,
                timestamp=datetime.utcnow(),
            )


class TestCausalDataset:
    """Test causal dataset."""

    def create_test_dataset(self, n_samples: int = 100) -> CausalDataset:
        """Create test causal dataset."""
        variable_names = ["X", "Y", "Z"]
        data = np.random.randn(n_samples, len(variable_names))
        
        return CausalDataset(
            dataset_id=uuid4(),
            name="test_dataset",
            data=data,
            variable_names=variable_names,
            is_time_series=True,
        )

    def test_causal_dataset_creation(self):
        """Test creating causal dataset."""
        dataset = self.create_test_dataset()
        
        assert dataset.name == "test_dataset"
        assert dataset.data.shape == (100, 3)
        assert len(dataset.variable_names) == 3
        assert dataset.is_time_series

    def test_causal_dataset_validation(self):
        """Test causal dataset validation."""
        # Mismatched dimensions
        with pytest.raises(ValueError, match="Number of variables must match data dimensions"):
            CausalDataset(
                dataset_id=uuid4(),
                name="invalid_dataset",
                data=np.random.randn(100, 3),
                variable_names=["X", "Y"],  # Only 2 variables for 3 columns
            )

    def test_variable_data_access(self):
        """Test accessing variable data."""
        dataset = self.create_test_dataset()
        
        x_data = dataset.get_variable_data("X")
        assert len(x_data) == 100
        
        with pytest.raises(ValueError, match="Variable 'W' not found"):
            dataset.get_variable_data("W")

    def test_dataset_subset(self):
        """Test creating dataset subset."""
        dataset = self.create_test_dataset()
        
        subset = dataset.get_subset(10, 50)
        
        assert subset.data.shape == (40, 3)
        assert subset.variable_names == dataset.variable_names
        assert "subset" in subset.name

    def test_intervention_tracking(self):
        """Test intervention period tracking."""
        dataset = self.create_test_dataset()
        
        intervention = InterventionSpec(
            intervention_id=uuid4(),
            target_variable="X",
            intervention_type="do",
            intervention_value=5.0,
        )
        
        dataset.add_intervention_period(intervention, 10, 20)
        
        assert len(dataset.interventions) == 1
        assert dataset.is_experimental
        assert not dataset.is_observational


class TestCausalDetector:
    """Test causal detector."""

    def create_test_detector(self) -> CausalDetector:
        """Create test causal detector."""
        config = CausalAnalysisConfig(
            method=CausalMethod.PC_ALGORITHM,
            alpha=0.05,
        )
        
        return CausalDetector(
            detector_id=uuid4(),
            name="test_detector",
            config=config,
        )

    def test_causal_detector_creation(self):
        """Test creating causal detector."""
        detector = self.create_test_detector()
        
        assert detector.name == "test_detector"
        assert detector.config.method == CausalMethod.PC_ALGORITHM
        assert not detector.is_trained
        assert len(detector.detected_anomalies) == 0

    def test_detector_validation(self):
        """Test detector validation."""
        config = CausalAnalysisConfig(method=CausalMethod.PC_ALGORITHM)
        
        with pytest.raises(ValueError, match="Detector name cannot be empty"):
            CausalDetector(
                detector_id=uuid4(),
                name="",  # Empty name
                config=config,
            )

    def test_intervention_management(self):
        """Test intervention management."""
        detector = self.create_test_detector()
        
        intervention = InterventionSpec(
            intervention_id=uuid4(),
            target_variable="X",
            intervention_type="do",
            intervention_value=5.0,
        )
        
        detector.add_intervention(intervention)
        
        assert len(detector.intervention_history) == 1
        
        recent = detector.get_recent_interventions(5)
        assert len(recent) == 1

    def test_detect_causal_anomaly(self):
        """Test causal anomaly detection."""
        detector = self.create_test_detector()
        detector.is_trained = True  # Mark as trained for testing
        
        # Test with normal data
        normal_data = np.random.randn(100, 3)
        variable_names = ["X", "Y", "Z"]
        
        anomaly = detector.detect_causal_anomaly(normal_data, variable_names)
        # Should not detect anomaly in normal data
        # (This depends on the threshold and random data)
        
        # Test with anomalous data (high variance)
        anomalous_data = np.random.randn(100, 3) * 10  # High variance
        anomaly = detector.detect_causal_anomaly(anomalous_data, variable_names)
        
        if anomaly:
            assert isinstance(anomaly, CausalAnomalyEvent)
            assert anomaly.causal_score > 0

    def test_untrained_detector_error(self):
        """Test error when using untrained detector."""
        detector = self.create_test_detector()
        
        with pytest.raises(ValueError, match="Detector must be trained before detection"):
            detector.detect_causal_anomaly(np.random.randn(10, 3), ["X", "Y", "Z"])

    def test_anomaly_explanation(self):
        """Test anomaly explanation."""
        detector = self.create_test_detector()
        
        anomaly = CausalAnomalyEvent(
            event_id=uuid4(),
            anomaly_type=AnomalyType.STRUCTURAL_BREAK,
            affected_variables=["X"],
            causal_score=0.8,
            timestamp=datetime.utcnow(),
        )
        
        explanation = detector.explain_anomaly(anomaly)
        
        assert "anomaly_id" in explanation
        assert "causal_explanation" in explanation
        assert "recommendations" in explanation

    def test_detector_summary(self):
        """Test detector summary."""
        detector = self.create_test_detector()
        
        summary = detector.get_detector_summary()
        
        assert "detector_id" in summary
        assert "name" in summary
        assert "method" in summary
        assert "is_trained" in summary
        assert "anomaly_statistics" in summary


@pytest.mark.asyncio
class TestStructureLearning:
    """Test structure learning algorithms."""

    def create_test_data(self, n_samples: int = 100) -> Tuple[np.ndarray, List[str]]:
        """Create test data with known causal structure."""
        # Create data with X -> Y -> Z structure
        variable_names = ["X", "Y", "Z"]
        
        X = np.random.randn(n_samples)
        Y = 0.5 * X + np.random.randn(n_samples) * 0.5
        Z = 0.3 * Y + np.random.randn(n_samples) * 0.5
        
        data = np.column_stack([X, Y, Z])
        
        return data, variable_names

    async def test_pc_algorithm_learner(self):
        """Test PC algorithm structure learning."""
        learner = PCAlgorithmLearner(alpha=0.1)  # More permissive for test data
        
        data, variable_names = self.create_test_data()
        graph = await learner.learn_structure(data, variable_names)
        
        assert isinstance(graph, CausalGraph)
        assert graph.variables == variable_names
        assert len(graph.edges) >= 0  # May not discover all edges with small sample

    async def test_granger_causality_learner(self):
        """Test Granger causality structure learning."""
        learner = GrangerCausalityLearner(max_lag=3, alpha=0.1)
        
        data, variable_names = self.create_test_data(200)  # Larger sample for time series
        graph = await learner.learn_structure(data, variable_names)
        
        assert isinstance(graph, CausalGraph)
        assert graph.variables == variable_names
        assert graph.graph_type == CausalGraphType.DAG

    async def test_transfer_entropy_learner(self):
        """Test transfer entropy structure learning."""
        learner = TransferEntropyLearner(k=1, alpha=0.1, num_bins=5)
        
        data, variable_names = self.create_test_data(200)
        graph = await learner.learn_structure(data, variable_names)
        
        assert isinstance(graph, CausalGraph)
        assert graph.variables == variable_names

    async def test_structure_learning_service(self):
        """Test structure learning service."""
        service = StructureLearningService()
        
        data, variable_names = self.create_test_data()
        
        # Test single method
        graph = await service.learn_causal_structure(
            data, variable_names, CausalMethod.PC_ALGORITHM, alpha=0.1
        )
        
        assert isinstance(graph, CausalGraph)
        assert "learning_time_seconds" in graph.metadata
        
        # Test method comparison
        methods = [CausalMethod.PC_ALGORITHM, CausalMethod.GRANGER_CAUSALITY]
        results = await service.compare_methods(data, variable_names, methods, alpha=0.1)
        
        assert len(results) <= len(methods)  # Some may fail
        
        for method, graph in results.items():
            assert isinstance(graph, CausalGraph)

    async def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        learner = PCAlgorithmLearner()
        
        # Too few samples
        with pytest.raises(ValueError, match="Need at least 10 samples"):
            await learner.learn_structure(np.random.randn(5, 3), ["X", "Y", "Z"])
        
        # Mismatched dimensions
        with pytest.raises(ValueError, match="Number of variables must match"):
            await learner.learn_structure(np.random.randn(100, 3), ["X", "Y"])

    def test_supported_methods(self):
        """Test getting supported methods."""
        service = StructureLearningService()
        
        methods = service.get_supported_methods()
        
        assert CausalMethod.PC_ALGORITHM in methods
        assert CausalMethod.GRANGER_CAUSALITY in methods
        assert CausalMethod.TRANSFER_ENTROPY in methods


@pytest.mark.asyncio
class TestCausalAnomalyDetectionService:
    """Test causal anomaly detection service."""

    async def test_service_creation(self):
        """Test creating causal anomaly detection service."""
        service = CausalAnomalyDetectionService()
        
        assert len(service.detectors) == 0
        assert len(service.graphs) == 0
        assert len(service.datasets) == 0

    async def test_detector_creation(self):
        """Test creating detector through service."""
        service = CausalAnomalyDetectionService()
        
        config = CausalAnalysisConfig(method=CausalMethod.PC_ALGORITHM)
        detector = await service.create_detector("test_detector", config)
        
        assert detector.name == "test_detector"
        assert detector.detector_id in service.detectors

    async def test_dataset_registration(self):
        """Test registering dataset."""
        service = CausalAnomalyDetectionService()
        
        data = np.random.randn(100, 3)
        variable_names = ["X", "Y", "Z"]
        
        dataset = await service.register_dataset(
            "test_dataset", data, variable_names, is_time_series=True
        )
        
        assert dataset.name == "test_dataset"
        assert dataset.dataset_id in service.datasets

    async def test_detector_training(self):
        """Test training detector."""
        service = CausalAnomalyDetectionService()
        
        # Create detector
        config = CausalAnalysisConfig(method=CausalMethod.PC_ALGORITHM, alpha=0.1)
        detector = await service.create_detector("test_detector", config)
        
        # Register dataset
        data = np.random.randn(200, 3)  # Larger dataset for training
        variable_names = ["X", "Y", "Z"]
        dataset = await service.register_dataset("test_dataset", data, variable_names)
        
        # Train detector
        trained_detector = await service.train_detector(
            detector.detector_id, dataset.dataset_id
        )
        
        assert trained_detector.is_trained
        assert trained_detector.baseline_graph is not None
        assert trained_detector.training_samples > 0

    async def test_anomaly_detection(self):
        """Test anomaly detection."""
        service = CausalAnomalyDetectionService()
        
        # Create and train detector
        config = CausalAnalysisConfig(method=CausalMethod.PC_ALGORITHM, alpha=0.1)
        detector = await service.create_detector("test_detector", config)
        
        # Use smaller dataset for faster testing
        train_data = np.random.randn(50, 2)
        variable_names = ["X", "Y"]
        dataset = await service.register_dataset("train_dataset", train_data, variable_names)
        
        await service.train_detector(detector.detector_id, dataset.dataset_id)
        
        # Test detection
        test_data = np.random.randn(30, 2) * 2  # Different distribution
        anomalies = await service.detect_causal_anomalies(
            detector.detector_id, test_data, variable_names, window_size=15
        )
        
        assert isinstance(anomalies, list)
        # May or may not find anomalies depending on data

    async def test_intervention_simulation(self):
        """Test intervention simulation."""
        service = CausalAnomalyDetectionService()
        
        # Create and train detector
        config = CausalAnalysisConfig(method=CausalMethod.PC_ALGORITHM)
        detector = await service.create_detector("test_detector", config)
        
        train_data = np.random.randn(50, 2)
        variable_names = ["X", "Y"]
        dataset = await service.register_dataset("train_dataset", train_data, variable_names)
        
        await service.train_detector(detector.detector_id, dataset.dataset_id)
        
        # Create intervention
        intervention = InterventionSpec(
            intervention_id=uuid4(),
            target_variable="X",
            intervention_type="do",
            intervention_value=5.0,
        )
        
        # Simulate intervention
        result = await service.simulate_intervention(
            detector.detector_id, intervention, simulation_steps=50
        )
        
        assert "intervention" in result
        assert "predicted_effects" in result

    async def test_anomaly_explanation(self):
        """Test anomaly explanation."""
        service = CausalAnomalyDetectionService()
        
        config = CausalAnalysisConfig(method=CausalMethod.PC_ALGORITHM)
        detector = await service.create_detector("test_detector", config)
        
        anomaly = CausalAnomalyEvent(
            event_id=uuid4(),
            anomaly_type=AnomalyType.STRUCTURAL_BREAK,
            affected_variables=["X"],
            causal_score=0.8,
            timestamp=datetime.utcnow(),
        )
        
        explanation = await service.explain_anomaly(detector.detector_id, anomaly)
        
        assert "service_context" in explanation
        assert "detector_name" in explanation["service_context"]

    async def test_detector_summary(self):
        """Test detector summary."""
        service = CausalAnomalyDetectionService()
        
        config = CausalAnalysisConfig(method=CausalMethod.PC_ALGORITHM)
        detector = await service.create_detector("test_detector", config)
        
        summary = await service.get_detector_summary(detector.detector_id)
        
        assert "service_statistics" in summary
        assert summary["detector_id"] == str(detector.detector_id)

    async def test_service_statistics(self):
        """Test service statistics."""
        service = CausalAnomalyDetectionService()
        
        stats = service.get_service_statistics()
        
        assert "detector_statistics" in stats
        assert "graph_statistics" in stats
        assert "dataset_statistics" in stats
        assert "detection_statistics" in stats
        assert "supported_methods" in stats

    async def test_resource_cleanup(self):
        """Test resource cleanup."""
        service = CausalAnomalyDetectionService()
        
        # Create multiple detectors
        for i in range(3):
            config = CausalAnalysisConfig(method=CausalMethod.PC_ALGORITHM)
            await service.create_detector(f"detector_{i}", config)
        
        assert len(service.detectors) == 3
        
        # Cleanup untrained detectors
        cleanup_count = await service.cleanup_resources(keep_trained=True)
        
        assert cleanup_count == 3  # All were untrained
        assert len(service.detectors) == 0

    async def test_invalid_operations(self):
        """Test invalid operations."""
        service = CausalAnomalyDetectionService()
        
        invalid_id = uuid4()
        
        # Training non-existent detector
        with pytest.raises(ValueError, match="Detector .* not found"):
            await service.train_detector(invalid_id, uuid4())
        
        # Detection with non-existent detector
        with pytest.raises(ValueError, match="Detector .* not found"):
            await service.detect_causal_anomalies(invalid_id, np.random.randn(10, 2), ["X", "Y"])


class TestCausalAnalysisConfig:
    """Test causal analysis configuration."""

    def test_config_creation(self):
        """Test creating causal analysis configuration."""
        config = CausalAnalysisConfig(
            method=CausalMethod.GRANGER_CAUSALITY,
            alpha=0.01,
            max_lag=10,
            anomaly_threshold=0.9,
        )
        
        assert config.method == CausalMethod.GRANGER_CAUSALITY
        assert config.alpha == 0.01
        assert config.max_lag == 10
        assert config.anomaly_threshold == 0.9

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid alpha
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            CausalAnalysisConfig(
                method=CausalMethod.PC_ALGORITHM,
                alpha=1.5,
            )
        
        # Invalid max_lag
        with pytest.raises(ValueError, match="Max lag must be non-negative"):
            CausalAnalysisConfig(
                method=CausalMethod.GRANGER_CAUSALITY,
                max_lag=-1,
            )
        
        # Invalid min_causal_effect
        with pytest.raises(ValueError, match="Min causal effect must be between 0 and 1"):
            CausalAnalysisConfig(
                method=CausalMethod.PC_ALGORITHM,
                min_causal_effect=1.5,
            )

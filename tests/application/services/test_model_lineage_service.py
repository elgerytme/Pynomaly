"""
Tests for model lineage service.

This module provides comprehensive tests for the model lineage tracking service,
ensuring proper lineage tracking, dependency management, and impact analysis.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from src.pynomaly.application.services.model_lineage_service import (
    CircularDependencyError,
    LineageGraph,
    LineageTrackingError,
    ModelLineageService,
)
from src.pynomaly.domain.entities.dataset import Dataset
from src.pynomaly.domain.entities.model import Model


class TestModelLineageService:
    """Test cases for ModelLineageService."""

    @pytest.fixture
    def mock_model_repository(self):
        """Mock model repository."""
        repository = Mock()
        repository.get_by_id = AsyncMock()
        repository.save = AsyncMock()
        repository.list_all = AsyncMock()
        return repository

    @pytest.fixture
    def mock_dataset_repository(self):
        """Mock dataset repository."""
        repository = Mock()
        repository.get_by_id = AsyncMock()
        repository.save = AsyncMock()
        repository.list_all = AsyncMock()
        return repository

    @pytest.fixture
    def lineage_service(self, mock_model_repository, mock_dataset_repository):
        """Create ModelLineageService instance with mocked dependencies."""
        return ModelLineageService(
            model_repository=mock_model_repository,
            dataset_repository=mock_dataset_repository,
        )

    @pytest.fixture
    def sample_model(self):
        """Sample model for testing."""
        return Model(
            id="model_123",
            name="Test Model",
            algorithm="IsolationForest",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for testing."""
        return Dataset(
            id="dataset_456",
            name="Test Dataset",
            description="Test dataset for lineage tracking",
            created_at=datetime.now(UTC),
        )


class TestLineageTracking:
    """Test lineage tracking functionality."""

    @pytest.mark.asyncio
    async def test_add_model_dataset_relationship(
        self, lineage_service, sample_model, sample_dataset
    ):
        """Test adding model-dataset relationship."""
        # Execute
        await lineage_service.add_model_dataset_relationship(
            model_id=sample_model.id,
            dataset_id=sample_dataset.id,
            relationship_type="training",
        )

        # Verify
        relationships = await lineage_service.get_model_lineage(sample_model.id)
        assert len(relationships.edges) > 0

        # Check relationship exists
        edge_found = False
        for edge in relationships.edges:
            if (
                edge.source_id == sample_model.id
                and edge.target_id == sample_dataset.id
                and edge.relationship_type == "training"
            ):
                edge_found = True
                break

        assert edge_found is True

    @pytest.mark.asyncio
    async def test_add_model_model_relationship(self, lineage_service):
        """Test adding model-to-model relationship."""
        parent_model_id = "parent_model_123"
        child_model_id = "child_model_456"

        # Execute
        await lineage_service.add_model_relationship(
            parent_model_id=parent_model_id,
            child_model_id=child_model_id,
            relationship_type="ensemble_member",
        )

        # Verify
        lineage = await lineage_service.get_model_lineage(child_model_id)

        # Check parent relationship exists
        parent_found = False
        for edge in lineage.edges:
            if (
                edge.source_id == parent_model_id
                and edge.target_id == child_model_id
                and edge.relationship_type == "ensemble_member"
            ):
                parent_found = True
                break

        assert parent_found is True

    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, lineage_service):
        """Test detection of circular dependencies."""
        model_a = "model_a"
        model_b = "model_b"
        model_c = "model_c"

        # Create chain: A -> B -> C
        await lineage_service.add_model_relationship(model_a, model_b, "derived_from")
        await lineage_service.add_model_relationship(model_b, model_c, "derived_from")

        # Attempting to add C -> A should raise CircularDependencyError
        with pytest.raises(CircularDependencyError):
            await lineage_service.add_model_relationship(
                model_c, model_a, "derived_from"
            )

    @pytest.mark.asyncio
    async def test_remove_relationship(self, lineage_service):
        """Test removing lineage relationships."""
        model_id = "model_123"
        dataset_id = "dataset_456"

        # Add relationship
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id, dataset_id=dataset_id, relationship_type="training"
        )

        # Remove relationship
        await lineage_service.remove_relationship(
            source_id=model_id, target_id=dataset_id, relationship_type="training"
        )

        # Verify removal
        lineage = await lineage_service.get_model_lineage(model_id)

        # Check relationship no longer exists
        relationship_exists = False
        for edge in lineage.edges:
            if (
                edge.source_id == model_id
                and edge.target_id == dataset_id
                and edge.relationship_type == "training"
            ):
                relationship_exists = True
                break

        assert relationship_exists is False


class TestLineageQuery:
    """Test lineage query functionality."""

    @pytest.mark.asyncio
    async def test_get_model_lineage(self, lineage_service):
        """Test getting complete model lineage."""
        model_id = "model_123"
        dataset_id = "dataset_456"
        parent_model_id = "parent_model_789"

        # Add relationships
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id, dataset_id=dataset_id, relationship_type="training"
        )

        await lineage_service.add_model_relationship(
            parent_model_id=parent_model_id,
            child_model_id=model_id,
            relationship_type="derived_from",
        )

        # Execute
        lineage = await lineage_service.get_model_lineage(model_id)

        # Verify
        assert len(lineage.nodes) >= 3  # model, dataset, parent_model
        assert len(lineage.edges) >= 2  # two relationships

        # Check all expected nodes exist
        node_ids = {node.id for node in lineage.nodes}
        assert model_id in node_ids
        assert dataset_id in node_ids
        assert parent_model_id in node_ids

    @pytest.mark.asyncio
    async def test_get_upstream_dependencies(self, lineage_service):
        """Test getting upstream dependencies."""
        model_id = "model_123"
        dataset1_id = "dataset_456"
        dataset2_id = "dataset_789"
        parent_model_id = "parent_model_abc"

        # Create dependency chain
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id, dataset_id=dataset1_id, relationship_type="training"
        )

        await lineage_service.add_model_dataset_relationship(
            model_id=model_id, dataset_id=dataset2_id, relationship_type="validation"
        )

        await lineage_service.add_model_relationship(
            parent_model_id=parent_model_id,
            child_model_id=model_id,
            relationship_type="derived_from",
        )

        # Execute
        upstream = await lineage_service.get_upstream_dependencies(model_id)

        # Verify
        assert len(upstream) >= 3  # 2 datasets + 1 parent model

        upstream_ids = {dep.id for dep in upstream}
        assert dataset1_id in upstream_ids
        assert dataset2_id in upstream_ids
        assert parent_model_id in upstream_ids

    @pytest.mark.asyncio
    async def test_get_downstream_dependencies(self, lineage_service):
        """Test getting downstream dependencies."""
        model_id = "model_123"
        child_model1_id = "child_model_456"
        child_model2_id = "child_model_789"
        ensemble_model_id = "ensemble_model_abc"

        # Create downstream dependencies
        await lineage_service.add_model_relationship(
            parent_model_id=model_id,
            child_model_id=child_model1_id,
            relationship_type="derived_from",
        )

        await lineage_service.add_model_relationship(
            parent_model_id=model_id,
            child_model_id=child_model2_id,
            relationship_type="derived_from",
        )

        await lineage_service.add_model_relationship(
            parent_model_id=model_id,
            child_model_id=ensemble_model_id,
            relationship_type="ensemble_member",
        )

        # Execute
        downstream = await lineage_service.get_downstream_dependencies(model_id)

        # Verify
        assert len(downstream) >= 3

        downstream_ids = {dep.id for dep in downstream}
        assert child_model1_id in downstream_ids
        assert child_model2_id in downstream_ids
        assert ensemble_model_id in downstream_ids

    @pytest.mark.asyncio
    async def test_find_impact_analysis(self, lineage_service):
        """Test impact analysis for model changes."""
        # Create a complex dependency graph
        root_model = "root_model"
        dataset = "dataset_123"
        child_model1 = "child_model_1"
        child_model2 = "child_model_2"
        ensemble_model = "ensemble_model"

        # Build dependency chain
        await lineage_service.add_model_dataset_relationship(
            model_id=root_model, dataset_id=dataset, relationship_type="training"
        )

        await lineage_service.add_model_relationship(
            parent_model_id=root_model,
            child_model_id=child_model1,
            relationship_type="derived_from",
        )

        await lineage_service.add_model_relationship(
            parent_model_id=root_model,
            child_model_id=child_model2,
            relationship_type="derived_from",
        )

        await lineage_service.add_model_relationship(
            parent_model_id=child_model1,
            child_model_id=ensemble_model,
            relationship_type="ensemble_member",
        )

        await lineage_service.add_model_relationship(
            parent_model_id=child_model2,
            child_model_id=ensemble_model,
            relationship_type="ensemble_member",
        )

        # Execute impact analysis
        impact = await lineage_service.analyze_impact(root_model)

        # Verify
        assert len(impact.affected_models) >= 3  # child1, child2, ensemble
        assert len(impact.affected_datasets) == 0  # No datasets should be affected

        affected_model_ids = {model.id for model in impact.affected_models}
        assert child_model1 in affected_model_ids
        assert child_model2 in affected_model_ids
        assert ensemble_model in affected_model_ids


class TestLineageGraph:
    """Test lineage graph operations."""

    @pytest.mark.asyncio
    async def test_lineage_graph_creation(self, lineage_service):
        """Test creation of lineage graph."""
        model_id = "model_123"
        dataset_id = "dataset_456"

        # Add relationship
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id, dataset_id=dataset_id, relationship_type="training"
        )

        # Get lineage graph
        graph = await lineage_service.get_lineage_graph()

        # Verify graph structure
        assert isinstance(graph, LineageGraph)
        assert len(graph.nodes) >= 2
        assert len(graph.edges) >= 1

        # Check nodes exist
        node_ids = {node.id for node in graph.nodes}
        assert model_id in node_ids
        assert dataset_id in node_ids

    @pytest.mark.asyncio
    async def test_graph_traversal(self, lineage_service):
        """Test graph traversal functionality."""
        # Create a multi-level dependency chain
        models = ["model_a", "model_b", "model_c", "model_d"]

        # Create linear chain: A -> B -> C -> D
        for i in range(len(models) - 1):
            await lineage_service.add_model_relationship(
                parent_model_id=models[i],
                child_model_id=models[i + 1],
                relationship_type="derived_from",
            )

        # Test depth-first traversal
        visited = await lineage_service._traverse_graph(
            start_node=models[0], direction="downstream"
        )

        # Should visit all models in the chain
        assert len(visited) == len(models)
        for model_id in models:
            assert model_id in visited

    @pytest.mark.asyncio
    async def test_lineage_graph_export(self, lineage_service):
        """Test exporting lineage graph."""
        model_id = "model_123"
        dataset_id = "dataset_456"

        # Add relationship
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id, dataset_id=dataset_id, relationship_type="training"
        )

        # Export graph
        graph_dict = await lineage_service.export_lineage_graph(format="dict")

        # Verify export structure
        assert "nodes" in graph_dict
        assert "edges" in graph_dict
        assert len(graph_dict["nodes"]) >= 2
        assert len(graph_dict["edges"]) >= 1

        # Check node and edge structure
        node = graph_dict["nodes"][0]
        assert "id" in node
        assert "type" in node
        assert "metadata" in node

        edge = graph_dict["edges"][0]
        assert "source_id" in edge
        assert "target_id" in edge
        assert "relationship_type" in edge


class TestLineageMetadata:
    """Test lineage metadata functionality."""

    @pytest.mark.asyncio
    async def test_add_relationship_metadata(self, lineage_service):
        """Test adding metadata to relationships."""
        model_id = "model_123"
        dataset_id = "dataset_456"
        metadata = {
            "training_accuracy": 0.95,
            "training_date": "2025-06-26",
            "data_version": "v1.2",
        }

        # Add relationship with metadata
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id,
            dataset_id=dataset_id,
            relationship_type="training",
            metadata=metadata,
        )

        # Verify metadata is stored
        lineage = await lineage_service.get_model_lineage(model_id)

        relationship_edge = None
        for edge in lineage.edges:
            if edge.source_id == model_id and edge.target_id == dataset_id:
                relationship_edge = edge
                break

        assert relationship_edge is not None
        assert relationship_edge.metadata == metadata

    @pytest.mark.asyncio
    async def test_update_relationship_metadata(self, lineage_service):
        """Test updating relationship metadata."""
        model_id = "model_123"
        dataset_id = "dataset_456"

        initial_metadata = {"version": "1.0"}
        updated_metadata = {"version": "1.1", "performance": "improved"}

        # Add relationship with initial metadata
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id,
            dataset_id=dataset_id,
            relationship_type="training",
            metadata=initial_metadata,
        )

        # Update metadata
        await lineage_service.update_relationship_metadata(
            source_id=model_id,
            target_id=dataset_id,
            relationship_type="training",
            metadata=updated_metadata,
        )

        # Verify update
        lineage = await lineage_service.get_model_lineage(model_id)

        relationship_edge = None
        for edge in lineage.edges:
            if edge.source_id == model_id and edge.target_id == dataset_id:
                relationship_edge = edge
                break

        assert relationship_edge is not None
        assert relationship_edge.metadata == updated_metadata


class TestLineageSearch:
    """Test lineage search functionality."""

    @pytest.mark.asyncio
    async def test_search_by_relationship_type(self, lineage_service):
        """Test searching relationships by type."""
        model_id = "model_123"
        training_dataset = "training_dataset"
        validation_dataset = "validation_dataset"
        test_dataset = "test_dataset"

        # Add different relationship types
        await lineage_service.add_model_dataset_relationship(
            model_id, training_dataset, "training"
        )
        await lineage_service.add_model_dataset_relationship(
            model_id, validation_dataset, "validation"
        )
        await lineage_service.add_model_dataset_relationship(
            model_id, test_dataset, "testing"
        )

        # Search for training relationships
        training_relationships = await lineage_service.find_relationships_by_type(
            "training"
        )

        # Verify search results
        assert len(training_relationships) >= 1

        training_found = False
        for edge in training_relationships:
            if (
                edge.source_id == model_id
                and edge.target_id == training_dataset
                and edge.relationship_type == "training"
            ):
                training_found = True
                break

        assert training_found is True

    @pytest.mark.asyncio
    async def test_search_by_node_type(self, lineage_service):
        """Test searching by node type."""
        model_id = "model_123"
        dataset_id = "dataset_456"

        # Add relationship
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id, dataset_id=dataset_id, relationship_type="training"
        )

        # Search for model nodes
        model_nodes = await lineage_service.find_nodes_by_type("model")

        # Verify search results
        assert len(model_nodes) >= 1

        model_found = False
        for node in model_nodes:
            if node.id == model_id and node.type == "model":
                model_found = True
                break

        assert model_found is True

    @pytest.mark.asyncio
    async def test_search_by_metadata(self, lineage_service):
        """Test searching by metadata criteria."""
        model_id = "model_123"
        dataset_id = "dataset_456"
        metadata = {"accuracy": 0.95, "algorithm": "RandomForest"}

        # Add relationship with metadata
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id,
            dataset_id=dataset_id,
            relationship_type="training",
            metadata=metadata,
        )

        # Search by metadata criteria
        high_accuracy_relationships = (
            await lineage_service.find_relationships_by_metadata(
                {"accuracy": {"$gte": 0.9}}
            )
        )

        # Verify search results
        assert len(high_accuracy_relationships) >= 1

        relationship_found = False
        for edge in high_accuracy_relationships:
            if (
                edge.source_id == model_id
                and edge.target_id == dataset_id
                and edge.metadata.get("accuracy", 0) >= 0.9
            ):
                relationship_found = True
                break

        assert relationship_found is True


class TestLineagePerformance:
    """Test lineage performance and scalability."""

    @pytest.mark.asyncio
    async def test_large_graph_performance(self, lineage_service):
        """Test performance with large lineage graphs."""
        # Create a large number of relationships
        num_models = 50
        num_datasets = 20

        models = [f"model_{i}" for i in range(num_models)]
        datasets = [f"dataset_{i}" for i in range(num_datasets)]

        # Add relationships
        for i, model_id in enumerate(models):
            dataset_idx = i % num_datasets
            await lineage_service.add_model_dataset_relationship(
                model_id=model_id,
                dataset_id=datasets[dataset_idx],
                relationship_type="training",
            )

        # Test query performance
        import time

        start_time = time.time()

        lineage = await lineage_service.get_lineage_graph()

        end_time = time.time()
        query_time = end_time - start_time

        # Should complete within reasonable time (< 1 second)
        assert query_time < 1.0
        assert len(lineage.nodes) == num_models + num_datasets
        assert len(lineage.edges) == num_models

    @pytest.mark.asyncio
    async def test_deep_dependency_chain_performance(self, lineage_service):
        """Test performance with deep dependency chains."""
        # Create a deep chain of model dependencies
        chain_length = 20
        models = [f"model_{i}" for i in range(chain_length)]

        # Create linear chain
        for i in range(chain_length - 1):
            await lineage_service.add_model_relationship(
                parent_model_id=models[i],
                child_model_id=models[i + 1],
                relationship_type="derived_from",
            )

        # Test traversal performance
        import time

        start_time = time.time()

        downstream = await lineage_service.get_downstream_dependencies(models[0])

        end_time = time.time()
        traversal_time = end_time - start_time

        # Should complete within reasonable time
        assert traversal_time < 0.5
        assert len(downstream) == chain_length - 1  # All models except the root


class TestErrorHandling:
    """Test error handling in lineage service."""

    @pytest.mark.asyncio
    async def test_invalid_model_id_error(self, lineage_service):
        """Test handling of invalid model IDs."""
        with pytest.raises(LineageTrackingError):
            await lineage_service.get_model_lineage("nonexistent_model")

    @pytest.mark.asyncio
    async def test_duplicate_relationship_handling(self, lineage_service):
        """Test handling of duplicate relationships."""
        model_id = "model_123"
        dataset_id = "dataset_456"

        # Add relationship
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id, dataset_id=dataset_id, relationship_type="training"
        )

        # Adding same relationship should not create duplicate
        await lineage_service.add_model_dataset_relationship(
            model_id=model_id, dataset_id=dataset_id, relationship_type="training"
        )

        # Verify only one relationship exists
        lineage = await lineage_service.get_model_lineage(model_id)

        training_relationships = [
            edge
            for edge in lineage.edges
            if (
                edge.source_id == model_id
                and edge.target_id == dataset_id
                and edge.relationship_type == "training"
            )
        ]

        assert len(training_relationships) == 1

    @pytest.mark.asyncio
    async def test_self_reference_prevention(self, lineage_service):
        """Test prevention of self-referencing relationships."""
        model_id = "model_123"

        # Attempting to create self-reference should raise error
        with pytest.raises(LineageTrackingError, match="self-reference"):
            await lineage_service.add_model_relationship(
                parent_model_id=model_id,
                child_model_id=model_id,
                relationship_type="derived_from",
            )

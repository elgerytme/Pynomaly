"""Model lineage tracking service."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

from pynomaly.domain.entities.lineage_record import (
    LineageEdge,
    LineageGraph,
    LineageNode,
    LineageQuery,
    LineageRecord,
    LineageRelationType,
    LineageStatistics,
    LineageTransformation,
    TransformationType,
)
from pynomaly.infrastructure.repositories.protocols import (
    ModelRepositoryProtocol,
    ModelVersionRepositoryProtocol,
)


class ModelLineageService:
    """Service for tracking and analyzing model lineage."""

    def __init__(
        self,
        model_repository: ModelRepositoryProtocol,
        model_version_repository: ModelVersionRepositoryProtocol,
        lineage_repository: Any,  # LineageRepositoryProtocol when implemented
    ):
        """Initialize the lineage service.
        
        Args:
            model_repository: Model repository
            model_version_repository: Model version repository
            lineage_repository: Lineage repository
        """
        self.model_repository = model_repository
        self.model_version_repository = model_version_repository
        self.lineage_repository = lineage_repository

    async def create_lineage_record(
        self,
        child_model_id: UUID,
        parent_model_ids: list[UUID],
        relation_type: LineageRelationType,
        transformation: LineageTransformation,
        created_by: str,
        experiment_id: UUID | None = None,
        run_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageRecord:
        """Create a new lineage record.
        
        Args:
            child_model_id: Child model identifier
            parent_model_ids: Parent model identifiers
            relation_type: Type of relationship
            transformation: Transformation details
            created_by: User creating the record
            experiment_id: Associated experiment ID
            run_id: Associated run ID
            tags: Record tags
            metadata: Additional metadata
            
        Returns:
            Created lineage record
        """
        # Validate models exist
        await self._validate_models_exist(parent_model_ids + [child_model_id])
        
        # Create lineage record
        record = LineageRecord(
            child_model_id=child_model_id,
            parent_model_ids=parent_model_ids,
            relation_type=relation_type,
            transformation=transformation,
            created_by=created_by,
            experiment_id=experiment_id,
            run_id=run_id,
            tags=tags or [],
            metadata=metadata or {},
        )
        
        # Store in repository
        stored_record = await self.lineage_repository.create(record)
        
        return stored_record

    async def track_model_derivation(
        self,
        parent_model_id: UUID,
        child_model_id: UUID,
        transformation_type: TransformationType,
        transformation_metadata: dict[str, Any],
        created_by: str,
        algorithm: str | None = None,
        tool: str | None = None,
        execution_time: float | None = None,
        resource_usage: dict[str, Any] | None = None,
    ) -> LineageRecord:
        """Track a simple parent-child model derivation.
        
        Args:
            parent_model_id: Parent model ID
            child_model_id: Child model ID
            transformation_type: Type of transformation
            transformation_metadata: Transformation parameters
            created_by: User creating the record
            algorithm: Algorithm used
            tool: Tool or framework used
            execution_time: Execution time in seconds
            resource_usage: Resource usage information
            
        Returns:
            Created lineage record
        """
        # Determine relation type from transformation type
        relation_type_mapping = {
            TransformationType.FINE_TUNING: LineageRelationType.FINE_TUNED_FROM,
            TransformationType.TRANSFER_LEARNING: LineageRelationType.DERIVED_FROM,
            TransformationType.DISTILLATION: LineageRelationType.DISTILLED_FROM,
            TransformationType.PRUNING: LineageRelationType.PRUNED_FROM,
            TransformationType.QUANTIZATION: LineageRelationType.QUANTIZED_FROM,
            TransformationType.FEDERATED_LEARNING: LineageRelationType.FEDERATED_FROM,
            TransformationType.MODEL_MERGING: LineageRelationType.MERGED_FROM,
        }
        
        relation_type = relation_type_mapping.get(
            transformation_type, LineageRelationType.DERIVED_FROM
        )
        
        transformation = LineageTransformation(
            type=transformation_type,
            parameters=transformation_metadata,
            algorithm=algorithm,
            tool=tool,
            execution_time=execution_time,
            resource_usage=resource_usage or {},
        )
        
        return await self.create_lineage_record(
            child_model_id=child_model_id,
            parent_model_ids=[parent_model_id],
            relation_type=relation_type,
            transformation=transformation,
            created_by=created_by,
        )

    async def track_ensemble_creation(
        self,
        ensemble_model_id: UUID,
        component_model_ids: list[UUID],
        ensemble_metadata: dict[str, Any],
        created_by: str,
        algorithm: str = "ensemble",
        tool: str | None = None,
    ) -> LineageRecord:
        """Track ensemble model creation.
        
        Args:
            ensemble_model_id: Ensemble model ID
            component_model_ids: Component model IDs
            ensemble_metadata: Ensemble configuration
            created_by: User creating the record
            algorithm: Ensemble algorithm
            tool: Tool used for ensemble
            
        Returns:
            Created lineage record
        """
        transformation = LineageTransformation(
            type=TransformationType.ENSEMBLE,
            parameters=ensemble_metadata,
            algorithm=algorithm,
            tool=tool,
        )
        
        return await self.create_lineage_record(
            child_model_id=ensemble_model_id,
            parent_model_ids=component_model_ids,
            relation_type=LineageRelationType.ENSEMBLE_OF,
            transformation=transformation,
            created_by=created_by,
        )

    async def get_model_lineage(
        self,
        model_id: UUID,
        include_ancestors: bool = True,
        include_descendants: bool = True,
        max_depth: int = 10,
    ) -> LineageGraph:
        """Get complete lineage graph for a model.
        
        Args:
            model_id: Target model ID
            include_ancestors: Include ancestor models
            include_descendants: Include descendant models
            max_depth: Maximum depth to traverse
            
        Returns:
            Complete lineage graph
        """
        # Get all lineage records related to the model
        query = LineageQuery(
            model_id=model_id,
            include_ancestors=include_ancestors,
            include_descendants=include_descendants,
            max_depth=max_depth,
        )
        
        lineage_records = await self.lineage_repository.query_lineage(query)
        
        # Build nodes and edges
        nodes = {}
        edges = []
        
        # Collect all model IDs
        all_model_ids = set()
        for record in lineage_records:
            all_model_ids.update(record.get_all_model_ids())
        
        # Create nodes
        for model_id in all_model_ids:
            model = await self.model_repository.get_by_id(model_id)
            if model:
                nodes[model_id] = LineageNode(
                    model_id=model_id,
                    model_name=model.name,
                    model_version=model.current_version or "unknown",
                    created_at=model.created_at,
                    metadata=model.metadata,
                )
        
        # Create edges
        for record in lineage_records:
            for parent_id in record.parent_model_ids:
                edges.append(
                    LineageEdge(
                        parent_id=parent_id,
                        child_id=record.child_model_id,
                        relation_type=record.relation_type,
                        transformation=record.transformation,
                        created_at=record.created_at,
                        metadata=record.metadata,
                    )
                )
        
        # Calculate depth
        depth = self._calculate_graph_depth(model_id, edges)
        
        return LineageGraph(
            root_model_id=model_id,
            nodes=nodes,
            edges=edges,
            depth=depth,
        )

    async def get_model_ancestors(
        self, model_id: UUID, max_depth: int = 10
    ) -> list[UUID]:
        """Get all ancestor models.
        
        Args:
            model_id: Target model ID
            max_depth: Maximum depth to traverse
            
        Returns:
            List of ancestor model IDs
        """
        lineage_graph = await self.get_model_lineage(
            model_id, include_ancestors=True, include_descendants=False, max_depth=max_depth
        )
        return list(lineage_graph.get_ancestors(model_id))

    async def get_model_descendants(
        self, model_id: UUID, max_depth: int = 10
    ) -> list[UUID]:
        """Get all descendant models.
        
        Args:
            model_id: Target model ID
            max_depth: Maximum depth to traverse
            
        Returns:
            List of descendant model IDs
        """
        lineage_graph = await self.get_model_lineage(
            model_id, include_ancestors=False, include_descendants=True, max_depth=max_depth
        )
        return list(lineage_graph.get_descendants(model_id))

    async def find_common_ancestor(
        self, model_id1: UUID, model_id2: UUID
    ) -> UUID | None:
        """Find common ancestor of two models.
        
        Args:
            model_id1: First model ID
            model_id2: Second model ID
            
        Returns:
            Common ancestor model ID or None
        """
        ancestors1 = set(await self.get_model_ancestors(model_id1))
        ancestors2 = set(await self.get_model_ancestors(model_id2))
        
        common_ancestors = ancestors1.intersection(ancestors2)
        
        if not common_ancestors:
            return None
        
        # Return the most recent common ancestor
        # This would need additional logic to determine recency
        return next(iter(common_ancestors))

    async def get_lineage_path(
        self, from_model_id: UUID, to_model_id: UUID
    ) -> list[UUID] | None:
        """Find lineage path between two models.
        
        Args:
            from_model_id: Source model ID
            to_model_id: Target model ID
            
        Returns:
            Path as list of model IDs or None if no path exists
        """
        # Get lineage graph that includes both models
        lineage_graph = await self.get_model_lineage(from_model_id, max_depth=20)
        
        # Check if target model is in the graph
        if to_model_id not in lineage_graph.nodes:
            # Try from the other direction
            lineage_graph = await self.get_model_lineage(to_model_id, max_depth=20)
            if from_model_id not in lineage_graph.nodes:
                return None
        
        return lineage_graph.find_path(from_model_id, to_model_id)

    async def get_lineage_statistics(self) -> LineageStatistics:
        """Get overall lineage statistics.
        
        Returns:
            Lineage statistics
        """
        # This would be implemented based on the lineage repository
        # For now, return a placeholder implementation
        total_records = await self.lineage_repository.count()
        
        return LineageStatistics(
            total_models=await self.model_repository.count(),
            total_relationships=total_records,
            max_depth=0,  # Would be calculated
            avg_branching_factor=0.0,  # Would be calculated
            relation_type_counts={},  # Would be calculated
            transformation_type_counts={},  # Would be calculated
            orphaned_models=0,  # Would be calculated
            most_derived_model=None,  # Would be calculated
            oldest_lineage=None,  # Would be calculated
            newest_lineage=None,  # Would be calculated
        )

    async def query_lineage(self, query: LineageQuery) -> list[LineageRecord]:
        """Query lineage records with filters.
        
        Args:
            query: Lineage query with filters
            
        Returns:
            Matching lineage records
        """
        return await self.lineage_repository.query_lineage(query)

    async def delete_lineage_record(self, record_id: UUID) -> bool:
        """Delete a lineage record.
        
        Args:
            record_id: Record ID to delete
            
        Returns:
            Success status
        """
        return await self.lineage_repository.delete(record_id)

    async def bulk_import_lineage(
        self, lineage_records: list[LineageRecord]
    ) -> list[LineageRecord]:
        """Bulk import lineage records.
        
        Args:
            lineage_records: Records to import
            
        Returns:
            Imported records
        """
        # Validate all models exist
        all_model_ids = set()
        for record in lineage_records:
            all_model_ids.update(record.get_all_model_ids())
        
        await self._validate_models_exist(list(all_model_ids))
        
        # Import records
        imported_records = []
        for record in lineage_records:
            imported_record = await self.lineage_repository.create(record)
            imported_records.append(imported_record)
        
        return imported_records

    async def _validate_models_exist(self, model_ids: list[UUID]) -> None:
        """Validate that all models exist.
        
        Args:
            model_ids: Model IDs to validate
            
        Raises:
            ValueError: If any model doesn't exist
        """
        # Check models in parallel
        validation_tasks = [
            self.model_repository.get_by_id(model_id) for model_id in model_ids
        ]
        
        models = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        for i, result in enumerate(models):
            if isinstance(result, Exception) or result is None:
                raise ValueError(f"Model {model_ids[i]} does not exist")

    def _calculate_graph_depth(self, root_id: UUID, edges: list[LineageEdge]) -> int:
        """Calculate the maximum depth of a lineage graph.
        
        Args:
            root_id: Root model ID
            edges: Graph edges
            
        Returns:
            Maximum depth
        """
        # Build adjacency list
        children = {}
        for edge in edges:
            if edge.parent_id not in children:
                children[edge.parent_id] = []
            children[edge.parent_id].append(edge.child_id)
        
        # Calculate depth using DFS
        def dfs(node_id: UUID, current_depth: int) -> int:
            max_depth = current_depth
            for child_id in children.get(node_id, []):
                child_depth = dfs(child_id, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            return max_depth
        
        return dfs(root_id, 0)
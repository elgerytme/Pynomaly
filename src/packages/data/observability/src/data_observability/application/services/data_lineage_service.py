"""
Data Lineage Service

Provides application-level services for managing data lineage tracking,
including lineage creation, updates, queries, and impact analysis.
"""

from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from uuid import UUID

from ...domain.entities.data_lineage import (
    DataLineage,
    LineageNode,
    LineageEdge,
    LineageNodeType,
    LineageRelationType,
    LineageMetadata
)


from ...domain.repositories.data_lineage_repository import DataLineageRepository
from ...infrastructure.errors.exceptions import LineageError


class DataLineageService:
    """Service for managing data lineage operations."""
    
    def __init__(self, repository: DataLineageRepository):
        self._repository = repository
    
    async def create_lineage(self, name: str, description: str = None, namespace: str = "default") -> DataLineage:
        """Create a new data lineage graph."""
        lineage = DataLineage(
            name=name,
            description=description,
            namespace=namespace
        )
        
        return await self._repository.save_lineage(lineage)
    
    async def get_lineage(self, lineage_id: UUID) -> Optional[DataLineage]:
        """Get a lineage by ID."""
        return await self._repository.get_lineage_by_id(lineage_id)
    
    async def get_lineage_by_name(self, name: str, namespace: str = "default") -> Optional[DataLineage]:
        """Get a lineage by name and namespace."""
        return await self._repository.get_lineage_by_name(name, namespace)
    
    async def list_lineages(self, namespace: str = None) -> List[DataLineage]:
        """List all lineages, optionally filtered by namespace."""
        return await self._repository.list_lineages(namespace)
    
    async def add_node(self, lineage_id: UUID, node: LineageNode) -> None:
        """Add a node to a lineage."""
        await self._repository.add_node(lineage_id, node)
    
    async def add_edge(self, lineage_id: UUID, edge: LineageEdge) -> None:
        """Add an edge to a lineage."""
        await self._repository.add_edge(lineage_id, edge)
    
    def create_node(
        self,
        name: str,
        node_type: LineageNodeType,
        namespace: str,
        description: str = None,
        schema: Dict[str, Any] = None,
        location: str = None,
        owner: str = None,
        **kwargs
    ) -> LineageNode:
        """Create a new lineage node."""
        metadata = LineageMetadata(created_by=owner)
        if asset_id:
            metadata.set_property("asset_id", asset_id)
        
        node = LineageNode(
            name=name,
            type=node_type,
            namespace=namespace,
            description=description,
            asset_id=asset_id,
            metadata=metadata,
            **kwargs
        )
        
        return node
    
    def create_edge(
        self,
        source_node_id: UUID,
        target_node_id: UUID,
        relationship_type: LineageRelationType,
        transform_logic: str = None,
        column_mapping: Dict[str, str] = None,
        pipeline_id: UUID = None,
        **kwargs
    ) -> LineageEdge:
        """Create a new lineage edge."""
        edge = LineageEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship_type,
            transform_logic=transform_logic,
            column_mapping=column_mapping,
            pipeline_id=pipeline_id,
            **kwargs
        )
        
        return edge
    
    async def find_nodes_by_name(self, name: str) -> List[LineageNode]:
        """Find all nodes with a given name."""
        return await self._repository.find_nodes_by_name(name)
    
    async def find_nodes_by_type(self, node_type: LineageNodeType) -> List[LineageNode]:
        """Find all nodes of a given type."""
        return await self._repository.find_nodes_by_type(node_type.value)
    
    async def get_impact_analysis(self, lineage_id: UUID, node_id: UUID) -> Dict[str, Any]:
        """Get comprehensive impact analysis for a node."""
        lineage = await self._repository.get_lineage_by_id(lineage_id)
        if not lineage:
            raise LineageError(f"Lineage {lineage_id} not found")
        
        node = await self._repository.get_node_by_id(node_id)
        if not node:
            raise LineageError(f"Node {node_id} not found in lineage")
        
        # These methods (get_upstream_nodes, get_downstream_nodes, get_critical_path, validate_lineage) are on the DataLineage entity
        # They operate on the in-memory graph representation. If the graph is large, this might be inefficient.
        # For now, we assume the DataLineage object returned by the repository is complete enough for these operations.
        upstream_nodes = lineage.get_upstream_nodes(node_id)
        downstream_nodes = lineage.get_downstream_nodes(node_id)
        
        # Calculate impact metrics
        impact_metrics = {
            "upstream_count": len(upstream_nodes),
            "downstream_count": len(downstream_nodes),
            "total_dependencies": len(upstream_nodes) + len(downstream_nodes),
            "criticality_score": self._calculate_criticality_score(lineage, node_id),
            "quality_impact": self._calculate_quality_impact(lineage, node_id)
        }
        
        return {
            "node": node,
            "upstream_nodes": upstream_nodes,
            "downstream_nodes": downstream_nodes,
            "impact_metrics": impact_metrics,
            "recommendations": self._generate_impact_recommendations(lineage, node_id)
        }
    
    async def get_data_flow_path(self, lineage_id: UUID, source_node_id: UUID, target_node_id: UUID) -> Dict[str, Any]:
        """Get the data flow path between two nodes."""
        lineage = await self._repository.get_lineage_by_id(lineage_id)
        if not lineage:
            raise LineageError(f"Lineage {lineage_id} not found")
        
        path_nodes = lineage.get_path(source_node_id, target_node_id)
        
        if not path_nodes:
            return {
                "path_exists": False,
                "path_nodes": [],
                "path_edges": [],
                "path_metrics": {}
            }
        
        # Get edges in the path
        path_edges = []
        for i in range(len(path_nodes) - 1):
            current_node = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            # Find the edge between these nodes
            for edge in lineage.edges.values(): # This assumes edges are loaded with the lineage
                if edge.source_node_id == current_node.id and edge.target_node_id == next_node.id:
                    path_edges.append(edge)
                    break
        
        # Calculate path metrics
        path_metrics = self._calculate_path_metrics(path_nodes, path_edges)
        
        return {
            "path_exists": True,
            "path_nodes": path_nodes,
            "path_edges": path_edges,
            "path_metrics": path_metrics
        }
    
    async def update_node_quality(self, lineage_id: UUID, node_id: UUID, quality_score: float) -> None:
        """Update the quality score for a node."""
        node = await self._repository.get_node_by_id(node_id)
        if not node:
            raise LineageError(f"Node {node_id} not found")
        
        node.update_quality_score(quality_score)
        await self._repository.update_node(node)
        
        # Propagate quality impact to downstream nodes
        lineage = await self._repository.get_lineage_by_id(lineage_id)
        if lineage:
            self._propagate_quality_impact(lineage, node_id, quality_score)
    
    async def track_data_transformation(
        self,
        lineage_id: UUID,
        source_node_id: UUID,
        target_node_id: UUID,
        transform_logic: str,
        column_mapping: Dict[str, str] = None,
        execution_time: float = None,
        error_rate: float = None
    ) -> None:
        """Track a data transformation between two nodes."""
        lineage = await self._repository.get_lineage_by_id(lineage_id)
        if not lineage:
            raise LineageError(f"Lineage {lineage_id} not found")
        
        # Find existing edge or create new one
        edge = None
        for e in lineage.edges.values(): # This assumes edges are loaded with the lineage
            if e.source_node_id == source_node_id and e.target_node_id == target_node_id:
                edge = e
                break
        
        if not edge:
            edge = self.create_edge(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                relationship_type=LineageRelationType.TRANSFORMS_TO
            )
            await self._repository.add_edge(lineage_id, edge)
        
        # Update edge with transformation details
        edge.set_transform_logic(transform_logic)
        if column_mapping:
            edge.set_column_mapping(column_mapping)
        if execution_time is not None:
            edge.update_execution_stats(execution_time, error_rate)
        
        await self._repository.update_edge(edge)

    async def get_lineage_health_report(self, lineage_id: UUID) -> Dict[str, Any]:
        """Generate a health report for a lineage."""
        lineage = await self._repository.get_lineage_by_id(lineage_id)
        if not lineage:
            raise LineageError(f"Lineage {lineage_id} not found")
        
        validation_errors = lineage.validate_lineage()
        
        # Calculate health metrics
        total_nodes = len(lineage.nodes)
        total_edges = len(lineage.edges)
        
        nodes_with_quality = sum(1 for node in lineage.nodes.values() if node.quality_score is not None)
        avg_quality_score = None
        if nodes_with_quality > 0:
            avg_quality_score = sum(
                node.quality_score for node in lineage.nodes.values()
                if node.quality_score is not None
            ) / nodes_with_quality
        
        critical_nodes = [node for node in lineage.get_critical_path()[:5]]  # Top 5 critical nodes
        
        stale_nodes = [
            node for node in lineage.nodes.values()
            if node.last_updated and (datetime.utcnow() - node.last_updated).days > 7
        ]
        
        return {
            "lineage_id": lineage_id,
            "lineage_name": lineage.name,
            "validation_errors": validation_errors,
            "health_metrics": {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "nodes_with_quality": nodes_with_quality,
                "avg_quality_score": avg_quality_score,
                "critical_nodes_count": len(critical_nodes),
                "stale_nodes_count": len(stale_nodes)
            },
            "critical_nodes": critical_nodes,
            "stale_nodes": stale_nodes,
            "recommendations": self._generate_health_recommendations(lineage)
        }
    
    def _calculate_criticality_score(self, lineage: DataLineage, node_id: UUID) -> float:
        """Calculate criticality score for a node."""
        downstream_count = len(lineage.get_downstream_nodes(node_id))
        upstream_count = len(lineage.get_upstream_nodes(node_id))
        
        node = lineage.get_node(node_id)
        quality_factor = node.quality_score or 1.0
        
        # Criticality based on dependencies and quality
        criticality = (downstream_count * 2 + upstream_count) * quality_factor
        
        # Normalize to 0-1 range
        max_possible = len(lineage.nodes) * 2
        return min(criticality / max_possible, 1.0) if max_possible > 0 else 0.0
    
    def _calculate_quality_impact(self, lineage: DataLineage, node_id: UUID) -> float:
        """Calculate quality impact score for a node."""
        node = lineage.get_node(node_id)
        if not node or node.quality_score is None:
            return 0.0
        
        downstream_nodes = lineage.get_downstream_nodes(node_id)
        if not downstream_nodes:
            return 0.0
        
        # Quality impact is the inverse of quality score times downstream count
        quality_impact = (1.0 - node.quality_score) * len(downstream_nodes)
        
        # Normalize to 0-1 range
        max_possible = len(lineage.nodes)
        return min(quality_impact / max_possible, 1.0) if max_possible > 0 else 0.0
    
    def _calculate_path_metrics(self, path_nodes: List[LineageNode], path_edges: List[LineageEdge]) -> Dict[str, Any]:
        """Calculate metrics for a data flow path."""
        metrics = {
            "path_length": len(path_nodes),
            "transformation_count": len(path_edges),
            "avg_quality_score": None,
            "min_quality_score": None,
            "total_execution_time": None,
            "avg_error_rate": None
        }
        
        # Quality metrics
        quality_scores = [node.quality_score for node in path_nodes if node.quality_score is not None]
        if quality_scores:
            metrics["avg_quality_score"] = sum(quality_scores) / len(quality_scores)
            metrics["min_quality_score"] = min(quality_scores)
        
        # Execution metrics
        execution_times = [edge.execution_time for edge in path_edges if edge.execution_time is not None]
        if execution_times:
            metrics["total_execution_time"] = sum(execution_times)
        
        error_rates = [edge.error_rate for edge in path_edges if edge.error_rate is not None]
        if error_rates:
            metrics["avg_error_rate"] = sum(error_rates) / len(error_rates)
        
        return metrics
    
    def _propagate_quality_impact(self, lineage: DataLineage, node_id: UUID, quality_score: float) -> None:
        """Propagate quality impact to downstream nodes."""
        downstream_nodes = lineage.get_downstream_nodes(node_id)
        
        for downstream_node in downstream_nodes:
            # Find the edge between the nodes
            for edge in lineage.edges.values():
                if edge.source_node_id == node_id and edge.target_node_id == downstream_node.id:
                    # Update quality impact on the edge
                    edge.data_quality_impact = 1.0 - quality_score
                    break
    
    def _generate_impact_recommendations(self, lineage: DataLineage, node_id: UUID) -> List[str]:
        """Generate recommendations based on impact analysis."""
        recommendations = []
        
        node = lineage.get_node(node_id)
        downstream_count = len(lineage.get_downstream_nodes(node_id))
        
        if downstream_count > 5:
            recommendations.append("High downstream dependency - consider implementing data quality monitoring")
        
        if node.quality_score and node.quality_score < 0.8:
            recommendations.append("Low quality score - investigate data quality issues")
        
        if not node.last_validated or (datetime.utcnow() - node.last_validated).days > 7:
            recommendations.append("Data validation is stale - consider more frequent validation")
        
        return recommendations
    
    def _generate_health_recommendations(self, lineage: DataLineage) -> List[str]:
        """Generate health recommendations for a lineage."""
        recommendations = []
        
        validation_errors = lineage.validate_lineage()
        if validation_errors:
            recommendations.append("Fix validation errors to ensure lineage consistency")
        
        nodes_without_quality = [
            node for node in lineage.nodes.values()
            if node.quality_score is None
        ]
        
        if len(nodes_without_quality) > len(lineage.nodes) * 0.3:
            recommendations.append("Many nodes lack quality scores - implement quality monitoring")
        
        return recommendations
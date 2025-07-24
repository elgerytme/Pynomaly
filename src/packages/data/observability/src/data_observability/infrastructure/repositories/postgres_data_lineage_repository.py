"""PostgreSQL repository implementation for data lineage."""

from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, insert, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...domain.entities.data_lineage import DataLineage, LineageNode, LineageEdge, LineageNodeType, LineageRelationType, LineageMetadata
from ...domain.repositories.data_lineage_repository import DataLineageRepository
from ...infrastructure.persistence.models import DataLineageNodeModel, DataLineageEdgeModel


class PostgresDataLineageRepository(DataLineageRepository):
    """PostgreSQL implementation of data lineage repository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session."""
        self._session = session

    def _node_to_model(self, node: LineageNode) -> DataLineageNodeModel:
        """Convert LineageNode entity to DataLineageNodeModel."""
        # Ensure lineage_graph_id is set when converting to model
        lineage_graph_id = node.metadata.properties.get("lineage_graph_id")
        if not lineage_graph_id:
            raise ValueError("LineageNode must have 'lineage_graph_id' in its metadata properties to be saved.")

        return DataLineageNodeModel(
            id=node.id,
            lineage_graph_id=lineage_graph_id,
            name=node.name,
            asset_id=node.asset_id,
            node_type=node.type.value,
            description=node.description,
            metadata=node.metadata.dict()
        )

    def _model_to_node(self, model: DataLineageNodeModel) -> LineageNode:
        """Convert DataLineageNodeModel to LineageNode entity."""
        # Reconstruct LineageMetadata from model.metadata
        metadata = LineageMetadata(**model.metadata)
        # Add lineage_graph_id to metadata properties for consistency
        metadata.set_property("lineage_graph_id", model.lineage_graph_id)

        return LineageNode(
            id=model.id,
            name=model.name,
            type=LineageNodeType(model.node_type),
            namespace=metadata.properties.get("namespace", "default"), # Assuming namespace is in metadata
            description=model.description,
            asset_id=model.asset_id,
            metadata=metadata
        )

    def _edge_to_model(self, edge: LineageEdge) -> DataLineageEdgeModel:
        """Convert LineageEdge entity to DataLineageEdgeModel."""
        lineage_graph_id = edge.metadata.properties.get("lineage_graph_id")
        if not lineage_graph_id:
            raise ValueError("LineageEdge must have 'lineage_graph_id' in its metadata properties to be saved.")

        return DataLineageEdgeModel(
            id=edge.id,
            lineage_graph_id=lineage_graph_id,
            source_node_id=edge.source_node_id,
            target_node_id=edge.target_node_id,
            relationship_type=edge.relationship_type.value,
            metadata=edge.metadata.dict()
        )

    def _model_to_edge(self, model: DataLineageEdgeModel) -> LineageEdge:
        """Convert DataLineageEdgeModel to LineageEdge entity."""
        metadata = LineageMetadata(**model.metadata)
        metadata.set_property("lineage_graph_id", model.lineage_graph_id)

        return LineageEdge(
            id=model.id,
            source_node_id=model.source_node_id,
            target_node_id=model.target_node_id,
            relationship_type=LineageRelationType(model.relationship_type),
            metadata=metadata
        )

    async def save_lineage(self, lineage: DataLineage) -> DataLineage:
        """Save a data lineage graph."""
        # Check if lineage graph already exists
        existing_graph = await self._session.execute(
            select(DataLineageGraphModel).filter(DataLineageGraphModel.id == lineage.id)
        )
        graph_model = existing_graph.scalar_one_or_none()

        if graph_model:
            # Update existing graph
            graph_model.name = lineage.name
            graph_model.description = lineage.description
            graph_model.namespace = lineage.namespace
            graph_model.metadata = lineage.metadata.dict()
        else:
            # Create new graph
            graph_model = DataLineageGraphModel(
                id=lineage.id,
                name=lineage.name,
                description=lineage.description,
                namespace=lineage.namespace,
                metadata=lineage.metadata.dict()
            )
            self._session.add(graph_model)
        
        await self._session.flush() # Ensure graph_model has an ID if new

        # Save/update nodes
        for node_entity in lineage.nodes.values():
            # Ensure lineage_graph_id is set in node metadata for _node_to_model
            node_entity.metadata.set_property("lineage_graph_id", lineage.id)
            node_model = self._node_to_model(node_entity)
            
            existing_node_model = await self._session.execute(
                select(DataLineageNodeModel).filter(DataLineageNodeModel.id == node_model.id)
            )
            if existing_node_model.scalar_one_or_none():
                await self.update_node(node_entity)
            else:
                self._session.add(node_model)
        
        # Save/update edges
        for edge_entity in lineage.edges.values():
            # Ensure lineage_graph_id is set in edge metadata for _edge_to_model
            edge_entity.metadata.set_property("lineage_graph_id", lineage.id)
            edge_model = self._edge_to_model(edge_entity)

            existing_edge_model = await self._session.execute(
                select(DataLineageEdgeModel).filter(DataLineageEdgeModel.id == edge_model.id)
            )
            if existing_edge_model.scalar_one_or_none():
                await self.update_edge(edge_entity)
            else:
                self._session.add(edge_model)

        await self._session.commit()
        return lineage

    async def get_lineage_by_id(self, lineage_id: UUID) -> Optional[DataLineage]:
        """Get lineage graph by ID."""
        result = await self._session.execute(
            select(DataLineageGraphModel)
            .filter(DataLineageGraphModel.id == lineage_id)
            .options(selectinload(DataLineageGraphModel.nodes))
            .options(selectinload(DataLineageGraphModel.edges))
        )
        graph_model = result.scalar_one_or_none()

        if not graph_model:
            return None

        lineage = DataLineage(
            id=graph_model.id,
            name=graph_model.name,
            description=graph_model.description,
            namespace=graph_model.namespace,
            metadata=LineageMetadata(**graph_model.metadata)
        )

        for node_model in graph_model.nodes:
            lineage.add_node(self._model_to_node(node_model))
        for edge_model in graph_model.edges:
            lineage.add_edge(self._model_to_edge(edge_model))

        return lineage

    async def get_lineage_by_name(self, name: str, namespace: str = "default") -> Optional[DataLineage]:
        """Get lineage graph by name and namespace."""
        result = await self._session.execute(
            select(DataLineageGraphModel)
            .filter(DataLineageGraphModel.name == name, DataLineageGraphModel.namespace == namespace)
            .options(selectinload(DataLineageGraphModel.nodes))
            .options(selectinload(DataLineageGraphModel.edges))
        )
        graph_model = result.scalar_one_or_none()

        if not graph_model:
            return None

        lineage = DataLineage(
            id=graph_model.id,
            name=graph_model.name,
            description=graph_model.description,
            namespace=graph_model.namespace,
            metadata=LineageMetadata(**graph_model.metadata)
        )

        for node_model in graph_model.nodes:
            lineage.add_node(self._model_to_node(node_model))
        for edge_model in graph_model.edges:
            lineage.add_edge(self._model_to_edge(edge_model))

        return lineage

    async def list_lineages(self, namespace: str = None) -> List[DataLineage]:
        """List all lineages, optionally filtered by namespace."""
        query = select(DataLineageGraphModel)
        if namespace:
            query = query.filter(DataLineageGraphModel.namespace == namespace)
        
        result = await self._session.execute(query)
        graph_models = result.scalars().all()

        lineages = []
        for graph_model in graph_models:
            # For listing, we might not need to load all nodes/edges immediately
            # but for consistency with get_lineage_by_id/name, we'll load them.
            lineage = DataLineage(
                id=graph_model.id,
                name=graph_model.name,
                description=graph_model.description,
                namespace=graph_model.namespace,
                metadata=LineageMetadata(**graph_model.metadata)
            )
            # To avoid N+1 problem, consider using selectinload or joinedload if performance is an issue
            # For now, fetching nodes/edges separately for each lineage in the list
            nodes_result = await self._session.execute(
                select(DataLineageNodeModel).filter(DataLineageNodeModel.lineage_graph_id == graph_model.id)
            )
            for node_model in nodes_result.scalars().all():
                lineage.add_node(self._model_to_node(node_model))
            
            edges_result = await self._session.execute(
                select(DataLineageEdgeModel).filter(DataLineageEdgeModel.lineage_graph_id == graph_model.id)
            )
            for edge_model in edges_result.scalars().all():
                lineage.add_edge(self._model_to_edge(edge_model))
            
            lineages.append(lineage)
        return lineages

    async def add_node(self, lineage_id: UUID, node: LineageNode) -> None:
        """Add a node to a lineage."""
        # Ensure the node has the lineage_graph_id set in its metadata
        node.metadata.set_property("lineage_graph_id", lineage_id)
        model = self._node_to_model(node)

        # Check if node already exists
        existing_node = await self._session.execute(
            select(DataLineageNodeModel).filter(DataLineageNodeModel.id == node.id)
        )
        if existing_node.scalar_one_or_none():
            await self.update_node(node)
        else:
            self._session.add(model)
            await self._session.flush()

    async def add_edge(self, lineage_id: UUID, edge: LineageEdge) -> None:
        """Add an edge to a lineage."""
        # Ensure the edge has the lineage_graph_id set in its metadata
        edge.metadata.set_property("lineage_graph_id", lineage_id)
        model = self._edge_to_model(edge)

        # Check if edge already exists
        existing_edge = await self._session.execute(
            select(DataLineageEdgeModel).filter(DataLineageEdgeModel.id == edge.id)
        )
        if existing_edge.scalar_one_or_none():
            await self.update_edge(edge)
        else:
            self._session.add(model)
            await self._session.flush()

    async def get_node_by_id(self, node_id: UUID) -> Optional[LineageNode]:
        """Get a node by ID."""
        result = await self._session.execute(
            select(DataLineageNodeModel).filter(DataLineageNodeModel.id == node_id)
        )
        model = result.scalar_one_or_none()
        return self._model_to_node(model) if model else None

    async def get_edge_by_id(self, edge_id: UUID) -> Optional[LineageEdge]:
        """Get an edge by ID."""
        result = await self._session.execute(
            select(DataLineageEdgeModel).filter(DataLineageEdgeModel.id == edge_id)
        )
        model = result.scalar_one_or_none()
        return self._model_to_edge(model) if model else None

    async def find_nodes_by_name(self, name: str) -> List[LineageNode]:
        """Find all nodes with a given name across all lineages."""
        result = await self._session.execute(
            select(DataLineageNodeModel).filter(DataLineageNodeModel.name == name)
        )
        return [self._model_to_node(model) for model in result.scalars().all()]

    async def find_nodes_by_type(self, node_type: str) -> List[LineageNode]:
        """Find all nodes of a given type across all lineages."""
        result = await self._session.execute(
            select(DataLineageNodeModel).filter(DataLineageNodeModel.node_type == node_type)
        )
        return [self._model_to_node(model) for model in result.scalars().all()]

    async def get_nodes_by_asset_id(self, asset_id: UUID) -> List[LineageNode]:
        """Get all lineage nodes associated with a given asset ID."""
        result = await self._session.execute(
            select(DataLineageNodeModel).filter(DataLineageNodeModel.asset_id == asset_id)
        )
        return [self._model_to_node(model) for model in result.scalars().all()]

    async def update_node(self, node: LineageNode) -> LineageNode:
        """Update an existing lineage node."""
        model_data = self._node_to_model(node).dict(exclude_unset=True)
        model_data.pop("id", None)
        model_data.pop("created_at", None)
        model_data.pop("updated_at", None)

        result = await self._session.execute(
            update(DataLineageNodeModel)
            .filter(DataLineageNodeModel.id == node.id)
            .values(**model_data)
        )
        await self._session.flush()
        if result.rowcount > 0:
            return await self.get_node_by_id(node.id)
        else:
            raise ValueError(f"Node with ID {node.id} not found for update")

    async def update_edge(self, edge: LineageEdge) -> LineageEdge:
        """Update an existing lineage edge."""
        model_data = self._edge_to_model(edge).dict(exclude_unset=True)
        model_data.pop("id", None)
        model_data.pop("created_at", None)

        result = await self._session.execute(
            update(DataLineageEdgeModel)
            .filter(DataLineageEdgeModel.id == edge.id)
            .values(**model_data)
        )
        await self._session.flush()
        if result.rowcount > 0:
            return await self.get_edge_by_id(edge.id)
        else:
            raise ValueError(f"Edge with ID {edge.id} not found for update")

    async def delete_lineage(self, lineage_id: UUID) -> bool:
        """Delete a lineage graph by ID."""
        result = await self._session.execute(
            delete(DataLineageGraphModel).filter(DataLineageGraphModel.id == lineage_id)
        )
        await self._session.commit()
        return result.rowcount > 0

    async def delete_node(self, node_id: UUID) -> bool:
        """Delete a node by ID."""
        result = await self._session.execute(
            delete(DataLineageNodeModel).filter(DataLineageNodeModel.id == node_id)
        )
        await self._session.flush()
        return result.rowcount > 0

    async def delete_edge(self, edge_id: UUID) -> bool:
        """Delete an edge by ID."""
        result = await self._session.execute(
            delete(DataLineageEdgeModel).filter(DataLineageEdgeModel.id == edge_id)
        )
        await self._session.flush()
        return result.rowcount > 0

"""Service for managing workflow pipelines."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class WorkflowNode(BaseModel):
    """A single node in a workflow pipeline."""
    id: str
    type: str
    label: str
    position: Dict[str, float]  # x, y coordinates
    inputs: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}
    config: Dict[str, Any] = {}


class WorkflowEdge(BaseModel):
    """Connection between workflow nodes."""
    id: str
    source: str
    target: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None


class Workflow(BaseModel):
    """Complete workflow definition."""
    id: str
    name: str
    description: Optional[str] = None
    nodes: List[WorkflowNode] = []
    edges: List[WorkflowEdge] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    version: int = 1
    is_active: bool = True


class WorkflowService:
    """Service for managing workflow pipelines."""

    def __init__(self):
        self._workflows = {}

    def create_workflow(
        self,
        name: str,
        description: Optional[str] = None,
        nodes: List[WorkflowNode] = None,
        edges: List[WorkflowEdge] = None,
        metadata: Dict[str, Any] = None
    ) -> Workflow:
        """Create a new workflow."""
        workflow_id = f"wf_{len(self._workflows) + 1}"
        now = datetime.utcnow()
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            nodes=nodes or [],
            edges=edges or [],
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            version=1,
            is_active=True
        )
        
        self._workflows[workflow_id] = workflow
        return workflow

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    def list_workflows(self, active_only: bool = True) -> List[Workflow]:
        """List all workflows."""
        workflows = list(self._workflows.values())
        if active_only:
            workflows = [w for w in workflows if w.is_active]
        return workflows

    def update_workflow(
        self,
        workflow_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        nodes: Optional[List[WorkflowNode]] = None,
        edges: Optional[List[WorkflowEdge]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Workflow]:
        """Update an existing workflow."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None

        if name is not None:
            workflow.name = name
        if description is not None:
            workflow.description = description
        if nodes is not None:
            workflow.nodes = nodes
        if edges is not None:
            workflow.edges = edges
        if metadata is not None:
            workflow.metadata.update(metadata)
        
        workflow.updated_at = datetime.utcnow()
        workflow.version += 1
        
        return workflow

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow (soft delete by setting is_active=False)."""
        workflow = self._workflows.get(workflow_id)
        if workflow:
            workflow.is_active = False
            workflow.updated_at = datetime.utcnow()
            return True
        return False

    def export_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Export a workflow as JSON."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
        
        return workflow.dict()

    def import_workflow(self, workflow_data: Dict[str, Any]) -> Workflow:
        """Import a workflow from JSON data."""
        # Generate new ID for imported workflow
        workflow_id = f"wf_{len(self._workflows) + 1}"
        now = datetime.utcnow()
        
        # Parse nodes and edges
        nodes = [WorkflowNode(**node) for node in workflow_data.get('nodes', [])]
        edges = [WorkflowEdge(**edge) for edge in workflow_data.get('edges', [])]
        
        workflow = Workflow(
            id=workflow_id,
            name=workflow_data.get('name', 'Imported Workflow'),
            description=workflow_data.get('description'),
            nodes=nodes,
            edges=edges,
            metadata=workflow_data.get('metadata', {}),
            created_at=now,
            updated_at=now,
            version=1,
            is_active=True
        )
        
        self._workflows[workflow_id] = workflow
        return workflow

    def validate_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Validate a workflow for execution."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return {"valid": False, "errors": ["Workflow not found"]}
        
        errors = []
        
        # Check for orphaned nodes
        node_ids = {node.id for node in workflow.nodes}
        for edge in workflow.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge references unknown source node: {edge.source}")
            if edge.target not in node_ids:
                errors.append(f"Edge references unknown target node: {edge.target}")
        
        # Check for cycles (basic check)
        # TODO: Implement proper cycle detection
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "node_count": len(workflow.nodes),
            "edge_count": len(workflow.edges)
        }

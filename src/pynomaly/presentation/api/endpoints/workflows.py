"""API router for workflows."""

from typing import Dict, List
from fastapi import APIRouter
from pydantic import BaseModel

class Workflow(BaseModel):
    id: int
    name: str
    data: Dict

router = APIRouter()

workflows = []

@router.post("/workflows")
async def create_workflow(workflow: Workflow):
    workflows.append(workflow)
    return workflow

@router.get("/workflows", response_model=List[Workflow])
async def list_workflows():
    return workflows

@router.get("/workflows/{workflow_id}", response_model=Workflow)
async def get_workflow(workflow_id: int):
    for workflow in workflows:
        if workflow.id == workflow_id:
            return workflow
    return {"error": "Workflow not found"}

@router.put("/workflows/{workflow_id}", response_model=Workflow)
async def update_workflow(workflow_id: int, updated_workflow: Workflow):
    for i, workflow in enumerate(workflows):
        if workflow.id == workflow_id:
            workflows[i] = updated_workflow
            return updated_workflow
    return {"error": "Workflow not found"}

@router.delete("/workflows/{workflow_id}", response_model=Dict)
async def delete_workflow(workflow_id: int):
    for i, workflow in enumerate(workflows):
        if workflow.id == workflow_id:
            del workflows[i]
            return {"success": True}
    return {"error": "Workflow not found"}

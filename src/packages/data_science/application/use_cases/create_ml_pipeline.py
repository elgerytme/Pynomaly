"""Use case for creating machine learning pipelines."""

from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime

from ..dto.ml_pipeline_dto import (
    CreatePipelineRequestDTO,
    CreatePipelineResponseDTO
)
from ...domain.entities.machine_learning_pipeline import (
    MachineLearningPipeline,
    PipelineType,
    PipelineStatus
)
from ...domain.repositories.machine_learning_pipeline_repository import IMachineLearningPipelineRepository


class CreateMLPipelineUseCase:
    """Use case for creating machine learning pipelines."""
    
    def __init__(self, pipeline_repository: IMachineLearningPipelineRepository):
        self._repository = pipeline_repository
    
    async def execute(self, request: CreatePipelineRequestDTO) -> CreatePipelineResponseDTO:
        """Execute pipeline creation use case.
        
        Args:
            request: Pipeline creation request parameters
            
        Returns:
            Pipeline creation response with pipeline details
            
        Raises:
            PipelineCreationError: If pipeline creation fails
        """
        try:
            # Validate pipeline steps
            self._validate_pipeline_steps(request.steps)
            
            # Map pipeline type
            pipeline_type = self._map_pipeline_type(request.pipeline_type)
            
            # Create pipeline entity
            pipeline = MachineLearningPipeline(
                name=request.name,
                pipeline_type=pipeline_type,
                description=request.description,
                status=PipelineStatus.DRAFT,
                version_number="1.0.0",
                steps=request.steps,
                dependencies=request.dependencies or [],
                parameters=request.parameters or {},
                input_schema={},
                output_schema={},
                resource_requirements={
                    "cpu_cores": 2,
                    "memory_gb": 4,
                    "gpu_required": False
                }
            )
            
            # Save pipeline
            await self._repository.save(pipeline)
            
            return CreatePipelineResponseDTO(
                pipeline_id=pipeline.pipeline_id.value,
                name=pipeline.name,
                status=pipeline.status.value,
                version_number=pipeline.version_number,
                created_at=pipeline.created_at
            )
            
        except Exception as e:
            raise ValueError(f"Failed to create pipeline: {str(e)}")
    
    def _validate_pipeline_steps(self, steps: List[Dict[str, Any]]) -> None:
        """Validate pipeline steps configuration."""
        if not steps:
            raise ValueError("Pipeline must have at least one step")
        
        required_fields = ["name", "type", "config"]
        step_names = set()
        
        for i, step in enumerate(steps):
            # Check required fields
            for field in required_fields:
                if field not in step:
                    raise ValueError(f"Step {i} missing required field: {field}")
            
            # Check for duplicate step names
            step_name = step["name"]
            if step_name in step_names:
                raise ValueError(f"Duplicate step name: {step_name}")
            step_names.add(step_name)
            
            # Validate step type
            valid_step_types = [
                "data_preprocessing",
                "feature_engineering", 
                "model_training",
                "model_validation",
                "model_deployment",
                "data_quality_check",
                "statistical_analysis"
            ]
            
            if step["type"] not in valid_step_types:
                raise ValueError(f"Invalid step type: {step['type']}")
    
    def _map_pipeline_type(self, pipeline_type_str: str) -> PipelineType:
        """Map string pipeline type to enum."""
        type_mapping = {
            "training": PipelineType.TRAINING,
            "inference": PipelineType.INFERENCE, 
            "batch_prediction": PipelineType.BATCH_PREDICTION,
            "feature_engineering": PipelineType.FEATURE_ENGINEERING,
            "data_validation": PipelineType.DATA_VALIDATION,
            "model_validation": PipelineType.MODEL_VALIDATION,
            "automated_ml": PipelineType.AUTOMATED_ML,
            "experimentation": PipelineType.EXPERIMENTATION
        }
        
        if pipeline_type_str not in type_mapping:
            raise ValueError(f"Invalid pipeline type: {pipeline_type_str}")
        
        return type_mapping[pipeline_type_str]
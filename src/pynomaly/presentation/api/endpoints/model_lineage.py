"""Model lineage tracking API endpoints."""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from pynomaly.application.services.model_lineage_service import ModelLineageService
from pynomaly.domain.entities.lineage_record import (
    LineageGraph,
    LineageQuery,
    LineageRecord,
    LineageRelationType,
    LineageStatistics,
    LineageTransformation,
    TransformationType,
)
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.auth_deps import get_container_simple_simple
from pynomaly.presentation.api.docs.response_models import (
    ErrorResponse,
    HTTPResponses,
    SuccessResponse,
)

router = APIRouter(
    prefix="/lineage",
    tags=["Model Lineage"],
    responses={
        401: HTTPResponses.unauthorized_401(),
        403: HTTPResponses.forbidden_403(),
        404: HTTPResponses.not_found_404(),
        500: HTTPResponses.server_error_500(),
    },
)


class CreateLineageRequest(BaseModel):
    """Request for creating lineage record."""

    child_model_id: UUID = Field(..., description="Child model identifier")
    parent_model_ids: list[UUID] = Field(..., description="Parent model identifiers")
    relation_type: LineageRelationType = Field(..., description="Type of relationship")
    transformation: LineageTransformation = Field(
        ..., description="Transformation details"
    )
    experiment_id: UUID | None = Field(None, description="Associated experiment ID")
    run_id: str | None = Field(None, description="Associated run ID")
    tags: list[str] = Field(default_factory=list, description="Lineage tags")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class TrackDerivationRequest(BaseModel):
    """Request for tracking model derivation."""

    parent_model_id: UUID = Field(..., description="Parent model ID")
    child_model_id: UUID = Field(..., description="Child model ID")
    transformation_type: TransformationType = Field(
        ..., description="Type of transformation"
    )
    transformation_metadata: dict[str, Any] = Field(
        ..., description="Transformation parameters"
    )
    algorithm: str | None = Field(None, description="Algorithm used")
    tool: str | None = Field(None, description="Tool or framework used")
    execution_time: float | None = Field(None, description="Execution time in seconds")
    resource_usage: dict[str, Any] = Field(
        default_factory=dict, description="Resource usage"
    )


class TrackEnsembleRequest(BaseModel):
    """Request for tracking ensemble creation."""

    ensemble_model_id: UUID = Field(..., description="Ensemble model ID")
    component_model_ids: list[UUID] = Field(..., description="Component model IDs")
    ensemble_metadata: dict[str, Any] = Field(..., description="Ensemble configuration")
    algorithm: str = Field(default="ensemble", description="Ensemble algorithm")
    tool: str | None = Field(None, description="Tool used for ensemble")


class LineageQueryRequest(BaseModel):
    """Request for querying lineage."""

    model_id: UUID = Field(..., description="Target model identifier")
    include_ancestors: bool = Field(True, description="Include ancestor models")
    include_descendants: bool = Field(True, description="Include descendant models")
    max_depth: int = Field(10, description="Maximum depth to traverse")
    relation_types: list[LineageRelationType] | None = Field(
        None, description="Filter by relation types"
    )
    transformation_types: list[TransformationType] | None = Field(
        None, description="Filter by transformation types"
    )
    created_after: datetime | None = Field(None, description="Filter by creation date")
    created_before: datetime | None = Field(None, description="Filter by creation date")
    created_by: str | None = Field(None, description="Filter by creator")
    tags: list[str] | None = Field(None, description="Filter by tags")


async def get_lineage_service(
    container: Container = Depends(get_container_simple),
) -> ModelLineageService:
    """Get model lineage service."""
    # This would be properly injected in a real implementation
    # For now, create a mock service
    return ModelLineageService(
        model_repository=container.model_repository(),
        model_version_repository=container.model_version_repository(),
        lineage_repository=None,  # Would be injected
    )


@router.post(
    "/records",
    response_model=SuccessResponse[LineageRecord],
    summary="Create Lineage Record",
    description="""
    Create a new model lineage record to track relationships between models.
    
    This endpoint allows you to document how models are related through various
    transformations such as fine-tuning, ensemble creation, or distillation.
    
    **Use Cases:**
    - Track parent-child relationships between models
    - Document transformation processes and parameters
    - Link models to experiments and runs
    - Maintain audit trail for model evolution
    
    **Example Relationships:**
    - Fine-tuning: Model B fine-tuned from Model A
    - Ensemble: Model C is ensemble of Models A and B
    - Distillation: Model B distilled from Model A
    """,
    responses={
        201: HTTPResponses.created_201("Lineage record created successfully"),
        400: HTTPResponses.bad_request_400("Invalid lineage data"),
    },
)
async def create_lineage_record(
    request: CreateLineageRequest,
    created_by: str = Query(..., description="User creating the record"),
    lineage_service: ModelLineageService = Depends(get_lineage_service),
) -> SuccessResponse[LineageRecord]:
    """Create a new lineage record."""
    try:
        record = await lineage_service.create_lineage_record(
            child_model_id=request.child_model_id,
            parent_model_ids=request.parent_model_ids,
            relation_type=request.relation_type,
            transformation=request.transformation,
            created_by=created_by,
            experiment_id=request.experiment_id,
            run_id=request.run_id,
            tags=request.tags,
            metadata=request.metadata,
        )

        return SuccessResponse(
            data=record, message="Lineage record created successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create lineage record: {str(e)}"
        )


@router.post(
    "/track-derivation",
    response_model=SuccessResponse[LineageRecord],
    summary="Track Model Derivation",
    description="""
    Track a simple parent-child model derivation with transformation details.
    
    This is a simplified endpoint for the common case of tracking how one model
    was derived from another through a specific transformation process.
    
    **Common Use Cases:**
    - Fine-tuning a pre-trained model
    - Transfer learning from a base model
    - Model compression through pruning or quantization
    - Knowledge distillation
    """,
    responses={
        201: HTTPResponses.created_201("Model derivation tracked successfully"),
        400: HTTPResponses.bad_request_400("Invalid derivation data"),
    },
)
async def track_model_derivation(
    request: TrackDerivationRequest,
    created_by: str = Query(..., description="User tracking the derivation"),
    lineage_service: ModelLineageService = Depends(get_lineage_service),
) -> SuccessResponse[LineageRecord]:
    """Track a model derivation."""
    try:
        record = await lineage_service.track_model_derivation(
            parent_model_id=request.parent_model_id,
            child_model_id=request.child_model_id,
            transformation_type=request.transformation_type,
            transformation_metadata=request.transformation_metadata,
            created_by=created_by,
            algorithm=request.algorithm,
            tool=request.tool,
            execution_time=request.execution_time,
            resource_usage=request.resource_usage,
        )

        return SuccessResponse(
            data=record, message="Model derivation tracked successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to track derivation: {str(e)}"
        )


@router.post(
    "/track-ensemble",
    response_model=SuccessResponse[LineageRecord],
    summary="Track Ensemble Creation",
    description="""
    Track the creation of an ensemble model from multiple component models.
    
    This endpoint specifically handles ensemble models where multiple models
    are combined to create a new ensemble model.
    
    **Ensemble Types:**
    - Voting ensembles
    - Stacking ensembles
    - Boosting ensembles
    - Weighted averages
    """,
    responses={
        201: HTTPResponses.created_201("Ensemble creation tracked successfully"),
        400: HTTPResponses.bad_request_400("Invalid ensemble data"),
    },
)
async def track_ensemble_creation(
    request: TrackEnsembleRequest,
    created_by: str = Query(..., description="User tracking the ensemble"),
    lineage_service: ModelLineageService = Depends(get_lineage_service),
) -> SuccessResponse[LineageRecord]:
    """Track ensemble model creation."""
    try:
        record = await lineage_service.track_ensemble_creation(
            ensemble_model_id=request.ensemble_model_id,
            component_model_ids=request.component_model_ids,
            ensemble_metadata=request.ensemble_metadata,
            created_by=created_by,
            algorithm=request.algorithm,
            tool=request.tool,
        )

        return SuccessResponse(
            data=record, message="Ensemble creation tracked successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to track ensemble: {str(e)}"
        )


@router.get(
    "/models/{model_id}/graph",
    response_model=SuccessResponse[LineageGraph],
    summary="Get Model Lineage Graph",
    description="""
    Get the complete lineage graph for a model, including ancestors and descendants.
    
    This returns a comprehensive view of how a model relates to other models
    in your system, showing the full lineage tree with relationships and
    transformation details.
    
    **Graph Structure:**
    - **Nodes**: Individual models with metadata
    - **Edges**: Relationships with transformation details
    - **Depth**: Maximum depth of the lineage tree
    
    **Use Cases:**
    - Visualize model evolution
    - Understand model dependencies
    - Track model heritage for compliance
    - Identify model families
    """,
)
async def get_model_lineage_graph(
    model_id: UUID,
    include_ancestors: bool = Query(True, description="Include ancestor models"),
    include_descendants: bool = Query(True, description="Include descendant models"),
    max_depth: int = Query(10, description="Maximum depth to traverse"),
    lineage_service: ModelLineageService = Depends(get_lineage_service),
) -> SuccessResponse[LineageGraph]:
    """Get complete lineage graph for a model."""
    try:
        graph = await lineage_service.get_model_lineage(
            model_id=model_id,
            include_ancestors=include_ancestors,
            include_descendants=include_descendants,
            max_depth=max_depth,
        )

        return SuccessResponse(
            data=graph,
            message=f"Retrieved lineage graph with {len(graph.nodes)} models and {len(graph.edges)} relationships",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get lineage graph: {str(e)}"
        )


@router.get(
    "/models/{model_id}/ancestors",
    response_model=SuccessResponse[list[UUID]],
    summary="Get Model Ancestors",
    description="""
    Get all ancestor models for a given model.
    
    Returns a list of model IDs that are ancestors (parents, grandparents, etc.)
    of the specified model.
    """,
)
async def get_model_ancestors(
    model_id: UUID,
    max_depth: int = Query(10, description="Maximum depth to traverse"),
    lineage_service: ModelLineageService = Depends(get_lineage_service),
) -> SuccessResponse[list[UUID]]:
    """Get all ancestor models."""
    try:
        ancestors = await lineage_service.get_model_ancestors(model_id, max_depth)

        return SuccessResponse(
            data=ancestors, message=f"Found {len(ancestors)} ancestor models"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get ancestors: {str(e)}"
        )


@router.get(
    "/models/{model_id}/descendants",
    response_model=SuccessResponse[list[UUID]],
    summary="Get Model Descendants",
    description="""
    Get all descendant models for a given model.
    
    Returns a list of model IDs that are descendants (children, grandchildren, etc.)
    of the specified model.
    """,
)
async def get_model_descendants(
    model_id: UUID,
    max_depth: int = Query(10, description="Maximum depth to traverse"),
    lineage_service: ModelLineageService = Depends(get_lineage_service),
) -> SuccessResponse[list[UUID]]:
    """Get all descendant models."""
    try:
        descendants = await lineage_service.get_model_descendants(model_id, max_depth)

        return SuccessResponse(
            data=descendants, message=f"Found {len(descendants)} descendant models"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get descendants: {str(e)}"
        )


@router.get(
    "/models/{from_model_id}/path/{to_model_id}",
    response_model=SuccessResponse[list[UUID]],
    summary="Find Lineage Path",
    description="""
    Find the lineage path between two models.
    
    Returns the shortest path of model IDs that connect the source model
    to the target model through lineage relationships.
    
    Returns null if no path exists between the models.
    """,
)
async def find_lineage_path(
    from_model_id: UUID,
    to_model_id: UUID,
    lineage_service: ModelLineageService = Depends(get_lineage_service),
) -> SuccessResponse[list[UUID] | None]:
    """Find lineage path between two models."""
    try:
        path = await lineage_service.get_lineage_path(from_model_id, to_model_id)

        if path:
            return SuccessResponse(
                data=path, message=f"Found path with {len(path)} models"
            )
        else:
            return SuccessResponse(
                data=None, message="No lineage path found between models"
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find path: {str(e)}")


@router.post(
    "/query",
    response_model=SuccessResponse[list[LineageRecord]],
    summary="Query Lineage Records",
    description="""
    Query lineage records with advanced filtering options.
    
    This endpoint provides flexible querying capabilities to find lineage
    records based on various criteria such as model IDs, relationship types,
    transformation types, dates, creators, and tags.
    
    **Filter Options:**
    - **Model-based**: Find records involving specific models
    - **Type-based**: Filter by relationship or transformation types
    - **Time-based**: Filter by creation date ranges
    - **User-based**: Filter by record creators
    - **Tag-based**: Filter by associated tags
    """,
)
async def query_lineage_records(
    query: LineageQueryRequest,
    lineage_service: ModelLineageService = Depends(get_lineage_service),
) -> SuccessResponse[list[LineageRecord]]:
    """Query lineage records with filters."""
    try:
        lineage_query = LineageQuery(
            model_id=query.model_id,
            include_ancestors=query.include_ancestors,
            include_descendants=query.include_descendants,
            max_depth=query.max_depth,
            relation_types=query.relation_types,
            transformation_types=query.transformation_types,
            created_after=query.created_after,
            created_before=query.created_before,
            created_by=query.created_by,
            tags=query.tags,
        )

        records = await lineage_service.query_lineage(lineage_query)

        return SuccessResponse(
            data=records, message=f"Found {len(records)} lineage records"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to query lineage: {str(e)}"
        )


@router.get(
    "/statistics",
    response_model=SuccessResponse[LineageStatistics],
    summary="Get Lineage Statistics",
    description="""
    Get overall statistics about model lineage in the system.
    
    Provides insights into the lineage ecosystem including:
    - Total number of models and relationships
    - Maximum lineage depth
    - Distribution of relationship types
    - Distribution of transformation types
    - Orphaned models (without lineage)
    """,
)
async def get_lineage_statistics(
    lineage_service: ModelLineageService = Depends(get_lineage_service),
) -> SuccessResponse[LineageStatistics]:
    """Get lineage statistics."""
    try:
        stats = await lineage_service.get_lineage_statistics()

        return SuccessResponse(data=stats, message="Retrieved lineage statistics")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )


@router.delete(
    "/records/{record_id}",
    response_model=SuccessResponse[bool],
    summary="Delete Lineage Record",
    description="""
    Delete a specific lineage record.
    
    **Warning**: This operation cannot be undone. Deleting lineage records
    will break the lineage chain and may affect lineage graph completeness.
    
    Use with caution and ensure you have proper authorization.
    """,
    responses={
        200: HTTPResponses.success_200("Lineage record deleted successfully"),
        404: HTTPResponses.not_found_404("Lineage record not found"),
    },
)
async def delete_lineage_record(
    record_id: UUID, lineage_service: ModelLineageService = Depends(get_lineage_service)
) -> SuccessResponse[bool]:
    """Delete a lineage record."""
    try:
        success = await lineage_service.delete_lineage_record(record_id)

        if success:
            return SuccessResponse(
                data=True, message="Lineage record deleted successfully"
            )
        else:
            raise HTTPException(status_code=404, detail="Lineage record not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete lineage record: {str(e)}"
        )


@router.post(
    "/bulk-import",
    response_model=SuccessResponse[list[LineageRecord]],
    summary="Bulk Import Lineage Records",
    description="""
    Import multiple lineage records in a single operation.
    
    This endpoint is useful for:
    - Migrating lineage data from other systems
    - Bulk loading historical lineage information
    - Batch processing of lineage creation
    
    **Validation**: All referenced models must exist before import.
    """,
    responses={
        201: HTTPResponses.created_201("Lineage records imported successfully"),
        400: HTTPResponses.bad_request_400("Invalid lineage data or missing models"),
    },
)
async def bulk_import_lineage(
    records: list[LineageRecord],
    lineage_service: ModelLineageService = Depends(get_lineage_service),
) -> SuccessResponse[list[LineageRecord]]:
    """Bulk import lineage records."""
    try:
        imported_records = await lineage_service.bulk_import_lineage(records)

        return SuccessResponse(
            data=imported_records,
            message=f"Successfully imported {len(imported_records)} lineage records",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to import lineage records: {str(e)}"
        )

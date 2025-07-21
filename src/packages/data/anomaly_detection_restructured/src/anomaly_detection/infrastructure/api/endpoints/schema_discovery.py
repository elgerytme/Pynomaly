"""Schema Discovery and Inference API Endpoints.

This module provides RESTful endpoints for schema discovery, type inference,
and schema evolution tracking.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from .security.authorization import require_permissions
from .dependencies.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schema-discovery", tags=["Schema Discovery"])

# Pydantic models for request/response
class SchemaDiscoveryRequest(BaseModel):
    """Request model for schema discovery."""
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    data: List[Dict[str, Any]] = Field(..., description="Dataset records as list of dictionaries")
    discovery_mode: str = Field(default="comprehensive", description="Discovery mode (quick, comprehensive, statistical)")
    include_samples: bool = Field(default=True, description="Include sample values in response")
    infer_relationships: bool = Field(default=False, description="Infer relationships between columns")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Schema discovery configuration")        schema_extra = {
            "example": {
                "dataset_id": "customer_data_2024",
                "data": [
                    {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com", "active": True},
                    {"id": 2, "name": "Jane Smith", "age": 25, "email": "jane@example.com", "active": False}
                ],
                "discovery_mode": "comprehensive",
                "include_samples": True,
                "infer_relationships": True,
                "config": {
                    "sample_size": 1000,
                    "confidence_threshold": 0.8,
                    "enable_semantic_inference": True
                }
            }
        }


class SchemaDiscoveryResponse(BaseModel):
    """Response model for schema discovery."""
    discovery_id: str = Field(..., description="Unique identifier for the discovery task")
    dataset_id: str = Field(..., description="Dataset identifier")
    schema_version: str = Field(..., description="Schema version")
    discovery_mode: str = Field(..., description="Discovery mode used")
    columns: List[Dict[str, Any]] = Field(..., description="Discovered column information")
    relationships: List[Dict[str, Any]] = Field(..., description="Inferred relationships")
    constraints: List[Dict[str, Any]] = Field(..., description="Discovered constraints")
    quality_metrics: Dict[str, Any] = Field(..., description="Schema quality metrics")
    recommendations: List[Dict[str, Any]] = Field(..., description="Schema improvement recommendations")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    created_at: str = Field(..., description="Discovery timestamp")        schema_extra = {
            "example": {
                "discovery_id": "sch_123456789",
                "dataset_id": "customer_data_2024",
                "schema_version": "1.0.0",
                "discovery_mode": "comprehensive",
                "columns": [
                    {
                        "name": "id",
                        "inferred_type": "integer",
                        "nullable": False,
                        "unique": True,
                        "primary_key": True,
                        "min_value": 1,
                        "max_value": 1000,
                        "samples": [1, 2, 3, 4, 5]
                    },
                    {
                        "name": "name",
                        "inferred_type": "string",
                        "nullable": False,
                        "unique": False,
                        "semantic_type": "person_name",
                        "max_length": 50,
                        "samples": ["John Doe", "Jane Smith", "Bob Johnson"]
                    },
                    {
                        "name": "email",
                        "inferred_type": "string",
                        "nullable": False,
                        "unique": True,
                        "semantic_type": "email_address",
                        "format": "email",
                        "samples": ["john@example.com", "jane@example.com"]
                    }
                ],
                "relationships": [
                    {
                        "type": "functional_dependency",
                        "source": "id",
                        "target": "name",
                        "confidence": 1.0,
                        "description": "ID functionally determines name"
                    }
                ],
                "constraints": [
                    {
                        "type": "primary_key",
                        "columns": ["id"],
                        "confidence": 1.0
                    },
                    {
                        "type": "unique",
                        "columns": ["email"],
                        "confidence": 0.95
                    }
                ],
                "quality_metrics": {
                    "completeness": 0.98,
                    "consistency": 0.92,
                    "conformity": 0.87,
                    "uniqueness": 0.95
                },
                "recommendations": [
                    {
                        "type": "constraint",
                        "description": "Add primary key constraint on id column",
                        "priority": "high",
                        "sql": "ALTER TABLE customer_data ADD CONSTRAINT pk_id PRIMARY KEY (id)"
                    }
                ],
                "processing_time_ms": 2345.7,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class SchemaInferenceRequest(BaseModel):
    """Request model for type inference."""
    dataset_id: str = Field(..., description="Dataset identifier")
    columns: List[Dict[str, Any]] = Field(..., description="Column data for inference")
    inference_method: str = Field(default="statistical", description="Inference method (statistical, ml, hybrid)")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence threshold")        schema_extra = {
            "example": {
                "dataset_id": "customer_data_2024",
                "columns": [
                    {
                        "name": "birth_date",
                        "values": ["1990-01-15", "1985-03-22", "1995-12-08", "invalid_date", "2000-05-30"]
                    },
                    {
                        "name": "score",
                        "values": ["85.5", "92.3", "78.9", "invalid", "95.1"]
                    }
                ],
                "inference_method": "statistical",
                "confidence_threshold": 0.8
            }
        }


class SchemaInferenceResponse(BaseModel):
    """Response model for type inference."""
    inference_id: str = Field(..., description="Inference task identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    inference_method: str = Field(..., description="Inference method used")
    column_inferences: List[Dict[str, Any]] = Field(..., description="Type inferences by column")
    semantic_inferences: List[Dict[str, Any]] = Field(..., description="Semantic type inferences")
    format_inferences: List[Dict[str, Any]] = Field(..., description="Format pattern inferences")
    confidence_summary: Dict[str, float] = Field(..., description="Confidence summary by column")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    created_at: str = Field(..., description="Inference timestamp")


class SchemaComparisonRequest(BaseModel):
    """Request model for schema comparison."""
    source_schema_id: str = Field(..., description="Source schema identifier")
    target_schema_id: str = Field(..., description="Target schema identifier")
    comparison_mode: str = Field(default="structural", description="Comparison mode (structural, semantic, full)")
    ignore_nullable: bool = Field(default=False, description="Ignore nullable differences")        schema_extra = {
            "example": {
                "source_schema_id": "sch_123456789",
                "target_schema_id": "sch_987654321",
                "comparison_mode": "full",
                "ignore_nullable": False
            }
        }


class SchemaComparisonResponse(BaseModel):
    """Response model for schema comparison."""
    comparison_id: str = Field(..., description="Comparison task identifier")
    source_schema_id: str = Field(..., description="Source schema identifier")
    target_schema_id: str = Field(..., description="Target schema identifier")
    comparison_mode: str = Field(..., description="Comparison mode used")
    compatibility_score: float = Field(..., description="Overall compatibility score")
    differences: List[Dict[str, Any]] = Field(..., description="Detected differences")
    additions: List[Dict[str, Any]] = Field(..., description="Columns added in target")
    removals: List[Dict[str, Any]] = Field(..., description="Columns removed in target")
    modifications: List[Dict[str, Any]] = Field(..., description="Column modifications")
    migration_suggestions: List[Dict[str, Any]] = Field(..., description="Migration suggestions")
    created_at: str = Field(..., description="Comparison timestamp")


# API Endpoints

@router.post(
    "/discover-schema",
    response_model=SchemaDiscoveryResponse,
    summary="Discover schema from data",
    description="Analyze data to discover schema, infer types, and identify relationships"
)
@require_permissions(["schema_discovery:read"])
async def discover_schema(
    request: SchemaDiscoveryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Discover schema from the provided data."""
    try:
        logger.info(f"Schema discovery request for dataset {request.dataset_id}")
        
        # Mock implementation - in real implementation, this would:
        # 1. Analyze the data structure
        # 2. Infer column types and constraints
        # 3. Detect relationships
        # 4. Generate schema recommendations
        
        columns = []
        relationships = []
        constraints = []
        
        # Analyze each column in the data
        if request.data:
            sample_record = request.data[0]
            for column_name, value in sample_record.items():
                column_info = {
                    "name": column_name,
                    "inferred_type": _infer_type(value),
                    "nullable": _check_nullable(column_name, request.data),
                    "unique": _check_unique(column_name, request.data),
                    "primary_key": column_name.lower() == "id",
                    "samples": _get_samples(column_name, request.data, request.include_samples)
                }
                
                # Add semantic type inference
                if request.config and request.config.get("enable_semantic_inference"):
                    column_info["semantic_type"] = _infer_semantic_type(column_name, value)
                
                # Add statistical information based on type
                if column_info["inferred_type"] == "integer":
                    column_info.update(_get_numeric_stats(column_name, request.data))
                elif column_info["inferred_type"] == "string":
                    column_info.update(_get_string_stats(column_name, request.data))
                
                columns.append(column_info)
        
        # Infer relationships if requested
        if request.infer_relationships:
            relationships = _infer_relationships(columns, request.data)
        
        # Generate constraints
        constraints = _generate_constraints(columns)
        
        # Calculate quality metrics
        quality_metrics = _calculate_quality_metrics(columns, request.data)
        
        # Generate recommendations
        recommendations = _generate_schema_recommendations(columns, constraints, quality_metrics)
        
        return SchemaDiscoveryResponse(
            discovery_id=str(uuid4()),
            dataset_id=request.dataset_id,
            schema_version="1.0.0",
            discovery_mode=request.discovery_mode,
            columns=columns,
            relationships=relationships,
            constraints=constraints,
            quality_metrics=quality_metrics,
            recommendations=recommendations,
            processing_time_ms=2345.7,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Schema discovery failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Schema discovery failed"
        )


@router.post(
    "/infer-types",
    response_model=SchemaInferenceResponse,
    summary="Infer column types",
    description="Infer data types for specific columns using various methods"
)
@require_permissions(["schema_discovery:read"])
async def infer_types(
    request: SchemaInferenceRequest,
    current_user: dict = Depends(get_current_user)
):
    """Infer data types for specified columns."""
    try:
        logger.info(f"Type inference request for dataset {request.dataset_id}")
        
        column_inferences = []
        semantic_inferences = []
        format_inferences = []
        confidence_summary = {}
        
        # Process each column
        for column_data in request.columns:
            column_name = column_data["name"]
            values = column_data["values"]
            
            # Perform type inference
            type_inference = _perform_type_inference(
                column_name, values, request.inference_method, request.confidence_threshold
            )
            column_inferences.append(type_inference)
            
            # Perform semantic inference
            semantic_inference = _perform_semantic_inference(column_name, values)
            semantic_inferences.append(semantic_inference)
            
            # Perform format inference
            format_inference = _perform_format_inference(column_name, values)
            format_inferences.append(format_inference)
            
            # Store confidence score
            confidence_summary[column_name] = type_inference.get("confidence", 0.0)
        
        return SchemaInferenceResponse(
            inference_id=str(uuid4()),
            dataset_id=request.dataset_id,
            inference_method=request.inference_method,
            column_inferences=column_inferences,
            semantic_inferences=semantic_inferences,
            format_inferences=format_inferences,
            confidence_summary=confidence_summary,
            processing_time_ms=856.3,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Type inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Type inference failed"
        )


@router.post(
    "/compare-schemas",
    response_model=SchemaComparisonResponse,
    summary="Compare two schemas",
    description="Compare two schemas and identify differences and migration paths"
)
@require_permissions(["schema_discovery:read"])
async def compare_schemas(
    request: SchemaComparisonRequest,
    current_user: dict = Depends(get_current_user)
):
    """Compare two schemas and identify differences."""
    try:
        logger.info(f"Schema comparison between {request.source_schema_id} and {request.target_schema_id}")
        
        # Mock implementation - in real implementation, this would:
        # 1. Load both schemas from storage
        # 2. Compare structures, types, and constraints
        # 3. Generate migration suggestions
        
        # Mock comparison results
        differences = [
            {
                "type": "type_change",
                "column": "age",
                "source_type": "string",
                "target_type": "integer",
                "severity": "medium",
                "description": "Column 'age' changed from string to integer"
            },
            {
                "type": "constraint_added",
                "column": "email",
                "constraint": "unique",
                "severity": "low",
                "description": "Unique constraint added to 'email' column"
            }
        ]
        
        additions = [
            {
                "column": "created_at",
                "type": "timestamp",
                "nullable": False,
                "description": "New timestamp column added"
            }
        ]
        
        removals = [
            {
                "column": "legacy_id",
                "type": "string",
                "description": "Legacy ID column removed"
            }
        ]
        
        modifications = [
            {
                "column": "name",
                "change_type": "length_increased",
                "source_length": 50,
                "target_length": 100,
                "description": "Maximum length increased from 50 to 100"
            }
        ]
        
        migration_suggestions = [
            {
                "type": "data_migration",
                "column": "age",
                "suggestion": "Convert string age values to integers",
                "sql": "UPDATE table SET age = CAST(age AS INTEGER)",
                "priority": "high"
            },
            {
                "type": "constraint_migration",
                "column": "email",
                "suggestion": "Add unique constraint after data cleanup",
                "sql": "ALTER TABLE table ADD CONSTRAINT unique_email UNIQUE (email)",
                "priority": "medium"
            }
        ]
        
        # Calculate compatibility score
        compatibility_score = _calculate_compatibility_score(differences, additions, removals, modifications)
        
        return SchemaComparisonResponse(
            comparison_id=str(uuid4()),
            source_schema_id=request.source_schema_id,
            target_schema_id=request.target_schema_id,
            comparison_mode=request.comparison_mode,
            compatibility_score=compatibility_score,
            differences=differences,
            additions=additions,
            removals=removals,
            modifications=modifications,
            migration_suggestions=migration_suggestions,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Schema comparison failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Schema comparison failed"
        )


@router.get(
    "/schemas/{schema_id}",
    response_model=SchemaDiscoveryResponse,
    summary="Get schema by ID",
    description="Retrieve a previously discovered schema by its ID"
)
@require_permissions(["schema_discovery:read"])
async def get_schema(
    schema_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get schema by ID."""
    try:
        logger.info(f"Retrieving schema {schema_id}")
        
        # Mock implementation - in real implementation, this would:
        # 1. Query the database for the schema
        # 2. Return the schema information
        
        # Mock response
        return SchemaDiscoveryResponse(
            discovery_id=schema_id,
            dataset_id="sample_dataset",
            schema_version="1.0.0",
            discovery_mode="comprehensive",
            columns=[
                {
                    "name": "id",
                    "inferred_type": "integer",
                    "nullable": False,
                    "unique": True,
                    "primary_key": True,
                    "samples": [1, 2, 3, 4, 5]
                }
            ],
            relationships=[],
            constraints=[
                {
                    "type": "primary_key",
                    "columns": ["id"],
                    "confidence": 1.0
                }
            ],
            quality_metrics={
                "completeness": 0.98,
                "consistency": 0.92,
                "conformity": 0.87,
                "uniqueness": 0.95
            },
            recommendations=[
                {
                    "type": "constraint",
                    "description": "Add primary key constraint on id column",
                    "priority": "high",
                    "sql": "ALTER TABLE customer_data ADD CONSTRAINT pk_id PRIMARY KEY (id)"
                }
            ],
            processing_time_ms=2345.7,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve schema: {e}")
        raise HTTPException(
            status_code=404,
            detail="Schema not found"
        )


@router.get(
    "/schemas",
    summary="List schemas",
    description="List all discovered schemas with filtering options"
)
@require_permissions(["schema_discovery:read"])
async def list_schemas(
    dataset_id: Optional[str] = None,
    discovery_mode: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """List schemas with optional filtering."""
    try:
        logger.info("Listing schemas")
        
        # Mock implementation - in real implementation, this would:
        # 1. Query the database with filters
        # 2. Return paginated results
        
        # Mock response
        schemas = [
            {
                "discovery_id": "sch_123456789",
                "dataset_id": "customer_data_2024",
                "schema_version": "1.0.0",
                "discovery_mode": "comprehensive",
                "columns_count": 5,
                "relationships_count": 2,
                "constraints_count": 3,
                "quality_score": 0.93,
                "created_at": "2024-01-15T10:30:00Z"
            },
            {
                "discovery_id": "sch_987654321",
                "dataset_id": "transaction_data_2024",
                "schema_version": "1.0.0",
                "discovery_mode": "quick",
                "columns_count": 8,
                "relationships_count": 1,
                "constraints_count": 2,
                "quality_score": 0.87,
                "created_at": "2024-01-15T09:15:00Z"
            }
        ]
        
        # Apply filters
        if dataset_id:
            schemas = [s for s in schemas if s["dataset_id"] == dataset_id]
        if discovery_mode:
            schemas = [s for s in schemas if s["discovery_mode"] == discovery_mode]
        
        # Apply pagination
        total_count = len(schemas)
        paginated_schemas = schemas[offset:offset + limit]
        
        return {
            "schemas": paginated_schemas,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to list schemas: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list schemas"
        )


# Helper functions

def _infer_type(value: Any) -> str:
    """Infer the type of a value."""
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        return "string"
    elif value is None:
        return "null"
    else:
        return "unknown"


def _check_nullable(column_name: str, data: List[Dict[str, Any]]) -> bool:
    """Check if a column is nullable."""
    for record in data:
        if record.get(column_name) is None:
            return True
    return False


def _check_unique(column_name: str, data: List[Dict[str, Any]]) -> bool:
    """Check if a column has unique values."""
    values = [record.get(column_name) for record in data]
    return len(values) == len(set(values))


def _get_samples(column_name: str, data: List[Dict[str, Any]], include_samples: bool) -> List[Any]:
    """Get sample values from a column."""
    if not include_samples:
        return []
    
    values = [record.get(column_name) for record in data]
    return list(set(values))[:5]  # Return up to 5 unique samples


def _infer_semantic_type(column_name: str, value: Any) -> str:
    """Infer semantic type from column name and value."""
    column_lower = column_name.lower()
    
    if "email" in column_lower:
        return "email_address"
    elif "name" in column_lower:
        return "person_name"
    elif "phone" in column_lower:
        return "phone_number"
    elif "address" in column_lower:
        return "address"
    elif "date" in column_lower:
        return "date"
    elif "id" in column_lower:
        return "identifier"
    else:
        return "general"


def _get_numeric_stats(column_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get numeric statistics for a column."""
    values = [record.get(column_name) for record in data if record.get(column_name) is not None]
    
    if not values:
        return {}
    
    return {
        "min_value": min(values),
        "max_value": max(values),
        "avg_value": sum(values) / len(values)
    }


def _get_string_stats(column_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get string statistics for a column."""
    values = [record.get(column_name) for record in data if record.get(column_name) is not None]
    
    if not values:
        return {}
    
    lengths = [len(str(v)) for v in values]
    return {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": sum(lengths) / len(lengths)
    }


def _infer_relationships(columns: List[Dict[str, Any]], data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Infer relationships between columns."""
    relationships = []
    
    # Simple functional dependency detection
    for source_col in columns:
        if source_col["unique"]:
            for target_col in columns:
                if source_col["name"] != target_col["name"]:
                    relationships.append({
                        "type": "functional_dependency",
                        "source": source_col["name"],
                        "target": target_col["name"],
                        "confidence": 1.0 if source_col["primary_key"] else 0.8,
                        "description": f"{source_col['name']} functionally determines {target_col['name']}"
                    })
    
    return relationships


def _generate_constraints(columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate constraints based on column analysis."""
    constraints = []
    
    for column in columns:
        if column["primary_key"]:
            constraints.append({
                "type": "primary_key",
                "columns": [column["name"]],
                "confidence": 1.0
            })
        elif column["unique"]:
            constraints.append({
                "type": "unique",
                "columns": [column["name"]],
                "confidence": 0.95
            })
        
        if not column["nullable"]:
            constraints.append({
                "type": "not_null",
                "columns": [column["name"]],
                "confidence": 1.0
            })
    
    return constraints


def _calculate_quality_metrics(columns: List[Dict[str, Any]], data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate schema quality metrics."""
    # Simple quality metric calculation
    total_columns = len(columns)
    nullable_columns = sum(1 for col in columns if col["nullable"])
    unique_columns = sum(1 for col in columns if col["unique"])
    
    completeness = (total_columns - nullable_columns) / total_columns if total_columns > 0 else 1.0
    uniqueness = unique_columns / total_columns if total_columns > 0 else 0.0
    
    return {
        "completeness": completeness,
        "consistency": 0.92,  # Mock value
        "conformity": 0.87,   # Mock value
        "uniqueness": uniqueness
    }


def _generate_schema_recommendations(columns: List[Dict[str, Any]], constraints: List[Dict[str, Any]], quality_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    """Generate schema improvement recommendations."""
    recommendations = []
    
    # Primary key recommendations
    primary_key_exists = any(constraint["type"] == "primary_key" for constraint in constraints)
    if not primary_key_exists:
        id_column = next((col for col in columns if col["name"].lower() == "id"), None)
        if id_column:
            recommendations.append({
                "type": "constraint",
                "description": "Add primary key constraint on id column",
                "priority": "high",
                "sql": "ALTER TABLE table_name ADD CONSTRAINT pk_id PRIMARY KEY (id)"
            })
    
    # Unique constraint recommendations
    for column in columns:
        if column["unique"] and not any(
            constraint["type"] == "unique" and column["name"] in constraint["columns"]
            for constraint in constraints
        ):
            recommendations.append({
                "type": "constraint",
                "description": f"Add unique constraint on {column['name']} column",
                "priority": "medium",
                "sql": f"ALTER TABLE table_name ADD CONSTRAINT unique_{column['name']} UNIQUE ({column['name']})"
            })
    
    # Data quality recommendations
    if quality_metrics.get("completeness", 1.0) < 0.9:
        recommendations.append({
            "type": "data_quality",
            "description": "Improve data completeness by addressing missing values",
            "priority": "medium",
            "action": "address_missing_values"
        })
    
    return recommendations


def _perform_type_inference(column_name: str, values: List[Any], method: str, threshold: float) -> Dict[str, Any]:
    """Perform type inference on a column."""
    # Simple type inference based on values
    non_null_values = [v for v in values if v is not None and v != ""]
    
    if not non_null_values:
        return {
            "column": column_name,
            "inferred_type": "unknown",
            "confidence": 0.0,
            "method": method,
            "issues": ["No valid values found"]
        }
    
    # Count type occurrences
    type_counts = {}
    for value in non_null_values:
        value_type = _infer_type(value)
        type_counts[value_type] = type_counts.get(value_type, 0) + 1
    
    # Find most common type
    most_common_type = max(type_counts, key=type_counts.get)
    confidence = type_counts[most_common_type] / len(non_null_values)
    
    return {
        "column": column_name,
        "inferred_type": most_common_type,
        "confidence": confidence,
        "method": method,
        "type_distribution": type_counts,
        "issues": [] if confidence >= threshold else ["Low confidence in type inference"]
    }


def _perform_semantic_inference(column_name: str, values: List[Any]) -> Dict[str, Any]:
    """Perform semantic type inference."""
    semantic_type = _infer_semantic_type(column_name, values[0] if values else None)
    
    return {
        "column": column_name,
        "semantic_type": semantic_type,
        "confidence": 0.85,
        "indicators": [column_name.lower()]
    }


def _perform_format_inference(column_name: str, values: List[Any]) -> Dict[str, Any]:
    """Perform format pattern inference."""
    # Simple format inference
    string_values = [str(v) for v in values if v is not None]
    
    if not string_values:
        return {
            "column": column_name,
            "format": "unknown",
            "confidence": 0.0,
            "pattern": None
        }
    
    # Check for common patterns
    if all("@" in v for v in string_values):
        return {
            "column": column_name,
            "format": "email",
            "confidence": 0.95,
            "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        }
    elif all(v.isdigit() for v in string_values):
        return {
            "column": column_name,
            "format": "numeric_string",
            "confidence": 0.90,
            "pattern": r"^\d+$"
        }
    else:
        return {
            "column": column_name,
            "format": "text",
            "confidence": 0.70,
            "pattern": None
        }


def _calculate_compatibility_score(differences: List[Dict[str, Any]], additions: List[Dict[str, Any]], removals: List[Dict[str, Any]], modifications: List[Dict[str, Any]]) -> float:
    """Calculate compatibility score between schemas."""
    # Simple compatibility scoring
    total_changes = len(differences) + len(additions) + len(removals) + len(modifications)
    
    if total_changes == 0:
        return 1.0
    
    # Weight different types of changes
    compatibility_impact = 0
    compatibility_impact += len(removals) * 0.3  # Removals have high impact
    compatibility_impact += len(differences) * 0.2  # Type changes have medium impact
    compatibility_impact += len(modifications) * 0.1  # Modifications have low impact
    compatibility_impact += len(additions) * 0.05  # Additions have minimal impact
    
    # Calculate score (higher is better)
    return max(0.0, 1.0 - (compatibility_impact / 10.0))
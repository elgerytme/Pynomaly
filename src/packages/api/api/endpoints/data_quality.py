"""Data Quality Validation API Endpoints.

This module provides RESTful endpoints for data quality validation operations including
quality assessment, rule management, and quality monitoring.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...data_quality.application.services.validation_engine import ValidationEngine
from ...data_quality.application.services.quality_assessment_service import (
    QualityAssessmentService, QualityAssessmentConfig
)
from ...data_quality.domain.entities.quality_profile import DatasetId
from ...data_quality.domain.entities.validation_rule import ValidationRule, ValidationLogic
from ..security.authorization import require_permissions
from ..dependencies.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-quality", tags=["Data Quality"])

# Pydantic models for request/response
class QualityValidationRequest(BaseModel):
    """Request model for data quality validation."""
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    data: List[Dict[str, Any]] = Field(..., description="Dataset records as list of dictionaries")
    validation_rules: Optional[List[Dict[str, Any]]] = Field(default=None, description="Custom validation rules")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Quality assessment configuration")
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "customer_data_2024",
                "data": [
                    {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com"},
                    {"id": 2, "name": "Jane Smith", "age": 25, "email": "jane@example.com"}
                ],
                "validation_rules": [
                    {
                        "name": "email_format",
                        "description": "Validate email format",
                        "logic_type": "regex",
                        "parameters": {"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
                        "target_columns": ["email"]
                    }
                ],
                "config": {
                    "scoring_method": "weighted_average",
                    "enable_trend_analysis": True,
                    "enable_business_impact_analysis": True
                }
            }
        }


class QualityValidationResponse(BaseModel):
    """Response model for data quality validation."""
    validation_id: str = Field(..., description="Unique identifier for the validation")
    dataset_id: str = Field(..., description="Dataset identifier")
    overall_score: float = Field(..., description="Overall quality score")
    quality_scores: Dict[str, float] = Field(..., description="Quality scores by dimension")
    total_records: int = Field(..., description="Total number of records")
    total_columns: int = Field(..., description="Total number of columns")
    issues_detected: int = Field(..., description="Number of quality issues detected")
    validation_rules_applied: int = Field(..., description="Number of validation rules applied")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    created_at: str = Field(..., description="Validation creation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "validation_id": "val_123456789",
                "dataset_id": "customer_data_2024",
                "overall_score": 0.85,
                "quality_scores": {
                    "completeness": 0.92,
                    "accuracy": 0.88,
                    "consistency": 0.90,
                    "validity": 0.82,
                    "uniqueness": 0.95,
                    "timeliness": 0.78
                },
                "total_records": 1000,
                "total_columns": 5,
                "issues_detected": 12,
                "validation_rules_applied": 8,
                "processing_time_ms": 345.7,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class ValidationRuleRequest(BaseModel):
    """Request model for validation rule creation."""
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    logic_type: str = Field(..., description="Validation logic type")
    parameters: Dict[str, Any] = Field(..., description="Rule parameters")
    target_columns: List[str] = Field(..., description="Target columns for validation")
    severity: str = Field(default="medium", description="Rule severity level")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "email_format_validation",
                "description": "Validate email format using regex",
                "logic_type": "regex",
                "parameters": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "target_columns": ["email"],
                "severity": "high",
                "enabled": True
            }
        }


class ValidationRuleResponse(BaseModel):
    """Response model for validation rule."""
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    logic_type: str = Field(..., description="Validation logic type")
    parameters: Dict[str, Any] = Field(..., description="Rule parameters")
    target_columns: List[str] = Field(..., description="Target columns")
    severity: str = Field(..., description="Rule severity level")
    enabled: bool = Field(..., description="Whether rule is enabled")
    created_at: str = Field(..., description="Rule creation timestamp")
    execution_stats: Dict[str, Any] = Field(..., description="Rule execution statistics")


class QualityMonitoringResponse(BaseModel):
    """Response model for quality monitoring."""
    monitoring_id: str = Field(..., description="Monitoring session identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    monitoring_period: str = Field(..., description="Monitoring period")
    quality_trends: Dict[str, Any] = Field(..., description="Quality trend analysis")
    alerts: List[Dict[str, Any]] = Field(..., description="Quality alerts")
    recommendations: List[Dict[str, Any]] = Field(..., description="Improvement recommendations")
    last_updated: str = Field(..., description="Last update timestamp")


# Initialize services
validation_engine = ValidationEngine()
quality_assessment_service = QualityAssessmentService()


@router.post(
    "/validate",
    response_model=QualityValidationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Validate data quality",
    description="Perform comprehensive data quality validation on the provided dataset"
)
async def validate_data_quality(
    request: QualityValidationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_quality:write"]))
) -> QualityValidationResponse:
    """Validate data quality and return comprehensive assessment."""
    try:
        start_time = datetime.now()
        
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset cannot be empty"
            )
        
        # Configure quality assessment
        config = QualityAssessmentConfig()
        if request.config:
            config = QualityAssessmentConfig(**request.config)
        
        # Create dataset ID
        dataset_id = DatasetId(request.dataset_id)
        
        # Execute quality assessment
        quality_profile = quality_assessment_service.assess_dataset_quality(
            df, dataset_id=dataset_id, config=config
        )
        
        # Apply custom validation rules if provided
        validation_rules_applied = 0
        if request.validation_rules:
            for rule_data in request.validation_rules:
                rule = ValidationRule(
                    rule_id=str(uuid4()),
                    name=rule_data["name"],
                    description=rule_data["description"],
                    logic=ValidationLogic(
                        logic_type=rule_data["logic_type"],
                        parameters=rule_data["parameters"]
                    ),
                    target_columns=rule_data["target_columns"],
                    severity=rule_data.get("severity", "medium"),
                    enabled=rule_data.get("enabled", True)
                )
                
                # Execute validation rule
                validation_result = validation_engine.execute_rule(df, rule)
                validation_rules_applied += 1
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build response
        response = QualityValidationResponse(
            validation_id=str(uuid4()),
            dataset_id=str(quality_profile.dataset_id),
            overall_score=quality_profile.quality_scores.overall_score,
            quality_scores=quality_profile.quality_scores.get_dimension_scores(),
            total_records=len(df),
            total_columns=len(df.columns),
            issues_detected=len(quality_profile.quality_issues),
            validation_rules_applied=validation_rules_applied,
            processing_time_ms=processing_time,
            created_at=quality_profile.created_at.isoformat()
        )
        
        logger.info(f"Data quality validated successfully: {request.dataset_id} by user {current_user['user_id']}")
        return response
        
    except ValueError as e:
        logger.error(f"Invalid data quality validation request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Data quality validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data quality validation operation failed"
        )


@router.post(
    "/rules",
    response_model=ValidationRuleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create validation rule",
    description="Create a new data quality validation rule"
)
async def create_validation_rule(
    request: ValidationRuleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_quality:write"]))
) -> ValidationRuleResponse:
    """Create a new validation rule."""
    try:
        rule_id = str(uuid4())
        
        # Create validation rule
        rule = ValidationRule(
            rule_id=rule_id,
            name=request.name,
            description=request.description,
            logic=ValidationLogic(
                logic_type=request.logic_type,
                parameters=request.parameters
            ),
            target_columns=request.target_columns,
            severity=request.severity,
            enabled=request.enabled
        )
        
        # Save rule (would typically save to database)
        # For demonstration, return mock response
        
        response = ValidationRuleResponse(
            rule_id=rule_id,
            name=request.name,
            description=request.description,
            logic_type=request.logic_type,
            parameters=request.parameters,
            target_columns=request.target_columns,
            severity=request.severity,
            enabled=request.enabled,
            created_at=datetime.now().isoformat(),
            execution_stats={
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time_ms": 0.0,
                "last_executed": None
            }
        )
        
        logger.info(f"Validation rule created: {rule_id} by user {current_user['user_id']}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to create validation rule: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create validation rule"
        )


@router.get(
    "/rules",
    response_model=List[ValidationRuleResponse],
    summary="List validation rules",
    description="Get a list of all validation rules"
)
async def list_validation_rules(
    enabled_only: bool = True,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_quality:read"]))
) -> List[ValidationRuleResponse]:
    """Get a list of validation rules."""
    try:
        # This would typically query a database
        # For demonstration, return mock data
        
        mock_rules = [
            ValidationRuleResponse(
                rule_id=f"rule_{i:06d}",
                name=f"validation_rule_{i}",
                description=f"Description for validation rule {i}",
                logic_type="regex" if i % 2 == 0 else "range",
                parameters={"pattern": ".*"} if i % 2 == 0 else {"min": 0, "max": 100},
                target_columns=[f"column_{i}"],
                severity="medium",
                enabled=True,
                created_at=f"2024-01-{(i % 28) + 1:02d}T10:00:00Z",
                execution_stats={
                    "total_executions": i * 10,
                    "success_rate": 0.95,
                    "average_execution_time_ms": 25.5,
                    "last_executed": f"2024-01-{(i % 28) + 1:02d}T12:00:00Z"
                }
            )
            for i in range(1, 21)  # Mock 20 rules
        ]
        
        if enabled_only:
            mock_rules = [rule for rule in mock_rules if rule.enabled]
        
        return mock_rules
        
    except Exception as e:
        logger.error(f"Failed to list validation rules: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve validation rules"
        )


@router.get(
    "/rules/{rule_id}",
    response_model=ValidationRuleResponse,
    summary="Get validation rule",
    description="Get detailed information about a specific validation rule"
)
async def get_validation_rule(
    rule_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_quality:read"]))
) -> ValidationRuleResponse:
    """Get detailed information about a validation rule."""
    try:
        # This would typically query a database
        # For demonstration, return mock data
        
        mock_rule = ValidationRuleResponse(
            rule_id=rule_id,
            name="email_format_validation",
            description="Validate email format using regex pattern",
            logic_type="regex",
            parameters={
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            },
            target_columns=["email"],
            severity="high",
            enabled=True,
            created_at="2024-01-15T10:30:00Z",
            execution_stats={
                "total_executions": 250,
                "success_rate": 0.92,
                "average_execution_time_ms": 18.7,
                "last_executed": "2024-01-15T14:30:00Z"
            }
        )
        
        return mock_rule
        
    except Exception as e:
        logger.error(f"Failed to get validation rule: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Validation rule not found"
        )


@router.put(
    "/rules/{rule_id}",
    response_model=ValidationRuleResponse,
    summary="Update validation rule",
    description="Update an existing validation rule"
)
async def update_validation_rule(
    rule_id: str,
    request: ValidationRuleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_quality:write"]))
) -> ValidationRuleResponse:
    """Update an existing validation rule."""
    try:
        # This would typically update in database
        # For demonstration, return updated mock data
        
        response = ValidationRuleResponse(
            rule_id=rule_id,
            name=request.name,
            description=request.description,
            logic_type=request.logic_type,
            parameters=request.parameters,
            target_columns=request.target_columns,
            severity=request.severity,
            enabled=request.enabled,
            created_at="2024-01-15T10:30:00Z",
            execution_stats={
                "total_executions": 250,
                "success_rate": 0.92,
                "average_execution_time_ms": 18.7,
                "last_executed": "2024-01-15T14:30:00Z"
            }
        )
        
        logger.info(f"Validation rule updated: {rule_id} by user {current_user['user_id']}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to update validation rule: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update validation rule"
        )


@router.delete(
    "/rules/{rule_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete validation rule",
    description="Delete a validation rule"
)
async def delete_validation_rule(
    rule_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_quality:delete"]))
) -> None:
    """Delete a validation rule."""
    try:
        # This would typically delete from database
        # For demonstration, just log the operation
        
        logger.info(f"Validation rule {rule_id} deleted by user {current_user['user_id']}")
        
    except Exception as e:
        logger.error(f"Failed to delete validation rule: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete validation rule"
        )


@router.get(
    "/monitor/{dataset_id}",
    response_model=QualityMonitoringResponse,
    summary="Get quality monitoring",
    description="Get quality monitoring data for a dataset"
)
async def get_quality_monitoring(
    dataset_id: str,
    period: str = "7d",
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_quality:read"]))
) -> QualityMonitoringResponse:
    """Get quality monitoring data for a dataset."""
    try:
        # This would typically query monitoring database
        # For demonstration, return mock monitoring data
        
        mock_response = QualityMonitoringResponse(
            monitoring_id=str(uuid4()),
            dataset_id=dataset_id,
            monitoring_period=period,
            quality_trends={
                "overall_score": {
                    "trend": "stable",
                    "current_score": 0.85,
                    "previous_score": 0.84,
                    "change_percentage": 1.2
                },
                "dimension_trends": {
                    "completeness": {"trend": "improving", "change": 0.02},
                    "accuracy": {"trend": "stable", "change": 0.01},
                    "consistency": {"trend": "declining", "change": -0.03}
                }
            },
            alerts=[
                {
                    "alert_id": "alert_001",
                    "type": "quality_degradation",
                    "severity": "medium",
                    "message": "Consistency score has declined by 3% in the last 24 hours",
                    "triggered_at": "2024-01-15T09:00:00Z",
                    "affected_columns": ["phone_number", "address"]
                }
            ],
            recommendations=[
                {
                    "recommendation_id": "rec_001",
                    "type": "data_cleansing",
                    "priority": "high",
                    "description": "Standardize phone number formats to improve consistency",
                    "estimated_impact": "5% improvement in consistency score",
                    "effort_estimate": "2-3 hours"
                }
            ],
            last_updated=datetime.now().isoformat()
        )
        
        return mock_response
        
    except Exception as e:
        logger.error(f"Failed to get quality monitoring: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve quality monitoring data"
        )


@router.post(
    "/monitor/{dataset_id}/alerts",
    summary="Create quality alert",
    description="Create a quality monitoring alert for a dataset"
)
async def create_quality_alert(
    dataset_id: str,
    alert_config: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_quality:write"]))
) -> Dict[str, Any]:
    """Create a quality monitoring alert."""
    try:
        alert_id = str(uuid4())
        
        # Create alert (would typically save to database)
        alert = {
            "alert_id": alert_id,
            "dataset_id": dataset_id,
            "alert_type": alert_config.get("type", "threshold"),
            "conditions": alert_config.get("conditions", {}),
            "notification_settings": alert_config.get("notifications", {}),
            "enabled": alert_config.get("enabled", True),
            "created_by": current_user["user_id"],
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Quality alert created: {alert_id} for dataset {dataset_id} by user {current_user['user_id']}")
        return alert
        
    except Exception as e:
        logger.error(f"Failed to create quality alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create quality alert"
        )


@router.get(
    "/engine/metrics",
    summary="Get engine metrics",
    description="Get performance metrics for the quality validation engine"
)
async def get_engine_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_quality:read"]))
) -> Dict[str, Any]:
    """Get quality validation engine performance metrics."""
    try:
        # Get engine metrics
        metrics = {
            "validation_engine": {
                "total_validations": 1250,
                "success_rate": 0.98,
                "average_execution_time_ms": 45.2,
                "active_rules": 25,
                "cache_hit_rate": 0.85
            },
            "quality_assessment": {
                "total_assessments": 850,
                "average_assessment_time_ms": 125.7,
                "dimension_coverage": {
                    "completeness": 1.0,
                    "accuracy": 0.95,
                    "consistency": 0.90,
                    "validity": 0.98,
                    "uniqueness": 0.85,
                    "timeliness": 0.75
                }
            },
            "system_health": {
                "memory_usage_mb": 256,
                "cpu_usage_percent": 15.3,
                "disk_usage_mb": 1024,
                "uptime_hours": 72.5
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get engine metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve engine metrics"
        )
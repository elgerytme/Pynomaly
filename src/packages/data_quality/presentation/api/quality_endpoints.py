"""Data Quality API endpoints for validation, cleansing, and monitoring."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pandas as pd
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...application.services.validation_engine import ValidationEngine
from ...application.services.data_cleansing_engine import DataCleansingEngine, CleansingAction, CleansingStrategy
from ...application.services.quality_monitoring_service import (
    QualityMonitoringService, MonitoringConfiguration, AlertSeverity,
    MonitoringStatus, MetricType
)
from ...domain.entities.quality_rule import (
    QualityRule, RuleType, LogicType, ValidationLogic, QualityThreshold,
    Severity, UserId, DatasetId, RuleId
)

logger = structlog.get_logger(__name__)

# Request Models
class CreateRuleRequest(BaseModel):
    """Request model for creating quality rules."""
    rule_name: str = Field(..., description="Name of the quality rule")
    rule_type: str = Field(..., description="Type of rule (completeness, uniqueness, etc.)")
    target_columns: List[str] = Field(default_factory=list, description="Target columns for validation")
    validation_logic: Dict[str, Any] = Field(..., description="Validation logic configuration")
    thresholds: Dict[str, float] = Field(..., description="Quality thresholds")
    severity: str = Field(default="medium", description="Rule severity level")
    description: Optional[str] = Field(None, description="Rule description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rule_name": "Email Format Validation",
                "rule_type": "validity",
                "target_columns": ["email"],
                "validation_logic": {
                    "logic_type": "regex",
                    "expression": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                    "parameters": {},
                    "error_message_template": "Invalid email format: {value}"
                },
                "thresholds": {
                    "pass_rate_threshold": 0.95,
                    "warning_threshold": 0.90,
                    "critical_threshold": 0.80
                },
                "severity": "high",
                "description": "Validates email addresses against standard format"
            }
        }

class ValidationRequest(BaseModel):
    """Request model for dataset validation."""
    dataset_id: str = Field(..., description="ID of the dataset to validate")
    rule_ids: List[str] = Field(..., description="List of rule IDs to apply")
    data: List[Dict[str, Any]] = Field(..., description="Dataset records")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
                "rule_ids": ["rule-001", "rule-002"],
                "data": [
                    {"id": 1, "email": "john@example.com", "age": 30},
                    {"id": 2, "email": "invalid-email", "age": 25}
                ]
            }
        }

class CleansingRequest(BaseModel):
    """Request model for data cleansing."""
    dataset_name: str = Field(..., description="Name of the dataset")
    data: List[Dict[str, Any]] = Field(..., description="Dataset records")
    cleansing_config: Dict[str, Any] = Field(..., description="Cleansing configuration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_name": "customer_data",
                "data": [
                    {"id": 1, "name": "John Doe", "email": "JOHN@EXAMPLE.COM"},
                    {"id": 2, "name": "jane smith", "email": "jane@example.com"}
                ],
                "cleansing_config": {
                    "actions": ["format_standardization", "text_cleaning"],
                    "columns": {
                        "email": {"strategy": "standardize", "to_lower": True},
                        "name": {"strategy": "standardize", "title_case": True}
                    }
                }
            }
        }

class MonitoringConfigRequest(BaseModel):
    """Request model for monitoring configuration."""
    dataset_id: str = Field(..., description="Dataset ID to monitor")
    rule_ids: List[str] = Field(..., description="Rule IDs to monitor")
    data_source_config: Dict[str, Any] = Field(..., description="Data source configuration")
    monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Monitoring settings")

# Response Models
class ValidationErrorResponse(BaseModel):
    """Response model for validation errors."""
    rule_id: str
    row_identifier: Optional[str]
    column_name: Optional[str]
    invalid_value: Optional[str]
    error_message: str
    error_code: str

class ValidationResultResponse(BaseModel):
    """Response model for validation results."""
    rule_id: str
    dataset_id: str
    status: str
    total_records: int
    records_passed: int
    records_failed: int
    pass_rate: float
    validation_errors: List[ValidationErrorResponse]
    execution_time_seconds: float

class CleansingResultResponse(BaseModel):
    """Response model for cleansing results."""
    action: str
    column_name: str
    records_affected: int
    strategy_used: str
    success: bool
    error_message: Optional[str]

class CleansingReportResponse(BaseModel):
    """Response model for cleansing reports."""
    dataset_name: str
    total_records: int
    total_columns: int
    cleansing_results: List[CleansingResultResponse]
    execution_time_seconds: float
    overall_success: bool
    quality_improvement: Dict[str, float]

class QualityAlertResponse(BaseModel):
    """Response model for quality alerts."""
    alert_id: str
    rule_id: str
    dataset_id: str
    severity: str
    alert_type: str
    message: str
    current_value: float
    threshold_value: float
    triggered_at: str
    acknowledged: bool

class MonitoringDashboardResponse(BaseModel):
    """Response model for monitoring dashboard."""
    monitoring_status: str
    monitored_datasets: int
    total_active_rules: int
    quality_summary: Dict[str, Any]
    alert_summary: Dict[str, Any]
    trends: Dict[str, Any]
    last_updated: str

# Router setup
router = APIRouter(
    prefix="/data-quality",
    tags=["Data Quality"],
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"},
    },
)

# Global instances (in production, use dependency injection)
validation_engine = ValidationEngine()
cleansing_engine = DataCleansingEngine()
monitoring_service = QualityMonitoringService(MonitoringConfiguration())

# Helper functions
def _create_quality_rule_from_request(request: CreateRuleRequest, created_by: UserId) -> QualityRule:
    """Create QualityRule entity from request."""
    return QualityRule(
        rule_id=RuleId(value=uuid4()),
        rule_name=request.rule_name,
        rule_type=RuleType(request.rule_type),
        target_columns=request.target_columns,
        validation_logic=ValidationLogic(
            logic_type=LogicType(request.validation_logic["logic_type"]),
            expression=request.validation_logic["expression"],
            parameters=request.validation_logic.get("parameters", {}),
            error_message_template=request.validation_logic.get("error_message_template", "Validation failed")
        ),
        thresholds=QualityThreshold(
            pass_rate_threshold=request.thresholds["pass_rate_threshold"],
            warning_threshold=request.thresholds["warning_threshold"],
            critical_threshold=request.thresholds["critical_threshold"]
        ),
        severity=Severity(request.severity),
        description=request.description,
        created_by=created_by,
        is_enabled=True
    )

# Endpoints
@router.post(
    "/rules",
    summary="Create Quality Rule",
    description="Create a new data quality validation rule.",
    responses={
        201: {"description": "Rule created successfully"},
        400: {"description": "Invalid rule configuration"}
    }
)
async def create_rule(request: CreateRuleRequest):
    """Create a new quality rule."""
    try:
        # Create rule entity
        rule = _create_quality_rule_from_request(request, UserId(value=uuid4()))
        
        # Store rule (in production, use repository)
        # For now, just return the created rule info
        
        return {
            "rule_id": str(rule.rule_id.value),
            "rule_name": rule.rule_name,
            "rule_type": rule.rule_type.value,
            "status": "created",
            "message": "Quality rule created successfully"
        }
        
    except Exception as e:
        logger.error("Failed to create rule", error=str(e))
        raise HTTPException(status_code=400, detail=f"Failed to create rule: {str(e)}")

@router.post(
    "/validate",
    response_model=List[ValidationResultResponse],
    summary="Validate Dataset",
    description="Execute validation rules against a dataset.",
    responses={
        200: {"description": "Validation completed successfully"},
        400: {"description": "Invalid validation request"}
    }
)
async def validate_dataset(request: ValidationRequest):
    """Validate dataset with specified rules."""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset cannot be empty")
        
        # Create mock rules for demonstration (in production, fetch from repository)
        rules = []
        for rule_id in request.rule_ids:
            # Create basic completeness rule as example
            rule = QualityRule(
                rule_id=RuleId(value=UUID(rule_id) if len(rule_id) == 36 else uuid4()),
                rule_name=f"Rule {rule_id}",
                rule_type=RuleType.COMPLETENESS,
                target_columns=list(df.columns),
                validation_logic=ValidationLogic(
                    logic_type=LogicType.PYTHON,
                    expression="df.notna().all(axis=1)",
                    parameters={},
                    error_message_template="Missing values found"
                ),
                thresholds=QualityThreshold(
                    pass_rate_threshold=0.95,
                    warning_threshold=0.90,
                    critical_threshold=0.80
                ),
                severity=Severity.MEDIUM,
                created_by=UserId(value=uuid4()),
                is_enabled=True
            )
            rules.append(rule)
        
        # Execute validation
        results = validation_engine.validate_dataset(
            rules=rules,
            df=df,
            dataset_id=DatasetId(value=UUID(request.dataset_id))
        )
        
        # Convert to response format
        response_results = []
        for result in results:
            error_responses = [
                ValidationErrorResponse(
                    rule_id=str(error.rule_id.value),
                    row_identifier=error.row_identifier,
                    column_name=error.column_name,
                    invalid_value=str(error.invalid_value) if error.invalid_value is not None else None,
                    error_message=error.error_message,
                    error_code=error.error_code
                )
                for error in result.validation_errors
            ]
            
            response_results.append(ValidationResultResponse(
                rule_id=str(result.rule_id.value),
                dataset_id=str(result.dataset_id.value),
                status=result.status.value,
                total_records=result.total_records,
                records_passed=result.records_passed,
                records_failed=result.records_failed,
                pass_rate=result.pass_rate,
                validation_errors=error_responses,
                execution_time_seconds=result.execution_time_seconds
            ))
        
        return response_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post(
    "/cleanse",
    response_model=CleansingReportResponse,
    summary="Cleanse Dataset",
    description="Apply data cleansing operations to improve data quality.",
    responses={
        200: {"description": "Cleansing completed successfully"},
        400: {"description": "Invalid cleansing request"}
    }
)
async def cleanse_dataset(request: CleansingRequest):
    """Cleanse dataset according to configuration."""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset cannot be empty")
        
        # Execute cleansing
        cleaned_df, report = cleansing_engine.clean_dataset(
            df=df,
            cleansing_config=request.cleansing_config,
            dataset_name=request.dataset_name
        )
        
        # Convert results to response format
        cleansing_results = [
            CleansingResultResponse(
                action=result.action.value,
                column_name=result.column_name,
                records_affected=result.records_affected,
                strategy_used=result.strategy_used.value,
                success=result.success,
                error_message=result.error_message
            )
            for result in report.cleansing_results
        ]
        
        return CleansingReportResponse(
            dataset_name=report.dataset_name,
            total_records=report.total_records,
            total_columns=report.total_columns,
            cleansing_results=cleansing_results,
            execution_time_seconds=report.execution_time_seconds,
            overall_success=report.overall_success,
            quality_improvement=report.quality_improvement
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Cleansing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cleansing failed: {str(e)}")

@router.post(
    "/monitoring/start",
    summary="Start Quality Monitoring",
    description="Start real-time quality monitoring for a dataset.",
    responses={
        200: {"description": "Monitoring started successfully"},
        400: {"description": "Invalid monitoring configuration"}
    }
)
async def start_monitoring(request: MonitoringConfigRequest):
    """Start quality monitoring for a dataset."""
    try:
        # Create mock rules for monitoring
        rules = []
        for rule_id in request.rule_ids:
            rule = QualityRule(
                rule_id=RuleId(value=UUID(rule_id) if len(rule_id) == 36 else uuid4()),
                rule_name=f"Monitoring Rule {rule_id}",
                rule_type=RuleType.COMPLETENESS,
                target_columns=[],
                validation_logic=ValidationLogic(
                    logic_type=LogicType.PYTHON,
                    expression="df.notna().all(axis=1)",
                    parameters={},
                    error_message_template="Quality issue detected"
                ),
                thresholds=QualityThreshold(
                    pass_rate_threshold=0.95,
                    warning_threshold=0.90,
                    critical_threshold=0.80
                ),
                severity=Severity.MEDIUM,
                created_by=UserId(value=uuid4()),
                is_enabled=True
            )
            rules.append(rule)
        
        # Add dataset to monitoring
        monitoring_service.add_dataset_monitoring(
            dataset_id=UUID(request.dataset_id),
            rules=rules,
            data_source_config=request.data_source_config
        )
        
        # Start monitoring if not already active
        if monitoring_service.status != MonitoringStatus.ACTIVE:
            monitoring_service.start_monitoring()
        
        return {
            "dataset_id": request.dataset_id,
            "status": "monitoring_started",
            "message": f"Quality monitoring started for dataset with {len(rules)} rules",
            "monitoring_status": monitoring_service.status.value
        }
        
    except Exception as e:
        logger.error("Failed to start monitoring", error=str(e))
        raise HTTPException(status_code=400, detail=f"Failed to start monitoring: {str(e)}")

@router.get(
    "/monitoring/dashboard",
    response_model=MonitoringDashboardResponse,
    summary="Get Monitoring Dashboard",
    description="Get comprehensive monitoring dashboard data.",
    responses={
        200: {"description": "Dashboard data retrieved successfully"}
    }
)
async def get_monitoring_dashboard(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID")
):
    """Get monitoring dashboard data."""
    try:
        dataset_uuid = UUID(dataset_id) if dataset_id else None
        dashboard_data = monitoring_service.get_quality_dashboard_data(dataset_uuid)
        
        return MonitoringDashboardResponse(**dashboard_data)
        
    except Exception as e:
        logger.error("Failed to get dashboard data", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard data: {str(e)}")

@router.get(
    "/monitoring/alerts",
    response_model=List[QualityAlertResponse],
    summary="Get Quality Alerts",
    description="Get active quality alerts with optional filtering.",
    responses={
        200: {"description": "Alerts retrieved successfully"}
    }
)
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment status")
):
    """Get quality alerts."""
    try:
        severity_filter = AlertSeverity(severity) if severity else None
        alerts = monitoring_service.alert_manager.get_active_alerts(severity_filter)
        
        # Apply acknowledgment filter
        if acknowledged is not None:
            alerts = [alert for alert in alerts if alert.acknowledged == acknowledged]
        
        # Convert to response format
        alert_responses = [
            QualityAlertResponse(
                alert_id=str(alert.alert_id),
                rule_id=str(alert.rule_id),
                dataset_id=str(alert.dataset_id),
                severity=alert.severity.value,
                alert_type=alert.alert_type,
                message=alert.message,
                current_value=alert.current_value,
                threshold_value=alert.threshold_value,
                triggered_at=alert.triggered_at.isoformat(),
                acknowledged=alert.acknowledged
            )
            for alert in alerts
        ]
        
        return alert_responses
        
    except Exception as e:
        logger.error("Failed to get alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alerts: {str(e)}")

@router.put(
    "/monitoring/alerts/{alert_id}/acknowledge",
    summary="Acknowledge Alert",
    description="Acknowledge a quality alert.",
    responses={
        200: {"description": "Alert acknowledged successfully"},
        404: {"description": "Alert not found"}
    }
)
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = Query(..., description="User acknowledging the alert")
):
    """Acknowledge a quality alert."""
    try:
        success = monitoring_service.alert_manager.acknowledge_alert(
            alert_id=UUID(alert_id),
            acknowledged_by=acknowledged_by
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "alert_id": alert_id,
            "status": "acknowledged",
            "acknowledged_by": acknowledged_by,
            "acknowledged_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to acknowledge alert", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.get(
    "/health",
    summary="Health Check",
    description="Check the health of the data quality service.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unavailable"}
    }
)
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "service": "data-quality",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "monitoring_status": monitoring_service.status.value,
            "components": {
                "validation_engine": "healthy",
                "cleansing_engine": "healthy",
                "monitoring_service": monitoring_service.status.value
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")
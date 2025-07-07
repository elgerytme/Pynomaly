"""
FastAPI router for A/B testing endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from pynomaly.infrastructure.testing.ab_testing_framework import (
    ABTestingService, ABTest, TestVariant, TestStatus, SplitStrategy, 
    MetricType, StatisticalResult, TestResult, TestMetric
)
from pynomaly.domain.entities.user import User
from pynomaly.domain.entities.detector import Detector
from pynomaly.shared.exceptions import TestingError, ValidationError
from pynomaly.shared.types import TenantId, UserId

# Router setup
router = APIRouter(prefix="/api/ab-testing", tags=["A/B Testing"])

# Global A/B testing service
ab_testing_service = ABTestingService()

# Request/Response Models
class CreateVariantRequest(BaseModel):
    """Request to create a test variant."""
    name: str = Field(..., description="Variant name")
    description: str = Field(..., description="Variant description")
    detector_id: str = Field(..., description="Detector ID to use")
    traffic_percentage: float = Field(..., ge=0.0, le=1.0, description="Traffic percentage (0.0-1.0)")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")
    is_control: bool = Field(default=False, description="Whether this is the control variant")


class CreateABTestRequest(BaseModel):
    """Request to create an A/B test."""
    name: str = Field(..., description="Test name")
    description: str = Field(..., description="Test description")
    variants: List[CreateVariantRequest] = Field(..., min_items=2, description="Test variants")
    split_strategy: SplitStrategy = Field(default=SplitStrategy.RANDOM, description="Traffic splitting strategy")
    metrics_to_collect: List[MetricType] = Field(
        default=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, MetricType.F1_SCORE],
        description="Metrics to collect"
    )
    minimum_sample_size: int = Field(default=100, ge=10, description="Minimum sample size")
    confidence_level: float = Field(default=0.95, ge=0.01, le=0.99, description="Statistical confidence level")
    significance_threshold: float = Field(default=0.05, ge=0.001, le=0.5, description="Statistical significance threshold")
    duration_days: Optional[int] = Field(default=None, ge=1, description="Test duration in days")


class ABTestResponse(BaseModel):
    """Response for A/B test information."""
    test_id: str
    name: str
    description: str
    status: TestStatus
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    duration_days: Optional[int]
    variants: List[Dict[str, Any]]
    split_strategy: SplitStrategy
    metrics_collected: List[MetricType]
    minimum_sample_size: int
    confidence_level: float
    significance_threshold: float
    total_executions: int
    variant_statistics: Dict[str, Any]


class ExecuteTestRequest(BaseModel):
    """Request to execute an A/B test."""
    dataset_id: str = Field(..., description="Dataset ID to test with")
    user_id: Optional[str] = Field(default=None, description="User ID for consistent splitting")
    ground_truth: Optional[List[bool]] = Field(default=None, description="Ground truth labels for evaluation")


class TestResultResponse(BaseModel):
    """Response for test execution result."""
    test_id: str
    variant_id: str
    dataset_id: str
    execution_time: float
    timestamp: datetime
    metrics: List[Dict[str, Any]]
    anomaly_count: int
    metadata: Dict[str, Any]


class StatisticalAnalysisResponse(BaseModel):
    """Response for statistical analysis."""
    metric_name: str
    control_variant: str
    treatment_variant: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    p_value: float
    confidence_interval: List[float]
    effect_size: float
    is_significant: bool
    statistical_power: float
    sample_size_control: int
    sample_size_treatment: int
    interpretation: str


class MetricSummaryResponse(BaseModel):
    """Response for metric summary."""
    metric_name: str
    metric_type: MetricType
    variant_summaries: Dict[str, Dict[str, float]]  # variant_id -> {mean, std, count}


# Dependencies
async def get_current_user() -> User:
    """Get current authenticated user."""
    # TODO: Implement authentication
    pass


async def require_testing_access(tenant_id: UUID, current_user: User = Depends(get_current_user)):
    """Require A/B testing access to specific tenant."""
    if not (current_user.is_super_admin() or 
            current_user.has_role_in_tenant(TenantId(str(tenant_id)), ["tenant_admin", "data_scientist"])):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied - requires A/B testing permissions"
        )
    return current_user


async def get_detector_by_id(detector_id: str) -> Detector:
    """Get detector by ID."""
    # TODO: Implement detector retrieval from repository
    from pynomaly.domain.entities.detector import Detector
    from pynomaly.shared.types import DetectorId
    
    return Detector(
        id=DetectorId(detector_id),
        name=f"Detector_{detector_id}",
        algorithm="isolation_forest",
        parameters={},
        is_trained=True,
        created_at=datetime.utcnow()
    )


async def get_dataset_by_id(dataset_id: str):
    """Get dataset by ID."""
    # TODO: Implement dataset retrieval
    from pynomaly.domain.entities.dataset import Dataset
    import pandas as pd
    import numpy as np
    
    # Generate sample data for demonstration
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.normal(0, 1, 1000)
    })
    
    return Dataset(
        id=dataset_id,
        data=data,
        metadata={"source": "sample_data"}
    )


# A/B Test Management Endpoints
@router.post("/tenants/{tenant_id}/tests", response_model=ABTestResponse, status_code=status.HTTP_201_CREATED)
async def create_ab_test(
    tenant_id: UUID,
    request: CreateABTestRequest,
    current_user: User = Depends(require_testing_access)
):
    """Create a new A/B test."""
    try:
        # Validate traffic percentages sum to 1.0
        total_traffic = sum(v.traffic_percentage for v in request.variants)
        if abs(total_traffic - 1.0) > 0.01:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Traffic percentages must sum to 1.0, got {total_traffic}"
            )
        
        # Create test variants
        variants = []
        for i, variant_req in enumerate(request.variants):
            detector = await get_detector_by_id(variant_req.detector_id)
            
            variant = TestVariant(
                id=f"variant_{i}_{variant_req.name.lower().replace(' ', '_')}",
                name=variant_req.name,
                description=variant_req.description,
                detector=detector,
                traffic_percentage=variant_req.traffic_percentage,
                configuration=variant_req.configuration,
                is_control=variant_req.is_control
            )
            variants.append(variant)
        
        # Create A/B test
        test = ab_testing_service.create_test(
            name=request.name,
            description=request.description,
            variants=variants,
            split_strategy=request.split_strategy,
            metrics_to_collect=request.metrics_to_collect,
            minimum_sample_size=request.minimum_sample_size,
            confidence_level=request.confidence_level,
            significance_threshold=request.significance_threshold,
            duration_days=request.duration_days,
            created_by=UserId(current_user.id) if hasattr(current_user, 'id') else None
        )
        
        summary = test.get_summary()
        
        return ABTestResponse(
            test_id=summary["test_id"],
            name=summary["name"],
            description=summary["description"],
            status=TestStatus(summary["status"]),
            created_at=datetime.fromisoformat(summary["created_at"]),
            started_at=datetime.fromisoformat(summary["started_at"]) if summary["started_at"] else None,
            ended_at=datetime.fromisoformat(summary["ended_at"]) if summary["ended_at"] else None,
            duration_days=summary["duration_days"],
            variants=summary["variants"],
            split_strategy=SplitStrategy(summary["split_strategy"]),
            metrics_collected=request.metrics_to_collect,
            minimum_sample_size=summary["minimum_sample_size"],
            confidence_level=summary["confidence_level"],
            significance_threshold=summary["significance_threshold"],
            total_executions=summary["total_executions"],
            variant_statistics=summary["variant_statistics"]
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create A/B test: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/tests", response_model=List[ABTestResponse])
async def list_ab_tests(
    tenant_id: UUID,
    status_filter: Optional[TestStatus] = None,
    current_user: User = Depends(require_testing_access)
):
    """List A/B tests for a tenant."""
    try:
        tests = ab_testing_service.list_tests(status_filter)
        
        response = []
        for test in tests:
            summary = test.get_summary()
            
            response.append(ABTestResponse(
                test_id=summary["test_id"],
                name=summary["name"],
                description=summary["description"],
                status=TestStatus(summary["status"]),
                created_at=datetime.fromisoformat(summary["created_at"]),
                started_at=datetime.fromisoformat(summary["started_at"]) if summary["started_at"] else None,
                ended_at=datetime.fromisoformat(summary["ended_at"]) if summary["ended_at"] else None,
                duration_days=summary["duration_days"],
                variants=summary["variants"],
                split_strategy=SplitStrategy(summary["split_strategy"]),
                metrics_collected=[MetricType(m) for m in summary["metrics_collected"]],
                minimum_sample_size=summary["minimum_sample_size"],
                confidence_level=summary["confidence_level"],
                significance_threshold=summary["significance_threshold"],
                total_executions=summary["total_executions"],
                variant_statistics=summary["variant_statistics"]
            ))
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list A/B tests: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/tests/{test_id}", response_model=ABTestResponse)
async def get_ab_test(
    tenant_id: UUID,
    test_id: str,
    current_user: User = Depends(require_testing_access)
):
    """Get details of a specific A/B test."""
    try:
        test = ab_testing_service.get_test(test_id)
        if not test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"A/B test {test_id} not found"
            )
        
        summary = test.get_summary()
        
        return ABTestResponse(
            test_id=summary["test_id"],
            name=summary["name"],
            description=summary["description"],
            status=TestStatus(summary["status"]),
            created_at=datetime.fromisoformat(summary["created_at"]),
            started_at=datetime.fromisoformat(summary["started_at"]) if summary["started_at"] else None,
            ended_at=datetime.fromisoformat(summary["ended_at"]) if summary["ended_at"] else None,
            duration_days=summary["duration_days"],
            variants=summary["variants"],
            split_strategy=SplitStrategy(summary["split_strategy"]),
            metrics_collected=[MetricType(m) for m in summary["metrics_collected"]],
            minimum_sample_size=summary["minimum_sample_size"],
            confidence_level=summary["confidence_level"],
            significance_threshold=summary["significance_threshold"],
            total_executions=summary["total_executions"],
            variant_statistics=summary["variant_statistics"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get A/B test: {str(e)}"
        )


@router.post("/tenants/{tenant_id}/tests/{test_id}/start")
async def start_ab_test(
    tenant_id: UUID,
    test_id: str,
    current_user: User = Depends(require_testing_access)
):
    """Start an A/B test."""
    try:
        ab_testing_service.start_test(test_id)
        
        return {"message": f"A/B test {test_id} started successfully"}
        
    except TestingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start A/B test: {str(e)}"
        )


@router.post("/tenants/{tenant_id}/tests/{test_id}/stop")
async def stop_ab_test(
    tenant_id: UUID,
    test_id: str,
    current_user: User = Depends(require_testing_access)
):
    """Stop an A/B test."""
    try:
        ab_testing_service.stop_test(test_id)
        
        return {"message": f"A/B test {test_id} stopped successfully"}
        
    except TestingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop A/B test: {str(e)}"
        )


@router.delete("/tenants/{tenant_id}/tests/{test_id}")
async def delete_ab_test(
    tenant_id: UUID,
    test_id: str,
    current_user: User = Depends(require_testing_access)
):
    """Delete an A/B test."""
    try:
        ab_testing_service.delete_test(test_id)
        
        return {"message": f"A/B test {test_id} deleted successfully"}
        
    except TestingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete A/B test: {str(e)}"
        )


# Test Execution Endpoints
@router.post("/tenants/{tenant_id}/tests/{test_id}/execute", response_model=TestResultResponse)
async def execute_ab_test(
    tenant_id: UUID,
    test_id: str,
    request: ExecuteTestRequest,
    current_user: User = Depends(require_testing_access)
):
    """Execute an A/B test with a dataset."""
    try:
        # Get dataset
        dataset = await get_dataset_by_id(request.dataset_id)
        
        # Execute test
        result = await ab_testing_service.execute_test(
            test_id=test_id,
            dataset=dataset,
            user_id=request.user_id,
            ground_truth=request.ground_truth
        )
        
        return TestResultResponse(
            test_id=result.test_id,
            variant_id=result.variant_id,
            dataset_id=result.dataset_id,
            execution_time=result.execution_time,
            timestamp=result.timestamp,
            metrics=[m.to_dict() for m in result.metrics],
            anomaly_count=len(result.detection_result.anomalies),
            metadata=result.metadata
        )
        
    except TestingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute A/B test: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/tests/{test_id}/results", response_model=List[TestResultResponse])
async def get_test_results(
    tenant_id: UUID,
    test_id: str,
    variant_id: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(require_testing_access)
):
    """Get results for an A/B test."""
    try:
        results = ab_testing_service.get_test_results(test_id)
        
        # Filter by variant if specified
        if variant_id:
            results = [r for r in results if r.variant_id == variant_id]
        
        # Limit results
        results = results[-limit:]
        
        response = []
        for result in results:
            response.append(TestResultResponse(
                test_id=result.test_id,
                variant_id=result.variant_id,
                dataset_id=result.dataset_id,
                execution_time=result.execution_time,
                timestamp=result.timestamp,
                metrics=[m.to_dict() for m in result.metrics],
                anomaly_count=len(result.detection_result.anomalies),
                metadata=result.metadata
            ))
        
        return response
        
    except TestingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get test results: {str(e)}"
        )


# Statistical Analysis Endpoints
@router.get("/tenants/{tenant_id}/tests/{test_id}/analysis", response_model=List[StatisticalAnalysisResponse])
async def get_statistical_analysis(
    tenant_id: UUID,
    test_id: str,
    current_user: User = Depends(require_testing_access)
):
    """Get statistical analysis for an A/B test."""
    try:
        analysis_results = ab_testing_service.get_statistical_analysis(test_id)
        
        response = []
        for result in analysis_results:
            # Generate interpretation
            interpretation = _generate_interpretation(result)
            
            response.append(StatisticalAnalysisResponse(
                metric_name=result.metric_name,
                control_variant=result.control_variant,
                treatment_variant=result.treatment_variant,
                control_mean=result.control_mean,
                treatment_mean=result.treatment_mean,
                control_std=result.control_std,
                treatment_std=result.treatment_std,
                p_value=result.p_value,
                confidence_interval=list(result.confidence_interval),
                effect_size=result.effect_size,
                is_significant=result.is_significant,
                statistical_power=result.statistical_power,
                sample_size_control=result.sample_size_control,
                sample_size_treatment=result.sample_size_treatment,
                interpretation=interpretation
            ))
        
        return response
        
    except TestingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistical analysis: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/tests/{test_id}/metrics", response_model=List[MetricSummaryResponse])
async def get_metric_summaries(
    tenant_id: UUID,
    test_id: str,
    current_user: User = Depends(require_testing_access)
):
    """Get metric summaries for an A/B test."""
    try:
        test = ab_testing_service.get_test(test_id)
        if not test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"A/B test {test_id} not found"
            )
        
        # Group metrics by name
        metric_groups = {}
        for metric in test.collected_metrics:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = {
                    "type": metric.type,
                    "values_by_variant": {}
                }
            
            if metric.variant_id not in metric_groups[metric.name]["values_by_variant"]:
                metric_groups[metric.name]["values_by_variant"][metric.variant_id] = []
            
            metric_groups[metric.name]["values_by_variant"][metric.variant_id].append(metric.value)
        
        # Calculate summaries
        response = []
        for metric_name, group in metric_groups.items():
            variant_summaries = {}
            
            for variant_id, values in group["values_by_variant"].items():
                if values:
                    import statistics
                    variant_summaries[variant_id] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "count": len(values),
                        "min": min(values),
                        "max": max(values)
                    }
            
            response.append(MetricSummaryResponse(
                metric_name=metric_name,
                metric_type=group["type"],
                variant_summaries=variant_summaries
            ))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metric summaries: {str(e)}"
        )


# Utility Functions
def _generate_interpretation(result: StatisticalResult) -> str:
    """Generate human-readable interpretation of statistical result."""
    interpretation_parts = []
    
    # Statistical significance
    if result.is_significant:
        interpretation_parts.append("The difference is statistically significant")
    else:
        interpretation_parts.append("The difference is not statistically significant")
    
    # Effect size interpretation
    abs_effect = abs(result.effect_size)
    if abs_effect < 0.2:
        effect_desc = "negligible"
    elif abs_effect < 0.5:
        effect_desc = "small"
    elif abs_effect < 0.8:
        effect_desc = "medium"
    else:
        effect_desc = "large"
    
    interpretation_parts.append(f"with a {effect_desc} effect size ({result.effect_size:.3f})")
    
    # Direction of effect
    if result.treatment_mean > result.control_mean:
        direction = "higher"
    else:
        direction = "lower"
    
    improvement = abs((result.treatment_mean - result.control_mean) / result.control_mean * 100)
    interpretation_parts.append(f"The treatment variant shows {improvement:.1f}% {direction} {result.metric_name}")
    
    # Statistical power
    if result.statistical_power < 0.8:
        interpretation_parts.append("Note: Statistical power is low, consider collecting more data")
    
    return ". ".join(interpretation_parts) + "."


# Health Check
@router.get("/health")
async def ab_testing_health_check():
    """Health check for A/B testing service."""
    try:
        active_tests = len(ab_testing_service.active_tests)
        total_tests = len(ab_testing_service.active_tests) + len(ab_testing_service.test_history)
        
        return {
            "status": "healthy",
            "active_tests": active_tests,
            "total_tests": total_tests,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
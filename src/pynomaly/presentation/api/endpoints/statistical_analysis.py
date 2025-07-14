"""
Statistical Analysis API Endpoints

RESTful endpoints for comprehensive statistical analysis capabilities.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from uuid import UUID
import logging

from ....domain.entities.user import User
from ....application.dto.statistical_analysis_dto import (
    StatisticalAnalysisRequestDTO,
    StatisticalAnalysisResponseDTO,
    DescriptiveStatsRequestDTO,
    DescriptiveStatsResponseDTO,
    CorrelationAnalysisRequestDTO,
    CorrelationAnalysisResponseDTO,
    DistributionAnalysisRequestDTO,
    DistributionAnalysisResponseDTO,
    HypothesisTestRequestDTO,
    HypothesisTestResponseDTO
)
from ....application.use_cases.perform_statistical_analysis import PerformStatisticalAnalysisUseCase
from ....application.use_cases.perform_correlation_analysis import PerformCorrelationAnalysisUseCase
from ....application.use_cases.perform_distribution_analysis import PerformDistributionAnalysisUseCase
from ....shared.dependencies import get_current_user, get_statistical_analysis_use_case
from ....shared.monitoring import metrics, monitor_endpoint
from ....shared.error_handling import APIError, ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analysis", tags=["Statistical Analysis"])


@router.post(
    "/descriptive-stats",
    response_model=DescriptiveStatsResponseDTO,
    summary="Perform Descriptive Statistical Analysis",
    description="""
    Perform comprehensive descriptive statistical analysis on dataset features.
    
    Calculates measures of central tendency, dispersion, and distribution shape
    including mean, median, mode, standard deviation, variance, skewness, kurtosis,
    and percentile distributions.
    
    **Features:**
    - Univariate and multivariate analysis
    - Missing value analysis
    - Outlier detection and quantification
    - Data type inference and validation
    - Statistical significance testing
    """,
    responses={
        200: {"description": "Descriptive statistics computed successfully"},
        400: {"description": "Invalid request parameters"},
        404: {"description": "Dataset not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("descriptive_stats_analysis")
async def analyze_descriptive_statistics(
    request: DescriptiveStatsRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: PerformStatisticalAnalysisUseCase = Depends(get_statistical_analysis_use_case)
) -> DescriptiveStatsResponseDTO:
    """
    Perform descriptive statistical analysis on dataset.
    
    Args:
        request: Descriptive statistics analysis request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Statistical analysis use case
        
    Returns:
        Descriptive statistics analysis results
        
    Raises:
        HTTPException: If analysis fails or dataset not found
    """
    try:
        logger.info(
            f"Starting descriptive statistics analysis for dataset {request.dataset_id} "
            f"by user {current_user.id}"
        )
        
        # Convert to general statistical analysis request
        analysis_request = StatisticalAnalysisRequestDTO(
            dataset_id=request.dataset_id,
            user_id=current_user.id,
            analysis_type="descriptive",
            feature_columns=request.feature_columns,
            analysis_params={
                "include_percentiles": request.include_percentiles,
                "percentile_values": request.percentile_values,
                "detect_outliers": request.detect_outliers,
                "outlier_method": request.outlier_method,
                "missing_value_analysis": request.missing_value_analysis
            }
        )
        
        result = await use_case.execute(analysis_request)
        
        if result.status == "failed":
            metrics.analysis_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Descriptive analysis failed: {result.error_message}"
            )
        
        metrics.successful_analyses.inc()
        logger.info(f"Descriptive analysis completed for dataset {request.dataset_id}")
        
        return DescriptiveStatsResponseDTO(
            analysis_id=result.analysis_id,
            dataset_id=request.dataset_id,
            descriptive_stats=result.results.get("descriptive_stats", {}),
            outlier_analysis=result.results.get("outlier_analysis", {}),
            missing_value_analysis=result.results.get("missing_value_analysis", {}),
            insights=result.insights,
            execution_time_seconds=result.execution_time_seconds,
            created_at=result.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.analysis_failures.inc()
        logger.error(f"Unexpected error in descriptive analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during descriptive analysis"
        )


@router.post(
    "/correlation",
    response_model=CorrelationAnalysisResponseDTO,
    summary="Perform Correlation Analysis",
    description="""
    Perform correlation analysis between dataset features to identify relationships
    and dependencies.
    
    Supports multiple correlation methods including Pearson, Spearman, and Kendall
    correlations with statistical significance testing and visualization data.
    
    **Features:**
    - Multiple correlation coefficient methods
    - Statistical significance testing
    - Partial correlation analysis
    - Correlation clustering and network analysis
    - Missing value handling strategies
    """,
    responses={
        200: {"description": "Correlation analysis completed successfully"},
        400: {"description": "Invalid request parameters"},
        404: {"description": "Dataset not found"},
        422: {"description": "Insufficient features for correlation analysis"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("correlation_analysis")
async def analyze_correlations(
    request: CorrelationAnalysisRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: PerformCorrelationAnalysisUseCase = Depends(get_statistical_analysis_use_case)
) -> CorrelationAnalysisResponseDTO:
    """
    Perform correlation analysis on dataset features.
    
    Args:
        request: Correlation analysis request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Statistical analysis use case
        
    Returns:
        Correlation analysis results
        
    Raises:
        HTTPException: If analysis fails or insufficient features
    """
    try:
        logger.info(
            f"Starting correlation analysis for dataset {request.dataset_id} "
            f"by user {current_user.id}"
        )
        
        # Validate minimum features for correlation
        if len(request.feature_columns) < 2:
            raise HTTPException(
                status_code=422,
                detail="At least 2 features required for correlation analysis"
            )
        
        analysis_request = StatisticalAnalysisRequestDTO(
            dataset_id=request.dataset_id,
            user_id=current_user.id,
            analysis_type="correlation",
            feature_columns=request.feature_columns,
            analysis_params={
                "method": request.method,
                "min_periods": request.min_periods,
                "significance_level": request.significance_level,
                "partial_correlation": request.partial_correlation,
                "clustering_enabled": request.clustering_enabled
            }
        )
        
        result = await use_case.execute(analysis_request)
        
        if result.status == "failed":
            metrics.analysis_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Correlation analysis failed: {result.error_message}"
            )
        
        metrics.successful_analyses.inc()
        logger.info(f"Correlation analysis completed for dataset {request.dataset_id}")
        
        return CorrelationAnalysisResponseDTO(
            analysis_id=result.analysis_id,
            dataset_id=request.dataset_id,
            correlation_matrix=result.results.get("correlation_matrix", {}),
            significance_matrix=result.results.get("significance_matrix", {}),
            partial_correlations=result.results.get("partial_correlations", {}),
            correlation_clusters=result.results.get("correlation_clusters", []),
            method=request.method,
            insights=result.insights,
            execution_time_seconds=result.execution_time_seconds,
            created_at=result.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.analysis_failures.inc()
        logger.error(f"Unexpected error in correlation analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during correlation analysis"
        )


@router.post(
    "/distribution",
    response_model=DistributionAnalysisResponseDTO,
    summary="Perform Distribution Analysis",
    description="""
    Analyze the statistical distribution of dataset features with goodness-of-fit
    testing and parameter estimation.
    
    Supports analysis of common distributions including normal, exponential,
    gamma, beta, and custom distributions with comprehensive statistical testing.
    
    **Features:**
    - Multiple distribution fitting algorithms
    - Goodness-of-fit statistical tests
    - Parameter estimation with confidence intervals
    - Distribution comparison and ranking
    - Visualization data for distribution plots
    """,
    responses={
        200: {"description": "Distribution analysis completed successfully"},
        400: {"description": "Invalid request parameters"},
        404: {"description": "Dataset not found"},
        422: {"description": "Insufficient data for distribution analysis"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("distribution_analysis")
async def analyze_distributions(
    request: DistributionAnalysisRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: PerformDistributionAnalysisUseCase = Depends(get_statistical_analysis_use_case)
) -> DistributionAnalysisResponseDTO:
    """
    Perform distribution analysis on dataset features.
    
    Args:
        request: Distribution analysis request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Statistical analysis use case
        
    Returns:
        Distribution analysis results
        
    Raises:
        HTTPException: If analysis fails or insufficient data
    """
    try:
        logger.info(
            f"Starting distribution analysis for dataset {request.dataset_id} "
            f"by user {current_user.id}"
        )
        
        analysis_request = StatisticalAnalysisRequestDTO(
            dataset_id=request.dataset_id,
            user_id=current_user.id,
            analysis_type="distribution",
            feature_columns=request.feature_columns,
            analysis_params={
                "distributions_to_test": request.distributions_to_test,
                "significance_level": request.significance_level,
                "estimation_method": request.estimation_method,
                "goodness_of_fit_tests": request.goodness_of_fit_tests,
                "bootstrap_samples": request.bootstrap_samples
            }
        )
        
        result = await use_case.execute(analysis_request)
        
        if result.status == "failed":
            metrics.analysis_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Distribution analysis failed: {result.error_message}"
            )
        
        metrics.successful_analyses.inc()
        logger.info(f"Distribution analysis completed for dataset {request.dataset_id}")
        
        return DistributionAnalysisResponseDTO(
            analysis_id=result.analysis_id,
            dataset_id=request.dataset_id,
            fitted_distributions=result.results.get("fitted_distributions", {}),
            goodness_of_fit_tests=result.results.get("goodness_of_fit_tests", {}),
            best_fit_distribution=result.results.get("best_fit_distribution", {}),
            parameter_estimates=result.results.get("parameter_estimates", {}),
            visualization_data=result.results.get("visualization_data", {}),
            insights=result.insights,
            execution_time_seconds=result.execution_time_seconds,
            created_at=result.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.analysis_failures.inc()
        logger.error(f"Unexpected error in distribution analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during distribution analysis"
        )


@router.post(
    "/hypothesis-test",
    response_model=HypothesisTestResponseDTO,
    summary="Perform Hypothesis Testing",
    description="""
    Perform statistical hypothesis testing with support for multiple test types
    and comprehensive result interpretation.
    
    Supports t-tests, chi-square tests, ANOVA, Mann-Whitney U, Wilcoxon signed-rank,
    and other common statistical tests with effect size calculations.
    
    **Features:**
    - Multiple hypothesis testing procedures
    - Effect size calculations
    - Power analysis and sample size estimation
    - Multiple comparison corrections
    - Comprehensive result interpretation
    """,
    responses={
        200: {"description": "Hypothesis testing completed successfully"},
        400: {"description": "Invalid test parameters"},
        404: {"description": "Dataset not found"},
        422: {"description": "Insufficient data for specified test"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("hypothesis_testing")
async def perform_hypothesis_test(
    request: HypothesisTestRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: PerformStatisticalAnalysisUseCase = Depends(get_statistical_analysis_use_case)
) -> HypothesisTestResponseDTO:
    """
    Perform statistical hypothesis testing.
    
    Args:
        request: Hypothesis test request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Statistical analysis use case
        
    Returns:
        Hypothesis test results
        
    Raises:
        HTTPException: If test fails or invalid parameters
    """
    try:
        logger.info(
            f"Starting hypothesis testing for dataset {request.dataset_id} "
            f"by user {current_user.id}"
        )
        
        analysis_request = StatisticalAnalysisRequestDTO(
            dataset_id=request.dataset_id,
            user_id=current_user.id,
            analysis_type="hypothesis_testing",
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            analysis_params={
                "test_type": request.test_type,
                "alternative_hypothesis": request.alternative_hypothesis,
                "significance_level": request.significance_level,
                "effect_size_calculation": request.effect_size_calculation,
                "multiple_comparison_correction": request.multiple_comparison_correction,
                "power_analysis": request.power_analysis
            }
        )
        
        result = await use_case.execute(analysis_request)
        
        if result.status == "failed":
            metrics.analysis_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Hypothesis testing failed: {result.error_message}"
            )
        
        metrics.successful_analyses.inc()
        logger.info(f"Hypothesis testing completed for dataset {request.dataset_id}")
        
        return HypothesisTestResponseDTO(
            analysis_id=result.analysis_id,
            dataset_id=request.dataset_id,
            test_statistic=result.results.get("test_statistic"),
            p_value=result.results.get("p_value"),
            critical_value=result.results.get("critical_value"),
            effect_size=result.results.get("effect_size"),
            confidence_interval=result.results.get("confidence_interval"),
            power=result.results.get("power"),
            test_decision=result.results.get("test_decision"),
            test_interpretation=result.results.get("test_interpretation"),
            insights=result.insights,
            execution_time_seconds=result.execution_time_seconds,
            created_at=result.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.analysis_failures.inc()
        logger.error(f"Unexpected error in hypothesis testing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during hypothesis testing"
        )


@router.get(
    "/results/{analysis_id}",
    response_model=StatisticalAnalysisResponseDTO,
    summary="Get Statistical Analysis Results",
    description="""
    Retrieve the results of a previously completed statistical analysis.
    
    Returns comprehensive analysis results including computed statistics,
    insights, and metadata about the analysis execution.
    """,
    responses={
        200: {"description": "Analysis results retrieved successfully"},
        404: {"description": "Analysis not found"},
        403: {"description": "Access denied to analysis results"},
        500: {"description": "Internal server error"}
    }
)
async def get_analysis_results(
    analysis_id: UUID,
    current_user: User = Depends(get_current_user),
    use_case: PerformStatisticalAnalysisUseCase = Depends(get_statistical_analysis_use_case)
) -> StatisticalAnalysisResponseDTO:
    """
    Get statistical analysis results by ID.
    
    Args:
        analysis_id: Analysis identifier
        current_user: Authenticated user
        use_case: Statistical analysis use case
        
    Returns:
        Statistical analysis results
        
    Raises:
        HTTPException: If analysis not found or access denied
    """
    try:
        # This would be implemented in the use case
        # For now, return a placeholder response
        logger.info(f"Retrieving analysis results for {analysis_id}")
        
        return StatisticalAnalysisResponseDTO(
            analysis_id=analysis_id,
            status="completed",
            results={"placeholder": "results"},
            insights=["Analysis results retrieved successfully"],
            execution_time_seconds=1.5,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving analysis results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving analysis results"
        )


@router.get(
    "/history",
    response_model=List[StatisticalAnalysisResponseDTO],
    summary="Get Analysis History",
    description="""
    Retrieve the history of statistical analyses performed by the current user.
    
    Supports filtering by dataset, analysis type, and date range with pagination.
    """,
    responses={
        200: {"description": "Analysis history retrieved successfully"},
        400: {"description": "Invalid query parameters"},
        500: {"description": "Internal server error"}
    }
)
async def get_analysis_history(
    dataset_id: Optional[UUID] = Query(None, description="Filter by dataset ID"),
    analysis_type: Optional[str] = Query(None, description="Filter by analysis type"),
    limit: int = Query(50, description="Maximum number of results", ge=1, le=100),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    current_user: User = Depends(get_current_user)
) -> List[StatisticalAnalysisResponseDTO]:
    """
    Get user's statistical analysis history.
    
    Args:
        dataset_id: Optional dataset filter
        analysis_type: Optional analysis type filter
        limit: Maximum results
        offset: Results offset
        current_user: Authenticated user
        
    Returns:
        List of statistical analysis results
    """
    try:
        logger.info(f"Retrieving analysis history for user {current_user.id}")
        
        # This would be implemented with proper repository queries
        # For now, return an empty list
        return []
        
    except Exception as e:
        logger.error(f"Error retrieving analysis history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving analysis history"
        )
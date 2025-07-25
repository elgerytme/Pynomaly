"""
Quality Assurance Service for the MLOps Marketplace.

Provides comprehensive quality assurance capabilities including automated testing,
security scanning, performance analysis, and certification management.
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel

from mlops_marketplace.domain.entities import (
    Solution,
    SolutionVersion,
    Certification,
    QualityReport,
    SecurityScan,
    PerformanceReport,
)
from mlops_marketplace.domain.value_objects import (
    SolutionId,
    CertificationId,
    Version,
)
from mlops_marketplace.domain.repositories import (
    SolutionRepository,
    CertificationRepository,
    QualityReportRepository,
)
from mlops_marketplace.domain.interfaces import (
    SecurityScanner,
    PerformanceAnalyzer,
    CodeAnalyzer,
    ModelValidator,
    ComplianceChecker,
    NotificationService,
)


class QualityCheckType(str, Enum):
    """Types of quality checks."""
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_TEST = "performance_test"
    CODE_ANALYSIS = "code_analysis"
    MODEL_VALIDATION = "model_validation"
    COMPLIANCE_CHECK = "compliance_check"
    INTEGRATION_TEST = "integration_test"
    LOAD_TEST = "load_test"
    VULNERABILITY_SCAN = "vulnerability_scan"


class QualityCheckStatus(str, Enum):
    """Status of quality checks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CertificationLevel(str, Enum):
    """Certification levels."""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class QualityCheckRequest(BaseModel):
    """Request for quality check execution."""
    solution_id: SolutionId
    version_id: UUID
    check_types: List[QualityCheckType]
    check_config: Dict[str, Any] = {}
    priority: int = 5  # 1-10, higher is more priority
    notify_on_completion: bool = True


class QualityCheckResult(BaseModel):
    """Result of a quality check."""
    check_id: UUID
    check_type: QualityCheckType
    status: QualityCheckStatus
    score: float  # 0-100
    passed: bool
    findings: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    recommendations: List[str] = []
    execution_time_seconds: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class CertificationRequest(BaseModel):
    """Request for solution certification."""
    solution_id: SolutionId
    version_id: UUID
    certification_level: CertificationLevel
    compliance_standards: List[str] = []
    custom_requirements: Dict[str, Any] = {}


class QualityAssuranceService:
    """Service for managing quality assurance operations."""
    
    def __init__(
        self,
        solution_repository: SolutionRepository,
        certification_repository: CertificationRepository,
        quality_report_repository: QualityReportRepository,
        security_scanner: SecurityScanner,
        performance_analyzer: PerformanceAnalyzer,
        code_analyzer: CodeAnalyzer,
        model_validator: ModelValidator,
        compliance_checker: ComplianceChecker,
        notification_service: NotificationService,
    ):
        """Initialize the quality assurance service."""
        self.solution_repository = solution_repository
        self.certification_repository = certification_repository
        self.quality_report_repository = quality_report_repository
        self.security_scanner = security_scanner
        self.performance_analyzer = performance_analyzer
        self.code_analyzer = code_analyzer
        self.model_validator = model_validator
        self.compliance_checker = compliance_checker
        self.notification_service = notification_service
        
        # Quality thresholds
        self.quality_thresholds = {
            CertificationLevel.BASIC: 70.0,
            CertificationLevel.STANDARD: 80.0,
            CertificationLevel.PREMIUM: 90.0,
            CertificationLevel.ENTERPRISE: 95.0,
        }
    
    async def run_quality_checks(
        self, 
        request: QualityCheckRequest
    ) -> List[QualityCheckResult]:
        """Run comprehensive quality checks on a solution version."""
        solution = await self.solution_repository.get_by_id(request.solution_id)
        if not solution:
            raise ValueError(f"Solution {request.solution_id} not found")
        
        version = solution.get_version(request.version_id)
        if not version:
            raise ValueError(f"Version {request.version_id} not found")
        
        # Create quality check tasks
        check_tasks = []
        
        for check_type in request.check_types:
            task = self._create_quality_check_task(
                check_type, solution, version, request.check_config
            )
            check_tasks.append(task)
        
        # Execute checks in parallel
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        quality_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create failed result for exception
                failed_result = QualityCheckResult(
                    check_id=uuid4(),
                    check_type=request.check_types[i],
                    status=QualityCheckStatus.FAILED,
                    score=0.0,
                    passed=False,
                    execution_time_seconds=0.0,
                    started_at=datetime.utcnow(),
                    error_message=str(result),
                )
                quality_results.append(failed_result)
            else:
                quality_results.append(result)
        
        # Save quality report
        await self._save_quality_report(solution, version, quality_results)
        
        # Update solution quality scores
        await self._update_solution_quality_scores(solution, version, quality_results)
        
        # Send notifications if requested
        if request.notify_on_completion:
            await self._send_quality_check_notifications(
                solution, version, quality_results
            )
        
        return quality_results
    
    async def request_certification(
        self, 
        request: CertificationRequest
    ) -> CertificationId:
        """Request certification for a solution version."""
        solution = await self.solution_repository.get_by_id(request.solution_id)
        if not solution:
            raise ValueError(f"Solution {request.solution_id} not found")
        
        version = solution.get_version(request.version_id)
        if not version:
            raise ValueError(f"Version {request.version_id} not found")
        
        # Check if solution meets minimum quality requirements
        quality_report = await self.quality_report_repository.get_latest_report(
            request.solution_id, request.version_id
        )
        
        if not quality_report:
            raise ValueError("No quality report found. Run quality checks first.")
        
        minimum_score = self.quality_thresholds[request.certification_level]
        if quality_report.overall_score < minimum_score:
            raise ValueError(
                f"Solution score {quality_report.overall_score} is below "
                f"minimum requirement {minimum_score} for {request.certification_level}"
            )
        
        # Create certification request
        certification = Certification(
            id=CertificationId.generate(),
            solution_id=request.solution_id,
            version_id=request.version_id,
            certification_level=request.certification_level,
            compliance_standards=request.compliance_standards,
            custom_requirements=request.custom_requirements,
            status="pending",
            requested_at=datetime.utcnow(),
        )
        
        # Save certification request
        await self.certification_repository.save(certification)
        
        # Start certification process
        await self._process_certification_request(certification)
        
        return certification.id
    
    async def get_quality_report(
        self, 
        solution_id: SolutionId,
        version_id: UUID
    ) -> Optional[QualityReport]:
        """Get the latest quality report for a solution version."""
        return await self.quality_report_repository.get_latest_report(
            solution_id, version_id
        )
    
    async def get_certification_status(
        self, 
        certification_id: CertificationId
    ) -> Optional[Certification]:
        """Get certification status and details."""
        return await self.certification_repository.get_by_id(certification_id)
    
    async def get_security_scan_results(
        self, 
        solution_id: SolutionId,
        version_id: UUID
    ) -> List[SecurityScan]:
        """Get security scan results for a solution version."""
        return await self.quality_report_repository.get_security_scans(
            solution_id, version_id
        )
    
    async def get_performance_test_results(
        self, 
        solution_id: SolutionId,
        version_id: UUID
    ) -> List[PerformanceReport]:
        """Get performance test results for a solution version."""
        return await self.quality_report_repository.get_performance_reports(
            solution_id, version_id
        )
    
    async def schedule_automated_checks(
        self, 
        solution_id: SolutionId,
        check_schedule: Dict[str, Any]
    ) -> None:
        """Schedule automated quality checks for a solution."""
        # This would integrate with a job scheduler like Celery
        # For now, we'll store the schedule configuration
        await self.solution_repository.update_check_schedule(
            solution_id, check_schedule
        )
    
    async def get_quality_trends(
        self, 
        solution_id: SolutionId,
        days: int = 30
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get quality trends over time for a solution."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        return await self.quality_report_repository.get_quality_trends(
            solution_id, start_date, end_date
        )
    
    async def _create_quality_check_task(
        self,
        check_type: QualityCheckType,
        solution: Solution,
        version: SolutionVersion,
        config: Dict[str, Any]
    ) -> QualityCheckResult:
        """Create and execute a quality check task."""
        check_id = uuid4()
        started_at = datetime.utcnow()
        
        try:
            if check_type == QualityCheckType.SECURITY_SCAN:
                result = await self._run_security_scan(solution, version, config)
            elif check_type == QualityCheckType.PERFORMANCE_TEST:
                result = await self._run_performance_test(solution, version, config)
            elif check_type == QualityCheckType.CODE_ANALYSIS:
                result = await self._run_code_analysis(solution, version, config)
            elif check_type == QualityCheckType.MODEL_VALIDATION:
                result = await self._run_model_validation(solution, version, config)
            elif check_type == QualityCheckType.COMPLIANCE_CHECK:
                result = await self._run_compliance_check(solution, version, config)
            else:
                raise ValueError(f"Unsupported check type: {check_type}")
            
            completed_at = datetime.utcnow()
            execution_time = (completed_at - started_at).total_seconds()
            
            return QualityCheckResult(
                check_id=check_id,
                check_type=check_type,
                status=QualityCheckStatus.COMPLETED,
                score=result["score"],
                passed=result["passed"],
                findings=result.get("findings", []),
                metrics=result.get("metrics", {}),
                recommendations=result.get("recommendations", []),
                execution_time_seconds=execution_time,
                started_at=started_at,
                completed_at=completed_at,
            )
        
        except Exception as e:
            completed_at = datetime.utcnow()
            execution_time = (completed_at - started_at).total_seconds()
            
            return QualityCheckResult(
                check_id=check_id,
                check_type=check_type,
                status=QualityCheckStatus.FAILED,
                score=0.0,
                passed=False,
                execution_time_seconds=execution_time,
                started_at=started_at,
                completed_at=completed_at,
                error_message=str(e),
            )
    
    async def _run_security_scan(
        self, 
        solution: Solution,
        version: SolutionVersion,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run security scan on solution version."""
        return await self.security_scanner.scan_solution(
            solution_id=solution.id,
            version_id=version.id,
            container_image=version.container_image,
            source_code_url=version.source_code_url,
            config=config,
        )
    
    async def _run_performance_test(
        self, 
        solution: Solution,
        version: SolutionVersion,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run performance test on solution version."""
        return await self.performance_analyzer.analyze_performance(
            solution_id=solution.id,
            version_id=version.id,
            container_image=version.container_image,
            api_specification=version.api_specification,
            config=config,
        )
    
    async def _run_code_analysis(
        self, 
        solution: Solution,
        version: SolutionVersion,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run code analysis on solution version."""
        return await self.code_analyzer.analyze_code(
            solution_id=solution.id,
            version_id=version.id,
            source_code_url=version.source_code_url,
            config=config,
        )
    
    async def _run_model_validation(
        self, 
        solution: Solution,
        version: SolutionVersion,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run model validation on solution version."""
        return await self.model_validator.validate_model(
            solution_id=solution.id,
            version_id=version.id,
            technical_spec=version.technical_spec,
            config=config,
        )
    
    async def _run_compliance_check(
        self, 
        solution: Solution,
        version: SolutionVersion,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run compliance check on solution version."""
        return await self.compliance_checker.check_compliance(
            solution_id=solution.id,
            version_id=version.id,
            license_type=version.license_type,
            metadata=solution.metadata,
            config=config,
        )
    
    async def _save_quality_report(
        self,
        solution: Solution,
        version: SolutionVersion,
        results: List[QualityCheckResult]
    ) -> None:
        """Save quality report to repository."""
        # Calculate overall score
        total_score = sum(result.score for result in results)
        overall_score = total_score / len(results) if results else 0.0
        
        # Determine overall status
        all_passed = all(result.passed for result in results)
        
        # Create quality report
        quality_report = QualityReport(
            id=uuid4(),
            solution_id=solution.id,
            version_id=version.id,
            overall_score=overall_score,
            overall_status="passed" if all_passed else "failed",
            check_results=results,
            generated_at=datetime.utcnow(),
        )
        
        await self.quality_report_repository.save(quality_report)
    
    async def _update_solution_quality_scores(
        self,
        solution: Solution,
        version: SolutionVersion,
        results: List[QualityCheckResult]
    ) -> None:
        """Update solution and version quality scores."""
        # Calculate scores by category
        security_results = [r for r in results if r.check_type == QualityCheckType.SECURITY_SCAN]
        performance_results = [r for r in results if r.check_type == QualityCheckType.PERFORMANCE_TEST]
        
        if security_results:
            security_score = sum(r.score for r in security_results) / len(security_results)
            version.security_score = security_score
        
        if performance_results:
            performance_score = sum(r.score for r in performance_results) / len(performance_results)
            version.performance_score = performance_score
        
        # Calculate overall quality score
        total_score = sum(result.score for result in results)
        overall_score = total_score / len(results) if results else 0.0
        version.quality_score = overall_score
        
        # Update in repository
        await self.solution_repository.update_version(version)
    
    async def _send_quality_check_notifications(
        self,
        solution: Solution,
        version: SolutionVersion,
        results: List[QualityCheckResult]
    ) -> None:
        """Send notifications about quality check completion."""
        provider = await self.solution_repository.get_provider(solution.provider_id)
        
        # Determine notification type based on results
        all_passed = all(result.passed for result in results)
        notification_type = "quality_check_passed" if all_passed else "quality_check_failed"
        
        # Send notification
        await self.notification_service.send_notification(
            recipient_id=solution.provider_id,
            notification_type=notification_type,
            data={
                "solution_name": solution.name,
                "version": str(version.version),
                "overall_score": sum(r.score for r in results) / len(results),
                "results_summary": {
                    "total_checks": len(results),
                    "passed_checks": sum(1 for r in results if r.passed),
                    "failed_checks": sum(1 for r in results if not r.passed),
                },
            },
        )
    
    async def _process_certification_request(
        self, 
        certification: Certification
    ) -> None:
        """Process a certification request."""
        # This would typically involve:
        # 1. Manual review by certification team
        # 2. Additional automated checks
        # 3. Compliance verification
        # 4. Documentation review
        
        # For now, we'll simulate the process
        certification.status = "in_review"
        certification.review_started_at = datetime.utcnow()
        await self.certification_repository.update(certification)
        
        # Schedule automated certification checks
        # This would be handled by background workers in a real implementation
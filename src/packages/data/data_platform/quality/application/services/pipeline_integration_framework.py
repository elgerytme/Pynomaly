"""
Pipeline integration framework for self-healing data quality in data pipelines.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from abc import ABC, abstractmethod

from data_quality.application.services.autonomous_quality_monitoring_service import AutonomousQualityMonitoringService
from data_quality.application.services.automated_remediation_engine import AutomatedRemediationEngine
from data_quality.application.services.adaptive_quality_controls import AdaptiveQualityControls
from data_quality.domain.entities.quality_anomaly import QualityAnomaly
from software.interfaces.data_quality_interface import DataQualityInterface
from software.interfaces.data_quality_interface import QualityReport


logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Data pipeline stages."""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    QUALITY_CHECK = "quality_check"
    ENRICHMENT = "enrichment"
    OUTPUT = "output"


class QualityAction(Enum):
    """Quality actions in pipeline."""
    VALIDATE = "validate"
    MONITOR = "monitor"
    REMEDIATE = "remediate"
    ALERT = "alert"
    BLOCK = "block"
    BYPASS = "bypass"


@dataclass
class PipelineContext:
    """Context information for pipeline execution."""
    pipeline_id: str
    stage: PipelineStage
    data_batch_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    previous_stage_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityCheckResult:
    """Result of quality check in pipeline."""
    check_id: str
    stage: PipelineStage
    success: bool
    quality_score: float
    issues_found: List[QualityAnomaly]
    execution_time: timedelta
    action_taken: QualityAction
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for quality protection."""
    name: str
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(minutes=5)
    
    def should_open(self) -> bool:
        """Check if circuit breaker should open."""
        return self.failure_count >= self.failure_threshold
    
    def should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.state == "OPEN" and self.last_failure_time:
            return datetime.utcnow() - self.last_failure_time > self.recovery_timeout
        return False


@dataclass
class QualityMiddleware:
    """Quality middleware configuration."""
    name: str
    enabled: bool = True
    priority: int = 0
    stages: List[PipelineStage] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable] = None


class PipelineIntegrator(ABC):
    """Abstract base class for pipeline integrators."""
    
    @abstractmethod
    async def initialize(self, pipeline_config: Dict[str, Any]) -> None:
        """Initialize the integrator with pipeline configuration."""
        pass
    
    @abstractmethod
    async def inject_quality_checks(self, stage: PipelineStage, context: PipelineContext) -> QualityCheckResult:
        """Inject quality checks into pipeline stage."""
        pass
    
    @abstractmethod
    async def handle_quality_failure(self, result: QualityCheckResult, context: PipelineContext) -> QualityAction:
        """Handle quality check failure."""
        pass


class AirflowIntegrator(PipelineIntegrator):
    """Airflow pipeline integrator."""
    
    def __init__(self, quality_framework):
        self.quality_framework = quality_framework
        self.dag_configs: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self, pipeline_config: Dict[str, Any]) -> None:
        """Initialize Airflow integrator."""
        self.dag_configs = pipeline_config.get("dags", {})
        logger.info(f"Initialized Airflow integrator with {len(self.dag_configs)} DAGs")
    
    async def inject_quality_checks(self, stage: PipelineStage, context: PipelineContext) -> QualityCheckResult:
        """Inject quality checks into Airflow DAG."""
        start_time = datetime.utcnow()
        
        try:
            # Get quality metrics from monitoring service
            quality_state = await self.quality_framework.monitoring_service.get_quality_state(
                context.data_batch_id
            )
            
            issues_found = []
            quality_score = 0.8  # Default score
            
            if quality_state:
                quality_score = quality_state.overall_score
                issues_found = quality_state.anomalies
            
            # Determine action based on quality score
            if quality_score < 0.5:
                action = QualityAction.BLOCK
            elif quality_score < 0.7:
                action = QualityAction.REMEDIATE
            elif quality_score < 0.9:
                action = QualityAction.ALERT
            else:
                action = QualityAction.VALIDATE
            
            execution_time = datetime.utcnow() - start_time
            
            return QualityCheckResult(
                check_id=f"airflow_{stage.value}_{context.data_batch_id}",
                stage=stage,
                success=quality_score >= 0.7,
                quality_score=quality_score,
                issues_found=issues_found,
                execution_time=execution_time,
                action_taken=action,
                recommendations=self._generate_airflow_recommendations(quality_score, issues_found)
            )
            
        except Exception as e:
            logger.error(f"Airflow quality check failed: {str(e)}")
            return QualityCheckResult(
                check_id=f"airflow_{stage.value}_{context.data_batch_id}",
                stage=stage,
                success=False,
                quality_score=0.0,
                issues_found=[],
                execution_time=datetime.utcnow() - start_time,
                action_taken=QualityAction.ALERT,
                metadata={"error": str(e)}
            )
    
    async def handle_quality_failure(self, result: QualityCheckResult, context: PipelineContext) -> QualityAction:
        """Handle quality failure in Airflow."""
        if result.action_taken == QualityAction.BLOCK:
            # Stop DAG execution
            logger.error(f"Blocking Airflow DAG execution due to quality failure: {result.check_id}")
            return QualityAction.BLOCK
        
        elif result.action_taken == QualityAction.REMEDIATE:
            # Attempt remediation
            logger.info(f"Attempting remediation for Airflow DAG: {result.check_id}")
            
            # Trigger remediation through framework
            for issue in result.issues_found:
                await self.quality_framework.remediation_engine.analyze_and_remediate(
                    issue, context.metadata.get("data")
                )
            
            return QualityAction.REMEDIATE
        
        return QualityAction.ALERT
    
    def _generate_airflow_recommendations(self, quality_score: float, issues: List[QualityAnomaly]) -> List[str]:
        """Generate Airflow-specific recommendations."""
        recommendations = []
        
        if quality_score < 0.5:
            recommendations.append("Consider adding upstream data validation tasks")
            recommendations.append("Implement data quality sensors in DAG")
        
        if issues:
            recommendations.append("Review data source quality")
            recommendations.append("Consider adding data cleansing tasks")
        
        return recommendations


class PrefectIntegrator(PipelineIntegrator):
    """Prefect pipeline integrator."""
    
    def __init__(self, quality_framework):
        self.quality_framework = quality_framework
        self.flow_configs: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self, pipeline_config: Dict[str, Any]) -> None:
        """Initialize Prefect integrator."""
        self.flow_configs = pipeline_config.get("flows", {})
        logger.info(f"Initialized Prefect integrator with {len(self.flow_configs)} flows")
    
    async def inject_quality_checks(self, stage: PipelineStage, context: PipelineContext) -> QualityCheckResult:
        """Inject quality checks into Prefect flow."""
        start_time = datetime.utcnow()
        
        try:
            # Similar to Airflow but with Prefect-specific logic
            quality_state = await self.quality_framework.monitoring_service.get_quality_state(
                context.data_batch_id
            )
            
            issues_found = []
            quality_score = 0.8
            
            if quality_state:
                quality_score = quality_state.overall_score
                issues_found = quality_state.anomalies
            
            # Prefect-specific action determination
            if quality_score < 0.6:
                action = QualityAction.BLOCK
            elif quality_score < 0.8:
                action = QualityAction.REMEDIATE
            else:
                action = QualityAction.VALIDATE
            
            execution_time = datetime.utcnow() - start_time
            
            return QualityCheckResult(
                check_id=f"prefect_{stage.value}_{context.data_batch_id}",
                stage=stage,
                success=quality_score >= 0.8,
                quality_score=quality_score,
                issues_found=issues_found,
                execution_time=execution_time,
                action_taken=action,
                recommendations=self._generate_prefect_recommendations(quality_score, issues_found)
            )
            
        except Exception as e:
            logger.error(f"Prefect quality check failed: {str(e)}")
            return QualityCheckResult(
                check_id=f"prefect_{stage.value}_{context.data_batch_id}",
                stage=stage,
                success=False,
                quality_score=0.0,
                issues_found=[],
                execution_time=datetime.utcnow() - start_time,
                action_taken=QualityAction.ALERT,
                metadata={"error": str(e)}
            )
    
    async def handle_quality_failure(self, result: QualityCheckResult, context: PipelineContext) -> QualityAction:
        """Handle quality failure in Prefect."""
        if result.action_taken == QualityAction.BLOCK:
            logger.error(f"Blocking Prefect flow execution: {result.check_id}")
            return QualityAction.BLOCK
        
        elif result.action_taken == QualityAction.REMEDIATE:
            logger.info(f"Attempting remediation for Prefect flow: {result.check_id}")
            
            for issue in result.issues_found:
                await self.quality_framework.remediation_engine.analyze_and_remediate(
                    issue, context.metadata.get("data")
                )
            
            return QualityAction.REMEDIATE
        
        return QualityAction.ALERT
    
    def _generate_prefect_recommendations(self, quality_score: float, issues: List[QualityAnomaly]) -> List[str]:
        """Generate Prefect-specific recommendations."""
        recommendations = []
        
        if quality_score < 0.6:
            recommendations.append("Add quality validation tasks to flow")
            recommendations.append("Implement state-based quality checks")
        
        if issues:
            recommendations.append("Use Prefect's retry mechanism for transient issues")
            recommendations.append("Consider data lineage tracking")
        
        return recommendations


class StreamingIntegrator(PipelineIntegrator):
    """Streaming pipeline integrator (Kafka, Kinesis, etc.)."""
    
    def __init__(self, quality_framework):
        self.quality_framework = quality_framework
        self.stream_configs: Dict[str, Dict[str, Any]] = {}
        self.processing_buffers: Dict[str, List[Any]] = {}
    
    async def initialize(self, pipeline_config: Dict[str, Any]) -> None:
        """Initialize streaming integrator."""
        self.stream_configs = pipeline_config.get("streams", {})
        logger.info(f"Initialized streaming integrator with {len(self.stream_configs)} streams")
    
    async def inject_quality_checks(self, stage: PipelineStage, context: PipelineContext) -> QualityCheckResult:
        """Inject quality checks into streaming pipeline."""
        start_time = datetime.utcnow()
        
        try:
            # Real-time quality monitoring for streaming data
            quality_state = await self.quality_framework.monitoring_service.get_quality_state(
                context.data_batch_id
            )
            
            issues_found = []
            quality_score = 0.85  # Higher default for streaming
            
            if quality_state:
                quality_score = quality_state.overall_score
                issues_found = quality_state.anomalies
            
            # Streaming-specific action determination (more lenient)
            if quality_score < 0.4:
                action = QualityAction.BLOCK
            elif quality_score < 0.6:
                action = QualityAction.REMEDIATE
            elif quality_score < 0.8:
                action = QualityAction.ALERT
            else:
                action = QualityAction.VALIDATE
            
            execution_time = datetime.utcnow() - start_time
            
            return QualityCheckResult(
                check_id=f"streaming_{stage.value}_{context.data_batch_id}",
                stage=stage,
                success=quality_score >= 0.6,
                quality_score=quality_score,
                issues_found=issues_found,
                execution_time=execution_time,
                action_taken=action,
                recommendations=self._generate_streaming_recommendations(quality_score, issues_found)
            )
            
        except Exception as e:
            logger.error(f"Streaming quality check failed: {str(e)}")
            return QualityCheckResult(
                check_id=f"streaming_{stage.value}_{context.data_batch_id}",
                stage=stage,
                success=False,
                quality_score=0.0,
                issues_found=[],
                execution_time=datetime.utcnow() - start_time,
                action_taken=QualityAction.ALERT,
                metadata={"error": str(e)}
            )
    
    async def handle_quality_failure(self, result: QualityCheckResult, context: PipelineContext) -> QualityAction:
        """Handle quality failure in streaming pipeline."""
        if result.action_taken == QualityAction.BLOCK:
            logger.error(f"Blocking streaming pipeline: {result.check_id}")
            # In streaming, blocking means dropping the current batch
            return QualityAction.BLOCK
        
        elif result.action_taken == QualityAction.REMEDIATE:
            logger.info(f"Attempting real-time remediation: {result.check_id}")
            
            # Fast remediation for streaming
            for issue in result.issues_found:
                await self.quality_framework.remediation_engine.analyze_and_remediate(
                    issue, context.metadata.get("data")
                )
            
            return QualityAction.REMEDIATE
        
        return QualityAction.ALERT
    
    def _generate_streaming_recommendations(self, quality_score: float, issues: List[QualityAnomaly]) -> List[str]:
        """Generate streaming-specific recommendations."""
        recommendations = []
        
        if quality_score < 0.4:
            recommendations.append("Implement dead letter queue for failed records")
            recommendations.append("Add stream processing quality filters")
        
        if issues:
            recommendations.append("Consider windowed quality aggregation")
            recommendations.append("Implement backpressure handling")
        
        return recommendations


class PipelineIntegrationFramework:
    """Framework for integrating quality controls into data pipelines."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the pipeline integration framework."""
        # Initialize service configuration
        self.config = config
        
        # Initialize quality services
        self.monitoring_service = AutonomousQualityMonitoringService(config.get("monitoring", {}))
        self.remediation_engine = AutomatedRemediationEngine(config.get("remediation", {}))
        self.adaptive_controls = AdaptiveQualityControls(config.get("adaptive_controls", {}))
        
        # Pipeline integrators
        self.integrators: Dict[str, PipelineIntegrator] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.quality_middleware: List[QualityMiddleware] = []
        
        # Performance tracking
        self.pipeline_metrics: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[QualityCheckResult] = []
        
        # Configuration
        self.max_execution_time = config.get("max_execution_time", 300)  # 5 minutes
        self.quality_threshold = config.get("quality_threshold", 0.7)
        
        # Initialize integrators
        self._initialize_integrators()
        self._initialize_circuit_breakers()
        self._initialize_middleware()
        
        # Start monitoring tasks
        asyncio.create_task(self._pipeline_monitoring_task())
    
    def _initialize_integrators(self) -> None:
        """Initialize pipeline integrators."""
        self.integrators = {
            "airflow": AirflowIntegrator(self),
            "prefect": PrefectIntegrator(self),
            "streaming": StreamingIntegrator(self)
        }
        
        logger.info(f"Initialized {len(self.integrators)} pipeline integrators")
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for quality protection."""
        breaker_configs = self.config.get("circuit_breakers", {})
        
        for name, config in breaker_configs.items():
            self.circuit_breakers[name] = CircuitBreakerState(
                name=name,
                failure_threshold=config.get("failure_threshold", 5),
                recovery_timeout=timedelta(seconds=config.get("recovery_timeout", 300))
            )
    
    def _initialize_middleware(self) -> None:
        """Initialize quality middleware."""
        middleware_configs = self.config.get("middleware", [])
        
        for config in middleware_configs:
            middleware = QualityMiddleware(
                name=config["name"],
                enabled=config.get("enabled", True),
                priority=config.get("priority", 0),
                stages=config.get("stages", []),
                config=config.get("config", {})
            )
            self.quality_middleware.append(middleware)
        
        # Sort by priority
        self.quality_middleware.sort(key=lambda x: x.priority)
    
    async def _pipeline_monitoring_task(self) -> None:
        """Monitor pipeline performance and quality."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Update circuit breaker states
                await self._update_circuit_breakers()
                
                # Collect pipeline metrics
                await self._collect_pipeline_metrics()
                
                # Clean up old execution history
                await self._cleanup_execution_history()
                
            except Exception as e:
                logger.error(f"Pipeline monitoring error: {str(e)}")
    
    async def _update_circuit_breakers(self) -> None:
        """Update circuit breaker states."""
        for breaker in self.circuit_breakers.values():
            if breaker.state == "OPEN" and breaker.should_attempt_reset():
                breaker.state = "HALF_OPEN"
                logger.info(f"Circuit breaker {breaker.name} attempting reset")
    
    async def _collect_pipeline_metrics(self) -> None:
        """Collect pipeline performance metrics."""
        # Calculate metrics from execution history
        recent_executions = [
            result for result in self.execution_history
            if datetime.utcnow() - result.execution_time < timedelta(hours=1)
        ]
        
        if recent_executions:
            success_rate = sum(1 for r in recent_executions if r.success) / len(recent_executions)
            avg_quality_score = sum(r.quality_score for r in recent_executions) / len(recent_executions)
            avg_execution_time = sum(r.execution_time.total_seconds() for r in recent_executions) / len(recent_executions)
            
            self.pipeline_metrics["overall"] = {
                "success_rate": success_rate,
                "avg_quality_score": avg_quality_score,
                "avg_execution_time": avg_execution_time,
                "total_executions": len(recent_executions)
            }
    
    async def _cleanup_execution_history(self) -> None:
        """Clean up old execution history."""
        cutoff_time = datetime.utcnow() - timedelta(days=1)
        self.execution_history = [
            result for result in self.execution_history
            if result.execution_time > cutoff_time
        ]
    
    # Error handling would be managed by interface implementation
    async def register_pipeline(self, pipeline_type: str, pipeline_config: Dict[str, Any]) -> bool:
        """Register a pipeline with the framework."""
        if pipeline_type not in self.integrators:
            logger.error(f"Unsupported pipeline type: {pipeline_type}")
            return False
        
        integrator = self.integrators[pipeline_type]
        await integrator.initialize(pipeline_config)
        
        logger.info(f"Registered {pipeline_type} pipeline with quality framework")
        return True
    
    # Error handling would be managed by interface implementation
    async def execute_quality_check(self, pipeline_type: str, stage: PipelineStage, 
                                   context: PipelineContext) -> QualityCheckResult:
        """Execute quality check for a pipeline stage."""
        if pipeline_type not in self.integrators:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
        
        integrator = self.integrators[pipeline_type]
        
        # Check circuit breaker
        breaker_name = f"{pipeline_type}_{stage.value}"
        if breaker_name in self.circuit_breakers:
            breaker = self.circuit_breakers[breaker_name]
            if breaker.state == "OPEN":
                logger.warning(f"Circuit breaker {breaker_name} is open, skipping quality check")
                return QualityCheckResult(
                    check_id=f"circuit_breaker_{breaker_name}",
                    stage=stage,
                    success=False,
                    quality_score=0.0,
                    issues_found=[],
                    execution_time=timedelta(seconds=0),
                    action_taken=QualityAction.BYPASS,
                    metadata={"circuit_breaker": "open"}
                )
        
        # Execute quality check
        start_time = datetime.utcnow()
        
        try:
            result = await integrator.inject_quality_checks(stage, context)
            
            # Update circuit breaker
            if breaker_name in self.circuit_breakers:
                breaker = self.circuit_breakers[breaker_name]
                if result.success:
                    breaker.success_count += 1
                    breaker.failure_count = 0
                    if breaker.state == "HALF_OPEN":
                        breaker.state = "CLOSED"
                        logger.info(f"Circuit breaker {breaker_name} closed")
                else:
                    breaker.failure_count += 1
                    breaker.last_failure_time = datetime.utcnow()
                    if breaker.should_open():
                        breaker.state = "OPEN"
                        logger.warning(f"Circuit breaker {breaker_name} opened")
            
            # Store execution history
            self.execution_history.append(result)
            
            # Handle failure if needed
            if not result.success:
                action = await integrator.handle_quality_failure(result, context)
                result.action_taken = action
            
            return result
            
        except Exception as e:
            logger.error(f"Quality check execution failed: {str(e)}")
            
            # Update circuit breaker for exception
            if breaker_name in self.circuit_breakers:
                breaker = self.circuit_breakers[breaker_name]
                breaker.failure_count += 1
                breaker.last_failure_time = datetime.utcnow()
                if breaker.should_open():
                    breaker.state = "OPEN"
            
            return QualityCheckResult(
                check_id=f"error_{pipeline_type}_{stage.value}",
                stage=stage,
                success=False,
                quality_score=0.0,
                issues_found=[],
                execution_time=datetime.utcnow() - start_time,
                action_taken=QualityAction.ALERT,
                metadata={"error": str(e)}
            )
    
    # Error handling would be managed by interface implementation
    async def create_quality_operator(self, pipeline_type: str, stage: PipelineStage) -> Dict[str, Any]:
        """Create a quality operator for specific pipeline type."""
        if pipeline_type == "airflow":
            return {
                "task_id": f"quality_check_{stage.value}",
                "python_callable": self._create_airflow_callable(stage),
                "provide_context": True
            }
        
        elif pipeline_type == "prefect":
            return {
                "name": f"quality_check_{stage.value}",
                "task_function": self._create_prefect_task(stage)
            }
        
        elif pipeline_type == "streaming":
            return {
                "processor": self._create_streaming_processor(stage),
                "config": {"batch_size": 1000, "timeout": 30}
            }
        
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
    
    def _create_airflow_callable(self, stage: PipelineStage) -> Callable:
        """Create Airflow callable for quality checks."""
        async def quality_check_callable(**context):
            pipeline_context = PipelineContext(
                pipeline_id=context.get("dag_id"),
                stage=stage,
                data_batch_id=context.get("run_id"),
                timestamp=datetime.utcnow(),
                metadata=context
            )
            
            result = await self.execute_quality_check("airflow", stage, pipeline_context)
            
            if not result.success and result.action_taken == QualityAction.BLOCK:
                raise Exception(f"Quality check failed: {result.check_id}")
            
            return result.success
        
        return quality_check_callable
    
    def _create_prefect_task(self, stage: PipelineStage) -> Callable:
        """Create Prefect task for quality checks."""
        async def quality_check_task(context: dict):
            pipeline_context = PipelineContext(
                pipeline_id=context.get("flow_id"),
                stage=stage,
                data_batch_id=context.get("run_id"),
                timestamp=datetime.utcnow(),
                metadata=context
            )
            
            result = await self.execute_quality_check("prefect", stage, pipeline_context)
            
            if not result.success and result.action_taken == QualityAction.BLOCK:
                raise Exception(f"Quality check failed: {result.check_id}")
            
            return result.success
        
        return quality_check_task
    
    def _create_streaming_processor(self, stage: PipelineStage) -> Callable:
        """Create streaming processor for quality checks."""
        async def quality_processor(data_batch):
            pipeline_context = PipelineContext(
                pipeline_id="streaming_pipeline",
                stage=stage,
                data_batch_id=f"batch_{datetime.utcnow().timestamp()}",
                timestamp=datetime.utcnow(),
                metadata={"data": data_batch}
            )
            
            result = await self.execute_quality_check("streaming", stage, pipeline_context)
            
            if result.action_taken == QualityAction.BLOCK:
                return None  # Drop the batch
            
            return data_batch
        
        return quality_processor
    
    # Error handling would be managed by interface implementation
    async def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health status."""
        health_status = {
            "overall_health": "healthy",
            "circuit_breakers": {},
            "pipeline_metrics": self.pipeline_metrics,
            "active_integrators": list(self.integrators.keys()),
            "middleware_status": []
        }
        
        # Check circuit breakers
        open_breakers = 0
        for name, breaker in self.circuit_breakers.items():
            health_status["circuit_breakers"][name] = {
                "state": breaker.state,
                "failure_count": breaker.failure_count,
                "success_count": breaker.success_count
            }
            if breaker.state == "OPEN":
                open_breakers += 1
        
        # Check middleware
        for middleware in self.quality_middleware:
            health_status["middleware_status"].append({
                "name": middleware.name,
                "enabled": middleware.enabled,
                "priority": middleware.priority
            })
        
        # Determine overall health
        if open_breakers > 0:
            health_status["overall_health"] = "degraded"
        
        overall_metrics = self.pipeline_metrics.get("overall", {})
        if overall_metrics.get("success_rate", 1.0) < 0.8:
            health_status["overall_health"] = "unhealthy"
        
        return health_status
    
    # Error handling would be managed by interface implementation
    async def get_execution_history(self, pipeline_type: Optional[str] = None, 
                                   stage: Optional[PipelineStage] = None) -> List[QualityCheckResult]:
        """Get execution history with optional filters."""
        results = self.execution_history
        
        if pipeline_type:
            results = [r for r in results if pipeline_type in r.check_id]
        
        if stage:
            results = [r for r in results if r.stage == stage]
        
        return results
    
    async def shutdown(self) -> None:
        """Shutdown the pipeline integration framework."""
        logger.info("Shutting down pipeline integration framework...")
        
        # Shutdown quality services
        await self.monitoring_service.shutdown()
        await self.remediation_engine.shutdown()
        await self.adaptive_controls.shutdown()
        
        # Clear data
        self.integrators.clear()
        self.circuit_breakers.clear()
        self.quality_middleware.clear()
        self.pipeline_metrics.clear()
        self.execution_history.clear()
        
        logger.info("Pipeline integration framework shutdown complete")
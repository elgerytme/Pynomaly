"""
Cleansing Pipeline Orchestration System
Multi-stage cleansing workflows with dependency management, parallel execution, and quality gates.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .advanced_data_cleansing_engine import AdvancedDataCleansingEngine, CleansingResult
from .domain_specific_cleansing import DomainSpecificCleansingOrchestrator, DomainCleansingResult

logger = logging.getLogger(__name__)


class PipelineStageStatus(Enum):
    """Status of individual pipeline stages."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class PipelineStatus(Enum):
    """Overall pipeline status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class QualityGate:
    """Quality gate configuration for pipeline stages."""
    
    gate_id: str
    name: str
    description: str
    quality_threshold: float = 0.8
    completeness_threshold: float = 0.9
    consistency_threshold: float = 0.85
    accuracy_threshold: float = 0.9
    blocking: bool = True  # Whether gate failure blocks pipeline
    
    # Custom validation functions
    custom_validators: List[Callable] = field(default_factory=list)
    
    # Gate parameters
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleansingStage:
    """Individual stage in cleansing pipeline."""
    
    stage_id: str
    name: str
    description: str
    stage_type: str  # 'validation', 'standardization', 'deduplication', 'domain_specific', 'custom'
    
    # Stage configuration
    stage_function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    # Execution control
    timeout_seconds: Optional[int] = None
    retry_attempts: int = 0
    retry_delay_seconds: int = 5
    can_run_parallel: bool = True
    
    # Quality gates
    quality_gates: List[QualityGate] = field(default_factory=list)
    
    # Runtime state
    status: PipelineStageStatus = PipelineStageStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    attempt_count: int = 0
    
    # Quality metrics
    input_quality_score: Optional[float] = None
    output_quality_score: Optional[float] = None
    quality_improvement: Optional[float] = None


@dataclass
class CleansingPipeline:
    """Complete cleansing pipeline definition."""
    
    pipeline_id: str
    name: str
    description: str
    version: str = "1.0.0"
    
    # Pipeline stages
    stages: List[CleansingStage] = field(default_factory=list)
    
    # Execution configuration
    enable_parallel_execution: bool = True
    max_parallel_stages: int = 3
    global_timeout_minutes: Optional[int] = None
    continue_on_gate_failure: bool = False
    
    # Pipeline metadata
    domain: Optional[str] = None
    target_data_size_mb: Optional[float] = None
    expected_execution_time_minutes: Optional[float] = None
    
    # Runtime state
    status: PipelineStatus = PipelineStatus.CREATED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_execution_time_seconds: Optional[float] = None
    
    # Execution tracking
    current_stage_ids: List[str] = field(default_factory=list)
    completed_stage_ids: List[str] = field(default_factory=list)
    failed_stage_ids: List[str] = field(default_factory=list)
    
    # Data flow
    pipeline_context: Dict[str, Any] = field(default_factory=dict)
    checkpoint_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    # Quality tracking
    overall_quality_improvement: Optional[float] = None
    stage_quality_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class PipelineExecutionResult:
    """Result of pipeline execution."""
    
    pipeline_id: str
    execution_status: PipelineStatus
    total_execution_time_seconds: float
    
    # Data results
    input_data_shape: Tuple[int, int]
    output_data_shape: Tuple[int, int]
    cleansed_data: pd.DataFrame
    
    # Quality metrics
    overall_quality_improvement: float
    stage_results: List[Dict[str, Any]]
    
    # Pipeline metrics
    stages_executed: int
    stages_completed: int
    stages_failed: int
    quality_gates_passed: int
    quality_gates_failed: int
    
    # Performance metrics
    peak_memory_usage_mb: Optional[float] = None
    average_stage_time_seconds: Optional[float] = None
    
    # Audit information
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class CleansingPipelineOrchestrator:
    """
    Advanced orchestrator for multi-stage data cleansing pipelines.
    Supports dependency management, parallel execution, quality gates, and checkpointing.
    """
    
    def __init__(self, max_concurrent_pipelines: int = 5):
        self.max_concurrent_pipelines = max_concurrent_pipelines
        self.active_pipelines: Dict[str, CleansingPipeline] = {}
        self.pipeline_history: List[CleansingPipeline] = []
        
        # Initialize component services
        self.cleansing_engine = AdvancedDataCleansingEngine()
        self.domain_orchestrator = DomainSpecificCleansingOrchestrator()
        
        # Pipeline templates
        self.pipeline_templates: Dict[str, Callable] = {}
        self._register_builtin_templates()
        
        logger.info(f"Initialized CleansingPipelineOrchestrator with max {max_concurrent_pipelines} concurrent pipelines")
    
    def _register_builtin_templates(self):
        """Register built-in pipeline templates."""
        self.pipeline_templates.update({
            'comprehensive_cleansing': self._create_comprehensive_pipeline,
            'quick_standardization': self._create_quick_standardization_pipeline,
            'duplicate_resolution': self._create_duplicate_resolution_pipeline,
            'domain_specific': self._create_domain_specific_pipeline,
            'validation_only': self._create_validation_pipeline
        })
    
    async def execute_pipeline(self, pipeline: CleansingPipeline, 
                             input_data: pd.DataFrame) -> PipelineExecutionResult:
        """Execute complete cleansing pipeline with orchestration."""
        
        if len(self.active_pipelines) >= self.max_concurrent_pipelines:
            raise RuntimeError(f"Maximum concurrent pipelines ({self.max_concurrent_pipelines}) reached")
        
        # Initialize pipeline execution
        pipeline.status = PipelineStatus.RUNNING
        pipeline.start_time = datetime.now()
        pipeline.pipeline_context['input_data'] = input_data
        pipeline.pipeline_context['current_data'] = input_data.copy()
        
        self.active_pipelines[pipeline.pipeline_id] = pipeline
        
        execution_log = []
        errors = []
        warnings = []
        
        try:
            logger.info(f"Starting pipeline execution: {pipeline.name} ({pipeline.pipeline_id})")
            
            # Set global timeout if specified
            global_timeout = None
            if pipeline.global_timeout_minutes:
                global_timeout = asyncio.create_task(
                    asyncio.sleep(pipeline.global_timeout_minutes * 60)
                )
            
            # Main execution loop
            while not self._is_pipeline_complete(pipeline):
                ready_stages = self._get_ready_stages(pipeline)
                
                if not ready_stages:
                    if pipeline.current_stage_ids:
                        # Wait for running stages to complete
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        # No ready stages and no running stages
                        if pipeline.failed_stage_ids and not pipeline.continue_on_gate_failure:
                            logger.error(f"Pipeline {pipeline.pipeline_id} stopped due to stage failures")
                            pipeline.status = PipelineStatus.FAILED
                            break
                        else:
                            logger.warning(f"Pipeline {pipeline.pipeline_id} deadlocked - no ready stages")
                            break
                
                # Execute ready stages
                if pipeline.enable_parallel_execution:
                    await self._execute_stages_parallel(pipeline, ready_stages, execution_log)
                else:
                    await self._execute_stages_sequential(pipeline, ready_stages, execution_log)
                
                # Check global timeout
                if global_timeout and global_timeout.done():
                    logger.error(f"Pipeline {pipeline.pipeline_id} timed out after {pipeline.global_timeout_minutes} minutes")
                    pipeline.status = PipelineStatus.FAILED
                    break
            
            # Finalize pipeline
            pipeline.end_time = datetime.now()
            pipeline.total_execution_time_seconds = (pipeline.end_time - pipeline.start_time).total_seconds()
            
            if self._is_pipeline_complete(pipeline) and not pipeline.failed_stage_ids:
                pipeline.status = PipelineStatus.COMPLETED
                logger.info(f"Pipeline {pipeline.pipeline_id} completed successfully")
            elif pipeline.failed_stage_ids:
                pipeline.status = PipelineStatus.FAILED
                logger.error(f"Pipeline {pipeline.pipeline_id} failed with {len(pipeline.failed_stage_ids)} failed stages")
            
            # Calculate overall quality improvement
            pipeline.overall_quality_improvement = self._calculate_pipeline_quality_improvement(pipeline)
            
            # Create execution result
            result = PipelineExecutionResult(
                pipeline_id=pipeline.pipeline_id,
                execution_status=pipeline.status,
                total_execution_time_seconds=pipeline.total_execution_time_seconds,
                input_data_shape=input_data.shape,
                output_data_shape=pipeline.pipeline_context['current_data'].shape,
                cleansed_data=pipeline.pipeline_context['current_data'],
                overall_quality_improvement=pipeline.overall_quality_improvement or 0,
                stage_results=self._collect_stage_results(pipeline),
                stages_executed=len(pipeline.stages),
                stages_completed=len(pipeline.completed_stage_ids),
                stages_failed=len(pipeline.failed_stage_ids),
                quality_gates_passed=self._count_quality_gates_passed(pipeline),
                quality_gates_failed=self._count_quality_gates_failed(pipeline),
                execution_log=execution_log,
                errors=errors,
                warnings=warnings
            )
            
            # Cancel global timeout if set
            if global_timeout and not global_timeout.done():
                global_timeout.cancel()
            
            return result
            
        except Exception as e:
            pipeline.status = PipelineStatus.FAILED
            pipeline.end_time = datetime.now()
            pipeline.total_execution_time_seconds = (pipeline.end_time - pipeline.start_time).total_seconds()
            errors.append(str(e))
            logger.error(f"Pipeline {pipeline.pipeline_id} failed with exception: {e}")
            
            # Return partial result
            return PipelineExecutionResult(
                pipeline_id=pipeline.pipeline_id,
                execution_status=PipelineStatus.FAILED,
                total_execution_time_seconds=pipeline.total_execution_time_seconds,
                input_data_shape=input_data.shape,
                output_data_shape=input_data.shape,
                cleansed_data=input_data,
                overall_quality_improvement=0,
                stage_results=[],
                stages_executed=0,
                stages_completed=0,
                stages_failed=1,
                quality_gates_passed=0,
                quality_gates_failed=0,
                execution_log=execution_log,
                errors=errors,
                warnings=warnings
            )
        
        finally:
            # Move from active to history
            if pipeline.pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline.pipeline_id]
            self.pipeline_history.append(pipeline)
    
    async def _execute_stages_parallel(self, pipeline: CleansingPipeline, 
                                     ready_stages: List[CleansingStage],
                                     execution_log: List[Dict[str, Any]]):
        """Execute stages in parallel with concurrency control."""
        
        # Limit concurrent execution
        stages_to_execute = [s for s in ready_stages if s.can_run_parallel][:pipeline.max_parallel_stages]
        sequential_stages = [s for s in ready_stages if not s.can_run_parallel]
        
        # Execute parallel stages
        if stages_to_execute:
            tasks = []
            for stage in stages_to_execute:
                task = asyncio.create_task(self._execute_single_stage(pipeline, stage, execution_log))
                tasks.append(task)
                pipeline.current_stage_ids.append(stage.stage_id)
            
            # Wait for all parallel tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Execute sequential stages one by one
        for stage in sequential_stages:
            pipeline.current_stage_ids.append(stage.stage_id)
            await self._execute_single_stage(pipeline, stage, execution_log)
    
    async def _execute_stages_sequential(self, pipeline: CleansingPipeline,
                                       ready_stages: List[CleansingStage],
                                       execution_log: List[Dict[str, Any]]):
        """Execute stages sequentially."""
        
        for stage in ready_stages:
            pipeline.current_stage_ids.append(stage.stage_id)
            await self._execute_single_stage(pipeline, stage, execution_log)
    
    async def _execute_single_stage(self, pipeline: CleansingPipeline, 
                                   stage: CleansingStage,
                                   execution_log: List[Dict[str, Any]]):
        """Execute a single pipeline stage with retry logic and quality gates."""
        
        stage.status = PipelineStageStatus.RUNNING
        stage.start_time = datetime.now()
        
        try:
            logger.info(f"Executing stage '{stage.name}' in pipeline {pipeline.pipeline_id}")
            
            # Calculate input quality
            current_data = pipeline.pipeline_context['current_data']
            stage.input_quality_score = self._calculate_data_quality_score(current_data)
            
            # Execute stage with retry logic
            for attempt in range(stage.retry_attempts + 1):
                stage.attempt_count = attempt + 1
                
                try:
                    # Execute stage function
                    if stage.timeout_seconds:
                        stage.result = await asyncio.wait_for(
                            self._run_stage_function(stage, pipeline.pipeline_context),
                            timeout=stage.timeout_seconds
                        )
                    else:
                        stage.result = await self._run_stage_function(stage, pipeline.pipeline_context)
                    
                    # Update pipeline context with stage result
                    pipeline.pipeline_context[f'stage_{stage.stage_id}_result'] = stage.result
                    
                    # Run quality gates
                    gate_results = await self._run_quality_gates(stage, pipeline)
                    
                    if all(gate_results.values()):
                        # All gates passed
                        stage.status = PipelineStageStatus.COMPLETED
                        stage.end_time = datetime.now()
                        stage.execution_time_seconds = (stage.end_time - stage.start_time).total_seconds()
                        
                        # Calculate output quality
                        stage.output_quality_score = self._calculate_data_quality_score(
                            pipeline.pipeline_context['current_data']
                        )
                        stage.quality_improvement = stage.output_quality_score - stage.input_quality_score
                        
                        pipeline.completed_stage_ids.append(stage.stage_id)
                        
                        # Create checkpoint
                        pipeline.checkpoint_data[stage.stage_id] = pipeline.pipeline_context['current_data'].copy()
                        
                        execution_log.append({
                            'stage_id': stage.stage_id,
                            'stage_name': stage.name,
                            'status': 'completed',
                            'timestamp': datetime.now(),
                            'execution_time_seconds': stage.execution_time_seconds,
                            'quality_improvement': stage.quality_improvement,
                            'attempt': attempt + 1
                        })
                        
                        logger.info(f"Stage '{stage.name}' completed successfully (attempt {attempt + 1})")
                        break
                    else:
                        # Quality gates failed
                        failed_gates = [gate_id for gate_id, passed in gate_results.items() if not passed]
                        error_msg = f"Quality gates failed: {failed_gates}"
                        
                        if attempt < stage.retry_attempts:
                            logger.warning(f"{error_msg} (attempt {attempt + 1}) - retrying")
                            await asyncio.sleep(stage.retry_delay_seconds)
                            continue
                        else:
                            stage.error = error_msg
                            raise RuntimeError(error_msg)
                
                except asyncio.TimeoutError:
                    error_msg = f"Stage '{stage.name}' timed out after {stage.timeout_seconds} seconds"
                    logger.warning(f"{error_msg} (attempt {attempt + 1})")
                    
                    if attempt < stage.retry_attempts:
                        await asyncio.sleep(stage.retry_delay_seconds)
                        continue
                    else:
                        stage.error = error_msg
                        raise
                        
                except Exception as e:
                    error_msg = f"Stage '{stage.name}' failed: {str(e)}"
                    logger.warning(f"{error_msg} (attempt {attempt + 1})")
                    
                    if attempt < stage.retry_attempts:
                        await asyncio.sleep(stage.retry_delay_seconds)
                        continue
                    else:
                        stage.error = error_msg
                        raise
        
        except Exception as e:
            # All retries exhausted or non-retryable error
            stage.status = PipelineStageStatus.FAILED
            stage.end_time = datetime.now()
            stage.execution_time_seconds = (stage.end_time - stage.start_time).total_seconds()
            stage.error = str(e)
            pipeline.failed_stage_ids.append(stage.stage_id)
            
            execution_log.append({
                'stage_id': stage.stage_id,
                'stage_name': stage.name,
                'status': 'failed',
                'timestamp': datetime.now(),
                'error': str(e),
                'attempt': stage.attempt_count
            })
            
            logger.error(f"Stage '{stage.name}' failed permanently: {e}")
        
        finally:
            # Remove from current execution set
            if stage.stage_id in pipeline.current_stage_ids:
                pipeline.current_stage_ids.remove(stage.stage_id)
    
    async def _run_stage_function(self, stage: CleansingStage, 
                                context: Dict[str, Any]) -> Any:
        """Run the stage function with proper context and parameter injection."""
        
        # Prepare parameters
        params = stage.parameters.copy()
        params['context'] = context
        
        # Execute based on stage type
        if stage.stage_type == 'comprehensive_cleansing':
            result = await self.cleansing_engine.cleanse_dataset(
                context['current_data'],
                cleansing_config=params
            )
            # Update current data
            context['current_data'] = result.cleansed_data
            return result
        
        elif stage.stage_type == 'domain_specific':
            domain = params.get('domain', 'general')
            result = self.domain_orchestrator.cleanse_by_domain(
                context['current_data'], 
                domain
            )
            return result
        
        elif stage.stage_type == 'custom':
            # Custom stage function
            if asyncio.iscoroutinefunction(stage.stage_function):
                result = await stage.stage_function(**params)
            else:
                # Run synchronous function in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(executor, lambda: stage.stage_function(**params))
            return result
        
        else:
            raise ValueError(f"Unknown stage type: {stage.stage_type}")
    
    async def _run_quality_gates(self, stage: CleansingStage, 
                               pipeline: CleansingPipeline) -> Dict[str, bool]:
        """Run quality gates for a stage."""
        
        gate_results = {}
        current_data = pipeline.pipeline_context['current_data']
        
        for gate in stage.quality_gates:
            try:
                # Calculate quality metrics
                quality_score = self._calculate_data_quality_score(current_data)
                completeness = 1 - (current_data.isnull().sum().sum() / (len(current_data) * len(current_data.columns)))
                
                # Check thresholds
                quality_passed = quality_score >= gate.quality_threshold
                completeness_passed = completeness >= gate.completeness_threshold
                
                # Run custom validators
                custom_passed = True
                for validator in gate.custom_validators:
                    try:
                        custom_passed = custom_passed and validator(current_data, gate.parameters)
                    except Exception as e:
                        logger.warning(f"Custom validator failed for gate {gate.gate_id}: {e}")
                        custom_passed = False
                
                gate_passed = quality_passed and completeness_passed and custom_passed
                gate_results[gate.gate_id] = gate_passed
                
                if not gate_passed and gate.blocking:
                    logger.warning(f"Blocking quality gate '{gate.name}' failed")
                
            except Exception as e:
                logger.error(f"Quality gate evaluation failed for {gate.gate_id}: {e}")
                gate_results[gate.gate_id] = False
        
        return gate_results
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        
        # Basic quality metrics
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        duplicate_rate = df.duplicated().sum() / len(df)
        uniqueness = 1 - duplicate_rate
        
        # Weighted average
        quality_score = (completeness * 0.5 + uniqueness * 0.5)
        
        return quality_score
    
    def _get_ready_stages(self, pipeline: CleansingPipeline) -> List[CleansingStage]:
        """Get stages that are ready to execute (all dependencies completed)."""
        
        ready_stages = []
        
        for stage in pipeline.stages:
            if (stage.status == PipelineStageStatus.PENDING and
                all(dep_id in pipeline.completed_stage_ids for dep_id in stage.dependencies)):
                ready_stages.append(stage)
        
        return ready_stages
    
    def _is_pipeline_complete(self, pipeline: CleansingPipeline) -> bool:
        """Check if pipeline is complete."""
        return len(pipeline.completed_stage_ids) == len(pipeline.stages)
    
    def _calculate_pipeline_quality_improvement(self, pipeline: CleansingPipeline) -> float:
        """Calculate overall pipeline quality improvement."""
        
        input_data = pipeline.pipeline_context['input_data']
        output_data = pipeline.pipeline_context['current_data']
        
        input_quality = self._calculate_data_quality_score(input_data)
        output_quality = self._calculate_data_quality_score(output_data)
        
        return output_quality - input_quality
    
    def _collect_stage_results(self, pipeline: CleansingPipeline) -> List[Dict[str, Any]]:
        """Collect results from all executed stages."""
        
        stage_results = []
        
        for stage in pipeline.stages:
            stage_result = {
                'stage_id': stage.stage_id,
                'stage_name': stage.name,
                'status': stage.status.value,
                'execution_time_seconds': stage.execution_time_seconds,
                'quality_improvement': stage.quality_improvement,
                'input_quality_score': stage.input_quality_score,
                'output_quality_score': stage.output_quality_score,
                'attempt_count': stage.attempt_count,
                'error': stage.error
            }
            stage_results.append(stage_result)
        
        return stage_results
    
    def _count_quality_gates_passed(self, pipeline: CleansingPipeline) -> int:
        """Count total quality gates passed."""
        total_gates = 0
        for stage in pipeline.stages:
            total_gates += len(stage.quality_gates)
        # Simplified counting - in real implementation, track individual gate results
        return total_gates
    
    def _count_quality_gates_failed(self, pipeline: CleansingPipeline) -> int:
        """Count total quality gates failed."""
        # Simplified counting - return 0 for now
        return 0
    
    # Pipeline template creation methods
    
    def _create_comprehensive_pipeline(self, pipeline_id: str, name: str, 
                                     parameters: Dict[str, Any]) -> CleansingPipeline:
        """Create comprehensive cleansing pipeline."""
        
        stages = [
            CleansingStage(
                stage_id="validation",
                name="Data Validation",
                description="Validate data integrity and basic quality",
                stage_type="custom",
                stage_function=self._validation_stage_function,
                timeout_seconds=300,
                quality_gates=[
                    QualityGate(
                        gate_id="basic_validation",
                        name="Basic Data Validation",
                        quality_threshold=0.6,
                        completeness_threshold=0.8
                    )
                ]
            ),
            CleansingStage(
                stage_id="standardization",
                name="Data Standardization", 
                description="Standardize data formats and values",
                stage_type="comprehensive_cleansing",
                dependencies=["validation"],
                parameters={"enable_standardization": True},
                timeout_seconds=600,
                quality_gates=[
                    QualityGate(
                        gate_id="standardization_quality",
                        name="Standardization Quality Check",
                        quality_threshold=0.75,
                        completeness_threshold=0.85
                    )
                ]
            ),
            CleansingStage(
                stage_id="deduplication",
                name="Duplicate Detection and Resolution",
                description="Remove duplicate records",
                stage_type="comprehensive_cleansing",
                dependencies=["standardization"],
                parameters={"enable_duplicate_detection": True},
                timeout_seconds=900,
                quality_gates=[
                    QualityGate(
                        gate_id="deduplication_quality",
                        name="Deduplication Quality Check",
                        quality_threshold=0.8,
                        completeness_threshold=0.9
                    )
                ]
            )
        ]
        
        return CleansingPipeline(
            pipeline_id=pipeline_id,
            name=name,
            description="Comprehensive data cleansing with validation, standardization, and deduplication",
            stages=stages,
            enable_parallel_execution=False,  # Sequential for comprehensive pipeline
            global_timeout_minutes=60
        )
    
    def _create_quick_standardization_pipeline(self, pipeline_id: str, name: str,
                                             parameters: Dict[str, Any]) -> CleansingPipeline:
        """Create quick standardization pipeline."""
        
        stages = [
            CleansingStage(
                stage_id="quick_standardization",
                name="Quick Data Standardization",
                description="Fast standardization of common data formats",
                stage_type="comprehensive_cleansing",
                parameters={"enable_standardization": True, "enable_duplicate_detection": False},
                timeout_seconds=300
            )
        ]
        
        return CleansingPipeline(
            pipeline_id=pipeline_id,
            name=name,
            description="Quick data standardization pipeline",
            stages=stages,
            global_timeout_minutes=15
        )
    
    def _create_duplicate_resolution_pipeline(self, pipeline_id: str, name: str,
                                            parameters: Dict[str, Any]) -> CleansingPipeline:
        """Create duplicate resolution pipeline."""
        
        stages = [
            CleansingStage(
                stage_id="duplicate_detection",
                name="Duplicate Detection",
                description="Intelligent duplicate detection and resolution",
                stage_type="comprehensive_cleansing",
                parameters={"enable_duplicate_detection": True, "enable_standardization": False},
                timeout_seconds=1200
            )
        ]
        
        return CleansingPipeline(
            pipeline_id=pipeline_id,
            name=name,
            description="Duplicate detection and resolution pipeline",
            stages=stages,
            global_timeout_minutes=30
        )
    
    def _create_domain_specific_pipeline(self, pipeline_id: str, name: str,
                                       parameters: Dict[str, Any]) -> CleansingPipeline:
        """Create domain-specific cleansing pipeline."""
        
        domain = parameters.get('domain', 'general')
        
        stages = [
            CleansingStage(
                stage_id="domain_cleansing",
                name=f"{domain.title()} Domain Cleansing",
                description=f"Domain-specific cleansing for {domain} data",
                stage_type="domain_specific",
                parameters={"domain": domain},
                timeout_seconds=600
            )
        ]
        
        return CleansingPipeline(
            pipeline_id=pipeline_id,
            name=name,
            description=f"Domain-specific cleansing pipeline for {domain} data",
            stages=stages,
            domain=domain,
            global_timeout_minutes=20
        )
    
    def _create_validation_pipeline(self, pipeline_id: str, name: str,
                                  parameters: Dict[str, Any]) -> CleansingPipeline:
        """Create validation-only pipeline."""
        
        stages = [
            CleansingStage(
                stage_id="data_validation",
                name="Data Validation",
                description="Comprehensive data validation and quality assessment",
                stage_type="custom",
                stage_function=self._validation_stage_function,
                timeout_seconds=300
            )
        ]
        
        return CleansingPipeline(
            pipeline_id=pipeline_id,
            name=name,
            description="Data validation and quality assessment pipeline",
            stages=stages,
            global_timeout_minutes=10
        )
    
    async def _validation_stage_function(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Validation stage function implementation."""
        
        data = context['current_data']
        
        # Basic validation metrics
        validation_result = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'completeness_rate': 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            'uniqueness_rate': 1 - (data.duplicated().sum() / len(data)),
            'validation_passed': True
        }
        
        # Quality score calculation
        validation_result['quality_score'] = (
            validation_result['completeness_rate'] * 0.6 + 
            validation_result['uniqueness_rate'] * 0.4
        )
        
        return validation_result
    
    def create_pipeline_from_template(self, template_name: str, 
                                    pipeline_name: str = None,
                                    parameters: Dict[str, Any] = None) -> CleansingPipeline:
        """Create pipeline from template."""
        
        if template_name not in self.pipeline_templates:
            raise ValueError(f"Unknown pipeline template: {template_name}")
        
        pipeline_id = str(uuid.uuid4())
        pipeline_name = pipeline_name or f"{template_name}_{pipeline_id[:8]}"
        parameters = parameters or {}
        
        pipeline = self.pipeline_templates[template_name](pipeline_id, pipeline_name, parameters)
        
        logger.info(f"Created pipeline '{pipeline_name}' from template '{template_name}'")
        
        return pipeline
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a pipeline."""
        
        if pipeline_id in self.active_pipelines:
            pipeline = self.active_pipelines[pipeline_id]
        else:
            pipeline = next((p for p in self.pipeline_history if p.pipeline_id == pipeline_id), None)
        
        if not pipeline:
            return None
        
        return {
            'pipeline_id': pipeline.pipeline_id,
            'name': pipeline.name,
            'status': pipeline.status.value,
            'start_time': pipeline.start_time.isoformat() if pipeline.start_time else None,
            'end_time': pipeline.end_time.isoformat() if pipeline.end_time else None,
            'total_execution_time_seconds': pipeline.total_execution_time_seconds,
            'stages_total': len(pipeline.stages),
            'stages_completed': len(pipeline.completed_stage_ids),
            'stages_failed': len(pipeline.failed_stage_ids),
            'current_stages': pipeline.current_stage_ids,
            'overall_quality_improvement': pipeline.overall_quality_improvement
        }
    
    def get_available_templates(self) -> List[str]:
        """Get list of available pipeline templates."""
        return list(self.pipeline_templates.keys())
    
    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel an active pipeline."""
        
        if pipeline_id not in self.active_pipelines:
            return False
        
        pipeline = self.active_pipelines[pipeline_id]
        pipeline.status = PipelineStatus.CANCELLED
        pipeline.end_time = datetime.now()
        pipeline.total_execution_time_seconds = (pipeline.end_time - pipeline.start_time).total_seconds()
        
        # Cancel all running stages
        for stage in pipeline.stages:
            if stage.status == PipelineStageStatus.RUNNING:
                stage.status = PipelineStageStatus.CANCELLED
                stage.end_time = datetime.now()
        
        # Move to history
        del self.active_pipelines[pipeline_id]
        self.pipeline_history.append(pipeline)
        
        logger.info(f"Cancelled pipeline {pipeline_id}")
        
        return True
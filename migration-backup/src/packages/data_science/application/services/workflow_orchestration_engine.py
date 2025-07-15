"""
Workflow Orchestration Engine - Phase 2.8 End-to-End Orchestration
Manages complex data science workflows with dependency management, parallel execution, and error recovery.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

logger = logging.getLogger(__name__)


class WorkflowStepStatus(Enum):
    """Status of individual workflow steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Overall workflow status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    
    step_id: str
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    retry_attempts: int = 0
    retry_delay_seconds: int = 5
    
    # Runtime state
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    attempt_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        return {
            'step_id': self.step_id,
            'name': self.name,
            'dependencies': self.dependencies,
            'parameters': self.parameters,
            'timeout_seconds': self.timeout_seconds,
            'retry_attempts': self.retry_attempts,
            'retry_delay_seconds': self.retry_delay_seconds,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error': self.error,
            'attempt_count': self.attempt_count
        }


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow."""
    
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    global_timeout_minutes: Optional[int] = None
    enable_parallel_execution: bool = True
    max_parallel_steps: int = 4
    continue_on_error: bool = False
    
    # Runtime state
    status: WorkflowStatus = WorkflowStatus.CREATED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_step_ids: Set[str] = field(default_factory=set)
    completed_step_ids: Set[str] = field(default_factory=set)
    failed_step_ids: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID."""
        return next((step for step in self.steps if step.step_id == step_id), None)
    
    def get_ready_steps(self) -> List[WorkflowStep]:
        """Get steps that are ready to execute (all dependencies completed)."""
        ready_steps = []
        
        for step in self.steps:
            if (step.status == WorkflowStepStatus.PENDING and
                all(dep_id in self.completed_step_ids for dep_id in step.dependencies)):
                ready_steps.append(step)
        
        return ready_steps
    
    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return len(self.completed_step_ids) == len(self.steps)
    
    def has_failures(self) -> bool:
        """Check if workflow has failed steps."""
        return len(self.failed_step_ids) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary representation."""
        return {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'global_timeout_minutes': self.global_timeout_minutes,
            'enable_parallel_execution': self.enable_parallel_execution,
            'max_parallel_steps': self.max_parallel_steps,
            'continue_on_error': self.continue_on_error,
            'steps': [step.to_dict() for step in self.steps],
            'current_step_ids': list(self.current_step_ids),
            'completed_step_ids': list(self.completed_step_ids),
            'failed_step_ids': list(self.failed_step_ids),
            'context': self.context
        }


class WorkflowOrchestrationEngine:
    """
    Advanced workflow orchestration engine for data science pipelines.
    Supports parallel execution, dependency management, error recovery, and monitoring.
    """
    
    def __init__(self, max_concurrent_workflows: int = 10):
        self.max_concurrent_workflows = max_concurrent_workflows
        self.active_workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_history: List[WorkflowDefinition] = []
        self.workflow_templates: Dict[str, Callable] = {}
        
        # Register built-in workflow templates
        self._register_builtin_templates()
        
        logger.info(f"Initialized WorkflowOrchestrationEngine with max {max_concurrent_workflows} concurrent workflows")
    
    def _register_builtin_templates(self):
        """Register built-in workflow templates."""
        self.workflow_templates.update({
            'comprehensive_data_science': self._create_comprehensive_workflow,
            'data_profiling_only': self._create_profiling_workflow,
            'ml_pipeline_only': self._create_ml_workflow,
            'quality_assessment_only': self._create_quality_workflow
        })
    
    async def create_workflow(self, template_name: str, workflow_name: str = None, 
                             parameters: Dict[str, Any] = None) -> WorkflowDefinition:
        """Create a new workflow from a template."""
        if template_name not in self.workflow_templates:
            raise ValueError(f"Unknown workflow template: {template_name}")
        
        workflow_id = str(uuid.uuid4())
        workflow_name = workflow_name or f"{template_name}_{workflow_id[:8]}"
        parameters = parameters or {}
        
        # Create workflow from template
        workflow = self.workflow_templates[template_name](workflow_id, workflow_name, parameters)
        
        logger.info(f"Created workflow '{workflow_name}' (ID: {workflow_id}) from template '{template_name}'")
        
        return workflow
    
    async def execute_workflow(self, workflow: WorkflowDefinition) -> WorkflowDefinition:
        """Execute a workflow with full orchestration."""
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            raise RuntimeError(f"Maximum concurrent workflows ({self.max_concurrent_workflows}) reached")
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.start_time = datetime.now()
        self.active_workflows[workflow.workflow_id] = workflow
        
        try:
            logger.info(f"Starting workflow execution: {workflow.name} ({workflow.workflow_id})")
            
            # Set global timeout if specified
            global_timeout = None
            if workflow.global_timeout_minutes:
                global_timeout = asyncio.create_task(
                    asyncio.sleep(workflow.global_timeout_minutes * 60)
                )
            
            # Main execution loop
            while not workflow.is_complete() and not workflow.has_failures():
                ready_steps = workflow.get_ready_steps()
                
                if not ready_steps:
                    if workflow.current_step_ids:
                        # Wait for running steps to complete
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        # No ready steps and no running steps - check if we can continue
                        if workflow.continue_on_error and workflow.has_failures():
                            logger.warning(f"Workflow {workflow.workflow_id} continuing despite failures")
                            break
                        else:
                            logger.error(f"Workflow {workflow.workflow_id} deadlocked - no ready steps")
                            workflow.status = WorkflowStatus.FAILED
                            break
                
                # Execute ready steps (parallel if enabled)
                if workflow.enable_parallel_execution:
                    await self._execute_steps_parallel(workflow, ready_steps)
                else:
                    await self._execute_steps_sequential(workflow, ready_steps)
                
                # Check global timeout
                if global_timeout and global_timeout.done():
                    logger.error(f"Workflow {workflow.workflow_id} timed out after {workflow.global_timeout_minutes} minutes")
                    workflow.status = WorkflowStatus.FAILED
                    break
            
            # Finalize workflow
            if workflow.is_complete() and not workflow.has_failures():
                workflow.status = WorkflowStatus.COMPLETED
                logger.info(f"Workflow {workflow.workflow_id} completed successfully")
            elif workflow.has_failures() and not workflow.continue_on_error:
                workflow.status = WorkflowStatus.FAILED
                logger.error(f"Workflow {workflow.workflow_id} failed with errors")
            
            workflow.end_time = datetime.now()
            
            # Cancel global timeout if set
            if global_timeout and not global_timeout.done():
                global_timeout.cancel()
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = datetime.now()
            logger.error(f"Workflow {workflow.workflow_id} failed with exception: {e}")
        
        finally:
            # Move from active to history
            if workflow.workflow_id in self.active_workflows:
                del self.active_workflows[workflow.workflow_id]
            self.workflow_history.append(workflow)
        
        return workflow
    
    async def _execute_steps_parallel(self, workflow: WorkflowDefinition, ready_steps: List[WorkflowStep]):
        """Execute steps in parallel with concurrency control."""
        # Limit concurrent execution
        steps_to_execute = ready_steps[:workflow.max_parallel_steps]
        
        # Create tasks for each step
        tasks = []
        for step in steps_to_execute:
            task = asyncio.create_task(self._execute_single_step(workflow, step))
            tasks.append(task)
            workflow.current_step_ids.add(step.step_id)
        
        # Wait for all tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_steps_sequential(self, workflow: WorkflowDefinition, ready_steps: List[WorkflowStep]):
        """Execute steps sequentially."""
        for step in ready_steps:
            workflow.current_step_ids.add(step.step_id)
            await self._execute_single_step(workflow, step)
    
    async def _execute_single_step(self, workflow: WorkflowDefinition, step: WorkflowStep):
        """Execute a single workflow step with retry logic."""
        step.status = WorkflowStepStatus.RUNNING
        step.start_time = datetime.now()
        
        try:
            logger.info(f"Executing step '{step.name}' in workflow {workflow.workflow_id}")
            
            for attempt in range(step.retry_attempts + 1):
                step.attempt_count = attempt + 1
                
                try:
                    # Set up timeout for individual step
                    if step.timeout_seconds:
                        step.result = await asyncio.wait_for(
                            self._run_step_function(step, workflow.context),
                            timeout=step.timeout_seconds
                        )
                    else:
                        step.result = await self._run_step_function(step, workflow.context)
                    
                    # Step completed successfully
                    step.status = WorkflowStepStatus.COMPLETED
                    step.end_time = datetime.now()
                    workflow.completed_step_ids.add(step.step_id)
                    
                    logger.info(f"Step '{step.name}' completed successfully (attempt {attempt + 1})")
                    break
                    
                except asyncio.TimeoutError:
                    error_msg = f"Step '{step.name}' timed out after {step.timeout_seconds} seconds"
                    logger.warning(f"{error_msg} (attempt {attempt + 1})")
                    
                    if attempt < step.retry_attempts:
                        await asyncio.sleep(step.retry_delay_seconds)
                        continue
                    else:
                        step.error = error_msg
                        raise
                        
                except Exception as e:
                    error_msg = f"Step '{step.name}' failed: {str(e)}"
                    logger.warning(f"{error_msg} (attempt {attempt + 1})")
                    
                    if attempt < step.retry_attempts:
                        await asyncio.sleep(step.retry_delay_seconds)
                        continue
                    else:
                        step.error = error_msg
                        raise
        
        except Exception as e:
            # All retries exhausted or non-retryable error
            step.status = WorkflowStepStatus.FAILED
            step.end_time = datetime.now()
            step.error = str(e)
            workflow.failed_step_ids.add(step.step_id)
            
            logger.error(f"Step '{step.name}' failed permanently: {e}")
        
        finally:
            # Remove from current execution set
            workflow.current_step_ids.discard(step.step_id)
    
    async def _run_step_function(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Run the step function with proper context and parameter injection."""
        # Prepare parameters with context
        params = step.parameters.copy()
        params['context'] = context
        
        # If function is async, await it
        if asyncio.iscoroutinefunction(step.function):
            result = await step.function(**params)
        else:
            # Run synchronous function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, lambda: step.function(**params))
        
        # Update context with step result
        context[f"step_{step.step_id}_result"] = result
        context[f"step_{step.name}_result"] = result
        
        return result
    
    def _create_comprehensive_workflow(self, workflow_id: str, name: str, 
                                     parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Create a comprehensive data science workflow."""
        steps = [
            WorkflowStep(
                step_id="data_loading",
                name="Load and Validate Data",
                function=self._data_loading_step,
                parameters=parameters,
                timeout_seconds=300,
                retry_attempts=1
            ),
            WorkflowStep(
                step_id="data_profiling",
                name="Data Profiling Analysis",
                function=self._data_profiling_step,
                dependencies=["data_loading"],
                timeout_seconds=600,
                retry_attempts=1
            ),
            WorkflowStep(
                step_id="quality_assessment", 
                name="Data Quality Assessment",
                function=self._quality_assessment_step,
                dependencies=["data_loading", "data_profiling"],
                timeout_seconds=300,
                retry_attempts=1
            ),
            WorkflowStep(
                step_id="statistical_analysis",
                name="Statistical Analysis",
                function=self._statistical_analysis_step,
                dependencies=["data_loading"],
                timeout_seconds=300,
                retry_attempts=1
            ),
            WorkflowStep(
                step_id="feature_engineering",
                name="Feature Engineering",
                function=self._feature_engineering_step,
                dependencies=["data_profiling", "quality_assessment"],
                timeout_seconds=600,
                retry_attempts=1
            ),
            WorkflowStep(
                step_id="ml_modeling",
                name="Machine Learning Modeling",
                function=self._ml_modeling_step,
                dependencies=["feature_engineering", "statistical_analysis"],
                timeout_seconds=1800,
                retry_attempts=1
            ),
            WorkflowStep(
                step_id="model_evaluation",
                name="Model Evaluation",
                function=self._model_evaluation_step,
                dependencies=["ml_modeling"],
                timeout_seconds=300,
                retry_attempts=1
            ),
            WorkflowStep(
                step_id="report_generation",
                name="Generate Analysis Report",
                function=self._report_generation_step,
                dependencies=["model_evaluation"],
                timeout_seconds=180,
                retry_attempts=2
            )
        ]
        
        return WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description="Comprehensive data science analysis workflow",
            steps=steps,
            global_timeout_minutes=60,
            enable_parallel_execution=True,
            max_parallel_steps=3,
            continue_on_error=False
        )
    
    def _create_profiling_workflow(self, workflow_id: str, name: str, 
                                 parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Create a data profiling-only workflow."""
        steps = [
            WorkflowStep(
                step_id="data_loading",
                name="Load Data",
                function=self._data_loading_step,
                parameters=parameters,
                timeout_seconds=300
            ),
            WorkflowStep(
                step_id="data_profiling", 
                name="Comprehensive Data Profiling",
                function=self._data_profiling_step,
                dependencies=["data_loading"],
                timeout_seconds=600
            ),
            WorkflowStep(
                step_id="profiling_report",
                name="Generate Profiling Report",
                function=self._profiling_report_step,
                dependencies=["data_profiling"],
                timeout_seconds=180
            )
        ]
        
        return WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description="Data profiling and analysis workflow",
            steps=steps,
            global_timeout_minutes=30,
            enable_parallel_execution=False
        )
    
    def _create_ml_workflow(self, workflow_id: str, name: str, 
                          parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Create an ML pipeline-only workflow."""
        steps = [
            WorkflowStep(
                step_id="data_loading",
                name="Load Training Data",
                function=self._data_loading_step,
                parameters=parameters,
                timeout_seconds=300
            ),
            WorkflowStep(
                step_id="data_preprocessing",
                name="Data Preprocessing",
                function=self._data_preprocessing_step,
                dependencies=["data_loading"],
                timeout_seconds=600
            ),
            WorkflowStep(
                step_id="feature_engineering",
                name="Feature Engineering", 
                function=self._feature_engineering_step,
                dependencies=["data_preprocessing"],
                timeout_seconds=600
            ),
            WorkflowStep(
                step_id="model_training",
                name="Model Training",
                function=self._ml_modeling_step,
                dependencies=["feature_engineering"],
                timeout_seconds=3600
            ),
            WorkflowStep(
                step_id="model_validation",
                name="Model Validation",
                function=self._model_evaluation_step,
                dependencies=["model_training"],
                timeout_seconds=300
            )
        ]
        
        return WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description="Machine learning pipeline workflow",
            steps=steps,
            global_timeout_minutes=90,
            enable_parallel_execution=True,
            max_parallel_steps=2
        )
    
    def _create_quality_workflow(self, workflow_id: str, name: str, 
                               parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Create a data quality assessment workflow."""
        steps = [
            WorkflowStep(
                step_id="data_loading",
                name="Load Data",
                function=self._data_loading_step,
                parameters=parameters,
                timeout_seconds=300
            ),
            WorkflowStep(
                step_id="quality_assessment",
                name="Data Quality Assessment",
                function=self._quality_assessment_step,
                dependencies=["data_loading"],
                timeout_seconds=300
            ),
            WorkflowStep(
                step_id="data_cleansing",
                name="Data Cleansing",
                function=self._data_cleansing_step,
                dependencies=["quality_assessment"],
                timeout_seconds=600
            ),
            WorkflowStep(
                step_id="quality_validation",
                name="Quality Validation",
                function=self._quality_validation_step,
                dependencies=["data_cleansing"],
                timeout_seconds=300
            )
        ]
        
        return WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description="Data quality assessment and cleansing workflow",
            steps=steps,
            global_timeout_minutes=30,
            enable_parallel_execution=False
        )
    
    # Step function implementations
    async def _data_loading_step(self, context: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Load data step implementation."""
        data_source = kwargs.get('data_source')
        
        if isinstance(data_source, pd.DataFrame):
            return data_source
        elif isinstance(data_source, str):
            # Load from file
            if data_source.endswith('.csv'):
                return pd.read_csv(data_source)
            elif data_source.endswith('.parquet'):
                return pd.read_parquet(data_source)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        else:
            raise ValueError("data_source must be DataFrame or file path")
    
    async def _data_profiling_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Data profiling step implementation."""
        data = context.get('step_data_loading_result')
        if data is None:
            raise ValueError("No data available from previous step")
        
        # Placeholder for actual profiling logic
        return {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'summary_stats': data.describe().to_dict()
        }
    
    async def _quality_assessment_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Quality assessment step implementation."""
        data = context.get('step_data_loading_result')
        profiling_result = context.get('step_data_profiling_result', {})
        
        if data is None:
            raise ValueError("No data available from previous step")
        
        # Calculate basic quality metrics
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        
        duplicate_rows = data.duplicated().sum()
        uniqueness = (data.shape[0] - duplicate_rows) / data.shape[0]
        
        overall_quality = (completeness + uniqueness) / 2
        
        return {
            'completeness_score': completeness,
            'uniqueness_score': uniqueness,
            'overall_quality_score': overall_quality,
            'total_rows': data.shape[0],
            'missing_values': missing_cells,
            'duplicate_rows': duplicate_rows
        }
    
    async def _statistical_analysis_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Statistical analysis step implementation."""
        data = context.get('step_data_loading_result')
        
        if data is None:
            raise ValueError("No data available from previous step")
        
        numeric_data = data.select_dtypes(include=['number'])
        
        return {
            'descriptive_stats': numeric_data.describe().to_dict(),
            'correlations': numeric_data.corr().to_dict() if len(numeric_data.columns) > 1 else {},
            'numeric_columns': list(numeric_data.columns),
            'categorical_columns': list(data.select_dtypes(include=['object']).columns)
        }
    
    async def _feature_engineering_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Feature engineering step implementation."""
        data = context.get('step_data_loading_result')
        profiling_result = context.get('step_data_profiling_result', {})
        quality_result = context.get('step_quality_assessment_result', {})
        
        if data is None:
            raise ValueError("No data available from previous step")
        
        # Basic feature engineering
        engineered_features = []
        processed_data = data.copy()
        
        # Handle missing values
        for col in data.columns:
            if data[col].isnull().any():
                if data[col].dtype in ['int64', 'float64']:
                    processed_data[col].fillna(data[col].median(), inplace=True)
                else:
                    processed_data[col].fillna('missing', inplace=True)
                engineered_features.append(f"filled_missing_{col}")
        
        return {
            'processed_data': processed_data,
            'engineered_features': engineered_features,
            'feature_count': len(processed_data.columns)
        }
    
    async def _ml_modeling_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """ML modeling step implementation."""
        # Get processed data from feature engineering or raw data
        processed_data = context.get('step_feature_engineering_result', {}).get('processed_data')
        if processed_data is None:
            processed_data = context.get('step_data_loading_result')
        
        if processed_data is None:
            raise ValueError("No data available from previous steps")
        
        target_column = kwargs.get('target_column')
        if not target_column or target_column not in processed_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Placeholder for actual ML modeling
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X = processed_data.drop(columns=[target_column])
        y = processed_data[target_column]
        
        # Handle categorical variables (simple encoding)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.Categorical(X[col]).codes
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'feature_names': list(X.columns),
            'target_column': target_column,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    async def _model_evaluation_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Model evaluation step implementation."""
        ml_result = context.get('step_ml_modeling_result') or context.get('step_model_training_result')
        
        if ml_result is None:
            raise ValueError("No ML model available from previous step")
        
        accuracy = ml_result.get('accuracy', 0)
        
        # Determine performance category
        if accuracy > 0.9:
            performance_category = "excellent"
        elif accuracy > 0.8:
            performance_category = "good"
        elif accuracy > 0.7:
            performance_category = "fair"
        else:
            performance_category = "poor"
        
        return {
            'model_performance': {
                'accuracy': accuracy,
                'performance_category': performance_category
            },
            'evaluation_summary': f"Model achieved {accuracy:.3f} accuracy ({performance_category} performance)",
            'recommendations': self._generate_model_recommendations(accuracy)
        }
    
    async def _report_generation_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Report generation step implementation."""
        # Collect results from all previous steps
        report_data = {
            'workflow_summary': {},
            'data_overview': {},
            'quality_assessment': {},
            'statistical_analysis': {},
            'model_performance': {},
            'recommendations': []
        }
        
        # Data overview
        if 'step_data_loading_result' in context:
            data = context['step_data_loading_result']
            report_data['data_overview'] = {
                'total_rows': data.shape[0],
                'total_columns': data.shape[1],
                'data_types': data.dtypes.value_counts().to_dict()
            }
        
        # Quality assessment
        if 'step_quality_assessment_result' in context:
            report_data['quality_assessment'] = context['step_quality_assessment_result']
        
        # Statistical analysis
        if 'step_statistical_analysis_result' in context:
            report_data['statistical_analysis'] = context['step_statistical_analysis_result']
        
        # Model performance
        if 'step_model_evaluation_result' in context:
            report_data['model_performance'] = context['step_model_evaluation_result']
        
        return report_data
    
    async def _profiling_report_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Profiling report generation step."""
        profiling_result = context.get('step_data_profiling_result')
        
        if profiling_result is None:
            raise ValueError("No profiling result available")
        
        return {
            'report_type': 'data_profiling',
            'profiling_summary': profiling_result,
            'generated_at': datetime.now().isoformat()
        }
    
    async def _data_preprocessing_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Data preprocessing step implementation."""
        data = context.get('step_data_loading_result')
        
        if data is None:
            raise ValueError("No data available from previous step")
        
        # Basic preprocessing
        processed_data = data.copy()
        
        # Remove completely empty rows/columns
        processed_data = processed_data.dropna(how='all')
        processed_data = processed_data.dropna(axis=1, how='all')
        
        return {
            'processed_data': processed_data,
            'original_shape': data.shape,
            'processed_shape': processed_data.shape
        }
    
    async def _data_cleansing_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Data cleansing step implementation."""
        data = context.get('step_data_loading_result')
        quality_result = context.get('step_quality_assessment_result', {})
        
        if data is None:
            raise ValueError("No data available from previous step")
        
        cleansed_data = data.copy()
        
        # Remove duplicates if quality is poor
        if quality_result.get('uniqueness_score', 1.0) < 0.9:
            cleansed_data = cleansed_data.drop_duplicates()
        
        return {
            'cleansed_data': cleansed_data,
            'original_rows': len(data),
            'cleansed_rows': len(cleansed_data),
            'rows_removed': len(data) - len(cleansed_data)
        }
    
    async def _quality_validation_step(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Quality validation step implementation."""
        cleansed_data = context.get('step_data_cleansing_result', {}).get('cleansed_data')
        original_quality = context.get('step_quality_assessment_result', {})
        
        if cleansed_data is None:
            raise ValueError("No cleansed data available from previous step")
        
        # Re-calculate quality metrics
        total_cells = cleansed_data.shape[0] * cleansed_data.shape[1]
        missing_cells = cleansed_data.isnull().sum().sum()
        new_completeness = (total_cells - missing_cells) / total_cells
        
        duplicate_rows = cleansed_data.duplicated().sum()
        new_uniqueness = (cleansed_data.shape[0] - duplicate_rows) / cleansed_data.shape[0]
        
        new_overall_quality = (new_completeness + new_uniqueness) / 2
        
        improvement = new_overall_quality - original_quality.get('overall_quality_score', 0)
        
        return {
            'validation_results': {
                'new_completeness': new_completeness,
                'new_uniqueness': new_uniqueness,
                'new_overall_quality': new_overall_quality,
                'quality_improvement': improvement,
                'validation_passed': new_overall_quality > 0.8
            }
        }
    
    def _generate_model_recommendations(self, accuracy: float) -> List[str]:
        """Generate recommendations based on model performance."""
        recommendations = []
        
        if accuracy < 0.6:
            recommendations.extend([
                "Consider feature engineering to improve model performance",
                "Try different algorithms (XGBoost, Neural Networks)",
                "Collect more training data if possible",
                "Review data quality and remove noise"
            ])
        elif accuracy < 0.8:
            recommendations.extend([
                "Model shows good potential - consider hyperparameter tuning",
                "Try ensemble methods to boost performance",
                "Feature selection might help reduce overfitting"
            ])
        else:
            recommendations.extend([
                "Excellent model performance achieved",
                "Consider validating on additional test sets",
                "Monitor for model drift in production"
            ])
        
        return recommendations
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow."""
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id].to_dict()
        
        # Check history
        for workflow in self.workflow_history:
            if workflow.workflow_id == workflow_id:
                return workflow.to_dict()
        
        return None
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all currently active workflows."""
        return [workflow.to_dict() for workflow in self.active_workflows.values()]
    
    def get_workflow_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        return [workflow.to_dict() for workflow in self.workflow_history[-limit:]]
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow."""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        workflow.end_time = datetime.now()
        
        # Cancel all running steps
        for step in workflow.steps:
            if step.status == WorkflowStepStatus.RUNNING:
                step.status = WorkflowStepStatus.CANCELLED
                step.end_time = datetime.now()
        
        # Move to history
        del self.active_workflows[workflow_id]
        self.workflow_history.append(workflow)
        
        logger.info(f"Cancelled workflow {workflow_id}")
        return True
    
    def get_available_templates(self) -> List[str]:
        """Get list of available workflow templates."""
        return list(self.workflow_templates.keys())
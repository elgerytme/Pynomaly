"""Machine Learning Pipeline domain service interface.

This service orchestrates ML pipeline operations including training, validation,
deployment, and monitoring of machine learning models within the data science framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID
from datetime import datetime

from ..entities.machine_learning_pipeline import MachineLearningPipeline, PipelineStatus
from ..entities.analysis_job import AnalysisJob


class IMLPipelineService(ABC):
    """Domain service for machine learning pipeline orchestration.
    
    This service provides high-level ML pipeline operations that involve
    multiple domain entities and complex business logic beyond simple CRUD.
    """
    
    @abstractmethod
    async def create_pipeline(self, name: str, pipeline_type: str,
                            steps_config: List[Dict[str, Any]], 
                            created_by: str,
                            description: Optional[str] = None,
                            parameters: Optional[Dict[str, Any]] = None) -> MachineLearningPipeline:
        """Create a new ML pipeline with validation.
        
        Args:
            name: Pipeline name
            pipeline_type: Type of pipeline (training, inference, etc.)
            steps_config: List of step configurations
            created_by: User who created the pipeline
            description: Optional description
            parameters: Optional pipeline parameters
            
        Returns:
            Created and validated ML pipeline
            
        Raises:
            ValidationError: If pipeline configuration is invalid
            PipelineError: If pipeline creation fails
        """
        pass
    
    @abstractmethod
    async def execute_pipeline(self, pipeline_id: UUID, 
                             input_data: Any,
                             execution_config: Optional[Dict[str, Any]] = None,
                             user_id: Optional[UUID] = None) -> str:
        """Execute a machine learning pipeline.
        
        Args:
            pipeline_id: ID of pipeline to execute
            input_data: Input data for pipeline execution
            execution_config: Optional execution configuration overrides
            user_id: User initiating the execution
            
        Returns:
            Execution ID for tracking
            
        Raises:
            PipelineError: If execution fails to start
            ValidationError: If pipeline is not executable
        """
        pass
    
    @abstractmethod
    async def get_execution_status(self, pipeline_id: UUID, 
                                 execution_id: str) -> Dict[str, Any]:
        """Get detailed execution status and progress.
        
        Args:
            pipeline_id: Pipeline ID
            execution_id: Execution ID
            
        Returns:
            Execution status with progress details
        """
        pass
    
    @abstractmethod
    async def train_model(self, pipeline_id: UUID,
                        model_config: Dict[str, Any],
                        training_data: Any,
                        validation_data: Optional[Any] = None,
                        hyperparameter_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train a machine learning model within pipeline.
        
        Args:
            pipeline_id: Pipeline context
            model_config: Model configuration (algorithm, parameters)
            training_data: Training dataset
            validation_data: Optional validation dataset
            hyperparameter_config: Hyperparameter optimization config
            
        Returns:
            Training results with model artifacts and metrics
            
        Raises:
            TrainingError: If model training fails
        """
        pass
    
    @abstractmethod
    async def optimize_hyperparameters(self, pipeline_id: UUID,
                                     model_config: Dict[str, Any],
                                     training_data: Any,
                                     validation_data: Any,
                                     optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model hyperparameters using various strategies.
        
        Args:
            pipeline_id: Pipeline context
            model_config: Base model configuration
            training_data: Training dataset
            validation_data: Validation dataset
            optimization_config: Optimization strategy and parameters
            
        Returns:
            Optimization results with best parameters and metrics
            
        Raises:
            OptimizationError: If hyperparameter optimization fails
        """
        pass
    
    @abstractmethod
    async def evaluate_model(self, pipeline_id: UUID,
                           model_id: UUID,
                           test_data: Any,
                           evaluation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate a trained model comprehensively.
        
        Args:
            pipeline_id: Pipeline context
            model_id: Model to evaluate
            test_data: Test dataset
            evaluation_config: Evaluation configuration
            
        Returns:
            Comprehensive evaluation results and metrics
            
        Raises:
            EvaluationError: If model evaluation fails
        """
        pass
    
    @abstractmethod
    async def compare_models(self, pipeline_id: UUID,
                           model_ids: List[UUID],
                           comparison_data: Any,
                           comparison_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compare multiple models and recommend best performer.
        
        Args:
            pipeline_id: Pipeline context
            model_ids: List of models to compare
            comparison_data: Dataset for comparison
            comparison_config: Comparison configuration and metrics
            
        Returns:
            Model comparison results with recommendations
        """
        pass
    
    @abstractmethod
    async def create_ensemble(self, pipeline_id: UUID,
                            model_ids: List[UUID],
                            ensemble_config: Dict[str, Any],
                            validation_data: Any) -> Dict[str, Any]:
        """Create an ensemble from multiple models.
        
        Args:
            pipeline_id: Pipeline context
            model_ids: Models to include in ensemble
            ensemble_config: Ensemble configuration (voting, stacking, etc.)
            validation_data: Data for ensemble validation
            
        Returns:
            Ensemble model results and performance metrics
            
        Raises:
            EnsembleError: If ensemble creation fails
        """
        pass
    
    @abstractmethod
    async def deploy_model(self, pipeline_id: UUID,
                         model_id: UUID,
                         deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a trained model for inference.
        
        Args:
            pipeline_id: Pipeline context
            model_id: Model to deploy
            deployment_config: Deployment configuration
            
        Returns:
            Deployment results with endpoint information
            
        Raises:
            DeploymentError: If model deployment fails
        """
        pass
    
    @abstractmethod
    async def run_automl(self, pipeline_id: UUID,
                       training_data: Any,
                       target_metric: str,
                       automl_config: Dict[str, Any],
                       time_budget_minutes: Optional[int] = None) -> Dict[str, Any]:
        """Run automated machine learning pipeline.
        
        Args:
            pipeline_id: Pipeline context
            training_data: Training dataset
            target_metric: Metric to optimize
            automl_config: AutoML configuration
            time_budget_minutes: Time budget for optimization
            
        Returns:
            AutoML results with best models and insights
            
        Raises:
            AutoMLError: If automated ML fails
        """
        pass
    
    @abstractmethod
    async def validate_pipeline_configuration(self, pipeline: MachineLearningPipeline) -> Dict[str, Any]:
        """Validate pipeline configuration and dependencies.
        
        Args:
            pipeline: Pipeline to validate
            
        Returns:
            Validation results with errors and warnings
        """
        pass
    
    @abstractmethod
    async def get_pipeline_lineage(self, pipeline_id: UUID) -> Dict[str, Any]:
        """Get complete pipeline lineage including data and model dependencies.
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            Lineage information with dependency graph
        """
        pass
    
    @abstractmethod
    async def monitor_pipeline_performance(self, pipeline_id: UUID,
                                         monitoring_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Monitor pipeline performance and detect drift.
        
        Args:
            pipeline_id: Pipeline to monitor
            monitoring_config: Monitoring configuration
            
        Returns:
            Performance monitoring results
        """
        pass
    
    @abstractmethod
    async def feature_engineering(self, pipeline_id: UUID,
                                raw_data: Any,
                                feature_config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Perform automated feature engineering.
        
        Args:
            pipeline_id: Pipeline context
            raw_data: Raw input data
            feature_config: Feature engineering configuration
            
        Returns:
            Tuple of (engineered_features, feature_metadata)
            
        Raises:
            FeatureEngineeringError: If feature engineering fails
        """
        pass
    
    @abstractmethod
    async def preprocess_data(self, pipeline_id: UUID,
                            raw_data: Any,
                            preprocessing_config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Preprocess data according to pipeline configuration.
        
        Args:
            pipeline_id: Pipeline context
            raw_data: Raw input data
            preprocessing_config: Preprocessing configuration
            
        Returns:
            Tuple of (processed_data, preprocessing_metadata)
        """
        pass
    
    @abstractmethod
    async def schedule_pipeline(self, pipeline_id: UUID,
                              schedule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule pipeline for recurring execution.
        
        Args:
            pipeline_id: Pipeline to schedule
            schedule_config: Schedule configuration (cron, interval, etc.)
            
        Returns:
            Scheduling results and next execution time
        """
        pass
    
    @abstractmethod
    async def generate_pipeline_report(self, pipeline_id: UUID,
                                     execution_id: Optional[str] = None,
                                     report_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution report.
        
        Args:
            pipeline_id: Pipeline ID
            execution_id: Specific execution ID (latest if None)
            report_config: Report generation configuration
            
        Returns:
            Comprehensive pipeline report
        """
        pass
    
    @abstractmethod
    async def pause_pipeline(self, pipeline_id: UUID, execution_id: str) -> None:
        """Pause a running pipeline execution.
        
        Args:
            pipeline_id: Pipeline ID
            execution_id: Execution to pause
            
        Raises:
            PipelineError: If pipeline cannot be paused
        """
        pass
    
    @abstractmethod
    async def resume_pipeline(self, pipeline_id: UUID, execution_id: str) -> None:
        """Resume a paused pipeline execution.
        
        Args:
            pipeline_id: Pipeline ID
            execution_id: Execution to resume
            
        Raises:
            PipelineError: If pipeline cannot be resumed
        """
        pass
    
    @abstractmethod
    async def stop_pipeline(self, pipeline_id: UUID, execution_id: str,
                          reason: Optional[str] = None) -> None:
        """Stop a running pipeline execution.
        
        Args:
            pipeline_id: Pipeline ID
            execution_id: Execution to stop
            reason: Optional reason for stopping
            
        Raises:
            PipelineError: If pipeline cannot be stopped
        """
        pass
    
    @abstractmethod
    async def retry_failed_step(self, pipeline_id: UUID, execution_id: str,
                              step_name: str, retry_config: Optional[Dict[str, Any]] = None) -> None:
        """Retry a failed pipeline step.
        
        Args:
            pipeline_id: Pipeline ID
            execution_id: Execution ID
            step_name: Step to retry
            retry_config: Optional retry configuration
            
        Raises:
            PipelineError: If step cannot be retried
        """
        pass
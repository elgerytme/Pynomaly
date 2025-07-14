"""Data pipeline use case implementation."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Union
from pathlib import Path

import pandas as pd

from ...domain.entities.transformation_pipeline import TransformationPipeline
from ...domain.value_objects.pipeline_config import PipelineConfig
from ...domain.value_objects.transformation_step import TransformationStep, StepType
from ...domain.services.data_cleaning_service import DataCleaningService
from ...infrastructure.adapters.data_source_adapter import DataSourceAdapter
from ...infrastructure.processors.feature_processor import FeatureProcessor
from ..dto.pipeline_result import PipelineResult

logger = logging.getLogger(__name__)


class DataPipelineUseCase:
    """
    Use case for executing complete data transformation pipelines.
    
    Orchestrates the entire data transformation workflow from source to
    processed output, including data loading, cleaning, feature engineering,
    and export operations.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        data_source_adapter: Optional[DataSourceAdapter] = None,
        feature_processor: Optional[FeatureProcessor] = None,
        cleaning_service: Optional[DataCleaningService] = None
    ) -> None:
        """
        Initialize the data pipeline use case.
        
        Args:
            config: Pipeline configuration
            data_source_adapter: Adapter for data source operations
            feature_processor: Processor for feature engineering
            cleaning_service: Service for data cleaning operations
        """
        self.config = config
        self.data_source_adapter = data_source_adapter or DataSourceAdapter()
        self.feature_processor = feature_processor or FeatureProcessor()
        self.cleaning_service = cleaning_service or DataCleaningService()
        
        self._pipeline: Optional[TransformationPipeline] = None
    
    def execute(
        self,
        source: Union[str, Path, pd.DataFrame],
        config: Optional[PipelineConfig] = None
    ) -> PipelineResult:
        """
        Execute the complete data transformation pipeline.
        
        Args:
            source: Data source (file path or DataFrame)
            config: Optional pipeline configuration (overrides instance config)
            
        Returns:
            Pipeline execution result with transformed data and metadata
        """
        # Use provided config or instance config
        pipeline_config = config or self.config
        if pipeline_config is None:
            raise ValueError("Pipeline configuration is required")
        
        # Create pipeline entity
        self._pipeline = self._create_pipeline(pipeline_config, source)
        
        logger.info(f"Starting pipeline execution: {self._pipeline.name}")
        
        try:
            # Execute pipeline steps
            result = self._execute_pipeline_steps(source)
            
            # Mark pipeline as completed
            self._pipeline.complete_execution(
                records_processed=result.records_processed,
                features_created=result.features_created,
                execution_time=result.execution_time_seconds,
                memory_usage=result.memory_usage_mb
            )
            
            logger.info(f"Pipeline completed successfully: {self._pipeline.name}")
            return result
            
        except Exception as e:
            # Mark pipeline as failed
            self._pipeline.fail_execution(str(e))
            logger.error(f"Pipeline failed: {self._pipeline.name} - {str(e)}")
            raise
    
    def get_pipeline_status(self) -> Optional[TransformationPipeline]:
        """Get current pipeline status."""
        return self._pipeline
    
    def _create_pipeline(
        self,
        config: PipelineConfig,
        source: Union[str, Path, pd.DataFrame]
    ) -> TransformationPipeline:
        """Create a transformation pipeline entity."""
        source_name = str(source) if not isinstance(source, pd.DataFrame) else "DataFrame"
        
        pipeline = TransformationPipeline(
            name=f"data_transformation_{int(time.time())}",
            description=f"Data transformation pipeline for {source_name}",
            config=config
        )
        
        # Add pipeline steps based on configuration
        self._add_pipeline_steps(pipeline, config)
        
        return pipeline
    
    def _add_pipeline_steps(
        self,
        pipeline: TransformationPipeline,
        config: PipelineConfig
    ) -> None:
        """Add transformation steps to the pipeline based on configuration."""
        
        # Data loading step
        pipeline.add_step(
            TransformationStep.create_data_loading_step(
                source_path=config.source_path or "",
                source_type=config.source_type,
                **config.source_params
            )
        )
        
        # Data cleaning step
        if config.cleaning_strategy.value != "none":
            pipeline.add_step(
                TransformationStep.create_cleaning_step(
                    strategy=config.cleaning_strategy.value,
                    missing_threshold=config.missing_value_threshold,
                    outlier_method=config.outlier_detection_method,
                    outlier_threshold=config.outlier_threshold
                )
            )
        
        # Feature engineering step
        if config.feature_engineering:
            methods = []
            if config.polynomial_features:
                methods.append("polynomial")
            if config.interaction_features:
                methods.append("interactions")
            if config.time_features:
                methods.append("temporal")
            
            pipeline.add_step(
                TransformationStep.create_feature_engineering_step(
                    methods=methods,
                    polynomial_degree=config.polynomial_degree
                )
            )
        
        # Feature scaling step
        if config.scaling_method.value != "none":
            pipeline.add_step(
                TransformationStep.create_scaling_step(
                    method=config.scaling_method.value
                )
            )
        
        # Data export step
        pipeline.add_step(
            TransformationStep(
                name="data_export",
                step_type=StepType.DATA_EXPORT,
                description="Export processed data",
                parameters={
                    "format": config.output_format,
                    "compression": config.compression
                },
                order=100
            )
        )
    
    def _execute_pipeline_steps(
        self,
        source: Union[str, Path, pd.DataFrame]
    ) -> PipelineResult:
        """Execute all pipeline steps in order."""
        if not self._pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        self._pipeline.start_execution()
        
        # Track execution metrics
        start_time = time.time()
        current_data = None
        
        # Sort steps by order
        sorted_steps = sorted(self._pipeline.steps, key=lambda x: x.order)
        
        # Execute each step
        for step in sorted_steps:
            logger.info(f"Executing step: {step.name}")
            
            step.start_execution()
            step_start_time = time.time()
            
            try:
                # Execute step based on type
                if step.step_type == StepType.DATA_LOADING:
                    current_data = self._execute_data_loading_step(step, source)
                elif step.step_type == StepType.DATA_CLEANING:
                    current_data = self._execute_cleaning_step(step, current_data)
                elif step.step_type == StepType.FEATURE_ENGINEERING:
                    current_data = self._execute_feature_engineering_step(step, current_data)
                elif step.step_type == StepType.FEATURE_SCALING:
                    current_data = self._execute_scaling_step(step, current_data)
                elif step.step_type == StepType.DATA_EXPORT:
                    # Export step doesn't modify data
                    pass
                else:
                    logger.warning(f"Unknown step type: {step.step_type}")
                
                # Mark step as completed
                step_execution_time = time.time() - step_start_time
                
                if current_data is not None:
                    step.complete_execution(
                        input_records=step.input_records or 0,
                        output_records=len(current_data),
                        input_features=step.input_features or 0,
                        output_features=len(current_data.columns) if hasattr(current_data, 'columns') else 0,
                        execution_time=step_execution_time
                    )
                else:
                    step.complete_execution(0, 0, 0, 0, step_execution_time)
                
                logger.info(f"Step completed: {step.name}")
                
            except Exception as e:
                step.fail_execution(str(e))
                logger.error(f"Step failed: {step.name} - {str(e)}")
                raise
        
        # Calculate final metrics
        total_execution_time = time.time() - start_time
        
        # Create pipeline result
        return PipelineResult(
            data=current_data,
            pipeline_id=self._pipeline.id,
            records_processed=len(current_data) if current_data is not None else 0,
            features_created=len(current_data.columns) if current_data is not None and hasattr(current_data, 'columns') else 0,
            execution_time_seconds=total_execution_time,
            memory_usage_mb=self._estimate_memory_usage(current_data),
            steps_completed=len([s for s in sorted_steps if s.is_completed]),
            steps_failed=len([s for s in sorted_steps if s.has_failed]),
            metadata=self._collect_pipeline_metadata()
        )
    
    def _execute_data_loading_step(
        self,
        step: TransformationStep,
        source: Union[str, Path, pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute data loading step."""
        if isinstance(source, pd.DataFrame):
            data = source.copy()
        else:
            data = self.data_source_adapter.load_data(
                source_path=str(source),
                source_type=step.parameters.get("source_type", "csv"),
                **step.parameters
            )
        
        step.input_records = 0
        step.input_features = 0
        
        return data
    
    def _execute_cleaning_step(
        self,
        step: TransformationStep,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Execute data cleaning step."""
        step.input_records = len(data)
        step.input_features = len(data.columns)
        
        cleaned_data, cleaning_report = self.cleaning_service.clean_data(
            data=data,
            strategy=step.parameters.get("strategy", "auto"),
            missing_value_threshold=step.parameters.get("missing_threshold", 0.5),
            outlier_detection_method=step.parameters.get("outlier_method", "iqr"),
            outlier_threshold=step.parameters.get("outlier_threshold", 1.5)
        )
        
        # Store cleaning report in step metadata
        step.metadata["cleaning_report"] = cleaning_report
        
        return cleaned_data
    
    def _execute_feature_engineering_step(
        self,
        step: TransformationStep,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Execute feature engineering step."""
        step.input_records = len(data)
        step.input_features = len(data.columns)
        
        methods = step.parameters.get("methods", [])
        
        # Apply feature engineering based on methods
        processed_data = data.copy()
        
        if "polynomial" in methods:
            processed_data = self.feature_processor.create_polynomial_features(
                processed_data,
                degree=step.parameters.get("polynomial_degree", 2)
            )
        
        if "interactions" in methods:
            processed_data = self.feature_processor.create_interaction_features(processed_data)
        
        if "temporal" in methods:
            processed_data = self.feature_processor.create_temporal_features(processed_data)
        
        return processed_data
    
    def _execute_scaling_step(
        self,
        step: TransformationStep,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Execute feature scaling step."""
        step.input_records = len(data)
        step.input_features = len(data.columns)
        
        scaling_method = step.parameters.get("method", "standard")
        
        scaled_data = self.feature_processor.scale_features(
            data=data,
            method=scaling_method
        )
        
        return scaled_data
    
    def _estimate_memory_usage(self, data: Optional[pd.DataFrame]) -> float:
        """Estimate memory usage in MB."""
        if data is None:
            return 0.0
        
        try:
            return data.memory_usage(deep=True).sum() / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _collect_pipeline_metadata(self) -> Dict[str, Any]:
        """Collect metadata from pipeline execution."""
        if not self._pipeline:
            return {}
        
        return {
            "total_steps": len(self._pipeline.steps),
            "completed_steps": len([s for s in self._pipeline.steps if s.is_completed]),
            "failed_steps": len([s for s in self._pipeline.steps if s.has_failed]),
            "step_details": [step.to_dict() for step in self._pipeline.steps],
            "config": self._pipeline.config.to_dict()
        }
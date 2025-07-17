#!/usr/bin/env python3
"""
Pipeline Orchestrator - Core orchestration logic only
Coordinates pipeline execution but delegates actual work to specialized services
"""

import logging
from datetime import datetime

import pandas as pd

from monorepo.application.services.data_processing_service import DataProcessingService
from monorepo.application.services.model_optimization_service import (
    ModelOptimizationService,
)
from monorepo.application.services.pipeline_config_service import PipelineConfigService
from monorepo.application.services.pipeline_monitoring_service import (
    PipelineMonitoringService,
)
from monorepo.application.services.pipeline_result_service import PipelineResultService
from monorepo.domain.models.pipeline_models import (
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates pipeline execution by coordinating specialized services"""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

        # Initialize specialized services
        self.config_service = PipelineConfigService(self.config)
        self.monitoring_service = PipelineMonitoringService()
        self.data_service = DataProcessingService(self.config)
        self.optimization_service = ModelOptimizationService(self.config)
        self.result_service = PipelineResultService()

        # Current pipeline tracking
        self.current_pipeline: PipelineResult | None = None

    async def run_complete_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        pipeline_id: str | None = None,
    ) -> PipelineResult:
        """
        Run the complete pipeline by orchestrating specialized services

        Args:
            X: Input features
            y: Target variable (optional for unsupervised)
            pipeline_id: Optional pipeline identifier

        Returns:
            Complete pipeline result
        """
        # Initialize pipeline
        if pipeline_id is None:
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"🚀 Starting pipeline: {pipeline_id}")

        # Create pipeline result
        result = PipelineResult(
            pipeline_id=pipeline_id, config=self.config, start_time=datetime.now()
        )

        self.current_pipeline = result

        try:
            # Run pipeline stages
            await self._execute_pipeline_stages(result, X, y)

            # Finalize pipeline
            result.final_stage = PipelineStage.COMPLETED
            result.end_time = datetime.now()
            result.total_duration_seconds = (
                result.end_time - result.start_time
            ).total_seconds()

            # Save results
            await self.result_service.save_pipeline_result(result)

            logger.info(f"✅ Pipeline completed: {pipeline_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            result.final_stage = PipelineStage.FAILED
            result.end_time = datetime.now()
            result.total_duration_seconds = (
                result.end_time - result.start_time
            ).total_seconds()
            raise

    async def _execute_pipeline_stages(
        self, result: PipelineResult, X: pd.DataFrame, y: pd.Series | None
    ) -> None:
        """Execute all pipeline stages in order"""

        # Stage 1: Data Processing
        processed_data = await self.monitoring_service.run_monitored_stage(
            result, PipelineStage.DATA_VALIDATION, self.data_service.process_data, X, y
        )

        if processed_data:
            X = processed_data.get("X", X)
            y = processed_data.get("y", y)

        # Stage 2: Processor Optimization
        optimization_result = await self.monitoring_service.run_monitored_stage(
            result,
            PipelineStage.HYPERPARAMETER_OPTIMIZATION,
            self.optimization_service.optimize_processors,
            X,
            y,
        )

        if optimization_result:
            result.best_processor = optimization_result.get("best_processor")
            result.best_processor_params = optimization_result.get("best_params", {})
            result.best_processor_performance = optimization_result.get(
                "best_performance", {}
            )

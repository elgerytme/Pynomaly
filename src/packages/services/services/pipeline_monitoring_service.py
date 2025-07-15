#!/usr/bin/env python3
"""
Pipeline Monitoring Service - Handles stage execution monitoring and resource tracking
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pynomaly.domain.models.pipeline_models import (
    PipelineResult,
    PipelineStage,
    PipelineStageResult,
)

logger = logging.getLogger(__name__)


class PipelineMonitoringService:
    """Service responsible for monitoring pipeline execution"""

    def __init__(self):
        self.resource_monitor = ResourceMonitor()

    async def run_monitored_stage(
        self,
        result: PipelineResult,
        stage: PipelineStage,
        stage_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Run a pipeline stage with monitoring and error handling"""

        logger.info(f"🔧 Running stage: {stage.value}")

        stage_result = PipelineStageResult(
            stage=stage, status="running", start_time=datetime.now()
        )

        result.stage_results[stage] = stage_result

        try:
            # Monitor resources before stage
            memory_before = self.resource_monitor.get_memory_usage()

            # Run stage function
            stage_output = await stage_func(*args, **kwargs)

            # Update stage result
            stage_result.end_time = datetime.now()
            stage_result.duration_seconds = (
                stage_result.end_time - stage_result.start_time
            ).total_seconds()
            stage_result.status = "success"
            stage_result.memory_usage_mb = (
                self.resource_monitor.get_memory_usage() - memory_before
            )

            if isinstance(stage_output, dict):
                stage_result.outputs.update(stage_output)
            elif stage_output is not None:
                stage_result.outputs["result"] = stage_output

            logger.info(
                f"✅ Stage completed: {stage.value} ({stage_result.duration_seconds:.2f}s)"
            )

            return stage_output

        except Exception as e:
            stage_result.end_time = datetime.now()
            stage_result.duration_seconds = (
                stage_result.end_time - stage_result.start_time
            ).total_seconds()
            stage_result.status = "failed"
            stage_result.error_message = str(e)

            logger.error(f"❌ Stage failed: {stage.value} - {e}")
            raise


class ResourceMonitor:
    """Monitor resource usage during pipeline execution"""

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil

            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0

#!/usr/bin/env python3
"""
Pipeline Result Service - Handles pipeline result storage and retrieval
"""

import json
import logging
from pathlib import Path

from pynomaly.domain.models.pipeline_models import PipelineResult

logger = logging.getLogger(__name__)


class PipelineResultService:
    """Service responsible for pipeline result management"""

    def __init__(self, artifacts_dir: str = "artifacts/automl_pipelines"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    async def save_pipeline_result(self, result: PipelineResult) -> str:
        """Save pipeline result to disk and return file path"""

        results_path = self.artifacts_dir / f"{result.pipeline_id}_results.json"

        # Convert result to serializable format
        serializable_result = {
            "pipeline_id": result.pipeline_id,
            "config": result.config.__dict__,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "total_duration_seconds": result.total_duration_seconds,
            "final_stage": result.final_stage.value,
            "best_model_params": result.best_model_params,
            "best_model_performance": result.best_model_performance,
            "ensemble_performance": result.ensemble_performance,
            "cross_validation_scores": result.cross_validation_scores,
            "improvement_recommendations": result.improvement_recommendations,
            "production_readiness_score": result.production_readiness_score,
            "stage_results": {
                stage.value: {
                    "status": stage_result.status,
                    "duration_seconds": stage_result.duration_seconds,
                    "outputs": stage_result.outputs,
                    "error_message": stage_result.error_message,
                }
                for stage, stage_result in result.stage_results.items()
            },
        }

        with open(results_path, "w") as f:
            json.dump(serializable_result, f, indent=2)

        logger.info(f"💾 Pipeline results saved: {results_path}")
        return str(results_path)

    def get_pipeline_summary(self, pipeline_id: str) -> dict | None:
        """Get summary of a completed pipeline"""

        results_path = self.artifacts_dir / f"{pipeline_id}_results.json"

        if not results_path.exists():
            return None

        try:
            with open(results_path) as f:
                result_data = json.load(f)

            return {
                "pipeline_id": result_data["pipeline_id"],
                "status": result_data["final_stage"],
                "duration_seconds": result_data["total_duration_seconds"],
                "best_model_performance": result_data["best_model_performance"],
                "production_readiness_score": result_data["production_readiness_score"],
                "recommendations": result_data["improvement_recommendations"],
            }

        except Exception as e:
            logger.error(f"Failed to load pipeline summary for {pipeline_id}: {e}")
            return None

    def list_pipeline_results(self) -> list[dict]:
        """List all saved pipeline results"""

        results = []

        for results_file in self.artifacts_dir.glob("*_results.json"):
            try:
                with open(results_file) as f:
                    result_data = json.load(f)

                results.append({
                    "pipeline_id": result_data["pipeline_id"],
                    "final_stage": result_data["final_stage"],
                    "start_time": result_data["start_time"],
                    "duration_seconds": result_data["total_duration_seconds"],
                    "production_readiness_score": result_data.get("production_readiness_score", 0.0),
                })

            except Exception as e:
                logger.warning(f"Failed to read result file {results_file}: {e}")
                continue

        # Sort by start time (most recent first)
        results.sort(key=lambda x: x["start_time"], reverse=True)
        return results

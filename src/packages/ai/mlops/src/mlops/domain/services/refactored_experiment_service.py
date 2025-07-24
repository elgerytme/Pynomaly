"""Refactored experiment service using hexagonal architecture."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd

from mlops.domain.interfaces.experiment_tracking_operations import (
    ExperimentTrackingPort,
    ExperimentRunPort,
    ArtifactManagementPort,
    ExperimentAnalysisPort,
    MetricsTrackingPort,
    ExperimentSearchPort,
    ExperimentConfig,
    ExperimentInfo,
    ExperimentStatus,
    RunConfig,
    RunInfo,
    RunStatus,
    RunMetrics,
    ComparisonRequest,
    ComparisonResult
)

logger = logging.getLogger(__name__)


class ExperimentService:
    """Clean domain service for experiment management using dependency injection."""
    
    def __init__(
        self,
        experiment_tracking_port: ExperimentTrackingPort,
        experiment_run_port: ExperimentRunPort,
        artifact_management_port: ArtifactManagementPort,
        experiment_analysis_port: ExperimentAnalysisPort,
        metrics_tracking_port: MetricsTrackingPort,
        experiment_search_port: ExperimentSearchPort
    ):
        """Initialize service with injected dependencies.
        
        Args:
            experiment_tracking_port: Port for experiment tracking operations
            experiment_run_port: Port for experiment run operations  
            artifact_management_port: Port for artifact management
            experiment_analysis_port: Port for experiment analysis
            metrics_tracking_port: Port for metrics tracking
            experiment_search_port: Port for experiment search
        """
        self._experiment_tracking_port = experiment_tracking_port
        self._experiment_run_port = experiment_run_port
        self._artifact_management_port = artifact_management_port
        self._experiment_analysis_port = experiment_analysis_port
        self._metrics_tracking_port = metrics_tracking_port
        self._experiment_search_port = experiment_search_port
        
        logger.info("ExperimentService initialized with clean architecture")
    
    async def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_by: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new experiment with business logic validation.
        
        Args:
            name: Experiment name
            description: Optional description
            tags: Optional tags for categorization
            created_by: User creating the experiment
            metadata: Optional metadata
            
        Returns:
            Experiment ID
            
        Raises:
            ValueError: If validation fails
        """
        # Domain validation
        if not name or not name.strip():
            raise ValueError("Experiment name cannot be empty")
        
        if len(name) > 100:
            raise ValueError("Experiment name cannot exceed 100 characters")
        
        # Apply business rules
        processed_tags = self._process_experiment_tags(tags or [])
        validated_metadata = self._validate_experiment_metadata(metadata or {})
        
        config = ExperimentConfig(
            name=name.strip(),
            description=description,
            tags=processed_tags,
            metadata=validated_metadata
        )
        
        try:
            experiment_id = await self._experiment_tracking_port.create_experiment(config, created_by)
            logger.info(f"Created experiment {experiment_id} for user {created_by}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def start_experiment_run(
        self,
        experiment_id: str,
        detector_name: str,
        dataset_name: str,
        parameters: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """Start a new experiment run with validation.
        
        Args:
            experiment_id: ID of the experiment
            detector_name: Name of the detector/algorithm
            dataset_name: Name of the dataset
            parameters: Run parameters
            tags: Optional run tags
            
        Returns:
            Run ID
            
        Raises:
            ValueError: If validation fails
        """
        # Validate experiment exists
        experiment = await self._experiment_tracking_port.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if experiment.status == ExperimentStatus.CANCELLED:
            raise ValueError("Cannot start run on cancelled experiment")
        
        # Domain validation
        if not detector_name or not detector_name.strip():
            raise ValueError("Detector name cannot be empty")
        
        if not dataset_name or not dataset_name.strip():
            raise ValueError("Dataset name cannot be empty")
        
        # Validate parameters
        validated_parameters = self._validate_run_parameters(parameters)
        
        run_config = RunConfig(
            experiment_id=experiment_id,
            detector_name=detector_name.strip(),
            dataset_name=dataset_name.strip(),
            parameters=validated_parameters,
            tags=tags
        )
        
        try:
            run_id = await self._experiment_run_port.create_run(run_config)
            
            # Update experiment status if needed
            if experiment.status == ExperimentStatus.CREATED:
                await self._experiment_tracking_port.update_experiment_status(
                    experiment_id, ExperimentStatus.RUNNING
                )
            
            logger.info(f"Started run {run_id} for experiment {experiment_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start experiment run: {e}")
            raise
    
    async def log_run_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log metrics for a run with business logic.
        
        Args:
            run_id: ID of the run
            metrics: Metrics to log
            step: Optional step number
        """
        # Validate run exists and is active
        run = await self._experiment_run_port.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        
        if run.status not in [RunStatus.STARTED, RunStatus.RUNNING]:
            raise ValueError(f"Cannot log metrics for run in status {run.status.value}")
        
        # Validate and process metrics
        validated_metrics = self._validate_metrics(metrics)
        
        try:
            # Log individual metrics with timestamps
            await self._metrics_tracking_port.log_metrics_batch(
                run_id, validated_metrics, step
            )
            
            # Update run metrics summary
            run_metrics = self._create_run_metrics_from_dict(validated_metrics)
            await self._experiment_run_port.update_run_metrics(run_id, run_metrics)
            
            # Update run status to running if it was just started
            if run.status == RunStatus.STARTED:
                await self._experiment_run_port.update_run_status(run_id, RunStatus.RUNNING)
            
            logger.info(f"Logged metrics for run {run_id}: {list(validated_metrics.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to log run metrics: {e}")
            raise
    
    async def finish_experiment_run(
        self,
        run_id: str,
        final_metrics: Dict[str, float],
        status: RunStatus = RunStatus.COMPLETED
    ) -> RunInfo:
        """Finish an experiment run with final metrics.
        
        Args:
            run_id: ID of the run
            final_metrics: Final metrics for the run
            status: Final status (completed or failed)
            
        Returns:
            Updated run information
        """
        # Validate run exists
        run = await self._experiment_run_port.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        
        if run.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
            raise ValueError(f"Run {run_id} is already finished with status {run.status.value}")
        
        # Validate final metrics
        validated_metrics = self._validate_metrics(final_metrics)
        run_metrics = self._create_run_metrics_from_dict(validated_metrics)
        
        try:
            # Finish the run
            await self._experiment_run_port.finish_run(run_id, run_metrics)
            
            # Log final metrics
            await self._metrics_tracking_port.log_metrics_batch(run_id, validated_metrics)
            
            # Get updated run info
            updated_run = await self._experiment_run_port.get_run(run_id)
            
            # Check if experiment should be marked as completed
            await self._check_experiment_completion(run.experiment_id)
            
            logger.info(f"Finished run {run_id} with status {status.value}")
            return updated_run
            
        except Exception as e:
            logger.error(f"Failed to finish experiment run: {e}")
            raise
    
    async def compare_experiment_runs(
        self,
        experiment_id: str,
        run_ids: Optional[List[str]] = None,
        metric: str = "f1_score",
        include_parameters: bool = True
    ) -> ComparisonResult:
        """Compare runs within an experiment.
        
        Args:
            experiment_id: ID of the experiment
            run_ids: Specific runs to compare (None = all)
            metric: Primary metric for comparison
            include_parameters: Include parameters in comparison
            
        Returns:
            Comparison results
        """
        # Validate experiment exists
        experiment = await self._experiment_tracking_port.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Validate runs exist if specified
        if run_ids:
            for run_id in run_ids:
                run = await self._experiment_run_port.get_run(run_id)
                if not run:
                    raise ValueError(f"Run {run_id} not found")
                if run.experiment_id != experiment_id:
                    raise ValueError(f"Run {run_id} does not belong to experiment {experiment_id}")
        
        request = ComparisonRequest(
            experiment_id=experiment_id,
            run_ids=run_ids,
            include_parameters=include_parameters,
            sort_by=metric
        )
        
        try:
            comparison_result = await self._experiment_analysis_port.compare_runs(request)
            
            # Apply business logic to enhance comparison
            enhanced_result = self._enhance_comparison_result(comparison_result)
            
            logger.info(f"Compared runs for experiment {experiment_id}")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Failed to compare experiment runs: {e}")
            raise
    
    async def get_best_run(
        self,
        experiment_id: str,
        metric: str = "f1_score",
        higher_is_better: bool = True
    ) -> Optional[RunInfo]:
        """Get the best performing run from an experiment.
        
        Args:
            experiment_id: ID of the experiment
            metric: Metric to optimize for
            higher_is_better: Whether higher values are better
            
        Returns:
            Best run information or None
        """
        # Validate experiment exists
        experiment = await self._experiment_tracking_port.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        try:
            best_run = await self._experiment_analysis_port.get_best_run(
                experiment_id, metric, higher_is_better
            )
            
            if best_run:
                logger.info(f"Found best run {best_run.run_id} for experiment {experiment_id}")
            else:
                logger.warning(f"No runs found for experiment {experiment_id}")
            
            return best_run
            
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            raise
    
    async def search_experiments(
        self,
        query: str,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[ExperimentInfo]:
        """Search experiments with enhanced business logic.
        
        Args:
            query: Search query
            created_by: Filter by creator
            tags: Filter by tags
            limit: Maximum results
            
        Returns:
            List of matching experiments
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        try:
            # Prepare filters
            filters = {}
            if created_by:
                filters["created_by"] = created_by
            if tags:
                filters["tags"] = tags
            
            results = await self._experiment_search_port.search_experiments(
                query.strip(),
                filters=filters,
                limit=limit
            )
            
            # Apply business logic to rank results
            ranked_results = self._rank_search_results(results, query)
            
            logger.info(f"Found {len(ranked_results)} experiments for query '{query}'")
            return ranked_results
            
        except Exception as e:
            logger.error(f"Failed to search experiments: {e}")
            raise
    
    async def generate_experiment_report(
        self,
        experiment_id: str,
        include_artifacts: bool = False,
        include_detailed_metrics: bool = True
    ) -> str:
        """Generate comprehensive experiment report.
        
        Args:
            experiment_id: ID of the experiment
            include_artifacts: Include artifact information
            include_detailed_metrics: Include detailed metrics
            
        Returns:
            Report content in markdown format
        """
        # Validate experiment exists
        experiment = await self._experiment_tracking_port.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        try:
            # Get basic report
            base_report = await self._experiment_analysis_port.generate_experiment_report(
                experiment_id, include_artifacts
            )
            
            # Enhance with business context
            enhanced_report = await self._enhance_experiment_report(
                experiment, base_report, include_detailed_metrics
            )
            
            logger.info(f"Generated report for experiment {experiment_id}")
            return enhanced_report
            
        except Exception as e:
            logger.error(f"Failed to generate experiment report: {e}")
            raise
    
    # Private helper methods
    
    def _process_experiment_tags(self, tags: List[str]) -> List[str]:
        """Process and validate experiment tags."""
        processed_tags = []
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                clean_tag = tag.strip().lower().replace(" ", "_")
                if len(clean_tag) <= 50 and clean_tag not in processed_tags:
                    processed_tags.append(clean_tag)
        return processed_tags
    
    def _validate_experiment_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experiment metadata."""
        validated = {}
        for key, value in metadata.items():
            if isinstance(key, str) and len(key) <= 100:
                # Only allow simple types in metadata
                if isinstance(value, (str, int, float, bool)):
                    validated[key] = value
                elif isinstance(value, list) and all(isinstance(x, (str, int, float)) for x in value):
                    validated[key] = value
        return validated
    
    def _validate_run_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate run parameters."""
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary")
        
        validated = {}
        for key, value in parameters.items():
            if not isinstance(key, str):
                raise ValueError(f"Parameter key must be string: {key}")
            
            # Validate parameter values
            if isinstance(value, (str, int, float, bool)):
                validated[key] = value
            elif isinstance(value, list) and all(isinstance(x, (str, int, float)) for x in value):
                validated[key] = value
            else:
                logger.warning(f"Skipping unsupported parameter type: {key}={type(value)}")
        
        return validated
    
    def _validate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Validate metrics dictionary."""
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be a dictionary")
        
        validated = {}
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise ValueError(f"Metric name must be string: {key}")
            
            try:
                float_value = float(value)
                if not (float('-inf') < float_value < float('inf')):
                    raise ValueError(f"Invalid metric value: {key}={value}")
                validated[key] = float_value
            except (ValueError, TypeError):
                raise ValueError(f"Metric value must be numeric: {key}={value}")
        
        return validated
    
    def _create_run_metrics_from_dict(self, metrics_dict: Dict[str, float]) -> RunMetrics:
        """Create RunMetrics object from dictionary."""
        return RunMetrics(
            accuracy=metrics_dict.get("accuracy"),
            precision=metrics_dict.get("precision"),
            recall=metrics_dict.get("recall"),
            f1_score=metrics_dict.get("f1_score"),
            auc_roc=metrics_dict.get("auc_roc"),
            loss=metrics_dict.get("loss"),
            training_time=metrics_dict.get("training_time"),
            inference_time=metrics_dict.get("inference_time"),
            custom_metrics={
                k: v for k, v in metrics_dict.items()
                if k not in ["accuracy", "precision", "recall", "f1_score", "auc_roc", "loss", "training_time", "inference_time"]
            }
        )
    
    async def _check_experiment_completion(self, experiment_id: str) -> None:
        """Check if experiment should be marked as completed."""
        try:
            runs = await self._experiment_run_port.list_runs(experiment_id)
            
            if runs:
                # Check if all runs are finished
                active_runs = [
                    run for run in runs 
                    if run.status in [RunStatus.STARTED, RunStatus.RUNNING]
                ]
                
                if not active_runs:
                    # All runs are finished, mark experiment as completed
                    await self._experiment_tracking_port.update_experiment_status(
                        experiment_id, ExperimentStatus.COMPLETED
                    )
                    logger.info(f"Marked experiment {experiment_id} as completed")
        
        except Exception as e:
            logger.warning(f"Failed to check experiment completion: {e}")
    
    def _enhance_comparison_result(self, result: ComparisonResult) -> ComparisonResult:
        """Enhance comparison result with business insights."""
        # Add business logic like statistical significance, confidence intervals, etc.
        if not result.comparison_data.empty:
            # Add rank column if not present
            if "rank" not in result.comparison_data.columns:
                result.comparison_data["rank"] = range(1, len(result.comparison_data) + 1)
            
            # Add performance categories
            if result.metric_used in result.comparison_data.columns:
                metric_values = result.comparison_data[result.metric_used]
                result.comparison_data["performance_category"] = pd.cut(
                    metric_values,
                    bins=3,
                    labels=["Low", "Medium", "High"]
                )
        
        return result
    
    def _rank_search_results(self, results: List[ExperimentInfo], query: str) -> List[ExperimentInfo]:
        """Rank search results by relevance."""
        # Simple ranking by name match and recency
        query_lower = query.lower()
        
        def relevance_score(experiment: ExperimentInfo) -> float:
            score = 0.0
            
            # Name exact match
            if query_lower in experiment.name.lower():
                score += 10.0
            
            # Tag match
            for tag in experiment.tags:
                if query_lower in tag.lower():
                    score += 5.0
            
            # Description match
            if experiment.description and query_lower in experiment.description.lower():
                score += 3.0
            
            # Recency bonus (more recent = higher score)
            days_old = (datetime.utcnow() - experiment.created_at).days
            recency_score = max(0, 30 - days_old) / 30.0
            score += recency_score
            
            return score
        
        # Sort by relevance score
        ranked_results = sorted(results, key=relevance_score, reverse=True)
        return ranked_results
    
    async def _enhance_experiment_report(
        self,
        experiment: ExperimentInfo,
        base_report: str,
        include_detailed_metrics: bool
    ) -> str:
        """Enhance experiment report with additional business context."""
        try:
            # Add executive summary
            runs = await self._experiment_run_port.list_runs(experiment.experiment_id)
            
            summary_section = f"""
# Executive Summary

**Experiment Status**: {experiment.status.value.title()}
**Total Runs**: {len(runs)}
**Duration**: {(experiment.updated_at - experiment.created_at).days} days
**Created By**: {experiment.created_by}

## Key Insights
"""
            
            if runs:
                completed_runs = [r for r in runs if r.status == RunStatus.COMPLETED]
                if completed_runs:
                    # Find best performing run
                    best_run = max(
                        completed_runs,
                        key=lambda r: r.metrics.f1_score or 0,
                        default=None
                    )
                    
                    if best_run and best_run.metrics.f1_score:
                        summary_section += f"- Best F1-score achieved: {best_run.metrics.f1_score:.3f}\n"
                        summary_section += f"- Best performing algorithm: {best_run.detector_name}\n"
                
                summary_section += f"- Success rate: {len(completed_runs)}/{len(runs)} runs completed\n"
            else:
                summary_section += "- No runs completed yet\n"
            
            # Combine with base report
            enhanced_report = summary_section + "\n" + base_report
            
            return enhanced_report
            
        except Exception as e:
            logger.warning(f"Failed to enhance report: {e}")
            return base_report
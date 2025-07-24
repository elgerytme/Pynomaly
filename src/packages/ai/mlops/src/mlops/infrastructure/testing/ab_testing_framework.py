"""
A/B Testing Framework for ML Models

Comprehensive framework for conducting A/B tests on ML models with statistical
analysis, experiment tracking, and automated decision making.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
from collections import defaultdict
import statistics

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import structlog

from mlops.domain.entities.model import Model, ModelVersion
from mlops.infrastructure.serving.realtime_inference_engine import InferenceRequest, InferenceResponse


class ExperimentStatus(Enum):
    """A/B test experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AllocationMethod(Enum):
    """Traffic allocation methods."""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    WEIGHTED = "weighted"
    STRATIFIED = "stratified"


class SignificanceTest(Enum):
    """Statistical significance tests."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    CHI_SQUARE = "chi_square"
    BOOTSTRAP = "bootstrap"


@dataclass
class ExperimentArm:
    """Individual experiment arm (model variant)."""
    arm_id: str
    name: str
    model_id: str
    model_version: str
    traffic_percentage: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    total_requests: int = 0
    successful_requests: int = 0
    total_latency_ms: float = 0.0
    
    # Business metrics
    conversions: int = 0
    revenue: float = 0.0
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def get_avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests
    
    def get_conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.total_requests == 0:
            return 0.0
        return self.conversions / self.total_requests


@dataclass
class ExperimentConfig:
    """A/B test experiment configuration."""
    name: str
    description: str
    arms: List[ExperimentArm]
    
    # Traffic allocation
    allocation_method: AllocationMethod = AllocationMethod.RANDOM
    allocation_key: str = "user_id"  # Key for hash-based allocation
    
    # Duration and sample size
    min_duration_hours: int = 24
    max_duration_hours: int = 168  # 1 week
    min_sample_size_per_arm: int = 1000
    target_sample_size_per_arm: int = 10000
    
    # Statistical configuration
    significance_level: float = 0.05
    statistical_power: float = 0.8
    minimum_detectable_effect: float = 0.05  # 5% relative change
    significance_test: SignificanceTest = SignificanceTest.T_TEST
    
    # Business metrics
    primary_metric: str = "conversion_rate"
    secondary_metrics: List[str] = field(default_factory=list)
    guardrail_metrics: List[str] = field(default_factory=list)
    
    # Safety and monitoring
    enable_early_stopping: bool = True
    early_stopping_threshold: float = 0.01  # Stop if p-value < 0.01
    enable_traffic_ramping: bool = True
    initial_traffic_percentage: float = 10.0  # Start with 10% traffic
    ramp_duration_hours: int = 24
    
    # Filtering and segmentation
    inclusion_filters: Dict[str, Any] = field(default_factory=dict)
    exclusion_filters: Dict[str, Any] = field(default_factory=dict)
    segment_definitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results of an A/B test experiment."""
    experiment_id: str
    arm_id: str
    metric_name: str
    value: float
    sample_size: int
    confidence_interval_lower: float
    confidence_interval_upper: float
    p_value: Optional[float] = None
    statistical_significance: bool = False
    practical_significance: bool = False
    effect_size: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExperimentSummary:
    """Summary of experiment results."""
    experiment_id: str
    status: ExperimentStatus
    started_at: datetime
    ended_at: Optional[datetime]
    duration_hours: float
    total_participants: int
    
    # Results per arm
    arm_results: Dict[str, Dict[str, ExperimentResult]] = field(default_factory=dict)
    
    # Overall conclusions
    winning_arm: Optional[str] = None
    confidence_in_winner: float = 0.0
    recommendation: str = ""
    statistical_significance_achieved: bool = False
    practical_significance_achieved: bool = False


class ABTestingFramework:
    """Comprehensive A/B testing framework for ML models."""
    
    def __init__(self, inference_engine=None):
        self.inference_engine = inference_engine
        self.logger = structlog.get_logger(__name__)
        
        # Experiment management
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_status: Dict[str, ExperimentStatus] = {}
        self.experiment_data: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        
        # Traffic allocation
        self.traffic_allocator = TrafficAllocator()
        
        # Statistical analysis
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Background monitoring
        self.monitoring_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Results storage
        self.experiment_results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self.experiment_summaries: Dict[str, ExperimentSummary] = {}
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment."""
        
        experiment_id = str(uuid.uuid4())
        
        # Validate configuration
        self._validate_experiment_config(config)
        
        # Store experiment
        self.experiments[experiment_id] = config
        self.experiment_status[experiment_id] = ExperimentStatus.DRAFT
        
        # Initialize data storage
        for arm in config.arms:
            self.experiment_data[experiment_id][arm.arm_id] = []
        
        self.logger.info(
            "A/B test experiment created",
            experiment_id=experiment_id,
            experiment_name=config.name,
            arms=len(config.arms),
            primary_metric=config.primary_metric
        )
        
        return experiment_id
    
    async def start_experiment(self, experiment_id: str) -> None:
        """Start an A/B test experiment."""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if self.experiment_status[experiment_id] != ExperimentStatus.DRAFT:
            raise ValueError(f"Experiment {experiment_id} cannot be started (status: {self.experiment_status[experiment_id]})")
        
        config = self.experiments[experiment_id]
        
        # Update status
        self.experiment_status[experiment_id] = ExperimentStatus.RUNNING
        
        # Initialize experiment summary
        self.experiment_summaries[experiment_id] = ExperimentSummary(
            experiment_id=experiment_id,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.utcnow(),
            ended_at=None,
            duration_hours=0.0,
            total_participants=0
        )
        
        # Start monitoring task
        if not self.is_running:
            await self.start_monitoring()
        
        self.logger.info(
            "A/B test experiment started",
            experiment_id=experiment_id,
            experiment_name=config.name
        )
    
    async def stop_experiment(self, experiment_id: str, reason: str = "") -> ExperimentSummary:
        """Stop an A/B test experiment and generate final results."""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        
        # Update status
        self.experiment_status[experiment_id] = ExperimentStatus.COMPLETED
        
        # Generate final analysis
        summary = await self._generate_experiment_summary(experiment_id)
        summary.ended_at = datetime.utcnow()
        summary.status = ExperimentStatus.COMPLETED
        summary.duration_hours = (summary.ended_at - summary.started_at).total_seconds() / 3600
        
        self.experiment_summaries[experiment_id] = summary
        
        self.logger.info(
            "A/B test experiment stopped",
            experiment_id=experiment_id,
            experiment_name=config.name,
            reason=reason,
            winning_arm=summary.winning_arm,
            confidence=summary.confidence_in_winner
        )
        
        return summary
    
    async def allocate_traffic(self, 
                             experiment_id: str,
                             request: InferenceRequest) -> Optional[str]:
        """Allocate traffic to experiment arm."""
        
        if experiment_id not in self.experiments:
            return None
        
        if self.experiment_status[experiment_id] != ExperimentStatus.RUNNING:
            return None
        
        config = self.experiments[experiment_id]
        
        # Check inclusion/exclusion filters
        if not self._passes_filters(request, config):
            return None
        
        # Allocate to arm
        arm_id = await self.traffic_allocator.allocate(
            experiment_id, config, request
        )
        
        return arm_id
    
    async def record_experiment_data(self,
                                   experiment_id: str,
                                   arm_id: str,
                                   request: InferenceRequest,
                                   response: InferenceResponse,
                                   business_metrics: Dict[str, Any] = None) -> None:
        """Record data for experiment analysis."""
        
        if experiment_id not in self.experiments:
            return
        
        config = self.experiments[experiment_id]
        
        # Find arm
        arm = next((a for a in config.arms if a.arm_id == arm_id), None)
        if not arm:
            return
        
        # Update arm statistics
        arm.total_requests += 1
        if response.warnings:
            # Consider requests with warnings as failed
            pass
        else:
            arm.successful_requests += 1
            arm.total_latency_ms += response.processing_time_ms
        
        # Record business metrics
        if business_metrics:
            if "conversion" in business_metrics:
                if business_metrics["conversion"]:
                    arm.conversions += 1
            
            if "revenue" in business_metrics:
                arm.revenue += business_metrics["revenue"]
            
            # Record custom metrics
            for metric_name, value in business_metrics.items():
                if metric_name not in ["conversion", "revenue"]:
                    if metric_name not in arm.custom_metrics:
                        arm.custom_metrics[metric_name] = []
                    arm.custom_metrics[metric_name].append(value)
        
        # Store detailed data for analysis
        data_point = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request.request_id,
            "model_id": response.model_id,
            "model_version": response.model_version,
            "processing_time_ms": response.processing_time_ms,
            "success": len(response.warnings) == 0,
            "business_metrics": business_metrics or {}
        }
        
        self.experiment_data[experiment_id][arm_id].append(data_point)
        
        # Update experiment summary
        if experiment_id in self.experiment_summaries:
            self.experiment_summaries[experiment_id].total_participants += 1
    
    async def get_experiment_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Get current experiment results."""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        return await self._analyze_experiment(experiment_id)
    
    async def _analyze_experiment(self, experiment_id: str) -> List[ExperimentResult]:
        """Analyze experiment data and calculate results."""
        
        config = self.experiments[experiment_id]
        results = []
        
        # Calculate results for each metric
        metrics_to_analyze = [config.primary_metric] + config.secondary_metrics
        
        for metric_name in metrics_to_analyze:
            metric_results = await self._analyze_metric(experiment_id, metric_name)
            results.extend(metric_results)
        
        # Store results
        self.experiment_results[experiment_id] = results
        
        return results
    
    async def _analyze_metric(self, experiment_id: str, metric_name: str) -> List[ExperimentResult]:
        """Analyze a specific metric across experiment arms."""
        
        config = self.experiments[experiment_id]
        results = []
        
        # Collect metric values for each arm
        arm_values = {}
        
        for arm in config.arms:
            values = self._extract_metric_values(experiment_id, arm.arm_id, metric_name)
            if values:
                arm_values[arm.arm_id] = values
        
        # Calculate results for each arm
        for arm_id, values in arm_values.items():
            if not values:
                continue
            
            # Calculate basic statistics
            mean_value = statistics.mean(values)
            sample_size = len(values)
            
            # Calculate confidence interval
            if sample_size > 1:
                std_error = statistics.stdev(values) / (sample_size ** 0.5)
                t_critical = stats.t.ppf(1 - config.significance_level / 2, sample_size - 1)
                margin_error = t_critical * std_error
                
                ci_lower = mean_value - margin_error
                ci_upper = mean_value + margin_error
            else:
                ci_lower = mean_value
                ci_upper = mean_value
            
            result = ExperimentResult(
                experiment_id=experiment_id,
                arm_id=arm_id,
                metric_name=metric_name,
                value=mean_value,
                sample_size=sample_size,
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper
            )
            
            results.append(result)
        
        # Perform statistical significance tests between arms
        if len(arm_values) >= 2:
            results = await self._perform_significance_tests(
                experiment_id, metric_name, arm_values, results, config
            )
        
        return results
    
    def _extract_metric_values(self, experiment_id: str, arm_id: str, metric_name: str) -> List[float]:
        """Extract metric values for an arm."""
        
        values = []
        data_points = self.experiment_data[experiment_id][arm_id]
        
        for point in data_points:
            if metric_name == "conversion_rate":
                # Conversion is binary (0 or 1)
                conversion = point["business_metrics"].get("conversion", False)
                values.append(1.0 if conversion else 0.0)
            
            elif metric_name == "revenue":
                revenue = point["business_metrics"].get("revenue", 0.0)
                values.append(revenue)
            
            elif metric_name == "response_time":
                values.append(point["processing_time_ms"])
            
            elif metric_name == "success_rate":
                values.append(1.0 if point["success"] else 0.0)
            
            elif metric_name in point["business_metrics"]:
                metric_value = point["business_metrics"][metric_name]
                if isinstance(metric_value, (int, float)):
                    values.append(float(metric_value))
        
        return values
    
    async def _perform_significance_tests(self,
                                        experiment_id: str,
                                        metric_name: str,
                                        arm_values: Dict[str, List[float]],
                                        results: List[ExperimentResult],
                                        config: ExperimentConfig) -> List[ExperimentResult]:
        """Perform statistical significance tests between arms."""
        
        arm_ids = list(arm_values.keys())
        
        # Compare each pair of arms
        for i in range(len(arm_ids)):
            for j in range(i + 1, len(arm_ids)):
                arm_a = arm_ids[i]
                arm_b = arm_ids[j]
                
                values_a = arm_values[arm_a]
                values_b = arm_values[arm_b]
                
                if len(values_a) < 10 or len(values_b) < 10:
                    continue  # Need minimum sample size
                
                # Perform statistical test
                p_value, effect_size = await self.statistical_analyzer.compare_arms(
                    values_a, values_b, config.significance_test
                )
                
                # Update results with significance information
                for result in results:
                    if result.arm_id == arm_a and result.metric_name == metric_name:
                        result.p_value = p_value
                        result.statistical_significance = p_value < config.significance_level
                        result.effect_size = effect_size
                        result.practical_significance = abs(effect_size) >= config.minimum_detectable_effect
                    
                    elif result.arm_id == arm_b and result.metric_name == metric_name:
                        result.p_value = p_value
                        result.statistical_significance = p_value < config.significance_level
                        result.effect_size = -effect_size  # Reverse for second arm
                        result.practical_significance = abs(effect_size) >= config.minimum_detectable_effect
        
        return results
    
    async def _generate_experiment_summary(self, experiment_id: str) -> ExperimentSummary:
        """Generate comprehensive experiment summary."""
        
        config = self.experiments[experiment_id]
        existing_summary = self.experiment_summaries.get(experiment_id)
        
        if not existing_summary:
            existing_summary = ExperimentSummary(
                experiment_id=experiment_id,
                status=self.experiment_status[experiment_id],
                started_at=datetime.utcnow(),
                ended_at=None,
                duration_hours=0.0,
                total_participants=0
            )
        
        # Analyze current results
        results = await self._analyze_experiment(experiment_id)
        
        # Group results by arm and metric
        arm_results = defaultdict(dict)
        for result in results:
            arm_results[result.arm_id][result.metric_name] = result
        
        existing_summary.arm_results = dict(arm_results)
        
        # Determine winning arm
        primary_metric_results = [
            r for r in results 
            if r.metric_name == config.primary_metric
        ]
        
        if primary_metric_results:
            # Find arm with best primary metric value
            if config.primary_metric in ["conversion_rate", "revenue", "success_rate"]:
                # Higher is better
                best_result = max(primary_metric_results, key=lambda r: r.value)
            else:
                # Lower is better (e.g., response_time)
                best_result = min(primary_metric_results, key=lambda r: r.value)
            
            if best_result.statistical_significance and best_result.practical_significance:
                existing_summary.winning_arm = best_result.arm_id
                existing_summary.confidence_in_winner = 1.0 - (best_result.p_value or 0.05)
                existing_summary.statistical_significance_achieved = True
                existing_summary.practical_significance_achieved = True
                existing_summary.recommendation = f"Deploy arm {best_result.arm_id} - statistically and practically significant improvement"
            else:
                existing_summary.recommendation = "No clear winner - consider running experiment longer or investigate results"
        
        return existing_summary
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> None:
        """Validate experiment configuration."""
        
        # Check arm percentages sum to 100%
        total_percentage = sum(arm.traffic_percentage for arm in config.arms)
        if abs(total_percentage - 100.0) > 0.01:
            raise ValueError(f"Arm traffic percentages must sum to 100%, got {total_percentage}")
        
        # Check for duplicate arm IDs
        arm_ids = [arm.arm_id for arm in config.arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("Duplicate arm IDs found")
        
        # Validate significance level
        if not 0 < config.significance_level < 1:
            raise ValueError("Significance level must be between 0 and 1")
        
        # Validate statistical power
        if not 0 < config.statistical_power < 1:
            raise ValueError("Statistical power must be between 0 and 1")
    
    def _passes_filters(self, request: InferenceRequest, config: ExperimentConfig) -> bool:
        """Check if request passes inclusion/exclusion filters."""
        
        # Check inclusion filters
        for filter_key, filter_value in config.inclusion_filters.items():
            request_value = request.metadata.get(filter_key)
            if request_value != filter_value:
                return False
        
        # Check exclusion filters
        for filter_key, filter_value in config.exclusion_filters.items():
            request_value = request.metadata.get(filter_key)
            if request_value == filter_value:
                return False
        
        return True
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._monitor_experiments()),
            asyncio.create_task(self._check_early_stopping()),
            asyncio.create_task(self._update_traffic_ramping())
        ]
        
        self.logger.info("A/B testing monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.logger.info("A/B testing monitoring stopped")
    
    async def _monitor_experiments(self) -> None:
        """Monitor running experiments."""
        while self.is_running:
            try:
                for experiment_id, status in self.experiment_status.items():
                    if status != ExperimentStatus.RUNNING:
                        continue
                    
                    config = self.experiments[experiment_id]
                    summary = self.experiment_summaries.get(experiment_id)
                    
                    if summary:
                        # Check duration limits
                        duration_hours = (datetime.utcnow() - summary.started_at).total_seconds() / 3600
                        
                        if duration_hours >= config.max_duration_hours:
                            await self.stop_experiment(experiment_id, "Maximum duration reached")
                        
                        # Check sample size
                        min_samples_reached = True
                        for arm in config.arms:
                            if arm.total_requests < config.min_sample_size_per_arm:
                                min_samples_reached = False
                                break
                        
                        if (min_samples_reached and 
                            duration_hours >= config.min_duration_hours):
                            # Check if we have significant results
                            results = await self._analyze_experiment(experiment_id)
                            primary_results = [r for r in results if r.metric_name == config.primary_metric]
                            
                            if any(r.statistical_significance for r in primary_results):
                                await self.stop_experiment(experiment_id, "Statistical significance achieved")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error("Error in experiment monitoring", error=str(e))
    
    async def _check_early_stopping(self) -> None:
        """Check for early stopping conditions."""
        while self.is_running:
            try:
                for experiment_id, status in self.experiment_status.items():
                    if status != ExperimentStatus.RUNNING:
                        continue
                    
                    config = self.experiments[experiment_id]
                    
                    if not config.enable_early_stopping:
                        continue
                    
                    # Analyze current results
                    results = await self._analyze_experiment(experiment_id)
                    primary_results = [r for r in results if r.metric_name == config.primary_metric]
                    
                    for result in primary_results:
                        if (result.p_value and 
                            result.p_value < config.early_stopping_threshold):
                            await self.stop_experiment(
                                experiment_id, 
                                f"Early stopping triggered (p-value: {result.p_value:.4f})"
                            )
                            break
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                self.logger.error("Error in early stopping check", error=str(e))
    
    async def _update_traffic_ramping(self) -> None:
        """Update traffic ramping for experiments."""
        while self.is_running:
            try:
                for experiment_id, status in self.experiment_status.items():
                    if status != ExperimentStatus.RUNNING:
                        continue
                    
                    config = self.experiments[experiment_id]
                    
                    if not config.enable_traffic_ramping:
                        continue
                    
                    summary = self.experiment_summaries.get(experiment_id)
                    if not summary:
                        continue
                    
                    # Calculate ramp progress
                    duration_hours = (datetime.utcnow() - summary.started_at).total_seconds() / 3600
                    ramp_progress = min(1.0, duration_hours / config.ramp_duration_hours)
                    
                    # Update traffic percentages
                    target_total = 100.0
                    current_total = config.initial_traffic_percentage + (
                        (target_total - config.initial_traffic_percentage) * ramp_progress
                    )
                    
                    # Scale arm percentages
                    scale_factor = current_total / 100.0
                    for arm in config.arms:
                        arm.traffic_percentage = (
                            arm.traffic_percentage / sum(a.traffic_percentage for a in config.arms)
                        ) * current_total
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                self.logger.error("Error in traffic ramping", error=str(e))
    
    async def get_experiment_dashboard(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment dashboard data."""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        summary = self.experiment_summaries.get(experiment_id)
        results = await self._analyze_experiment(experiment_id)
        
        # Calculate arm performance
        arm_performance = {}
        for arm in config.arms:
            arm_performance[arm.arm_id] = {
                "name": arm.name,
                "model_id": arm.model_id,
                "model_version": arm.model_version,
                "traffic_percentage": arm.traffic_percentage,
                "total_requests": arm.total_requests,
                "success_rate": arm.get_success_rate(),
                "avg_latency_ms": arm.get_avg_latency_ms(),
                "conversion_rate": arm.get_conversion_rate(),
                "revenue": arm.revenue
            }
        
        # Group results by metric
        metric_results = defaultdict(dict)
        for result in results:
            metric_results[result.metric_name][result.arm_id] = {
                "value": result.value,
                "sample_size": result.sample_size,
                "confidence_interval": [result.confidence_interval_lower, result.confidence_interval_upper],
                "p_value": result.p_value,
                "statistical_significance": result.statistical_significance,
                "practical_significance": result.practical_significance,
                "effect_size": result.effect_size
            }
        
        return {
            "experiment_id": experiment_id,
            "config": {
                "name": config.name,
                "description": config.description,
                "primary_metric": config.primary_metric,
                "secondary_metrics": config.secondary_metrics,
                "significance_level": config.significance_level,
                "minimum_detectable_effect": config.minimum_detectable_effect
            },
            "status": self.experiment_status[experiment_id].value,
            "summary": {
                "started_at": summary.started_at.isoformat() if summary else None,
                "ended_at": summary.ended_at.isoformat() if summary and summary.ended_at else None,
                "duration_hours": summary.duration_hours if summary else 0,
                "total_participants": summary.total_participants if summary else 0,
                "winning_arm": summary.winning_arm if summary else None,
                "confidence_in_winner": summary.confidence_in_winner if summary else 0,
                "recommendation": summary.recommendation if summary else ""
            },
            "arm_performance": arm_performance,
            "metric_results": dict(metric_results)
        }


class TrafficAllocator:
    """Handles traffic allocation for A/B tests."""
    
    def __init__(self):
        self.allocation_cache: Dict[str, str] = {}  # user_id -> arm_id mapping
    
    async def allocate(self, 
                      experiment_id: str,
                      config: ExperimentConfig,
                      request: InferenceRequest) -> Optional[str]:
        """Allocate request to experiment arm."""
        
        if config.allocation_method == AllocationMethod.RANDOM:
            return self._random_allocation(config)
        
        elif config.allocation_method == AllocationMethod.HASH_BASED:
            return self._hash_based_allocation(config, request)
        
        elif config.allocation_method == AllocationMethod.WEIGHTED:
            return self._weighted_allocation(config)
        
        else:
            return self._random_allocation(config)
    
    def _random_allocation(self, config: ExperimentConfig) -> str:
        """Random traffic allocation."""
        import random
        
        rand_val = random.random() * 100
        cumulative = 0
        
        for arm in config.arms:
            cumulative += arm.traffic_percentage
            if rand_val <= cumulative:
                return arm.arm_id
        
        return config.arms[0].arm_id  # Fallback
    
    def _hash_based_allocation(self, config: ExperimentConfig, request: InferenceRequest) -> str:
        """Hash-based consistent allocation."""
        
        allocation_key_value = request.metadata.get(config.allocation_key, request.request_id)
        
        # Create hash
        hash_value = int(hashlib.md5(str(allocation_key_value).encode()).hexdigest(), 16)
        bucket = (hash_value % 100) + 1  # 1-100
        
        # Allocate based on bucket
        cumulative = 0
        for arm in config.arms:
            cumulative += arm.traffic_percentage
            if bucket <= cumulative:
                return arm.arm_id
        
        return config.arms[0].arm_id  # Fallback
    
    def _weighted_allocation(self, config: ExperimentConfig) -> str:
        """Weighted random allocation."""
        import random
        
        weights = [arm.traffic_percentage for arm in config.arms]
        return random.choices(config.arms, weights=weights)[0].arm_id


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B tests."""
    
    async def compare_arms(self,
                          values_a: List[float],
                          values_b: List[float],
                          test_type: SignificanceTest) -> Tuple[float, float]:
        """Compare two experiment arms statistically."""
        
        if test_type == SignificanceTest.T_TEST:
            return self._t_test(values_a, values_b)
        
        elif test_type == SignificanceTest.MANN_WHITNEY:
            return self._mann_whitney_test(values_a, values_b)
        
        elif test_type == SignificanceTest.CHI_SQUARE:
            return self._chi_square_test(values_a, values_b)
        
        elif test_type == SignificanceTest.BOOTSTRAP:
            return self._bootstrap_test(values_a, values_b)
        
        else:
            return self._t_test(values_a, values_b)
    
    def _t_test(self, values_a: List[float], values_b: List[float]) -> Tuple[float, float]:
        """Perform independent t-test."""
        try:
            statistic, p_value = stats.ttest_ind(values_a, values_b)
            
            # Calculate effect size (Cohen's d)
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            pooled_std = ((statistics.variance(values_a) + statistics.variance(values_b)) / 2) ** 0.5
            
            if pooled_std > 0:
                effect_size = (mean_a - mean_b) / pooled_std
            else:
                effect_size = 0.0
            
            return p_value, effect_size
            
        except Exception:
            return 1.0, 0.0  # No significance
    
    def _mann_whitney_test(self, values_a: List[float], values_b: List[float]) -> Tuple[float, float]:
        """Perform Mann-Whitney U test."""
        try:
            statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
            
            # Calculate effect size (rank-biserial correlation)
            n_a, n_b = len(values_a), len(values_b)
            effect_size = 1 - (2 * statistic) / (n_a * n_b)
            
            return p_value, effect_size
            
        except Exception:
            return 1.0, 0.0
    
    def _chi_square_test(self, values_a: List[float], values_b: List[float]) -> Tuple[float, float]:
        """Perform chi-square test (for binary outcomes)."""
        try:
            # Convert to binary outcomes
            successes_a = sum(1 for v in values_a if v > 0)
            failures_a = len(values_a) - successes_a
            successes_b = sum(1 for v in values_b if v > 0)
            failures_b = len(values_b) - failures_b
            
            # Create contingency table
            observed = np.array([[successes_a, failures_a], [successes_b, failures_b]])
            
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            
            # Calculate Cramér's V
            n = observed.sum()
            effect_size = np.sqrt(chi2 / (n * (min(observed.shape) - 1)))
            
            return p_value, effect_size
            
        except Exception:
            return 1.0, 0.0
    
    def _bootstrap_test(self, values_a: List[float], values_b: List[float]) -> Tuple[float, float]:
        """Perform bootstrap test."""
        try:
            import random
            
            observed_diff = statistics.mean(values_a) - statistics.mean(values_b)
            
            # Combine samples for null hypothesis
            combined = values_a + values_b
            n_a, n_b = len(values_a), len(values_b)
            
            # Bootstrap resampling
            n_bootstrap = 1000
            bootstrap_diffs = []
            
            for _ in range(n_bootstrap):
                # Resample under null hypothesis
                resampled = random.choices(combined, k=n_a + n_b)
                resample_a = resampled[:n_a]
                resample_b = resampled[n_a:]
                
                diff = statistics.mean(resample_a) - statistics.mean(resample_b)
                bootstrap_diffs.append(diff)
            
            # Calculate p-value
            extreme_diffs = sum(1 for d in bootstrap_diffs if abs(d) >= abs(observed_diff))
            p_value = extreme_diffs / n_bootstrap
            
            # Effect size is the standardized difference
            pooled_std = statistics.stdev(combined)
            effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0
            
            return p_value, effect_size
            
        except Exception:
            return 1.0, 0.0
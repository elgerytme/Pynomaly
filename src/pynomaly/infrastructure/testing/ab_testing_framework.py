"""
A/B Testing Framework for algorithm comparison and performance evaluation.

This module provides comprehensive A/B testing capabilities for comparing
different anomaly detection algorithms, configurations, and model versions.
"""

import asyncio
import json
import logging
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from scipy import stats

from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.shared.exceptions import TestingError, ValidationError
from pynomaly.shared.types import TenantId, UserId

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Status of A/B test."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SplitStrategy(str, Enum):
    """Strategy for splitting traffic between variants."""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    USER_ID_HASH = "user_id_hash"
    TEMPORAL = "temporal"  # Time-based splitting


class MetricType(str, Enum):
    """Types of metrics to collect."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    TRUE_NEGATIVE_RATE = "true_negative_rate"
    PROCESSING_TIME = "processing_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CUSTOM = "custom"


@dataclass
class TestVariant:
    """A variant in an A/B test."""
    id: str
    name: str
    description: str
    detector: Detector
    traffic_percentage: float  # 0.0 to 1.0
    configuration: Dict[str, Any] = field(default_factory=dict)
    is_control: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "detector_id": self.detector.id,
            "detector_name": self.detector.name,
            "traffic_percentage": self.traffic_percentage,
            "configuration": self.configuration,
            "is_control": self.is_control
        }


@dataclass
class TestMetric:
    """A metric collected during testing."""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    variant_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "variant_id": self.variant_id,
            "metadata": self.metadata
        }


@dataclass
class TestResult:
    """Result of a completed test execution."""
    test_id: str
    variant_id: str
    dataset_id: str
    detection_result: DetectionResult
    metrics: List[TestMetric]
    execution_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalResult:
    """Statistical analysis result."""
    metric_name: str
    control_variant: str
    treatment_variant: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    is_significant: bool
    statistical_power: float
    sample_size_control: int
    sample_size_treatment: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "control_variant": self.control_variant,
            "treatment_variant": self.treatment_variant,
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "control_std": self.control_std,
            "treatment_std": self.treatment_std,
            "p_value": self.p_value,
            "confidence_interval": list(self.confidence_interval),
            "effect_size": self.effect_size,
            "is_significant": self.is_significant,
            "statistical_power": self.statistical_power,
            "sample_size_control": self.sample_size_control,
            "sample_size_treatment": self.sample_size_treatment
        }


class TrafficSplitter:
    """Handles traffic splitting between test variants."""
    
    def __init__(self, strategy: SplitStrategy = SplitStrategy.RANDOM):
        self.strategy = strategy
        self._round_robin_counter = 0
    
    def get_variant(
        self, 
        variants: List[TestVariant], 
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> TestVariant:
        """Get variant for a given request."""
        if not variants:
            raise TestingError("No variants available")
        
        if len(variants) == 1:
            return variants[0]
        
        if self.strategy == SplitStrategy.RANDOM:
            return self._random_split(variants)
        elif self.strategy == SplitStrategy.ROUND_ROBIN:
            return self._round_robin_split(variants)
        elif self.strategy == SplitStrategy.WEIGHTED:
            return self._weighted_split(variants)
        elif self.strategy == SplitStrategy.USER_ID_HASH:
            return self._user_hash_split(variants, user_id)
        elif self.strategy == SplitStrategy.TEMPORAL:
            return self._temporal_split(variants, timestamp)
        else:
            return self._random_split(variants)
    
    def _random_split(self, variants: List[TestVariant]) -> TestVariant:
        """Random traffic splitting."""
        cumulative_weights = []
        total = 0
        
        for variant in variants:
            total += variant.traffic_percentage
            cumulative_weights.append(total)
        
        # Normalize if weights don't sum to 1.0
        if total > 0:
            cumulative_weights = [w / total for w in cumulative_weights]
        
        random_value = np.random.random()
        
        for i, weight in enumerate(cumulative_weights):
            if random_value <= weight:
                return variants[i]
        
        return variants[-1]  # Fallback
    
    def _round_robin_split(self, variants: List[TestVariant]) -> TestVariant:
        """Round-robin traffic splitting."""
        variant = variants[self._round_robin_counter % len(variants)]
        self._round_robin_counter += 1
        return variant
    
    def _weighted_split(self, variants: List[TestVariant]) -> TestVariant:
        """Weighted traffic splitting."""
        return self._random_split(variants)  # Same as random split
    
    def _user_hash_split(self, variants: List[TestVariant], user_id: Optional[str]) -> TestVariant:
        """Consistent user-based splitting."""
        if not user_id:
            return self._random_split(variants)
        
        # Use hash of user_id for consistent assignment
        hash_value = hash(user_id) % 1000000
        normalized_hash = hash_value / 1000000
        
        cumulative_weights = []
        total = 0
        
        for variant in variants:
            total += variant.traffic_percentage
            cumulative_weights.append(total)
        
        if total > 0:
            cumulative_weights = [w / total for w in cumulative_weights]
        
        for i, weight in enumerate(cumulative_weights):
            if normalized_hash <= weight:
                return variants[i]
        
        return variants[-1]
    
    def _temporal_split(self, variants: List[TestVariant], timestamp: Optional[datetime]) -> TestVariant:
        """Time-based traffic splitting."""
        if not timestamp:
            timestamp = datetime.utcnow()
        
        # Use hour of day to determine variant
        hour = timestamp.hour
        variant_index = hour % len(variants)
        return variants[variant_index]


class MetricsCalculator:
    """Calculates various performance metrics."""
    
    @staticmethod
    def calculate_accuracy(detection_result: DetectionResult, ground_truth: List[bool]) -> float:
        """Calculate accuracy metric."""
        if not ground_truth or len(detection_result.anomalies) == 0:
            return 0.0
        
        # Convert detection result to binary predictions
        predictions = [False] * len(ground_truth)
        for anomaly in detection_result.anomalies:
            if anomaly.index < len(predictions):
                predictions[anomaly.index] = True
        
        correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
        return correct / len(ground_truth)
    
    @staticmethod
    def calculate_precision(detection_result: DetectionResult, ground_truth: List[bool]) -> float:
        """Calculate precision metric."""
        if not detection_result.anomalies:
            return 0.0
        
        true_positives = 0
        for anomaly in detection_result.anomalies:
            if anomaly.index < len(ground_truth) and ground_truth[anomaly.index]:
                true_positives += 1
        
        return true_positives / len(detection_result.anomalies)
    
    @staticmethod
    def calculate_recall(detection_result: DetectionResult, ground_truth: List[bool]) -> float:
        """Calculate recall metric."""
        true_anomalies = sum(ground_truth)
        if true_anomalies == 0:
            return 0.0
        
        true_positives = 0
        for anomaly in detection_result.anomalies:
            if anomaly.index < len(ground_truth) and ground_truth[anomaly.index]:
                true_positives += 1
        
        return true_positives / true_anomalies
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_false_positive_rate(detection_result: DetectionResult, ground_truth: List[bool]) -> float:
        """Calculate false positive rate."""
        true_negatives = sum(1 for gt in ground_truth if not gt)
        if true_negatives == 0:
            return 0.0
        
        false_positives = 0
        for anomaly in detection_result.anomalies:
            if anomaly.index < len(ground_truth) and not ground_truth[anomaly.index]:
                false_positives += 1
        
        return false_positives / true_negatives


class ABTest:
    """A/B test configuration and execution."""
    
    def __init__(
        self,
        test_id: str,
        name: str,
        description: str,
        variants: List[TestVariant],
        split_strategy: SplitStrategy = SplitStrategy.RANDOM,
        metrics_to_collect: List[MetricType] = None,
        minimum_sample_size: int = 100,
        confidence_level: float = 0.95,
        significance_threshold: float = 0.05,
        duration_days: Optional[int] = None,
        created_by: Optional[UserId] = None
    ):
        self.test_id = test_id
        self.name = name
        self.description = description
        self.variants = variants
        self.split_strategy = split_strategy
        self.metrics_to_collect = metrics_to_collect or [
            MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, 
            MetricType.F1_SCORE, MetricType.PROCESSING_TIME
        ]
        self.minimum_sample_size = minimum_sample_size
        self.confidence_level = confidence_level
        self.significance_threshold = significance_threshold
        self.duration_days = duration_days
        self.created_by = created_by
        
        # Test state
        self.status = TestStatus.DRAFT
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        
        # Results storage
        self.test_results: List[TestResult] = []
        self.collected_metrics: List[TestMetric] = []
        
        # Traffic splitter
        self.traffic_splitter = TrafficSplitter(split_strategy)
        
        # Validate configuration
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate test configuration."""
        if len(self.variants) < 2:
            raise ValidationError("A/B test requires at least 2 variants")
        
        # Check traffic percentages
        total_traffic = sum(v.traffic_percentage for v in self.variants)
        if abs(total_traffic - 1.0) > 0.01:  # Allow small floating point differences
            raise ValidationError(f"Variant traffic percentages must sum to 1.0, got {total_traffic}")
        
        # Check for control variant
        control_variants = [v for v in self.variants if v.is_control]
        if len(control_variants) > 1:
            raise ValidationError("Only one control variant is allowed")
        
        # Validate metric types
        for metric_type in self.metrics_to_collect:
            if not isinstance(metric_type, MetricType):
                raise ValidationError(f"Invalid metric type: {metric_type}")
    
    def start(self) -> None:
        """Start the A/B test."""
        if self.status != TestStatus.DRAFT:
            raise TestingError(f"Cannot start test in status: {self.status}")
        
        self.status = TestStatus.RUNNING
        self.started_at = datetime.utcnow()
        logger.info(f"Started A/B test: {self.test_id}")
    
    def pause(self) -> None:
        """Pause the A/B test."""
        if self.status != TestStatus.RUNNING:
            raise TestingError(f"Cannot pause test in status: {self.status}")
        
        self.status = TestStatus.PAUSED
        logger.info(f"Paused A/B test: {self.test_id}")
    
    def resume(self) -> None:
        """Resume the A/B test."""
        if self.status != TestStatus.PAUSED:
            raise TestingError(f"Cannot resume test in status: {self.status}")
        
        self.status = TestStatus.RUNNING
        logger.info(f"Resumed A/B test: {self.test_id}")
    
    def stop(self) -> None:
        """Stop the A/B test."""
        if self.status not in [TestStatus.RUNNING, TestStatus.PAUSED]:
            raise TestingError(f"Cannot stop test in status: {self.status}")
        
        self.status = TestStatus.COMPLETED
        self.ended_at = datetime.utcnow()
        logger.info(f"Stopped A/B test: {self.test_id}")
    
    def cancel(self) -> None:
        """Cancel the A/B test."""
        if self.status in [TestStatus.COMPLETED, TestStatus.CANCELLED]:
            raise TestingError(f"Cannot cancel test in status: {self.status}")
        
        self.status = TestStatus.CANCELLED
        self.ended_at = datetime.utcnow()
        logger.info(f"Cancelled A/B test: {self.test_id}")
    
    async def execute_variant(
        self, 
        dataset: Dataset,
        user_id: Optional[str] = None,
        ground_truth: Optional[List[bool]] = None
    ) -> TestResult:
        """Execute detection with selected variant."""
        if self.status != TestStatus.RUNNING:
            raise TestingError(f"Test is not running: {self.status}")
        
        # Select variant
        variant = self.traffic_splitter.get_variant(
            self.variants, 
            user_id=user_id,
            timestamp=datetime.utcnow()
        )
        
        # Execute detection
        start_time = datetime.utcnow()
        try:
            detection_result = await self._execute_detection(variant.detector, dataset)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                detection_result, 
                variant.id, 
                execution_time,
                ground_truth
            )
            
            # Create test result
            test_result = TestResult(
                test_id=self.test_id,
                variant_id=variant.id,
                dataset_id=dataset.id,
                detection_result=detection_result,
                metrics=metrics,
                execution_time=execution_time,
                timestamp=datetime.utcnow()
            )
            
            # Store results
            self.test_results.append(test_result)
            self.collected_metrics.extend(metrics)
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error executing variant {variant.id}: {e}")
            raise TestingError(f"Variant execution failed: {e}")
    
    async def _execute_detection(self, detector: Detector, dataset: Dataset) -> DetectionResult:
        """Execute anomaly detection with given detector."""
        # This would call the actual detector's predict method
        # For simulation purposes, we'll create a mock result
        
        # In a real implementation:
        # return await detector.detect(dataset)
        
        # Simulated detection for demonstration
        n_samples = len(dataset.data) if not dataset.data.empty else 0
        n_anomalies = max(1, int(n_samples * 0.05))  # 5% anomaly rate
        
        anomalies = []
        if n_samples > 0:
            anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
            
            for idx in anomaly_indices:
                from pynomaly.domain.entities.anomaly import Anomaly, Score
                
                anomaly = Anomaly(
                    index=int(idx),
                    score=Score(float(np.random.random())),
                    features=dataset.data.iloc[idx].to_dict() if idx < len(dataset.data) else {},
                    metadata={
                        "detector": detector.name,
                        "algorithm": detector.algorithm,
                        "test_id": self.test_id
                    }
                )
                anomalies.append(anomaly)
        
        return DetectionResult(
            anomalies=anomalies,
            algorithm=detector.algorithm,
            threshold=0.5,
            metadata={
                "detector_name": detector.name,
                "sample_count": n_samples,
                "test_execution": True
            }
        )
    
    def _calculate_metrics(
        self,
        detection_result: DetectionResult,
        variant_id: str,
        execution_time: float,
        ground_truth: Optional[List[bool]] = None
    ) -> List[TestMetric]:
        """Calculate metrics for a detection result."""
        metrics = []
        timestamp = datetime.utcnow()
        
        # Performance metrics
        if MetricType.PROCESSING_TIME in self.metrics_to_collect:
            metrics.append(TestMetric(
                name="processing_time",
                type=MetricType.PROCESSING_TIME,
                value=execution_time,
                timestamp=timestamp,
                variant_id=variant_id
            ))
        
        # Quality metrics (require ground truth)
        if ground_truth:
            calculator = MetricsCalculator()
            
            if MetricType.ACCURACY in self.metrics_to_collect:
                accuracy = calculator.calculate_accuracy(detection_result, ground_truth)
                metrics.append(TestMetric(
                    name="accuracy",
                    type=MetricType.ACCURACY,
                    value=accuracy,
                    timestamp=timestamp,
                    variant_id=variant_id
                ))
            
            if MetricType.PRECISION in self.metrics_to_collect:
                precision = calculator.calculate_precision(detection_result, ground_truth)
                metrics.append(TestMetric(
                    name="precision",
                    type=MetricType.PRECISION,
                    value=precision,
                    timestamp=timestamp,
                    variant_id=variant_id
                ))
            
            if MetricType.RECALL in self.metrics_to_collect:
                recall = calculator.calculate_recall(detection_result, ground_truth)
                metrics.append(TestMetric(
                    name="recall",
                    type=MetricType.RECALL,
                    value=recall,
                    timestamp=timestamp,
                    variant_id=variant_id
                ))
            
            if MetricType.F1_SCORE in self.metrics_to_collect:
                # Get precision and recall values
                precision = calculator.calculate_precision(detection_result, ground_truth)
                recall = calculator.calculate_recall(detection_result, ground_truth)
                f1 = calculator.calculate_f1_score(precision, recall)
                
                metrics.append(TestMetric(
                    name="f1_score",
                    type=MetricType.F1_SCORE,
                    value=f1,
                    timestamp=timestamp,
                    variant_id=variant_id
                ))
            
            if MetricType.FALSE_POSITIVE_RATE in self.metrics_to_collect:
                fpr = calculator.calculate_false_positive_rate(detection_result, ground_truth)
                metrics.append(TestMetric(
                    name="false_positive_rate",
                    type=MetricType.FALSE_POSITIVE_RATE,
                    value=fpr,
                    timestamp=timestamp,
                    variant_id=variant_id
                ))
        
        return metrics
    
    def get_statistical_analysis(self) -> List[StatisticalResult]:
        """Perform statistical analysis of test results."""
        if len(self.variants) != 2:
            raise TestingError("Statistical analysis currently supports only 2-variant tests")
        
        # Find control and treatment variants
        control_variant = next((v for v in self.variants if v.is_control), self.variants[0])
        treatment_variant = next((v for v in self.variants if v.id != control_variant.id), None)
        
        if not treatment_variant:
            raise TestingError("Could not identify treatment variant")
        
        results = []
        
        # Analyze each metric
        for metric_type in self.metrics_to_collect:
            try:
                result = self._analyze_metric(
                    metric_type.value,
                    control_variant.id,
                    treatment_variant.id
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze metric {metric_type}: {e}")
        
        return results
    
    def _analyze_metric(
        self,
        metric_name: str,
        control_variant_id: str,
        treatment_variant_id: str
    ) -> Optional[StatisticalResult]:
        """Analyze a specific metric statistically."""
        # Get metric values for both variants
        control_values = [
            m.value for m in self.collected_metrics
            if m.name == metric_name and m.variant_id == control_variant_id
        ]
        
        treatment_values = [
            m.value for m in self.collected_metrics
            if m.name == metric_name and m.variant_id == treatment_variant_id
        ]
        
        if len(control_values) < 2 or len(treatment_values) < 2:
            logger.warning(f"Insufficient data for metric {metric_name}")
            return None
        
        # Calculate basic statistics
        control_mean = statistics.mean(control_values)
        treatment_mean = statistics.mean(treatment_values)
        control_std = statistics.stdev(control_values) if len(control_values) > 1 else 0
        treatment_std = statistics.stdev(treatment_values) if len(treatment_values) > 1 else 0
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            
            # Calculate confidence interval for the difference
            pooled_std = np.sqrt(
                ((len(control_values) - 1) * control_std**2 + 
                 (len(treatment_values) - 1) * treatment_std**2) /
                (len(control_values) + len(treatment_values) - 2)
            )
            
            se_diff = pooled_std * np.sqrt(1/len(control_values) + 1/len(treatment_values))
            df = len(control_values) + len(treatment_values) - 2
            t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)
            
            mean_diff = treatment_mean - control_mean
            margin_error = t_critical * se_diff
            
            confidence_interval = (mean_diff - margin_error, mean_diff + margin_error)
            
            # Calculate effect size (Cohen's d)
            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # Determine statistical significance
            is_significant = p_value < self.significance_threshold
            
            # Estimate statistical power (simplified)
            statistical_power = self._estimate_power(
                control_values, treatment_values, self.significance_threshold
            )
            
            return StatisticalResult(
                metric_name=metric_name,
                control_variant=control_variant_id,
                treatment_variant=treatment_variant_id,
                control_mean=control_mean,
                treatment_mean=treatment_mean,
                control_std=control_std,
                treatment_std=treatment_std,
                p_value=p_value,
                confidence_interval=confidence_interval,
                effect_size=effect_size,
                is_significant=is_significant,
                statistical_power=statistical_power,
                sample_size_control=len(control_values),
                sample_size_treatment=len(treatment_values)
            )
            
        except Exception as e:
            logger.error(f"Statistical analysis failed for {metric_name}: {e}")
            return None
    
    def _estimate_power(
        self,
        control_values: List[float],
        treatment_values: List[float],
        alpha: float
    ) -> float:
        """Estimate statistical power (simplified calculation)."""
        try:
            # This is a simplified power calculation
            # In practice, you'd use more sophisticated methods
            
            n1, n2 = len(control_values), len(treatment_values)
            pooled_std = np.sqrt(
                (np.var(control_values) * (n1 - 1) + np.var(treatment_values) * (n2 - 1)) /
                (n1 + n2 - 2)
            )
            
            if pooled_std == 0:
                return 1.0
            
            effect_size = abs(statistics.mean(treatment_values) - statistics.mean(control_values)) / pooled_std
            
            # Rough power estimation based on effect size and sample size
            if effect_size < 0.2:
                return 0.1  # Very low power for small effects
            elif effect_size < 0.5:
                return min(0.8, 0.3 + (n1 + n2) / 200)  # Medium power
            else:
                return min(0.95, 0.6 + (n1 + n2) / 100)  # High power for large effects
            
        except Exception:
            return 0.5  # Default estimate
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        summary = {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_days": self.duration_days,
            "variants": [v.to_dict() for v in self.variants],
            "split_strategy": self.split_strategy.value,
            "metrics_collected": [m.value for m in self.metrics_to_collect],
            "minimum_sample_size": self.minimum_sample_size,
            "confidence_level": self.confidence_level,
            "significance_threshold": self.significance_threshold,
            "total_executions": len(self.test_results),
            "total_metrics": len(self.collected_metrics)
        }
        
        # Add variant-specific statistics
        variant_stats = {}
        for variant in self.variants:
            variant_results = [r for r in self.test_results if r.variant_id == variant.id]
            variant_stats[variant.id] = {
                "executions": len(variant_results),
                "traffic_percentage": variant.traffic_percentage,
                "is_control": variant.is_control
            }
        
        summary["variant_statistics"] = variant_stats
        
        return summary


class ABTestingService:
    """Service for managing A/B tests."""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTest] = {}
        self.test_history: Dict[str, ABTest] = {}
    
    def create_test(
        self,
        name: str,
        description: str,
        variants: List[TestVariant],
        **kwargs
    ) -> ABTest:
        """Create a new A/B test."""
        test_id = f"ab_test_{uuid.uuid4().hex[:8]}"
        
        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            variants=variants,
            **kwargs
        )
        
        self.active_tests[test_id] = test
        logger.info(f"Created A/B test: {test_id}")
        
        return test
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get an A/B test by ID."""
        return self.active_tests.get(test_id) or self.test_history.get(test_id)
    
    def list_tests(self, status_filter: Optional[TestStatus] = None) -> List[ABTest]:
        """List A/B tests with optional status filter."""
        all_tests = list(self.active_tests.values()) + list(self.test_history.values())
        
        if status_filter:
            return [t for t in all_tests if t.status == status_filter]
        
        return all_tests
    
    def start_test(self, test_id: str) -> None:
        """Start an A/B test."""
        test = self.get_test(test_id)
        if not test:
            raise TestingError(f"Test not found: {test_id}")
        
        test.start()
    
    def stop_test(self, test_id: str) -> None:
        """Stop an A/B test."""
        test = self.get_test(test_id)
        if not test:
            raise TestingError(f"Test not found: {test_id}")
        
        test.stop()
        
        # Move to history
        if test_id in self.active_tests:
            self.test_history[test_id] = self.active_tests.pop(test_id)
    
    def delete_test(self, test_id: str) -> None:
        """Delete an A/B test."""
        if test_id in self.active_tests:
            del self.active_tests[test_id]
        elif test_id in self.test_history:
            del self.test_history[test_id]
        else:
            raise TestingError(f"Test not found: {test_id}")
        
        logger.info(f"Deleted A/B test: {test_id}")
    
    async def execute_test(
        self,
        test_id: str,
        dataset: Dataset,
        user_id: Optional[str] = None,
        ground_truth: Optional[List[bool]] = None
    ) -> TestResult:
        """Execute an A/B test with given dataset."""
        test = self.get_test(test_id)
        if not test:
            raise TestingError(f"Test not found: {test_id}")
        
        return await test.execute_variant(dataset, user_id, ground_truth)
    
    def get_test_results(self, test_id: str) -> List[TestResult]:
        """Get results for a specific test."""
        test = self.get_test(test_id)
        if not test:
            raise TestingError(f"Test not found: {test_id}")
        
        return test.test_results
    
    def get_statistical_analysis(self, test_id: str) -> List[StatisticalResult]:
        """Get statistical analysis for a test."""
        test = self.get_test(test_id)
        if not test:
            raise TestingError(f"Test not found: {test_id}")
        
        return test.get_statistical_analysis()
    
    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """Get summary for a test."""
        test = self.get_test(test_id)
        if not test:
            raise TestingError(f"Test not found: {test_id}")
        
        return test.get_summary()
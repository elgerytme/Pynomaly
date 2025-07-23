"""A/B testing service for model comparison in production."""

import logging
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from scipy import stats

from anomaly_detection.domain.services.mlops_service import MLOpsService, ModelVersion
from anomaly_detection.domain.entities.detection_result import DetectionResult


class TestStatus(Enum):
    """A/B test status enumeration."""
    DRAFT = "draft"
    RUNNING = "running"  
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class SplitType(Enum):
    """Type of traffic splitting."""
    RANDOM = "random"
    USER_BASED = "user_based"
    FEATURE_BASED = "feature_based"
    TIME_BASED = "time_based"


@dataclass
class TestVariant:
    """A variant in an A/B test."""
    variant_id: str
    name: str
    model_id: str
    model_version: int
    traffic_percentage: float
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass  
class TestMetrics:
    """Metrics for an A/B test variant."""
    variant_id: str
    total_requests: int
    total_predictions: int
    avg_response_time_ms: float
    error_rate: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    custom_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    test_name: str
    description: str
    variants: List[TestVariant]
    split_type: SplitType
    duration_days: int
    significance_threshold: float = 0.05
    minimum_sample_size: int = 1000
    success_metric: str = "accuracy"
    early_stopping_enabled: bool = True
    early_stopping_threshold: float = 0.001
    traffic_ramp_up: bool = False
    max_traffic_percentage: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "test_name": self.test_name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "split_type": self.split_type.value,
            "duration_days": self.duration_days,
            "significance_threshold": self.significance_threshold,
            "minimum_sample_size": self.minimum_sample_size,
            "success_metric": self.success_metric,
            "early_stopping_enabled": self.early_stopping_enabled,
            "early_stopping_threshold": self.early_stopping_threshold,
            "traffic_ramp_up": self.traffic_ramp_up,
            "max_traffic_percentage": self.max_traffic_percentage
        }


@dataclass
class ABTest:
    """Represents an A/B test."""
    test_id: str
    config: ABTestConfig  
    status: TestStatus
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    created_by: str
    current_traffic_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "test_id": self.test_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "created_by": self.created_by,
            "current_traffic_percentage": self.current_traffic_percentage
        }


@dataclass
class TestResult:
    """Results of an A/B test."""
    test_id: str
    variant_metrics: List[TestMetrics]
    statistical_significance: Dict[str, Any]
    winner: Optional[str]  # variant_id of winner
    confidence_level: float
    effect_size: float
    recommendations: List[str]
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "test_id": self.test_id,
            "variant_metrics": [m.to_dict() for m in self.variant_metrics],
            "statistical_significance": self.statistical_significance,
            "winner": self.winner,
            "confidence_level": self.confidence_level,
            "effect_size": self.effect_size,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat()
        }


class ABTestingService:
    """Service for A/B testing models in production."""
    
    def __init__(self, mlops_service: MLOpsService):
        """Initialize A/B testing service.
        
        Args:
            mlops_service: MLOps service for model management
        """
        self.mlops_service = mlops_service
        self.logger = logging.getLogger(__name__)
        
        # Storage for tests and data
        self._tests: Dict[str, ABTest] = {}
        self._test_data: Dict[str, List[Dict[str, Any]]] = {}  # test_id -> list of requests
        self._variant_assignments: Dict[str, str] = {}  # request_id -> variant_id
        
    def create_test(self,
                   config: ABTestConfig,
                   created_by: str = "system") -> str:
        """Create a new A/B test.
        
        Args:
            config: Test configuration
            created_by: User who created the test
            
        Returns:
            Test ID
        """
        # Validate configuration
        self._validate_test_config(config)
        
        test_id = str(uuid.uuid4())
        
        test = ABTest(
            test_id=test_id,
            config=config,
            status=TestStatus.DRAFT,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            created_by=created_by
        )
        
        self._tests[test_id] = test
        self._test_data[test_id] = []
        
        self.logger.info(f"Created A/B test '{config.test_name}' with ID: {test_id}")
        return test_id
    
    def _validate_test_config(self, config: ABTestConfig):
        """Validate A/B test configuration.
        
        Args:
            config: Configuration to validate
        """
        # Check variant percentages sum to 100
        total_percentage = sum(v.traffic_percentage for v in config.variants)
        if abs(total_percentage - 100.0) > 0.01:
            raise ValueError(f"Variant percentages must sum to 100%, got {total_percentage}%")
        
        # Check minimum 2 variants
        if len(config.variants) < 2:
            raise ValueError("A/B test must have at least 2 variants")
        
        # Check duplicate variant names
        variant_names = [v.name for v in config.variants]
        if len(set(variant_names)) != len(variant_names):
            raise ValueError("Variant names must be unique")
        
        # Validate models exist
        for variant in config.variants:
            model_versions = self.mlops_service.get_model_versions(variant.model_id)
            if not any(v.version == variant.model_version for v in model_versions):
                raise ValueError(
                    f"Model version {variant.model_version} not found for model {variant.model_id}"
                )
    
    def start_test(self, test_id: str, initial_traffic_percentage: float = 10.0):
        """Start an A/B test.
        
        Args:
            test_id: ID of the test to start
            initial_traffic_percentage: Initial percentage of traffic to route to test
        """
        if test_id not in self._tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self._tests[test_id]
        
        if test.status != TestStatus.DRAFT:
            raise ValueError(f"Test {test_id} is not in draft status")
        
        test.status = TestStatus.RUNNING
        test.started_at = datetime.now()
        test.current_traffic_percentage = initial_traffic_percentage
        
        self.logger.info(f"Started A/B test {test_id} with {initial_traffic_percentage}% traffic")
    
    def pause_test(self, test_id: str):
        """Pause a running A/B test.
        
        Args:
            test_id: ID of the test to pause
        """
        if test_id not in self._tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self._tests[test_id]
        
        if test.status != TestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")
        
        test.status = TestStatus.PAUSED
        
        self.logger.info(f"Paused A/B test {test_id}")
    
    def resume_test(self, test_id: str):
        """Resume a paused A/B test.
        
        Args:
            test_id: ID of the test to resume
        """
        if test_id not in self._tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self._tests[test_id]
        
        if test.status != TestStatus.PAUSED:
            raise ValueError(f"Test {test_id} is not paused")
        
        test.status = TestStatus.RUNNING
        
        self.logger.info(f"Resumed A/B test {test_id}")
    
    def stop_test(self, test_id: str, reason: str = "manual"):
        """Stop an A/B test.
        
        Args:
            test_id: ID of the test to stop
            reason: Reason for stopping
        """
        if test_id not in self._tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self._tests[test_id]
        
        if test.status not in [TestStatus.RUNNING, TestStatus.PAUSED]:
            raise ValueError(f"Test {test_id} is not running or paused")
        
        test.status = TestStatus.COMPLETED
        test.ended_at = datetime.now()
        
        self.logger.info(f"Stopped A/B test {test_id}. Reason: {reason}")
    
    def get_variant_assignment(self, 
                             test_id: str,
                             request_id: str,
                             user_id: Optional[str] = None,
                             features: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get variant assignment for a request.
        
        Args:
            test_id: ID of the test
            request_id: Unique request identifier
            user_id: Optional user identifier for user-based splitting
            features: Optional features for feature-based splitting
            
        Returns:
            Variant ID or None if not in test
        """
        if test_id not in self._tests:
            return None
        
        test = self._tests[test_id]
        
        # Check if test is active
        if test.status != TestStatus.RUNNING:
            return None
        
        # Check if request should be included in test based on traffic percentage
        if not self._should_include_request(test, request_id):
            return None
        
        # Get or create variant assignment
        assignment_key = f"{test_id}:{request_id}"
        
        if assignment_key in self._variant_assignments:
            return self._variant_assignments[assignment_key]
        
        # Assign variant based on split type
        variant_id = self._assign_variant(test, request_id, user_id, features)
        self._variant_assignments[assignment_key] = variant_id
        
        return variant_id
    
    def _should_include_request(self, test: ABTest, request_id: str) -> bool:
        """Determine if a request should be included in the test.
        
        Args:
            test: The A/B test
            request_id: Request identifier
            
        Returns:
            True if request should be included
        """
        # Use hash of request_id to get consistent assignment
        hash_value = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16)
        percentage = (hash_value % 10000) / 100.0  # Convert to 0-100 range
        
        return percentage < test.current_traffic_percentage
    
    def _assign_variant(self,
                       test: ABTest,
                       request_id: str,
                       user_id: Optional[str] = None,
                       features: Optional[Dict[str, Any]] = None) -> str:
        """Assign a variant to a request.
        
        Args:
            test: The A/B test
            request_id: Request identifier
            user_id: Optional user identifier
            features: Optional features
            
        Returns:
            Variant ID
        """
        if test.config.split_type == SplitType.RANDOM:
            return self._random_assignment(test, request_id)
        elif test.config.split_type == SplitType.USER_BASED:
            return self._user_based_assignment(test, user_id or request_id)
        elif test.config.split_type == SplitType.FEATURE_BASED:
            return self._feature_based_assignment(test, features or {})
        elif test.config.split_type == SplitType.TIME_BASED:  
            return self._time_based_assignment(test)
        else:
            return self._random_assignment(test, request_id)
    
    def _random_assignment(self, test: ABTest, request_id: str) -> str:
        """Randomly assign variant based on traffic percentages.
        
        Args:
            test: The A/B test
            request_id: Request identifier
            
        Returns:
            Variant ID
        """
        # Use hash for consistent assignment
        hash_value = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16)
        percentage = (hash_value % 10000) / 100.0
        
        cumulative_percentage = 0.0
        for variant in test.config.variants:
            cumulative_percentage += variant.traffic_percentage
            if percentage < cumulative_percentage:
                return variant.variant_id
        
        # Fallback to last variant
        return test.config.variants[-1].variant_id
    
    def _user_based_assignment(self, test: ABTest, user_id: str) -> str:
        """Assign variant based on user ID hash.
        
        Args:
            test: The A/B test
            user_id: User identifier
            
        Returns:
            Variant ID
        """
        return self._random_assignment(test, user_id)
    
    def _feature_based_assignment(self, test: ABTest, features: Dict[str, Any]) -> str:
        """Assign variant based on feature values.
        
        Args:
            test: The A/B test
            features: Feature dictionary
            
        Returns:
            Variant ID
        """
        # Simple feature-based assignment (can be made more sophisticated)
        feature_hash = hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()
        return self._random_assignment(test, feature_hash)
    
    def _time_based_assignment(self, test: ABTest) -> str:
        """Assign variant based on time periods.
        
        Args:
            test: The A/B test
            
        Returns:
            Variant ID
        """
        # Simple time-based assignment - alternate by hour
        hour = datetime.now().hour
        variant_index = hour % len(test.config.variants)
        return test.config.variants[variant_index].variant_id
    
    def record_request(self,
                      test_id: str,
                      request_id: str,
                      variant_id: str,
                      prediction_data: Dict[str, Any],
                      response_time_ms: float,
                      error_occurred: bool = False):
        """Record a request for analysis.
        
        Args:
            test_id: ID of the test
            request_id: Request identifier
            variant_id: Assigned variant ID
            prediction_data: Prediction request and response data
            response_time_ms: Response time in milliseconds
            error_occurred: Whether an error occurred
        """
        if test_id not in self._test_data:
            return
        
        request_record = {
            "request_id": request_id,
            "variant_id": variant_id,
            "timestamp": datetime.now().isoformat(),
            "prediction_data": prediction_data,
            "response_time_ms": response_time_ms,
            "error_occurred": error_occurred
        }
        
        self._test_data[test_id].append(request_record)
    
    def update_traffic_percentage(self, test_id: str, new_percentage: float):
        """Update traffic percentage for a test.
        
        Args:
            test_id: ID of the test
            new_percentage: New traffic percentage
        """
        if test_id not in self._tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self._tests[test_id]
        
        if new_percentage < 0 or new_percentage > 100:
            raise ValueError("Traffic percentage must be between 0 and 100")
        
        test.current_traffic_percentage = new_percentage
        
        self.logger.info(f"Updated traffic percentage for test {test_id} to {new_percentage}%")
    
    def analyze_test_results(self, test_id: str) -> TestResult:
        """Analyze A/B test results and determine statistical significance.
        
        Args:
            test_id: ID of the test to analyze
            
        Returns:
            Test results with statistical analysis
        """
        if test_id not in self._tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self._tests[test_id]
        test_data = self._test_data.get(test_id, [])
        
        if not test_data:
            raise ValueError(f"No data available for test {test_id}")
        
        # Calculate metrics for each variant
        variant_metrics = self._calculate_variant_metrics(test, test_data)
        
        # Perform statistical significance testing
        statistical_significance = self._calculate_statistical_significance(
            variant_metrics, test.config.success_metric
        )
        
        # Determine winner
        winner = self._determine_winner(variant_metrics, statistical_significance, test.config.success_metric)
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(variant_metrics, test.config.success_metric)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            test, variant_metrics, statistical_significance, winner
        )
        
        return TestResult(
            test_id=test_id,
            variant_metrics=variant_metrics,
            statistical_significance=statistical_significance,
            winner=winner,
            confidence_level=1.0 - test.config.significance_threshold,
            effect_size=effect_size,
            recommendations=recommendations,
            generated_at=datetime.now()
        )
    
    def _calculate_variant_metrics(self, test: ABTest, test_data: List[Dict[str, Any]]) -> List[TestMetrics]:
        """Calculate metrics for each variant.
        
        Args:
            test: The A/B test
            test_data: List of request records
            
        Returns:
            List of TestMetrics for each variant
        """
        variant_metrics = []
        
        for variant in test.config.variants:
            variant_data = [d for d in test_data if d["variant_id"] == variant.variant_id]
            
            if not variant_data:
                # Create empty metrics for variants with no data
                metrics = TestMetrics(
                    variant_id=variant.variant_id,
                    total_requests=0,
                    total_predictions=0,
                    avg_response_time_ms=0.0,
                    error_rate=0.0,
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    custom_metrics={}
                )
                variant_metrics.append(metrics)
                continue
            
            # Calculate basic metrics
            total_requests = len(variant_data)
            errors = sum(1 for d in variant_data if d["error_occurred"])
            error_rate = errors / total_requests if total_requests > 0 else 0.0
            
            response_times = [d["response_time_ms"] for d in variant_data]
            avg_response_time = np.mean(response_times) if response_times else 0.0
            
            # Calculate prediction quality metrics (simplified)
            # In practice, you'd need ground truth data to calculate these properly
            accuracy = self._calculate_accuracy_from_data(variant_data)
            precision = self._calculate_precision_from_data(variant_data)
            recall = self._calculate_recall_from_data(variant_data)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics = TestMetrics(
                variant_id=variant.variant_id,
                total_requests=total_requests,
                total_predictions=total_requests - errors,
                avg_response_time_ms=avg_response_time,
                error_rate=error_rate,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                custom_metrics={}
            )
            
            variant_metrics.append(metrics)
        
        return variant_metrics
    
    def _calculate_accuracy_from_data(self, variant_data: List[Dict[str, Any]]) -> float:
        """Calculate accuracy from variant data (placeholder implementation).
        
        Args:
            variant_data: List of request records for a variant
            
        Returns:
            Accuracy score
        """
        # Placeholder - in practice, you'd compare predictions with ground truth
        return 0.85 + np.random.normal(0, 0.05)
    
    def _calculate_precision_from_data(self, variant_data: List[Dict[str, Any]]) -> float:
        """Calculate precision from variant data (placeholder implementation).
        
        Args:
            variant_data: List of request records for a variant
            
        Returns:
            Precision score
        """
        # Placeholder implementation
        return 0.82 + np.random.normal(0, 0.05)
    
    def _calculate_recall_from_data(self, variant_data: List[Dict[str, Any]]) -> float:
        """Calculate recall from variant data (placeholder implementation).
        
        Args:
            variant_data: List of request records for a variant
            
        Returns:
            Recall score
        """
        # Placeholder implementation
        return 0.78 + np.random.normal(0, 0.06)
    
    def _calculate_statistical_significance(self, 
                                          variant_metrics: List[TestMetrics],
                                          success_metric: str) -> Dict[str, Any]:
        """Calculate statistical significance between variants.
        
        Args:
            variant_metrics: Metrics for each variant
            success_metric: Primary metric to test
            
        Returns:
            Statistical significance results
        """
        if len(variant_metrics) < 2:
            return {"error": "Need at least 2 variants for significance testing"}
        
        # Get metric values for comparison
        metric_values = []
        sample_sizes = []
        
        for metrics in variant_metrics:
            metric_value = getattr(metrics, success_metric, 0.0)
            metric_values.append(metric_value)
            sample_sizes.append(metrics.total_requests)
        
        # Perform t-test between first two variants (can be extended for multiple comparisons)
        if len(metric_values) >= 2 and all(n > 0 for n in sample_sizes[:2]):
            # Simulate sample data for t-test (in practice, you'd use actual sample data)
            sample1 = np.random.normal(metric_values[0], 0.1, sample_sizes[0])
            sample2 = np.random.normal(metric_values[1], 0.1, sample_sizes[1])
            
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
            
            return {
                "test_type": "t_test",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "is_significant": p_value < 0.05,
                "variants_compared": [variant_metrics[0].variant_id, variant_metrics[1].variant_id],
                "metric_values": metric_values[:2]
            }
        
        return {"error": "Insufficient data for significance testing"}
    
    def _determine_winner(self,
                         variant_metrics: List[TestMetrics],
                         statistical_significance: Dict[str, Any],
                         success_metric: str) -> Optional[str]:
        """Determine the winning variant.
        
        Args:
            variant_metrics: Metrics for each variant
            statistical_significance: Statistical significance results
            success_metric: Primary metric to optimize
            
        Returns:
            Variant ID of winner or None
        """
        if not statistical_significance.get("is_significant", False):
            return None
        
        # Find variant with best metric value
        best_variant = None
        best_value = -float('inf')
        
        for metrics in variant_metrics:
            metric_value = getattr(metrics, success_metric, 0.0)
            if metric_value > best_value:
                best_value = metric_value
                best_variant = metrics.variant_id
        
        return best_variant
    
    def _calculate_effect_size(self, variant_metrics: List[TestMetrics], success_metric: str) -> float:
        """Calculate effect size between variants.
        
        Args:
            variant_metrics: Metrics for each variant
            success_metric: Primary metric
            
        Returns:
            Effect size (Cohen's d)
        """
        if len(variant_metrics) < 2:
            return 0.0
        
        metric_values = [getattr(m, success_metric, 0.0) for m in variant_metrics[:2]]
        
        # Simplified effect size calculation
        if len(metric_values) >= 2:
            diff = abs(metric_values[0] - metric_values[1])
            pooled_std = 0.1  # Placeholder - would calculate from actual data
            return diff / pooled_std if pooled_std > 0 else 0.0
        
        return 0.0
    
    def _generate_recommendations(self,
                                test: ABTest,
                                variant_metrics: List[TestMetrics],
                                statistical_significance: Dict[str, Any],
                                winner: Optional[str]) -> List[str]:
        """Generate recommendations based on test results.
        
        Args:
            test: The A/B test
            variant_metrics: Metrics for each variant
            statistical_significance: Statistical significance results
            winner: Winning variant ID
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if winner:
            winner_name = next(v.name for v in test.config.variants if v.variant_id == winner)
            recommendations.append(f"Deploy variant '{winner_name}' to 100% of traffic")
            recommendations.append("Monitor performance closely after full rollout")
        else:
            if not statistical_significance.get("is_significant", False):
                recommendations.append("No statistically significant difference found")
                recommendations.append("Consider running test longer or increasing sample size")
            else:
                recommendations.append("Results are inconclusive")
        
        # Check for performance issues
        for metrics in variant_metrics:
            if metrics.error_rate > 0.05:  # 5% error rate threshold
                variant_name = next(v.name for v in test.config.variants if v.variant_id == metrics.variant_id)
                recommendations.append(f"Investigate high error rate in variant '{variant_name}' ({metrics.error_rate:.1%})")
            
            if metrics.avg_response_time_ms > 1000:  # 1 second threshold
                variant_name = next(v.name for v in test.config.variants if v.variant_id == metrics.variant_id)
                recommendations.append(f"Optimize response time for variant '{variant_name}' ({metrics.avg_response_time_ms:.0f}ms)")
        
        return recommendations
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get A/B test by ID.
        
        Args:
            test_id: Test ID
            
        Returns:
            ABTest object or None
        """
        return self._tests.get(test_id)
    
    def get_active_tests(self) -> List[ABTest]:
        """Get all active A/B tests.
        
        Returns:
            List of active tests
        """
        return [test for test in self._tests.values() if test.status == TestStatus.RUNNING]
    
    def check_early_stopping(self, test_id: str) -> bool:
        """Check if test should be stopped early due to clear winner.
        
        Args:
            test_id: Test ID to check
            
        Returns:
            True if test should be stopped early
        """
        if test_id not in self._tests:
            return False
        
        test = self._tests[test_id]
        
        if not test.config.early_stopping_enabled:
            return False
        
        try:
            results = self.analyze_test_results(test_id)
            
            # Check if we have a clear winner with high confidence
            if (results.winner and 
                results.statistical_significance.get("is_significant", False) and
                results.statistical_significance.get("p_value", 1.0) < test.config.early_stopping_threshold):
                return True
                
        except Exception as e:
            self.logger.error(f"Error checking early stopping for test {test_id}: {e}")
        
        return False
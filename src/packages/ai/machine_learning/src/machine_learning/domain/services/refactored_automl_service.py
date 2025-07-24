"""Refactored AutoML service using hexagonal architecture.

This service demonstrates proper dependency injection and interface usage
following hexagonal architecture principles. It delegates to adapters
instead of directly importing external libraries.
"""

import logging
from typing import Any, Dict, List, Optional

from ..interfaces.automl_operations import (
    AutoMLOptimizationPort,
    ModelSelectionPort,
    OptimizationConfig,
    AlgorithmConfig,
    OptimizationResult,
    AlgorithmType,
    OptimizationMetric,
    SearchStrategy,
)
from ..interfaces.monitoring_operations import (
    DistributedTracingPort,
    MonitoringPort,
)
from ..entities.dataset import Dataset
from ..entities.prediction_result import PredictionResult

logger = logging.getLogger(__name__)


class AutoMLService:
    """AutoML service using hexagonal architecture.
    
    This service orchestrates machine learning optimization by delegating
    to infrastructure adapters while maintaining domain logic separation.
    """
    
    def __init__(
        self,
        automl_port: AutoMLOptimizationPort,
        model_selection_port: ModelSelectionPort,
        monitoring_port: Optional[MonitoringPort] = None,
        tracing_port: Optional[DistributedTracingPort] = None
    ):
        """Initialize AutoML service with dependency injection.
        
        Args:
            automl_port: Port for AutoML optimization operations
            model_selection_port: Port for model selection operations
            monitoring_port: Optional port for monitoring operations
            tracing_port: Optional port for distributed tracing
        """
        self._automl_port = automl_port
        self._model_selection_port = model_selection_port
        self._monitoring_port = monitoring_port
        self._tracing_port = tracing_port
        self._logger = logging.getLogger(__name__)
    
    async def optimize_prediction(
        self,
        dataset: Dataset,
        optimization_config: Optional[OptimizationConfig] = None,
        ground_truth: Optional[Any] = None,
    ) -> OptimizationResult:
        """Automatically optimize machine learning predictions for the given dataset.
        
        This method demonstrates clean domain logic that orchestrates
        infrastructure operations through well-defined interfaces.
        """
        if optimization_config is None:
            optimization_config = OptimizationConfig()
        
        # Start tracing if available
        trace_span = None
        if self._tracing_port:
            trace_span = await self._tracing_port.start_trace(
                "automl_optimization",
                tags={"dataset_size": len(dataset.data) if hasattr(dataset, 'data') else 0}
            )
        
        try:
            self._logger.info("Starting AutoML optimization")
            
            # Record optimization start metric
            if self._monitoring_port:
                await self._monitoring_port.increment_counter(
                    "automl_optimizations_started",
                    labels={"strategy": optimization_config.search_strategy.value}
                )
            
            # Delegate to AutoML adapter
            result = await self._automl_port.optimize_model(
                dataset, optimization_config, ground_truth
            )
            
            # Log success metrics
            if self._monitoring_port:
                await self._monitoring_port.increment_counter("automl_optimizations_completed")
                await self._monitoring_port.set_gauge(
                    "automl_best_score", 
                    result.best_score,
                    labels={"algorithm": result.best_algorithm_type.value}
                )
                await self._monitoring_port.record_histogram(
                    "automl_optimization_duration_seconds",
                    result.optimization_time_seconds
                )
            
            # Add trace tags
            if trace_span and self._tracing_port:
                await self._tracing_port.add_trace_tag(trace_span, "best_algorithm", result.best_algorithm_type.value)
                await self._tracing_port.add_trace_tag(trace_span, "best_score", result.best_score)
                await self._tracing_port.add_trace_tag(trace_span, "total_trials", result.total_trials)
            
            self._logger.info(
                f"AutoML optimization completed: {result.best_algorithm_type.value} "
                f"(score: {result.best_score:.4f}, trials: {result.total_trials})"
            )
            
            return result
            
        except Exception as e:
            # Record error metrics
            if self._monitoring_port:
                await self._monitoring_port.increment_counter(
                    "automl_optimizations_failed",
                    labels={"error_type": type(e).__name__}
                )
            
            # Add error to trace
            if trace_span and self._tracing_port:
                await self._tracing_port.finish_trace(trace_span, "error", str(e))
            
            self._logger.error(f"AutoML optimization failed: {e}")
            raise
        
        finally:
            # Finish trace
            if trace_span and self._tracing_port:
                await self._tracing_port.finish_trace(trace_span)
    
    async def auto_select_algorithm(
        self,
        dataset: Dataset,
        ground_truth: Optional[Any] = None,
        quick_mode: bool = False,
    ) -> tuple[AlgorithmType, AlgorithmConfig]:
        """Automatically select the best algorithm for a dataset.
        
        This method demonstrates domain orchestration without external coupling.
        """
        # Start tracing
        trace_span = None
        if self._tracing_port:
            trace_span = await self._tracing_port.start_trace(
                "algorithm_selection",
                tags={"quick_mode": quick_mode}
            )
        
        try:
            self._logger.info(f"Starting algorithm selection (quick_mode: {quick_mode})")
            
            if quick_mode:
                # Use heuristic-based selection
                requirements = {"performance": "balanced", "speed": "fast"}
                algorithm, config = await self._model_selection_port.select_best_model(
                    dataset, requirements, quick_mode=True
                )
            else:
                # Use full optimization
                optimization_config = OptimizationConfig(
                    max_trials=50,
                    search_strategy=SearchStrategy.RANDOM_SEARCH,
                )
                
                result = await self.optimize_prediction(dataset, optimization_config, ground_truth)
                algorithm = result.best_algorithm_type
                config = result.best_config
            
            # Add trace tags
            if trace_span and self._tracing_port:
                await self._tracing_port.add_trace_tag(trace_span, "selected_algorithm", algorithm.value)
            
            self._logger.info(f"Selected algorithm: {algorithm.value}")
            return algorithm, config
            
        except Exception as e:
            if trace_span and self._tracing_port:
                await self._tracing_port.finish_trace(trace_span, "error", str(e))
            raise
        
        finally:
            if trace_span and self._tracing_port:
                await self._tracing_port.finish_trace(trace_span)
    
    async def get_optimization_recommendations(
        self, 
        dataset: Dataset, 
        current_results: Optional[PredictionResult] = None
    ) -> Dict[str, List[str]]:
        """Get recommendations for improving prediction performance.
        
        This method shows how domain logic can remain clean while
        delegating complex analysis to adapters.
        """
        try:
            # Analyze dataset characteristics
            characteristics = await self._model_selection_port.analyze_dataset_characteristics(dataset)
            
            # Get performance-based recommendations
            current_performance = None
            if current_results:
                current_performance = {
                    "anomaly_rate": current_results.anomaly_count / current_results.total_samples,
                    "execution_time": getattr(current_results, 'execution_time', 0),
                }
            
            recommendations = await self._model_selection_port.get_model_recommendations(
                dataset, current_performance
            )
            
            # Add domain-specific business rules
            recommendations = self._enhance_recommendations_with_domain_logic(
                recommendations, characteristics, current_performance
            )
            
            return recommendations
            
        except Exception as e:
            self._logger.error(f"Failed to generate recommendations: {e}")
            # Return safe default recommendations
            return {
                "algorithms": ["Consider trying different algorithms"],
                "preprocessing": ["Ensure data quality"],
                "general": ["Monitor model performance over time"]
            }
    
    def _enhance_recommendations_with_domain_logic(
        self,
        base_recommendations: Dict[str, List[str]],
        characteristics: Dict[str, Any],
        current_performance: Optional[Dict[str, float]]
    ) -> Dict[str, List[str]]:
        """Enhance recommendations with domain-specific business logic.
        
        This method demonstrates how domain services can add business logic
        on top of infrastructure adapter results.
        """
        enhanced = base_recommendations.copy()
        
        # Add business rules based on dataset characteristics
        n_samples = characteristics.get("n_samples", 0)
        n_features = characteristics.get("n_features", 0)
        
        if n_samples > 100000:
            enhanced.setdefault("scalability", []).append(
                "Consider distributed processing for large datasets"
            )
        
        if n_features > 50:
            enhanced.setdefault("preprocessing", []).append(
                "Consider dimensionality reduction for high-dimensional data"
            )
        
        # Add performance-based recommendations
        if current_performance:
            anomaly_rate = current_performance.get("anomaly_rate", 0)
            if anomaly_rate > 0.3:
                enhanced.setdefault("tuning", []).append(
                    "Anomaly rate seems high - consider reducing contamination parameter"
                )
            elif anomaly_rate < 0.01:
                enhanced.setdefault("tuning", []).append(
                    "Anomaly rate seems low - consider increasing contamination parameter"
                )
            
            execution_time = current_performance.get("execution_time", 0)
            if execution_time > 60:  # More than 1 minute
                enhanced.setdefault("performance", []).append(
                    "Consider faster algorithms or parameter optimization for better performance"
                )
        
        # Add general domain best practices
        enhanced.setdefault("monitoring", []).extend([
            "Implement model performance monitoring",
            "Set up data drift detection",
            "Establish model retraining pipelines"
        ])
        
        return enhanced
    
    async def compare_optimization_results(
        self,
        results: List[OptimizationResult]
    ) -> Dict[str, Any]:
        """Compare multiple optimization results to provide insights.
        
        This method demonstrates domain analysis logic while delegating
        detailed comparisons to adapters.
        """
        if not results:
            return {"error": "No results to compare"}
        
        try:
            # Extract trial data for comparison
            all_trials = []
            for result in results:
                all_trials.extend(result.trial_history)
            
            # Delegate detailed comparison to adapter
            comparison = await self._model_selection_port.compare_algorithms(all_trials)
            
            # Add domain-specific insights
            domain_insights = self._generate_domain_insights(results)
            comparison["domain_insights"] = domain_insights
            
            return comparison
            
        except Exception as e:
            self._logger.error(f"Failed to compare optimization results: {e}")
            return {"error": f"Comparison failed: {e}"}
    
    def _generate_domain_insights(self, results: List[OptimizationResult]) -> List[str]:
        """Generate domain-specific insights from optimization results.
        
        This method shows how domain knowledge can be applied to
        interpret technical results in business context.
        """
        insights = []
        
        # Analyze performance consistency
        best_scores = [result.best_score for result in results]
        if len(set(best_scores)) == 1:
            insights.append("All optimizations achieved similar performance - results are consistent")
        elif max(best_scores) - min(best_scores) > 0.2:
            insights.append("Performance varies significantly between runs - consider ensemble methods")
        
        # Analyze algorithm preferences
        best_algorithms = [result.best_algorithm_type for result in results]
        algorithm_counts = {}
        for alg in best_algorithms:
            algorithm_counts[alg] = algorithm_counts.get(alg, 0) + 1
        
        most_common = max(algorithm_counts, key=algorithm_counts.get)
        if algorithm_counts[most_common] > len(results) / 2:
            insights.append(f"{most_common.value} consistently performs well for this dataset type")
        
        # Analyze optimization efficiency
        total_trials = [result.total_trials for result in results]
        avg_trials = sum(total_trials) / len(total_trials)
        if avg_trials < 20:
            insights.append("Quick convergence observed - dataset may have clear optimal parameters")
        elif avg_trials > 80:
            insights.append("Slow convergence - parameter space may be complex or noisy")
        
        return insights


# Factory function for creating service with proper dependencies
def create_automl_service(
    automl_port: AutoMLOptimizationPort,
    model_selection_port: ModelSelectionPort,
    monitoring_port: Optional[MonitoringPort] = None,
    tracing_port: Optional[DistributedTracingPort] = None
) -> AutoMLService:
    """Factory function for creating AutoML service with dependencies.
    
    This function demonstrates dependency injection pattern for clean
    service instantiation.
    """
    return AutoMLService(
        automl_port=automl_port,
        model_selection_port=model_selection_port,
        monitoring_port=monitoring_port,
        tracing_port=tracing_port
    )
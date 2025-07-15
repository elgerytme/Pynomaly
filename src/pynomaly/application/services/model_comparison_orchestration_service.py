"""
Model Comparison and Selection Orchestration Service

This service implements Issue #11 (A-003) by orchestrating multi-algorithm comparison workflows
with statistical significance testing, building upon existing comparison infrastructure.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer

from pynomaly.domain.services.model_selector import ModelSelector, ParetoOptimizer
from pynomaly.domain.services.metrics_calculator import MetricsCalculator
from pynomaly.application.services.ab_testing_service import ABTestingService
from pynomaly.application.services.performance_benchmarking_service import PerformanceBenchmarkingService
from pynomaly.application.services.experiment_tracking_service import ExperimentTrackingService
from packages.data_science.domain.services.model_validation_service import ModelValidationService


class ComparisonStrategy(str, Enum):
    """Strategies for model comparison"""
    CROSS_VALIDATION = "cross_validation"
    HOLDOUT_VALIDATION = "holdout_validation"
    BOOTSTRAP = "bootstrap"
    TEMPORAL_SPLIT = "temporal_split"
    STRATIFIED_SPLIT = "stratified_split"
    AB_TEST = "ab_test"


class SignificanceTest(str, Enum):
    """Statistical significance tests"""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    FRIEDMAN = "friedman"
    MCNEMAR = "mcnemar"
    PAIRED_T_TEST = "paired_t_test"
    MANN_WHITNEY = "mann_whitney"


class SelectionCriteria(str, Enum):
    """Model selection criteria"""
    PERFORMANCE = "performance"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    PARETO_OPTIMAL = "pareto_optimal"
    BUSINESS_METRICS = "business_metrics"
    COMBINED = "combined"


@dataclass
class ComparisonConfig:
    """Configuration for model comparison workflows"""
    primary_metric: str = "f1_score"
    secondary_metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall"])
    significance_level: float = 0.05
    min_effect_size: float = 0.01
    
    # Validation configuration
    validation_strategy: ComparisonStrategy = ComparisonStrategy.CROSS_VALIDATION
    cv_folds: int = 5
    validation_split: float = 0.2
    bootstrap_samples: int = 1000
    
    # Statistical testing
    significance_test: SignificanceTest = SignificanceTest.PAIRED_T_TEST
    multiple_comparison_correction: str = "bonferroni"
    
    # Selection criteria
    selection_criteria: SelectionCriteria = SelectionCriteria.COMBINED
    pareto_objectives: List[str] = field(default_factory=lambda: ["f1_score", "training_time"])
    
    # Performance constraints
    max_training_time: Optional[float] = None
    max_memory_usage: Optional[float] = None
    min_performance_threshold: Optional[float] = None
    
    # Business constraints
    business_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Parallel processing
    max_workers: int = 4
    timeout_seconds: int = 3600


@dataclass
class ModelComparisonResult:
    """Results from model comparison"""
    model_id: str
    algorithm_name: str
    performance_metrics: Dict[str, float]
    statistical_results: Dict[str, Any]
    training_time: float
    memory_usage: float
    validation_scores: List[float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    rank: int
    is_significant: bool
    is_pareto_optimal: bool
    selection_rationale: str


@dataclass
class ComparisonWorkflowResult:
    """Results from complete comparison workflow"""
    workflow_id: str
    algorithm_results: List[ModelComparisonResult]
    best_model: ModelComparisonResult
    statistical_analysis: Dict[str, Any]
    pareto_front: List[ModelComparisonResult]
    recommendations: List[str]
    execution_time: float
    metadata: Dict[str, Any]


class ModelComparisonOrchestrator:
    """
    Orchestrates multi-algorithm comparison workflows with statistical significance testing
    """
    
    def __init__(
        self,
        model_selector: ModelSelector,
        metrics_calculator: MetricsCalculator,
        ab_testing_service: ABTestingService,
        performance_benchmarking_service: PerformanceBenchmarkingService,
        experiment_tracking_service: ExperimentTrackingService,
        model_validation_service: ModelValidationService
    ):
        self.model_selector = model_selector
        self.metrics_calculator = metrics_calculator
        self.ab_testing_service = ab_testing_service
        self.performance_benchmarking_service = performance_benchmarking_service
        self.experiment_tracking_service = experiment_tracking_service
        self.model_validation_service = model_validation_service
        
        self.logger = logging.getLogger(__name__)
        self.pareto_optimizer = ParetoOptimizer()
        
        # Execution tracking
        self.active_workflows: Dict[str, ComparisonConfig] = {}
        self.workflow_results: Dict[str, ComparisonWorkflowResult] = {}
    
    async def execute_model_comparison_workflow(
        self,
        workflow_id: str,
        algorithms: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        config: Optional[ComparisonConfig] = None
    ) -> ComparisonWorkflowResult:
        """
        Execute complete model comparison workflow with statistical significance testing
        """
        start_time = datetime.utcnow()
        config = config or ComparisonConfig()
        
        try:
            self.logger.info(f"Starting model comparison workflow {workflow_id}")
            self.active_workflows[workflow_id] = config
            
            # Create experiment
            experiment_id = await self.experiment_tracking_service.create_experiment(
                name=f"model_comparison_{workflow_id}",
                description=f"Multi-algorithm comparison for {workflow_id}",
                tags=["model_comparison", "statistical_testing"]
            )
            
            # Step 1: Execute algorithm comparisons
            algorithm_results = await self._execute_algorithm_comparisons(
                algorithms, X_train, y_train, X_test, y_test, config, experiment_id
            )
            
            # Step 2: Perform statistical analysis
            statistical_analysis = await self._perform_statistical_analysis(
                algorithm_results, config
            )
            
            # Step 3: Identify Pareto front
            pareto_front = await self._identify_pareto_front(
                algorithm_results, config
            )
            
            # Step 4: Select best model
            best_model = await self._select_best_model(
                algorithm_results, statistical_analysis, config
            )
            
            # Step 5: Generate recommendations
            recommendations = await self._generate_recommendations(
                algorithm_results, statistical_analysis, best_model, config
            )
            
            # Create workflow result
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            workflow_result = ComparisonWorkflowResult(
                workflow_id=workflow_id,
                algorithm_results=algorithm_results,
                best_model=best_model,
                statistical_analysis=statistical_analysis,
                pareto_front=pareto_front,
                recommendations=recommendations,
                execution_time=execution_time,
                metadata={
                    "config": config,
                    "experiment_id": experiment_id,
                    "num_algorithms": len(algorithms),
                    "dataset_shape": X_train.shape
                }
            )
            
            self.workflow_results[workflow_id] = workflow_result
            
            # Log experiment completion
            await self.experiment_tracking_service.log_experiment_completion(
                experiment_id, workflow_result
            )
            
            self.logger.info(f"Completed model comparison workflow {workflow_id}")
            return workflow_result
            
        except Exception as e:
            self.logger.error(f"Error in model comparison workflow {workflow_id}: {str(e)}")
            raise
        finally:
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
    
    async def _execute_algorithm_comparisons(
        self,
        algorithms: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        config: ComparisonConfig,
        experiment_id: str
    ) -> List[ModelComparisonResult]:
        """Execute comparisons for all algorithms"""
        results = []
        
        # Execute algorithms in parallel
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_algorithm = {
                executor.submit(
                    self._evaluate_single_algorithm,
                    alg_name, algorithm, X_train, y_train, X_test, y_test, config, experiment_id
                ): alg_name
                for alg_name, algorithm in algorithms.items()
            }
            
            for future in as_completed(future_to_algorithm):
                algorithm_name = future_to_algorithm[future]
                try:
                    result = await asyncio.wrap_future(future)
                    results.append(result)
                    self.logger.info(f"Completed evaluation for {algorithm_name}")
                except Exception as e:
                    self.logger.error(f"Error evaluating {algorithm_name}: {str(e)}")
                    # Continue with other algorithms
        
        # Rank results
        results.sort(key=lambda x: x.performance_metrics[config.primary_metric], reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _evaluate_single_algorithm(
        self,
        algorithm_name: str,
        algorithm: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        config: ComparisonConfig,
        experiment_id: str
    ) -> ModelComparisonResult:
        """Evaluate a single algorithm"""
        try:
            # Record start time and memory
            start_time = datetime.utcnow()
            start_memory = self._get_memory_usage()
            
            # Perform validation based on strategy
            if config.validation_strategy == ComparisonStrategy.CROSS_VALIDATION:
                scores = self._cross_validation_evaluation(
                    algorithm, X_train, y_train, config
                )
            elif config.validation_strategy == ComparisonStrategy.HOLDOUT_VALIDATION:
                scores = self._holdout_validation_evaluation(
                    algorithm, X_train, y_train, X_test, y_test, config
                )
            elif config.validation_strategy == ComparisonStrategy.BOOTSTRAP:
                scores = self._bootstrap_evaluation(
                    algorithm, X_train, y_train, config
                )
            else:
                scores = self._cross_validation_evaluation(
                    algorithm, X_train, y_train, config
                )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(scores, config)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                scores, config.significance_level
            )
            
            # Record execution metrics
            training_time = (datetime.utcnow() - start_time).total_seconds()
            memory_usage = self._get_memory_usage() - start_memory
            
            # Log to experiment tracking
            run_id = f"{experiment_id}_{algorithm_name}"
            asyncio.create_task(self.experiment_tracking_service.log_run(
                experiment_id=experiment_id,
                run_id=run_id,
                parameters={"algorithm": algorithm_name},
                metrics=performance_metrics,
                artifacts={"validation_scores": scores}
            ))
            
            return ModelComparisonResult(
                model_id=f"{algorithm_name}_{experiment_id}",
                algorithm_name=algorithm_name,
                performance_metrics=performance_metrics,
                statistical_results={},  # Will be filled in statistical analysis
                training_time=training_time,
                memory_usage=memory_usage,
                validation_scores=scores,
                confidence_intervals=confidence_intervals,
                rank=0,  # Will be set later
                is_significant=False,  # Will be determined in statistical analysis
                is_pareto_optimal=False,  # Will be determined in Pareto analysis
                selection_rationale=""  # Will be generated later
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating {algorithm_name}: {str(e)}")
            raise
    
    def _cross_validation_evaluation(
        self,
        algorithm: Any,
        X: np.ndarray,
        y: np.ndarray,
        config: ComparisonConfig
    ) -> List[float]:
        """Perform cross-validation evaluation"""
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
        scorer = make_scorer(self._get_scoring_function(config.primary_metric))
        
        scores = cross_val_score(
            algorithm, X, y, cv=cv, scoring=scorer, n_jobs=-1
        )
        
        return scores.tolist()
    
    def _holdout_validation_evaluation(
        self,
        algorithm: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        config: ComparisonConfig
    ) -> List[float]:
        """Perform holdout validation evaluation"""
        if X_test is None or y_test is None:
            # Split training data
            from sklearn.model_selection import train_test_split
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=config.validation_split, random_state=42
            )
        else:
            X_train_split, X_val, y_train_split, y_val = X_train, X_test, y_train, y_test
        
        # Train and evaluate
        algorithm.fit(X_train_split, y_train_split)
        predictions = algorithm.predict(X_val)
        
        scoring_func = self._get_scoring_function(config.primary_metric)
        score = scoring_func(y_val, predictions)
        
        return [score]
    
    def _bootstrap_evaluation(
        self,
        algorithm: Any,
        X: np.ndarray,
        y: np.ndarray,
        config: ComparisonConfig
    ) -> List[float]:
        """Perform bootstrap evaluation"""
        from sklearn.utils import resample
        
        scores = []
        for _ in range(config.bootstrap_samples):
            # Bootstrap sample
            X_bootstrap, y_bootstrap = resample(X, y, random_state=None)
            
            # Out-of-bag validation
            oob_indices = list(set(range(len(X))) - set(np.random.choice(len(X), len(X))))
            if len(oob_indices) == 0:
                continue
            
            X_oob = X[oob_indices]
            y_oob = y[oob_indices]
            
            # Train and evaluate
            algorithm.fit(X_bootstrap, y_bootstrap)
            predictions = algorithm.predict(X_oob)
            
            scoring_func = self._get_scoring_function(config.primary_metric)
            score = scoring_func(y_oob, predictions)
            scores.append(score)
        
        return scores
    
    async def _perform_statistical_analysis(
        self,
        algorithm_results: List[ModelComparisonResult],
        config: ComparisonConfig
    ) -> Dict[str, Any]:
        """Perform statistical significance analysis"""
        if len(algorithm_results) < 2:
            return {"message": "Need at least 2 algorithms for statistical analysis"}
        
        # Prepare data for statistical tests
        algorithm_names = [result.algorithm_name for result in algorithm_results]
        scores_matrix = np.array([result.validation_scores for result in algorithm_results])
        
        # Perform pairwise statistical tests
        pairwise_results = {}
        for i in range(len(algorithm_results)):
            for j in range(i + 1, len(algorithm_results)):
                alg1_name = algorithm_names[i]
                alg2_name = algorithm_names[j]
                
                scores1 = algorithm_results[i].validation_scores
                scores2 = algorithm_results[j].validation_scores
                
                # Perform statistical test
                test_result = self._perform_significance_test(
                    scores1, scores2, config.significance_test
                )
                
                pairwise_results[f"{alg1_name}_vs_{alg2_name}"] = test_result
                
                # Update significance flags
                if test_result["p_value"] < config.significance_level:
                    if test_result["effect_size"] >= config.min_effect_size:
                        if np.mean(scores1) > np.mean(scores2):
                            algorithm_results[i].is_significant = True
                        else:
                            algorithm_results[j].is_significant = True
        
        # Perform overall statistical test (Friedman test for multiple algorithms)
        if len(algorithm_results) > 2:
            friedman_result = self._perform_friedman_test(scores_matrix)
        else:
            friedman_result = {"message": "Friedman test requires 3+ algorithms"}
        
        # Multiple comparison correction
        if config.multiple_comparison_correction:
            corrected_results = self._apply_multiple_comparison_correction(
                pairwise_results, config.multiple_comparison_correction
            )
        else:
            corrected_results = pairwise_results
        
        return {
            "pairwise_tests": corrected_results,
            "friedman_test": friedman_result,
            "significance_level": config.significance_level,
            "multiple_comparison_correction": config.multiple_comparison_correction
        }
    
    async def _identify_pareto_front(
        self,
        algorithm_results: List[ModelComparisonResult],
        config: ComparisonConfig
    ) -> List[ModelComparisonResult]:
        """Identify Pareto optimal models"""
        if len(config.pareto_objectives) < 2:
            return []
        
        # Prepare data for Pareto analysis
        objectives_data = []
        for result in algorithm_results:
            objectives = []
            for objective in config.pareto_objectives:
                if objective in result.performance_metrics:
                    objectives.append(result.performance_metrics[objective])
                elif objective == "training_time":
                    objectives.append(-result.training_time)  # Minimize time
                elif objective == "memory_usage":
                    objectives.append(-result.memory_usage)  # Minimize memory
                else:
                    objectives.append(0)  # Default value
            objectives_data.append(objectives)
        
        # Find Pareto front
        pareto_indices = self.pareto_optimizer.find_pareto_front(
            np.array(objectives_data)
        )
        
        # Mark Pareto optimal models
        pareto_models = []
        for idx in pareto_indices:
            algorithm_results[idx].is_pareto_optimal = True
            pareto_models.append(algorithm_results[idx])
        
        return pareto_models
    
    async def _select_best_model(
        self,
        algorithm_results: List[ModelComparisonResult],
        statistical_analysis: Dict[str, Any],
        config: ComparisonConfig
    ) -> ModelComparisonResult:
        """Select the best model based on selection criteria"""
        if config.selection_criteria == SelectionCriteria.PERFORMANCE:
            best_model = max(
                algorithm_results,
                key=lambda x: x.performance_metrics[config.primary_metric]
            )
            best_model.selection_rationale = f"Highest {config.primary_metric} score"
        
        elif config.selection_criteria == SelectionCriteria.STATISTICAL_SIGNIFICANCE:
            significant_models = [r for r in algorithm_results if r.is_significant]
            if significant_models:
                best_model = max(
                    significant_models,
                    key=lambda x: x.performance_metrics[config.primary_metric]
                )
                best_model.selection_rationale = "Highest performance among statistically significant models"
            else:
                best_model = algorithm_results[0]
                best_model.selection_rationale = "No statistically significant models found, selected best performer"
        
        elif config.selection_criteria == SelectionCriteria.PARETO_OPTIMAL:
            pareto_models = [r for r in algorithm_results if r.is_pareto_optimal]
            if pareto_models:
                best_model = max(
                    pareto_models,
                    key=lambda x: x.performance_metrics[config.primary_metric]
                )
                best_model.selection_rationale = "Highest performance among Pareto optimal models"
            else:
                best_model = algorithm_results[0]
                best_model.selection_rationale = "No Pareto optimal models found, selected best performer"
        
        elif config.selection_criteria == SelectionCriteria.BUSINESS_METRICS:
            best_model = self._select_by_business_criteria(algorithm_results, config)
        
        else:  # COMBINED
            best_model = self._select_by_combined_criteria(
                algorithm_results, statistical_analysis, config
            )
        
        return best_model
    
    async def _generate_recommendations(
        self,
        algorithm_results: List[ModelComparisonResult],
        statistical_analysis: Dict[str, Any],
        best_model: ModelComparisonResult,
        config: ComparisonConfig
    ) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        # Primary recommendation
        recommendations.append(
            f"Recommended model: {best_model.algorithm_name} "
            f"({best_model.selection_rationale})"
        )
        
        # Performance analysis
        best_score = best_model.performance_metrics[config.primary_metric]
        recommendations.append(
            f"Best {config.primary_metric} score: {best_score:.4f} "
            f"(rank {best_model.rank}/{len(algorithm_results)})"
        )
        
        # Statistical significance
        significant_models = [r for r in algorithm_results if r.is_significant]
        if significant_models:
            recommendations.append(
                f"Statistically significant models: {', '.join([r.algorithm_name for r in significant_models])}"
            )
        else:
            recommendations.append("No statistically significant differences found between models")
        
        # Pareto analysis
        pareto_models = [r for r in algorithm_results if r.is_pareto_optimal]
        if pareto_models:
            recommendations.append(
                f"Pareto optimal models: {', '.join([r.algorithm_name for r in pareto_models])}"
            )
        
        # Performance constraints
        if config.max_training_time:
            fast_models = [r for r in algorithm_results if r.training_time <= config.max_training_time]
            if fast_models:
                recommendations.append(
                    f"Models meeting training time constraint: {', '.join([r.algorithm_name for r in fast_models])}"
                )
        
        # Top performers
        top_3 = algorithm_results[:3]
        recommendations.append(
            f"Top 3 performers: {', '.join([f'{r.algorithm_name} ({r.performance_metrics[config.primary_metric]:.4f})' for r in top_3])}"
        )
        
        return recommendations
    
    def _perform_significance_test(
        self,
        scores1: List[float],
        scores2: List[float],
        test_type: SignificanceTest
    ) -> Dict[str, Any]:
        """Perform statistical significance test"""
        if test_type == SignificanceTest.T_TEST:
            statistic, p_value = stats.ttest_ind(scores1, scores2)
        elif test_type == SignificanceTest.PAIRED_T_TEST:
            statistic, p_value = stats.ttest_rel(scores1, scores2)
        elif test_type == SignificanceTest.WILCOXON:
            statistic, p_value = stats.wilcoxon(scores1, scores2)
        elif test_type == SignificanceTest.MANN_WHITNEY:
            statistic, p_value = stats.mannwhitneyu(scores1, scores2)
        else:
            statistic, p_value = stats.ttest_ind(scores1, scores2)
        
        # Calculate effect size (Cohen's d)
        effect_size = (np.mean(scores1) - np.mean(scores2)) / np.sqrt(
            (np.std(scores1)**2 + np.std(scores2)**2) / 2
        )
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "test_type": test_type.value
        }
    
    def _perform_friedman_test(self, scores_matrix: np.ndarray) -> Dict[str, Any]:
        """Perform Friedman test for multiple algorithms"""
        try:
            statistic, p_value = stats.friedmanchisquare(*scores_matrix)
            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "test_type": "friedman"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _apply_multiple_comparison_correction(
        self,
        pairwise_results: Dict[str, Any],
        correction_method: str
    ) -> Dict[str, Any]:
        """Apply multiple comparison correction"""
        p_values = [result["p_value"] for result in pairwise_results.values()]
        
        if correction_method == "bonferroni":
            corrected_p_values = [p * len(p_values) for p in p_values]
        elif correction_method == "holm":
            # Holm-Bonferroni correction
            sorted_indices = np.argsort(p_values)
            corrected_p_values = [0] * len(p_values)
            for i, idx in enumerate(sorted_indices):
                corrected_p_values[idx] = p_values[idx] * (len(p_values) - i)
        else:
            corrected_p_values = p_values
        
        # Update results with corrected p-values
        corrected_results = {}
        for i, (key, result) in enumerate(pairwise_results.items()):
            corrected_result = result.copy()
            corrected_result["corrected_p_value"] = min(1.0, corrected_p_values[i])
            corrected_results[key] = corrected_result
        
        return corrected_results
    
    def _select_by_business_criteria(
        self,
        algorithm_results: List[ModelComparisonResult],
        config: ComparisonConfig
    ) -> ModelComparisonResult:
        """Select model based on business criteria"""
        # Example business criteria implementation
        best_model = algorithm_results[0]
        best_model.selection_rationale = "Selected based on business criteria"
        return best_model
    
    def _select_by_combined_criteria(
        self,
        algorithm_results: List[ModelComparisonResult],
        statistical_analysis: Dict[str, Any],
        config: ComparisonConfig
    ) -> ModelComparisonResult:
        """Select model using combined criteria"""
        # Calculate combined score
        for result in algorithm_results:
            score = result.performance_metrics[config.primary_metric]
            
            # Bonus for statistical significance
            if result.is_significant:
                score += 0.1
            
            # Bonus for Pareto optimality
            if result.is_pareto_optimal:
                score += 0.05
            
            # Penalty for long training time
            if config.max_training_time and result.training_time > config.max_training_time:
                score -= 0.1
            
            result.combined_score = score
        
        best_model = max(algorithm_results, key=lambda x: getattr(x, 'combined_score', 0))
        best_model.selection_rationale = "Selected using combined criteria (performance + significance + Pareto optimality)"
        return best_model
    
    def _calculate_performance_metrics(
        self,
        scores: List[float],
        config: ComparisonConfig
    ) -> Dict[str, float]:
        """Calculate performance metrics from validation scores"""
        return {
            config.primary_metric: np.mean(scores),
            f"{config.primary_metric}_std": np.std(scores),
            f"{config.primary_metric}_min": np.min(scores),
            f"{config.primary_metric}_max": np.max(scores)
        }
    
    def _calculate_confidence_intervals(
        self,
        scores: List[float],
        significance_level: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for scores"""
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        n = len(scores)
        
        # Calculate confidence interval
        confidence_level = 1 - significance_level
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        margin_of_error = t_value * (std_score / np.sqrt(n))
        
        return {
            "mean": (mean_score - margin_of_error, mean_score + margin_of_error)
        }
    
    def _get_scoring_function(self, metric_name: str):
        """Get scoring function for metric"""
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        
        scoring_functions = {
            "f1_score": f1_score,
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score
        }
        
        return scoring_functions.get(metric_name, f1_score)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    async def get_workflow_results(self, workflow_id: str) -> Optional[ComparisonWorkflowResult]:
        """Get results for a specific workflow"""
        return self.workflow_results.get(workflow_id)
    
    async def list_active_workflows(self) -> List[str]:
        """List all active workflow IDs"""
        return list(self.active_workflows.keys())
    
    async def export_results(
        self,
        workflow_id: str,
        format: str = "json"
    ) -> Optional[str]:
        """Export workflow results in specified format"""
        if workflow_id not in self.workflow_results:
            return None
        
        result = self.workflow_results[workflow_id]
        
        if format == "json":
            import json
            return json.dumps(result, default=str, indent=2)
        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame([
                {
                    "algorithm": r.algorithm_name,
                    "rank": r.rank,
                    "performance": r.performance_metrics.get("f1_score", 0),
                    "training_time": r.training_time,
                    "memory_usage": r.memory_usage,
                    "is_significant": r.is_significant,
                    "is_pareto_optimal": r.is_pareto_optimal
                }
                for r in result.algorithm_results
            ])
            return df.to_csv(index=False)
        else:
            return None


# Factory function
def create_model_comparison_orchestrator(
    model_selector: ModelSelector,
    metrics_calculator: MetricsCalculator,
    ab_testing_service: ABTestingService,
    performance_benchmarking_service: PerformanceBenchmarkingService,
    experiment_tracking_service: ExperimentTrackingService,
    model_validation_service: ModelValidationService
) -> ModelComparisonOrchestrator:
    """Create a configured model comparison orchestrator"""
    return ModelComparisonOrchestrator(
        model_selector=model_selector,
        metrics_calculator=metrics_calculator,
        ab_testing_service=ab_testing_service,
        performance_benchmarking_service=performance_benchmarking_service,
        experiment_tracking_service=experiment_tracking_service,
        model_validation_service=model_validation_service
    )
"""Advanced experiment analysis and visualization service."""

import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from uuid import UUID
import math

from pynomaly_mlops.domain.entities.experiment import ExperimentRun, ExperimentRunStatus
from pynomaly_mlops.domain.repositories.experiment_repository import ExperimentRepository


class ExperimentAnalysisService:
    """Service for advanced experiment analysis and insights."""
    
    def __init__(self, experiment_repository: ExperimentRepository):
        """Initialize service with repository.
        
        Args:
            experiment_repository: Repository for experiment data
        """
        self.experiment_repository = experiment_repository
    
    async def analyze_experiment_performance(
        self,
        experiment_id: UUID,
        metric_key: str,
        analysis_window_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze experiment performance over time.
        
        Args:
            experiment_id: Experiment ID
            metric_key: Metric to analyze
            analysis_window_days: Optional window in days to analyze
            
        Returns:
            Performance analysis results
        """
        # Get runs within the analysis window
        runs = await self.experiment_repository.list_runs(
            experiment_id=experiment_id,
            limit=1000  # Large limit to get all runs
        )
        
        # Filter by time window if specified
        if analysis_window_days:
            cutoff_date = datetime.utcnow() - timedelta(days=analysis_window_days)
            runs = [run for run in runs if run.start_time >= cutoff_date]
        
        # Filter completed runs with the metric
        metric_runs = [
            run for run in runs 
            if run.status == ExperimentRunStatus.COMPLETED 
            and run.metrics 
            and metric_key in run.metrics
        ]
        
        if not metric_runs:
            return {
                "metric_key": metric_key,
                "analysis_period": analysis_window_days,
                "total_runs": len(runs),
                "runs_with_metric": 0,
                "error": "No completed runs with the specified metric found"
            }
        
        # Extract metric values and timestamps
        metric_data = []
        for run in metric_runs:
            metric_data.append({
                "run_id": str(run.id),
                "value": run.metrics[metric_key],
                "timestamp": run.start_time,
                "parameters": run.parameters
            })
        
        # Sort by timestamp
        metric_data.sort(key=lambda x: x["timestamp"])
        
        values = [d["value"] for d in metric_data]
        
        # Calculate statistics
        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "range": max(values) - min(values),
            "coefficient_of_variation": statistics.stdev(values) / statistics.mean(values) if len(values) > 1 and statistics.mean(values) != 0 else 0
        }
        
        # Calculate trend analysis
        trend_analysis = self._calculate_trend(metric_data)
        
        # Identify best performing runs
        best_runs = sorted(metric_data, key=lambda x: x["value"], reverse=True)[:5]
        worst_runs = sorted(metric_data, key=lambda x: x["value"])[:5]
        
        # Parameter impact analysis
        parameter_impact = await self._analyze_parameter_impact(metric_runs, metric_key)
        
        return {
            "metric_key": metric_key,
            "analysis_period_days": analysis_window_days,
            "total_runs": len(runs),
            "runs_with_metric": len(metric_runs),
            "statistics": stats,
            "trend_analysis": trend_analysis,
            "best_performing_runs": [
                {
                    "run_id": run["run_id"],
                    "value": run["value"],
                    "timestamp": run["timestamp"].isoformat(),
                    "key_parameters": {k: v for k, v in list(run["parameters"].items())[:3]}
                }
                for run in best_runs
            ],
            "worst_performing_runs": [
                {
                    "run_id": run["run_id"],
                    "value": run["value"],
                    "timestamp": run["timestamp"].isoformat(),
                    "key_parameters": {k: v for k, v in list(run["parameters"].items())[:3]}
                }
                for run in worst_runs
            ],
            "parameter_impact": parameter_impact
        }
    
    async def compare_experiments(
        self,
        experiment_ids: List[UUID],
        metric_keys: List[str],
        statistical_tests: bool = True
    ) -> Dict[str, Any]:
        """Compare multiple experiments across metrics.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric_keys: List of metrics to compare
            statistical_tests: Whether to include statistical significance tests
            
        Returns:
            Experiment comparison results
        """
        experiment_data = {}
        
        # Collect data for each experiment
        for exp_id in experiment_ids:
            runs = await self.experiment_repository.list_runs(
                experiment_id=exp_id,
                limit=1000
            )
            
            completed_runs = [
                run for run in runs 
                if run.status == ExperimentRunStatus.COMPLETED and run.metrics
            ]
            
            experiment_data[str(exp_id)] = {
                "experiment_id": str(exp_id),
                "total_runs": len(runs),
                "completed_runs": len(completed_runs),
                "runs": completed_runs
            }
        
        # Compare metrics across experiments
        metric_comparisons = {}
        
        for metric_key in metric_keys:
            metric_comparison = {
                "metric_key": metric_key,
                "experiment_stats": {},
                "winner": None,
                "statistical_significance": None
            }
            
            experiment_values = {}
            
            for exp_id_str, exp_data in experiment_data.items():
                values = [
                    run.metrics.get(metric_key) 
                    for run in exp_data["runs"] 
                    if run.metrics and metric_key in run.metrics
                ]
                values = [v for v in values if v is not None]
                
                if values:
                    stats = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values),
                        "confidence_interval_95": self._calculate_confidence_interval(values, 0.95)
                    }
                    
                    metric_comparison["experiment_stats"][exp_id_str] = stats
                    experiment_values[exp_id_str] = values
            
            # Determine winner (assuming higher is better for now)
            if metric_comparison["experiment_stats"]:
                winner_exp = max(
                    metric_comparison["experiment_stats"].keys(),
                    key=lambda x: metric_comparison["experiment_stats"][x]["mean"]
                )
                metric_comparison["winner"] = winner_exp
            
            # Statistical significance testing
            if statistical_tests and len(experiment_values) == 2:
                exp_ids = list(experiment_values.keys())
                values1 = experiment_values[exp_ids[0]]
                values2 = experiment_values[exp_ids[1]]
                
                if len(values1) > 1 and len(values2) > 1:
                    metric_comparison["statistical_significance"] = self._t_test(values1, values2)
            
            metric_comparisons[metric_key] = metric_comparison
        
        # Overall experiment ranking
        overall_ranking = self._calculate_overall_ranking(experiment_data, metric_keys)
        
        return {
            "experiment_ids": [str(eid) for eid in experiment_ids],
            "metric_keys": metric_keys,
            "experiment_summaries": {
                exp_id: {
                    "total_runs": data["total_runs"],
                    "completed_runs": data["completed_runs"],
                    "success_rate": data["completed_runs"] / data["total_runs"] if data["total_runs"] > 0 else 0
                }
                for exp_id, data in experiment_data.items()
            },
            "metric_comparisons": metric_comparisons,
            "overall_ranking": overall_ranking
        }
    
    async def detect_anomalous_runs(
        self,
        experiment_id: UUID,
        metric_key: str,
        threshold_std_devs: float = 2.0
    ) -> Dict[str, Any]:
        """Detect anomalous experiment runs based on metrics.
        
        Args:
            experiment_id: Experiment ID
            metric_key: Metric to analyze for anomalies
            threshold_std_devs: Number of standard deviations for anomaly threshold
            
        Returns:
            Anomaly detection results
        """
        runs = await self.experiment_repository.list_runs(
            experiment_id=experiment_id,
            limit=1000
        )
        
        metric_runs = [
            run for run in runs 
            if run.status == ExperimentRunStatus.COMPLETED 
            and run.metrics 
            and metric_key in run.metrics
        ]
        
        if len(metric_runs) < 3:
            return {
                "metric_key": metric_key,
                "threshold_std_devs": threshold_std_devs,
                "error": "Need at least 3 completed runs with the metric for anomaly detection"
            }
        
        values = [run.metrics[metric_key] for run in metric_runs]
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        # Calculate anomaly thresholds
        upper_threshold = mean_value + (threshold_std_devs * std_dev)
        lower_threshold = mean_value - (threshold_std_devs * std_dev)
        
        # Identify anomalous runs
        anomalous_runs = []
        normal_runs = []
        
        for run in metric_runs:
            metric_value = run.metrics[metric_key]
            
            if metric_value > upper_threshold or metric_value < lower_threshold:
                anomaly_type = "high" if metric_value > upper_threshold else "low"
                anomaly_score = abs(metric_value - mean_value) / std_dev
                
                anomalous_runs.append({
                    "run_id": str(run.id),
                    "metric_value": metric_value,
                    "anomaly_type": anomaly_type,
                    "anomaly_score": anomaly_score,
                    "timestamp": run.start_time.isoformat(),
                    "parameters": run.parameters
                })
            else:
                normal_runs.append({
                    "run_id": str(run.id),
                    "metric_value": metric_value,
                    "timestamp": run.start_time.isoformat()
                })
        
        return {
            "metric_key": metric_key,
            "threshold_std_devs": threshold_std_devs,
            "total_runs": len(metric_runs),
            "normal_runs": len(normal_runs),
            "anomalous_runs": len(anomalous_runs),
            "detection_thresholds": {
                "upper": upper_threshold,
                "lower": lower_threshold,
                "mean": mean_value,
                "std_dev": std_dev
            },
            "anomalies": sorted(anomalous_runs, key=lambda x: x["anomaly_score"], reverse=True),
            "anomaly_rate": len(anomalous_runs) / len(metric_runs)
        }
    
    async def analyze_hyperparameter_optimization(
        self,
        experiment_id: UUID,
        target_metric: str,
        parameter_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze hyperparameter optimization progress.
        
        Args:
            experiment_id: Experiment ID
            target_metric: Target metric to optimize
            parameter_names: Optional list of parameter names to analyze
            
        Returns:
            Hyperparameter optimization analysis
        """
        runs = await self.experiment_repository.list_runs(
            experiment_id=experiment_id,
            limit=1000
        )
        
        completed_runs = [
            run for run in runs 
            if run.status == ExperimentRunStatus.COMPLETED 
            and run.metrics 
            and target_metric in run.metrics
            and run.parameters
        ]
        
        if not completed_runs:
            return {
                "target_metric": target_metric,
                "error": "No completed runs with target metric and parameters found"
            }
        
        # Sort runs by metric value (assuming higher is better)
        completed_runs.sort(key=lambda x: x.metrics[target_metric], reverse=True)
        
        # Analyze parameter importance
        if not parameter_names:
            # Extract all parameter names
            all_params = set()
            for run in completed_runs:
                all_params.update(run.parameters.keys())
            parameter_names = list(all_params)
        
        parameter_analysis = {}
        
        for param_name in parameter_names:
            param_values = []
            metric_values = []
            
            for run in completed_runs:
                if param_name in run.parameters:
                    param_values.append(run.parameters[param_name])
                    metric_values.append(run.metrics[target_metric])
            
            if len(param_values) > 2:
                # Calculate correlation
                correlation = self._calculate_correlation(param_values, metric_values)
                
                # Analyze parameter ranges for best performing runs
                top_10_percent = max(1, len(completed_runs) // 10)
                top_runs = completed_runs[:top_10_percent]
                
                top_param_values = [
                    run.parameters.get(param_name) 
                    for run in top_runs 
                    if param_name in run.parameters
                ]
                
                if top_param_values:
                    if isinstance(top_param_values[0], (int, float)):
                        param_stats = {
                            "min": min(top_param_values),
                            "max": max(top_param_values),
                            "mean": statistics.mean(top_param_values),
                            "median": statistics.median(top_param_values)
                        }
                    else:
                        # Categorical parameter
                        param_counts = {}
                        for val in top_param_values:
                            param_counts[str(val)] = param_counts.get(str(val), 0) + 1
                        param_stats = {
                            "most_frequent": max(param_counts, key=param_counts.get),
                            "frequency_distribution": param_counts
                        }
                    
                    parameter_analysis[param_name] = {
                        "correlation_with_target": correlation,
                        "top_performers_stats": param_stats,
                        "sample_size": len(param_values)
                    }
        
        # Optimization progress over time
        progress_analysis = self._analyze_optimization_progress(completed_runs, target_metric)
        
        return {
            "target_metric": target_metric,
            "total_completed_runs": len(completed_runs),
            "best_run": {
                "run_id": str(completed_runs[0].id),
                "metric_value": completed_runs[0].metrics[target_metric],
                "parameters": completed_runs[0].parameters,
                "timestamp": completed_runs[0].start_time.isoformat()
            },
            "parameter_analysis": parameter_analysis,
            "optimization_progress": progress_analysis,
            "recommendations": self._generate_optimization_recommendations(parameter_analysis)
        }
    
    def _calculate_trend(self, metric_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trend analysis for metric over time."""
        if len(metric_data) < 2:
            return {"trend": "insufficient_data"}
        
        values = [d["value"] for d in metric_data]
        
        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope using least squares
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"
        
        # Calculate improvement rate (percentage change)
        first_value = values[0]
        last_value = values[-1]
        improvement_rate = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
        
        return {
            "trend": trend_direction,
            "slope": slope,
            "improvement_rate_percent": improvement_rate,
            "first_value": first_value,
            "last_value": last_value,
            "total_runs_analyzed": n
        }
    
    async def _analyze_parameter_impact(
        self, 
        runs: List[ExperimentRun], 
        metric_key: str
    ) -> Dict[str, Any]:
        """Analyze the impact of parameters on the target metric."""
        # Extract all parameter names
        all_params = set()
        for run in runs:
            all_params.update(run.parameters.keys())
        
        parameter_impacts = {}
        
        for param_name in all_params:
            param_values = []
            metric_values = []
            
            for run in runs:
                if param_name in run.parameters:
                    param_values.append(run.parameters[param_name])
                    metric_values.append(run.metrics[metric_key])
            
            if len(param_values) > 2:
                # Check if parameter is numeric
                if all(isinstance(v, (int, float)) for v in param_values):
                    correlation = self._calculate_correlation(param_values, metric_values)
                    parameter_impacts[param_name] = {
                        "type": "numeric",
                        "correlation": correlation,
                        "sample_size": len(param_values)
                    }
                else:
                    # Categorical parameter - analyze groups
                    groups = {}
                    for i, param_val in enumerate(param_values):
                        if param_val not in groups:
                            groups[param_val] = []
                        groups[param_val].append(metric_values[i])
                    
                    group_stats = {}
                    for group_val, group_metrics in groups.items():
                        if group_metrics:
                            group_stats[str(group_val)] = {
                                "count": len(group_metrics),
                                "mean": statistics.mean(group_metrics),
                                "std_dev": statistics.stdev(group_metrics) if len(group_metrics) > 1 else 0
                            }
                    
                    parameter_impacts[param_name] = {
                        "type": "categorical",
                        "group_statistics": group_stats,
                        "sample_size": len(param_values)
                    }
        
        return parameter_impacts
    
    def _calculate_correlation(self, x_values: List[Union[int, float]], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        x_var = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        y_var = sum((y_values[i] - y_mean) ** 2 for i in range(n))
        
        denominator = math.sqrt(x_var * y_var)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_confidence_interval(self, values: List[float], confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        if len(values) < 2:
            mean_val = values[0] if values else 0
            return (mean_val, mean_val)
        
        mean_val = statistics.mean(values)
        std_err = statistics.stdev(values) / math.sqrt(len(values))
        
        # Using t-distribution critical value approximation for 95% CI
        t_critical = 1.96 if len(values) > 30 else 2.0  # Simplified
        
        margin_of_error = t_critical * std_err
        
        return (mean_val - margin_of_error, mean_val + margin_of_error)
    
    def _t_test(self, values1: List[float], values2: List[float]) -> Dict[str, Any]:
        """Perform two-sample t-test (simplified)."""
        if len(values1) < 2 or len(values2) < 2:
            return {"error": "Insufficient sample size for t-test"}
        
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)
        
        var1 = statistics.variance(values1)
        var2 = statistics.variance(values2)
        
        n1, n2 = len(values1), len(values2)
        
        # Pooled standard error
        pooled_se = math.sqrt(var1/n1 + var2/n2)
        
        if pooled_se == 0:
            return {"error": "Cannot compute t-test with zero variance"}
        
        t_stat = (mean1 - mean2) / pooled_se
        
        # Simplified p-value approximation
        p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + math.sqrt(n1 + n2 - 2)))
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_at_0_05": p_value < 0.05,
            "mean_difference": mean1 - mean2,
            "sample_sizes": [n1, n2]
        }
    
    def _calculate_overall_ranking(
        self, 
        experiment_data: Dict[str, Any], 
        metric_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """Calculate overall experiment ranking across multiple metrics."""
        rankings = []
        
        for exp_id, data in experiment_data.items():
            score = 0
            valid_metrics = 0
            
            for metric_key in metric_keys:
                values = [
                    run.metrics.get(metric_key) 
                    for run in data["runs"] 
                    if run.metrics and metric_key in run.metrics
                ]
                values = [v for v in values if v is not None]
                
                if values:
                    score += statistics.mean(values)
                    valid_metrics += 1
            
            avg_score = score / valid_metrics if valid_metrics > 0 else 0
            
            rankings.append({
                "experiment_id": exp_id,
                "average_score": avg_score,
                "valid_metrics": valid_metrics,
                "total_metrics": len(metric_keys),
                "completed_runs": data["completed_runs"]
            })
        
        # Sort by average score (descending)
        rankings.sort(key=lambda x: x["average_score"], reverse=True)
        
        # Add rank
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1
        
        return rankings
    
    def _analyze_optimization_progress(
        self, 
        runs: List[ExperimentRun], 
        target_metric: str
    ) -> Dict[str, Any]:
        """Analyze optimization progress over time."""
        # Sort by start time
        runs.sort(key=lambda x: x.start_time)
        
        values = [run.metrics[target_metric] for run in runs]
        
        # Calculate running best
        running_best = []
        current_best = float('-inf')
        
        for value in values:
            if value > current_best:
                current_best = value
            running_best.append(current_best)
        
        # Calculate improvement events
        improvements = 0
        for i in range(1, len(running_best)):
            if running_best[i] > running_best[i-1]:
                improvements += 1
        
        # Calculate convergence
        if len(values) >= 10:
            last_10_best = running_best[-10:]
            last_10_variation = max(last_10_best) - min(last_10_best)
            is_converging = last_10_variation < (max(values) - min(values)) * 0.05
        else:
            is_converging = False
        
        return {
            "total_runs": len(runs),
            "improvement_events": improvements,
            "improvement_rate": improvements / len(runs) if len(runs) > 0 else 0,
            "initial_best": running_best[0] if running_best else None,
            "final_best": running_best[-1] if running_best else None,
            "total_improvement": running_best[-1] - running_best[0] if len(running_best) >= 2 else 0,
            "is_converging": is_converging,
            "convergence_analysis": {
                "runs_analyzed": len(last_10_best) if len(values) >= 10 else len(values),
                "variation_in_recent_runs": last_10_variation if len(values) >= 10 else None
            }
        }
    
    def _generate_optimization_recommendations(
        self, 
        parameter_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations based on parameter analysis."""
        recommendations = []
        
        for param_name, analysis in parameter_analysis.items():
            if analysis["type"] == "numeric" and "correlation" in analysis:
                correlation = analysis["correlation"]
                
                if abs(correlation) > 0.5:
                    if correlation > 0:
                        recommendations.append(
                            f"Parameter '{param_name}' shows strong positive correlation ({correlation:.3f}) "
                            f"with target metric. Consider exploring higher values."
                        )
                    else:
                        recommendations.append(
                            f"Parameter '{param_name}' shows strong negative correlation ({correlation:.3f}) "
                            f"with target metric. Consider exploring lower values."
                        )
                elif abs(correlation) < 0.1:
                    recommendations.append(
                        f"Parameter '{param_name}' shows little correlation ({correlation:.3f}) "
                        f"with target metric. Consider fixing this parameter or removing it from optimization."
                    )
            
            elif analysis["type"] == "categorical" and "group_statistics" in analysis:
                groups = analysis["group_statistics"]
                if len(groups) > 1:
                    best_group = max(groups.items(), key=lambda x: x[1]["mean"])
                    worst_group = min(groups.items(), key=lambda x: x[1]["mean"])
                    
                    recommendations.append(
                        f"For parameter '{param_name}', value '{best_group[0]}' performs best "
                        f"(mean: {best_group[1]['mean']:.4f}) while '{worst_group[0]}' performs worst "
                        f"(mean: {worst_group[1]['mean']:.4f})."
                    )
        
        if not recommendations:
            recommendations.append("No clear parameter optimization patterns detected. Consider running more experiments.")
        
        return recommendations
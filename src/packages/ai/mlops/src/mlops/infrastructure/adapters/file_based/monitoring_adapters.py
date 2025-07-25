"""File-based implementations for monitoring operations."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
import statistics

from mlops.domain.interfaces.mlops_monitoring_operations import (
    ModelPerformanceMonitoringPort,
    InfrastructureMonitoringPort,
    DataQualityMonitoringPort,
    DataDriftMonitoringPort,
    AlertingPort,
    HealthCheckPort,
    PerformanceMetrics,
    InfrastructureMetrics,
    DataQualityMetrics,
    DriftDetectionResult,
    MonitoringAlert,
    ModelHealthReport,
    MonitoringRule,
    MonitoringAlertSeverity,
    MonitoringAlertStatus,
    ModelHealthStatus,
    DataDriftType
)

logger = logging.getLogger(__name__)


class FileBasedModelPerformanceMonitoring(ModelPerformanceMonitoringPort):
    """File-based implementation for model performance monitoring."""
    
    def __init__(self, storage_path: str = "./mlops_data/performance"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileBasedModelPerformanceMonitoring initialized at {storage_path}")
    
    def _get_metrics_file(self, model_id: str, deployment_id: Optional[str] = None) -> Path:
        """Get the metrics file path for a model/deployment."""
        key = f"{model_id}_{deployment_id or 'default'}"
        return self._storage_path / f"{key}_performance.json"
    
    def _load_metrics(self, metrics_file: Path) -> List[Dict[str, Any]]:
        """Load metrics from file."""
        if not metrics_file.exists():
            return []
        
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                # Convert timestamp strings back to datetime objects
                for entry in data:
                    if 'timestamp' in entry:
                        entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                return data
        except Exception as e:
            logger.warning(f"Failed to load metrics from {metrics_file}: {e}")
            return []
    
    def _save_metrics(self, metrics_file: Path, metrics_data: List[Dict[str, Any]]) -> None:
        """Save metrics to file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = []
            for entry in metrics_data:
                serializable_entry = entry.copy()
                if 'timestamp' in entry and isinstance(entry['timestamp'], datetime):
                    serializable_entry['timestamp'] = entry['timestamp'].isoformat()
                serializable_data.append(serializable_entry)
            
            with open(metrics_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics to {metrics_file}: {e}")
    
    async def log_prediction_metrics(
        self,
        model_id: str,
        deployment_id: Optional[str],
        metrics: PerformanceMetrics
    ) -> None:
        """Log model prediction performance metrics."""
        metrics_file = self._get_metrics_file(model_id, deployment_id)
        metrics_data = self._load_metrics(metrics_file)
        
        # Convert PerformanceMetrics to dict
        metrics_dict = {
            "model_id": model_id,
            "deployment_id": deployment_id,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "auc_roc": metrics.auc_roc,
            "latency_p50": metrics.latency_p50,
            "latency_p95": metrics.latency_p95,
            "latency_p99": metrics.latency_p99,
            "throughput": metrics.throughput,
            "error_rate": metrics.error_rate,
            "custom_metrics": metrics.custom_metrics or {},
            "timestamp": metrics.timestamp or datetime.utcnow()
        }
        
        metrics_data.append(metrics_dict)
        
        # Keep only last 1000 entries to prevent unbounded growth
        if len(metrics_data) > 1000:
            metrics_data = metrics_data[-1000:]
        
        self._save_metrics(metrics_file, metrics_data)
        logger.info(f"Logged performance metrics for {model_id}")
    
    async def get_performance_metrics(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation_window: Optional[timedelta] = None
    ) -> List[PerformanceMetrics]:
        """Get historical performance metrics."""
        metrics_file = self._get_metrics_file(model_id, deployment_id)
        metrics_data = self._load_metrics(metrics_file)
        
        # Apply time filters
        filtered_data = []
        for entry in metrics_data:
            timestamp = entry.get('timestamp')
            if not timestamp:
                continue
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            filtered_data.append(entry)
        
        # Convert back to PerformanceMetrics objects
        performance_metrics = []
        for entry in filtered_data:
            metrics = PerformanceMetrics(
                accuracy=entry.get('accuracy'),
                precision=entry.get('precision'),
                recall=entry.get('recall'),
                f1_score=entry.get('f1_score'),
                auc_roc=entry.get('auc_roc'),
                latency_p50=entry.get('latency_p50'),
                latency_p95=entry.get('latency_p95'),
                latency_p99=entry.get('latency_p99'),
                throughput=entry.get('throughput'),
                error_rate=entry.get('error_rate'),
                custom_metrics=entry.get('custom_metrics', {}),
                timestamp=entry.get('timestamp')
            )
            performance_metrics.append(metrics)
        
        logger.info(f"Retrieved {len(performance_metrics)} performance metrics for {model_id}")
        return performance_metrics
    
    async def calculate_performance_degradation(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        baseline_period: timedelta = timedelta(days=7),
        comparison_period: timedelta = timedelta(days=1)
    ) -> Dict[str, float]:
        """Calculate performance degradation compared to baseline."""
        current_time = datetime.utcnow()
        
        # Get baseline metrics
        baseline_start = current_time - baseline_period - comparison_period
        baseline_end = current_time - comparison_period
        baseline_metrics = await self.get_performance_metrics(
            model_id, deployment_id, baseline_start, baseline_end
        )
        
        # Get comparison metrics
        comparison_start = current_time - comparison_period
        comparison_metrics = await self.get_performance_metrics(
            model_id, deployment_id, comparison_start, current_time
        )
        
        if not baseline_metrics or not comparison_metrics:
            logger.warning(f"Insufficient data for degradation calculation: {model_id}")
            return {"overall_health_score": 0.8}  # Default health score when insufficient data
        
        # Calculate averages for each metric
        degradation_metrics = {}
        
        metric_names = ["accuracy", "precision", "recall", "f1_score", "auc_roc", "latency_p95", "throughput", "error_rate"]
        
        for metric_name in metric_names:
            baseline_values = [getattr(m, metric_name) for m in baseline_metrics if getattr(m, metric_name) is not None]
            comparison_values = [getattr(m, metric_name) for m in comparison_metrics if getattr(m, metric_name) is not None]
            
            if baseline_values and comparison_values:
                baseline_avg = statistics.mean(baseline_values)
                comparison_avg = statistics.mean(comparison_values)
                
                # For error rate and latency, degradation is increase (positive is bad)
                # For other metrics, degradation is decrease (negative is bad)
                if metric_name in ["error_rate", "latency_p95"]:
                    degradation = comparison_avg - baseline_avg
                else:
                    degradation = baseline_avg - comparison_avg
                
                degradation_metrics[f"{metric_name}_degradation"] = degradation
        
        # Calculate overall health score
        if degradation_metrics:
            # Simple health score based on number of degraded metrics
            degraded_count = sum(1 for v in degradation_metrics.values() if v > 0)
            total_metrics = len(degradation_metrics)
            health_score = max(0.0, 1.0 - (degraded_count / total_metrics))
            degradation_metrics["overall_health_score"] = health_score
        
        logger.info(f"Calculated performance degradation for {model_id}")
        return degradation_metrics
    
    async def detect_performance_anomalies(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        lookback_window: timedelta = timedelta(hours=24),
        sensitivity: float = 0.95
    ) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        current_time = datetime.utcnow()
        start_time = current_time - lookback_window
        
        metrics = await self.get_performance_metrics(model_id, deployment_id, start_time, current_time)
        
        if len(metrics) < 10:  # Need minimum data points
            logger.warning(f"Insufficient data for anomaly detection: {model_id}")
            return []
        
        anomalies = []
        metric_names = ["accuracy", "precision", "recall", "f1_score", "latency_p95", "throughput", "error_rate"]
        
        for metric_name in metric_names:
            values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
            
            if len(values) < 5:
                continue
            
            # Simple anomaly detection using statistical thresholds
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            # Define thresholds based on sensitivity
            threshold_multiplier = 2.0 if sensitivity > 0.9 else 3.0 if sensitivity > 0.8 else 4.0
            upper_threshold = mean_val + (threshold_multiplier * std_val)
            lower_threshold = mean_val - (threshold_multiplier * std_val)
            
            # Check recent values for anomalies
            recent_values = values[-5:]  # Check last 5 values
            for i, value in enumerate(recent_values):
                if value > upper_threshold or value < lower_threshold:
                    anomaly_score = abs(value - mean_val) / (std_val + 1e-10)  # Avoid division by zero
                    
                    anomaly = {
                        "anomaly_id": f"anomaly_{model_id}_{metric_name}_{str(uuid4())[:8]}",
                        "metric_name": metric_name,
                        "anomaly_score": min(1.0, anomaly_score / threshold_multiplier),
                        "detected_at": current_time - timedelta(minutes=5 * (len(recent_values) - i)),
                        "value": value,
                        "expected_range": {"min": lower_threshold, "max": upper_threshold},
                        "description": f"Anomalous {metric_name} value detected",
                        "severity": "high" if anomaly_score > threshold_multiplier * 1.5 else "medium",
                        "suggested_actions": [
                            f"Investigate {metric_name} performance",
                            "Check for data quality issues",
                            "Review model predictions"
                        ]
                    }
                    anomalies.append(anomaly)
        
        logger.info(f"Detected {len(anomalies)} performance anomalies for {model_id}")
        return anomalies


class FileBasedInfrastructureMonitoring(InfrastructureMonitoringPort):
    """File-based implementation for infrastructure monitoring."""
    
    def __init__(self, storage_path: str = "./mlops_data/infrastructure"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileBasedInfrastructureMonitoring initialized at {storage_path}")
    
    def _get_metrics_file(self, deployment_id: str) -> Path:
        """Get the metrics file path for a deployment."""
        return self._storage_path / f"{deployment_id}_infrastructure.json"
    
    def _load_metrics(self, metrics_file: Path) -> List[Dict[str, Any]]:
        """Load metrics from file."""
        if not metrics_file.exists():
            return []
        
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                # Convert timestamp strings back to datetime objects
                for entry in data:
                    if 'timestamp' in entry:
                        entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                return data
        except Exception as e:
            logger.warning(f"Failed to load metrics from {metrics_file}: {e}")
            return []
    
    def _save_metrics(self, metrics_file: Path, metrics_data: List[Dict[str, Any]]) -> None:
        """Save metrics to file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = []
            for entry in metrics_data:
                serializable_entry = entry.copy()
                if 'timestamp' in entry and isinstance(entry['timestamp'], datetime):
                    serializable_entry['timestamp'] = entry['timestamp'].isoformat()
                serializable_data.append(serializable_entry)
            
            with open(metrics_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics to {metrics_file}: {e}")
    
    async def log_infrastructure_metrics(
        self,
        deployment_id: str,
        metrics: InfrastructureMetrics
    ) -> None:
        """Log infrastructure metrics."""
        metrics_file = self._get_metrics_file(deployment_id)
        metrics_data = self._load_metrics(metrics_file)
        
        # Convert InfrastructureMetrics to dict
        metrics_dict = {
            "deployment_id": deployment_id,
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "gpu_usage": metrics.gpu_usage,
            "disk_usage": metrics.disk_usage,
            "network_io": metrics.network_io,
            "request_queue_size": metrics.request_queue_size,
            "active_connections": metrics.active_connections,
            "replica_count": metrics.replica_count,
            "custom_metrics": metrics.custom_metrics or {},
            "timestamp": metrics.timestamp or datetime.utcnow()
        }
        
        metrics_data.append(metrics_dict)
        
        # Keep only last 1000 entries
        if len(metrics_data) > 1000:
            metrics_data = metrics_data[-1000:]
        
        self._save_metrics(metrics_file, metrics_data)
        logger.info(f"Logged infrastructure metrics for deployment {deployment_id}")
    
    async def get_infrastructure_metrics(
        self,
        deployment_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation_window: Optional[timedelta] = None
    ) -> List[InfrastructureMetrics]:
        """Get historical infrastructure metrics."""
        metrics_file = self._get_metrics_file(deployment_id)
        metrics_data = self._load_metrics(metrics_file)
        
        # Apply time filters
        filtered_data = []
        for entry in metrics_data:
            timestamp = entry.get('timestamp')
            if not timestamp:
                continue
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            filtered_data.append(entry)
        
        # Convert back to InfrastructureMetrics objects
        infrastructure_metrics = []
        for entry in filtered_data:
            metrics = InfrastructureMetrics(
                cpu_usage=entry.get('cpu_usage'),
                memory_usage=entry.get('memory_usage'),
                gpu_usage=entry.get('gpu_usage'),
                disk_usage=entry.get('disk_usage'),
                network_io=entry.get('network_io'),
                request_queue_size=entry.get('request_queue_size'),
                active_connections=entry.get('active_connections'),
                replica_count=entry.get('replica_count'),
                custom_metrics=entry.get('custom_metrics', {}),
                timestamp=entry.get('timestamp')
            )
            infrastructure_metrics.append(metrics)
        
        logger.info(f"Retrieved {len(infrastructure_metrics)} infrastructure metrics for {deployment_id}")
        return infrastructure_metrics
    
    async def check_resource_utilization(
        self,
        deployment_id: str,
        threshold_config: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check resource utilization against thresholds."""
        # Get recent metrics (last 5 minutes)
        current_time = datetime.utcnow()
        start_time = current_time - timedelta(minutes=5)
        
        metrics = await self.get_infrastructure_metrics(deployment_id, start_time, current_time)
        
        if not metrics:
            return {
                "deployment_id": deployment_id,
                "error": "No recent metrics available",
                "check_timestamp": current_time.isoformat()
            }
        
        # Get latest metrics
        latest_metrics = metrics[-1]
        
        resource_status = {}
        overall_status = "healthy"
        recommendations = []
        
        # Check each resource type
        resource_checks = {
            "cpu": ("cpu_usage", "CPU usage"),
            "memory": ("memory_usage", "Memory usage"),
            "disk": ("disk_usage", "Disk usage"),
            "gpu": ("gpu_usage", "GPU usage")
        }
        
        for resource_key, (metric_attr, display_name) in resource_checks.items():
            current_value = getattr(latest_metrics, metric_attr)
            threshold = threshold_config.get(resource_key, 80.0)  # Default 80%
            
            if current_value is not None:
                status = "normal"
                if current_value > threshold * 1.2:  # 20% above threshold is critical
                    status = "critical"
                    overall_status = "unhealthy"
                elif current_value > threshold:
                    status = "warning"
                    if overall_status == "healthy":
                        overall_status = "degraded"
                
                resource_status[resource_key] = {
                    "current": current_value,
                    "threshold": threshold,
                    "status": status
                }
                
                if status != "normal":
                    recommendations.append(f"{display_name} is {status}: {current_value:.1f}% (threshold: {threshold}%)")
        
        if not recommendations:
            recommendations.append("All resource utilization within normal ranges")
        
        return {
            "deployment_id": deployment_id,
            "check_timestamp": current_time.isoformat(),
            "overall_status": overall_status,
            "resource_status": resource_status,
            "recommendations": recommendations
        }
    
    async def predict_resource_needs(
        self,
        deployment_id: str,
        forecast_horizon: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Predict future resource needs."""
        # Get historical metrics for trend analysis
        current_time = datetime.utcnow()
        lookback_period = timedelta(days=3)  # Look back 3 days for trend
        start_time = current_time - lookback_period
        
        metrics = await self.get_infrastructure_metrics(deployment_id, start_time, current_time)
        
        if len(metrics) < 10:
            return {
                "deployment_id": deployment_id,
                "error": "Insufficient historical data for prediction",
                "forecast_horizon_hours": forecast_horizon.total_seconds() / 3600
            }
        
        # Simple linear trend prediction
        predictions = {}
        
        resource_metrics = ["cpu_usage", "memory_usage", "request_queue_size"]
        
        for metric_name in resource_metrics:
            values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
            
            if len(values) >= 5:
                # Calculate simple trend
                recent_avg = statistics.mean(values[-5:])  # Last 5 values
                older_avg = statistics.mean(values[:5])    # First 5 values
                trend = recent_avg - older_avg
                
                # Project trend forward
                forecast_multiplier = forecast_horizon.total_seconds() / lookback_period.total_seconds()
                predicted_value = recent_avg + (trend * forecast_multiplier)
                
                predictions[metric_name] = {
                    "predicted_max": max(predicted_value * 1.2, recent_avg),  # 20% buffer
                    "predicted_avg": predicted_value,
                    "confidence": 0.7 if len(values) > 20 else 0.5  # Higher confidence with more data
                }
        
        # Generate scaling recommendations
        scaling_recommendations = {
            "recommended_replicas": 1,
            "scale_trigger_threshold": 0.8,
            "confidence": 0.6
        }
        
        # Simple heuristic: if CPU or memory predicted to exceed 80%, recommend scaling
        if "cpu_usage" in predictions and predictions["cpu_usage"]["predicted_max"] > 80:
            scaling_recommendations["recommended_replicas"] = 2
            scaling_recommendations["confidence"] = 0.8
        
        return {
            "deployment_id": deployment_id,
            "forecast_horizon_hours": forecast_horizon.total_seconds() / 3600,
            "predictions": predictions,
            "scaling_recommendations": scaling_recommendations
        }


class FileBasedDataQualityMonitoring(DataQualityMonitoringPort):
    """File-based implementation for data quality monitoring."""
    
    def __init__(self, storage_path: str = "./mlops_data/data_quality"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileBasedDataQualityMonitoring initialized at {storage_path}")
    
    def _get_metrics_file(self, model_id: str, deployment_id: Optional[str] = None) -> Path:
        """Get the metrics file path for a model/deployment."""
        key = f"{model_id}_{deployment_id or 'default'}"
        return self._storage_path / f"{key}_data_quality.json"
    
    def _load_metrics(self, metrics_file: Path) -> List[Dict[str, Any]]:
        """Load metrics from file."""
        if not metrics_file.exists():
            return []
        
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                # Convert timestamp strings back to datetime objects
                for entry in data:
                    if 'timestamp' in entry:
                        entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                    # Convert data_freshness back to timedelta
                    if 'data_freshness_seconds' in entry:
                        entry['data_freshness'] = timedelta(seconds=entry['data_freshness_seconds'])
                return data
        except Exception as e:
            logger.warning(f"Failed to load metrics from {metrics_file}: {e}")
            return []
    
    def _save_metrics(self, metrics_file: Path, metrics_data: List[Dict[str, Any]]) -> None:
        """Save metrics to file."""
        try:
            # Convert datetime and timedelta objects to serializable format
            serializable_data = []
            for entry in metrics_data:
                serializable_entry = entry.copy()
                if 'timestamp' in entry and isinstance(entry['timestamp'], datetime):
                    serializable_entry['timestamp'] = entry['timestamp'].isoformat()
                if 'data_freshness' in entry and isinstance(entry['data_freshness'], timedelta):
                    serializable_entry['data_freshness_seconds'] = entry['data_freshness'].total_seconds()
                    del serializable_entry['data_freshness']  # Remove original to avoid confusion
                serializable_data.append(serializable_entry)
            
            with open(metrics_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics to {metrics_file}: {e}")
    
    async def log_data_quality_metrics(
        self,
        model_id: str,
        deployment_id: Optional[str],
        metrics: DataQualityMetrics
    ) -> None:
        """Log data quality metrics."""
        metrics_file = self._get_metrics_file(model_id, deployment_id)
        metrics_data = self._load_metrics(metrics_file)
        
        # Convert DataQualityMetrics to dict
        metrics_dict = {
            "model_id": model_id,
            "deployment_id": deployment_id,
            "missing_values_ratio": metrics.missing_values_ratio,
            "outlier_ratio": metrics.outlier_ratio,
            "schema_violations": metrics.schema_violations,
            "data_freshness": metrics.data_freshness,
            "row_count": metrics.row_count,
            "column_count": metrics.column_count,
            "duplicate_ratio": metrics.duplicate_ratio,
            "data_type_violations": metrics.data_type_violations,
            "custom_checks": metrics.custom_checks or {},
            "timestamp": metrics.timestamp or datetime.utcnow()
        }
        
        metrics_data.append(metrics_dict)
        
        # Keep only last 1000 entries
        if len(metrics_data) > 1000:
            metrics_data = metrics_data[-1000:]
        
        self._save_metrics(metrics_file, metrics_data)
        logger.info(f"Logged data quality metrics for {model_id}")
    
    async def get_data_quality_metrics(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[DataQualityMetrics]:
        """Get historical data quality metrics."""
        metrics_file = self._get_metrics_file(model_id, deployment_id)
        metrics_data = self._load_metrics(metrics_file)
        
        # Apply time filters
        filtered_data = []
        for entry in metrics_data:
            timestamp = entry.get('timestamp')
            if not timestamp:
                continue
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            filtered_data.append(entry)
        
        # Convert back to DataQualityMetrics objects
        quality_metrics = []
        for entry in filtered_data:
            metrics = DataQualityMetrics(
                missing_values_ratio=entry.get('missing_values_ratio'),
                outlier_ratio=entry.get('outlier_ratio'),
                schema_violations=entry.get('schema_violations'),
                data_freshness=entry.get('data_freshness'),
                row_count=entry.get('row_count'),
                column_count=entry.get('column_count'),
                duplicate_ratio=entry.get('duplicate_ratio'),
                data_type_violations=entry.get('data_type_violations'),
                custom_checks=entry.get('custom_checks', {}),
                timestamp=entry.get('timestamp')
            )
            quality_metrics.append(metrics)
        
        logger.info(f"Retrieved {len(quality_metrics)} data quality metrics for {model_id}")
        return quality_metrics
    
    async def validate_input_data(
        self,
        model_id: str,
        data_sample: Dict[str, Any],
        schema_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate input data against expected schema."""
        validation_result = {
            "model_id": model_id,
            "validation_timestamp": datetime.utcnow().isoformat(),
            "overall_valid": True,
            "schema_validation": {
                "required_fields_present": True,
                "data_types_correct": True,
                "value_ranges_valid": True,
                "missing_fields": [],
                "invalid_types": [],
                "out_of_range_values": []
            },
            "data_quality_checks": {
                "missing_values": 0,
                "outliers_detected": 0,
                "duplicates_found": 0,
                "quality_score": 1.0
            },
            "recommendations": []
        }
        
        # Check required fields
        required_fields = schema_config.get("required_fields", [])
        for field in required_fields:
            if field not in data_sample:
                validation_result["schema_validation"]["missing_fields"].append(field)
                validation_result["schema_validation"]["required_fields_present"] = False
                validation_result["overall_valid"] = False
        
        # Check data types
        expected_types = schema_config.get("field_types", {})
        for field, expected_type in expected_types.items():
            if field in data_sample:
                actual_value = data_sample[field]
                if expected_type == "string" and not isinstance(actual_value, str):
                    validation_result["schema_validation"]["invalid_types"].append(f"{field}: expected string, got {type(actual_value).__name__}")
                    validation_result["schema_validation"]["data_types_correct"] = False
                    validation_result["overall_valid"] = False
                elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                    validation_result["schema_validation"]["invalid_types"].append(f"{field}: expected number, got {type(actual_value).__name__}")
                    validation_result["schema_validation"]["data_types_correct"] = False
                    validation_result["overall_valid"] = False
        
        # Check value ranges
        value_ranges = schema_config.get("value_ranges", {})
        for field, range_config in value_ranges.items():
            if field in data_sample:
                value = data_sample[field]
                if isinstance(value, (int, float)):
                    min_val = range_config.get("min")
                    max_val = range_config.get("max")
                    if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                        validation_result["schema_validation"]["out_of_range_values"].append(f"{field}: {value} not in range [{min_val}, {max_val}]")
                        validation_result["schema_validation"]["value_ranges_valid"] = False
                        validation_result["overall_valid"] = False
        
        # Check for missing values
        missing_count = sum(1 for v in data_sample.values() if v is None or v == "")
        validation_result["data_quality_checks"]["missing_values"] = missing_count
        
        # Simple outlier detection (values more than 3 std devs from mean)
        numeric_values = [v for v in data_sample.values() if isinstance(v, (int, float))]
        if len(numeric_values) > 2:
            mean_val = statistics.mean(numeric_values)
            std_val = statistics.stdev(numeric_values)
            outliers = [v for v in numeric_values if abs(v - mean_val) > 3 * std_val]
            validation_result["data_quality_checks"]["outliers_detected"] = len(outliers)
        
        # Calculate quality score
        issues = (
            len(validation_result["schema_validation"]["missing_fields"]) +
            len(validation_result["schema_validation"]["invalid_types"]) +
            len(validation_result["schema_validation"]["out_of_range_values"]) +
            validation_result["data_quality_checks"]["missing_values"] +
            validation_result["data_quality_checks"]["outliers_detected"]
        )
        
        total_fields = len(data_sample)
        quality_score = max(0.0, 1.0 - (issues / max(total_fields, 1)))
        validation_result["data_quality_checks"]["quality_score"] = quality_score
        
        # Generate recommendations
        if not validation_result["overall_valid"]:
            validation_result["recommendations"].append("Fix schema validation errors before processing")
        if quality_score < 0.8:
            validation_result["recommendations"].append("Review data quality - multiple issues detected")
        if not validation_result["recommendations"]:
            validation_result["recommendations"].append("Data validation passed successfully")
        
        logger.info(f"Validated input data for model {model_id}: valid={validation_result['overall_valid']}")
        return validation_result
    
    async def detect_data_quality_issues(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        check_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Detect data quality issues."""
        # Get recent data quality metrics
        current_time = datetime.utcnow()
        lookback_period = timedelta(hours=24)
        start_time = current_time - lookback_period
        
        metrics = await self.get_data_quality_metrics(model_id, deployment_id, start_time, current_time)
        
        if not metrics:
            logger.warning(f"No recent data quality metrics for {model_id}")
            return []
        
        issues = []
        thresholds = check_config or {
            "missing_values_threshold": 0.05,    # 5%
            "outlier_threshold": 0.1,            # 10%
            "schema_violations_threshold": 5,     # 5 violations
            "duplicate_threshold": 0.02           # 2%
        }
        
        # Check latest metrics against thresholds
        latest_metrics = metrics[-1]
        
        # Missing values check
        if latest_metrics.missing_values_ratio and latest_metrics.missing_values_ratio > thresholds.get("missing_values_threshold", 0.05):
            issues.append({
                "issue_id": f"quality_issue_{str(uuid4())[:8]}",
                "issue_type": "missing_values",
                "severity": "high" if latest_metrics.missing_values_ratio > 0.1 else "medium",
                "detected_at": current_time.isoformat(),
                "description": f"High missing values ratio: {latest_metrics.missing_values_ratio:.3f}",
                "current_value": latest_metrics.missing_values_ratio,
                "threshold": thresholds.get("missing_values_threshold", 0.05),
                "affected_features": ["multiple_features"],
                "impact_assessment": "May impact model prediction accuracy",
                "recommended_actions": [
                    "Investigate data source for missing value causes",
                    "Implement data imputation strategies",
                    "Review data collection process"
                ]
            })
        
        # Outliers check
        if latest_metrics.outlier_ratio and latest_metrics.outlier_ratio > thresholds.get("outlier_threshold", 0.1):
            issues.append({
                "issue_id": f"quality_issue_{str(uuid4())[:8]}",
                "issue_type": "outliers",
                "severity": "medium",
                "detected_at": current_time.isoformat(),
                "description": f"High outlier ratio: {latest_metrics.outlier_ratio:.3f}",
                "current_value": latest_metrics.outlier_ratio,
                "threshold": thresholds.get("outlier_threshold", 0.1),
                "affected_features": ["numeric_features"],
                "impact_assessment": "Outliers may skew model predictions",
                "recommended_actions": [
                    "Review outlier detection rules",
                    "Investigate data source changes",
                    "Consider outlier treatment strategies"
                ]
            })
        
        # Schema violations check
        if latest_metrics.schema_violations and latest_metrics.schema_violations > thresholds.get("schema_violations_threshold", 5):
            issues.append({
                "issue_id": f"quality_issue_{str(uuid4())[:8]}",
                "issue_type": "schema_drift",
                "severity": "high",
                "detected_at": current_time.isoformat(),
                "description": f"Schema violations detected: {latest_metrics.schema_violations}",
                "current_value": latest_metrics.schema_violations,
                "threshold": thresholds.get("schema_violations_threshold", 5),
                "affected_features": ["schema_fields"],
                "impact_assessment": "Schema changes may break model pipeline",
                "recommended_actions": [
                    "Review data schema changes",
                    "Update data validation rules",
                    "Coordinate with data source teams"
                ]
            })
        
        logger.info(f"Detected {len(issues)} data quality issues for {model_id}")
        return issues


class FileBasedDataDriftMonitoring(DataDriftMonitoringPort):
    """File-based implementation for data drift monitoring."""
    
    def __init__(self, storage_path: str = "./mlops_data/drift"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileBasedDataDriftMonitoring initialized at {storage_path}")
    
    def _get_drift_file(self, model_id: str, drift_type: str) -> Path:
        """Get the drift results file path."""
        return self._storage_path / f"{model_id}_{drift_type}_drift.json"
    
    def _load_drift_results(self, drift_file: Path) -> List[Dict[str, Any]]:
        """Load drift results from file."""
        if not drift_file.exists():
            return []
        
        try:
            with open(drift_file, 'r') as f:
                data = json.load(f)
                # Convert timestamp strings back to datetime objects
                for entry in data:
                    if 'timestamp' in entry:
                        entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                    # Convert period dictionaries back to datetime
                    for period_key in ['reference_period', 'detection_period']:
                        if period_key in entry:
                            period = entry[period_key]
                            if 'start' in period:
                                period['start'] = datetime.fromisoformat(period['start'])
                            if 'end' in period:
                                period['end'] = datetime.fromisoformat(period['end'])
                return data
        except Exception as e:
            logger.warning(f"Failed to load drift results from {drift_file}: {e}")
            return []
    
    def _save_drift_results(self, drift_file: Path, drift_data: List[Dict[str, Any]]) -> None:
        """Save drift results to file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = []
            for entry in drift_data:
                serializable_entry = entry.copy()
                if 'timestamp' in entry and isinstance(entry['timestamp'], datetime):
                    serializable_entry['timestamp'] = entry['timestamp'].isoformat()
                # Convert period datetime objects
                for period_key in ['reference_period', 'detection_period']:
                    if period_key in entry:
                        period = entry[period_key].copy()
                        if 'start' in period and isinstance(period['start'], datetime):
                            period['start'] = period['start'].isoformat()
                        if 'end' in period and isinstance(period['end'], datetime):
                            period['end'] = period['end'].isoformat()
                        serializable_entry[period_key] = period
                serializable_data.append(serializable_entry)
            
            with open(drift_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save drift results to {drift_file}: {e}")
    
    async def detect_feature_drift(
        self,
        model_id: str,
        reference_data: Dict[str, Any],
        current_data: Dict[str, Any],
        drift_config: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        """Detect feature drift."""
        import random  # Simple simulation for file-based implementation
        
        # Simple drift detection simulation
        # In a real implementation, this would use statistical tests like KS test, Chi-square, etc.
        
        drift_threshold = drift_config.get("drift_threshold", 0.1) if drift_config else 0.1
        
        # Calculate a simple "drift score" based on feature differences
        drift_score = random.uniform(0.0, 0.3)  # Simulate drift calculation
        is_drift_detected = drift_score > drift_threshold
        
        # Simulate affected features
        all_features = list(set(reference_data.keys()) | set(current_data.keys()))
        affected_features = []
        if is_drift_detected:
            # Randomly select some features as affected
            num_affected = min(len(all_features), random.randint(1, 3))
            affected_features = random.sample(all_features, num_affected)
        
        # Create result
        result = DriftDetectionResult(
            drift_type=DataDriftType.FEATURE_DRIFT,
            is_drift_detected=is_drift_detected,
            drift_score=drift_score,
            confidence=random.uniform(0.8, 0.95),
            affected_features=affected_features,
            reference_period={
                "start": datetime.utcnow() - timedelta(days=7),
                "end": datetime.utcnow() - timedelta(days=1)
            },
            detection_period={
                "start": datetime.utcnow() - timedelta(days=1),
                "end": datetime.utcnow()
            },
            statistical_tests={
                "kolmogorov_smirnov": {"statistic": random.uniform(0.1, 0.5), "p_value": random.uniform(0.01, 0.1)},
                "chi_square": {"statistic": random.uniform(10, 50), "p_value": random.uniform(0.01, 0.1)},
                "jensen_shannon_divergence": {"distance": random.uniform(0.1, 0.4)}
            },
            timestamp=datetime.utcnow()
        )
        
        # Store result
        drift_file = self._get_drift_file(model_id, "feature")
        drift_data = self._load_drift_results(drift_file)
        
        result_dict = {
            "drift_type": result.drift_type.value,
            "is_drift_detected": result.is_drift_detected,
            "drift_score": result.drift_score,
            "confidence": result.confidence,
            "affected_features": result.affected_features,
            "reference_period": result.reference_period,
            "detection_period": result.detection_period,
            "statistical_tests": result.statistical_tests,
            "timestamp": result.timestamp
        }
        
        drift_data.append(result_dict)
        
        # Keep only last 100 results
        if len(drift_data) > 100:
            drift_data = drift_data[-100:]
        
        self._save_drift_results(drift_file, drift_data)
        
        logger.info(f"Detected feature drift for model {model_id}: drift={is_drift_detected}, score={drift_score:.3f}")
        return result
    
    async def detect_target_drift(
        self,
        model_id: str,
        reference_targets: List[Any],
        current_targets: List[Any],
        drift_config: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        """Detect target drift."""
        import random
        
        drift_threshold = drift_config.get("drift_threshold", 0.1) if drift_config else 0.1
        
        # Simple target drift simulation
        drift_score = random.uniform(0.0, 0.25)
        is_drift_detected = drift_score > drift_threshold
        
        result = DriftDetectionResult(
            drift_type=DataDriftType.TARGET_DRIFT,
            is_drift_detected=is_drift_detected,
            drift_score=drift_score,
            confidence=random.uniform(0.85, 0.98),
            affected_features=["target"],
            reference_period={
                "start": datetime.utcnow() - timedelta(days=7),
                "end": datetime.utcnow() - timedelta(days=1)
            },
            detection_period={
                "start": datetime.utcnow() - timedelta(days=1),
                "end": datetime.utcnow()
            },
            statistical_tests={
                "population_stability_index": {"psi": random.uniform(0.1, 0.25)},
                "wasserstein_distance": {"distance": random.uniform(0.1, 0.5)}
            },
            timestamp=datetime.utcnow()
        )
        
        # Store result
        drift_file = self._get_drift_file(model_id, "target")
        drift_data = self._load_drift_results(drift_file)
        
        result_dict = {
            "drift_type": result.drift_type.value,
            "is_drift_detected": result.is_drift_detected,
            "drift_score": result.drift_score,
            "confidence": result.confidence,
            "affected_features": result.affected_features,
            "reference_period": result.reference_period,
            "detection_period": result.detection_period,
            "statistical_tests": result.statistical_tests,
            "timestamp": result.timestamp
        }
        
        drift_data.append(result_dict)
        
        if len(drift_data) > 100:
            drift_data = drift_data[-100:]
        
        self._save_drift_results(drift_file, drift_data)
        
        logger.info(f"Detected target drift for model {model_id}: drift={is_drift_detected}, score={drift_score:.3f}")
        return result
    
    async def detect_prediction_drift(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        reference_period: timedelta = timedelta(days=7),
        comparison_period: timedelta = timedelta(days=1),
        drift_config: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        """Detect prediction drift."""
        import random
        
        drift_threshold = drift_config.get("drift_threshold", 0.1) if drift_config else 0.1
        
        # Simple prediction drift simulation
        drift_score = random.uniform(0.0, 0.2)
        is_drift_detected = drift_score > drift_threshold
        
        result = DriftDetectionResult(
            drift_type=DataDriftType.PREDICTION_DRIFT,
            is_drift_detected=is_drift_detected,
            drift_score=drift_score,
            confidence=random.uniform(0.8, 0.93),
            affected_features=["predictions"],
            reference_period={
                "start": datetime.utcnow() - reference_period - comparison_period,
                "end": datetime.utcnow() - comparison_period
            },
            detection_period={
                "start": datetime.utcnow() - comparison_period,
                "end": datetime.utcnow()
            },
            statistical_tests={
                "prediction_stability": {"stability_score": random.uniform(0.7, 0.95)},
                "output_distribution": {"kl_divergence": random.uniform(0.1, 0.4)}
            },
            timestamp=datetime.utcnow()
        )
        
        # Store result
        drift_file = self._get_drift_file(model_id, "prediction")
        drift_data = self._load_drift_results(drift_file)
        
        result_dict = {
            "drift_type": result.drift_type.value,
            "is_drift_detected": result.is_drift_detected,
            "drift_score": result.drift_score,
            "confidence": result.confidence,
            "affected_features": result.affected_features,
            "reference_period": result.reference_period,
            "detection_period": result.detection_period,
            "statistical_tests": result.statistical_tests,
            "timestamp": result.timestamp
        }
        
        drift_data.append(result_dict)
        
        if len(drift_data) > 100:
            drift_data = drift_data[-100:]
        
        self._save_drift_results(drift_file, drift_data)
        
        logger.info(f"Detected prediction drift for model {model_id}: drift={is_drift_detected}, score={drift_score:.3f}")
        return result
    
    async def get_drift_history(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        drift_type: Optional[DataDriftType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[DriftDetectionResult]:
        """Get historical drift detection results."""
        all_results = []
        
        # Determine which drift types to include
        if drift_type:
            drift_types = [drift_type]
        else:
            drift_types = [DataDriftType.FEATURE_DRIFT, DataDriftType.TARGET_DRIFT, DataDriftType.PREDICTION_DRIFT]
        
        # Load results for each drift type
        for dt in drift_types:
            drift_file = self._get_drift_file(model_id, dt.value.split('_')[0])  # feature, target, prediction
            drift_data = self._load_drift_results(drift_file)
            
            for entry in drift_data:
                # Apply time filters
                timestamp = entry.get('timestamp')
                if timestamp:
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                
                # Convert back to DriftDetectionResult
                result = DriftDetectionResult(
                    drift_type=DataDriftType(entry['drift_type']),
                    is_drift_detected=entry['is_drift_detected'],
                    drift_score=entry['drift_score'],
                    confidence=entry['confidence'],
                    affected_features=entry['affected_features'],
                    reference_period=entry['reference_period'],
                    detection_period=entry['detection_period'],
                    statistical_tests=entry['statistical_tests'],
                    timestamp=entry['timestamp']
                )
                all_results.append(result)
        
        # Sort by timestamp
        all_results.sort(key=lambda x: x.timestamp or datetime.min)
        
        logger.info(f"Retrieved {len(all_results)} drift detection results for {model_id}")
        return all_results


class FileBasedAlerting(AlertingPort):
    """File-based implementation for alerting."""
    
    def __init__(self, storage_path: str = "./mlops_data/alerts"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._rules_file = self._storage_path / "monitoring_rules.json"
        self._alerts_file = self._storage_path / "alerts.json"
        self._rules = self._load_rules()
        self._alerts = self._load_alerts()
        logger.info(f"FileBasedAlerting initialized at {storage_path}")
    
    def _load_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load monitoring rules from file."""
        if not self._rules_file.exists():
            return {}
        
        try:
            with open(self._rules_file, 'r') as f:
                data = json.load(f)
                # Convert timedelta strings back to timedelta objects
                for rule_data in data.values():
                    if 'evaluation_window_seconds' in rule_data:
                        rule_data['evaluation_window'] = timedelta(seconds=rule_data['evaluation_window_seconds'])
                    if 'alert_frequency_seconds' in rule_data:
                        rule_data['alert_frequency'] = timedelta(seconds=rule_data['alert_frequency_seconds'])
                return data
        except Exception as e:
            logger.warning(f"Failed to load rules from {self._rules_file}: {e}")
            return {}
    
    def _save_rules(self) -> None:
        """Save monitoring rules to file."""
        try:
            # Convert timedelta objects to seconds for JSON serialization
            serializable_data = {}
            for rule_id, rule_data in self._rules.items():
                serializable_rule = rule_data.copy()
                if 'evaluation_window' in rule_data and isinstance(rule_data['evaluation_window'], timedelta):
                    serializable_rule['evaluation_window_seconds'] = rule_data['evaluation_window'].total_seconds()
                    del serializable_rule['evaluation_window']
                if 'alert_frequency' in rule_data and isinstance(rule_data['alert_frequency'], timedelta):
                    serializable_rule['alert_frequency_seconds'] = rule_data['alert_frequency'].total_seconds()
                    del serializable_rule['alert_frequency']
                serializable_data[rule_id] = serializable_rule
            
            with open(self._rules_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rules to {self._rules_file}: {e}")
    
    def _load_alerts(self) -> Dict[str, Dict[str, Any]]:
        """Load alerts from file."""
        if not self._alerts_file.exists():
            return {}
        
        try:
            with open(self._alerts_file, 'r') as f:
                data = json.load(f)
                # Convert timestamp strings back to datetime objects
                for alert_data in data.values():
                    for timestamp_field in ['triggered_at', 'acknowledged_at', 'resolved_at']:
                        if timestamp_field in alert_data and alert_data[timestamp_field]:
                            alert_data[timestamp_field] = datetime.fromisoformat(alert_data[timestamp_field])
                return data
        except Exception as e:
            logger.warning(f"Failed to load alerts from {self._alerts_file}: {e}")
            return {}
    
    def _save_alerts(self) -> None:
        """Save alerts to file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = {}
            for alert_id, alert_data in self._alerts.items():
                serializable_alert = alert_data.copy()
                for timestamp_field in ['triggered_at', 'acknowledged_at', 'resolved_at']:
                    if timestamp_field in alert_data and isinstance(alert_data[timestamp_field], datetime):
                        serializable_alert[timestamp_field] = alert_data[timestamp_field].isoformat()
                    elif timestamp_field in alert_data and alert_data[timestamp_field] is None:
                        serializable_alert[timestamp_field] = None
                serializable_data[alert_id] = serializable_alert
            
            with open(self._alerts_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alerts to {self._alerts_file}: {e}")
    
    async def create_monitoring_rule(self, rule: MonitoringRule) -> str:
        """Create a new monitoring rule."""
        # Convert MonitoringRule to dict for storage
        rule_dict = {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "description": rule.description,
            "model_id": rule.model_id,
            "deployment_id": rule.deployment_id,
            "thresholds": [
                {
                    "metric_name": t.metric_name,
                    "operator": t.operator,
                    "threshold_value": t.threshold_value,
                    "severity": t.severity.value,
                    "description": t.description
                }
                for t in rule.thresholds
            ],
            "evaluation_window": rule.evaluation_window,
            "alert_frequency": rule.alert_frequency,
            "enabled": rule.enabled,
            "tags": rule.tags
        }
        
        self._rules[rule.rule_id] = rule_dict
        self._save_rules()
        
        logger.info(f"Created monitoring rule {rule.rule_id}")
        return rule.rule_id
    
    async def get_monitoring_rule(self, rule_id: str) -> Optional[MonitoringRule]:
        """Get monitoring rule configuration."""
        rule_data = self._rules.get(rule_id)
        if not rule_data:
            return None
        
        # Convert back to MonitoringRule object
        from mlops.domain.interfaces.mlops_monitoring_operations import MetricThreshold
        
        thresholds = []
        for t_data in rule_data.get("thresholds", []):
            threshold = MetricThreshold(
                metric_name=t_data["metric_name"],
                operator=t_data["operator"],
                threshold_value=t_data["threshold_value"],
                severity=MonitoringAlertSeverity(t_data["severity"]),
                description=t_data["description"]
            )
            thresholds.append(threshold)
        
        rule = MonitoringRule(
            rule_id=rule_data["rule_id"],
            name=rule_data["name"],
            description=rule_data["description"],
            model_id=rule_data["model_id"],
            deployment_id=rule_data["deployment_id"],
            thresholds=thresholds,
            evaluation_window=rule_data.get("evaluation_window", timedelta(minutes=5)),
            alert_frequency=rule_data.get("alert_frequency", timedelta(minutes=15)),
            enabled=rule_data["enabled"],
            tags=rule_data["tags"]
        )
        
        return rule
    
    async def update_monitoring_rule(
        self,
        rule_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update monitoring rule configuration."""
        if rule_id not in self._rules:
            return False
        
        rule_data = self._rules[rule_id]
        
        # Update allowed fields
        allowed_updates = ["name", "description", "enabled", "tags"]
        for key, value in updates.items():
            if key in allowed_updates:
                rule_data[key] = value
        
        self._save_rules()
        logger.info(f"Updated monitoring rule {rule_id}")
        return True
    
    async def delete_monitoring_rule(self, rule_id: str) -> bool:
        """Delete monitoring rule."""
        if rule_id not in self._rules:
            return False
        
        del self._rules[rule_id]
        self._save_rules()
        
        logger.info(f"Deleted monitoring rule {rule_id}")
        return True
    
    async def trigger_alert(
        self,
        rule_id: str,
        alert_data: Dict[str, Any]
    ) -> str:
        """Trigger a monitoring alert."""
        alert_id = f"alert_{str(uuid4())[:8]}"
        
        # Create alert dict for storage
        alert_dict = {
            "alert_id": alert_id,
            "rule_id": rule_id,
            "severity": alert_data.get("severity", "medium"),
            "status": MonitoringAlertStatus.ACTIVE.value,
            "title": alert_data.get("title", "Monitoring Alert"),
            "description": alert_data.get("description", "Alert triggered by monitoring rule"),
            "model_id": alert_data.get("model_id"),
            "deployment_id": alert_data.get("deployment_id"),
            "triggered_at": datetime.utcnow(),
            "acknowledged_at": None,
            "resolved_at": None,
            "metadata": alert_data.get("metadata", {}),
            "remediation_suggestions": alert_data.get("remediation_suggestions", [])
        }
        
        self._alerts[alert_id] = alert_dict
        self._save_alerts()
        
        logger.info(f"Triggered alert {alert_id} for rule {rule_id}")
        return alert_id
    
    async def get_alert(self, alert_id: str) -> Optional[MonitoringAlert]:
        """Get alert information."""
        alert_data = self._alerts.get(alert_id)
        if not alert_data:
            return None
        
        # Convert back to MonitoringAlert object
        alert = MonitoringAlert(
            alert_id=alert_data["alert_id"],
            rule_id=alert_data["rule_id"],
            severity=MonitoringAlertSeverity(alert_data["severity"]),
            status=MonitoringAlertStatus(alert_data["status"]),
            title=alert_data["title"],
            description=alert_data["description"],
            model_id=alert_data["model_id"],
            deployment_id=alert_data["deployment_id"],
            triggered_at=alert_data["triggered_at"],
            acknowledged_at=alert_data["acknowledged_at"],
            resolved_at=alert_data["resolved_at"],
            metadata=alert_data["metadata"],
            remediation_suggestions=alert_data["remediation_suggestions"]
        )
        
        return alert
    
    async def list_alerts(
        self,
        model_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        severity: Optional[MonitoringAlertSeverity] = None,
        status: Optional[MonitoringAlertStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[MonitoringAlert]:
        """List alerts with optional filters."""
        alerts = []
        
        for alert_data in self._alerts.values():
            # Apply filters
            if model_id and alert_data.get("model_id") != model_id:
                continue
            if deployment_id and alert_data.get("deployment_id") != deployment_id:
                continue
            if severity and alert_data["severity"] != severity.value:
                continue
            if status and alert_data["status"] != status.value:
                continue
            if start_time and alert_data["triggered_at"] < start_time:
                continue
            if end_time and alert_data["triggered_at"] > end_time:
                continue
            
            # Convert to MonitoringAlert object
            alert = MonitoringAlert(
                alert_id=alert_data["alert_id"],
                rule_id=alert_data["rule_id"],
                severity=MonitoringAlertSeverity(alert_data["severity"]),
                status=MonitoringAlertStatus(alert_data["status"]),
                title=alert_data["title"],
                description=alert_data["description"],
                model_id=alert_data["model_id"],
                deployment_id=alert_data["deployment_id"],
                triggered_at=alert_data["triggered_at"],
                acknowledged_at=alert_data["acknowledged_at"],
                resolved_at=alert_data["resolved_at"],
                metadata=alert_data["metadata"],
                remediation_suggestions=alert_data["remediation_suggestions"]
            )
            alerts.append(alert)
        
        # Sort by triggered_at descending
        alerts.sort(key=lambda x: x.triggered_at, reverse=True)
        
        # Apply pagination
        paginated_alerts = alerts[offset:offset + limit]
        
        logger.info(f"Listed {len(paginated_alerts)} alerts")
        return paginated_alerts
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        acknowledgment_note: Optional[str] = None
    ) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self._alerts:
            return False
        
        alert_data = self._alerts[alert_id]
        alert_data["status"] = MonitoringAlertStatus.ACKNOWLEDGED.value
        alert_data["acknowledged_at"] = datetime.utcnow()
        alert_data["acknowledged_by"] = acknowledged_by
        if acknowledgment_note:
            alert_data["acknowledgment_note"] = acknowledgment_note
        
        self._save_alerts()
        logger.info(f"Acknowledged alert {alert_id} by {acknowledged_by}")
        return True
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_note: Optional[str] = None
    ) -> bool:
        """Resolve an alert."""
        if alert_id not in self._alerts:
            return False
        
        alert_data = self._alerts[alert_id]
        alert_data["status"] = MonitoringAlertStatus.RESOLVED.value
        alert_data["resolved_at"] = datetime.utcnow()
        alert_data["resolved_by"] = resolved_by
        if resolution_note:
            alert_data["resolution_note"] = resolution_note
        
        self._save_alerts()
        logger.info(f"Resolved alert {alert_id} by {resolved_by}")
        return True


class FileBasedHealthCheck(HealthCheckPort):
    """File-based implementation for health checks."""
    
    def __init__(self, storage_path: str = "./mlops_data/health"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileBasedHealthCheck initialized at {storage_path}")
    
    async def check_model_health(
        self,
        model_id: str,
        deployment_id: Optional[str] = None
    ) -> ModelHealthReport:
        """Perform comprehensive model health check."""
        import random
        
        # Generate simulated health report
        overall_health = random.choice([
            ModelHealthStatus.HEALTHY,
            ModelHealthStatus.HEALTHY,  # Higher chance of healthy
            ModelHealthStatus.DEGRADED,
            ModelHealthStatus.UNHEALTHY
        ])
        
        # Generate component health statuses
        performance_health = random.choice(list(ModelHealthStatus))
        infrastructure_health = random.choice(list(ModelHealthStatus))
        data_quality_health = random.choice(list(ModelHealthStatus))
        drift_health = random.choice(list(ModelHealthStatus))
        
        # Generate performance metrics
        performance_metrics = PerformanceMetrics(
            accuracy=random.uniform(0.75, 0.95),
            precision=random.uniform(0.75, 0.95),
            recall=random.uniform(0.75, 0.95),
            f1_score=random.uniform(0.75, 0.95),
            latency_p95=random.uniform(80, 250),
            throughput=random.uniform(40, 120),
            error_rate=random.uniform(0.001, 0.02),
            timestamp=datetime.utcnow()
        )
        
        # Generate infrastructure metrics
        infrastructure_metrics = InfrastructureMetrics(
            cpu_usage=random.uniform(20, 85),
            memory_usage=random.uniform(30, 90),
            gpu_usage=random.uniform(0, 70),
            replica_count=random.randint(1, 5),
            timestamp=datetime.utcnow()
        )
        
        # Generate data quality metrics
        data_quality_metrics = DataQualityMetrics(
            missing_values_ratio=random.uniform(0, 0.03),
            outlier_ratio=random.uniform(0.01, 0.08),
            schema_violations=random.randint(0, 3),
            row_count=random.randint(1000, 10000),
            column_count=random.randint(5, 25),
            timestamp=datetime.utcnow()
        )
        
        # Generate recommendations based on health status
        recommendations = []
        if overall_health == ModelHealthStatus.UNHEALTHY:
            recommendations.extend([
                "Model requires immediate attention",
                "Review performance and infrastructure metrics",
                "Consider model retraining or rollback"
            ])
        elif overall_health == ModelHealthStatus.DEGRADED:
            recommendations.extend([
                "Monitor model performance closely",
                "Investigate potential issues",
                "Plan for maintenance window"
            ])
        else:
            recommendations.append("Model is operating within expected parameters")
        
        # Add specific recommendations based on metrics
        if performance_metrics.accuracy and performance_metrics.accuracy < 0.8:
            recommendations.append("Low accuracy detected - consider model retraining")
        
        if infrastructure_metrics.cpu_usage and infrastructure_metrics.cpu_usage > 80:
            recommendations.append("High CPU usage - consider scaling or optimization")
        
        if infrastructure_metrics.memory_usage and infrastructure_metrics.memory_usage > 85:
            recommendations.append("High memory usage - investigate memory leaks")
        
        report = ModelHealthReport(
            model_id=model_id,
            deployment_id=deployment_id,
            overall_health=overall_health,
            performance_health=performance_health,
            infrastructure_health=infrastructure_health,
            data_quality_health=data_quality_health,
            drift_health=drift_health,
            recent_alerts=[],  # Would be populated with actual alerts
            performance_metrics=performance_metrics,
            infrastructure_metrics=infrastructure_metrics,
            data_quality_metrics=data_quality_metrics,
            drift_results=[],  # Would be populated with actual drift results
            recommendations=recommendations,
            report_timestamp=datetime.utcnow()
        )
        
        # Save report to file for history
        report_file = self._storage_path / f"{model_id}_health_reports.json"
        reports_data = []
        
        if report_file.exists():
            try:
                with open(report_file, 'r') as f:
                    reports_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing health reports: {e}")
        
        # Convert report to dict for storage
        report_dict = {
            "model_id": model_id,
            "deployment_id": deployment_id,
            "overall_health": overall_health.value,
            "performance_health": performance_health.value,
            "infrastructure_health": infrastructure_health.value,
            "data_quality_health": data_quality_health.value,
            "drift_health": drift_health.value,
            "recommendations": recommendations,
            "report_timestamp": datetime.utcnow().isoformat()
        }
        
        reports_data.append(report_dict)
        
        # Keep only last 50 reports
        if len(reports_data) > 50:
            reports_data = reports_data[-50:]
        
        try:
            with open(report_file, 'w') as f:
                json.dump(reports_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save health report: {e}")
        
        logger.info(f"Generated health report for model {model_id} (status: {overall_health.value})")
        return report
    
    async def check_deployment_health(
        self,
        deployment_id: str
    ) -> Dict[str, Any]:
        """Check deployment health status."""
        import random
        
        overall_status = random.choice(["healthy", "degraded", "unhealthy"])
        
        health_status = {
            "deployment_id": deployment_id,
            "overall_status": overall_status,
            "check_timestamp": datetime.utcnow().isoformat(),
            "components": {
                "api_endpoint": {
                    "status": random.choice(["healthy", "degraded"]),
                    "response_time_ms": random.uniform(50, 200),
                    "success_rate": random.uniform(0.95, 1.0)
                },
                "model_loading": {
                    "status": "healthy",
                    "load_time_ms": random.uniform(1000, 5000)
                },
                "prediction_service": {
                    "status": random.choice(["healthy", "degraded"]),
                    "average_latency_ms": random.uniform(80, 250),
                    "throughput_rps": random.uniform(10, 100)
                },
                "resource_usage": {
                    "status": random.choice(["healthy", "warning"]),
                    "cpu_usage": random.uniform(20, 85),
                    "memory_usage": random.uniform(30, 90)
                }
            },
            "metrics": {
                "uptime_seconds": random.randint(3600, 86400 * 7),
                "total_requests": random.randint(1000, 100000),
                "successful_requests": random.randint(950, 99500),
                "average_response_time": random.uniform(80, 200)
            },
            "recommendations": []
        }
        
        # Add recommendations based on status
        if overall_status == "unhealthy":
            health_status["recommendations"].extend([
                "Deployment requires immediate attention",
                "Check component health status",
                "Consider restarting deployment"
            ])
        elif overall_status == "degraded":
            health_status["recommendations"].extend([
                "Monitor deployment closely",
                "Investigate performance issues"
            ])
        else:
            health_status["recommendations"].append("Deployment is operating normally")
        
        logger.info(f"Checked deployment health for {deployment_id}: {overall_status}")
        return health_status
    
    async def run_health_diagnostics(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        diagnostic_tests: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run detailed health diagnostics."""
        import random
        
        default_tests = [
            "model_availability",
            "prediction_accuracy",
            "response_time",
            "resource_utilization",
            "error_rate",
            "data_quality",
            "drift_detection"
        ]
        
        tests_to_run = diagnostic_tests or default_tests
        results = {}
        
        for test in tests_to_run:
            if test == "model_availability":
                results[test] = {
                    "status": "pass",
                    "details": "Model is loaded and accessible",
                    "latency_ms": random.uniform(10, 50)
                }
            elif test == "prediction_accuracy":
                accuracy = random.uniform(0.75, 0.95)
                results[test] = {
                    "status": "pass" if accuracy > 0.8 else "warning",
                    "details": f"Current accuracy: {accuracy:.3f}",
                    "value": accuracy
                }
            elif test == "response_time":
                response_time = random.uniform(50, 300)
                results[test] = {
                    "status": "pass" if response_time < 200 else "warning",
                    "details": f"Average response time: {response_time:.1f}ms",
                    "value": response_time
                }
            elif test == "resource_utilization":
                cpu_usage = random.uniform(20, 90)
                results[test] = {
                    "status": "pass" if cpu_usage < 80 else "warning",
                    "details": f"CPU usage: {cpu_usage:.1f}%",
                    "value": cpu_usage
                }
            elif test == "error_rate":
                error_rate = random.uniform(0.001, 0.05)
                results[test] = {
                    "status": "pass" if error_rate < 0.02 else "warning",
                    "details": f"Error rate: {error_rate:.3f}",
                    "value": error_rate
                }
            else:
                # Generic test result
                results[test] = {
                    "status": random.choice(["pass", "warning"]),
                    "details": f"{test.replace('_', ' ').title()} check completed"
                }
        
        # Calculate overall score
        passed_tests = sum(1 for result in results.values() if result["status"] == "pass")
        overall_score = passed_tests / len(results)
        
        diagnostics = {
            "model_id": model_id,
            "deployment_id": deployment_id,
            "diagnostic_timestamp": datetime.utcnow().isoformat(),
            "tests_run": tests_to_run,
            "results": results,
            "overall_score": overall_score,
            "recommendations": []
        }
        
        # Generate recommendations
        if overall_score >= 0.9:
            diagnostics["recommendations"].append("All diagnostic tests passed - model is healthy")
        elif overall_score >= 0.7:
            diagnostics["recommendations"].extend([
                "Most diagnostic tests passed",
                "Monitor warning conditions closely"
            ])
        else:
            diagnostics["recommendations"].extend([
                "Multiple diagnostic tests failed",
                "Model requires immediate attention"
            ])
        
        logger.info(f"Ran health diagnostics for model {model_id}: score={overall_score:.2f}")
        return diagnostics
    
    async def get_health_history(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ModelHealthReport]:
        """Get historical health reports."""
        report_file = self._storage_path / f"{model_id}_health_reports.json"
        
        if not report_file.exists():
            return []
        
        try:
            with open(report_file, 'r') as f:
                reports_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load health history: {e}")
            return []
        
        # Convert back to ModelHealthReport objects (simplified)
        history = []
        for report_data in reports_data:
            # Apply time filters
            report_time = datetime.fromisoformat(report_data["report_timestamp"])
            if start_time and report_time < start_time:
                continue
            if end_time and report_time > end_time:
                continue
            
            # Create simplified health report for history
            report = ModelHealthReport(
                model_id=report_data["model_id"],
                deployment_id=report_data.get("deployment_id"),
                overall_health=ModelHealthStatus(report_data["overall_health"]),
                performance_health=ModelHealthStatus(report_data["performance_health"]),
                infrastructure_health=ModelHealthStatus(report_data["infrastructure_health"]),
                data_quality_health=ModelHealthStatus(report_data["data_quality_health"]),
                drift_health=ModelHealthStatus(report_data["drift_health"]),
                recent_alerts=[],
                performance_metrics=None,  # Would need to store full metrics
                infrastructure_metrics=None,
                data_quality_metrics=None,
                drift_results=[],
                recommendations=report_data["recommendations"],
                report_timestamp=report_time
            )
            history.append(report)
        
        logger.info(f"Retrieved {len(history)} historical health reports for {model_id}")
        return history
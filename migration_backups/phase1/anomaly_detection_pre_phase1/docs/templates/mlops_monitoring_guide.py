#!/usr/bin/env python3
"""
MLOps and Production Monitoring Guide
=====================================

Complete guide for MLOps practices, model monitoring, drift detection,
and automated retraining for anomaly detection systems in production.

Usage:
    python mlops_monitoring_guide.py

Requirements:
    - anomaly_detection
    - prometheus_client (optional)
    - mlflow (optional)
    - wandb (optional)
    - evidently (optional)
"""

import os
import json
import time
import logging
import threading
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import schedule

import numpy as np
import pandas as pd

from anomaly_detection import DetectionService, EnsembleService
from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.infrastructure.monitoring.metrics_collector import MetricsCollector


@dataclass
class ModelMetrics:
    """Metrics for model performance monitoring."""
    
    model_id: str
    timestamp: datetime
    predictions_count: int
    anomaly_rate: float
    processing_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    drift_score: Optional[float] = None


@dataclass
class AlertConfig:
    """Configuration for monitoring alerts."""
    
    anomaly_rate_threshold: Tuple[float, float] = (0.01, 0.5)  # Min, Max
    processing_time_threshold: float = 5000.0  # milliseconds
    memory_threshold: float = 2000.0  # MB
    drift_threshold: float = 0.3
    error_rate_threshold: float = 0.05
    alert_cooldown_minutes: int = 30


class MLOpsMonitor:
    """Comprehensive MLOps monitoring system for anomaly detection."""
    
    def __init__(self, model_registry_path: str = "model_registry"):
        self.model_registry_path = Path(model_registry_path)
        self.model_registry_path.mkdir(exist_ok=True)
        
        # Services
        self.detection_service = DetectionService()
        self.metrics_collector = MetricsCollector()
        
        # Monitoring state
        self.active_models: Dict[str, Any] = {}
        self.model_metrics: List[ModelMetrics] = []
        self.alert_config = AlertConfig()
        self.last_alerts: Dict[str, datetime] = {}
        
        # Background threads
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        
        self._setup_logging()
        self._setup_prometheus_metrics()
        self._load_active_models()
    
    def _setup_logging(self):
        """Setup structured logging for MLOps monitoring."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('mlops_monitor.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("MLOps monitor initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring."""
        try:
            from prometheus_client import Counter, Histogram, Gauge, start_http_server
            
            # Model performance metrics
            self.prediction_counter = Counter(
                'anomaly_predictions_total',
                'Total number of predictions made',
                ['model_id', 'algorithm']
            )
            
            self.anomaly_counter = Counter(
                'anomalies_detected_total',
                'Total number of anomalies detected',
                ['model_id', 'algorithm']
            )
            
            self.processing_time_histogram = Histogram(
                'prediction_processing_seconds',
                'Time spent processing predictions',
                ['model_id', 'algorithm']
            )
            
            self.model_accuracy_gauge = Gauge(
                'model_accuracy',
                'Current model accuracy',
                ['model_id', 'algorithm']
            )
            
            self.drift_score_gauge = Gauge(
                'model_drift_score',
                'Model drift score (0-1)',
                ['model_id', 'algorithm']
            )
            
            # System metrics
            self.active_models_gauge = Gauge('active_models_count', 'Number of active models')
            self.memory_usage_gauge = Gauge('memory_usage_mb', 'Memory usage in MB')
            
            # Start Prometheus server
            start_http_server(9090)
            self.logger.info("Prometheus metrics server started on port 9090")
            
        except ImportError:
            self.logger.warning("Prometheus client not available")
            self.prometheus_enabled = False
        else:
            self.prometheus_enabled = True
    
    def _load_active_models(self):
        """Load active models from registry."""
        registry_file = self.model_registry_path / "active_models.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    model_configs = json.load(f)
                
                for model_id, config in model_configs.items():
                    self.active_models[model_id] = config
                    self.logger.info(f"Loaded active model: {model_id}")
                    
            except Exception as e:
                self.logger.error(f"Error loading model registry: {e}")
    
    def register_model(
        self,
        model_id: str,
        algorithm: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a new model for monitoring.
        
        Args:
            model_id: Unique identifier for the model
            algorithm: Algorithm name
            model_path: Path to the saved model
            metadata: Additional model metadata
        """
        model_config = {
            'model_id': model_id,
            'algorithm': algorithm,
            'model_path': model_path,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'status': 'active'
        }
        
        self.active_models[model_id] = model_config
        
        # Save to registry
        registry_file = self.model_registry_path / "active_models.json"
        with open(registry_file, 'w') as f:
            json.dump(self.active_models, f, indent=2, default=str)
        
        self.logger.info(f"Registered model {model_id} with algorithm {algorithm}")
    
    def predict_with_monitoring(
        self,
        model_id: str,
        data: np.ndarray,
        ground_truth: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Make predictions with comprehensive monitoring.
        
        Args:
            model_id: ID of the model to use
            data: Input data for prediction
            ground_truth: True labels (if available) for accuracy calculation
            
        Returns:
            Dictionary containing predictions and monitoring metrics
        """
        if model_id not in self.active_models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_config = self.active_models[model_id]
        algorithm = model_config['algorithm']
        
        # Start timing
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            # Make prediction
            result = self.detection_service.detect_anomalies(
                data=data,
                algorithm=algorithm,
                contamination=model_config.get('contamination', 0.1)
            )
            
            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000
            
            # Calculate metrics
            anomaly_rate = np.mean(result.predictions == -1)
            memory_usage = self._get_memory_usage()
            
            # Calculate accuracy if ground truth provided
            accuracy = precision = recall = f1_score = None
            if ground_truth is not None:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as f1
                
                y_true = (ground_truth == -1).astype(int)
                y_pred = (result.predictions == -1).astype(int)
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1_score = f1(y_true, y_pred, zero_division=0)
            
            # Create metrics record
            metrics = ModelMetrics(
                model_id=model_id,
                timestamp=datetime.now(),
                predictions_count=len(data),
                anomaly_rate=anomaly_rate,
                processing_time_ms=processing_time_ms,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=self._get_cpu_usage(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score
            )
            
            # Store metrics
            self.model_metrics.append(metrics)
            
            # Update Prometheus metrics
            if self.prometheus_enabled:
                self.prediction_counter.labels(model_id=model_id, algorithm=algorithm).inc(len(data))
                self.anomaly_counter.labels(model_id=model_id, algorithm=algorithm).inc(np.sum(result.predictions == -1))
                self.processing_time_histogram.labels(model_id=model_id, algorithm=algorithm).observe(processing_time_ms / 1000)
                
                if accuracy is not None:
                    self.model_accuracy_gauge.labels(model_id=model_id, algorithm=algorithm).set(accuracy)
            
            # Check for alerts
            self._check_alerts(metrics)
            
            # Return prediction results with monitoring info
            return {
                'predictions': result.predictions.tolist(),
                'anomaly_scores': result.anomaly_scores.tolist() if hasattr(result, 'anomaly_scores') else None,
                'anomaly_count': result.anomaly_count,
                'anomaly_rate': anomaly_rate,
                'monitoring': {
                    'processing_time_ms': processing_time_ms,
                    'memory_usage_mb': memory_usage,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error for model {model_id}: {e}")
            
            # Record error metrics
            error_metrics = ModelMetrics(
                model_id=model_id,
                timestamp=datetime.now(),
                predictions_count=0,
                anomaly_rate=0.0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                memory_usage_mb=self._get_memory_usage(),
                cpu_usage_percent=self._get_cpu_usage()
            )
            
            self.model_metrics.append(error_metrics)
            
            raise
    
    def detect_data_drift(
        self,
        model_id: str,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        method: str = "statistical"
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.
        
        Args:
            model_id: ID of the model
            reference_data: Reference dataset (training data)
            current_data: Current data to compare
            method: Drift detection method
            
        Returns:
            Drift detection results
        """
        drift_results = {
            'model_id': model_id,
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'drift_score': 0.0,
            'feature_drifts': []
        }
        
        try:
            if method == "statistical":
                # Use KS test for each feature
                from scipy.stats import ks_2samp
                
                feature_drifts = []
                drift_scores = []
                
                for i in range(reference_data.shape[1]):
                    ref_feature = reference_data[:, i]
                    cur_feature = current_data[:, i]
                    
                    # Kolmogorov-Smirnov test
                    statistic, p_value = ks_2samp(ref_feature, cur_feature)
                    
                    drift_detected = p_value < 0.05  # 5% significance level
                    drift_scores.append(statistic)
                    
                    feature_drifts.append({
                        'feature_index': i,
                        'ks_statistic': statistic,
                        'p_value': p_value,
                        'drift_detected': drift_detected
                    })
                
                # Overall drift score
                overall_drift_score = np.mean(drift_scores)
                overall_drift_detected = overall_drift_score > self.alert_config.drift_threshold
                
                drift_results.update({
                    'drift_detected': overall_drift_detected,
                    'drift_score': overall_drift_score,
                    'feature_drifts': feature_drifts
                })
                
            elif method == "evidently":
                # Use Evidently AI for drift detection
                try:
                    from evidently.report import Report
                    from evidently.metrics import DataDriftMetric
                    
                    # Convert to DataFrames
                    ref_df = pd.DataFrame(reference_data, columns=[f'feature_{i}' for i in range(reference_data.shape[1])])
                    cur_df = pd.DataFrame(current_data, columns=[f'feature_{i}' for i in range(current_data.shape[1])])
                    
                    # Generate drift report
                    report = Report(metrics=[DataDriftMetric()])
                    report.run(reference_data=ref_df, current_data=cur_df)
                    
                    # Extract results
                    report_dict = report.as_dict()
                    drift_info = report_dict['metrics'][0]['result']
                    
                    drift_results.update({
                        'drift_detected': drift_info['dataset_drift'],
                        'drift_score': drift_info['drift_share'],
                        'evidently_details': drift_info
                    })
                    
                except ImportError:
                    self.logger.warning("Evidently AI not available, falling back to statistical method")
                    return self.detect_data_drift(model_id, reference_data, current_data, "statistical")
            
            # Update model metrics with drift information
            if model_id in self.active_models:
                latest_metrics = ModelMetrics(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    predictions_count=0,
                    anomaly_rate=0.0,
                    processing_time_ms=0.0,
                    memory_usage_mb=self._get_memory_usage(),
                    cpu_usage_percent=self._get_cpu_usage(),
                    drift_score=drift_results['drift_score']
                )
                
                self.model_metrics.append(latest_metrics)
                
                # Update Prometheus metrics
                if self.prometheus_enabled:
                    self.drift_score_gauge.labels(
                        model_id=model_id,
                        algorithm=self.active_models[model_id]['algorithm']
                    ).set(drift_results['drift_score'])
            
            # Check for drift alerts
            if drift_results['drift_detected']:
                self._send_alert(
                    'data_drift',
                    f"Data drift detected for model {model_id}",
                    {'drift_score': drift_results['drift_score']}
                )
            
            self.logger.info(f"Drift detection for {model_id}: score={drift_results['drift_score']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Drift detection error: {e}")
            drift_results['error'] = str(e)
        
        return drift_results
    
    def schedule_retraining(
        self,
        model_id: str,
        training_data: np.ndarray,
        training_labels: Optional[np.ndarray] = None,
        trigger_conditions: Optional[Dict[str, Any]] = None
    ):
        """
        Schedule automatic model retraining based on conditions.
        
        Args:
            model_id: ID of the model to retrain
            training_data: New training data
            training_labels: Training labels (if available)
            trigger_conditions: Conditions for triggering retraining
        """
        if model_id not in self.active_models:
            raise ValueError(f"Model {model_id} not found")
        
        model_config = self.active_models[model_id]
        conditions = trigger_conditions or {
            'drift_threshold': 0.3,
            'accuracy_threshold': 0.8,
            'min_samples': 1000,
            'schedule': 'weekly'
        }
        
        def should_retrain() -> bool:
            """Check if retraining should be triggered."""
            recent_metrics = [m for m in self.model_metrics if m.model_id == model_id and 
                            (datetime.now() - m.timestamp).days < 7]
            
            if not recent_metrics:
                return False
            
            # Check drift condition
            if 'drift_threshold' in conditions:
                latest_drift = max((m.drift_score for m in recent_metrics if m.drift_score is not None), default=0)
                if latest_drift > conditions['drift_threshold']:
                    self.logger.info(f"Triggering retraining for {model_id}: drift score {latest_drift}")
                    return True
            
            # Check accuracy condition
            if 'accuracy_threshold' in conditions and training_labels is not None:
                recent_accuracy = [m.accuracy for m in recent_metrics if m.accuracy is not None]
                if recent_accuracy and np.mean(recent_accuracy) < conditions['accuracy_threshold']:
                    self.logger.info(f"Triggering retraining for {model_id}: accuracy below threshold")
                    return True
            
            # Check minimum samples condition
            if len(training_data) >= conditions.get('min_samples', 1000):
                self.logger.info(f"Triggering retraining for {model_id}: sufficient new samples")
                return True
            
            return False
        
        def retrain_model():
            """Perform model retraining."""
            try:
                self.logger.info(f"Starting retraining for model {model_id}")
                
                algorithm = model_config['algorithm']
                
                # Retrain the model
                if training_labels is not None:
                    # Supervised retraining (if labels available)
                    # This would require extending the detection service
                    self.logger.warning("Supervised retraining not implemented yet")
                else:
                    # Unsupervised retraining
                    self.detection_service.fit(
                        data=training_data,
                        algorithm=algorithm,
                        contamination=model_config.get('contamination', 0.1)
                    )
                
                # Save new model
                new_model_path = self.model_registry_path / f"{model_id}_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                
                # Update model config
                model_config['last_retrained'] = datetime.now().isoformat()
                model_config['retrain_count'] = model_config.get('retrain_count', 0) + 1
                model_config['model_path'] = str(new_model_path)
                
                self.logger.info(f"Model {model_id} retrained successfully")
                
                # Send notification
                self._send_alert(
                    'model_retrained',
                    f"Model {model_id} has been retrained",
                    {'retrain_count': model_config['retrain_count']}
                )
                
            except Exception as e:
                self.logger.error(f"Retraining failed for {model_id}: {e}")
                self._send_alert('retraining_failed', f"Retraining failed for {model_id}", {'error': str(e)})
        
        # Schedule retraining check
        schedule_type = conditions.get('schedule', 'daily')
        if schedule_type == 'daily':
            schedule.every().day.at("02:00").do(lambda: retrain_model() if should_retrain() else None)
        elif schedule_type == 'weekly':
            schedule.every().week.do(lambda: retrain_model() if should_retrain() else None)
        elif schedule_type == 'continuous':
            # Check conditions continuously
            def continuous_check():
                while self.is_monitoring:
                    if should_retrain():
                        retrain_model()
                    time.sleep(3600)  # Check hourly
            
            threading.Thread(target=continuous_check, daemon=True).start()
    
    def _check_alerts(self, metrics: ModelMetrics):
        """Check metrics against alert thresholds."""
        alert_type = None
        message = None
        details = {}
        
        # Check anomaly rate
        if (metrics.anomaly_rate < self.alert_config.anomaly_rate_threshold[0] or 
            metrics.anomaly_rate > self.alert_config.anomaly_rate_threshold[1]):
            alert_type = 'anomaly_rate'
            message = f"Anomaly rate {metrics.anomaly_rate:.3f} outside normal range"
            details['anomaly_rate'] = metrics.anomaly_rate
        
        # Check processing time
        elif metrics.processing_time_ms > self.alert_config.processing_time_threshold:
            alert_type = 'processing_time'
            message = f"Processing time {metrics.processing_time_ms:.1f}ms exceeds threshold"
            details['processing_time_ms'] = metrics.processing_time_ms
        
        # Check memory usage
        elif metrics.memory_usage_mb > self.alert_config.memory_threshold:
            alert_type = 'memory_usage'
            message = f"Memory usage {metrics.memory_usage_mb:.1f}MB exceeds threshold"
            details['memory_usage_mb'] = metrics.memory_usage_mb
        
        # Check accuracy (if available)
        elif metrics.accuracy is not None and metrics.accuracy < 0.7:
            alert_type = 'low_accuracy'
            message = f"Model accuracy {metrics.accuracy:.3f} below acceptable threshold"
            details['accuracy'] = metrics.accuracy
        
        if alert_type:
            self._send_alert(alert_type, message, details)
    
    def _send_alert(self, alert_type: str, message: str, details: Dict[str, Any]):
        """Send monitoring alert."""
        # Check cooldown
        cooldown_key = f"{alert_type}_{details.get('model_id', 'global')}"
        last_alert = self.last_alerts.get(cooldown_key)
        
        if last_alert and (datetime.now() - last_alert).total_seconds() < self.alert_config.alert_cooldown_minutes * 60:
            return  # Still in cooldown
        
        # Log alert
        self.logger.warning(f"ALERT [{alert_type}]: {message} - {details}")
        
        # Update last alert time
        self.last_alerts[cooldown_key] = datetime.now()
        
        # Here you would integrate with your alerting system
        # Examples: Slack, PagerDuty, email, etc.
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
    
    def get_model_health_report(self, model_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive health report for a model.
        
        Args:
            model_id: ID of the model
            days: Number of days to include in report
            
        Returns:
            Comprehensive health report
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [m for m in self.model_metrics if m.model_id == model_id and m.timestamp > cutoff_date]
        
        if not recent_metrics:
            return {'error': f'No metrics found for model {model_id} in last {days} days'}
        
        # Calculate aggregated metrics
        processing_times = [m.processing_time_ms for m in recent_metrics if m.processing_time_ms > 0]
        anomaly_rates = [m.anomaly_rate for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics if m.accuracy is not None]
        drift_scores = [m.drift_score for m in recent_metrics if m.drift_score is not None]
        
        report = {
            'model_id': model_id,
            'report_period_days': days,
            'total_predictions': sum(m.predictions_count for m in recent_metrics),
            'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0,
            'p95_processing_time_ms': np.percentile(processing_times, 95) if processing_times else 0,
            'avg_anomaly_rate': np.mean(anomaly_rates),
            'anomaly_rate_std': np.std(anomaly_rates),
            'avg_accuracy': np.mean(accuracies) if accuracies else None,
            'accuracy_trend': self._calculate_trend(accuracies) if len(accuracies) > 1 else None,
            'latest_drift_score': drift_scores[-1] if drift_scores else None,
            'max_drift_score': max(drift_scores) if drift_scores else None,
            'health_score': self._calculate_health_score(recent_metrics),
            'recommendations': self._generate_recommendations(recent_metrics),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a metric."""
        if len(values) < 2:
            return 'stable'
        
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        change = (second_half - first_half) / first_half if first_half != 0 else 0
        
        if change > 0.05:
            return 'improving'
        elif change < -0.05:
            return 'degrading'
        else:
            return 'stable'
    
    def _calculate_health_score(self, metrics: List[ModelMetrics]) -> float:
        """Calculate overall health score (0-100)."""
        if not metrics:
            return 0.0
        
        score = 100.0
        
        # Processing time penalty
        avg_processing_time = np.mean([m.processing_time_ms for m in metrics if m.processing_time_ms > 0])
        if avg_processing_time > 1000:  # > 1 second
            score -= min(20, (avg_processing_time - 1000) / 100)
        
        # Memory usage penalty
        avg_memory = np.mean([m.memory_usage_mb for m in metrics])
        if avg_memory > 1000:  # > 1GB
            score -= min(15, (avg_memory - 1000) / 100)
        
        # Accuracy bonus/penalty
        accuracies = [m.accuracy for m in metrics if m.accuracy is not None]
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            if avg_accuracy < 0.8:
                score -= (0.8 - avg_accuracy) * 50
            elif avg_accuracy > 0.9:
                score += (avg_accuracy - 0.9) * 20
        
        # Drift penalty
        drift_scores = [m.drift_score for m in metrics if m.drift_score is not None]
        if drift_scores:
            max_drift = max(drift_scores)
            if max_drift > 0.3:
                score -= (max_drift - 0.3) * 30
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, metrics: List[ModelMetrics]) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        if not metrics:
            return ["No metrics available for analysis"]
        
        # Processing time recommendations
        avg_processing_time = np.mean([m.processing_time_ms for m in metrics if m.processing_time_ms > 0])
        if avg_processing_time > 2000:
            recommendations.append("Consider optimizing model for faster inference")
        
        # Memory recommendations
        avg_memory = np.mean([m.memory_usage_mb for m in metrics])
        if avg_memory > 1500:
            recommendations.append("Monitor memory usage - consider model compression")
        
        # Accuracy recommendations
        accuracies = [m.accuracy for m in metrics if m.accuracy is not None]
        if accuracies and np.mean(accuracies) < 0.8:
            recommendations.append("Model accuracy is declining - consider retraining")
        
        # Drift recommendations
        drift_scores = [m.drift_score for m in metrics if m.drift_score is not None]
        if drift_scores and max(drift_scores) > 0.3:
            recommendations.append("Data drift detected - evaluate need for retraining")
        
        # Stability recommendations
        anomaly_rates = [m.anomaly_rate for m in metrics]
        if np.std(anomaly_rates) > 0.1:
            recommendations.append("High variance in anomaly detection rates - investigate data quality")
        
        if not recommendations:
            recommendations.append("Model is performing well - continue monitoring")
        
        return recommendations


def demonstrate_mlops_practices():
    """Demonstrate MLOps best practices for anomaly detection."""
    
    print("🔧 MLOps and Production Monitoring Guide")
    print("=" * 50)
    
    print("\n📊 Key MLOps Practices:")
    
    practices = [
        "• Model Registry: Centralized storage and versioning of models",
        "• Continuous Monitoring: Real-time tracking of model performance",
        "• Data Drift Detection: Identifying when input data changes",
        "• Automated Retraining: Triggering retraining based on conditions",
        "• A/B Testing: Comparing model versions in production",
        "• Feature Store: Centralized feature management",
        "• Model Lineage: Tracking model ancestry and dependencies",
        "• Automated Testing: CI/CD pipelines for model validation",
        "• Alerting System: Notifications for performance degradation",
        "• Rollback Capability: Quick recovery from bad deployments"
    ]
    
    for practice in practices:
        print(f"   {practice}")
    
    print("\n📈 Monitoring Metrics:")
    
    metrics = [
        "• Model Performance: Accuracy, precision, recall, F1-score",
        "• System Performance: Latency, throughput, resource usage",
        "• Data Quality: Missing values, outliers, schema violations",
        "• Data Drift: Statistical tests, distribution comparisons",
        "• Model Drift: Performance degradation over time",
        "• Business Metrics: Domain-specific KPIs",
        "• Operational Metrics: Error rates, availability, SLA compliance",
        "• Security Metrics: Authentication, authorization, data privacy"
    ]
    
    for metric in metrics:
        print(f"   {metric}")
    
    print("\n🚨 Alert Conditions:")
    
    alerts = [
        "• Accuracy drops below threshold (e.g., < 80%)",
        "• Processing time exceeds SLA (e.g., > 5 seconds)",
        "• Memory usage spikes (e.g., > 2GB)",
        "• Error rate increases (e.g., > 5%)",
        "• Data drift detected (KS test p-value < 0.05)",
        "• Anomaly rate outside expected range",
        "• Model not updated for extended period",
        "• System resource exhaustion"
    ]
    
    for alert in alerts:
        print(f"   {alert}")
    
    print("\n🔄 Retraining Triggers:")
    
    triggers = [
        "• Scheduled retraining (daily, weekly, monthly)",
        "• Performance degradation below threshold",
        "• Significant data drift detected",
        "• New labeled data available",
        "• Concept drift in target distribution",
        "• Business rule changes",
        "• Model age exceeds policy limit",
        "• Manual trigger by data scientist"
    ]
    
    for trigger in triggers:
        print(f"   {trigger}")


def main():
    """Main function to demonstrate MLOps monitoring."""
    
    demonstrate_mlops_practices()
    
    print(f"\n🚀 Ready to set up MLOps monitoring?")
    choice = input("Run monitoring demo? (y/n): ").lower().strip()
    
    if choice == 'y':
        # Initialize monitor
        monitor = MLOpsMonitor()
        
        # Register a sample model
        monitor.register_model(
            model_id="fraud_detector_v1",
            algorithm="isolation_forest",
            model_path="models/fraud_detector.pkl",
            metadata={"version": "1.0", "training_date": "2024-01-15"}
        )
        
        # Generate sample data and make predictions
        print("\n📊 Running sample predictions with monitoring...")
        
        np.random.seed(42)
        sample_data = np.random.randn(100, 5)
        
        result = monitor.predict_with_monitoring("fraud_detector_v1", sample_data)
        
        print(f"Predictions completed:")
        print(f"  Anomalies detected: {result['anomaly_count']}")
        print(f"  Processing time: {result['monitoring']['processing_time_ms']:.1f}ms")
        print(f"  Memory usage: {result['monitoring']['memory_usage_mb']:.1f}MB")
        
        # Generate health report
        print("\n📋 Generating health report...")
        health_report = monitor.get_model_health_report("fraud_detector_v1")
        
        print(f"Health Score: {health_report['health_score']:.1f}/100")
        print(f"Recommendations: {len(health_report['recommendations'])}")
        for rec in health_report['recommendations']:
            print(f"  • {rec}")
        
        print(f"\n✅ MLOps monitoring demo complete!")
        print(f"Check mlops_monitor.log for detailed logs")
        
    else:
        print("Demo skipped. Use the MLOpsMonitor class to implement your own monitoring!")


if __name__ == "__main__":
    main()
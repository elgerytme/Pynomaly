"""
AI-driven Operations (AIOps) for Predictive Maintenance
Implements machine learning models to predict system failures and maintenance needs
"""

import asyncio
import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from kubernetes import client, config as k8s_config
from prometheus_client.parser import text_string_to_metric_families
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class SystemMetric:
    """Represents a system metric data point"""
    timestamp: datetime
    component: str
    metric_name: str
    value: float
    labels: Dict[str, str]


@dataclass
class PredictionResult:
    """Represents a maintenance prediction result"""
    component: str
    prediction_type: str  # failure, maintenance, performance_degradation
    probability: float
    confidence: float
    predicted_time: Optional[datetime]
    recommended_actions: List[str]
    severity: str  # low, medium, high, critical
    features_used: List[str]


@dataclass
class MaintenanceRecommendation:
    """Represents a maintenance recommendation"""
    id: str
    component: str
    type: str  # preventive, corrective, predictive
    priority: int  # 1-5, 5 being highest
    description: str
    estimated_downtime: timedelta
    cost_estimate: float
    deadline: datetime
    dependencies: List[str]


class AIOpsPredictor:
    """AI-driven operations predictor for system maintenance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_history = {}
        self.predictions_history = []
        
        # Model configuration
        self.model_config = {
            'anomaly_detection': {
                'algorithm': 'isolation_forest',
                'contamination': 0.1,
                'retrain_interval': 3600  # 1 hour
            },
            'failure_prediction': {
                'algorithm': 'random_forest',
                'n_estimators': 100,
                'retrain_interval': 86400  # 24 hours
            },
            'performance_prediction': {
                'algorithm': 'regression',
                'retrain_interval': 21600  # 6 hours
            }
        }
        
        # Initialize components
        self._init_kubernetes_client()
        self._init_models()
        
    def _init_kubernetes_client(self):
        """Initialize Kubernetes client for metrics collection"""
        try:
            if self.config.get('kubeconfig_path'):
                k8s_config.load_kube_config(config_file=self.config['kubeconfig_path'])
            else:
                k8s_config.load_incluster_config()
                
            self.k8s_client = client.CoreV1Api()
            self.k8s_apps_client = client.AppsV1Api()
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            self.k8s_client = None

    def _init_models(self):
        """Initialize machine learning models"""
        try:
            # Anomaly detection model
            self.models['anomaly_detection'] = IsolationForest(
                contamination=self.model_config['anomaly_detection']['contamination'],
                random_state=42,
                n_jobs=-1
            )
            
            # Failure prediction model
            self.models['failure_prediction'] = RandomForestClassifier(
                n_estimators=self.model_config['failure_prediction']['n_estimators'],
                random_state=42,
                n_jobs=-1
            )
            
            # Initialize scalers
            self.scalers['standard'] = StandardScaler()
            self.scalers['minmax'] = MinMaxScaler()
            
            logger.info("Initialized AIOps ML models")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    async def collect_system_metrics(self) -> List[SystemMetric]:
        """Collect comprehensive system metrics from various sources"""
        metrics = []
        
        try:
            # Collect Kubernetes metrics
            k8s_metrics = await self._collect_kubernetes_metrics()
            metrics.extend(k8s_metrics)
            
            # Collect Prometheus metrics
            prometheus_metrics = await self._collect_prometheus_metrics()
            metrics.extend(prometheus_metrics)
            
            # Collect application-specific metrics
            app_metrics = await self._collect_application_metrics()
            metrics.extend(app_metrics)
            
            # Collect infrastructure metrics  
            infra_metrics = await self._collect_infrastructure_metrics()
            metrics.extend(infra_metrics)
            
            logger.info(f"Collected {len(metrics)} system metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return []

    async def _collect_kubernetes_metrics(self) -> List[SystemMetric]:
        """Collect Kubernetes cluster metrics"""
        metrics = []
        
        if not self.k8s_client:
            return metrics
            
        try:
            # Node metrics
            nodes = self.k8s_client.list_node()
            for node in nodes.items:
                node_metrics = await self._extract_node_metrics(node)
                metrics.extend(node_metrics)
                
            # Pod metrics
            pods = self.k8s_client.list_pod_for_all_namespaces()
            for pod in pods.items:
                pod_metrics = await self._extract_pod_metrics(pod)
                metrics.extend(pod_metrics)
                
        except Exception as e:
            logger.error(f"Failed to collect Kubernetes metrics: {e}")
            
        return metrics

    async def _extract_node_metrics(self, node) -> List[SystemMetric]:
        """Extract metrics from a Kubernetes node"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Node capacity and allocatable resources
            capacity = node.status.capacity
            allocatable = node.status.allocatable
            
            for resource in ['cpu', 'memory', 'pods']:
                if resource in capacity:
                    metrics.append(SystemMetric(
                        timestamp=timestamp,
                        component=f"node/{node.metadata.name}",
                        metric_name=f"{resource}_capacity",
                        value=self._parse_resource_value(capacity[resource]),
                        labels={'node': node.metadata.name, 'resource': resource}
                    ))
                    
                if resource in allocatable:
                    metrics.append(SystemMetric(
                        timestamp=timestamp,
                        component=f"node/{node.metadata.name}",
                        metric_name=f"{resource}_allocatable",
                        value=self._parse_resource_value(allocatable[resource]),
                        labels={'node': node.metadata.name, 'resource': resource}
                    ))
                    
            # Node conditions
            if node.status.conditions:
                for condition in node.status.conditions:
                    metrics.append(SystemMetric(
                        timestamp=timestamp,
                        component=f"node/{node.metadata.name}",
                        metric_name=f"condition_{condition.type.lower()}",
                        value=1.0 if condition.status == "True" else 0.0,
                        labels={'node': node.metadata.name, 'condition': condition.type}
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to extract node metrics: {e}")
            
        return metrics

    async def _extract_pod_metrics(self, pod) -> List[SystemMetric]:
        """Extract metrics from a Kubernetes pod"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Pod phase
            phase_mapping = {'Running': 1.0, 'Pending': 0.5, 'Succeeded': 0.8, 'Failed': 0.0, 'Unknown': -1.0}
            metrics.append(SystemMetric(
                timestamp=timestamp,
                component=f"pod/{pod.metadata.namespace}/{pod.metadata.name}",
                metric_name="pod_phase",
                value=phase_mapping.get(pod.status.phase, -1.0),
                labels={'namespace': pod.metadata.namespace, 'pod': pod.metadata.name, 'phase': pod.status.phase}
            ))
            
            # Container metrics
            if pod.status.container_statuses:
                for container in pod.status.container_statuses:
                    metrics.append(SystemMetric(
                        timestamp=timestamp,
                        component=f"container/{pod.metadata.namespace}/{pod.metadata.name}/{container.name}",
                        metric_name="container_ready",
                        value=1.0 if container.ready else 0.0,
                        labels={'namespace': pod.metadata.namespace, 'pod': pod.metadata.name, 'container': container.name}
                    ))
                    
                    metrics.append(SystemMetric(
                        timestamp=timestamp,
                        component=f"container/{pod.metadata.namespace}/{pod.metadata.name}/{container.name}",
                        metric_name="container_restart_count",
                        value=float(container.restart_count),
                        labels={'namespace': pod.metadata.namespace, 'pod': pod.metadata.name, 'container': container.name}
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to extract pod metrics: {e}")
            
        return metrics

    def _parse_resource_value(self, resource_str: str) -> float:
        """Parse Kubernetes resource string to numeric value"""
        try:
            if resource_str.endswith('Ki'):
                return float(resource_str[:-2]) * 1024
            elif resource_str.endswith('Mi'):
                return float(resource_str[:-2]) * 1024 * 1024
            elif resource_str.endswith('Gi'):
                return float(resource_str[:-2]) * 1024 * 1024 * 1024
            elif resource_str.endswith('m'):
                return float(resource_str[:-1]) / 1000
            else:
                return float(resource_str)
        except:
            return 0.0

    async def _collect_prometheus_metrics(self) -> List[SystemMetric]:
        """Collect metrics from Prometheus"""
        metrics = []
        
        try:
            import aiohttp
            
            prometheus_url = self.config.get('prometheus_url', 'http://prometheus:9090')
            queries = [
                'up',
                'cpu_usage_percent',
                'memory_usage_percent',
                'disk_usage_percent',
                'network_bytes_sent',
                'network_bytes_received',
                'http_requests_total',
                'http_request_duration_seconds'
            ]
            
            async with aiohttp.ClientSession() as session:
                for query in queries:
                    try:
                        url = f"{prometheus_url}/api/v1/query"
                        params = {'query': query}
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                prometheus_metrics = self._parse_prometheus_response(data, query)
                                metrics.extend(prometheus_metrics)
                                
                    except Exception as e:
                        logger.error(f"Failed to query Prometheus for {query}: {e}")
                        
        except ImportError:
            logger.warning("aiohttp not available, skipping Prometheus metrics collection")
        except Exception as e:
            logger.error(f"Failed to collect Prometheus metrics: {e}")
            
        return metrics

    def _parse_prometheus_response(self, data: Dict, query: str) -> List[SystemMetric]:
        """Parse Prometheus API response"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            if data.get('status') == 'success' and 'data' in data:
                result = data['data'].get('result', [])
                
                for item in result:
                    metric_labels = item.get('metric', {})
                    value_data = item.get('value', [])
                    
                    if len(value_data) >= 2:
                        value = float(value_data[1])
                        
                        metrics.append(SystemMetric(
                            timestamp=timestamp,
                            component=f"prometheus/{query}",
                            metric_name=query,
                            value=value,
                            labels=metric_labels
                        ))
                        
        except Exception as e:
            logger.error(f"Failed to parse Prometheus response: {e}")
            
        return metrics

    async def _collect_application_metrics(self) -> List[SystemMetric]:
        """Collect application-specific metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Mock application metrics (replace with actual application monitoring)
            app_metrics = {
                'request_latency_ms': np.random.normal(50, 10),
                'error_rate_percent': np.random.uniform(0, 5),
                'throughput_rps': np.random.normal(1000, 100),
                'active_connections': np.random.randint(100, 1000),
                'queue_length': np.random.randint(0, 50),
                'cache_hit_rate': np.random.uniform(0.7, 0.95),
                'database_connection_pool_usage': np.random.uniform(0.3, 0.8)
            }
            
            for metric_name, value in app_metrics.items():
                metrics.append(SystemMetric(
                    timestamp=timestamp,
                    component="application/mlops",
                    metric_name=metric_name,
                    value=float(value),
                    labels={'app': 'mlops', 'version': 'v1.0.0'}
                ))
                
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            
        return metrics

    async def _collect_infrastructure_metrics(self) -> List[SystemMetric]:
        """Collect infrastructure metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Mock infrastructure metrics (replace with actual infrastructure monitoring)
            infra_metrics = {
                'load_balancer_response_time': np.random.normal(10, 2),
                'database_query_time': np.random.normal(25, 5),
                'storage_iops': np.random.randint(500, 2000),
                'network_latency_ms': np.random.normal(5, 1),
                'ssl_certificate_days_remaining': np.random.randint(30, 365),
                'backup_success_rate': np.random.uniform(0.95, 1.0),
                'log_error_count': np.random.randint(0, 10)
            }
            
            for metric_name, value in infra_metrics.items():
                metrics.append(SystemMetric(
                    timestamp=timestamp,
                    component="infrastructure/aws",
                    metric_name=metric_name,
                    value=float(value),
                    labels={'provider': 'aws', 'region': 'us-east-1'}
                ))
                
        except Exception as e:
            logger.error(f"Failed to collect infrastructure metrics: {e}")
            
        return metrics

    async def analyze_and_predict(self, metrics: List[SystemMetric]) -> List[PredictionResult]:
        """Analyze metrics and generate predictions"""
        predictions = []
        
        try:
            # Prepare feature matrix
            feature_matrix = await self._prepare_feature_matrix(metrics)
            
            if len(feature_matrix) == 0:
                logger.warning("No features available for prediction")
                return predictions
                
            # Anomaly detection
            anomaly_predictions = await self._detect_anomalies(feature_matrix)
            predictions.extend(anomaly_predictions)
            
            # Failure prediction
            failure_predictions = await self._predict_failures(feature_matrix)
            predictions.extend(failure_predictions)
            
            # Performance degradation prediction
            performance_predictions = await self._predict_performance_issues(feature_matrix)
            predictions.extend(performance_predictions)
            
            # Store predictions for historical analysis
            self.predictions_history.extend(predictions)
            
            # Keep only recent predictions (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.predictions_history = [
                p for p in self.predictions_history 
                if p.predicted_time and p.predicted_time > cutoff_time
            ]
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to analyze and predict: {e}")
            return []

    async def _prepare_feature_matrix(self, metrics: List[SystemMetric]) -> pd.DataFrame:
        """Prepare feature matrix from metrics"""
        try:
            # Convert metrics to DataFrame
            metric_data = []
            for metric in metrics:
                metric_data.append({
                    'timestamp': metric.timestamp,
                    'component': metric.component,
                    'metric_name': metric.metric_name,
                    'value': metric.value,
                    **metric.labels
                })
                
            if not metric_data:
                return pd.DataFrame()
                
            df = pd.DataFrame(metric_data)
            
            # Pivot to create feature matrix
            feature_matrix = df.pivot_table(
                index=['timestamp', 'component'],
                columns='metric_name',
                values='value',
                aggfunc='mean'
            ).fillna(0)
            
            # Add time-based features
            feature_matrix = feature_matrix.reset_index()
            feature_matrix['hour'] = feature_matrix['timestamp'].dt.hour
            feature_matrix['day_of_week'] = feature_matrix['timestamp'].dt.dayofweek
            feature_matrix['minute'] = feature_matrix['timestamp'].dt.minute
            
            # Add rolling statistics
            numeric_columns = feature_matrix.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col not in ['hour', 'day_of_week', 'minute']:
                    feature_matrix[f'{col}_rolling_mean'] = feature_matrix[col].rolling(window=5, min_periods=1).mean()
                    feature_matrix[f'{col}_rolling_std'] = feature_matrix[col].rolling(window=5, min_periods=1).std().fillna(0)
                    
            return feature_matrix
            
        except Exception as e:
            logger.error(f"Failed to prepare feature matrix: {e}")
            return pd.DataFrame()

    async def _detect_anomalies(self, feature_matrix: pd.DataFrame) -> List[PredictionResult]:
        """Detect anomalies in system metrics"""
        predictions = []
        
        try:
            if len(feature_matrix) < 10:  # Need minimum data for anomaly detection
                return predictions
                
            # Select numeric features
            numeric_features = feature_matrix.select_dtypes(include=[np.number]).columns
            X = feature_matrix[numeric_features].fillna(0)
            
            # Scale features
            X_scaled = self.scalers['standard'].fit_transform(X)
            
            # Detect anomalies
            anomaly_scores = self.models['anomaly_detection'].fit_predict(X_scaled)
            decision_scores = self.models['anomaly_detection'].decision_function(X_scaled)
            
            # Create predictions for anomalies
            for i, (idx, row) in enumerate(feature_matrix.iterrows()):
                if anomaly_scores[i] == -1:  # Anomaly detected
                    confidence = abs(decision_scores[i])
                    severity = self._determine_severity(confidence)
                    
                    prediction = PredictionResult(
                        component=row['component'],
                        prediction_type='anomaly',
                        probability=min(confidence * 100, 100),
                        confidence=confidence,
                        predicted_time=datetime.now(),
                        recommended_actions=self._get_anomaly_recommendations(row['component']),
                        severity=severity,
                        features_used=list(numeric_features)
                    )
                    predictions.append(prediction)
                    
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            
        return predictions

    async def _predict_failures(self, feature_matrix: pd.DataFrame) -> List[PredictionResult]:
        """Predict component failures"""
        predictions = []
        
        try:
            if len(feature_matrix) < 20:  # Need more data for failure prediction
                return predictions
                
            # Create synthetic failure labels for training (in production, use historical failure data)
            failure_indicators = self._create_failure_labels(feature_matrix)
            
            if len(failure_indicators) == 0:
                return predictions
                
            # Select features
            numeric_features = feature_matrix.select_dtypes(include=[np.number]).columns
            X = feature_matrix[numeric_features].fillna(0)
            y = failure_indicators
            
            # Split data
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Train model
                self.models['failure_prediction'].fit(X_train, y_train)
                
                # Predict failures
                failure_probabilities = self.models['failure_prediction'].predict_proba(X_test)
                
                # Create predictions
                for i, prob in enumerate(failure_probabilities):
                    if prob[1] > 0.7:  # High probability of failure
                        component = feature_matrix.iloc[X_test.index[i]]['component']
                        
                        prediction = PredictionResult(
                            component=component,
                            prediction_type='failure',
                            probability=prob[1] * 100,
                            confidence=prob[1],
                            predicted_time=datetime.now() + timedelta(hours=np.random.randint(1, 48)),
                            recommended_actions=self._get_failure_recommendations(component),
                            severity='high' if prob[1] > 0.9 else 'medium',
                            features_used=list(numeric_features)
                        )
                        predictions.append(prediction)
                        
        except Exception as e:
            logger.error(f"Failed to predict failures: {e}")
            
        return predictions

    async def _predict_performance_issues(self, feature_matrix: pd.DataFrame) -> List[PredictionResult]:
        """Predict performance degradation"""
        predictions = []
        
        try:
            # Analyze performance trends
            performance_metrics = ['cpu_usage_percent', 'memory_usage_percent', 'request_latency_ms', 'error_rate_percent']
            
            for metric in performance_metrics:
                if metric in feature_matrix.columns:
                    values = feature_matrix[metric].dropna()
                    if len(values) > 5:
                        # Simple trend analysis
                        trend = np.polyfit(range(len(values)), values, 1)[0]
                        
                        # If trend is increasing for negative metrics or decreasing for positive metrics
                        is_degrading = False
                        if metric in ['request_latency_ms', 'error_rate_percent'] and trend > 0:
                            is_degrading = True
                        elif metric in ['cpu_usage_percent', 'memory_usage_percent'] and trend > 0 and values.iloc[-1] > 80:
                            is_degrading = True
                            
                        if is_degrading:
                            confidence = min(abs(trend) * 10, 1.0)
                            
                            prediction = PredictionResult(
                                component=f"performance/{metric}",
                                prediction_type='performance_degradation',
                                probability=confidence * 100,
                                confidence=confidence,
                                predicted_time=datetime.now() + timedelta(hours=2),
                                recommended_actions=self._get_performance_recommendations(metric),
                                severity='medium' if confidence > 0.7 else 'low',
                                features_used=[metric]
                            )
                            predictions.append(prediction)
                            
        except Exception as e:
            logger.error(f"Failed to predict performance issues: {e}")
            
        return predictions

    def _create_failure_labels(self, feature_matrix: pd.DataFrame) -> List[int]:
        """Create synthetic failure labels for training"""
        labels = []
        
        try:
            for idx, row in feature_matrix.iterrows():
                # Simple heuristic for failure probability
                failure_score = 0
                
                # High CPU usage
                if 'cpu_usage_percent' in row and row['cpu_usage_percent'] > 90:
                    failure_score += 0.3
                    
                # High memory usage
                if 'memory_usage_percent' in row and row['memory_usage_percent'] > 85:
                    failure_score += 0.3
                    
                # High error rate
                if 'error_rate_percent' in row and row['error_rate_percent'] > 10:
                    failure_score += 0.4
                    
                # High restart count
                if 'container_restart_count' in row and row['container_restart_count'] > 3:
                    failure_score += 0.5
                    
                # Pod not ready
                if 'container_ready' in row and row['container_ready'] == 0:
                    failure_score += 0.6
                    
                labels.append(1 if failure_score > 0.5 else 0)
                
        except Exception as e:
            logger.error(f"Failed to create failure labels: {e}")
            
        return labels

    def _determine_severity(self, confidence: float) -> str:
        """Determine severity level based on confidence"""
        if confidence > 0.9:
            return 'critical'
        elif confidence > 0.7:
            return 'high'
        elif confidence > 0.5:
            return 'medium'
        else:
            return 'low'

    def _get_anomaly_recommendations(self, component: str) -> List[str]:
        """Get recommendations for anomaly remediation"""
        recommendations = [
            "Investigate recent changes to the component",
            "Check resource utilization and scaling policies",
            "Review logs for error patterns",
            "Verify network connectivity and dependencies"
        ]
        
        if 'database' in component.lower():
            recommendations.extend([
                "Check database connection pool settings",
                "Analyze slow query logs",
                "Verify database storage capacity"
            ])
        elif 'api' in component.lower():
            recommendations.extend([
                "Check API rate limiting configuration",
                "Verify authentication and authorization",
                "Analyze request patterns and payload sizes"
            ])
            
        return recommendations

    def _get_failure_recommendations(self, component: str) -> List[str]:
        """Get recommendations for failure prevention"""
        recommendations = [
            "Immediate health check and diagnostic review",
            "Prepare rollback strategy",
            "Scale up resources if applicable",
            "Notify on-call team",
            "Create maintenance window if needed"
        ]
        
        if 'node' in component.lower():
            recommendations.extend([
                "Check node resource availability",
                "Drain and cordon node if necessary",
                "Prepare replacement node"
            ])
        elif 'pod' in component.lower():
            recommendations.extend([
                "Restart pod with health checks",
                "Check resource requests and limits",
                "Verify persistent volume availability"
            ])
            
        return recommendations

    def _get_performance_recommendations(self, metric: str) -> List[str]:
        """Get recommendations for performance optimization"""
        recommendations = []
        
        if metric == 'cpu_usage_percent':
            recommendations = [
                "Implement horizontal pod autoscaling",
                "Optimize application CPU usage",
                "Consider upgrading instance types",
                "Review and optimize algorithms"
            ]
        elif metric == 'memory_usage_percent':
            recommendations = [
                "Implement memory-based autoscaling",
                "Analyze memory leaks",
                "Optimize memory allocation patterns",
                "Consider increasing memory limits"
            ]
        elif metric == 'request_latency_ms':
            recommendations = [
                "Implement caching strategies",
                "Optimize database queries",
                "Add load balancing",
                "Consider CDN for static content"
            ]
        elif metric == 'error_rate_percent':
            recommendations = [
                "Implement circuit breaker patterns",
                "Improve error handling and retries",
                "Review API documentation and validation",
                "Implement graceful degradation"
            ]
            
        return recommendations

    async def generate_maintenance_schedule(self, predictions: List[PredictionResult]) -> List[MaintenanceRecommendation]:
        """Generate optimized maintenance schedule based on predictions"""
        recommendations = []
        
        try:
            # Sort predictions by severity and probability
            critical_predictions = [p for p in predictions if p.severity in ['critical', 'high']]
            critical_predictions.sort(key=lambda x: (x.probability, x.confidence), reverse=True)
            
            for i, prediction in enumerate(critical_predictions):
                # Determine maintenance type
                maintenance_type = 'predictive' if prediction.prediction_type == 'failure' else 'preventive'
                
                # Calculate priority
                priority = 5 if prediction.severity == 'critical' else 4 if prediction.severity == 'high' else 3
                
                # Estimate downtime and cost
                estimated_downtime = self._estimate_downtime(prediction.component, maintenance_type)
                cost_estimate = self._estimate_cost(prediction.component, maintenance_type, estimated_downtime)
                
                # Set deadline
                deadline = prediction.predicted_time or (datetime.now() + timedelta(hours=24))
                
                recommendation = MaintenanceRecommendation(
                    id=f"maint_{int(time.time())}_{i}",
                    component=prediction.component,
                    type=maintenance_type,
                    priority=priority,
                    description=f"{maintenance_type.title()} maintenance for {prediction.component} - {prediction.prediction_type}",
                    estimated_downtime=estimated_downtime,
                    cost_estimate=cost_estimate,
                    deadline=deadline,
                    dependencies=self._identify_dependencies(prediction.component)
                )
                
                recommendations.append(recommendation)
                
            # Optimize schedule to minimize total downtime and cost
            optimized_recommendations = self._optimize_maintenance_schedule(recommendations)
            
            logger.info(f"Generated {len(optimized_recommendations)} maintenance recommendations")
            return optimized_recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate maintenance schedule: {e}")
            return []

    def _estimate_downtime(self, component: str, maintenance_type: str) -> timedelta:
        """Estimate maintenance downtime"""
        base_times = {
            'predictive': timedelta(minutes=30),
            'preventive': timedelta(minutes=15),
            'corrective': timedelta(hours=2)
        }
        
        base_time = base_times.get(maintenance_type, timedelta(minutes=30))
        
        # Adjust based on component type
        if 'database' in component.lower():
            base_time *= 2
        elif 'node' in component.lower():
            base_time *= 3
        elif 'network' in component.lower():
            base_time *= 1.5
            
        return base_time

    def _estimate_cost(self, component: str, maintenance_type: str, downtime: timedelta) -> float:
        """Estimate maintenance cost"""
        # Base cost per hour of downtime
        hourly_cost = 1000.0  # $1000/hour
        
        # Maintenance type multiplier
        type_multipliers = {
            'predictive': 0.5,  # Cheaper because planned
            'preventive': 0.3,  # Cheapest
            'corrective': 2.0   # Most expensive
        }
        
        multiplier = type_multipliers.get(maintenance_type, 1.0)
        downtime_hours = downtime.total_seconds() / 3600
        
        return hourly_cost * downtime_hours * multiplier

    def _identify_dependencies(self, component: str) -> List[str]:
        """Identify component dependencies"""
        dependencies = []
        
        # Simple dependency mapping (in production, use service mesh or dependency graph)
        dependency_map = {
            'api': ['database', 'cache'],
            'frontend': ['api', 'cdn'],
            'database': ['storage'],
            'cache': ['memory'],
            'worker': ['queue', 'api']
        }
        
        for comp_type in dependency_map:
            if comp_type in component.lower():
                dependencies.extend(dependency_map[comp_type])
                break
                
        return dependencies

    def _optimize_maintenance_schedule(self, recommendations: List[MaintenanceRecommendation]) -> List[MaintenanceRecommendation]:
        """Optimize maintenance schedule to minimize impact"""
        # Simple optimization: sort by priority and deadline
        optimized = sorted(recommendations, key=lambda x: (x.priority, x.deadline), reverse=True)
        
        # Adjust schedules to avoid conflicts
        scheduled_times = {}
        for rec in optimized:
            # Find optimal time slot
            optimal_time = self._find_optimal_maintenance_time(rec, scheduled_times)
            rec.deadline = optimal_time
            scheduled_times[rec.id] = optimal_time
            
        return optimized

    def _find_optimal_maintenance_time(self, recommendation: MaintenanceRecommendation, 
                                     scheduled_times: Dict[str, datetime]) -> datetime:
        """Find optimal time for maintenance to minimize business impact"""
        # Prefer off-peak hours (assuming 2-6 AM local time)
        now = datetime.now()
        
        # Try to schedule in next available off-peak window
        next_offpeak = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if next_offpeak <= now:
            next_offpeak += timedelta(days=1)
            
        # Check for conflicts with already scheduled maintenance
        while any(abs((next_offpeak - scheduled_time).total_seconds()) < 3600 
                 for scheduled_time in scheduled_times.values()):
            next_offpeak += timedelta(hours=1)
            
        return next_offpeak

    async def execute_automated_remediation(self, prediction: PredictionResult) -> Dict[str, Any]:
        """Execute automated remediation actions"""
        remediation_result = {
            'prediction_id': f"{prediction.component}_{int(time.time())}",
            'component': prediction.component,
            'actions_taken': [],
            'success': False,
            'error': None
        }
        
        try:
            logger.info(f"Executing automated remediation for {prediction.component}")
            
            # Based on prediction type, execute appropriate actions
            if prediction.prediction_type == 'anomaly':
                result = await self._handle_anomaly_remediation(prediction)
            elif prediction.prediction_type == 'failure':
                result = await self._handle_failure_remediation(prediction)
            elif prediction.prediction_type == 'performance_degradation':
                result = await self._handle_performance_remediation(prediction)
            else:
                result = {'actions': ['Log and alert'], 'success': True}
                
            remediation_result['actions_taken'] = result.get('actions', [])
            remediation_result['success'] = result.get('success', False)
            
        except Exception as e:
            remediation_result['error'] = str(e)
            logger.error(f"Failed to execute automated remediation: {e}")
            
        return remediation_result

    async def _handle_anomaly_remediation(self, prediction: PredictionResult) -> Dict[str, Any]:
        """Handle anomaly remediation"""
        actions = []
        success = True
        
        try:
            # Restart unhealthy pods
            if 'pod' in prediction.component:
                actions.append("Restart unhealthy pod")
                # Implementation would call Kubernetes API to restart pod
                
            # Scale up if resource constrained
            if prediction.probability > 80:
                actions.append("Scale up resources")
                # Implementation would trigger autoscaling
                
            # Alert operations team
            actions.append("Send alert to operations team")
            
        except Exception as e:
            success = False
            actions.append(f"Remediation failed: {e}")
            
        return {'actions': actions, 'success': success}

    async def _handle_failure_remediation(self, prediction: PredictionResult) -> Dict[str, Any]:
        """Handle failure remediation"""
        actions = []
        success = True
        
        try:
            # Immediate actions for predicted failures
            actions.append("Create emergency maintenance window")
            actions.append("Notify on-call engineer")
            actions.append("Prepare rollback plan")
            
            # Proactive scaling
            if 'pod' in prediction.component or 'container' in prediction.component:
                actions.append("Increase replica count")
                
            # Redirect traffic if applicable
            if 'api' in prediction.component or 'service' in prediction.component:
                actions.append("Redirect traffic to healthy instances")
                
        except Exception as e:
            success = False
            actions.append(f"Failure remediation failed: {e}")
            
        return {'actions': actions, 'success': success}

    async def _handle_performance_remediation(self, prediction: PredictionResult) -> Dict[str, Any]:
        """Handle performance degradation remediation"""
        actions = []
        success = True
        
        try:
            # Performance optimization actions
            actions.append("Enable performance monitoring")
            actions.append("Adjust resource limits")
            
            if 'cpu' in prediction.component:
                actions.append("Increase CPU allocation")
            elif 'memory' in prediction.component:
                actions.append("Increase memory allocation")
            elif 'latency' in prediction.component:
                actions.append("Enable caching")
                
        except Exception as e:
            success = False
            actions.append(f"Performance remediation failed: {e}")
            
        return {'actions': actions, 'success': success}

    def save_models(self, model_dir: str):
        """Save trained models to disk"""
        try:
            model_path = Path(model_dir)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save models
            for name, model in self.models.items():
                joblib.dump(model, model_path / f"{name}_model.pkl")
                
            # Save scalers
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, model_path / f"{name}_scaler.pkl")
                
            logger.info(f"Saved models to {model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_models(self, model_dir: str):
        """Load trained models from disk"""
        try:
            model_path = Path(model_dir)
            
            # Load models
            for name in self.models.keys():
                model_file = model_path / f"{name}_model.pkl"
                if model_file.exists():
                    self.models[name] = joblib.load(model_file)
                    
            # Load scalers
            for name in self.scalers.keys():
                scaler_file = model_path / f"{name}_scaler.pkl"
                if scaler_file.exists():
                    self.scalers[name] = joblib.load(scaler_file)
                    
            logger.info(f"Loaded models from {model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


# Example usage and testing
async def main():
    """Example usage of AIOpsPredictor"""
    config = {
        'kubeconfig_path': '/path/to/kubeconfig',
        'prometheus_url': 'http://prometheus:9090',
        'model_retrain_interval': 3600
    }
    
    predictor = AIOpsPredictor(config)
    
    # Collect metrics
    metrics = await predictor.collect_system_metrics()
    print(f"Collected {len(metrics)} metrics")
    
    # Generate predictions
    predictions = await predictor.analyze_and_predict(metrics)
    print(f"Generated {len(predictions)} predictions")
    
    for prediction in predictions:
        print(f"  - {prediction.component}: {prediction.prediction_type} ({prediction.probability:.1f}% probability)")
        
    # Generate maintenance schedule
    maintenance_schedule = await predictor.generate_maintenance_schedule(predictions)
    print(f"Generated {len(maintenance_schedule)} maintenance recommendations")
    
    # Execute automated remediation for critical predictions
    critical_predictions = [p for p in predictions if p.severity == 'critical']
    for prediction in critical_predictions:
        result = await predictor.execute_automated_remediation(prediction)
        print(f"Remediation result: {result}")
        
    # Save models
    predictor.save_models('./models')


if __name__ == "__main__":
    asyncio.run(main())
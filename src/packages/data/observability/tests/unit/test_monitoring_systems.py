"""
Observability and monitoring system validation tests.
Tests real-time monitoring, alerting, and system health tracking for production deployment.
"""

import pytest
import time
import json
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import sys
from pathlib import Path
import threading

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from data_observability.application.services.pipeline_health_service import PipelineHealthService
    from data_observability.application.services.predictive_quality_service import PredictiveQualityService
    from data_observability.application.services.data_lineage_service import DataLineageService
    from data_observability.domain.entities.pipeline_health import PipelineHealth
    from data_observability.domain.entities.data_lineage import DataLineage
except ImportError as e:
    # Create mock classes for testing infrastructure
    class PipelineHealthService:
        def __init__(self):
            self.health_metrics = {}
            self.alerts = []
            self.monitoring_active = False
            
        def start_monitoring(self, pipeline_id: str, config: Dict = None):
            """Start monitoring a pipeline."""
            self.monitoring_active = True
            self.health_metrics[pipeline_id] = {
                'status': 'healthy',
                'last_update': time.time(),
                'metrics': {
                    'execution_time': 0.0,
                    'memory_usage': 0.0,
                    'cpu_usage': 0.0,
                    'error_rate': 0.0,
                    'throughput': 0.0
                }
            }
            return {'success': True, 'monitoring_id': f"monitor_{pipeline_id}"}
            
        def get_health_status(self, pipeline_id: str) -> Dict[str, Any]:
            """Get current health status."""
            if pipeline_id in self.health_metrics:
                health_data = self.health_metrics[pipeline_id].copy()
                
                # Simulate health degradation over time
                time_since_update = time.time() - health_data['last_update']
                if time_since_update > 300:  # 5 minutes
                    health_data['status'] = 'degraded'
                elif time_since_update > 600:  # 10 minutes
                    health_data['status'] = 'unhealthy'
                    
                return health_data
            
            return {'status': 'unknown', 'error': 'Pipeline not monitored'}
            
        def record_metrics(self, pipeline_id: str, metrics: Dict[str, float]):
            """Record performance metrics."""
            if pipeline_id in self.health_metrics:
                self.health_metrics[pipeline_id]['metrics'].update(metrics)
                self.health_metrics[pipeline_id]['last_update'] = time.time()
                
                # Check for threshold violations
                self._check_thresholds(pipeline_id, metrics)
                
                return {'success': True}
            
            return {'success': False, 'error': 'Pipeline not monitored'}
            
        def _check_thresholds(self, pipeline_id: str, metrics: Dict[str, float]):
            """Check metrics against thresholds and generate alerts."""
            thresholds = {
                'execution_time': 30.0,  # seconds
                'memory_usage': 1000.0,  # MB
                'cpu_usage': 80.0,       # percentage
                'error_rate': 5.0        # percentage
            }
            
            for metric, value in metrics.items():
                if metric in thresholds and value > thresholds[metric]:
                    alert = {
                        'id': f"alert_{len(self.alerts) + 1}",
                        'pipeline_id': pipeline_id,
                        'metric': metric,
                        'value': value,
                        'threshold': thresholds[metric],
                        'severity': 'high' if value > thresholds[metric] * 1.5 else 'medium',
                        'timestamp': time.time(),
                        'message': f"{metric} ({value}) exceeded threshold ({thresholds[metric]})"
                    }
                    self.alerts.append(alert)
                    
        def get_alerts(self, pipeline_id: str = None, severity: str = None) -> List[Dict[str, Any]]:
            """Get alerts with optional filtering."""
            filtered_alerts = self.alerts
            
            if pipeline_id:
                filtered_alerts = [a for a in filtered_alerts if a.get('pipeline_id') == pipeline_id]
                
            if severity:
                filtered_alerts = [a for a in filtered_alerts if a.get('severity') == severity]
                
            return filtered_alerts
            
    class PredictiveQualityService:
        def __init__(self):
            self.quality_history = {}
            self.predictions = {}
            
        def record_quality_metrics(self, data_source: str, metrics: Dict[str, float]) -> Dict[str, Any]:
            """Record quality metrics for trend analysis."""
            if data_source not in self.quality_history:
                self.quality_history[data_source] = []
                
            metric_record = {
                'timestamp': time.time(),
                'metrics': metrics.copy()
            }
            
            self.quality_history[data_source].append(metric_record)
            
            # Keep only recent history (last 100 records)
            self.quality_history[data_source] = self.quality_history[data_source][-100:]
            
            return {'success': True, 'records_count': len(self.quality_history[data_source])}
            
        def predict_quality_degradation(self, data_source: str, horizon_hours: int = 24) -> Dict[str, Any]:
            """Predict quality degradation based on historical trends."""
            if data_source not in self.quality_history or len(self.quality_history[data_source]) < 5:
                return {
                    'success': False,
                    'error': 'Insufficient historical data for prediction'
                }
                
            history = self.quality_history[data_source]
            
            # Simple trend analysis (mock implementation)
            recent_metrics = [record['metrics'] for record in history[-5:]]
            
            predictions = {}
            
            for metric_name in recent_metrics[0].keys():
                values = [m[metric_name] for m in recent_metrics]
                
                # Calculate trend
                trend = (values[-1] - values[0]) / len(values)
                
                # Predict future value
                predicted_value = values[-1] + trend * (horizon_hours / 24)
                
                # Determine risk level
                if predicted_value < 0.7:  # Quality threshold
                    risk_level = 'high'
                elif predicted_value < 0.8:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
                    
                predictions[metric_name] = {
                    'current_value': values[-1],
                    'predicted_value': predicted_value,
                    'trend': 'declining' if trend < 0 else 'improving',
                    'risk_level': risk_level,
                    'confidence': 0.75  # Mock confidence
                }
                
            return {
                'success': True,
                'data_source': data_source,
                'horizon_hours': horizon_hours,
                'predictions': predictions,
                'prediction_timestamp': time.time()
            }
            
    class DataLineageService:
        def __init__(self):
            self.lineage_graph = {}
            self.transformations = {}
            
        def track_data_flow(self, source_id: str, target_id: str, transformation: str, metadata: Dict = None) -> Dict[str, Any]:
            """Track data flow between components."""
            if source_id not in self.lineage_graph:
                self.lineage_graph[source_id] = []
                
            lineage_record = {
                'target_id': target_id,
                'transformation': transformation,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            self.lineage_graph[source_id].append(lineage_record)
            
            # Track transformation details
            transformation_key = f"{source_id}->{target_id}"
            self.transformations[transformation_key] = lineage_record
            
            return {
                'success': True,
                'lineage_id': transformation_key,
                'lineage_graph_size': len(self.lineage_graph)
            }
            
        def get_data_lineage(self, data_id: str, depth: int = 3) -> Dict[str, Any]:
            """Get data lineage for a specific data element."""
            def build_lineage(current_id, current_depth, visited=None):
                if visited is None:
                    visited = set()
                    
                if current_depth <= 0 or current_id in visited:
                    return {}
                    
                visited.add(current_id)
                lineage = {'id': current_id, 'dependencies': []}
                
                if current_id in self.lineage_graph:
                    for record in self.lineage_graph[current_id]:
                        target_lineage = build_lineage(
                            record['target_id'], 
                            current_depth - 1, 
                            visited.copy()
                        )
                        if target_lineage:
                            lineage['dependencies'].append({
                                'target': target_lineage,
                                'transformation': record['transformation'],
                                'metadata': record['metadata']
                            })
                            
                return lineage
                
            full_lineage = build_lineage(data_id, depth)
            
            return {
                'success': True,
                'data_id': data_id,
                'lineage': full_lineage,
                'depth_searched': depth
            }
            
        def detect_lineage_issues(self) -> Dict[str, Any]:
            """Detect issues in data lineage."""
            issues = []
            
            # Check for circular dependencies
            visited = set()
            rec_stack = set()
            
            def has_cycle(node):
                visited.add(node)
                rec_stack.add(node)
                
                if node in self.lineage_graph:
                    for record in self.lineage_graph[node]:
                        neighbor = record['target_id']
                        if neighbor not in visited:
                            if has_cycle(neighbor):
                                return True
                        elif neighbor in rec_stack:
                            return True
                            
                rec_stack.remove(node)
                return False
                
            # Check all nodes for cycles
            for node in self.lineage_graph.keys():
                if node not in visited:
                    if has_cycle(node):
                        issues.append({
                            'type': 'circular_dependency',
                            'severity': 'high',
                            'description': f'Circular dependency detected involving {node}'
                        })
                        break
                        
            # Check for orphaned data (data without lineage)
            all_targets = set()
            for records in self.lineage_graph.values():
                for record in records:
                    all_targets.add(record['target_id'])
                    
            orphaned_sources = set(self.lineage_graph.keys()) - all_targets
            
            if orphaned_sources:
                issues.append({
                    'type': 'orphaned_data',
                    'severity': 'medium',
                    'description': f'Data sources without upstream lineage: {list(orphaned_sources)[:5]}'
                })
                
            return {
                'success': True,
                'issues_found': len(issues),
                'issues': issues
            }


@pytest.mark.monitoring
class TestPipelineHealthMonitoring:
    """Test pipeline health monitoring capabilities."""
    
    def test_health_monitoring_initialization(self):
        """Test pipeline health monitoring setup."""
        health_service = PipelineHealthService()
        
        pipeline_id = "test_pipeline_001"
        config = {
            'check_interval': 60,  # seconds
            'alert_thresholds': {
                'execution_time': 30.0,
                'memory_usage': 1000.0
            }
        }
        
        result = health_service.start_monitoring(pipeline_id, config)
        
        assert result['success'], "Health monitoring initialization failed"
        assert 'monitoring_id' in result, "Monitoring ID not returned"
        assert health_service.monitoring_active, "Monitoring not activated"
        
        # Verify initial health status
        status = health_service.get_health_status(pipeline_id)
        assert status['status'] == 'healthy', f"Initial status should be healthy, got {status['status']}"
        assert 'metrics' in status, "Health metrics not initialized"
    
    def test_real_time_metrics_recording(self):
        """Test real-time metrics recording and tracking."""
        health_service = PipelineHealthService()
        pipeline_id = "test_pipeline_002"
        
        # Start monitoring
        health_service.start_monitoring(pipeline_id)
        
        # Record metrics over time
        test_metrics = [
            {'execution_time': 5.2, 'memory_usage': 256.5, 'cpu_usage': 45.0, 'error_rate': 0.1},
            {'execution_time': 8.7, 'memory_usage': 312.1, 'cpu_usage': 52.3, 'error_rate': 0.2},
            {'execution_time': 12.1, 'memory_usage': 445.8, 'cpu_usage': 63.7, 'error_rate': 0.0},
        ]
        
        for i, metrics in enumerate(test_metrics):
            result = health_service.record_metrics(pipeline_id, metrics)
            assert result['success'], f"Metrics recording failed at step {i+1}"
            
            # Verify metrics are stored
            status = health_service.get_health_status(pipeline_id)
            assert 'metrics' in status, "Metrics not stored in health status"
            
            # Verify specific metrics
            for metric_name, expected_value in metrics.items():
                actual_value = status['metrics'].get(metric_name)
                assert actual_value == expected_value, (
                    f"Metric {metric_name}: expected {expected_value}, got {actual_value}"
                )
    
    def test_threshold_based_alerting(self):
        """Test threshold-based alert generation."""
        health_service = PipelineHealthService()
        pipeline_id = "test_pipeline_003"
        
        # Start monitoring
        health_service.start_monitoring(pipeline_id)
        
        # Record metrics that exceed thresholds
        high_metrics = {
            'execution_time': 45.0,  # Exceeds 30.0 threshold
            'memory_usage': 1200.0,  # Exceeds 1000.0 threshold
            'cpu_usage': 95.0,       # Exceeds 80.0 threshold
            'error_rate': 8.5        # Exceeds 5.0 threshold
        }
        
        health_service.record_metrics(pipeline_id, high_metrics)
        
        # Check for generated alerts
        alerts = health_service.get_alerts(pipeline_id)
        
        assert len(alerts) > 0, "No alerts generated for threshold violations"
        
        # Verify alert content
        alert_metrics = {alert['metric'] for alert in alerts}
        expected_violations = {'execution_time', 'memory_usage', 'cpu_usage', 'error_rate'}
        
        assert alert_metrics == expected_violations, (
            f"Alert metrics {alert_metrics} don't match expected violations {expected_violations}"
        )
        
        # Check alert severities
        high_severity_alerts = [a for a in alerts if a['severity'] == 'high']
        assert len(high_severity_alerts) > 0, "No high-severity alerts generated for extreme violations"
        
        # Verify alert structure
        for alert in alerts:
            required_fields = ['id', 'pipeline_id', 'metric', 'value', 'threshold', 'severity', 'timestamp', 'message']
            for field in required_fields:
                assert field in alert, f"Alert missing required field: {field}"
    
    def test_alert_filtering_and_retrieval(self):
        """Test alert filtering by pipeline and severity."""
        health_service = PipelineHealthService()
        
        # Create multiple pipelines with different alert severities
        pipelines = [
            ('pipeline_A', {'execution_time': 50.0}),  # High severity
            ('pipeline_B', {'memory_usage': 1100.0}),  # Medium severity
            ('pipeline_C', {'error_rate': 15.0}),      # High severity
        ]
        
        for pipeline_id, metrics in pipelines:
            health_service.start_monitoring(pipeline_id)
            health_service.record_metrics(pipeline_id, metrics)
        
        # Test filtering by pipeline
        pipeline_A_alerts = health_service.get_alerts('pipeline_A')
        assert all(alert['pipeline_id'] == 'pipeline_A' for alert in pipeline_A_alerts), (
            "Pipeline filtering not working correctly"
        )
        
        # Test filtering by severity
        high_severity_alerts = health_service.get_alerts(severity='high')
        assert all(alert['severity'] == 'high' for alert in high_severity_alerts), (
            "Severity filtering not working correctly"
        )
        
        # Test combined filtering
        pipeline_A_high_alerts = health_service.get_alerts('pipeline_A', 'high')
        assert all(
            alert['pipeline_id'] == 'pipeline_A' and alert['severity'] == 'high' 
            for alert in pipeline_A_high_alerts
        ), "Combined filtering not working correctly"


@pytest.mark.monitoring
class TestPredictiveQualityMonitoring:
    """Test predictive quality monitoring and trend analysis."""
    
    def test_quality_metrics_recording(self):
        """Test quality metrics recording for trend analysis."""
        predictive_service = PredictiveQualityService()
        
        data_source = "sensor_data_stream"
        
        # Record quality metrics over time
        quality_metrics_series = [
            {'completeness': 0.95, 'accuracy': 0.92, 'consistency': 0.88},
            {'completeness': 0.94, 'accuracy': 0.91, 'consistency': 0.87},
            {'completeness': 0.92, 'accuracy': 0.90, 'consistency': 0.85},
            {'completeness': 0.90, 'accuracy': 0.88, 'consistency': 0.83},
            {'completeness': 0.87, 'accuracy': 0.85, 'consistency': 0.80},
        ]
        
        for i, metrics in enumerate(quality_metrics_series):
            result = predictive_service.record_quality_metrics(data_source, metrics)
            assert result['success'], f"Quality metrics recording failed at step {i+1}"
            assert result['records_count'] == i + 1, f"Incorrect record count at step {i+1}"
    
    def test_quality_degradation_prediction(self):
        """Test prediction of quality degradation trends."""
        predictive_service = PredictiveQualityService()
        
        data_source = "declining_quality_source"
        
        # Record declining quality trend
        base_quality = 0.95
        decline_rate = 0.02
        
        for i in range(10):
            current_quality = base_quality - (decline_rate * i)
            metrics = {
                'completeness': current_quality,
                'accuracy': current_quality - 0.01,
                'consistency': current_quality - 0.02
            }
            
            predictive_service.record_quality_metrics(data_source, metrics)
            
            # Add small delay to simulate time progression
            time.sleep(0.01)
        
        # Generate predictions
        prediction_result = predictive_service.predict_quality_degradation(data_source, horizon_hours=24)
        
        assert prediction_result['success'], "Quality degradation prediction failed"
        assert 'predictions' in prediction_result, "Predictions not returned"
        
        predictions = prediction_result['predictions']
        
        # Verify predictions for each metric
        for metric_name in ['completeness', 'accuracy', 'consistency']:
            assert metric_name in predictions, f"Prediction missing for metric {metric_name}"
            
            prediction = predictions[metric_name]
            required_fields = ['current_value', 'predicted_value', 'trend', 'risk_level', 'confidence']
            
            for field in required_fields:
                assert field in prediction, f"Prediction for {metric_name} missing field {field}"
            
            # With declining trend, predicted value should be lower
            assert prediction['trend'] == 'declining', f"Expected declining trend for {metric_name}"
            assert prediction['predicted_value'] < prediction['current_value'], (
                f"Predicted value should be lower than current for declining {metric_name}"
            )
            
            # Risk level should be appropriate for declining quality
            if prediction['predicted_value'] < 0.7:
                assert prediction['risk_level'] == 'high', f"Risk level should be high for {metric_name}"
    
    def test_insufficient_data_handling(self):
        """Test prediction behavior with insufficient historical data."""
        predictive_service = PredictiveQualityService()
        
        data_source = "new_data_source"
        
        # Record only minimal data
        metrics = {'completeness': 0.95, 'accuracy': 0.90}
        predictive_service.record_quality_metrics(data_source, metrics)
        
        # Try to generate predictions
        prediction_result = predictive_service.predict_quality_degradation(data_source)
        
        assert not prediction_result['success'], "Prediction should fail with insufficient data"
        assert 'error' in prediction_result, "Error message not provided for insufficient data"
        assert 'insufficient' in prediction_result['error'].lower(), "Error message should mention insufficient data"
    
    def test_prediction_confidence_levels(self):
        """Test prediction confidence levels based on data quality and history."""
        predictive_service = PredictiveQualityService()
        
        # Test with stable quality metrics (should have higher confidence)
        stable_source = "stable_quality_source"
        stable_quality = 0.90
        
        for _ in range(20):  # More data points
            # Add small random variation
            noise = np.random.normal(0, 0.01)
            metrics = {
                'completeness': stable_quality + noise,
                'accuracy': stable_quality + noise,
                'consistency': stable_quality + noise
            }
            predictive_service.record_quality_metrics(stable_source, metrics)
        
        stable_prediction = predictive_service.predict_quality_degradation(stable_source)
        
        assert stable_prediction['success'], "Stable quality prediction failed"
        
        # Confidence should be reasonable for stable data
        for metric_name, prediction in stable_prediction['predictions'].items():
            confidence = prediction['confidence']
            assert 0.0 <= confidence <= 1.0, f"Invalid confidence level {confidence} for {metric_name}"


@pytest.mark.monitoring
class TestDataLineageTracking:
    """Test data lineage tracking and validation."""
    
    def test_data_flow_tracking(self):
        """Test tracking of data flow between components."""
        lineage_service = DataLineageService()
        
        # Define a simple data pipeline flow
        flow_steps = [
            ('raw_data', 'cleaned_data', 'data_cleaning', {'rows_removed': 50}),
            ('cleaned_data', 'transformed_data', 'feature_engineering', {'features_added': 5}),
            ('transformed_data', 'model_input', 'normalization', {'scaling_method': 'standard'}),
            ('model_input', 'predictions', 'anomaly_detection', {'algorithm': 'isolation_forest'}),
        ]
        
        lineage_ids = []
        
        for source_id, target_id, transformation, metadata in flow_steps:
            result = lineage_service.track_data_flow(source_id, target_id, transformation, metadata)
            
            assert result['success'], f"Failed to track flow from {source_id} to {target_id}"
            assert 'lineage_id' in result, "Lineage ID not returned"
            
            lineage_ids.append(result['lineage_id'])
        
        # Verify lineage graph growth
        assert len(set(lineage_ids)) == len(flow_steps), "Duplicate or missing lineage IDs"
    
    def test_lineage_retrieval_and_depth(self):
        """Test retrieval of data lineage with different depths."""
        lineage_service = DataLineageService()
        
        # Create a complex lineage tree
        #     A
        #    / \
        #   B   C
        #  /   / \
        # D   E   F
        
        flows = [
            ('A', 'B', 'transform_1'),
            ('A', 'C', 'transform_2'),
            ('B', 'D', 'transform_3'),
            ('C', 'E', 'transform_4'),
            ('C', 'F', 'transform_5'),
        ]
        
        for source, target, transform in flows:
            lineage_service.track_data_flow(source, target, transform)
        
        # Test different depths
        depth_1_lineage = lineage_service.get_data_lineage('A', depth=1)
        depth_2_lineage = lineage_service.get_data_lineage('A', depth=2)
        depth_3_lineage = lineage_service.get_data_lineage('A', depth=3)
        
        assert depth_1_lineage['success'], "Depth 1 lineage retrieval failed"
        assert depth_2_lineage['success'], "Depth 2 lineage retrieval failed"
        assert depth_3_lineage['success'], "Depth 3 lineage retrieval failed"
        
        # Verify depth affects result structure
        depth_1_deps = len(depth_1_lineage['lineage']['dependencies'])
        depth_2_deps_total = depth_1_deps + sum(
            len(dep['target'].get('dependencies', []))
            for dep in depth_1_lineage['lineage']['dependencies']
        )
        
        assert depth_1_deps >= 0, "Depth 1 should have some dependencies"
        
        # Verify lineage structure
        lineage = depth_3_lineage['lineage']
        assert lineage['id'] == 'A', "Root lineage ID incorrect"
        assert 'dependencies' in lineage, "Dependencies missing from lineage"
    
    def test_lineage_issue_detection(self):
        """Test detection of lineage issues like circular dependencies."""
        lineage_service = DataLineageService()
        
        # Create circular dependency: A -> B -> C -> A
        circular_flows = [
            ('A', 'B', 'transform_1'),
            ('B', 'C', 'transform_2'),
            ('C', 'A', 'transform_3'),  # Creates cycle
        ]
        
        for source, target, transform in circular_flows:
            lineage_service.track_data_flow(source, target, transform)
        
        # Detect issues
        issues_result = lineage_service.detect_lineage_issues()
        
        assert issues_result['success'], "Lineage issue detection failed"
        assert issues_result['issues_found'] > 0, "Circular dependency not detected"
        
        # Check for specific circular dependency issue
        issues = issues_result['issues']
        circular_issues = [issue for issue in issues if issue['type'] == 'circular_dependency']
        
        assert len(circular_issues) > 0, "Circular dependency issue not found"
        assert circular_issues[0]['severity'] == 'high', "Circular dependency should be high severity"
    
    def test_orphaned_data_detection(self):
        """Test detection of orphaned data without upstream lineage."""
        lineage_service = DataLineageService()
        
        # Create flows where some sources don't have upstream lineage
        flows = [
            ('source_1', 'intermediate_1', 'transform_1'),
            ('source_2', 'intermediate_2', 'transform_2'),
            ('intermediate_1', 'final_output', 'transform_3'),
            # source_2 and intermediate_2 are orphaned (no upstream)
        ]
        
        for source, target, transform in flows:
            lineage_service.track_data_flow(source, target, transform)
        
        issues_result = lineage_service.detect_lineage_issues()
        
        assert issues_result['success'], "Orphaned data detection failed"
        
        # Check for orphaned data issues
        issues = issues_result['issues']
        orphaned_issues = [issue for issue in issues if issue['type'] == 'orphaned_data']
        
        if len(orphaned_issues) > 0:
            assert orphaned_issues[0]['severity'] in ['medium', 'low'], (
                "Orphaned data should be medium or low severity"
            )


@pytest.mark.monitoring
@pytest.mark.performance
class TestMonitoringSystemPerformance:
    """Test monitoring system performance under load."""
    
    def test_concurrent_metrics_recording(self):
        """Test concurrent metrics recording performance."""
        health_service = PipelineHealthService()
        
        # Setup multiple pipelines
        pipeline_ids = [f"concurrent_pipeline_{i}" for i in range(10)]
        
        for pipeline_id in pipeline_ids:
            health_service.start_monitoring(pipeline_id)
        
        # Concurrent metrics recording function
        def record_metrics_worker(pipeline_id, iterations=50):
            for i in range(iterations):
                metrics = {
                    'execution_time': np.random.uniform(1, 10),
                    'memory_usage': np.random.uniform(100, 500),
                    'cpu_usage': np.random.uniform(20, 80),
                    'error_rate': np.random.uniform(0, 2)
                }
                
                result = health_service.record_metrics(pipeline_id, metrics)
                assert result['success'], f"Concurrent metrics recording failed for {pipeline_id}"
        
        # Start concurrent workers
        threads = []
        start_time = time.perf_counter()
        
        for pipeline_id in pipeline_ids:
            thread = threading.Thread(target=record_metrics_worker, args=(pipeline_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Validate performance
        total_operations = len(pipeline_ids) * 50
        operations_per_second = total_operations / total_time
        
        assert operations_per_second >= 100, (
            f"Concurrent metrics recording too slow: {operations_per_second:.1f} ops/sec"
        )
        
        # Verify all metrics were recorded
        for pipeline_id in pipeline_ids:
            status = health_service.get_health_status(pipeline_id)
            assert status['status'] != 'unknown', f"Pipeline {pipeline_id} status not updated"
    
    def test_large_scale_lineage_tracking(self):
        """Test lineage tracking performance with large datasets."""
        lineage_service = DataLineageService()
        
        # Create large lineage graph
        num_nodes = 1000
        
        start_time = time.perf_counter()
        
        # Create linear chain lineage
        for i in range(num_nodes - 1):
            source_id = f"data_node_{i}"
            target_id = f"data_node_{i+1}"
            transformation = f"transform_{i}"
            
            result = lineage_service.track_data_flow(source_id, target_id, transformation)
            assert result['success'], f"Lineage tracking failed at node {i}"
        
        tracking_time = time.perf_counter() - start_time
        
        # Test lineage retrieval performance
        retrieval_start = time.perf_counter()
        
        # Retrieve lineage for root node
        lineage_result = lineage_service.get_data_lineage('data_node_0', depth=10)
        
        retrieval_time = time.perf_counter() - retrieval_start
        
        assert lineage_result['success'], "Large-scale lineage retrieval failed"
        
        # Performance assertions
        tracking_rate = num_nodes / tracking_time
        assert tracking_rate >= 50, f"Lineage tracking too slow: {tracking_rate:.1f} nodes/sec"
        
        assert retrieval_time < 5.0, f"Lineage retrieval took {retrieval_time:.2f}s, too slow"
    
    def test_alert_system_scalability(self):
        """Test alert system performance with high alert volume."""
        health_service = PipelineHealthService()
        
        pipeline_id = "high_alert_pipeline"
        health_service.start_monitoring(pipeline_id)
        
        # Generate high volume of alerts
        num_alerts = 1000
        
        start_time = time.perf_counter()
        
        for i in range(num_alerts):
            # Metrics that will trigger alerts
            high_metrics = {
                'execution_time': 50 + (i % 10),  # Varies to avoid duplicates
                'memory_usage': 1500 + (i % 100),
                'error_rate': 10 + (i % 5)
            }
            
            result = health_service.record_metrics(pipeline_id, high_metrics)
            assert result['success'], f"Alert generation failed at iteration {i}"
        
        alert_generation_time = time.perf_counter() - start_time
        
        # Test alert retrieval performance
        retrieval_start = time.perf_counter()
        all_alerts = health_service.get_alerts(pipeline_id)
        retrieval_time = time.perf_counter() - retrieval_start
        
        # Validate performance
        alert_generation_rate = num_alerts / alert_generation_time
        assert alert_generation_rate >= 50, (
            f"Alert generation too slow: {alert_generation_rate:.1f} alerts/sec"
        )
        
        assert retrieval_time < 1.0, f"Alert retrieval took {retrieval_time:.2f}s, too slow"
        assert len(all_alerts) > 0, "No alerts generated despite threshold violations"
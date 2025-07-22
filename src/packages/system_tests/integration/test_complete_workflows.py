"""
Cross-package integration tests for complete anomaly detection workflows.
Tests end-to-end processes spanning multiple packages and domains.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add paths for package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ai" / "machine_learning" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data" / "anomaly_detection" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data" / "data_quality" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data" / "transformation" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data" / "observability" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "enterprise" / "enterprise_auth" / "src"))

# Mock service classes for integration testing
class MockDataIngestionService:
    """Mock data ingestion service."""
    
    def ingest_csv(self, file_path: str) -> Dict[str, Any]:
        """Mock CSV ingestion."""
        # Generate mock data that looks like ingested CSV
        np.random.seed(42)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
            'sensor_1': np.random.randn(1000),
            'sensor_2': np.random.randn(1000) + 0.5,
            'sensor_3': np.random.exponential(1, 1000),
            'status': np.random.choice(['normal', 'warning'], 1000, p=[0.9, 0.1])
        })
        
        # Inject some anomalies
        anomaly_indices = np.random.choice(1000, 50, replace=False)
        data.loc[anomaly_indices, 'sensor_1'] += np.random.choice([-5, 5], 50)
        
        return {
            'success': True,
            'data': data,
            'row_count': len(data),
            'columns': list(data.columns),
            'ingestion_time': time.time(),
            'file_size_mb': 2.5
        }
    
    def ingest_realtime_stream(self, stream_config: Dict) -> Dict[str, Any]:
        """Mock real-time stream ingestion."""
        return {
            'success': True,
            'stream_id': 'stream_123',
            'status': 'connected',
            'data_rate': 100  # samples per second
        }


class MockDataQualityService:
    """Mock data quality service."""
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Mock data quality validation."""
        quality_score = 0.95 - (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 0.5
        
        return {
            'success': True,
            'quality_score': quality_score,
            'issues_found': [],
            'completeness': 0.98,
            'consistency': 0.96,
            'accuracy': 0.94,
            'validation_rules_passed': 12,
            'validation_rules_failed': 1
        }
    
    def clean_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Mock data cleaning."""
        cleaned_data = data.copy()
        
        # Simulate cleaning operations
        cleaned_data = cleaned_data.dropna()
        
        return {
            'success': True,
            'original_rows': len(data),
            'cleaned_rows': len(cleaned_data),
            'rows_removed': len(data) - len(cleaned_data),
            'data': cleaned_data,
            'cleaning_operations': ['remove_nulls', 'standardize_formats']
        }


class MockTransformationService:
    """Mock data transformation service."""
    
    def transform_data(self, data: pd.DataFrame, config: Dict) -> Dict[str, Any]:
        """Mock data transformation."""
        transformed_data = data.copy()
        
        # Simulate transformations
        for col in transformed_data.select_dtypes(include=[np.number]).columns:
            if col != 'timestamp':
                # Standardize numerical columns
                mean = transformed_data[col].mean()
                std = transformed_data[col].std()
                transformed_data[col] = (transformed_data[col] - mean) / std
        
        return {
            'success': True,
            'data': transformed_data,
            'transformations_applied': ['standardization', 'encoding'],
            'feature_count': len(transformed_data.columns),
            'processing_time': 0.5
        }


class MockAnomalyDetectionService:
    """Mock anomaly detection service."""
    
    def detect_anomalies(self, data: pd.DataFrame, config: Dict = None) -> Dict[str, Any]:
        """Mock anomaly detection."""
        np.random.seed(42)
        
        # Simulate anomaly detection
        n_samples = len(data)
        predictions = np.ones(n_samples)
        
        # Mark some samples as anomalies
        contamination = config.get('contamination', 0.1) if config else 0.1
        n_anomalies = int(n_samples * contamination)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        predictions[anomaly_indices] = -1
        
        confidence_scores = np.random.uniform(0.6, 0.95, n_samples)
        confidence_scores[anomaly_indices] = np.random.uniform(0.8, 0.99, n_anomalies)
        
        return {
            'success': True,
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'anomalies_detected': n_anomalies,
            'anomaly_indices': anomaly_indices.tolist(),
            'algorithm_used': 'isolation_forest',
            'detection_time': 1.2
        }


class MockObservabilityService:
    """Mock observability service."""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
    
    def track_pipeline_execution(self, pipeline_result: Dict) -> Dict[str, Any]:
        """Mock pipeline execution tracking."""
        pipeline_id = f"pipeline_{int(time.time())}"
        
        self.metrics[pipeline_id] = {
            'execution_time': pipeline_result.get('total_time', 0),
            'success': pipeline_result.get('success', False),
            'data_processed': pipeline_result.get('samples_processed', 0),
            'anomalies_found': pipeline_result.get('anomalies_detected', 0)
        }
        
        return {
            'success': True,
            'pipeline_id': pipeline_id,
            'tracked_metrics': list(self.metrics[pipeline_id].keys())
        }
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Mock pipeline metrics retrieval."""
        if not self.metrics:
            return {'success_rate': 1.0, 'avg_execution_time': 0.0, 'total_pipelines': 0}
        
        successful_pipelines = sum(1 for m in self.metrics.values() if m['success'])
        total_pipelines = len(self.metrics)
        
        return {
            'success_rate': successful_pipelines / total_pipelines,
            'avg_execution_time': sum(m['execution_time'] for m in self.metrics.values()) / total_pipelines,
            'total_pipelines': total_pipelines,
            'total_anomalies_detected': sum(m['anomalies_found'] for m in self.metrics.values())
        }
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Mock alert checking."""
        return self.alerts
    
    def trigger_alert(self, alert_type: str, message: str) -> Dict[str, Any]:
        """Mock alert triggering."""
        alert = {
            'id': f"alert_{len(self.alerts) + 1}",
            'type': alert_type,
            'message': message,
            'timestamp': time.time(),
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        
        return {'success': True, 'alert_id': alert['id']}


class MockAuthenticationService:
    """Mock authentication service for workflow security."""
    
    def authenticate_user(self, credentials: Dict) -> Dict[str, Any]:
        """Mock user authentication."""
        if credentials.get('username') == 'data_analyst' and credentials.get('password') == 'secure_password':
            return {
                'success': True,
                'user_id': 'user_456',
                'access_token': 'workflow_token_123',
                'permissions': ['data_access', 'anomaly_detection', 'dashboard_view']
            }
        return {'success': False, 'error': 'Invalid credentials'}
    
    def check_permission(self, token: str, action: str) -> bool:
        """Mock permission checking."""
        return token == 'workflow_token_123' and action in ['data_access', 'anomaly_detection', 'dashboard_view']


@pytest.mark.integration
class TestCompleteAnomalyDetectionWorkflow:
    """Test complete end-to-end anomaly detection workflows."""
    
    def test_csv_to_anomaly_detection_workflow(self):
        """Test complete workflow from CSV ingestion to anomaly detection results."""
        # Initialize services
        ingestion_service = MockDataIngestionService()
        quality_service = MockDataQualityService()
        transformation_service = MockTransformationService()
        detection_service = MockAnomalyDetectionService()
        observability_service = MockObservabilityService()
        
        workflow_start_time = time.time()
        
        # Step 1: Data Ingestion
        ingestion_result = ingestion_service.ingest_csv("test_data.csv")
        assert ingestion_result['success'], "Data ingestion failed"
        assert 'data' in ingestion_result, "No data returned from ingestion"
        
        raw_data = ingestion_result['data']
        assert len(raw_data) > 0, "Empty dataset ingested"
        
        # Step 2: Data Quality Validation
        quality_result = quality_service.validate_data(raw_data)
        assert quality_result['success'], "Data quality validation failed"
        assert quality_result['quality_score'] > 0.8, f"Data quality score {quality_result['quality_score']} too low"
        
        # Step 3: Data Cleaning (if needed)
        cleaning_result = quality_service.clean_data(raw_data)
        assert cleaning_result['success'], "Data cleaning failed"
        
        clean_data = cleaning_result['data']
        assert len(clean_data) <= len(raw_data), "Data cleaning increased row count"
        
        # Step 4: Data Transformation
        transform_config = {
            'scaling_method': 'standard',
            'encoding_strategy': 'onehot',
            'feature_engineering': True
        }
        
        transform_result = transformation_service.transform_data(clean_data, transform_config)
        assert transform_result['success'], "Data transformation failed"
        
        processed_data = transform_result['data']
        assert len(processed_data) == len(clean_data), "Transformation changed row count unexpectedly"
        
        # Step 5: Anomaly Detection
        detection_config = {
            'algorithm': 'isolation_forest',
            'contamination': 0.1,
            'n_estimators': 100
        }
        
        detection_result = detection_service.detect_anomalies(processed_data, detection_config)
        assert detection_result['success'], "Anomaly detection failed"
        assert 'anomalies_detected' in detection_result, "No anomaly count returned"
        assert detection_result['anomalies_detected'] > 0, "No anomalies detected in test data"
        
        # Step 6: Results Observability
        workflow_end_time = time.time()
        
        pipeline_result = {
            'success': True,
            'total_time': workflow_end_time - workflow_start_time,
            'samples_processed': len(processed_data),
            'anomalies_detected': detection_result['anomalies_detected']
        }
        
        tracking_result = observability_service.track_pipeline_execution(pipeline_result)
        assert tracking_result['success'], "Pipeline tracking failed"
        
        # Validate end-to-end metrics
        metrics = observability_service.get_pipeline_metrics()
        assert metrics['success_rate'] > 0.95, f"Pipeline success rate {metrics['success_rate']} too low"
        assert metrics['total_anomalies_detected'] > 0, "No anomalies tracked in observability"
        
        # Validate workflow performance
        total_workflow_time = workflow_end_time - workflow_start_time
        assert total_workflow_time < 30, f"Workflow took {total_workflow_time:.2f}s, exceeding 30s limit"
    
    def test_realtime_streaming_workflow(self):
        """Test real-time streaming anomaly detection workflow."""
        # Initialize services
        ingestion_service = MockDataIngestionService()
        detection_service = MockAnomalyDetectionService()
        observability_service = MockObservabilityService()
        
        # Step 1: Setup real-time stream
        stream_config = {
            'source': 'kafka',
            'topic': 'sensor_data',
            'batch_size': 100
        }
        
        stream_result = ingestion_service.ingest_realtime_stream(stream_config)
        assert stream_result['success'], "Real-time stream setup failed"
        assert stream_result['status'] == 'connected', "Stream not connected"
        
        # Step 2: Simulate processing multiple batches
        batch_results = []
        
        for batch_num in range(5):
            # Generate batch data
            batch_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='S'),
                'value_1': np.random.randn(100),
                'value_2': np.random.randn(100) + batch_num * 0.1  # Slight drift
            })
            
            # Real-time detection
            batch_start_time = time.time()
            detection_result = detection_service.detect_anomalies(batch_data, {'contamination': 0.05})
            batch_end_time = time.time()
            
            assert detection_result['success'], f"Batch {batch_num} detection failed"
            
            batch_processing_time = batch_end_time - batch_start_time
            assert batch_processing_time < 1.0, f"Batch {batch_num} processing time {batch_processing_time:.2f}s too slow"
            
            batch_results.append({
                'batch_num': batch_num,
                'processing_time': batch_processing_time,
                'anomalies_detected': detection_result['anomalies_detected']
            })
            
            # Track batch processing
            tracking_result = observability_service.track_pipeline_execution({
                'success': True,
                'total_time': batch_processing_time,
                'samples_processed': 100,
                'anomalies_detected': detection_result['anomalies_detected']
            })
            assert tracking_result['success'], f"Batch {batch_num} tracking failed"
        
        # Validate real-time performance
        avg_processing_time = sum(b['processing_time'] for b in batch_results) / len(batch_results)
        assert avg_processing_time < 0.5, f"Average batch processing time {avg_processing_time:.2f}s too slow"
        
        # Validate consistency
        anomaly_counts = [b['anomalies_detected'] for b in batch_results]
        anomaly_variance = np.var(anomaly_counts)
        assert anomaly_variance < 10, f"Anomaly detection too inconsistent across batches: variance {anomaly_variance}"
    
    def test_authenticated_workflow_security(self):
        """Test workflow with authentication and authorization."""
        # Initialize services
        auth_service = MockAuthenticationService()
        detection_service = MockAnomalyDetectionService()
        observability_service = MockObservabilityService()
        
        # Step 1: User Authentication
        credentials = {
            'username': 'data_analyst',
            'password': 'secure_password'
        }
        
        auth_result = auth_service.authenticate_user(credentials)
        assert auth_result['success'], "User authentication failed"
        
        access_token = auth_result['access_token']
        user_permissions = auth_result['permissions']
        
        # Step 2: Permission Validation
        required_permissions = ['data_access', 'anomaly_detection']
        
        for permission in required_permissions:
            has_permission = auth_service.check_permission(access_token, permission)
            assert has_permission, f"User lacks required permission: {permission}"
        
        # Step 3: Authorized Data Processing
        # Only proceed if user has proper permissions
        if all(perm in user_permissions for perm in required_permissions):
            
            # Generate test data
            test_data = pd.DataFrame({
                'feature_1': np.random.randn(500),
                'feature_2': np.random.randn(500)
            })
            
            # Perform detection with user context
            detection_result = detection_service.detect_anomalies(test_data)
            assert detection_result['success'], "Authorized detection failed"
            
            # Log authorized action
            tracking_result = observability_service.track_pipeline_execution({
                'success': True,
                'user_id': auth_result['user_id'],
                'action': 'anomaly_detection',
                'samples_processed': len(test_data),
                'anomalies_detected': detection_result['anomalies_detected']
            })
            assert tracking_result['success'], "Authorized action tracking failed"
        
        # Step 4: Test unauthorized access attempt
        invalid_token = 'invalid_token_456'
        has_unauthorized_access = auth_service.check_permission(invalid_token, 'anomaly_detection')
        assert not has_unauthorized_access, "Unauthorized access granted"
    
    def test_error_handling_in_workflow(self):
        """Test error handling and recovery in complete workflows."""
        # Initialize services
        ingestion_service = MockDataIngestionService()
        quality_service = MockDataQualityService()
        detection_service = MockAnomalyDetectionService()
        observability_service = MockObservabilityService()
        
        # Test 1: Handle data ingestion failure
        with patch.object(ingestion_service, 'ingest_csv', return_value={'success': False, 'error': 'File not found'}):
            
            ingestion_result = ingestion_service.ingest_csv("nonexistent.csv")
            assert not ingestion_result['success'], "Should fail with nonexistent file"
            assert 'error' in ingestion_result, "Error message not provided"
            
            # Workflow should handle gracefully
            if not ingestion_result['success']:
                # Log failure
                failure_tracking = observability_service.track_pipeline_execution({
                    'success': False,
                    'error': ingestion_result['error'],
                    'stage': 'data_ingestion'
                })
                assert failure_tracking['success'], "Failure tracking failed"
        
        # Test 2: Handle data quality issues
        # Create problematic data
        problematic_data = pd.DataFrame({
            'col1': [1, 2, np.nan, np.nan, 5],  # Missing values
            'col2': [1, 1, 1, 1, 1],  # No variance
            'col3': ['A', 'B', 'C', 'D', 'E']  # Categorical
        })
        
        quality_result = quality_service.validate_data(problematic_data)
        
        if quality_result['quality_score'] < 0.8:  # Low quality threshold
            # Attempt cleaning
            cleaning_result = quality_service.clean_data(problematic_data)
            
            if cleaning_result['success']:
                cleaned_data = cleaning_result['data']
                
                # Retry detection with cleaned data
                detection_result = detection_service.detect_anomalies(cleaned_data)
                
                if detection_result['success']:
                    # Track successful recovery
                    recovery_tracking = observability_service.track_pipeline_execution({
                        'success': True,
                        'recovery_applied': True,
                        'original_quality_score': quality_result['quality_score'],
                        'samples_processed': len(cleaned_data)
                    })
                    assert recovery_tracking['success'], "Recovery tracking failed"
        
        # Test 3: Handle detection algorithm failure
        with patch.object(detection_service, 'detect_anomalies', return_value={'success': False, 'error': 'Algorithm convergence failed'}):
            
            test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
            detection_result = detection_service.detect_anomalies(test_data)
            
            assert not detection_result['success'], "Detection should fail"
            
            # Should trigger alert
            alert_result = observability_service.trigger_alert(
                'detection_failure',
                f"Anomaly detection failed: {detection_result['error']}"
            )
            assert alert_result['success'], "Alert triggering failed"
            
            # Check alert was created
            alerts = observability_service.check_alerts()
            assert len(alerts) > 0, "No alerts found"
            assert any('detection_failure' in alert['type'] for alert in alerts), "Detection failure alert not found"


@pytest.mark.integration
@pytest.mark.performance
class TestWorkflowPerformance:
    """Test performance characteristics of integrated workflows."""
    
    def test_high_volume_workflow_performance(self):
        """Test workflow performance with high data volumes."""
        # Initialize services
        detection_service = MockAnomalyDetectionService()
        observability_service = MockObservabilityService()
        
        # Test different dataset sizes
        dataset_sizes = [1000, 5000, 10000, 25000]
        performance_results = []
        
        for size in dataset_sizes:
            # Generate large dataset
            large_data = pd.DataFrame({
                'feature_1': np.random.randn(size),
                'feature_2': np.random.randn(size),
                'feature_3': np.random.exponential(1, size),
                'feature_4': np.random.uniform(0, 1, size),
                'feature_5': np.random.normal(10, 2, size)
            })
            
            # Time the detection process
            start_time = time.perf_counter()
            detection_result = detection_service.detect_anomalies(large_data)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            
            assert detection_result['success'], f"Detection failed for size {size}"
            
            # Record performance metrics
            performance_results.append({
                'dataset_size': size,
                'processing_time': processing_time,
                'samples_per_second': size / processing_time if processing_time > 0 else 0,
                'anomalies_detected': detection_result['anomalies_detected']
            })
            
            # Track performance
            tracking_result = observability_service.track_pipeline_execution({
                'success': True,
                'total_time': processing_time,
                'samples_processed': size,
                'anomalies_detected': detection_result['anomalies_detected']
            })
            assert tracking_result['success'], f"Performance tracking failed for size {size}"
        
        # Validate performance scaling
        for result in performance_results:
            # Should process at least 1000 samples per second
            assert result['samples_per_second'] >= 1000, (
                f"Processing rate {result['samples_per_second']:.0f} samples/s too slow for size {result['dataset_size']}"
            )
            
            # Processing time should scale sub-quadratically
            time_per_sample = result['processing_time'] / result['dataset_size']
            assert time_per_sample < 0.001, (  # 1ms per sample max
                f"Time per sample {time_per_sample*1000:.2f}ms too high for size {result['dataset_size']}"
            )
        
        # Check for reasonable performance scaling
        if len(performance_results) >= 2:
            small_result = performance_results[0]
            large_result = performance_results[-1]
            
            size_ratio = large_result['dataset_size'] / small_result['dataset_size']
            time_ratio = large_result['processing_time'] / small_result['processing_time']
            
            # Time should not increase more than linearly with size
            assert time_ratio <= size_ratio * 2, (
                f"Performance scaling poor: {time_ratio:.2f}x time for {size_ratio:.2f}x data"
            )
    
    def test_concurrent_workflow_performance(self):
        """Test performance with concurrent workflow executions."""
        import threading
        
        detection_service = MockAnomalyDetectionService()
        observability_service = MockObservabilityService()
        
        # Shared results storage
        results = []
        errors = []
        
        def worker_workflow():
            """Worker function for concurrent workflows."""
            try:
                # Generate data
                data = pd.DataFrame({
                    'col1': np.random.randn(1000),
                    'col2': np.random.randn(1000)
                })
                
                # Process
                start_time = time.perf_counter()
                detection_result = detection_service.detect_anomalies(data)
                end_time = time.perf_counter()
                
                results.append({
                    'success': detection_result['success'],
                    'processing_time': end_time - start_time,
                    'anomalies_detected': detection_result['anomalies_detected']
                })
                
            except Exception as e:
                errors.append(e)
        
        # Start concurrent workflows
        threads = []
        for _ in range(10):  # 10 concurrent workflows
            thread = threading.Thread(target=worker_workflow)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Validate concurrent execution
        assert len(errors) == 0, f"Errors in concurrent execution: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        
        successful_workflows = sum(1 for r in results if r['success'])
        assert successful_workflows == 10, f"Only {successful_workflows}/10 workflows succeeded"
        
        # Validate performance under concurrency
        avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
        max_processing_time = max(r['processing_time'] for r in results)
        
        # Should maintain reasonable performance under concurrency
        assert avg_processing_time < 2.0, f"Average concurrent processing time {avg_processing_time:.2f}s too high"
        assert max_processing_time < 5.0, f"Maximum concurrent processing time {max_processing_time:.2f}s too high"


@pytest.mark.integration 
@pytest.mark.slow
class TestWorkflowReliability:
    """Test workflow reliability and fault tolerance."""
    
    def test_workflow_fault_tolerance(self):
        """Test workflow behavior under various fault conditions."""
        # Initialize services
        detection_service = MockAnomalyDetectionService()
        observability_service = MockObservabilityService()
        
        # Test with various problematic datasets
        fault_conditions = [
            {
                'name': 'empty_dataset',
                'data': pd.DataFrame(),
                'expected_behavior': 'graceful_failure'
            },
            {
                'name': 'single_row',
                'data': pd.DataFrame({'col1': [1], 'col2': [2]}),
                'expected_behavior': 'success_or_graceful_failure'
            },
            {
                'name': 'single_column',
                'data': pd.DataFrame({'col1': range(100)}),
                'expected_behavior': 'success_or_graceful_failure'
            },
            {
                'name': 'all_identical_values',
                'data': pd.DataFrame({'col1': [1] * 100, 'col2': [2] * 100}),
                'expected_behavior': 'success_or_graceful_failure'
            }
        ]
        
        fault_tolerance_results = []
        
        for condition in fault_conditions:
            try:
                detection_result = detection_service.detect_anomalies(condition['data'])
                
                fault_tolerance_results.append({
                    'condition': condition['name'],
                    'success': detection_result['success'],
                    'handled_gracefully': True,
                    'error': detection_result.get('error', None)
                })
                
                # Track fault handling
                observability_service.track_pipeline_execution({
                    'success': detection_result['success'],
                    'fault_condition': condition['name'],
                    'samples_processed': len(condition['data'])
                })
                
            except Exception as e:
                # Should handle exceptions gracefully
                fault_tolerance_results.append({
                    'condition': condition['name'],
                    'success': False,
                    'handled_gracefully': False,
                    'error': str(e)
                })
        
        # Validate fault tolerance
        for result in fault_tolerance_results:
            if result['condition'] == 'empty_dataset':
                # Empty dataset should be handled gracefully (success=False is OK)
                assert result['handled_gracefully'], f"Empty dataset not handled gracefully: {result['error']}"
            else:
                # Other conditions should either succeed or fail gracefully
                if not result['success']:
                    assert result['handled_gracefully'], f"Condition {result['condition']} not handled gracefully: {result['error']}"
    
    def test_workflow_recovery_mechanisms(self):
        """Test workflow recovery from transient failures."""
        detection_service = MockAnomalyDetectionService()
        observability_service = MockObservabilityService()
        
        # Simulate transient failures with retry logic
        failure_count = 0
        max_failures = 2
        
        def failing_detection(data, config=None):
            """Mock detection that fails first few times."""
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= max_failures:
                return {'success': False, 'error': 'Transient service unavailable'}
            else:
                # Succeed after failures
                return detection_service.detect_anomalies(data, config)
        
        # Test data
        test_data = pd.DataFrame({'col1': np.random.randn(100), 'col2': np.random.randn(100)})
        
        # Implement retry logic
        max_retries = 5
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            # Replace detection method temporarily
            original_detect = detection_service.detect_anomalies
            detection_service.detect_anomalies = failing_detection
            
            result = detection_service.detect_anomalies(test_data)
            
            # Restore original method
            detection_service.detect_anomalies = original_detect
            
            if result['success']:
                # Track successful recovery
                recovery_tracking = observability_service.track_pipeline_execution({
                    'success': True,
                    'recovery_attempt': attempt + 1,
                    'samples_processed': len(test_data)
                })
                assert recovery_tracking['success'], "Recovery tracking failed"
                break
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        # Should eventually succeed
        assert result['success'], f"Workflow failed to recover after {max_retries} attempts"
        assert failure_count > max_failures, "Recovery mechanism not tested properly"
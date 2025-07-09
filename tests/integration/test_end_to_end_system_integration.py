"""End-to-end system integration tests for Pynomaly."""

import pytest
import asyncio
import numpy as np
import pandas as pd
import tempfile
import shutil
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from pynomaly.domain.entities import Dataset, DetectionResult, Anomaly, Detector, Model
from pynomaly.domain.value_objects import AnomalyScore, AnomalyType, SemanticVersion
from pynomaly.features.advanced_analytics import get_analytics_engine, AnalysisType
from pynomaly.features.model_management import get_model_registry, ModelStatus, DeploymentConfig
from pynomaly.features.real_time_processing import get_stream_processor, StreamingConfig, ProcessingMode
from pynomaly.features.api_gateway import get_api_gateway, APIRequest, HTTPMethod
from pynomaly.features.feature_engineering import FeatureEngineer


class MockProductionDetector(Detector):
    """Production-grade mock detector for end-to-end testing."""
    
    def __init__(self, algorithm: str = "production_detector"):
        self.algorithm = algorithm
        self.is_fitted = False
        self.model_params = {
            "contamination": 0.1,
            "n_estimators": 100,
            "max_samples": "auto",
            "random_state": 42
        }
        self.performance_metrics = {
            "precision": 0.85,
            "recall": 0.78,
            "f1_score": 0.81,
            "accuracy": 0.92
        }
        self.training_time = 0.0
        self.prediction_count = 0
    
    def fit(self, dataset: Dataset) -> None:
        """Fit the detector to training data."""
        start_time = datetime.now()
        
        # Simulate training time
        import time
        time.sleep(0.1)  # Simulate training
        
        self.is_fitted = True
        self.training_time = (datetime.now() - start_time).total_seconds()
    
    def predict(self, dataset: Dataset) -> DetectionResult:
        """Predict anomalies in the dataset."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        start_time = datetime.now()
        self.prediction_count += 1
        
        anomalies = []
        
        # Simulate realistic anomaly detection
        # Look for outliers in numeric columns
        numeric_cols = dataset.data.select_dtypes(include=[np.number]).columns
        
        for idx, row in dataset.data.iterrows():
            anomaly_score = 0.0
            feature_scores = {}
            
            # Calculate anomaly score based on feature deviations
            for col in numeric_cols:
                if pd.notna(row[col]):
                    col_mean = dataset.data[col].mean()
                    col_std = dataset.data[col].std()
                    
                    if col_std > 0:
                        z_score = abs((row[col] - col_mean) / col_std)
                        feature_scores[col] = z_score
                        anomaly_score += z_score
            
            # Normalize anomaly score
            if len(feature_scores) > 0:
                anomaly_score = anomaly_score / len(feature_scores)
                anomaly_score = min(anomaly_score / 4, 1.0)  # Cap at 1.0
                
                # Create anomaly if score exceeds threshold
                if anomaly_score > self.model_params["contamination"] * 3:
                    anomaly = Anomaly(
                        id=f"anomaly_{self.prediction_count}_{idx}",
                        score=AnomalyScore(anomaly_score),
                        type=AnomalyType.POINT,
                        timestamp=datetime.now(),
                        data_point=row.to_dict(),
                        feature_scores=feature_scores
                    )
                    anomalies.append(anomaly)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return DetectionResult(
            anomalies=anomalies,
            threshold=self.model_params["contamination"] * 3,
            metadata={
                "algorithm": self.algorithm,
                "execution_time_ms": execution_time,
                "prediction_count": self.prediction_count,
                "features_analyzed": list(numeric_cols),
                "model_params": self.model_params,
                "performance_metrics": self.performance_metrics,
            }
        )
    
    async def detect(self, dataset: Dataset) -> DetectionResult:
        """Async version of predict for real-time processing."""
        return self.predict(dataset)


@pytest.fixture
def production_dataset():
    """Create production-like dataset for testing."""
    # Generate 1 year of hourly data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    n_samples = len(dates)
    
    # Create realistic time series with patterns
    time_index = np.arange(n_samples)
    
    # Base signal with trend and seasonality
    trend = time_index * 0.001
    daily_seasonal = np.sin(2 * np.pi * time_index / 24) * 5
    weekly_seasonal = np.sin(2 * np.pi * time_index / (24 * 7)) * 3
    yearly_seasonal = np.sin(2 * np.pi * time_index / (24 * 365)) * 10
    
    # Add noise
    noise = np.random.normal(0, 1, n_samples)
    
    # Create main metric
    main_metric = 100 + trend + daily_seasonal + weekly_seasonal + yearly_seasonal + noise
    
    # Add anomalies at specific times
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    main_metric[anomaly_indices] += np.random.choice([-1, 1], len(anomaly_indices)) * np.random.uniform(20, 50, len(anomaly_indices))
    
    # Create supporting metrics
    data = pd.DataFrame({
        'timestamp': dates,
        'cpu_usage': np.clip(main_metric * 0.8 + np.random.normal(0, 5, n_samples), 0, 100),
        'memory_usage': np.clip(main_metric * 0.7 + np.random.normal(0, 8, n_samples), 0, 100),
        'network_io': np.maximum(0, main_metric * 0.5 + np.random.normal(0, 10, n_samples)),
        'disk_io': np.maximum(0, main_metric * 0.6 + np.random.normal(0, 7, n_samples)),
        'response_time': np.maximum(0, main_metric * 0.02 + np.random.normal(0, 0.5, n_samples)),
        'error_rate': np.clip(np.random.exponential(0.5, n_samples), 0, 10),
        'service_id': np.random.choice(['web-1', 'web-2', 'api-1', 'api-2', 'db-1'], n_samples),
        'datacenter': np.random.choice(['us-east-1', 'us-west-2', 'eu-west-1'], n_samples),
        'environment': np.random.choice(['production', 'staging'], n_samples, p=[0.8, 0.2]),
    })
    
    return Dataset(
        name="production_system_metrics",
        data=data,
        description="Production system metrics for end-to-end testing",
        target_column="cpu_usage"
    )


@pytest.fixture
def temp_system_dir():
    """Create temporary directory for system testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
class TestEndToEndDataPipeline:
    """Test complete data processing pipeline."""
    
    async def test_complete_data_pipeline(self, production_dataset, temp_system_dir):
        """Test end-to-end data processing pipeline."""
        # Step 1: Feature Engineering
        feature_engineer = FeatureEngineer()
        
        feature_config = [
            {
                'step': 'extract_temporal',
                'params': {
                    'timestamp_col': 'timestamp'
                }
            },
            {
                'step': 'extract_statistical',
                'params': {
                    'numeric_columns': ['cpu_usage', 'memory_usage', 'network_io'],
                    'window_size': 24  # 24-hour rolling window
                }
            },
            {
                'step': 'normalize',
                'params': {
                    'columns': ['cpu_usage', 'memory_usage', 'network_io', 'disk_io', 'response_time'],
                    'method': 'zscore'
                }
            },
            {
                'step': 'encode_categorical',
                'params': {
                    'columns': ['service_id', 'datacenter', 'environment'],
                    'method': 'onehot'
                }
            },
            {
                'step': 'select_by_variance',
                'params': {
                    'threshold': 0.01
                }
            }
        ]
        
        engineered_dataset = await feature_engineer.engineer_features(
            production_dataset, 
            feature_config
        )
        
        # Verify feature engineering
        assert len(engineered_dataset.data.columns) > len(production_dataset.data.columns)
        assert 'hour' in engineered_dataset.data.columns
        assert 'cpu_usage_rolling_mean' in engineered_dataset.data.columns
        
        # Step 2: Model Training
        detector = MockProductionDetector()
        detector.fit(engineered_dataset)
        
        assert detector.is_fitted
        assert detector.training_time > 0
        
        # Step 3: Model Registration
        model_registry = get_model_registry()
        
        model = Model(
            name="production_anomaly_detector",
            algorithm=detector.algorithm,
            parameters=detector.model_params,
            description="Production-trained anomaly detector",
        )
        
        model_metadata = await model_registry.register_model(
            model,
            name="prod_detector_v1",
            version=SemanticVersion("1.0.0")
        )
        
        assert model_metadata.status == ModelStatus.REGISTERED
        
        # Step 4: Model Validation and Deployment
        # Simulate validation by running detection
        validation_data = engineered_dataset.data.sample(n=1000, random_state=42)
        validation_dataset = Dataset(
            name="validation_data",
            data=validation_data,
            description="Validation data for model testing"
        )
        
        validation_result = detector.predict(validation_dataset)
        
        # If validation passes, mark as validated
        if len(validation_result.anomalies) > 0:
            await model_registry.update_model_metadata(
                "prod_detector_v1",
                SemanticVersion("1.0.0"),
                status=ModelStatus.VALIDATED
            )
        
        # Step 5: Analytics and Insights
        analytics_engine = get_analytics_engine()
        
        analytics_report = await analytics_engine.analyze_dataset(
            engineered_dataset,
            [validation_result]
        )
        
        # Verify analytics report
        assert 'dataset_name' in analytics_report
        assert 'analyses_performed' in analytics_report
        assert 'results' in analytics_report
        assert 'summary' in analytics_report
        
        # Step 6: Generate Feature Engineering Report
        feature_report = await feature_engineer.generate_feature_report(
            engineered_dataset,
            production_dataset
        )
        
        assert feature_report['original_features'] == len(production_dataset.data.columns)
        assert feature_report['engineered_features'] >= feature_report['original_features']
        
        # Verify complete pipeline success
        pipeline_results = {
            'feature_engineering': {
                'original_features': len(production_dataset.data.columns),
                'engineered_features': len(engineered_dataset.data.columns),
                'feature_report': feature_report
            },
            'model_training': {
                'algorithm': detector.algorithm,
                'training_time': detector.training_time,
                'is_fitted': detector.is_fitted
            },
            'model_registry': {
                'model_name': model_metadata.name,
                'model_version': str(model_metadata.version),
                'model_status': model_metadata.status.value
            },
            'validation': {
                'samples_tested': len(validation_dataset.data),
                'anomalies_detected': len(validation_result.anomalies),
                'detection_rate': len(validation_result.anomalies) / len(validation_dataset.data)
            },
            'analytics': {
                'analyses_performed': analytics_report['analyses_performed'],
                'total_analyses': analytics_report['summary']['total_analyses']
            }
        }
        
        # All pipeline steps should complete successfully
        assert pipeline_results['feature_engineering']['engineered_features'] > 0
        assert pipeline_results['model_training']['is_fitted']
        assert pipeline_results['model_registry']['model_status'] == 'validated'
        assert pipeline_results['validation']['anomalies_detected'] >= 0
        assert pipeline_results['analytics']['total_analyses'] > 0
        
        return pipeline_results


@pytest.mark.asyncio
class TestEndToEndRealTimeSystem:
    """Test end-to-end real-time anomaly detection system."""
    
    async def test_real_time_anomaly_detection_system(self, production_dataset, temp_system_dir):
        """Test complete real-time anomaly detection system."""
        # Step 1: Prepare detector
        detector = MockProductionDetector()
        
        # Train on subset of data
        training_data = production_dataset.data.sample(n=5000, random_state=42)
        training_dataset = Dataset(
            name="training_data",
            data=training_data,
            description="Training data for real-time system"
        )
        
        detector.fit(training_dataset)
        
        # Step 2: Setup real-time streaming
        stream_processor = get_stream_processor()
        
        streaming_config = StreamingConfig(
            buffer_size=1000,
            batch_size=10,
            batch_timeout_seconds=1.0,
            processing_mode=ProcessingMode.REAL_TIME,
            enable_backpressure=True,
            max_memory_mb=256,
            heartbeat_interval_seconds=5.0
        )
        
        # Create streaming pipeline
        pipeline_created = await stream_processor.create_pipeline(
            "production_anomaly_pipeline",
            detector,
            streaming_config
        )
        assert pipeline_created
        
        # Start pipeline
        pipeline_started = await stream_processor.start_pipeline("production_anomaly_pipeline")
        assert pipeline_started
        
        # Step 3: Simulate real-time data streaming
        real_time_data = production_dataset.data.sample(n=100, random_state=123)
        detected_anomalies = []
        
        # Custom anomaly handler to collect results
        async def anomaly_collection_handler(event):
            if event.event_type.value == "anomaly_detected":
                detected_anomalies.append(event.data)
        
        # Register custom handler (would need access to pipeline internals)
        # For testing, we'll submit data and check pipeline stats
        
        successful_submissions = 0
        for idx, row in real_time_data.iterrows():
            data_point = row.to_dict()
            
            success = await stream_processor.submit_data(
                "production_anomaly_pipeline",
                data_point,
                source="production_system"
            )
            
            if success:
                successful_submissions += 1
            
            # Small delay to simulate real-time streaming
            await asyncio.sleep(0.01)
        
        # Wait for processing to complete
        await asyncio.sleep(3)
        
        # Step 4: Check pipeline status and results
        pipeline_status = await stream_processor.get_all_pipeline_status()
        
        assert pipeline_status["active_pipelines"] == 1
        assert "production_anomaly_pipeline" in pipeline_status["pipelines"]
        
        production_pipeline_stats = pipeline_status["pipelines"]["production_anomaly_pipeline"]
        assert production_pipeline_stats["is_running"]
        assert production_pipeline_stats["buffer_stats"]["total_received"] >= successful_submissions * 0.8
        assert production_pipeline_stats["detector_stats"]["total_detections"] > 0
        
        # Step 5: API Gateway Integration
        api_gateway = get_api_gateway()
        await api_gateway.register_anomaly_detection_endpoints()
        
        # Test health endpoint
        health_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/health",
            headers={"Accept": "application/json"},
            query_params={},
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        health_response = await api_gateway.handle_request(health_request)
        assert health_response.status_code == 200
        
        # Test detection endpoint
        detection_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.POST,
            path="/v1/detect",
            headers={"Content-Type": "application/json"},
            query_params={},
            body={
                "data": real_time_data.iloc[:5].to_dict('records'),
                "algorithm": "production_detector"
            },
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        detection_response = await api_gateway.handle_request(detection_request)
        assert detection_response.status_code == 200
        
        # Step 6: Analytics on real-time results
        analytics_engine = get_analytics_engine()
        
        # Create detection result for analytics
        batch_result = detector.predict(Dataset(
            name="real_time_batch",
            data=real_time_data,
            description="Real-time data batch for analytics"
        ))
        
        real_time_analytics = await analytics_engine.analyze_dataset(
            Dataset(name="real_time_data", data=real_time_data, description="Real-time data"),
            [batch_result]
        )
        
        # Step 7: Stop streaming pipeline
        pipeline_stopped = await stream_processor.stop_pipeline("production_anomaly_pipeline")
        assert pipeline_stopped
        
        # Verify system performance
        system_performance = {
            'streaming': {
                'data_submitted': successful_submissions,
                'data_processed': production_pipeline_stats["buffer_stats"]["total_processed"],
                'detections_performed': production_pipeline_stats["detector_stats"]["total_detections"],
                'processing_rate': successful_submissions / 3.0,  # submissions per second
            },
            'api_gateway': {
                'health_check': health_response.status_code == 200,
                'detection_endpoint': detection_response.status_code == 200,
            },
            'analytics': {
                'real_time_analysis': len(real_time_analytics['analyses_performed']) > 0,
                'anomalies_analyzed': len(batch_result.anomalies),
            }
        }
        
        # Verify system meets performance requirements
        assert system_performance['streaming']['processing_rate'] > 10  # >10 samples/second
        assert system_performance['api_gateway']['health_check']
        assert system_performance['api_gateway']['detection_endpoint']
        assert system_performance['analytics']['real_time_analysis']
        
        return system_performance


@pytest.mark.asyncio
class TestEndToEndModelLifecycle:
    """Test complete model lifecycle management."""
    
    async def test_complete_model_lifecycle(self, production_dataset, temp_system_dir):
        """Test complete model lifecycle from development to production."""
        # Step 1: Model Development
        detector_v1 = MockProductionDetector("isolation_forest_v1")
        detector_v1.model_params["contamination"] = 0.1
        
        # Train initial model
        detector_v1.fit(production_dataset)
        
        # Step 2: Model Registration and Versioning
        model_registry = get_model_registry()
        
        model_v1 = Model(
            name="system_monitor",
            algorithm=detector_v1.algorithm,
            parameters=detector_v1.model_params,
            description="Initial system monitoring model",
        )
        
        v1_metadata = await model_registry.register_model(
            model_v1,
            name="system_monitor",
            version=SemanticVersion("1.0.0")
        )
        
        assert v1_metadata.version.major == 1
        assert v1_metadata.version.minor == 0
        assert v1_metadata.version.patch == 0
        
        # Step 3: Model Validation
        validation_data = production_dataset.data.sample(n=2000, random_state=42)
        validation_dataset = Dataset(
            name="validation_set",
            data=validation_data,
            description="Model validation dataset"
        )
        
        v1_results = detector_v1.predict(validation_dataset)
        
        # Validate model performance
        if len(v1_results.anomalies) > 0:
            await model_registry.update_model_metadata(
                "system_monitor",
                SemanticVersion("1.0.0"),
                status=ModelStatus.VALIDATED
            )
        
        # Step 4: Model Deployment
        from pynomaly.features.model_management import ModelDeployment
        deployment_service = ModelDeployment(model_registry)
        
        deployment_config = DeploymentConfig(
            environment="staging",
            replicas=2,
            cpu_limit="500m",
            memory_limit="512Mi",
            auto_scale=True,
            health_check_enabled=True
        )
        
        v1_deployment_id = await deployment_service.deploy_model(
            "system_monitor",
            SemanticVersion("1.0.0"),
            deployment_config
        )
        
        assert v1_deployment_id is not None
        
        # Step 5: Model Monitoring
        from pynomaly.features.model_management import ModelMonitoring, PerformanceMetrics
        monitoring_service = ModelMonitoring(model_registry)
        
        monitor_id = await monitoring_service.start_monitoring(
            "system_monitor",
            SemanticVersion("1.0.0"),
            validation_dataset
        )
        
        # Record performance metrics
        v1_metrics = PerformanceMetrics(
            accuracy=0.92,
            precision=0.85,
            recall=0.78,
            f1_score=0.81,
            execution_time_ms=50.0,
            memory_usage_mb=128.0,
            throughput_per_second=1000.0
        )
        
        await monitoring_service.record_metrics(monitor_id, v1_metrics)
        
        # Step 6: Model Improvement (Version 2)
        detector_v2 = MockProductionDetector("isolation_forest_v2")
        detector_v2.model_params["contamination"] = 0.08  # Improved parameter
        detector_v2.model_params["n_estimators"] = 200    # More trees
        
        detector_v2.fit(production_dataset)
        
        model_v2 = Model(
            name="system_monitor",
            algorithm=detector_v2.algorithm,
            parameters=detector_v2.model_params,
            description="Improved system monitoring model with better parameters",
        )
        
        v2_metadata = await model_registry.register_model(
            model_v2,
            name="system_monitor",
            version=SemanticVersion("1.1.0")
        )
        
        assert v2_metadata.version.minor == 1
        
        # Step 7: A/B Testing
        v2_results = detector_v2.predict(validation_dataset)
        
        # Compare model performance
        v1_anomaly_count = len(v1_results.anomalies)
        v2_anomaly_count = len(v2_results.anomalies)
        
        v1_avg_score = np.mean([a.score.value for a in v1_results.anomalies]) if v1_results.anomalies else 0
        v2_avg_score = np.mean([a.score.value for a in v2_results.anomalies]) if v2_results.anomalies else 0
        
        # Step 8: Model Promotion
        # If v2 performs better, promote to production
        if v2_avg_score >= v1_avg_score:
            await model_registry.update_model_metadata(
                "system_monitor",
                SemanticVersion("1.1.0"),
                status=ModelStatus.VALIDATED
            )
            
            # Deploy v2 to production
            prod_config = DeploymentConfig(
                environment="production",
                replicas=3,
                cpu_limit="1000m",
                memory_limit="1Gi",
                auto_scale=True,
                health_check_enabled=True
            )
            
            v2_deployment_id = await deployment_service.deploy_model(
                "system_monitor",
                SemanticVersion("1.1.0"),
                prod_config
            )
            
            # Start monitoring v2
            v2_monitor_id = await monitoring_service.start_monitoring(
                "system_monitor",
                SemanticVersion("1.1.0"),
                validation_dataset
            )
            
            # Gradually retire v1
            await deployment_service.stop_deployment(v1_deployment_id)
        
        # Step 9: Model Versioning Analysis
        from pynomaly.features.model_management import ModelVersioning
        versioning_service = ModelVersioning(model_registry)
        
        version_history = await versioning_service.get_version_history("system_monitor")
        assert len(version_history) >= 2
        
        version_comparison = await versioning_service.compare_versions(
            "system_monitor",
            SemanticVersion("1.0.0"),
            SemanticVersion("1.1.0")
        )
        
        assert "parameter_changes" in version_comparison
        assert version_comparison["parameter_changes"]["contamination"]["old"] == 0.1
        assert version_comparison["parameter_changes"]["contamination"]["new"] == 0.08
        
        # Step 10: Comprehensive Analytics
        analytics_engine = get_analytics_engine()
        
        # Analyze both model versions
        v1_analytics = await analytics_engine.analyze_dataset(validation_dataset, [v1_results])
        v2_analytics = await analytics_engine.analyze_dataset(validation_dataset, [v2_results])
        
        # Generate model lifecycle report
        lifecycle_report = {
            'model_versions': {
                'v1.0.0': {
                    'anomalies_detected': v1_anomaly_count,
                    'average_score': v1_avg_score,
                    'deployment_id': v1_deployment_id,
                    'analytics': v1_analytics
                },
                'v1.1.0': {
                    'anomalies_detected': v2_anomaly_count,
                    'average_score': v2_avg_score,
                    'deployment_id': v2_deployment_id if v2_avg_score >= v1_avg_score else None,
                    'analytics': v2_analytics
                }
            },
            'version_comparison': version_comparison,
            'monitoring': {
                'v1_monitor_id': monitor_id,
                'v2_monitor_id': v2_monitor_id if v2_avg_score >= v1_avg_score else None
            },
            'performance_improvement': {
                'score_improvement': v2_avg_score - v1_avg_score,
                'detection_rate_change': (v2_anomaly_count - v1_anomaly_count) / max(v1_anomaly_count, 1)
            }
        }
        
        # Clean up monitoring
        await monitoring_service.stop_monitoring(monitor_id)
        if v2_avg_score >= v1_avg_score and 'v2_monitor_id' in locals():
            await monitoring_service.stop_monitoring(v2_monitor_id)
        
        # Verify lifecycle completed successfully
        assert len(lifecycle_report['model_versions']) == 2
        assert lifecycle_report['version_comparison']['parameter_changes']['contamination']['old'] != \
               lifecycle_report['version_comparison']['parameter_changes']['contamination']['new']
        
        return lifecycle_report


@pytest.mark.asyncio
class TestEndToEndScalabilityAndPerformance:
    """Test system scalability and performance under load."""
    
    async def test_system_under_load(self, production_dataset):
        """Test system performance under high load."""
        # Step 1: Setup multiple concurrent pipelines
        stream_processor = get_stream_processor()
        api_gateway = get_api_gateway()
        
        # Create multiple detectors for concurrent processing
        detectors = []
        for i in range(3):
            detector = MockProductionDetector(f"load_test_detector_{i}")
            
            # Train on subset to speed up testing
            training_subset = production_dataset.data.sample(n=1000, random_state=i)
            training_dataset = Dataset(
                name=f"training_subset_{i}",
                data=training_subset,
                description=f"Training subset {i} for load testing"
            )
            detector.fit(training_dataset)
            detectors.append(detector)
        
        # Create streaming pipelines
        pipeline_ids = []
        for i, detector in enumerate(detectors):
            pipeline_id = f"load_test_pipeline_{i}"
            
            config = StreamingConfig(
                buffer_size=500,
                batch_size=20,
                batch_timeout_seconds=0.5,
                processing_mode=ProcessingMode.MICRO_BATCH
            )
            
            created = await stream_processor.create_pipeline(pipeline_id, detector, config)
            assert created
            
            started = await stream_processor.start_pipeline(pipeline_id)
            assert started
            
            pipeline_ids.append(pipeline_id)
        
        # Step 2: Concurrent data streaming
        test_data = production_dataset.data.sample(n=1000, random_state=999)
        
        async def stream_data_to_pipeline(pipeline_id: str, data_chunk: pd.DataFrame):
            successful_submissions = 0
            start_time = datetime.now()
            
            for idx, row in data_chunk.iterrows():
                data_point = row.to_dict()
                
                success = await stream_processor.submit_data(
                    pipeline_id,
                    data_point,
                    source=f"load_test_{pipeline_id}"
                )
                
                if success:
                    successful_submissions += 1
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                'pipeline_id': pipeline_id,
                'submissions': successful_submissions,
                'processing_time': processing_time,
                'throughput': successful_submissions / processing_time if processing_time > 0 else 0
            }
        
        # Split data among pipelines
        data_chunks = np.array_split(test_data, len(pipeline_ids))
        
        # Run concurrent streaming
        start_time = datetime.now()
        
        streaming_tasks = [
            stream_data_to_pipeline(pipeline_id, chunk)
            for pipeline_id, chunk in zip(pipeline_ids, data_chunks)
        ]
        
        streaming_results = await asyncio.gather(*streaming_tasks)
        
        concurrent_processing_time = (datetime.now() - start_time).total_seconds()
        
        # Step 3: Concurrent API requests
        await api_gateway.register_anomaly_detection_endpoints()
        
        async def make_api_request(request_id: int):
            sample_data = test_data.sample(n=5, random_state=request_id)
            
            request = APIRequest(
                request_id=f"load_test_request_{request_id}",
                method=HTTPMethod.POST,
                path="/v1/detect",
                headers={"Content-Type": "application/json"},
                query_params={},
                body={
                    "data": sample_data.to_dict('records'),
                    "algorithm": "load_test"
                },
                timestamp=datetime.now(),
                client_ip=f"192.168.1.{(request_id % 254) + 1}"
            )
            
            start_time = datetime.now()
            response = await api_gateway.handle_request(request)
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'request_id': request_id,
                'status_code': response.status_code,
                'response_time': response_time,
                'success': response.status_code == 200
            }
        
        # Make concurrent API requests
        num_concurrent_requests = 50
        api_start_time = datetime.now()
        
        api_tasks = [make_api_request(i) for i in range(num_concurrent_requests)]
        api_results = await asyncio.gather(*api_tasks)
        
        api_processing_time = (datetime.now() - api_start_time).total_seconds()
        
        # Step 4: Wait for streaming processing to complete
        await asyncio.sleep(5)
        
        # Get final pipeline statistics
        final_status = await stream_processor.get_all_pipeline_status()
        
        # Step 5: Analytics on load test results
        analytics_engine = get_analytics_engine()
        
        # Analyze system performance
        total_submissions = sum(result['submissions'] for result in streaming_results)
        total_throughput = sum(result['throughput'] for result in streaming_results)
        
        successful_api_requests = sum(1 for result in api_results if result['success'])
        average_api_response_time = np.mean([result['response_time'] for result in api_results])
        
        # Step 6: Cleanup
        for pipeline_id in pipeline_ids:
            await stream_processor.stop_pipeline(pipeline_id)
        
        # Performance metrics
        load_test_results = {
            'streaming_performance': {
                'total_submissions': total_submissions,
                'total_throughput': total_throughput,
                'concurrent_processing_time': concurrent_processing_time,
                'pipelines_tested': len(pipeline_ids),
                'average_throughput_per_pipeline': total_throughput / len(pipeline_ids)
            },
            'api_performance': {
                'total_requests': num_concurrent_requests,
                'successful_requests': successful_api_requests,
                'success_rate': successful_api_requests / num_concurrent_requests,
                'average_response_time': average_api_response_time,
                'requests_per_second': num_concurrent_requests / api_processing_time,
                'total_processing_time': api_processing_time
            },
            'system_performance': {
                'concurrent_pipelines': len(pipeline_ids),
                'total_data_processed': len(test_data),
                'overall_processing_time': max(concurrent_processing_time, api_processing_time),
                'system_utilization': final_status['active_pipelines'] / final_status['total_pipelines']
            }
        }
        
        # Performance assertions
        assert load_test_results['streaming_performance']['total_throughput'] > 50  # >50 samples/second total
        assert load_test_results['api_performance']['success_rate'] > 0.95  # >95% success rate
        assert load_test_results['api_performance']['average_response_time'] < 2.0  # <2 seconds avg
        assert load_test_results['api_performance']['requests_per_second'] > 10  # >10 requests/second
        
        return load_test_results


@pytest.mark.asyncio
class TestEndToEndErrorRecovery:
    """Test system error handling and recovery."""
    
    async def test_system_error_recovery(self, production_dataset):
        """Test system resilience under various error conditions."""
        # Step 1: Setup system components
        stream_processor = get_stream_processor()
        api_gateway = get_api_gateway()
        model_registry = get_model_registry()
        
        # Create detector that can simulate failures
        class FailureSimulatingDetector(MockProductionDetector):
            def __init__(self):
                super().__init__("failure_test_detector")
                self.failure_count = 0
                self.max_failures = 3
            
            def predict(self, dataset: Dataset) -> DetectionResult:
                self.failure_count += 1
                
                # Simulate failures for first few predictions
                if self.failure_count <= self.max_failures:
                    raise RuntimeError(f"Simulated failure #{self.failure_count}")
                
                # After failures, work normally
                return super().predict(dataset)
            
            async def detect(self, dataset: Dataset) -> DetectionResult:
                return self.predict(dataset)
        
        detector = FailureSimulatingDetector()
        
        # Train detector
        training_data = production_dataset.data.sample(n=500, random_state=42)
        training_dataset = Dataset(
            name="failure_test_training",
            data=training_data,
            description="Training data for failure testing"
        )
        detector.fit(training_dataset)
        
        # Step 2: Test streaming pipeline error recovery
        config = StreamingConfig(
            buffer_size=100,
            batch_size=10,
            batch_timeout_seconds=1.0,
            error_threshold=5  # Allow some errors before shutdown
        )
        
        created = await stream_processor.create_pipeline("error_recovery_test", detector, config)
        assert created
        
        started = await stream_processor.start_pipeline("error_recovery_test")
        assert started
        
        # Submit data that will cause failures
        test_data = production_dataset.data.sample(n=20, random_state=123)
        
        submission_results = []
        for idx, row in test_data.iterrows():
            data_point = row.to_dict()
            
            success = await stream_processor.submit_data(
                "error_recovery_test",
                data_point,
                source="error_recovery_test"
            )
            submission_results.append(success)
            
            await asyncio.sleep(0.1)
        
        # Wait for processing (including failures and recovery)
        await asyncio.sleep(5)
        
        # Check pipeline status after error recovery
        status = await stream_processor.get_all_pipeline_status()
        pipeline_stats = status["pipelines"]["error_recovery_test"]
        
        # Pipeline should still be running despite initial failures
        assert pipeline_stats["is_running"]
        
        # Step 3: Test API gateway error handling
        await api_gateway.register_anomaly_detection_endpoints()
        
        # Test various error scenarios
        error_test_requests = [
            # Invalid request body
            APIRequest(
                request_id="error_test_1",
                method=HTTPMethod.POST,
                path="/v1/detect",
                headers={"Content-Type": "application/json"},
                body={"invalid": "data_format"},
                timestamp=datetime.now(),
                client_ip="127.0.0.1"
            ),
            # Missing request body
            APIRequest(
                request_id="error_test_2",
                method=HTTPMethod.POST,
                path="/v1/detect",
                headers={"Content-Type": "application/json"},
                body=None,
                timestamp=datetime.now(),
                client_ip="127.0.0.1"
            ),
            # Non-existent endpoint
            APIRequest(
                request_id="error_test_3",
                method=HTTPMethod.GET,
                path="/v1/nonexistent",
                headers={},
                timestamp=datetime.now(),
                client_ip="127.0.0.1"
            ),
        ]
        
        api_error_responses = []
        for request in error_test_requests:
            try:
                response = await api_gateway.handle_request(request)
                api_error_responses.append({
                    'request_id': request.request_id,
                    'status_code': response.status_code,
                    'handled_gracefully': response.status_code in [400, 404, 500]
                })
            except Exception as e:
                api_error_responses.append({
                    'request_id': request.request_id,
                    'status_code': None,
                    'handled_gracefully': False,
                    'error': str(e)
                })
        
        # Step 4: Test model registry error recovery
        # Try to register invalid model
        try:
            invalid_model = Model(
                name="",  # Invalid empty name
                algorithm="test",
                parameters={},
                description="Invalid model for testing"
            )
            
            await model_registry.register_model(
                invalid_model,
                name="invalid_model",
                version=SemanticVersion("1.0.0")
            )
            model_registry_handled_error = False
        except Exception:
            model_registry_handled_error = True
        
        # Try to get non-existent model
        nonexistent_model = await model_registry.get_model("nonexistent", SemanticVersion("1.0.0"))
        
        # Step 5: Test analytics error recovery
        analytics_engine = get_analytics_engine()
        
        # Try analytics with empty dataset
        empty_dataset = Dataset(
            name="empty_test",
            data=pd.DataFrame(),
            description="Empty dataset for error testing"
        )
        
        try:
            empty_analytics = await analytics_engine.analyze_dataset(empty_dataset, [])
            analytics_handled_empty = True
        except Exception:
            analytics_handled_empty = False
        
        # Step 6: System recovery verification
        # After errors, system should still function normally
        
        # Test normal streaming
        normal_data = production_dataset.data.sample(n=5, random_state=456)
        normal_submissions = 0
        
        for idx, row in normal_data.iterrows():
            success = await stream_processor.submit_data(
                "error_recovery_test",
                row.to_dict(),
                source="recovery_test"
            )
            if success:
                normal_submissions += 1
        
        await asyncio.sleep(2)
        
        # Test normal API request
        normal_request = APIRequest(
            request_id="recovery_test",
            method=HTTPMethod.GET,
            path="/health",
            headers={"Accept": "application/json"},
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        normal_response = await api_gateway.handle_request(normal_request)
        
        # Cleanup
        await stream_processor.stop_pipeline("error_recovery_test")
        
        # Compile error recovery results
        error_recovery_results = {
            'streaming_recovery': {
                'pipeline_survived_errors': pipeline_stats["is_running"],
                'normal_operations_after_errors': normal_submissions > 0,
                'error_handling_effective': detector.failure_count > detector.max_failures
            },
            'api_error_handling': {
                'total_error_scenarios': len(error_test_requests),
                'gracefully_handled': sum(1 for r in api_error_responses if r['handled_gracefully']),
                'normal_operation_after_errors': normal_response.status_code == 200
            },
            'model_registry_recovery': {
                'handled_invalid_input': model_registry_handled_error,
                'handled_nonexistent_queries': nonexistent_model is None
            },
            'analytics_recovery': {
                'handled_empty_dataset': analytics_handled_empty
            }
        }
        
        # Verify error recovery
        assert error_recovery_results['streaming_recovery']['pipeline_survived_errors']
        assert error_recovery_results['streaming_recovery']['normal_operations_after_errors']
        assert error_recovery_results['api_error_handling']['gracefully_handled'] >= 2
        assert error_recovery_results['api_error_handling']['normal_operation_after_errors']
        assert error_recovery_results['model_registry_recovery']['handled_invalid_input']
        assert error_recovery_results['model_registry_recovery']['handled_nonexistent_queries']
        
        return error_recovery_results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
"""Enhanced tests for application layer services - Phase 1 Coverage Enhancement."""

from __future__ import annotations

import asyncio
import numpy as np
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import UUID, uuid4
import tempfile
import json

from pynomaly.application.services import (
    DetectionService,
    EnsembleService,
    ModelPersistenceService,
    ExperimentTrackingService,
)
from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Anomaly
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore, ConfidenceInterval
from pynomaly.domain.services import AnomalyScorer, ThresholdCalculator, EnsembleAggregator
from pynomaly.infrastructure.repositories import (
    InMemoryDetectorRepository,
    InMemoryResultRepository,
    InMemoryDatasetRepository,
)


@pytest.fixture
def large_dataset():
    """Create a larger dataset for comprehensive testing."""
    features = np.random.RandomState(42).normal(0, 1, (1000, 10))
    targets = np.random.RandomState(42).choice([0, 1], size=1000, p=[0.92, 0.08])
    return Dataset(name="large_test_dataset", features=features, targets=targets)


@pytest.fixture
def multiple_detectors():
    """Create multiple detectors for testing."""
    detectors = []
    algorithms = ["isolation_forest", "local_outlier_factor", "one_class_svm", "copod", "knn"]
    
    for i, algo in enumerate(algorithms):
        detector = Detector(
            name=f"detector_{i}_{algo}",
            algorithm=algo,
            contamination=ContaminationRate(0.05 + i * 0.01),  # Varying contamination
            hyperparameters={"random_state": 42 + i}
        )
        detectors.append(detector)
    
    return detectors


@pytest.fixture
def complex_detection_service():
    """Create a DetectionService with enhanced functionality."""
    detector_repo = InMemoryDetectorRepository()
    result_repo = InMemoryResultRepository()
    scorer = AnomalyScorer()
    threshold_calc = ThresholdCalculator()
    
    return DetectionService(
        detector_repository=detector_repo,
        result_repository=result_repo,
        anomaly_scorer=scorer,
        threshold_calculator=threshold_calc
    )


class TestDetectionServiceEnhanced:
    """Enhanced tests for DetectionService covering advanced functionality."""
    
    @pytest.mark.asyncio
    async def test_detect_with_multiple_detectors_parallel(self, complex_detection_service, multiple_detectors, large_dataset):
        """Test parallel detection with multiple detectors."""
        # Save detectors
        detector_ids = []
        for detector in multiple_detectors:
            await complex_detection_service.detector_repository.save(detector)
            detector_ids.append(detector.id)
        
        # Mock the actual detection algorithms
        with patch.object(complex_detection_service.anomaly_scorer, 'compute_scores') as mock_compute:
            mock_compute.return_value = np.random.RandomState(42).random(len(large_dataset.features))
            
            # Test parallel detection
            results = await complex_detection_service.detect_with_multiple_detectors(
                detector_ids=detector_ids,
                dataset=large_dataset,
                save_results=True
            )
            
            # Verify results
            assert len(results) == len(detector_ids)
            assert all(isinstance(result, DetectionResult) for result in results.values())
            assert all(detector_id in results for detector_id in detector_ids)
            
            # Each detector should have been called
            assert mock_compute.call_count == len(detector_ids)
    
    @pytest.mark.asyncio
    async def test_batch_detection_with_performance_tracking(self, complex_detection_service, multiple_detectors):
        """Test batch detection with performance metrics tracking."""
        detector = multiple_detectors[0]
        await complex_detection_service.detector_repository.save(detector)
        
        # Create multiple datasets with different sizes
        datasets = []
        sizes = [100, 500, 1000, 2000]
        for i, size in enumerate(sizes):
            features = np.random.RandomState(42 + i).normal(0, 1, (size, 8))
            dataset = Dataset(name=f"batch_dataset_{i}", features=features)
            datasets.append(dataset)
        
        with patch.object(complex_detection_service.anomaly_scorer, 'compute_scores') as mock_compute:
            def mock_compute_side_effect(features):
                return np.random.random(len(features))
            
            mock_compute.side_effect = mock_compute_side_effect
            
            start_time = datetime.now()
            results = await complex_detection_service.batch_detection(
                detector_id=detector.id,
                datasets=datasets,
                save_results=False,
                track_performance=True
            )
            end_time = datetime.now()
            
            # Verify results
            assert len(results) == len(datasets)
            processing_time = (end_time - start_time).total_seconds()
            assert processing_time > 0
            
            # Verify each dataset was processed
            for i, result in enumerate(results):
                assert result.dataset.name == f"batch_dataset_{i}"
                assert len(result.scores) == sizes[i]
    
    @pytest.mark.asyncio
    async def test_detection_with_quality_assessment(self, complex_detection_service, multiple_detectors, large_dataset):
        """Test detection with comprehensive data quality assessment."""
        detector = multiple_detectors[0]
        await complex_detection_service.detector_repository.save(detector)
        
        # Add some data quality issues to the dataset
        corrupted_features = large_dataset.features.copy()
        corrupted_features[10:15, 0] = np.nan  # Missing values
        corrupted_features[20:25, 1] = np.inf  # Infinite values
        corrupted_features[50:100, 2] = corrupted_features[50, 2]  # Constant values
        
        corrupted_dataset = Dataset(
            name="corrupted_test_dataset",
            features=corrupted_features,
            targets=large_dataset.targets
        )
        
        with patch.object(complex_detection_service.anomaly_scorer, 'compute_scores') as mock_compute:
            mock_compute.return_value = np.random.random(len(corrupted_features))
            
            # Test detection with quality assessment
            result = await complex_detection_service.run_detection_with_quality_check(
                detector_id=detector.id,
                dataset=corrupted_dataset,
                quality_threshold=0.7,
                auto_clean=True
            )
            
            # Verify quality assessment was performed
            assert hasattr(result, 'quality_report')
            assert hasattr(result, 'data_cleaning_applied')
    
    @pytest.mark.asyncio
    async def test_detection_history_analysis(self, complex_detection_service, multiple_detectors, large_dataset):
        """Test detection history tracking and analysis."""
        detector = multiple_detectors[0]
        await complex_detection_service.detector_repository.save(detector)
        
        # Run multiple detections over time
        detection_results = []
        with patch.object(complex_detection_service.anomaly_scorer, 'compute_scores') as mock_compute:
            for i in range(5):
                # Simulate different detection runs
                mock_scores = np.random.RandomState(42 + i).random(len(large_dataset.features))
                mock_compute.return_value = mock_scores
                
                result = await complex_detection_service.run_detection(
                    detector_id=detector.id,
                    dataset=large_dataset
                )
                detection_results.append(result)
                
                # Simulate time passing
                await asyncio.sleep(0.01)
        
        # Test history analysis
        history = await complex_detection_service.get_detection_history(
            detector_id=detector.id,
            limit=10,
            include_statistics=True
        )
        
        assert len(history) == 5
        assert all(r.detector.id == detector.id for r in history)
        
        # Test performance trends
        trends = await complex_detection_service.analyze_performance_trends(
            detector_id=detector.id,
            time_window=timedelta(minutes=1)
        )
        
        assert 'average_processing_time' in trends
        assert 'detection_count' in trends
    
    @pytest.mark.asyncio
    async def test_adaptive_threshold_detection(self, complex_detection_service, multiple_detectors, large_dataset):
        """Test detection with adaptive thresholding."""
        detector = multiple_detectors[0]
        await complex_detection_service.detector_repository.save(detector)
        
        with patch.object(complex_detection_service.threshold_calculator, 'calculate_adaptive_threshold') as mock_threshold:
            with patch.object(complex_detection_service.anomaly_scorer, 'compute_scores') as mock_compute:
                # Mock scores with known distribution
                scores = np.random.beta(2, 8, len(large_dataset.features))  # Skewed distribution
                mock_compute.return_value = scores
                
                # Mock adaptive threshold calculation
                mock_threshold.return_value = np.percentile(scores, 95)  # Top 5%
                
                result = await complex_detection_service.run_detection_with_adaptive_threshold(
                    detector_id=detector.id,
                    dataset=large_dataset,
                    adaptation_method="statistical",
                    min_threshold=0.5,
                    max_threshold=0.99
                )
                
                # Verify adaptive thresholding was applied
                assert isinstance(result, DetectionResult)
                mock_threshold.assert_called_once()
                
                # Verify threshold is within bounds
                calculated_threshold = mock_threshold.return_value
                assert 0.5 <= calculated_threshold <= 0.99


class TestEnsembleServiceEnhanced:
    """Enhanced tests for EnsembleService covering advanced ensemble techniques."""
    
    @pytest.fixture
    def ensemble_service(self):
        """Create an EnsembleService with mocked dependencies."""
        detector_repo = InMemoryDetectorRepository()
        aggregator = EnsembleAggregator()
        scorer = AnomalyScorer()
        
        return EnsembleService(
            detector_repository=detector_repo,
            ensemble_aggregator=aggregator,
            anomaly_scorer=scorer
        )
    
    @pytest.mark.asyncio
    async def test_create_weighted_ensemble(self, ensemble_service, multiple_detectors, large_dataset):
        """Test creating a weighted ensemble with performance-based weights."""
        # Save detectors
        detector_ids = []
        for detector in multiple_detectors:
            await ensemble_service.detector_repository.save(detector)
            detector_ids.append(detector.id)
        
        # Define performance-based weights
        weights = {
            detector_ids[0]: 0.3,  # Best performer
            detector_ids[1]: 0.25,
            detector_ids[2]: 0.2,
            detector_ids[3]: 0.15,
            detector_ids[4]: 0.1   # Worst performer
        }
        
        with patch.object(ensemble_service.ensemble_aggregator, 'aggregate') as mock_agg:
            mock_agg.return_value = np.random.random(len(large_dataset.features))
            
            ensemble_detector = await ensemble_service.create_weighted_ensemble(
                name="Performance Weighted Ensemble",
                detector_ids=detector_ids,
                weights=weights,
                aggregation_method="weighted_average"
            )
            
            # Verify ensemble creation
            assert ensemble_detector.name == "Performance Weighted Ensemble"
            assert len(ensemble_detector.component_detectors) == len(detector_ids)
            assert ensemble_detector.weights == weights
    
    @pytest.mark.asyncio
    async def test_ensemble_performance_evaluation(self, ensemble_service, multiple_detectors, large_dataset):
        """Test comprehensive ensemble performance evaluation."""
        # Save detectors
        detector_ids = []
        for detector in multiple_detectors:
            await ensemble_service.detector_repository.save(detector)
            detector_ids.append(detector.id)
        
        # Mock individual detector performances
        individual_performances = {
            detector_ids[0]: {"accuracy": 0.85, "f1_score": 0.78, "precision": 0.82},
            detector_ids[1]: {"accuracy": 0.82, "f1_score": 0.75, "precision": 0.79},
            detector_ids[2]: {"accuracy": 0.88, "f1_score": 0.81, "precision": 0.85},
            detector_ids[3]: {"accuracy": 0.80, "f1_score": 0.73, "precision": 0.77},
            detector_ids[4]: {"accuracy": 0.83, "f1_score": 0.76, "precision": 0.80},
        }
        
        with patch.object(ensemble_service, '_evaluate_individual_detectors') as mock_eval:
            mock_eval.return_value = individual_performances
            
            # Test ensemble evaluation
            evaluation = await ensemble_service.evaluate_ensemble_performance(
                detector_ids=detector_ids,
                test_dataset=large_dataset,
                aggregation_methods=["average", "weighted_average", "voting"],
                cross_validation=True,
                cv_folds=5
            )
            
            # Verify evaluation results
            assert "individual_performances" in evaluation
            assert "ensemble_performances" in evaluation
            assert "diversity_metrics" in evaluation
            assert "optimal_weights" in evaluation
            
            # Check that all aggregation methods were tested
            assert len(evaluation["ensemble_performances"]) == 3
    
    @pytest.mark.asyncio
    async def test_dynamic_ensemble_adaptation(self, ensemble_service, multiple_detectors, large_dataset):
        """Test dynamic ensemble weight adaptation based on recent performance."""
        detector_ids = []
        for detector in multiple_detectors:
            await ensemble_service.detector_repository.save(detector)
            detector_ids.append(detector.id)
        
        # Simulate performance history over time
        performance_history = []
        for time_step in range(10):
            step_performance = {}
            for i, detector_id in enumerate(detector_ids):
                # Simulate performance drift over time
                base_performance = 0.8 + i * 0.02
                drift = 0.05 * np.sin(time_step * 0.5) if i == 0 else 0  # First detector drifts
                step_performance[detector_id] = base_performance + drift
            performance_history.append(step_performance)
        
        with patch.object(ensemble_service, '_get_performance_history') as mock_history:
            mock_history.return_value = performance_history
            
            # Test dynamic adaptation
            adapted_weights = await ensemble_service.adapt_ensemble_weights(
                detector_ids=detector_ids,
                adaptation_method="exponential_smoothing",
                adaptation_rate=0.1,
                min_weight=0.05,
                max_weight=0.4
            )
            
            # Verify weight adaptation
            assert len(adapted_weights) == len(detector_ids)
            assert all(0.05 <= weight <= 0.4 for weight in adapted_weights.values())
            assert abs(sum(adapted_weights.values()) - 1.0) < 1e-10  # Weights sum to 1
    
    @pytest.mark.asyncio
    async def test_ensemble_diversity_analysis(self, ensemble_service, multiple_detectors, large_dataset):
        """Test ensemble diversity analysis and optimization."""
        detector_ids = []
        for detector in multiple_detectors:
            await ensemble_service.detector_repository.save(detector)
            detector_ids.append(detector.id)
        
        # Mock individual detector predictions
        mock_predictions = {}
        for i, detector_id in enumerate(detector_ids):
            # Create diverse prediction patterns
            predictions = np.random.RandomState(42 + i).choice([0, 1], size=len(large_dataset.features), p=[0.9, 0.1])
            mock_predictions[detector_id] = predictions
        
        with patch.object(ensemble_service, '_get_detector_predictions') as mock_pred:
            mock_pred.return_value = mock_predictions
            
            # Test diversity analysis
            diversity_analysis = await ensemble_service.analyze_ensemble_diversity(
                detector_ids=detector_ids,
                dataset=large_dataset,
                diversity_metrics=["disagreement", "q_statistic", "correlation", "entropy"]
            )
            
            # Verify diversity analysis
            assert "pairwise_diversity" in diversity_analysis
            assert "overall_diversity" in diversity_analysis
            assert "diversity_ranking" in diversity_analysis
            assert "optimal_subset" in diversity_analysis
            
            # Check that all requested metrics are included
            for metric in ["disagreement", "q_statistic", "correlation", "entropy"]:
                assert metric in diversity_analysis["overall_diversity"]


class TestModelPersistenceServiceEnhanced:
    """Enhanced tests for ModelPersistenceService covering advanced functionality."""
    
    @pytest.fixture
    def enhanced_persistence_service(self):
        """Create ModelPersistenceService with temporary storage and enhanced features."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            detector_repo = InMemoryDetectorRepository()
            service = ModelPersistenceService(
                detector_repository=detector_repo,
                storage_path=Path(tmp_dir),
                compression=True,
                encryption_key=None,  # No encryption for tests
                backup_versions=3
            )
            yield service
    
    @pytest.mark.asyncio
    async def test_model_versioning(self, enhanced_persistence_service, multiple_detectors):
        """Test model versioning and backup functionality."""
        detector = multiple_detectors[0]
        await enhanced_persistence_service.detector_repository.save(detector)
        
        # Save multiple versions of the same model
        model_versions = []
        for version in range(1, 5):
            model_data = {
                "version": version,
                "weights": np.random.random(100).tolist(),
                "hyperparameters": {"n_estimators": 100 * version},
                "training_metadata": {
                    "accuracy": 0.8 + version * 0.02,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            with patch('pickle.dump') as mock_dump:
                file_path = await enhanced_persistence_service.save_model_version(
                    detector_id=detector.id,
                    model_data=model_data,
                    version=f"v{version}",
                    description=f"Model version {version} with improved accuracy"
                )
                model_versions.append(file_path)
        
        # Test version listing
        with patch('pickle.load') as mock_load:
            mock_load.return_value = {"version": 1, "weights": []}
            
            versions = await enhanced_persistence_service.list_model_versions(detector.id)
            
            assert len(versions) >= 3  # Should keep backup_versions
            assert all("version" in version_info for version_info in versions)
    
    @pytest.mark.asyncio
    async def test_model_metadata_management(self, enhanced_persistence_service, multiple_detectors):
        """Test comprehensive model metadata management."""
        detector = multiple_detectors[0]
        await enhanced_persistence_service.detector_repository.save(detector)
        
        # Create comprehensive model metadata
        model_metadata = {
            "model_info": {
                "algorithm": detector.algorithm,
                "version": "2.1.0",
                "framework": "scikit-learn",
                "model_size_mb": 15.7
            },
            "training_info": {
                "dataset_name": "fraud_detection_v2",
                "training_samples": 50000,
                "training_time_seconds": 127.5,
                "convergence_epoch": 45,
                "hyperparameters": detector.hyperparameters
            },
            "performance_metrics": {
                "accuracy": 0.924,
                "precision": 0.887,
                "recall": 0.856,
                "f1_score": 0.871,
                "roc_auc": 0.945
            },
            "data_profile": {
                "feature_count": 30,
                "categorical_features": 8,
                "missing_value_ratio": 0.023,
                "outlier_ratio": 0.087
            },
            "deployment_info": {
                "created_by": "data_scientist_001",
                "environment": "production",
                "deployment_target": "fraud_api_v2"
            }
        }
        
        model_data = {"weights": np.random.random(100).tolist()}
        
        with patch('pickle.dump') as mock_dump:
            with patch('json.dump') as mock_json_dump:
                # Save model with comprehensive metadata
                file_path = await enhanced_persistence_service.save_model_with_metadata(
                    detector_id=detector.id,
                    model_data=model_data,
                    metadata=model_metadata
                )
                
                # Verify both model and metadata were saved
                assert mock_dump.called
                assert mock_json_dump.called
        
        # Test metadata retrieval
        with patch('json.load') as mock_json_load:
            mock_json_load.return_value = model_metadata
            
            retrieved_metadata = await enhanced_persistence_service.get_model_metadata(detector.id)
            
            assert retrieved_metadata["model_info"]["algorithm"] == detector.algorithm
            assert retrieved_metadata["performance_metrics"]["accuracy"] == 0.924
            assert retrieved_metadata["training_info"]["training_samples"] == 50000
    
    @pytest.mark.asyncio
    async def test_model_compression_and_optimization(self, enhanced_persistence_service, multiple_detectors):
        """Test model compression and storage optimization."""
        detector = multiple_detectors[0]
        await enhanced_persistence_service.detector_repository.save(detector)
        
        # Create large model data to test compression
        large_model_data = {
            "weights": np.random.random(10000).tolist(),
            "bias": np.random.random(1000).tolist(),
            "feature_names": [f"feature_{i}" for i in range(1000)],
            "training_history": {
                "epoch_losses": np.random.random(200).tolist(),
                "validation_scores": np.random.random(200).tolist()
            }
        }
        
        with patch('pickle.dump') as mock_dump:
            with patch('gzip.open') as mock_gzip:  # Mock compression
                # Test compressed save
                file_path = await enhanced_persistence_service.save_model_compressed(
                    detector_id=detector.id,
                    model_data=large_model_data,
                    compression_level=6
                )
                
                # Verify compression was used
                assert mock_gzip.called
                assert file_path.suffix == '.pkl.gz'
        
        # Test storage optimization
        optimization_report = await enhanced_persistence_service.optimize_model_storage(
            detector_id=detector.id,
            remove_redundant_data=True,
            compress_weights=True
        )
        
        assert "original_size_mb" in optimization_report
        assert "optimized_size_mb" in optimization_report
        assert "compression_ratio" in optimization_report


class TestExperimentTrackingServiceEnhanced:
    """Enhanced tests for ExperimentTrackingService covering advanced functionality."""
    
    @pytest.fixture
    def enhanced_tracking_service(self):
        """Create ExperimentTrackingService with temporary storage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = ExperimentTrackingService(
                tracking_path=Path(tmp_dir),
                auto_log_system_metrics=True,
                enable_distributed_tracking=False
            )
            yield service
    
    @pytest.mark.asyncio
    async def test_comprehensive_experiment_tracking(self, enhanced_tracking_service, multiple_detectors, large_dataset):
        """Test comprehensive experiment tracking with multiple runs."""
        # Start a comprehensive experiment
        experiment_config = {
            "experiment_type": "hyperparameter_optimization",
            "algorithms": [d.algorithm for d in multiple_detectors],
            "dataset_info": {
                "name": large_dataset.name,
                "samples": len(large_dataset.features),
                "features": large_dataset.features.shape[1]
            },
            "optimization_target": "f1_score",
            "search_space": {
                "contamination": [0.05, 0.1, 0.15],
                "n_estimators": [100, 200, 300]
            }
        }
        
        experiment_id = await enhanced_tracking_service.start_experiment(
            name="Advanced Fraud Detection Optimization",
            description="Comprehensive hyperparameter optimization for fraud detection",
            config=experiment_config,
            tags=["fraud_detection", "hyperparameter_tuning", "production"]
        )
        
        # Simulate multiple experiment runs
        run_results = []
        for i, detector in enumerate(multiple_detectors):
            run_config = {
                "algorithm": detector.algorithm,
                "contamination": 0.05 + i * 0.025,
                "hyperparameters": detector.hyperparameters
            }
            
            run_id = await enhanced_tracking_service.start_run(
                experiment_id=experiment_id,
                run_name=f"run_{i}_{detector.algorithm}",
                config=run_config
            )
            
            # Log comprehensive metrics for each run
            metrics = {
                "accuracy": 0.85 + i * 0.02 + np.random.normal(0, 0.01),
                "precision": 0.82 + i * 0.015 + np.random.normal(0, 0.01),
                "recall": 0.78 + i * 0.02 + np.random.normal(0, 0.01),
                "f1_score": 0.80 + i * 0.018 + np.random.normal(0, 0.01),
                "roc_auc": 0.88 + i * 0.02 + np.random.normal(0, 0.01),
                "training_time": 45.0 + i * 5 + np.random.normal(0, 2),
                "memory_usage_mb": 128 + i * 16 + np.random.normal(0, 5)
            }
            
            for metric_name, value in metrics.items():
                await enhanced_tracking_service.log_metric(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    metric_name=metric_name,
                    value=float(value),
                    step=1
                )
            
            # Log run artifacts
            await enhanced_tracking_service.log_artifact(
                experiment_id=experiment_id,
                run_id=run_id,
                artifact_name=f"model_{detector.algorithm}",
                artifact_data={"model_state": "trained", "version": "1.0"}
            )
            
            await enhanced_tracking_service.finish_run(
                experiment_id=experiment_id,
                run_id=run_id,
                status="completed"
            )
            
            run_results.append({
                "run_id": run_id,
                "algorithm": detector.algorithm,
                "metrics": metrics
            })
        
        # Finish experiment
        await enhanced_tracking_service.finish_experiment(
            experiment_id=experiment_id,
            status="completed"
        )
        
        # Test experiment analysis
        analysis = await enhanced_tracking_service.analyze_experiment(
            experiment_id=experiment_id,
            target_metric="f1_score",
            include_statistical_tests=True
        )
        
        assert "best_run" in analysis
        assert "metric_statistics" in analysis
        assert "algorithm_comparison" in analysis
        assert "hyperparameter_importance" in analysis
    
    @pytest.mark.asyncio
    async def test_experiment_comparison_and_ranking(self, enhanced_tracking_service, multiple_detectors):
        """Test comparing multiple experiments and ranking performance."""
        # Create multiple experiments
        experiment_ids = []
        experiment_results = []
        
        for exp_idx in range(3):
            config = {
                "dataset_version": f"v{exp_idx + 1}",
                "preprocessing": f"pipeline_{exp_idx}",
                "validation_strategy": "cross_validation"
            }
            
            exp_id = await enhanced_tracking_service.start_experiment(
                name=f"Experiment_{exp_idx + 1}",
                config=config
            )
            experiment_ids.append(exp_id)
            
            # Add runs to each experiment
            exp_metrics = []
            for detector in multiple_detectors[:3]:  # Use first 3 detectors
                run_id = await enhanced_tracking_service.start_run(
                    experiment_id=exp_id,
                    run_name=f"run_{detector.algorithm}",
                    config={"algorithm": detector.algorithm}
                )
                
                # Simulate different performance levels for each experiment
                base_performance = 0.8 + exp_idx * 0.05
                metrics = {
                    "f1_score": base_performance + np.random.normal(0, 0.02),
                    "accuracy": base_performance + 0.05 + np.random.normal(0, 0.02),
                    "training_time": 30 + exp_idx * 10 + np.random.normal(0, 5)
                }
                
                for metric_name, value in metrics.items():
                    await enhanced_tracking_service.log_metric(
                        experiment_id=exp_id,
                        run_id=run_id,
                        metric_name=metric_name,
                        value=float(value)
                    )
                
                exp_metrics.append(metrics)
                
                await enhanced_tracking_service.finish_run(exp_id, run_id, "completed")
            
            experiment_results.append(exp_metrics)
            await enhanced_tracking_service.finish_experiment(exp_id, "completed")
        
        # Test experiment comparison
        comparison = await enhanced_tracking_service.compare_experiments(
            experiment_ids=experiment_ids,
            metrics=["f1_score", "accuracy", "training_time"],
            statistical_tests=["t_test", "wilcoxon"],
            significance_level=0.05
        )
        
        assert "experiment_rankings" in comparison
        assert "statistical_significance" in comparison
        assert "performance_summary" in comparison
        assert "best_configurations" in comparison
        
        # Test leaderboard generation
        leaderboard = await enhanced_tracking_service.generate_leaderboard(
            experiments=experiment_ids,
            ranking_metric="f1_score",
            include_confidence_intervals=True,
            top_k=10
        )
        
        assert len(leaderboard) <= 10
        assert all("rank" in entry for entry in leaderboard)
        assert all("confidence_interval" in entry for entry in leaderboard)
    
    @pytest.mark.asyncio
    async def test_distributed_experiment_tracking(self, enhanced_tracking_service):
        """Test distributed experiment tracking across multiple workers."""
        # Simulate distributed experiment
        experiment_id = await enhanced_tracking_service.start_experiment(
            name="Distributed Training Experiment",
            config={"distributed": True, "workers": 4}
        )
        
        # Simulate multiple workers reporting metrics
        worker_results = []
        for worker_id in range(4):
            worker_metrics = {}
            
            # Each worker reports different metrics
            for epoch in range(10):
                metrics = {
                    "worker_loss": np.random.exponential(1.0) * (1 - epoch * 0.05),
                    "worker_accuracy": 0.5 + (epoch * 0.04) + np.random.normal(0, 0.01),
                    "processing_speed": 100 + np.random.normal(0, 10)
                }
                
                for metric_name, value in metrics.items():
                    await enhanced_tracking_service.log_distributed_metric(
                        experiment_id=experiment_id,
                        worker_id=worker_id,
                        metric_name=metric_name,
                        value=float(value),
                        timestamp=datetime.now(),
                        step=epoch
                    )
            
            worker_results.append(worker_metrics)
        
        # Test metric aggregation across workers
        aggregated_metrics = await enhanced_tracking_service.aggregate_distributed_metrics(
            experiment_id=experiment_id,
            aggregation_methods=["mean", "median", "std", "min", "max"]
        )
        
        assert "worker_loss" in aggregated_metrics
        assert "worker_accuracy" in aggregated_metrics
        assert all(method in aggregated_metrics["worker_loss"] for method in ["mean", "std", "min", "max"])
        
        await enhanced_tracking_service.finish_experiment(experiment_id, "completed")
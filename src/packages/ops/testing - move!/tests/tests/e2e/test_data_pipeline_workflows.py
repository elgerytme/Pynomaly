"""Data pipeline workflow end-to-end tests.

This module tests complete data processing pipelines from data ingestion
through preprocessing, validation, and anomaly detection.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from monorepo.infrastructure.config import create_container
from monorepo.presentation.api.app import create_app


class TestDataPipelineWorkflows:
    """Test complete data processing pipelines."""

    @pytest.fixture
    def app_client(self):
        """Create test client for API."""
        container = create_container()
        app = create_app(container)
        return TestClient(app)

    @pytest.fixture
    def raw_datasets(self):
        """Create various raw datasets for pipeline testing."""
        np.random.seed(42)

        datasets = {}

        # Dataset with missing values
        data_with_missing = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
                "feature_3": np.random.normal(0, 1, 100),
            }
        )
        # Introduce missing values
        missing_indices = np.random.choice(100, 15, replace=False)
        data_with_missing.loc[missing_indices, "feature_2"] = np.nan
        datasets["missing_values"] = data_with_missing

        # Dataset with categorical features
        categorical_data = pd.DataFrame(
            {
                "numeric_1": np.random.normal(0, 1, 200),
                "numeric_2": np.random.normal(0, 1, 200),
                "category_1": np.random.choice(["A", "B", "C"], 200),
                "category_2": np.random.choice(["X", "Y"], 200),
                "mixed_feature": np.random.choice([1, 2, 3, "other"], 200),
            }
        )
        datasets["categorical"] = categorical_data

        # Time series dataset
        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        time_series = pd.DataFrame(
            {
                "timestamp": dates,
                "value_1": np.sin(np.arange(500) * 0.1) + np.random.normal(0, 0.1, 500),
                "value_2": np.cos(np.arange(500) * 0.1) + np.random.normal(0, 0.1, 500),
                "trend": np.arange(500) * 0.01 + np.random.normal(0, 0.05, 500),
            }
        )
        # Add some time-based anomalies
        anomaly_indices = [100, 200, 300, 400]
        time_series.loc[anomaly_indices, "value_1"] += 3
        datasets["time_series"] = time_series

        # Large dataset for scalability testing
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.normal(0, 1, 5000) for i in range(10)}
        )
        # Add structured anomalies
        anomaly_mask = np.random.choice(5000, 250, replace=False)
        large_data.loc[anomaly_mask, :] *= 3
        datasets["large_scale"] = large_data

        # Dataset with extreme values
        extreme_data = pd.DataFrame(
            {
                "normal_feature": np.random.normal(0, 1, 300),
                "extreme_feature": np.concatenate(
                    [
                        np.random.normal(0, 1, 250),
                        np.array([100, -100, 1000, -1000, 50]),  # Extreme outliers
                    ]
                ),
                "bounded_feature": np.random.beta(2, 2, 300) * 10,
            }
        )
        datasets["extreme_values"] = extreme_data

        return datasets

    def test_complete_data_pipeline_workflow(self, app_client, raw_datasets):
        """Test complete data pipeline from raw data to detection results."""
        # Test with missing values dataset
        raw_data = raw_datasets["missing_values"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            raw_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Step 1: Upload raw dataset
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("raw_data.csv", file, "text/csv")},
                    data={"name": "Raw Pipeline Dataset"},
                )
            assert upload_response.status_code == 200
            raw_dataset_id = upload_response.json()["id"]

            # Step 2: Analyze data quality
            quality_response = app_client.post(
                "/api/data/analyze-quality",
                json={"dataset_id": raw_dataset_id, "include_recommendations": True},
            )
            assert quality_response.status_code == 200
            quality_result = quality_response.json()

            # Verify quality analysis
            assert "missing_values" in quality_result
            assert "data_types" in quality_result
            assert "outliers" in quality_result
            assert "recommendations" in quality_result

            # Step 3: Apply preprocessing pipeline
            preprocessing_config = {
                "dataset_id": raw_dataset_id,
                "pipeline_steps": [
                    {
                        "step": "handle_missing",
                        "method": "mean_imputation",
                        "columns": ["feature_2"],
                    },
                    {
                        "step": "normalize",
                        "method": "standard_scaler",
                        "columns": ["feature_1", "feature_2", "feature_3"],
                    },
                    {"step": "outlier_removal", "method": "iqr", "threshold": 3.0},
                ],
                "output_name": "Processed Pipeline Dataset",
            }

            preprocessing_response = app_client.post(
                "/api/data/preprocess", json=preprocessing_config
            )
            assert preprocessing_response.status_code == 200
            preprocessing_result = preprocessing_response.json()

            # Verify preprocessing
            assert "processed_dataset_id" in preprocessing_result
            assert "transformation_log" in preprocessing_result
            assert "quality_improvement" in preprocessing_result

            processed_dataset_id = preprocessing_result["processed_dataset_id"]

            # Step 4: Create and configure detector
            detector_data = {
                "name": "Pipeline Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1},
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            # Step 5: Train on processed data
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train",
                json={"dataset_id": processed_dataset_id},
            )
            assert train_response.status_code == 200

            # Step 6: Run detection
            detect_response = app_client.post(
                f"/api/detectors/{detector_id}/detect",
                json={"dataset_id": processed_dataset_id},
            )
            assert detect_response.status_code == 200
            detect_response.json()

            # Step 7: Generate pipeline report
            report_response = app_client.post(
                "/api/pipeline/generate-report",
                json={
                    "raw_dataset_id": raw_dataset_id,
                    "processed_dataset_id": processed_dataset_id,
                    "detector_id": detector_id,
                    "include_visualizations": True,
                },
            )
            assert report_response.status_code == 200
            report_result = report_response.json()

            # Verify complete pipeline report
            assert "data_quality_improvement" in report_result
            assert "preprocessing_impact" in report_result
            assert "detection_results" in report_result
            assert "pipeline_performance" in report_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_categorical_data_pipeline(self, app_client, raw_datasets):
        """Test pipeline handling categorical and mixed data types."""
        categorical_data = raw_datasets["categorical"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            categorical_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Upload categorical dataset
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("categorical_data.csv", file, "text/csv")},
                    data={"name": "Categorical Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Analyze categorical features
            analysis_response = app_client.post(
                "/api/data/analyze-categorical",
                json={
                    "dataset_id": dataset_id,
                    "include_cardinality": True,
                    "detect_ordinal": True,
                },
            )
            assert analysis_response.status_code == 200
            analysis_result = analysis_response.json()

            # Verify categorical analysis
            assert "categorical_features" in analysis_result
            assert "numeric_features" in analysis_result
            assert "encoding_recommendations" in analysis_result

            # Apply categorical preprocessing
            categorical_config = {
                "dataset_id": dataset_id,
                "encoding_strategy": "auto",
                "categorical_columns": ["category_1", "category_2", "mixed_feature"],
                "encoding_methods": {
                    "category_1": "one_hot",
                    "category_2": "label",
                    "mixed_feature": "target",
                },
                "handle_unknown": "ignore",
                "output_name": "Encoded Categorical Dataset",
            }

            encoding_response = app_client.post(
                "/api/data/encode-categorical", json=categorical_config
            )
            assert encoding_response.status_code == 200
            encoding_result = encoding_response.json()

            # Verify encoding results
            assert "encoded_dataset_id" in encoding_result
            assert "encoding_mapping" in encoding_result
            assert "feature_expansion" in encoding_result

            encoded_dataset_id = encoding_result["encoded_dataset_id"]

            # Train detector on encoded data
            detector_data = {
                "name": "Categorical Detector",
                "algorithm_name": "LocalOutlierFactor",
                "parameters": {"contamination": 0.1},
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train",
                json={"dataset_id": encoded_dataset_id},
            )
            assert train_response.status_code == 200

            detect_response = app_client.post(
                f"/api/detectors/{detector_id}/detect",
                json={"dataset_id": encoded_dataset_id},
            )
            assert detect_response.status_code == 200

            # Verify categorical feature handling worked
            detection_result = detect_response.json()
            assert "anomalies" in detection_result
            assert len(detection_result["anomalies"]) > 0

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_time_series_pipeline(self, app_client, raw_datasets):
        """Test time series data processing pipeline."""
        time_series_data = raw_datasets["time_series"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            time_series_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Upload time series dataset
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("timeseries_data.csv", file, "text/csv")},
                    data={"name": "Time Series Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Analyze time series characteristics
            ts_analysis_response = app_client.post(
                "/api/data/analyze-timeseries",
                json={
                    "dataset_id": dataset_id,
                    "timestamp_column": "timestamp",
                    "value_columns": ["value_1", "value_2", "trend"],
                    "detect_seasonality": True,
                    "detect_trend": True,
                },
            )
            assert ts_analysis_response.status_code == 200
            ts_analysis_result = ts_analysis_response.json()

            # Verify time series analysis
            assert "seasonality" in ts_analysis_result
            assert "trend_analysis" in ts_analysis_result
            assert "stationarity_test" in ts_analysis_result
            assert "anomaly_patterns" in ts_analysis_result

            # Apply time series preprocessing
            ts_preprocessing = {
                "dataset_id": dataset_id,
                "timestamp_column": "timestamp",
                "operations": [
                    {
                        "operation": "create_features",
                        "features": ["hour", "day_of_week", "month", "rolling_mean_7d"],
                    },
                    {"operation": "detrend", "method": "linear", "columns": ["trend"]},
                    {
                        "operation": "normalize",
                        "method": "min_max",
                        "columns": ["value_1", "value_2"],
                    },
                ],
                "output_name": "Processed Time Series Dataset",
            }

            ts_preprocessing_response = app_client.post(
                "/api/data/preprocess-timeseries", json=ts_preprocessing
            )
            assert ts_preprocessing_response.status_code == 200
            ts_preprocessing_result = ts_preprocessing_response.json()

            # Verify time series preprocessing
            assert "processed_dataset_id" in ts_preprocessing_result
            assert "feature_engineering_log" in ts_preprocessing_result
            assert "temporal_features_created" in ts_preprocessing_result

            processed_ts_id = ts_preprocessing_result["processed_dataset_id"]

            # Create time series specific detector
            ts_detector_data = {
                "name": "Time Series Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {
                    "contamination": 0.08,  # Lower contamination for time series
                    "random_state": 42,
                },
            }

            create_response = app_client.post("/api/detectors/", json=ts_detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            # Train and detect
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train",
                json={"dataset_id": processed_ts_id},
            )
            assert train_response.status_code == 200

            detect_response = app_client.post(
                f"/api/detectors/{detector_id}/detect",
                json={"dataset_id": processed_ts_id},
            )
            assert detect_response.status_code == 200

            # Analyze temporal patterns in anomalies
            temporal_analysis_response = app_client.post(
                "/api/analysis/temporal-anomalies",
                json={
                    "dataset_id": processed_ts_id,
                    "detector_id": detector_id,
                    "timestamp_column": "timestamp",
                    "analyze_patterns": True,
                },
            )
            assert temporal_analysis_response.status_code == 200
            temporal_result = temporal_analysis_response.json()

            # Verify temporal analysis
            assert "temporal_distribution" in temporal_result
            assert "seasonal_anomalies" in temporal_result
            assert "anomaly_clusters" in temporal_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_large_scale_pipeline(self, app_client, raw_datasets):
        """Test pipeline scalability with large datasets."""
        large_data = raw_datasets["large_scale"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            large_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Upload large dataset
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("large_data.csv", file, "text/csv")},
                    data={"name": "Large Scale Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Test chunked processing for large datasets
            chunked_config = {
                "dataset_id": dataset_id,
                "chunk_size": 1000,
                "processing_strategy": "parallel",
                "operations": [
                    {"operation": "standardize", "method": "robust_scaler"},
                    {
                        "operation": "dimensionality_reduction",
                        "method": "pca",
                        "n_components": 5,
                    },
                ],
                "output_name": "Processed Large Dataset",
            }

            chunked_response = app_client.post(
                "/api/data/process-chunked", json=chunked_config
            )
            assert chunked_response.status_code == 200
            chunked_result = chunked_response.json()

            # Verify chunked processing
            assert "processed_dataset_id" in chunked_result
            assert "processing_statistics" in chunked_result
            assert "memory_efficiency" in chunked_result

            processed_large_id = chunked_result["processed_dataset_id"]

            # Create scalable detector
            scalable_detector = {
                "name": "Scalable Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {
                    "contamination": 0.05,
                    "n_estimators": 100,
                    "max_samples": "auto",
                    "n_jobs": -1,  # Use all available cores
                },
            }

            create_response = app_client.post("/api/detectors/", json=scalable_detector)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            # Test batch training for large datasets
            batch_train_config = {
                "dataset_id": processed_large_id,
                "batch_size": 1000,
                "training_strategy": "incremental",
            }

            batch_train_response = app_client.post(
                f"/api/detectors/{detector_id}/train-batch", json=batch_train_config
            )
            assert batch_train_response.status_code == 200

            # Test batch detection
            batch_detect_config = {
                "dataset_id": processed_large_id,
                "batch_size": 1000,
                "parallel_processing": True,
            }

            batch_detect_response = app_client.post(
                f"/api/detectors/{detector_id}/detect-batch", json=batch_detect_config
            )
            assert batch_detect_response.status_code == 200
            batch_detect_result = batch_detect_response.json()

            # Verify scalable detection
            assert "batch_results" in batch_detect_result
            assert "processing_time" in batch_detect_result
            assert "memory_usage" in batch_detect_result
            assert "throughput" in batch_detect_result

            # Performance should be reasonable for large dataset
            throughput = batch_detect_result["throughput"]
            assert throughput > 100  # samples per second minimum

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_data_validation_pipeline(self, app_client, raw_datasets):
        """Test comprehensive data validation pipeline."""
        extreme_data = raw_datasets["extreme_values"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            extreme_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Upload dataset with extreme values
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("extreme_data.csv", file, "text/csv")},
                    data={"name": "Extreme Values Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Run comprehensive data validation
            validation_config = {
                "dataset_id": dataset_id,
                "validation_rules": [
                    {
                        "rule": "check_data_types",
                        "expected_types": {
                            "normal_feature": "float",
                            "extreme_feature": "float",
                            "bounded_feature": "float",
                        },
                    },
                    {
                        "rule": "check_ranges",
                        "feature_ranges": {
                            "normal_feature": {"min": -5, "max": 5},
                            "bounded_feature": {"min": 0, "max": 10},
                        },
                    },
                    {
                        "rule": "check_distributions",
                        "expected_distributions": {
                            "normal_feature": "normal",
                            "bounded_feature": "beta",
                        },
                    },
                    {
                        "rule": "detect_outliers",
                        "methods": ["iqr", "zscore", "isolation_forest"],
                        "threshold": 0.05,
                    },
                ],
                "generate_report": True,
            }

            validation_response = app_client.post(
                "/api/data/validate", json=validation_config
            )
            assert validation_response.status_code == 200
            validation_result = validation_response.json()

            # Verify validation results
            assert "validation_summary" in validation_result
            assert "rule_results" in validation_result
            assert "data_quality_score" in validation_result
            assert "recommendations" in validation_result

            # Check that extreme values were detected
            rule_results = validation_result["rule_results"]
            outlier_results = next(
                (r for r in rule_results if r["rule"] == "detect_outliers"), None
            )
            assert outlier_results is not None
            assert outlier_results["violations_detected"] > 0

            # Apply data cleaning based on validation
            cleaning_config = {
                "dataset_id": dataset_id,
                "cleaning_strategy": "conservative",
                "rules": [
                    {
                        "action": "cap_outliers",
                        "method": "percentile",
                        "percentiles": [1, 99],
                        "columns": ["extreme_feature"],
                    },
                    {
                        "action": "validate_ranges",
                        "enforce_bounds": True,
                        "columns": ["bounded_feature"],
                    },
                ],
                "output_name": "Validated Dataset",
            }

            cleaning_response = app_client.post("/api/data/clean", json=cleaning_config)
            assert cleaning_response.status_code == 200
            cleaning_result = cleaning_response.json()

            # Verify cleaning results
            assert "cleaned_dataset_id" in cleaning_result
            assert "cleaning_log" in cleaning_result
            assert "quality_improvement" in cleaning_result

            # Re-validate cleaned dataset
            cleaned_id = cleaning_result["cleaned_dataset_id"]
            revalidation_response = app_client.post(
                "/api/data/validate",
                json={**validation_config, "dataset_id": cleaned_id},
            )
            assert revalidation_response.status_code == 200
            revalidation_result = revalidation_response.json()

            # Verify data quality improved
            original_score = validation_result["data_quality_score"]
            cleaned_score = revalidation_result["data_quality_score"]
            assert cleaned_score > original_score

        finally:
            Path(dataset_file).unlink(missing_ok=True)

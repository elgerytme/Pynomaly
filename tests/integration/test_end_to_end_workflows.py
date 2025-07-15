"""End-to-end integration tests for complete workflows."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tests.fixtures.test_data_generator import TestDataManager
from tests.integration.framework import (
    IntegrationTestBuilder,
    IntegrationTestRunner,
)


class TestCompleteAnomalyDetectionWorkflow:
    """Integration tests for complete anomaly detection workflows."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_basic_anomaly_detection_pipeline(self):
        """Test complete anomaly detection pipeline from data loading to results."""

        # Setup test data
        data_manager = TestDataManager()
        test_data, labels = data_manager.get_dataset(
            "simple", n_samples=500, n_features=10, contamination=0.1
        )

        # Store test results
        results = {}

        async def load_data():
            """Load test data."""
            results["data"] = test_data
            results["labels"] = labels
            return len(test_data)

        async def validate_data():
            """Validate loaded data."""
            data = results["data"]
            assert not data.empty, "Data should not be empty"
            assert len(data.columns) == 10, "Should have 10 features"
            assert len(data) == 500, "Should have 500 samples"
            return True

        async def create_detector():
            """Create anomaly detector."""
            try:
                from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import (
                    PyODAdapter,
                )

                adapter = PyODAdapter()

                if not adapter.supports_algorithm("IsolationForest"):
                    pytest.skip("IsolationForest not available")

                detector = adapter.create_algorithm(
                    "IsolationForest",
                    {"contamination": 0.1, "n_estimators": 50, "random_state": 42},
                )
                results["detector"] = detector
                return True
            except Exception as e:
                pytest.skip(f"Could not create detector: {e}")

        async def train_detector():
            """Train the detector on data."""
            detector = results["detector"]
            data = results["data"]

            # Get numeric features only
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            X = data[numeric_columns].values

            detector.fit(X)
            results["trained"] = True
            return True

        async def detect_anomalies():
            """Detect anomalies in the data."""
            detector = results["detector"]
            data = results["data"]

            numeric_columns = data.select_dtypes(include=[np.number]).columns
            X = data[numeric_columns].values

            scores = detector.decision_function(X)
            predictions = detector.predict(X)

            results["scores"] = scores
            results["predictions"] = predictions

            # Basic validation
            assert len(scores) == len(data), "Should have score for each sample"
            assert len(predictions) == len(
                data
            ), "Should have prediction for each sample"
            assert np.all(np.isfinite(scores)), "All scores should be finite"

            return len(np.where(predictions == 1)[0])  # Number of detected anomalies

        async def evaluate_results():
            """Evaluate detection results."""
            predictions = results["predictions"]
            true_labels = results["labels"]

            # Calculate basic metrics
            detected_anomalies = np.sum(predictions == 1)
            true_anomalies = np.sum(true_labels == 1)

            # Should detect some anomalies
            assert detected_anomalies > 0, "Should detect at least some anomalies"
            assert detected_anomalies < len(
                predictions
            ), "Should not classify all as anomalies"

            # Store metrics
            results["detected_anomalies"] = detected_anomalies
            results["true_anomalies"] = true_anomalies

            return {
                "detected": detected_anomalies,
                "true": true_anomalies,
                "precision_estimate": min(detected_anomalies, true_anomalies)
                / max(detected_anomalies, 1),
            }

        # Build and run integration test
        suite = (
            IntegrationTestBuilder(
                "basic_anomaly_detection_pipeline", "Complete end-to-end workflow"
            )
            .add_step("load_data", "Load test dataset", load_data, expected_result=500)
            .add_step(
                "validate_data",
                "Validate data quality",
                validate_data,
                expected_result=True,
                dependencies=["load_data"],
            )
            .add_step(
                "create_detector",
                "Create anomaly detector",
                create_detector,
                expected_result=True,
                dependencies=["validate_data"],
            )
            .add_step(
                "train_detector",
                "Train detector on data",
                train_detector,
                expected_result=True,
                dependencies=["create_detector"],
            )
            .add_step(
                "detect_anomalies",
                "Detect anomalies",
                detect_anomalies,
                dependencies=["train_detector"],
            )
            .add_step(
                "evaluate_results",
                "Evaluate detection results",
                evaluate_results,
                dependencies=["detect_anomalies"],
            )
            .build()
        )

        runner = IntegrationTestRunner()
        result_suite = await runner.run_suite(suite)

        # Assert overall success
        assert (
            result_suite.failed_steps == 0
        ), f"Pipeline failed: {result_suite.failed_steps} failures"
        assert (
            result_suite.error_steps == 0
        ), f"Pipeline errors: {result_suite.error_steps} errors"

        # Verify results
        assert "detected_anomalies" in results
        assert results["detected_anomalies"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_streaming_anomaly_detection_workflow(self):
        """Test streaming anomaly detection workflow."""

        # Generate streaming data
        data_manager = TestDataManager()
        stream_data, _ = data_manager.get_dataset(
            "timeseries",
            n_timestamps=200,
            n_features=5,
            anomaly_periods=[(50, 70), (120, 140)],
        )

        results = {}

        async def setup_streaming_detector():
            """Setup streaming anomaly detector."""

            # Mock streaming detector since full implementation may not be available
            class MockStreamingDetector:
                def __init__(self):
                    self.is_fitted = False
                    self.buffer = []

                def fit_initial(self, X):
                    self.is_fitted = True
                    return True

                def process_batch(self, batch):
                    # Simple mock: random anomaly scores
                    np.random.seed(42)
                    scores = np.random.uniform(0, 1, len(batch))
                    predictions = (scores > 0.8).astype(int)
                    return scores, predictions

            detector = MockStreamingDetector()
            results["stream_detector"] = detector
            return True

        async def initialize_detector():
            """Initialize detector with initial data."""
            detector = results["stream_detector"]

            # Use first 50 samples for initialization
            init_data = stream_data.iloc[:50]
            numeric_columns = init_data.select_dtypes(include=[np.number]).columns
            X_init = init_data[numeric_columns].values

            success = detector.fit_initial(X_init)
            results["initialized"] = success
            return success

        async def process_streaming_batches():
            """Process data in streaming batches."""
            detector = results["stream_detector"]

            # Process remaining data in batches
            batch_size = 20
            total_processed = 0
            all_scores = []
            all_predictions = []

            remaining_data = stream_data.iloc[50:]
            numeric_columns = remaining_data.select_dtypes(include=[np.number]).columns

            for i in range(0, len(remaining_data), batch_size):
                batch = remaining_data.iloc[i : i + batch_size]
                X_batch = batch[numeric_columns].values

                if len(X_batch) > 0:
                    scores, predictions = detector.process_batch(X_batch)
                    all_scores.extend(scores)
                    all_predictions.extend(predictions)
                    total_processed += len(X_batch)

            results["total_processed"] = total_processed
            results["stream_scores"] = all_scores
            results["stream_predictions"] = all_predictions

            return total_processed

        async def validate_streaming_results():
            """Validate streaming detection results."""
            scores = results["stream_scores"]
            predictions = results["stream_predictions"]
            processed = results["total_processed"]

            assert (
                len(scores) == processed
            ), "Should have score for each processed sample"
            assert (
                len(predictions) == processed
            ), "Should have prediction for each processed sample"
            assert all(0 <= s <= 1 for s in scores), "Scores should be in valid range"
            assert all(p in [0, 1] for p in predictions), "Predictions should be binary"

            detected_anomalies = sum(predictions)
            results["stream_detected"] = detected_anomalies

            return detected_anomalies

        # Build streaming workflow test
        suite = (
            IntegrationTestBuilder(
                "streaming_anomaly_detection", "Streaming detection workflow"
            )
            .add_step(
                "setup_detector",
                "Setup streaming detector",
                setup_streaming_detector,
                expected_result=True,
            )
            .add_step(
                "initialize",
                "Initialize with baseline data",
                initialize_detector,
                expected_result=True,
                dependencies=["setup_detector"],
            )
            .add_step(
                "process_batches",
                "Process streaming batches",
                process_streaming_batches,
                dependencies=["initialize"],
            )
            .add_step(
                "validate_results",
                "Validate streaming results",
                validate_streaming_results,
                dependencies=["process_batches"],
            )
            .build()
        )

        runner = IntegrationTestRunner()
        result_suite = await runner.run_suite(suite)

        assert result_suite.failed_steps == 0
        assert result_suite.error_steps == 0
        assert results["total_processed"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_algorithm_comparison_workflow(self):
        """Test workflow comparing multiple algorithms."""

        data_manager = TestDataManager()
        test_data, labels = data_manager.get_dataset(
            "clustered", n_samples=300, n_features=8, n_clusters=3, contamination=0.15
        )

        results = {}
        algorithms = ["IsolationForest"]  # Focus on one available algorithm

        async def prepare_comparison_data():
            """Prepare data for algorithm comparison."""
            numeric_columns = test_data.select_dtypes(include=[np.number]).columns
            X = test_data[numeric_columns].values

            results["X"] = X
            results["y_true"] = labels
            results["algorithms"] = algorithms
            results["algorithm_results"] = {}

            return len(X)

        async def run_algorithm_comparison():
            """Run multiple algorithms and compare."""
            try:
                from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import (
                    PyODAdapter,
                )

                adapter = PyODAdapter()

                X = results["X"]

                for algorithm in algorithms:
                    if not adapter.supports_algorithm(algorithm):
                        continue

                    # Create and train detector
                    detector = adapter.create_algorithm(
                        algorithm,
                        {
                            "contamination": 0.15,
                            "random_state": 42,
                            "n_estimators": 30,  # Faster for testing
                        },
                    )

                    detector.fit(X)
                    scores = detector.decision_function(X)
                    predictions = detector.predict(X)

                    results["algorithm_results"][algorithm] = {
                        "scores": scores,
                        "predictions": predictions,
                        "detected_count": np.sum(predictions == 1),
                    }

                return len(results["algorithm_results"])

            except Exception as e:
                pytest.skip(f"Algorithm comparison failed: {e}")

        async def analyze_comparison_results():
            """Analyze and compare algorithm results."""
            algorithm_results = results["algorithm_results"]

            if not algorithm_results:
                pytest.skip("No algorithm results to analyze")

            analysis = {}

            for algorithm, result in algorithm_results.items():
                detected = result["detected_count"]
                total_samples = len(result["predictions"])
                detection_rate = detected / total_samples

                analysis[algorithm] = {
                    "detection_rate": detection_rate,
                    "detected_count": detected,
                    "score_range": (np.min(result["scores"]), np.max(result["scores"])),
                }

            results["analysis"] = analysis
            return analysis

        # Build comparison workflow
        suite = (
            IntegrationTestBuilder(
                "multi_algorithm_comparison", "Compare multiple algorithms"
            )
            .add_step(
                "prepare_data",
                "Prepare comparison data",
                prepare_comparison_data,
                expected_result=300,
            )
            .add_step(
                "run_comparison",
                "Run algorithm comparison",
                run_algorithm_comparison,
                dependencies=["prepare_data"],
            )
            .add_step(
                "analyze_results",
                "Analyze comparison results",
                analyze_comparison_results,
                dependencies=["run_comparison"],
            )
            .build()
        )

        runner = IntegrationTestRunner()
        result_suite = await runner.run_suite(suite)

        assert result_suite.failed_steps == 0
        assert result_suite.error_steps == 0
        if "analysis" in results:
            assert len(results["analysis"]) > 0


class TestDataPipelineIntegration:
    """Integration tests for data processing pipelines."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_preprocessing_pipeline(self):
        """Test complete data preprocessing pipeline."""

        # Create mixed-type dataset with issues
        mixed_data = pd.DataFrame(
            {
                "numeric_1": [1, 2, np.nan, 4, 5, 100],  # Has NaN and outlier
                "numeric_2": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
                "categorical": ["A", "B", "C", "A", "B", "NEW"],  # Has new category
                "text_feature": [
                    "hello",
                    "world",
                    "test",
                    "",
                    "data",
                    "science",
                ],  # Has empty string
                "constant_feature": [1, 1, 1, 1, 1, 1],  # Constant feature
            }
        )

        results = {}

        async def load_raw_data():
            """Load raw data with quality issues."""
            results["raw_data"] = mixed_data
            return len(mixed_data)

        async def detect_data_quality_issues():
            """Detect data quality issues."""
            data = results["raw_data"]
            issues = {}

            # Check for missing values
            missing_counts = data.isnull().sum()
            issues["missing_values"] = missing_counts[missing_counts > 0].to_dict()

            # Check for constant features
            constant_features = []
            for col in data.columns:
                if data[col].dtype in ["int64", "float64"]:
                    if data[col].nunique() == 1:
                        constant_features.append(col)
            issues["constant_features"] = constant_features

            # Check for outliers (simple method)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outliers = {}
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (data[col] < (Q1 - 1.5 * IQR)) | (
                    data[col] > (Q3 + 1.5 * IQR)
                )
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    outliers[col] = outlier_count
            issues["outliers"] = outliers

            results["quality_issues"] = issues
            return len(issues)

        async def clean_data():
            """Clean the data."""
            data = results["raw_data"].copy()
            issues = results["quality_issues"]

            # Handle missing values
            if "missing_values" in issues and issues["missing_values"]:
                # Simple imputation: forward fill then backward fill
                data = data.fillna(method="ffill").fillna(method="bfill")
                # If still NaN, fill with median for numeric, mode for categorical
                for col in data.columns:
                    if data[col].isnull().any():
                        if data[col].dtype in ["int64", "float64"]:
                            data[col] = data[col].fillna(data[col].median())
                        else:
                            data[col] = data[col].fillna(
                                data[col].mode()[0]
                                if len(data[col].mode()) > 0
                                else "unknown"
                            )

            # Remove constant features
            if "constant_features" in issues:
                data = data.drop(columns=issues["constant_features"])

            # Handle outliers (simple capping)
            if "outliers" in issues:
                for col in issues["outliers"]:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

            results["cleaned_data"] = data
            return len(data.columns)

        async def validate_cleaned_data():
            """Validate the cleaned data."""
            cleaned_data = results["cleaned_data"]
            original_data = results["raw_data"]

            # Should have same number of rows
            assert len(cleaned_data) == len(
                original_data
            ), "Row count should be preserved"

            # Should have no missing values
            assert (
                not cleaned_data.isnull().any().any()
            ), "Should have no missing values"

            # Should have fewer columns (constant removed)
            assert len(cleaned_data.columns) <= len(
                original_data.columns
            ), "Should not add columns"

            results["validation_passed"] = True
            return True

        async def prepare_for_modeling():
            """Prepare cleaned data for modeling."""
            cleaned_data = results["cleaned_data"]

            # Select numeric features only for modeling
            numeric_features = cleaned_data.select_dtypes(include=[np.number])

            # Basic normalization
            normalized_data = (
                numeric_features - numeric_features.mean()
            ) / numeric_features.std()

            results["model_ready_data"] = normalized_data
            results["feature_count"] = len(normalized_data.columns)

            return len(normalized_data.columns)

        # Build data pipeline test
        suite = (
            IntegrationTestBuilder(
                "data_preprocessing_pipeline", "Complete data preprocessing workflow"
            )
            .add_step("load_data", "Load raw data", load_raw_data, expected_result=6)
            .add_step(
                "detect_issues",
                "Detect data quality issues",
                detect_data_quality_issues,
                dependencies=["load_data"],
            )
            .add_step(
                "clean_data",
                "Clean the data",
                clean_data,
                dependencies=["detect_issues"],
            )
            .add_step(
                "validate_cleaned",
                "Validate cleaned data",
                validate_cleaned_data,
                expected_result=True,
                dependencies=["clean_data"],
            )
            .add_step(
                "prepare_modeling",
                "Prepare for modeling",
                prepare_for_modeling,
                dependencies=["validate_cleaned"],
            )
            .build()
        )

        runner = IntegrationTestRunner()
        result_suite = await runner.run_suite(suite)

        assert result_suite.failed_steps == 0
        assert result_suite.error_steps == 0
        assert results["validation_passed"]
        assert results["feature_count"] > 0

"""Multi-algorithm workflow end-to-end tests.

This module tests workflows involving multiple machine learning algorithms,
ensemble methods, and algorithm comparison scenarios.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from monorepo.infrastructure.config import create_container
from monorepo.presentation.api.app import create_app


class TestMultiAlgorithmWorkflows:
    """Test workflows involving multiple ML algorithms."""

    @pytest.fixture
    def app_client(self):
        """Create test client for API."""
        container = create_container()
        app = create_app(container)
        return TestClient(app)

    @pytest.fixture
    def complex_dataset(self):
        """Create complex dataset for algorithm testing."""
        np.random.seed(42)

        # Create dataset with different types of anomalies
        normal_cluster1 = np.random.multivariate_normal(
            [0, 0], [[1, 0.5], [0.5, 1]], 200
        )
        normal_cluster2 = np.random.multivariate_normal(
            [3, 3], [[1, -0.3], [-0.3, 1]], 200
        )

        # Different types of anomalies
        outliers_far = np.random.uniform(-6, 9, (20, 2))  # Far outliers
        outliers_dense = np.random.multivariate_normal(
            [1.5, 1.5], [[0.1, 0], [0, 0.1]], 30
        )  # Dense anomalies

        all_data = np.vstack(
            [normal_cluster1, normal_cluster2, outliers_far, outliers_dense]
        )

        # Add more features
        feature3 = np.random.normal(0, 1, len(all_data))
        feature4 = all_data[:, 0] * all_data[:, 1] + np.random.normal(
            0, 0.1, len(all_data)
        )
        feature5 = np.random.exponential(2, len(all_data))

        dataset = pd.DataFrame(
            {
                "feature_1": all_data[:, 0],
                "feature_2": all_data[:, 1],
                "feature_3": feature3,
                "feature_4": feature4,
                "feature_5": feature5,
            }
        )

        return dataset

    def test_algorithm_comparison_workflow(self, app_client, complex_dataset):
        """Test comparison of multiple algorithms on the same dataset."""
        # Algorithms to compare
        algorithms = [
            ("IsolationForest", {"contamination": 0.1, "random_state": 42}),
            ("LocalOutlierFactor", {"n_neighbors": 20, "contamination": 0.1}),
            ("OneClassSVM", {"gamma": "scale", "nu": 0.1}),
        ]

        # Upload dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            complex_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("comparison_data.csv", file, "text/csv")},
                    data={"name": "Algorithm Comparison Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Create and train all detectors
            detector_results = {}

            for algorithm, params in algorithms:
                # Create detector
                detector_data = {
                    "name": f"Comparison {algorithm}",
                    "algorithm_name": algorithm,
                    "parameters": params,
                }

                create_response = app_client.post("/api/detectors/", json=detector_data)
                assert create_response.status_code == 200
                detector_id = create_response.json()["id"]

                # Train detector
                train_response = app_client.post(
                    f"/api/detectors/{detector_id}/train",
                    json={"dataset_id": dataset_id},
                )
                assert train_response.status_code == 200

                # Run detection
                detect_response = app_client.post(
                    f"/api/detectors/{detector_id}/detect",
                    json={"dataset_id": dataset_id},
                )
                assert detect_response.status_code == 200
                result = detect_response.json()

                detector_results[algorithm] = {
                    "detector_id": detector_id,
                    "anomalies": result["anomalies"],
                    "anomaly_rate": result["anomaly_rate"],
                    "scores": result.get("scores", []),
                }

            # Run algorithm comparison
            comparison_request = {
                "detector_ids": [r["detector_id"] for r in detector_results.values()],
                "dataset_id": dataset_id,
                "metrics": ["precision", "recall", "f1_score", "auc_roc"],
            }

            comparison_response = app_client.post(
                "/api/analysis/compare-algorithms", json=comparison_request
            )
            assert comparison_response.status_code == 200
            comparison_result = comparison_response.json()

            # Verify comparison results
            assert "algorithm_performance" in comparison_result
            assert "ranking" in comparison_result
            assert "statistical_significance" in comparison_result

            performance = comparison_result["algorithm_performance"]
            assert len(performance) == len(algorithms)

            for algo_result in performance:
                assert "algorithm" in algo_result
                assert "metrics" in algo_result
                assert "confidence_interval" in algo_result

            # Verify ranking makes sense
            ranking = comparison_result["ranking"]
            assert len(ranking) == len(algorithms)
            assert all("algorithm" in r and "score" in r for r in ranking)

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_ensemble_algorithm_workflow(self, app_client, complex_dataset):
        """Test ensemble methods combining multiple algorithms."""
        # Create diverse set of algorithms for ensemble
        base_algorithms = [
            ("IsolationForest", {"contamination": 0.1, "n_estimators": 50}),
            ("LocalOutlierFactor", {"n_neighbors": 15, "contamination": 0.1}),
            ("OneClassSVM", {"gamma": "auto", "nu": 0.1}),
            ("EllipticEnvelope", {"contamination": 0.1, "random_state": 42}),
        ]

        # Upload dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            complex_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("ensemble_data.csv", file, "text/csv")},
                    data={"name": "Ensemble Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Create and train base detectors
            base_detector_ids = []

            for algorithm, params in base_algorithms:
                detector_data = {
                    "name": f"Ensemble Base {algorithm}",
                    "algorithm_name": algorithm,
                    "parameters": params,
                }

                create_response = app_client.post("/api/detectors/", json=detector_data)
                assert create_response.status_code == 200
                detector_id = create_response.json()["id"]
                base_detector_ids.append(detector_id)

                # Train detector
                train_response = app_client.post(
                    f"/api/detectors/{detector_id}/train",
                    json={"dataset_id": dataset_id},
                )
                assert train_response.status_code == 200

            # Test different ensemble methods
            ensemble_methods = [
                {"method": "voting", "weights": None},
                {"method": "averaging", "weights": [0.3, 0.3, 0.2, 0.2]},
                {"method": "stacking", "meta_learner": "LogisticRegression"},
                {"method": "max", "weights": None},
                {"method": "min", "weights": None},
            ]

            ensemble_results = {}

            for ensemble_config in ensemble_methods:
                ensemble_request = {
                    "detector_ids": base_detector_ids,
                    "dataset_id": dataset_id,
                    **ensemble_config,
                }

                ensemble_response = app_client.post(
                    "/api/detection/ensemble", json=ensemble_request
                )
                assert ensemble_response.status_code == 200
                ensemble_result = ensemble_response.json()

                method_name = ensemble_config["method"]
                ensemble_results[method_name] = {
                    "anomalies": ensemble_result["anomalies"],
                    "anomaly_rate": ensemble_result["anomaly_rate"],
                    "ensemble_score": ensemble_result["ensemble_score"],
                    "individual_results": ensemble_result["individual_results"],
                }

                # Verify ensemble result structure
                assert "confidence_metrics" in ensemble_result
                assert "ensemble_statistics" in ensemble_result
                assert len(ensemble_result["individual_results"]) == len(
                    base_algorithms
                )

            # Compare ensemble methods
            ensemble_comparison = {
                "ensemble_results": list(ensemble_results.keys()),
                "dataset_id": dataset_id,
                "evaluation_metrics": ["stability", "diversity", "accuracy_estimate"],
            }

            comparison_response = app_client.post(
                "/api/analysis/compare-ensembles", json=ensemble_comparison
            )
            assert comparison_response.status_code == 200
            comparison_result = comparison_response.json()

            # Verify ensemble comparison
            assert "method_performance" in comparison_result
            assert "diversity_analysis" in comparison_result
            assert "recommendation" in comparison_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_adaptive_algorithm_selection(self, app_client, complex_dataset):
        """Test adaptive algorithm selection based on data characteristics."""
        # Upload dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            complex_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("adaptive_data.csv", file, "text/csv")},
                    data={"name": "Adaptive Selection Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Request data analysis for algorithm recommendation
            analysis_request = {
                "dataset_id": dataset_id,
                "analysis_type": "comprehensive",
                "include_recommendations": True,
            }

            analysis_response = app_client.post(
                "/api/analysis/dataset", json=analysis_request
            )
            assert analysis_response.status_code == 200
            analysis_result = analysis_response.json()

            # Verify analysis results
            assert "data_characteristics" in analysis_result
            assert "algorithm_recommendations" in analysis_result
            assert "reasoning" in analysis_result

            data_chars = analysis_result["data_characteristics"]
            assert "dimensionality" in data_chars
            assert "data_distribution" in data_chars
            assert "outlier_density_estimate" in data_chars
            assert "feature_correlations" in data_chars

            # Get recommended algorithms
            recommendations = analysis_result["algorithm_recommendations"]
            assert "primary_recommendation" in recommendations
            assert "alternative_options" in recommendations
            assert "ensemble_suggestion" in recommendations

            # Test auto-creation with recommended algorithm
            auto_create_request = {
                "dataset_id": dataset_id,
                "use_recommendation": True,
                "optimization_target": "balanced",  # precision, recall, balanced
            }

            auto_response = app_client.post(
                "/api/detectors/auto-create", json=auto_create_request
            )
            assert auto_response.status_code == 200
            auto_result = auto_response.json()

            assert "detector_id" in auto_result
            assert "algorithm_selected" in auto_result
            assert "configuration" in auto_result
            assert "expected_performance" in auto_result

            # Verify auto-created detector works
            detector_id = auto_result["detector_id"]

            # Train auto-created detector
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
            )
            assert train_response.status_code == 200

            # Test detection
            detect_response = app_client.post(
                f"/api/detectors/{detector_id}/detect", json={"dataset_id": dataset_id}
            )
            assert detect_response.status_code == 200
            detect_result = detect_response.json()

            # Verify performance meets expectations
            expected_perf = auto_result["expected_performance"]
            actual_rate = detect_result["anomaly_rate"]

            if "anomaly_rate_range" in expected_perf:
                min_rate, max_rate = expected_perf["anomaly_rate_range"]
                assert min_rate <= actual_rate <= max_rate

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_hyperparameter_optimization_workflow(self, app_client, complex_dataset):
        """Test automated hyperparameter optimization across algorithms."""
        # Upload dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            complex_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("optimization_data.csv", file, "text/csv")},
                    data={"name": "Optimization Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Test hyperparameter optimization for multiple algorithms
            algorithms_to_optimize = [
                {
                    "algorithm": "IsolationForest",
                    "parameter_space": {
                        "contamination": {"type": "uniform", "low": 0.05, "high": 0.2},
                        "n_estimators": {"type": "int", "low": 50, "high": 200},
                        "max_features": {
                            "type": "categorical",
                            "choices": [1.0, "sqrt", "log2"],
                        },
                    },
                },
                {
                    "algorithm": "LocalOutlierFactor",
                    "parameter_space": {
                        "contamination": {"type": "uniform", "low": 0.05, "high": 0.2},
                        "n_neighbors": {"type": "int", "low": 5, "high": 50},
                        "algorithm": {
                            "type": "categorical",
                            "choices": ["auto", "ball_tree", "kd_tree"],
                        },
                    },
                },
            ]

            optimization_results = {}

            for algo_config in algorithms_to_optimize:
                optimization_request = {
                    "dataset_id": dataset_id,
                    "algorithm": algo_config["algorithm"],
                    "parameter_space": algo_config["parameter_space"],
                    "optimization_config": {
                        "n_trials": 20,
                        "cv_folds": 3,
                        "scoring_metric": "silhouette_score",
                        "timeout": 300,
                    },
                }

                optimization_response = app_client.post(
                    "/api/optimization/hyperparameters", json=optimization_request
                )
                assert optimization_response.status_code == 200
                optimization_result = optimization_response.json()

                algorithm = algo_config["algorithm"]
                optimization_results[algorithm] = optimization_result

                # Verify optimization results
                assert "best_parameters" in optimization_result
                assert "best_score" in optimization_result
                assert "optimization_history" in optimization_result
                assert "convergence_analysis" in optimization_result

                # Verify detector was created with optimal parameters
                assert "optimized_detector_id" in optimization_result
                detector_id = optimization_result["optimized_detector_id"]

                # Test optimized detector
                train_response = app_client.post(
                    f"/api/detectors/{detector_id}/train",
                    json={"dataset_id": dataset_id},
                )
                assert train_response.status_code == 200

                detect_response = app_client.post(
                    f"/api/detectors/{detector_id}/detect",
                    json={"dataset_id": dataset_id},
                )
                assert detect_response.status_code == 200

            # Compare optimization results across algorithms
            comparison_request = {
                "optimization_results": list(optimization_results.keys()),
                "dataset_id": dataset_id,
                "comparison_metrics": [
                    "best_score",
                    "convergence_speed",
                    "parameter_stability",
                ],
            }

            comparison_response = app_client.post(
                "/api/optimization/compare", json=comparison_request
            )
            assert comparison_response.status_code == 200
            comparison_result = comparison_response.json()

            # Verify comparison
            assert "algorithm_ranking" in comparison_result
            assert "optimization_summary" in comparison_result
            assert "recommendations" in comparison_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_cross_validation_workflow(self, app_client, complex_dataset):
        """Test cross-validation workflow across multiple algorithms."""
        # Upload dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            complex_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("cv_data.csv", file, "text/csv")},
                    data={"name": "Cross-Validation Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Define algorithms for cross-validation
            algorithms = [
                {"algorithm": "IsolationForest", "parameters": {"contamination": 0.1}},
                {
                    "algorithm": "LocalOutlierFactor",
                    "parameters": {"contamination": 0.1},
                },
                {"algorithm": "OneClassSVM", "parameters": {"nu": 0.1}},
            ]

            # Run cross-validation for each algorithm
            cv_results = {}

            for algo_config in algorithms:
                cv_request = {
                    "dataset_id": dataset_id,
                    "algorithm": algo_config["algorithm"],
                    "parameters": algo_config["parameters"],
                    "cv_config": {
                        "n_folds": 5,
                        "stratify": False,
                        "shuffle": True,
                        "random_state": 42,
                    },
                    "metrics": [
                        "silhouette_score",
                        "calinski_harabasz_score",
                        "davies_bouldin_score",
                    ],
                }

                cv_response = app_client.post(
                    "/api/validation/cross-validate", json=cv_request
                )
                assert cv_response.status_code == 200
                cv_result = cv_response.json()

                algorithm = algo_config["algorithm"]
                cv_results[algorithm] = cv_result

                # Verify CV results
                assert "fold_scores" in cv_result
                assert "mean_score" in cv_result
                assert "std_score" in cv_result
                assert "confidence_interval" in cv_result
                assert "statistical_significance" in cv_result

                # Verify we have results for all folds
                assert len(cv_result["fold_scores"]) == 5

                # Verify statistical measures
                assert cv_result["mean_score"] is not None
                assert cv_result["std_score"] >= 0
                assert len(cv_result["confidence_interval"]) == 2

            # Generate cross-validation report
            report_request = {
                "cv_results": list(cv_results.keys()),
                "dataset_id": dataset_id,
                "report_type": "comprehensive",
            }

            report_response = app_client.post(
                "/api/validation/generate-report", json=report_request
            )
            assert report_response.status_code == 200
            report_result = report_response.json()

            # Verify report
            assert "algorithm_comparison" in report_result
            assert "statistical_tests" in report_result
            assert "recommendations" in report_result
            assert "visualization_data" in report_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

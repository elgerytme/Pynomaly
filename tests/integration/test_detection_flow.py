"""Integration tests for complete detection flow."""

from __future__ import annotations

import pandas as pd
import pytest
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.infrastructure.config import Container


class TestDetectionFlow:
    """Test complete detection workflow."""

    @pytest.mark.asyncio
    async def test_full_detection_workflow(self, container: Container):
        """Test complete workflow from data loading to results."""
        # 1. Create and save detector
        detector = Detector(
            name="Integration Test Detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1, "random_state": 42},
        )
        detector_repo = container.detector_repository()
        detector_repo.save(detector)

        # 2. Create and save dataset
        data = pd.DataFrame(
            {
                "feature1": list(range(100)) + [200, 201, 202],  # Add outliers
                "feature2": list(range(100)) + [-50, -51, -52],  # Add outliers
            }
        )
        dataset = Dataset(name="Integration Test Data", data=data)
        dataset_repo = container.dataset_repository()
        dataset_repo.save(dataset)

        # 3. Train detector
        train_use_case = container.train_detector_use_case()
        from pynomaly.application.use_cases import TrainDetectorRequest

        train_request = TrainDetectorRequest(
            detector_id=detector.id,
            dataset=dataset,
            validate_data=True,
            save_model=True,
        )
        train_response = await train_use_case.execute(train_request)

        assert train_response.success
        assert train_response.training_time_ms > 0

        # 4. Run detection
        detect_use_case = container.detect_anomalies_use_case()
        from pynomaly.application.use_cases import DetectAnomaliesRequest

        detect_request = DetectAnomaliesRequest(
            detector_id=detector.id,
            dataset=dataset,
            validate_features=True,
            save_results=True,
        )
        detect_response = await detect_use_case.execute(detect_request)

        assert detect_response.success
        assert detect_response.result.n_anomalies > 0
        assert detect_response.result.n_anomalies <= 15  # ~10% contamination

        # 5. Verify results are saved
        result_repo = container.result_repository()
        results = result_repo.find_by_detector(detector.id)
        assert len(results) == 1
        assert results[0].id == detect_response.result.id

    @pytest.mark.asyncio
    async def test_ensemble_detection(self, container: Container):
        """Test ensemble detection with multiple algorithms."""
        # Create dataset
        data = pd.DataFrame(
            {
                "x": list(range(100)) + [150, 160, 170],
                "y": list(range(100)) + [-80, -90, -100],
            }
        )
        dataset = Dataset(name="Ensemble Test Data", data=data)
        dataset_repo = container.dataset_repository()
        dataset_repo.save(dataset)

        # Create multiple detectors
        algorithms = ["IsolationForest", "LOF", "OCSVM"]
        detectors = []

        for algo in algorithms:
            detector = Detector(
                name=f"{algo} Detector",
                algorithm=algo,
                parameters={"contamination": 0.05},
            )
            detector_repo = container.detector_repository()
            detector_repo.save(detector)
            detectors.append(detector)

        # Train all detectors
        train_use_case = container.train_detector_use_case()
        from pynomaly.application.use_cases import TrainDetectorRequest

        for detector in detectors:
            train_request = TrainDetectorRequest(
                detector_id=detector.id,
                dataset=dataset,
                validate_data=True,
                save_model=True,
            )
            await train_use_case.execute(train_request)

        # Run ensemble detection
        ensemble_service = container.ensemble_service()
        ensemble_result = await ensemble_service.detect_with_ensemble(
            detector_ids=[d.id for d in detectors],
            dataset=dataset,
            aggregation_method="average",
        )

        assert ensemble_result is not None
        assert ensemble_result.n_anomalies > 0
        assert "ensemble_scores" in ensemble_result.metadata

    @pytest.mark.asyncio
    async def test_experiment_tracking(self, container: Container):
        """Test experiment tracking functionality."""
        # Create experiment
        experiment_service = container.experiment_tracking_service()

        experiment_id = await experiment_service.create_experiment(
            name="Integration Test Experiment",
            description="Testing experiment tracking",
            tags=["test", "integration"],
        )

        # Create detector and dataset
        detector = Detector(
            name="Experiment Detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1},
        )
        dataset = Dataset(
            name="Experiment Data",
            data=pd.DataFrame({"a": range(100), "b": range(100, 200)}),
        )

        # Save entities
        container.detector_repository().save(detector)
        container.dataset_repository().save(dataset)

        # Train and evaluate
        train_use_case = container.train_detector_use_case()
        evaluate_use_case = container.evaluate_model_use_case()

        from pynomaly.application.use_cases import (
            EvaluateModelRequest,
            TrainDetectorRequest,
        )

        # Add labels for evaluation
        dataset.data["label"] = [0] * 95 + [1] * 5
        dataset.target_column = "label"

        # Train
        await train_use_case.execute(
            TrainDetectorRequest(
                detector_id=detector.id,
                dataset=dataset,
                validate_data=True,
                save_model=True,
            )
        )

        # Evaluate
        eval_response = await evaluate_use_case.execute(
            EvaluateModelRequest(
                detector_id=detector.id,
                test_dataset=dataset,
                cross_validate=False,
                metrics=["precision", "recall", "f1"],
            )
        )

        # Log run to experiment
        run_id = await experiment_service.log_run(
            experiment_id=experiment_id,
            detector_name=detector.name,
            dataset_name=dataset.name,
            parameters=detector.parameters,
            metrics=eval_response.metrics,
        )

        # Verify experiment contains run
        experiments = await experiment_service.list_experiments()
        assert experiment_id in experiments
        assert len(experiments[experiment_id]["runs"]) == 1
        assert experiments[experiment_id]["runs"][0]["run_id"] == run_id

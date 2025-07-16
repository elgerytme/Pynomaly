"""ONNX adapter for anomaly detection models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.shared.protocols import DetectorProtocol

logger = logging.getLogger(__name__)


class ONNXAdapter(DetectorProtocol):
    """Adapter for ONNX-based anomaly detection models."""

    def __init__(self, session, model_path: Path):
        """Initialize ONNX adapter.

        Args:
            session: ONNX runtime session
            model_path: Path to ONNX model
        """
        self.session = session
        self.model_path = model_path
        self._id = uuid4()
        self._name = f"ONNX_Model_{self.model_path.stem}"
        self._algorithm_name = "ONNX"
        self._is_fitted = True  # ONNX models are pre-trained
        self._contamination_rate = ContaminationRate(0.1)
        self._parameters = {
            "model_path": str(model_path),
            "input_names": [input.name for input in session.get_inputs()],
            "output_names": [output.name for output in session.get_outputs()],
        }
        self._supports_streaming = True
        self._requires_fitting = False

    @property
    def id(self):
        """Model ID."""
        return self._id

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def algorithm_name(self) -> str:
        """Algorithm name."""
        return self._algorithm_name

    @property
    def is_fitted(self) -> bool:
        """Whether model is fitted."""
        return self._is_fitted

    @property
    def contamination_rate(self) -> ContaminationRate:
        """Contamination rate."""
        return self._contamination_rate

    @property
    def parameters(self) -> dict[str, Any]:
        """Model parameters."""
        return self._parameters

    @property
    def supports_streaming(self) -> bool:
        """Whether model supports streaming."""
        return self._supports_streaming

    @property
    def requires_fitting(self) -> bool:
        """Whether model requires fitting."""
        return self._requires_fitting

    def fit(self, dataset: Dataset) -> DetectorProtocol:
        """Fit the model (ONNX models are pre-trained).

        Args:
            dataset: Training dataset

        Returns:
            Self
        """
        logger.info(f"ONNX model {self.name} is pre-trained, skipping fit")
        return self

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies using ONNX model.

        Args:
            dataset: Dataset to analyze

        Returns:
            Detection results
        """
        # Get anomaly scores
        scores = self.score(dataset)

        # Calculate threshold (simple percentile-based approach)
        score_values = [s.value for s in scores]
        threshold = np.percentile(
            score_values, (1 - self._contamination_rate.value) * 100
        )

        # Create labels
        labels = [1 if score.value > threshold else 0 for score in scores]

        # Create anomaly objects
        from pynomaly.domain.entities import Anomaly

        anomalies = []
        for i, (score, label) in enumerate(zip(scores, labels, strict=False)):
            if label == 1:
                anomaly = Anomaly(
                    id=uuid4(),
                    data_point_index=i,
                    score=score,
                    features=dataset.data.iloc[i].to_dict(),
                    timestamp=None,
                    explanation=f"ONNX model detected anomaly with score {score.value:.3f}",
                )
                anomalies.append(anomaly)

        return DetectionResult(
            detector_id=self.id,
            dataset_name=dataset.name,
            scores=scores,
            labels=np.array(labels),
            anomalies=anomalies,
            threshold=threshold,
            execution_time_ms=0.0,  # Would need timing implementation
            metadata={"model_type": "ONNX", "model_path": str(self.model_path)},
        )

    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores using ONNX model.

        Args:
            dataset: Dataset to score

        Returns:
            List of anomaly scores
        """
        try:
            # Prepare input data
            input_data = self._prepare_input_data(dataset.data)

            # Run inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name

            result = self.session.run([output_name], {input_name: input_data})
            raw_scores = result[0].flatten()

            # Convert to AnomalyScore objects
            scores = []
            for score in raw_scores:
                scores.append(
                    AnomalyScore(
                        value=float(score),
                        confidence=0.8,  # Default confidence
                        metadata={"source": "ONNX_inference"},
                    )
                )

            return scores

        except Exception as e:
            logger.error(f"Error during ONNX inference: {e}")
            # Return default scores if inference fails
            return [
                AnomalyScore(value=0.0, confidence=0.0)
                for _ in range(len(dataset.data))
            ]

    def _prepare_input_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare input data for ONNX model.

        Args:
            data: Input dataframe

        Returns:
            Prepared numpy array
        """
        # Convert to numpy array
        input_array = data.select_dtypes(include=[np.number]).values

        # Ensure correct data type
        input_array = input_array.astype(np.float32)

        # Check if we need to reshape for model input
        expected_shape = self.session.get_inputs()[0].shape
        if len(expected_shape) > 2:
            # Reshape for models expecting different input dimensions
            input_array = input_array.reshape(input_array.shape[0], -1)

        return input_array

    def predict(self, dataset: Dataset) -> np.ndarray:
        """Predict anomaly labels.

        Args:
            dataset: Dataset to predict

        Returns:
            Predicted labels
        """
        result = self.detect(dataset)
        return result.labels

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """Predict anomaly probabilities.

        Args:
            dataset: Dataset to predict

        Returns:
            Predicted probabilities
        """
        scores = self.score(dataset)
        return np.array([s.value for s in scores])

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance (not supported for ONNX models).

        Returns:
            Empty dict
        """
        logger.warning("Feature importance not supported for ONNX models")
        return {}

    def clone(self) -> DetectorProtocol:
        """Clone the detector.

        Returns:
            Cloned detector
        """
        return ONNXAdapter(self.session, self.model_path)

    def __repr__(self) -> str:
        """String representation."""
        return f"ONNXAdapter(name={self.name}, model_path={self.model_path})"

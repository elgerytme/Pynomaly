"""PyOD algorithm adapter for Pynomaly."""

from typing import Any

import numpy as np
import numpy.typing as npt

from .. import ArrayLike, PredictionArray


class PyODAdapter:
    """Adapter for PyOD algorithms."""

    def __init__(self, algorithm: str = "IForest", **kwargs: Any) -> None:
        """Initialize PyOD adapter.

        Args:
            algorithm: PyOD algorithm name
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.kwargs = kwargs
        self._model: Any | None = None
        self._trained = False

    def fit(self, data: ArrayLike, **kwargs: Any) -> "PyODAdapter":
        """Train the PyOD model.

        Args:
            data: Training data
            **kwargs: Additional parameters

        Returns:
            self: Fitted adapter
        """
        try:
            from pyod.models.iforest import IForest
            from pyod.models.knn import KNN
            from pyod.models.lof import LOF
            from pyod.models.ocsvm import OCSVM

            # Create model based on algorithm
            if self.algorithm == "IForest":
                self._model = IForest(**self.kwargs)
            elif self.algorithm == "LOF":
                self._model = LOF(**self.kwargs)
            elif self.algorithm == "OCSVM":
                self._model = OCSVM(**self.kwargs)
            elif self.algorithm == "KNN":
                self._model = KNN(**self.kwargs)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            # Train the model
            self._model.fit(data)
            self._trained = True
            return self

        except ImportError as e:
            raise ImportError("PyOD is required for PyOD algorithms") from e

    def predict(self, data: ArrayLike, **kwargs: Any) -> PredictionArray:
        """Predict anomalies using the trained model.

        Args:
            data: Data to predict on
            **kwargs: Additional parameters

        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        if not self._trained or not self._model:
            raise ValueError("Model must be trained before prediction")

        # Get predictions (0 for normal, 1 for anomaly)
        predictions = self._model.predict(data)
        return predictions.astype(int)

    def detect(self, data: ArrayLike, **kwargs: Any) -> PredictionArray:
        """Detect anomalies in data (fit and predict in one call).

        Args:
            data: Input data
            **kwargs: Additional parameters

        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        self.fit(data, **kwargs)
        return self.predict(data, **kwargs)

    def decision_function(self, data: ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Get anomaly scores.

        Args:
            data: Input data

        Returns:
            Array of anomaly scores
        """
        if not self._trained or not self._model:
            raise ValueError("Model must be trained before prediction")

        return self._model.decision_function(data)

"""Scikit-learn algorithm adapter for Pynomaly."""

from typing import Any

import numpy as np
import numpy.typing as npt

from .. import ArrayLike, PredictionArray


class SklearnAdapter:
    """Adapter for scikit-learn algorithms."""

    def __init__(self, algorithm: str = "IsolationForest", **kwargs: Any) -> None:
        """Initialize sklearn adapter.

        Args:
            algorithm: Sklearn algorithm name
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.kwargs = kwargs
        self._model: Any | None = None
        self._trained = False

    def fit(self, data: ArrayLike, **kwargs: Any) -> "SklearnAdapter":
        """Train the sklearn model.

        Args:
            data: Training data
            **kwargs: Additional parameters

        Returns:
            self: Fitted adapter
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.svm import OneClassSVM

            # Create model based on algorithm
            if self.algorithm == "IsolationForest":
                self._model = IsolationForest(**self.kwargs)
            elif self.algorithm == "LocalOutlierFactor":
                self._model = LocalOutlierFactor(**self.kwargs)
            elif self.algorithm == "OneClassSVM":
                self._model = OneClassSVM(**self.kwargs)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            # Train the model
            if hasattr(self._model, 'fit'):
                self._model.fit(data)
                self._trained = True
            return self

        except ImportError as e:
            raise ImportError("scikit-learn is required for sklearn algorithms") from e

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

        # Get predictions (-1 for anomaly, 1 for normal in sklearn)
        predictions = self._model.predict(data)
        # Convert to 1 for anomaly, 0 for normal
        return (predictions == -1).astype(int)

    def detect(self, data: ArrayLike, **kwargs: Any) -> PredictionArray:
        """Detect anomalies in data (fit and predict in one call).

        Args:
            data: Input data
            **kwargs: Additional parameters

        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        # For LocalOutlierFactor, we can't separate fit and predict
        if self.algorithm == "LocalOutlierFactor":
            try:
                from sklearn.neighbors import LocalOutlierFactor
                model = LocalOutlierFactor(**self.kwargs)
                predictions = model.fit_predict(data)
                return (predictions == -1).astype(int)
            except ImportError as e:
                msg = "scikit-learn is required for sklearn algorithms"
                raise ImportError(msg) from e
        else:
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

        if hasattr(self._model, 'decision_function'):
            return self._model.decision_function(data)
        elif hasattr(self._model, 'score_samples'):
            return self._model.score_samples(data)
        else:
            msg = f"Algorithm {self.algorithm} doesn't support decision function"
            raise ValueError(msg)

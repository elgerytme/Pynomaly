"""Time series anomaly detection service with LSTM and Prophet-based algorithms."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
from datetime import datetime, timedelta

from ..entities.detection_result import DetectionResult
from ...infrastructure.logging import get_logger, log_decorator, timing_decorator
from ...infrastructure.logging.error_handler import ErrorHandler, InputValidationError, AlgorithmError
from ...infrastructure.monitoring import MetricsCollector, get_metrics_collector

logger = get_logger(__name__)
error_handler = ErrorHandler(logger._logger)
metrics_collector = get_metrics_collector()


class TimeSeriesDetectionService:
    """Service for time series-specific anomaly detection algorithms.
    
    Supports LSTM-based, Prophet-based, and statistical time series anomaly detection.
    """
    
    def __init__(self):
        """Initialize time series detection service."""
        self._fitted_models: Dict[str, Any] = {}
        self._preprocessing_params: Dict[str, Any] = {}
    
    @log_decorator(operation="time_series_detection", log_args=True, log_duration=True)
    def detect_anomalies(
        self,
        data: Union[npt.NDArray[np.floating], pd.DataFrame, pd.Series],
        algorithm: str = "lstm_autoencoder",
        timestamps: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
        contamination: float = 0.1,
        **kwargs: Any
    ) -> DetectionResult:
        """Detect anomalies in time series data.
        
        Args:
            data: Time series data. Can be univariate (Series/1D array) or multivariate (DataFrame/2D array)
            algorithm: Algorithm to use ('lstm_autoencoder', 'prophet', 'statistical', 'isolation_forest_ts')
            timestamps: Timestamps for the data (optional, will be generated if not provided)
            contamination: Expected proportion of anomalies
            **kwargs: Additional algorithm parameters
            
        Returns:
            DetectionResult with time series-specific metadata
        """
        try:
            # Preprocess and validate input
            processed_data, processed_timestamps = self._preprocess_input(data, timestamps)
            
            # Validate inputs
            self._validate_inputs(processed_data, algorithm, contamination)
            
            # Select and run algorithm
            if algorithm == "lstm_autoencoder":
                predictions, scores = self._lstm_autoencoder_detect(processed_data, contamination, **kwargs)
            elif algorithm == "prophet":
                predictions, scores = self._prophet_detect(processed_data, processed_timestamps, contamination, **kwargs)
            elif algorithm == "statistical":
                predictions, scores = self._statistical_detect(processed_data, contamination, **kwargs)
            elif algorithm == "isolation_forest_ts":
                predictions, scores = self._isolation_forest_ts_detect(processed_data, contamination, **kwargs)
            else:
                raise AlgorithmError(
                    f"Unknown time series algorithm: {algorithm}",
                    details={
                        "requested_algorithm": algorithm,
                        "available_algorithms": self.list_available_algorithms()
                    }
                )
            
            # Create result with time series metadata
            result = DetectionResult(
                predictions=predictions,
                confidence_scores=scores,
                algorithm=algorithm,
                metadata={
                    "contamination": contamination,
                    "data_shape": processed_data.shape,
                    "algorithm_params": kwargs,
                    "timestamps": processed_timestamps,
                    "is_time_series": True,
                    "time_range": {
                        "start": processed_timestamps[0] if len(processed_timestamps) > 0 else None,
                        "end": processed_timestamps[-1] if len(processed_timestamps) > 0 else None,
                        "frequency": self._infer_frequency(processed_timestamps) if len(processed_timestamps) > 1 else None
                    }
                }
            )
            
            # Log detection statistics
            logger.info("Time series anomaly detection completed successfully", 
                       algorithm=algorithm,
                       total_samples=result.total_samples,
                       anomalies_detected=result.anomaly_count,
                       anomaly_rate=result.anomaly_rate,
                       time_range_hours=self._calculate_time_range_hours(processed_timestamps))
            
            return result
            
        except Exception as e:
            return error_handler.handle_error(
                error=e,
                context={
                    "algorithm": algorithm,
                    "data_shape": data.shape if hasattr(data, 'shape') else len(data),
                    "contamination": contamination
                },
                operation="time_series_anomaly_detection",
                reraise=True
            )
    
    @timing_decorator(operation="lstm_autoencoder_detection")
    def _lstm_autoencoder_detect(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """LSTM Autoencoder-based anomaly detection for time series."""
        try:
            # Try to import required libraries
            try:
                import tensorflow as tf
                from tensorflow import keras
                from sklearn.preprocessing import MinMaxScaler
            except ImportError as e:
                # Fallback to statistical method if TensorFlow not available
                logger.warning("TensorFlow not available, falling back to statistical method")
                return self._statistical_detect(data, contamination, **kwargs)
            
            # Parameters
            sequence_length = kwargs.get('sequence_length', 10)
            epochs = kwargs.get('epochs', 50)
            batch_size = kwargs.get('batch_size', 32)
            encoding_dim = kwargs.get('encoding_dim', min(8, data.shape[1] if len(data.shape) > 1 else 4))
            
            logger.debug("Running LSTM Autoencoder detection", 
                        sequence_length=sequence_length,
                        epochs=epochs,
                        data_samples=data.shape[0])
            
            # Prepare data for LSTM (ensure 2D)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            # Normalize data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences for LSTM
            X_sequences = self._create_sequences(scaled_data, sequence_length)
            
            if len(X_sequences) < 10:
                # Not enough data for LSTM, fall back to statistical
                logger.warning("Not enough data for LSTM, falling back to statistical method")
                return self._statistical_detect(data, contamination, **kwargs)
            
            # Build LSTM Autoencoder
            model = self._build_lstm_autoencoder(
                sequence_length=sequence_length,
                n_features=data.shape[1],
                encoding_dim=encoding_dim
            )
            
            # Train the model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    X_sequences, X_sequences,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    validation_split=0.1
                )
            
            # Get reconstruction errors
            reconstructed = model.predict(X_sequences, verbose=0)
            mse = np.mean(np.square(X_sequences - reconstructed), axis=(1, 2))
            
            # Pad reconstruction errors to match original data length
            reconstruction_errors = np.zeros(len(data))
            reconstruction_errors[sequence_length-1:] = mse
            reconstruction_errors[:sequence_length-1] = mse[0]  # Use first error for initial points
            
            # Determine threshold and predictions
            threshold = np.percentile(reconstruction_errors, (1 - contamination) * 100)
            predictions = np.where(reconstruction_errors > threshold, -1, 1).astype(np.integer)
            
            # Convert reconstruction errors to confidence scores (0-1, higher = more anomalous)
            max_error = np.max(reconstruction_errors)
            min_error = np.min(reconstruction_errors)
            if max_error > min_error:
                confidence_scores = (reconstruction_errors - min_error) / (max_error - min_error)
            else:
                confidence_scores = np.zeros_like(reconstruction_errors)
            
            logger.debug("LSTM Autoencoder detection completed", 
                        anomalies_found=np.sum(predictions == -1),
                        threshold=threshold,
                        avg_reconstruction_error=np.mean(reconstruction_errors))
            
            return predictions, confidence_scores.astype(np.float64)
            
        except Exception as e:
            raise AlgorithmError(
                f"LSTM Autoencoder detection failed: {str(e)}",
                details={"contamination": contamination, "data_shape": data.shape},
                original_error=e
            )
    
    @timing_decorator(operation="prophet_detection")
    def _prophet_detect(
        self,
        data: npt.NDArray[np.floating],
        timestamps: List[datetime],
        contamination: float,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Prophet-based anomaly detection for time series."""
        try:
            # Try to import Prophet
            try:
                from prophet import Prophet
            except ImportError:
                logger.warning("Prophet not available, falling back to statistical method")
                return self._statistical_detect(data, contamination, **kwargs)
            
            # Parameters
            uncertainty_samples = kwargs.get('uncertainty_samples', 100)
            interval_width = kwargs.get('interval_width', 0.99)
            
            logger.debug("Running Prophet detection", 
                        data_samples=data.shape[0],
                        interval_width=interval_width)
            
            # Prophet requires univariate data
            if len(data.shape) > 1 and data.shape[1] > 1:
                # Use first column or PCA
                logger.info("Using first column for Prophet (univariate only)")
                values = data[:, 0]
            else:
                values = data.flatten()
            
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': timestamps,
                'y': values
            })
            
            # Initialize and fit Prophet model
            model = Prophet(
                uncertainty_samples=uncertainty_samples,
                interval_width=interval_width,
                daily_seasonality=kwargs.get('daily_seasonality', 'auto'),
                weekly_seasonality=kwargs.get('weekly_seasonality', 'auto'),
                yearly_seasonality=kwargs.get('yearly_seasonality', 'auto')
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(df)
            
            # Make predictions
            forecast = model.predict(df)
            
            # Calculate anomaly scores based on prediction intervals
            actual = df['y'].values
            predicted = forecast['yhat'].values
            upper_bound = forecast['yhat_upper'].values
            lower_bound = forecast['yhat_lower'].values
            
            # Points outside prediction interval are potential anomalies
            outside_interval = (actual > upper_bound) | (actual < lower_bound)
            
            # Calculate residuals and normalize
            residuals = np.abs(actual - predicted)
            interval_width_vals = upper_bound - lower_bound
            
            # Normalized residuals (how far outside the interval)
            normalized_residuals = np.where(
                interval_width_vals > 0,
                residuals / interval_width_vals,
                residuals
            )
            
            # Determine threshold and predictions
            threshold = np.percentile(normalized_residuals, (1 - contamination) * 100)
            predictions = np.where(normalized_residuals > threshold, -1, 1).astype(np.integer)
            
            # Confidence scores
            max_residual = np.max(normalized_residuals)
            if max_residual > 0:
                confidence_scores = normalized_residuals / max_residual
            else:
                confidence_scores = np.zeros_like(normalized_residuals)
            
            logger.debug("Prophet detection completed", 
                        anomalies_found=np.sum(predictions == -1),
                        outside_interval_count=np.sum(outside_interval),
                        avg_residual=np.mean(residuals))
            
            return predictions, confidence_scores.astype(np.float64)
            
        except Exception as e:
            raise AlgorithmError(
                f"Prophet detection failed: {str(e)}",
                details={"contamination": contamination, "data_shape": data.shape},
                original_error=e
            )
    
    @timing_decorator(operation="statistical_detection")
    def _statistical_detect(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Statistical anomaly detection for time series using moving averages and Z-scores."""
        try:
            # Parameters
            window_size = kwargs.get('window_size', min(50, len(data) // 4))
            z_threshold = kwargs.get('z_threshold', 3.0)
            
            logger.debug("Running statistical detection", 
                        window_size=window_size,
                        z_threshold=z_threshold,
                        data_samples=data.shape[0])
            
            # Handle multivariate data
            if len(data.shape) > 1 and data.shape[1] > 1:
                # Calculate anomaly scores for each feature and combine
                all_scores = []
                for i in range(data.shape[1]):
                    feature_scores = self._calculate_statistical_scores(data[:, i], window_size, z_threshold)
                    all_scores.append(feature_scores)
                
                # Combine scores (max or average)
                combined_scores = np.max(all_scores, axis=0)
            else:
                values = data.flatten()
                combined_scores = self._calculate_statistical_scores(values, window_size, z_threshold)
            
            # Determine threshold and predictions
            threshold = np.percentile(combined_scores, (1 - contamination) * 100)
            predictions = np.where(combined_scores > threshold, -1, 1).astype(np.integer)
            
            # Normalize confidence scores
            max_score = np.max(combined_scores)
            if max_score > 0:
                confidence_scores = combined_scores / max_score
            else:
                confidence_scores = np.zeros_like(combined_scores)
            
            logger.debug("Statistical detection completed", 
                        anomalies_found=np.sum(predictions == -1),
                        threshold=threshold,
                        avg_score=np.mean(combined_scores))
            
            return predictions, confidence_scores.astype(np.float64)
            
        except Exception as e:
            raise AlgorithmError(
                f"Statistical detection failed: {str(e)}",
                details={"contamination": contamination, "data_shape": data.shape},
                original_error=e
            )
    
    @timing_decorator(operation="isolation_forest_ts_detection")
    def _isolation_forest_ts_detect(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Isolation Forest with time series feature engineering."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Parameters
            window_size = kwargs.get('window_size', min(10, len(data) // 10))
            n_estimators = kwargs.get('n_estimators', 100)
            
            logger.debug("Running Isolation Forest with TS features", 
                        window_size=window_size,
                        n_estimators=n_estimators,
                        data_samples=data.shape[0])
            
            # Extract time series features
            ts_features = self._extract_time_series_features(data, window_size)
            
            # Extract random_state to avoid duplicate keyword argument
            model_kwargs = kwargs.copy()
            random_state = model_kwargs.pop('random_state', 42)
            
            # Apply Isolation Forest
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=random_state,
                **{k: v for k, v in model_kwargs.items() if k not in ['window_size']}
            )
            
            predictions = model.fit_predict(ts_features)
            
            # Get anomaly scores
            scores = model.decision_function(ts_features)
            
            # Convert to confidence scores (0-1, higher = more anomalous)
            threshold = np.percentile(scores, contamination * 100)
            normalized_scores = np.clip((threshold - scores) / (np.max(scores) - threshold + 1e-10), 0, 1)
            
            logger.debug("Isolation Forest TS detection completed", 
                        anomalies_found=np.sum(predictions == -1),
                        features_extracted=ts_features.shape[1])
            
            return predictions.astype(np.integer), normalized_scores.astype(np.float64)
            
        except ImportError as e:
            raise AlgorithmError(
                "scikit-learn required for Isolation Forest",
                details={"missing_dependency": "scikit-learn"},
                original_error=e
            )
        except Exception as e:
            raise AlgorithmError(
                f"Isolation Forest TS detection failed: {str(e)}",
                details={"contamination": contamination, "data_shape": data.shape},
                original_error=e
            )
    
    def _preprocess_input(
        self, 
        data: Union[npt.NDArray[np.floating], pd.DataFrame, pd.Series],
        timestamps: Optional[Union[pd.DatetimeIndex, List[datetime]]]
    ) -> Tuple[npt.NDArray[np.floating], List[datetime]]:
        """Preprocess input data and timestamps."""
        # Convert data to numpy array
        if isinstance(data, pd.DataFrame):
            processed_data = data.values.astype(np.float64)
        elif isinstance(data, pd.Series):
            processed_data = data.values.astype(np.float64).reshape(-1, 1)
        else:
            processed_data = np.array(data, dtype=np.float64)
            if len(processed_data.shape) == 1:
                processed_data = processed_data.reshape(-1, 1)
        
        # Generate or process timestamps
        if timestamps is None:
            # Generate default timestamps (hourly)
            start_time = datetime.now()
            processed_timestamps = [
                start_time + timedelta(hours=i) for i in range(len(processed_data))
            ]
        elif isinstance(timestamps, pd.DatetimeIndex):
            processed_timestamps = timestamps.to_pydatetime().tolist()
        else:
            processed_timestamps = list(timestamps)
        
        return processed_data, processed_timestamps
    
    def _validate_inputs(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float
    ) -> None:
        """Validate inputs for time series detection."""
        if data.size == 0:
            raise InputValidationError("Input data cannot be empty")
        
        if len(data.shape) > 2:
            raise InputValidationError(
                f"Data must be 1D or 2D, got shape {data.shape}"
            )
        
        if data.shape[0] < 5:
            raise InputValidationError(
                f"Need at least 5 time points, got {data.shape[0]}"
            )
        
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise InputValidationError("Algorithm name must be a non-empty string")
        
        if not (0.001 <= contamination <= 0.5):
            raise InputValidationError(
                f"Contamination must be between 0.001 and 0.5, got {contamination}"
            )
        
        # Check for non-finite values
        if not np.isfinite(data).all():
            raise InputValidationError("Input data contains non-finite values (NaN or inf)")
    
    def _create_sequences(self, data: npt.NDArray[np.floating], sequence_length: int) -> npt.NDArray[np.floating]:
        """Create sequences for LSTM input."""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def _build_lstm_autoencoder(self, sequence_length: int, n_features: int, encoding_dim: int):
        """Build LSTM autoencoder model."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Encoder
            inputs = keras.Input(shape=(sequence_length, n_features))
            encoded = layers.LSTM(encoding_dim, return_sequences=False)(inputs)
            
            # Decoder
            decoded = layers.RepeatVector(sequence_length)(encoded)
            decoded = layers.LSTM(n_features, return_sequences=True)(decoded)
            
            # Create model
            autoencoder = keras.Model(inputs, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            
            return autoencoder
            
        except ImportError as e:
            raise AlgorithmError(
                "TensorFlow required for LSTM autoencoder",
                details={"missing_dependency": "tensorflow"},
                original_error=e
            )
    
    def _calculate_statistical_scores(self, values: npt.NDArray[np.floating], window_size: int, z_threshold: float) -> npt.NDArray[np.floating]:
        """Calculate statistical anomaly scores for univariate time series."""
        scores = np.zeros(len(values))
        
        # Rolling statistics
        for i in range(len(values)):
            start_idx = max(0, i - window_size + 1)
            window_data = values[start_idx:i+1]
            
            if len(window_data) > 1:
                mean_val = np.mean(window_data)
                std_val = np.std(window_data)
                
                if std_val > 0:
                    z_score = abs((values[i] - mean_val) / std_val)
                    scores[i] = z_score
                else:
                    scores[i] = 0
            else:
                scores[i] = 0
        
        return scores
    
    def _extract_time_series_features(self, data: npt.NDArray[np.floating], window_size: int) -> npt.NDArray[np.floating]:
        """Extract time series features for Isolation Forest."""
        n_samples, n_features = data.shape
        features = []
        
        for i in range(n_samples):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            
            sample_features = []
            
            for j in range(n_features):
                window_data = data[start_idx:end_idx, j]
                
                # Basic statistics
                sample_features.extend([
                    np.mean(window_data),
                    np.std(window_data),
                    np.min(window_data),
                    np.max(window_data)
                ])
                
                # Current value
                sample_features.append(data[i, j])
                
                # Trend (if enough points)
                if len(window_data) > 1:
                    trend = window_data[-1] - window_data[0]
                    sample_features.append(trend)
                else:
                    sample_features.append(0)
            
            features.append(sample_features)
        
        return np.array(features)
    
    def _infer_frequency(self, timestamps: List[datetime]) -> Optional[str]:
        """Infer frequency from timestamps."""
        if len(timestamps) < 2:
            return None
        
        # Calculate time differences
        diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(min(10, len(timestamps)-1))]
        avg_diff = np.mean(diffs)
        
        # Common frequencies
        if abs(avg_diff - 60) < 10:  # ~1 minute
            return "1min"
        elif abs(avg_diff - 3600) < 300:  # ~1 hour
            return "1H"
        elif abs(avg_diff - 86400) < 3600:  # ~1 day
            return "1D"
        else:
            return f"{avg_diff:.0f}s"
    
    def _calculate_time_range_hours(self, timestamps: List[datetime]) -> float:
        """Calculate time range in hours."""
        if len(timestamps) < 2:
            return 0
        return (timestamps[-1] - timestamps[0]).total_seconds() / 3600
    
    def list_available_algorithms(self) -> List[str]:
        """List all available time series algorithms."""
        return ["lstm_autoencoder", "prophet", "statistical", "isolation_forest_ts"]
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """Get information about a time series algorithm."""
        info = {"name": algorithm, "type": "time_series"}
        
        if algorithm == "lstm_autoencoder":
            info.update({
                "description": "LSTM-based autoencoder for sequence anomaly detection",
                "requires": ["tensorflow", "scikit-learn"],
                "supports_multivariate": True,
                "supports_online": False
            })
        elif algorithm == "prophet":
            info.update({
                "description": "Prophet-based time series forecasting for anomaly detection",
                "requires": ["prophet"],
                "supports_multivariate": False,
                "supports_online": False
            })
        elif algorithm == "statistical":
            info.update({
                "description": "Statistical methods using moving averages and Z-scores",
                "requires": [],
                "supports_multivariate": True,
                "supports_online": True
            })
        elif algorithm == "isolation_forest_ts":
            info.update({
                "description": "Isolation Forest with time series feature engineering",
                "requires": ["scikit-learn"],
                "supports_multivariate": True,
                "supports_online": False
            })
        
        return info
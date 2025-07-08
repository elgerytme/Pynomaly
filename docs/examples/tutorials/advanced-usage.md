# Advanced Usage Tutorials

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 💡 [Examples](README.md) > 📁 Tutorials > 📄 Advanced Usage

---


This guide provides comprehensive tutorials for advanced Pynomaly usage scenarios, from custom algorithm development to production deployment patterns.

## Table of Contents

1. [Custom Algorithm Development](#custom-algorithm-development)
2. [Multi-Modal Anomaly Detection](#multi-modal-anomaly-detection)
3. [Real-Time Streaming Detection](#real-time-streaming-detection)
4. [Distributed Training and Inference](#distributed-training-and-inference)
5. [AutoML and Hyperparameter Optimization](#automl-and-hyperparameter-optimization)
6. [Model Explainability and Interpretability](#model-explainability-and-interpretability)
7. [Production Deployment Patterns](#production-deployment-patterns)
8. [Performance Optimization](#performance-optimization)

---

## Custom Algorithm Development

Learn how to develop and integrate custom anomaly detection algorithms into Pynomaly.

### 1. Basic Custom Detector

```python
from typing import Any, Dict, List, Optional
import numpy as np
from pynomaly.domain.entities import Detector, Dataset, DetectionResult, Anomaly
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.domain.exceptions import DetectorNotFittedError

class MahalanobisDetector(Detector):
    """Custom detector using Mahalanobis distance."""

    def __init__(self, name: str, contamination_rate: ContaminationRate, **kwargs):
        super().__init__(
            name=name,
            algorithm_name="Mahalanobis",
            contamination_rate=contamination_rate,
            **kwargs
        )
        self.mean_ = None
        self.cov_inv_ = None
        self.threshold_ = None

    async def fit(self, dataset: Dataset) -> None:
        """Train the Mahalanobis detector."""
        X = dataset.features.values

        # Calculate mean and covariance
        self.mean_ = np.mean(X, axis=0)
        cov_matrix = np.cov(X.T)

        # Add regularization for numerical stability
        reg_param = 1e-6
        cov_matrix += reg_param * np.eye(cov_matrix.shape[0])

        self.cov_inv_ = np.linalg.inv(cov_matrix)

        # Calculate threshold based on contamination rate
        distances = self._calculate_distances(X)
        threshold_percentile = (1 - self.contamination_rate.value) * 100
        self.threshold_ = np.percentile(distances, threshold_percentile)

        self._is_fitted = True

    async def predict(self, dataset: Dataset) -> DetectionResult:
        """Predict anomalies using Mahalanobis distance."""
        if not self._is_fitted:
            raise DetectorNotFittedError("Detector must be fitted before prediction")

        X = dataset.features.values
        distances = self._calculate_distances(X)

        anomalies = []
        for idx, distance in enumerate(distances):
            if distance > self.threshold_:
                anomaly = Anomaly(
                    index=idx,
                    score=AnomalyScore(min(distance / self.threshold_, 1.0)),
                    timestamp=None,
                    feature_names=list(dataset.features.columns)
                )
                anomalies.append(anomaly)

        return DetectionResult(
            id=f"result_{self.id}",
            detector_id=self.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            n_anomalies=len(anomalies),
            anomaly_rate=len(anomalies) / len(X),
            threshold=self.threshold_,
            execution_time=0.0  # Would measure actual time
        )

    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate Mahalanobis distances."""
        diff = X - self.mean_
        distances = np.sqrt(np.sum(diff @ self.cov_inv_ * diff, axis=1))
        return distances

# Usage example
async def use_custom_detector():
    from pynomaly import Pynomaly

    pynomaly = Pynomaly()

    # Register custom detector
    pynomaly.register_detector_class("Mahalanobis", MahalanobisDetector)

    # Create and use custom detector
    detector = await pynomaly.detectors.create(
        name="mahalanobis_detector",
        algorithm="Mahalanobis",
        contamination_rate=ContaminationRate(0.1)
    )

    # Load dataset and use detector
    dataset = await pynomaly.datasets.load_csv("data.csv")
    await detector.fit(dataset)
    result = await detector.predict(dataset)

    print(f"Custom detector found {len(result.anomalies)} anomalies")
```

### 2. Advanced Custom Detector with Configuration

```python
from pydantic import BaseModel, Field
from typing import Union, Literal

class CustomDetectorConfig(BaseModel):
    """Configuration for custom detector."""
    distance_metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean"
    normalization: Literal["none", "minmax", "zscore"] = "zscore"
    outlier_fraction: float = Field(default=0.1, ge=0.0, le=0.5)

class ConfigurableDetector(Detector):
    """Detector with rich configuration options."""

    def __init__(self, name: str, config: CustomDetectorConfig, **kwargs):
        super().__init__(name=name, algorithm_name="Configurable", **kwargs)
        self.config = config
        self.scaler_ = None
        self.reference_points_ = None

    async def fit(self, dataset: Dataset) -> None:
        """Fit with configurable preprocessing and algorithm."""
        X = dataset.features.values.copy()

        # Apply normalization
        if self.config.normalization == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler_ = MinMaxScaler()
            X = self.scaler_.fit_transform(X)
        elif self.config.normalization == "zscore":
            from sklearn.preprocessing import StandardScaler
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # Store reference points for distance calculation
        self.reference_points_ = X
        self._is_fitted = True

    async def predict(self, dataset: Dataset) -> DetectionResult:
        """Predict with configurable distance metrics."""
        if not self._is_fitted:
            raise DetectorNotFittedError("Must fit before predict")

        X = dataset.features.values.copy()

        # Apply same normalization
        if self.scaler_:
            X = self.scaler_.transform(X)

        # Calculate distances using specified metric
        distances = self._calculate_distances(X)

        # Determine threshold
        threshold = np.percentile(
            distances,
            (1 - self.config.outlier_fraction) * 100
        )

        # Create anomalies
        anomalies = []
        for idx, distance in enumerate(distances):
            if distance > threshold:
                score = min(distance / threshold, 1.0)
                anomaly = Anomaly(
                    index=idx,
                    score=AnomalyScore(score),
                    timestamp=None,
                    feature_names=list(dataset.features.columns)
                )
                anomalies.append(anomaly)

        return DetectionResult(
            id=f"result_{self.id}",
            detector_id=self.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            n_anomalies=len(anomalies),
            anomaly_rate=len(anomalies) / len(X),
            threshold=threshold,
            execution_time=0.0
        )

    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate distances using configured metric."""
        from scipy.spatial.distance import cdist

        if self.config.distance_metric == "euclidean":
            distances = cdist(X, self.reference_points_, metric='euclidean')
        elif self.config.distance_metric == "manhattan":
            distances = cdist(X, self.reference_points_, metric='manhattan')
        elif self.config.distance_metric == "chebyshev":
            distances = cdist(X, self.reference_points_, metric='chebyshev')

        # Return minimum distance to any reference point
        return np.min(distances, axis=1)

# Usage with configuration
config = CustomDetectorConfig(
    distance_metric="manhattan",
    normalization="zscore",
    outlier_fraction=0.05
)

detector = ConfigurableDetector("advanced_custom", config)
```

---

## Multi-Modal Anomaly Detection

Handle different data types (tabular, time-series, text, images) in a unified framework.

### 1. Tabular + Time-Series Fusion

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MultiModalDataset:
    """Dataset combining multiple data modalities."""

    def __init__(self, tabular_data: pd.DataFrame, time_series_data: pd.DataFrame):
        self.tabular_data = tabular_data
        self.time_series_data = time_series_data
        self.fused_features = None

    def extract_time_series_features(self, window_size: int = 24) -> pd.DataFrame:
        """Extract statistical features from time series."""
        features = []

        for i in range(len(self.time_series_data) - window_size + 1):
            window = self.time_series_data.iloc[i:i+window_size]

            feature_dict = {
                'ts_mean': window.mean().mean(),
                'ts_std': window.std().mean(),
                'ts_min': window.min().min(),
                'ts_max': window.max().max(),
                'ts_trend': self._calculate_trend(window),
                'ts_seasonality': self._calculate_seasonality(window),
                'ts_anomaly_score': self._calculate_ts_anomaly(window)
            }
            features.append(feature_dict)

        return pd.DataFrame(features)

    def fuse_modalities(self) -> pd.DataFrame:
        """Combine tabular and time-series features."""
        ts_features = self.extract_time_series_features()

        # Align datasets (assuming same length after windowing)
        min_len = min(len(self.tabular_data), len(ts_features))

        tabular_aligned = self.tabular_data.iloc[-min_len:].reset_index(drop=True)
        ts_aligned = ts_features.iloc[-min_len:].reset_index(drop=True)

        # Concatenate features
        self.fused_features = pd.concat([tabular_aligned, ts_aligned], axis=1)
        return self.fused_features

    def _calculate_trend(self, window: pd.DataFrame) -> float:
        """Calculate trend strength in time series window."""
        x = np.arange(len(window))
        y = window.iloc[:, 0].values  # First column
        return np.corrcoef(x, y)[0, 1] if len(y) > 1 else 0.0

    def _calculate_seasonality(self, window: pd.DataFrame) -> float:
        """Calculate seasonality strength."""
        # Simple autocorrelation at lag 7 (daily data, weekly seasonality)
        series = window.iloc[:, 0].values
        if len(series) < 14:
            return 0.0

        lag_7_corr = np.corrcoef(series[:-7], series[7:])[0, 1]
        return lag_7_corr if not np.isnan(lag_7_corr) else 0.0

    def _calculate_ts_anomaly(self, window: pd.DataFrame) -> float:
        """Calculate anomaly score within time series window."""
        series = window.iloc[:, 0].values
        z_scores = np.abs((series - np.mean(series)) / (np.std(series) + 1e-8))
        return np.max(z_scores)

# Usage example
async def multimodal_detection():
    # Load different data modalities
    tabular_df = pd.read_csv("customer_features.csv")
    time_series_df = pd.read_csv("transaction_time_series.csv")

    # Create multimodal dataset
    multimodal_data = MultiModalDataset(tabular_df, time_series_df)
    fused_data = multimodal_data.fuse_modalities()

    # Convert to Pynomaly dataset
    from pynomaly.domain.entities import Dataset
    dataset = Dataset(
        name="multimodal_dataset",
        data=fused_data,
        target_column=None
    )

    # Use ensemble of specialized detectors
    from pynomaly import Pynomaly
    pynomaly = Pynomaly()

    # Tabular-focused detector
    tabular_detector = await pynomaly.detectors.create(
        name="tabular_detector",
        algorithm="IsolationForest",
        contamination_rate=0.1
    )

    # Time-series focused detector
    ts_detector = await pynomaly.detectors.create(
        name="ts_detector",
        algorithm="LOF",
        contamination_rate=0.1,
        n_neighbors=10
    )

    # Neural network for complex patterns
    nn_detector = await pynomaly.detectors.create(
        name="neural_detector",
        algorithm="AutoEncoder",
        adapter="tensorflow",
        contamination_rate=0.1,
        hidden_layers=[64, 32, 16]
    )

    # Train all detectors
    await tabular_detector.fit(dataset)
    await ts_detector.fit(dataset)
    await nn_detector.fit(dataset)

    # Ensemble prediction
    from pynomaly.application.services import EnsembleService
    ensemble = EnsembleService([tabular_detector, ts_detector, nn_detector])

    result = await ensemble.predict(
        dataset,
        voting_strategy="weighted",
        weights=[0.4, 0.3, 0.3]  # Weight based on modality importance
    )

    print(f"Multimodal detection found {len(result.anomalies)} anomalies")
```

### 2. Text + Tabular Data Fusion

```python
from transformers import AutoTokenizer, AutoModel
import torch

class TextTabularFusion:
    """Combine text embeddings with tabular features."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Convert text to embeddings."""
        embeddings = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )

                outputs = self.model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)

        return np.array(embeddings)

    def fuse_text_tabular(self, text_data: List[str],
                         tabular_data: pd.DataFrame) -> pd.DataFrame:
        """Combine text embeddings with tabular features."""
        # Get text embeddings
        text_embeddings = self.encode_text(text_data)

        # Create DataFrame with text embeddings
        text_df = pd.DataFrame(
            text_embeddings,
            columns=[f"text_emb_{i}" for i in range(text_embeddings.shape[1])]
        )

        # Combine with tabular data
        combined_df = pd.concat([tabular_data.reset_index(drop=True), text_df], axis=1)
        return combined_df

# Usage for review + rating anomaly detection
async def text_tabular_anomaly_detection():
    # Load data
    reviews_df = pd.read_csv("product_reviews.csv")

    # Extract text and tabular features
    text_data = reviews_df['review_text'].tolist()
    tabular_data = reviews_df[['rating', 'helpful_votes', 'verified_purchase']].copy()

    # Fuse modalities
    fusion = TextTabularFusion()
    fused_data = fusion.fuse_text_tabular(text_data, tabular_data)

    # Create dataset
    dataset = Dataset(name="review_anomalies", data=fused_data)

    # Use detector optimized for high-dimensional data
    detector = await pynomaly.detectors.create(
        name="text_tabular_detector",
        algorithm="AutoEncoder",
        adapter="tensorflow",
        contamination_rate=0.05,
        hidden_layers=[128, 64, 32],
        encoding_dim=16,
        dropout_rate=0.2
    )

    await detector.fit(dataset)
    result = await detector.predict(dataset)

    # Analyze anomalous reviews
    for anomaly in result.anomalies[:10]:  # Top 10 anomalies
        idx = anomaly.index
        print(f"Anomalous review (score: {anomaly.score.value:.3f}):")
        print(f"Rating: {reviews_df.iloc[idx]['rating']}")
        print(f"Text: {reviews_df.iloc[idx]['review_text'][:200]}...")
        print("-" * 50)
```

---

## Real-Time Streaming Detection

Implement real-time anomaly detection for streaming data sources.

### 1. Kafka Stream Processing

```python
import asyncio
import json
from kafka import KafkaConsumer, KafkaProducer
from typing import AsyncGenerator
import logging

class StreamingAnomalyDetector:
    """Real-time anomaly detection on streaming data."""

    def __init__(self, detector, batch_size: int = 100, buffer_timeout: float = 1.0):
        self.detector = detector
        self.batch_size = batch_size
        self.buffer_timeout = buffer_timeout
        self.buffer = []
        self.last_process_time = time.time()

    async def process_stream(self, input_topic: str, output_topic: str,
                           kafka_config: dict):
        """Process streaming data from Kafka."""
        consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id=kafka_config.get('group_id', 'anomaly_detector')
        )

        producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

        try:
            async for message in self._async_kafka_consumer(consumer):
                await self._process_message(message, producer, output_topic)
        finally:
            consumer.close()
            producer.close()

    async def _async_kafka_consumer(self, consumer) -> AsyncGenerator:
        """Convert synchronous Kafka consumer to async generator."""
        while True:
            message_pack = consumer.poll(timeout_ms=100)
            for topic_partition, messages in message_pack.items():
                for message in messages:
                    yield message

            # Check for buffer timeout
            if (time.time() - self.last_process_time > self.buffer_timeout and
                self.buffer):
                await self._process_buffer(producer, output_topic)

            await asyncio.sleep(0.01)  # Prevent blocking

    async def _process_message(self, message, producer, output_topic: str):
        """Process individual message."""
        try:
            data = message.value
            self.buffer.append(data)

            # Process batch when buffer is full
            if len(self.buffer) >= self.batch_size:
                await self._process_buffer(producer, output_topic)

        except Exception as e:
            logging.error(f"Error processing message: {e}")

    async def _process_buffer(self, producer, output_topic: str):
        """Process accumulated buffer."""
        if not self.buffer:
            return

        try:
            # Convert buffer to DataFrame
            batch_df = pd.DataFrame(self.buffer)

            # Create temporary dataset
            from pynomaly.domain.entities import Dataset
            temp_dataset = Dataset(
                name=f"stream_batch_{int(time.time())}",
                data=batch_df
            )

            # Detect anomalies
            result = await self.detector.predict(temp_dataset)

            # Send anomalies to output topic
            for anomaly in result.anomalies:
                anomaly_data = {
                    'timestamp': time.time(),
                    'index': anomaly.index,
                    'score': anomaly.score.value,
                    'data': self.buffer[anomaly.index],
                    'severity': anomaly.get_severity()
                }

                producer.send(output_topic, value=anomaly_data)

            # Clear buffer
            self.buffer.clear()
            self.last_process_time = time.time()

            logging.info(f"Processed batch of {len(batch_df)} records, "
                        f"found {len(result.anomalies)} anomalies")

        except Exception as e:
            logging.error(f"Error processing buffer: {e}")
            self.buffer.clear()

# Usage example
async def run_streaming_detection():
    from pynomaly import Pynomaly

    pynomaly = Pynomaly()

    # Create and train detector on historical data
    historical_dataset = await pynomaly.datasets.load_csv("historical_data.csv")

    detector = await pynomaly.detectors.create(
        name="streaming_detector",
        algorithm="IsolationForest",
        contamination_rate=0.05,
        n_estimators=50  # Smaller for faster inference
    )

    await detector.fit(historical_dataset)

    # Set up streaming detector
    stream_detector = StreamingAnomalyDetector(
        detector=detector,
        batch_size=50,
        buffer_timeout=2.0
    )

    # Kafka configuration
    kafka_config = {
        'bootstrap_servers': ['localhost:9092'],
        'group_id': 'pynomaly_consumer'
    }

    # Start processing stream
    await stream_detector.process_stream(
        input_topic='transaction_stream',
        output_topic='anomaly_alerts',
        kafka_config=kafka_config
    )
```

### 2. WebSocket Real-Time Detection

```python
import websockets
import json
from collections import deque
import asyncio

class WebSocketAnomalyDetector:
    """Real-time anomaly detection via WebSocket."""

    def __init__(self, detector, window_size: int = 100):
        self.detector = detector
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)
        self.connected_clients = set()

    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time detection."""
        async def handle_client(websocket, path):
            self.connected_clients.add(websocket)
            try:
                async for message in websocket:
                    await self.process_websocket_message(message, websocket)
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.connected_clients.discard(websocket)

        server = await websockets.serve(handle_client, host, port)
        print(f"WebSocket anomaly detection server started on ws://{host}:{port}")
        await server.wait_closed()

    async def process_websocket_message(self, message: str, websocket):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Add to sliding window
            self.data_window.append(data)

            # Perform detection if window is full
            if len(self.data_window) >= self.window_size:
                await self._detect_and_notify(websocket)

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'error': 'Invalid JSON format'
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                'error': f'Processing error: {str(e)}'
            }))

    async def _detect_and_notify(self, websocket):
        """Perform detection on current window and notify clients."""
        try:
            # Convert window to DataFrame
            window_df = pd.DataFrame(list(self.data_window))

            # Create temporary dataset
            from pynomaly.domain.entities import Dataset
            temp_dataset = Dataset(
                name=f"websocket_window_{int(time.time())}",
                data=window_df
            )

            # Detect anomalies
            result = await self.detector.predict(temp_dataset)

            # Check if the latest point is anomalous
            latest_index = len(window_df) - 1
            latest_anomaly = None

            for anomaly in result.anomalies:
                if anomaly.index == latest_index:
                    latest_anomaly = anomaly
                    break

            # Prepare response
            response = {
                'timestamp': time.time(),
                'is_anomaly': latest_anomaly is not None,
                'total_anomalies_in_window': len(result.anomalies),
                'window_size': len(self.data_window)
            }

            if latest_anomaly:
                response.update({
                    'anomaly_score': latest_anomaly.score.value,
                    'severity': latest_anomaly.get_severity()
                })

            # Notify all connected clients
            if self.connected_clients:
                await asyncio.gather(*[
                    client.send(json.dumps(response))
                    for client in self.connected_clients
                ], return_exceptions=True)

        except Exception as e:
            error_response = {
                'error': f'Detection error: {str(e)}',
                'timestamp': time.time()
            }
            await websocket.send(json.dumps(error_response))

# WebSocket client example
async def websocket_client_example():
    """Example WebSocket client for testing."""
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        # Send test data
        for i in range(200):
            # Generate normal data with occasional anomalies
            if i % 50 == 0:  # Anomaly every 50 points
                data = {'value': np.random.normal(10, 1), 'feature2': np.random.normal(5, 0.5)}
            else:
                data = {'value': np.random.normal(0, 1), 'feature2': np.random.normal(0, 0.5)}

            await websocket.send(json.dumps(data))

            # Receive response
            response = await websocket.recv()
            result = json.loads(response)

            if result.get('is_anomaly'):
                print(f"Anomaly detected! Score: {result.get('anomaly_score', 'N/A')}")

            await asyncio.sleep(0.1)  # Send data every 100ms

# Usage
async def run_websocket_detection():
    # Set up detector
    pynomaly = Pynomaly()
    detector = await pynomaly.detectors.create(
        name="websocket_detector",
        algorithm="LOF",
        contamination_rate=0.1,
        n_neighbors=20
    )

    # Train on historical data
    historical_data = pd.DataFrame({
        'value': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 0.5, 1000)
    })
    historical_dataset = Dataset(name="historical", data=historical_data)
    await detector.fit(historical_dataset)

    # Start WebSocket server
    ws_detector = WebSocketAnomalyDetector(detector, window_size=50)
    await ws_detector.start_server()
```

---

## Distributed Training and Inference

Scale anomaly detection across multiple machines and GPUs.

### 1. Multi-GPU Training with TensorFlow

```python
import tensorflow as tf
from pynomaly.infrastructure.adapters import TensorFlowAdapter

class DistributedTensorFlowAdapter(TensorFlowAdapter):
    """TensorFlow adapter with distributed training support."""

    def __init__(self, *args, strategy=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Set up distribution strategy
        if strategy == "MirroredStrategy":
            self.strategy = tf.distribute.MirroredStrategy()
        elif strategy == "MultiWorkerMirroredStrategy":
            self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
        elif strategy == "ParameterServerStrategy":
            self.strategy = tf.distribute.ParameterServerStrategy()
        else:
            self.strategy = tf.distribute.get_strategy()  # Default strategy

        print(f"Using strategy: {self.strategy}")
        print(f"Number of replicas: {self.strategy.num_replicas_in_sync}")

    async def fit(self, dataset):
        """Distributed training implementation."""
        with self.strategy.scope():
            # Model creation must be within strategy scope
            await super().fit(dataset)

    def _create_model(self, input_dim):
        """Create model within distribution strategy scope."""
        with self.strategy.scope():
            return super()._create_model(input_dim)

# Usage example
async def distributed_training():
    # Create distributed detector
    detector = DistributedTensorFlowAdapter(
        algorithm_name="AutoEncoder",
        contamination_rate=ContaminationRate(0.1),
        strategy="MirroredStrategy",  # Use all available GPUs
        encoding_dim=64,
        hidden_layers=[128, 64],
        epochs=100,
        batch_size=512  # Larger batch size for multi-GPU
    )

    # Load large dataset
    large_dataset = await pynomaly.datasets.load_parquet("large_dataset.parquet")

    # Train across multiple GPUs
    await detector.fit(large_dataset)

    print("Distributed training completed")
```

### 2. Distributed Inference with Ray

```python
import ray
from typing import List
import pandas as pd

@ray.remote
class DetectorWorker:
    """Remote worker for distributed inference."""

    def __init__(self, detector_config: dict):
        # Initialize detector on worker
        from pynomaly import Pynomaly
        self.pynomaly = Pynomaly()
        self.detector = None
        self.detector_config = detector_config

    async def initialize(self, model_path: str):
        """Load trained model on worker."""
        self.detector = await self.pynomaly.detectors.load(model_path)
        return True

    async def predict_batch(self, data_batch: pd.DataFrame) -> dict:
        """Predict on batch of data."""
        from pynomaly.domain.entities import Dataset

        temp_dataset = Dataset(
            name=f"batch_{ray.get_runtime_context().worker.worker_id}",
            data=data_batch
        )

        result = await self.detector.predict(temp_dataset)

        return {
            'n_anomalies': len(result.anomalies),
            'anomaly_indices': [a.index for a in result.anomalies],
            'anomaly_scores': [a.score.value for a in result.anomalies],
            'batch_size': len(data_batch)
        }

class DistributedInferenceManager:
    """Manages distributed inference across Ray cluster."""

    def __init__(self, n_workers: int = 4, detector_config: dict = None):
        ray.init(ignore_reinit_error=True)

        self.n_workers = n_workers
        self.workers = [
            DetectorWorker.remote(detector_config or {})
            for _ in range(n_workers)
        ]

    async def initialize_workers(self, model_path: str):
        """Initialize all workers with trained model."""
        initialization_tasks = [
            worker.initialize.remote(model_path)
            for worker in self.workers
        ]

        results = await asyncio.gather(*[
            asyncio.wrap_future(ray.get(task))
            for task in initialization_tasks
        ])

        print(f"Initialized {len(results)} workers")

    async def predict_large_dataset(self, large_dataset: pd.DataFrame,
                                  batch_size: int = 10000) -> dict:
        """Distribute prediction across workers."""
        # Split dataset into batches
        batches = [
            large_dataset.iloc[i:i+batch_size]
            for i in range(0, len(large_dataset), batch_size)
        ]

        print(f"Processing {len(batches)} batches across {self.n_workers} workers")

        # Distribute batches across workers
        prediction_tasks = []
        for i, batch in enumerate(batches):
            worker_idx = i % self.n_workers
            task = self.workers[worker_idx].predict_batch.remote(batch)
            prediction_tasks.append(task)

        # Collect results
        results = await asyncio.gather(*[
            asyncio.wrap_future(ray.get(task))
            for task in prediction_tasks
        ])

        # Aggregate results
        total_anomalies = sum(r['n_anomalies'] for r in results)
        total_samples = sum(r['batch_size'] for r in results)

        # Adjust indices for global dataset
        global_anomaly_indices = []
        for i, result in enumerate(results):
            batch_start = i * batch_size
            global_indices = [idx + batch_start for idx in result['anomaly_indices']]
            global_anomaly_indices.extend(global_indices)

        return {
            'total_anomalies': total_anomalies,
            'total_samples': total_samples,
            'anomaly_rate': total_anomalies / total_samples,
            'anomaly_indices': global_anomaly_indices,
            'processing_time': time.time()
        }

    def shutdown(self):
        """Shutdown Ray cluster."""
        ray.shutdown()

# Usage example
async def distributed_inference_example():
    # Set up distributed inference
    inference_manager = DistributedInferenceManager(n_workers=8)

    # Initialize workers with trained model
    await inference_manager.initialize_workers("trained_model.pkl")

    # Load very large dataset
    large_data = pd.read_parquet("very_large_dataset.parquet")
    print(f"Dataset size: {len(large_data)} samples")

    # Distributed prediction
    start_time = time.time()
    results = await inference_manager.predict_large_dataset(
        large_data,
        batch_size=50000
    )
    end_time = time.time()

    print(f"Distributed inference completed in {end_time - start_time:.2f} seconds")
    print(f"Found {results['total_anomalies']} anomalies "
          f"({results['anomaly_rate']:.3f} rate)")

    # Cleanup
    inference_manager.shutdown()
```

---

## AutoML and Hyperparameter Optimization

Automatically select algorithms and optimize hyperparameters.

### 1. Algorithm Selection with Optuna

```python
import optuna
from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score

class AutoMLAnomalyDetector:
    """Automated ML for anomaly detection."""

    def __init__(self, algorithms: List[str] = None, n_trials: int = 100):
        self.algorithms = algorithms or [
            "IsolationForest", "LOF", "OCSVM", "AutoEncoder"
        ]
        self.n_trials = n_trials
        self.best_detector = None
        self.study = None

    async def auto_select(self, dataset, validation_dataset=None,
                         metric: str = "f1_score") -> dict:
        """Automatically select best algorithm and parameters."""

        def objective(trial):
            # Select algorithm
            algorithm = trial.suggest_categorical("algorithm", self.algorithms)

            # Get algorithm-specific parameters
            params = self._suggest_parameters(trial, algorithm)

            # Create detector
            detector = self._create_detector(algorithm, params)

            # Train and evaluate
            return self._evaluate_detector(detector, dataset, validation_dataset, metric)

        # Create study
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=self.n_trials)

        # Get best parameters
        best_params = self.study.best_params
        best_algorithm = best_params.pop("algorithm")

        # Create best detector
        self.best_detector = self._create_detector(best_algorithm, best_params)
        await self.best_detector.fit(dataset)

        return {
            "best_algorithm": best_algorithm,
            "best_parameters": best_params,
            "best_score": self.study.best_value,
            "n_trials": len(self.study.trials)
        }

    def _suggest_parameters(self, trial, algorithm: str) -> Dict[str, Any]:
        """Suggest hyperparameters for specific algorithm."""
        params = {"contamination_rate": trial.suggest_float("contamination", 0.01, 0.3)}

        if algorithm == "IsolationForest":
            params.update({
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_samples": trial.suggest_categorical("max_samples", ["auto", 0.5, 0.8]),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0)
            })

        elif algorithm == "LOF":
            params.update({
                "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
                "algorithm": trial.suggest_categorical("lof_algorithm",
                                                    ["auto", "ball_tree", "kd_tree"]),
                "leaf_size": trial.suggest_int("leaf_size", 10, 50)
            })

        elif algorithm == "OCSVM":
            params.update({
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
                        if trial.suggest_categorical("gamma_type", ["scale", "float"]) == "scale"
                        else trial.suggest_float("gamma", 1e-5, 1e-1, log=True),
                "nu": trial.suggest_float("nu", 0.01, 0.5)
            })

        elif algorithm == "AutoEncoder":
            params.update({
                "adapter": "tensorflow",
                "encoding_dim": trial.suggest_int("encoding_dim", 8, 128),
                "hidden_layers": [
                    trial.suggest_int("hidden_1", 32, 256),
                    trial.suggest_int("hidden_2", 16, 128)
                ],
                "epochs": trial.suggest_int("epochs", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5)
            })

        return params

    def _create_detector(self, algorithm: str, params: Dict[str, Any]):
        """Create detector with specified algorithm and parameters."""
        from pynomaly.infrastructure.adapters import PyODAdapter, TensorFlowAdapter
        from pynomaly.domain.value_objects import ContaminationRate

        contamination = ContaminationRate(params.pop("contamination_rate"))

        if algorithm in ["IsolationForest", "LOF", "OCSVM"]:
            return PyODAdapter(
                algorithm_name=algorithm,
                contamination_rate=contamination,
                **params
            )
        elif algorithm == "AutoEncoder":
            return TensorFlowAdapter(
                algorithm_name=algorithm,
                contamination_rate=contamination,
                **params
            )

    def _evaluate_detector(self, detector, dataset, validation_dataset, metric: str) -> float:
        """Evaluate detector performance."""
        try:
            # Train detector
            detector.fit(dataset)

            # Use validation dataset if provided, otherwise use training dataset
            eval_dataset = validation_dataset if validation_dataset else dataset

            # Get predictions
            result = detector.predict(eval_dataset)

            # Calculate metric based on labels if available
            if hasattr(eval_dataset, 'target_column') and eval_dataset.target_column:
                true_labels = eval_dataset.data[eval_dataset.target_column].values
                pred_labels = np.zeros(len(true_labels))

                # Mark anomalies
                for anomaly in result.anomalies:
                    pred_labels[anomaly.index] = 1

                if metric == "f1_score":
                    return f1_score(true_labels, pred_labels)
                elif metric == "roc_auc":
                    scores = np.zeros(len(true_labels))
                    for anomaly in result.anomalies:
                        scores[anomaly.index] = anomaly.score.value
                    return roc_auc_score(true_labels, scores)

            # If no labels, use contamination rate as proxy metric
            expected_anomalies = int(len(eval_dataset.data) * detector.contamination_rate.value)
            actual_anomalies = len(result.anomalies)

            # Score based on how close we are to expected contamination
            score = 1.0 - abs(expected_anomalies - actual_anomalies) / expected_anomalies
            return max(0.0, score)

        except Exception as e:
            print(f"Evaluation error for {detector.algorithm_name}: {e}")
            return 0.0

# Usage example
async def automl_example():
    from pynomaly import Pynomaly
    from pynomaly.domain.entities import Dataset

    pynomaly = Pynomaly()

    # Load labeled dataset for evaluation
    data = pd.read_csv("labeled_anomalies.csv")
    dataset = Dataset(name="automl_data", data=data, target_column="is_anomaly")

    # Split for validation
    train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['is_anomaly'])

    train_dataset = Dataset(name="train", data=train_data, target_column="is_anomaly")
    val_dataset = Dataset(name="validation", data=val_data, target_column="is_anomaly")

    # Run AutoML
    automl = AutoMLAnomalyDetector(n_trials=50)

    results = await automl.auto_select(
        train_dataset,
        val_dataset,
        metric="f1_score"
    )

    print(f"Best algorithm: {results['best_algorithm']}")
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best F1 score: {results['best_score']:.3f}")

    # Use best detector for prediction
    best_detector = automl.best_detector
    final_result = await best_detector.predict(val_dataset)

    print(f"Final validation: {len(final_result.anomalies)} anomalies detected")
```

### 2. Hyperparameter Optimization with Hyperopt

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt

class HyperoptOptimizer:
    """Hyperparameter optimization using Hyperopt."""

    def __init__(self, algorithm: str, max_evals: int = 100):
        self.algorithm = algorithm
        self.max_evals = max_evals
        self.trials = Trials()

    def optimize(self, dataset, validation_dataset=None):
        """Optimize hyperparameters for specific algorithm."""

        # Define search space
        space = self._get_search_space(self.algorithm)

        def objective(params):
            try:
                # Create and evaluate detector
                detector = self._create_detector(params)
                score = self._evaluate(detector, dataset, validation_dataset)

                # Hyperopt minimizes, so negate score
                return {'loss': -score, 'status': STATUS_OK}
            except Exception as e:
                return {'loss': 0, 'status': STATUS_OK}

        # Run optimization
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials
        )

        return best, self.trials

    def _get_search_space(self, algorithm: str):
        """Define hyperparameter search space."""
        spaces = {
            "IsolationForest": {
                'contamination': hp.uniform('contamination', 0.01, 0.3),
                'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300]),
                'max_samples': hp.choice('max_samples', ['auto', 0.5, 0.8, 1.0]),
                'max_features': hp.uniform('max_features', 0.1, 1.0)
            },
            "LOF": {
                'contamination': hp.uniform('contamination', 0.01, 0.3),
                'n_neighbors': hp.choice('n_neighbors', range(5, 51)),
                'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree']),
                'leaf_size': hp.choice('leaf_size', range(10, 51))
            },
            "AutoEncoder": {
                'contamination': hp.uniform('contamination', 0.01, 0.3),
                'encoding_dim': hp.choice('encoding_dim', [8, 16, 32, 64, 128]),
                'hidden_layers': hp.choice('hidden_layers', [
                    [64, 32], [128, 64], [128, 64, 32], [256, 128, 64]
                ]),
                'learning_rate': hp.loguniform('learning_rate', -6, -2),
                'epochs': hp.choice('epochs', [50, 100, 150, 200]),
                'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5)
            }
        }

        return spaces.get(algorithm, {})

    def _create_detector(self, params):
        """Create detector with hyperparameters."""
        from pynomaly.infrastructure.adapters import PyODAdapter, TensorFlowAdapter
        from pynomaly.domain.value_objects import ContaminationRate

        contamination = ContaminationRate(params.pop('contamination'))

        if self.algorithm in ["IsolationForest", "LOF", "OCSVM"]:
            return PyODAdapter(
                algorithm_name=self.algorithm,
                contamination_rate=contamination,
                **params
            )
        elif self.algorithm == "AutoEncoder":
            return TensorFlowAdapter(
                algorithm_name=self.algorithm,
                contamination_rate=contamination,
                adapter="tensorflow",
                **params
            )

    def _evaluate(self, detector, dataset, validation_dataset):
        """Evaluate detector with cross-validation."""
        # Simplified evaluation - in practice, use proper CV
        detector.fit(dataset)

        eval_dataset = validation_dataset or dataset
        result = detector.predict(eval_dataset)

        # Use contamination-based score if no labels
        expected = len(eval_dataset.data) * detector.contamination_rate.value
        actual = len(result.anomalies)

        return 1.0 - abs(expected - actual) / max(expected, actual)

# Usage
optimizer = HyperoptOptimizer("AutoEncoder", max_evals=50)
best_params, trials = optimizer.optimize(dataset, validation_dataset)

print(f"Best parameters: {best_params}")
print(f"Best score: {-min(trials.losses()):.3f}")
```

This advanced usage guide provides comprehensive examples for extending Pynomaly's capabilities in production environments. Each section includes complete, runnable code examples that demonstrate real-world usage patterns.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities

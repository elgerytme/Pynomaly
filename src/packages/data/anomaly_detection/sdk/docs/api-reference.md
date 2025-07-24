# API Reference

Complete reference documentation for all Anomaly Detection SDK methods, types, and configuration options.

## Table of Contents

- [Client Classes](#client-classes)
- [Core Methods](#core-methods)
- [Data Types](#data-types)
- [Error Types](#error-types)
- [Configuration](#configuration)
- [Utilities](#utilities)

## Client Classes

### AnomalyDetectionClient

The main HTTP client for synchronous anomaly detection operations.

#### Constructor

```python
# Python
AnomalyDetectionClient(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    headers: Optional[Dict[str, str]] = None
)
```

```javascript
// JavaScript/TypeScript
new AnomalyDetectionClient({
    baseUrl: string,
    apiKey?: string,
    timeout?: number,        // milliseconds
    maxRetries?: number,
    headers?: Record<string, string>
})
```

```go
// Go
NewClient(config ClientConfig) *Client

type ClientConfig struct {
    BaseURL    string
    APIKey     *string
    Timeout    time.Duration
    MaxRetries int
    Headers    map[string]string
}
```

### AsyncAnomalyDetectionClient

Asynchronous client for non-blocking operations (Python and JavaScript only).

#### Constructor

```python
# Python
AsyncAnomalyDetectionClient(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    headers: Optional[Dict[str, str]] = None
)
```

```javascript
// JavaScript/TypeScript
// Same as AnomalyDetectionClient - all methods are async
```

### StreamingClient

WebSocket client for real-time anomaly detection.

#### Constructor

```python
# Python
StreamingClient(
    ws_url: str,
    config: Optional[StreamingConfig] = None,
    api_key: Optional[str] = None,
    auto_reconnect: bool = True,
    reconnect_delay: float = 5.0
)
```

```javascript
// JavaScript/TypeScript
new StreamingClient({
    wsUrl: string,
    bufferSize?: number,
    detectionThreshold?: number,
    batchSize?: number,
    algorithm?: AlgorithmType,
    autoRetrain?: boolean,
    apiKey?: string,
    autoReconnect?: boolean,
    reconnectDelay?: number
})
```

```go
// Go
NewStreamingClient(config StreamingClientConfig) *StreamingClient

type StreamingClientConfig struct {
    WSURL           string
    APIKey          *string
    Config          StreamingConfig
    AutoReconnect   bool
    ReconnectDelay  time.Duration
}
```

## Core Methods

### detect_anomalies / detectAnomalies / DetectAnomalies

Detect anomalies in the provided data.

#### Signature

```python
# Python
def detect_anomalies(
    data: List[List[float]],
    algorithm: AlgorithmType = AlgorithmType.ISOLATION_FOREST,
    parameters: Optional[Dict[str, Any]] = None,
    return_explanations: bool = False
) -> DetectionResult
```

```javascript
// JavaScript/TypeScript
async detectAnomalies(
    data: number[][],
    algorithm: AlgorithmType = AlgorithmType.ISOLATION_FOREST,
    parameters?: Record<string, any>,
    returnExplanations: boolean = false
): Promise<DetectionResult>
```

```go
// Go
func (c *Client) DetectAnomalies(
    ctx context.Context,
    data [][]float64,
    algorithm AlgorithmType,
    parameters map[string]interface{},
    returnExplanations bool
) (*DetectionResult, error)
```

#### Parameters

- **`data`**: 2D array of data points (required)
- **`algorithm`**: Algorithm to use for detection (default: IsolationForest)
- **`parameters`**: Algorithm-specific parameters (optional)
- **`return_explanations`**: Whether to include explanations (default: false)

#### Returns

[`DetectionResult`](#detectionresult) object containing:
- List of anomalies with indices and scores
- Total points analyzed
- Anomaly count
- Algorithm used
- Execution time
- Metadata

#### Example

```python
# Python
result = client.detect_anomalies(
    data=[[1, 2], [1.1, 2.1], [10, 20]],
    algorithm=AlgorithmType.ISOLATION_FOREST,
    parameters={'contamination': 0.3}
)
```

### batch_detect / batchDetect / BatchDetect

Process a batch detection request with additional options.

#### Signature

```python
# Python
def batch_detect(request: BatchProcessingRequest) -> DetectionResult
```

```javascript
// JavaScript/TypeScript
async batchDetect(request: BatchProcessingRequest): Promise<DetectionResult>
```

```go
// Go
func (c *Client) BatchDetect(ctx context.Context, request BatchProcessingRequest) (*DetectionResult, error)
```

#### Parameters

- **`request`**: [`BatchProcessingRequest`](#batchprocessingrequest) object

#### Example

```python
# Python
from anomaly_detection_sdk.models import BatchProcessingRequest

request = BatchProcessingRequest(
    data=data,
    algorithm=AlgorithmType.ISOLATION_FOREST,
    parameters={'contamination': 0.1},
    return_explanations=True
)
result = client.batch_detect(request)
```

### train_model / trainModel / TrainModel

Train a new anomaly detection model.

#### Signature

```python
# Python
def train_model(request: TrainingRequest) -> TrainingResult
```

```javascript
// JavaScript/TypeScript
async trainModel(request: TrainingRequest): Promise<TrainingResult>
```

```go
// Go
func (c *Client) TrainModel(ctx context.Context, request TrainingRequest) (*TrainingResult, error)
```

#### Parameters

- **`request`**: [`TrainingRequest`](#trainingrequest) object

#### Returns

[`TrainingResult`](#trainingresult) object with model information and metrics.

### get_model / getModel / GetModel

Retrieve information about a specific model.

#### Signature

```python
# Python
def get_model(model_id: str) -> ModelInfo
```

```javascript
// JavaScript/TypeScript
async getModel(modelId: string): Promise<ModelInfo>
```

```go
// Go
func (c *Client) GetModel(ctx context.Context, modelID string) (*ModelInfo, error)
```

### list_models / listModels / ListModels

List all available models.

#### Signature

```python
# Python
def list_models() -> List[ModelInfo]
```

```javascript
// JavaScript/TypeScript
async listModels(): Promise<ModelInfo[]>
```

```go
// Go
func (c *Client) ListModels(ctx context.Context) ([]ModelInfo, error)
```

### delete_model / deleteModel / DeleteModel

Delete a model.

#### Signature

```python
# Python
def delete_model(model_id: str) -> Dict[str, str]
```

```javascript
// JavaScript/TypeScript
async deleteModel(modelId: string): Promise<{message: string}>
```

```go
// Go
func (c *Client) DeleteModel(ctx context.Context, modelID string) error
```

### explain_anomaly / explainAnomaly / ExplainAnomaly

Get explanation for why a data point is anomalous.

#### Signature

```python
# Python
def explain_anomaly(
    data_point: List[float],
    model_id: Optional[str] = None,
    algorithm: AlgorithmType = AlgorithmType.ISOLATION_FOREST,
    method: str = "shap"
) -> ExplanationResult
```

```javascript
// JavaScript/TypeScript
async explainAnomaly(
    dataPoint: number[],
    options: {
        modelId?: string,
        algorithm?: AlgorithmType,
        method?: string
    } = {}
): Promise<ExplanationResult>
```

```go
// Go
func (c *Client) ExplainAnomaly(
    ctx context.Context,
    dataPoint []float64,
    options ExplainOptions
) (*ExplanationResult, error)
```

### get_health / getHealth / GetHealth

Get service health status.

#### Signature

```python
# Python
def get_health() -> HealthStatus
```

```javascript
// JavaScript/TypeScript
async getHealth(): Promise<HealthStatus>
```

```go
// Go
func (c *Client) GetHealth(ctx context.Context) (*HealthStatus, error)
```

### get_metrics / getMetrics / GetMetrics

Get service metrics.

#### Signature

```python
# Python
def get_metrics() -> Dict[str, Any]
```

```javascript
// JavaScript/TypeScript
async getMetrics(): Promise<Record<string, any>>
```

```go
// Go
func (c *Client) GetMetrics(ctx context.Context) (map[string]interface{}, error)
```

### upload_data / uploadData / UploadData

Upload training data to the service.

#### Signature

```python
# Python
def upload_data(
    data: List[List[float]],
    dataset_name: str,
    description: Optional[str] = None
) -> Dict[str, str]
```

```javascript
// JavaScript/TypeScript
async uploadData(
    data: number[][],
    datasetName: string,
    description?: string
): Promise<{datasetId: string, message: string}>
```

```go
// Go
func (c *Client) UploadData(
    ctx context.Context,
    data [][]float64,
    datasetName string,
    description *string
) (*UploadResult, error)
```

## Streaming Methods

### start / start / Start

Start the streaming client.

#### Signature

```python
# Python
def start() -> None
```

```javascript
// JavaScript/TypeScript
async start(): Promise<void>
```

```go
// Go
func (sc *StreamingClient) Start() error
```

### stop / stop / Stop

Stop the streaming client.

#### Signature

```python
# Python
def stop() -> None
```

```javascript
// JavaScript/TypeScript
stop(): void
```

```go
// Go
func (sc *StreamingClient) Stop()
```

### send_data / sendData / SendData

Send a data point for anomaly detection.

#### Signature

```python
# Python
def send_data(data_point: List[float]) -> None
```

```javascript
// JavaScript/TypeScript
sendData(dataPoint: number[]): void
```

```go
// Go
func (sc *StreamingClient) SendData(dataPoint []float64) error
```

### Event Handlers (Streaming)

#### Python

```python
@client.on_anomaly
def handle_anomaly(anomaly_data: AnomalyData):
    pass

@client.on_connect
def handle_connect():
    pass

@client.on_disconnect
def handle_disconnect():
    pass

@client.on_error
def handle_error(error: Exception):
    pass
```

#### JavaScript/TypeScript

```javascript
client.on('anomaly', (anomaly: AnomalyData) => {});
client.on('connect', () => {});
client.on('disconnect', () => {});
client.on('error', (error: Error) => {});
client.on('message', (data: any) => {});
```

#### Go

```go
handlers := anomaly.StreamingHandlers{
    OnAnomaly:    func(anomaly anomaly.AnomalyData) {},
    OnConnect:    func() {},
    OnDisconnect: func() {},
    OnError:      func(err error) {},
    OnMessage:    func(data map[string]interface{}) {},
}
client.SetHandlers(handlers)
```

## Data Types

### AlgorithmType

Enumeration of available algorithms.

```python
# Python
class AlgorithmType(str, Enum):
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    AUTOENCODER = "autoencoder"
    ENSEMBLE = "ensemble"
```

```javascript
// JavaScript/TypeScript
enum AlgorithmType {
    ISOLATION_FOREST = 'isolation_forest',
    LOCAL_OUTLIER_FACTOR = 'local_outlier_factor',
    ONE_CLASS_SVM = 'one_class_svm',
    ELLIPTIC_ENVELOPE = 'elliptic_envelope',
    AUTOENCODER = 'autoencoder',
    ENSEMBLE = 'ensemble'
}
```

```go
// Go
type AlgorithmType string

const (
    IsolationForest     AlgorithmType = "isolation_forest"
    LocalOutlierFactor  AlgorithmType = "local_outlier_factor"
    OneClassSVM         AlgorithmType = "one_class_svm"
    EllipticEnvelope    AlgorithmType = "elliptic_envelope"
    Autoencoder         AlgorithmType = "autoencoder"
    Ensemble            AlgorithmType = "ensemble"
)
```

### AnomalyData

Represents an individual anomaly detection.

```python
# Python
class AnomalyData(BaseModel):
    index: int
    score: float
    data_point: List[float]
    confidence: Optional[float] = None
    timestamp: Optional[datetime] = None
```

```javascript
// JavaScript/TypeScript
interface AnomalyData {
    index: number;
    score: number;
    dataPoint: number[];
    confidence?: number;
    timestamp?: string;
}
```

```go
// Go
type AnomalyData struct {
    Index      int       `json:"index"`
    Score      float64   `json:"score"`
    DataPoint  []float64 `json:"data_point"`
    Confidence *float64  `json:"confidence,omitempty"`
    Timestamp  *time.Time `json:"timestamp,omitempty"`
}
```

### DetectionResult

Result of anomaly detection operation.

```python
# Python
class DetectionResult(BaseModel):
    anomalies: List[AnomalyData]
    total_points: int
    anomaly_count: int
    algorithm_used: AlgorithmType
    execution_time: float
    model_version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

```javascript
// JavaScript/TypeScript
interface DetectionResult {
    anomalies: AnomalyData[];
    totalPoints: number;
    anomalyCount: number;
    algorithmUsed: AlgorithmType;
    executionTime: number;
    modelVersion?: string;
    metadata: Record<string, any>;
}
```

```go
// Go
type DetectionResult struct {
    Anomalies     []AnomalyData      `json:"anomalies"`
    TotalPoints   int                `json:"total_points"`
    AnomalyCount  int                `json:"anomaly_count"`
    AlgorithmUsed AlgorithmType      `json:"algorithm_used"`
    ExecutionTime float64            `json:"execution_time"`
    ModelVersion  *string            `json:"model_version,omitempty"`
    Metadata      map[string]interface{} `json:"metadata"`
}
```

### ModelInfo

Information about a trained model.

```python
# Python
class ModelInfo(BaseModel):
    model_id: str
    algorithm: AlgorithmType
    created_at: datetime
    training_data_size: int
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    version: str
    status: str
```

```javascript
// JavaScript/TypeScript
interface ModelInfo {
    modelId: string;
    algorithm: AlgorithmType;
    createdAt: string;
    trainingDataSize: number;
    performanceMetrics: Record<string, number>;
    hyperparameters: Record<string, any>;
    version: string;
    status: string;
}
```

```go
// Go
type ModelInfo struct {
    ModelID            string                 `json:"model_id"`
    Algorithm          AlgorithmType          `json:"algorithm"`
    CreatedAt          time.Time              `json:"created_at"`
    TrainingDataSize   int                    `json:"training_data_size"`
    PerformanceMetrics map[string]float64     `json:"performance_metrics"`
    Hyperparameters    map[string]interface{} `json:"hyperparameters"`
    Version            string                 `json:"version"`
    Status             string                 `json:"status"`
}
```

### StreamingConfig

Configuration for streaming detection.

```python
# Python
class StreamingConfig(BaseModel):
    buffer_size: int = 100
    detection_threshold: float = 0.5
    batch_size: int = 10
    algorithm: AlgorithmType = AlgorithmType.ISOLATION_FOREST
    auto_retrain: bool = False
```

```javascript
// JavaScript/TypeScript
interface StreamingConfig {
    bufferSize?: number;
    detectionThreshold?: number;
    batchSize?: number;
    algorithm?: AlgorithmType;
    autoRetrain?: boolean;
}
```

```go
// Go
type StreamingConfig struct {
    BufferSize          int           `json:"buffer_size,omitempty"`
    DetectionThreshold  float64       `json:"detection_threshold,omitempty"`
    BatchSize           int           `json:"batch_size,omitempty"`
    Algorithm           AlgorithmType `json:"algorithm,omitempty"`
    AutoRetrain         bool          `json:"auto_retrain,omitempty"`
}
```

### ExplanationResult

Result of anomaly explanation.

```python
# Python
class ExplanationResult(BaseModel):
    anomaly_index: int
    feature_importance: Dict[str, float]
    shap_values: Optional[List[float]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    explanation_text: str
    confidence: float
```

```javascript
// JavaScript/TypeScript
interface ExplanationResult {
    anomalyIndex: number;
    featureImportance: Record<string, number>;
    shapValues?: number[];
    limeExplanation?: Record<string, any>;
    explanationText: string;
    confidence: number;
}
```

```go
// Go
type ExplanationResult struct {
    AnomalyIndex      int                    `json:"anomaly_index"`
    FeatureImportance map[string]float64     `json:"feature_importance"`
    ShapValues        []float64              `json:"shap_values,omitempty"`
    LimeExplanation   map[string]interface{} `json:"lime_explanation,omitempty"`
    ExplanationText   string                 `json:"explanation_text"`
    Confidence        float64                `json:"confidence"`
}
```

### HealthStatus

Health status of the service.

```python
# Python
class HealthStatus(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime: float
    components: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
```

```javascript
// JavaScript/TypeScript
interface HealthStatus {
    status: string;
    timestamp: string;
    version: string;
    uptime: number;
    components: Record<string, string>;
    metrics: Record<string, number | string>;
}
```

```go
// Go
type HealthStatus struct {
    Status     string                    `json:"status"`
    Timestamp  time.Time                 `json:"timestamp"`
    Version    string                    `json:"version"`
    Uptime     float64                   `json:"uptime"`
    Components map[string]string         `json:"components"`
    Metrics    map[string]interface{}    `json:"metrics"`
}
```

### BatchProcessingRequest

Request for batch processing.

```python
# Python
class BatchProcessingRequest(BaseModel):
    data: List[List[float]]
    algorithm: AlgorithmType = AlgorithmType.ISOLATION_FOREST
    parameters: Dict[str, Any] = Field(default_factory=dict)
    return_explanations: bool = False
```

```javascript
// JavaScript/TypeScript
interface BatchProcessingRequest {
    data: number[][];
    algorithm?: AlgorithmType;
    parameters?: Record<string, any>;
    returnExplanations?: boolean;
}
```

```go
// Go
type BatchProcessingRequest struct {
    Data               [][]float64            `json:"data"`
    Algorithm          AlgorithmType          `json:"algorithm,omitempty"`
    Parameters         map[string]interface{} `json:"parameters,omitempty"`
    ReturnExplanations bool                   `json:"return_explanations,omitempty"`
}
```

### TrainingRequest

Request for model training.

```python
# Python
class TrainingRequest(BaseModel):
    data: List[List[float]]
    algorithm: AlgorithmType
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    validation_split: float = 0.2
    model_name: Optional[str] = None
```

```javascript
// JavaScript/TypeScript
interface TrainingRequest {
    data: number[][];
    algorithm: AlgorithmType;
    hyperparameters?: Record<string, any>;
    validationSplit?: number;
    modelName?: string;
}
```

```go
// Go
type TrainingRequest struct {
    Data            [][]float64            `json:"data"`
    Algorithm       AlgorithmType          `json:"algorithm"`
    Hyperparameters map[string]interface{} `json:"hyperparameters,omitempty"`
    ValidationSplit float64                `json:"validation_split,omitempty"`
    ModelName       *string                `json:"model_name,omitempty"`
}
```

### TrainingResult

Result of model training.

```python
# Python
class TrainingResult(BaseModel):
    model_id: str
    training_time: float
    performance_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    model_info: ModelInfo
```

```javascript
// JavaScript/TypeScript
interface TrainingResult {
    modelId: string;
    trainingTime: number;
    performanceMetrics: Record<string, number>;
    validationMetrics: Record<string, number>;
    modelInfo: ModelInfo;
}
```

```go
// Go
type TrainingResult struct {
    ModelID           string             `json:"model_id"`
    TrainingTime      float64            `json:"training_time"`
    PerformanceMetrics map[string]float64 `json:"performance_metrics"`
    ValidationMetrics map[string]float64 `json:"validation_metrics"`
    ModelInfo         ModelInfo          `json:"model_info"`
}
```

## Error Types

### AnomalyDetectionSDKError / AnomalyDetectionError / SDKError

Base error class for all SDK errors.

```python
# Python
class AnomalyDetectionSDKError(Exception):
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
```

```javascript
// JavaScript/TypeScript
class AnomalyDetectionError extends Error {
    public readonly code?: string;
    public readonly details?: Record<string, any>;
    
    constructor(message: string, code?: string, details?: Record<string, any>) {
        super(message);
        this.code = code;
        this.details = details;
    }
}
```

```go
// Go
type SDKError struct {
    Message string
    Code    *string
    Details map[string]interface{}
}

func (e *SDKError) Error() string {
    if e.Code != nil {
        return "[" + *e.Code + "] " + e.Message
    }
    return e.Message
}
```

### APIError

Error from the API service.

```python
# Python
class APIError(AnomalyDetectionSDKError):
    def __init__(self, message: str, status_code: int, response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, f"HTTP_{status_code}", response_data)
        self.status_code = status_code
        self.response_data = response_data or {}
```

```javascript
// JavaScript/TypeScript
class APIError extends AnomalyDetectionError {
    public readonly statusCode: number;
    public readonly responseData?: Record<string, any>;
    
    constructor(message: string, statusCode: number, responseData?: Record<string, any>) {
        super(message, `HTTP_${statusCode}`, responseData);
        this.statusCode = statusCode;
        this.responseData = responseData;
    }
}
```

```go
// Go
type APIError struct {
    *SDKError
    StatusCode   int
    ResponseData map[string]interface{}
}
```

### ValidationError

Data validation error.

```python
# Python
class ValidationError(AnomalyDetectionSDKError):
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})
        self.field = field
        self.value = value
```

```javascript
// JavaScript/TypeScript
class ValidationError extends AnomalyDetectionError {
    public readonly field?: string;
    public readonly value?: any;
    
    constructor(message: string, field?: string, value?: any) {
        super(message, 'VALIDATION_ERROR', { field, value });
        this.field = field;
        this.value = value;
    }
}
```

```go
// Go
type ValidationError struct {
    *SDKError
    Field *string
    Value interface{}
}
```

### ConnectionError

Network connection error.

```python
# Python
class ConnectionError(AnomalyDetectionSDKError):
    def __init__(self, message: str, url: Optional[str] = None):
        super().__init__(message, "CONNECTION_ERROR", {"url": url})
        self.url = url
```

```javascript
// JavaScript/TypeScript
class ConnectionError extends AnomalyDetectionError {
    public readonly url?: string;
    
    constructor(message: string, url?: string) {
        super(message, 'CONNECTION_ERROR', { url });
        this.url = url;
    }
}
```

```go
// Go
type ConnectionError struct {
    *SDKError
    URL *string
}
```

### TimeoutError

Request timeout error.

```python
# Python
class TimeoutError(AnomalyDetectionSDKError):
    def __init__(self, message: str, timeout_duration: Optional[float] = None):
        super().__init__(message, "TIMEOUT_ERROR", {"timeout_duration": timeout_duration})
        self.timeout_duration = timeout_duration
```

```javascript
// JavaScript/TypeScript
class TimeoutError extends AnomalyDetectionError {
    public readonly timeoutDuration?: number;
    
    constructor(message: string, timeoutDuration?: number) {
        super(message, 'TIMEOUT_ERROR', { timeoutDuration });
        this.timeoutDuration = timeoutDuration;
    }
}
```

```go
// Go
type TimeoutError struct {
    *SDKError
    Duration time.Duration
}
```

### StreamingError

Streaming connection error.

```python
# Python
class StreamingError(AnomalyDetectionSDKError):
    def __init__(self, message: str, connection_status: Optional[str] = None):
        super().__init__(message, "STREAMING_ERROR", {"connection_status": connection_status})
        self.connection_status = connection_status
```

```javascript
// JavaScript/TypeScript
class StreamingError extends AnomalyDetectionError {
    public readonly connectionStatus?: string;
    
    constructor(message: string, connectionStatus?: string) {
        super(message, 'STREAMING_ERROR', { connectionStatus });
        this.connectionStatus = connectionStatus;
    }
}
```

```go
// Go
type StreamingError struct {
    *SDKError
    ConnectionStatus *string
}
```

## Configuration

### Algorithm Parameters

Each algorithm supports specific parameters:

#### Isolation Forest
```python
parameters = {
    'n_estimators': 100,        # Number of trees
    'contamination': 0.1,       # Expected anomaly ratio
    'max_samples': 'auto',      # Samples per tree
    'max_features': 1.0,        # Features per tree
    'bootstrap': False,         # Bootstrap sampling
    'random_state': 42          # Random seed
}
```

#### Local Outlier Factor
```python
parameters = {
    'n_neighbors': 20,          # Number of neighbors
    'contamination': 0.1,       # Expected anomaly ratio
    'algorithm': 'auto',        # Neighbor search algorithm
    'leaf_size': 30,           # Leaf size for tree algorithms
    'metric': 'minkowski',     # Distance metric
    'p': 2                     # Power parameter for Minkowski
}
```

#### One-Class SVM
```python
parameters = {
    'kernel': 'rbf',           # Kernel type
    'nu': 0.1,                 # Upper bound on anomalies
    'gamma': 'scale',          # Kernel coefficient
    'degree': 3,               # Polynomial degree
    'coef0': 0.0,             # Independent term
    'tol': 1e-3,              # Tolerance
    'shrinking': True,         # Use shrinking heuristic
    'cache_size': 200          # Cache size in MB
}
```

#### Elliptic Envelope
```python
parameters = {
    'contamination': 0.1,      # Expected anomaly ratio
    'store_precision': True,   # Store precision matrix
    'assume_centered': False,  # Assume centered data
    'support_fraction': None,  # Support fraction
    'random_state': 42        # Random seed
}
```

#### Autoencoder
```python
parameters = {
    'hidden_neurons': [64, 32, 64],  # Hidden layer sizes
    'epochs': 100,                   # Training epochs
    'batch_size': 32,               # Training batch size
    'learning_rate': 0.001,         # Learning rate
    'validation_size': 0.1,         # Validation split
    'preprocessing': True,          # Standardize data
    'l2_regularizer': 0.1,         # L2 regularization
    'dropout_rate': 0.2,           # Dropout rate
    'contamination': 0.1           # Expected anomaly ratio
}
```

#### Ensemble
```python
parameters = {
    'base_estimators': [         # List of base estimators
        ('isolation_forest', {'n_estimators': 50}),
        ('local_outlier_factor', {'n_neighbors': 10}),
        ('one_class_svm', {'nu': 0.1})
    ],
    'combination': 'average',    # Combination method
    'standardization_flag_list': [False, True, False],  # Standardization flags
    'contamination': 0.1        # Expected anomaly ratio
}
```

## Utilities

All SDKs provide utility functions for data processing and validation.

### Data Validation

```python
# Python
from anomaly_detection_sdk.utils import validate_data_format

try:
    validate_data_format(data)
    print("Data format is valid")
except ValidationError as e:
    print(f"Invalid data: {e.message}")
```

### Data Normalization

```python
# Python
from anomaly_detection_sdk.utils import normalize_data, apply_normalization

normalized_data, params = normalize_data(raw_data)
new_normalized = apply_normalization(new_data, params)
```

### Statistics Calculation

```python
# Python
from anomaly_detection_sdk.utils import calculate_data_statistics

stats = calculate_data_statistics(data)
print(f"Dataset: {stats.num_samples} samples, {stats.num_features} features")
```

### Sample Data Generation

```python
# Python
from anomaly_detection_sdk.utils import generate_sample_data

sample_data, labels = generate_sample_data(1000, 5, 0.1)  # 1000 samples, 5 features, 10% anomalies
```

### Train/Validation Split

```python
# Python
from anomaly_detection_sdk.utils import train_validation_split

split_data = train_validation_split(data, validation_ratio=0.2, shuffle=True)
```

---

For more examples and detailed usage patterns, see the [Examples](../examples/) directory and [SDK-specific guides](sdk-guides/).
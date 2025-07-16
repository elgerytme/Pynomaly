# Getting Started with Monorepo API

This guide will help you get started with the Monorepo API for anomaly detection. You'll learn how to authenticate, upload data, create detectors, and perform anomaly detection.

## Prerequisites

- Python 3.7+ (for Python examples)
- Node.js 14+ (for JavaScript examples)
- curl (for command-line examples)
- A Monorepo account with API access

## Step 1: Authentication

### Obtain API Credentials

1. Log in to your Monorepo account
2. Navigate to Settings > API Keys
3. Create a new API key with appropriate permissions
4. Store your API key securely (never commit it to version control)

### Authentication Methods

#### JWT Bearer Token (Recommended)

```bash
# Login to get JWT token
curl -X POST https://api.monorepo.com/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your_email@example.com&password=your_password"
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### API Key Authentication

```bash
# Using API key in header (recommended)
curl -H "X-API-Key: your_api_key" https://api.monorepo.com/v1/health

# Using API key in query parameter
curl "https://api.monorepo.com/v1/health?api_key=your_api_key"
```

## Step 2: Check API Health

Before starting, verify that the API is accessible:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  https://api.monorepo.com/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 86400.0,
  "checks": {
    "database": {
      "status": "healthy",
      "response_time": 15.2,
      "last_checked": "2024-01-01T12:00:00Z"
    },
    "redis": {
      "status": "healthy",
      "response_time": 5.1,
      "last_checked": "2024-01-01T12:00:00Z"
    }
  }
}
```

## Step 3: Upload a Dataset

### Prepare Your Data

Your dataset should be in one of the supported formats:
- CSV (comma-separated values)
- JSON (array of objects)
- Parquet (Apache Parquet format)
- Excel (XLSX format)

Example CSV format:
```csv
timestamp,temperature,humidity,pressure,anomaly_label
2024-01-01 00:00:00,25.5,60.2,1013.2,0
2024-01-01 01:00:00,25.1,61.8,1012.9,0
2024-01-01 02:00:00,24.8,63.1,1012.5,0
2024-01-01 03:00:00,45.2,25.3,1050.1,1
```

### Upload via API

```bash
curl -X POST https://api.monorepo.com/v1/datasets/upload \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@sensor_data.csv" \
  -F "name=IoT Sensor Data" \
  -F "description=Temperature and humidity sensor readings" \
  -F "tags=iot,sensors,temperature"
```

### Python Example

```python
import requests

headers = {"Authorization": "Bearer YOUR_JWT_TOKEN"}

# Upload dataset
with open('sensor_data.csv', 'rb') as f:
    files = {'file': f}
    data = {
        'name': 'IoT Sensor Data',
        'description': 'Temperature and humidity sensor readings',
        'tags': ['iot', 'sensors', 'temperature']
    }
    
    response = requests.post(
        'https://api.monorepo.com/v1/datasets/upload',
        headers=headers,
        files=files,
        data=data
    )
    
    dataset = response.json()
    dataset_id = dataset['id']
    print(f"Dataset uploaded with ID: {dataset_id}")
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('name', 'IoT Sensor Data');
formData.append('description', 'Temperature and humidity sensor readings');
formData.append('tags', JSON.stringify(['iot', 'sensors', 'temperature']));

const response = await fetch('https://api.monorepo.com/v1/datasets/upload', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer YOUR_JWT_TOKEN'
    },
    body: formData
});

const dataset = await response.json();
console.log('Dataset uploaded:', dataset);
```

## Step 4: Create an Anomaly Detector

### Choose an Algorithm

Monorepo supports multiple anomaly detection algorithms:

- **Isolation Forest**: Good for high-dimensional data
- **Local Outlier Factor**: Effective for local anomalies
- **One-Class SVM**: Robust for various data types
- **Autoencoder**: Neural network-based detection
- **DBSCAN**: Density-based clustering
- **And many more...**

### Create Detector

```bash
curl -X POST https://api.monorepo.com/v1/detectors/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "IoT Anomaly Detector",
    "description": "Detector for IoT sensor anomalies",
    "algorithm": "isolation_forest",
    "dataset_id": "YOUR_DATASET_ID",
    "hyperparameters": {
      "n_estimators": 100,
      "contamination": 0.1,
      "random_state": 42
    }
  }'
```

### Python Example

```python
detector_data = {
    "name": "IoT Anomaly Detector",
    "description": "Detector for IoT sensor anomalies",
    "algorithm": "isolation_forest",
    "dataset_id": dataset_id,
    "hyperparameters": {
        "n_estimators": 100,
        "contamination": 0.1,
        "random_state": 42
    }
}

response = requests.post(
    'https://api.monorepo.com/v1/detectors/',
    headers=headers,
    json=detector_data
)

detector = response.json()
detector_id = detector['id']
print(f"Detector created with ID: {detector_id}")
```

## Step 5: Train the Detector

### Start Training

```bash
curl -X POST https://api.monorepo.com/v1/detection/train \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detector_id": "YOUR_DETECTOR_ID",
    "dataset_id": "YOUR_DATASET_ID",
    "validation_split": 0.2,
    "cross_validation": true
  }'
```

### Python Example

```python
train_data = {
    "detector_id": detector_id,
    "dataset_id": dataset_id,
    "validation_split": 0.2,
    "cross_validation": True
}

response = requests.post(
    'https://api.monorepo.com/v1/detection/train',
    headers=headers,
    json=train_data
)

training_job = response.json()
job_id = training_job['job_id']
print(f"Training started with job ID: {job_id}")
```

### Monitor Training Progress

```python
import time

while True:
    response = requests.get(
        f'https://api.monorepo.com/v1/detection/train/{job_id}',
        headers=headers
    )
    
    job_status = response.json()
    print(f"Training progress: {job_status['progress']:.1%}")
    
    if job_status['status'] == 'completed':
        print("Training completed successfully!")
        break
    elif job_status['status'] == 'failed':
        print(f"Training failed: {job_status['error_message']}")
        break
    
    time.sleep(10)  # Check every 10 seconds
```

## Step 6: Detect Anomalies

### Single Detection

```bash
curl -X POST https://api.monorepo.com/v1/detection/detect \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detector_id": "YOUR_DETECTOR_ID",
    "data": [
      {
        "timestamp": "2024-01-01 12:00:00",
        "temperature": 25.5,
        "humidity": 60.2,
        "pressure": 1013.2
      },
      {
        "timestamp": "2024-01-01 13:00:00",
        "temperature": 45.2,
        "humidity": 25.3,
        "pressure": 1050.1
      }
    ],
    "return_scores": true,
    "return_explanations": true
  }'
```

### Python Example

```python
detection_data = {
    "detector_id": detector_id,
    "data": [
        {
            "timestamp": "2024-01-01 12:00:00",
            "temperature": 25.5,
            "humidity": 60.2,
            "pressure": 1013.2
        },
        {
            "timestamp": "2024-01-01 13:00:00",
            "temperature": 45.2,  # Anomalous temperature
            "humidity": 25.3,
            "pressure": 1050.1
        }
    ],
    "return_scores": True,
    "return_explanations": True
}

response = requests.post(
    'https://api.monorepo.com/v1/detection/detect',
    headers=headers,
    json=detection_data
)

results = response.json()
print("Detection results:")
for i, anomaly in enumerate(results['anomalies']):
    print(f"Data point {i}: {'ANOMALY' if anomaly['is_anomaly'] else 'NORMAL'}")
    print(f"  Score: {anomaly['score']:.3f}")
    print(f"  Confidence: {anomaly['confidence']:.3f}")
    if anomaly.get('explanation'):
        print(f"  Explanation: {anomaly['explanation']}")
```

### Expected Response

```json
{
  "detection_id": "det_123456789",
  "detector_id": "YOUR_DETECTOR_ID",
  "anomalies": [
    {
      "index": 0,
      "score": 0.15,
      "is_anomaly": false,
      "confidence": 0.85,
      "original_data": {
        "timestamp": "2024-01-01 12:00:00",
        "temperature": 25.5,
        "humidity": 60.2,
        "pressure": 1013.2
      }
    },
    {
      "index": 1,
      "score": 0.89,
      "is_anomaly": true,
      "confidence": 0.92,
      "explanation": {
        "feature_importance": {
          "temperature": 0.85,
          "pressure": 0.12,
          "humidity": 0.03
        },
        "reason": "Temperature value significantly exceeds normal range"
      },
      "original_data": {
        "timestamp": "2024-01-01 13:00:00",
        "temperature": 45.2,
        "humidity": 25.3,
        "pressure": 1050.1
      }
    }
  ],
  "statistics": {
    "total_points": 2,
    "anomaly_count": 1,
    "anomaly_rate": 0.5,
    "processing_time": 0.15
  },
  "threshold": 0.8,
  "created_at": "2024-01-01T12:00:00Z"
}
```

## Step 7: Batch Detection

For processing large amounts of data, use batch detection:

```python
batch_data = {
    "detections": [
        {
            "detector_id": detector_id,
            "data": batch_1_data
        },
        {
            "detector_id": detector_id,
            "data": batch_2_data
        }
    ],
    "parallel": True,
    "notification_webhook": "https://your-app.com/webhook/batch-complete"
}

response = requests.post(
    'https://api.monorepo.com/v1/detection/detect/batch',
    headers=headers,
    json=batch_data
)

batch_job = response.json()
print(f"Batch detection started: {batch_job['batch_id']}")
```

## Step 8: Evaluate Detector Performance

If you have labeled data, evaluate your detector's performance:

```python
evaluation_data = {
    "detector_id": detector_id,
    "test_dataset_id": test_dataset_id,
    "ground_truth_column": "anomaly_label",
    "metrics": ["precision", "recall", "f1_score", "auc_roc"]
}

response = requests.post(
    'https://api.monorepo.com/v1/detection/evaluate',
    headers=headers,
    json=evaluation_data
)

evaluation = response.json()
print("Performance metrics:")
print(f"  Precision: {evaluation['metrics']['precision']:.3f}")
print(f"  Recall: {evaluation['metrics']['recall']:.3f}")
print(f"  F1-Score: {evaluation['metrics']['f1_score']:.3f}")
print(f"  AUC-ROC: {evaluation['metrics']['auc_roc']:.3f}")
```

## Error Handling

Always implement proper error handling:

```python
try:
    response = requests.post(
        'https://api.monorepo.com/v1/detection/detect',
        headers=headers,
        json=detection_data
    )
    
    # Check for HTTP errors
    response.raise_for_status()
    
    results = response.json()
    
    # Check for API errors
    if 'error' in results:
        print(f"API Error: {results['error']} - {results['message']}")
        return
    
    # Process results
    process_detection_results(results)
    
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### 1. Data Preparation

- **Clean your data**: Remove duplicates and handle missing values
- **Feature engineering**: Create relevant features for better detection
- **Normalization**: Consider normalizing numerical features
- **Validation**: Use a validation set to tune hyperparameters

### 2. Algorithm Selection

- **Start simple**: Begin with Isolation Forest or LOF
- **Consider data characteristics**: Choose algorithms suitable for your data type
- **Use AutoML**: Let Monorepo automatically select the best algorithm
- **Ensemble methods**: Combine multiple algorithms for better performance

### 3. Performance Optimization

- **Batch processing**: Use batch detection for large datasets
- **Caching**: Cache detector objects to avoid re-training
- **Parallel processing**: Enable parallel processing for batch operations
- **Monitoring**: Monitor API usage and performance metrics

### 4. Security

- **API key management**: Store API keys securely
- **Environment variables**: Use environment variables for sensitive data
- **Rate limiting**: Respect API rate limits
- **HTTPS**: Always use HTTPS for API calls

## Next Steps

- **Explore AutoML**: Use automated machine learning for optimal results
- **Try ensemble methods**: Combine multiple detectors for better accuracy
- **Set up monitoring**: Monitor your detectors in production
- **Implement streaming**: Use WebSocket for real-time detection
- **Add explainability**: Understand why anomalies are detected

## Additional Resources

- [Complete API Reference](./openapi.yaml)
- [Python SDK Documentation](./python-sdk.md)
- [JavaScript SDK Documentation](./javascript-sdk.md)
- [Advanced Examples](./examples/)
- [Troubleshooting Guide](./troubleshooting.md)

## Support

If you need help:
- Check the [FAQ](./faq.md)
- Visit our [Community Forum](https://community.monorepo.com)
- Contact support at [support@monorepo.com](mailto:support@monorepo.com)
- Report issues on [GitHub](https://github.com/monorepo/monorepo/issues)
# Pynomaly API Documentation

Welcome to the Pynomaly API documentation! This directory contains comprehensive documentation for the Pynomaly anomaly detection platform.

## 📁 Documentation Structure

```
docs/api/
├── API_DOCUMENTATION.md         # Main API documentation
├── generated/                   # Generated documentation files
│   ├── openapi.json            # OpenAPI 3.0 specification (JSON)
│   ├── openapi.yaml            # OpenAPI 3.0 specification (YAML)
│   ├── index.html              # Interactive Swagger UI
│   ├── examples/               # Code examples
│   │   ├── python/            # Python examples
│   │   ├── javascript/        # JavaScript examples
│   │   └── curl/              # cURL examples
│   └── pynomaly_api.postman_collection.json  # Postman collection
└── README.md                   # This file
```

## 🚀 Getting Started

### 1. Interactive Documentation

Open `generated/index.html` in your browser to explore the API interactively with Swagger UI.

### 2. API Specification

- **OpenAPI JSON**: `generated/openapi.json`
- **OpenAPI YAML**: `generated/openapi.yaml`

### 3. Code Examples

Choose your preferred language:

- **Python**: `generated/examples/python/`
- **JavaScript**: `generated/examples/javascript/`
- **cURL**: `generated/examples/curl/`

### 4. Postman Collection

Import `generated/pynomaly_api.postman_collection.json` into Postman for easy API testing.

## 📖 API Overview

The Pynomaly API provides comprehensive REST endpoints for:

- **Health Monitoring**: System health and status checks
- **Detector Management**: Create, update, and manage anomaly detectors
- **Detection Operations**: Run anomaly detection on data
- **AutoML**: Automated machine learning optimization
- **Dataset Management**: Handle training and testing datasets
- **Metrics & Monitoring**: Performance and system metrics
- **Explainability**: Model interpretability and explanations

## 🔐 Authentication

All API endpoints require authentication via API key:

```bash
# Header authentication (recommended)
curl -H "X-API-Key: your-api-key" https://api.pynomaly.com/api/v1/detectors

# Query parameter authentication
curl "https://api.pynomaly.com/api/v1/detectors?api_key=your-api-key"
```

## 📚 Quick Examples

### Python

```python
import requests

headers = {"X-API-Key": "your-api-key"}
response = requests.get("https://api.pynomaly.com/health", headers=headers)
print(response.json())
```

### JavaScript

```javascript
const response = await fetch('https://api.pynomaly.com/health', {
    headers: { 'X-API-Key': 'your-api-key' }
});
const data = await response.json();
console.log(data);
```

### cURL

```bash
curl -H "X-API-Key: your-api-key" https://api.pynomaly.com/health
```

## 🌐 Base URLs

- **Production**: `https://api.pynomaly.com`
- **Staging**: `https://staging-api.pynomaly.com`
- **Development**: `http://localhost:8000`

## 📊 Rate Limits

- **Default**: 1000 requests per minute
- **Burst**: 100 requests per second
- **Training**: 10 concurrent jobs per user
- **Detection**: 10,000 requests per minute

## 📧 Support

- **Documentation**: https://docs.pynomaly.com
- **API Status**: https://status.pynomaly.com
- **Support**: support@pynomaly.com
- **Community**: https://github.com/pynomaly/pynomaly

## 🔄 Regenerating Documentation

To regenerate this documentation:

```bash
cd /path/to/pynomaly
python scripts/generate_api_docs.py
```

---

*Last updated: 2025-07-09 16:28:17*

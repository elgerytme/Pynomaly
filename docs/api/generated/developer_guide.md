# Pynomaly API Developer Guide

## Overview

The Pynomaly API provides comprehensive anomaly detection capabilities with built-in authentication, multi-factor authentication (MFA), and administrative features.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.pynomaly.com`

## Authentication

### JWT Token Authentication

1. **Login** to get JWT token:
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

2. **Use token** in subsequent requests:
```bash
curl -X GET http://localhost:8000/api/v1/health \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### API Key Authentication

1. **Create API key** (requires admin privileges):
```bash
curl -X POST http://localhost:8000/api/v1/api-keys \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My API Key",
    "scopes": ["detection:read", "detection:write"],
    "expires_at": "2024-12-31T23:59:59Z"
  }'
```

2. **Use API key** in requests:
```bash
curl -X GET http://localhost:8000/api/v1/health \
  -H "X-API-Key: YOUR_API_KEY"
```

## Multi-Factor Authentication (MFA)

### TOTP Setup

1. **Initialize TOTP**:
```bash
curl -X POST http://localhost:8000/api/v1/mfa/totp/setup \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "app_name": "Pynomaly",
    "issuer": "Pynomaly Security"
  }'
```

2. **Verify TOTP code**:
```bash
curl -X POST http://localhost:8000/api/v1/mfa/totp/verify \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"totp_code": "123456"}'
```

### SMS Authentication

1. **Setup SMS**:
```bash
curl -X POST http://localhost:8000/api/v1/mfa/sms/setup \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "+1234567890"}'
```

2. **Verify SMS code**:
```bash
curl -X POST http://localhost:8000/api/v1/mfa/sms/verify \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"sms_code": "123456"}'
```

## Anomaly Detection

### Basic Detection

```bash
curl -X POST http://localhost:8000/api/v1/detection/detect \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
    "algorithm": "isolation_forest",
    "parameters": {
      "contamination": 0.1,
      "random_state": 42
    }
  }'
```

### Batch Detection

```bash
curl -X POST http://localhost:8000/api/v1/detection/batch \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "datasets": [
      {
        "name": "dataset1",
        "data": [1, 2, 3, 4, 5, 100]
      },
      {
        "name": "dataset2",
        "data": [10, 20, 30, 40, 50, 1000]
      }
    ],
    "algorithm": "isolation_forest"
  }'
```

## Error Handling

The API uses RFC 7807 Problem Details for HTTP APIs format:

```json
{
  "type": "https://api.pynomaly.com/problems/validation-error",
  "title": "Validation Error",
  "detail": "The request body contains invalid data",
  "status": 400,
  "instance": "/api/v1/detection/detect"
}
```

## Rate Limiting

The API implements rate limiting:
- **Authenticated users**: 1000 requests per hour
- **API keys**: 5000 requests per hour
- **Admin users**: 10000 requests per hour

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## SDKs and Code Examples

### Python SDK

```python
import requests
from typing import List, Dict, Any

class PynomaliAPI:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def detect_anomalies(self, data: List[float], algorithm: str = "isolation_forest") -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/v1/detection/detect",
            headers=self.headers,
            json={
                "data": data,
                "algorithm": algorithm
            }
        )
        return response.json()
    
    def get_health(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/v1/health", headers=self.headers)
        return response.json()

# Usage
api = PynomaliAPI("http://localhost:8000", "your_jwt_token")
result = api.detect_anomalies([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
print(result)
```

### JavaScript SDK

```javascript
class PynomaliAPI {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async detectAnomalies(data, algorithm = 'isolation_forest') {
        const response = await fetch(`${this.baseUrl}/api/v1/detection/detect`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                data: data,
                algorithm: algorithm
            })
        });
        return await response.json();
    }
    
    async getHealth() {
        const response = await fetch(`${this.baseUrl}/api/v1/health`, {
            headers: this.headers
        });
        return await response.json();
    }
}

// Usage
const api = new PynomaliAPI('http://localhost:8000', 'your_jwt_token');
api.detectAnomalies([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
    .then(result => console.log(result));
```

## Best Practices

1. **Always use HTTPS** in production
2. **Implement proper error handling** for all API calls
3. **Use API keys** for server-to-server communication
4. **Enable MFA** for enhanced security
5. **Monitor rate limits** and implement backoff strategies
6. **Keep tokens secure** and refresh them regularly
7. **Use appropriate scopes** for API keys

## Support

For support and questions:
- GitHub Issues: https://github.com/pynomaly/pynomaly/issues
- Email: support@pynomaly.com
- Documentation: https://docs.pynomaly.com

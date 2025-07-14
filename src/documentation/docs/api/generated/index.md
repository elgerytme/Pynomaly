# Pynomaly API Documentation

Generated on: 2025-07-09 16:40:37

## Quick Start

1. **[Developer Guide](developer_guide.md)** - Complete integration guide
2. **[Authentication Guide](authentication_guide.md)** - Security implementation
3. **[API Versioning Strategy](versioning_strategy.md)** - Version management

## API Specifications

- **[OpenAPI JSON](openapi.json)** - Machine-readable API specification
- **[OpenAPI YAML](openapi.yaml)** - Human-readable API specification
- **[Postman Collection](pynomaly_api.postman_collection.json)** - Ready-to-use API testing

## API Overview

### Authentication Endpoints
- JWT token authentication
- Multi-factor authentication (MFA)
- API key management
- Password reset functionality

### Detection Endpoints
- Real-time anomaly detection
- Batch processing
- Algorithm selection
- Custom parameters

### Admin Endpoints
- User management
- Role assignment
- System monitoring
- Configuration management

### Health Endpoints
- Service health checks
- Detailed system status
- Performance metrics

## Getting Started

### 1. Authentication
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### 2. Basic Detection
```bash
curl -X POST http://localhost:8000/api/v1/detection/detect \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
    "algorithm": "isolation_forest"
  }'
```

### 3. Health Check
```bash
curl -X GET http://localhost:8000/api/v1/health \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Development Tools

### Import Postman Collection
1. Open Postman
2. Click "Import"
3. Select `pynomaly_api.postman_collection.json`
4. Set environment variables:
   - `base_url`: Your API base URL
   - `auth_token`: Your JWT token

### Generate SDK
Use the OpenAPI specification to generate SDKs:
```bash
# Python SDK
openapi-generator-cli generate -i openapi.json -g python -o python-sdk/

# JavaScript SDK
openapi-generator-cli generate -i openapi.json -g javascript -o javascript-sdk/
```

## Support

- **GitHub**: https://github.com/pynomaly/pynomaly
- **Issues**: https://github.com/pynomaly/pynomaly/issues
- **Email**: support@pynomaly.com

## Security

- Always use HTTPS in production
- Implement proper token management
- Enable MFA for enhanced security
- Follow rate limiting guidelines
- Keep API keys secure

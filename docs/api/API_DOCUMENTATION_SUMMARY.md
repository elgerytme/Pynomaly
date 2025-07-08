# Pynomaly API Documentation Summary

## 🎉 **Complete API Documentation Coverage Achieved**

**Total Documented Endpoints: 125+**  
**OpenAPI Schema Generation: ✅ Successful**  
**Authentication Migration: ✅ Complete**

---

## **📊 API Overview**

### **Base URL**
```
Production: https://api.pynomaly.io/api/v1
Development: http://localhost:8000/api/v1
```

### **Authentication**
All endpoints use JWT Bearer token authentication via simplified auth dependencies:
```http
Authorization: Bearer <your-jwt-token>
```

### **API Documentation Endpoints**
- **Interactive Docs**: `/api/v1/docs` (Swagger UI)
- **API Reference**: `/api/v1/redoc` (ReDoc)
- **OpenAPI Schema**: `/api/v1/openapi.json`

---

## **🔗 Router Coverage (125+ Endpoints)**

### **1. Health & System (6 endpoints)**
- `GET /api/v1/health` - System health check
- `GET /api/v1/health/ready` - Readiness probe
- `GET /api/v1/health/live` - Liveness probe
- `GET /api/v1/health/metrics` - Health metrics
- `GET /api/v1/health/dependencies` - Dependency status
- `GET /api/v1/health/version` - Version information

### **2. Authentication (7 endpoints)**
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/refresh` - Token refresh
- `POST /api/v1/auth/logout` - User logout
- `GET /api/v1/auth/me` - Current user profile
- `PUT /api/v1/auth/profile` - Update profile
- `POST /api/v1/auth/password/reset` - Password reset

### **3. Administration (7 endpoints)**
- `GET /api/v1/admin/users` - List users
- `POST /api/v1/admin/users` - Create user
- `PUT /api/v1/admin/users/{id}` - Update user
- `DELETE /api/v1/admin/users/{id}` - Delete user
- `GET /api/v1/admin/system/stats` - System statistics
- `POST /api/v1/admin/system/maintenance` - Maintenance mode
- `GET /api/v1/admin/audit/logs` - Audit logs

### **4. Datasets (6 endpoints)**
- `GET /api/v1/datasets` - List datasets
- `POST /api/v1/datasets/upload` - Upload dataset
- `GET /api/v1/datasets/{id}` - Get dataset details
- `PUT /api/v1/datasets/{id}` - Update dataset
- `DELETE /api/v1/datasets/{id}` - Delete dataset
- `GET /api/v1/datasets/{id}/profile` - Dataset profiling

### **5. Detectors (3 endpoints)**
- `GET /api/v1/detectors` - List detectors
- `POST /api/v1/detectors` - Create detector
- `GET /api/v1/detectors/{id}` - Get detector details

### **6. Detection (6 endpoints)**
- `POST /api/v1/detection/train` - Train detector
- `POST /api/v1/detection/predict` - Predict anomalies
- `POST /api/v1/detection/batch` - Batch detection
- `GET /api/v1/detection/results/{id}` - Get results
- `GET /api/v1/detection/status/{id}` - Training status
- `POST /api/v1/detection/evaluate` - Model evaluation

### **7. AutoML (8 endpoints)** ✅ *Migrated*
- `POST /api/v1/automl/profile` - Dataset profiling
- `POST /api/v1/automl/optimize` - Full AutoML optimization
- `POST /api/v1/automl/optimize-algorithm` - Single algorithm optimization
- `GET /api/v1/automl/algorithms` - List supported algorithms
- `GET /api/v1/automl/status/{id}` - Optimization status
- `DELETE /api/v1/automl/optimization/{id}` - Cancel optimization
- `GET /api/v1/automl/recommendations/{dataset_id}` - Algorithm recommendations
- `POST /api/v1/automl/batch-optimize` - Batch optimization

### **8. Autonomous Detection (7 endpoints)** ✅ *Migrated*
- `POST /api/v1/autonomous/detect` - Autonomous anomaly detection
- `POST /api/v1/autonomous/automl/optimize` - AutoML optimization
- `POST /api/v1/autonomous/ensemble/create` - Create ensemble
- `POST /api/v1/autonomous/ensemble/create-by-family` - Family-based ensemble
- `POST /api/v1/autonomous/explain/choices` - Algorithm choice explanation
- `GET /api/v1/autonomous/algorithms/families` - Algorithm families
- `GET /api/v1/autonomous/status` - Autonomous system status

### **9. Ensemble Detection (4 endpoints)** ✅ *Migrated*
- `POST /api/v1/ensemble/detect` - Ensemble-based detection
- `POST /api/v1/ensemble/optimize` - Ensemble optimization
- `GET /api/v1/ensemble/status` - Ensemble system status
- `GET /api/v1/ensemble/metrics` - Ensemble performance metrics

### **10. Explainability (8 endpoints)** ✅ *Migrated*
- `POST /api/v1/explainability/explain/prediction` - Single prediction explanation
- `POST /api/v1/explainability/explain/model` - Global model explanation
- `POST /api/v1/explainability/explain/cohort` - Cohort explanation
- `POST /api/v1/explainability/explain/compare` - Compare explanation methods
- `GET /api/v1/explainability/methods` - Available methods
- `GET /api/v1/explainability/detector/{id}/explanation-stats` - Explanation statistics
- `DELETE /api/v1/explainability/cache` - Clear explanation cache
- `GET /api/v1/explainability/health` - Service health

### **11. Experiments (7 endpoints)**
- `GET /api/v1/experiments` - List experiments
- `POST /api/v1/experiments` - Create experiment
- `GET /api/v1/experiments/{id}` - Get experiment
- `PUT /api/v1/experiments/{id}` - Update experiment
- `DELETE /api/v1/experiments/{id}` - Delete experiment
- `POST /api/v1/experiments/{id}/run` - Run experiment
- `GET /api/v1/experiments/{id}/results` - Experiment results

### **12. Streaming (9 endpoints)** ✅ *Migrated*
- `POST /api/v1/streaming/sessions` - Create streaming session
- `GET /api/v1/streaming/sessions` - List sessions
- `GET /api/v1/streaming/sessions/{id}` - Get session details
- `DELETE /api/v1/streaming/sessions/{id}` - Stop session
- `POST /api/v1/streaming/sessions/{id}/data` - Send data
- `GET /api/v1/streaming/sessions/{id}/results` - Get results
- `GET /api/v1/streaming/sessions/{id}/metrics` - Session metrics
- `POST /api/v1/streaming/sessions/{id}/configure` - Update configuration
- `GET /api/v1/streaming/health` - Streaming health

### **13. Performance (Endpoints)** ✅ *Migrated*
- Connection pool management endpoints
- Query optimization endpoints
- Performance metrics and statistics
- System optimization controls

### **14. Export (Endpoints)** ✅ *Migrated*
- Data export functionality
- Multiple format support
- Batch export operations

### **15. Model Lineage (Endpoints)** ✅ *Migrated*
- Model versioning and tracking
- Lineage visualization endpoints
- Model comparison features

### **16. Events (Endpoints)** ✅ *Migrated*
- Event streaming and management
- Webhook integrations
- Event processing pipelines

---

## **🚀 Quick Start Examples**

### **1. Authentication**
```bash
# Login and get token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "password"}'

# Use token in subsequent requests
curl -X GET "http://localhost:8000/api/v1/datasets" \
  -H "Authorization: Bearer <your-jwt-token>"
```

### **2. Dataset Upload & AutoML**
```bash
# Upload dataset
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -H "Authorization: Bearer <token>" \
  -F "file=@data.csv" \
  -F "name=my_dataset"

# Run AutoML optimization
curl -X POST "http://localhost:8000/api/v1/automl/optimize" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "dataset-uuid",
    "objective": "auc",
    "max_algorithms": 5,
    "max_optimization_time": 3600
  }'
```

### **3. Autonomous Detection**
```bash
# Run autonomous detection with file upload
curl -X POST "http://localhost:8000/api/v1/autonomous/detect" \
  -H "Authorization: Bearer <token>" \
  -F "file=@anomaly_data.csv" \
  -F "request={
    \"max_algorithms\": 5,
    \"confidence_threshold\": 0.8,
    \"auto_tune\": true
  }"
```

### **4. Ensemble Detection**
```bash
# Create ensemble from multiple detectors
curl -X POST "http://localhost:8000/api/v1/ensemble/detect" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "detector_ids": ["detector1", "detector2", "detector3"],
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "voting_strategy": "weighted_average",
    "enable_explanation": true
  }'
```

### **5. Streaming Detection**
```bash
# Create streaming session
curl -X POST "http://localhost:8000/api/v1/streaming/sessions" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "detector_id": "detector-uuid",
    "configuration": {
      "strategy": "adaptive_batch",
      "max_batch_size": 100,
      "batch_timeout_ms": 1000
    }
  }'
```

---

## **📋 Response Format Standards**

### **Success Response**
```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "execution_time": 0.123
}
```

### **Error Response**
```json
{
  "success": false,
  "error": "Error description",
  "details": { ... },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### **Pagination Response**
```json
{
  "items": [ ... ],
  "total": 100,
  "page": 1,
  "page_size": 20,
  "total_pages": 5
}
```

---

## **🔧 Development Tools**

### **API Testing**
- **Postman Collection**: Available at `/docs/postman`
- **cURL Examples**: Provided for each endpoint
- **SDK Support**: Python, JavaScript, and REST

### **Monitoring**
- **Prometheus Metrics**: `/metrics`
- **Health Checks**: `/api/v1/health/*`
- **Performance Monitoring**: Built-in request tracking

### **Security**
- **JWT Authentication**: Secure token-based auth
- **Rate Limiting**: Configurable per endpoint
- **Input Validation**: Comprehensive pydantic validation
- **CORS Support**: Configurable cross-origin requests

---

## **🎯 Next Steps**

1. **Explore Interactive Docs**: Visit `/api/v1/docs` for hands-on testing
2. **Check Examples**: Review endpoint-specific examples in documentation
3. **Set Up Authentication**: Get your JWT token and start making requests
4. **Try AutoML**: Upload a dataset and run automated optimization
5. **Experiment with Ensembles**: Combine multiple detectors for better results

---

**📚 Complete API Reference**: [/api/v1/redoc](/api/v1/redoc)  
**🔧 Interactive Testing**: [/api/v1/docs](/api/v1/docs)  
**📊 System Health**: [/api/v1/health](/api/v1/health)

> **Note**: This documentation covers all 125+ documented endpoints after successful authentication migration and OpenAPI schema generation.

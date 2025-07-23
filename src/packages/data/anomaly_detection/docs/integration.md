# Integration Examples and Workflows Guide

This guide provides comprehensive examples and workflows for integrating the Anomaly Detection package with various systems, platforms, and existing infrastructure components.

!!! info "Prerequisites"
    - **Basic understanding needed?** Start with the [Getting Started Guide](getting-started/index.md)
    - **API reference required?** Check the [API Documentation](api.md) for endpoint details
    - **CLI integration?** See the [CLI Guide](cli.md) for command-line automation
    - **Need deployment context?** Review the [Deployment Guide](deployment.md) first

!!! tip "Integration Pathways"
    - **Real-time processing?** Combine with [Streaming Detection](streaming.md) patterns
    - **Production deployment?** Use [Security & Privacy](security.md) for enterprise integration
    - **Performance critical?** Apply [Performance Optimization](performance.md) techniques

## Table of Contents

1. [Overview](#overview)
2. [API Integration](#api-integration)
3. [Database Integration](#database-integration)
4. [Message Queue Integration](#message-queue-integration)
5. [Cloud Platform Integration](#cloud-platform-integration)
6. [Monitoring System Integration](#monitoring-system-integration)
7. [Data Pipeline Integration](#data-pipeline-integration)
8. [Web Framework Integration](#web-framework-integration)
9. [Notebook and Analytics Integration](#notebook-and-analytics-integration)
10. [Enterprise System Integration](#enterprise-system-integration)
11. [Workflow Orchestration](#workflow-orchestration)
12. [Best Practices](#best-practices)

## Overview

The Anomaly Detection package is designed for seamless integration with modern data infrastructure. This guide demonstrates practical integration patterns, code examples, and best practices for various use cases.

### Integration Patterns

- **Event-Driven Integration**: React to data changes and events
- **API-First Integration**: RESTful and GraphQL API patterns
- **Stream Processing Integration**: Real-time data processing pipelines
- **Batch Processing Integration**: Scheduled and ETL workflows
- **Microservices Integration**: Service-to-service communication
- **Plugin Architecture**: Extensible integration points

### Common Integration Scenarios

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

class IntegrationType(Enum):
    API_WEBHOOK = "api_webhook"
    MESSAGE_QUEUE = "message_queue"
    DATABASE_TRIGGER = "database_trigger"
    FILE_WATCHER = "file_watcher"
    STREAM_PROCESSOR = "stream_processor"
    SCHEDULED_BATCH = "scheduled_batch"

@dataclass
class IntegrationConfig:
    """Configuration for different integration types."""
    
    name: str
    integration_type: IntegrationType
    source_config: Dict[str, Any]
    processing_config: Dict[str, Any]
    destination_config: Dict[str, Any]
    error_handling: Dict[str, Any]
    monitoring: Dict[str, Any]

# Example configurations
INTEGRATION_CONFIGS = {
    'kafka_streaming': IntegrationConfig(
        name="Kafka Real-time Processing",
        integration_type=IntegrationType.STREAM_PROCESSOR,
        source_config={
            'bootstrap_servers': ['kafka1:9092', 'kafka2:9092'],
            'topic': 'sensor-data',
            'consumer_group': 'anomaly-detectors',
            'auto_offset_reset': 'latest'
        },
        processing_config={
            'algorithm': 'isolation_forest',
            'batch_size': 100,
            'window_size': 1000,
            'model_path': '/models/streaming_model.pkl'
        },
        destination_config={
            'output_topic': 'anomaly-alerts',
            'alert_webhook': 'https://alerts.company.com/webhook',
            'database_url': 'postgresql://user:pass@db:5432/alerts'
        },
        error_handling={
            'retry_attempts': 3,
            'dead_letter_topic': 'anomaly-errors',
            'circuit_breaker_threshold': 5
        },
        monitoring={
            'metrics_enabled': True,
            'health_check_interval': 30,
            'prometheus_port': 8080
        }
    ),
    
    'api_webhook': IntegrationConfig(
        name="API Webhook Integration",
        integration_type=IntegrationType.API_WEBHOOK,
        source_config={
            'webhook_url': '/api/v1/detect-anomaly',
            'authentication': 'bearer_token',
            'rate_limit': 1000,  # requests per minute
            'max_payload_size': '10MB'
        },
        processing_config={
            'algorithms': ['isolation_forest', 'local_outlier_factor'],
            'ensemble_method': 'voting',
            'confidence_threshold': 0.8
        },
        destination_config={
            'response_format': 'json',
            'include_explanations': True,
            'callback_url': None  # Optional callback
        },
        error_handling={
            'validation_enabled': True,
            'timeout_seconds': 30,
            'error_response_format': 'json'
        },
        monitoring={
            'request_logging': True,
            'performance_tracking': True,
            'alert_on_high_latency': True
        }
    )
}
```

## API Integration

### RESTful API Integration

```python
# integrations/api/fastapi_integration.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime

from anomaly_detection.core.application.services.ml_lifecycle_service import MLLifecycleService
from anomaly_detection.core.domain.entities.model import Model
from anomaly_detection.infrastructure.repositories.in_memory_model_repository import InMemoryModelRepository


class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    data: List[List[float]] = Field(..., description="Input data for anomaly detection")
    features: Optional[List[str]] = Field(None, description="Feature names")
    algorithm: str = Field("isolation_forest", description="Algorithm to use")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Algorithm parameters")
    include_explanations: bool = Field(False, description="Include explanations in response")
    model_id: Optional[str] = Field(None, description="Pre-trained model ID to use")

class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection."""
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]
    model_info: Dict[str, str]
    processing_time_ms: float
    timestamp: datetime
    explanations: Optional[List[Dict[str, Any]]] = None

class ModelTrainingRequest(BaseModel):
    """Request model for model training."""
    training_data: List[List[float]]
    features: Optional[List[str]] = None
    algorithm: str = "isolation_forest"
    parameters: Optional[Dict[str, Any]] = None
    model_name: str
    description: Optional[str] = None

class ModelTrainingResponse(BaseModel):
    """Response model for model training."""
    model_id: str
    status: str
    training_metrics: Dict[str, float]
    model_info: Dict[str, Any]
    timestamp: datetime

# FastAPI application
app = FastAPI(
    title="Anomaly Detection API",
    description="Production-ready anomaly detection service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()

# Service dependencies
ml_service = MLLifecycleService(InMemoryModelRepository())

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token."""
    token = credentials.credentials
    # Implement your token verification logic here
    if not token or len(token) < 10:  # Simplified validation
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return token

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time to response headers."""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.post("/api/v1/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Detect anomalies in provided data."""
    start_time = datetime.now()
    
    try:
        # Convert input data
        X = np.array(request.data)
        
        # Load or create model
        if request.model_id:
            # Use pre-trained model
            model = await ml_service.load_model(request.model_id)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
        else:
            # Create and train model on-the-fly
            model = await ml_service.create_model(
                name=f"temp_model_{start_time.timestamp()}",
                algorithm=request.algorithm,
                parameters=request.parameters or {}
            )
            await ml_service.train_model(model.id, X)
        
        # Perform anomaly detection
        predictions = await ml_service.predict_anomalies(model.id, X)
        
        # Format predictions
        formatted_predictions = []
        for i, (data_point, prediction, score) in enumerate(zip(X, predictions['predictions'], predictions['scores'])):
            formatted_predictions.append({
                'index': i,
                'data': data_point.tolist(),
                'is_anomaly': bool(prediction),
                'anomaly_score': float(score),
                'confidence': abs(float(score))  # Use absolute score as confidence
            })
        
        # Calculate summary statistics
        num_anomalies = sum(1 for p in formatted_predictions if p['is_anomaly'])
        summary = {
            'total_samples': len(formatted_predictions),
            'anomalies_detected': num_anomalies,
            'anomaly_rate': num_anomalies / len(formatted_predictions) if formatted_predictions else 0,
            'average_score': float(np.mean(predictions['scores'])),
            'min_score': float(np.min(predictions['scores'])),
            'max_score': float(np.max(predictions['scores']))
        }
        
        # Add explanations if requested
        explanations = None
        if request.include_explanations:
            # Generate explanations for anomalous points
            anomaly_indices = [i for i, p in enumerate(formatted_predictions) if p['is_anomaly']]
            explanations = await _generate_explanations(model, X, anomaly_indices)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log API usage in background
        background_tasks.add_task(
            _log_api_usage,
            endpoint="detect",
            user_token=token,
            processing_time_ms=processing_time,
            data_size=len(request.data),
            anomalies_found=num_anomalies
        )
        
        return AnomalyDetectionResponse(
            predictions=formatted_predictions,
            summary=summary,
            model_info={
                'model_id': model.id,
                'algorithm': model.algorithm,
                'version': model.version
            },
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            explanations=explanations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/v1/models/train", response_model=ModelTrainingResponse)
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Train a new anomaly detection model."""
    start_time = datetime.now()
    
    try:
        # Convert training data
        X_train = np.array(request.training_data)
        
        # Create model
        model = await ml_service.create_model(
            name=request.model_name,
            algorithm=request.algorithm,
            parameters=request.parameters or {},
            description=request.description
        )
        
        # Train model
        training_result = await ml_service.train_model(model.id, X_train)
        
        # Calculate training metrics
        training_metrics = {
            'training_samples': len(X_train),
            'features': X_train.shape[1],
            'training_time_seconds': training_result.get('training_time', 0),
            'model_size_bytes': training_result.get('model_size', 0)
        }
        
        # Log training activity
        background_tasks.add_task(
            _log_training_activity,
            model_id=model.id,
            user_token=token,
            training_samples=len(X_train),
            algorithm=request.algorithm
        )
        
        return ModelTrainingResponse(
            model_id=model.id,
            status="completed",
            training_metrics=training_metrics,
            model_info={
                'name': model.name,
                'algorithm': model.algorithm,
                'version': model.version,
                'created_at': model.created_at.isoformat()
            },
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/api/v1/models", response_model=List[Dict[str, Any]])
async def list_models(
    algorithm: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    token: str = Depends(verify_token)
):
    """List available models."""
    try:
        models = await ml_service.list_models(
            algorithm=algorithm,
            limit=limit,
            offset=offset
        )
        
        return [
            {
                'id': model.id,
                'name': model.name,
                'algorithm': model.algorithm,
                'version': model.version,
                'created_at': model.created_at.isoformat(),
                'updated_at': model.updated_at.isoformat(),
                'description': model.description
            }
            for model in models
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.delete("/api/v1/models/{model_id}")
async def delete_model(
    model_id: str,
    token: str = Depends(verify_token)
):
    """Delete a model."""
    try:
        success = await ml_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {"message": f"Model {model_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "anomaly-detection-api",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Implementation would return Prometheus-formatted metrics
    return {"metrics": "prometheus_formatted_metrics"}

async def _generate_explanations(model, X, anomaly_indices):
    """Generate explanations for anomalous points."""
    # Simplified explanation generation
    explanations = []
    for idx in anomaly_indices:
        explanations.append({
            'index': idx,
            'explanation': f"Sample {idx} deviates significantly from normal patterns",
            'confidence': 0.8,  # Placeholder
            'feature_importance': {}  # Would include actual feature importance
        })
    return explanations

async def _log_api_usage(endpoint, user_token, processing_time_ms, data_size, anomalies_found):
    """Log API usage for monitoring and analytics."""
    # Implementation would log to database or monitoring system
    print(f"API Usage: {endpoint}, user: {user_token[:8]}..., "
          f"time: {processing_time_ms}ms, data: {data_size}, anomalies: {anomalies_found}")

async def _log_training_activity(model_id, user_token, training_samples, algorithm):
    """Log model training activity."""
    # Implementation would log to database or monitoring system
    print(f"Model Training: {model_id}, user: {user_token[:8]}..., "
          f"samples: {training_samples}, algorithm: {algorithm}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### GraphQL Integration

```python
# integrations/api/graphql_integration.py
import strawberry
from strawberry.fastapi import GraphQLRouter
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime

from anomaly_detection.core.application.services.ml_lifecycle_service import MLLifecycleService


@strawberry.type
class AnomalyPrediction:
    index: int
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    data_point: List[float]

@strawberry.type
class DetectionResult:
    predictions: List[AnomalyPrediction]
    total_samples: int
    anomalies_detected: int
    anomaly_rate: float
    processing_time_ms: float
    timestamp: datetime

@strawberry.type
class ModelInfo:
    id: str
    name: str
    algorithm: str
    version: str
    created_at: datetime
    description: Optional[str] = None

@strawberry.input
class DetectionInput:
    data: List[List[float]]
    algorithm: str = "isolation_forest"
    parameters: Optional[str] = None  # JSON string
    model_id: Optional[str] = None

@strawberry.input
class TrainingInput:
    training_data: List[List[float]]
    model_name: str
    algorithm: str = "isolation_forest"
    parameters: Optional[str] = None  # JSON string
    description: Optional[str] = None

@strawberry.type
class Query:
    @strawberry.field
    async def models(
        self,
        algorithm: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ModelInfo]:
        """Get list of available models."""
        ml_service = MLLifecycleService()
        models = await ml_service.list_models(algorithm=algorithm, limit=limit, offset=offset)
        
        return [
            ModelInfo(
                id=model.id,
                name=model.name,
                algorithm=model.algorithm,
                version=model.version,
                created_at=model.created_at,
                description=model.description
            )
            for model in models
        ]
    
    @strawberry.field
    async def model(self, model_id: str) -> Optional[ModelInfo]:
        """Get specific model by ID."""
        ml_service = MLLifecycleService()
        model = await ml_service.get_model(model_id)
        
        if not model:
            return None
        
        return ModelInfo(
            id=model.id,
            name=model.name,
            algorithm=model.algorithm,
            version=model.version,
            created_at=model.created_at,
            description=model.description
        )

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def detect_anomalies(self, input: DetectionInput) -> DetectionResult:
        """Detect anomalies in provided data."""
        import json
        import numpy as np
        
        start_time = datetime.now()
        ml_service = MLLifecycleService()
        
        # Convert input data
        X = np.array(input.data)
        parameters = json.loads(input.parameters) if input.parameters else {}
        
        # Load or create model
        if input.model_id:
            model = await ml_service.load_model(input.model_id)
            if not model:
                raise Exception("Model not found")
        else:
            model = await ml_service.create_model(
                name=f"temp_model_{start_time.timestamp()}",
                algorithm=input.algorithm,
                parameters=parameters
            )
            await ml_service.train_model(model.id, X)
        
        # Perform detection
        predictions = await ml_service.predict_anomalies(model.id, X)
        
        # Format results
        formatted_predictions = [
            AnomalyPrediction(
                index=i,
                is_anomaly=bool(pred),
                anomaly_score=float(score),
                confidence=abs(float(score)),
                data_point=data_point.tolist()
            )
            for i, (data_point, pred, score) in enumerate(
                zip(X, predictions['predictions'], predictions['scores'])
            )
        ]
        
        num_anomalies = sum(1 for p in formatted_predictions if p.is_anomaly)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return DetectionResult(
            predictions=formatted_predictions,
            total_samples=len(formatted_predictions),
            anomalies_detected=num_anomalies,
            anomaly_rate=num_anomalies / len(formatted_predictions) if formatted_predictions else 0,
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )
    
    @strawberry.mutation
    async def train_model(self, input: TrainingInput) -> ModelInfo:
        """Train a new anomaly detection model."""
        import json
        import numpy as np
        
        ml_service = MLLifecycleService()
        
        # Convert input data
        X_train = np.array(input.training_data)
        parameters = json.loads(input.parameters) if input.parameters else {}
        
        # Create and train model
        model = await ml_service.create_model(
            name=input.model_name,
            algorithm=input.algorithm,
            parameters=parameters,
            description=input.description
        )
        
        await ml_service.train_model(model.id, X_train)
        
        return ModelInfo(
            id=model.id,
            name=model.name,
            algorithm=model.algorithm,
            version=model.version,
            created_at=model.created_at,
            description=model.description
        )
    
    @strawberry.mutation
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        ml_service = MLLifecycleService()
        return await ml_service.delete_model(model_id)

# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Create GraphQL router for FastAPI
graphql_app = GraphQLRouter(schema)

# Usage example queries:
"""
# Query to list models
query {
  models(algorithm: "isolation_forest", limit: 10) {
    id
    name
    algorithm
    createdAt
  }
}

# Mutation to detect anomalies
mutation {
  detectAnomalies(input: {
    data: [[1.0, 2.0], [1.1, 2.1], [10.0, 20.0]]
    algorithm: "isolation_forest"
  }) {
    predictions {
      index
      isAnomaly
      anomalyScore
      confidence
    }
    totalSamples
    anomaliesDetected
    anomalyRate
  }
}
"""
```

## Database Integration

### SQL Database Integration

```python
# integrations/database/sql_integration.py
import asyncio
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

from anomaly_detection.core.application.services.ml_lifecycle_service import MLLifecycleService


class PostgreSQLAnomalyDetector:
    """PostgreSQL integration for anomaly detection."""
    
    def __init__(self, connection_string: str, ml_service: MLLifecycleService):
        self.connection_string = connection_string
        self.ml_service = ml_service
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        
        # Create necessary tables
        await self.create_tables()
    
    async def create_tables(self):
        """Create tables for anomaly detection results."""
        async with self.pool.acquire() as conn:
            # Create anomaly detection results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_detection_results (
                    id SERIAL PRIMARY KEY,
                    source_table VARCHAR(255) NOT NULL,
                    source_record_id INTEGER,
                    detection_timestamp TIMESTAMP DEFAULT NOW(),
                    is_anomaly BOOLEAN NOT NULL,
                    anomaly_score FLOAT NOT NULL,
                    confidence FLOAT NOT NULL,
                    model_id VARCHAR(255) NOT NULL,
                    algorithm VARCHAR(100) NOT NULL,
                    features JSONB,
                    raw_data JSONB,
                    metadata JSONB DEFAULT '{}'::jsonb
                );
            """)
            
            # Create indexes for better performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_anomaly_results_timestamp 
                ON anomaly_detection_results(detection_timestamp);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_anomaly_results_source 
                ON anomaly_detection_results(source_table, source_record_id);
            """)
            
            # Create model registry table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    algorithm VARCHAR(100) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    source_table VARCHAR(255),
                    feature_columns TEXT[],
                    parameters JSONB DEFAULT '{}'::jsonb,
                    performance_metrics JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE
                );
            """)
            
            # Create scheduled jobs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_detection_jobs (
                    id SERIAL PRIMARY KEY,
                    job_name VARCHAR(255) NOT NULL UNIQUE,
                    source_table VARCHAR(255) NOT NULL,
                    feature_columns TEXT[] NOT NULL,
                    model_id VARCHAR(255) REFERENCES model_registry(model_id),
                    schedule_cron VARCHAR(100),
                    last_run TIMESTAMP,
                    next_run TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    configuration JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
    
    async def detect_anomalies_in_table(
        self,
        table_name: str,
        feature_columns: List[str],
        model_id: str,
        where_clause: Optional[str] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Detect anomalies in a database table."""
        
        results = {
            'processed_records': 0,
            'anomalies_found': 0,
            'processing_time_seconds': 0,
            'batch_results': []
        }
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            # Build query
            feature_cols = ", ".join(feature_columns)
            base_query = f"SELECT id, {feature_cols} FROM {table_name}"
            
            if where_clause:
                base_query += f" WHERE {where_clause}"
            
            # Get total count
            count_query = f"SELECT COUNT(*) FROM ({base_query}) as subq"
            total_records = await conn.fetchval(count_query)
            
            # Process in batches
            offset = 0
            while offset < total_records:
                batch_query = f"{base_query} LIMIT {batch_size} OFFSET {offset}"
                rows = await conn.fetch(batch_query)
                
                if not rows:
                    break
                
                # Extract data and IDs
                record_ids = [row['id'] for row in rows]
                feature_data = np.array([
                    [row[col] for col in feature_columns] 
                    for row in rows
                ])
                
                # Perform anomaly detection
                predictions = await self.ml_service.predict_anomalies(
                    model_id, feature_data
                )
                
                # Store results
                batch_anomalies = 0
                anomaly_records = []
                
                for i, (record_id, is_anomaly, score) in enumerate(zip(
                    record_ids, predictions['predictions'], predictions['scores']
                )):
                    if is_anomaly:
                        batch_anomalies += 1
                    
                    anomaly_records.append({
                        'source_table': table_name,
                        'source_record_id': record_id,
                        'is_anomaly': bool(is_anomaly),
                        'anomaly_score': float(score),
                        'confidence': abs(float(score)),
                        'model_id': model_id,
                        'algorithm': 'isolation_forest',  # Get from model
                        'features': dict(zip(feature_columns, feature_data[i].tolist())),
                        'raw_data': dict(rows[i])
                    })
                
                # Bulk insert results
                await self.bulk_insert_results(conn, anomaly_records)
                
                results['processed_records'] += len(rows)
                results['anomalies_found'] += batch_anomalies
                results['batch_results'].append({
                    'batch_number': offset // batch_size + 1,
                    'records_processed': len(rows),
                    'anomalies_found': batch_anomalies,
                    'anomaly_rate': batch_anomalies / len(rows)
                })
                
                offset += batch_size
        
        results['processing_time_seconds'] = (datetime.now() - start_time).total_seconds()
        results['overall_anomaly_rate'] = (
            results['anomalies_found'] / results['processed_records']
            if results['processed_records'] > 0 else 0
        )
        
        return results
    
    async def bulk_insert_results(self, conn, anomaly_records: List[Dict]):
        """Bulk insert anomaly detection results."""
        if not anomaly_records:
            return
        
        columns = [
            'source_table', 'source_record_id', 'is_anomaly', 'anomaly_score',
            'confidence', 'model_id', 'algorithm', 'features', 'raw_data'
        ]
        
        values = [
            (
                record['source_table'],
                record['source_record_id'],
                record['is_anomaly'],
                record['anomaly_score'],
                record['confidence'],
                record['model_id'],
                record['algorithm'],
                json.dumps(record['features']),
                json.dumps(record['raw_data'])
            )
            for record in anomaly_records
        ]
        
        await conn.executemany(
            """
            INSERT INTO anomaly_detection_results 
            (source_table, source_record_id, is_anomaly, anomaly_score, 
             confidence, model_id, algorithm, features, raw_data)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            values
        )
    
    async def train_model_from_table(
        self,
        table_name: str,
        feature_columns: List[str],
        model_name: str,
        algorithm: str = "isolation_forest",
        parameters: Optional[Dict] = None,
        where_clause: Optional[str] = None
    ) -> str:
        """Train a model from database table data."""
        
        async with self.pool.acquire() as conn:
            # Build query
            feature_cols = ", ".join(feature_columns)
            query = f"SELECT {feature_cols} FROM {table_name}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
            
            # Fetch training data
            rows = await conn.fetch(query)
            
            if not rows:
                raise ValueError("No training data found")
            
            # Convert to numpy array
            training_data = np.array([
                [row[col] for col in feature_columns] 
                for row in rows
            ])
        
        # Create and train model
        model = await self.ml_service.create_model(
            name=model_name,
            algorithm=algorithm,
            parameters=parameters or {}
        )
        
        await self.ml_service.train_model(model.id, training_data)
        
        # Register model in database
        await self.register_model(
            model_id=model.id,
            name=model_name,
            algorithm=algorithm,
            source_table=table_name,
            feature_columns=feature_columns,
            parameters=parameters or {}
        )
        
        return model.id
    
    async def register_model(
        self,
        model_id: str,
        name: str,
        algorithm: str,
        source_table: str,
        feature_columns: List[str],
        parameters: Dict,
        performance_metrics: Optional[Dict] = None
    ):
        """Register model in the database."""
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_registry 
                (model_id, name, algorithm, source_table, feature_columns, 
                 parameters, performance_metrics, version)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (model_id) DO UPDATE SET
                    updated_at = NOW(),
                    is_active = TRUE
                """,
                model_id, name, algorithm, source_table, feature_columns,
                json.dumps(parameters), json.dumps(performance_metrics or {}), "1.0"
            )
    
    async def schedule_anomaly_detection(
        self,
        job_name: str,
        table_name: str,
        feature_columns: List[str],
        model_id: str,
        cron_schedule: str,
        configuration: Optional[Dict] = None
    ):
        """Schedule recurring anomaly detection job."""
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO anomaly_detection_jobs 
                (job_name, source_table, feature_columns, model_id, 
                 schedule_cron, configuration)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (job_name) DO UPDATE SET
                    source_table = EXCLUDED.source_table,
                    feature_columns = EXCLUDED.feature_columns,
                    model_id = EXCLUDED.model_id,
                    schedule_cron = EXCLUDED.schedule_cron,
                    configuration = EXCLUDED.configuration,
                    is_active = TRUE
                """,
                job_name, table_name, feature_columns, model_id,
                cron_schedule, json.dumps(configuration or {})
            )
    
    async def get_anomaly_summary(
        self,
        table_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get anomaly detection summary statistics."""
        
        conditions = []
        params = []
        param_count = 0
        
        if table_name:
            param_count += 1
            conditions.append(f"source_table = ${param_count}")
            params.append(table_name)
        
        if start_date:
            param_count += 1
            conditions.append(f"detection_timestamp >= ${param_count}")
            params.append(start_date)
        
        if end_date:
            param_count += 1
            conditions.append(f"detection_timestamp <= ${param_count}")
            params.append(end_date)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        async with self.pool.acquire() as conn:
            # Get summary statistics
            summary_query = f"""
                SELECT 
                    COUNT(*) as total_records,
                    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as total_anomalies,
                    AVG(CASE WHEN is_anomaly THEN 1.0 ELSE 0.0 END) as anomaly_rate,
                    AVG(anomaly_score) as avg_anomaly_score,
                    AVG(confidence) as avg_confidence,
                    MIN(detection_timestamp) as earliest_detection,
                    MAX(detection_timestamp) as latest_detection
                FROM anomaly_detection_results
                WHERE {where_clause}
            """
            
            summary = await conn.fetchrow(summary_query, *params)
            
            # Get breakdown by table
            breakdown_query = f"""
                SELECT 
                    source_table,
                    COUNT(*) as records,
                    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomalies,
                    AVG(CASE WHEN is_anomaly THEN 1.0 ELSE 0.0 END) as anomaly_rate
                FROM anomaly_detection_results
                WHERE {where_clause}
                GROUP BY source_table
                ORDER BY anomaly_rate DESC
            """
            
            breakdown = await conn.fetch(breakdown_query, *params)
            
            return {
                'summary': dict(summary),
                'breakdown_by_table': [dict(row) for row in breakdown]
            }
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()

# Usage example
async def main():
    # Initialize
    ml_service = MLLifecycleService()
    detector = PostgreSQLAnomalyDetector(
        "postgresql://user:password@localhost/database",
        ml_service
    )
    
    await detector.initialize()
    
    try:
        # Train model from existing data
        model_id = await detector.train_model_from_table(
            table_name="sensor_readings",
            feature_columns=["temperature", "humidity", "pressure"],
            model_name="sensor_anomaly_detector",
            algorithm="isolation_forest",
            where_clause="created_at >= NOW() - INTERVAL '30 days'"
        )
        
        # Schedule regular anomaly detection
        await detector.schedule_anomaly_detection(
            job_name="daily_sensor_check",
            table_name="sensor_readings",
            feature_columns=["temperature", "humidity", "pressure"],
            model_id=model_id,
            cron_schedule="0 2 * * *",  # Daily at 2 AM
            configuration={
                "where_clause": "created_at >= NOW() - INTERVAL '1 day'",
                "batch_size": 5000
            }
        )
        
        # Run immediate detection
        results = await detector.detect_anomalies_in_table(
            table_name="sensor_readings",
            feature_columns=["temperature", "humidity", "pressure"],
            model_id=model_id,
            where_clause="created_at >= NOW() - INTERVAL '1 hour'"
        )
        
        print(f"Processed {results['processed_records']} records")
        print(f"Found {results['anomalies_found']} anomalies")
        print(f"Anomaly rate: {results['overall_anomaly_rate']:.2%}")
        
        # Get summary statistics
        summary = await detector.get_anomaly_summary(
            table_name="sensor_readings",
            start_date=datetime.now() - timedelta(days=7)
        )
        
        print("Summary:", summary)
        
    finally:
        await detector.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Message Queue Integration

### Apache Kafka Integration

```python
# integrations/messaging/kafka_integration.py
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
import aiokafka

from anomaly_detection.core.application.services.ml_lifecycle_service import MLLifecycleService


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: str
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_ca_location: Optional[str] = None
    client_id: str = "anomaly-detector"


class KafkaAnomalyProcessor:
    """Kafka-based real-time anomaly detection processor."""
    
    def __init__(
        self,
        kafka_config: KafkaConfig,
        ml_service: MLLifecycleService,
        input_topic: str,
        output_topic: str,
        error_topic: str,
        consumer_group: str = "anomaly-detection-group"
    ):
        self.kafka_config = kafka_config
        self.ml_service = ml_service
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.error_topic = error_topic
        self.consumer_group = consumer_group
        
        self.consumer = None
        self.producer = None
        self.admin_client = None
        self.running = False
        
        # Processing statistics
        self.stats = {
            'messages_processed': 0,
            'anomalies_detected': 0,
            'errors': 0,
            'start_time': None,
            'last_processed': None
        }
        
        # Model cache
        self.model_cache = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize Kafka components."""
        # Create admin client
        admin_config = {
            'bootstrap.servers': self.kafka_config.bootstrap_servers,
            'security.protocol': self.kafka_config.security_protocol,
            'client.id': f"{self.kafka_config.client_id}-admin"
        }
        
        if self.kafka_config.sasl_mechanism:
            admin_config.update({
                'sasl.mechanism': self.kafka_config.sasl_mechanism,
                'sasl.username': self.kafka_config.sasl_username,
                'sasl.password': self.kafka_config.sasl_password
            })
        
        self.admin_client = AdminClient(admin_config)
        
        # Create topics if they don't exist
        await self.create_topics()
        
        # Create consumer
        consumer_config = {
            'bootstrap.servers': self.kafka_config.bootstrap_servers,
            'group.id': self.consumer_group,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': False,
            'security.protocol': self.kafka_config.security_protocol,
            'client.id': f"{self.kafka_config.client_id}-consumer"
        }
        
        if self.kafka_config.sasl_mechanism:
            consumer_config.update({
                'sasl.mechanism': self.kafka_config.sasl_mechanism,
                'sasl.username': self.kafka_config.sasl_username,
                'sasl.password': self.kafka_config.sasl_password
            })
        
        self.consumer = Consumer(consumer_config)
        
        # Create producer
        producer_config = {
            'bootstrap.servers': self.kafka_config.bootstrap_servers,
            'security.protocol': self.kafka_config.security_protocol,
            'client.id': f"{self.kafka_config.client_id}-producer",
            'acks': 'all',
            'retries': 3,
            'retry.backoff.ms': 1000
        }
        
        if self.kafka_config.sasl_mechanism:
            producer_config.update({
                'sasl.mechanism': self.kafka_config.sasl_mechanism,
                'sasl.username': self.kafka_config.sasl_username,
                'sasl.password': self.kafka_config.sasl_password
            })
        
        self.producer = Producer(producer_config)
        
        self.logger.info("Kafka components initialized successfully")
    
    async def create_topics(self):
        """Create required Kafka topics."""
        topics = [
            NewTopic(self.input_topic, num_partitions=3, replication_factor=1),
            NewTopic(self.output_topic, num_partitions=3, replication_factor=1),
            NewTopic(self.error_topic, num_partitions=1, replication_factor=1)
        ]
        
        # Create topics
        fs = self.admin_client.create_topics(topics, validate_only=False)
        
        # Wait for topics to be created
        for topic, f in fs.items():
            try:
                f.result()  # The result itself is None
                self.logger.info(f"Topic {topic} created successfully")
            except Exception as e:
                if "already exists" in str(e):
                    self.logger.info(f"Topic {topic} already exists")
                else:
                    self.logger.error(f"Failed to create topic {topic}: {e}")
    
    async def start_processing(self, model_id: str, feature_columns: List[str]):
        """Start processing messages from Kafka."""
        if self.running:
            self.logger.warning("Processor is already running")
            return
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        # Subscribe to input topic
        self.consumer.subscribe([self.input_topic])
        
        self.logger.info(f"Started processing messages from topic: {self.input_topic}")
        
        try:
            while self.running:
                # Poll for messages
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        self.logger.error(f"Consumer error: {msg.error()}")
                        continue
                
                # Process message
                await self.process_message(msg, model_id, feature_columns)
                
                # Commit offset
                self.consumer.commit(msg)
                
        except KeyboardInterrupt:
            self.logger.info("Stopping processor due to keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in processing loop: {e}")
        finally:
            await self.stop_processing()
    
    async def process_message(self, msg, model_id: str, feature_columns: List[str]):
        """Process individual Kafka message."""
        try:
            # Parse message
            message_data = json.loads(msg.value().decode('utf-8'))
            
            # Extract features
            features = []
            for col in feature_columns:
                if col not in message_data:
                    raise ValueError(f"Missing feature column: {col}")
                features.append(float(message_data[col]))
            
            # Convert to numpy array
            X = np.array([features])
            
            # Get model (use cache for performance)
            if model_id not in self.model_cache:
                model = await self.ml_service.load_model(model_id)
                if not model:
                    raise ValueError(f"Model not found: {model_id}")
                self.model_cache[model_id] = model
            
            # Perform anomaly detection
            predictions = await self.ml_service.predict_anomalies(model_id, X)
            
            # Create result message
            result = {
                'message_id': message_data.get('id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'original_data': message_data,
                'features': dict(zip(feature_columns, features)),
                'is_anomaly': bool(predictions['predictions'][0]),
                'anomaly_score': float(predictions['scores'][0]),
                'confidence': abs(float(predictions['scores'][0])),
                'model_id': model_id,
                'processing_metadata': {
                    'kafka_topic': msg.topic(),
                    'kafka_partition': msg.partition(),
                    'kafka_offset': msg.offset(),
                    'processing_time': datetime.now().isoformat()
                }
            }
            
            # Send result to output topic
            await self.send_result(result)
            
            # Update statistics
            self.stats['messages_processed'] += 1
            self.stats['last_processed'] = datetime.now()
            
            if result['is_anomaly']:
                self.stats['anomalies_detected'] += 1
                # Send to alert topic for high-confidence anomalies
                if result['confidence'] > 0.8:
                    await self.send_alert(result)
            
            # Log every 100 messages
            if self.stats['messages_processed'] % 100 == 0:
                self.logger.info(f"Processed {self.stats['messages_processed']} messages, "
                               f"found {self.stats['anomalies_detected']} anomalies")
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            await self.send_error(msg, str(e))
            self.stats['errors'] += 1
    
    async def send_result(self, result: Dict[str, Any]):
        """Send result to output topic."""
        try:
            message = json.dumps(result, default=str)
            self.producer.produce(
                topic=self.output_topic,
                key=result.get('message_id', '').encode('utf-8'),
                value=message.encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)  # Trigger delivery reports
            
        except Exception as e:
            self.logger.error(f"Error sending result: {e}")
    
    async def send_alert(self, result: Dict[str, Any]):
        """Send high-confidence anomaly alert."""
        alert_topic = f"{self.output_topic}-alerts"
        
        alert = {
            'alert_type': 'anomaly_detected',
            'severity': 'high' if result['confidence'] > 0.9 else 'medium',
            'timestamp': datetime.now().isoformat(),
            'anomaly_data': result,
            'alert_id': f"anomaly_{result.get('message_id')}_{int(datetime.now().timestamp())}"
        }
        
        try:
            message = json.dumps(alert, default=str)
            self.producer.produce(
                topic=alert_topic,
                key=alert['alert_id'].encode('utf-8'),
                value=message.encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    async def send_error(self, original_msg, error_message: str):
        """Send error message to error topic."""
        error = {
            'error_type': 'processing_error',
            'timestamp': datetime.now().isoformat(),
            'error_message': error_message,
            'original_message': {
                'topic': original_msg.topic(),
                'partition': original_msg.partition(),
                'offset': original_msg.offset(),
                'value': original_msg.value().decode('utf-8', errors='replace')
            }
        }
        
        try:
            message = json.dumps(error, default=str)
            self.producer.produce(
                topic=self.error_topic,
                value=message.encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
            
        except Exception as e:
            self.logger.error(f"Error sending error message: {e}")
    
    def delivery_report(self, err, msg):
        """Kafka delivery report callback."""
        if err is not None:
            self.logger.error(f"Message delivery failed: {err}")
        # Don't log successful deliveries to avoid spam
    
    async def stop_processing(self):
        """Stop processing messages."""
        self.running = False
        
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.flush()
            
        self.logger.info("Processor stopped")
        
        # Log final statistics
        if self.stats['start_time']:
            duration = datetime.now() - self.stats['start_time']
            self.logger.info(f"Processing statistics:")
            self.logger.info(f"  Duration: {duration}")
            self.logger.info(f"  Messages processed: {self.stats['messages_processed']}")
            self.logger.info(f"  Anomalies detected: {self.stats['anomalies_detected']}")
            self.logger.info(f"  Errors: {self.stats['errors']}")
            if self.stats['messages_processed'] > 0:
                rate = self.stats['messages_processed'] / duration.total_seconds()
                anomaly_rate = self.stats['anomalies_detected'] / self.stats['messages_processed']
                self.logger.info(f"  Processing rate: {rate:.2f} msg/sec")
                self.logger.info(f"  Anomaly rate: {anomaly_rate:.2%}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.stats.copy()
        
        if stats['start_time']:
            stats['uptime_seconds'] = (datetime.now() - stats['start_time']).total_seconds()
            
            if stats['messages_processed'] > 0:
                stats['processing_rate'] = stats['messages_processed'] / stats['uptime_seconds']
                stats['anomaly_rate'] = stats['anomalies_detected'] / stats['messages_processed']
                stats['error_rate'] = stats['errors'] / stats['messages_processed']
        
        return stats

# Usage example
async def main():
    # Configuration
    kafka_config = KafkaConfig(
        bootstrap_servers="localhost:9092",
        client_id="anomaly-detector-demo"
    )
    
    # Initialize services
    ml_service = MLLifecycleService()
    processor = KafkaAnomalyProcessor(
        kafka_config=kafka_config,
        ml_service=ml_service,
        input_topic="sensor-data",
        output_topic="anomaly-results",
        error_topic="anomaly-errors",
        consumer_group="anomaly-detection-group"
    )
    
    # Initialize processor
    await processor.initialize()
    
    # Train or load model
    # model_id = await ml_service.create_model(...)
    model_id = "existing_model_id"
    feature_columns = ["temperature", "humidity", "pressure", "vibration"]
    
    try:
        # Start processing
        await processor.start_processing(model_id, feature_columns)
    except KeyboardInterrupt:
        print("Stopping processor...")
    finally:
        await processor.stop_processing()
        
        # Print final statistics
        stats = processor.get_statistics()
        print(f"Final statistics: {stats}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

This comprehensive integration guide provides practical examples for integrating the anomaly detection package with various systems and platforms. The examples include error handling, monitoring, and production-ready patterns that can be adapted to specific use cases and environments.
#!/usr/bin/env python3
"""
Simple Web API implementation for Pynomaly anomaly detection.

This module provides a minimal FastAPI-based web API for anomaly detection.
"""

from typing import Any, Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import json
from io import StringIO

try:
    from pynomaly_detection import AnomalyDetector, __version__
except ImportError:
    # Fallback for development
    import sys
    from pathlib import Path
    package_root = Path(__file__).parent.parent
    sys.path.insert(0, str(package_root))
    from pynomaly_detection import AnomalyDetector, __version__

# Pydantic models for request/response
class DetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    data: List[List[float]]
    algorithm: str = "isolation_forest"
    contamination: float = 0.1

class DetectionResponse(BaseModel):
    """Response model for anomaly detection."""
    predictions: List[int]
    anomaly_count: int
    total_count: int
    anomaly_rate: float
    algorithm: str

class TrainingRequest(BaseModel):
    """Request model for training."""
    data: List[List[float]]
    algorithm: str = "isolation_forest"
    contamination: float = 0.1

class TrainingResponse(BaseModel):
    """Response model for training."""
    status: str
    message: str

class VersionResponse(BaseModel):
    """Response model for version information."""
    version: str
    api_version: str
    description: str

class AlgorithmInfo(BaseModel):
    """Information about an algorithm."""
    name: str
    description: str
    status: str

class AlgorithmsResponse(BaseModel):
    """Response model for available algorithms."""
    algorithms: List[AlgorithmInfo]

# Create FastAPI app
app = FastAPI(
    title="Pynomaly Web API",
    version=__version__,
    description="Simple Web API for anomaly detection using Pynomaly",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance (in production, this would be session-based)
detector = None

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Pynomaly Web API",
        "version": __version__,
        "api_version": "v1",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}

@app.get("/version", response_model=VersionResponse)
async def get_version():
    """Get version information."""
    return VersionResponse(
        version=__version__,
        api_version="v1",
        description="Pynomaly - Advanced Anomaly Detection Platform"
    )

@app.get("/algorithms", response_model=AlgorithmsResponse)
async def get_algorithms():
    """Get available algorithms."""
    algorithms = [
        AlgorithmInfo(
            name="isolation_forest",
            description="Isolation Forest (default)",
            status="✓ Working"
        ),
        AlgorithmInfo(
            name="lof",
            description="Local Outlier Factor",
            status="⚠ Requires sklearn"
        ),
        AlgorithmInfo(
            name="ocsvm",
            description="One-Class SVM",
            status="⚠ Requires sklearn"
        ),
        AlgorithmInfo(
            name="autoencoder",
            description="Neural Network Autoencoder",
            status="✗ Not implemented"
        ),
        AlgorithmInfo(
            name="ensemble",
            description="Ensemble Methods",
            status="✗ Not implemented"
        )
    ]
    return AlgorithmsResponse(algorithms=algorithms)

@app.post("/detect", response_model=DetectionResponse)
async def detect_anomalies(request: DetectionRequest):
    """Detect anomalies in the provided data."""
    try:
        # Convert data to numpy array
        data = np.array(request.data)
        
        # Create detector
        detector = AnomalyDetector()
        
        # Detect anomalies
        predictions = detector.detect(
            data,
            algorithm=request.algorithm,
            contamination=request.contamination
        )
        
        # Calculate metrics
        anomaly_count = int(np.sum(predictions))
        total_count = len(predictions)
        anomaly_rate = anomaly_count / total_count
        
        return DetectionResponse(
            predictions=predictions.tolist(),
            anomaly_count=anomaly_count,
            total_count=total_count,
            anomaly_rate=anomaly_rate,
            algorithm=request.algorithm
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train_detector(request: TrainingRequest):
    """Train an anomaly detector."""
    try:
        # Convert data to numpy array
        data = np.array(request.data)
        
        # Create and train detector
        global detector
        detector = AnomalyDetector()
        detector.fit(data, contamination=request.contamination)
        
        return TrainingResponse(
            status="success",
            message=f"Detector trained successfully with {request.algorithm}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=DetectionResponse)
async def predict_anomalies(request: DetectionRequest):
    """Predict anomalies using a trained detector."""
    try:
        global detector
        if detector is None:
            raise HTTPException(status_code=400, detail="Detector not trained. Call /train first.")
        
        # Convert data to numpy array
        data = np.array(request.data)
        
        # Predict anomalies
        predictions = detector.predict(data)
        
        # Calculate metrics
        anomaly_count = int(np.sum(predictions))
        total_count = len(predictions)
        anomaly_rate = anomaly_count / total_count
        
        return DetectionResponse(
            predictions=predictions.tolist(),
            anomaly_count=anomaly_count,
            total_count=total_count,
            anomaly_rate=anomaly_rate,
            algorithm=request.algorithm
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a data file for processing."""
    try:
        contents = await file.read()
        
        # Parse based on file type
        if file.filename.endswith('.csv'):
            # Parse CSV
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
            data = df.select_dtypes(include=[np.number]).values
        elif file.filename.endswith('.json'):
            # Parse JSON
            json_data = json.loads(contents.decode('utf-8'))
            data = np.array(json_data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        return {
            "message": "File uploaded successfully",
            "shape": data.shape,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run the API server."""
    uvicorn.run(
        "pynomaly_detection.infrastructure.web_api:app",
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    run_api()
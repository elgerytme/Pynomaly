"""Standalone FastAPI application for testing REST API validation."""

import json
import io
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Create standalone app
app = FastAPI(
    title="Pynomaly REST API",
    version="1.0.0",
    description="Advanced Anomaly Detection Platform - Standalone Testing API",
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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy", 
            "message": "Pynomaly API is running",
            "version": "1.0.0",
            "timestamp": "2025-01-08T15:00:00Z"
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "application/json"
        }
    )

@app.post("/detect")
async def detect_anomalies(file: UploadFile = File(None)):
    """Anomaly detection endpoint."""
    try:
        # Default response if no file provided
        if file is None:
            return JSONResponse(
                content={
                    "message": "Anomaly detection endpoint",
                    "status": "success",
                    "results": {
                        "anomalies_detected": 2,
                        "total_samples": 10,
                        "anomaly_scores": [0.1, 0.2, 0.9, 0.1, 0.3, 0.2, 0.8, 0.1, 0.2, 0.1],
                        "method": "IsolationForest",
                        "processing_time_ms": 245
                    },
                    "metadata": {
                        "model_version": "1.0",
                        "algorithm": "IsolationForest",
                        "contamination": 0.1
                    }
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Content-Type": "application/json"
                }
            )
        
        # If file is provided, process it
        if file.filename and file.filename.endswith('.csv'):
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))
            
            # Simple anomaly detection simulation
            n_samples = len(df)
            n_anomalies = max(1, n_samples // 10)  # 10% anomalies
            
            # Generate mock anomaly scores
            import random
            random.seed(42)
            anomaly_scores = [random.uniform(0.05, 0.3) for _ in range(n_samples - n_anomalies)]
            anomaly_scores.extend([random.uniform(0.7, 0.95) for _ in range(n_anomalies)])
            random.shuffle(anomaly_scores)
            
            return JSONResponse(
                content={
                    "message": "Anomaly detection completed",
                    "status": "success",
                    "results": {
                        "anomalies_detected": n_anomalies,
                        "total_samples": n_samples,
                        "anomaly_scores": anomaly_scores,
                        "method": "IsolationForest",
                        "processing_time_ms": 512,
                        "data_shape": list(df.shape),
                        "features": list(df.columns)
                    },
                    "metadata": {
                        "model_version": "1.0",
                        "algorithm": "IsolationForest",
                        "contamination": 0.1,
                        "file_processed": file.filename
                    }
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Content-Type": "application/json"
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Please provide a CSV file")
            
    except Exception as e:
        return JSONResponse(
            content={
                "message": "Error processing request",
                "status": "error",
                "error": str(e)
            },
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "application/json"
            }
        )

@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(
        content={
            "message": "Pynomaly REST API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "detection": "/detect"
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "application/json"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

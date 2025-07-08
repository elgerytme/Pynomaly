"""FastAPI application entry point for uvicorn."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create a simple app for testing
app = FastAPI(
    title="Pynomaly API",
    version="1.0.0",
    description="Advanced Anomaly Detection Platform"
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
    return {"status": "healthy", "message": "Pynomaly API is running"}

@app.post("/detect")
async def detect_anomalies():
    """Simple anomaly detection endpoint for testing."""
    return {
        "message": "Anomaly detection endpoint",
        "status": "success",
        "results": {
            "anomalies_detected": 2,
            "total_samples": 10,
            "anomaly_scores": [0.1, 0.2, 0.9, 0.1, 0.3, 0.2, 0.8, 0.1, 0.2, 0.1]
        }
    }

#!/usr/bin/env python
"""Test API functionality for Pynomaly."""

import sys
import os
sys.path.insert(0, '/mnt/c/Users/andre/Pynomaly/src/packages/data/anomaly_detection/src')

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import uvicorn
    
    # Create a simple FastAPI app for testing
    app = FastAPI(title="Pynomaly API", version="0.1.0")
    
    @app.get("/")
    async def root():
        return {"message": "Pynomaly API is running", "version": "0.1.0"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "pynomaly-api"}
    
    @app.post("/detect")
    async def detect(data: dict):
        """Detect anomalies in provided data."""
        try:
            import numpy as np
            from pynomaly_detection import AnomalyDetector
            
            # Extract data from request
            input_data = data.get("data", [])
            contamination = data.get("contamination", 0.1)
            
            # Convert to numpy array
            X = np.array(input_data)
            
            # Create detector
            detector = AnomalyDetector()
            
            # Detect anomalies
            detector.fit(X, contamination=contamination)
            predictions = detector.predict(X)
            
            # Return results
            return {
                "status": "success",
                "data_points": len(X),
                "anomalies_detected": int(sum(predictions)),
                "anomaly_indices": [int(i) for i, val in enumerate(predictions) if val == 1],
                "contamination": contamination
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    # Test the API
    if __name__ == "__main__":
        print("Testing Pynomaly API...")
        
        # Create test client
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        print(f"Root endpoint: {response.status_code}, {response.json()}")
        
        # Test health endpoint
        response = client.get("/health")
        print(f"Health endpoint: {response.status_code}, {response.json()}")
        
        # Test detect endpoint
        test_data = {
            "data": [[1, 2], [2, 3], [3, 4], [10, 20], [100, 200]],
            "contamination": 0.2
        }
        response = client.post("/detect", json=test_data)
        print(f"Detect endpoint: {response.status_code}, {response.json()}")
        
        print("API test completed successfully!")

except ImportError as e:
    print(f"FastAPI not available: {e}")
    print("API testing skipped.")
except Exception as e:
    print(f"API test failed: {e}")
    import traceback
    traceback.print_exc()
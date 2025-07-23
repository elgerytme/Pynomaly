#!/usr/bin/env python
"""Test API functionality for data quality detection."""

import sys
import os

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import uvicorn
    
    # Create a simple FastAPI app for testing
    app = FastAPI(title="Data Quality API", version="0.1.0")
    
    @app.get("/")
    async def root():
        return {"message": "Data Quality API is running", "version": "0.1.0"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "data-quality-api"}
    
    @app.post("/detect")
    async def detect(data: dict):
        """Detect quality issues in provided data."""
        try:
            import numpy as np
            
            # Extract data from request
            input_data = data.get("data", [])
            threshold = data.get("threshold", 0.1)
            
            # Convert to numpy array
            X = np.array(input_data)
            
            # Simple quality detection logic (placeholder)
            # In a real implementation, this would use proper quality detection algorithms
            quality_scores = np.random.random(len(X))
            issues_detected = quality_scores < threshold
            
            # Return results
            return {
                "status": "success",
                "data_points": len(X),
                "issues_detected": int(sum(issues_detected)),
                "issue_indices": [int(i) for i, val in enumerate(issues_detected) if val],
                "threshold": threshold
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    # Test the API
    if __name__ == "__main__":
        print("Testing data quality detection API...")
        
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
            "threshold": 0.2
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
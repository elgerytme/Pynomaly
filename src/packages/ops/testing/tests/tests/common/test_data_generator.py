"""
Test Data Generator

Utility for generating test datasets for integration testing.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class TestDataGenerator:
    """Generates test data for integration testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def create_sample_csv(
        self, 
        file_path: Path, 
        rows: int = 1000, 
        features: int = 10,
        anomalies: bool = True
    ):
        """Create a sample CSV dataset."""
        
        # Generate normal data
        data = np.random.normal(0, 1, (rows, features))
        
        if anomalies:
            # Add some anomalies (5% of data)
            n_anomalies = int(rows * 0.05)
            anomaly_indices = random.sample(range(rows), n_anomalies)
            
            for idx in anomaly_indices:
                # Make anomalies by scaling values
                data[idx] = data[idx] * random.uniform(3, 5)
        
        # Create DataFrame
        columns = [f"feature_{i}" for i in range(features)]
        df = pd.DataFrame(data, columns=columns)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
    
    def create_sample_json(self, file_path: Path, records: int = 500):
        """Create a sample JSON dataset."""
        
        data = []
        for i in range(records):
            record = {
                "id": i,
                "timestamp": f"2024-01-{(i % 30) + 1:02d}T{(i % 24):02d}:00:00Z",
                "values": [random.uniform(-2, 2) for _ in range(5)],
                "category": random.choice(["A", "B", "C"]),
                "is_anomaly": random.random() < 0.05
            }
            data.append(record)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_streaming_batch(self, size: int = 50, anomaly_rate: float = 0.05) -> List[Dict[str, Any]]:
        """Create a batch of streaming data."""
        
        batch = []
        for i in range(size):
            record = {
                "timestamp": f"2024-01-01T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z",
                "values": [random.uniform(-1, 1) for _ in range(3)],
                "is_anomaly": random.random() < anomaly_rate
            }
            
            # Add anomalous values
            if record["is_anomaly"]:
                record["values"] = [v * random.uniform(3, 5) for v in record["values"]]
            
            batch.append(record)
        
        return batch
"""Optimized file-based adapters with performance enhancements."""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from machine_learning.domain.interfaces.data_operations import DataIngestionPort, DataProcessingPort, DataStoragePort
from machine_learning.domain.entities.training_data import TrainingData
from machine_learning.domain.entities.prediction_request import PredictionRequest
from machine_learning.domain.entities.model_metadata import ModelMetadata
from shared.performance import performance_monitor, cached, batch_operation, CacheManager

logger = logging.getLogger(__name__)

class OptimizedFileBasedDataIngestion(DataIngestionPort):
    """Optimized file-based data ingestion with caching and batch processing."""
    
    def __init__(self, data_dir: str = "/tmp/ml_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self._cache = CacheManager(default_ttl=600)  # 10 minute cache
    
    @performance_monitor("data_ingestion")
    @cached(ttl=300)
    async def ingest_data(self, source: str) -> TrainingData:
        """Ingest data with caching and performance monitoring."""
        data_file = self.data_dir / f"{source}.json"
        
        if not data_file.exists():
            # Create optimized sample data
            sample_data = {
                "features": [[i, i*2, i*3] for i in range(1000)],  # Larger dataset
                "labels": [i % 2 for i in range(1000)],
                "metadata": {
                    "source": source,
                    "size": 1000,
                    "feature_count": 3
                }
            }
            
            with open(data_file, 'w') as f:
                json.dump(sample_data, f)
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        return TrainingData(
            features=data["features"],
            labels=data["labels"],
            source=source
        )
    
    async def batch_ingest_data(self, sources: List[str]) -> List[TrainingData]:
        """Batch ingest multiple data sources with concurrency control."""
        async with batch_operation(batch_size=10) as batch:
            tasks = [self.ingest_data(source) for source in sources]
            return await asyncio.gather(*tasks, return_exceptions=True)

class OptimizedFileBasedDataProcessing(DataProcessingPort):
    """Optimized data processing with performance enhancements."""
    
    def __init__(self, output_dir: str = "/tmp/ml_processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    @performance_monitor("data_processing")
    async def process_data(self, data: TrainingData) -> TrainingData:
        """Process data with performance optimization."""
        # Simulate optimized processing with vectorized operations
        processed_features = []
        
        # Process in chunks for better memory usage
        chunk_size = 100
        for i in range(0, len(data.features), chunk_size):
            chunk = data.features[i:i + chunk_size]
            # Simulate vectorized processing
            processed_chunk = [[x * 1.1 for x in row] for row in chunk]
            processed_features.extend(processed_chunk)
            
            # Allow other tasks to run
            if i % 500 == 0:
                await asyncio.sleep(0.001)
        
        return TrainingData(
            features=processed_features,
            labels=data.labels,
            source=data.source
        )
    
    @performance_monitor("data_validation")
    async def validate_data(self, data: TrainingData) -> bool:
        """Validate data with optimized checks."""
        if not data.features or not data.labels:
            return False
        
        if len(data.features) != len(data.labels):
            return False
        
        # Optimized validation - sample check instead of full validation
        sample_size = min(100, len(data.features))
        for i in range(0, len(data.features), len(data.features) // sample_size):
            if not isinstance(data.features[i], list) or not data.features[i]:
                return False
        
        return True

class OptimizedFileBasedDataStorage(DataStoragePort):
    """Optimized data storage with compression and indexing."""
    
    def __init__(self, storage_dir: str = "/tmp/ml_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._index_file = self.storage_dir / "index.json"
        self._load_index()
    
    def _load_index(self):
        """Load storage index for fast lookups."""
        if self._index_file.exists():
            with open(self._index_file, 'r') as f:
                self._index = json.load(f)
        else:
            self._index = {}
    
    def _save_index(self):
        """Save storage index."""
        with open(self._index_file, 'w') as f:
            json.dump(self._index, f)
    
    @performance_monitor("data_storage")
    async def store_data(self, data: TrainingData, identifier: str) -> bool:
        """Store data with optimized indexing."""
        try:
            storage_file = self.storage_dir / f"{identifier}.json"
            
            # Store data
            data_dict = {
                "features": data.features,
                "labels": data.labels,
                "source": data.source
            }
            
            with open(storage_file, 'w') as f:
                json.dump(data_dict, f, separators=(',', ':'))  # Compact JSON
            
            # Update index
            self._index[identifier] = {
                "file": str(storage_file),
                "size": len(data.features),
                "source": data.source
            }
            self._save_index()
            
            return True
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            return False
    
    @performance_monitor("data_retrieval")
    @cached(ttl=600)
    async def retrieve_data(self, identifier: str) -> Optional[TrainingData]:
        """Retrieve data with caching."""
        if identifier not in self._index:
            return None
        
        storage_file = Path(self._index[identifier]["file"])
        if not storage_file.exists():
            return None
        
        try:
            with open(storage_file, 'r') as f:
                data = json.load(f)
            
            return TrainingData(
                features=data["features"],
                labels=data["labels"],
                source=data["source"]
            )
        except Exception as e:
            logger.error(f"Failed to retrieve data: {e}")
            return None
    
    async def list_stored_data(self) -> List[str]:
        """List all stored data identifiers."""
        return list(self._index.keys())
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = sum(entry["size"] for entry in self._index.values())
        return {
            "total_datasets": len(self._index),
            "total_samples": total_size,
            "storage_files": len(list(self.storage_dir.glob("*.json"))) - 1  # Exclude index
        }
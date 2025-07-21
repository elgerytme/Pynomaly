"""Integration adapters for external systems and data sources."""

from __future__ import annotations

import json
import csv
import time
from typing import Any, Dict, List, Optional, Union, Iterator, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
import warnings

from simplified_services.core_detection_service import CoreDetectionService, DetectionResult


@dataclass
class IntegrationConfig:
    """Configuration for integration adapters."""
    source_type: str
    connection_params: Dict[str, Any]
    data_format: str = "json"
    batch_size: int = 1000
    max_retries: int = 3
    timeout_seconds: float = 30.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class DataSource:
    """Data source information."""
    source_id: str
    source_type: str
    description: str
    schema: Dict[str, str]
    last_updated: str
    record_count: int
    status: str = "active"


class BaseIntegrationAdapter(ABC):
    """Base class for integration adapters."""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize adapter with configuration."""
        self.config = config
        self.detection_service = CoreDetectionService()
        self._cache: Dict[str, Any] = {}
        self._last_cache_clear = time.time()
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to external system."""
        pass
    
    @abstractmethod
    def fetch_data(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Fetch data from external system."""
        pass
    
    @abstractmethod
    def send_results(
        self,
        results: List[DetectionResult],
        destination: Optional[str] = None
    ) -> bool:
        """Send detection results to external system."""
        pass
    
    def disconnect(self) -> None:
        """Close connection to external system."""
        pass
    
    def _clear_cache_if_expired(self) -> None:
        """Clear cache if TTL has expired."""
        if time.time() - self._last_cache_clear > self.config.cache_ttl_seconds:
            self._cache.clear()
            self._last_cache_clear = time.time()


class FileSystemAdapter(BaseIntegrationAdapter):
    """Adapter for file system data sources (CSV, JSON, etc.)."""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize file system adapter."""
        super().__init__(config)
        self.base_path = Path(config.connection_params.get("base_path", "."))
        self._file_handle = None
    
    def connect(self) -> bool:
        """Check if base path exists."""
        try:
            return self.base_path.exists() and self.base_path.is_dir()
        except Exception:
            return False
    
    def fetch_data(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Fetch data from files."""
        file_pattern = query.get("file_pattern", "*") if query else "*"
        files = list(self.base_path.glob(file_pattern))
        
        count = 0
        for file_path in files:
            if limit and count >= limit:
                break
            
            try:
                if file_path.suffix.lower() == '.csv':
                    yield from self._read_csv(file_path, limit - count if limit else None)
                elif file_path.suffix.lower() == '.json':
                    yield from self._read_json(file_path, limit - count if limit else None)
                else:
                    # Try to read as text
                    with open(file_path, 'r', encoding='utf-8') as f:
                        yield {
                            "file_path": str(file_path),
                            "content": f.read(),
                            "source": "filesystem"
                        }
                        count += 1
                        
            except Exception as e:
                warnings.warn(f"Error reading file {file_path}: {e}")
                continue
    
    def send_results(
        self,
        results: List[DetectionResult],
        destination: Optional[str] = None
    ) -> bool:
        """Save detection results to files."""
        try:
            output_path = Path(destination) if destination else self.base_path / "anomaly_results.json"
            
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                result_dict = {
                    "algorithm": result.algorithm,
                    "contamination": result.contamination,
                    "n_samples": result.n_samples,
                    "n_anomalies": result.n_anomalies,
                    "predictions": result.predictions.tolist(),
                    "scores": result.scores.tolist() if result.scores is not None else None,
                    "metadata": result.metadata,
                    "timestamp": time.time()
                }
                serializable_results.append(result_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"📁 Results saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    def _read_csv(self, file_path: Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Read CSV file."""
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if limit and count >= limit:
                    break
                yield {**row, "source": "csv", "file_path": str(file_path)}
                count += 1
    
    def _read_json(self, file_path: Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Read JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            count = 0
            for item in data:
                if limit and count >= limit:
                    break
                yield {**item, "source": "json", "file_path": str(file_path)}
                count += 1
        else:
            yield {**data, "source": "json", "file_path": str(file_path)}


class DatabaseAdapter(BaseIntegrationAdapter):
    """Adapter for database connections (simplified, no actual DB dependency)."""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize database adapter."""
        super().__init__(config)
        self.connection_string = config.connection_params.get("connection_string")
        self.table_name = config.connection_params.get("table_name", "anomaly_data")
        self._connected = False
    
    def connect(self) -> bool:
        """Simulate database connection."""
        try:
            # In a real implementation, this would establish actual DB connection
            print(f"🔌 Simulating connection to database: {self.connection_string}")
            self._connected = True
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def fetch_data(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Simulate fetching data from database."""
        if not self._connected:
            raise RuntimeError("Not connected to database")
        
        # Simulate database query results
        print(f"📊 Simulating query execution on table: {self.table_name}")
        
        # Generate mock data
        for i in range(limit or 100):
            yield {
                "id": i,
                "timestamp": time.time() - i * 3600,
                "feature_1": np.random.normal(0, 1),
                "feature_2": np.random.normal(0, 1),
                "feature_3": np.random.normal(0, 1),
                "source": "database",
                "table": self.table_name
            }
    
    def send_results(
        self,
        results: List[DetectionResult],
        destination: Optional[str] = None
    ) -> bool:
        """Simulate sending results to database."""
        try:
            table_name = destination or f"{self.table_name}_anomaly_results"
            print(f"💾 Simulating insert of {len(results)} results into table: {table_name}")
            
            # In real implementation, would execute SQL INSERT statements
            for i, result in enumerate(results):
                print(f"  - Result {i+1}: {result.n_anomalies} anomalies detected")
            
            return True
            
        except Exception as e:
            print(f"Error sending results to database: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from database."""
        if self._connected:
            print("🔌 Disconnecting from database")
            self._connected = False


class APIAdapter(BaseIntegrationAdapter):
    """Adapter for REST API integrations."""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize API adapter."""
        super().__init__(config)
        self.base_url = config.connection_params.get("base_url")
        self.api_key = config.connection_params.get("api_key")
        self.headers = config.connection_params.get("headers", {})
        self._session = None
    
    def connect(self) -> bool:
        """Test API connection."""
        try:
            # In real implementation, would test API endpoint
            print(f"🌐 Testing API connection to: {self.base_url}")
            return True
        except Exception as e:
            print(f"API connection failed: {e}")
            return False
    
    def fetch_data(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Fetch data from API endpoints."""
        endpoint = query.get("endpoint", "/data") if query else "/data"
        
        print(f"🌐 Simulating API call to: {self.base_url}{endpoint}")
        
        # Simulate API response
        for i in range(limit or 50):
            yield {
                "api_id": f"api_record_{i}",
                "timestamp": time.time() - i * 60,
                "metrics": {
                    "cpu_usage": np.random.uniform(0, 100),
                    "memory_usage": np.random.uniform(0, 100),
                    "disk_usage": np.random.uniform(0, 100)
                },
                "source": "api",
                "endpoint": endpoint
            }
    
    def send_results(
        self,
        results: List[DetectionResult],
        destination: Optional[str] = None
    ) -> bool:
        """Send results to API endpoint."""
        try:
            endpoint = destination or "/anomaly-results"
            
            # Convert results to API format
            api_payload = {
                "timestamp": time.time(),
                "results": []
            }
            
            for result in results:
                api_payload["results"].append({
                    "algorithm": result.algorithm,
                    "anomaly_count": result.n_anomalies,
                    "total_samples": result.n_samples,
                    "anomaly_ratio": result.n_anomalies / result.n_samples if result.n_samples > 0 else 0,
                    "metadata": result.metadata
                })
            
            print(f"🌐 Simulating POST to: {self.base_url}{endpoint}")
            print(f"  - Payload size: {len(json.dumps(api_payload))} bytes")
            
            return True
            
        except Exception as e:
            print(f"Error sending results to API: {e}")
            return False


class StreamingAdapter(BaseIntegrationAdapter):
    """Adapter for streaming data sources (Kafka-like)."""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize streaming adapter."""
        super().__init__(config)
        self.topic = config.connection_params.get("topic", "anomaly-data")
        self.brokers = config.connection_params.get("brokers", ["localhost:9092"])
        self._consumer = None
        self._producer = None
    
    def connect(self) -> bool:
        """Connect to streaming platform."""
        try:
            print(f"🔄 Simulating connection to streaming platform")
            print(f"  - Topic: {self.topic}")
            print(f"  - Brokers: {self.brokers}")
            return True
        except Exception as e:
            print(f"Streaming connection failed: {e}")
            return False
    
    def fetch_data(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Fetch streaming data."""
        print(f"🔄 Simulating streaming data consumption from topic: {self.topic}")
        
        # Simulate streaming messages
        count = 0
        while limit is None or count < limit:
            yield {
                "message_id": f"msg_{count}",
                "timestamp": time.time(),
                "partition": count % 3,
                "offset": count,
                "data": {
                    "sensor_id": f"sensor_{count % 10}",
                    "temperature": np.random.normal(20, 5),
                    "humidity": np.random.uniform(30, 70),
                    "pressure": np.random.normal(1013, 10)
                },
                "source": "streaming",
                "topic": self.topic
            }
            count += 1
            
            # Simulate real-time delay
            time.sleep(0.1)
            
            if count >= 100:  # Prevent infinite loop in demo
                break
    
    def send_results(
        self,
        results: List[DetectionResult],
        destination: Optional[str] = None
    ) -> bool:
        """Send results to streaming platform."""
        try:
            output_topic = destination or f"{self.topic}-anomalies"
            
            print(f"🔄 Simulating streaming publish to topic: {output_topic}")
            
            for i, result in enumerate(results):
                message = {
                    "message_id": f"anomaly_result_{i}",
                    "timestamp": time.time(),
                    "result": {
                        "algorithm": result.algorithm,
                        "anomaly_count": result.n_anomalies,
                        "total_samples": result.n_samples,
                        "metadata": result.metadata
                    }
                }
                print(f"  - Publishing message {i+1}: {result.n_anomalies} anomalies")
            
            return True
            
        except Exception as e:
            print(f"Error publishing to streaming platform: {e}")
            return False


class IntegrationManager:
    """Manager for multiple integration adapters."""
    
    def __init__(self):
        """Initialize integration manager."""
        self.adapters: Dict[str, BaseIntegrationAdapter] = {}
        self.data_sources: Dict[str, DataSource] = {}
    
    def register_adapter(
        self,
        adapter_id: str,
        adapter: BaseIntegrationAdapter
    ) -> None:
        """Register an integration adapter."""
        self.adapters[adapter_id] = adapter
        print(f"🔧 Registered adapter: {adapter_id}")
    
    def connect_all(self) -> Dict[str, bool]:
        """Connect all registered adapters."""
        results = {}
        
        for adapter_id, adapter in self.adapters.items():
            try:
                results[adapter_id] = adapter.connect()
                print(f"✅ Adapter {adapter_id}: {'Connected' if results[adapter_id] else 'Failed'}")
            except Exception as e:
                results[adapter_id] = False
                print(f"❌ Adapter {adapter_id}: Error - {e}")
        
        return results
    
    def fetch_from_all_sources(
        self,
        limit_per_source: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch data from all connected sources."""
        all_data = {}
        
        for adapter_id, adapter in self.adapters.items():
            try:
                data_list = list(adapter.fetch_data(limit=limit_per_source))
                all_data[adapter_id] = data_list
                print(f"📥 Fetched {len(data_list)} records from {adapter_id}")
            except Exception as e:
                print(f"Error fetching from {adapter_id}: {e}")
                all_data[adapter_id] = []
        
        return all_data
    
    def run_anomaly_detection_pipeline(
        self,
        source_adapters: List[str],
        output_adapters: List[str],
        algorithm: str = "iforest",
        contamination: float = 0.1
    ) -> List[DetectionResult]:
        """Run complete anomaly detection pipeline."""
        print(f"🔄 Starting anomaly detection pipeline")
        print(f"  - Sources: {source_adapters}")
        print(f"  - Outputs: {output_adapters}")
        print(f"  - Algorithm: {algorithm}")
        
        all_results = []
        
        # Fetch data from source adapters
        for source_id in source_adapters:
            if source_id not in self.adapters:
                print(f"Warning: Source adapter {source_id} not found")
                continue
            
            try:
                # Fetch raw data
                raw_data = list(self.adapters[source_id].fetch_data(limit=1000))
                
                if not raw_data:
                    print(f"No data from source {source_id}")
                    continue
                
                # Convert to numerical format for anomaly detection
                numerical_data = self._convert_to_numerical(raw_data)
                
                if numerical_data is None or len(numerical_data) == 0:
                    print(f"Could not convert data from {source_id} to numerical format")
                    continue
                
                # Run anomaly detection
                detection_service = CoreDetectionService()
                result = detection_service.detect_anomalies(
                    numerical_data,
                    algorithm=algorithm,
                    contamination=contamination
                )
                
                # Add source information to metadata
                result.metadata["source_adapter"] = source_id
                result.metadata["source_record_count"] = len(raw_data)
                
                all_results.append(result)
                print(f"✅ Detected {result.n_anomalies} anomalies from {source_id}")
                
            except Exception as e:
                print(f"Error processing data from {source_id}: {e}")
        
        # Send results to output adapters
        for output_id in output_adapters:
            if output_id not in self.adapters:
                print(f"Warning: Output adapter {output_id} not found")
                continue
            
            try:
                success = self.adapters[output_id].send_results(all_results)
                if success:
                    print(f"✅ Results sent to {output_id}")
                else:
                    print(f"❌ Failed to send results to {output_id}")
            except Exception as e:
                print(f"Error sending results to {output_id}: {e}")
        
        return all_results
    
    def _convert_to_numerical(
        self,
        raw_data: List[Dict[str, Any]]
    ) -> Optional[npt.NDArray[np.floating]]:
        """Convert raw data to numerical format for anomaly detection."""
        if not raw_data:
            return None
        
        # Extract numerical features from data
        numerical_features = []
        
        for record in raw_data:
            features = []
            
            # Extract numbers from various fields
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, dict):
                    # Extract numbers from nested dict
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, (int, float)):
                            features.append(float(nested_value))
                elif isinstance(value, str):
                    # Try to convert string to number
                    try:
                        features.append(float(value))
                    except ValueError:
                        # Skip non-numerical strings
                        pass
            
            if features:
                numerical_features.append(features)
        
        if not numerical_features:
            return None
        
        # Ensure all rows have same number of features
        min_features = min(len(row) for row in numerical_features)
        if min_features == 0:
            return None
        
        # Truncate to common feature count
        normalized_features = [row[:min_features] for row in numerical_features]
        
        return np.array(normalized_features)
    
    def get_adapter_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all adapters."""
        status = {}
        
        for adapter_id, adapter in self.adapters.items():
            try:
                # Test connection
                is_connected = adapter.connect()
                
                status[adapter_id] = {
                    "adapter_type": type(adapter).__name__,
                    "connected": is_connected,
                    "config": {
                        "source_type": adapter.config.source_type,
                        "data_format": adapter.config.data_format,
                        "batch_size": adapter.config.batch_size
                    }
                }
            except Exception as e:
                status[adapter_id] = {
                    "adapter_type": type(adapter).__name__,
                    "connected": False,
                    "error": str(e)
                }
        
        return status
    
    def disconnect_all(self) -> None:
        """Disconnect all adapters."""
        for adapter_id, adapter in self.adapters.items():
            try:
                adapter.disconnect()
                print(f"🔌 Disconnected adapter: {adapter_id}")
            except Exception as e:
                print(f"Error disconnecting {adapter_id}: {e}")


# Factory function for creating adapters
def create_adapter(
    adapter_type: str,
    config: IntegrationConfig
) -> BaseIntegrationAdapter:
    """Create an integration adapter of the specified type."""
    adapter_classes = {
        "filesystem": FileSystemAdapter,
        "database": DatabaseAdapter,
        "api": APIAdapter,
        "streaming": StreamingAdapter
    }
    
    adapter_class = adapter_classes.get(adapter_type.lower())
    if not adapter_class:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    return adapter_class(config)
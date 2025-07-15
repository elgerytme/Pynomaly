"""Integration testing configuration and utilities."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests."""
    
    # Test data configuration
    sample_size: int = 1000
    feature_count: int = 10
    anomaly_rate: float = 0.1
    random_seed: int = 42
    
    # Performance test configuration
    max_execution_time_seconds: float = 30.0
    max_memory_usage_mb: float = 500.0
    min_throughput_ops_per_sec: float = 1.0
    max_error_rate_percent: float = 1.0
    
    # Load test configuration
    load_test_users: int = 10
    load_test_duration_seconds: int = 60
    load_test_ramp_up_seconds: int = 10
    
    # Security test configuration
    enable_security_tests: bool = True
    enable_compliance_validation: bool = True
    audit_trail_required: bool = True
    
    # Multi-tenant test configuration
    tenant_count: int = 3
    tenant_isolation_strict: bool = True
    
    # Disaster recovery test configuration
    enable_disaster_recovery_tests: bool = True
    backup_retention_days: int = 30
    recovery_time_objective_minutes: int = 15
    
    # API contract test configuration
    api_version: str = "v1"
    validate_response_schemas: bool = True
    test_backward_compatibility: bool = True
    
    # Test data paths
    test_data_dir: Optional[Path] = None
    temp_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize test directories."""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="pynomaly_integration_"))
        
        if self.test_data_dir is None:
            self.test_data_dir = self.temp_dir / "test_data"
            self.test_data_dir.mkdir(exist_ok=True)

    @classmethod
    def from_environment(cls) -> 'IntegrationTestConfig':
        """Create configuration from environment variables."""
        return cls(
            sample_size=int(os.getenv('INTEGRATION_SAMPLE_SIZE', '1000')),
            feature_count=int(os.getenv('INTEGRATION_FEATURE_COUNT', '10')),
            anomaly_rate=float(os.getenv('INTEGRATION_ANOMALY_RATE', '0.1')),
            random_seed=int(os.getenv('INTEGRATION_RANDOM_SEED', '42')),
            max_execution_time_seconds=float(os.getenv('INTEGRATION_MAX_EXEC_TIME', '30.0')),
            max_memory_usage_mb=float(os.getenv('INTEGRATION_MAX_MEMORY_MB', '500.0')),
            min_throughput_ops_per_sec=float(os.getenv('INTEGRATION_MIN_THROUGHPUT', '1.0')),
            load_test_users=int(os.getenv('INTEGRATION_LOAD_USERS', '10')),
            load_test_duration_seconds=int(os.getenv('INTEGRATION_LOAD_DURATION', '60')),
            enable_security_tests=os.getenv('INTEGRATION_SECURITY_TESTS', 'true').lower() == 'true',
            enable_disaster_recovery_tests=os.getenv('INTEGRATION_DR_TESTS', 'true').lower() == 'true',
        )


class TestDataGenerator:
    """Utility class for generating test data."""
    
    def __init__(self, config: IntegrationTestConfig):
        self.config = config
        np.random.seed(config.random_seed)

    def generate_anomaly_dataset(
        self,
        size: Optional[int] = None,
        features: Optional[int] = None,
        anomaly_rate: Optional[float] = None,
    ) -> np.ndarray:
        """Generate synthetic anomaly detection dataset."""
        size = size or self.config.sample_size
        features = features or self.config.feature_count
        anomaly_rate = anomaly_rate or self.config.anomaly_rate
        
        # Generate normal data
        normal_size = int(size * (1 - anomaly_rate))
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(features),
            cov=np.eye(features),
            size=normal_size
        )
        
        # Generate anomalous data
        anomaly_size = size - normal_size
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(features) * 3,  # Shifted mean for anomalies
            cov=np.eye(features) * 2,    # Different covariance
            size=anomaly_size
        )
        
        # Combine and shuffle
        data = np.vstack([normal_data, anomaly_data])
        indices = np.random.permutation(len(data))
        
        return data[indices]

    def generate_streaming_data(
        self,
        batch_size: int = 100,
        num_batches: int = 10,
        anomaly_rate: float = 0.05,
    ) -> list[np.ndarray]:
        """Generate streaming data batches."""
        batches = []
        
        for _ in range(num_batches):
            batch = self.generate_anomaly_dataset(
                size=batch_size,
                anomaly_rate=anomaly_rate
            )
            batches.append(batch)
        
        return batches

    def generate_time_series_data(
        self,
        length: int = 1000,
        features: int = 5,
        seasonal_period: int = 50,
        anomaly_positions: Optional[list[int]] = None,
    ) -> np.ndarray:
        """Generate time series data with seasonal patterns and anomalies."""
        t = np.arange(length)
        
        # Base trend
        trend = 0.001 * t
        
        # Seasonal component
        seasonal = np.sin(2 * np.pi * t / seasonal_period)
        
        # Generate multiple features
        data = np.zeros((length, features))
        for f in range(features):
            # Random walk component
            noise = np.cumsum(np.random.normal(0, 0.1, length))
            
            # Combine components
            data[:, f] = trend + seasonal + noise
            
            # Add feature-specific variations
            if f > 0:
                data[:, f] += np.sin(2 * np.pi * t / (seasonal_period * (f + 1))) * 0.5
        
        # Inject anomalies
        if anomaly_positions:
            for pos in anomaly_positions:
                if 0 <= pos < length:
                    # Spike anomaly
                    data[pos, :] += np.random.normal(0, 3, features)
        elif anomaly_positions is None:
            # Random anomalies
            num_anomalies = int(length * 0.02)  # 2% anomalies
            positions = np.random.choice(length, num_anomalies, replace=False)
            for pos in positions:
                data[pos, :] += np.random.normal(0, 3, features)
        
        return data

    def generate_multi_tenant_data(self) -> Dict[str, np.ndarray]:
        """Generate data for multiple tenants."""
        tenant_data = {}
        
        for i in range(self.config.tenant_count):
            tenant_id = f"tenant_{chr(ord('A') + i)}"
            
            # Each tenant has slightly different data characteristics
            mean_shift = np.ones(self.config.feature_count) * i
            cov_scale = 1 + i * 0.2
            
            tenant_data[tenant_id] = self.generate_tenant_specific_data(
                mean_shift=mean_shift,
                cov_scale=cov_scale
            )
        
        return tenant_data

    def generate_tenant_specific_data(
        self,
        mean_shift: np.ndarray,
        cov_scale: float,
    ) -> np.ndarray:
        """Generate data specific to a tenant."""
        size = self.config.sample_size // self.config.tenant_count
        
        # Normal data with tenant-specific characteristics
        normal_data = np.random.multivariate_normal(
            mean=mean_shift,
            cov=np.eye(len(mean_shift)) * cov_scale,
            size=int(size * 0.9)
        )
        
        # Anomalous data
        anomaly_data = np.random.multivariate_normal(
            mean=mean_shift + 3,
            cov=np.eye(len(mean_shift)) * cov_scale * 2,
            size=int(size * 0.1)
        )
        
        data = np.vstack([normal_data, anomaly_data])
        indices = np.random.permutation(len(data))
        
        return data[indices]

    def save_dataset_to_csv(self, data: np.ndarray, filename: str) -> Path:
        """Save dataset to CSV file."""
        filepath = self.config.test_data_dir / filename
        
        # Create DataFrame with column names
        columns = [f"feature_{i}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=columns)
        
        # Add metadata columns
        df['timestamp'] = pd.date_range(
            start=datetime.now(),
            periods=len(df),
            freq='1min'
        )
        df['sample_id'] = range(len(df))
        
        df.to_csv(filepath, index=False)
        return filepath

    def save_dataset_to_json(self, data: np.ndarray, filename: str) -> Path:
        """Save dataset to JSON file."""
        filepath = self.config.test_data_dir / filename
        
        # Convert to list of dictionaries
        records = []
        for i, row in enumerate(data):
            record = {
                f"feature_{j}": float(row[j]) for j in range(len(row))
            }
            record.update({
                "timestamp": (datetime.now().timestamp() + i * 60),  # 1 minute intervals
                "sample_id": i,
            })
            records.append(record)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(records, f, indent=2)
        
        return filepath


class TestEnvironmentManager:
    """Manages test environment setup and cleanup."""
    
    def __init__(self, config: IntegrationTestConfig):
        self.config = config
        self.data_generator = TestDataGenerator(config)
        self.generated_files: list[Path] = []

    def setup_test_environment(self) -> Dict[str, Any]:
        """Set up complete test environment."""
        # Generate standard datasets
        datasets = {}
        
        # Standard anomaly detection dataset
        standard_data = self.data_generator.generate_anomaly_dataset()
        standard_csv = self.data_generator.save_dataset_to_csv(
            standard_data, "standard_dataset.csv"
        )
        datasets["standard_csv"] = standard_csv
        self.generated_files.append(standard_csv)
        
        # Time series dataset
        ts_data = self.data_generator.generate_time_series_data()
        ts_csv = self.data_generator.save_dataset_to_csv(
            ts_data, "time_series_dataset.csv"
        )
        datasets["time_series_csv"] = ts_csv
        self.generated_files.append(ts_csv)
        
        # Multi-tenant datasets
        tenant_data = self.data_generator.generate_multi_tenant_data()
        for tenant_id, data in tenant_data.items():
            tenant_csv = self.data_generator.save_dataset_to_csv(
                data, f"{tenant_id}_dataset.csv"
            )
            datasets[f"{tenant_id}_csv"] = tenant_csv
            self.generated_files.append(tenant_csv)
        
        # Streaming data
        streaming_batches = self.data_generator.generate_streaming_data()
        datasets["streaming_batches"] = streaming_batches
        
        # Performance test data (larger dataset)
        large_data = self.data_generator.generate_anomaly_dataset(
            size=5000, features=20
        )
        large_csv = self.data_generator.save_dataset_to_csv(
            large_data, "large_dataset.csv"
        )
        datasets["large_csv"] = large_csv
        self.generated_files.append(large_csv)
        
        return {
            "config": self.config,
            "datasets": datasets,
            "data_generator": self.data_generator,
            "temp_dir": self.config.temp_dir,
        }

    def cleanup_test_environment(self):
        """Clean up test environment."""
        # Remove generated files
        for filepath in self.generated_files:
            try:
                if filepath.exists():
                    filepath.unlink()
            except Exception:
                pass
        
        # Remove temporary directory
        try:
            import shutil
            if self.config.temp_dir and self.config.temp_dir.exists():
                shutil.rmtree(self.config.temp_dir)
        except Exception:
            pass


@dataclass
class TestResult:
    """Test result data structure."""
    
    test_name: str
    success: bool
    execution_time: float
    memory_usage_mb: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TestResultCollector:
    """Collects and manages test results."""
    
    def __init__(self):
        self.results: list[TestResult] = []

    def add_result(self, result: TestResult):
        """Add a test result."""
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all test results."""
        if not self.results:
            return {"message": "No test results available"}
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        avg_execution_time = sum(r.execution_time for r in self.results) / total_tests
        avg_memory_usage = sum(r.memory_usage_mb for r in self.results) / total_tests
        
        summary = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate_percent": (successful_tests / total_tests) * 100,
            "avg_execution_time_seconds": avg_execution_time,
            "avg_memory_usage_mb": avg_memory_usage,
            "failed_test_names": [r.test_name for r in self.results if not r.success],
        }
        
        return summary

    def export_results(self, filepath: Path):
        """Export results to JSON file."""
        import json
        
        export_data = {
            "summary": self.get_summary(),
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "memory_usage_mb": r.memory_usage_mb,
                    "error_message": r.error_message,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self.results
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


# Global test configuration instance
TEST_CONFIG = IntegrationTestConfig.from_environment()
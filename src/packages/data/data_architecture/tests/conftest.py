"""
Pytest configuration for Data Architecture package testing.
Provides fixtures for data pipeline, schema validation, and integration testing.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
from uuid import uuid4
from datetime import datetime, timedelta
from unittest.mock import Mock
import json


@pytest.fixture
def sample_data_schema() -> Dict[str, Any]:
    """Sample data schema for testing."""
    return {
        'name': 'customer_anomaly_data',
        'version': '1.0.0',
        'description': 'Customer behavior data for anomaly detection',
        'fields': [
            {
                'name': 'customer_id',
                'type': 'string',
                'required': True,
                'description': 'Unique customer identifier'
            },
            {
                'name': 'transaction_amount',
                'type': 'float',
                'required': True,
                'constraints': {
                    'min': 0.01,
                    'max': 10000.00
                }
            },
            {
                'name': 'transaction_count',
                'type': 'integer',
                'required': True,
                'constraints': {
                    'min': 1,
                    'max': 1000
                }
            },
            {
                'name': 'account_age_days',
                'type': 'integer',
                'required': True,
                'constraints': {
                    'min': 0
                }
            },
            {
                'name': 'risk_score',
                'type': 'float',
                'required': False,
                'constraints': {
                    'min': 0.0,
                    'max': 1.0
                }
            },
            {
                'name': 'created_at',
                'type': 'datetime',
                'required': True
            }
        ]
    }


@pytest.fixture
def sample_pipeline_config() -> Dict[str, Any]:
    """Sample data pipeline configuration."""
    return {
        'pipeline_name': 'customer_anomaly_pipeline',
        'version': '1.2.0',
        'description': 'End-to-end customer anomaly detection pipeline',
        'stages': [
            {
                'name': 'data_ingestion',
                'type': 'source',
                'config': {
                    'source_type': 'database',
                    'connection_string': 'postgresql://localhost:5432/analytics',
                    'query': 'SELECT * FROM customer_transactions WHERE date >= CURRENT_DATE - INTERVAL \'7 days\'',
                    'batch_size': 10000
                },
                'outputs': ['raw_customer_data']
            },
            {
                'name': 'data_validation',
                'type': 'transformer',
                'config': {
                    'schema_validation': True,
                    'data_quality_checks': True,
                    'reject_invalid_rows': False
                },
                'inputs': ['raw_customer_data'],
                'outputs': ['validated_customer_data']
            },
            {
                'name': 'feature_engineering',
                'type': 'transformer',
                'config': {
                    'aggregation_window': '24h',
                    'features': [
                        'transaction_velocity',
                        'spending_pattern_deviation',
                        'time_based_features'
                    ]
                },
                'inputs': ['validated_customer_data'],
                'outputs': ['engineered_features']
            },
            {
                'name': 'anomaly_detection',
                'type': 'model',
                'config': {
                    'model_type': 'isolation_forest',
                    'contamination': 0.05,
                    'n_estimators': 100
                },
                'inputs': ['engineered_features'],
                'outputs': ['anomaly_scores']
            },
            {
                'name': 'results_storage',
                'type': 'sink',
                'config': {
                    'destination_type': 'database',
                    'table_name': 'anomaly_results',
                    'write_mode': 'append'
                },
                'inputs': ['anomaly_scores'],
                'outputs': []
            }
        ],
        'dependencies': {
            'data_validation': ['data_ingestion'],
            'feature_engineering': ['data_validation'],
            'anomaly_detection': ['feature_engineering'],
            'results_storage': ['anomaly_detection']
        },
        'error_handling': {
            'retry_attempts': 3,
            'retry_delay_seconds': 30,
            'dead_letter_queue': True
        },
        'monitoring': {
            'enable_metrics': True,
            'alert_thresholds': {
                'processing_time_seconds': 300,
                'error_rate_percentage': 5
            }
        }
    }


@pytest.fixture
def sample_test_data() -> pd.DataFrame:
    """Generate sample test data matching the schema."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': [f'CUST_{i:06d}' for i in range(n_samples)],
        'transaction_amount': np.random.lognormal(mean=3, sigma=1, size=n_samples),
        'transaction_count': np.random.poisson(lam=5, size=n_samples) + 1,
        'account_age_days': np.random.exponential(scale=365, size=n_samples).astype(int),
        'risk_score': np.random.beta(a=2, b=5, size=n_samples),
        'created_at': pd.date_range(
            start='2024-01-01', 
            end='2024-01-07', 
            periods=n_samples
        )
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def invalid_test_data() -> pd.DataFrame:
    """Generate test data with schema violations for validation testing."""
    np.random.seed(42)
    
    # Create data with various violations
    data = {
        'customer_id': ['CUST_001', None, 'CUST_003', '', 'CUST_005'],  # Null and empty values
        'transaction_amount': [100.50, -50.0, 15000.0, 25.75, np.nan],  # Negative and out-of-range values
        'transaction_count': [5, 0, 1500, 3, 10],  # Zero and out-of-range values
        'account_age_days': [365, -10, 100, 50, 730],  # Negative values
        'risk_score': [0.75, 1.5, -0.2, 0.3, 0.9],  # Out-of-range values
        'created_at': [
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-02'),
            None,  # Null datetime
            pd.Timestamp('2024-01-04'),
            pd.Timestamp('2024-01-05')
        ]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def data_transformation_rules() -> List[Dict[str, Any]]:
    """Sample data transformation rules."""
    return [
        {
            'name': 'calculate_transaction_velocity',
            'type': 'aggregation',
            'config': {
                'group_by': ['customer_id'],
                'time_window': '24h',
                'aggregations': {
                    'transaction_count': 'sum',
                    'transaction_amount': 'sum'
                },
                'output_column': 'daily_transaction_velocity'
            }
        },
        {
            'name': 'normalize_amounts',
            'type': 'normalization',
            'config': {
                'method': 'z_score',
                'columns': ['transaction_amount'],
                'output_suffix': '_normalized'
            }
        },
        {
            'name': 'encode_categorical',
            'type': 'encoding',
            'config': {
                'method': 'one_hot',
                'columns': ['account_type', 'region'],
                'drop_original': True
            }
        },
        {
            'name': 'create_time_features',
            'type': 'feature_extraction',
            'config': {
                'datetime_column': 'created_at',
                'features': ['hour', 'day_of_week', 'is_weekend'],
                'timezone': 'UTC'
            }
        }
    ]


@pytest.fixture
def integration_test_scenarios() -> List[Dict[str, Any]]:
    """Integration test scenarios for data architecture."""
    return [
        {
            'scenario_name': 'end_to_end_pipeline_success',
            'description': 'Complete pipeline execution with valid data',
            'input_data_size': 1000,
            'expected_output_size': 1000,
            'expected_success_rate': 1.0,
            'max_processing_time_seconds': 60,
            'data_quality_threshold': 0.95
        },
        {
            'scenario_name': 'partial_data_failure_handling',
            'description': 'Pipeline handling of partial data failures',
            'input_data_size': 1000,
            'invalid_data_percentage': 0.10,
            'expected_output_size': 900,
            'expected_success_rate': 0.90,
            'max_processing_time_seconds': 70,
            'error_handling_required': True
        },
        {
            'scenario_name': 'high_volume_processing',
            'description': 'Pipeline performance with large data volumes',
            'input_data_size': 50000,
            'expected_output_size': 50000,
            'expected_success_rate': 0.98,
            'max_processing_time_seconds': 300,
            'memory_limit_mb': 2048
        },
        {
            'scenario_name': 'schema_evolution_compatibility',
            'description': 'Pipeline handling of schema changes',
            'schema_changes': [
                {'action': 'add_column', 'column': 'new_feature', 'type': 'float'},
                {'action': 'modify_constraint', 'column': 'transaction_amount', 'new_max': 50000}
            ],
            'backward_compatibility_required': True,
            'migration_strategy': 'gradual'
        }
    ]


@pytest.fixture
def performance_benchmarks() -> Dict[str, Any]:
    """Performance benchmarks for data architecture components."""
    return {
        'data_ingestion': {
            'throughput_rows_per_second': 10000,
            'latency_p95_milliseconds': 100,
            'memory_usage_mb_per_1k_rows': 50
        },
        'data_validation': {
            'throughput_rows_per_second': 50000,
            'latency_p95_milliseconds': 50,
            'validation_accuracy': 0.999
        },
        'feature_engineering': {
            'throughput_rows_per_second': 5000,
            'latency_p95_milliseconds': 200,
            'feature_quality_score': 0.95
        },
        'anomaly_detection': {
            'throughput_rows_per_second': 1000,
            'latency_p95_milliseconds': 1000,
            'detection_accuracy': 0.85,
            'false_positive_rate': 0.05
        },
        'data_storage': {
            'throughput_rows_per_second': 20000,
            'latency_p95_milliseconds': 200,
            'storage_efficiency': 0.80
        }
    }


@pytest.fixture
def mock_data_sources():
    """Mock data sources for testing."""
    class MockDataSource:
        def __init__(self, source_type: str):
            self.source_type = source_type
            self.connection_active = False
            self.data_cache = {}
            
        def connect(self) -> Dict[str, Any]:
            self.connection_active = True
            return {'success': True, 'connection_id': str(uuid4())}
            
        def disconnect(self):
            self.connection_active = False
            
        def read_data(self, query: str = None, limit: int = None) -> Dict[str, Any]:
            if not self.connection_active:
                return {'success': False, 'error': 'Not connected'}
                
            # Mock data generation based on source type
            if self.source_type == 'database':
                data_size = limit or 1000
                mock_data = self._generate_database_data(data_size)
            elif self.source_type == 'file':
                mock_data = self._generate_file_data()
            elif self.source_type == 'api':
                mock_data = self._generate_api_data()
            else:
                mock_data = pd.DataFrame()
                
            return {
                'success': True,
                'data': mock_data,
                'rows_read': len(mock_data),
                'query': query
            }
            
        def _generate_database_data(self, size: int) -> pd.DataFrame:
            np.random.seed(42)
            return pd.DataFrame({
                'id': range(size),
                'customer_id': [f'CUST_{i:06d}' for i in range(size)],
                'amount': np.random.lognormal(3, 1, size),
                'timestamp': pd.date_range('2024-01-01', periods=size, freq='h')
            })
            
        def _generate_file_data(self) -> pd.DataFrame:
            return pd.DataFrame({
                'file_data': ['sample1', 'sample2', 'sample3'],
                'processed': [True, False, True]
            })
            
        def _generate_api_data(self) -> pd.DataFrame:
            return pd.DataFrame({
                'api_response': [{'status': 'ok'}, {'status': 'error'}],
                'response_time_ms': [150, 300]
            })
    
    return {
        'database': MockDataSource('database'),
        'file': MockDataSource('file'),
        'api': MockDataSource('api')
    }


@pytest.fixture
def mock_data_sinks():
    """Mock data sinks for testing."""
    class MockDataSink:
        def __init__(self, sink_type: str):
            self.sink_type = sink_type
            self.connection_active = False
            self.written_data = []
            
        def connect(self) -> Dict[str, Any]:
            self.connection_active = True
            return {'success': True, 'sink_id': str(uuid4())}
            
        def disconnect(self):
            self.connection_active = False
            
        def write_data(self, data: pd.DataFrame, mode: str = 'append') -> Dict[str, Any]:
            if not self.connection_active:
                return {'success': False, 'error': 'Sink not connected'}
                
            try:
                if mode == 'overwrite':
                    self.written_data = []
                    
                self.written_data.append({
                    'data': data.copy(),
                    'timestamp': datetime.utcnow(),
                    'rows_written': len(data)
                })
                
                return {
                    'success': True,
                    'rows_written': len(data),
                    'total_rows': sum(entry['rows_written'] for entry in self.written_data)
                }
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
                
        def get_written_data(self) -> List[pd.DataFrame]:
            return [entry['data'] for entry in self.written_data]
            
        def get_write_statistics(self) -> Dict[str, Any]:
            if not self.written_data:
                return {'total_writes': 0, 'total_rows': 0}
                
            return {
                'total_writes': len(self.written_data),
                'total_rows': sum(entry['rows_written'] for entry in self.written_data),
                'first_write': self.written_data[0]['timestamp'],
                'last_write': self.written_data[-1]['timestamp']
            }
    
    return {
        'database': MockDataSink('database'),
        'file': MockDataSink('file'),
        'stream': MockDataSink('stream')
    }


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return self.end_time - self.start_time
    
    return Timer()


@pytest.fixture
def large_test_dataset() -> pd.DataFrame:
    """Generate large test dataset for performance testing."""
    np.random.seed(42)
    n_samples = 100000
    
    data = {
        'customer_id': [f'CUST_{i:08d}' for i in range(n_samples)],
        'transaction_amount': np.random.lognormal(mean=4, sigma=1.5, size=n_samples),
        'transaction_count': np.random.poisson(lam=3, size=n_samples) + 1,
        'account_age_days': np.random.exponential(scale=500, size=n_samples).astype(int),
        'risk_score': np.random.beta(a=2, b=3, size=n_samples),
        'region': np.random.choice(['NA', 'EU', 'APAC', 'LATAM'], size=n_samples),
        'account_type': np.random.choice(['personal', 'business', 'premium'], size=n_samples, p=[0.6, 0.3, 0.1]),
        'created_at': pd.date_range(
            start='2023-01-01', 
            end='2024-12-31', 
            periods=n_samples
        )
    }
    
    return pd.DataFrame(data)


def pytest_configure(config):
    """Configure pytest markers for data architecture testing."""
    markers = [
        "architecture: Data architecture tests",
        "integration: Integration testing",
        "pipeline: Data pipeline tests",
        "schema_validation: Schema validation tests",
        "data_transformation: Data transformation tests",
        "performance: Architecture performance tests"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)

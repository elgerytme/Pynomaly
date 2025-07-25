"""Comprehensive test configuration and fixtures for data_quality package."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime, timezone

# Import from shared test utilities if available
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[4] / "shared" / "test_utilities"))

try:
    from fixtures import mock_logger, temp_directory, sample_data
    from helpers import generate_test_data, generate_test_dataframe
    from factories import BaseFactory
    SHARED_TEST_UTILITIES_AVAILABLE = True
except ImportError:
    SHARED_TEST_UTILITIES_AVAILABLE = False

# Import data_quality modules
from src.data_quality.domain.entities.data_profile import DataProfile, ColumnProfile, ProfileStatistics, DataType, ProfileStatus
from src.data_quality.domain.entities.data_quality_check import DataQualityCheck, CheckType, CheckStatus, CheckSeverity, CheckResult
from src.data_quality.domain.entities.data_quality_rule import DataQualityRule, RuleType, RuleSeverity, RuleCondition, RuleOperator
from src.data_quality.application.services.data_profiling_service import DataProfilingService
from src.data_quality.application.services.data_quality_check_service import DataQualityCheckService
from src.data_quality.application.services.data_quality_rule_service import DataQualityRuleService
from src.data_quality.application.services.rule_evaluator import RuleEvaluator
from src.data_quality.infrastructure.adapters.pandas_csv_adapter import PandasCSVAdapter


# ==========================================
# CORE PYTEST CONFIGURATION
# ==========================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "database: Tests requiring database")
    config.addinivalue_line("markers", "external: Tests requiring external services")
    config.addinivalue_line("markers", "performance: Performance benchmark tests")


# ==========================================
# DATA FIXTURES
# ==========================================

@pytest.fixture
def sample_csv_data() -> pd.DataFrame:
    """Create sample CSV data for testing."""
    np.random.seed(42)
    
    return pd.DataFrame({
        'id': range(1, 101),
        'name': [f'Person_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'email': [f'person{i}@example.com' for i in range(1, 101)],
        'salary': np.random.uniform(30000, 120000, 100),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 100),
        'join_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'is_active': np.random.choice([True, False], 100, p=[0.8, 0.2]),
        'rating': np.random.uniform(1.0, 5.0, 100),
        'notes': [f'Notes for person {i}' if i % 3 == 0 else None for i in range(1, 101)]
    })


@pytest.fixture
def corrupted_csv_data() -> pd.DataFrame:
    """Create corrupted CSV data for testing validation rules."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'id': [1, 2, 3, None, 5, 6, 7, 8, 9, 10],  # Missing values
        'name': ['Alice', 'Bob', '', 'David', 'Eve', 'Frank', None, 'Grace', 'Henry', 'Ivy'],  # Empty and None
        'age': [-5, 25, 30, 150, 40, 45, 50, 55, 60, 65],  # Invalid ages
        'email': ['alice@example.com', 'invalid-email', 'charlie@example.com', 
                 'david@example.com', '', 'frank@example.com', 'grace@invalid', 
                 'henry@example.com', 'ivy@example.com', None],  # Invalid emails
        'salary': [50000, -10000, 75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000],  # Negative salary
        'department': ['Engineering', 'Unknown', 'Marketing', 'HR', '', 'Engineering', 
                      'Sales', 'Marketing', 'HR', 'Engineering'],  # Invalid department
        'rating': [4.5, 6.0, 3.2, 2.8, 4.1, 3.9, -1.0, 4.7, 3.5, 4.2]  # Invalid rating
    })
    
    return df


@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Create a large dataset for performance testing."""
    np.random.seed(42)
    size = 10000
    
    return pd.DataFrame({
        'id': range(1, size + 1),
        'value': np.random.uniform(0, 1000, size),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size),
        'timestamp': pd.date_range('2020-01-01', periods=size, freq='H'),
        'flag': np.random.choice([True, False], size),
        'score': np.random.normal(100, 15, size)
    })


@pytest.fixture
def temp_csv_file(sample_csv_data, tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test_data.csv"
    sample_csv_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def corrupted_csv_file(corrupted_csv_data, tmp_path):
    """Create a temporary corrupted CSV file for testing."""
    csv_file = tmp_path / "corrupted_data.csv"
    corrupted_csv_data.to_csv(csv_file, index=False)
    return str(csv_file)


# ==========================================
# DOMAIN ENTITY FIXTURES
# ==========================================

@pytest.fixture
def sample_data_profile():
    """Create a sample DataProfile entity."""
    profile = DataProfile(dataset_name="test_dataset")
    profile.start_profiling()
    
    # Add sample column profiles
    col_profile1 = ColumnProfile(column_name="id", data_type=DataType.INTEGER)
    col_profile1.statistics = ProfileStatistics(
        total_count=100,
        null_count=0,
        distinct_count=100
    )
    
    col_profile2 = ColumnProfile(column_name="name", data_type=DataType.STRING)
    col_profile2.statistics = ProfileStatistics(
        total_count=100,
        null_count=5,
        distinct_count=95
    )
    
    profile.add_column_profile(col_profile1)
    profile.add_column_profile(col_profile2)
    profile.complete_profiling()
    
    return profile


@pytest.fixture
def sample_data_quality_rule():
    """Create a sample DataQualityRule entity."""
    rule = DataQualityRule(
        name="age_range_check",
        description="Check if age is within valid range",
        rule_type=RuleType.RANGE,
        severity=RuleSeverity.ERROR,
        dataset_name="test_dataset"
    )
    
    # Add rule conditions
    condition = RuleCondition(
        column_name="age",
        operator=RuleOperator.BETWEEN,
        value="18,80"
    )
    rule.add_condition(condition)
    
    return rule


@pytest.fixture
def sample_data_quality_check(sample_data_quality_rule):
    """Create a sample DataQualityCheck entity."""
    check = DataQualityCheck(
        name="age_validation_check",
        description="Validate age values in dataset",
        check_type=CheckType.VALIDATION,
        rule_id=sample_data_quality_rule.id,
        dataset_name="test_dataset"
    )
    
    return check


@pytest.fixture
def sample_check_result(sample_data_quality_check):
    """Create a sample CheckResult entity."""
    result = CheckResult(
        check_id=sample_data_quality_check.id,
        dataset_name="test_dataset",
        passed=True,
        score=0.95,
        error_count=5,
        total_records=100,
        message="Age validation completed successfully"
    )
    
    return result


# ==========================================
# SERVICE MOCK FIXTURES
# ==========================================

@pytest.fixture
def mock_data_profile_repository():
    """Mock DataProfileRepository."""
    mock_repo = Mock()
    mock_repo.get_by_id.return_value = None
    mock_repo.save.return_value = None
    return mock_repo


@pytest.fixture
def mock_data_quality_check_repository():
    """Mock DataQualityCheckRepository."""
    mock_repo = Mock()
    mock_repo.get_by_id.return_value = None
    mock_repo.save.return_value = None
    return mock_repo


@pytest.fixture
def mock_data_quality_rule_repository():
    """Mock DataQualityRuleRepository."""
    mock_repo = Mock()
    mock_repo.get_by_id.return_value = None
    mock_repo.save.return_value = None
    return mock_repo


@pytest.fixture
def mock_pandas_csv_adapter():
    """Mock PandasCSVAdapter."""
    mock_adapter = Mock(spec=PandasCSVAdapter)
    return mock_adapter


@pytest.fixture
def mock_rule_evaluator():
    """Mock RuleEvaluator."""
    mock_evaluator = Mock(spec=RuleEvaluator)
    mock_evaluator.evaluate_record.return_value = True
    return mock_evaluator


# ==========================================
# SERVICE FIXTURES
# ==========================================

@pytest.fixture
def data_profiling_service(mock_data_profile_repository):
    """Create DataProfilingService with mocked dependencies."""
    return DataProfilingService(mock_data_profile_repository)


@pytest.fixture
def data_quality_check_service(
    mock_data_quality_check_repository,
    mock_data_quality_rule_repository,
    mock_pandas_csv_adapter,
    mock_rule_evaluator
):
    """Create DataQualityCheckService with mocked dependencies."""
    return DataQualityCheckService(
        mock_data_quality_check_repository,
        mock_data_quality_rule_repository,
        mock_pandas_csv_adapter,
        mock_rule_evaluator
    )


@pytest.fixture
def data_quality_rule_service(mock_data_quality_rule_repository):
    """Create DataQualityRuleService with mocked dependencies."""
    return DataQualityRuleService(mock_data_quality_rule_repository)


# ==========================================
# DATABASE FIXTURES
# ==========================================

@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = MagicMock()
    session.query.return_value = session
    session.filter_by.return_value = session
    session.first.return_value = None
    session.commit.return_value = None
    session.rollback.return_value = None
    session.close.return_value = None
    return session


@pytest.fixture
def in_memory_database():
    """Create in-memory SQLite database for testing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    try:
        from src.data_quality.infrastructure.database.models import Base
        
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
    except ImportError:
        # Fallback if models don't exist
        yield mock_db_session()


# ==========================================
# ASYNC FIXTURES
# ==========================================

@pytest.fixture
async def async_mock_service():
    """Create async mock service."""
    service = AsyncMock()
    return service


# ==========================================
# PERFORMANCE TESTING FIXTURES
# ==========================================

@pytest.fixture
def performance_monitor():
    """Create performance monitoring fixture."""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process(os.getpid())
        
        def start(self):
            self.start_time = time.perf_counter()
            self.start_memory = self.process.memory_info().rss
        
        def stop(self):
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss
            
            return {
                'duration_seconds': end_time - self.start_time,
                'memory_delta_mb': (end_memory - self.start_memory) / 1024 / 1024,
                'peak_memory_mb': end_memory / 1024 / 1024
            }
    
    return PerformanceMonitor()


# ==========================================
# INTEGRATION TEST FIXTURES
# ==========================================

@pytest.fixture
def integration_test_data():
    """Create comprehensive test data for integration tests."""
    return {
        'datasets': {
            'valid_dataset': pd.DataFrame({
                'id': range(1, 11),
                'name': [f'Item_{i}' for i in range(1, 11)],
                'value': np.random.uniform(10, 100, 10),
                'category': ['A', 'B', 'C'] * 3 + ['A']
            }),
            'invalid_dataset': pd.DataFrame({
                'id': [1, 2, None, 4, 5],
                'name': ['Item_1', '', 'Item_3', None, 'Item_5'],
                'value': [-5, 50, 75, 200, 25],
                'category': ['A', 'Invalid', 'C', 'D', 'A']
            })
        },
        'rules': [
            {
                'name': 'id_not_null',
                'type': RuleType.NOT_NULL,
                'column': 'id',
                'severity': RuleSeverity.ERROR
            },
            {
                'name': 'value_range',
                'type': RuleType.RANGE,
                'column': 'value',
                'severity': RuleSeverity.WARNING,
                'min_value': 0,
                'max_value': 150
            }
        ]
    }


# ==========================================
# E2E TEST FIXTURES
# ==========================================

@pytest.fixture
def e2e_test_environment(tmp_path):
    """Create end-to-end test environment."""
    env = {
        'temp_dir': tmp_path,
        'input_files': {},
        'output_files': {},
        'config': {
            'batch_size': 1000,
            'max_errors': 10,
            'timeout_seconds': 30
        }
    }
    
    # Create sample input files
    sample_data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'order_amount': np.random.uniform(10, 1000, 1000),
        'order_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'status': np.random.choice(['completed', 'pending', 'cancelled'], 1000)
    })
    
    input_file = tmp_path / "orders.csv"
    sample_data.to_csv(input_file, index=False)
    env['input_files']['orders'] = str(input_file)
    
    return env


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def create_test_rule(
    name: str,
    rule_type: RuleType,
    column_name: str,
    operator: RuleOperator = RuleOperator.EQUALS,
    value: str = "test_value",
    severity: RuleSeverity = RuleSeverity.ERROR
) -> DataQualityRule:
    """Factory function to create test rules."""
    rule = DataQualityRule(
        name=name,
        description=f"Test rule for {column_name}",
        rule_type=rule_type,
        severity=severity,
        dataset_name="test_dataset"
    )
    
    condition = RuleCondition(
        column_name=column_name,
        operator=operator,
        value=value
    )
    rule.add_condition(condition)
    
    return rule


def create_test_profile(dataset_name: str, columns: List[Dict[str, Any]]) -> DataProfile:
    """Factory function to create test profiles."""
    profile = DataProfile(dataset_name=dataset_name)
    profile.start_profiling()
    
    for col_info in columns:
        col_profile = ColumnProfile(
            column_name=col_info['name'],
            data_type=col_info.get('data_type', DataType.STRING)
        )
        
        if 'statistics' in col_info:
            stats = col_info['statistics']
            col_profile.statistics = ProfileStatistics(
                total_count=stats.get('total_count', 100),
                null_count=stats.get('null_count', 0),
                distinct_count=stats.get('distinct_count', 100)
            )
        
        profile.add_column_profile(col_profile)
    
    profile.complete_profiling()
    return profile


# ==========================================
# ASSERTION HELPERS
# ==========================================

def assert_profile_valid(profile: DataProfile):
    """Assert that a DataProfile is valid."""
    assert profile is not None, "Profile should not be None"
    assert profile.dataset_name, "Profile should have a dataset name"
    assert profile.status in [ProfileStatus.COMPLETED, ProfileStatus.IN_PROGRESS], "Profile should have valid status"
    assert len(profile.column_profiles) > 0, "Profile should have column profiles"


def assert_check_result_valid(result: CheckResult):
    """Assert that a CheckResult is valid."""
    assert result is not None, "Result should not be None"
    assert result.check_id is not None, "Result should have a check ID"
    assert result.dataset_name, "Result should have a dataset name"
    assert isinstance(result.passed, bool), "Result should have boolean passed status"
    assert 0 <= result.score <= 1, "Result score should be between 0 and 1"


def assert_rule_valid(rule: DataQualityRule):
    """Assert that a DataQualityRule is valid."""
    assert rule is not None, "Rule should not be None"
    assert rule.name, "Rule should have a name"
    assert rule.rule_type in RuleType, "Rule should have valid type"
    assert rule.severity in RuleSeverity, "Rule should have valid severity"
    assert len(rule.conditions) > 0, "Rule should have conditions"


# ==========================================
# PARAMETRIZE HELPERS
# ==========================================

# Common test parameters
RULE_TYPES = [RuleType.NOT_NULL, RuleType.UNIQUE, RuleType.RANGE, RuleType.PATTERN]
RULE_SEVERITIES = [RuleSeverity.INFO, RuleSeverity.WARNING, RuleSeverity.ERROR, RuleSeverity.CRITICAL]
DATA_TYPES = [DataType.INTEGER, DataType.FLOAT, DataType.STRING, DataType.BOOLEAN, DataType.DATE]
RULE_OPERATORS = [RuleOperator.EQUALS, RuleOperator.NOT_EQUALS, RuleOperator.GREATER_THAN, RuleOperator.LESS_THAN]


# ==========================================
# CLEANUP
# ==========================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Cleanup code runs after each test
    # Close any open resources, clean temporary files, etc.
    import gc
    gc.collect()


if __name__ == "__main__":
    # Test fixture creation
    print("Testing fixture creation...")
    
    # This can be used to validate fixtures work correctly
    sample_df = pd.DataFrame({'test': [1, 2, 3]})
    print(f"Sample DataFrame created: {len(sample_df)} rows")
    
    sample_rule = create_test_rule("test_rule", RuleType.NOT_NULL, "test_column")
    print(f"Sample rule created: {sample_rule.name}")
    
    print("All fixtures working correctly!")
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.packages.data_profiling.application.services.schema_analysis_service import SchemaAnalysisService
from src.packages.data_profiling.domain.entities.data_profile import (
    DataType, CardinalityLevel, SemanticType
)


class TestSchemaAnalysisService:
    """Test SchemaAnalysisService."""
    
    @pytest.fixture
    def service(self):
        """Create SchemaAnalysisService instance."""
        return SchemaAnalysisService()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        data = {
            'id': range(1, 1001),
            'name': [f'User_{i}' for i in range(1, 1001)],
            'email': [f'user{i}@example.com' for i in range(1, 1001)],
            'phone': [f'+1-555-{1000+i:04d}' for i in range(1, 1001)],
            'age': np.random.randint(18, 80, 1000),
            'salary': np.random.normal(50000, 15000, 1000),
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 1000),
            'is_active': np.random.choice([True, False], 1000),
            'created_at': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'score': np.random.uniform(0, 100, 1000)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mixed_types_dataframe(self):
        """Create DataFrame with mixed types for testing."""
        data = {
            'string_col': ['text1', 'text2', 'text3'] * 100,
            'int_col': list(range(300)),
            'float_col': np.random.normal(0, 1, 300),
            'bool_col': [True, False, True] * 100,
            'datetime_col': pd.date_range('2023-01-01', periods=300),
            'mixed_col': ['123', 'abc', '456'] * 100,
            'null_col': [None] * 300
        }
        return pd.DataFrame(data)


class TestSchemaInference:
    """Test schema inference functionality."""
    
    def test_infer_basic_schema(self, service, sample_dataframe):
        """Test basic schema inference."""
        schema_profile = service.infer(sample_dataframe)
        
        assert schema_profile.total_tables == 1
        assert schema_profile.total_columns == len(sample_dataframe.columns)
        assert schema_profile.total_rows == len(sample_dataframe)
        assert len(schema_profile.columns) == len(sample_dataframe.columns)
    
    def test_column_profiles_created(self, service, sample_dataframe):
        """Test that column profiles are created for all columns."""
        schema_profile = service.infer(sample_dataframe)
        
        expected_columns = set(sample_dataframe.columns)
        actual_columns = {col.column_name for col in schema_profile.columns}
        
        assert expected_columns == actual_columns
    
    def test_data_type_inference(self, service, mixed_types_dataframe):
        """Test data type inference for different column types."""
        schema_profile = service.infer(mixed_types_dataframe)
        
        column_types = {col.column_name: col.data_type for col in schema_profile.columns}
        
        assert column_types['string_col'] == DataType.STRING
        assert column_types['int_col'] == DataType.INTEGER
        assert column_types['float_col'] == DataType.FLOAT
        assert column_types['bool_col'] == DataType.BOOLEAN
        assert column_types['datetime_col'] == DataType.DATETIME
        # mixed_col should be inferred as string since it contains mixed values
        assert column_types['mixed_col'] == DataType.STRING
    
    def test_cardinality_inference(self, service, sample_dataframe):
        """Test cardinality level inference."""
        schema_profile = service.infer(sample_dataframe)
        
        column_cardinalities = {col.column_name: col.cardinality_level for col in schema_profile.columns}
        
        # ID should be high cardinality (unique values)
        assert column_cardinalities['id'] == CardinalityLevel.HIGH
        
        # Department should be low cardinality (few unique values)
        assert column_cardinalities['department'] == CardinalityLevel.LOW
        
        # Boolean column should be low cardinality
        assert column_cardinalities['is_active'] == CardinalityLevel.LOW
    
    def test_semantic_type_inference(self, service, sample_dataframe):
        """Test semantic type inference."""
        schema_profile = service.infer(sample_dataframe)
        
        column_semantic_types = {col.column_name: col.semantic_type for col in schema_profile.columns}
        
        # Email should be detected as PII_EMAIL
        assert column_semantic_types['email'] == SemanticType.PII_EMAIL
        
        # Phone should be detected as PII_PHONE
        assert column_semantic_types['phone'] == SemanticType.PII_PHONE
        
        # ID should be detected as IDENTIFIER
        assert column_semantic_types['id'] == SemanticType.IDENTIFIER
        
        # Salary should be detected as FINANCIAL_AMOUNT
        assert column_semantic_types['salary'] == SemanticType.FINANCIAL_AMOUNT
        
        # Created_at should be detected as TIMESTAMP
        assert column_semantic_types['created_at'] == SemanticType.TIMESTAMP


class TestSemanticTypeDetection:
    """Test semantic type detection functionality."""
    
    def test_pii_email_detection(self, service):
        """Test email PII detection."""
        email_series = pd.Series(['user@example.com', 'test@domain.org', 'admin@company.net'])
        
        # Mock the analyze_column method to focus on semantic type detection
        column_profile = service._analyze_column(email_series, 'email_col', len(email_series))
        
        assert column_profile.semantic_type == SemanticType.PII_EMAIL
    
    def test_pii_phone_detection(self, service):
        """Test phone PII detection."""
        phone_series = pd.Series(['+1-555-1234', '+1-555-5678', '+1-555-9012'])
        
        column_profile = service._analyze_column(phone_series, 'phone_col', len(phone_series))
        
        assert column_profile.semantic_type == SemanticType.PII_PHONE
    
    def test_identifier_detection(self, service):
        """Test identifier detection."""
        id_series = pd.Series(range(1, 1001))
        
        column_profile = service._analyze_column(id_series, 'id', len(id_series))
        
        assert column_profile.semantic_type == SemanticType.IDENTIFIER
    
    def test_financial_amount_detection(self, service):
        """Test financial amount detection."""
        salary_series = pd.Series(np.random.normal(50000, 15000, 1000))
        
        column_profile = service._analyze_column(salary_series, 'salary', len(salary_series))
        
        assert column_profile.semantic_type == SemanticType.FINANCIAL_AMOUNT
    
    def test_timestamp_detection(self, service):
        """Test timestamp detection."""
        timestamp_series = pd.date_range('2023-01-01', periods=1000)
        
        column_profile = service._analyze_column(timestamp_series, 'created_at', len(timestamp_series))
        
        assert column_profile.semantic_type == SemanticType.TIMESTAMP


class TestAdvancedRelationships:
    """Test advanced relationship detection."""
    
    def test_detect_advanced_relationships(self, service, sample_dataframe):
        """Test advanced relationship detection."""
        relationships = service.detect_advanced_relationships(sample_dataframe)
        
        assert 'functional_dependencies' in relationships
        assert 'correlation_relationships' in relationships
        assert 'hierarchical_relationships' in relationships
        assert 'categorical_relationships' in relationships
        assert 'temporal_relationships' in relationships
        assert 'inclusion_dependencies' in relationships
    
    def test_functional_dependency_detection(self, service):
        """Test functional dependency detection."""
        # Create data with clear functional dependency (employee_id -> name)
        data = {
            'employee_id': [1, 2, 3, 1, 2, 3] * 50,
            'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Charlie'] * 50,
            'department': ['Eng', 'Sales', 'HR', 'Eng', 'Sales', 'HR'] * 50,
            'random_value': np.random.random(300)
        }
        df = pd.DataFrame(data)
        
        functional_deps = service._detect_functional_dependencies(df)
        
        assert isinstance(functional_deps, list)
        # Should detect that employee_id determines name and department
    
    def test_correlation_relationship_detection(self, service, sample_dataframe):
        """Test correlation relationship detection."""
        correlation_relationships = service._detect_correlation_relationships(sample_dataframe)
        
        assert isinstance(correlation_relationships, list)
        # Should find correlations between numeric columns
    
    def test_hierarchical_relationship_detection(self, service):
        """Test hierarchical relationship detection."""
        # Create hierarchical data
        data = {
            'country': ['USA', 'USA', 'Canada', 'Canada'] * 25,
            'state': ['CA', 'NY', 'ON', 'BC'] * 25,
            'city': ['LA', 'NYC', 'Toronto', 'Vancouver'] * 25
        }
        df = pd.DataFrame(data)
        
        hierarchical_relationships = service._detect_hierarchical_relationships(df)
        
        assert isinstance(hierarchical_relationships, list)
    
    def test_categorical_relationship_detection(self, service, sample_dataframe):
        """Test categorical relationship detection."""
        categorical_relationships = service._detect_categorical_relationships(sample_dataframe)
        
        assert isinstance(categorical_relationships, list)
    
    def test_temporal_relationship_detection(self, service, sample_dataframe):
        """Test temporal relationship detection."""
        temporal_relationships = service._detect_temporal_relationships(sample_dataframe)
        
        assert isinstance(temporal_relationships, list)
    
    def test_inclusion_dependency_detection(self, service):
        """Test inclusion dependency detection."""
        # Create data with inclusion dependencies
        data = {
            'all_employees': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'managers': ['Alice', 'Bob'],  # Subset of all_employees
            'random_names': ['Frank', 'Grace', 'Henry', 'Ivy', 'Jack']
        }
        # Repeat to have sufficient data
        df = pd.DataFrame({
            col: values * 20 for col, values in data.items()
        })
        
        inclusion_dependencies = service._detect_inclusion_dependencies(df)
        
        assert isinstance(inclusion_dependencies, list)


class TestConstraintDetection:
    """Test constraint detection functionality."""
    
    def test_detect_primary_keys(self, service, sample_dataframe):
        """Test primary key detection."""
        column_profiles = [
            service._analyze_column(sample_dataframe[col], col, len(sample_dataframe))
            for col in sample_dataframe.columns
        ]
        
        primary_keys = service._detect_primary_keys(sample_dataframe, column_profiles)
        
        assert isinstance(primary_keys, list)
        # ID column should be detected as primary key (unique values)
        assert 'id' in primary_keys
    
    def test_detect_foreign_keys(self, service, sample_dataframe):
        """Test foreign key detection."""
        column_profiles = [
            service._analyze_column(sample_dataframe[col], col, len(sample_dataframe))
            for col in sample_dataframe.columns
        ]
        
        foreign_keys = service._detect_foreign_keys(sample_dataframe, column_profiles)
        
        assert isinstance(foreign_keys, dict)
    
    def test_detect_constraints(self, service, sample_dataframe):
        """Test general constraint detection."""
        column_profiles = [
            service._analyze_column(sample_dataframe[col], col, len(sample_dataframe))
            for col in sample_dataframe.columns
        ]
        
        constraints = service._detect_constraints(sample_dataframe, column_profiles)
        
        assert isinstance(constraints, list)
    
    def test_recommend_indexes(self, service, sample_dataframe):
        """Test index recommendation."""
        column_profiles = [
            service._analyze_column(sample_dataframe[col], col, len(sample_dataframe))
            for col in sample_dataframe.columns
        ]
        
        indexes = service._recommend_indexes(sample_dataframe, column_profiles)
        
        assert isinstance(indexes, list)


class TestSizeMetrics:
    """Test size metrics calculation."""
    
    def test_calculate_size_metrics(self, service, sample_dataframe):
        """Test size metrics calculation."""
        size_metrics = service._calculate_size_metrics(sample_dataframe)
        
        assert size_metrics is not None
        assert hasattr(size_metrics, 'estimated_size_bytes')
        assert hasattr(size_metrics, 'memory_usage_mb')
        assert size_metrics.estimated_size_bytes > 0


class TestTableRelationships:
    """Test table relationship detection."""
    
    def test_detect_table_relationships(self, service, sample_dataframe):
        """Test table relationship detection."""
        column_profiles = [
            service._analyze_column(sample_dataframe[col], col, len(sample_dataframe))
            for col in sample_dataframe.columns
        ]
        
        relationships = service._detect_table_relationships(sample_dataframe, column_profiles)
        
        assert isinstance(relationships, list)


class TestPrivacyDetection:
    """Test privacy and PII detection."""
    
    def test_pii_pattern_matching(self, service):
        """Test PII pattern matching."""
        # Test email pattern
        email_pattern = service.pii_patterns[SemanticType.PII_EMAIL]
        assert email_pattern.match('user@example.com') is not None
        assert email_pattern.match('invalid-email') is None
        
        # Test phone pattern
        phone_pattern = service.pii_patterns[SemanticType.PII_PHONE]
        assert phone_pattern.match('+1-555-1234') is not None
        assert phone_pattern.match('not-a-phone') is None
        
        # Test SSN pattern
        ssn_pattern = service.pii_patterns[SemanticType.PII_SSN]
        assert ssn_pattern.match('123-45-6789') is not None
        assert ssn_pattern.match('123456789') is None
    
    def test_semantic_indicator_matching(self, service):
        """Test semantic indicator matching."""
        # Test financial amount indicators
        financial_indicators = service.semantic_indicators[SemanticType.FINANCIAL_AMOUNT]
        
        assert any(indicator in 'salary_amount' for indicator in financial_indicators)
        assert any(indicator in 'total_price' for indicator in financial_indicators)
        
        # Test geographic location indicators
        geo_indicators = service.semantic_indicators[SemanticType.GEOGRAPHIC_LOCATION]
        
        assert any(indicator in 'user_country' for indicator in geo_indicators)
        assert any(indicator in 'billing_state' for indicator in geo_indicators)


class TestColumnAnalysis:
    """Test individual column analysis."""
    
    def test_analyze_column_complete(self, service):
        """Test complete column analysis."""
        test_series = pd.Series([1, 2, 3, 4, 5, 1, 2, 3])
        
        column_profile = service._analyze_column(test_series, 'test_col', len(test_series))
        
        assert column_profile.column_name == 'test_col'
        assert column_profile.data_type == DataType.INTEGER
        assert column_profile.cardinality_level is not None
        assert column_profile.semantic_type is not None
        assert column_profile.is_nullable is not None
        assert column_profile.is_primary_key is not None
        assert column_profile.is_foreign_key is not None
    
    def test_analyze_empty_column(self, service):
        """Test analysis of empty column."""
        empty_series = pd.Series([], dtype=object)
        
        column_profile = service._analyze_column(empty_series, 'empty_col', 0)
        
        assert column_profile.column_name == 'empty_col'
        assert column_profile.data_type == DataType.UNKNOWN
    
    def test_analyze_null_column(self, service):
        """Test analysis of column with only null values."""
        null_series = pd.Series([None, None, None])
        
        column_profile = service._analyze_column(null_series, 'null_col', len(null_series))
        
        assert column_profile.column_name == 'null_col'
        assert column_profile.is_nullable is True


class TestSchemaEvolution:
    """Test schema evolution tracking."""
    
    def test_schema_evolution_creation(self, service, sample_dataframe):
        """Test schema evolution object creation."""
        schema_profile = service.infer(sample_dataframe)
        
        assert schema_profile.schema_evolution is not None
        assert schema_profile.schema_evolution.current_version == "1.0.0"
        assert isinstance(schema_profile.schema_evolution.changes_detected, list)
        assert isinstance(schema_profile.schema_evolution.breaking_changes, list)
        assert schema_profile.schema_evolution.last_change_date is not None
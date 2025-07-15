"""Comprehensive test suite for Data Quality Validation Engine.

Tests for validation engine, data cleansing service, and comprehensive quality scoring engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import the services we're testing
from src.packages.data_quality.application.services.validation_engine import (
    ValidationEngine, ValidationEngineConfig
)
from src.packages.data_quality.application.services.data_cleansing_service import (
    DataCleansingService, DataCleansingConfig, CleansingRule, CleansingAction, CleansingStrategy
)
from src.packages.data_quality.application.services.comprehensive_quality_scoring_engine import (
    ComprehensiveQualityScoringEngine, ComprehensiveQualityScoringConfig, 
    QualityDimension, QualityContext, ScoringAlgorithm
)
from src.packages.data_quality.application.services.rule_management_service import (
    RuleManagementService, RuleTemplate, RuleTestStatus
)

# Import domain entities
from src.packages.data_quality.domain.entities.validation_rule import (
    QualityRule, ValidationLogic, ValidationResult, ValidationError,
    RuleId, RuleType, LogicType, Severity, QualityCategory, 
    SuccessCriteria, UserId, ValidationStatus
)
from src.packages.data_quality.domain.entities.quality_profile import DatasetId


class TestValidationEngine:
    """Test suite for Validation Engine."""
    
    @pytest.fixture
    def validation_config(self):
        """Validation engine configuration."""
        return ValidationEngineConfig(
            enable_parallel_processing=True,
            max_workers=2,
            timeout_seconds=30,
            enable_caching=True,
            chunk_size=1000
        )
    
    @pytest.fixture
    def validation_engine(self, validation_config):
        """Validation engine instance."""
        return ValidationEngine(validation_config)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'id': range(1, 101),
            'name': [f'Name_{i}' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'salary': np.random.randint(30000, 120000, 100),
            'email': [f'user{i}@example.com' for i in range(1, 101)],
            'active': np.random.choice([True, False], 100),
            'created_at': pd.date_range('2020-01-01', periods=100, freq='D')
        })
    
    @pytest.fixture
    def sample_rules(self):
        """Sample validation rules."""
        return [
            QualityRule(
                rule_id=RuleId(),
                rule_name="Age Range Check",
                description="Check if age is within valid range",
                rule_type=RuleType.RANGE,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.STATISTICAL,
                    expression="age >= 18 and age <= 100",
                    parameters={'column_name': 'age', 'stat_type': 'range', 'min_value': 18, 'max_value': 100},
                    success_criteria=SuccessCriteria(min_pass_rate=0.95),
                    error_message="Age must be between 18 and 100"
                ),
                target_columns=['age'],
                severity=Severity.HIGH,
                category=QualityCategory.BUSINESS_RULES,
                created_by=UserId("test_user"),
                is_active=True
            ),
            QualityRule(
                rule_id=RuleId(),
                rule_name="Email Format Check",
                description="Check email format",
                rule_type=RuleType.PATTERN,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.REGEX,
                    expression=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    parameters={'column_name': 'email'},
                    success_criteria=SuccessCriteria(min_pass_rate=0.90),
                    error_message="Invalid email format"
                ),
                target_columns=['email'],
                severity=Severity.MEDIUM,
                category=QualityCategory.DATA_INTEGRITY,
                created_by=UserId("test_user"),
                is_active=True
            ),
            QualityRule(
                rule_id=RuleId(),
                rule_name="Salary Minimum Check",
                description="Check minimum salary",
                rule_type=RuleType.RANGE,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.COMPARISON,
                    expression="salary >= 25000",
                    parameters={'column1': 'salary', 'operator': '>=', 'value': 25000},
                    success_criteria=SuccessCriteria(min_pass_rate=0.85),
                    error_message="Salary must be at least 25000"
                ),
                target_columns=['salary'],
                severity=Severity.MEDIUM,
                category=QualityCategory.BUSINESS_RULES,
                created_by=UserId("test_user"),
                is_active=True
            )
        ]
    
    def test_validation_engine_initialization(self, validation_engine):
        """Test validation engine initialization."""
        assert validation_engine is not None
        assert validation_engine.config.enable_parallel_processing is True
        assert validation_engine.config.max_workers == 2
        assert validation_engine.config.timeout_seconds == 30
        assert validation_engine._cache is not None
    
    def test_validate_dataset_sequential(self, validation_engine, sample_dataframe, sample_rules):
        """Test sequential dataset validation."""
        # Disable parallel processing for this test
        validation_engine.config.enable_parallel_processing = False
        
        dataset_id = DatasetId("test_dataset")
        results = validation_engine.validate_dataset(sample_dataframe, sample_rules, dataset_id)
        
        assert len(results) == len(sample_rules)
        
        # Check that all rules were executed
        for result in results:
            assert result.dataset_id == dataset_id
            assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED, ValidationStatus.WARNING]
            assert result.total_records == len(sample_dataframe)
            assert result.passed_records + result.failed_records == result.total_records
    
    def test_validate_dataset_parallel(self, validation_engine, sample_dataframe, sample_rules):
        """Test parallel dataset validation."""
        # Enable parallel processing
        validation_engine.config.enable_parallel_processing = True
        
        dataset_id = DatasetId("test_dataset_parallel")
        results = validation_engine.validate_dataset(sample_dataframe, sample_rules, dataset_id)
        
        assert len(results) == len(sample_rules)
        
        # Check that all rules were executed
        for result in results:
            assert result.dataset_id == dataset_id
            assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED, ValidationStatus.WARNING]
    
    def test_validate_single_rule(self, validation_engine, sample_dataframe, sample_rules):
        """Test single rule validation."""
        dataset_id = DatasetId("test_single_rule")
        rule = sample_rules[0]  # Age range check
        
        result = validation_engine.validate_single_rule(sample_dataframe, rule, dataset_id)
        
        assert result.rule_id == rule.rule_id
        assert result.dataset_id == dataset_id
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED, ValidationStatus.WARNING]
        assert result.total_records == len(sample_dataframe)
    
    def test_python_rule_execution(self, validation_engine, sample_dataframe):
        """Test Python rule execution."""
        python_rule = QualityRule(
            rule_id=RuleId(),
            rule_name="Python Age Check",
            description="Check age using Python expression",
            rule_type=RuleType.CUSTOM,
            validation_logic=ValidationLogic(
                logic_type=LogicType.PYTHON,
                expression="age >= 18",
                parameters={'row_wise': True},
                success_criteria=SuccessCriteria(min_pass_rate=0.95),
                error_message="Age must be >= 18"
            ),
            target_columns=['age'],
            severity=Severity.HIGH,
            category=QualityCategory.BUSINESS_RULES,
            created_by=UserId("test_user"),
            is_active=True
        )
        
        dataset_id = DatasetId("test_python_rule")
        result = validation_engine.validate_single_rule(sample_dataframe, python_rule, dataset_id)
        
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED, ValidationStatus.WARNING]
        assert result.total_records == len(sample_dataframe)
    
    def test_regex_rule_execution(self, validation_engine, sample_dataframe, sample_rules):
        """Test regex rule execution."""
        regex_rule = sample_rules[1]  # Email format check
        
        dataset_id = DatasetId("test_regex_rule")
        result = validation_engine.validate_single_rule(sample_dataframe, regex_rule, dataset_id)
        
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED, ValidationStatus.WARNING]
        assert result.total_records == len(sample_dataframe)
    
    def test_statistical_rule_execution(self, validation_engine, sample_dataframe):
        """Test statistical rule execution."""
        statistical_rule = QualityRule(
            rule_id=RuleId(),
            rule_name="Salary Outlier Check",
            description="Check for salary outliers",
            rule_type=RuleType.VALIDITY,
            validation_logic=ValidationLogic(
                logic_type=LogicType.STATISTICAL,
                expression="outlier_detection",
                parameters={'column_name': 'salary', 'stat_type': 'outlier'},
                success_criteria=SuccessCriteria(min_pass_rate=0.85),
                error_message="Salary outlier detected"
            ),
            target_columns=['salary'],
            severity=Severity.LOW,
            category=QualityCategory.DATA_INTEGRITY,
            created_by=UserId("test_user"),
            is_active=True
        )
        
        dataset_id = DatasetId("test_statistical_rule")
        result = validation_engine.validate_single_rule(sample_dataframe, statistical_rule, dataset_id)
        
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED, ValidationStatus.WARNING]
    
    def test_caching_functionality(self, validation_engine, sample_dataframe, sample_rules):
        """Test caching functionality."""
        # Enable caching
        validation_engine.config.enable_caching = True
        
        dataset_id = DatasetId("test_caching")
        rule = sample_rules[0]
        
        # First execution - should miss cache
        result1 = validation_engine.validate_single_rule(sample_dataframe, rule, dataset_id)
        
        # Second execution - should hit cache
        result2 = validation_engine.validate_single_rule(sample_dataframe, rule, dataset_id)
        
        # Results should be the same
        assert result1.status == result2.status
        assert result1.passed_records == result2.passed_records
        assert result1.failed_records == result2.failed_records
    
    def test_execution_metrics(self, validation_engine, sample_dataframe, sample_rules):
        """Test execution metrics collection."""
        dataset_id = DatasetId("test_metrics")
        
        # Execute some validations
        validation_engine.validate_dataset(sample_dataframe, sample_rules, dataset_id)
        
        # Get metrics
        metrics = validation_engine.get_execution_metrics()
        
        assert 'total_validations' in metrics
        assert 'successful_validations' in metrics
        assert 'failed_validations' in metrics
        assert 'total_execution_time' in metrics
        assert 'success_rate' in metrics
        assert 'failure_rate' in metrics
        assert metrics['total_validations'] > 0
    
    def test_error_handling(self, validation_engine, sample_dataframe):
        """Test error handling for invalid rules."""
        invalid_rule = QualityRule(
            rule_id=RuleId(),
            rule_name="Invalid Rule",
            description="Rule with invalid logic",
            rule_type=RuleType.CUSTOM,
            validation_logic=ValidationLogic(
                logic_type=LogicType.PYTHON,
                expression="invalid_python_code(",
                parameters={},
                success_criteria=SuccessCriteria(min_pass_rate=0.95),
                error_message="Invalid rule"
            ),
            target_columns=['age'],
            severity=Severity.HIGH,
            category=QualityCategory.BUSINESS_RULES,
            created_by=UserId("test_user"),
            is_active=True
        )
        
        dataset_id = DatasetId("test_error_handling")
        result = validation_engine.validate_single_rule(sample_dataframe, invalid_rule, dataset_id)
        
        assert result.status == ValidationStatus.ERROR
        assert len(result.error_details) > 0


class TestDataCleansingService:
    """Test suite for Data Cleansing Service."""
    
    @pytest.fixture
    def cleansing_config(self):
        """Data cleansing service configuration."""
        return DataCleansingConfig(
            assess_quality_before=True,
            assess_quality_after=True,
            create_backup=True,
            validate_after_cleansing=True,
            detailed_reporting=True
        )
    
    @pytest.fixture
    def cleansing_service(self, cleansing_config):
        """Data cleansing service instance."""
        return DataCleansingService(cleansing_config)
    
    @pytest.fixture
    def dirty_dataframe(self):
        """DataFrame with quality issues for testing."""
        np.random.seed(42)
        
        # Create data with various quality issues
        data = {
            'id': list(range(1, 91)) + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Duplicates
            'name': ['John Doe', 'Jane Smith', '  Bob Johnson  ', 'alice brown', 'CHARLIE WILSON'] * 20,
            'age': list(range(18, 80)) + [150, 200, -5, -10, None, None, None, None, None, None],  # Outliers and nulls
            'salary': list(range(30000, 92000)) + [0, -1000, 500000, 1000000, None, None, None, None, None, None],
            'email': ['john@example.com', 'jane@test.com', 'invalid-email', 'bob@domain', 'alice@.com'] * 20,
            'phone': ['555-1234', '555-5678', '123', 'invalid', '', '555-9999'] * 16 + ['555-1111', '555-2222', '555-3333', '555-4444'],
            'status': ['active', 'inactive', 'ACTIVE', 'Inactive', 'unknown'] * 20
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def cleansing_rules(self):
        """Sample cleansing rules."""
        return [
            CleansingRule(
                action=CleansingAction.MISSING_VALUES,
                strategy=CleansingStrategy.IMPUTE,
                target_columns=['age', 'salary'],
                parameters={'impute_method': 'median'},
                priority=3
            ),
            CleansingRule(
                action=CleansingAction.DUPLICATES,
                strategy=CleansingStrategy.REMOVE,
                target_columns=[],
                parameters={'keep': 'first'},
                priority=2
            ),
            CleansingRule(
                action=CleansingAction.OUTLIERS,
                strategy=CleansingStrategy.TRANSFORM,
                target_columns=['age', 'salary'],
                parameters={'outlier_method': 'iqr'},
                priority=1
            ),
            CleansingRule(
                action=CleansingAction.WHITESPACE,
                strategy=CleansingStrategy.STANDARDIZE,
                target_columns=['name'],
                priority=4
            ),
            CleansingRule(
                action=CleansingAction.INCONSISTENT_CASING,
                strategy=CleansingStrategy.STANDARDIZE,
                target_columns=['name', 'status'],
                parameters={'case_method': 'lower'},
                priority=5
            )
        ]
    
    def test_cleansing_service_initialization(self, cleansing_service):
        """Test cleansing service initialization."""
        assert cleansing_service is not None
        assert cleansing_service.config.assess_quality_before is True
        assert len(cleansing_service._processors) > 0
        assert len(cleansing_service._processor_map) > 0
    
    def test_cleanse_dataset(self, cleansing_service, dirty_dataframe, cleansing_rules):
        """Test dataset cleansing with rules."""
        original_length = len(dirty_dataframe)
        
        cleaned_df, result = cleansing_service.cleanse_dataset(
            dirty_dataframe, cleansing_rules, DatasetId("test_cleansing")
        )
        
        assert result.original_records == original_length
        assert result.cleaned_records > 0
        assert len(result.actions_applied) > 0
        assert result.retention_rate > 0
        assert result.execution_time_seconds > 0
        
        # Check that data was actually cleaned
        assert len(cleaned_df) <= original_length  # Some records may be removed
        assert cleaned_df.isnull().sum().sum() < dirty_dataframe.isnull().sum().sum()  # Fewer nulls
    
    def test_missing_values_processing(self, cleansing_service, dirty_dataframe):
        """Test missing values processing."""
        rules = [
            CleansingRule(
                action=CleansingAction.MISSING_VALUES,
                strategy=CleansingStrategy.IMPUTE,
                target_columns=['age'],
                parameters={'impute_method': 'median'},
                priority=1
            )
        ]
        
        cleaned_df, result = cleansing_service.cleanse_dataset(dirty_dataframe, rules)
        
        # Check that age column has no missing values
        assert cleaned_df['age'].isnull().sum() == 0
        assert result.modified_records > 0
    
    def test_duplicate_removal(self, cleansing_service, dirty_dataframe):
        """Test duplicate removal."""
        rules = [
            CleansingRule(
                action=CleansingAction.DUPLICATES,
                strategy=CleansingStrategy.REMOVE,
                target_columns=[],
                parameters={'keep': 'first'},
                priority=1
            )
        ]
        
        original_length = len(dirty_dataframe)
        cleaned_df, result = cleansing_service.cleanse_dataset(dirty_dataframe, rules)
        
        # Should have fewer records due to duplicate removal
        assert len(cleaned_df) < original_length
        assert result.removed_records > 0
    
    def test_outlier_processing(self, cleansing_service, dirty_dataframe):
        """Test outlier processing."""
        rules = [
            CleansingRule(
                action=CleansingAction.OUTLIERS,
                strategy=CleansingStrategy.TRANSFORM,
                target_columns=['age'],
                parameters={'outlier_method': 'iqr'},
                priority=1
            )
        ]
        
        cleaned_df, result = cleansing_service.cleanse_dataset(dirty_dataframe, rules)
        
        # Check that extreme outliers are handled
        assert cleaned_df['age'].max() <= 100  # Should cap extreme values
        assert cleaned_df['age'].min() >= 0    # Should fix negative values
        assert result.modified_records > 0
    
    def test_format_standardization(self, cleansing_service, dirty_dataframe):
        """Test format standardization."""
        rules = [
            CleansingRule(
                action=CleansingAction.WHITESPACE,
                strategy=CleansingStrategy.STANDARDIZE,
                target_columns=['name'],
                priority=1
            )
        ]
        
        cleaned_df, result = cleansing_service.cleanse_dataset(dirty_dataframe, rules)
        
        # Check that whitespace is cleaned
        for name in cleaned_df['name'].dropna():
            assert name == name.strip()
        
        assert result.modified_records > 0
    
    def test_create_default_rules(self, cleansing_service, dirty_dataframe):
        """Test creation of default cleansing rules."""
        rules = cleansing_service.create_default_cleansing_rules(dirty_dataframe)
        
        assert len(rules) > 0
        assert any(rule.action == CleansingAction.MISSING_VALUES for rule in rules)
        assert any(rule.action == CleansingAction.DUPLICATES for rule in rules)
        assert any(rule.action == CleansingAction.OUTLIERS for rule in rules)
        assert any(rule.action == CleansingAction.WHITESPACE for rule in rules)
    
    def test_recommend_cleansing_rules(self, cleansing_service, dirty_dataframe):
        """Test rule recommendations."""
        rules = cleansing_service.recommend_cleansing_rules(dirty_dataframe, quality_threshold=0.8)
        
        assert len(rules) > 0
        
        # Should recommend rules based on detected issues
        has_missing_rule = any(rule.action == CleansingAction.MISSING_VALUES for rule in rules)
        has_duplicate_rule = any(rule.action == CleansingAction.DUPLICATES for rule in rules)
        
        assert has_missing_rule or has_duplicate_rule
    
    def test_processor_info(self, cleansing_service):
        """Test processor information retrieval."""
        info = cleansing_service.get_processor_info()
        
        assert 'total_processors' in info
        assert 'supported_actions' in info
        assert 'supported_strategies' in info
        assert 'processor_capabilities' in info
        assert info['total_processors'] > 0


class TestComprehensiveQualityScoringEngine:
    """Test suite for Comprehensive Quality Scoring Engine."""
    
    @pytest.fixture
    def scoring_config(self):
        """Quality scoring engine configuration."""
        return ComprehensiveQualityScoringConfig(
            scoring_algorithm=ScoringAlgorithm.WEIGHTED_AVERAGE,
            enable_benchmarking=True,
            enable_trend_analysis=True,
            detailed_reporting=True
        )
    
    @pytest.fixture
    def scoring_engine(self, scoring_config):
        """Quality scoring engine instance."""
        return ComprehensiveQualityScoringEngine(scoring_config)
    
    @pytest.fixture
    def sample_dataframe_for_scoring(self):
        """Sample DataFrame for quality scoring."""
        np.random.seed(42)
        return pd.DataFrame({
            'id': range(1, 101),
            'name': [f'Name_{i}' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'salary': np.random.randint(30000, 120000, 100),
            'email': [f'user{i}@example.com' for i in range(1, 101)],
            'active': np.random.choice([True, False], 100),
            'score': np.random.uniform(0, 100, 100)
        })
    
    @pytest.fixture
    def quality_context(self):
        """Quality context for scoring."""
        return QualityContext(
            business_domain="finance",
            data_sensitivity="high",
            regulatory_requirements=["GDPR", "SOX"],
            usage_patterns={
                'critical_columns': ['id', 'email'],
                'consistency_rules': {
                    'age_salary_correlation': {
                        'columns': ['age', 'salary'],
                        'type': 'correlation'
                    }
                }
            }
        )
    
    @pytest.fixture
    def mock_validation_results(self):
        """Mock validation results."""
        return [
            ValidationResult(
                validation_id="val_1",
                rule_id=RuleId(),
                dataset_id=DatasetId("test"),
                status=ValidationStatus.PASSED,
                passed_records=95,
                failed_records=5,
                failure_rate=0.05,
                total_records=100,
                execution_time=timedelta(seconds=1),
                validated_at=datetime.now()
            ),
            ValidationResult(
                validation_id="val_2",
                rule_id=RuleId(),
                dataset_id=DatasetId("test"),
                status=ValidationStatus.FAILED,
                passed_records=80,
                failed_records=20,
                failure_rate=0.20,
                total_records=100,
                execution_time=timedelta(seconds=2),
                validated_at=datetime.now()
            )
        ]
    
    def test_scoring_engine_initialization(self, scoring_engine):
        """Test scoring engine initialization."""
        assert scoring_engine is not None
        assert scoring_engine.config.scoring_algorithm == ScoringAlgorithm.WEIGHTED_AVERAGE
        assert len(scoring_engine._calculators) > 0
        assert len(scoring_engine._weights_lookup) > 0
    
    def test_calculate_comprehensive_score(self, scoring_engine, sample_dataframe_for_scoring, 
                                         quality_context, mock_validation_results):
        """Test comprehensive score calculation."""
        score = scoring_engine.calculate_comprehensive_score(
            sample_dataframe_for_scoring, 
            "test_dataset", 
            mock_validation_results, 
            quality_context
        )
        
        assert score is not None
        assert score.dataset_id == "test_dataset"
        assert 0 <= score.overall_score <= 1
        assert 0 <= score.weighted_score <= 1
        assert 0 <= score.confidence <= 1
        assert len(score.dimension_scores) > 0
        assert score.calculation_duration_ms > 0
    
    def test_dimension_score_calculation(self, scoring_engine, sample_dataframe_for_scoring, 
                                       quality_context, mock_validation_results):
        """Test individual dimension score calculation."""
        score = scoring_engine.calculate_comprehensive_score(
            sample_dataframe_for_scoring, 
            "test_dataset", 
            mock_validation_results, 
            quality_context
        )
        
        # Check that key dimensions are calculated
        assert QualityDimension.COMPLETENESS in score.dimension_scores
        assert QualityDimension.ACCURACY in score.dimension_scores
        assert QualityDimension.CONSISTENCY in score.dimension_scores
        
        # Check dimension score properties
        for dimension, dim_score in score.dimension_scores.items():
            assert 0 <= dim_score.score <= 1
            assert 0 <= dim_score.confidence <= 1
            assert dim_score.weight > 0
            assert dim_score.dimension == dimension
    
    def test_scoring_algorithms(self, scoring_engine, sample_dataframe_for_scoring, quality_context):
        """Test different scoring algorithms."""
        # Test weighted average
        scoring_engine.config.scoring_algorithm = ScoringAlgorithm.WEIGHTED_AVERAGE
        score_weighted = scoring_engine.calculate_comprehensive_score(
            sample_dataframe_for_scoring, "test_weighted", [], quality_context
        )
        
        # Test geometric mean
        scoring_engine.config.scoring_algorithm = ScoringAlgorithm.GEOMETRIC_MEAN
        score_geometric = scoring_engine.calculate_comprehensive_score(
            sample_dataframe_for_scoring, "test_geometric", [], quality_context
        )
        
        # Test harmonic mean
        scoring_engine.config.scoring_algorithm = ScoringAlgorithm.HARMONIC_MEAN
        score_harmonic = scoring_engine.calculate_comprehensive_score(
            sample_dataframe_for_scoring, "test_harmonic", [], quality_context
        )
        
        # All should produce valid scores
        assert 0 <= score_weighted.overall_score <= 1
        assert 0 <= score_geometric.overall_score <= 1
        assert 0 <= score_harmonic.overall_score <= 1
        
        # Harmonic mean should typically be most conservative
        assert score_harmonic.overall_score <= score_weighted.overall_score
    
    def test_quality_dashboard_generation(self, scoring_engine, sample_dataframe_for_scoring, 
                                        quality_context, mock_validation_results):
        """Test quality dashboard data generation."""
        score = scoring_engine.calculate_comprehensive_score(
            sample_dataframe_for_scoring, 
            "test_dashboard", 
            mock_validation_results, 
            quality_context
        )
        
        dashboard = score.get_quality_dashboard()
        
        assert 'dataset_id' in dashboard
        assert 'overall_assessment' in dashboard
        assert 'dimensional_breakdown' in dashboard
        assert 'critical_analysis' in dashboard
        assert 'business_context' in dashboard
        assert 'metadata' in dashboard
        
        # Check overall assessment
        overall = dashboard['overall_assessment']
        assert 'score' in overall
        assert 'quality_level' in overall
        assert 'confidence' in overall
        
        # Check dimensional breakdown
        dimensional = dashboard['dimensional_breakdown']
        assert len(dimensional) > 0
        
        for dim_name, dim_data in dimensional.items():
            assert 'score' in dim_data
            assert 'quality_level' in dim_data
            assert 'is_critical' in dim_data
    
    def test_business_impact_calculation(self, scoring_engine, sample_dataframe_for_scoring, 
                                       quality_context, mock_validation_results):
        """Test business impact calculation."""
        score = scoring_engine.calculate_comprehensive_score(
            sample_dataframe_for_scoring, 
            "test_business_impact", 
            mock_validation_results, 
            quality_context
        )
        
        assert 0 <= score.business_impact_score <= 1
        assert len(score.risk_assessment) > 0
        assert 'overall_risk' in score.risk_assessment
        assert 'risk_factors' in score.risk_assessment
    
    def test_improvement_opportunities(self, scoring_engine, sample_dataframe_for_scoring, 
                                     quality_context, mock_validation_results):
        """Test improvement opportunities identification."""
        score = scoring_engine.calculate_comprehensive_score(
            sample_dataframe_for_scoring, 
            "test_improvement", 
            mock_validation_results, 
            quality_context
        )
        
        opportunities = score.get_top_improvement_opportunities()
        
        assert len(opportunities) > 0
        assert len(opportunities) <= 5  # Top 5
        
        # Check that opportunities are sorted by potential impact
        for i in range(len(opportunities) - 1):
            assert opportunities[i][1] >= opportunities[i + 1][1]
    
    def test_critical_dimensions_identification(self, scoring_engine):
        """Test critical dimensions identification."""
        # Create a DataFrame with obvious quality issues
        low_quality_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': [None, None, None, None, None],  # All missing
            'age': [150, 200, -5, -10, 999],          # All outliers
            'email': ['invalid', 'bad', 'wrong', 'nope', 'fail']  # All invalid
        })
        
        score = scoring_engine.calculate_comprehensive_score(
            low_quality_df, 
            "test_critical", 
            [], 
            QualityContext()
        )
        
        critical_dims = score.get_critical_dimensions()
        
        # Should identify critical dimensions
        assert len(critical_dims) > 0
        
        # Check that identified dimensions are actually critical
        for dim_score in critical_dims:
            assert dim_score.is_critical()
    
    def test_scoring_configuration(self, scoring_engine):
        """Test scoring configuration retrieval."""
        config = scoring_engine.get_scoring_configuration()
        
        assert 'scoring_algorithm' in config
        assert 'quality_weights' in config
        assert 'features_enabled' in config
        assert 'reporting_options' in config
        
        # Check quality weights
        weights = config['quality_weights']
        assert len(weights) > 0
        
        for dim_name, weight_info in weights.items():
            assert 'weight' in weight_info
            assert 'critical_threshold' in weight_info
            assert 0 <= weight_info['weight'] <= 1


class TestRuleManagementService:
    """Test suite for Rule Management Service."""
    
    @pytest.fixture
    def rule_management_service(self):
        """Rule management service instance."""
        return RuleManagementService()
    
    @pytest.fixture
    def sample_rule_template(self):
        """Sample rule template."""
        return RuleTemplate(
            template_id="tmpl_001",
            template_name="Range Check Template",
            template_description="Template for range validation rules",
            rule_type=RuleType.RANGE,
            logic_type=LogicType.STATISTICAL,
            expression_template="{column} >= {min_value} AND {column} <= {max_value}",
            parameter_definitions={
                'column': {'type': 'string', 'required': True},
                'min_value': {'type': 'number', 'required': True},
                'max_value': {'type': 'number', 'required': True}
            },
            examples=['age >= 18 AND age <= 100', 'salary >= 0 AND salary <= 1000000']
        )
    
    @pytest.fixture
    def sample_test_dataframe(self):
        """Sample DataFrame for rule testing."""
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 150, -5],  # Some invalid values
            'salary': [50000, 60000, 70000, 80000, 90000, 2000000, -1000]  # Some invalid values
        })
    
    def test_rule_template_creation(self, rule_management_service, sample_rule_template):
        """Test rule template functionality."""
        # Test template creation
        rule = sample_rule_template.create_rule(
            rule_name="Age Range Rule",
            description="Validate age range",
            parameters={'column': 'age', 'min_value': 18, 'max_value': 100},
            created_by=UserId("test_user"),
            severity=Severity.HIGH,
            category=QualityCategory.BUSINESS_RULES
        )
        
        assert rule.rule_name == "Age Range Rule"
        assert rule.description == "Validate age range"
        assert rule.rule_type == RuleType.RANGE
        assert rule.validation_logic.logic_type == LogicType.STATISTICAL
        assert rule.severity == Severity.HIGH
        assert rule.category == QualityCategory.BUSINESS_RULES
    
    def test_rule_testing(self, rule_management_service, sample_rule_template, sample_test_dataframe):
        """Test rule testing functionality."""
        # Create a rule from template
        rule = sample_rule_template.create_rule(
            rule_name="Age Range Test",
            description="Test age range validation",
            parameters={'column': 'age', 'min_value': 18, 'max_value': 100},
            created_by=UserId("test_user")
        )
        
        # Test the rule
        test_result = rule_management_service.test_rule(
            rule, 
            sample_test_dataframe, 
            DatasetId("test_dataset")
        )
        
        assert test_result is not None
        assert test_result.rule_id == rule.rule_id
        assert test_result.test_status in [RuleTestStatus.PASSED, RuleTestStatus.FAILED, RuleTestStatus.WARNING]
        assert test_result.test_dataset_size == len(sample_test_dataframe)
        assert test_result.validation_result is not None


class TestIntegrationWorkflow:
    """Integration tests for the complete validation workflow."""
    
    @pytest.fixture
    def complete_workflow_setup(self):
        """Set up complete workflow components."""
        return {
            'validation_engine': ValidationEngine(),
            'cleansing_service': DataCleansingService(),
            'scoring_engine': ComprehensiveQualityScoringEngine(),
            'rule_management': RuleManagementService()
        }
    
    @pytest.fixture
    def workflow_dataframe(self):
        """DataFrame for workflow testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'customer_id': range(1, 1001),
            'name': [f'Customer_{i}' for i in range(1, 1001)],
            'age': np.random.randint(18, 80, 1000),
            'email': [f'customer{i}@example.com' for i in range(1, 1001)],
            'salary': np.random.randint(30000, 120000, 1000),
            'join_date': pd.date_range('2020-01-01', periods=1000, freq='D')
        })
    
    def test_end_to_end_workflow(self, complete_workflow_setup, workflow_dataframe):
        """Test complete end-to-end workflow."""
        components = complete_workflow_setup
        
        # 1. Create validation rules
        validation_rules = [
            QualityRule(
                rule_id=RuleId(),
                rule_name="Age Validation",
                description="Validate customer age",
                rule_type=RuleType.RANGE,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.STATISTICAL,
                    expression="age >= 18 and age <= 100",
                    parameters={'column_name': 'age', 'stat_type': 'range', 'min_value': 18, 'max_value': 100},
                    success_criteria=SuccessCriteria(min_pass_rate=0.95),
                    error_message="Age must be between 18 and 100"
                ),
                target_columns=['age'],
                severity=Severity.HIGH,
                category=QualityCategory.BUSINESS_RULES,
                created_by=UserId("system"),
                is_active=True
            )
        ]
        
        # 2. Execute validation
        dataset_id = DatasetId("workflow_test")
        validation_results = components['validation_engine'].validate_dataset(
            workflow_dataframe, validation_rules, dataset_id
        )
        
        # 3. Calculate quality score
        quality_score = components['scoring_engine'].calculate_comprehensive_score(
            workflow_dataframe, 
            "workflow_test", 
            validation_results, 
            QualityContext(business_domain="customer_management")
        )
        
        # 4. Apply cleansing if needed
        if quality_score.overall_score < 0.8:
            cleansing_rules = components['cleansing_service'].recommend_cleansing_rules(
                workflow_dataframe, quality_threshold=0.8
            )
            
            if cleansing_rules:
                cleaned_df, cleansing_result = components['cleansing_service'].cleanse_dataset(
                    workflow_dataframe, cleansing_rules, dataset_id
                )
                
                # Re-validate after cleansing
                validation_results_after = components['validation_engine'].validate_dataset(
                    cleaned_df, validation_rules, dataset_id
                )
                
                # Re-calculate quality score
                quality_score_after = components['scoring_engine'].calculate_comprehensive_score(
                    cleaned_df, 
                    "workflow_test_cleaned", 
                    validation_results_after, 
                    QualityContext(business_domain="customer_management")
                )
                
                # Quality should improve after cleansing
                assert quality_score_after.overall_score >= quality_score.overall_score
        
        # Verify workflow completed successfully
        assert len(validation_results) > 0
        assert quality_score.overall_score > 0
        assert quality_score.dataset_id == "workflow_test"
        assert len(quality_score.dimension_scores) > 0
    
    def test_workflow_with_poor_quality_data(self, complete_workflow_setup):
        """Test workflow with intentionally poor quality data."""
        components = complete_workflow_setup
        
        # Create poor quality data
        poor_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 1, 2, 3],  # Duplicates
            'name': ['  John  ', 'jane', 'BOB', None, ''],  # Format issues
            'age': [25, 150, -5, None, 200],  # Outliers and nulls
            'email': ['john@test.com', 'invalid', 'bob@', None, 'test@test']  # Invalid emails
        })
        
        # 1. Initial validation
        validation_rules = [
            QualityRule(
                rule_id=RuleId(),
                rule_name="Email Format",
                description="Validate email format",
                rule_type=RuleType.PATTERN,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.REGEX,
                    expression=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    parameters={'column_name': 'email'},
                    success_criteria=SuccessCriteria(min_pass_rate=0.80),
                    error_message="Invalid email format"
                ),
                target_columns=['email'],
                severity=Severity.HIGH,
                category=QualityCategory.DATA_INTEGRITY,
                created_by=UserId("system"),
                is_active=True
            )
        ]
        
        dataset_id = DatasetId("poor_quality_test")
        validation_results = components['validation_engine'].validate_dataset(
            poor_data, validation_rules, dataset_id
        )
        
        # 2. Calculate initial quality score
        initial_score = components['scoring_engine'].calculate_comprehensive_score(
            poor_data, 
            "poor_quality_test", 
            validation_results, 
            QualityContext()
        )
        
        # 3. Apply aggressive cleansing
        cleansing_rules = [
            CleansingRule(
                action=CleansingAction.DUPLICATES,
                strategy=CleansingStrategy.REMOVE,
                target_columns=[],
                priority=1
            ),
            CleansingRule(
                action=CleansingAction.MISSING_VALUES,
                strategy=CleansingStrategy.REMOVE,
                target_columns=[],
                priority=2
            ),
            CleansingRule(
                action=CleansingAction.OUTLIERS,
                strategy=CleansingStrategy.REMOVE,
                target_columns=['age'],
                parameters={'outlier_method': 'iqr'},
                priority=3
            ),
            CleansingRule(
                action=CleansingAction.WHITESPACE,
                strategy=CleansingStrategy.STANDARDIZE,
                target_columns=['name'],
                priority=4
            )
        ]
        
        cleaned_df, cleansing_result = components['cleansing_service'].cleanse_dataset(
            poor_data, cleansing_rules, dataset_id
        )
        
        # 4. Validate after cleansing
        validation_results_after = components['validation_engine'].validate_dataset(
            cleaned_df, validation_rules, dataset_id
        )
        
        # 5. Calculate final quality score
        final_score = components['scoring_engine'].calculate_comprehensive_score(
            cleaned_df, 
            "poor_quality_test_cleaned", 
            validation_results_after, 
            QualityContext()
        )
        
        # Verify improvement
        assert final_score.overall_score > initial_score.overall_score
        assert cleansing_result.removed_records > 0
        assert cleansing_result.retention_rate < 1.0
        assert len(cleaned_df) < len(poor_data)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
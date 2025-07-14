import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock

from ..application.services.quality_assessment_service import QualityAssessmentService
from ..domain.entities.data_profile import (
    SchemaProfile, ColumnProfile, ValueDistribution, StatisticalSummary,
    DataType, CardinalityLevel, QualityIssue, QualityIssueType, QualityAssessment
)


class TestQualityAssessmentService:
    
    def setup_method(self):
        self.service = QualityAssessmentService()
    
    def create_mock_column_profile(self, column_name: str, data_type: DataType,
                                  null_count: int = 0, total_count: int = 100,
                                  unique_count: int = 50, quality_issues: list = None):
        """Helper to create mock column profiles."""
        distribution = ValueDistribution(
            unique_count=unique_count,
            null_count=null_count,
            total_count=total_count,
            completeness_ratio=(total_count - null_count) / total_count,
            top_values={}
        )
        
        return ColumnProfile(
            column_name=column_name,
            data_type=data_type,
            inferred_type=None,
            nullable=null_count > 0,
            distribution=distribution,
            cardinality=CardinalityLevel.MEDIUM,
            statistical_summary=None,
            patterns=[],
            quality_score=1.0,
            quality_issues=quality_issues or [],
            semantic_type=None,
            business_meaning=None
        )
    
    def create_mock_schema_profile(self, columns: list):
        """Helper to create mock schema profile."""
        return SchemaProfile(
            table_name="test_table",
            total_columns=len(columns),
            total_rows=100,
            columns=columns,
            primary_keys=[],
            foreign_keys={},
            unique_constraints=[],
            check_constraints=[],
            estimated_size_bytes=1024,
            compression_ratio=None
        )
    
    def test_completeness_score_calculation(self):
        """Test completeness score calculation."""
        # Create columns with different completeness levels
        columns = [
            self.create_mock_column_profile("col1", DataType.STRING, null_count=0, total_count=100),  # 100% complete
            self.create_mock_column_profile("col2", DataType.INTEGER, null_count=20, total_count=100),  # 80% complete
            self.create_mock_column_profile("col3", DataType.FLOAT, null_count=50, total_count=100),  # 50% complete
        ]
        
        schema_profile = self.create_mock_schema_profile(columns)
        df = pd.DataFrame()  # Empty for this test
        
        completeness = self.service._calculate_completeness_score(schema_profile, df)
        
        # Average completeness: (1.0 + 0.8 + 0.5) / 3 = 0.767
        assert abs(completeness - 0.767) < 0.01
    
    def test_assess_quality_basic(self):
        """Test basic quality assessment."""
        columns = [
            self.create_mock_column_profile("good_col", DataType.STRING, null_count=0, total_count=100),
            self.create_mock_column_profile("bad_col", DataType.INTEGER, null_count=50, total_count=100)
        ]
        
        schema_profile = self.create_mock_schema_profile(columns)
        df = pd.DataFrame({
            'good_col': ['A'] * 100,
            'bad_col': [1] * 50 + [None] * 50
        })
        
        assessment = self.service.assess_quality(schema_profile, df)
        
        assert isinstance(assessment, QualityAssessment)
        assert 0 <= assessment.overall_score <= 1
        assert 0 <= assessment.completeness_score <= 1
        assert 0 <= assessment.consistency_score <= 1
        assert 0 <= assessment.accuracy_score <= 1
        assert 0 <= assessment.validity_score <= 1
        assert 0 <= assessment.uniqueness_score <= 1
        
        # Should have recommendations
        assert isinstance(assessment.recommendations, list)
    
    def test_uniqueness_score_calculation(self):
        """Test uniqueness score calculation."""
        df = pd.DataFrame({
            'unique_id': [1, 2, 3, 4, 5],  # All unique
            'duplicate_values': [1, 1, 2, 2, 3],  # Some duplicates
            'category': ['A', 'A', 'B', 'B', 'C']  # Expected duplicates
        })
        
        columns = [
            self.create_mock_column_profile("unique_id", DataType.INTEGER),
            self.create_mock_column_profile("duplicate_values", DataType.INTEGER),
            self.create_mock_column_profile("category", DataType.STRING)
        ]
        schema_profile = self.create_mock_schema_profile(columns)
        
        uniqueness = self.service._calculate_uniqueness_score(schema_profile, df)
        
        # Should be high since no row duplicates and unique_id is unique
        assert uniqueness > 0.8
    
    def test_column_quality_assessment_detailed(self):
        """Test detailed column quality assessment."""
        # Create test data with quality issues
        test_data = pd.Series([1, 2, 3, None, None, 100, 200])  # Has nulls and outliers
        
        column_profile = self.create_mock_column_profile(
            "test_col", DataType.INTEGER, null_count=2, total_count=7
        )
        
        quality_score, issues = self.service.assess_column_quality_detailed(column_profile, test_data)
        
        assert 0 <= quality_score <= 1
        assert isinstance(issues, list)
        
        # Should detect missing values
        missing_issues = [issue for issue in issues if issue.issue_type == QualityIssueType.MISSING_VALUES]
        assert len(missing_issues) > 0
        assert missing_issues[0].affected_rows == 2
    
    def test_email_validity_checking(self):
        """Test email validity checking."""
        emails = pd.Series([
            'valid@example.com',
            'also.valid@test.org',
            'invalid-email',
            'missing@',
            '@missing-domain.com'
        ])
        
        result = self.service._check_email_validity(emails)
        
        assert 'count' in result
        assert 'percentage' in result
        assert 'examples' in result
        
        # Should detect invalid emails
        assert result['count'] >= 3  # At least 3 invalid emails
        assert result['percentage'] > 0
    
    def test_phone_validity_checking(self):
        """Test phone number validity checking."""
        phones = pd.Series([
            '+1-555-123-4567',
            '(555) 987-6543',
            'not-a-phone',
            '123',  # Too short
            'abcd-efgh-ijkl'  # Invalid format
        ])
        
        result = self.service._check_phone_validity(phones)
        
        assert 'count' in result
        assert 'percentage' in result
        assert 'examples' in result
        
        # Should detect some invalid phones
        assert result['count'] >= 2
    
    def test_consistency_score_calculation(self):
        """Test consistency score calculation."""
        # Create data with consistency issues
        df = pd.DataFrame({
            'mixed_case': ['Apple', 'BANANA', 'cherry', 'GRAPE'],  # Inconsistent casing
            'consistent': ['apple', 'banana', 'cherry', 'grape'],  # Consistent
            'numeric_strings': ['1', '2', 'three', '4']  # Mixed numeric/text
        })
        
        columns = [
            self.create_mock_column_profile("mixed_case", DataType.STRING),
            self.create_mock_column_profile("consistent", DataType.STRING),
            self.create_mock_column_profile("numeric_strings", DataType.STRING)
        ]
        schema_profile = self.create_mock_schema_profile(columns)
        
        consistency = self.service._calculate_consistency_score(schema_profile, df)
        
        # Should detect consistency issues
        assert 0 <= consistency <= 1
    
    def test_accuracy_score_with_outliers(self):
        """Test accuracy score calculation with outliers."""
        # Create data with clear outliers
        normal_data = list(range(1, 11))  # Normal range 1-10
        outlier_data = [1000, 2000]  # Clear outliers
        
        df = pd.DataFrame({
            'with_outliers': normal_data + outlier_data,
            'without_outliers': normal_data + [11, 12]  # No outliers
        })
        
        columns = [
            self.create_mock_column_profile("with_outliers", DataType.INTEGER),
            self.create_mock_column_profile("without_outliers", DataType.INTEGER)
        ]
        schema_profile = self.create_mock_schema_profile(columns)
        
        accuracy = self.service._calculate_accuracy_score(schema_profile, df)
        
        # Should penalize for outliers
        assert 0 <= accuracy <= 1
    
    def test_validity_score_calculation(self):
        """Test validity score calculation."""
        df = pd.DataFrame({
            'integers': [1, 2, 3, 4, 5],  # Valid integers
            'fake_integers': [1.5, 2.7, 3.1],  # Should be integers but aren't
            'negative_ages': [-5, -10, 25, 30]  # Negative values where they shouldn't be
        })
        
        columns = [
            self.create_mock_column_profile("integers", DataType.INTEGER),
            self.create_mock_column_profile("fake_integers", DataType.INTEGER),
            self.create_mock_column_profile("negative_ages", DataType.INTEGER)  # Assume this is an age column
        ]
        schema_profile = self.create_mock_schema_profile(columns)
        
        validity = self.service._calculate_validity_score(schema_profile, df)
        
        # Should detect validity issues
        assert 0 <= validity <= 1
    
    def test_recommendations_generation(self):
        """Test quality improvement recommendations generation."""
        # Create data with various quality issues
        df = pd.DataFrame({
            'high_nulls': [1, None, None, None, None],  # High missing rate
            'email_col': ['user@test.com', 'invalid-email'],  # Potential PII
            'inconsistent_case': ['Apple', 'BANANA', 'cherry']  # Case inconsistency
        })
        
        columns = [
            self.create_mock_column_profile("high_nulls", DataType.INTEGER, null_count=4, total_count=5),
            self.create_mock_column_profile("email_col", DataType.STRING),
            self.create_mock_column_profile("inconsistent_case", DataType.STRING)
        ]
        schema_profile = self.create_mock_schema_profile(columns)
        
        recommendations = self.service._generate_recommendations(schema_profile, df)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should include recommendation about high missing data
        high_missing_rec = any('missing' in rec.lower() for rec in recommendations)
        assert high_missing_rec
    
    def test_issue_severity_determination(self):
        """Test issue severity determination."""
        # Test different severity levels
        assert self.service._determine_severity(3, [5, 15, 30]) == "low"
        assert self.service._determine_severity(10, [5, 15, 30]) == "medium"
        assert self.service._determine_severity(20, [5, 15, 30]) == "high"
        assert self.service._determine_severity(40, [5, 15, 30]) == "critical"
    
    def test_missing_value_action_suggestions(self):
        """Test missing value action suggestions."""
        assert "removing" in self.service._suggest_missing_value_action(3).lower()
        assert "imputation" in self.service._suggest_missing_value_action(15).lower()
        assert "dropping column" in self.service._suggest_missing_value_action(25).lower()
    
    def test_string_consistency_assessment(self):
        """Test string consistency assessment."""
        # Mixed case strings
        mixed_case_series = pd.Series(['Apple', 'BANANA', 'cherry'])
        issues = self.service._assess_string_consistency(mixed_case_series)
        
        # Should detect consistency issues
        assert len(issues) > 0
        assert any(issue.issue_type == QualityIssueType.INCONSISTENT_VALUES for issue in issues)
        
        # Consistent strings
        consistent_series = pd.Series(['apple', 'banana', 'cherry'])
        issues = self.service._assess_string_consistency(consistent_series)
        
        # Should not detect issues
        assert len(issues) == 0
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()
        empty_schema = self.create_mock_schema_profile([])
        
        assessment = self.service.assess_quality(empty_schema, empty_df)
        
        # Should handle gracefully
        assert isinstance(assessment, QualityAssessment)
        assert assessment.overall_score >= 0
    
    def test_custom_dimension_weights(self):
        """Test custom quality dimension weights."""
        custom_weights = {
            'completeness': 0.5,
            'consistency': 0.2,
            'accuracy': 0.1,
            'validity': 0.1,
            'uniqueness': 0.1
        }
        
        columns = [
            self.create_mock_column_profile("test_col", DataType.STRING)
        ]
        schema_profile = self.create_mock_schema_profile(columns)
        df = pd.DataFrame({'test_col': ['A', 'B', 'C']})
        
        assessment = self.service.assess_quality(schema_profile, df, custom_weights)
        
        assert assessment.dimension_weights == custom_weights
        # Overall score should reflect the custom weighting
        assert 0 <= assessment.overall_score <= 1
    
    def test_issue_counting(self):
        """Test quality issue counting by severity."""
        # Create columns with various quality issues
        high_issue = QualityIssue(
            issue_type=QualityIssueType.MISSING_VALUES,
            severity="high",
            description="High severity issue",
            affected_rows=10,
            affected_percentage=50.0,
            examples=[],
            suggested_action="Fix this"
        )
        
        medium_issue = QualityIssue(
            issue_type=QualityIssueType.DUPLICATE_VALUES,
            severity="medium",
            description="Medium severity issue",
            affected_rows=5,
            affected_percentage=25.0,
            examples=[],
            suggested_action="Consider fixing"
        )
        
        columns = [
            self.create_mock_column_profile("col1", DataType.STRING, quality_issues=[high_issue]),
            self.create_mock_column_profile("col2", DataType.STRING, quality_issues=[medium_issue])
        ]
        schema_profile = self.create_mock_schema_profile(columns)
        
        counts = self.service._count_issues_by_severity(schema_profile)
        
        assert counts['high'] == 1
        assert counts['medium'] == 1
        assert counts['low'] == 0
        assert counts['critical'] == 0
    
    @pytest.mark.parametrize("null_percentage,expected_severity", [
        (3, "low"),
        (10, "medium"),
        (25, "high"),
        (50, "critical")
    ])
    def test_severity_levels(self, null_percentage, expected_severity):
        """Test different severity levels for quality issues."""
        thresholds = [5, 15, 30]
        severity = self.service._determine_severity(null_percentage, thresholds)
        assert severity == expected_severity
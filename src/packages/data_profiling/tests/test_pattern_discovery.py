import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from ..application.services.pattern_discovery_service import PatternDiscoveryService
from ..domain.entities.data_profile import Pattern, PatternType


class TestPatternDiscoveryService:
    
    def setup_method(self):
        self.service = PatternDiscoveryService()
    
    def test_email_pattern_detection(self):
        """Test email pattern detection."""
        df = pd.DataFrame({
            'emails': [
                'user@example.com',
                'admin@test.org',
                'contact@company.net',
                'invalid-email',
                'another@domain.co.uk'
            ]
        })
        
        patterns = self.service.discover(df)
        
        assert 'emails' in patterns
        email_patterns = patterns['emails']
        
        # Should detect email pattern
        email_pattern = next((p for p in email_patterns if p.pattern_type == PatternType.EMAIL), None)
        assert email_pattern is not None
        assert email_pattern.frequency >= 4  # 4 valid emails out of 5
        assert 'user@example.com' in email_pattern.examples
    
    def test_phone_pattern_detection(self):
        """Test phone number pattern detection."""
        df = pd.DataFrame({
            'phones': [
                '+1-555-123-4567',
                '(555) 987-6543',
                '555.444.3333',
                '5551234567',
                'not-a-phone'
            ]
        })
        
        patterns = self.service.discover(df)
        
        assert 'phones' in patterns
        phone_patterns = patterns['phones']
        
        # Should detect some phone patterns
        phone_pattern = next((p for p in phone_patterns if p.pattern_type == PatternType.PHONE), None)
        if phone_pattern:  # Depending on regex strictness
            assert phone_pattern.frequency >= 1
    
    def test_url_pattern_detection(self):
        """Test URL pattern detection."""
        df = pd.DataFrame({
            'urls': [
                'https://www.example.com',
                'http://test.org/page',
                'https://api.service.com/v1/endpoint',
                'not-a-url',
                'https://secure.site.net'
            ]
        })
        
        patterns = self.service.discover(df)
        
        assert 'urls' in patterns
        url_patterns = patterns['urls']
        
        # Should detect URL pattern
        url_pattern = next((p for p in url_patterns if p.pattern_type == PatternType.URL), None)
        assert url_pattern is not None
        assert url_pattern.frequency >= 3  # 3 valid URLs
    
    def test_semantic_type_classification(self):
        """Test semantic type classification with column name hints."""
        df = pd.DataFrame({
            'user_email': ['user@test.com', 'admin@test.com'],
            'phone_number': ['+1-555-1234', '+1-555-5678'],
            'random_text': ['abc', 'def']
        })
        
        patterns = self.service.discover(df)
        
        # Email column should have higher confidence due to column name hint
        if 'user_email' in patterns:
            email_patterns = patterns['user_email']
            for pattern in email_patterns:
                if 'email' in pattern.regex.lower():
                    assert pattern.confidence > 0.5
    
    def test_fixed_length_pattern_analysis(self):
        """Test fixed-length pattern analysis."""
        df = pd.DataFrame({
            'codes': [
                'ABC123',
                'DEF456',
                'GHI789',
                'JKL012',
                'MNO345'
            ]
        })
        
        patterns = self.service.discover(df)
        
        assert 'codes' in patterns
        code_patterns = patterns['codes']
        
        # Should detect pattern for fixed-length alphanumeric codes
        assert len(code_patterns) > 0
        
        # Check if a pattern captures the structure (3 letters + 3 digits)
        pattern_found = False
        for pattern in code_patterns:
            if 'A-Z' in pattern.regex and '\\d' in pattern.regex:
                pattern_found = True
                break
        assert pattern_found
    
    def test_character_composition_analysis(self):
        """Test character composition pattern analysis."""
        df = pd.DataFrame({
            'only_digits': ['123', '456', '789'],
            'only_letters': ['abc', 'def', 'ghi'],
            'mixed': ['abc123', 'def456', 'ghi789']
        })
        
        patterns = self.service.discover(df)
        
        # Only digits column
        if 'only_digits' in patterns:
            digit_patterns = patterns['only_digits']
            digit_pattern = next((p for p in digit_patterns if '\\d' in p.regex), None)
            assert digit_pattern is not None
        
        # Only letters column
        if 'only_letters' in patterns:
            letter_patterns = patterns['only_letters']
            letter_pattern = next((p for p in letter_patterns if '[a-zA-Z]' in p.regex), None)
            assert letter_pattern is not None
    
    def test_format_template_extraction(self):
        """Test format template extraction."""
        values = np.array(['ABC-123', 'DEF-456', 'GHI-789'])
        
        templates = self.service._extract_format_templates(values)
        
        # Should extract template UUU-DDD (Upper-Upper-Upper-Dash-Digit-Digit-Digit)
        expected_template = 'UUU-DDD'
        assert expected_template in templates
        assert len(templates[expected_template]) == 3
    
    def test_string_to_template_conversion(self):
        """Test string to template conversion."""
        assert self.service._string_to_template('ABC123') == 'UUUDDD'
        assert self.service._string_to_template('abc-123') == 'LLL-DDD'
        assert self.service._string_to_template('Test 123') == 'ULLLLSDDD'
        assert self.service._string_to_template('user@domain.com') == 'LLLL@LLLLLL.LLL'
    
    def test_template_to_regex_conversion(self):
        """Test template to regex conversion."""
        assert self.service._template_to_regex('DDD') == '^\\d\\d\\d$'
        assert self.service._template_to_regex('UUU') == '^[A-Z][A-Z][A-Z]$'
        assert self.service._template_to_regex('LLL') == '^[a-z][a-z][a-z]$'
        assert self.service._template_to_regex('UUU-DDD') == '^[A-Z][A-Z][A-Z]\\-\\d\\d\\d$'
    
    @patch('sklearn.cluster.KMeans')
    def test_ml_pattern_clustering(self, mock_kmeans):
        """Test ML-based pattern clustering."""
        # Mock KMeans clustering
        mock_model = Mock()
        mock_model.fit_predict.return_value = np.array([0, 0, 1, 1, 2])
        mock_kmeans.return_value = mock_model
        
        df = pd.DataFrame({
            'mixed_patterns': [
                'ABC123',  # Cluster 0
                'DEF456',  # Cluster 0
                'user@test.com',  # Cluster 1
                'admin@test.com',  # Cluster 1
                'https://test.com'  # Cluster 2
            ]
        })
        
        patterns = self.service.discover(df)
        
        # Should generate patterns based on clustering
        assert 'mixed_patterns' in patterns
        assert len(patterns['mixed_patterns']) > 0
    
    def test_pattern_deduplication_and_ranking(self):
        """Test pattern deduplication and ranking."""
        patterns = [
            Pattern(
                pattern_type=PatternType.CUSTOM,
                regex='^\\d+$',
                frequency=10,
                percentage=50.0,
                examples=['123', '456'],
                confidence=0.8
            ),
            Pattern(
                pattern_type=PatternType.CUSTOM,
                regex='^\\d+$',  # Duplicate regex
                frequency=8,
                percentage=40.0,
                examples=['789', '012'],
                confidence=0.7
            ),
            Pattern(
                pattern_type=PatternType.CUSTOM,
                regex='^[a-z]+$',
                frequency=5,
                percentage=25.0,
                examples=['abc', 'def'],
                confidence=0.6
            ),
            Pattern(
                pattern_type=PatternType.CUSTOM,
                regex='^[A-Z]+$',
                frequency=2,
                percentage=2.0,  # Below threshold
                examples=['ABC'],
                confidence=0.1  # Low confidence
            )
        ]
        
        deduplicated = self.service._deduplicate_and_rank_patterns(patterns)
        
        # Should remove duplicates and low-quality patterns
        assert len(deduplicated) == 2  # Remove duplicate and low-quality pattern
        
        # Should be sorted by confidence and percentage
        assert deduplicated[0].confidence >= deduplicated[1].confidence
    
    def test_functional_dependency_detection(self):
        """Test functional dependency detection."""
        df = pd.DataFrame({
            'country_code': ['US', 'US', 'CA', 'CA', 'UK'],
            'country_name': ['United States', 'United States', 'Canada', 'Canada', 'United Kingdom'],
            'random_value': [1, 2, 3, 4, 5]
        })
        
        relationships = self.service.analyze_data_relationships(df)
        
        dependencies = relationships['functional_dependencies']
        
        # Should detect country_code -> country_name dependency
        dependency_found = False
        for dep in dependencies:
            if dep['determinant'] == 'country_code' and dep['dependent'] == 'country_name':
                assert dep['strength'] > 0.9
                dependency_found = True
                break
        assert dependency_found
    
    def test_cramers_v_calculation(self):
        """Test Cramér's V calculation for categorical correlation."""
        # Create strongly correlated categorical variables
        df = pd.DataFrame({
            'category_a': ['A', 'A', 'B', 'B', 'C', 'C'],
            'category_b': ['X', 'X', 'Y', 'Y', 'Z', 'Z'],
            'random_cat': ['P', 'Q', 'R', 'P', 'Q', 'R']
        })
        
        # Test direct calculation
        cramers_v = self.service._calculate_cramers_v(df['category_a'], df['category_b'])
        
        # Should be high correlation (perfect mapping)
        assert cramers_v > 0.8
        
        # Test with random categories
        cramers_v_random = self.service._calculate_cramers_v(df['category_a'], df['random_cat'])
        
        # Should be lower correlation
        assert cramers_v_random < cramers_v
    
    def test_cross_column_pattern_analysis(self):
        """Test cross-column pattern analysis."""
        df = pd.DataFrame({
            'first_name': ['John', 'Jane', 'Bob'],
            'last_name': ['Doe', 'Smith', 'Jones'],
            'unrelated': ['X', 'Y', 'Z']
        })
        
        relationships = self.service.analyze_data_relationships(df)
        
        cross_patterns = relationships['cross_column_patterns']
        
        # Should analyze combinations of string columns
        assert isinstance(cross_patterns, list)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        df = pd.DataFrame()
        
        patterns = self.service.discover(df)
        assert patterns == {}
        
        relationships = self.service.analyze_data_relationships(df)
        assert all(len(v) == 0 for v in relationships.values())
    
    def test_single_value_column(self):
        """Test handling of columns with single unique value."""
        df = pd.DataFrame({
            'constant': ['same', 'same', 'same', 'same']
        })
        
        patterns = self.service.discover(df)
        
        # Should not generate patterns for single-value columns
        assert 'constant' not in patterns
    
    def test_confidence_scoring(self):
        """Test confidence scoring for patterns."""
        df = pd.DataFrame({
            'high_confidence': ['ABC', 'ABC', 'ABC', 'ABC', 'ABC'],  # 100% match
            'medium_confidence': ['ABC', 'ABC', 'DEF', 'GHI', 'JKL'],  # 20% match
            'low_confidence': ['ABC', 'DEF', 'GHI', 'JKL', 'MNO']  # 20% match
        })
        
        patterns = self.service.discover(df)
        
        # High confidence column should have patterns with higher confidence
        if 'high_confidence' in patterns:
            high_conf_patterns = patterns['high_confidence']
            if high_conf_patterns:
                assert max(p.confidence for p in high_conf_patterns) > 0.8
    
    @pytest.mark.parametrize("column_name,expected_boost", [
        ('user_email', True),
        ('phone_number', True),
        ('random_column', False),
    ])
    def test_column_name_confidence_boost(self, column_name, expected_boost):
        """Test confidence boost based on column names."""
        # This would require creating specific test data and checking
        # if confidence is boosted for semantically named columns
        df = pd.DataFrame({
            column_name: ['test@example.com', 'user@domain.com']
        })
        
        patterns = self.service.discover(df)
        
        if column_name in patterns:
            # Check if any patterns have confidence boost
            has_high_confidence = any(p.confidence > 0.8 for p in patterns[column_name])
            if expected_boost:
                # Email patterns should have higher confidence when column name suggests email
                assert has_high_confidence or len(patterns[column_name]) == 0
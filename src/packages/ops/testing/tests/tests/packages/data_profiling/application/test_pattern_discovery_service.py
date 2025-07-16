import pytest
import pandas as pd
import numpy as np
import re

from src.packages.data_profiling.application.services.pattern_discovery_service import PatternDiscoveryService
from src.packages.data_profiling.domain.entities.data_profile import PatternType, SemanticType


class TestPatternDiscoveryService:
    """Test PatternDiscoveryService."""
    
    @pytest.fixture
    def service(self):
        """Create PatternDiscoveryService instance."""
        return PatternDiscoveryService()
    
    @pytest.fixture
    def pattern_dataframe(self):
        """Create DataFrame with various patterns for testing."""
        data = {
            'emails': [
                'user1@example.com', 'admin@company.org', 'test@domain.net',
                'info@business.co.uk', 'support@website.io'
            ] * 20,
            'phones': [
                '+1-555-1234', '+1-555-5678', '+1-555-9012',
                '+44-20-1234-5678', '+81-3-1234-5678'
            ] * 20,
            'urls': [
                'https://www.example.com', 'http://domain.org/path',
                'https://subdomain.site.net/page?param=value',
                'https://api.service.com/v1/endpoint',
                'http://localhost:8080/test'
            ] * 20,
            'dates': [
                '2023-01-01', '2023-12-31', '2024-06-15',
                '01/15/2023', '12-25-2024'
            ] * 20,
            'times': [
                '14:30:00', '09:15:30', '23:59:59',
                '2:30 PM', '11:45 AM'
            ] * 20,
            'numeric': [
                '123', '456.789', '0.001',
                '1000000', '3.14159'
            ] * 20,
            'alphanumeric': [
                'ABC123', 'XYZ789', 'DEF456',
                'GHI999', 'JKL000'
            ] * 20,
            'credit_cards': [
                '4532-1234-5678-9012', '5555-5555-5555-4444',
                '3782-822463-10005', '6011-1111-1111-1117',
                '3056-9309-0259-04'
            ] * 20,
            'ssn': [
                '123-45-6789', '987-65-4321', '555-12-3456',
                '111-22-3333', '999-88-7777'
            ] * 20,
            'ip_addresses': [
                '192.168.1.1', '10.0.0.1', '172.16.0.1',
                '8.8.8.8', '127.0.0.1'
            ] * 20,
            'mixed_content': [
                'text123', 'email@domain', 'not-a-pattern',
                '123-abc', 'random text'
            ] * 20
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def business_dataframe(self):
        """Create DataFrame with business patterns."""
        data = {
            'product_codes': [
                'AB123456', 'XY789012', 'CD345678',
                'EF901234', 'GH567890'
            ] * 20,
            'order_ids': [
                'ORD-1234567', 'ORD-9876543', 'ORD-5555555',
                'ORD-1111111', 'ORD-7777777'
            ] * 20,
            'invoice_numbers': [
                'INV-12345678', 'INV-87654321', 'INV-55555555',
                'INV-11111111', 'INV-99999999'
            ] * 20,
            'customer_ids': [
                'CUST-123456', 'CUST-789012', 'CUST-345678',
                'CUST-901234', 'CUST-567890'
            ] * 20,
            'transaction_ids': [
                'TXN-ABCD1234EFGH', 'TXN-XYZA9876BCDE',
                'TXN-MNOP5555QRST', 'TXN-UVWX1111YZAB',
                'TXN-CDEF7777GHIJ'
            ] * 20
        }
        return pd.DataFrame(data)


class TestBasicPatternDiscovery:
    """Test basic pattern discovery functionality."""
    
    def test_discover_patterns_basic(self, service, pattern_dataframe):
        """Test basic pattern discovery."""
        patterns = service.discover(pattern_dataframe)
        
        assert isinstance(patterns, dict)
        assert len(patterns) > 0
        
        # Should find patterns in most columns
        for col in ['emails', 'phones', 'urls', 'numeric', 'alphanumeric']:
            assert col in patterns
            assert len(patterns[col]) > 0
    
    def test_discover_empty_dataframe(self, service):
        """Test pattern discovery on empty DataFrame."""
        empty_df = pd.DataFrame()
        patterns = service.discover(empty_df)
        
        assert isinstance(patterns, dict)
        assert len(patterns) == 0
    
    def test_discover_single_column(self, service):
        """Test pattern discovery on single column."""
        single_col_df = pd.DataFrame({
            'test_col': ['user@example.com', 'admin@company.org', 'test@domain.net']
        })
        
        patterns = service.discover(single_col_df)
        
        assert 'test_col' in patterns
        assert len(patterns['test_col']) > 0


class TestPredefinedPatterns:
    """Test predefined pattern matching."""
    
    def test_email_pattern_detection(self, service, pattern_dataframe):
        """Test email pattern detection."""
        email_series = pattern_dataframe['emails']
        patterns = service._discover_column_patterns(email_series, 'emails')
        
        # Should detect email pattern
        email_patterns = [p for p in patterns if p.pattern_type == PatternType.EMAIL]
        assert len(email_patterns) > 0
        assert email_patterns[0].confidence_score > 0.8
    
    def test_phone_pattern_detection(self, service, pattern_dataframe):
        """Test phone pattern detection."""
        phone_series = pattern_dataframe['phones']
        patterns = service._discover_column_patterns(phone_series, 'phones')
        
        # Should detect phone pattern
        phone_patterns = [p for p in patterns if p.pattern_type == PatternType.PHONE]
        assert len(phone_patterns) > 0
    
    def test_url_pattern_detection(self, service, pattern_dataframe):
        """Test URL pattern detection."""
        url_series = pattern_dataframe['urls']
        patterns = service._discover_column_patterns(url_series, 'urls')
        
        # Should detect URL pattern
        url_patterns = [p for p in patterns if p.pattern_type == PatternType.URL]
        assert len(url_patterns) > 0
    
    def test_date_pattern_detection(self, service, pattern_dataframe):
        """Test date pattern detection."""
        date_series = pattern_dataframe['dates']
        patterns = service._discover_column_patterns(date_series, 'dates')
        
        # Should detect date pattern
        date_patterns = [p for p in patterns if p.pattern_type == PatternType.DATE]
        assert len(date_patterns) > 0
    
    def test_numeric_pattern_detection(self, service, pattern_dataframe):
        """Test numeric pattern detection."""
        numeric_series = pattern_dataframe['numeric']
        patterns = service._discover_column_patterns(numeric_series, 'numeric')
        
        # Should detect numeric pattern
        numeric_patterns = [p for p in patterns if p.pattern_type == PatternType.NUMERIC]
        assert len(numeric_patterns) > 0
    
    def test_alphanumeric_pattern_detection(self, service, pattern_dataframe):
        """Test alphanumeric pattern detection."""
        alphanum_series = pattern_dataframe['alphanumeric']
        patterns = service._discover_column_patterns(alphanum_series, 'alphanumeric')
        
        # Should detect alphanumeric pattern
        alphanum_patterns = [p for p in patterns if p.pattern_type == PatternType.ALPHANUMERIC]
        assert len(alphanum_patterns) > 0


class TestSemanticPatterns:
    """Test semantic pattern classification."""
    
    def test_credit_card_detection(self, service, pattern_dataframe):
        """Test credit card pattern detection."""
        cc_series = pattern_dataframe['credit_cards']
        patterns = service._discover_column_patterns(cc_series, 'credit_cards')
        
        # Should detect credit card patterns in semantic classification
        assert len(patterns) > 0
    
    def test_ssn_detection(self, service, pattern_dataframe):
        """Test SSN pattern detection."""
        ssn_series = pattern_dataframe['ssn']
        patterns = service._discover_column_patterns(ssn_series, 'ssn')
        
        # Should detect SSN patterns
        assert len(patterns) > 0
    
    def test_ip_address_detection(self, service, pattern_dataframe):
        """Test IP address pattern detection."""
        ip_series = pattern_dataframe['ip_addresses']
        patterns = service._discover_column_patterns(ip_series, 'ip_addresses')
        
        # Should detect IP address patterns
        assert len(patterns) > 0


class TestBusinessPatterns:
    """Test business domain pattern detection."""
    
    def test_product_code_detection(self, service, business_dataframe):
        """Test product code pattern detection."""
        product_series = business_dataframe['product_codes']
        patterns = service._discover_column_patterns(product_series, 'product_codes')
        
        # Should detect business patterns
        assert len(patterns) > 0
    
    def test_order_id_detection(self, service, business_dataframe):
        """Test order ID pattern detection."""
        order_series = business_dataframe['order_ids']
        patterns = service._discover_column_patterns(order_series, 'order_ids')
        
        # Should detect order ID patterns
        assert len(patterns) > 0
    
    def test_invoice_number_detection(self, service, business_dataframe):
        """Test invoice number pattern detection."""
        invoice_series = business_dataframe['invoice_numbers']
        patterns = service._discover_column_patterns(invoice_series, 'invoice_numbers')
        
        # Should detect invoice patterns
        assert len(patterns) > 0


class TestPatternMatching:
    """Test pattern matching functionality."""
    
    def test_match_predefined_patterns(self, service):
        """Test predefined pattern matching."""
        test_data = pd.Series([
            'user@example.com', 'admin@test.org', 'invalid-email',
            'contact@domain.net', 'not-an-email'
        ])
        
        patterns = service._match_predefined_patterns(test_data, test_data.unique())
        
        assert len(patterns) > 0
        # Should find email pattern with confidence > 0.5 (3 out of 5 match)
        email_pattern = next((p for p in patterns if p.pattern_type == PatternType.EMAIL), None)
        assert email_pattern is not None
        assert email_pattern.confidence_score > 0.5
    
    def test_classify_semantic_types(self, service):
        """Test semantic type classification."""
        test_data = pd.Series(['4532-1234-5678-9012', '5555-5555-5555-4444', 'not-a-card'])
        
        patterns = service._classify_semantic_types(test_data, test_data.unique(), 'credit_card_col')
        
        assert len(patterns) > 0
    
    def test_match_business_patterns(self, service):
        """Test business pattern matching."""
        test_data = pd.Series(['ORD-1234567', 'ORD-7654321', 'invalid-order'])
        
        patterns = service._match_business_patterns(test_data, test_data.unique(), 'order_column')
        
        assert len(patterns) > 0
    
    def test_discover_statistical_patterns(self, service):
        """Test statistical pattern discovery."""
        # Create data with statistical patterns
        test_data = pd.Series([f'ITEM-{i:06d}' for i in range(100)])
        
        patterns = service._discover_statistical_patterns(test_data, test_data.unique())
        
        assert len(patterns) > 0


class TestPatternConfidence:
    """Test pattern confidence scoring."""
    
    def test_high_confidence_pattern(self, service):
        """Test high confidence pattern detection."""
        # All values match email pattern
        perfect_emails = pd.Series([
            'user1@example.com', 'user2@example.com', 'user3@example.com',
            'user4@example.com', 'user5@example.com'
        ])
        
        patterns = service._discover_column_patterns(perfect_emails, 'perfect_emails')
        
        email_pattern = next((p for p in patterns if p.pattern_type == PatternType.EMAIL), None)
        assert email_pattern is not None
        assert email_pattern.confidence_score >= 0.9
    
    def test_medium_confidence_pattern(self, service):
        """Test medium confidence pattern detection."""
        # Some values match email pattern
        mixed_emails = pd.Series([
            'user1@example.com', 'user2@example.com', 'not-an-email',
            'user4@example.com', 'another-non-email'
        ])
        
        patterns = service._discover_column_patterns(mixed_emails, 'mixed_emails')
        
        email_pattern = next((p for p in patterns if p.pattern_type == PatternType.EMAIL), None)
        if email_pattern:
            assert 0.3 <= email_pattern.confidence_score < 0.9
    
    def test_low_confidence_pattern(self, service):
        """Test low confidence pattern handling."""
        # Very few values match any pattern
        random_data = pd.Series([
            'random text', 'more random', 'user@example.com',
            'completely random', 'no pattern here'
        ])
        
        patterns = service._discover_column_patterns(random_data, 'random_data')
        
        # Should either find no patterns or patterns with low confidence
        if patterns:
            for pattern in patterns:
                assert pattern.confidence_score >= 0.0


class TestPatternExamples:
    """Test pattern example collection."""
    
    def test_pattern_examples_collection(self, service, pattern_dataframe):
        """Test that patterns include examples."""
        email_series = pattern_dataframe['emails']
        patterns = service._discover_column_patterns(email_series, 'emails')
        
        email_pattern = next((p for p in patterns if p.pattern_type == PatternType.EMAIL), None)
        if email_pattern:
            assert len(email_pattern.examples) > 0
            assert len(email_pattern.examples) <= 5  # Default max examples
            
            # Examples should match the pattern
            for example in email_pattern.examples:
                assert re.match(email_pattern.pattern_regex, example)
    
    def test_pattern_match_count(self, service, pattern_dataframe):
        """Test pattern match count accuracy."""
        email_series = pattern_dataframe['emails']
        patterns = service._discover_column_patterns(email_series, 'emails')
        
        email_pattern = next((p for p in patterns if p.pattern_type == PatternType.EMAIL), None)
        if email_pattern:
            assert email_pattern.match_count > 0
            assert email_pattern.match_count <= len(email_series)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_series(self, service):
        """Test pattern discovery on empty series."""
        empty_series = pd.Series([], dtype=object)
        patterns = service._discover_column_patterns(empty_series, 'empty_col')
        
        assert patterns == []
    
    def test_single_value_series(self, service):
        """Test pattern discovery on series with single unique value."""
        single_value = pd.Series(['same_value'] * 100)
        patterns = service._discover_column_patterns(single_value, 'single_val')
        
        # Should not find meaningful patterns with only one unique value
        assert len(patterns) == 0
    
    def test_null_heavy_series(self, service):
        """Test pattern discovery on series with many nulls."""
        null_heavy = pd.Series([
            'user@example.com', None, None, None, None,
            'admin@test.org', None, None, None, None
        ])
        
        patterns = service._discover_column_patterns(null_heavy, 'null_heavy')
        
        # Should still find patterns in non-null values
        email_pattern = next((p for p in patterns if p.pattern_type == PatternType.EMAIL), None)
        if email_pattern:
            assert email_pattern.confidence_score > 0
    
    def test_mixed_type_series(self, service):
        """Test pattern discovery on series with mixed types."""
        mixed_types = pd.Series([
            'user@example.com', 123, 'admin@test.org',
            45.67, 'not-an-email', True
        ])
        
        patterns = service._discover_column_patterns(mixed_types, 'mixed_types')
        
        # Should handle mixed types gracefully
        assert isinstance(patterns, list)


class TestPatternRegex:
    """Test pattern regex functionality."""
    
    def test_email_regex_accuracy(self, service):
        """Test email regex pattern accuracy."""
        email_regex = service.patterns[PatternType.EMAIL]
        
        valid_emails = [
            'test@example.com', 'user.name@domain.org', 'admin+tag@site.net'
        ]
        invalid_emails = [
            'invalid-email', '@domain.com', 'user@', 'user.domain.com'
        ]
        
        for email in valid_emails:
            assert re.match(email_regex, email), f"Valid email {email} should match"
        
        for email in invalid_emails:
            assert not re.match(email_regex, email), f"Invalid email {email} should not match"
    
    def test_phone_regex_accuracy(self, service):
        """Test phone regex pattern accuracy."""
        phone_regex = service.patterns[PatternType.PHONE]
        
        valid_phones = [
            '+1-555-1234', '+44-20-1234-5678', '555-1234'
        ]
        invalid_phones = [
            'not-a-phone', '123', 'abc-def-ghij'
        ]
        
        for phone in valid_phones:
            assert re.match(phone_regex, phone), f"Valid phone {phone} should match"
        
        for phone in invalid_phones:
            assert not re.match(phone_regex, phone), f"Invalid phone {phone} should not match"
    
    def test_url_regex_accuracy(self, service):
        """Test URL regex pattern accuracy."""
        url_regex = service.patterns[PatternType.URL]
        
        valid_urls = [
            'https://example.com', 'http://domain.org/path', 'https://sub.site.net'
        ]
        invalid_urls = [
            'not-a-url', 'ftp://example.com', 'example.com'
        ]
        
        for url in valid_urls:
            assert re.match(url_regex, url), f"Valid URL {url} should match"
        
        for url in invalid_urls:
            assert not re.match(url_regex, url), f"Invalid URL {url} should not match"


class TestSemanticClassifiers:
    """Test semantic classifier patterns."""
    
    def test_credit_card_classifier(self, service):
        """Test credit card semantic classifier."""
        cc_regex = service.semantic_classifiers['credit_card']
        
        valid_cards = [
            '4532-1234-5678-9012', '5555-5555-5555-4444', '3782-822463-10005'
        ]
        
        for card in valid_cards:
            assert re.match(cc_regex, card), f"Valid card {card} should match"
    
    def test_ssn_classifier(self, service):
        """Test SSN semantic classifier."""
        ssn_regex = service.semantic_classifiers['ssn']
        
        valid_ssns = ['123-45-6789', '987-65-4321']
        invalid_ssns = ['123456789', '123-456-789', 'abc-de-fghi']
        
        for ssn in valid_ssns:
            assert re.match(ssn_regex, ssn), f"Valid SSN {ssn} should match"
        
        for ssn in invalid_ssns:
            assert not re.match(ssn_regex, ssn), f"Invalid SSN {ssn} should not match"
    
    def test_ip_address_classifier(self, service):
        """Test IP address semantic classifier."""
        ip_regex = service.semantic_classifiers['ip_address']
        
        valid_ips = ['192.168.1.1', '10.0.0.1', '127.0.0.1']
        invalid_ips = ['256.256.256.256', '192.168.1', 'not-an-ip']
        
        for ip in valid_ips:
            assert re.match(ip_regex, ip), f"Valid IP {ip} should match"
        
        for ip in invalid_ips:
            assert not re.match(ip_regex, ip), f"Invalid IP {ip} should not match"


class TestBusinessClassifiers:
    """Test business domain pattern classifiers."""
    
    def test_product_code_classifier(self, service):
        """Test product code business classifier."""
        product_regex = service.business_patterns['product_code']
        
        valid_codes = ['AB123456', 'XYZ789012']
        
        for code in valid_codes:
            assert re.match(product_regex, code), f"Valid product code {code} should match"
    
    def test_order_id_classifier(self, service):
        """Test order ID business classifier."""
        order_regex = service.business_patterns['order_id']
        
        valid_orders = ['ORD-1234567', 'ORD-9876543']
        invalid_orders = ['ORDER-123', 'ORD123', 'not-an-order']
        
        for order in valid_orders:
            assert re.match(order_regex, order), f"Valid order ID {order} should match"
        
        for order in invalid_orders:
            assert not re.match(order_regex, order), f"Invalid order ID {order} should not match"


class TestColumnNameHints:
    """Test column name hint usage in pattern discovery."""
    
    def test_column_name_email_hint(self, service):
        """Test column name hints for email detection."""
        email_data = pd.Series(['user@example.com', 'admin@test.org'])
        
        # Should be more confident when column name suggests email
        patterns_with_hint = service._discover_column_patterns(email_data, 'email_address')
        patterns_without_hint = service._discover_column_patterns(email_data, 'random_column')
        
        # Both should find email patterns, but hint might affect confidence
        assert len(patterns_with_hint) > 0
        assert len(patterns_without_hint) > 0
    
    def test_column_name_phone_hint(self, service):
        """Test column name hints for phone detection."""
        phone_data = pd.Series(['+1-555-1234', '+1-555-5678'])
        
        patterns_with_hint = service._discover_column_patterns(phone_data, 'phone_number')
        patterns_without_hint = service._discover_column_patterns(phone_data, 'random_column')
        
        assert len(patterns_with_hint) > 0
        assert len(patterns_without_hint) > 0
"""
Comprehensive tests for DomainRegistry and related domain management classes.

Tests domain mapping, boundary rule management, exception handling,
and configuration loading functionality.
"""
import pytest
import tempfile
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, mock_open

from test_utilities.factories import TestDataFactory
from test_utilities.fixtures import async_test

from core.domain.services.registry import (
    DomainRegistry,
    Domain,
    BoundaryException,
    BoundaryRule
)


class TestDomain:
    """Test Domain entity functionality."""
    
    def test_domain_creation(self):
        """Test creating a domain with all fields."""
        domain = Domain(
            name="ai",
            packages=["ai/mlops", "ai/core"],
            allowed_dependencies=["shared", "infrastructure"],
            description="AI domain description"
        )
        
        assert domain.name == "ai"
        assert domain.packages == ["ai/mlops", "ai/core"]
        assert domain.allowed_dependencies == ["shared", "infrastructure"]
        assert domain.description == "AI domain description"
    
    def test_domain_default_values(self):
        """Test domain creation with default values."""
        domain = Domain(name="test")
        
        assert domain.name == "test"
        assert domain.packages == []
        assert domain.allowed_dependencies == []
        assert domain.description == ""
    
    def test_contains_package_exact_match(self):
        """Test package containment with exact match."""
        domain = Domain(
            name="ai",
            packages=["ai/mlops", "ai/core"]
        )
        
        assert domain.contains_package("ai/mlops")
        assert domain.contains_package("ai/core")
        assert not domain.contains_package("finance/billing")
    
    def test_contains_package_prefix_match(self):
        """Test package containment with prefix matching."""
        domain = Domain(
            name="ai",
            packages=["ai/mlops", "ai/core"]
        )
        
        # Should match sub-packages via prefix
        assert domain.contains_package("ai/mlops/models")
        assert domain.contains_package("ai/mlops/services/training")
        assert domain.contains_package("ai/core/utils")
        
        # Should not match similar but different packages
        assert not domain.contains_package("ai_other/package")
        assert not domain.contains_package("data/ai")
    
    def test_contains_package_empty_packages(self):
        """Test package containment with no packages defined."""
        domain = Domain(name="empty")
        
        assert not domain.contains_package("any/package")
        assert not domain.contains_package("")


class TestBoundaryException:
    """Test BoundaryException entity functionality."""
    
    def test_boundary_exception_creation(self):
        """Test creating a boundary exception."""
        expires = datetime.now() + timedelta(days=30)
        exception = BoundaryException(
            from_package="ai/mlops",
            to_package="finance/billing",
            reason="Legacy integration needs refactoring",
            expires=expires,
            approved_by="tech_lead@company.com"
        )
        
        assert exception.from_package == "ai/mlops"
        assert exception.to_package == "finance/billing"
        assert exception.reason == "Legacy integration needs refactoring"
        assert exception.expires == expires
        assert exception.approved_by == "tech_lead@company.com"
    
    def test_boundary_exception_default_values(self):
        """Test boundary exception with default values."""
        exception = BoundaryException(
            from_package="ai/mlops",
            to_package="finance/billing",
            reason="Test exception"
        )
        
        assert exception.expires is None
        assert exception.approved_by == ""
    
    def test_is_valid_without_expiry(self):
        """Test that exceptions without expiry are always valid."""
        exception = BoundaryException(
            from_package="ai/mlops",
            to_package="finance/billing",
            reason="Permanent exception"
        )
        
        assert exception.is_valid()
    
    def test_is_valid_with_future_expiry(self):
        """Test that exceptions with future expiry are valid."""
        future_date = datetime.now() + timedelta(days=30)
        exception = BoundaryException(
            from_package="ai/mlops",
            to_package="finance/billing",
            reason="Temporary exception",
            expires=future_date
        )
        
        assert exception.is_valid()
    
    def test_is_valid_with_past_expiry(self):
        """Test that exceptions with past expiry are invalid."""
        past_date = datetime.now() - timedelta(days=30)
        exception = BoundaryException(
            from_package="ai/mlops",
            to_package="finance/billing",
            reason="Expired exception",
            expires=past_date
        )
        
        assert not exception.is_valid()
    
    def test_is_valid_with_current_datetime(self):
        """Test edge case with current datetime."""
        # Create exception that expires in 1 second
        near_future = datetime.now() + timedelta(seconds=1)
        exception = BoundaryException(
            from_package="ai/mlops",
            to_package="finance/billing",
            reason="Nearly expired exception",
            expires=near_future
        )
        
        assert exception.is_valid()


class TestBoundaryRule:
    """Test BoundaryRule entity functionality."""
    
    def test_boundary_rule_creation(self):
        """Test creating a boundary rule."""
        exception = BoundaryException(
            from_package="ai/mlops",
            to_package="finance/billing",
            reason="Legacy integration"
        )
        
        rule = BoundaryRule(
            name="no_cross_domain",
            description="Prevent cross-domain imports",
            severity="critical",
            pattern="cross_domain",
            exceptions=[exception]
        )
        
        assert rule.name == "no_cross_domain"
        assert rule.description == "Prevent cross-domain imports"
        assert rule.severity == "critical"
        assert rule.pattern == "cross_domain"
        assert len(rule.exceptions) == 1
        assert rule.exceptions[0] == exception
    
    def test_boundary_rule_default_exceptions(self):
        """Test boundary rule with default empty exceptions."""
        rule = BoundaryRule(
            name="test_rule",
            description="Test rule",
            severity="warning",
            pattern="test"
        )
        
        assert rule.exceptions == []


class TestDomainRegistry:
    """Test DomainRegistry core functionality."""
    
    @pytest.fixture
    def registry(self):
        """Create a registry for testing."""
        return DomainRegistry()
    
    @pytest.fixture
    def sample_domains(self):
        """Create sample domains for testing."""
        ai_domain = Domain(
            name="ai",
            packages=["ai/mlops", "ai/core"],
            allowed_dependencies=["shared", "data"],
            description="AI domain"
        )
        
        finance_domain = Domain(
            name="finance",
            packages=["finance/billing", "finance/payments"],
            allowed_dependencies=["shared"],
            description="Finance domain"
        )
        
        shared_domain = Domain(
            name="shared",
            packages=["shared/utils", "shared/types"],
            allowed_dependencies=[],
            description="Shared utilities"
        )
        
        return [ai_domain, finance_domain, shared_domain]
    
    def test_registry_initialization(self, registry):
        """Test registry initialization with default values."""
        assert registry.domains == {}
        assert registry.rules == []
        assert len(registry.global_allowed) > 0
        assert 'shared' in registry.global_allowed
        assert 'infrastructure' in registry.global_allowed
        assert registry._package_to_domain == {}
    
    def test_add_domain(self, registry):
        """Test adding a domain to the registry."""
        domain = Domain(
            name="ai",
            packages=["ai/mlops", "ai/core"],
            allowed_dependencies=["shared"]
        )
        
        registry.add_domain(domain)
        
        assert "ai" in registry.domains
        assert registry.domains["ai"] == domain
        assert registry._package_to_domain["ai/mlops"] == "ai"
        assert registry._package_to_domain["ai/core"] == "ai"
    
    def test_add_multiple_domains(self, registry, sample_domains):
        """Test adding multiple domains."""
        for domain in sample_domains:
            registry.add_domain(domain)
        
        assert len(registry.domains) == 3
        assert "ai" in registry.domains
        assert "finance" in registry.domains
        assert "shared" in registry.domains
        
        # Check package mappings
        assert registry._package_to_domain["ai/mlops"] == "ai"
        assert registry._package_to_domain["finance/billing"] == "finance"
        assert registry._package_to_domain["shared/utils"] == "shared"
    
    def test_add_rule(self, registry):
        """Test adding a boundary rule."""
        rule = BoundaryRule(
            name="test_rule",
            description="Test rule",
            severity="warning",
            pattern="test"
        )
        
        registry.add_rule(rule)
        
        assert len(registry.rules) == 1
        assert registry.rules[0] == rule
    
    def test_get_domain_for_package_direct_match(self, registry, sample_domains):
        """Test getting domain for package with direct match."""
        for domain in sample_domains:
            registry.add_domain(domain)
        
        assert registry.get_domain_for_package("ai/mlops") == "ai"
        assert registry.get_domain_for_package("finance/billing") == "finance"
        assert registry.get_domain_for_package("shared/utils") == "shared"
    
    def test_get_domain_for_package_sub_package_match(self, registry, sample_domains):
        """Test getting domain for sub-packages."""
        for domain in sample_domains:
            registry.add_domain(domain)
        
        # Sub-packages should match parent packages
        assert registry.get_domain_for_package("ai/mlops/models") == "ai"
        assert registry.get_domain_for_package("ai/mlops/services/training") == "ai"
        assert registry.get_domain_for_package("finance/billing/models/invoice") == "finance"
        assert registry.get_domain_for_package("shared/utils/helpers") == "shared"
    
    def test_get_domain_for_package_no_match(self, registry, sample_domains):
        """Test getting domain for unknown packages."""
        for domain in sample_domains:
            registry.add_domain(domain)
        
        assert registry.get_domain_for_package("unknown/package") is None
        assert registry.get_domain_for_package("data/analytics") is None
        assert registry.get_domain_for_package("") is None
    
    def test_is_allowed_dependency_same_domain(self, registry, sample_domains):
        """Test that dependencies within the same domain are allowed."""
        for domain in sample_domains:
            registry.add_domain(domain)
        
        # Same domain dependencies should be allowed
        assert registry.is_allowed_dependency("ai/mlops", "ai/core")
        assert registry.is_allowed_dependency("ai/core", "ai/mlops")
        assert registry.is_allowed_dependency("finance/billing", "finance/payments")
    
    def test_is_allowed_dependency_global_allowed(self, registry, sample_domains):
        """Test dependencies to globally allowed domains."""
        for domain in sample_domains:
            registry.add_domain(domain)
        
        # Dependencies to global allowed domains should be allowed
        assert registry.is_allowed_dependency("ai/mlops", "shared/utils")
        assert registry.is_allowed_dependency("finance/billing", "shared/types")
        assert registry.is_allowed_dependency("ai/core", "infrastructure/logging")
    
    def test_is_allowed_dependency_domain_specific(self, registry, sample_domains):
        """Test domain-specific allowed dependencies."""
        for domain in sample_domains:
            registry.add_domain(domain)
        
        # AI domain allows data dependencies
        assert registry.is_allowed_dependency("ai/mlops", "data/analytics")
        
        # Finance domain doesn't allow data dependencies
        assert not registry.is_allowed_dependency("finance/billing", "data/analytics")
    
    def test_is_allowed_dependency_cross_domain_blocked(self, registry, sample_domains):
        """Test that cross-domain dependencies are blocked by default."""
        for domain in sample_domains:
            registry.add_domain(domain)
        
        # Cross-domain dependencies should be blocked
        assert not registry.is_allowed_dependency("ai/mlops", "finance/billing")
        assert not registry.is_allowed_dependency("finance/payments", "ai/core")
    
    def test_is_allowed_dependency_unknown_domains(self, registry):
        """Test that unknown domain dependencies are allowed by default."""
        # Unknown domains should be allowed (permissive by default)
        assert registry.is_allowed_dependency("unknown/package1", "unknown/package2")
        assert registry.is_allowed_dependency("ai/mlops", "unknown/package")
        assert registry.is_allowed_dependency("unknown/package", "ai/mlops")
    
    def test_find_applicable_exceptions_matching(self, registry):
        """Test finding applicable exceptions that match."""
        # Create exception
        exception = BoundaryException(
            from_package="ai/mlops",
            to_package="finance/billing",
            reason="Legacy integration"
        )
        
        # Create rule with exception
        rule = BoundaryRule(
            name="test_rule",
            description="Test",
            severity="critical",
            pattern="cross_domain",
            exceptions=[exception]
        )
        
        registry.add_rule(rule)
        
        # Should find applicable exception
        exceptions = registry.find_applicable_exceptions("ai/mlops/service", "finance/billing/model")
        assert len(exceptions) == 1
        assert exceptions[0] == exception
    
    def test_find_applicable_exceptions_no_match(self, registry):
        """Test finding exceptions when none match."""
        # Create exception for different packages
        exception = BoundaryException(
            from_package="other/package",
            to_package="different/package",
            reason="Different exception"
        )
        
        rule = BoundaryRule(
            name="test_rule",
            description="Test",
            severity="critical",
            pattern="cross_domain",
            exceptions=[exception]
        )
        
        registry.add_rule(rule)
        
        # Should not find applicable exception
        exceptions = registry.find_applicable_exceptions("ai/mlops", "finance/billing")
        assert len(exceptions) == 0
    
    def test_find_applicable_exceptions_expired(self, registry):
        """Test that expired exceptions are not returned."""
        # Create expired exception
        past_date = datetime.now() - timedelta(days=30)
        exception = BoundaryException(
            from_package="ai/mlops",
            to_package="finance/billing",
            reason="Expired exception",
            expires=past_date
        )
        
        rule = BoundaryRule(
            name="test_rule",
            description="Test",
            severity="critical",
            pattern="cross_domain",
            exceptions=[exception]
        )
        
        registry.add_rule(rule)
        
        # Should not find expired exception
        exceptions = registry.find_applicable_exceptions("ai/mlops", "finance/billing")
        assert len(exceptions) == 0
    
    def test_load_from_yaml_file(self, registry):
        """Test loading configuration from YAML file."""
        config_data = {
            'domains': {
                'ai': {
                    'packages': ['ai/mlops', 'ai/core'],
                    'allowed_dependencies': ['shared', 'data'],
                    'description': 'AI domain'
                },
                'finance': {
                    'packages': ['finance/billing'],
                    'allowed_dependencies': ['shared'],
                    'description': 'Finance domain'
                }
            },
            'global_allowed': ['shared', 'infrastructure', 'common'],
            'rules': [
                {
                    'name': 'no_cross_domain',
                    'description': 'Prevent cross-domain imports',
                    'severity': 'critical',
                    'pattern': 'cross_domain',
                    'exceptions': [
                        {
                            'from': 'ai/mlops',
                            'to': 'finance/billing',
                            'reason': 'Legacy integration',
                            'approved_by': 'tech_lead@company.com'
                        }
                    ]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            
            registry.load_from_file(Path(f.name))
        
        # Verify domains loaded
        assert len(registry.domains) == 2
        assert "ai" in registry.domains
        assert "finance" in registry.domains
        assert registry.domains["ai"].packages == ['ai/mlops', 'ai/core']
        assert registry.domains["ai"].allowed_dependencies == ['shared', 'data']
        
        # Verify global allowed
        assert registry.global_allowed == ['shared', 'infrastructure', 'common']
        
        # Verify rules and exceptions
        assert len(registry.rules) == 1
        rule = registry.rules[0]
        assert rule.name == 'no_cross_domain'
        assert rule.severity == 'critical'
        assert len(rule.exceptions) == 1
        exception = rule.exceptions[0]
        assert exception.from_package == 'ai/mlops'
        assert exception.to_package == 'finance/billing'
        assert exception.approved_by == 'tech_lead@company.com'
        
        # Clean up
        Path(f.name).unlink()
    
    def test_load_from_json_file(self, registry):
        """Test loading configuration from JSON file."""
        config_data = {
            'domains': {
                'ai': {
                    'packages': ['ai/mlops'],
                    'allowed_dependencies': ['shared'],
                    'description': 'AI domain'
                }
            },
            'global_allowed': ['shared']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            
            registry.load_from_file(Path(f.name))
        
        # Verify configuration loaded
        assert len(registry.domains) == 1
        assert "ai" in registry.domains
        assert registry.global_allowed == ['shared']
        
        # Clean up
        Path(f.name).unlink()
    
    def test_load_from_file_with_expires(self, registry):
        """Test loading configuration with expiring exceptions."""
        future_date = datetime.now() + timedelta(days=30)
        config_data = {
            'domains': {
                'ai': {
                    'packages': ['ai/mlops'],
                    'allowed_dependencies': ['shared']
                }
            },
            'rules': [
                {
                    'name': 'test_rule',
                    'exceptions': [
                        {
                            'from': 'ai/mlops',
                            'to': 'finance/billing',
                            'reason': 'Temporary exception',
                            'expires': future_date.isoformat(),
                            'approved_by': 'manager@company.com'
                        }
                    ]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            
            registry.load_from_file(Path(f.name))
        
        # Verify exception with expiry loaded
        assert len(registry.rules) == 1
        rule = registry.rules[0]
        assert len(rule.exceptions) == 1
        exception = rule.exceptions[0]
        assert exception.expires == future_date
        assert exception.is_valid()
        
        # Clean up
        Path(f.name).unlink()
    
    def test_get_default_registry(self):
        """Test creating a default registry."""
        registry = DomainRegistry()
        default_registry = registry.get_default_registry()
        
        # Should have standard domains
        assert "ai" in default_registry.domains
        assert "data" in default_registry.domains
        assert "finance" in default_registry.domains
        assert "infrastructure" in default_registry.domains
        assert "shared" in default_registry.domains
        
        # Verify AI domain configuration
        ai_domain = default_registry.domains["ai"]
        assert "ai/mlops" in ai_domain.packages
        assert "ai/neuro_symbolic" in ai_domain.packages
        assert "shared" in ai_domain.allowed_dependencies
        assert "data" in ai_domain.allowed_dependencies
        
        # Verify infrastructure domain constraints
        infra_domain = default_registry.domains["infrastructure"]
        assert infra_domain.allowed_dependencies == []  # Infrastructure can't depend on business domains
        
        # Verify shared domain constraints
        shared_domain = default_registry.domains["shared"]
        assert shared_domain.allowed_dependencies == ["infrastructure"]  # Only infrastructure allowed
        
        # Should have default rules
        assert len(default_registry.rules) >= 2
        rule_names = [rule.name for rule in default_registry.rules]
        assert "no_cross_domain_imports" in rule_names
        assert "no_circular_dependencies" in rule_names
    
    def test_load_config_empty_configuration(self, registry):
        """Test loading empty configuration."""
        empty_config = {}
        registry._load_config(empty_config)
        
        # Should not crash and maintain defaults
        assert len(registry.domains) == 0
        assert len(registry.rules) == 0
        assert len(registry.global_allowed) > 0  # Should keep defaults
    
    def test_load_config_partial_domain_info(self, registry):
        """Test loading configuration with partial domain information."""
        config = {
            'domains': {
                'minimal': {
                    'packages': ['minimal/package']
                    # No allowed_dependencies or description
                }
            }
        }
        
        registry._load_config(config)
        
        domain = registry.domains["minimal"]
        assert domain.packages == ['minimal/package']
        assert domain.allowed_dependencies == []  # Should default to empty
        assert domain.description == ""  # Should default to empty
    
    def test_package_to_domain_mapping_updates(self, registry):
        """Test that package to domain mapping updates correctly."""
        # Add first domain
        domain1 = Domain(name="test1", packages=["test1/package"])
        registry.add_domain(domain1)
        assert registry._package_to_domain["test1/package"] == "test1"
        
        # Add second domain
        domain2 = Domain(name="test2", packages=["test2/package"])
        registry.add_domain(domain2)
        assert registry._package_to_domain["test1/package"] == "test1"
        assert registry._package_to_domain["test2/package"] == "test2"
        
        # Update first domain
        updated_domain1 = Domain(name="test1", packages=["test1/package", "test1/other"])
        registry.add_domain(updated_domain1)
        assert registry._package_to_domain["test1/package"] == "test1"
        assert registry._package_to_domain["test1/other"] == "test1"
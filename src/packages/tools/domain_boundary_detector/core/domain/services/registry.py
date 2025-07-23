"""Domain registry for mapping packages to their domains and managing rules."""

from typing import Dict, List, Set, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import yaml
import json


@dataclass
class Domain:
    """Represents a domain in the system."""
    name: str
    packages: List[str] = field(default_factory=list)
    allowed_dependencies: List[str] = field(default_factory=list)
    description: str = ""
    
    def contains_package(self, package: str) -> bool:
        """Check if a package belongs to this domain."""
        return any(package.startswith(p) for p in self.packages)


@dataclass
class BoundaryException:
    """Represents an approved exception to boundary rules."""
    from_package: str
    to_package: str
    reason: str
    expires: Optional[datetime] = None
    approved_by: str = ""
    
    def is_valid(self) -> bool:
        """Check if the exception is still valid."""
        if self.expires:
            return datetime.now() < self.expires
        return True


@dataclass
class BoundaryRule:
    """Represents a boundary rule."""
    name: str
    description: str
    severity: str  # critical, warning, info
    pattern: str
    exceptions: List[BoundaryException] = field(default_factory=list)


class DomainRegistry:
    """Registry for managing domains and their boundaries."""
    
    def __init__(self):
        self.domains: Dict[str, Domain] = {}
        self.rules: List[BoundaryRule] = []
        self.global_allowed: List[str] = ['shared', 'common', 'infrastructure', 'interfaces']
        self._package_to_domain: Dict[str, str] = {}
        
    def add_domain(self, domain: Domain) -> None:
        """Add a domain to the registry."""
        self.domains[domain.name] = domain
        # Update package to domain mapping
        for package in domain.packages:
            self._package_to_domain[package] = domain.name
            
    def add_rule(self, rule: BoundaryRule) -> None:
        """Add a boundary rule."""
        self.rules.append(rule)
        
    def get_domain_for_package(self, package: str) -> Optional[str]:
        """Get the domain name for a package."""
        # Direct lookup first
        if package in self._package_to_domain:
            return self._package_to_domain[package]
            
        # Check if package is a sub-package
        for pkg, domain in self._package_to_domain.items():
            if package.startswith(pkg + '/'):
                return domain
                
        return None
        
    def is_allowed_dependency(self, from_package: str, to_package: str) -> bool:
        """Check if a dependency is allowed between packages."""
        from_domain = self.get_domain_for_package(from_package)
        to_domain = self.get_domain_for_package(to_package)
        
        if not from_domain or not to_domain:
            return True  # Unknown domains are allowed by default
            
        # Same domain is always allowed
        if from_domain == to_domain:
            return True
            
        # Check global allowed
        if to_domain in self.global_allowed:
            return True
            
        # Check domain-specific allowed dependencies
        domain = self.domains.get(from_domain)
        if domain and to_domain in domain.allowed_dependencies:
            return True
            
        return False
        
    def find_applicable_exceptions(self, from_package: str, to_package: str) -> List[BoundaryException]:
        """Find exceptions that apply to a given dependency."""
        applicable = []
        
        for rule in self.rules:
            for exception in rule.exceptions:
                if (exception.from_package in from_package and 
                    exception.to_package in to_package and
                    exception.is_valid()):
                    applicable.append(exception)
                    
        return applicable
        
    def load_from_file(self, config_path: Path) -> None:
        """Load domain configuration from a YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
                
        self._load_config(config)
        
    def _load_config(self, config: Dict[str, Any]) -> None:
        """Load configuration from a dictionary."""
        # Load domains
        if 'domains' in config:
            for domain_name, domain_config in config['domains'].items():
                domain = Domain(
                    name=domain_name,
                    packages=domain_config.get('packages', []),
                    allowed_dependencies=domain_config.get('allowed_dependencies', []),
                    description=domain_config.get('description', '')
                )
                self.add_domain(domain)
                
        # Load global allowed
        if 'global_allowed' in config:
            self.global_allowed = config['global_allowed']
            
        # Load rules
        if 'rules' in config:
            for rule_config in config['rules']:
                rule = BoundaryRule(
                    name=rule_config['name'],
                    description=rule_config.get('description', ''),
                    severity=rule_config.get('severity', 'warning'),
                    pattern=rule_config.get('pattern', '')
                )
                
                # Load exceptions
                if 'exceptions' in rule_config:
                    for exc_config in rule_config['exceptions']:
                        expires = None
                        if 'expires' in exc_config:
                            expires = datetime.fromisoformat(exc_config['expires'])
                            
                        exception = BoundaryException(
                            from_package=exc_config['from'],
                            to_package=exc_config['to'],
                            reason=exc_config.get('reason', ''),
                            expires=expires,
                            approved_by=exc_config.get('approved_by', '')
                        )
                        rule.exceptions.append(exception)
                        
                self.add_rule(rule)
                
    def get_default_registry(self) -> 'DomainRegistry':
        """Create a default registry with common domains."""
        registry = DomainRegistry()
        
        # AI domain
        ai_domain = Domain(
            name='ai',
            packages=['ai/mlops', 'ai/ml_platform', 'ai/neuro_symbolic'],
            allowed_dependencies=['shared', 'infrastructure', 'data'],
            description='Artificial Intelligence and Machine Learning'
        )
        registry.add_domain(ai_domain)
        
        # Data domain
        data_domain = Domain(
            name='data',
            packages=['data/analytics', 'data/ingestion', 'data/quality', 'data/pipelines'],
            allowed_dependencies=['shared', 'infrastructure'],
            description='Data processing and management'
        )
        registry.add_domain(data_domain)
        
        # Finance domain
        finance_domain = Domain(
            name='finance',
            packages=['finance/billing', 'finance/payments', 'finance/accounting'],
            allowed_dependencies=['shared', 'infrastructure', 'data'],
            description='Financial services and billing'
        )
        registry.add_domain(finance_domain)
        
        # Infrastructure domain
        infra_domain = Domain(
            name='infrastructure',
            packages=['infrastructure/logging', 'infrastructure/monitoring', 'infrastructure/config'],
            allowed_dependencies=[],  # Infrastructure can't depend on business domains
            description='Technical infrastructure and cross-cutting concerns'
        )
        registry.add_domain(infra_domain)
        
        # Shared domain
        shared_domain = Domain(
            name='shared',
            packages=['shared/utils', 'shared/types', 'shared/constants'],
            allowed_dependencies=['infrastructure'],  # Shared can only depend on infrastructure
            description='Shared utilities and common code'
        )
        registry.add_domain(shared_domain)
        
        # Add default rules
        no_cross_domain = BoundaryRule(
            name='no_cross_domain_imports',
            description='Prevent direct imports between different business domains',
            severity='critical',
            pattern='cross_domain'
        )
        registry.add_rule(no_cross_domain)
        
        no_circular = BoundaryRule(
            name='no_circular_dependencies',
            description='Prevent circular dependencies between packages',
            severity='critical',
            pattern='circular'
        )
        registry.add_rule(no_circular)
        
        return registry
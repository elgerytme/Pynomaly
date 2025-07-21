#!/usr/bin/env python3
"""
Intelligent Package Generator - Domain-Aware Self-Contained Package Creation
===========================================================================
Integrates with domain boundary system to create intelligent, self-contained packages
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import subprocess
import re
import logging
from datetime import datetime

# Add the project root to sys.path to import our domain utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.domain_boundary_validator import DomainBoundaryValidator, DomainSuggestion
    from scripts.create_domain_package import DomainPackageCreator
except ImportError as e:
    print(f"Warning: Could not import domain utilities: {e}")
    DomainBoundaryValidator = None
    DomainPackageCreator = None


@dataclass
class IntelligentPackageConfig:
    """Enhanced configuration for intelligent package generation"""
    # Basic package information
    package_name: str
    package_description: str
    domain_name: str
    author_name: str
    author_email: str
    github_username: str
    repository_name: str
    
    # Technical configuration
    docker_registry: str = "docker.io"
    package_version: str = "0.1.0"
    python_version: str = "3.11"
    license: str = "MIT"
    
    # Architecture decisions
    architecture_pattern: str = "hexagonal"  # hexagonal, clean, layered
    async_framework: bool = True
    web_framework: str = "fastapi"  # fastapi, flask, django
    
    # Infrastructure components
    use_database: bool = True
    database_type: str = "postgresql"
    use_cache: bool = True
    cache_type: str = "redis"
    use_message_queue: bool = False
    message_queue_type: str = "rabbitmq"
    use_search_engine: bool = False
    search_engine_type: str = "elasticsearch"
    
    # Deployment and operations
    use_kubernetes: bool = True
    use_docker: bool = True
    use_monitoring: bool = True
    use_tracing: bool = True
    use_logging: bool = True
    
    # Quality and security
    use_security_scanning: bool = True
    use_performance_testing: bool = True
    use_mutation_testing: bool = False
    code_coverage_threshold: int = 90
    
    # AI/ML specific (for anomaly detection packages)
    use_mlflow: bool = False
    use_tensorflow: bool = False
    use_pytorch: bool = False
    use_scikit_learn: bool = True
    use_jupyter: bool = False
    
    # Domain-specific features
    domain_features: List[str] = None
    integration_points: List[str] = None
    external_apis: List[str] = None


class IntelligentPackageGenerator:
    """
    Intelligent package generator that creates domain-aware, self-contained packages
    with AI-driven architecture decisions and comprehensive automation
    """
    
    def __init__(self, templates_dir: Path, output_dir: Path):
        self.templates_dir = templates_dir
        self.output_dir = output_dir
        self.template_vars: Dict[str, Any] = {}
        self.logger = self._setup_logging()
        
        # Initialize domain validator if available
        self.domain_validator = None
        if DomainBoundaryValidator:
            try:
                self.domain_validator = DomainBoundaryValidator()
            except Exception as e:
                self.logger.warning(f"Could not initialize domain validator: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the generator"""
        logger = logging.getLogger('intelligent_package_generator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_domain_requirements(self, package_name: str, description: str) -> Dict[str, Any]:
        """Analyze domain requirements using AI-driven analysis"""
        self.logger.info(f"Analyzing domain requirements for {package_name}")
        
        analysis = {
            'domain_type': self._classify_domain_type(package_name, description),
            'architectural_patterns': self._suggest_architecture_patterns(description),
            'infrastructure_needs': self._analyze_infrastructure_needs(description),
            'integration_requirements': self._identify_integration_points(description),
            'scalability_requirements': self._assess_scalability_needs(description),
            'security_requirements': self._analyze_security_needs(description),
            'compliance_requirements': self._identify_compliance_needs(description)
        }
        
        # Use domain validator for additional insights
        if self.domain_validator:
            try:
                domain_suggestions = self.domain_validator.suggest_new_domains([description])
                if domain_suggestions:
                    analysis['domain_suggestions'] = [
                        {
                            'name': suggestion.name,
                            'confidence': suggestion.confidence,
                            'reasoning': suggestion.reasoning
                        }
                        for suggestion in domain_suggestions
                    ]
            except Exception as e:
                self.logger.warning(f"Could not get domain suggestions: {e}")
        
        self.logger.info(f"Domain analysis complete: {analysis['domain_type']}")
        return analysis
    
    def _classify_domain_type(self, package_name: str, description: str) -> str:
        """Classify the domain type based on package name and description"""
        text = f"{package_name} {description}".lower()
        
        # Domain classification patterns
        domain_patterns = {
            'data_processing': ['data', 'etl', 'transform', 'process', 'pipeline', 'ingestion'],
            'anomaly_detection': ['anomaly', 'detection', 'outlier', 'fraud', 'monitoring'],
            'machine_learning': ['ml', 'model', 'training', 'prediction', 'classification'],
            'api_service': ['api', 'service', 'endpoint', 'rest', 'graphql', 'web'],
            'authentication': ['auth', 'login', 'security', 'oauth', 'jwt', 'token'],
            'notification': ['notification', 'alert', 'message', 'email', 'sms'],
            'analytics': ['analytics', 'metrics', 'reporting', 'dashboard', 'insights'],
            'storage': ['storage', 'database', 'persistence', 'repository', 'cache'],
            'integration': ['integration', 'connector', 'adapter', 'bridge', 'sync'],
            'workflow': ['workflow', 'orchestration', 'task', 'job', 'scheduler']
        }
        
        for domain_type, keywords in domain_patterns.items():
            if any(keyword in text for keyword in keywords):
                return domain_type
        
        return 'generic'
    
    def _suggest_architecture_patterns(self, description: str) -> List[str]:
        """Suggest appropriate architecture patterns"""
        patterns = []
        description_lower = description.lower()
        
        # Pattern analysis
        if any(word in description_lower for word in ['domain', 'business', 'entity']):
            patterns.append('domain_driven_design')
        
        if any(word in description_lower for word in ['event', 'message', 'async']):
            patterns.append('event_driven')
        
        if any(word in description_lower for word in ['microservice', 'service', 'api']):
            patterns.append('microservices')
        
        if any(word in description_lower for word in ['pipeline', 'stream', 'flow']):
            patterns.append('pipeline')
        
        # Default to hexagonal if no specific pattern detected
        if not patterns:
            patterns.append('hexagonal')
        
        return patterns
    
    def _analyze_infrastructure_needs(self, description: str) -> Dict[str, bool]:
        """Analyze infrastructure requirements from description"""
        description_lower = description.lower()
        
        return {
            'database': any(word in description_lower for word in [
                'database', 'persistent', 'store', 'data', 'record'
            ]),
            'cache': any(word in description_lower for word in [
                'cache', 'fast', 'performance', 'redis', 'memory'
            ]),
            'message_queue': any(word in description_lower for word in [
                'queue', 'message', 'async', 'background', 'task'
            ]),
            'search': any(word in description_lower for word in [
                'search', 'index', 'elasticsearch', 'query', 'find'
            ]),
            'monitoring': any(word in description_lower for word in [
                'monitor', 'metric', 'observe', 'track', 'alert'
            ]),
            'tracing': any(word in description_lower for word in [
                'trace', 'debug', 'distributed', 'observability'
            ])
        }
    
    def _identify_integration_points(self, description: str) -> List[str]:
        """Identify potential integration points"""
        integrations = []
        description_lower = description.lower()
        
        integration_patterns = {
            'external_api': ['api', 'rest', 'graphql', 'webhook', 'http'],
            'database': ['database', 'sql', 'nosql', 'postgres', 'mongo'],
            'file_system': ['file', 'storage', 's3', 'disk', 'upload'],
            'message_queue': ['queue', 'kafka', 'rabbitmq', 'sqs', 'pubsub'],
            'email': ['email', 'smtp', 'mail', 'notification'],
            'authentication': ['auth', 'oauth', 'saml', 'ldap', 'sso']
        }
        
        for integration_type, keywords in integration_patterns.items():
            if any(keyword in description_lower for keyword in keywords):
                integrations.append(integration_type)
        
        return integrations
    
    def _assess_scalability_needs(self, description: str) -> Dict[str, Any]:
        """Assess scalability requirements"""
        description_lower = description.lower()
        
        high_scale_indicators = [
            'high volume', 'thousands', 'millions', 'concurrent', 'distributed',
            'cluster', 'load', 'performance', 'scale'
        ]
        
        is_high_scale = any(indicator in description_lower for indicator in high_scale_indicators)
        
        return {
            'expected_scale': 'high' if is_high_scale else 'medium',
            'horizontal_scaling': is_high_scale,
            'caching_strategy': 'distributed' if is_high_scale else 'local',
            'database_sharding': is_high_scale,
            'load_balancing': is_high_scale
        }
    
    def _analyze_security_needs(self, description: str) -> Dict[str, bool]:
        """Analyze security requirements"""
        description_lower = description.lower()
        
        security_indicators = {
            'authentication_required': ['auth', 'login', 'user', 'secure'],
            'authorization_required': ['permission', 'role', 'access', 'privilege'],
            'encryption_required': ['encrypt', 'secure', 'sensitive', 'private'],
            'audit_required': ['audit', 'log', 'track', 'compliance'],
            'rate_limiting_required': ['limit', 'throttle', 'ddos', 'abuse']
        }
        
        security_needs = {}
        for requirement, keywords in security_indicators.items():
            security_needs[requirement] = any(keyword in description_lower for keyword in keywords)
        
        return security_needs
    
    def _identify_compliance_needs(self, description: str) -> List[str]:
        """Identify compliance requirements"""
        description_lower = description.lower()
        compliance_types = []
        
        compliance_patterns = {
            'gdpr': ['gdpr', 'privacy', 'personal data', 'european'],
            'hipaa': ['hipaa', 'health', 'medical', 'patient'],
            'pci_dss': ['pci', 'payment', 'credit card', 'financial'],
            'sox': ['sox', 'financial reporting', 'audit'],
            'iso_27001': ['iso', 'information security', 'security management']
        }
        
        for compliance_type, keywords in compliance_patterns.items():
            if any(keyword in description_lower for keyword in keywords):
                compliance_types.append(compliance_type)
        
        return compliance_types
    
    def generate_intelligent_package(self, config: IntelligentPackageConfig) -> None:
        """Generate an intelligent, domain-aware package"""
        self.logger.info(f"🧠 Generating intelligent package: {config.package_name}")
        
        # Analyze domain requirements
        domain_analysis = self.analyze_domain_requirements(
            config.package_name, 
            config.package_description
        )
        
        # Apply intelligent defaults based on analysis
        config = self._apply_intelligent_defaults(config, domain_analysis)
        
        # Prepare template variables
        self._prepare_intelligent_template_vars(config, domain_analysis)
        
        # Create output directory
        package_dir = self.output_dir / config.package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate domain boundary integration
        self._integrate_domain_boundaries(package_dir, config)
        
        # Copy and process templates with intelligent customization
        self._process_intelligent_templates(package_dir, config, domain_analysis)
        
        # Create intelligent package structure
        self._create_intelligent_package_structure(package_dir, config, domain_analysis)
        
        # Generate intelligent configuration
        self._generate_intelligent_configs(package_dir, config, domain_analysis)
        
        # Generate domain-specific code
        self._generate_domain_specific_code(package_dir, config, domain_analysis)
        
        # Set up development environment
        self._setup_intelligent_dev_environment(package_dir, config)
        
        # Generate documentation
        self._generate_intelligent_documentation(package_dir, config, domain_analysis)
        
        # Initialize repository
        self._initialize_intelligent_repository(package_dir, config)
        
        self.logger.info(f"✅ Intelligent package '{config.package_name}' generated successfully!")
        self._print_next_steps(package_dir, config)
    
    def _apply_intelligent_defaults(
        self, 
        config: IntelligentPackageConfig, 
        domain_analysis: Dict[str, Any]
    ) -> IntelligentPackageConfig:
        """Apply intelligent defaults based on domain analysis"""
        
        # Apply infrastructure decisions
        infra_needs = domain_analysis.get('infrastructure_needs', {})
        if infra_needs.get('database'):
            config.use_database = True
        if infra_needs.get('cache'):
            config.use_cache = True
        if infra_needs.get('message_queue'):
            config.use_message_queue = True
        if infra_needs.get('search'):
            config.use_search_engine = True
        
        # Apply security defaults
        security_needs = domain_analysis.get('security_requirements', {})
        if any(security_needs.values()):
            config.use_security_scanning = True
        
        # Apply scalability defaults
        scalability = domain_analysis.get('scalability_requirements', {})
        if scalability.get('expected_scale') == 'high':
            config.use_kubernetes = True
            config.use_monitoring = True
            config.use_tracing = True
        
        # Domain-specific defaults
        domain_type = domain_analysis.get('domain_type')
        if domain_type == 'machine_learning':
            config.use_mlflow = True
            config.use_jupyter = True
            config.use_scikit_learn = True
        elif domain_type == 'anomaly_detection':
            config.use_scikit_learn = True
            config.use_monitoring = True
            config.use_performance_testing = True
        elif domain_type == 'api_service':
            config.web_framework = 'fastapi'
            config.use_monitoring = True
            config.use_security_scanning = True
        
        return config
    
    def _prepare_intelligent_template_vars(
        self, 
        config: IntelligentPackageConfig, 
        domain_analysis: Dict[str, Any]
    ) -> None:
        """Prepare intelligent template variables"""
        self.template_vars = asdict(config)
        
        # Add analysis results
        self.template_vars.update({
            'domain_analysis': domain_analysis,
            'domain_type': domain_analysis.get('domain_type', 'generic'),
            'architecture_patterns': domain_analysis.get('architectural_patterns', []),
            'integration_points': domain_analysis.get('integration_requirements', []),
            'scalability_level': domain_analysis.get('scalability_requirements', {}).get('expected_scale', 'medium'),
            'security_level': 'high' if domain_analysis.get('security_requirements', {}) else 'standard',
            'compliance_types': domain_analysis.get('compliance_requirements', []),
        })
        
        # Add computed variables
        self.template_vars.update({
            'package_name_upper': config.package_name.upper(),
            'package_name_title': config.package_name.title(),
            'package_slug': re.sub(r'[^a-zA-Z0-9]', '-', config.package_name).lower(),
            'python_module': config.package_name.replace('-', '_'),
            'domain_module': config.domain_name.replace('-', '_'),
            'current_year': str(datetime.now().year),
            'generation_timestamp': datetime.now().isoformat(),
        })
    
    def _integrate_domain_boundaries(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Integrate with domain boundary system"""
        if not DomainPackageCreator:
            self.logger.warning("Domain package creator not available, skipping domain integration")
            return
        
        self.logger.info("🏗️ Integrating with domain boundary system")
        
        try:
            domain_creator = DomainPackageCreator()
            
            # Create domain package structure if it doesn't exist
            domain_package_path = self.output_dir / f"src/packages/{config.domain_name}"
            if not domain_package_path.exists():
                domain_creator.create_domain_package(
                    domain_name=config.domain_name,
                    description=f"Domain package for {config.domain_name}",
                    base_path=str(self.output_dir)
                )
            
            # Add package to domain
            package_path = domain_package_path / config.package_name
            package_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Package integrated into domain: {config.domain_name}")
            
        except Exception as e:
            self.logger.warning(f"Could not integrate with domain boundaries: {e}")
    
    def _process_intelligent_templates(
        self, 
        package_dir: Path, 
        config: IntelligentPackageConfig, 
        domain_analysis: Dict[str, Any]
    ) -> None:
        """Process templates with intelligent customization"""
        self.logger.info("📋 Processing intelligent templates")
        
        template_files = list(self.templates_dir.rglob("*"))
        processed_count = 0
        
        for template_file in template_files:
            if template_file.is_file():
                output_file = self._get_intelligent_output_path(template_file, package_dir)
                
                # Skip files based on intelligent analysis
                if self._should_skip_intelligent_file(template_file, config, domain_analysis):
                    continue
                
                # Create output directory
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Process template with intelligence
                if template_file.suffix == '.template':
                    self._process_intelligent_template_file(template_file, output_file, config, domain_analysis)
                else:
                    shutil.copy2(template_file, output_file)
                
                processed_count += 1
        
        self.logger.info(f"   Processed {processed_count} template files with intelligence")
    
    def _get_intelligent_output_path(self, template_file: Path, package_dir: Path) -> Path:
        """Get intelligent output path for template file"""
        relative_path = template_file.relative_to(self.templates_dir)
        
        # Remove .template extension
        if relative_path.suffix == '.template':
            relative_path = relative_path.with_suffix('')
        
        # Replace template placeholders in path
        path_str = str(relative_path)
        for key, value in self.template_vars.items():
            if isinstance(value, str):
                path_str = path_str.replace(f'{{{key}}}', value)
        
        return package_dir / path_str
    
    def _should_skip_intelligent_file(
        self, 
        template_file: Path, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> bool:
        """Intelligently determine if a file should be skipped"""
        path_str = str(template_file).lower()
        
        # Domain-specific skipping
        domain_type = domain_analysis.get('domain_type')
        if 'ml' in path_str and domain_type not in ['machine_learning', 'anomaly_detection']:
            return True
        
        # Infrastructure-based skipping
        if not config.use_database and any(db in path_str for db in ['postgres', 'database', 'alembic']):
            return True
        
        if not config.use_cache and any(cache in path_str for cache in ['redis', 'cache']):
            return True
        
        if not config.use_message_queue and any(queue in path_str for queue in ['rabbitmq', 'celery', 'queue']):
            return True
        
        if not config.use_kubernetes and ('k8s' in path_str or 'kubernetes' in path_str):
            return True
        
        if not config.use_monitoring and any(mon in path_str for mon in ['prometheus', 'grafana', 'monitoring']):
            return True
        
        # Security-based skipping
        security_needs = domain_analysis.get('security_requirements', {})
        if not any(security_needs.values()) and 'security' in path_str:
            return True
        
        return False
    
    def _process_intelligent_template_file(
        self, 
        template_file: Path, 
        output_file: Path, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> None:
        """Process template file with intelligent customization"""
        try:
            content = template_file.read_text(encoding='utf-8')
            
            # Replace template variables
            for key, value in self.template_vars.items():
                if isinstance(value, str):
                    content = content.replace(f'{{{key}}}', value)
                elif isinstance(value, (list, dict)):
                    content = content.replace(f'{{{key}}}', json.dumps(value, indent=2))
            
            # Apply domain-specific customizations
            content = self._apply_domain_specific_customizations(content, config, domain_analysis)
            
            output_file.write_text(content, encoding='utf-8')
            
            # Set appropriate permissions
            if any(ext in template_file.name for ext in ['.sh', 'health-check']):
                output_file.chmod(0o755)
                
        except Exception as e:
            self.logger.error(f"Error processing {template_file}: {e}")
    
    def _apply_domain_specific_customizations(
        self, 
        content: str, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Apply domain-specific customizations to content"""
        domain_type = domain_analysis.get('domain_type')
        
        if domain_type == 'anomaly_detection':
            # Add anomaly detection specific imports and configurations
            if 'import' in content and 'fastapi' in content:
                content = content.replace(
                    'from fastapi import FastAPI',
                    'from fastapi import FastAPI\nfrom sklearn.ensemble import IsolationForest\nfrom sklearn.preprocessing import StandardScaler'
                )
        
        elif domain_type == 'machine_learning':
            # Add ML specific configurations
            if 'dependencies' in content and 'scikit-learn' not in content:
                content = content.replace(
                    '"fastapi>=0.104.0",',
                    '"fastapi>=0.104.0",\n    "scikit-learn>=1.3.0",\n    "pandas>=2.1.0",\n    "numpy>=1.24.0",'
                )
        
        # Apply security enhancements based on requirements
        security_needs = domain_analysis.get('security_requirements', {})
        if security_needs.get('authentication_required') and 'fastapi' in content:
            content = content.replace(
                'from fastapi import FastAPI',
                'from fastapi import FastAPI, Depends, HTTPException, status\nfrom fastapi.security import HTTPBearer, HTTPAuthorizationCredentials'
            )
        
        return content
    
    def _create_intelligent_package_structure(
        self, 
        package_dir: Path, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> None:
        """Create intelligent package structure based on analysis"""
        self.logger.info("📁 Creating intelligent package structure")
        
        # Base directories
        directories = [
            f"src/{config.package_name}",
            f"src/{config.package_name}/domain",
            f"src/{config.package_name}/infrastructure",
            f"src/{config.package_name}/application",
            f"src/{config.package_name}/interfaces",
            "tests/unit",
            "tests/integration",
            "tests/e2e",
            "docs",
            "config",
            "scripts",
            "data",
            "logs",
            "reports",
        ]
        
        # Add domain-specific directories
        domain_type = domain_analysis.get('domain_type')
        if domain_type in ['machine_learning', 'anomaly_detection']:
            directories.extend([
                f"src/{config.package_name}/models",
                f"src/{config.package_name}/training",
                f"src/{config.package_name}/evaluation",
                "notebooks",
                "experiments",
                "models/saved",
                "data/raw",
                "data/processed",
                "data/features",
            ])
        
        if domain_type == 'api_service':
            directories.extend([
                f"src/{config.package_name}/api",
                f"src/{config.package_name}/middleware",
                f"src/{config.package_name}/schemas",
            ])
        
        # Add infrastructure-specific directories
        if config.use_monitoring:
            directories.extend([
                "monitoring/prometheus/rules",
                "monitoring/grafana/dashboards",
                "monitoring/alerts",
            ])
        
        if config.use_kubernetes:
            directories.extend([
                "k8s/base",
                "k8s/staging",
                "k8s/production",
            ])
        
        if config.use_security_scanning:
            directories.append("security")
        
        # Create directories
        for directory in directories:
            (package_dir / directory).mkdir(parents=True, exist_ok=True)
        
        # Create intelligent __init__.py files
        self._create_intelligent_init_files(package_dir, config, directories)
    
    def _create_intelligent_init_files(
        self, 
        package_dir: Path, 
        config: IntelligentPackageConfig,
        directories: List[str]
    ) -> None:
        """Create intelligent __init__.py files with proper imports"""
        
        # Main package __init__.py
        main_init = package_dir / f"src/{config.package_name}/__init__.py"
        main_init_content = f'''"""
{config.package_name} - {config.package_description}

A self-contained, domain-bounded package with intelligent architecture.

Generated by Intelligent Package Generator on {datetime.now().isoformat()}
"""

__version__ = "{config.package_version}"
__author__ = "{config.author_name}"
__email__ = "{config.author_email}"

# Package metadata
PACKAGE_NAME = "{config.package_name}"
DOMAIN_NAME = "{config.domain_name}"
DESCRIPTION = "{config.package_description}"
'''
        main_init.write_text(main_init_content)
        
        # Create other __init__.py files
        for directory in directories:
            if directory.startswith(f"src/{config.package_name}"):
                init_file = package_dir / directory / "__init__.py"
                if not init_file.exists():
                    module_name = directory.split('/')[-1]
                    init_content = f'"""{module_name.title()} module for {config.package_name}."""\n'
                    init_file.write_text(init_content)
    
    def _generate_intelligent_configs(
        self, 
        package_dir: Path, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> None:
        """Generate intelligent configuration files"""
        self.logger.info("⚙️ Generating intelligent configurations")
        
        # Generate environment-specific configurations
        self._generate_intelligent_env_files(package_dir, config, domain_analysis)
        
        # Generate intelligent logging configuration
        self._generate_logging_config(package_dir, config)
        
        # Generate intelligent monitoring configuration
        if config.use_monitoring:
            self._generate_monitoring_config(package_dir, config)
    
    def _generate_intelligent_env_files(
        self, 
        package_dir: Path, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> None:
        """Generate intelligent environment files"""
        environments = ['development', 'staging', 'production']
        
        for env in environments:
            env_file = package_dir / f".env.{env}"
            env_content = self._generate_env_content_for_environment(env, config, domain_analysis)
            env_file.write_text(env_content)
        
        # Create default .env pointing to development
        default_env = package_dir / ".env"
        default_env.write_text("# Default environment configuration\n# Copy .env.development and customize as needed\n")
    
    def _generate_env_content_for_environment(
        self, 
        environment: str, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Generate environment-specific configuration"""
        lines = [
            f"# {environment.title()} Environment Configuration",
            f"# Generated for {config.package_name} - {config.domain_name} domain",
            "",
            f"ENVIRONMENT={environment}",
            f"PACKAGE_NAME={config.package_name}",
            f"DOMAIN_NAME={config.domain_name}",
            f"PACKAGE_VERSION={config.package_version}",
            "",
            "# Application Settings",
            f"DEBUG={'true' if environment == 'development' else 'false'}",
            f"LOG_LEVEL={'DEBUG' if environment == 'development' else 'INFO'}",
            "",
        ]
        
        # Add database configuration
        if config.use_database:
            db_host = "localhost" if environment == "development" else f"{config.package_name}-{config.database_type}"
            lines.extend([
                "# Database Configuration",
                f"DATABASE_URL={config.database_type}://user:password@{db_host}:5432/{config.package_name}_{environment}",
                f"DATABASE_POOL_SIZE={'5' if environment == 'development' else '20'}",
                "",
            ])
        
        # Add cache configuration
        if config.use_cache:
            cache_host = "localhost" if environment == "development" else f"{config.package_name}-{config.cache_type}"
            lines.extend([
                "# Cache Configuration",
                f"REDIS_URL=redis://{cache_host}:6379/0",
                f"CACHE_TTL={'300' if environment == 'development' else '3600'}",
                "",
            ])
        
        # Add security configuration based on analysis
        security_needs = domain_analysis.get('security_requirements', {})
        if any(security_needs.values()):
            lines.extend([
                "# Security Configuration",
                f"SECRET_KEY={'dev-secret-key' if environment == 'development' else 'CHANGE-IN-PRODUCTION'}",
                f"JWT_SECRET={'dev-jwt-secret' if environment == 'development' else 'CHANGE-IN-PRODUCTION'}",
                f"JWT_EXPIRATION_HOURS={'24' if environment == 'development' else '1'}",
                "",
            ])
        
        # Add monitoring configuration
        if config.use_monitoring:
            lines.extend([
                "# Monitoring Configuration",
                f"METRICS_ENABLED={'true' if environment != 'development' else 'false'}",
                f"PROMETHEUS_PORT=8080",
                f"HEALTH_CHECK_PORT=8081",
                "",
            ])
        
        return '\n'.join(lines)
    
    def _generate_logging_config(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Generate intelligent logging configuration"""
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "simple": {
                    "format": "%(levelname)s - %(message)s"
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "formatter": "json",
                    "filename": f"logs/{config.package_name}.log"
                }
            },
            "loggers": {
                config.package_name: {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"]
            }
        }
        
        config_file = package_dir / "config/logging.yml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(logging_config, f, default_flow_style=False)
    
    def _generate_monitoring_config(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Generate intelligent monitoring configuration"""
        # This would generate Prometheus, Grafana configs based on the domain analysis
        # Implementation details depend on the specific monitoring stack
        pass
    
    def _generate_domain_specific_code(
        self, 
        package_dir: Path, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> None:
        """Generate domain-specific code templates"""
        self.logger.info("🎯 Generating domain-specific code")
        
        domain_type = domain_analysis.get('domain_type')
        
        if domain_type == 'anomaly_detection':
            self._generate_anomaly_detection_code(package_dir, config)
        elif domain_type == 'machine_learning':
            self._generate_ml_code(package_dir, config)
        elif domain_type == 'api_service':
            self._generate_api_service_code(package_dir, config)
    
    def _generate_anomaly_detection_code(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Generate anomaly detection specific code"""
        # Create base anomaly detection classes
        anomaly_detector_code = f'''"""
Anomaly Detection Domain Logic for {config.package_name}
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors"""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the anomaly detector to the data"""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> List[bool]:
        """Predict anomalies in the data"""
        pass
    
    @abstractmethod
    def score(self, data: np.ndarray) -> List[float]:
        """Return anomaly scores for the data"""
        pass


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest based anomaly detector"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def fit(self, data: np.ndarray) -> None:
        """Fit the isolation forest to the data"""
        scaled_data = self.scaler.fit_transform(data)
        self.model.fit(scaled_data)
        self._is_fitted = True
    
    def predict(self, data: np.ndarray) -> List[bool]:
        """Predict anomalies (True for anomaly, False for normal)"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        scaled_data = self.scaler.transform(data)
        predictions = self.model.predict(scaled_data)
        return [pred == -1 for pred in predictions]
    
    def score(self, data: np.ndarray) -> List[float]:
        """Return anomaly scores (lower scores indicate anomalies)"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        scaled_data = self.scaler.transform(data)
        return self.model.decision_function(scaled_data).tolist()
'''
        
        detector_file = package_dir / f"src/{config.package_name}/domain/anomaly_detector.py"
        detector_file.write_text(anomaly_detector_code)
    
    def _generate_ml_code(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Generate machine learning specific code"""
        # Implementation for ML-specific code generation
        pass
    
    def _generate_api_service_code(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Generate API service specific code"""
        # Implementation for API service code generation
        pass
    
    def _setup_intelligent_dev_environment(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Set up intelligent development environment"""
        self.logger.info("🔧 Setting up intelligent development environment")
        
        # Create pre-commit configuration
        self._create_precommit_config(package_dir, config)
        
        # Create VS Code configuration
        self._create_vscode_config(package_dir, config)
        
        # Create development scripts
        self._create_dev_scripts(package_dir, config)
    
    def _create_precommit_config(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Create pre-commit configuration"""
        precommit_config = {
            "repos": [
                {
                    "repo": "https://github.com/pre-commit/pre-commit-hooks",
                    "rev": "v4.4.0",
                    "hooks": [
                        {"id": "trailing-whitespace"},
                        {"id": "end-of-file-fixer"},
                        {"id": "check-yaml"},
                        {"id": "check-added-large-files"}
                    ]
                },
                {
                    "repo": "https://github.com/psf/black",
                    "rev": "23.3.0",
                    "hooks": [{"id": "black"}]
                },
                {
                    "repo": "https://github.com/pycqa/isort",
                    "rev": "5.12.0",
                    "hooks": [{"id": "isort"}]
                },
                {
                    "repo": "https://github.com/charliermarsh/ruff-pre-commit",
                    "rev": "v0.0.272",
                    "hooks": [{"id": "ruff"}]
                }
            ]
        }
        
        if config.use_security_scanning:
            precommit_config["repos"].append({
                "repo": "https://github.com/PyCQA/bandit",
                "rev": "1.7.5",
                "hooks": [{"id": "bandit"}]
            })
        
        config_file = package_dir / ".pre-commit-config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(precommit_config, f, default_flow_style=False)
    
    def _create_vscode_config(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Create VS Code configuration"""
        vscode_dir = package_dir / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        # Settings
        settings = {
            "python.defaultInterpreterPath": "./venv/bin/python",
            "python.linting.enabled": True,
            "python.linting.pylintEnabled": False,
            "python.linting.flake8Enabled": False,
            "python.linting.mypyEnabled": True,
            "python.formatting.provider": "black",
            "python.sortImports.path": "isort",
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": True
            }
        }
        
        settings_file = vscode_dir / "settings.json"
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
    
    def _create_dev_scripts(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Create development scripts"""
        # Create setup script
        setup_script = f'''#!/bin/bash
# {config.package_name} - Development Environment Setup

set -e

echo "Setting up development environment for {config.package_name}..."

# Create virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Created virtual environment"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  source venv/bin/activate"
echo "  make test"
echo "  make dev-up"
'''
        
        setup_file = package_dir / "scripts/setup-dev.sh"
        setup_file.write_text(setup_script)
        setup_file.chmod(0o755)
    
    def _generate_intelligent_documentation(
        self, 
        package_dir: Path, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> None:
        """Generate intelligent documentation"""
        self.logger.info("📚 Generating intelligent documentation")
        
        # Generate README
        readme_content = self._generate_intelligent_readme(config, domain_analysis)
        readme_file = package_dir / "README.md"
        readme_file.write_text(readme_content)
        
        # Generate architecture documentation
        arch_content = self._generate_architecture_docs(config, domain_analysis)
        arch_file = package_dir / "docs/ARCHITECTURE.md"
        arch_file.write_text(arch_content)
        
        # Generate API documentation if applicable
        if config.web_framework:
            api_content = self._generate_api_docs(config, domain_analysis)
            api_file = package_dir / "docs/API.md"
            api_file.write_text(api_content)
    
    def _generate_intelligent_readme(
        self, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Generate intelligent README content"""
        
        domain_type = domain_analysis.get('domain_type', 'generic')
        architecture_patterns = ', '.join(domain_analysis.get('architectural_patterns', []))
        
        return f'''# {config.package_name}

{config.package_description}

## Overview

This is a self-contained, domain-bounded package implementing the **{config.domain_name}** domain with **{domain_type}** capabilities. The package follows {architecture_patterns} architectural patterns and is designed for high maintainability, testability, and scalability.

### Key Features

- 🏗️ **Domain-Driven Design**: Properly bounded context with clear domain boundaries
- 🎯 **Self-Contained**: All dependencies and configurations included
- 🚀 **Production Ready**: Comprehensive monitoring, logging, and observability
- 🔒 **Secure**: Built-in security scanning and best practices
- 📊 **Observable**: Metrics, tracing, and health checks included
- 🧪 **Well Tested**: Unit, integration, E2E, and performance tests

### Architecture

This package implements the following architectural patterns:
- {', '.join(domain_analysis.get('architectural_patterns', ['Clean Architecture']))}

### Technology Stack

- **Language**: Python {config.python_version}
- **Web Framework**: {config.web_framework.title() if config.web_framework else 'N/A'}
- **Database**: {config.database_type.title() if config.use_database else 'N/A'}
- **Cache**: {config.cache_type.title() if config.use_cache else 'N/A'}
- **Containerization**: Docker {'+ Kubernetes' if config.use_kubernetes else ''}
- **Monitoring**: Prometheus + Grafana {'+ Jaeger' if config.use_tracing else ''}

## Quick Start

### Prerequisites

- Python {config.python_version}+
- Docker and Docker Compose
{'- Kubernetes cluster (for production deployment)' if config.use_kubernetes else ''}

### Development Setup

1. **Clone and setup the development environment:**
   ```bash
   git clone {config.repository_name}
   cd {config.package_name}
   ./scripts/setup-dev.sh
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Run tests:**
   ```bash
   make test
   ```

4. **Start development environment:**
   ```bash
   make dev-up
   ```

5. **Access the application:**
   - Application: http://localhost:8000
   - Health checks: http://localhost:8081/health
   - Metrics: http://localhost:8080/metrics
   {'- Grafana: http://localhost:3000' if config.use_monitoring else ''}

## Usage

### Running the Application

```bash
# Development
make dev-up

# Production
make deploy-prod
```

### Testing

```bash
# All tests
make test-all

# Unit tests only
make test-unit

# Integration tests
make test-integration

# E2E tests
make test-e2e

# Performance tests
make test-performance
```

### Monitoring

The application includes comprehensive monitoring:

- **Health Checks**: `/health`, `/ready`, `/live`
- **Metrics**: Prometheus metrics at `/metrics`
- **Logs**: Structured JSON logging
{'- **Tracing**: Distributed tracing with Jaeger' if config.use_tracing else ''}

## Domain Information

- **Domain**: {config.domain_name}
- **Domain Type**: {domain_type}
- **Bounded Context**: This package represents a single bounded context within the {config.domain_name} domain
{'- **Integration Points**: ' + ', '.join(domain_analysis.get('integration_requirements', [])) if domain_analysis.get('integration_requirements') else ''}

## Development

### Project Structure

```
{config.package_name}/
├── src/{config.package_name}/          # Main package code
│   ├── domain/                         # Domain logic
│   ├── application/                    # Application services
│   ├── infrastructure/                 # Infrastructure concerns
│   └── interfaces/                     # External interfaces
├── tests/                              # Test suites
├── docs/                               # Documentation
├── config/                             # Configuration files
├── k8s/                               # Kubernetes manifests
├── monitoring/                         # Monitoring configs
└── scripts/                           # Development scripts
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test-all`)
5. Run quality checks (`make quality-check`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Available Make Targets

Run `make help` to see all available targets.

## Deployment

### Local Development

```bash
make dev-up
```

### Staging

```bash
make deploy-staging
```

### Production

```bash
make deploy-prod
```

## License

This project is licensed under the {config.license} License - see the [LICENSE](LICENSE) file for details.

## Generated by Intelligent Package Generator

This package was generated by the Intelligent Package Generator on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with the following analysis:

- **Domain Type**: {domain_type}
- **Architecture Patterns**: {', '.join(domain_analysis.get('architectural_patterns', []))}
- **Infrastructure Needs**: {', '.join(k for k, v in domain_analysis.get('infrastructure_needs', {}).items() if v)}
- **Security Level**: {'High' if domain_analysis.get('security_requirements') else 'Standard'}
- **Scalability Level**: {domain_analysis.get('scalability_requirements', {}).get('expected_scale', 'Medium').title()}

---

**Author**: {config.author_name} <{config.author_email}>
**Domain**: {config.domain_name}
**Generated**: {datetime.now().isoformat()}
'''
    
    def _generate_architecture_docs(
        self, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Generate architecture documentation"""
        return f'''# Architecture Documentation

## Overview

{config.package_name} implements a {domain_analysis.get('domain_type', 'generic')} system using {', '.join(domain_analysis.get('architectural_patterns', []))} architectural patterns.

## Domain Model

### Bounded Context

This package represents the **{config.domain_name}** bounded context, which is responsible for:

- {config.package_description}
- Domain-specific business logic
- Data consistency within the domain boundary

### Domain Analysis Results

**Domain Type**: {domain_analysis.get('domain_type', 'generic')}

**Architectural Patterns**: 
{chr(10).join('- ' + pattern for pattern in domain_analysis.get('architectural_patterns', []))}

**Infrastructure Requirements**:
{chr(10).join('- ' + k + ': ' + ('Required' if v else 'Optional') for k, v in domain_analysis.get('infrastructure_needs', {}).items())}

**Integration Points**:
{chr(10).join('- ' + integration for integration in domain_analysis.get('integration_requirements', []))}

## System Architecture

### Layers

1. **Domain Layer** (`src/{config.package_name}/domain/`)
   - Core business logic
   - Domain entities and value objects
   - Domain services
   - Domain events

2. **Application Layer** (`src/{config.package_name}/application/`)
   - Application services
   - Use cases
   - Command and query handlers
   - Application events

3. **Infrastructure Layer** (`src/{config.package_name}/infrastructure/`)
   - Data access
   - External service integrations
   - Configuration
   - Cross-cutting concerns

4. **Interface Layer** (`src/{config.package_name}/interfaces/`)
   - REST API endpoints
   - Event handlers
   - Web UI (if applicable)
   - CLI commands

### Data Flow

```
External Request → Interface Layer → Application Layer → Domain Layer
                     ↓
Infrastructure Layer ← Application Layer ← Domain Layer
```

## Security Architecture

{self._generate_security_architecture_section(config, domain_analysis)}

## Scalability Design

{self._generate_scalability_section(config, domain_analysis)}

## Monitoring and Observability

{self._generate_observability_section(config, domain_analysis)}

## Decision Records

### ADR-001: Domain Boundary Selection
- **Status**: Accepted
- **Context**: Need to define clear domain boundaries
- **Decision**: Implement {config.domain_name} as a separate bounded context
- **Consequences**: Clear separation of concerns, independent deployment

### ADR-002: Architecture Pattern Selection
- **Status**: Accepted
- **Context**: Need to choose architectural pattern
- **Decision**: Implement {', '.join(domain_analysis.get('architectural_patterns', []))} pattern(s)
- **Consequences**: {self._get_architecture_consequences(domain_analysis.get('architectural_patterns', []))}

---

*Generated by Intelligent Package Generator on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
'''
    
    def _generate_security_architecture_section(
        self, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Generate security architecture section"""
        security_needs = domain_analysis.get('security_requirements', {})
        
        if not any(security_needs.values()):
            return "Standard security practices applied with basic input validation and HTTPS enforcement."
        
        sections = []
        
        if security_needs.get('authentication_required'):
            sections.append("- **Authentication**: JWT-based authentication with secure token management")
        
        if security_needs.get('authorization_required'):
            sections.append("- **Authorization**: Role-based access control (RBAC) implementation")
        
        if security_needs.get('encryption_required'):
            sections.append("- **Encryption**: Data encryption at rest and in transit")
        
        if security_needs.get('audit_required'):
            sections.append("- **Audit Logging**: Comprehensive audit trail for all operations")
        
        if security_needs.get('rate_limiting_required'):
            sections.append("- **Rate Limiting**: API rate limiting and DDoS protection")
        
        return '\n'.join(sections) if sections else "Standard security practices applied."
    
    def _generate_scalability_section(
        self, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Generate scalability section"""
        scalability = domain_analysis.get('scalability_requirements', {})
        scale_level = scalability.get('expected_scale', 'medium')
        
        if scale_level == 'high':
            return """
### High-Scale Design

- **Horizontal Scaling**: Stateless application design for easy horizontal scaling
- **Caching Strategy**: Distributed caching with Redis cluster
- **Database Design**: Read replicas and potential sharding strategy
- **Load Balancing**: Application-level load balancing
- **Resource Management**: Kubernetes-based auto-scaling
"""
        else:
            return """
### Standard Scale Design

- **Vertical Scaling**: Primary scaling approach
- **Local Caching**: In-memory caching for frequently accessed data
- **Database Optimization**: Query optimization and connection pooling
- **Resource Monitoring**: Basic resource monitoring and alerting
"""
    
    def _generate_observability_section(
        self, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Generate observability section"""
        sections = [
            "### Monitoring Stack",
            "",
            "- **Metrics**: Prometheus metrics collection",
            "- **Visualization**: Grafana dashboards",
            "- **Logging**: Structured JSON logging with log aggregation",
            "- **Health Checks**: Kubernetes-style health checks",
        ]
        
        if config.use_tracing:
            sections.append("- **Tracing**: Distributed tracing with Jaeger")
        
        sections.extend([
            "",
            "### Key Metrics",
            "",
            "- Request rate and response time",
            "- Error rates and status codes",
            "- Resource utilization (CPU, memory, disk)",
            "- Business metrics specific to the domain",
        ])
        
        return '\n'.join(sections)
    
    def _get_architecture_consequences(self, patterns: List[str]) -> str:
        """Get consequences of architecture pattern choices"""
        if 'hexagonal' in patterns:
            return "High testability, loose coupling, framework independence"
        elif 'clean' in patterns:
            return "Clear separation of concerns, testable business logic"
        elif 'microservices' in patterns:
            return "Independent deployment, technology diversity, operational complexity"
        else:
            return "Standard layered architecture benefits"
    
    def _generate_api_docs(
        self, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Generate API documentation"""
        return f'''# API Documentation

## Overview

{config.package_name} provides a RESTful API built with {config.web_framework.title()}.

## Base URL

- **Development**: http://localhost:8000
- **Staging**: https://{config.package_name}-staging.example.com
- **Production**: https://{config.package_name}.example.com

## Authentication

{self._generate_auth_docs(config, domain_analysis)}

## Health Endpoints

### Health Check
- **GET** `/health`
- **Description**: Overall application health
- **Response**: 200 OK with health status

### Readiness Check
- **GET** `/ready`
- **Description**: Application readiness for traffic
- **Response**: 200 OK when ready

### Liveness Check
- **GET** `/live`
- **Description**: Application liveness
- **Response**: 200 OK when alive

## Metrics

### Prometheus Metrics
- **GET** `/metrics`
- **Description**: Prometheus-formatted metrics
- **Content-Type**: text/plain

## API Endpoints

{self._generate_domain_specific_api_docs(config, domain_analysis)}

## Error Handling

All API endpoints return consistent error responses:

```json
{{
  "error": {{
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {{
      "field": "Additional error details"
    }},
    "timestamp": "2023-01-01T00:00:00Z",
    "request_id": "unique-request-id"
  }}
}}
```

### Common Error Codes

- `VALIDATION_ERROR`: Input validation failed
- `RESOURCE_NOT_FOUND`: Requested resource not found
- `UNAUTHORIZED`: Authentication required
- `FORBIDDEN`: Access denied
- `INTERNAL_ERROR`: Internal server error

## Rate Limiting

{self._generate_rate_limiting_docs(config, domain_analysis)}

---

*Generated by Intelligent Package Generator*
'''
    
    def _generate_auth_docs(
        self, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Generate authentication documentation"""
        security_needs = domain_analysis.get('security_requirements', {})
        
        if not security_needs.get('authentication_required'):
            return "No authentication required for this service."
        
        return '''
### JWT Authentication

All protected endpoints require a valid JWT token in the Authorization header:

```
Authorization: Bearer <jwt-token>
```

#### Getting a Token

**POST** `/auth/login`

```json
{
  "username": "user@example.com",
  "password": "password"
}
```

**Response:**
```json
{
  "access_token": "jwt-token-here",
  "token_type": "bearer",
  "expires_in": 3600
}
```
'''
    
    def _generate_domain_specific_api_docs(
        self, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Generate domain-specific API documentation"""
        domain_type = domain_analysis.get('domain_type')
        
        if domain_type == 'anomaly_detection':
            return '''
### Anomaly Detection

#### Detect Anomalies
- **POST** `/anomaly/detect`
- **Description**: Detect anomalies in the provided data
- **Request Body**:
  ```json
  {
    "data": [[1.2, 3.4, 5.6], [2.1, 4.3, 6.5]],
    "model_params": {
      "contamination": 0.1
    }
  }
  ```
- **Response**:
  ```json
  {
    "anomalies": [false, true],
    "scores": [-0.123, -0.456],
    "metadata": {
      "model_version": "1.0.0",
      "processed_at": "2023-01-01T00:00:00Z"
    }
  }
  ```

#### Train Model
- **POST** `/anomaly/train`
- **Description**: Train anomaly detection model
- **Request Body**:
  ```json
  {
    "training_data": [[1.2, 3.4], [2.1, 4.3]],
    "model_params": {
      "contamination": 0.1,
      "random_state": 42
    }
  }
  ```
'''
        elif domain_type == 'machine_learning':
            return '''
### Machine Learning

#### Make Prediction
- **POST** `/ml/predict`
- **Description**: Make predictions using trained model
- **Request Body**:
  ```json
  {
    "features": [1.2, 3.4, 5.6],
    "model_id": "model-v1.0.0"
  }
  ```

#### Train Model
- **POST** `/ml/train`
- **Description**: Train a new model
- **Request Body**:
  ```json
  {
    "training_data": "path/to/training/data",
    "model_params": {
      "algorithm": "random_forest",
      "hyperparams": {...}
    }
  }
  ```
'''
        else:
            return '''
### Domain Endpoints

Domain-specific endpoints will be documented here based on the business requirements.

Example endpoints structure:
- **GET** `/api/v1/resources`
- **POST** `/api/v1/resources`
- **GET** `/api/v1/resources/{id}`
- **PUT** `/api/v1/resources/{id}`
- **DELETE** `/api/v1/resources/{id}`
'''
    
    def _generate_rate_limiting_docs(
        self, 
        config: IntelligentPackageConfig,
        domain_analysis: Dict[str, Any]
    ) -> str:
        """Generate rate limiting documentation"""
        security_needs = domain_analysis.get('security_requirements', {})
        
        if not security_needs.get('rate_limiting_required'):
            return "No rate limiting applied to this service."
        
        return '''
The API implements rate limiting to prevent abuse:

- **Authenticated users**: 1000 requests per hour
- **Anonymous users**: 100 requests per hour
- **Burst limit**: 10 requests per second

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Remaining requests in current window  
- `X-RateLimit-Reset`: Window reset time (Unix timestamp)

When rate limit is exceeded, the API returns `429 Too Many Requests`.
'''
    
    def _initialize_intelligent_repository(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Initialize intelligent repository with domain integration"""
        self.logger.info("🔧 Initializing intelligent repository")
        
        if self._ask_yes_no("Initialize git repository with intelligent hooks?", default=True):
            try:
                # Initialize git repository
                subprocess.run(['git', 'init'], cwd=package_dir, check=True, capture_output=True)
                
                # Configure git
                subprocess.run(['git', 'config', 'user.name', config.author_name], cwd=package_dir, check=True, capture_output=True)
                subprocess.run(['git', 'config', 'user.email', config.author_email], cwd=package_dir, check=True, capture_output=True)
                
                # Add files
                subprocess.run(['git', 'add', '.'], cwd=package_dir, check=True, capture_output=True)
                
                # Initial commit
                commit_message = f'''Initial commit for {config.package_name}

Generated by Intelligent Package Generator with:
- Domain: {config.domain_name}
- Type: {self.template_vars.get('domain_type', 'generic')}
- Architecture: {', '.join(self.template_vars.get('architecture_patterns', []))}

🤖 Generated with Intelligent Package Generator
'''
                subprocess.run(['git', 'commit', '-m', commit_message], cwd=package_dir, check=True, capture_output=True)
                
                self.logger.info("   ✅ Git repository initialized with intelligent commit")
                
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"   ⚠️ Could not initialize git repository: {e}")
    
    def _print_next_steps(self, package_dir: Path, config: IntelligentPackageConfig) -> None:
        """Print intelligent next steps"""
        print(f"""
🎉 Intelligent package '{config.package_name}' generated successfully!

📍 **Location**: {package_dir.absolute()}

🏗️ **Architecture**: {', '.join(self.template_vars.get('architecture_patterns', []))}
🎯 **Domain**: {config.domain_name} ({self.template_vars.get('domain_type', 'generic')})
🔒 **Security Level**: {self.template_vars.get('security_level', 'standard')}
📈 **Scale Level**: {self.template_vars.get('scalability_level', 'medium')}

🚀 **Next Steps**:

1. **Set up development environment:**
   ```bash
   cd {package_dir}
   ./scripts/setup-dev.sh
   source venv/bin/activate
   ```

2. **Run tests:**
   ```bash
   make test-all
   ```

3. **Start development environment:**
   ```bash
   make dev-up
   ```

4. **Access your application:**
   - Application: http://localhost:8000
   - Health: http://localhost:8081/health
   - Metrics: http://localhost:8080/metrics
   {'- Grafana: http://localhost:3000 (admin/admin)' if config.use_monitoring else ''}

5. **Review generated documentation:**
   - README.md: Project overview
   - docs/ARCHITECTURE.md: Architecture decisions
   {'- docs/API.md: API documentation' if config.web_framework else ''}

6. **Customize for your needs:**
   - Update domain logic in src/{config.package_name}/domain/
   - Add business rules and entities
   - Configure environment variables in .env files
   - Customize monitoring dashboards

💡 **Tips**:
- Run `make help` to see all available commands
- Use `make quality-check` before committing
- Check logs with `make logs` when running
- Use `make monitor` to start monitoring stack

Happy coding! 🎯
        """)
    
    def _ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question"""
        default_str = "Y/n" if default else "y/N"
        while True:
            try:
                response = input(f"{question} [{default_str}]: ").strip().lower()
                if not response:
                    return default
                if response in ('y', 'yes'):
                    return True
                if response in ('n', 'no'):
                    return False
                print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                return False


def create_intelligent_config() -> IntelligentPackageConfig:
    """Create intelligent configuration interactively"""
    print("🧠 Intelligent Package Generator")
    print("=" * 50)
    print("This generator will analyze your requirements and create an optimized package.")
    print("")
    
    # Basic package information
    package_name = input("📦 Package name: ").strip()
    while not package_name or not re.match(r'^[a-z][a-z0-9\-_]*$', package_name):
        print("Package name must start with a letter and contain only lowercase letters, numbers, hyphens, and underscores")
        package_name = input("📦 Package name: ").strip()
    
    package_description = input("📝 Package description: ").strip()
    domain_name = input("🏗️ Domain name: ").strip()
    
    print(f"\n🔍 Analyzing requirements for '{package_description}'...")
    
    # Create a temporary generator to analyze requirements
    temp_generator = IntelligentPackageGenerator(Path("."), Path("."))
    analysis = temp_generator.analyze_domain_requirements(package_name, package_description)
    
    print(f"✅ Analysis complete!")
    print(f"   Domain Type: {analysis.get('domain_type')}")
    print(f"   Suggested Architecture: {', '.join(analysis.get('architectural_patterns', []))}")
    print(f"   Infrastructure Needs: {', '.join(k for k, v in analysis.get('infrastructure_needs', {}).items() if v)}")
    print(f"   Scale Level: {analysis.get('scalability_requirements', {}).get('expected_scale', 'medium').title()}")
    
    # Get author information
    print("\n👤 Author Information")
    author_name = input("Author name: ").strip()
    author_email = input("Author email: ").strip()
    github_username = input("GitHub username: ").strip()
    repository_name = input(f"Repository name [{package_name}]: ").strip() or package_name
    
    # Advanced configuration
    print("\n⚙️ Advanced Configuration")
    print("The generator has made intelligent defaults based on your requirements.")
    print("You can accept the defaults or customize further.")
    
    customize = input("Customize advanced settings? [y/N]: ").strip().lower() in ('y', 'yes')
    
    config = IntelligentPackageConfig(
        package_name=package_name,
        package_description=package_description,
        domain_name=domain_name,
        author_name=author_name,
        author_email=author_email,
        github_username=github_username,
        repository_name=repository_name,
    )
    
    if customize:
        config = temp_generator._apply_intelligent_defaults(config, analysis)
        config = _customize_intelligent_config(config, analysis)
    else:
        config = temp_generator._apply_intelligent_defaults(config, analysis)
        print("✅ Using intelligent defaults based on analysis")
    
    return config


def _customize_intelligent_config(config: IntelligentPackageConfig, analysis: Dict[str, Any]) -> IntelligentPackageConfig:
    """Customize intelligent configuration"""
    print("\n🛠️ Customizing Configuration")
    
    # Architecture customization
    suggested_patterns = analysis.get('architectural_patterns', [])
    print(f"Suggested architecture patterns: {', '.join(suggested_patterns)}")
    if input("Use different architecture pattern? [y/N]: ").strip().lower() in ('y', 'yes'):
        print("Available patterns: hexagonal, clean, layered, microservices, event_driven")
        pattern = input("Enter pattern: ").strip()
        if pattern:
            config.architecture_pattern = pattern
    
    # Infrastructure customization
    infra_needs = analysis.get('infrastructure_needs', {})
    print(f"\n🏗️ Infrastructure Configuration")
    print(f"Detected needs: {', '.join(k for k, v in infra_needs.items() if v)}")
    
    if infra_needs.get('database') and input("Use different database? [y/N]: ").strip().lower() in ('y', 'yes'):
        db_types = ['postgresql', 'mysql', 'sqlite', 'mongodb']
        print(f"Available: {', '.join(db_types)}")
        db_type = input("Database type [postgresql]: ").strip() or "postgresql"
        config.database_type = db_type
    
    # Security customization
    security_needs = analysis.get('security_requirements', {})
    if any(security_needs.values()):
        print(f"\n🔒 Security Configuration")
        print("Enhanced security features detected and enabled")
        if input("Disable security scanning? [y/N]: ").strip().lower() in ('y', 'yes'):
            config.use_security_scanning = False
    
    # ML/AI customization
    domain_type = analysis.get('domain_type')
    if domain_type in ['machine_learning', 'anomaly_detection']:
        print(f"\n🤖 ML/AI Configuration")
        if domain_type == 'machine_learning':
            if input("Include MLflow for experiment tracking? [Y/n]: ").strip().lower() not in ('n', 'no'):
                config.use_mlflow = True
            if input("Include Jupyter notebooks? [Y/n]: ").strip().lower() not in ('n', 'no'):
                config.use_jupyter = True
        
        # ML framework selection
        print("ML Framework selection:")
        if input("Include TensorFlow? [y/N]: ").strip().lower() in ('y', 'yes'):
            config.use_tensorflow = True
        if input("Include PyTorch? [y/N]: ").strip().lower() in ('y', 'yes'):
            config.use_pytorch = True
    
    return config


def main():
    """Main entry point for intelligent package generator"""
    parser = argparse.ArgumentParser(description="Intelligent Package Generator - AI-Powered Self-Contained Package Creation")
    parser.add_argument('--config', type=str, help="Configuration file (JSON)")
    parser.add_argument('--templates', type=str, default="templates/self_contained_package", 
                       help="Templates directory")
    parser.add_argument('--output', type=str, default=".", help="Output directory")
    parser.add_argument('--interactive', action='store_true', help="Interactive configuration with AI analysis")
    parser.add_argument('--analyze-only', action='store_true', help="Only analyze requirements, don't generate")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger('intelligent_package_generator').setLevel(logging.DEBUG)
    
    # Determine script location
    script_dir = Path(__file__).parent
    templates_dir = script_dir.parent / args.templates
    output_dir = Path(args.output)
    
    if not templates_dir.exists():
        print(f"❌ Templates directory not found: {templates_dir}")
        return 1
    
    # Initialize generator
    generator = IntelligentPackageGenerator(templates_dir, output_dir)
    
    # Get configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = IntelligentPackageConfig(**config_data)
    elif args.interactive:
        config = create_intelligent_config()
    else:
        print("❌ Must provide either --config or --interactive")
        return 1
    
    # Analyze only mode
    if args.analyze_only:
        analysis = generator.analyze_domain_requirements(config.package_name, config.package_description)
        print(json.dumps(analysis, indent=2))
        return 0
    
    # Generate package
    try:
        generator.generate_intelligent_package(config)
        return 0
    except Exception as e:
        generator.logger.error(f"Package generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
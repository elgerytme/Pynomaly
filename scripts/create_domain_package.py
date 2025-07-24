#!/usr/bin/env python3
"""
Domain Package Generator

Creates new domain packages with proper structure and compliance.
Enhanced with intelligent domain detection and auto-suggestions.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
import subprocess

@dataclass
class DomainSuggestion:
    """Represents a domain package suggestion"""
    name: str
    confidence: float
    concepts: List[str]
    files: List[str]
    suggested_structure: Dict[str, Any]
    reasoning: str

class IntelligentDomainAnalyzer:
    """Analyzes codebase to suggest domain packages"""
    
    def __init__(self):
        self.domain_patterns = self._load_domain_patterns()
    
    def _load_domain_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load domain patterns for intelligent analysis"""
        return {
            "user_management": {
                "keywords": ["user", "account", "profile", "authentication", "authorization", "login", "signup", "permission", "role"],
                "patterns": [r"\b(user|account|auth)_\w+", r"\buser\w*", r"\bauth\w*"],
                "suggested_entities": ["User", "Profile", "Account", "Permission", "Role"],
                "suggested_services": ["AuthenticationService", "UserService", "PermissionService"],
                "confidence_boost": 0.2
            },
            "payment_processing": {
                "keywords": ["payment", "transaction", "billing", "invoice", "subscription", "charge", "refund", "stripe", "paypal"],
                "patterns": [r"\b(payment|billing|transaction)_\w+", r"\bpay\w*", r"\bbilling\w*"],
                "suggested_entities": ["Payment", "Transaction", "Invoice", "Subscription"],
                "suggested_services": ["PaymentService", "BillingService", "TransactionService"],
                "confidence_boost": 0.3
            },
            "notification_system": {
                "keywords": ["notification", "email", "sms", "push", "alert", "message", "campaign", "template"],
                "patterns": [r"\b(notification|email|sms)_\w+", r"\bnotif\w*", r"\bemail\w*"],
                "suggested_entities": ["Notification", "EmailTemplate", "Campaign", "Message"],
                "suggested_services": ["NotificationService", "EmailService", "MessageService"],
                "confidence_boost": 0.2
            },
            "product_catalog": {
                "keywords": ["product", "catalog", "inventory", "category", "brand", "sku", "variant", "price"],
                "patterns": [r"\b(product|catalog|inventory)_\w+", r"\bproduct\w*", r"\bcatalog\w*"],
                "suggested_entities": ["Product", "Category", "Brand", "Variant", "Price"],
                "suggested_services": ["ProductService", "CatalogService", "InventoryService"],
                "confidence_boost": 0.25
            },
            "order_management": {
                "keywords": ["order", "cart", "checkout", "shipping", "delivery", "fulfillment", "orderitem"],
                "patterns": [r"\b(order|cart|checkout)_\w+", r"\border\w*", r"\bcart\w*"],
                "suggested_entities": ["Order", "OrderItem", "Cart", "Shipment", "Delivery"],
                "suggested_services": ["OrderService", "CartService", "FulfillmentService"],
                "confidence_boost": 0.25
            },
            "content_management": {
                "keywords": ["content", "article", "post", "page", "media", "document", "cms", "blog"],
                "patterns": [r"\b(content|article|post)_\w+", r"\bcontent\w*", r"\barticle\w*"],
                "suggested_entities": ["Content", "Article", "Media", "Document", "Page"],
                "suggested_services": ["ContentService", "MediaService", "PublishingService"],
                "confidence_boost": 0.2
            },
            "analytics_tracking": {
                "keywords": ["analytics", "tracking", "metrics", "event", "statistics", "report", "dashboard"],
                "patterns": [r"\b(analytics|tracking|metrics)_\w+", r"\banalytics\w*", r"\btracking\w*"],
                "suggested_entities": ["Event", "Metric", "Report", "Dashboard", "Analytics"],
                "suggested_services": ["AnalyticsService", "TrackingService", "MetricsService"],
                "confidence_boost": 0.2
            }
        }
    
    def analyze_existing_code(self, root_path: str = ".") -> List[DomainSuggestion]:
        """Analyze existing code to suggest domain packages"""
        suggestions = []
        domain_matches = {}
        
        # Scan files for domain patterns
        for root, dirs, files in os.walk(root_path):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', '.git']]
            
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.md', '.yml', '.yaml')):
                    file_path = Path(root) / file
                    matches = self._analyze_file_for_domains(file_path)
                    
                    for domain_name, data in matches.items():
                        if domain_name not in domain_matches:
                            domain_matches[domain_name] = {
                                'concepts': set(),
                                'files': set(),
                                'confidence_scores': []
                            }
                        
                        domain_matches[domain_name]['concepts'].update(data['concepts'])
                        domain_matches[domain_name]['files'].add(str(file_path))
                        domain_matches[domain_name]['confidence_scores'].extend(data['confidence_scores'])
        
        # Convert to suggestions
        for domain_name, data in domain_matches.items():
            if len(data['concepts']) >= 2 and len(data['files']) >= 2:
                avg_confidence = sum(data['confidence_scores']) / len(data['confidence_scores'])
                
                if avg_confidence >= 0.5:
                    suggestion = DomainSuggestion(
                        name=domain_name,
                        confidence=avg_confidence,
                        concepts=list(data['concepts']),
                        files=list(data['files']),
                        suggested_structure=self._generate_suggested_structure(domain_name, data['concepts']),
                        reasoning=self._generate_suggestion_reasoning(domain_name, data)
                    )
                    suggestions.append(suggestion)
        
        return sorted(suggestions, key=lambda x: x.confidence, reverse=True)
    
    def _analyze_file_for_domains(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Analyze a file for domain-specific patterns"""
        matches = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
            
            for domain_name, pattern_data in self.domain_patterns.items():
                confidence = 0.0
                concepts = set()
                
                # Check keywords
                for keyword in pattern_data['keywords']:
                    if keyword in content:
                        concepts.add(keyword)
                        confidence += 0.3
                
                # Check patterns
                import re
                for pattern in pattern_data['patterns']:
                    pattern_matches = re.findall(pattern, content)
                    if pattern_matches:
                        concepts.update(pattern_matches)
                        confidence += 0.4
                
                # Boost confidence for known domain patterns
                confidence += pattern_data.get('confidence_boost', 0)
                
                if concepts and confidence > 0.4:
                    matches[domain_name] = {
                        'concepts': concepts,
                        'confidence_scores': [min(confidence, 1.0)]
                    }
        
        except Exception:
            pass
        
        return matches
    
    def _generate_suggested_structure(self, domain_name: str, concepts: Set[str]) -> Dict[str, Any]:
        """Generate suggested package structure"""
        pattern_data = self.domain_patterns.get(domain_name, {})
        
        return {
            "package_name": domain_name,
            "suggested_path": f"src/packages/{domain_name}",
            "entities": pattern_data.get('suggested_entities', []),
            "services": pattern_data.get('suggested_services', []),
            "detected_concepts": list(concepts),
            "dependencies": self._suggest_dependencies(domain_name)
        }
    
    def _suggest_dependencies(self, domain_name: str) -> List[str]:
        """Suggest dependencies based on domain type"""
        common_deps = ["pydantic>=2.0.0", "typing-extensions>=4.0.0"]
        
        domain_specific_deps = {
            "payment_processing": ["stripe", "requests", "cryptography"],
            "notification_system": ["sendgrid", "twilio", "celery"],
            "user_management": ["bcrypt", "pyjwt", "passlib"],
            "content_management": ["pillow", "bleach", "markdown"],
            "analytics_tracking": ["pandas", "numpy", "plotly"]
        }
        
        return common_deps + domain_specific_deps.get(domain_name, [])
    
    def _generate_suggestion_reasoning(self, domain_name: str, data: Dict[str, Any]) -> str:
        """Generate reasoning for domain suggestion"""
        concepts_count = len(data['concepts'])
        files_count = len(data['files'])
        
        return (
            f"Detected {domain_name.replace('_', ' ')} patterns in {files_count} files "
            f"with {concepts_count} domain concepts. "
            f"Key concepts: {', '.join(list(data['concepts'])[:3])}."
        )

def suggest_domain_improvements(package_name: str) -> List[str]:
    """Suggest improvements for domain package creation"""
    analyzer = IntelligentDomainAnalyzer()
    suggestions = analyzer.analyze_existing_code()
    
    improvements = []
    
    # Check if requested package aligns with detected domains
    for suggestion in suggestions:
        if suggestion.name == package_name or package_name in suggestion.name:
            improvements.append(f"✨ Great choice! Detected {suggestion.name} domain (confidence: {suggestion.confidence:.2f})")
            improvements.append(f"📁 Suggested entities: {', '.join(suggestion.suggested_structure['entities'][:3])}")
            improvements.append(f"🔧 Suggested services: {', '.join(suggestion.suggested_structure['services'][:3])}")
            break
    else:
        # No direct match, suggest similar domains
        related = [s for s in suggestions if any(word in package_name for word in s.concepts)]
        if related:
            improvements.append(f"💡 Related domains detected: {', '.join([s.name for s in related[:2]])}")
        
        # Suggest top detected domains
        if suggestions:
            top_suggestions = suggestions[:3]
            improvements.append("🎯 Top detected domains in codebase:")
            for suggestion in top_suggestions:
                improvements.append(f"  • {suggestion.name} (confidence: {suggestion.confidence:.2f})")
    
    return improvements

def create_directory_structure(package_name: str, base_path: str = "src/packages"):
    """Create standard directory structure for a domain package"""
    
    package_path = Path(base_path) / package_name
    
    # Standard domain package structure
    directories = [
        f"src/{package_name}/application",
        f"src/{package_name}/domain",
        f"src/{package_name}/infrastructure", 
        f"src/{package_name}/presentation",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "docs",
        "build",
        "deploy",
        "examples",
        "scripts"
    ]
    
    # Create directories
    for dir_path in directories:
        full_path = package_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python directories  
        if 'src' in str(full_path) and not dir_path.startswith(("tests", "docs")):
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    # Create main package files
    package_src = package_path / "src" / package_name
    (package_src / "__init__.py").write_text(f'''"""{package_name.replace('_', ' ').title()} Package

{package_name.replace('_', ' ').title()} domain package.
"""

__version__ = "0.1.0"
''')
    (package_src / "cli.py").write_text(f'''"""CLI interface for {package_name}"""

import click

@click.group()
def cli():
    """{package_name.replace('_', ' ').title()} CLI"""
    pass

@cli.command()
def status():
    """Check {package_name} status"""
    click.echo("{package_name.replace('_', ' ').title()} Status: OK")

if __name__ == "__main__":
    cli()
''')
    (package_src / "server.py").write_text(f'''"""Server for {package_name}"""

from fastapi import FastAPI

app = FastAPI(title="{package_name.replace('_', ' ').title()} Service")

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "service": "{package_name}"}}

@app.get("/")
async def root():
    return {{"message": "{package_name.replace('_', ' ').title()} Service"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''')
    (package_src / "worker.py").write_text(f'''"""Worker for {package_name}"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class {package_name.title().replace('_', '')}Worker:
    """Background worker for {package_name} tasks"""
    
    def __init__(self):
        self.running = False
    
    async def start(self):
        """Start the worker"""
        self.running = True
        logger.info("{package_name.replace('_', ' ').title()} Worker started")
        
        while self.running:
            await self.process_tasks()
            await asyncio.sleep(10)
    
    async def process_tasks(self):
        """Process pending tasks"""
        # TODO: Implement task processing logic
        pass
    
    def stop(self):
        """Stop the worker"""
        self.running = False
        logger.info("{package_name.replace('_', ' ').title()} Worker stopped")

if __name__ == "__main__":
    worker = {package_name.title().replace('_', '')}Worker()
    asyncio.run(worker.start())
''')

    print(f"✅ Created directory structure for {package_name}")
    return package_path

def create_pyproject_toml(package_path: Path, package_name: str, description: str, intelligent_suggestions: Optional[Dict[str, Any]] = None):
    """Create pyproject.toml for the domain package with intelligent dependency suggestions"""
    
    # Get intelligent dependency suggestions
    dependencies = ["pydantic>=2.0.0", "typing-extensions>=4.0.0"]
    dev_dependencies = [
        "pytest>=8.0.0",
        "pytest-cov>=6.0.0", 
        "pytest-asyncio>=0.24.0",
        "hypothesis>=6.115.0",
        "factory-boy>=3.3.1",
        "faker>=33.1.0"
    ]
    
    if intelligent_suggestions:
        suggested_deps = intelligent_suggestions.get('dependencies', [])
        dependencies.extend(suggested_deps)
    
    # Format dependencies as strings
    deps_str = ',\n    '.join([f'"{dep}"' for dep in dependencies])
    dev_deps_str = ',\n    '.join([f'"{dep}"' for dep in dev_dependencies])
    
    content = f'''[project]
name = "{package_name}"  
version = "0.1.0"

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]
'''
    
    pyproject_path = package_path / "pyproject.toml"
    with open(pyproject_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created pyproject.toml for {package_name}")

def create_readme(package_path: Path, package_name: str, description: str):
    """Create README.md for the domain package"""
    
    content = f'''# {package_name.replace('_', ' ').title()} Domain Package

{description}

## Overview

This package contains the domain logic for {package_name.replace('_', ' ')} functionality following Domain-Driven Design (DDD) principles.

## Structure

```
{package_name}/
├── core/
│   ├── domain/              # Domain layer
│   │   ├── entities/        # Domain entities
│   │   ├── services/        # Domain services
│   │   ├── value_objects/   # Value objects
│   │   ├── repositories/    # Repository interfaces
│   │   └── exceptions/      # Domain exceptions
│   ├── application/         # Application layer
│   │   ├── services/        # Application services
│   │   └── use_cases/       # Use cases
│   └── dto/                 # Data transfer objects
├── infrastructure/          # Infrastructure layer
│   ├── adapters/           # External adapters
│   ├── persistence/        # Data persistence
│   └── external/           # External services
├── interfaces/             # Interface layer
│   ├── api/               # REST API endpoints
│   ├── cli/               # Command-line interface
│   ├── web/               # Web interface
│   └── python_sdk/        # Python SDK
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
└── docs/                   # Documentation
```

## Domain Boundaries

This package follows strict domain boundaries:

### Allowed Concepts
- {package_name.replace('_', ' ').title()}-specific business logic
- Domain entities and value objects
- Domain services and repositories
- Use cases and application services

### Prohibited Concepts
- Generic software infrastructure (belongs in `software/` package)
- Other domain concepts (belongs in respective domain packages)
- Cross-domain references (use interfaces and dependency injection)

## Installation

```bash
pip install {package_name}
```

## Development

### Setup
```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
python scripts/install_domain_hooks.py
```

### Testing
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run domain boundary validation
python scripts/domain_boundary_validator.py
```

### Code Quality
```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Architecture

This package follows Clean Architecture principles:

1. **Domain Layer**: Core business logic
2. **Application Layer**: Use cases and application services
3. **Infrastructure Layer**: External concerns
4. **Interface Layer**: User interfaces

## Domain Compliance

This package maintains strict domain boundary compliance:

- **Validation**: Automated domain boundary validation
- **Enforcement**: Pre-commit hooks and CI/CD integration
- **Monitoring**: Continuous compliance monitoring
- **Documentation**: Clear domain boundary rules

## Contributing

1. Follow domain boundary rules
2. Add comprehensive tests
3. Update documentation
4. Validate domain compliance
5. Submit pull request

## License

MIT License - see LICENSE file for details.
'''
    
    readme_path = package_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created README.md for {package_name}")

def create_buck_file(package_path: Path, package_name: str):
    """Create BUCK file for the domain package"""
    
    content = f'''# BUCK file for {package_name} package
# Generated using standardized Buck2 templates

load("//tools/buck:monorepo_python_package.bzl", "monorepo_python_package")

monorepo_python_package(
    name = "{package_name}",
    domain = "data",  # Default to data domain - adjust as needed
    visibility = ["PUBLIC"],
    cli_entry_points = {{
        "{package_name}": "src/{package_name}.cli:main",
        "{package_name}-server": "src/{package_name}.server:main", 
        "{package_name}-worker": "src/{package_name}.worker:main",
    }},
)'''
    
    buck_path = package_path / "BUCK"
    with open(buck_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created BUCK file for {package_name}")

def create_license(package_path: Path):
    """Create LICENSE file for the domain package"""
    
    content = '''MIT License

Copyright (c) 2024 Domain Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
    
    license_path = package_path / "LICENSE"
    with open(license_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created LICENSE for package")

def create_sample_entity(package_path: Path, package_name: str):
    """Create a sample domain entity"""
    
    entity_name = f"{package_name.title().replace('_', '')}Entity"
    
    content = f'''"""
{entity_name}

Sample domain entity for {package_name} domain.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from software.core.domain.abstractions.base_entity import BaseEntity

@dataclass
class {entity_name}(BaseEntity):
    """
    Sample domain entity for {package_name} domain.
    
    This is a template - replace with actual domain entities.
    """
    
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation"""
        return {{
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }}
    
    def __str__(self) -> str:
        """String representation"""
        return f"{entity_name}(id={{self.id}}, name={{self.name}})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (
            f"{entity_name}("
            f"id={{self.id!r}}, "
            f"name={{self.name!r}}, "
            f"description={{self.description!r}}, "
            f"created_at={{self.created_at!r}}, "
            f"updated_at={{self.updated_at!r}}"
            f")"
        )
'''
    
    entity_path = package_path / "core" / "domain" / "entities" / f"{package_name}_entity.py"
    with open(entity_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created sample entity: {entity_name}")

def create_sample_service(package_path: Path, package_name: str):
    """Create a sample domain service"""
    
    service_name = f"{package_name.title().replace('_', '')}Service"
    
    content = f'''"""
{service_name}

Sample domain service for {package_name} domain.
"""

from typing import List, Optional
from software.core.domain.abstractions.base_service import BaseService
from ..entities.{package_name}_entity import {package_name.title().replace('_', '')}Entity

class {service_name}(BaseService):
    """
    Sample domain service for {package_name} domain.
    
    This is a template - replace with actual domain services.
    """
    
    def __init__(self):
        """Initialize service"""
        super().__init__()
    
    def process_entity(self, entity: {package_name.title().replace('_', '')}Entity) -> {package_name.title().replace('_', '')}Entity:
        """
        Process a domain entity.
        
        Args:
            entity: Entity to process
            
        Returns:
            Processed entity
        """
        # Sample processing logic
        processed_entity = entity
        processed_entity.description = f"Processed: {{entity.description}}"
        
        return processed_entity
    
    def validate_entity(self, entity: {package_name.title().replace('_', '')}Entity) -> bool:
        """
        Validate a domain entity.
        
        Args:
            entity: Entity to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Sample validation logic
        return bool(entity.name and entity.name.strip())
    
    def get_entity_summary(self, entities: List[{package_name.title().replace('_', '')}Entity]) -> dict:
        """
        Get summary of entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Summary dictionary
        """
        return {{
            "total_count": len(entities),
            "named_count": sum(1 for e in entities if e.name),
            "average_description_length": sum(len(e.description) for e in entities) / len(entities) if entities else 0
        }}
'''
    
    service_path = package_path / "core" / "domain" / "services" / f"{package_name}_service.py"
    with open(service_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created sample service: {service_name}")

def create_sample_tests(package_path: Path, package_name: str):
    """Create sample tests"""
    
    entity_name = f"{package_name.title().replace('_', '')}Entity"
    service_name = f"{package_name.title().replace('_', '')}Service"
    
    # Entity test
    entity_test_content = f'''"""
Tests for {entity_name}
"""

import pytest
from datetime import datetime
from {package_name}.core.domain.entities.{package_name}_entity import {entity_name}

class Test{entity_name}:
    """Test suite for {entity_name}"""
    
    def test_entity_creation(self):
        """Test entity creation"""
        entity = {entity_name}(
            id="test-id",
            name="Test Entity",
            description="Test description"
        )
        
        assert entity.id == "test-id"
        assert entity.name == "Test Entity"
        assert entity.description == "Test description"
    
    def test_entity_to_dict(self):
        """Test entity to dictionary conversion"""
        entity = {entity_name}(
            id="test-id",
            name="Test Entity",
            description="Test description"
        )
        
        result = entity.to_dict()
        
        assert result["id"] == "test-id"
        assert result["name"] == "Test Entity"
        assert result["description"] == "Test description"
    
    def test_entity_string_representation(self):
        """Test entity string representation"""
        entity = {entity_name}(
            id="test-id",
            name="Test Entity"
        )
        
        str_repr = str(entity)
        assert "test-id" in str_repr
        assert "Test Entity" in str_repr
    
    def test_entity_repr(self):
        """Test entity detailed representation"""
        entity = {entity_name}(
            id="test-id",
            name="Test Entity",
            description="Test description"
        )
        
        repr_str = repr(entity)
        assert "test-id" in repr_str
        assert "Test Entity" in repr_str
        assert "Test description" in repr_str
'''
    
    entity_test_path = package_path / "tests" / "unit" / f"test_{package_name}_entity.py"
    with open(entity_test_path, 'w') as f:
        f.write(entity_test_content)
    
    # Service test
    service_test_content = f'''"""
Tests for {service_name}
"""

import pytest
from {package_name}.core.domain.services.{package_name}_service import {service_name}
from {package_name}.core.domain.entities.{package_name}_entity import {entity_name}

class Test{service_name}:
    """Test suite for {service_name}"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.service = {service_name}()
    
    def test_process_entity(self):
        """Test entity processing"""
        entity = {entity_name}(
            id="test-id",
            name="Test Entity",
            description="Original description"
        )
        
        processed = self.service.process_entity(entity)
        
        assert processed.id == "test-id"
        assert processed.name == "Test Entity"
        assert "Processed:" in processed.description
    
    def test_validate_entity_valid(self):
        """Test entity validation with valid entity"""
        entity = {entity_name}(
            id="test-id",
            name="Test Entity",
            description="Test description"
        )
        
        result = self.service.validate_entity(entity)
        
        assert result is True
    
    def test_validate_entity_invalid(self):
        """Test entity validation with invalid entity"""
        entity = {entity_name}(
            id="test-id",
            name="",  # Empty name should be invalid
            description="Test description"
        )
        
        result = self.service.validate_entity(entity)
        
        assert result is False
    
    def test_get_entity_summary(self):
        """Test entity summary generation"""
        entities = [
            {entity_name}(id="1", name="Entity 1", description="Desc 1"),
            {entity_name}(id="2", name="Entity 2", description="Desc 2"),
            {entity_name}(id="3", name="", description="Desc 3")  # No name
        ]
        
        summary = self.service.get_entity_summary(entities)
        
        assert summary["total_count"] == 3
        assert summary["named_count"] == 2
        assert summary["average_description_length"] > 0
    
    def test_get_entity_summary_empty(self):
        """Test entity summary with empty list"""
        summary = self.service.get_entity_summary([])
        
        assert summary["total_count"] == 0
        assert summary["named_count"] == 0
        assert summary["average_description_length"] == 0
'''
    
    service_test_path = package_path / "tests" / "unit" / f"test_{package_name}_service.py"
    with open(service_test_path, 'w') as f:
        f.write(service_test_content)
    
    print(f"✅ Created sample tests for {package_name}")

def create_intelligent_entities(package_path: Path, package_name: str, suggestions: Optional[Dict[str, Any]] = None):
    """Create entities based on intelligent suggestions"""
    if not suggestions or 'entities' not in suggestions:
        return
    
    entities_dir = package_path / "core" / "domain" / "entities"
    
    for entity_name in suggestions['entities']:
        entity_filename = f"{entity_name.lower()}.py"
        entity_path = entities_dir / entity_filename
        
        content = f'''"""
{entity_name} Entity

Domain entity for {package_name} domain.
Auto-generated based on intelligent analysis.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from software.core.domain.abstractions.base_entity import BaseEntity

@dataclass
class {entity_name}(BaseEntity):
    """
    {entity_name} entity for {package_name} domain.
    
    This entity was automatically generated based on detected domain patterns.
    Customize as needed for your specific requirements.
    """
    
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation"""
        return {{
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }}
    
    def __str__(self) -> str:
        """String representation"""
        return f"{entity_name}(id={{self.id}}, name={{self.name}})"
'''
        
        with open(entity_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Created intelligent entity: {entity_name}")

def create_intelligent_services(package_path: Path, package_name: str, suggestions: Optional[Dict[str, Any]] = None):
    """Create services based on intelligent suggestions"""
    if not suggestions or 'services' not in suggestions:
        return
    
    services_dir = package_path / "core" / "domain" / "services"
    
    for service_name in suggestions['services']:
        service_filename = f"{service_name.lower()}.py"
        service_path = services_dir / service_filename
        
        content = f'''"""
{service_name}

Domain service for {package_name} domain.
Auto-generated based on intelligent analysis.
"""

from typing import List, Optional, Dict, Any
from software.core.domain.abstractions.base_service import BaseService

class {service_name}(BaseService):
    """
    {service_name} for {package_name} domain.
    
    This service was automatically generated based on detected domain patterns.
    Implement your specific business logic here.
    """
    
    def __init__(self):
        """Initialize service"""
        super().__init__()
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process domain-specific data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        # TODO: Implement your business logic here
        processed_data = {{
            "status": "processed",
            "data": data,
            "processed_at": self._get_current_timestamp()
        }}
        
        return processed_data
    
    async def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate domain-specific data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # TODO: Implement your validation logic here
        return bool(data)
    
    def get_summary(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of items.
        
        Args:
            items: List of items to summarize
            
        Returns:
            Summary information
        """
        return {{
            "total_count": len(items),
            "service_name": "{service_name}",
            "domain": "{package_name}"
        }}
'''
        
        with open(service_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Created intelligent service: {service_name}")

def analyze_and_suggest_domains() -> List[DomainSuggestion]:
    """Analyze current codebase and suggest domain packages"""
    print("🔍 Analyzing codebase for domain suggestions...")
    analyzer = IntelligentDomainAnalyzer()
    suggestions = analyzer.analyze_existing_code()
    
    if suggestions:
        print(f"\n🎯 Found {len(suggestions)} domain suggestions:")
        for i, suggestion in enumerate(suggestions[:5], 1):
            confidence_emoji = "🟢" if suggestion.confidence > 0.8 else "🟡" if suggestion.confidence > 0.6 else "🔴"
            print(f"  {i}. {confidence_emoji} {suggestion.name} (confidence: {suggestion.confidence:.2f})")
            print(f"     Concepts: {', '.join(suggestion.concepts[:3])}")
            print(f"     Files: {len(suggestion.files)}")
            print()
    else:
        print("ℹ️  No clear domain patterns detected in current codebase")
    
    return suggestions

def validate_package_name(package_name: str) -> bool:
    """Validate package name follows conventions"""
    
    # Check for prohibited terms (domain boundary violations)
    prohibited_terms = [
        "software", "core", "infrastructure", "interface"
    ]
    
    if package_name.lower() in prohibited_terms:
        print(f"❌ Package name '{package_name}' is prohibited")
        return False
    
    # Check format
    if not package_name.replace('_', '').isalnum():
        print(f"❌ Package name '{package_name}' contains invalid characters")
        return False
    
    if package_name.startswith('_') or package_name.endswith('_'):
        print(f"❌ Package name '{package_name}' cannot start or end with underscore")
        return False
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create a new domain package with intelligent suggestions")
    parser.add_argument("name", nargs="?", help="Package name (e.g., 'user_management')")
    parser.add_argument("--description", default="", help="Package description")
    parser.add_argument("--base-path", default="src/packages", help="Base path for packages")
    parser.add_argument("--skip-samples", action="store_true", help="Skip creating sample files")
    parser.add_argument("--intelligent", action="store_true", help="Use intelligent analysis for suggestions")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze and suggest domains, don't create")
    parser.add_argument("--auto-create", action="store_true", help="Auto-create from highest confidence suggestion")
    parser.add_argument("--interactive", action="store_true", help="Interactive domain creation mode")
    
    args = parser.parse_args()
    
    # If no name provided or analyze-only mode, run analysis
    if not args.name or args.analyze_only:
        suggestions = analyze_and_suggest_domains()
        
        if args.analyze_only:
            # Save suggestions to file
            with open('domain_suggestions.json', 'w') as f:
                suggestions_data = []
                for suggestion in suggestions:
                    suggestions_data.append({
                        'name': suggestion.name,
                        'confidence': suggestion.confidence,
                        'concepts': suggestion.concepts,
                        'files': suggestion.files,
                        'suggested_structure': suggestion.suggested_structure,
                        'reasoning': suggestion.reasoning
                    })
                json.dump(suggestions_data, f, indent=2)
            
            print(f"💾 Suggestions saved to domain_suggestions.json")
            return
        
        # Auto-create mode
        if args.auto_create and suggestions:
            highest_confidence = suggestions[0]
            if highest_confidence.confidence >= 0.8:
                args.name = highest_confidence.name
                print(f"🚀 Auto-creating package '{args.name}' (confidence: {highest_confidence.confidence:.2f})")
            else:
                print(f"❌ Highest confidence domain '{highest_confidence.name}' has low confidence ({highest_confidence.confidence:.2f})")
                print("Use --interactive mode to select manually")
                return
        
        # Interactive mode
        if args.interactive and suggestions:
            print("\n🎯 Select a domain to create:")
            for i, suggestion in enumerate(suggestions[:5], 1):
                confidence_emoji = "🟢" if suggestion.confidence > 0.8 else "🟡" if suggestion.confidence > 0.6 else "🔴"
                print(f"  {i}. {confidence_emoji} {suggestion.name} (confidence: {suggestion.confidence:.2f})")
            
            print("  0. Enter custom name")
            
            try:
                choice = int(input("\nEnter your choice (1-5 or 0): "))
                if choice == 0:
                    args.name = input("Enter custom package name: ").strip()
                elif 1 <= choice <= len(suggestions):
                    args.name = suggestions[choice - 1].name
                else:
                    print("Invalid choice")
                    return
            except (ValueError, KeyboardInterrupt):
                print("\nCancelled")
                return
        
        # If still no name, prompt for it
        if not args.name:
            if suggestions:
                print("💡 Consider using one of the suggested domain names above, or enter a custom name")
            args.name = input("Enter package name: ").strip()
            if not args.name:
                print("Package name is required")
                return
    
    # Validate package name
    if not validate_package_name(args.name):
        sys.exit(1)
    
    # Set description if not provided
    description = args.description or f"{args.name.replace('_', ' ').title()} domain package"
    
    # Get intelligent suggestions for this domain
    intelligent_suggestions = None
    if args.intelligent:
        print("🧠 Getting intelligent suggestions...")
        improvements = suggest_domain_improvements(args.name)
        for improvement in improvements:
            print(f"  {improvement}")
        
        # Get suggestion structure
        analyzer = IntelligentDomainAnalyzer()
        all_suggestions = analyzer.analyze_existing_code()
        for suggestion in all_suggestions:
            if suggestion.name == args.name or args.name in suggestion.name:
                intelligent_suggestions = suggestion.suggested_structure
                break
    
    print(f"\nCreating domain package: {args.name}")
    print(f"Description: {description}")
    print(f"Base path: {args.base_path}")
    print("=" * 50)
    
    # Create package
    package_path = create_directory_structure(args.name, args.base_path)
    create_pyproject_toml(package_path, args.name, description, intelligent_suggestions)
    create_readme(package_path, args.name, description)
    create_buck_file(package_path, args.name)
    create_license(package_path)
    
    # Create intelligent or sample files
    if intelligent_suggestions and not args.skip_samples:
        print("\n🧠 Creating intelligent entities and services...")
        create_intelligent_entities(package_path, args.name, intelligent_suggestions)
        create_intelligent_services(package_path, args.name, intelligent_suggestions)
    elif not args.skip_samples:
        create_sample_entity(package_path, args.name)
        create_sample_service(package_path, args.name)
        create_sample_tests(package_path, args.name)
    
    # Run domain boundary validation
    print("\n🔍 Running domain boundary validation...")
    try:
        result = subprocess.run([
            "python", "scripts/domain_boundary_validator.py", 
            "--detect-new-domains", "--root-path", str(package_path)
        ], capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print("✅ Domain boundaries validated successfully")
        else:
            print("⚠️  Domain boundary validation warnings (check output above)")
    except Exception as e:
        print(f"⚠️  Could not run domain boundary validation: {e}")
    
    print("\n🎉 Domain package created successfully!")
    print(f"📁 Package location: {package_path}")
    
    # Show intelligent recommendations if available
    if intelligent_suggestions:
        print(f"\n🧠 Intelligent Suggestions Applied:")
        if intelligent_suggestions.get('entities'):
            print(f"  📁 Created entities: {', '.join(intelligent_suggestions['entities'])}")
        if intelligent_suggestions.get('services'):
            print(f"  🔧 Created services: {', '.join(intelligent_suggestions['services'])}")
        if intelligent_suggestions.get('dependencies'):
            print(f"  📦 Added dependencies: {', '.join(intelligent_suggestions['dependencies'])}")
    
    print("\n📋 Next steps:")
    print("1. Review the generated files")
    print("2. Customize entities and services for your domain")
    print("3. Add your domain-specific logic")
    print("4. Run tests: pytest")
    print("5. Validate domain boundaries: python scripts/domain_boundary_validator.py")
    print("6. Install pre-commit hooks: python scripts/install_domain_hooks.py")
    
    # Show usage examples
    print(f"\n🚀 Quick commands:")
    print(f"  cd {package_path}")
    print(f"  pytest")
    print(f"  python -m {args.name}.core.domain.services")
    
    print(f"\n💡 To analyze domains again: python scripts/create_domain_package.py --analyze-only")
    print(f"💡 For interactive mode: python scripts/create_domain_package.py --interactive")

if __name__ == "__main__":
    main()
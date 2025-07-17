#!/usr/bin/env python3
"""
Domain Package Generator

Creates new domain packages with proper structure and compliance.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

def create_directory_structure(package_name: str, base_path: str = "src/packages"):
    """Create standard directory structure for a domain package"""
    
    package_path = Path(base_path) / package_name
    
    # Standard domain package structure
    directories = [
        "core/domain/entities",
        "core/domain/services", 
        "core/domain/value_objects",
        "core/domain/repositories",
        "core/domain/exceptions",
        "core/application/services",
        "core/application/use_cases",
        "core/dto",
        "infrastructure/adapters",
        "infrastructure/persistence",
        "infrastructure/external",
        "interfaces/api/endpoints",
        "interfaces/cli/commands",
        "interfaces/web/handlers",
        "interfaces/python_sdk/examples",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "docs"
    ]
    
    # Create directories
    for dir_path in directories:
        full_path = package_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python directories
        if not dir_path.startswith(("tests", "docs")):
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    print(f"✅ Created directory structure for {package_name}")
    return package_path

def create_pyproject_toml(package_path: Path, package_name: str, description: str):
    """Create pyproject.toml for the domain package"""
    
    content = f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{package_name}"
version = "0.1.0"
description = "{description}"
authors = [{{name = "Domain Team", email = "team@{package_name.replace('_', '-')}.io"}}]
license = {{text = "MIT"}}
readme = "README.md"
requires-python = ">=3.11"
keywords = ["{package_name}", "domain", "architecture"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Typing :: Typed",
]

dependencies = [
    "pydantic>=2.0.0",
    "typing-extensions>=4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "hypothesis>=6.115.0",
    "factory-boy>=3.3.1",
    "faker>=33.1.0",
]

test = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "hypothesis>=6.115.0",
    "factory-boy>=3.3.1",
    "faker>=33.1.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["B008"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
minversion = "8.0"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--tb=short",
    "--disable-warnings",
    "--color=yes",
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=80",
]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*", "*Tests"]
python_functions = ["test_*"]

[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/site-packages/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
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

def validate_package_name(package_name: str) -> bool:
    """Validate package name follows conventions"""
    
    # Check for prohibited terms (domain boundary violations)
    prohibited_terms = [
        "pynomaly", "software", "core", "infrastructure", "interface"
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
    parser = argparse.ArgumentParser(description="Create a new domain package")
    parser.add_argument("name", help="Package name (e.g., 'anomaly_detection')")
    parser.add_argument("--description", default="", help="Package description")
    parser.add_argument("--base-path", default="src/packages", help="Base path for packages")
    parser.add_argument("--skip-samples", action="store_true", help="Skip creating sample files")
    
    args = parser.parse_args()
    
    # Validate package name
    if not validate_package_name(args.name):
        sys.exit(1)
    
    # Set description if not provided
    description = args.description or f"{args.name.replace('_', ' ').title()} domain package"
    
    print(f"Creating domain package: {args.name}")
    print(f"Description: {description}")
    print(f"Base path: {args.base_path}")
    print("=" * 50)
    
    # Create package
    package_path = create_directory_structure(args.name, args.base_path)
    create_pyproject_toml(package_path, args.name, description)
    create_readme(package_path, args.name, description)
    create_license(package_path)
    
    if not args.skip_samples:
        create_sample_entity(package_path, args.name)
        create_sample_service(package_path, args.name)
        create_sample_tests(package_path, args.name)
    
    print("\n🎉 Domain package created successfully!")
    print(f"📁 Package location: {package_path}")
    print("\nNext steps:")
    print("1. Review the generated files")
    print("2. Customize entities and services for your domain")
    print("3. Add your domain-specific logic")
    print("4. Run tests: pytest")
    print("5. Validate domain boundaries: python scripts/domain_boundary_validator.py")
    print("6. Install pre-commit hooks: python scripts/install_domain_hooks.py")

if __name__ == "__main__":
    main()
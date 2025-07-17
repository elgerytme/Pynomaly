# Architecture

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Clean Architecture](https://img.shields.io/badge/architecture-clean-green.svg)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
[![Domain Driven Design](https://img.shields.io/badge/design-DDD-orange.svg)](https://www.domainlanguage.com/ddd/)

## Overview

Architectural patterns, design principles, and structural blueprints for the Pynomaly platform.

**Architecture Layer**: Meta-Architecture Layer  
**Package Type**: Architectural Patterns & Design Principles  
**Status**: Production Ready

## Purpose

This package provides the architectural foundation and design patterns that guide the structure and organization of the entire Pynomaly platform. It includes architectural decision records, design patterns, and structural blueprints that ensure consistency, maintainability, and scalability across all components.

### Key Features

- **Clean Architecture Patterns**: Implementation of hexagonal architecture principles
- **Domain-Driven Design**: Domain modeling and bounded context patterns
- **Microservices Architecture**: Service decomposition and communication patterns
- **Event-Driven Architecture**: Event sourcing and CQRS patterns
- **Layered Architecture**: Clear separation of concerns and dependency management
- **Modular Architecture**: Package organization and dependency injection patterns

### Use Cases

- Implementing new components following architectural patterns
- Understanding the overall system design and structure
- Making architectural decisions with documented patterns
- Maintaining consistency across development teams
- Scaling the system with proven architectural approaches

## Architecture Patterns

### Clean Architecture Implementation

```
src/packages/software/architecture/
├── architecture/               # Main package source
│   ├── patterns/              # Architectural patterns
│   │   ├── clean/            # Clean architecture patterns
│   │   ├── hexagonal/        # Hexagonal architecture (ports & adapters)
│   │   ├── layered/          # Layered architecture patterns
│   │   ├── microservices/    # Microservices patterns
│   │   └── event_driven/     # Event-driven patterns
│   ├── principles/           # Design principles
│   │   ├── solid/            # SOLID principles implementation
│   │   ├── dry/              # Don't Repeat Yourself patterns
│   │   ├── kiss/             # Keep It Simple, Stupid patterns
│   │   └── yagni/            # You Aren't Gonna Need It patterns
│   ├── blueprints/           # Architectural blueprints
│   │   ├── domain/           # Domain architecture blueprints
│   │   ├── application/      # Application layer blueprints
│   │   ├── infrastructure/   # Infrastructure layer blueprints
│   │   └── presentation/     # Presentation layer blueprints
│   ├── decisions/            # Architecture Decision Records (ADRs)
│   │   ├── adr-001-architecture-style.md
│   │   ├── adr-002-database-strategy.md
│   │   ├── adr-003-messaging-patterns.md
│   │   └── adr-004-deployment-architecture.md
│   └── validators/           # Architecture validation tools
│       ├── dependency_validator.py
│       ├── layer_validator.py
│       └── pattern_validator.py
├── tests/                    # Package tests
├── docs/                     # Architecture documentation
└── examples/                 # Architecture examples
```

### Dependencies

- **Internal Dependencies**: None (meta-architecture layer)
- **External Dependencies**: structlog, pydantic, dependency-injector
- **Optional Dependencies**: PlantUML (for diagrams), GraphViz (for visualization)

### Design Principles

1. **Separation of Concerns**: Clear boundaries between different responsibilities
2. **Dependency Inversion**: High-level modules should not depend on low-level modules
3. **Single Responsibility**: Each component should have one reason to change
4. **Open/Closed Principle**: Open for extension, closed for modification
5. **Interface Segregation**: Many specific interfaces are better than one general-purpose interface
6. **Liskov Substitution**: Subtypes must be substitutable for their base types

## Installation

### Prerequisites

- Python 3.11 or higher
- Understanding of architectural patterns and design principles
- Familiarity with clean architecture concepts

### Package Installation

```bash
# Install from source (development)
cd src/packages/software/architecture
pip install -e .

# Install with visualization tools
pip install pynomaly-architecture[visualization]

# Install with validation tools
pip install pynomaly-architecture[validation]
```

### Pynomaly Installation

```bash
# Install entire Pynomaly platform with this package
cd /path/to/pynomaly
pip install -e ".[architecture]"
```

## Usage

### Quick Start

```python
from pynomaly.architecture.patterns.clean import CleanArchitecturePattern
from pynomaly.architecture.validators import DependencyValidator
from pynomaly.architecture.blueprints import DomainBlueprint

# Apply clean architecture pattern
pattern = CleanArchitecturePattern()
domain_structure = pattern.create_domain_structure("anomaly_detection")

# Validate architectural compliance
validator = DependencyValidator()
violations = validator.validate_package_dependencies("src/packages/")

# Create domain blueprint
blueprint = DomainBlueprint("new_domain")
blueprint.generate_structure()
```

### Basic Examples

#### Example 1: Clean Architecture Pattern

```python
from pynomaly.architecture.patterns.clean import CleanArchitectureBuilder
from pynomaly.architecture.patterns.hexagonal import HexagonalArchitecture

# Build clean architecture structure
builder = CleanArchitectureBuilder()
architecture = builder \
    .with_domain_layer("anomaly_detection") \
    .with_application_layer("use_cases") \
    .with_infrastructure_layer("adapters") \
    .with_presentation_layer("controllers") \
    .build()

# Apply hexagonal architecture
hexagonal = HexagonalArchitecture()
ports = hexagonal.define_ports([
    "DetectionPort",
    "NotificationPort", 
    "PersistencePort"
])
adapters = hexagonal.define_adapters([
    "DatabaseAdapter",
    "EmailAdapter",
    "FileAdapter"
])
```

#### Example 2: Domain-Driven Design

```python
from pynomaly.architecture.patterns.ddd import DomainDrivenDesign
from pynomaly.architecture.blueprints import BoundedContextBlueprint

# Create bounded context
ddd = DomainDrivenDesign()
bounded_context = ddd.create_bounded_context(
    name="anomaly_detection",
    entities=["Anomaly", "Dataset", "Detector"],
    value_objects=["AnomalyScore", "Threshold"],
    services=["DetectionService", "ScoringService"]
)

# Generate bounded context blueprint
blueprint = BoundedContextBlueprint(bounded_context)
blueprint.generate_structure()
```

### Advanced Usage

#### Architecture Decision Records (ADRs)

```python
from pynomaly.architecture.decisions import ADRManager
from pynomaly.architecture.decisions.templates import ADRTemplate

# Create architecture decision record
adr_manager = ADRManager()
adr = adr_manager.create_adr(
    title="Database Selection for Data Storage",
    status="Proposed",
    context="Need to choose database for storing application data",
    decision="Use PostgreSQL with time-series extensions",
    consequences=[
        "Pros: ACID compliance, mature ecosystem, time-series support",
        "Cons: Requires PostgreSQL expertise, potential scalability limits"
    ]
)

# Generate ADR document
adr_template = ADRTemplate()
adr_document = adr_template.generate(adr)
```

#### Architecture Validation

```python
from pynomaly.architecture.validators import (
    LayerValidator,
    DependencyValidator,
    PatternValidator
)

# Validate layer boundaries
layer_validator = LayerValidator()
violations = layer_validator.validate_layers([
    "domain",
    "application", 
    "infrastructure",
    "presentation"
])

# Validate dependency directions
dependency_validator = DependencyValidator()
dependency_issues = dependency_validator.validate_dependencies(
    "src/packages/",
    allowed_patterns=["domain <- application <- infrastructure"]
)

# Validate architectural patterns
pattern_validator = PatternValidator()
pattern_compliance = pattern_validator.validate_clean_architecture(
    "src/packages/software/core"
)
```

## API Reference

### Core Classes

#### Architecture Patterns

- **`CleanArchitecturePattern`**: Implementation of clean architecture principles
- **`HexagonalArchitecture`**: Ports and adapters pattern implementation
- **`LayeredArchitecture`**: Traditional layered architecture patterns
- **`MicroservicesArchitecture`**: Microservices decomposition patterns
- **`EventDrivenArchitecture`**: Event sourcing and CQRS patterns

#### Design Principles

- **`SOLIDPrinciples`**: SOLID principles implementation and validation
- **`DRYPrinciple`**: Don't Repeat Yourself pattern enforcement
- **`KISSPrinciple`**: Keep It Simple, Stupid pattern guidance
- **`YAGNIPrinciple`**: You Aren't Gonna Need It pattern validation

#### Architecture Blueprints

- **`DomainBlueprint`**: Domain layer structure generation
- **`ApplicationBlueprint`**: Application layer structure generation
- **`InfrastructureBlueprint`**: Infrastructure layer structure generation
- **`PresentationBlueprint`**: Presentation layer structure generation

### Key Functions

```python
# Pattern application functions
from pynomaly.architecture.patterns import (
    apply_clean_architecture,
    apply_hexagonal_architecture,
    apply_layered_architecture,
    apply_microservices_architecture
)

# Validation functions
from pynomaly.architecture.validators import (
    validate_layer_boundaries,
    validate_dependency_directions,
    validate_pattern_compliance
)

# Blueprint generation functions
from pynomaly.architecture.blueprints import (
    generate_domain_structure,
    generate_application_structure,
    generate_infrastructure_structure
)
```

## Development

### Testing

```bash
# Run all tests
pytest tests/

# Run pattern validation tests
pytest tests/patterns/

# Run architecture compliance tests
pytest tests/validators/

# Run with coverage
pytest --cov=architecture --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format architecture/

# Type checking
mypy architecture/

# Architecture compliance check
python -m architecture.validators --check-all
```

## Troubleshooting

### Common Issues

**Issue**: Architecture validation fails
**Solution**: Review layer boundaries and dependency directions

**Issue**: Pattern implementation doesn't match expected structure
**Solution**: Check blueprint generation and compare with examples

**Issue**: Circular dependencies detected
**Solution**: Use dependency injection and interface segregation

### Debug Mode

```python
from pynomaly.architecture.config import enable_debug_mode

# Enable debug mode for architecture validation
enable_debug_mode(
    pattern_validation=True,
    dependency_checking=True,
    layer_validation=True
)
```

## Contributing

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Branch**: Create a feature branch (`git checkout -b feature/architecture-enhancement`)
3. **Develop**: Follow architectural patterns and principles
4. **Test**: Add comprehensive tests for new patterns
5. **Document**: Update architecture documentation and ADRs
6. **Commit**: Use conventional commit messages
7. **Pull Request**: Submit a PR with clear description

### Adding New Patterns

Follow the architectural pattern template:

```python
from pynomaly.architecture.patterns.base import BasePattern

class NewArchitecturalPattern(BasePattern):
    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.validate_pattern_requirements()
    
    def apply_pattern(self, target: str) -> PatternResult:
        # Implement pattern application logic
        pass
    
    def validate_pattern(self, implementation: str) -> ValidationResult:
        # Implement pattern validation logic
        pass
```

## Support

- **Documentation**: [Package docs](docs/)
- **Architecture Guide**: [Architecture Decision Records](docs/adr/)
- **Pattern Examples**: [Pattern Examples](examples/)
- **Issues**: [GitHub Issues](../../../issues)
- **Discussions**: [GitHub Discussions](../../../discussions)

## License

MIT License. See [LICENSE](../../../LICENSE) file for details.

---

**Part of the [Pynomaly](../../../) monorepo** - Advanced platform architecture
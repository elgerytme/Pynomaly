# Open Source Project Structure

## Core Architecture

The monorepo follows a domain-driven design with self-contained packages:

```
monorepo/
├── src/packages/              # Domain packages
│   ├── ai/                   # AI/ML domain
│   ├── data/                 # Data processing domain
│   ├── infrastructure/       # Infrastructure domain
│   └── core/                 # Shared core utilities
├── tools/                    # Development and automation tools
│   ├── package-generator/    # Intelligent package creation
│   ├── domain-analyzer/      # Domain boundary analysis
│   └── infrastructure/       # Infrastructure automation
├── docs/                     # Documentation
│   ├── architecture/         # System architecture docs
│   ├── development/          # Development guides
│   └── open-source/          # Open source governance
└── .github/                  # GitHub workflows and templates
```

## Package Organization

Each package in `src/packages/` is:
- **Self-contained**: Independent deployment and operation
- **Domain-bounded**: Clear business domain boundaries
- **Production-ready**: Complete CI/CD, monitoring, security

### Package Structure
```
package-name/
├── src/                      # Source code
├── tests/                    # Test suites
├── docs/                     # Package documentation
├── infrastructure/           # Deployment configs
├── monitoring/               # Observability setup
├── pyproject.toml           # Python dependencies
├── Dockerfile               # Container definition
├── docker-compose.yml       # Local development
└── k8s/                     # Kubernetes manifests
```

## Automation Tools

### Package Generator
- **Location**: `tools/package-generator/`
- **Purpose**: AI-driven package creation with domain analysis
- **Features**: Template customization, architecture patterns, tech stack selection

### Independence Validator
- **Location**: `tools/package-independence-validator/`
- **Purpose**: Enforce package boundaries and self-containment
- **Features**: Dependency analysis, violation detection, automated enforcement

### Infrastructure Templates
- **Location**: `tools/infrastructure-templates/`
- **Purpose**: Production-ready deployment configurations
- **Features**: Docker, Kubernetes, monitoring, security scanning

## Development Workflow

1. **Package Creation**: Use package generator for new domains
2. **Independence Validation**: Automated checks in CI/CD
3. **Quality Assurance**: Comprehensive testing and security scanning
4. **Deployment**: Automated containerized deployment with monitoring

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed contribution guidelines.
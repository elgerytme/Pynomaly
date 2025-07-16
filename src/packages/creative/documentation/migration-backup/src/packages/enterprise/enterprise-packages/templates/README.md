# Enterprise Package Templates

This directory contains Jinja2 templates for generating new enterprise packages with consistent structure and best practices.

## Available Templates

### 1. Service Template (`service-template/`)

Creates a complete service package with:

- Clean architecture patterns
- Domain-driven design support
- Service and handler implementations
- Comprehensive configuration management
- Health checks and monitoring integration
- Event-driven architecture support
- Full test suite

**Use cases:**

- Business logic services
- Domain services
- Application services
- Microservices

### 2. Adapter Template (`adapter-template/`)

Creates an adapter package for external service integration with:

- HTTP client with connection pooling
- Circuit breaker and retry logic
- Rate limiting and backoff strategies
- Health checks and monitoring
- Comprehensive error handling
- Async/await support throughout

**Use cases:**

- External API integrations
- Third-party service adapters
- Protocol adapters
- Data source connectors

### 3. Infrastructure Template (`infrastructure-template/`)

Creates infrastructure components with:

- Monitoring and metrics collection
- Security and authentication utilities
- System health monitoring
- Performance tracking
- Resource management
- Prometheus integration

**Use cases:**

- Monitoring infrastructure
- Security components
- System utilities
- Performance tools

### 4. Base Template (`package-template/`)

Generic package template with:

- Basic package structure
- Configuration management
- Standard project files
- CI/CD integration

**Use cases:**

- Custom package types
- Simple utility packages
- Experimentation

## Template Structure

Each template directory contains:

```
template-name/
├── config.json              # Template configuration
├── src/
│   └── {{package_module}}/  # Source code templates
│       ├── __init__.py.template
│       ├── config.py.template
│       └── ...
├── tests/                   # Test templates
│   ├── test_*.py.template
│   └── ...
└── README.md.template       # Package documentation
```

## Template Variables

All templates support these common variables:

- `package_name`: Package name (e.g., "user-service")
- `package_module`: Python module name (e.g., "user_service")
- `package_title`: Human-readable title (e.g., "User Service")
- `package_description`: Package description
- `author_name`: Author name
- `author_email`: Author email
- `keywords`: List of keywords
- `base_dependencies`: Core dependencies
- `optional_dependencies`: Dict of optional dependency groups
- `features`: List of feature descriptions
- `repository_url`: Git repository URL
- `documentation_url`: Documentation URL

Template-specific variables are defined in each template's `config.json`.

## Usage

### Using the Package Generator

```bash
# Generate a service package
python tools/package-generator/generate_package.py my-service --template service

# Generate an adapter package
python tools/package-generator/generate_package.py my-adapter --template adapter

# Generate an infrastructure package
python tools/package-generator/generate_package.py my-infra --template infrastructure
```

### Manual Template Processing

Templates use Jinja2 syntax and can be processed manually:

```python
from jinja2 import Environment, FileSystemLoader
import json

# Load template configuration
with open("service-template/config.json") as f:
    config = json.load(f)

# Add package-specific variables
config.update({
    "package_name": "user-service",
    "package_module": "user_service",
    "author_name": "Your Name",
    "author_email": "your@email.com"
})

# Process template
env = Environment(loader=FileSystemLoader("service-template"))
template = env.get_template("src/{{package_module}}/__init__.py.template")
output = template.render(**config)
```

## Creating Custom Templates

1. Create a new directory: `your-template/`
2. Add `config.json` with template configuration
3. Create template files with `.template` extension
4. Use Jinja2 syntax for variable substitution
5. Test with the package generator

### Example Custom Template

```json
{
  "package_description": "Custom package template",
  "keywords": ["custom", "template"],
  "base_dependencies": ["\"requests>=2.31.0\""],
  "optional_dependencies": {
    "dev": ["\"pytest>=8.0.0\""]
  },
  "features": [
    "• Custom functionality",
    "• Extensible design"
  ]
}
```

## Best Practices

1. **Consistent Structure**: Follow the established patterns
2. **Comprehensive Tests**: Include test templates for all components
3. **Documentation**: Provide clear README templates
4. **Configuration**: Use environment-based configuration
5. **Dependencies**: Specify exact version ranges
6. **Security**: Include security best practices
7. **Monitoring**: Add health checks and metrics
8. **Error Handling**: Implement proper error handling
9. **Async Support**: Use async/await patterns
10. **Type Hints**: Include comprehensive type annotations

## Template Development

### Jinja2 Features Used

- Variable substitution: `{{variable}}`
- Loops: `{% for item in items %}`
- Conditionals: `{% if condition %}`
- Filters: `{{name|title|replace('-', ' ')}}`
- Comments: `{# This is a comment #}`

### Common Filters

- `title`: Capitalize words
- `upper`/`lower`: Change case
- `replace(old, new)`: String replacement
- `join(separator)`: Join lists

### Template Testing

Test templates by generating packages and verifying:

- All files are created correctly
- Package can be installed
- Tests pass
- Documentation is accurate
- CI/CD workflows work

## Integration with CI/CD

Templates include GitHub Actions workflows that:

- Run tests on multiple Python versions
- Perform security scans
- Check code quality
- Build and publish packages
- Generate documentation

## Maintenance

Templates are maintained alongside the core packages:

- Update dependencies regularly
- Add new features as patterns emerge
- Improve based on user feedback
- Keep documentation current
- Test with latest Python versions

## Support

For questions about templates:

1. Check existing template examples
2. Review the package generator documentation
3. Look at generated package examples
4. Open an issue for template improvements

# Pynomaly Configuration Management

This directory contains all configuration files for the Pynomaly project, following a standardized configuration management approach.

## Directory Structure

```
config/
├── README.md                    # This file
├── environments/                # Environment-specific configurations
│   ├── development/            # Development environment
│   ├── testing/               # Testing environment
│   ├── staging/               # Staging environment
│   └── production/            # Production environment
├── tools/                      # Tool-specific configurations
│   ├── pytest.toml           # Pytest configuration
│   ├── ruff.toml             # Ruff linting configuration
│   ├── mypy.toml             # MyPy type checking configuration
│   └── coverage.toml         # Coverage configuration
├── deployment/                 # Deployment configurations
│   ├── docker/               # Docker configurations
│   ├── kubernetes/           # Kubernetes configurations
│   └── terraform/            # Infrastructure as Code
└── schemas/                   # Configuration schemas and validation
    ├── environment.schema.json
    ├── deployment.schema.json
    └── security.schema.json
```

## Configuration Principles

### 1. Single Source of Truth
- Each configuration setting has one canonical location
- No duplicate configurations across files
- Clear inheritance and override patterns

### 2. Environment Separation
- Environment-specific settings isolated in dedicated directories
- Base configurations with environment overrides
- Secure secrets management

### 3. Tool Integration
- All tool configurations consolidated in `pyproject.toml`
- External tool configs only when necessary
- Consistent formatting and validation

### 4. Validation and Schemas
- JSON schemas for configuration validation
- Automated configuration testing
- Clear error messages for misconfigurations

## Migration from Legacy Configuration

This configuration structure replaces the previous scattered configuration files:

### Consolidated Files
- `pytest.ini` → `[tool.pytest.ini_options]` in `pyproject.toml`
- Multiple `docker-compose.yml` → `config/deployment/docker/`
- Scattered `.env` files → `config/environments/*/`
- Tool-specific configs → `[tool.*]` sections in `pyproject.toml`

### Benefits
- **Reduced Complexity**: From ~9,800+ config files to manageable structure
- **Improved Maintainability**: Single location for each configuration type
- **Better Validation**: Schema-based validation and error checking
- **Environment Consistency**: Standardized environment management
- **Developer Experience**: Clear documentation and easy navigation

## Usage

### Environment Configuration
```bash
# Load development environment
export PYNOMALY_ENV=development

# Load production environment
export PYNOMALY_ENV=production
```

### Tool Configuration
All tools are configured through `pyproject.toml`:
- Testing: `[tool.pytest.ini_options]`
- Linting: `[tool.ruff]`
- Type Checking: `[tool.mypy]`
- Coverage: `[tool.coverage.*]`
- Formatting: `[tool.black]`, `[tool.isort]`

### Deployment Configuration
```bash
# Development deployment
docker-compose -f config/deployment/docker/development.yml up

# Production deployment
docker-compose -f config/deployment/docker/production.yml up
```

## Configuration Schema

All configurations follow defined schemas for validation:

- **Environment Schema**: Validates environment-specific settings
- **Deployment Schema**: Validates deployment configurations
- **Security Schema**: Validates security-related settings

## Best Practices

1. **Use Environment Variables**: For sensitive or environment-specific values
2. **Document Changes**: Update this README when adding new configurations
3. **Validate Configurations**: Test configurations before deployment
4. **Follow Naming Conventions**: Use consistent naming across all files
5. **Version Control**: All configurations are version controlled

## Contributing

When adding new configurations:

1. Check if the setting belongs in `pyproject.toml`
2. Use appropriate environment directory for env-specific settings
3. Add schema validation if creating new configuration types
4. Update this documentation
5. Test configuration changes in development environment

## Security Considerations

- **No Secrets in Files**: Use environment variables for secrets
- **Secure Defaults**: All configurations use secure defaults
- **Access Controls**: Proper file permissions and access controls
- **Audit Trail**: Configuration changes are tracked in version control

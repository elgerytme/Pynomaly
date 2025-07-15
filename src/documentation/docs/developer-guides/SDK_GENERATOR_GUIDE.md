# Pynomaly SDK Generator Guide

The Pynomaly SDK Generator is a comprehensive tool for automatically generating client libraries in multiple programming languages from the OpenAPI specification. This guide covers installation, usage, configuration, and customization of the SDK generator.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Supported Languages](#supported-languages)
5. [Configuration](#configuration)
6. [CLI Usage](#cli-usage)
7. [Generated SDK Features](#generated-sdk-features)
8. [Customization](#customization)
9. [Testing and Validation](#testing-and-validation)
10. [Publishing](#publishing)
11. [Troubleshooting](#troubleshooting)
12. [Advanced Usage](#advanced-usage)

## Overview

The SDK Generator automatically creates production-ready client libraries from the Pynomaly OpenAPI specification. Each generated SDK includes:

- Complete API coverage with type-safe client methods
- Authentication handling (JWT and API Key)
- Error handling with retry logic and rate limiting
- Comprehensive documentation and examples
- Unit tests and CI/CD configuration
- Package management setup

## Installation

### Prerequisites

The SDK generator requires the following tools:

1. **Python 3.8+** - For running the generator
2. **Node.js 16+** - For OpenAPI Generator and TypeScript/JavaScript SDKs
3. **OpenAPI Generator CLI** - For code generation
4. **Git** - For version control

### Installing Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install OpenAPI Generator CLI
npm install -g @openapitools/openapi-generator-cli

# Verify installation
python scripts/cli/sdk_cli.py validate-environment
```

### Additional Language Dependencies

For specific languages, you may need additional tools:

- **Java**: Maven 3.6+ or Gradle 7+
- **Go**: Go 1.19+
- **C#**: .NET 6.0+
- **PHP**: PHP 8.0+ and Composer
- **Ruby**: Ruby 3.0+ and Bundler
- **Rust**: Rust 1.60+

## Quick Start

### Generate All SDKs

```bash
# Generate SDKs for all supported languages
python scripts/cli/sdk_cli.py generate --all

# Generate specific languages
python scripts/cli/sdk_cli.py generate --languages python typescript java

# Validate OpenAPI spec only
python scripts/cli/sdk_cli.py generate --validate-only
```

### Test Generated SDKs

```bash
# Test a specific SDK
python scripts/cli/sdk_cli.py test python

# Validate SDK quality
python scripts/cli/sdk_cli.py validate python

# Get SDK information
python scripts/cli/sdk_cli.py info python
```

### Publish SDKs

```bash
# Dry run (preview what would be published)
python scripts/cli/sdk_cli.py publish python --dry-run

# Publish to default registry
python scripts/cli/sdk_cli.py publish python

# Publish to specific registry
python scripts/cli/sdk_cli.py publish python --registry pypi
```

## Supported Languages

The SDK generator supports the following programming languages:

| Language   | Generator        | Package Manager | Package Format |
|------------|------------------|-----------------|----------------|
| Python     | python           | pip             | PyPI           |
| TypeScript | typescript-fetch | npm             | npm            |
| Java       | java             | Maven/Gradle    | Maven Central  |
| Go         | go               | go modules      | Go modules     |
| C#         | csharp           | NuGet           | NuGet          |
| PHP        | php              | Composer        | Packagist      |
| Ruby       | ruby             | Bundler         | RubyGems       |
| Rust       | rust             | Cargo           | crates.io      |

### Language-Specific Features

#### Python

- Async/await support with `asyncio`
- Type hints and Pydantic models
- pytest test suite
- Black code formatting
- mypy type checking

#### TypeScript

- Full TypeScript definitions
- Support for Node.js and browsers
- Jest test framework
- ESLint configuration
- Webpack/Rollup bundling

#### Java

- Maven and Gradle support
- Java 11+ compatibility
- JUnit 5 tests
- OkHttp client library
- Jackson JSON serialization

#### Go

- Go modules support
- Context-based cancellation
- Standard library HTTP client
- Built-in testing framework

#### C #

- .NET 6.0+ compatibility
- NuGet package support
- xUnit test framework
- HttpClient implementation

## Configuration

### Configuration File

The SDK generator uses a YAML configuration file located at `config/sdk_generator_config.yaml`. This file controls all aspects of SDK generation:

```yaml
# Global configuration
global:
  openapi_spec: "docs/api/openapi.yaml"
  output_directory: "sdks"
  package:
    vendor: "pynomaly"
    version: "1.0.0"
    license: "MIT"

# Language-specific configuration
languages:
  python:
    enabled: true
    package:
      name: "pynomaly_client"
      pypi_name: "pynomaly-client"
    features:
      async_support: true
      retry_logic: true
      rate_limiting: true
```

### Customizing Language Settings

#### Enable/Disable Languages

```yaml
languages:
  python:
    enabled: true    # Generate Python SDK
  java:
    enabled: false   # Skip Java SDK
```

#### Package Configuration

```yaml
languages:
  python:
    package:
      name: "pynomaly_client"
      pypi_name: "pynomaly-client"
      import_name: "pynomaly_client"
    configuration:
      python_version: "3.8"
      library: "requests"
```

#### Feature Toggles

```yaml
languages:
  python:
    features:
      async_support: true     # Generate async methods
      retry_logic: true       # Include retry mechanisms
      rate_limiting: true     # Handle rate limits
      logging: true          # Include logging
      testing: true          # Generate test suites
      documentation: true    # Generate docs
      ci_cd: true           # Include CI/CD configs
```

## CLI Usage

### Basic Commands

```bash
# List supported languages
python scripts/cli/sdk_cli.py list-languages

# Generate SDKs
python scripts/cli/sdk_cli.py generate [options]

# Test SDKs
python scripts/cli/sdk_cli.py test <language>

# Validate SDKs
python scripts/cli/sdk_cli.py validate <language>

# Publish SDKs
python scripts/cli/sdk_cli.py publish <language> [options]

# Get SDK information
python scripts/cli/sdk_cli.py info <language>

# Check environment
python scripts/cli/sdk_cli.py validate-environment

# Generate status report
python scripts/cli/sdk_cli.py status [--output report.json]
```

### Generation Options

```bash
# Generate specific languages
python scripts/cli/sdk_cli.py generate --languages python typescript

# Generate all supported languages
python scripts/cli/sdk_cli.py generate --all

# Validate spec without generating
python scripts/cli/sdk_cli.py generate --validate-only
```

### Publishing Options

```bash
# Dry run (preview only)
python scripts/cli/sdk_cli.py publish python --dry-run

# Publish to specific registry
python scripts/cli/sdk_cli.py publish python --registry pypi
python scripts/cli/sdk_cli.py publish typescript --registry npm
```

## Generated SDK Features

### Authentication

All generated SDKs support multiple authentication methods:

```python
# Python example
from pynomaly_client import PynomaliClient

# JWT Authentication
client = PynomaliClient(base_url="https://api.pynomaly.com")
token_response = client.auth.login("username", "password")

# API Key Authentication
client = PynomaliClient(
    base_url="https://api.pynomaly.com",
    api_key="your-api-key"
)
```

### Error Handling

SDKs include comprehensive error handling:

```python
# Python example
from pynomaly_client.exceptions import ApiException

try:
    result = client.detection.detect(data, algorithm)
except ApiException as e:
    print(f"API Error: {e.status} - {e.reason}")
    print(f"Details: {e.body}")
```

### Rate Limiting

Automatic rate limit handling with exponential backoff:

```python
# Automatically handled by the SDK
result = client.detection.detect(data, algorithm)
# SDK will retry with backoff if rate limited
```

### Async Support

Languages that support async operations include async methods:

```python
# Python async example
import asyncio
from pynomaly_client import AsyncPynomaliClient

async def main():
    client = AsyncPynomaliClient()
    await client.auth.login("username", "password")
    result = await client.detection.detect(data, algorithm)
    await client.close()

asyncio.run(main())
```

## Customization

### Custom Templates

You can customize generated code using custom templates:

1. Create template directory: `templates/python/`
2. Add Mustache templates: `templates/python/README.md.mustache`
3. Configure in `sdk_generator_config.yaml`:

```yaml
templates:
  overrides:
    - language: "python"
      file: "README.md"
      template: "templates/python/README.md.mustache"
```

### Post-Generation Scripts

Run custom scripts after generation:

```yaml
post_generation:
  scripts:
    python:
      - "python -m black ."
      - "python -m isort ."
      - "python -m mypy --install-types --non-interactive"
```

### Additional Files

Add custom files to generated SDKs:

```yaml
post_generation:
  additional_files:
    - source: "templates/common/CONTRIBUTING.md"
      destination: "CONTRIBUTING.md"
      languages: ["python", "typescript"]
```

## Testing and Validation

### Quality Gates

The generator includes quality gates to ensure SDK quality:

```yaml
quality_gates:
  requirements:
    - name: "authentication_support"
      description: "Must support JWT and API key authentication"
      mandatory: true
    
    - name: "error_handling"
      description: "Must include comprehensive error handling"
      mandatory: true
    
    - name: "comprehensive_tests"
      description: "Must include unit tests for core functionality"
      mandatory: true
```

### Running Tests

```bash
# Test specific SDK
python scripts/cli/sdk_cli.py test python

# Test all generated SDKs
for lang in python typescript java; do
    python scripts/cli/sdk_cli.py test $lang
done
```

### Validation Checks

```bash
# Validate SDK against quality gates
python scripts/cli/sdk_cli.py validate python

# Check compilation/build
python scripts/cli/sdk_cli.py validate typescript
```

## Publishing

### Registry Configuration

Configure package registries in the config file:

```yaml
ci_cd:
  registries:
    python:
      - name: "PyPI"
        url: "https://pypi.org/"
        auth_token_secret: "PYPI_API_TOKEN"
    
    typescript:
      - name: "npm"
        url: "https://registry.npmjs.org/"
        auth_token_secret: "NPM_TOKEN"
```

### Publishing Process

1. **Validate SDK**: Ensure quality gates pass
2. **Run Tests**: Execute test suite
3. **Build Package**: Create distribution package
4. **Publish**: Upload to registry

```bash
# Complete publishing workflow
python scripts/cli/sdk_cli.py validate python
python scripts/cli/sdk_cli.py test python
python scripts/cli/sdk_cli.py publish python
```

### CI/CD Integration

Generated SDKs include GitHub Actions workflows:

```yaml
# .github/workflows/ci.yml
name: Python SDK CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
```

## Troubleshooting

### Common Issues

#### OpenAPI Generator Not Found

```bash
# Install OpenAPI Generator
npm install -g @openapitools/openapi-generator-cli

# Verify installation
openapi-generator-cli version
```

#### Generation Fails

```bash
# Check environment
python scripts/cli/sdk_cli.py validate-environment

# Validate OpenAPI spec
python scripts/cli/sdk_cli.py generate --validate-only

# Check logs
python scripts/cli/sdk_cli.py generate --languages python --verbose
```

#### Tests Fail

```bash
# Check SDK structure
python scripts/cli/sdk_cli.py info python

# Run individual test commands
cd sdks/python && python -m pytest -v
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run generator with debug output
generator = SDKGenerator()
generator.generate_all_sdks(["python"])
```

### Getting Help

1. Check the troubleshooting section above
2. Review the generated SDK structure with `info` command
3. Validate environment with `validate-environment`
4. Check GitHub issues for similar problems
5. Create a new issue with detailed error information

## Advanced Usage

### Custom Generator

Create a custom SDK generator for specialized needs:

```python
from scripts.sdk_generator import SDKGenerator

class CustomSDKGenerator(SDKGenerator):
    def __init__(self):
        super().__init__()
        # Add custom language configurations
        self.languages["kotlin"] = {
            "generator": "kotlin",
            "package_name": "com.pynomaly.client",
            "client_name": "PynomaliClient",
            "additional_properties": {
                "packageName": "com.pynomaly.client"
            }
        }
    
    def _post_process_sdk(self, language: str, output_path: Path, spec: Dict[str, Any]):
        super()._post_process_sdk(language, output_path, spec)
        
        # Add custom post-processing
        if language == "kotlin":
            self._add_kotlin_specific_files(output_path)
```

### Programmatic Usage

Use the SDK generator programmatically:

```python
from scripts.sdk_generator import SDKGenerator
from pathlib import Path

# Initialize generator
generator = SDKGenerator()

# Load OpenAPI spec
spec = generator.load_openapi_spec()

# Generate specific SDKs
results = {}
for language in ["python", "typescript"]:
    results[language] = generator.generate_sdk(language, spec)

# Check results
successful = [lang for lang, success in results.items() if success]
print(f"Successfully generated: {successful}")
```

### Batch Operations

Generate and validate multiple SDKs:

```bash
#!/bin/bash
# batch_generate.sh

LANGUAGES=("python" "typescript" "java" "go")

for lang in "${LANGUAGES[@]}"; do
    echo "Generating $lang SDK..."
    python scripts/cli/sdk_cli.py generate --languages $lang
    
    echo "Testing $lang SDK..."
    python scripts/cli/sdk_cli.py test $lang
    
    echo "Validating $lang SDK..."
    python scripts/cli/sdk_cli.py validate $lang
done

echo "Generating status report..."
python scripts/cli/sdk_cli.py status --output batch_report.json
```

### Integration with CI/CD

```yaml
# .github/workflows/sdk-generation.yml
name: SDK Generation

on:
  push:
    paths:
      - 'docs/api/openapi.yaml'
      - 'config/sdk_generator_config.yaml'

jobs:
  generate-sdks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          npm install -g @openapitools/openapi-generator-cli
      
      - name: Generate SDKs
        run: |
          python scripts/cli/sdk_cli.py generate --all
      
      - name: Test SDKs
        run: |
          for lang in python typescript java; do
            python scripts/cli/sdk_cli.py test $lang
          done
      
      - name: Upload SDKs
        uses: actions/upload-artifact@v3
        with:
          name: generated-sdks
          path: sdks/
```

This comprehensive guide covers all aspects of the Pynomaly SDK Generator. For additional help or feature requests, please refer to the project documentation or create an issue in the repository.

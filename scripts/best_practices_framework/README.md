# Best Practices Framework

[![PyPI version](https://badge.fury.io/py/best-practices-framework.svg)](https://badge.fury.io/py/best-practices-framework)
[![Python versions](https://img.shields.io/pypi/pyversions/best-practices-framework.svg)](https://pypi.org/project/best-practices-framework/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/best-practices-framework/best-practices-framework/workflows/Tests/badge.svg)](https://github.com/best-practices-framework/best-practices-framework/actions)
[![Coverage](https://codecov.io/gh/best-practices-framework/best-practices-framework/branch/main/graph/badge.svg)](https://codecov.io/gh/best-practices-framework/best-practices-framework)

A comprehensive, automated framework for enforcing software engineering best practices across architecture, security, testing, DevOps, and Site Reliability Engineering (SRE) domains.

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install best-practices-framework

# Full installation with all features
pip install best-practices-framework[full]

# Development installation
pip install best-practices-framework[dev]
```

### Basic Usage

```python
from best_practices_framework import BestPracticesValidator

# Initialize validator
validator = BestPracticesValidator(project_root=".")

# Run comprehensive validation
report = await validator.validate_all()

print(f"Overall Score: {report.compliance_score.overall_score:.1f}%")
print(f"Grade: {report.compliance_score.grade}")

# Check quality gate
if validator.quality_gate(report):
    print("✅ Quality gate PASSED")
else:
    print("❌ Quality gate FAILED")
```

### Command Line Interface

```bash
# Validate all categories
best-practices validate

# Validate specific category
best-practices validate --category security

# Generate HTML report
best-practices report --format html --output report.html

# Initialize configuration
best-practices init --profile fintech
```

## 📊 What It Validates

### 🏗️ Architecture
- **Clean Architecture**: Dependency inversion, layer separation, SOLID principles
- **Domain-Driven Design**: Bounded contexts, domain entities, ubiquitous language  
- **Microservices**: Service independence, data ownership, API contracts
- **Scalability**: Performance patterns, caching, load balancing

### 🔒 Security
- **OWASP Top 10**: Injection attacks, authentication, sensitive data exposure
- **Secrets Detection**: Hardcoded passwords, API keys, certificates
- **Vulnerability Scanning**: Dependency vulnerabilities, security misconfigurations
- **Compliance**: SOC2, ISO27001, PCI DSS, GDPR requirements

### 🧪 Testing
- **Test Pyramid**: Unit (70%), Integration (20%), E2E (10%) distribution
- **Coverage Thresholds**: Configurable coverage requirements per test type
- **Quality Gates**: Performance benchmarks, flaky test detection
- **Test-Driven Development**: Red-Green-Refactor cycle validation

### 🛠️ Engineering
- **Code Quality**: Complexity metrics, naming conventions, documentation
- **Version Control**: Git workflow, branch protection, commit standards
- **Technical Debt**: Maintainability index, code duplication detection
- **Documentation**: API docs, architecture decisions, runbooks

### 🚀 DevOps
- **CI/CD Pipelines**: Automated builds, testing, security gates, deployment
- **Infrastructure as Code**: Version control, immutable infrastructure, drift detection
- **Deployment Strategies**: Blue-green, canary deployments, rollback capabilities
- **Configuration Management**: Environment parity, secrets management

### 📈 Site Reliability Engineering
- **Observability**: Metrics, logging, distributed tracing, alerting
- **Service Level Objectives**: SLIs, SLOs, error budget management
- **Incident Management**: Runbooks, post-mortems, on-call procedures
- **Capacity Planning**: Resource forecasting, auto-scaling, cost optimization

## 🎯 Key Features

### ✅ Comprehensive Coverage
- **6+ Categories**: Architecture, Security, Testing, Engineering, DevOps, SRE
- **100+ Built-in Rules**: Industry-standard best practices out of the box
- **Multi-Language Support**: Python, JavaScript/TypeScript, Java, Go, C#, and more

### ⚙️ Highly Configurable  
- **YAML Configuration**: Easy-to-customize rules and thresholds
- **Industry Profiles**: Pre-configured settings for fintech, healthcare, e-commerce
- **Compliance Frameworks**: Built-in support for major compliance standards

### 🔌 Extensible Architecture
- **Plugin System**: Create custom validators and integrations
- **Rule Engine**: Add domain-specific rules without code changes
- **Integration APIs**: Connect with existing tools and workflows

### 📊 Rich Reporting
- **Multiple Formats**: HTML, JSON, SARIF, JUnit, Markdown
- **Interactive Dashboards**: Web-based visualization of results
- **Trend Analysis**: Track improvements over time
- **CI/CD Integration**: Native support for all major CI/CD platforms

## 🛠️ Configuration

Create a `.best-practices.yml` configuration file:

```yaml
# Enable/disable categories
enabled_categories:
  - architecture
  - security
  - testing
  - devops

# Global settings
global:
  enforcement_level: strict
  fail_on_critical: true
  max_violations_per_category: 10

# Category-specific configuration
security:
  owasp:
    enabled: true
    top_10_compliance: true
  secrets_detection:
    enabled: true
    scan_all_files: true

testing:
  coverage:
    unit_test_minimum: 80
    integration_test_minimum: 60
  test_pyramid:
    unit_tests_percentage: 70
    integration_tests_percentage: 20

# Industry profile (optional)
profile: fintech

# Compliance frameworks (optional)
compliance:
  soc2: true
  pci_dss: true
```

## 🚦 CI/CD Integration

### GitHub Actions

```yaml
name: Best Practices Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Best Practices Framework
      uses: best-practices-framework/setup-action@v1
      with:
        version: 'latest'
    
    - name: Run Validation
      run: |
        best-practices validate --format json --output results.json
        best-practices report --format html --output report.html
    
    - name: Quality Gate
      run: best-practices quality-gate --enforce-critical
    
    - name: Upload Results
      uses: actions/upload-artifact@v4
      with:
        name: validation-results
        path: |
          results.json
          report.html
```

### GitLab CI

```yaml
stages:
  - validate

best_practices:
  stage: validate
  image: python:3.11
  before_script:
    - pip install best-practices-framework[full]
  script:
    - best-practices validate --format json --output results.json
    - best-practices quality-gate --enforce-critical
  artifacts:
    reports:
      junit: results.xml
    paths:
      - results.json
      - report.html
  rules:
    - if: '$CI_PIPELINE_SOURCE == "push"'
```

## 📈 Scoring System

The framework uses a weighted scoring system:

- **Security**: 30% weight (highest priority)
- **Architecture**: 25% weight  
- **Testing**: 20% weight
- **DevOps**: 15% weight
- **Engineering**: 10% weight

**Violation Penalties:**
- Critical: -20 points
- High: -15 points  
- Medium: -10 points
- Low: -5 points
- Info: -1 point

**Grade Scale:**
- A+ (95-100%): Excellent
- A (90-94%): Very Good  
- B (75-89%): Good
- C (60-74%): Needs Improvement
- D (50-59%): Poor
- F (<50%): Failing

## 🔧 Extending the Framework

### Custom Validators

```python
from best_practices_framework import BaseValidator

class MyCustomValidator(BaseValidator):
    def get_name(self) -> str:
        return "my_custom_validator"
    
    def get_category(self) -> str:
        return "architecture"
    
    def get_description(self) -> str:
        return "Validates custom architecture rules"
    
    async def validate(self) -> ValidationResult:
        # Your validation logic here
        if self.violates_custom_rule():
            self.add_violation(
                rule_id="CUSTOM_001",
                severity="high", 
                message="Custom rule violation detected",
                suggestion="Fix the custom rule violation"
            )
        
        return self.create_result(execution_time=0.1)
```

### Custom Integrations

```python
from best_practices_framework.integrations import BaseIntegration

class MyToolIntegration(BaseIntegration):
    def get_name(self) -> str:
        return "my_tool"
    
    async def export_results(self, report: ValidationReport):
        # Export results to your tool
        pass
    
    async def import_config(self) -> dict:
        # Import configuration from your tool
        return {}
```

## 📚 Documentation

- **[User Guide](https://docs.bestpractices.dev/user-guide/)**: Comprehensive usage documentation
- **[API Reference](https://docs.bestpractices.dev/api/)**: Complete API documentation  
- **[Rule Catalog](https://docs.bestpractices.dev/rules/)**: All built-in rules and their descriptions
- **[Integration Guide](https://docs.bestpractices.dev/integrations/)**: CI/CD and tool integrations
- **[Examples](https://github.com/best-practices-framework/examples)**: Sample projects and configurations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/best-practices-framework/best-practices-framework.git
cd best-practices-framework

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run validation on itself
best-practices validate --category all
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OWASP](https://owasp.org/) for security best practices
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) by Robert C. Martin
- [Site Reliability Engineering](https://sre.google/) practices from Google
- [DevOps Research and Assessment (DORA)](https://www.devops-research.com/research.html) metrics

## 📞 Support

- **Documentation**: https://docs.bestpractices.dev
- **Issues**: https://github.com/best-practices-framework/best-practices-framework/issues
- **Discussions**: https://github.com/best-practices-framework/best-practices-framework/discussions
- **Email**: support@bestpractices.dev
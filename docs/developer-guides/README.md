# Developer Guides

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 👨‍💻 [Developer Guides](README.md)

---


Technical documentation for developers, integrators, and contributors working with Pynomaly's codebase and APIs.

## 📋 Quick Navigation

### 🏗️ **[Architecture](architecture/)**
System design, patterns, and architectural principles.
- **[Overview](architecture/overview.md)** - Clean architecture and design principles
- **[Continuous Learning](architecture/continuous-learning-framework.md)** - Online learning systems
- **[Deployment Pipeline](architecture/deployment-pipeline-framework.md)** - CI/CD architecture
- **[Model Persistence](architecture/model-persistence-framework.md)** - ML model management
- **[ADRs](architecture/adr/)** - Architectural decision records

### 🔌 **[API Integration](api-integration/)**
Programming interfaces and integration patterns.
- **[REST API](api-integration/rest-api.md)** - HTTP API reference and examples
- **[Python SDK](api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](api-integration/cli.md)** - Command-line interface
- **[Authentication](api-integration/authentication.md)** - Security and auth patterns
- **[Domain API](api-integration/domain.md)** - Core domain interfaces
- **[OpenAPI Spec](api-integration/openapi.yaml)** - API specification
- **[Quick Reference](api-integration/API_QUICK_REFERENCE.md)** - API cheat sheet
- **[Web API Setup](api-integration/WEB_API_SETUP_GUIDE.md)** - Complete setup guide

### 🤝 **[Contributing](contributing/)**
Development setup, standards, and contribution guidelines.
- **[Contributing Guidelines](contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](contributing/README.md)** - Local development environment
- **[Git Workflow](contributing/CI_CD_BRANCH_COMPLIANCE.md)** - Branch naming conventions, CI/CD integration, and development workflow
- **[Hatch Guide](contributing/HATCH_GUIDE.md)** - Build system and environment management
- **[Implementation Guide](contributing/IMPLEMENTATION_GUIDE.md)** - Architecture and coding patterns
- **[File Organization](contributing/FILE_ORGANIZATION_STANDARDS.md)** - Project structure standards
- **[Environment Management](contributing/ENVIRONMENT_MANAGEMENT_MIGRATION.md)** - Environment setup
- **[Dependency Management](contributing/DEPENDENCY_GUIDE.md)** - Managing dependencies
- **[Test Analysis](contributing/COMPREHENSIVE_TEST_ANALYSIS.md)** - Testing infrastructure
- **[Troubleshooting](contributing/troubleshooting/)** - Development troubleshooting

---

## 🎯 Developer Journey Paths

### **New Contributor**
1. **[Contributing Guidelines](contributing/CONTRIBUTING.md)** - Understand the process
2. **[Development Setup](contributing/README.md)** - Set up local environment
3. **[Git Workflow](contributing/CI_CD_BRANCH_COMPLIANCE.md)** - Master branch naming and CI/CD workflow
4. **[Hatch Guide](contributing/HATCH_GUIDE.md)** - Master the build system
5. **[Architecture Overview](architecture/overview.md)** - Understand system design

### **API Integration Developer**
1. **[REST API](api-integration/rest-api.md)** - Understand HTTP endpoints
2. **[Authentication](api-integration/authentication.md)** - Implement security
3. **[Python SDK](api-integration/python-sdk.md)** - Use client library
4. **[OpenAPI Spec](api-integration/openapi.yaml)** - Reference specification

### **Platform Developer**
1. **[Architecture](architecture/)** - Master system design
2. **[Implementation Guide](contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards
3. **[Model Persistence](architecture/model-persistence-framework.md)** - Data layer
4. **[Deployment Pipeline](architecture/deployment-pipeline-framework.md)** - CI/CD

### **DevOps Engineer**
1. **[Deployment Pipeline](architecture/deployment-pipeline-framework.md)** - CI/CD architecture
2. **[Environment Management](contributing/ENVIRONMENT_MANAGEMENT_MIGRATION.md)** - Environment setup
3. **[File Organization](contributing/FILE_ORGANIZATION_STANDARDS.md)** - Project structure
4. **[Dependency Management](contributing/DEPENDENCY_GUIDE.md)** - Package management

---

## 🏗️ Architecture Overview

Pynomaly follows **Clean Architecture** principles with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   REST API  │ │     CLI     │ │     Progressive Web App │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ Use Cases   │ │   App       │ │       Autonomous        │ │
│  │             │ │ Services    │ │        Services         │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                     Domain Layer                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │  Entities   │ │   Value     │ │      Domain             │ │
│  │             │ │  Objects    │ │      Services           │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │  Adapters   │ │ Persistence │ │    External Services    │ │
│  │  (PyOD,etc) │ │             │ │                         │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### **Key Principles**
- **Domain-Driven Design** - Business logic in domain layer
- **Dependency Inversion** - Dependencies point inward
- **Hexagonal Architecture** - Ports and adapters pattern
- **Clean Code** - SOLID principles throughout

---

## 🔧 Development Stack

### **Build System**
- **[Hatch](contributing/HATCH_GUIDE.md)** - Modern Python build system and environment management
- **Poetry** - Alternative dependency management (legacy support)
- **Pre-commit** - Git hooks for code quality
- **GitHub Actions** - CI/CD pipeline automation

### **Code Quality**
- **Ruff** - Lightning-fast linting and formatting
- **MyPy** - Static type checking with strict mode
- **Black** - Code formatting (via Ruff)
- **isort** - Import sorting (via Ruff)

### **Testing**
- **pytest** - Test framework with extensive plugins
- **pytest-cov** - Coverage reporting
- **pytest-asyncio** - Async test support
- **Hypothesis** - Property-based testing
- **Playwright** - Browser automation and UI testing

### **Documentation**
- **MkDocs** - Documentation site generation
- **Sphinx** - API documentation generation
- **OpenAPI** - API specification and documentation
- **Storybook** - UI component documentation

---

## 📚 Technical Resources

### **Architecture Patterns**
- **[Clean Architecture](architecture/overview.md)** - Robert C. Martin's architecture
- **[Hexagonal Architecture](architecture/overview.md)** - Ports and adapters
- **[Domain-Driven Design](architecture/overview.md)** - Business-focused design
- **[CQRS Pattern](architecture/overview.md)** - Command Query Responsibility Segregation

### **API Design**
- **[REST API Best Practices](api-integration/rest-api.md)** - HTTP API design
- **[OpenAPI Specification](api-integration/openapi.yaml)** - API documentation
- **[Authentication Patterns](api-integration/authentication.md)** - Security implementation
- **[SDK Design](api-integration/python-sdk.md)** - Client library patterns

### **Testing Strategies**
- **[Test-Driven Development](contributing/COMPREHENSIVE_TEST_ANALYSIS.md)** - TDD practices
- **[Testing Pyramid](contributing/COMPREHENSIVE_TEST_ANALYSIS.md)** - Test strategy
- **[Integration Testing](contributing/COMPREHENSIVE_TEST_ANALYSIS.md)** - System testing
- **[Property-Based Testing](contributing/COMPREHENSIVE_TEST_ANALYSIS.md)** - Hypothesis testing

---

## 🚀 Advanced Topics

### **Performance Optimization**
- **Async Programming** - High-performance async/await patterns
- **Memory Management** - Efficient data processing
- **Caching Strategies** - Redis and in-memory caching
- **Database Optimization** - SQLAlchemy performance tuning

### **Scalability Patterns**
- **Microservices** - Service decomposition
- **Event-Driven Architecture** - Async messaging
- **Load Balancing** - Traffic distribution
- **Horizontal Scaling** - Multi-instance deployment

### **Security Implementation**
- **JWT Authentication** - Token-based auth
- **Role-Based Access Control** - Authorization patterns
- **Input Validation** - Data sanitization
- **Audit Logging** - Security event tracking

---

## 🔗 Related Documentation

### **User Documentation**
- **[User Guides](../user-guides/)** - Feature usage and best practices
- **[Getting Started](../getting-started/)** - Installation and setup
- **[Examples](../examples/)** - Real-world use cases

### **Reference Documentation**
- **[Algorithm Reference](../reference/algorithms/)** - Algorithm documentation
- **[Configuration Reference](../reference/configuration/)** - System configuration
- **[API Reference](api-integration/)** - Complete API documentation

### **Operations**
- **[Deployment](../deployment/)** - Production deployment
- **[Monitoring](../user-guides/basic-usage/monitoring.md)** - System observability
- **[Security](../deployment/SECURITY.md)** - Security best practices

---

## 💡 Development Best Practices

### **Code Quality**
- Follow **[Implementation Guide](contributing/IMPLEMENTATION_GUIDE.md)** standards
- Use **[File Organization](contributing/FILE_ORGANIZATION_STANDARDS.md)** conventions
- Implement comprehensive testing with **[Test Analysis](contributing/COMPREHENSIVE_TEST_ANALYSIS.md)**
- Maintain 100% type hint coverage

### **Contribution Workflow**
1. Read **[Contributing Guidelines](contributing/CONTRIBUTING.md)**
2. Set up **[Development Environment](contributing/README.md)**
3. Follow **[Hatch Workflow](contributing/HATCH_GUIDE.md)**
4. Submit quality pull requests

### **Architecture Compliance**
- Respect **[Clean Architecture](architecture/overview.md)** boundaries
- Follow **[Domain-Driven Design](architecture/overview.md)** principles
- Use **[Dependency Injection](architecture/overview.md)** patterns
- Document **[Architectural Decisions](architecture/adr/)**

---

**Ready to contribute?** Start with our **[Contributing Guidelines](contributing/CONTRIBUTING.md)** and **[Development Setup](contributing/README.md)** guides.

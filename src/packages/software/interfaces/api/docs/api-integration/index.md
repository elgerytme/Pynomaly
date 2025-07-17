# API Integration

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 💻 [Developer Guides](../index.md) > 🔌 API Integration

---

## 🎯 API Integration Overview

This section provides comprehensive guidance for integrating with Software's APIs. Whether you're building applications, creating integrations, or developing custom solutions, these guides will help you understand and use all available APIs effectively.

---

## 📋 Quick Navigation

### 🚀 **Quick Start**
- **[API Quick Reference →](API_QUICK_REFERENCE.md)** - Fast API reference
- **[Web API Setup →](WEB_API_SETUP_GUIDE.md)** - Web API configuration

### 🔌 **API Types**
- **[REST API →](rest-api.md)** - RESTful API integration
- **[Python SDK →](python-sdk.md)** - Python SDK usage
- **[CLI →](cli.md)** - Command line interface
- **[REST →](rest.md)** - REST API details

### 📚 **Reference**
- **[API Reference →](reference.md)** - Complete API documentation
- **[Domain →](domain.md)** - Domain concepts and models
- **[OpenAPI Spec →](openapi.yaml)** - OpenAPI specification

### 🔐 **Security**
- **[Authentication →](authentication.md)** - Security and authentication

---

## 🚀 Quick Start Guide

### **⚡ 30-Second Setup**
Get started with the API in 30 seconds:

1. **[API Quick Reference](API_QUICK_REFERENCE.md)** - Essential API endpoints
2. **[Web API Setup](WEB_API_SETUP_GUIDE.md)** - Initial configuration
3. **[Authentication](authentication.md)** - Security setup

### **🔧 Development Setup**
Set up your development environment:

```python
# Install Software SDK
pip install software

# Quick API test
from pynomaly.client import PynamolyClient
client = PynamolyClient(api_key="your-api-key")
```

👉 **Learn more**: [Python SDK Guide](python-sdk.md)

---

## 🔌 API Integration Types

### **🌐 REST API**
HTTP-based REST API integration:
- **[REST API Guide](rest-api.md)** - Complete REST API integration
- **[REST Details](rest.md)** - REST API specifics
- **[OpenAPI Spec](openapi.yaml)** - Machine-readable API specification
- **[API Reference](reference.md)** - Complete endpoint documentation

### **🐍 Python SDK**
Native Python SDK integration:
- **[Python SDK Guide](python-sdk.md)** - Python SDK usage
- **[Domain Models](domain.md)** - Domain-driven design concepts
- **[Authentication](authentication.md)** - SDK authentication

### **💻 Command Line Interface**
CLI integration and automation:
- **[CLI Guide](cli.md)** - Command line interface
- **[CLI Automation](cli.md#automation)** - Scripting and automation
- **[CLI Reference](cli.md#reference)** - Command reference

---

## 🎯 Integration Patterns

### **🔗 Application Integration**
Integrate Software into your applications:

#### **Web Applications**
- **[REST API](rest-api.md)** - HTTP integration
- **[Web API Setup](WEB_API_SETUP_GUIDE.md)** - Web configuration
- **[Authentication](authentication.md)** - Web security

#### **Python Applications**
- **[Python SDK](python-sdk.md)** - Native integration
- **[Domain Models](domain.md)** - Domain concepts
- **[API Reference](reference.md)** - SDK reference

#### **Microservices**
- **[REST API](rest-api.md)** - Service-to-service
- **[Authentication](authentication.md)** - Service authentication
- **[OpenAPI Spec](openapi.yaml)** - Service contracts

### **🤖 Automation Integration**
Automate anomaly processing workflows:

#### **CI/CD Pipelines**
- **[CLI Integration](cli.md)** - Pipeline automation
- **[REST API](rest-api.md)** - HTTP automation
- **[Authentication](authentication.md)** - Pipeline security

#### **Monitoring Systems**
- **[Python SDK](python-sdk.md)** - Monitoring integration
- **[API Reference](reference.md)** - Monitoring APIs
- **[Domain Models](domain.md)** - Monitoring concepts

---

## 🔐 Security & Authentication

### **🔑 Authentication Methods**
Secure your API integration:
- **[Authentication Guide](authentication.md)** - Complete authentication
- **API Keys** - Simple key-based authentication
- **JWT Tokens** - JSON Web Token authentication
- **OAuth 2.0** - OAuth integration

### **🛡️ Security Best Practices**
- **[Security Guide](../../security/)** - Security best practices
- **[API Security](authentication.md#security)** - API-specific security
- **[Production Security](../../deployment/SECURITY.md)** - Production hardening

---

## 📊 API Usage Examples

### **🎯 Basic Examples**
Common API usage patterns:

#### **REST API Example**
```bash
# Get algorithm list
curl -X GET "https://api.pynomaly.com/v1/algorithms" \
  -H "Authorization: Bearer YOUR_API_KEY"

# Detect anomalies
curl -X POST "https://api.pynomaly.com/v1/detect" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "IsolationForest", "data": [...]}'
```

#### **Python SDK Example**
```python
from pynomaly.client import PynamolyClient
from pynomaly.models import Dataset

client = PynamolyClient(api_key="your-api-key")
data_collection = DataCollection.from_csv("data.csv")
result = client.detect_anomalies(data_collection, algorithm="IsolationForest")
```

#### **CLI Example**
```bash
# Detect anomalies via CLI
software detect --algorithm IsolationForest --input data.csv --output results.json

# List available algorithms
software algorithms list
```

### **🚀 Advanced Examples**
Advanced integration scenarios:
- **[Banking Integration](../../examples/banking/)** - Financial services
- **[Data Quality](../../examples/Data_Quality_Anomaly_Detection_Guide.md)** - Data validation
- **[Autonomous Mode](../../user-guides/basic-usage/autonomous-mode.md)** - Automated processing

---

## 🔗 Related Documentation

### **User Guides**
- **[Basic Usage](../../user-guides/basic-usage/)** - Using APIs
- **[Advanced Features](../../user-guides/advanced-features/)** - Advanced API usage
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - API monitoring

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/)** - Algorithm details
- **[Configuration](../../reference/configuration/)** - API configuration
- **[API Reference](../../reference/api/)** - Complete API docs

### **Examples**
- **[Examples](../../examples/)** - Real-world examples
- **[Tutorials](../../examples/tutorials/)** - Step-by-step guides
- **[Banking Examples](../../examples/banking/)** - Financial use cases

---

## 🆘 Getting Help

### **API Support**
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report API issues
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions
- **[API Reference](reference.md)** - Complete documentation

### **Integration Help**
- **[API Quick Reference](API_QUICK_REFERENCE.md)** - Quick help
- **[Web API Setup](WEB_API_SETUP_GUIDE.md)** - Setup help
- **[Authentication](authentication.md)** - Security help

---

## 🚀 Quick Start

Ready to integrate? Choose your path:

### **🌐 For REST API**
Start with: **[REST API Guide](rest-api.md)**

### **🐍 For Python Development**
Start with: **[Python SDK Guide](python-sdk.md)**

### **💻 For CLI Integration**
Start with: **[CLI Guide](cli.md)**

### **⚡ For Quick Reference**
Start with: **[API Quick Reference](API_QUICK_REFERENCE.md)**

---

**Last Updated**: 2025-01-09  
**Next Review**: 2025-02-09
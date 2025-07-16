# Basic Usage

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../index.md) > 🎓 Basic Usage

---

## 🎯 Basic Usage Overview

This section covers the fundamental concepts and operations for using Pynomaly effectively. Whether you're new to anomaly detection or new to Pynomaly, these guides will help you understand the core functionality and get productive quickly.

---

## 📋 Essential Guides

### 📊 **Working with Data**
- **[Datasets →](datasets.md)** - Loading, preparing, and managing datasets in Pynomaly

### 🤖 **Automated Detection**
- **[Autonomous Mode →](autonomous-mode.md)** - Automated anomaly detection workflows

### 📈 **System Monitoring**
- **[Monitoring →](monitoring.md)** - Observability, metrics, and system health

---

## 🎓 Learning Path

### **Step 1: Data Preparation**
Start with understanding how to work with data:
- **[Datasets Guide](datasets.md)** - Learn data loading, validation, and preparation
- **Key Topics**: Data formats, preprocessing, validation
- **Time**: 15-20 minutes

### **Step 2: Autonomous Detection**
Learn automated anomaly detection:
- **[Autonomous Mode Guide](autonomous-mode.md)** - Understand automated workflows
- **Key Topics**: Auto-selection, configuration, execution
- **Time**: 20-25 minutes

### **Step 3: System Monitoring**
Monitor your anomaly detection system:
- **[Monitoring Guide](monitoring.md)** - Set up observability and monitoring
- **Key Topics**: Metrics, alerts, dashboards
- **Time**: 25-30 minutes

---

## 🚀 Quick Start Workflow

### **1. Load Your Data**
```python
from pynomaly import Dataset
dataset = Dataset.from_csv("your_data.csv")
```
👉 **Learn more**: [Datasets Guide](datasets.md)

### **2. Run Autonomous Detection**
```python
from pynomaly import autonomous_detect
results = autonomous_detect(dataset)
```
👉 **Learn more**: [Autonomous Mode Guide](autonomous-mode.md)

### **3. Monitor Results**
```python
from pynomaly import MonitoringService
monitor = MonitoringService()
monitor.track_results(results)
```
👉 **Learn more**: [Monitoring Guide](monitoring.md)

---

## 📊 Core Concepts

### **🗂️ Datasets**
Understanding data management:
- **Data Loading** - Import from various sources
- **Data Validation** - Ensure data quality
- **Data Preprocessing** - Prepare for analysis
- **Data Formats** - Supported file types

### **🤖 Autonomous Mode**
Automated anomaly detection:
- **Auto-Selection** - Automatic algorithm selection
- **Configuration** - Automated parameter tuning
- **Execution** - Hands-free operation
- **Results** - Automated result interpretation

### **📈 Monitoring**
System observability:
- **Metrics Collection** - Performance and health metrics
- **Alerting** - Proactive issue detection
- **Dashboards** - Visual monitoring
- **Logging** - Audit and debugging

---

## 🎯 Use Cases

### **👩‍💼 Business Analyst**
Focus on business value:
1. **[Datasets](datasets.md)** - Load business data
2. **[Autonomous Mode](autonomous-mode.md)** - Automated analysis
3. **[Monitoring](monitoring.md)** - Track business metrics

### **👨‍💻 Data Scientist**
Technical analysis focus:
1. **[Datasets](datasets.md)** - Data engineering
2. **[Autonomous Mode](autonomous-mode.md)** - Model selection
3. **[Monitoring](monitoring.md)** - Model performance

### **🔧 Operations Engineer**
System reliability focus:
1. **[Monitoring](monitoring.md)** - System health
2. **[Datasets](datasets.md)** - Data pipeline health
3. **[Autonomous Mode](autonomous-mode.md)** - Automated operations

---

## 🔗 Related Documentation

### **Advanced Usage**
- **[Advanced Features](../advanced-features/)** - Advanced capabilities
- **[Performance Tuning](../advanced-features/performance-tuning.md)** - Optimization
- **[Explainability](../advanced-features/explainability.md)** - Understanding results

### **Technical Reference**
- **[Algorithm Reference](../../reference/algorithms/)** - Algorithm details
- **[API Reference](../../reference/api/)** - API documentation
- **[Configuration](../../reference/configuration/)** - Configuration options

### **Examples**
- **[Banking Examples](../../examples/banking/)** - Financial use cases
- **[Data Quality](../../examples/Data_Quality_Anomaly_Detection_Guide.md)** - Data validation
- **[Tutorials](../../examples/tutorials/)** - Step-by-step guides

---

## 🆘 Getting Help

### **Common Issues**
- **Data Loading Problems** - See [Datasets Guide](datasets.md#troubleshooting)
- **Autonomous Mode Issues** - See [Autonomous Mode Guide](autonomous-mode.md#troubleshooting)
- **Monitoring Problems** - See [Monitoring Guide](monitoring.md#troubleshooting)

### **Support Resources**
- **[Troubleshooting](../troubleshooting/)** - Common problems and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions

---

## 🚀 Next Steps

After mastering basic usage:

### **📈 Advanced Features**
- **[Performance Tuning](../advanced-features/performance-tuning.md)** - Optimize performance
- **[Explainability](../advanced-features/explainability.md)** - Understand results
- **[AutoML](../advanced-features/automl-and-intelligence.md)** - Advanced automation

### **🔧 Technical Integration**
- **[API Integration](../../developer-guides/api-integration/)** - Integrate with applications
- **[CLI Usage](../../developer-guides/api-integration/cli.md)** - Command line interface
- **[Python SDK](../../developer-guides/api-integration/python-sdk.md)** - SDK development

### **🚀 Production Deployment**
- **[Deployment](../../deployment/)** - Production deployment
- **[Security](../../security/)** - Security best practices
- **[Monitoring](monitoring.md)** - Production monitoring

---

**Last Updated**: 2025-01-09  
**Next Review**: 2025-02-09
🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 💡 [Examples](README.md)

---

# Examples & Use Cases

Real-world examples, tutorials, and industry-specific guides for implementing Pynomaly in production environments.

## 🏦 **Industry-Specific Guides**

### Banking & Financial Services
- **[Banking Anomaly Detection Guide](banking/Banking_Anomaly_Detection_Guide.md)** - Comprehensive guide for fraud detection, AML compliance, and risk management in banking
- **[Banking Examples](banking/)** - Financial fraud detection examples and templates

### Data Quality Management
- **[Data Quality Anomaly Detection Guide](Data_Quality_Anomaly_Detection_Guide.md)** - Enterprise guide for automated data quality assurance

## 📚 **Tutorials & Learning**

- **[Process Guide](tutorials/01-pynomaly-process-guide.md)** - Step-by-step implementation process
- **[Architecture Guide](tutorials/02-pynomaly-architecture-guide.md)** - Understanding system architecture
- **[Autonomous Mode Guide](tutorials/04-autonomous-mode-guide.md)** - Implementing autonomous detection
- **[Algorithm Selection Guide](tutorials/05-algorithm-rationale-selection-guide.md)** - Choosing the right algorithms
- **[Business User Testing Procedures](tutorials/06-business-user-monthly-testing-procedures.md)** - Monthly testing workflows
- **[Autonomous Classifier Selection](tutorials/09-autonomous-classifier-selection-guide.md)** - Advanced classifier selection
- **[Advanced Usage](tutorials/advanced-usage.md)** - Advanced features and techniques

---

## 🚀 **Quick Start Examples**

### Banking Fraud Detection
```python
from pynomaly import BankingAnomalyDetector

# Initialize banking-specific detector
detector = BankingAnomalyDetector(
    use_case="fraud_detection",
    regulatory_compliance="AML"
)

# Detect transaction anomalies
anomalies = detector.detect_transaction_anomalies(transactions)
print(f"Found {len(anomalies)} suspicious transactions")
```

### Data Quality Monitoring
```python
from pynomaly import DataQualityMonitor

# Monitor data quality in real-time
monitor = DataQualityMonitor()
quality_issues = monitor.assess_data_quality(dataset)
print(f"Data quality score: {quality_issues.overall_score}")
```

---

## 🔗 **Related Documentation**

- **[Getting Started](../getting-started/README.md)** - Installation and setup
- **[User Guides](../user-guides/README.md)** - Feature documentation
- **[Algorithm Reference](../reference/algorithms/README.md)** - Algorithm details
- **[Developer Guides](../developer-guides/README.md)** - Development documentation

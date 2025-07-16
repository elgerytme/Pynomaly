# ML Governance Guide

This comprehensive guide covers Pynomaly's ML governance framework, which ensures responsible AI practices, compliance, and robust model lifecycle management for enterprise deployments.

## 🎯 Overview

The ML Governance framework provides:
- **Model Lifecycle Management**: End-to-end tracking from development to retirement
- **Compliance & Audit Trails**: Meet regulatory requirements and internal policies
- **Approval Workflows**: Multi-stakeholder review and approval processes
- **Risk Assessment**: Automated compliance checking and risk evaluation
- **Deployment Controls**: Safe, controlled model deployments with rollback capabilities
- **Documentation Standards**: Model cards, data sheets, and comprehensive reporting

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML Governance Framework                      │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ Model Registry  │ Approval Engine │ Compliance Checker          │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ Audit & Reports │ Risk Assessment │ Deployment Manager          │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 🚀 Getting Started

### Basic Setup
```python
from pynomaly.infrastructure.ml_governance import MLGovernanceFramework
from pynomaly.application.services.ml_governance_service import MLGovernanceApplicationService

# Initialize governance framework
governance_framework = MLGovernanceFramework()
governance_service = MLGovernanceApplicationService(governance_framework)
```

### Configuration
```yaml
# governance_config.yaml
governance:
  compliance_level: "moderate"  # basic, moderate, strict
  required_approvers: 2
  approval_roles:
    - ml_engineer
    - data_scientist
    - product_owner
  auto_approval_conditions:
    performance_improvement: 0.05
    no_security_issues: true
    passes_bias_tests: true
```

## 📋 Model Lifecycle Stages

### 1. Development Stage
- Initial model development and experimentation
- Basic validation and compliance checks
- Documentation requirements minimal

### 2. Staging Stage
- Comprehensive testing and validation
- Performance benchmarking
- Security and bias testing
- Stakeholder review initiation

### 3. Production Stage
- Full approval required
- Comprehensive documentation
- Monitoring and alerting enabled
- Rollback procedures in place

### 4. Archived Stage
- Model retirement
- Data retention compliance
- Audit trail preservation

## 🔄 Governance Workflow

### Step 1: Model Onboarding
```python
import pandas as pd
from pynomaly.domain.entities.model import Model

# Create model
model = Model(
    name="fraud_detection_v2",
    algorithm="isolation_forest",
    parameters={"n_estimators": 100, "contamination": 0.05}
)

# Prepare model information
model_info = {
    "name": "Advanced Fraud Detection Model",
    "description": "ML model for real-time fraud detection in financial transactions",
    "intended_use": "Detect fraudulent credit card transactions in real-time",
    "limitations": "May have reduced accuracy for new fraud patterns not seen in training",
    "training_data": {
        "samples": 100000,
        "features": 25,
        "time_period": "2023-01-01 to 2024-01-01",
        "sources": ["transaction_logs", "user_profiles", "merchant_data"]
    },
    "performance_metrics": {
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90,
        "auc_roc": 0.94
    },
    "ethical_considerations": "Model trained on balanced dataset with bias testing completed",
    "caveats_and_recommendations": "Monitor for concept drift. Retrain monthly."
}

# Onboard model to governance
validation_data = pd.read_csv('validation_dataset.csv')
record = await governance_service.onboard_model(
    model=model,
    validation_data=validation_data,
    model_info=model_info,
    created_by="ml_engineer_alice"
)

print(f"Model onboarded with record ID: {record.record_id}")
print(f"Initial status: {record.status}")
```

### Step 2: Approval Workflow
```python
# Request approvals from stakeholders
approval_requests = await governance_service.request_model_approval(
    record_id=record.record_id,
    requested_by="ml_engineer_alice"
)

print(f"Created {len(approval_requests)} approval requests")

# Simulate approval process
approvers = [
    ("data_scientist_bob", "Reviewed model architecture and performance metrics"),
    ("ml_engineer_charlie", "Validated training pipeline and reproducibility"),
    ("product_owner_diana", "Confirmed business requirements and success criteria")
]

for approval_request in approval_requests:
    approver, comments = approvers[0]  # Simplified for demo
    
    approval = await governance_service.approve_model_deployment(
        record_id=record.record_id,
        approval_id=approval_request["approval_id"],
        approver=approver,
        comments=comments
    )
    
    print(f"✅ Approved by {approver}")
```

### Step 3: Deployment Management
```python
from pynomaly.infrastructure.ml_governance import ModelStage, DeploymentStrategy

# Deploy to staging
staging_result = await governance_service.deploy_model_to_stage(
    record_id=record.record_id,
    target_stage=ModelStage.STAGING,
    deployment_strategy=DeploymentStrategy.BLUE_GREEN
)

print(f"Staging deployment: {staging_result['status']}")

# Deploy to production with canary strategy
production_result = await governance_service.deploy_model_to_stage(
    record_id=record.record_id,
    target_stage=ModelStage.PRODUCTION,
    deployment_strategy=DeploymentStrategy.CANARY
)

print(f"Production deployment: {production_result['status']}")
```

## 📊 Compliance & Audit

### Compliance Checking
```python
# Run comprehensive compliance check
compliance_result = await governance_service.governance_framework.run_compliance_check(
    record_id=record.record_id
)

print(f"Compliance Score: {compliance_result['overall_score']:.2f}")
print(f"Checks Passed: {len(compliance_result['checks_passed'])}")
print(f"Checks Failed: {len(compliance_result['checks_failed'])}")

# Review failed checks
for check in compliance_result['checks_failed']:
    print(f"❌ {check['name']}: {check['description']}")
```

### Governance Audit
```python
# Generate comprehensive audit report
audit_report = await governance_service.run_governance_audit(record.record_id)

print(f"Governance Score: {audit_report['overall_governance_score']:.2f}")
print(f"Compliance Score: {audit_report['compliance_summary']['latest_compliance_score']:.2f}")
print(f"Approvals: {audit_report['approval_summary']['approved_count']}/{audit_report['approval_summary']['total_approvals']}")

if audit_report['audit_findings']:
    print("\n⚠️ Audit Findings:")
    for finding in audit_report['audit_findings']:
        print(f"  • {finding}")

if audit_report['recommendations']:
    print("\n💡 Recommendations:")
    for rec in audit_report['recommendations']:
        print(f"  • {rec}")
```

## 📈 Monitoring & Dashboards

### Governance Dashboard
```python
# Get governance dashboard data
dashboard = await governance_service.get_governance_dashboard()

print(f"Total Models: {dashboard['total_models']}")
print(f"Governance Health Score: {dashboard['governance_health_score']:.2f}")

print("\nModels by Stage:")
for stage, count in dashboard['models_by_stage'].items():
    print(f"  {stage}: {count}")

print("\nModels by Status:")
for status, count in dashboard['models_by_status'].items():
    print(f"  {status}: {count}")

print(f"\nCompliant Models: {dashboard['compliance_overview']['compliant_models']}")
print(f"Non-compliant Models: {dashboard['compliance_overview']['non_compliant_models']}")
```

### Bulk Operations
```python
# Run bulk compliance check across all production models
bulk_results = await governance_service.bulk_compliance_check(
    stage=ModelStage.PRODUCTION
)

print(f"Checked {bulk_results['total_models']} production models")
print(f"Compliant: {bulk_results['compliant_models']}")
print(f"Non-compliant: {bulk_results['non_compliant_models']}")

# Review non-compliant models
for result in bulk_results['results']:
    if result['status'] == 'non_compliant':
        print(f"⚠️ Model {result['record_id']}: Score {result['compliance_score']:.2f}")
```

## 🔧 Advanced Features

### Automated Model Promotion
```python
# Promote model through all stages automatically
promotion_results = await governance_service.promote_model_through_stages(
    record_id=record.record_id,
    auto_approve=True  # For demo; typically requires manual approval
)

print(f"Promoted through {len(promotion_results)} stages:")
for result in promotion_results:
    print(f"  {result['stage']}: {result['deployment']['status']}")
```

### Custom Governance Policies
```python
from pynomaly.infrastructure.ml_governance import GovernancePolicy, ComplianceLevel

# Create custom governance policy
custom_policy = GovernancePolicy(
    name="High-Risk Model Policy",
    description="Strict governance for high-risk ML models",
    compliance_level=ComplianceLevel.STRICT,
    required_approvers=3,
    approval_roles=["ml_engineer", "data_scientist", "product_owner", "security_officer"],
    auto_approve_conditions={
        "performance_improvement": 0.10,  # Require 10% improvement
        "no_security_issues": True,
        "passes_bias_tests": True,
        "regulatory_compliance": True
    },
    deployment_strategy=DeploymentStrategy.CANARY,
    rollback_conditions={
        "error_rate_spike": 0.01,  # Very low tolerance
        "performance_degradation": 0.02
    }
)

# Apply custom policy to model
governance_framework.policies["high_risk"] = custom_policy
```

### Model Cards and Documentation
```python
# Access model card
model_card = record.model_card
print(f"Model: {model_card.model_name}")
print(f"Version: {model_card.version}")
print(f"Description: {model_card.description}")
print(f"Intended Use: {model_card.intended_use}")
print(f"Limitations: {model_card.limitations}")
print(f"Performance: {model_card.performance_metrics}")
```

## 🛡️ Security & Compliance

### Regulatory Compliance
```yaml
# Configuration for regulatory compliance
compliance:
  regulations:
    gdpr_compliance: true        # GDPR (EU)
    ccpa_compliance: true        # CCPA (California)
    ai_act_compliance: true      # EU AI Act
    sox_compliance: false        # Sarbanes-Oxley (if financial)
    hipaa_compliance: false      # HIPAA (if healthcare)
    
  documentation:
    model_card_required: true
    data_sheet_required: true
    algorithm_explanation: true
    performance_benchmarks: true
    limitations_disclosure: true
    
  audit:
    log_all_changes: true
    change_approval_required: true
    periodic_audits: "quarterly"
```

### Bias and Fairness Testing
```python
# Fairness assessment configuration
fairness_config = {
    "protected_attributes": ["age", "gender", "race", "location"],
    "fairness_metrics": [
        "demographic_parity",
        "equalized_odds",
        "individual_fairness"
    ],
    "bias_threshold": 0.1  # 10% disparity threshold
}

# This would be integrated into compliance checking
```

## 📊 Reporting & Analytics

### Generate Comprehensive Report
```python
# Generate detailed governance report
report = await governance_framework.generate_governance_report(record.record_id)

print(f"""
Governance Report for Model {report['model_id']}
==============================================
Current Stage: {report['current_stage']}
Governance Status: {report['governance_status']}
Created: {report['created_at']}
Updated: {report['updated_at']}

Compliance Summary:
- Total Checks: {report['compliance_summary']['total_checks']}
- Passed: {report['compliance_summary']['passed_checks']}
- Latest Score: {report['compliance_summary']['latest_compliance_score']:.2f}

Approval Summary:
- Total Approvals: {report['approval_summary']['total_approvals']}
- Approved: {report['approval_summary']['approved_count']}
- Pending: {report['approval_summary']['pending_count']}

Deployment Summary:
- Total Deployments: {report['deployment_summary']['deployment_count']}
- Latest: {report['deployment_summary']['latest_deployment']['status'] if report['deployment_summary']['latest_deployment'] else 'None'}

Documentation:
- Model Card: {'✅' if report['documentation_status']['model_card_exists'] else '❌'}
- Data Sheet: {'✅' if report['documentation_status']['data_sheet_exists'] else '❌'}
""")
```

## 🚀 Best Practices

### 1. Early Integration
```python
# Integrate governance from the start of ML development
def train_model_with_governance():
    # 1. Register model early in development
    record = governance_service.register_model(model, policy="default")
    
    # 2. Document model card during development
    create_model_card(record.record_id, model_info)
    
    # 3. Run validation and compliance checks
    validate_model(record.record_id, validation_data)
    
    # 4. Request approvals before deployment
    request_approvals(record.record_id)
    
    return record
```

### 2. Continuous Monitoring
```python
# Set up continuous governance monitoring
async def governance_monitoring_loop():
    while True:
        # Check all production models
        results = await governance_service.bulk_compliance_check(
            stage=ModelStage.PRODUCTION
        )
        
        # Alert on compliance issues
        for result in results['results']:
            if result['compliance_score'] < 0.8:
                send_alert(f"Model {result['record_id']} compliance issue")
        
        # Wait before next check
        await asyncio.sleep(3600)  # Check hourly
```

### 3. Documentation Standards
```python
# Comprehensive model documentation template
MODEL_CARD_TEMPLATE = {
    "model_details": {
        "model_name": "",
        "model_version": "",
        "model_type": "",
        "algorithm_family": "",
        "training_framework": "",
        "model_size": "",
        "inference_time": ""
    },
    "intended_use": {
        "primary_uses": [],
        "primary_users": [],
        "out_of_scope_uses": []
    },
    "factors": {
        "relevant_factors": [],
        "evaluation_factors": []
    },
    "metrics": {
        "model_performance_measures": {},
        "decision_thresholds": {},
        "variation_approaches": []
    },
    "training_data": {
        "datasets": [],
        "motivation": "",
        "preprocessing": []
    },
    "evaluation_data": {
        "datasets": [],
        "motivation": "",
        "preprocessing": []
    },
    "ethical_considerations": {
        "sensitive_data": [],
        "human_life": "",
        "mitigations": [],
        "risks_and_harms": []
    },
    "caveats_and_recommendations": {
        "additional_information": []
    }
}
```

## 🔧 Integration Examples

### CI/CD Integration
```yaml
# .github/workflows/ml-governance.yml
name: ML Governance Pipeline
on:
  push:
    paths: ['models/**']

jobs:
  governance-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
          
      - name: Install Pynomaly
        run: pip install pynomaly[governance]
        
      - name: Run Governance Validation
        run: |
          python scripts/validate_model_governance.py \
            --model-path models/latest/ \
            --policy production \
            --strict
            
      - name: Generate Governance Report
        run: |
          python scripts/generate_governance_report.py \
            --output governance-report.json
            
      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: governance-report
          path: governance-report.json
```

### Monitoring Integration
```python
# Prometheus metrics for governance
from prometheus_client import Counter, Histogram, Gauge

# Governance metrics
model_registrations = Counter('pynomaly_model_registrations_total', 'Total model registrations')
approval_time = Histogram('pynomaly_approval_time_seconds', 'Time to complete approval')
compliance_score = Gauge('pynomaly_compliance_score', 'Current compliance score', ['model_id'])
governance_health = Gauge('pynomaly_governance_health', 'Overall governance health score')

# Update metrics
async def update_governance_metrics():
    dashboard = await governance_service.get_governance_dashboard()
    governance_health.set(dashboard['governance_health_score'])
    
    for model_id, score in get_compliance_scores().items():
        compliance_score.labels(model_id=model_id).set(score)
```

## 🎯 Next Steps

### Immediate Actions
1. **Set up governance framework** in your development environment
2. **Define governance policies** for your organization
3. **Create model card templates** for your use cases
4. **Integrate with CI/CD** for automated compliance checking

### Advanced Implementation
1. **Custom compliance rules** for your industry requirements
2. **Integration with model registries** like MLflow or Kubeflow
3. **Automated monitoring** and alerting systems
4. **Regulatory compliance reporting** automation

### Learn More
- [Production Deployment](production-deployment.md): Deploy governed models safely
- [Monitoring Guide](monitoring.md): Set up comprehensive monitoring
- [API Reference](reference/api-reference.md): Complete API documentation
- [Tutorials](tutorials/): Domain-specific governance examples

The ML Governance framework ensures your models meet the highest standards of reliability, compliance, and ethical AI practices. Start with basic governance and gradually implement advanced features as your organization's ML maturity grows.
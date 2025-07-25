#!/usr/bin/env python3
"""
Compliance Automation Test Suite

Standalone test runner for compliance automation framework.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from compliance_automation import (
    ComplianceAutomationEngine,
    MLGovernanceFramework,
    ComplianceFramework,
    AutomationTrigger,
    RemediationAction,
    GovernanceRisk
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_compliance_automation():
    """Test the compliance automation framework."""
    
    print("🔧 Testing MLOps Compliance Automation Framework")
    print("=" * 60)
    
    # Initialize framework
    governance = MLGovernanceFramework()
    automation = ComplianceAutomationEngine(governance)
    
    # Add test compliance rules
    print("\n📋 Adding compliance rules...")
    
    # GDPR data access rule
    gdpr_rule = await automation.add_compliance_rule(
        name="GDPR Data Access Monitoring",
        description="Monitor access to personal data for GDPR compliance",
        compliance_frameworks=[ComplianceFramework.GDPR],
        trigger=AutomationTrigger.CONTINUOUS,
        conditions={
            "data_type": "personal_data",
            "access_count": {"operator": "gt", "value": 10}
        },
        remediation_actions=[RemediationAction.ALERT, RemediationAction.AUDIT_LOG],
        auto_execute=True,
        requires_approval=False
    )
    print(f"✅ Added GDPR rule: {gdpr_rule}")
    
    # Model deployment security rule
    security_rule = await automation.add_compliance_rule(
        name="Model Deployment Security Check",
        description="Ensure model deployments meet security standards",
        compliance_frameworks=[ComplianceFramework.ISO_27001],
        trigger=AutomationTrigger.EVENT_DRIVEN,
        conditions={
            "deployment_environment": "production",
            "security_scan_passed": False
        },
        remediation_actions=[RemediationAction.QUARANTINE_MODEL, RemediationAction.ESCALATE],
        auto_execute=False,
        requires_approval=True
    )
    print(f"✅ Added security rule: {security_rule}")
    
    # Start automation
    print("\n🚀 Starting compliance automation...")
    await automation.start_automation()
    
    # Simulate compliance events
    print("\n📊 Simulating compliance events...")
    
    # Simulate data access events
    for i in range(15):
        await governance.log_audit_event(
            event_type="data_access",
            user_id=f"user_{i % 3}",
            action="read_personal_data",
            resource="customer_database",
            details={
                "data_type": "personal_data",
                "records_accessed": 100 + i * 10
            }
        )
    
    # Simulate model deployment event
    await governance.log_audit_event(
        event_type="model_deployment",
        user_id="deployer_001",
        action="deploy_model",
        resource="production_cluster",
        details={
            "deployment_environment": "production",
            "security_scan_passed": False,
            "model_id": "fraud_detection_v2"
        }
    )
    
    # Test rule evaluation
    print("\n🔍 Evaluating compliance rules...")
    
    context = {
        "data_type": "personal_data",
        "access_count": 12,
        "deployment_environment": "production",
        "security_scan_passed": False,
        "model_id": "fraud_detection_v2"
    }
    
    violations = await automation.evaluate_compliance_rules(context)
    
    print(f"📋 Found {len(violations)} compliance violations:")
    for violation in violations:
        print(f"  ⚠️  {violation.violation_type}: {violation.description}")
        print(f"      Severity: {violation.severity.value}")
        print(f"      Actions taken: {violation.remediation_actions_taken}")
    
    # Wait for background processing
    print("\n⏳ Processing automation tasks...")
    await asyncio.sleep(2)
    
    # Check metrics
    print("\n📈 Compliance Metrics:")
    registry = automation.registry
    for collector in registry._collector_to_names:
        for metric in collector.collect():
            if hasattr(metric, 'samples') and metric.samples:
                for sample in metric.samples:
                    if sample.value > 0:
                        print(f"  📊 {sample.name}: {sample.value}")
    
    # Stop automation
    print("\n🛑 Stopping compliance automation...")
    await automation.stop_automation()
    
    print("\n✅ Compliance automation test completed successfully!")
    return True

async def demo_enterprise_features():
    """Demonstrate enterprise compliance features."""
    
    print("\n🏢 Enterprise Compliance Features Demo")
    print("=" * 50)
    
    governance = MLGovernanceFramework()
    automation = ComplianceAutomationEngine(governance, config={
        "enterprise_mode": True,
        "real_time_monitoring": True,
        "automated_reporting": True
    })
    
    # Add sophisticated compliance rules
    print("\n📋 Adding enterprise compliance rules...")
    
    # Multi-framework compliance rule
    multi_compliance_rule = await automation.add_compliance_rule(
        name="Multi-Framework Data Governance",
        description="Comprehensive data governance across GDPR, HIPAA, and CCPA",
        compliance_frameworks=[
            ComplianceFramework.GDPR,
            ComplianceFramework.HIPAA,
            ComplianceFramework.CCPA
        ],
        trigger=AutomationTrigger.THRESHOLD_BASED,
        conditions={
            "sensitive_data_access": {"operator": "threshold"},
            "data_retention_period": {"operator": "gt", "value": 365},
            "consent_status": "expired"
        },
        threshold=5.0,
        remediation_actions=[
            RemediationAction.AUTOMATIC_FIX,
            RemediationAction.AUDIT_LOG,
            RemediationAction.REQUIRE_APPROVAL
        ],
        auto_execute=True,
        requires_approval=True,
        priority=9
    )
    print(f"✅ Added multi-framework rule: {multi_compliance_rule}")
    
    # Anomaly-based compliance rule
    anomaly_rule = await automation.add_compliance_rule(
        name="Anomalous Model Behavior Detection",
        description="Detect and respond to anomalous ML model behavior",
        compliance_frameworks=[ComplianceFramework.ISO_27001],
        trigger=AutomationTrigger.ANOMALY_DETECTED,
        conditions={
            "model_accuracy_drop": {"operator": "gt", "value": 0.1},
            "prediction_drift": {"operator": "gt", "value": 0.15},
            "resource_usage_spike": {"operator": "gt", "value": 2.0}
        },
        remediation_actions=[
            RemediationAction.QUARANTINE_MODEL,
            RemediationAction.ESCALATE,
            RemediationAction.AUDIT_LOG
        ],
        auto_execute=False,
        requires_approval=True,
        priority=8
    )
    print(f"✅ Added anomaly detection rule: {anomaly_rule}")
    
    print("\n🎯 Enterprise features configured successfully!")
    return True

async def main():
    """Main test runner."""
    
    print("🧪 MLOps Compliance Automation Test Suite")
    print("🔒 Testing enterprise-grade compliance automation")
    print("⚡ Real-time monitoring and automated remediation")
    print()
    
    try:
        # Run basic compliance tests
        await test_compliance_automation()
        
        # Run enterprise feature demo
        await demo_enterprise_features()
        
        print("\n🎉 All tests passed! Compliance automation framework is operational.")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        print(f"\n❌ Test failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
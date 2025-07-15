"""Simple test to verify data quality functionality works."""

import pandas as pd
from uuid import uuid4

# Test imports
from application.services.validation_engine import ValidationEngine
from application.services.quality_monitoring_service import QualityMonitoringService, MonitoringConfiguration
from domain.entities.quality_rule import (
    QualityRule, RuleType, LogicType, ValidationLogic, QualityThreshold,
    Severity, UserId, DatasetId, RuleId
)

def test_basic_functionality():
    """Test basic validation functionality."""
    print("🧪 Testing Data Quality Package...")
    
    # Create sample data
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', None, 'Bob', ''],
        'email': ['john@test.com', 'jane@test.com', 'invalid', 'bob@test.com', None]
    })
    
    print(f"📊 Sample dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Create validation rule
    rule = QualityRule(
        rule_id=RuleId(value=uuid4()),
        rule_name="Name Completeness",
        rule_type=RuleType.COMPLETENESS,
        target_columns=['name'],
        validation_logic=ValidationLogic(
            logic_type=LogicType.PYTHON,
            expression="df['name'].notna() & (df['name'].str.strip() != '')",
            parameters={},
            error_message_template="Missing name"
        ),
        thresholds=QualityThreshold(
            pass_rate_threshold=0.8,
            warning_threshold=0.7,
            critical_threshold=0.5
        ),
        severity=Severity.HIGH,
        created_by=UserId(value=uuid4()),
        is_enabled=True
    )
    
    print(f"📋 Created rule: {rule.rule_name}")
    
    # Test validation engine
    engine = ValidationEngine()
    results = engine.validate_dataset([rule], df, DatasetId(value=uuid4()))
    
    print(f"✅ Validation completed:")
    for result in results:
        print(f"   - Rule: {result.rule_id}")
        print(f"   - Status: {result.status.value}")
        print(f"   - Pass rate: {result.pass_rate:.1%}")
        print(f"   - Records passed: {result.records_passed}")
        print(f"   - Records failed: {result.records_failed}")
    
    # Test monitoring service
    config = MonitoringConfiguration()
    monitoring = QualityMonitoringService(config)
    
    # Add dataset to monitoring
    dataset_id = uuid4()
    monitoring.add_dataset_monitoring(
        dataset_id=dataset_id,
        rules=[rule],
        data_source_config={'type': 'dataframe'}
    )
    
    print(f"📡 Added dataset to monitoring: {dataset_id}")
    
    # Get dashboard data
    dashboard = monitoring.get_quality_dashboard_data()
    print(f"📊 Dashboard - Monitored datasets: {dashboard['monitored_datasets']}")
    print(f"📊 Dashboard - Active rules: {dashboard['total_active_rules']}")
    
    print("✅ All tests passed successfully!")

if __name__ == "__main__":
    test_basic_functionality()
#!/usr/bin/env python3
"""Test the new domain entities (Model, Experiment, Pipeline, Alert)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta
from uuid import uuid4

def test_new_entities():
    """Test all new domain entities."""
    print("🧪 Testing New Domain Entities\n")
    
    try:
        # Test Model entity
        print("1. Testing Model entity...")
        from pynomaly.domain.entities import Model, ModelType, ModelStage
        
        model = Model(
            name="fraud_detection_model",
            description="ML model for detecting fraudulent transactions",
            model_type=ModelType.SUPERVISED,
            algorithm_family="ensemble",
            created_by="data_team",
            team="fraud_prevention",
            use_cases=["real_time_fraud_detection", "batch_transaction_scoring"]
        )
        
        print(f"✓ Model created: {model.name}")
        print(f"  Type: {model.model_type.value}")
        print(f"  Stage: {model.stage.value}")
        print(f"  In production: {model.is_in_production}")
        
        # Test model operations
        model.add_tag("production_ready")
        model.update_stage(ModelStage.STAGING)
        version_id = uuid4()
        model.promote_to_production(version_id, "deploy_team")
        
        print(f"  After promotion: stage={model.stage.value}, in_production={model.is_in_production}")
        
        # Test deployment readiness check
        can_deploy, issues = model.can_deploy()
        print(f"  Can deploy: {can_deploy}")
        if issues:
            print(f"  Issues: {issues}")
        
        # Test Experiment entity
        print("\n2. Testing Experiment entity...")
        from pynomaly.domain.entities import Experiment, ExperimentRun, ExperimentType, ExperimentStatus
        
        experiment = Experiment(
            name="algorithm_comparison_v1",
            description="Compare IsolationForest vs LOF vs OneClassSVM performance",
            experiment_type=ExperimentType.ALGORITHM_COMPARISON,
            objective="Find best algorithm for anomaly detection on customer transaction data",
            created_by="ml_engineer"
        )
        
        print(f"✓ Experiment created: {experiment.name}")
        print(f"  Type: {experiment.experiment_type.value}")
        print(f"  Status: {experiment.status.value}")
        
        # Add optimization metrics
        experiment.add_optimization_metric("f1_score", "maximize")
        experiment.add_optimization_metric("precision", "maximize")
        experiment.add_optimization_metric("training_time", "minimize")
        
        # Create and add experiment runs
        run1 = ExperimentRun(
            name="isolation_forest_run",
            detector_id=uuid4(),
            dataset_id=uuid4(),
            parameters={"contamination": 0.1, "n_estimators": 100}
        )
        run1.start()
        run1.complete({"f1_score": 0.85, "precision": 0.80, "training_time": 5.2})
        
        run2 = ExperimentRun(
            name="lof_run",
            detector_id=uuid4(),
            dataset_id=uuid4(),
            parameters={"contamination": 0.1, "n_neighbors": 20}
        )
        run2.start()
        run2.complete({"f1_score": 0.78, "precision": 0.85, "training_time": 8.1})
        
        experiment.add_run(run1)
        experiment.add_run(run2)
        experiment.complete_experiment()
        
        print(f"  Runs: {experiment.run_count}")
        print(f"  Success rate: {experiment.success_rate:.1%}")
        print(f"  Best run: {experiment.best_run_id}")
        
        # Test metrics summary
        f1_summary = experiment.get_metric_summary("f1_score")
        print(f"  F1 score summary: mean={f1_summary.get('mean', 0):.3f}, max={f1_summary.get('max', 0):.3f}")
        
        # Test Pipeline entity
        print("\n3. Testing Pipeline entity...")
        from pynomaly.domain.entities import Pipeline, PipelineStep, PipelineType, StepType, PipelineStatus
        
        pipeline = Pipeline(
            name="daily_anomaly_detection",
            description="Daily batch anomaly detection pipeline",
            pipeline_type=PipelineType.BATCH_PROCESSING,
            created_by="ml_ops_team",
            environment="production"
        )
        
        print(f"✓ Pipeline created: {pipeline.name}")
        print(f"  Type: {pipeline.pipeline_type.value}")
        print(f"  Status: {pipeline.status.value}")
        
        # Add pipeline steps
        step1 = PipelineStep(
            name="load_data",
            step_type=StepType.DATA_LOADING,
            description="Load transaction data from database",
            order=1,
            configuration={"table": "transactions", "batch_size": 10000}
        )
        
        step2 = PipelineStep(
            name="preprocess_data",
            step_type=StepType.DATA_PREPROCESSING,
            description="Clean and normalize transaction data",
            order=2,
            dependencies=[step1.id],
            configuration={"scaling": "standard", "handle_missing": "drop"}
        )
        
        step3 = PipelineStep(
            name="detect_anomalies",
            step_type=StepType.PREDICTION,
            description="Run anomaly detection on processed data",
            order=3,
            dependencies=[step2.id],
            configuration={"model_id": str(model.id), "threshold": 0.8}
        )
        
        pipeline.add_step(step1)
        pipeline.add_step(step2)
        pipeline.add_step(step3)
        
        print(f"  Steps: {pipeline.step_count}")
        print(f"  Enabled steps: {len(pipeline.enabled_steps)}")
        
        # Test pipeline validation and activation
        is_valid, validation_issues = pipeline.validate_pipeline()
        print(f"  Valid: {is_valid}")
        if validation_issues:
            print(f"  Validation issues: {validation_issues}")
        
        if is_valid:
            pipeline.activate()
            print(f"  Activated: status={pipeline.status.value}")
        
        # Set schedule
        pipeline.set_schedule("0 2 * * *")  # Daily at 2 AM
        print(f"  Scheduled: {pipeline.is_scheduled}")
        
        # Test Alert entity
        print("\n4. Testing Alert entity...")
        from pynomaly.domain.entities import Alert, AlertCondition, AlertType, AlertSeverity, AlertStatus
        
        # Create alert condition
        condition = AlertCondition(
            metric_name="anomaly_rate",
            operator="gt",
            threshold=0.05,
            time_window_minutes=10,
            consecutive_breaches=2,
            description="Anomaly rate exceeds 5% for 10 minutes"
        )
        
        alert = Alert(
            name="high_anomaly_rate_alert",
            description="Alert when anomaly detection rate is unusually high",
            alert_type=AlertType.ANOMALY_DETECTION,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="monitoring_team",
            source="anomaly_detection_service"
        )
        
        print(f"✓ Alert created: {alert.name}")
        print(f"  Type: {alert.alert_type.value}")
        print(f"  Severity: {alert.severity.value}")
        print(f"  Status: {alert.status.value}")
        print(f"  Condition: {alert.condition.get_description()}")
        
        # Test alert lifecycle
        alert.trigger("monitoring_system", {"current_value": 0.07, "threshold": 0.05})
        print(f"  After trigger: status={alert.status.value}")
        
        alert.acknowledge("on_call_engineer", "Investigating anomaly spike")
        print(f"  After acknowledgment: status={alert.status.value}")
        
        alert.resolve("on_call_engineer", "False positive due to data processing delay")
        print(f"  After resolution: status={alert.status.value}")
        
        # Test alert metrics
        print(f"  Response time: {alert.response_time_minutes:.1f} minutes")
        print(f"  Resolution time: {alert.resolution_time_minutes:.1f} minutes")
        
        # Test condition evaluation
        test_values = [0.03, 0.06, 0.04, 0.08]
        print(f"  Condition evaluation: {[condition.evaluate(v) for v in test_values]}")
        
        # Test comprehensive info methods
        print("\n5. Testing entity info methods...")
        
        model_info = model.get_info()
        print(f"✓ Model info keys: {len(model_info)} fields")
        
        experiment_info = experiment.get_info()
        print(f"✓ Experiment info keys: {len(experiment_info)} fields")
        
        pipeline_info = pipeline.get_info()
        print(f"✓ Pipeline info keys: {len(pipeline_info)} fields")
        
        alert_info = alert.get_info()
        print(f"✓ Alert info keys: {len(alert_info)} fields")
        
        # Test timeline for alert
        timeline = alert.get_timeline()
        print(f"✓ Alert timeline: {len(timeline)} events")
        
        print("\n✅ All new entities working correctly!")
        print("\n📋 Entity Summary:")
        print(f"   ✓ Model: {model.name} ({model.stage.value})")
        print(f"   ✓ Experiment: {experiment.name} ({experiment.run_count} runs)")
        print(f"   ✓ Pipeline: {pipeline.name} ({pipeline.step_count} steps)")
        print(f"   ✓ Alert: {alert.name} ({alert.severity.value})")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Entity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_entities()
    print(f"\nResult: {'🎉 SUCCESS' if success else '💥 FAILED'}")
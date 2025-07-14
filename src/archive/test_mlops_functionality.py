#!/usr/bin/env python3
"""
Test MLOps functionality without external dependencies
"""

import asyncio
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_model_deployment():
    """Test model deployment functionality"""
    print("🔍 Testing model deployment...")
    
    try:
        from pynomaly.mlops.model_deployment import deployment_manager, DeploymentEnvironment
        
        # Create a test deployment
        deployment_id = await deployment_manager.create_deployment(
            model_id="test_model",
            model_version="1.0.0",
            environment=DeploymentEnvironment.DEVELOPMENT
        )
        
        print(f"✅ Created deployment: {deployment_id}")
        
        # List deployments
        deployments = await deployment_manager.list_deployments()
        print(f"✅ Found {len(deployments)} deployments")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment test failed: {e}")
        return False

async def test_automl_pipeline():
    """Test AutoML pipeline"""
    print("\n🔍 Testing AutoML pipeline...")
    
    try:
        from pynomaly.application.services.automl_pipeline_orchestrator import (
            AutoMLPipelineOrchestrator, PipelineConfig, PipelineMode
        )
        
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(100, 5), 
            columns=[f"feature_{i}" for i in range(5)]
        )
        y = pd.Series(np.random.choice([0, 1], size=100))
        
        # Configure pipeline
        config = PipelineConfig(
            mode=PipelineMode.FAST,
            optimization_time_budget_minutes=1,
            max_models_to_evaluate=2
        )
        
        orchestrator = AutoMLPipelineOrchestrator(config)
        print("✅ AutoML orchestrator created")
        
        # Test data validation
        validation_result = await orchestrator._validate_data(X, y)
        print(f"✅ Data validation completed: {validation_result['valid']}")
        
        return True
        
    except Exception as e:
        print(f"❌ AutoML pipeline test failed: {e}")
        return False

async def test_monitoring():
    """Test monitoring functionality"""
    print("\n🔍 Testing monitoring...")
    
    try:
        from pynomaly.mlops.monitoring import mlops_monitor
        
        # Test metrics collection
        metrics_count = len(mlops_monitor.metrics_collector.metrics_storage)
        print(f"✅ Found {metrics_count} metrics in storage")
        
        # Test alert management
        alerts = mlops_monitor.alert_manager.get_active_alerts()
        print(f"✅ Found {len(alerts)} active alerts")
        
        # Test dashboard management
        dashboards = mlops_monitor.dashboard_manager.list_dashboards()
        print(f"✅ Found {len(dashboards)} dashboards")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        return False

async def test_data_quality():
    """Test data quality monitoring"""
    print("\n🔍 Testing data quality...")
    
    try:
        from pynomaly.infrastructure.data_quality.data_validation import data_pipeline_monitor
        
        # Create test data
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Test data quality validation
        report = data_pipeline_monitor.validate_and_monitor(data, "test_dataset")
        print(f"✅ Data quality report created: score {report.data_quality_metrics.overall_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False

async def run_functionality_tests():
    """Run all functionality tests"""
    print("🚀 Testing MLOps functionality...\n")
    
    tests = [
        test_model_deployment,
        test_automl_pipeline,
        test_monitoring,
        test_data_quality,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
    
    print(f"\n📊 Functionality Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All functionality tests passed! MLOps platform is operational.")
        return True
    else:
        print("⚠️  Some functionality tests failed.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_functionality_tests())
    print(f"\n✅ MLOps deployment validation {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
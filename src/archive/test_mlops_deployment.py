#!/usr/bin/env python3
"""
Test script to validate MLOps deployment components
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_mlops_imports():
    """Test that all MLOps modules can be imported"""
    print("🔍 Testing MLOps module imports...")
    
    try:
        # Test core MLOps imports
        from pynomaly.mlops.model_deployment import deployment_manager
        print("✅ Model deployment module imported successfully")
        
        from pynomaly.mlops.monitoring import mlops_monitor
        print("✅ Monitoring module imported successfully")
        
        from pynomaly.mlops.model_serving import ModelServingEngine
        print("✅ Model serving module imported successfully")
        
        from pynomaly.mlops.model_registry import ModelRegistry
        print("✅ Model registry module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

async def test_main_server():
    """Test that main MLOps server can be initialized"""
    print("\n🔍 Testing main MLOps server initialization...")
    
    try:
        from pynomaly.mlops.main_server import app
        print("✅ Main server app created successfully")
        
        # Test that we can access the app routes
        routes = [route.path for route in app.routes]
        print(f"✅ Found {len(routes)} routes: {routes[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Server initialization error: {e}")
        traceback.print_exc()
        return False

async def test_deployment_manager():
    """Test deployment manager functionality"""
    print("\n🔍 Testing deployment manager...")
    
    try:
        from pynomaly.mlops.model_deployment import deployment_manager
        
        # Test listing deployments
        deployments = await deployment_manager.list_deployments()
        print(f"✅ Successfully listed {len(deployments)} deployments")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment manager error: {e}")
        traceback.print_exc()
        return False

async def test_automl_orchestrator():
    """Test AutoML pipeline orchestrator"""
    print("\n🔍 Testing AutoML pipeline orchestrator...")
    
    try:
        from pynomaly.application.services.automl_pipeline_orchestrator import AutoMLPipelineOrchestrator, PipelineConfig
        
        config = PipelineConfig()
        orchestrator = AutoMLPipelineOrchestrator(config)
        print("✅ AutoML orchestrator created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ AutoML orchestrator error: {e}")
        traceback.print_exc()
        return False

async def test_data_validation():
    """Test data validation components"""
    print("\n🔍 Testing data validation...")
    
    try:
        from pynomaly.infrastructure.data_quality.data_validation import data_pipeline_monitor
        print("✅ Data pipeline monitor imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation error: {e}")
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run all deployment tests"""
    print("🚀 Starting MLOps deployment validation tests...\n")
    
    tests = [
        test_mlops_imports,
        test_main_server,
        test_deployment_manager,
        test_automl_orchestrator,
        test_data_validation,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! MLOps deployment is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
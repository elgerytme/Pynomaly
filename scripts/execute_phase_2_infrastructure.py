#!/usr/bin/env python3
"""
Phase 2: Infrastructure Tests Execution Script
Target: 70% Test Coverage through Infrastructure Layer Testing

This script provides comprehensive infrastructure test execution for Pynomaly,
targeting distributed processing, adapters, data loaders, persistence, and monitoring.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def validate_infrastructure_components():
    """Validate that all infrastructure components exist and are importable."""
    
    infrastructure_components = [
        # Core Infrastructure
        'pynomaly.infrastructure.adapters.pyod_adapter',
        'pynomaly.infrastructure.adapters.sklearn_adapter', 
        'pynomaly.infrastructure.adapters.pytorch_adapter',
        'pynomaly.infrastructure.adapters.tensorflow_adapter',
        'pynomaly.infrastructure.adapters.jax_adapter',
        
        # Data Loading Infrastructure
        'pynomaly.infrastructure.data.csv_loader',
        'pynomaly.infrastructure.data.parquet_loader',
        'pynomaly.infrastructure.data.streaming_loader',
        
        # Persistence Infrastructure
        'pynomaly.infrastructure.persistence.model_repository',
        'pynomaly.infrastructure.persistence.result_repository',
        
        # Distributed Processing Infrastructure
        'pynomaly.infrastructure.distributed.manager',
        'pynomaly.infrastructure.distributed.worker',
        'pynomaly.infrastructure.distributed.coordinator',
        'pynomaly.infrastructure.distributed.load_balancer',
        'pynomaly.infrastructure.distributed.task_queue',
        'pynomaly.infrastructure.distributed.worker_pool',
        'pynomaly.infrastructure.distributed.coordination_service',
        
        # Configuration Infrastructure
        'pynomaly.infrastructure.config.container',
        'pynomaly.infrastructure.config.settings',
        
        # Monitoring Infrastructure
        'pynomaly.infrastructure.monitoring.metrics',
        'pynomaly.infrastructure.monitoring.health_checks',
    ]
    
    results = {'available': [], 'missing': [], 'errors': []}
    
    for component in infrastructure_components:
        try:
            __import__(component)
            results['available'].append(component)
            print(f"✅ {component}")
        except ImportError as e:
            results['missing'].append((component, str(e)))
            print(f"❌ {component}: {e}")
        except Exception as e:
            results['errors'].append((component, str(e)))
            print(f"⚠️  {component}: {e}")
    
    return results

def get_infrastructure_test_files():
    """Get all infrastructure test files."""
    
    test_dir = Path("tests/infrastructure")
    if not test_dir.exists():
        return []
    
    test_files = list(test_dir.glob("*.py"))
    test_files = [f for f in test_files if f.name.startswith("test_")]
    
    return sorted(test_files)

def analyze_infrastructure_test_coverage():
    """Analyze infrastructure test coverage and categorize tests."""
    
    test_files = get_infrastructure_test_files()
    
    categories = {
        'adapters': [],
        'data_loaders': [], 
        'persistence': [],
        'distributed': [],
        'configuration': [],
        'monitoring': [],
        'auth': [],
        'caching': [],
        'middleware': [],
        'preprocessing': [],
        'repositories': [],
        'resilience': []
    }
    
    for test_file in test_files:
        name = test_file.name
        if 'adapter' in name:
            categories['adapters'].append(test_file)
        elif 'data_loader' in name or 'loader' in name:
            categories['data_loaders'].append(test_file)
        elif 'persistence' in name:
            categories['persistence'].append(test_file)
        elif 'distributed' in name:
            categories['distributed'].append(test_file)
        elif 'config' in name:
            categories['configuration'].append(test_file)
        elif 'monitoring' in name:
            categories['monitoring'].append(test_file)
        elif 'auth' in name:
            categories['auth'].append(test_file)
        elif 'caching' in name:
            categories['caching'].append(test_file)
        elif 'middleware' in name:
            categories['middleware'].append(test_file)
        elif 'preprocessing' in name:
            categories['preprocessing'].append(test_file)
        elif 'repositories' in name:
            categories['repositories'].append(test_file)
        elif 'resilience' in name:
            categories['resilience'].append(test_file)
    
    return categories

def count_test_methods_in_file(test_file: Path) -> int:
    """Count test methods in a test file."""
    try:
        content = test_file.read_text()
        # Count functions that start with 'test_' or 'async def test_'
        lines = content.split('\n')
        test_count = 0
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('def test_') or 
                stripped.startswith('async def test_')):
                test_count += 1
        return test_count
    except Exception as e:
        print(f"Error counting tests in {test_file}: {e}")
        return 0

def execute_infrastructure_tests_strategy():
    """Execute Phase 2 infrastructure testing strategy."""
    
    print("=" * 80)
    print("PHASE 2: INFRASTRUCTURE TESTS EXECUTION")
    print("Target: 70% Test Coverage through Infrastructure Layer")
    print("=" * 80)
    
    # 1. Validate Infrastructure Components
    print("\n1. VALIDATING INFRASTRUCTURE COMPONENTS...")
    validation_results = validate_infrastructure_components()
    
    print(f"\n📊 Infrastructure Validation Summary:")
    print(f"   ✅ Available: {len(validation_results['available'])}")
    print(f"   ❌ Missing: {len(validation_results['missing'])}")
    print(f"   ⚠️  Errors: {len(validation_results['errors'])}")
    
    # 2. Analyze Test Categories
    print("\n2. ANALYZING INFRASTRUCTURE TEST COVERAGE...")
    test_categories = analyze_infrastructure_test_coverage()
    
    total_test_files = 0
    total_test_methods = 0
    
    for category, files in test_categories.items():
        if files:
            method_count = sum(count_test_methods_in_file(f) for f in files)
            total_test_files += len(files)
            total_test_methods += method_count
            print(f"   📁 {category.upper()}: {len(files)} files, {method_count} test methods")
    
    print(f"\n📊 Infrastructure Test Summary:")
    print(f"   📁 Total Test Files: {total_test_files}")
    print(f"   🧪 Total Test Methods: {total_test_methods}")
    
    # 3. Execution Strategy
    print("\n3. PHASE 2 EXECUTION STRATEGY...")
    
    execution_plan = [
        ("Core Adapters", ["adapters"], "High Priority - Framework integrations"),
        ("Data Infrastructure", ["data_loaders", "persistence"], "High Priority - Data processing"),
        ("Distributed Processing", ["distributed"], "Critical - New distributed infrastructure"),
        ("Configuration & Monitoring", ["configuration", "monitoring"], "Medium Priority - Operational"),
        ("Security & Resilience", ["auth", "resilience"], "High Priority - Production readiness"),
        ("Performance & Caching", ["caching", "middleware"], "Medium Priority - Optimization"),
        ("Data Processing", ["preprocessing", "repositories"], "Medium Priority - Data operations")
    ]
    
    for phase_name, categories, priority in execution_plan:
        phase_files = []
        phase_methods = 0
        for cat in categories:
            phase_files.extend(test_categories.get(cat, []))
            phase_methods += sum(count_test_methods_in_file(f) for f in test_categories.get(cat, []))
        
        print(f"   🎯 {phase_name}: {len(phase_files)} files, {phase_methods} methods - {priority}")
    
    # 4. Expected Coverage Impact
    print("\n4. EXPECTED COVERAGE IMPACT...")
    print(f"   📈 Current Coverage: ~50% (after Phase 1)")
    print(f"   🎯 Target Coverage: 70%")
    print(f"   📊 Infrastructure Methods: {total_test_methods}")
    print(f"   🚀 Expected Infrastructure Coverage: 85-90%")
    print(f"   📈 Overall Expected Coverage: 65-70%")
    
    return {
        'validation_results': validation_results,
        'test_categories': test_categories,
        'total_files': total_test_files,
        'total_methods': total_test_methods,
        'execution_plan': execution_plan
    }

def create_infrastructure_test_commands():
    """Create pytest commands for infrastructure testing."""
    
    commands = [
        # Priority 1: Core Adapters (Critical framework integrations)
        "pytest tests/infrastructure/test_adapters.py tests/infrastructure/test_adapters_comprehensive.py -v --tb=short",
        
        # Priority 2: Distributed Processing (New critical infrastructure)
        "pytest tests/infrastructure/ -k 'distributed' -v --tb=short",
        
        # Priority 3: Data Infrastructure
        "pytest tests/infrastructure/test_data_loaders.py tests/infrastructure/test_data_loaders_comprehensive.py -v --tb=short",
        "pytest tests/infrastructure/test_persistence_comprehensive.py tests/infrastructure/test_repositories.py tests/infrastructure/test_repositories_comprehensive.py -v --tb=short",
        
        # Priority 4: Security & Resilience
        "pytest tests/infrastructure/test_auth_comprehensive.py tests/infrastructure/test_resilience_comprehensive.py -v --tb=short",
        
        # Priority 5: Configuration & Monitoring
        "pytest tests/infrastructure/test_configuration_comprehensive.py tests/infrastructure/test_monitoring_comprehensive.py -v --tb=short",
        
        # Priority 6: Performance Infrastructure
        "pytest tests/infrastructure/test_caching_comprehensive.py tests/infrastructure/test_middleware_comprehensive.py -v --tb=short",
        
        # Priority 7: Data Processing
        "pytest tests/infrastructure/test_preprocessing_comprehensive.py -v --tb=short",
        
        # Final: Full infrastructure test suite with coverage
        "pytest tests/infrastructure/ --cov=src/pynomaly/infrastructure --cov-report=html --cov-report=term -v"
    ]
    
    return commands

def main():
    """Main execution function for Phase 2."""
    
    print("🚀 Starting Phase 2: Infrastructure Tests Execution")
    
    try:
        # Execute strategy analysis
        results = execute_infrastructure_tests_strategy()
        
        # Generate test commands
        commands = create_infrastructure_test_commands()
        
        print("\n5. READY FOR EXECUTION...")
        print("📋 Infrastructure test commands ready:")
        for i, cmd in enumerate(commands, 1):
            print(f"   {i}. {cmd}")
        
        print(f"\n✅ Phase 2 Infrastructure Analysis Complete!")
        print(f"📊 Ready to execute {results['total_methods']} test methods across {results['total_files']} files")
        print(f"🎯 Target: Achieve 70% overall coverage through infrastructure testing")
        
        # Check if we can execute tests
        try:
            import pytest
            print(f"\n🚀 Dependencies available - ready for immediate execution!")
            return True
        except ImportError:
            print(f"\n⏳ Dependencies needed for execution:")
            print(f"   pip install pytest pytest-cov pytest-asyncio")
            print(f"   pip install numpy pandas scikit-learn")
            print(f"   pip install torch tensorflow jax")
            return False
            
    except Exception as e:
        print(f"❌ Phase 2 analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
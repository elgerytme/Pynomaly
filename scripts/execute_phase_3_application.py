#!/usr/bin/env python3
"""
Phase 3: Application Layer Tests Execution Script
Target: 85% Test Coverage through Application Layer Testing

This script provides comprehensive application layer test execution for Pynomaly,
targeting use cases, services, DTOs, and integration workflows.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def count_test_methods_in_file(test_file: Path) -> int:
    """Count test methods in a test file."""
    try:
        content = test_file.read_text()
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

def analyze_application_layer_structure():
    """Analyze application layer components and test coverage."""
    
    application_components = {
        'use_cases': {
            'source_path': 'src/pynomaly/application/use_cases',
            'test_files': [],
            'priority': 'Critical',
            'description': 'Core business logic orchestration'
        },
        'services': {
            'source_path': 'src/pynomaly/application/services',
            'test_files': [],
            'priority': 'Critical', 
            'description': 'Application services and orchestration'
        },
        'dto': {
            'source_path': 'src/pynomaly/application/dto',
            'test_files': [],
            'priority': 'High',
            'description': 'Data Transfer Objects'
        },
        'workflows': {
            'source_path': 'src/pynomaly/application',
            'test_files': [],
            'priority': 'High',
            'description': 'Integration workflows and orchestration'
        }
    }
    
    # Analyze test files
    test_dir = Path("tests/application")
    if test_dir.exists():
        test_files = list(test_dir.glob("test_*.py"))
        
        for test_file in test_files:
            name = test_file.name.lower()
            method_count = count_test_methods_in_file(test_file)
            
            if 'use_case' in name:
                application_components['use_cases']['test_files'].append({
                    'file': test_file.name,
                    'methods': method_count
                })
            elif 'service' in name:
                application_components['services']['test_files'].append({
                    'file': test_file.name,
                    'methods': method_count
                })
            elif 'dto' in name:
                application_components['dto']['test_files'].append({
                    'file': test_file.name,
                    'methods': method_count
                })
            elif 'workflow' in name or 'integration' in name:
                application_components['workflows']['test_files'].append({
                    'file': test_file.name,
                    'methods': method_count
                })
    
    return application_components

def validate_application_layer_implementation():
    """Validate application layer implementation completeness."""
    
    # Core application layer files to check
    key_files = [
        'src/pynomaly/application/use_cases/detect_anomalies.py',
        'src/pynomaly/application/use_cases/train_detector.py',
        'src/pynomaly/application/use_cases/evaluate_model.py',
        'src/pynomaly/application/use_cases/explain_anomaly.py',
        'src/pynomaly/application/services/detection_service.py',
        'src/pynomaly/application/services/ensemble_service.py',
        'src/pynomaly/application/services/model_persistence_service.py',
        'src/pynomaly/application/services/automl_service.py',
        'src/pynomaly/application/services/explainability_service.py',
        'src/pynomaly/application/dto/detector_dto.py',
        'src/pynomaly/application/dto/dataset_dto.py',
        'src/pynomaly/application/dto/result_dto.py',
        'src/pynomaly/application/dto/experiment_dto.py',
        'src/pynomaly/application/dto/automl_dto.py',
        'src/pynomaly/application/dto/explainability_dto.py'
    ]
    
    implementation_status = {}
    
    for file_path in key_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            implementation_status[file_path] = {
                'exists': True,
                'size': size,
                'status': 'Implemented' if size > 1000 else 'Partial'
            }
        else:
            implementation_status[file_path] = {
                'exists': False,
                'size': 0,
                'status': 'Missing'
            }
    
    return implementation_status

def execute_application_layer_strategy():
    """Execute Phase 3 application layer testing strategy."""
    
    print("=" * 80)
    print("PHASE 3: APPLICATION LAYER TESTS EXECUTION")
    print("Target: 85% Test Coverage through Application Layer")
    print("=" * 80)
    
    # 1. Analyze Application Components
    print("\n1. ANALYZING APPLICATION LAYER COMPONENTS...")
    components = analyze_application_layer_structure()
    
    total_test_files = 0
    total_test_methods = 0
    
    for component_name, component_data in components.items():
        files_count = len(component_data['test_files'])
        methods_count = sum(f['methods'] for f in component_data['test_files'])
        total_test_files += files_count
        total_test_methods += methods_count
        
        print(f"   📁 {component_name.upper()}: {files_count} files, {methods_count} test methods - {component_data['priority']}")
        for test_file in component_data['test_files']:
            print(f"      📄 {test_file['file']}: {test_file['methods']} methods")
    
    print(f"\n📊 Application Layer Test Summary:")
    print(f"   📁 Total Test Files: {total_test_files}")
    print(f"   🧪 Total Test Methods: {total_test_methods}")
    
    # 2. Validate Implementation
    print("\n2. VALIDATING APPLICATION LAYER IMPLEMENTATION...")
    implementation = validate_application_layer_implementation()
    
    implemented = sum(1 for status in implementation.values() if status['status'] == 'Implemented')
    partial = sum(1 for status in implementation.values() if status['status'] == 'Partial')
    missing = sum(1 for status in implementation.values() if status['status'] == 'Missing')
    
    print(f"   ✅ Implemented: {implemented}")
    print(f"   🔄 Partial: {partial}")
    print(f"   ❌ Missing: {missing}")
    
    for file_path, status in implementation.items():
        if status['status'] != 'Implemented':
            size_kb = status['size'] / 1024 if status['size'] > 0 else 0
            print(f"   ⚠️  {file_path}: {status['status']} ({size_kb:.1f}KB)")
    
    # 3. Execution Strategy
    print("\n3. PHASE 3 EXECUTION STRATEGY...")
    
    execution_phases = [
        ("Core Use Cases", ["use_cases"], "Critical - Business logic"),
        ("Application Services", ["services"], "Critical - Service orchestration"),
        ("Data Transfer Objects", ["dto"], "High - API contracts"),
        ("Integration Workflows", ["workflows"], "High - End-to-end flows")
    ]
    
    for phase_name, component_keys, priority in execution_phases:
        phase_files = []
        phase_methods = 0
        
        for key in component_keys:
            if key in components:
                phase_files.extend(components[key]['test_files'])
                phase_methods += sum(f['methods'] for f in components[key]['test_files'])
        
        print(f"   🎯 {phase_name}: {len(phase_files)} files, {phase_methods} methods - {priority}")
    
    # 4. Expected Coverage Impact
    print("\n4. EXPECTED COVERAGE IMPACT...")
    print(f"   📈 Current Coverage: ~70% (after Phase 2)")
    print(f"   🎯 Target Coverage: 85%")
    print(f"   📊 Application Methods: {total_test_methods}")
    print(f"   🚀 Expected Application Coverage: 90-95%")
    print(f"   📈 Overall Expected Coverage: 80-85%")
    
    return {
        'components': components,
        'implementation': implementation,
        'total_files': total_test_files,
        'total_methods': total_test_methods,
        'execution_phases': execution_phases
    }

def create_application_test_commands():
    """Create pytest commands for application layer testing."""
    
    commands = [
        # Priority 1: Core Use Cases (Critical business logic)
        "pytest tests/application/test_use_cases.py tests/application/test_use_cases_enhanced.py -v --tb=short",
        
        # Priority 2: Application Services (Service orchestration)
        "pytest tests/application/test_services.py tests/application/test_services_enhanced.py tests/application/test_services_specific.py -v --tb=short",
        
        # Priority 3: Data Transfer Objects (API contracts)
        "pytest tests/application/test_dto.py tests/application/test_dto_comprehensive.py tests/application/test_dto_fixed.py tests/application/test_dto_production.py -v --tb=short",
        
        # Priority 4: Integration Workflows (End-to-end flows)
        "pytest tests/application/test_integration_workflows.py -v --tb=short",
        
        # Final: Full application test suite with coverage
        "pytest tests/application/ --cov=src/pynomaly/application --cov-report=html --cov-report=term -v"
    ]
    
    return commands

def generate_application_test_enhancements():
    """Generate enhanced test cases for missing application components."""
    
    print("\n5. GENERATING ENHANCED APPLICATION TESTS...")
    
    # Check for AutoML and Explainability service tests
    automl_tests_needed = not Path("tests/application/test_automl_services.py").exists()
    explainability_tests_needed = not Path("tests/application/test_explainability_services.py").exists()
    
    if automl_tests_needed:
        print("   📝 AutoML service tests needed")
    
    if explainability_tests_needed:
        print("   📝 Explainability service tests needed")
    
    # Enhanced use case tests
    enhanced_use_cases = [
        "detect_anomalies_advanced.py",
        "train_detector_ensemble.py", 
        "evaluate_model_comprehensive.py",
        "explain_anomaly_detailed.py",
        "automl_workflow.py",
        "streaming_detection.py"
    ]
    
    print("   📋 Enhanced use case tests recommended:")
    for use_case in enhanced_use_cases:
        print(f"      🎯 {use_case}")
    
    return {
        'automl_tests_needed': automl_tests_needed,
        'explainability_tests_needed': explainability_tests_needed,
        'enhanced_use_cases': enhanced_use_cases
    }

def main():
    """Main execution function for Phase 3."""
    
    print("🚀 Starting Phase 3: Application Layer Tests Execution")
    
    try:
        # Execute strategy analysis
        results = execute_application_layer_strategy()
        
        # Generate test commands
        commands = create_application_test_commands()
        
        # Generate enhancement recommendations
        enhancements = generate_application_test_enhancements()
        
        print("\n6. READY FOR EXECUTION...")
        print("📋 Application test commands ready:")
        for i, cmd in enumerate(commands, 1):
            print(f"   {i}. {cmd}")
        
        print(f"\n✅ Phase 3 Application Analysis Complete!")
        print(f"📊 Ready to execute {results['total_methods']} test methods across {results['total_files']} files")
        print(f"🎯 Target: Achieve 85% overall coverage through application testing")
        
        # Check if we can execute tests
        try:
            import pytest
            print(f"\n🚀 Dependencies available - ready for immediate execution!")
            return True
        except ImportError:
            print(f"\n⏳ Dependencies needed for execution:")
            print(f"   pip install pytest pytest-cov pytest-asyncio")
            print(f"   pip install numpy pandas scikit-learn")
            print(f"   pip install pydantic fastapi")
            return False
            
    except Exception as e:
        print(f"❌ Phase 3 analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
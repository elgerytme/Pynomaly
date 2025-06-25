#!/usr/bin/env python3
"""
Comprehensive Test Suite for Pynomaly

Consolidates testing functionality from multiple scripts:
- test_infrastructure_validation.py
- test_health_dashboard.py  
- test_algorithm_optimization.py
- test_dependencies.py
- test_domain_fixes.py
- test_performance_monitoring.py
- test_memory_optimization.py
- And other test_*.py scripts
"""

import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

@dataclass
class TestResult:
    """Test result data structure."""
    name: str
    category: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ComprehensiveTestSuite:
    """Comprehensive testing suite for all Pynomaly components."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.results: List[TestResult] = []
        
    def run_infrastructure_tests(self) -> List[TestResult]:
        """Test infrastructure components."""
        print("🔍 Testing Infrastructure Components...")
        
        infrastructure_results = []
        
        # Test container dependency injection
        start_time = time.time()
        try:
            from pynomaly.infrastructure.config.container import Container
            container = Container()
            
            infrastructure_results.append(TestResult(
                name="container_creation",
                category="infrastructure",
                passed=True,
                execution_time=time.time() - start_time,
                details={"container_type": str(type(container))}
            ))
            print("✅ Container creation successful")
            
        except Exception as e:
            infrastructure_results.append(TestResult(
                name="container_creation",
                category="infrastructure", 
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))
            print(f"❌ Container creation failed: {e}")
        
        # Test adapter imports
        adapters = ["pyod_adapter", "sklearn_adapter", "pytorch_adapter", "tensorflow_adapter"]
        for adapter in adapters:
            start_time = time.time()
            try:
                module_path = f"pynomaly.infrastructure.adapters.{adapter}"
                __import__(module_path)
                
                infrastructure_results.append(TestResult(
                    name=f"adapter_import_{adapter}",
                    category="infrastructure",
                    passed=True,
                    execution_time=time.time() - start_time
                ))
                print(f"✅ {adapter} import successful")
                
            except ImportError:
                infrastructure_results.append(TestResult(
                    name=f"adapter_import_{adapter}",
                    category="infrastructure",
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Optional adapter {adapter} not available"
                ))
                print(f"⚠️ {adapter} not available (optional)")
                
            except Exception as e:
                infrastructure_results.append(TestResult(
                    name=f"adapter_import_{adapter}",
                    category="infrastructure",
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ))
                print(f"❌ {adapter} import failed: {e}")
        
        return infrastructure_results
    
    def run_algorithm_tests(self) -> List[TestResult]:
        """Test algorithm components."""
        print("\n🔍 Testing Algorithm Components...")
        
        algorithm_results = []
        
        # Test domain services
        domain_services = [
            "anomaly_scorer", "threshold_calculator", 
            "feature_validator", "ensemble_aggregator"
        ]
        
        for service in domain_services:
            start_time = time.time()
            try:
                module_path = f"pynomaly.domain.services.{service}"
                __import__(module_path)
                
                algorithm_results.append(TestResult(
                    name=f"domain_service_{service}",
                    category="algorithm",
                    passed=True,
                    execution_time=time.time() - start_time
                ))
                print(f"✅ Domain service {service} import successful")
                
            except Exception as e:
                algorithm_results.append(TestResult(
                    name=f"domain_service_{service}",
                    category="algorithm",
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ))
                print(f"❌ Domain service {service} import failed: {e}")
        
        return algorithm_results
    
    def run_dependency_tests(self) -> List[TestResult]:
        """Test critical dependencies."""
        print("\n🔍 Testing Dependencies...")
        
        dependency_results = []
        
        # Test core dependencies
        core_deps = {
            "numpy": "numpy",
            "pandas": "pandas", 
            "scikit-learn": "sklearn",
            "pyod": "pyod",
            "fastapi": "fastapi",
            "typer": "typer"
        }
        
        for name, import_name in core_deps.items():
            start_time = time.time()
            try:
                __import__(import_name)
                
                dependency_results.append(TestResult(
                    name=f"dependency_{name}",
                    category="dependencies",
                    passed=True,
                    execution_time=time.time() - start_time
                ))
                print(f"✅ Core dependency {name} available")
                
            except ImportError as e:
                dependency_results.append(TestResult(
                    name=f"dependency_{name}",
                    category="dependencies",
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ))
                print(f"❌ Core dependency {name} missing: {e}")
        
        # Test optional dependencies
        optional_deps = {
            "torch": "torch",
            "tensorflow": "tensorflow",
            "jax": "jax",
            "polars": "polars"
        }
        
        for name, import_name in optional_deps.items():
            start_time = time.time()
            try:
                __import__(import_name)
                
                dependency_results.append(TestResult(
                    name=f"optional_dependency_{name}",
                    category="dependencies",
                    passed=True,
                    execution_time=time.time() - start_time
                ))
                print(f"✅ Optional dependency {name} available")
                
            except ImportError:
                dependency_results.append(TestResult(
                    name=f"optional_dependency_{name}",
                    category="dependencies",
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Optional dependency {name} not installed"
                ))
                print(f"⚠️ Optional dependency {name} not available")
        
        return dependency_results
    
    def run_performance_tests(self) -> List[TestResult]:
        """Test performance characteristics."""
        print("\n🔍 Testing Performance...")
        
        performance_results = []
        
        # Test import performance
        start_time = time.time()
        try:
            from pynomaly.application.services.detection_service import DetectionService
            import_time = time.time() - start_time
            
            performance_results.append(TestResult(
                name="detection_service_import",
                category="performance",
                passed=import_time < 2.0,
                execution_time=import_time,
                details={"import_time_threshold": 2.0}
            ))
            
            if import_time < 2.0:
                print(f"✅ Detection service import: {import_time:.3f}s (Good)")
            else:
                print(f"⚠️ Detection service import: {import_time:.3f}s (Slow)")
                
        except Exception as e:
            performance_results.append(TestResult(
                name="detection_service_import",
                category="performance",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))
            print(f"❌ Detection service import failed: {e}")
        
        return performance_results
    
    def run_memory_tests(self) -> List[TestResult]:
        """Test memory efficiency."""
        print("\n🔍 Testing Memory Efficiency...")
        
        memory_results = []
        
        # Test basic memory usage
        start_time = time.time()
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Import main components
            from pynomaly.infrastructure.config.container import Container
            container = Container()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            memory_results.append(TestResult(
                name="container_memory_usage",
                category="memory",
                passed=memory_increase < 100,  # Less than 100MB increase
                execution_time=time.time() - start_time,
                details={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase
                }
            ))
            
            if memory_increase < 100:
                print(f"✅ Container memory usage: +{memory_increase:.1f}MB (Good)")
            else:
                print(f"⚠️ Container memory usage: +{memory_increase:.1f}MB (High)")
                
        except ImportError:
            memory_results.append(TestResult(
                name="container_memory_usage",
                category="memory",
                passed=False,
                execution_time=time.time() - start_time,
                error_message="psutil not available for memory testing"
            ))
            print("⚠️ psutil not available for memory testing")
            
        except Exception as e:
            memory_results.append(TestResult(
                name="container_memory_usage", 
                category="memory",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))
            print(f"❌ Memory test failed: {e}")
        
        return memory_results
    
    def run_health_dashboard_tests(self) -> List[TestResult]:
        """Test health monitoring components."""
        print("\n🔍 Testing Health Dashboard...")
        
        health_results = []
        
        # Test health service
        start_time = time.time()
        try:
            from pynomaly.infrastructure.monitoring.health_service import HealthService
            
            health_results.append(TestResult(
                name="health_service_import",
                category="health",
                passed=True,
                execution_time=time.time() - start_time
            ))
            print("✅ Health service import successful")
            
        except Exception as e:
            health_results.append(TestResult(
                name="health_service_import",
                category="health",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))
            print(f"❌ Health service import failed: {e}")
        
        return health_results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        # Categorize results
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"passed": 0, "failed": 0, "total": 0}
            
            categories[result.category]["total"] += 1
            if result.passed:
                categories[result.category]["passed"] += 1
            else:
                categories[result.category]["failed"] += 1
        
        # Calculate overall metrics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate recommendations
        recommendations = []
        for result in self.results:
            if not result.passed and result.error_message:
                recommendations.append(f"Fix {result.category}.{result.name}: {result.error_message}")
        
        if not recommendations:
            recommendations.append("All tests passed! System is healthy.")
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": round(success_rate, 2),
                "execution_time": sum(r.execution_time for r in self.results)
            },
            "categories": categories,
            "detailed_results": [asdict(r) for r in self.results],
            "recommendations": recommendations
        }
        
        return report
    
    def run_full_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite."""
        print("🚀 Starting Comprehensive Test Suite...")
        print("=" * 60)
        
        # Run all test categories
        self.results.extend(self.run_infrastructure_tests())
        self.results.extend(self.run_algorithm_tests())
        self.results.extend(self.run_dependency_tests())
        self.results.extend(self.run_performance_tests())
        self.results.extend(self.run_memory_tests())
        self.results.extend(self.run_health_dashboard_tests())
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("📊 Comprehensive Test Suite Results:")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Success Rate: {report['summary']['success_rate']}%")
        print(f"Total Execution Time: {report['summary']['execution_time']:.2f}s")
        
        print("\n📋 Category Breakdown:")
        for category, stats in report['categories'].items():
            success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        if report['recommendations']:
            print("\n📋 Recommendations:")
            for rec in report['recommendations'][:5]:  # Show first 5
                print(f"  • {rec}")
            if len(report['recommendations']) > 5:
                print(f"  ... and {len(report['recommendations']) - 5} more")
        
        return report


def main():
    """Main entry point for comprehensive test suite."""
    suite = ComprehensiveTestSuite()
    report = suite.run_full_suite()
    
    # Save report to file
    report_file = PROJECT_ROOT / "comprehensive_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    success_rate = report['summary']['success_rate']
    if success_rate >= 70:
        print("🎉 Comprehensive test suite passed!")
        sys.exit(0)
    else:
        print("❌ Comprehensive test suite failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
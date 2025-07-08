"""
Utility functions for testing and validating the dependency injection system.

This module provides tools to help developers test their dependency setup
and troubleshoot issues with the forward-reference-free dependency system.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
from contextlib import contextmanager

from pynomaly.infrastructure.dependencies import (
    DependencyRegistry,
    register_dependency,
    register_dependency_provider,
    clear_dependencies,
    is_dependency_available,
    get_dependency,
)

logger = logging.getLogger(__name__)


class DependencyValidator:
    """Validates dependency setup and provides debugging information."""
    
    def __init__(self, registry: DependencyRegistry):
        """Initialize the validator.
        
        Args:
            registry: The dependency registry to validate
        """
        self.registry = registry
        self.validation_results: Dict[str, Any] = {}
    
    def validate_dependencies(self, expected_dependencies: List[str]) -> Dict[str, Any]:
        """Validate that expected dependencies are available.
        
        Args:
            expected_dependencies: List of dependency keys that should be available
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "available": [],
            "missing": [],
            "errors": [],
            "total_expected": len(expected_dependencies),
            "total_available": 0,
        }
        
        for dep_key in expected_dependencies:
            try:
                if is_dependency_available(dep_key):
                    results["available"].append(dep_key)
                    results["total_available"] += 1
                else:
                    results["missing"].append(dep_key)
                    results["valid"] = False
            except Exception as e:
                results["errors"].append({"dependency": dep_key, "error": str(e)})
                results["valid"] = False
        
        self.validation_results = results
        return results
    
    def test_dependency_creation(self, dependency_key: str) -> Dict[str, Any]:
        """Test that a dependency can be created successfully.
        
        Args:
            dependency_key: The dependency key to test
            
        Returns:
            Dictionary with test results
        """
        result = {
            "dependency": dependency_key,
            "success": False,
            "error": None,
            "instance": None,
            "type": None,
        }
        
        try:
            instance = get_dependency(dependency_key)
            result["success"] = True
            result["instance"] = instance
            result["type"] = type(instance).__name__ if instance else "None"
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def get_dependency_report(self) -> Dict[str, Any]:
        """Get a comprehensive report of all dependencies.
        
        Returns:
            Dictionary with dependency report
        """
        report = {
            "registry_initialized": self.registry.initialized,
            "total_dependencies": len(self.registry._dependencies),
            "total_providers": len(self.registry._providers),
            "dependencies": {},
            "providers": {},
        }
        
        # Test each registered dependency
        for key in self.registry._dependencies:
            report["dependencies"][key] = self.test_dependency_creation(key)
        
        # List providers
        for key in self.registry._providers:
            report["providers"][key] = {
                "provider": str(self.registry._providers[key]),
                "tested": key in report["dependencies"],
            }
        
        return report
    
    def print_report(self, report: Optional[Dict[str, Any]] = None) -> None:
        """Print a formatted dependency report.
        
        Args:
            report: Optional report to print (generates new one if None)
        """
        if report is None:
            report = self.get_dependency_report()
        
        print("=" * 60)
        print("DEPENDENCY INJECTION SYSTEM REPORT")
        print("=" * 60)
        
        print(f"Registry Initialized: {report['registry_initialized']}")
        print(f"Total Dependencies: {report['total_dependencies']}")
        print(f"Total Providers: {report['total_providers']}")
        print()
        
        print("DEPENDENCIES:")
        print("-" * 40)
        for key, info in report["dependencies"].items():
            status = "✓" if info["success"] else "✗"
            print(f"{status} {key}: {info['type']}")
            if info["error"]:
                print(f"    Error: {info['error']}")
        
        print()
        print("PROVIDERS:")
        print("-" * 40)
        for key, info in report["providers"].items():
            tested = "✓" if info["tested"] else "○"
            print(f"{tested} {key}: {info['provider']}")
        
        print("=" * 60)


@contextmanager
def test_dependency_context():
    """Context manager for testing dependencies in isolation."""
    original_dependencies = {}
    original_providers = {}
    
    # Save current state
    from pynomaly.infrastructure.dependencies.wrapper import _registry
    original_dependencies = _registry._dependencies.copy()
    original_providers = _registry._providers.copy()
    
    try:
        # Clear for testing
        clear_dependencies()
        yield
    finally:
        # Restore original state
        _registry._dependencies = original_dependencies
        _registry._providers = original_providers


def create_mock_dependencies() -> Dict[str, Any]:
    """Create mock dependencies for testing purposes.
    
    Returns:
        Dictionary of mock dependencies
    """
    class MockAuthService:
        def authenticate(self, token: str) -> bool:
            return token == "valid_token"
    
    class MockDetectionService:
        def detect(self, data: Any) -> List[int]:
            return [1, 2, 3]  # Mock anomaly indices
    
    class MockModelService:
        def train(self, data: Any) -> str:
            return "mock_model_id"
    
    class MockDatabaseService:
        def save(self, data: Any) -> str:
            return "mock_save_id"
    
    return {
        "auth_service": MockAuthService(),
        "detection_service": MockDetectionService(),
        "model_service": MockModelService(),
        "database_service": MockDatabaseService(),
    }


def setup_test_dependencies() -> None:
    """Setup mock dependencies for testing."""
    mock_deps = create_mock_dependencies()
    
    for key, instance in mock_deps.items():
        register_dependency(key, instance)


def validate_standard_dependencies() -> Dict[str, Any]:
    """Validate the standard set of dependencies expected by the system.
    
    Returns:
        Validation results
    """
    expected_deps = [
        "auth_service",
        "detection_service",
        "model_service",
        "database_service",
        "cache_service",
        "metrics_service",
        "user_service",
        "dataset_repository",
        "detector_repository",
        "result_repository",
        "detect_anomalies_use_case",
        "train_detector_use_case",
        "evaluate_model_use_case",
    ]
    
    from pynomaly.infrastructure.dependencies.wrapper import _registry
    validator = DependencyValidator(_registry)
    return validator.validate_dependencies(expected_deps)


def run_dependency_health_check() -> None:
    """Run a comprehensive health check of the dependency system."""
    print("Running dependency health check...")
    
    # Validate standard dependencies
    results = validate_standard_dependencies()
    
    print(f"Dependencies: {results['total_available']}/{results['total_expected']} available")
    
    if results["missing"]:
        print("Missing dependencies:")
        for dep in results["missing"]:
            print(f"  - {dep}")
    
    if results["errors"]:
        print("Errors:")
        for error in results["errors"]:
            print(f"  - {error['dependency']}: {error['error']}")
    
    # Generate full report
    from pynomaly.infrastructure.dependencies.wrapper import _registry
    validator = DependencyValidator(_registry)
    report = validator.get_dependency_report()
    validator.print_report(report)
    
    return results["valid"]


# Example usage functions
def example_test_dependency_setup():
    """Example of how to test dependency setup."""
    print("Testing dependency setup...")
    
    # Test in isolation
    with test_dependency_context():
        # Setup mock dependencies
        setup_test_dependencies()
        
        # Validate they were setup correctly
        results = validate_standard_dependencies()
        print(f"Mock setup successful: {results['valid']}")
        
        # Test individual dependency
        from pynomaly.infrastructure.dependencies.wrapper import _registry
        validator = DependencyValidator(_registry)
        test_result = validator.test_dependency_creation("auth_service")
        print(f"Auth service test: {test_result}")


def example_validate_production_dependencies():
    """Example of how to validate production dependencies."""
    print("Validating production dependencies...")
    
    # This would be called after your app startup
    is_healthy = run_dependency_health_check()
    
    if is_healthy:
        print("✓ All dependencies are healthy")
    else:
        print("✗ Some dependencies are missing or broken")
        
    return is_healthy


if __name__ == "__main__":
    # Run examples
    example_test_dependency_setup()
    print("\n" + "="*60 + "\n")
    example_validate_production_dependencies()

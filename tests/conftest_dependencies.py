"""Dependency management for tests."""

import pytest
import importlib
from typing import Dict, List


def _check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# Define dependency groups and their import checks
DEPENDENCY_GROUPS = {
    'torch': ['torch'],
    'tensorflow': ['tensorflow'],
    'jax': ['jax', 'jaxlib'],
    'redis': ['redis'],
    'auth': ['passlib'],
    'optuna': ['optuna'],
    'shap': ['shap'],
    'lime': ['lime'],
    'scikit-learn': ['sklearn'],
    'pyod': ['pyod'],
    'fastapi': ['fastapi'],
    'uvicorn': ['uvicorn'],
    'database': ['sqlalchemy', 'psycopg2'],
    'hypothesis': ['hypothesis'],
    'testing': ['pytest'],
}

# Check which dependencies are available
AVAILABLE_DEPENDENCIES = {}
for group, modules in DEPENDENCY_GROUPS.items():
    AVAILABLE_DEPENDENCIES[group] = all(_check_import(mod) for mod in modules)

# Core dependencies that should always be available
CORE_DEPENDENCIES = ['numpy', 'pandas', 'scipy', 'pydantic']
CORE_AVAILABLE = all(_check_import(dep) for dep in CORE_DEPENDENCIES)


def requires_dependency(dependency: str):
    """Skip test if dependency is not available."""
    return pytest.mark.skipif(
        not AVAILABLE_DEPENDENCIES.get(dependency, False),
        reason=f"Requires {dependency} dependency"
    )


def requires_dependencies(*dependencies: str):
    """Skip test if any of the dependencies are not available."""
    missing = [dep for dep in dependencies if not AVAILABLE_DEPENDENCIES.get(dep, False)]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"Requires dependencies: {', '.join(missing)}"
    )


def requires_core_dependencies():
    """Skip test if core dependencies are not available."""
    return pytest.mark.skipif(
        not CORE_AVAILABLE,
        reason="Requires core dependencies: numpy, pandas, scipy, pydantic"
    )


# Pytest markers for different dependency groups
pytestmark = [
    pytest.mark.torch,
    pytest.mark.tensorflow, 
    pytest.mark.jax,
    pytest.mark.redis,
    pytest.mark.auth,
    pytest.mark.optuna,
    pytest.mark.ml_optional,
    pytest.mark.infrastructure,
    pytest.mark.slow,
]


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "torch: tests requiring PyTorch")
    config.addinivalue_line("markers", "tensorflow: tests requiring TensorFlow") 
    config.addinivalue_line("markers", "jax: tests requiring JAX")
    config.addinivalue_line("markers", "redis: tests requiring Redis")
    config.addinivalue_line("markers", "auth: tests requiring authentication dependencies")
    config.addinivalue_line("markers", "optuna: tests requiring Optuna")
    config.addinivalue_line("markers", "shap: tests requiring SHAP")
    config.addinivalue_line("markers", "lime: tests requiring LIME")
    config.addinivalue_line("markers", "scikit-learn: tests requiring scikit-learn")
    config.addinivalue_line("markers", "pyod: tests requiring PyOD")
    config.addinivalue_line("markers", "fastapi: tests requiring FastAPI")
    config.addinivalue_line("markers", "ml_optional: tests requiring optional ML libraries")
    config.addinivalue_line("markers", "infrastructure: tests requiring infrastructure dependencies")
    config.addinivalue_line("markers", "slow: slow tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on dependencies."""
    for item in items:
        # Check test file path and add appropriate markers
        test_path = str(item.fspath)
        
        if 'infrastructure/test_adapters' in test_path:
            if 'torch' in test_path or 'pytorch' in test_path:
                item.add_marker(pytest.mark.torch)
            elif 'tensorflow' in test_path:
                item.add_marker(pytest.mark.tensorflow)
            elif 'jax' in test_path:
                item.add_marker(pytest.mark.jax)
                
        if 'auth' in test_path:
            item.add_marker(pytest.mark.auth)
            
        if 'redis' in test_path or 'cache' in test_path:
            item.add_marker(pytest.mark.redis)
            
        if 'automl' in test_path:
            item.add_marker(pytest.mark.optuna)
            
        if 'explainability' in test_path:
            item.add_marker(pytest.mark.shap)
            item.add_marker(pytest.mark.lime)


def get_dependency_report() -> Dict[str, bool]:
    """Get a report of available dependencies."""
    return AVAILABLE_DEPENDENCIES.copy()


def print_dependency_status():
    """Print status of all dependencies."""
    print("\n" + "="*50)
    print("DEPENDENCY STATUS")
    print("="*50)
    
    for group, available in AVAILABLE_DEPENDENCIES.items():
        status = "✅ AVAILABLE" if available else "❌ MISSING"
        modules = ", ".join(DEPENDENCY_GROUPS[group])
        print(f"{group:15} ({modules:20}) : {status}")
    
    print(f"\nCore dependencies: {'✅ AVAILABLE' if CORE_AVAILABLE else '❌ MISSING'}")
    print("="*50)
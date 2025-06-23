#!/usr/bin/env python3
"""
Test script to validate all dependencies and fix import issues.
This script checks if all required dependencies are available and 
identifies missing packages that need to be installed.
"""

import sys
import subprocess
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Core dependencies that must be available
CORE_DEPENDENCIES = [
    'numpy',
    'pandas', 
    'scipy',
    'scikit-learn',
    'pyod',
    'pydantic',
    'fastapi',
    'uvicorn',
    'typer',
    'structlog',
    'dependency_injector',
    'httpx',
    'requests',
    'rich',
    'jinja2',
    'psutil',
    'aiofiles',
    'pyjwt',
]

# Optional ML dependencies
ML_DEPENDENCIES = [
    'torch',
    'tensorflow', 
    'jax',
    'jaxlib',
    'optax',
]

# Infrastructure dependencies  
INFRASTRUCTURE_DEPENDENCIES = [
    'redis',
    'passlib',
    'sqlalchemy',
    'psycopg2',
]

# Explainability dependencies
EXPLAINABILITY_DEPENDENCIES = [
    'shap',
    'lime',
]

# AutoML dependencies
AUTOML_DEPENDENCIES = [
    'optuna',
]

# Testing dependencies
TESTING_DEPENDENCIES = [
    'pytest',
    'pytest_cov',
    'pytest_asyncio',
    'pytest_mock',
    'hypothesis',
]

def check_dependency(package: str) -> Tuple[bool, Optional[str]]:
    """Check if a package can be imported."""
    try:
        importlib.import_module(package)
        return True, None
    except ImportError as e:
        return False, str(e)

def install_package(package: str) -> bool:
    """Install a package using pip."""
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--user', package
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependency_group(group_name: str, dependencies: List[str]) -> Dict[str, Tuple[bool, Optional[str]]]:
    """Check a group of dependencies."""
    logger.info(f"Checking {group_name} dependencies...")
    results = {}
    
    for dep in dependencies:
        available, error = check_dependency(dep)
        results[dep] = (available, error)
        
        if available:
            logger.info(f"  ✓ {dep}")
        else:
            logger.warning(f"  ✗ {dep}: {error}")
    
    return results

def install_missing_dependencies(missing: List[str]) -> None:
    """Install missing dependencies."""
    if not missing:
        logger.info("No missing dependencies to install.")
        return
    
    logger.info(f"Installing {len(missing)} missing dependencies...")
    
    # Map package names to pip install names
    pip_mapping = {
        'dependency_injector': 'dependency-injector',
        'pytest_cov': 'pytest-cov',
        'pytest_asyncio': 'pytest-asyncio',
        'pytest_mock': 'pytest-mock',
        'psycopg2': 'psycopg2-binary',
    }
    
    for package in missing:
        pip_name = pip_mapping.get(package, package)
        logger.info(f"Installing {pip_name}...")
        
        if install_package(pip_name):
            logger.info(f"  ✓ {pip_name} installed successfully")
        else:
            logger.error(f"  ✗ Failed to install {pip_name}")

def test_imports() -> bool:
    """Test importing key Pynomaly modules."""
    logger.info("Testing Pynomaly module imports...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    
    test_imports = [
        'pynomaly',
        'pynomaly.domain.entities',
        'pynomaly.domain.value_objects', 
        'pynomaly.application.dto',
        'pynomaly.application.use_cases',
        'pynomaly.infrastructure.config',
        'pynomaly.shared.protocols',
    ]
    
    success_count = 0
    for module in test_imports:
        try:
            importlib.import_module(module)
            logger.info(f"  ✓ {module}")
            success_count += 1
        except ImportError as e:
            logger.error(f"  ✗ {module}: {e}")
    
    return success_count == len(test_imports)

def run_minimal_test() -> bool:
    """Run a minimal test to ensure basic functionality."""
    logger.info("Running minimal functionality test...")
    
    try:
        # Test domain entities
        import numpy as np
        from pynomaly.domain.entities import Dataset, Detector
        from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore
        
        # Create test objects
        contamination = ContaminationRate(0.1)
        dataset = Dataset(
            name="test_dataset",
            data=np.array([[1.0, 2.0], [3.0, 4.0]]),
            feature_names=["feature1", "feature2"]
        )
        detector = Detector(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=contamination
        )
        score = AnomalyScore(0.8)
        
        logger.info("  ✓ Domain objects created successfully")
        
        # Test DTO imports
        from pynomaly.application.dto import (
            CreateDetectorDTO, 
            DetectorResponseDTO,
            DatasetDTO
        )
        
        # Create test DTOs
        detector_dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest", 
            contamination_rate=0.1
        )
        
        logger.info("  ✓ Application DTOs created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Minimal test failed: {e}")
        return False

def main():
    """Main dependency testing function."""
    logger.info("Pynomaly Dependency Testing and Resolution")
    logger.info("=" * 50)
    
    all_results = {}
    missing_packages = []
    
    # Check all dependency groups
    dependency_groups = [
        ("Core", CORE_DEPENDENCIES),
        ("ML Frameworks", ML_DEPENDENCIES), 
        ("Infrastructure", INFRASTRUCTURE_DEPENDENCIES),
        ("Explainability", EXPLAINABILITY_DEPENDENCIES),
        ("AutoML", AUTOML_DEPENDENCIES),
        ("Testing", TESTING_DEPENDENCIES),
    ]
    
    for group_name, dependencies in dependency_groups:
        results = check_dependency_group(group_name, dependencies)
        all_results[group_name] = results
        
        # Collect missing packages
        for package, (available, _) in results.items():
            if not available:
                missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        logger.info(f"\nFound {len(missing_packages)} missing dependencies:")
        for package in missing_packages:
            logger.info(f"  - {package}")
        
        install_missing_dependencies(missing_packages)
        
        # Re-check core dependencies after installation
        logger.info("\nRe-checking core dependencies after installation...")
        core_results = check_dependency_group("Core (Recheck)", CORE_DEPENDENCIES)
        
    # Test Pynomaly imports
    logger.info("\n" + "=" * 50)
    import_success = test_imports()
    
    # Run minimal functionality test
    logger.info("\n" + "=" * 50)
    test_success = run_minimal_test()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("DEPENDENCY TEST SUMMARY")
    logger.info("=" * 50)
    
    total_deps = sum(len(deps) for _, deps in dependency_groups)
    available_deps = sum(
        sum(1 for available, _ in results.values() if available)
        for results in all_results.values()
    )
    
    logger.info(f"Dependencies available: {available_deps}/{total_deps}")
    logger.info(f"Import test: {'PASS' if import_success else 'FAIL'}")
    logger.info(f"Functionality test: {'PASS' if test_success else 'FAIL'}")
    
    overall_success = (
        len([p for p in CORE_DEPENDENCIES if check_dependency(p)[0]]) == len(CORE_DEPENDENCIES) and
        import_success and
        test_success
    )
    
    if overall_success:
        logger.info("\n🎉 All core dependencies are working correctly!")
        logger.info("You can now run tests with improved coverage.")
        return 0
    else:
        logger.error("\n❌ Some dependencies or tests failed.")
        logger.error("Please resolve the issues above before running tests.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
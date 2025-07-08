"""
Utilities for testing optional extras with pytest.importorskip.

This module provides parametrized test decorators and fixtures for testing
functionality that depends on optional extras: deep, automl, explainability, streaming.
"""

import pytest
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# Define extras and their required packages
EXTRAS_REQUIREMENTS = {
    "deep": {
        "torch": ["torch"],
        "tensorflow": ["tensorflow", "keras"],
        "jax": ["jax", "jaxlib", "optax"],
    },
    "automl": {
        "optuna": ["optuna"],
        "hyperopt": ["hyperopt"],
        "auto_sklearn": ["autosklearn"],
    },
    "explainability": {
        "shap": ["shap"],
        "lime": ["lime"],
    },
    "streaming": {
        "kafka": ["kafka"],
        "redis": ["redis"],
        "asyncio": ["asyncio"],
    },
}


def skip_if_missing_extra(extra: str, reason: Optional[str] = None) -> Callable:
    """
    Skip test if the specified extra is not available.
    
    Args:
        extra: Name of the extra (deep, automl, explainability, streaming)
        reason: Optional custom reason for skipping
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if extra not in EXTRAS_REQUIREMENTS:
                pytest.skip(f"Unknown extra: {extra}")
            
            # Check if any package from the extra is available
            extra_packages = EXTRAS_REQUIREMENTS[extra]
            available_packages = []
            
            for package_group, packages in extra_packages.items():
                for package in packages:
                    try:
                        pytest.importorskip(package)
                        available_packages.append(package)
                    except pytest.skip.Exception:
                        continue
            
            # If no packages are available, skip the test
            if not available_packages:
                skip_reason = reason or f"No packages available for {extra} extra"
                pytest.skip(skip_reason)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def parametrize_with_extras(
    extras: List[str],
    combinations: bool = False,
    ids: Optional[List[str]] = None
) -> Callable:
    """
    Parametrize test with different extras configurations.
    
    Args:
        extras: List of extras to test
        combinations: If True, test all combinations of extras
        ids: Optional list of test IDs
    
    Returns:
        Parametrized test decorator
    """
    if combinations:
        # Generate all combinations of extras
        import itertools
        all_combinations = []
        for r in range(1, len(extras) + 1):
            all_combinations.extend(itertools.combinations(extras, r))
        test_params = [list(combo) for combo in all_combinations]
    else:
        # Test each extra individually
        test_params = [[extra] for extra in extras]
    
    if ids is None:
        ids = ["+".join(param) for param in test_params]
    
    def decorator(func: Callable) -> Callable:
        @pytest.mark.parametrize("required_extras", test_params, ids=ids)
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if all required extras are available
            required_extras = kwargs.get("required_extras", args[-1] if args else [])
            
            missing_extras = []
            for extra in required_extras:
                if extra not in EXTRAS_REQUIREMENTS:
                    missing_extras.append(extra)
                    continue
                
                # Check if any package from the extra is available
                extra_packages = EXTRAS_REQUIREMENTS[extra]
                available = False
                
                for package_group, packages in extra_packages.items():
                    for package in packages:
                        try:
                            pytest.importorskip(package)
                            available = True
                            break
                        except pytest.skip.Exception:
                            continue
                    if available:
                        break
                
                if not available:
                    missing_extras.append(extra)
            
            if missing_extras:
                pytest.skip(f"Missing extras: {', '.join(missing_extras)}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Specific decorators for each extra
def requires_deep(reason: Optional[str] = None) -> Callable:
    """Skip test if deep learning packages are not available."""
    return skip_if_missing_extra("deep", reason)


def requires_automl(reason: Optional[str] = None) -> Callable:
    """Skip test if AutoML packages are not available."""
    return skip_if_missing_extra("automl", reason)


def requires_explainability(reason: Optional[str] = None) -> Callable:
    """Skip test if explainability packages are not available."""
    return skip_if_missing_extra("explainability", reason)


def requires_streaming(reason: Optional[str] = None) -> Callable:
    """Skip test if streaming packages are not available."""
    return skip_if_missing_extra("streaming", reason)


# Framework testing utilities
def check_graceful_degradation(
    func: Callable,
    extra: str,
    expected_fallback: Any = None,
    expected_error_type: type = ImportError
) -> Tuple[bool, Any]:
    """
    Check if a function gracefully degrades when extra is missing.
    
    Args:
        func: Function to test
        extra: Name of the extra
        expected_fallback: Expected fallback value/behavior
        expected_error_type: Expected error type if no graceful degradation
    
    Returns:
        Tuple of (graceful_degradation_works, result_or_error)
    """
    try:
        result = func()
        return True, result
    except expected_error_type as e:
        return False, e
    except Exception as e:
        return False, e


# Test fixtures for each extra
@pytest.fixture
def deep_learning_available():
    """Check if deep learning packages are available."""
    available_packages = {}
    
    # Check PyTorch
    try:
        torch = pytest.importorskip("torch")
        available_packages["torch"] = torch
    except pytest.skip.Exception:
        available_packages["torch"] = None
    
    # Check TensorFlow
    try:
        tensorflow = pytest.importorskip("tensorflow")
        available_packages["tensorflow"] = tensorflow
    except pytest.skip.Exception:
        available_packages["tensorflow"] = None
    
    # Check JAX
    try:
        jax = pytest.importorskip("jax")
        available_packages["jax"] = jax
    except pytest.skip.Exception:
        available_packages["jax"] = None
    
    if not any(available_packages.values()):
        pytest.skip("No deep learning packages available")
    
    return available_packages


@pytest.fixture
def automl_available():
    """Check if AutoML packages are available."""
    available_packages = {}
    
    # Check Optuna
    try:
        optuna = pytest.importorskip("optuna")
        available_packages["optuna"] = optuna
    except pytest.skip.Exception:
        available_packages["optuna"] = None
    
    # Check HyperOpt
    try:
        hyperopt = pytest.importorskip("hyperopt")
        available_packages["hyperopt"] = hyperopt
    except pytest.skip.Exception:
        available_packages["hyperopt"] = None
    
    if not any(available_packages.values()):
        pytest.skip("No AutoML packages available")
    
    return available_packages


@pytest.fixture
def explainability_available():
    """Check if explainability packages are available."""
    available_packages = {}
    
    # Check SHAP
    try:
        shap = pytest.importorskip("shap")
        available_packages["shap"] = shap
    except pytest.skip.Exception:
        available_packages["shap"] = None
    
    # Check LIME
    try:
        lime = pytest.importorskip("lime")
        available_packages["lime"] = lime
    except pytest.skip.Exception:
        available_packages["lime"] = None
    
    if not any(available_packages.values()):
        pytest.skip("No explainability packages available")
    
    return available_packages


@pytest.fixture
def streaming_available():
    """Check if streaming packages are available."""
    available_packages = {}
    
    # Check Redis
    try:
        redis = pytest.importorskip("redis")
        available_packages["redis"] = redis
    except pytest.skip.Exception:
        available_packages["redis"] = None
    
    # Check Kafka (kafka-python)
    try:
        kafka = pytest.importorskip("kafka")
        available_packages["kafka"] = kafka
    except pytest.skip.Exception:
        available_packages["kafka"] = None
    
    # asyncio is part of standard library
    import asyncio
    available_packages["asyncio"] = asyncio
    
    if not any(available_packages.values()):
        pytest.skip("No streaming packages available")
    
    return available_packages


# Combined extras fixture
@pytest.fixture
def extras_availability():
    """Check availability of all extras."""
    availability = {}
    
    # Deep learning
    try:
        deep_packages = {}
        for pkg in ["torch", "tensorflow", "jax"]:
            try:
                deep_packages[pkg] = pytest.importorskip(pkg)
            except pytest.skip.Exception:
                deep_packages[pkg] = None
        availability["deep"] = deep_packages
    except:
        availability["deep"] = {}
    
    # AutoML
    try:
        automl_packages = {}
        for pkg in ["optuna", "hyperopt"]:
            try:
                automl_packages[pkg] = pytest.importorskip(pkg)
            except pytest.skip.Exception:
                automl_packages[pkg] = None
        availability["automl"] = automl_packages
    except:
        availability["automl"] = {}
    
    # Explainability
    try:
        explainability_packages = {}
        for pkg in ["shap", "lime"]:
            try:
                explainability_packages[pkg] = pytest.importorskip(pkg)
            except pytest.skip.Exception:
                explainability_packages[pkg] = None
        availability["explainability"] = explainability_packages
    except:
        availability["explainability"] = {}
    
    # Streaming
    try:
        streaming_packages = {}
        for pkg in ["redis", "kafka"]:
            try:
                streaming_packages[pkg] = pytest.importorskip(pkg)
            except pytest.skip.Exception:
                streaming_packages[pkg] = None
        import asyncio
        streaming_packages["asyncio"] = asyncio
        availability["streaming"] = streaming_packages
    except:
        availability["streaming"] = {}
    
    return availability

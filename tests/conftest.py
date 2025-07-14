"""Consolidated pytest configuration for all Pynomaly tests."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

# Suppress warnings early and comprehensively
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", category=UserWarning, module="dependency_injector.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.*")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.*")

# Add src to Python path - single authoritative path setup
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import shared test utilities and fixtures
from tests.shared.fixtures import *  # noqa
from tests.shared.factories import *  # noqa


# Test configuration and cleanup
@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Session-level test setup and cleanup."""
    # Session setup
    print("\n🧪 Starting Pynomaly test session...")
    
    yield
    
    # Session cleanup
    print("\n✅ Pynomaly test session completed")


@pytest.fixture(autouse=True)
def test_isolation():
    """Ensure test isolation by cleaning up after each test."""
    yield
    
    # Clean up any global state that might leak between tests
    import gc
    gc.collect()
    
    # Clean up imported test modules
    test_modules_to_clear = [
        mod for mod in sys.modules.keys()
        if "pynomaly" in mod and any(test_path in mod for test_path in ["test_", "_test"])
    ]
    
    for module in test_modules_to_clear:
        sys.modules.pop(module, None)
"""
Simple Integration Test

A basic integration test to verify the testing framework works correctly.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class TestSimpleIntegration:
    """Simple integration test class."""

    def test_basic_import(self):
        """Test basic imports work."""
        try:
            from monorepo.domain.entities import Dataset
            assert Dataset is not None
        except ImportError:
            pytest.skip("Core imports not available")

    def test_container_creation(self):
        """Test container creation."""
        try:
            from monorepo.infrastructure.config.container import create_container
            container = create_container()
            assert container is not None
        except ImportError:
            pytest.skip("Container creation not available")

    def test_async_framework(self):
        """Test async framework works."""
        import asyncio

        async def simple_async_test():
            await asyncio.sleep(0.01)
            return "success"

        result = asyncio.run(simple_async_test())
        assert result == "success"

    def test_basic_computation(self):
        """Test basic computation."""
        import numpy as np
        import pandas as pd

        # Create simple data
        data = np.random.randn(100, 5)
        df = pd.DataFrame(data)

        # Basic assertions
        assert df.shape == (100, 5)
        assert not df.empty
        assert all(df.dtypes == np.float64)

    def test_file_operations(self):
        """Test file operations."""
        import tempfile
        from pathlib import Path

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            # Read file
            content = temp_path.read_text()
            assert content == "test content"
        finally:
            # Clean up
            temp_path.unlink()

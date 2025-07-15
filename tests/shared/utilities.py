"""Test utilities and helper functions."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class ResourceManager:
    """Manages test resources and ensures proper cleanup."""
    
    def __init__(self):
        self.temp_files: List[Path] = []
        self.temp_dirs: List[Path] = []
        self.patches: List[Any] = []
        self.async_tasks: List[asyncio.Task] = []
        self.processes: List[Any] = []
        
    def create_temp_file(self, suffix: str = ".txt", content: str = "") -> Path:
        """Create a temporary file with optional content."""
        temp_file = Path(tempfile.mktemp(suffix=suffix))
        temp_file.write_text(content)
        self.temp_files.append(temp_file)
        return temp_file
        
    def create_temp_dir(self) -> Path:
        """Create a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        return temp_dir
        
    def create_csv_file(self, data: pd.DataFrame, filename: Optional[str] = None) -> Path:
        """Create a temporary CSV file from DataFrame."""
        if filename is None:
            temp_file = self.create_temp_file(suffix=".csv")
        else:
            temp_dir = self.create_temp_dir()
            temp_file = temp_dir / filename
            
        data.to_csv(temp_file, index=False)
        if temp_file not in self.temp_files:
            self.temp_files.append(temp_file)
        return temp_file
        
    def create_json_file(self, data: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Create a temporary JSON file from dictionary."""
        if filename is None:
            temp_file = self.create_temp_file(suffix=".json")
        else:
            temp_dir = self.create_temp_dir()
            temp_file = temp_dir / filename
            
        temp_file.write_text(json.dumps(data, indent=2))
        if temp_file not in self.temp_files:
            self.temp_files.append(temp_file)
        return temp_file
        
    def patch_object(self, target: str, **kwargs) -> Any:
        """Create and track a patch object."""
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.patches.append(patcher)
        return mock
        
    def create_async_task(self, coro) -> asyncio.Task:
        """Create and track an async task."""
        task = asyncio.create_task(coro)
        self.async_tasks.append(task)
        return task
        
    def cleanup_all(self) -> None:
        """Clean up all managed resources."""
        # Cancel async tasks
        for task in self.async_tasks:
            if not task.done():
                task.cancel()
                
        # Stop patches
        for patcher in self.patches:
            try:
                patcher.stop()
            except RuntimeError:
                pass  # Already stopped
                
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except (OSError, PermissionError):
                pass
                
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except (OSError, PermissionError):
                pass
                
        # Terminate processes
        for process in self.processes:
            try:
                if hasattr(process, 'terminate'):
                    process.terminate()
                    process.wait(timeout=5)
            except Exception:
                pass
                
        # Clear all lists
        self.temp_files.clear()
        self.temp_dirs.clear()
        self.patches.clear()
        self.async_tasks.clear()
        self.processes.clear()


class TestUtilities:
    """Collection of utility functions for testing."""
    
    @staticmethod
    def assert_dataframes_equal(
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        check_dtype: bool = True,
        check_index: bool = True,
        check_names: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> None:
        """Assert two DataFrames are equal with configurable checks."""
        pd.testing.assert_frame_equal(
            df1, df2,
            check_dtype=check_dtype,
            check_index=check_index,
            check_names=check_names,
            rtol=rtol,
            atol=atol
        )
    
    @staticmethod
    def assert_arrays_equal(
        arr1: np.ndarray,
        arr2: np.ndarray,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> None:
        """Assert two numpy arrays are equal."""
        np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)
    
    @staticmethod
    def wait_for_condition(
        condition_func,
        timeout: float = 10.0,
        interval: float = 0.1,
        timeout_message: str = "Condition not met within timeout"
    ) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        raise TimeoutError(timeout_message)
    
    @staticmethod
    async def wait_for_async_condition(
        condition_func,
        timeout: float = 10.0,
        interval: float = 0.1,
        timeout_message: str = "Async condition not met within timeout"
    ) -> bool:
        """Wait for an async condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)
        raise TimeoutError(timeout_message)
    
    @staticmethod
    def create_mock_with_spec(spec_class: type, **kwargs) -> MagicMock:
        """Create a mock with proper spec."""
        return MagicMock(spec=spec_class, **kwargs)
    
    @staticmethod
    def verify_no_warnings(warning_list: List) -> None:
        """Verify that no warnings were raised."""
        if warning_list:
            warnings_text = "\n".join([str(w.message) for w in warning_list])
            raise AssertionError(f"Unexpected warnings raised:\n{warnings_text}")
    
    @staticmethod
    def generate_test_id(prefix: str = "test") -> str:
        """Generate a unique test ID."""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> tuple[Any, float]:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    async def measure_async_execution_time(coro) -> tuple[Any, float]:
        """Measure execution time of an async function."""
        start_time = time.perf_counter()
        result = await coro
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    def create_test_environment_variables() -> Dict[str, str]:
        """Create test environment variables."""
        return {
            "PYNOMALY_ENVIRONMENT": "test",
            "PYNOMALY_DEBUG": "false",
            "PYNOMALY_LOG_LEVEL": "INFO",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1"
        }
    
    @staticmethod
    def set_environment_variables(env_vars: Dict[str, str]) -> Dict[str, str]:
        """Set environment variables and return original values."""
        original_values = {}
        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value
        return original_values
    
    @staticmethod
    def restore_environment_variables(original_values: Dict[str, str]) -> None:
        """Restore environment variables to original values."""
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    @staticmethod
    def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate data against JSON schema (simplified validation)."""
        # This is a basic implementation - for full JSON schema validation,
        # consider using jsonschema library
        def validate_type(value, expected_type):
            if expected_type == "string":
                return isinstance(value, str)
            elif expected_type == "number":
                return isinstance(value, (int, float))
            elif expected_type == "integer":
                return isinstance(value, int)
            elif expected_type == "boolean":
                return isinstance(value, bool)
            elif expected_type == "array":
                return isinstance(value, list)
            elif expected_type == "object":
                return isinstance(value, dict)
            return True
        
        if "type" in schema:
            if not validate_type(data, schema["type"]):
                return False
        
        if "properties" in schema and isinstance(data, dict):
            for prop, prop_schema in schema["properties"].items():
                if prop in data:
                    if not TestUtilities.validate_json_schema(data[prop], prop_schema):
                        return False
        
        if "required" in schema and isinstance(data, dict):
            for required_prop in schema["required"]:
                if required_prop not in data:
                    return False
        
        return True
    
    @staticmethod
    def create_sample_config(config_type: str = "test") -> Dict[str, Any]:
        """Create sample configuration for testing."""
        configs = {
            "test": {
                "database": {"url": "sqlite:///:memory:"},
                "api": {"host": "localhost", "port": 8000},
                "logging": {"level": "INFO"}
            },
            "minimal": {
                "database": {"url": "sqlite:///:memory:"}
            },
            "full": {
                "database": {
                    "url": "postgresql://user:pass@localhost/test",
                    "pool_size": 5,
                    "echo": False
                },
                "api": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "workers": 4,
                    "debug": False
                },
                "cache": {
                    "backend": "redis",
                    "url": "redis://localhost:6379/0",
                    "ttl": 3600
                },
                "monitoring": {
                    "enabled": True,
                    "metrics_port": 9090,
                    "health_check_interval": 30
                },
                "logging": {
                    "level": "INFO",
                    "format": "json",
                    "handlers": ["console", "file"]
                }
            }
        }
        return configs.get(config_type, configs["test"])


class PerformanceProfiler:
    """Simple performance profiler for tests."""
    
    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}
    
    def measure(self, name: str, func, *args, **kwargs) -> Any:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        if name not in self.measurements:
            self.measurements[name] = []
        self.measurements[name].append(execution_time)
        
        return result
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a measurement."""
        if name not in self.measurements:
            return {}
        
        times = self.measurements[name]
        return {
            "count": len(times),
            "mean": np.mean(times),
            "median": np.median(times),
            "min": np.min(times),
            "max": np.max(times),
            "std": np.std(times)
        }
    
    def report(self) -> str:
        """Generate a performance report."""
        lines = ["Performance Report", "=" * 50]
        for name in sorted(self.measurements.keys()):
            stats = self.get_stats(name)
            lines.append(f"{name}:")
            lines.append(f"  Count: {stats['count']}")
            lines.append(f"  Mean: {stats['mean']:.4f}s")
            lines.append(f"  Median: {stats['median']:.4f}s")
            lines.append(f"  Min: {stats['min']:.4f}s")
            lines.append(f"  Max: {stats['max']:.4f}s")
            lines.append(f"  Std Dev: {stats['std']:.4f}s")
            lines.append("")
        
        return "\n".join(lines)